//! Subtitle cue parsing — SubRip, WebVTT, ASS/SSA → a common [`Cue`] list, plus the
//! markup-stripping every path shares (spec: subtitle support; RFC 9559 §12.7 the
//! Matroska S_TEXT/* mappings the demuxer announces).
//!
//! Two entry shapes, because a subtitle stream reaches us two ways:
//!
//! * **Whole-file parse** ([`parse_srt`] / [`parse_vtt`] / [`parse_ass`]): a standalone
//!   `.srt` / `.vtt` / `.ass` fed as one buffer through `filesrc`. The file carries its
//!   own timeline (`HH:MM:SS,mmm` cue markers), so the parser recovers every cue's
//!   `[start, end)` from the text.
//!
//! * **One-block-one-cue** ([`cue_body_srt`] / [`cue_body_vtt`] / [`cue_body_ass_dialogue`]):
//!   an mkv S_TEXT/* track, where the demuxer has *already framed* each cue as one buffer
//!   whose PTS + duration (`BlockDuration`) is the timing. Here there is no timeline to
//!   parse — only the cue **body** to normalise (strip markup to plain UTF-8 + newlines),
//!   and the timing rides the buffer. This is why the overlay is format-agnostic: `subparse`
//!   collapses all three markups to the same plain text, and the container gives the clock.
//!
//! v1 is **text only** — no positioning, karaoke, colour or bitmap (PGS) subs. Multi-line
//! cues survive (newlines are preserved); everything else is stripped to nothing. Markup
//! grammar is worked from the public format specs (clean-room — no libass/vlc source
//! consulted); citations sit at each parser.

// COLD: whole-file / one-block-one-cue subtitle parsing — runs once per stream or once per
// sparse cue Block (seconds apart), never on the per-video-frame overlay path. Every Vec/String
// here builds owned cue text, the natural shape of a subtitle document; not steady-state heap.
#![allow(clippy::disallowed_methods)]

use profluens_core::time::Timestamp;

/// One parsed subtitle cue: plain UTF-8 text (markup already stripped, `\n` between lines)
/// shown over `[start, end)`. `end == Timestamp::NONE` when a format left the span open
/// (the overlay then falls back to a default cue duration — see `overlay.rs`).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Cue {
    pub start: Timestamp,
    pub end: Timestamp,
    pub text: String,
}

impl Cue {
    /// Whether this cue is active at presentation time `t` (half-open `[start, end)`, the
    /// standard cue-interval convention — a cue ending exactly at `t` is already gone). A
    /// cue with an unset `end` is treated as still open (active for all `t >= start`).
    pub fn active_at(&self, t: Timestamp) -> bool {
        t >= self.start && (self.end.is_none() || t < self.end)
    }
}

// ============================ SubRip (.srt) ============================
//
// The SubRip format is a de-facto standard (no RFC / W3C spec; the reference is the SubRip
// tool and the widely-mirrored community grammar). A file is a sequence of cues separated by
// a blank line; each cue is:
//
//   <index>\n
//   HH:MM:SS,mmm --> HH:MM:SS,mmm [optional trailing coords, ignored]\n
//   line 1\n
//   line 2\n
//   \n
//
// Timestamps use a **comma** before the milliseconds. The index line is advisory (we do not
// require it to be numeric or monotonic — some muxers drop it). Inline markup is limited to
// a handful of HTML-ish tags (`<i>`, `<b>`, `<u>`, `<font …>`); we strip all `<...>` runs.

/// Parse a whole SubRip document into cues, in file order. Malformed cues (no `-->` line)
/// are skipped rather than erroring — a subtitle file with one bad block should still show
/// the rest.
pub fn parse_srt(input: &str) -> Vec<Cue> {
    let mut cues = Vec::new();
    // Split on blank lines (a run of whitespace-only lines) into cue blocks.
    for block in split_blocks(input) {
        let mut lines = block.lines();
        // The first line is the (advisory) index unless it is itself the timing line.
        let first = match lines.next() {
            Some(l) => l.trim(),
            None => continue,
        };
        // The timing line is either the first line (no index) or the second (index then
        // timing). Either way, once we have consumed up to and including it, the remaining
        // `lines` iterator is the cue body.
        let timing_line = if first.contains("-->") {
            first // no index line — the block opens with the timing
        } else {
            match lines.next() {
                Some(l) => l.trim(),
                None => continue, // index but no timing → not a cue
            }
        };
        let Some((start, end)) = parse_arrow_timing(timing_line, srt_time) else {
            continue;
        };
        let body = lines.collect::<Vec<_>>().join("\n");
        cues.push(Cue { start, end, text: strip_srt(body.trim()) });
    }
    cues
}

/// Normalise one SubRip cue *body* (the text after the timing line) to plain UTF-8 — the
/// one-block-one-cue entry (the mkv S_TEXT/UTF8 case, where timing rides the buffer). Strips
/// the HTML-ish inline tags SubRip permits.
pub fn cue_body_srt(body: &str) -> String {
    strip_srt(body.trim())
}

/// `HH:MM:SS,mmm` (SubRip) → nanoseconds. Hours may be more than two digits; the millisecond
/// field is exactly three. `None` on any malformation.
fn srt_time(s: &str) -> Option<u64> {
    // Split the comma between seconds and milliseconds, then the colons.
    let (hms, ms) = s.trim().split_once(',')?;
    let ms: u64 = ms.trim().parse().ok()?;
    hms_to_ns(hms, ms)
}

/// Strip SubRip/HTML-ish inline markup (`<i>`, `<b>`, `<font …>`, …): drop every `<...>` run,
/// keep the text between. Newlines are preserved (multi-line cues stack in the overlay).
fn strip_srt(s: &str) -> String {
    strip_angle_tags(s)
}

// ============================ WebVTT (.vtt) ============================
//
// WebVTT is a W3C spec ("WebVTT: The Web Video Text Tracks Format", W3C). A document opens
// with the `WEBVTT` signature line, may carry header metadata, then STYLE/NOTE/REGION blocks
// and cue blocks. A cue is an optional identifier line, then:
//
//   HH:MM:SS.mmm --> HH:MM:SS.mmm [cue settings, ignored]\n
//   payload line(s)\n
//
// Timestamps use a **dot** before milliseconds (vs SubRip's comma), and the hours field is
// optional (`MM:SS.mmm` is legal). The payload carries inline spans (`<c>`, `<v Speaker>`,
// `<b>`, timestamp tags `<00:01.000>`, …); we strip all `<...>` runs (W3C §"cue text span").
// NOTE/STYLE/REGION blocks are skipped (they contain no cue text).

/// Parse a whole WebVTT document into cues, in file order. The `WEBVTT` signature and any
/// header/STYLE/NOTE/REGION blocks are skipped; only timed cue blocks yield [`Cue`]s.
pub fn parse_vtt(input: &str) -> Vec<Cue> {
    let mut cues = Vec::new();
    let mut blocks = split_blocks(input).into_iter();
    // The first block holds the `WEBVTT` signature (+ optional header text) — skip it.
    let _signature = blocks.next();
    for block in blocks {
        // Skip non-cue blocks (comments / styles / regions have no timing line).
        let upper = block.trim_start();
        if upper.starts_with("NOTE") || upper.starts_with("STYLE") || upper.starts_with("REGION") {
            continue;
        }
        // A cue block: an optional identifier line, then the `-->` timing, then payload.
        let mut lines = block.lines().peekable();
        let mut timing_line = None;
        let mut body_lines: Vec<&str> = Vec::new();
        for line in &mut lines {
            if timing_line.is_none() {
                if line.contains("-->") {
                    timing_line = Some(line.trim());
                }
                // else: it's the identifier line — ignore it.
            } else {
                body_lines.push(line);
            }
        }
        let Some(tl) = timing_line else { continue };
        let Some((start, end)) = parse_arrow_timing(tl, vtt_time) else { continue };
        cues.push(Cue { start, end, text: strip_vtt(body_lines.join("\n").trim()) });
    }
    cues
}

/// Normalise one WebVTT cue *body* to plain UTF-8 — the one-block-one-cue entry (mkv
/// S_TEXT/WEBVTT, timing on the buffer). Strips inline `<...>` spans.
pub fn cue_body_vtt(body: &str) -> String {
    strip_vtt(body.trim())
}

/// `[HH:]MM:SS.mmm` (WebVTT) → nanoseconds. The hours field is optional. `None` on
/// malformation.
fn vtt_time(s: &str) -> Option<u64> {
    let (hms, ms) = s.trim().split_once('.')?;
    let ms: u64 = ms.trim().parse().ok()?;
    hms_to_ns(hms, ms)
}

/// Strip WebVTT inline markup: every `<...>` span (`<c>`, `<v …>`, `<b>`, timestamp tags),
/// keeping the text between. Newlines preserved.
fn strip_vtt(s: &str) -> String {
    strip_angle_tags(s)
}

// ============================ ASS / SSA (.ass) ============================
//
// Advanced SubStation Alpha (and its predecessor SubStation Alpha) — the Aegisub / libass
// grammar. An `.ass` file is INI-like sections; subtitle events live in `[Events]`:
//
//   [Events]
//   Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
//   Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\i1}hello{\i0}\Nsecond line
//
// The `Format:` line names the columns; `Dialogue:` rows carry the values comma-separated,
// **but the Text field is the last column and may itself contain commas**, so it is
// everything after the (fields-1)th comma. Timestamps are `H:MM:SS.cc` (centiseconds — two
// digits, not three). Text override blocks `{\...}` (styling/karaoke/positioning) are
// stripped; `\N` (hard break) and `\n` (soft break) become newlines; `\h` is a hard space.
//
// v1: text only. Styles, layers, positioning and karaoke are parsed-and-discarded.

/// Parse a whole ASS/SSA document's `[Events]` `Dialogue:` lines into cues, in file order.
/// The `Format:` line inside `[Events]` fixes the column order (so a non-default field layout
/// still finds Start/End/Text); absent it, the standard ASS order is assumed.
pub fn parse_ass(input: &str) -> Vec<Cue> {
    let mut cues = Vec::new();
    let mut in_events = false;
    // Column indices for Start / End / Text, from the `Format:` line (defaults = standard ASS).
    let (mut i_start, mut i_end, mut i_text) = (1usize, 2usize, 9usize);
    let mut field_count = 10usize;
    for raw in input.lines() {
        let line = raw.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_events = line.eq_ignore_ascii_case("[Events]");
            continue;
        }
        if !in_events {
            continue;
        }
        if let Some(rest) = strip_prefix_ci(line, "Format:") {
            let cols: Vec<&str> = rest.split(',').map(|c| c.trim()).collect();
            field_count = cols.len();
            for (i, c) in cols.iter().enumerate() {
                if c.eq_ignore_ascii_case("Start") {
                    i_start = i;
                } else if c.eq_ignore_ascii_case("End") {
                    i_end = i;
                } else if c.eq_ignore_ascii_case("Text") {
                    i_text = i;
                }
            }
            continue;
        }
        if let Some(rest) = strip_prefix_ci(line, "Dialogue:") {
            if let Some(cue) = parse_ass_dialogue(rest, i_start, i_end, i_text, field_count) {
                cues.push(cue);
            }
        }
    }
    cues
}

/// Parse one `Dialogue:` row body (everything after the `Dialogue:` prefix) into a [`Cue`],
/// using the column indices from the `Format:` line. The Text column is the tail (it may hold
/// commas), so only the first `field_count - 1` commas split fields.
fn parse_ass_dialogue(
    row: &str,
    i_start: usize,
    i_end: usize,
    i_text: usize,
    field_count: usize,
) -> Option<Cue> {
    // Split into exactly `field_count` fields: the first `field_count-1` on commas, the
    // remainder (the Text column) whole.
    let mut fields: Vec<&str> = Vec::with_capacity(field_count);
    let mut rest = row;
    for _ in 0..field_count.saturating_sub(1) {
        match rest.split_once(',') {
            Some((head, tail)) => {
                fields.push(head);
                rest = tail;
            }
            None => break,
        }
    }
    fields.push(rest); // the Text tail (or a short row's last field)
    let start = ass_time(fields.get(i_start)?.trim())?;
    let end = ass_time(fields.get(i_end)?.trim())?;
    let text = fields.get(i_text).copied().unwrap_or("");
    Some(Cue {
        start: Timestamp::from_nanos(start),
        end: Timestamp::from_nanos(end),
        text: strip_ass(text),
    })
}

/// Normalise one ASS `Dialogue:` cue text field to plain UTF-8 — the one-block-one-cue entry.
/// In an mkv S_TEXT/ASS track each Block's payload is the ASS *event line* body
/// (`ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text` per RFC 9559 §12.7 — no
/// Start/End, since timing is the block's), so the cue text is the field after the 8th comma.
/// We reuse the tail-after-N-commas rule with the mkv field count (9), then strip overrides.
pub fn cue_body_ass_dialogue(block_body: &str) -> String {
    // mkv S_TEXT/ASS Block layout (RFC 9559 §12.7): 8 leading fields then Text.
    const MKV_ASS_FIELDS: usize = 9;
    let mut rest = block_body;
    for _ in 0..MKV_ASS_FIELDS - 1 {
        match rest.split_once(',') {
            Some((_, tail)) => rest = tail,
            None => {
                rest = "";
                break;
            }
        }
    }
    strip_ass(rest)
}

/// `H:MM:SS.cc` (ASS — centiseconds) → nanoseconds. `None` on malformation.
///
/// The centisecond field is **checked**: `cc * 10` on an untrusted 19-digit field overflows
/// `u64` (a debug-build panic), and `hms_to_ns`'s `< 1000` range test only runs afterwards.
fn ass_time(s: &str) -> Option<u64> {
    let (hms, cc) = s.trim().split_once('.')?;
    // Centiseconds: two digits → milliseconds = cc * 10.
    let cc: u64 = cc.trim().parse().ok()?;
    hms_to_ns(hms, cc.checked_mul(10)?)
}

/// Strip ASS override blocks (`{\...}`) and convert `\N`/`\n` to newlines and `\h` to a space
/// (Aegisub/libass text conventions). Everything else is literal.
///
/// An **unbalanced** `{` is literal, not an opening brace: libass renders a brace that never
/// closes as text, and treating it as an override start would silently swallow the rest of the
/// cue (a bare `{` in dialogue is common enough that this is real caption loss). `last_close`
/// is computed once, so this stays linear. A stray `}` outside a block is likewise literal.
fn strip_ass(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    // The last byte offset a `}` occupies: a `{` after it can never be closed.
    let last_close = s.rfind('}');
    let mut chars = s.char_indices().peekable();
    let mut depth = 0u32; // inside a `{...}` override block
    while let Some((i, c)) = chars.next() {
        match c {
            '{' if depth > 0 || last_close.is_some_and(|j| j > i) => {
                depth = depth.saturating_add(1)
            }
            '}' if depth > 0 => depth -= 1,
            _ if depth > 0 => {} // swallow override-block content
            '\\' => match chars.peek().map(|&(_, c)| c) {
                Some('N') | Some('n') => {
                    chars.next();
                    out.push('\n');
                }
                Some('h') => {
                    chars.next();
                    out.push(' ');
                }
                // An unknown escape: keep the backslash literal (rare; not styling).
                _ => out.push('\\'),
            },
            _ => out.push(c),
        }
    }
    out
}

// ============================ shared helpers ============================

/// Split a document into blocks separated by one-or-more blank (whitespace-only) lines,
/// dropping empty blocks. Blocks keep their internal newlines. Handles both `\n` and `\r\n`.
fn split_blocks(input: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for line in input.lines() {
        if line.trim().is_empty() {
            if !cur.is_empty() {
                blocks.push(cur.join("\n"));
                cur.clear();
            }
        } else {
            cur.push(line);
        }
    }
    if !cur.is_empty() {
        blocks.push(cur.join("\n"));
    }
    blocks
}

/// Parse a `A --> B [settings]` line into `(start, end)` timestamps via `time` (the
/// format-specific `HH:MM:SS<sep>frac` parser). Trailing cue settings after the end time are
/// ignored. `None` if the arrow is absent or either side fails to parse.
fn parse_arrow_timing(line: &str, time: fn(&str) -> Option<u64>) -> Option<(Timestamp, Timestamp)> {
    let (lhs, rhs) = line.split_once("-->")?;
    let start = time(lhs.trim())?;
    // The end time is the first whitespace-delimited token on the rhs (cue settings, if any,
    // follow it). `split_whitespace` already skips leading whitespace, so no `trim` first.
    let end_tok = rhs.split_whitespace().next()?;
    let end = time(end_tok)?;
    Some((Timestamp::from_nanos(start), Timestamp::from_nanos(end)))
}

/// `[HH:]MM:SS` + a millisecond remainder → nanoseconds. Hours are optional (WebVTT allows
/// `MM:SS`); minutes/seconds are required. `None` on any non-numeric or out-of-range field.
///
/// Every step is **checked**: `h` comes from untrusted digits, and `h * 3600 * 1000 * 1_000_000`
/// overflows `u64` from about seven digits of hours upward (`6000000:00:00,000` is enough), which
/// is a debug-build panic and a silently wrapped timestamp in release. A subtitle file is
/// attacker-controlled input, so an out-of-range timestamp must be a rejected cue, not a crash.
fn hms_to_ns(hms: &str, ms: u64) -> Option<u64> {
    let parts: Vec<&str> = hms.trim().split(':').collect();
    let (h, m, s): (u64, u64, u64) = match parts.as_slice() {
        [h, m, s] => (h.trim().parse().ok()?, m.trim().parse().ok()?, s.trim().parse().ok()?),
        [m, s] => (0, m.trim().parse().ok()?, s.trim().parse().ok()?),
        _ => return None,
    };
    if m >= 60 || s >= 60 || ms >= 1000 {
        return None; // out-of-range field — a malformed timestamp
    }
    h.checked_mul(3600)?
        .checked_add(m * 60 + s)?
        .checked_mul(1000)?
        .checked_add(ms)?
        .checked_mul(1_000_000)
}

/// Drop every `<...>` run, keeping the text between (HTML-ish / WebVTT inline tags). A `<` with
/// **no `>` anywhere after it** cannot be a tag, so it is kept literally rather than swallowing
/// the rest of the cue — an unescaped `<` in dialogue ("5 < 6 minutes") is common in SubRip and
/// used to delete everything after it.
fn strip_angle_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let last_gt = s.rfind('>');
    let mut in_tag = false;
    for (i, c) in s.char_indices() {
        match c {
            '<' if !in_tag && last_gt.is_some_and(|j| j > i) => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if in_tag => {}
            _ => out.push(c),
        }
    }
    out
}

/// Case-insensitive `strip_prefix` for the ASS `Format:` / `Dialogue:` keywords.
///
/// Compares **bytes**, never `&str` slices: `s[..prefix.len()]` panics when the split point
/// lands inside a multi-byte character, and an `[Events]` section of untrusted text routinely
/// contains non-ASCII lines (a CJK caption line, a stray translated note). The keywords are
/// ASCII, so a byte-wise `eq_ignore_ascii_case` is exactly equivalent — and a prefix match on
/// ASCII bytes is always a char boundary, so the tail slice is safe.
fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let head = s.as_bytes().get(..prefix.len())?;
    if head.eq_ignore_ascii_case(prefix.as_bytes()) {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ns(h: u64, m: u64, s: u64, ms: u64) -> Timestamp {
        Timestamp::from_nanos(((h * 3600 + m * 60 + s) * 1000 + ms) * 1_000_000)
    }

    // ---- SubRip ------------------------------------------------------------------

    #[test]
    fn srt_parses_indexed_multiline_cues() {
        let doc = "1\n00:00:01,000 --> 00:00:03,500\nHello world\n\n\
                   2\n00:00:04,000 --> 00:00:05,000\nline one\nline two\n";
        let cues = parse_srt(doc);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0], Cue { start: ns(0, 0, 1, 0), end: ns(0, 0, 3, 500), text: "Hello world".into() });
        assert_eq!(cues[1].text, "line one\nline two", "multi-line cue keeps the break");
        assert_eq!(cues[1].start, ns(0, 0, 4, 0));
    }

    #[test]
    fn srt_strips_html_tags_and_tolerates_missing_index() {
        // No index line; italic markup.
        let doc = "00:01:00,000 --> 00:01:02,000\n<i>italic</i> and <font color=\"red\">red</font>\n";
        let cues = parse_srt(doc);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "italic and red");
        assert_eq!(cues[0].start, ns(0, 1, 0, 0));
    }

    #[test]
    fn srt_skips_a_malformed_block_but_keeps_the_rest() {
        let doc = "1\nthis is not a timing line\ntext\n\n\
                   2\n00:00:02,000 --> 00:00:03,000\ngood\n";
        let cues = parse_srt(doc);
        assert_eq!(cues.len(), 1, "the bad block is dropped, the good one survives");
        assert_eq!(cues[0].text, "good");
    }

    #[test]
    fn srt_cue_body_normalises_one_block() {
        assert_eq!(cue_body_srt("<b>bold</b>\nnext"), "bold\nnext");
    }

    // ---- WebVTT ------------------------------------------------------------------

    #[test]
    fn vtt_parses_dot_millis_and_optional_hours() {
        let doc = "WEBVTT\n\n\
                   00:01.000 --> 00:03.000\nno hours\n\n\
                   00:00:04.500 --> 00:00:06.000 line:90%\nwith settings\n";
        let cues = parse_vtt(doc);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0], Cue { start: ns(0, 0, 1, 0), end: ns(0, 0, 3, 0), text: "no hours".into() });
        assert_eq!(cues[1].start, ns(0, 0, 4, 500));
        assert_eq!(cues[1].text, "with settings", "cue settings after the end time are ignored");
    }

    #[test]
    fn vtt_skips_note_and_strips_inline_spans() {
        let doc = "WEBVTT\n\n\
                   NOTE this is a comment\nover two lines\n\n\
                   cue-id-1\n00:00.000 --> 00:02.000\n<v Bob>hi</v> <c.yellow>there</c>\n";
        let cues = parse_vtt(doc);
        assert_eq!(cues.len(), 1, "the NOTE block is not a cue");
        assert_eq!(cues[0].text, "hi there", "voice + class spans stripped; the id line skipped");
    }

    #[test]
    fn vtt_cue_body_strips_timestamp_tags() {
        assert_eq!(cue_body_vtt("<00:01.000>karaoke <c>word</c>"), "karaoke word");
    }

    // ---- ASS / SSA ---------------------------------------------------------------

    #[test]
    fn ass_parses_events_with_override_blocks_and_breaks() {
        let doc = "[Script Info]\nTitle: t\n\n\
                   [Events]\n\
                   Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
                   Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,{\\i1}hello{\\i0}\\Nsecond\n";
        let cues = parse_ass(doc);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].start, ns(0, 0, 1, 0));
        assert_eq!(cues[0].end, ns(0, 0, 3, 500), "centiseconds .50 → 500 ms");
        assert_eq!(cues[0].text, "hello\nsecond", "overrides stripped, \\N → newline");
    }

    #[test]
    fn ass_text_field_may_contain_commas() {
        let doc = "[Events]\n\
                   Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
                   Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,one, two, three\n";
        let cues = parse_ass(doc);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "one, two, three", "the Text column keeps its internal commas");
    }

    #[test]
    fn ass_honours_a_reordered_format_line() {
        // Text before Start/End — the Format line must drive the column mapping.
        let doc = "[Events]\n\
                   Format: Text, Start, End\n\
                   Dialogue: hi there,0:00:01.00,0:00:02.00\n";
        let cues = parse_ass(doc);
        assert_eq!(cues.len(), 1);
        // With Text as column 0 (not the tail), the tail-split still lands the text field.
        assert_eq!(cues[0].start, ns(0, 0, 1, 0));
        assert_eq!(cues[0].end, ns(0, 0, 2, 0));
    }

    #[test]
    fn ass_mkv_block_body_takes_text_after_eight_commas() {
        // mkv S_TEXT/ASS Block body: ReadOrder,Layer,Style,Name,ML,MR,MV,Effect,Text
        let body = "0,0,Default,,0,0,0,,{\\b1}bold{\\b0} caption";
        assert_eq!(cue_body_ass_dialogue(body), "bold caption");
    }

    #[test]
    fn active_at_is_half_open() {
        let c = Cue { start: ns(0, 0, 1, 0), end: ns(0, 0, 2, 0), text: "x".into() };
        assert!(!c.active_at(ns(0, 0, 0, 999)));
        assert!(c.active_at(ns(0, 0, 1, 0)), "active at start");
        assert!(c.active_at(ns(0, 0, 1, 999)));
        assert!(!c.active_at(ns(0, 0, 2, 0)), "inactive exactly at end (half-open)");
        // An open-ended cue is active for all t >= start.
        let open = Cue { start: ns(0, 0, 1, 0), end: Timestamp::NONE, text: "x".into() };
        assert!(open.active_at(ns(9, 0, 0, 0)));
    }

    #[test]
    fn timestamp_range_checks_reject_malformed() {
        // 61 seconds / 61 minutes / 1000 ms are all invalid.
        assert!(parse_srt("00:00:61,000 --> 00:00:62,000\nx\n").is_empty());
        assert!(parse_srt("00:61:00,000 --> 00:62:00,000\nx\n").is_empty());
    }

    // ---- untrusted-input robustness (spec: a crash on a downloaded file is a P0) -------

    /// A line inside `[Events]` that is not one of the ASS keywords used to panic: the
    /// case-insensitive prefix test sliced `line[..7]` / `line[..9]`, and a byte offset landing
    /// inside a multi-byte character is a hard `&str` slice panic. Any non-ASCII line in the
    /// events section triggers it — routine in CJK-subtitled files.
    #[test]
    fn ass_non_ascii_line_in_events_does_not_panic() {
        assert!(parse_ass("[Events]\n日本語あいうえお\n").is_empty());
        assert!(parse_ass("[Events]\n日本語\n").is_empty(), "9 bytes: the Dialogue: split point");
        assert!(parse_ass("[Events]\nÜbersetzung\n").is_empty());
        // A well-formed Dialogue line among the noise still parses.
        let doc = "[Events]\n\
                   Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
                   日本語のコメント行\n\
                   Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,ok\n";
        let cues = parse_ass(doc);
        assert_eq!(cues.len(), 1, "the junk line is skipped, the real cue survives");
        assert_eq!(cues[0].text, "ok");
    }

    /// A timestamp with an enormous hours field overflowed `h * 3600 * 1000 * 1_000_000` —
    /// a panic under debug overflow checks, a silently wrapped (wrong) timestamp in release.
    /// Seven digits of hours is enough; the field is attacker-controlled.
    #[test]
    fn absurd_timestamps_are_rejected_not_overflowed() {
        assert!(parse_srt("6000000:00:00,000 --> 6000000:00:01,000\nx\n").is_empty());
        assert!(parse_srt("18446744073709551615:00:00,000 --> 0:00:01,000\nx\n").is_empty());
        assert!(parse_vtt("WEBVTT\n\n6000000:00:00.000 --> 6000000:00:01.000\nx\n").is_empty());
        let ass = |t: &str| {
            format!(
                "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, \
                 Effect, Text\nDialogue: 0,{t},0:00:02.00,D,,0,0,0,,hi\n"
            )
        };
        assert!(parse_ass(&ass("6000000:00:00.00")).is_empty(), "ass hours overflow");
        assert!(
            parse_ass(&ass("0:00:00.1844674407370955162")).is_empty(),
            "ass centiseconds * 10 overflow"
        );
        // The largest timestamp that still fits is accepted, so the guard is not over-eager.
        let cues = parse_srt("5000:00:00,000 --> 5000:00:01,000\nx\n");
        assert_eq!(cues.len(), 1, "5000 h is large but representable");
    }

    /// A `{` with no matching `}` is literal text, not the start of an override block — the old
    /// parser swallowed everything after it, deleting the rest of the caption.
    #[test]
    fn unbalanced_ass_brace_does_not_eat_the_caption() {
        assert_eq!(strip_ass("50% {not an override at all"), "50% {not an override at all");
        assert_eq!(strip_ass("a } stray close"), "a } stray close");
        // A real override block still strips, including nested braces.
        assert_eq!(strip_ass("{\\i1}hi{\\i0} there"), "hi there");
        assert_eq!(strip_ass("{\\t({\\fs20})}text"), "text", "nested override block");
    }

    /// A `<` with no `>` after it is literal text: an unescaped `<` in SubRip dialogue used to
    /// delete the remainder of the cue.
    #[test]
    fn unterminated_angle_bracket_keeps_the_text() {
        assert_eq!(cue_body_srt("5 < 6 minutes left"), "5 < 6 minutes left");
        assert_eq!(cue_body_vtt("a < b"), "a < b");
        // A genuine tag still strips.
        assert_eq!(cue_body_srt("<i>x</i> < y"), "x < y");
    }

    /// A pile of degenerate documents: none may panic, all must terminate.
    #[test]
    fn degenerate_documents_never_panic() {
        for doc in [
            "",
            "\u{feff}",
            "\r\n\r\n\r\n",
            "-->",
            "1\n-->\n\n",
            "00:00:00,000 -->",
            "--> 00:00:00,000",
            "WEBVTT",
            "WEBVTT\n\n\n\n",
            "[Events]",
            "[Events]\nFormat:\nDialogue:\n",
            "[Events]\nFormat: Text\nDialogue: \n",
            "[",
            "{{{{{{{{",
            "<<<<<<<<",
            "\u{feff}[Events]\nDialogue: 0,0:00:00.00,0:00:01.00,D,,0,0,0,,x\n",
        ] {
            let _ = parse_srt(doc);
            let _ = parse_vtt(doc);
            let _ = parse_ass(doc);
            let _ = cue_body_srt(doc);
            let _ = cue_body_vtt(doc);
            let _ = cue_body_ass_dialogue(doc);
        }
    }

    /// A seeded mutation fuzz over all six entry points — the regression guard for the whole
    /// class of bug the checked arithmetic and byte-wise prefix test fix, not just the six
    /// inputs that happened to be found.
    ///
    /// The byte-level mutation is the realistic model: `subparse` hands these parsers
    /// `String::from_utf8_lossy(block)`, so **any** byte string is reachable input, and every
    /// invalid sequence becomes U+FFFD — a 3-byte non-ASCII character that lands on the
    /// `Format:`/`Dialogue:` split points. That is how the char-boundary panic was reachable
    /// from arbitrary binary data, not merely from a CJK-subtitled file.
    ///
    /// Deterministic (fixed seed, xorshift64*) so a failure is always reproducible, and sized
    /// to run in well under a second.
    #[test]
    fn mutation_fuzz_never_panics() {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 >> 12;
                self.0 ^= self.0 << 25;
                self.0 ^= self.0 >> 27;
                self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
            }
            fn below(&mut self, n: usize) -> usize {
                if n == 0 { 0 } else { (self.next() % n as u64) as usize }
            }
        }
        let seeds = [
            "1\n00:00:01,000 --> 00:00:03,500\n<i>Hello</i> world\n\n\
             2\n00:00:04,000 --> 00:00:05,000\nline one\nline two\n",
            "WEBVTT\n\nNOTE a comment\n\ncue-1\n00:01.000 --> 00:03.000 line:90%\n\
             <v Bob>hi</v> <c.yellow>there</c>\n",
            "[Script Info]\nTitle: t\n\n[Events]\n\
             Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
             Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,{\\i1}hi{\\i0}\\Nsecond, third\n",
            "0,0,Default,,0,0,0,,{\\b1}bold{\\b0} caption\n",
        ];
        // Tokens that steer the mutator into the interesting branches far more often than
        // chance would: the field separators, the markup delimiters, and numeric fields wide
        // enough to overflow every multiply in `hms_to_ns` / `ass_time`.
        let interesting: [&[u8]; 25] = [
            b"-->", b":", b",", b".", b"<", b">", b"{", b"}", b"\\", b"\n", b"\r\n", b"[Events]",
            b"Format:", b"Dialogue:", b"WEBVTT", b"NOTE", b"9", b"0", b"99999999999999999999",
            b"6000000", b"18446744073709551615", "\u{65e5}".as_bytes(), b"\xff", b"\xc3",
            b"\xe6\x97",
        ];
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for i in 0..20_000u32 {
            let mut buf = seeds[i as usize % seeds.len()].as_bytes().to_vec();
            for _ in 0..1 + rng.below(6) {
                match rng.below(6) {
                    0 if !buf.is_empty() => {
                        let j = rng.below(buf.len());
                        buf[j] ^= 1 << rng.below(8);
                    }
                    1 if !buf.is_empty() => {
                        let j = rng.below(buf.len());
                        buf.remove(j);
                    }
                    2 => {
                        let j = rng.below(buf.len() + 1);
                        buf.insert(j, rng.below(256) as u8);
                    }
                    3 => {
                        let tok = interesting[rng.below(interesting.len())];
                        let j = rng.below(buf.len() + 1);
                        buf.splice(j..j, tok.iter().copied());
                    }
                    4 if !buf.is_empty() => {
                        let j = rng.below(buf.len());
                        buf[j] = interesting[rng.below(interesting.len())][0];
                    }
                    _ if !buf.is_empty() => {
                        let n = rng.below(buf.len());
                        buf.truncate(n);
                    }
                    _ => {}
                }
            }
            // Exactly what `subparse` does with a Block payload.
            let s = String::from_utf8_lossy(&buf);
            let _ = parse_srt(&s);
            let _ = parse_vtt(&s);
            let _ = parse_ass(&s);
            let _ = cue_body_srt(&s);
            let _ = cue_body_vtt(&s);
            let _ = cue_body_ass_dialogue(&s);
        }
    }

    /// CRLF line endings (what a Windows-authored `.srt` actually ships) parse identically to LF.
    #[test]
    fn crlf_documents_parse_like_lf() {
        let lf = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:04,000 --> 00:00:05,000\nBye\n";
        let crlf = lf.replace('\n', "\r\n");
        assert_eq!(parse_srt(&crlf), parse_srt(lf), "CRLF and LF agree");
        assert_eq!(parse_srt(&crlf).len(), 2);
    }
}
