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

use streamcraft_core::time::Timestamp;

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
fn ass_time(s: &str) -> Option<u64> {
    let (hms, cc) = s.trim().split_once('.')?;
    // Centiseconds: two digits → milliseconds = cc * 10.
    let cc: u64 = cc.trim().parse().ok()?;
    hms_to_ns(hms, cc * 10)
}

/// Strip ASS override blocks (`{\...}`) and convert `\N`/`\n` to newlines and `\h` to a space
/// (Aegisub/libass text conventions). Everything else is literal.
fn strip_ass(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    let mut depth = 0u32; // inside a `{...}` override block
    while let Some(c) = chars.next() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth > 0 => {} // swallow override-block content
            '\\' => match chars.peek() {
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
/// `MM:SS`); minutes/seconds are required. `None` on any non-numeric field.
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
    Some(((h * 3600 + m * 60 + s) * 1000 + ms) * 1_000_000)
}

/// Drop every `<...>` run, keeping the text between (HTML-ish / WebVTT inline tags). An
/// unbalanced `<` with no `>` swallows to end-of-string (a truncated tag reveals no text).
fn strip_angle_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if in_tag => {}
            _ => out.push(c),
        }
    }
    out
}

/// Case-insensitive `strip_prefix` for the ASS `Format:` / `Dialogue:` keywords.
fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix) {
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
}
