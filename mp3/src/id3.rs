//! ID3v2 / ID3v1 / APEv2 tag parsing — pure `&[u8]` in, [`TagSink`] out.
//!
//! MP3 has no container of its own, so its metadata is bolted onto the elementary stream
//! from both ends: an **ID3v2** tag prepended to the file, an **ID3v1** 128-byte block
//! appended to it, and (for `mp3gain`-style ReplayGain) an **APEv2** tag sitting just
//! before that. All three are byte-aligned, self-delimiting side-cars that never touch
//! the audio, so this module is a set of free functions over a `&[u8]` window: no IO, no
//! pipeline types, no element state. The same code serves a standalone tag scanner and
//! (later) [`Mp3Dec`](crate::Mp3Dec).
//!
//! **Zero heap.** Values are handed to the sink either *borrowed* from the caller's tag
//! bytes or written into the caller's [`Arena`] — nothing here allocates (spec:
//! performance #1). The arena is only touched when the format forces a rewrite:
//! ISO-8859-1 → UTF-8 transcoding, UTF-16 decoding, and de-unsynchronisation. The caller
//! owns the arena's reset point; a whole tag's worth of rewrites lives at once.
//!
//! **Precedence.** A file may carry all three tag kinds. Callers emit **v2 first, then
//! APE, then v1** into one sink: [`TagList::get`](profluens_core::event::TagList::get)
//! returns the *first* value for a key, so the richer, unambiguously-encoded ID3v2 text
//! wins over the 30-byte Latin-1 truncations ID3v1 can offer, and the APEv2 ReplayGain
//! items (which ID3v2 usually lacks) still land.
//!
//! Parsing is **best-effort and total**: a malformed, truncated or hostile tag stops the
//! walk and emits what it understood — no panic, no error type (spec: Supervision — a
//! bad side-car must never take the stream down).
//!
//! ## Specifications
//!
//! ID3v2 is an informal standard published at <https://id3.org>; section numbers below
//! cite `id3v2.4.0-structure` / `id3v2.4.0-frames` (v2.4), `id3v2.3.0` (v2.3) and
//! `id3v2-00` (v2.2). ID3v1 is Eric Kemp's original 1996 layout plus Michael Mutschler's
//! v1.1 track-number amendment and the Winamp genre extension. APEv2 is specified on the
//! Hydrogenaudio wiki (<https://wiki.hydrogenaud.io/index.php?title=APE_key>,
//! `APEv2_specification`), the format's de-facto home.

use profluens_core::event::TagSink;
use profluens_core::memory::Arena;

/// Length of an ID3v2 header — and of the identical optional v2.4 footer (§3.1, §3.4).
const V2_HEADER_LEN: usize = 10;

// The canonical Vorbis-comment key vocabulary `TagSink` documents. Kept as constants so
// the frame table below reads as a mapping rather than a wall of string literals.
const TITLE: &str = "TITLE";
const ARTIST: &str = "ARTIST";
const ALBUM: &str = "ALBUM";
const ALBUMARTIST: &str = "ALBUMARTIST";
const TRACKNUMBER: &str = "TRACKNUMBER";
const DISCNUMBER: &str = "DISCNUMBER";
const GENRE: &str = "GENRE";
const DATE: &str = "DATE";
const COMPOSER: &str = "COMPOSER";
const COMMENT: &str = "COMMENT";

// ---------------------------------------------------------------------------------------
// ID3v2
// ---------------------------------------------------------------------------------------

/// Total byte length of the ID3v2 tag `prefix` starts with, or `None` if it does not start
/// with one (or the 10-byte header is not all there).
///
/// ID3v2 header (id3v2.4.0-structure §3.1): `"ID3"`, a major and a revision version byte
/// (neither may be `$FF`), one flags byte, then a 4-byte **synchsafe** size covering the
/// tag body — extended header, frames and padding — but *not* the header itself. Flags
/// bit 4 marks a 10-byte footer (§3.4), a header copy placed after the padding that the
/// size does not cover either, so it adds another `V2_HEADER_LEN`.
///
/// This is the seek-past-the-tag primitive: the returned length is exactly how many bytes
/// to skip to reach the first audio byte. The synchsafe encoding is validated (§6.2: the
/// high bit of every size byte is zero), so a chance `"ID3"` run inside audio data whose
/// following bytes are not synchsafe is rejected rather than skipped over.
pub fn v2_total_len(prefix: &[u8]) -> Option<usize> {
    let hdr = prefix.get(..V2_HEADER_LEN)?;
    if &hdr[..3] != b"ID3" {
        return None;
    }
    // §3.1: "the version ... will never be $FF" — for either byte.
    if hdr[3] == 0xFF || hdr[4] == 0xFF {
        return None;
    }
    let size = synchsafe(&hdr[6..10])?;
    let footer = if hdr[5] & 0x10 != 0 { V2_HEADER_LEN } else { 0 };
    Some(V2_HEADER_LEN + size as usize + footer)
}

/// Parse a whole ID3v2 tag — `tag` includes the 10-byte header, ideally sized by
/// [`v2_total_len`] — emitting its text frames and attached pictures into `sink`.
///
/// Handles v2.2, v2.3 and v2.4, which differ in three structural ways this walks over:
///
/// * **Frame header** — v2.2 (§3.2 of `id3v2-00`) uses 3-character IDs and a 3-byte size
///   with no flags; v2.3/v2.4 use 4-character IDs, a 4-byte size and 2 flag bytes. The
///   v2.4 size is *synchsafe* (id3v2.4.0-structure §4), v2.3's is a plain 32-bit
///   big-endian count (id3v2.3.0 §3.3) — reading one as the other is the classic ID3
///   parser bug, so the two are kept apart explicitly.
/// * **Extended header** — optional in v2.3 (§3.2: a size that *excludes* its own four
///   bytes) and v2.4 (§3.2: a synchsafe size that *includes* them). Skipped either way.
/// * **Unsynchronisation** (§6.1) — v2.2/v2.3 apply it to the whole tag body, v2.4 moved
///   it per frame. Both are reversed into `scratch` before the affected bytes are read,
///   so an `$FF $00` pair inside embedded cover art round-trips byte-exactly.
///
/// Frames that cannot be read as-is — compressed (zlib), encrypted, or carrying a group
/// identifier — are skipped, as are unknown IDs; the walk stops at the first padding byte
/// (a `$00` frame ID, §3.1). Nothing here fails: a truncated or corrupt tag simply ends
/// the walk.
pub fn parse_v2(tag: &[u8], scratch: &Arena, sink: &mut impl TagSink) {
    let Some(hdr) = tag.get(..V2_HEADER_LEN) else { return };
    if &hdr[..3] != b"ID3" {
        return;
    }
    let major = hdr[3];
    // v2.0/v2.1 predate the frame model and a v2.5+ layout does not exist yet; §3.1 tells
    // a reader to ignore a tag whose major version it does not know.
    if !matches!(major, 2..=4) {
        return;
    }
    let flags = hdr[5];
    let Some(size) = synchsafe(&hdr[6..10]) else { return };
    // Clamp the body to what is actually present: a caller may hand us a short prefix, and
    // a hostile tag may advertise a size far past EOF.
    let end = V2_HEADER_LEN.saturating_add(size as usize).min(tag.len());
    let mut body = &tag[V2_HEADER_LEN..end];

    // v2.2 §3.1 reuses flags bit 6 for whole-tag compression, and states that since no
    // compression scheme was ever defined "the ID3 decoder should just ignore the entire
    // tag if the compression bit is set".
    if major == 2 && flags & 0x40 != 0 {
        return;
    }
    let tag_unsync = flags & 0x80 != 0;
    // §6.1: in v2.2/v2.3 unsynchronisation covers the entire tag body (extended header
    // included), so reverse it once, before anything is located inside. v2.4 moved the
    // scheme to the individual frame (§4.1.2) and the tag-level bit there only asserts
    // that *every* frame is unsynchronised — folded into the per-frame check below.
    if tag_unsync && major < 4 {
        body = deunsync(body, scratch);
    }
    // Extended header (v2.3 §3.2 / v2.4 §3.2), flagged by bit 6. It carries CRC and
    // restriction data we do not use, so it is only measured and stepped over.
    if major >= 3 && flags & 0x40 != 0 {
        let Some(sz) = body.get(..4) else { return };
        let skip = if major == 4 {
            // v2.4's extended-header size is synchsafe and counts its own four bytes.
            match synchsafe(sz) {
                Some(n) => n as usize,
                None => return,
            }
        } else {
            // v2.3's is a plain 32-bit size that excludes the four size bytes themselves.
            (be32(sz) as usize).saturating_add(4)
        };
        let Some(rest) = body.get(skip..) else { return };
        body = rest;
    }

    // Frame header geometry, per version.
    let (id_len, size_len, flag_len) = if major == 2 { (3, 3, 0) } else { (4, 4, 2) };
    let frame_hdr = id_len + size_len + flag_len;
    let mut p = 0usize;
    while p + frame_hdr <= body.len() {
        let id = &body[p..p + id_len];
        // §3.1: the space between the last frame and the end of the tag is zero padding,
        // and a frame ID never starts with `$00` — so this is the end of the frames.
        if id[0] == 0 {
            break;
        }
        let sz_bytes = &body[p + id_len..p + id_len + size_len];
        let size = match major {
            4 => match synchsafe(sz_bytes) {
                Some(n) => n as usize,
                // A non-synchsafe v2.4 frame size means the frame boundaries are no longer
                // derivable; everything after it is unreadable, so stop rather than guess.
                None => break,
            },
            3 => be32(sz_bytes) as usize,
            _ => be24(sz_bytes) as usize,
        };
        let fflags = if flag_len == 2 { [body[p + 8], body[p + 9]] } else { [0, 0] };
        let start = p + frame_hdr;
        let Some(mut data) = start.checked_add(size).and_then(|e| body.get(start..e)) else {
            break; // size runs past the tag — truncated or corrupt
        };
        p = start + size;

        // Frame format flags: v2.3 §3.3.1 packs compression/encryption/grouping into the
        // top three bits of the second flag byte; v2.4 §4.1 re-laid them out and added
        // per-frame unsynchronisation and a data-length indicator.
        let (compressed, encrypted, grouped, frame_unsync, has_dli) = if major == 4 {
            (
                fflags[1] & 0x08 != 0,
                fflags[1] & 0x04 != 0,
                fflags[1] & 0x40 != 0,
                fflags[1] & 0x02 != 0,
                fflags[1] & 0x01 != 0,
            )
        } else {
            (fflags[1] & 0x80 != 0, fflags[1] & 0x40 != 0, fflags[1] & 0x20 != 0, false, false)
        };
        // zlib-compressed and encrypted frames need machinery we deliberately do not carry
        // (and grouped frames prepend a group byte no real-world tagger writes); all three
        // are vanishingly rare and skipping one costs only that frame's tag.
        if compressed || encrypted || grouped {
            continue;
        }
        if major == 4 && (frame_unsync || tag_unsync) {
            data = deunsync(data, scratch);
        }
        if has_dli {
            // §4.1.2: a 4-byte synchsafe data-length indicator prefixes the frame data.
            // (It is synchsafe, so unsynchronisation never touched it — order is moot.)
            let Some(rest) = data.get(4..) else { continue };
            data = rest;
        }

        match frame_kind(id) {
            // §4.2: multiple values in one text frame, separated by the string terminator,
            // are a v2.4 addition — earlier versions carry exactly one.
            Some(Kind::Text(key)) => text_frame(key, data, major == 4, scratch, sink),
            Some(Kind::Comment) => comment_frame(data, scratch, sink),
            Some(Kind::UserText) => user_text_frame(data, major == 4, scratch, sink),
            Some(Kind::Picture) => picture_frame(data, id_len == 3, sink),
            None => {}
        }
    }
}

/// What a recognised frame ID means.
enum Kind {
    /// A plain text-information frame carrying this canonical key (§4.2).
    Text(&'static str),
    /// `COMM` / `COM` — a comment with a language and a short description (§4.10).
    Comment,
    /// `TXXX` / `TXX` — a user-defined text frame; the description *is* the key (§4.2.6).
    /// This is how ReplayGain reaches an ID3v2 tag.
    UserText,
    /// `APIC` / `PIC` — an attached picture (§4.14 / `id3v2-00` §4.15).
    Picture,
}

/// The text-information frames we map, as `(frame ID, canonical key)`. v2.3/v2.4's
/// 4-character IDs and v2.2's 3-character ones share the table — their lengths differ, so
/// a lookup can never confuse the two.
static TEXT_FRAMES: [(&[u8], &str); 19] = [
    (b"TIT2", TITLE),
    (b"TT2", TITLE),
    (b"TPE1", ARTIST),
    (b"TP1", ARTIST),
    (b"TALB", ALBUM),
    (b"TAL", ALBUM),
    (b"TPE2", ALBUMARTIST),
    (b"TP2", ALBUMARTIST),
    (b"TRCK", TRACKNUMBER),
    (b"TRK", TRACKNUMBER),
    (b"TPOS", DISCNUMBER),
    (b"TPA", DISCNUMBER),
    // §4.2.3 `TCON` may carry the legacy `(17)`-style genre-index references. Real-world
    // taggers overwhelmingly write the plain name, and re-expanding the numeric form would
    // silently rewrite a user's text, so the value passes through raw.
    (b"TCON", GENRE),
    (b"TCO", GENRE),
    // v2.4 replaced the year-only `TYER` with the ISO-8601 `TDRC` (§4.2.5); both land on
    // DATE, and a file carrying both simply emits DATE twice (first wins on lookup).
    (b"TDRC", DATE),
    (b"TYER", DATE),
    (b"TYE", DATE),
    (b"TCOM", COMPOSER),
    (b"TCM", COMPOSER),
];

fn frame_kind(id: &[u8]) -> Option<Kind> {
    if id == b"COMM" || id == b"COM" {
        return Some(Kind::Comment);
    }
    if id == b"TXXX" || id == b"TXX" {
        return Some(Kind::UserText);
    }
    if id == b"APIC" || id == b"PIC" {
        return Some(Kind::Picture);
    }
    TEXT_FRAMES.iter().find(|(f, _)| *f == id).map(|&(_, key)| Kind::Text(key))
}

/// A text-information frame (§4.2): one encoding byte, then the text.
fn text_frame(key: &str, data: &[u8], multi: bool, scratch: &Arena, sink: &mut impl TagSink) {
    let Some((&enc, text)) = data.split_first() else { return };
    emit_values(key, enc, text, multi, scratch, sink);
}

/// Emit `rest` as one value (or, with `multi`, as terminator-separated values — the v2.4
/// multi-value rule, §4.2) under `key`. Empty values are dropped.
fn emit_values(
    key: &str,
    enc: u8,
    mut rest: &[u8],
    multi: bool,
    scratch: &Arena,
    sink: &mut impl TagSink,
) {
    loop {
        let (value, after) = split_terminated(rest, enc);
        if !value.is_empty() {
            if let Some(s) = decode_text(enc, value, scratch) {
                if !s.is_empty() {
                    sink.text(key, s);
                }
            }
        }
        // `after` is strictly shorter than `rest` whenever a terminator was found, and
        // empty when none was — so this always terminates.
        if !multi || after.is_empty() {
            break;
        }
        rest = after;
    }
}

/// `COMM` / `COM` (§4.10): encoding byte, a 3-byte ISO-639-2 language code, a
/// terminator-delimited short description, then the comment text.
fn comment_frame(data: &[u8], scratch: &Arena, sink: &mut impl TagSink) {
    let Some(&enc) = data.first() else { return };
    let Some(rest) = data.get(4..) else { return }; // skip the encoding byte + language
    let (_description, text) = split_terminated(rest, enc);
    if let Some(s) = decode_text(enc, text, scratch) {
        if !s.is_empty() {
            sink.text(COMMENT, s);
        }
    }
}

/// `TXXX` / `TXX` (§4.2.6): encoding byte, a terminator-delimited description, then the
/// value. The description is the key — uppercased, this is exactly how
/// `REPLAYGAIN_TRACK_GAIN` and friends arrive in an ID3v2 tag.
fn user_text_frame(data: &[u8], multi: bool, scratch: &Arena, sink: &mut impl TagSink) {
    let Some((&enc, rest)) = data.split_first() else { return };
    let (description, value) = split_terminated(rest, enc);
    let Some(description) = decode_text(enc, description, scratch) else { return };
    let Some(key) = upper_ascii(description, scratch) else { return };
    if key.is_empty() {
        return;
    }
    emit_values(key, enc, value, multi, scratch, sink);
}

/// `APIC` (§4.14) / `PIC` (`id3v2-00` §4.15): encoding byte, the image type, a picture-type
/// byte, a terminator-delimited description, then the raw image bytes.
///
/// The image type is where the two versions diverge: v2.3/v2.4 write a NUL-terminated
/// ISO-8859-1 MIME string, v2.2 writes a bare 3-character format code that has to be
/// mapped to one.
///
/// No arena: the MIME type is ASCII and the image bytes are handed over borrowed (the
/// caller copies if it retains them — that is [`TagSink::picture`]'s contract). A picture
/// under unsynchronisation was already rewritten by the frame walk.
fn picture_frame(data: &[u8], v22: bool, sink: &mut impl TagSink) {
    let Some((&enc, rest)) = data.split_first() else { return };
    let (mime, rest) = if v22 {
        let Some(format) = rest.get(..3) else { return };
        let mime = if format.eq_ignore_ascii_case(b"JPG") {
            "image/jpeg"
        } else if format.eq_ignore_ascii_case(b"PNG") {
            "image/png"
        } else {
            return; // `-->` (a URL link) and anything else: no image bytes to hand over
        };
        (mime, &rest[3..])
    } else {
        let (mime, rest) = split_terminated(rest, 0);
        // MIME types are ASCII, so a UTF-8 read of the ISO-8859-1 field is exact and a
        // non-ASCII byte means the field is not a MIME type at all.
        let Ok(mime) = std::str::from_utf8(mime) else { return };
        // §4.14: the MIME type `-->` means the frame holds a URL, not image bytes.
        if mime == "-->" {
            return;
        }
        (mime, rest)
    };
    let Some(rest) = rest.get(1..) else { return }; // picture type byte
    let (_description, image) = split_terminated(rest, enc);
    if image.is_empty() {
        return;
    }
    sink.picture(mime, image);
}

/// Reverse the unsynchronisation scheme (id3v2.4.0-structure §6.1) into `scratch`: the
/// encoder inserts a `$00` after every `$FF` that would otherwise start a false frame
/// sync, so the decoder drops the `$00` of every `$FF $00` pair. The result is never
/// longer than the input.
fn deunsync<'a>(src: &[u8], scratch: &'a Arena) -> &'a [u8] {
    let out = scratch.alloc_bytes(src.len());
    let mut n = 0;
    let mut i = 0;
    while i < src.len() {
        out[n] = src[i];
        n += 1;
        // `$FF $00` → `$FF`. Consuming both bytes is what makes `$FF $00 $00` (an encoded
        // `$FF $00`) come back as `$FF $00` rather than collapsing further.
        i += if src[i] == 0xFF && src.get(i + 1) == Some(&0x00) { 2 } else { 1 };
    }
    let out: &'a [u8] = out;
    &out[..n]
}

/// Decode an ID3v2 string in the encoding named by its `$00`..`$03` selector
/// (id3v2.4.0-structure §4), borrowing from `raw` where the bytes are already UTF-8 and
/// rewriting into `scratch` where they are not. `None` for an unknown selector or invalid
/// UTF-8.
fn decode_text<'a>(enc: u8, raw: &'a [u8], scratch: &'a Arena) -> Option<&'a str> {
    match enc {
        // $00 — ISO-8859-1.
        0 => latin1(raw, scratch),
        // $01 — UTF-16 with a mandatory BOM. RFC 2781 §3.2: `FF FE` is little-endian,
        // `FE FF` big-endian; §4.3 makes big-endian the default when the BOM is missing
        // (which some taggers do, so it must not be a parse failure).
        1 => {
            let (big_endian, body) = match raw.get(..2) {
                Some([0xFF, 0xFE]) => (false, &raw[2..]),
                Some([0xFE, 0xFF]) => (true, &raw[2..]),
                _ => (true, raw),
            };
            Some(utf16(body, big_endian, scratch))
        }
        // $02 — UTF-16BE without a BOM (v2.4 only).
        2 => Some(utf16(raw, true, scratch)),
        // $03 — UTF-8 (v2.4 only): validate and borrow.
        3 => std::str::from_utf8(raw).ok(),
        _ => None,
    }
}

/// ISO-8859-1 → UTF-8. The encoding maps 1:1 onto U+0000..U+00FF, so a pure-ASCII field is
/// already UTF-8 and is borrowed; anything else is transcoded into `scratch`, where each
/// byte costs at most the 2 UTF-8 bytes of U+0080..U+00FF (Unicode §3.9, table 3-6).
fn latin1<'a>(raw: &'a [u8], scratch: &'a Arena) -> Option<&'a str> {
    if raw.is_ascii() {
        return std::str::from_utf8(raw).ok();
    }
    let out = scratch.alloc_bytes(raw.len() * 2);
    let mut n = 0;
    for &b in raw {
        n += char::from(b).encode_utf8(&mut out[n..]).len();
    }
    let out: &'a [u8] = out;
    std::str::from_utf8(&out[..n]).ok()
}

/// UTF-16 → UTF-8, into `scratch`. Hand-rolled because the standard library has no
/// byte-slice UTF-16 decoder and the parse path may not allocate.
///
/// Surrogate pairing per RFC 2781 §2.2 / Unicode §3.8: a high surrogate (U+D800..U+DBFF)
/// followed by a low one (U+DC00..U+DFFF) combines to
/// `0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)`. An unpaired surrogate — a lone high
/// at the end of the field, a low without a high — becomes U+FFFD rather than failing the
/// whole tag (Unicode §3.9, U+FFFD substitution), and a trailing odd byte is dropped.
fn utf16<'a>(raw: &[u8], big_endian: bool, scratch: &'a Arena) -> &'a str {
    let unit = |i: usize| -> u16 {
        let (a, b) = (raw[i], raw[i + 1]);
        if big_endian { u16::from_be_bytes([a, b]) } else { u16::from_le_bytes([a, b]) }
    };
    // A BMP scalar is the worst case per code unit at 3 UTF-8 bytes; a surrogate pair
    // spends two units on 4 bytes, so 3 bytes/unit bounds both. The `+ 4` is slack.
    let out = scratch.alloc_bytes((raw.len() / 2) * 3 + 4);
    let mut n = 0;
    let mut i = 0;
    while i + 1 < raw.len() {
        let u = unit(i);
        i += 2;
        let cp = if (0xD800..0xDC00).contains(&u) {
            let lo = if i + 1 < raw.len() { unit(i) } else { 0 };
            if (0xDC00..0xE000).contains(&lo) {
                i += 2;
                0x1_0000u32 + ((u32::from(u) - 0xD800) << 10) + (u32::from(lo) - 0xDC00)
            } else {
                0xFFFD
            }
        } else if (0xDC00..0xE000).contains(&u) {
            0xFFFD
        } else {
            u32::from(u)
        };
        let c = char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER);
        n += c.encode_utf8(&mut out[n..]).len();
    }
    let out: &'a [u8] = out;
    // Valid by construction (every byte came from `char::encode_utf8`); the check is free
    // insurance against a future edit and keeps the function panic-free.
    std::str::from_utf8(&out[..n]).unwrap_or("")
}

/// Split `data` at its first string terminator, returning `(before, after)` and
/// `(data, "")` when there is none. Per id3v2.4.0-structure §4 the terminator is one `$00`
/// for the single-byte encodings and `$00 $00`, code-unit aligned, for the UTF-16 ones.
fn split_terminated(data: &[u8], enc: u8) -> (&[u8], &[u8]) {
    if enc == 1 || enc == 2 {
        let mut i = 0;
        while i + 1 < data.len() {
            if data[i] == 0 && data[i + 1] == 0 {
                return (&data[..i], &data[i + 2..]);
            }
            i += 2;
        }
        (data, &data[data.len()..])
    } else {
        match data.iter().position(|&b| b == 0) {
            Some(i) => (&data[..i], &data[i + 1..]),
            None => (data, &data[data.len()..]),
        }
    }
}

/// ASCII-uppercase `s` into `scratch`. ASCII case folding never changes a byte's length or
/// leaves the ASCII range, so the copy stays valid UTF-8 whatever `s` held.
fn upper_ascii<'a>(s: &str, scratch: &'a Arena) -> Option<&'a str> {
    let out = scratch.alloc_bytes(s.len());
    out.copy_from_slice(s.as_bytes());
    out.make_ascii_uppercase();
    let out: &'a [u8] = out;
    std::str::from_utf8(out).ok()
}

/// Read a 4-byte synchsafe integer (id3v2.4.0-structure §6.2): 7 bits per byte, high bit
/// always clear, so no synchsafe value can ever contain a false MPEG frame sync. `None` if
/// any high bit is set — which means the field is not synchsafe and must not be trusted.
fn synchsafe(b: &[u8]) -> Option<u32> {
    let b = b.get(..4)?;
    if b.iter().any(|&x| x & 0x80 != 0) {
        return None;
    }
    Some(
        (u32::from(b[0]) << 21)
            | (u32::from(b[1]) << 14)
            | (u32::from(b[2]) << 7)
            | u32::from(b[3]),
    )
}

/// Big-endian `u32`, 0 on a short slice (total by design — the parse path never panics).
fn be32(b: &[u8]) -> u32 {
    b.get(..4).map_or(0, |s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

/// Big-endian 24-bit size — the ID3v2.2 frame size field (`id3v2-00` §3.2).
fn be24(b: &[u8]) -> u32 {
    b.get(..3).map_or(0, |s| u32::from_be_bytes([0, s[0], s[1], s[2]]))
}

// ---------------------------------------------------------------------------------------
// ID3v1
// ---------------------------------------------------------------------------------------

/// Length of an ID3v1 tag: the last 128 bytes of the file.
const V1_LEN: usize = 128;

/// Parse the trailing ID3v1 tag out of `tail` — the file's **last 128 bytes** (a longer
/// slice is accepted; only its last 128 bytes are read) — into `sink`.
///
/// The 1996 layout (Eric Kemp's original `id3v1` note) is a flat 128-byte record: `"TAG"`,
/// then 30-byte title, artist and album fields, a 4-byte year, a 30-byte comment and a
/// single genre index. Fields are fixed-width ISO-8859-1 padded with NULs or spaces, so
/// each is cut at its first NUL and right-trimmed before emission; an empty field is not
/// emitted at all.
///
/// **ID3v1.1** (Michael Mutschler's amendment) steals the last two comment bytes: a `$00`
/// at `comment[28]` with a non-zero `comment[29]` means that last byte is the track
/// number, and the comment is only 28 bytes long.
///
/// Emit order matters to the caller, not to this function: run [`parse_v2`] (and
/// [`parse_ape`]) into the same sink **first**, because
/// [`TagList::get`](profluens_core::event::TagList::get) returns the first value for a key
/// and ID3v1's 30-byte Latin-1 truncations should never shadow a full ID3v2 string.
pub fn parse_v1(tail: &[u8], scratch: &Arena, sink: &mut impl TagSink) {
    if tail.len() < V1_LEN {
        return;
    }
    let t = &tail[tail.len() - V1_LEN..];
    if &t[..3] != b"TAG" {
        return;
    }
    if let Some(s) = v1_field(&t[3..33], scratch) {
        sink.text(TITLE, s);
    }
    if let Some(s) = v1_field(&t[33..63], scratch) {
        sink.text(ARTIST, s);
    }
    if let Some(s) = v1_field(&t[63..93], scratch) {
        sink.text(ALBUM, s);
    }
    if let Some(s) = v1_field(&t[93..97], scratch) {
        sink.text(DATE, s);
    }
    // The v1.1 track-number probe: comment[28] == $00 && comment[29] != $00.
    let (comment, track) =
        if t[125] == 0 && t[126] != 0 { (&t[97..125], Some(t[126])) } else { (&t[97..127], None) };
    if let Some(s) = v1_field(comment, scratch) {
        sink.text(COMMENT, s);
    }
    if let Some(n) = track {
        let mut buf = [0u8; 3];
        sink.text(TRACKNUMBER, decimal(n, &mut buf));
    }
    // The genre byte indexes the table below; $FF (and any index past it) means "unset".
    if let Some(genre) = GENRES.get(t[127] as usize) {
        sink.text(GENRE, genre);
    }
}

/// One fixed-width ID3v1 field: cut at the first NUL (taggers pad with NULs and some leave
/// junk behind them), right-trim the space padding, then transcode from ISO-8859-1.
/// `None` for an empty field.
fn v1_field<'a>(raw: &'a [u8], scratch: &'a Arena) -> Option<&'a str> {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    let field = &raw[..end];
    // The last non-space byte; `None` means the field is empty or all padding.
    let last = field.iter().rposition(|&b| b != b' ')?;
    latin1(&field[..=last], scratch)
}

/// Decimal-render `n` into a stack buffer — a `format!`-free `u8` → `&str` (spec:
/// allocation discipline; the parse path holds no heap strings).
fn decimal(n: u8, buf: &mut [u8; 3]) -> &str {
    let mut i = buf.len();
    let mut v = n;
    loop {
        i -= 1;
        buf[i] = b'0' + v % 10;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    std::str::from_utf8(&buf[i..]).unwrap_or("")
}

/// The ID3v1 genre index → name table.
///
/// Indices 0..=79 are the list in Eric Kemp's original ID3v1 note (typos and all —
/// `Psychadelic` is spelled that way in the source, and readers match on the index, not the
/// spelling). Indices 80..=147 are the de-facto Winamp extension, which every ID3v1 writer
/// since has followed.
///
/// One deliberate substitution: index 133 in the original Winamp list is a racial slur.
/// Later Winamp releases and every modern tagger publish it as `Afro-Punk`, which is what
/// this table emits — the index, which is what is actually stored in the file, is
/// unchanged.
static GENRES: [&str; 148] = [
    "Blues",
    "Classic Rock",
    "Country",
    "Dance",
    "Disco",
    "Funk",
    "Grunge",
    "Hip-Hop",
    "Jazz",
    "Metal",
    "New Age",
    "Oldies",
    "Other",
    "Pop",
    "R&B",
    "Rap",
    "Reggae",
    "Rock",
    "Techno",
    "Industrial",
    "Alternative",
    "Ska",
    "Death Metal",
    "Pranks",
    "Soundtrack",
    "Euro-Techno",
    "Ambient",
    "Trip-Hop",
    "Vocal",
    "Jazz+Funk",
    "Fusion",
    "Trance",
    "Classical",
    "Instrumental",
    "Acid",
    "House",
    "Game",
    "Sound Clip",
    "Gospel",
    "Noise",
    "AlternRock",
    "Bass",
    "Soul",
    "Punk",
    "Space",
    "Meditative",
    "Instrumental Pop",
    "Instrumental Rock",
    "Ethnic",
    "Gothic",
    "Darkwave",
    "Techno-Industrial",
    "Electronic",
    "Pop-Folk",
    "Eurodance",
    "Dream",
    "Southern Rock",
    "Comedy",
    "Cult",
    "Gangsta",
    "Top 40",
    "Christian Rap",
    "Pop/Funk",
    "Jungle",
    "Native American",
    "Cabaret",
    "New Wave",
    "Psychadelic",
    "Rave",
    "Showtunes",
    "Trailer",
    "Lo-Fi",
    "Tribal",
    "Acid Punk",
    "Acid Jazz",
    "Polka",
    "Retro",
    "Musical",
    "Rock & Roll",
    "Hard Rock",
    // --- Winamp extension (80..=147) ---
    "Folk",
    "Folk-Rock",
    "National Folk",
    "Swing",
    "Fast Fusion",
    "Bebob",
    "Latin",
    "Revival",
    "Celtic",
    "Bluegrass",
    "Avantgarde",
    "Gothic Rock",
    "Progressive Rock",
    "Psychedelic Rock",
    "Symphonic Rock",
    "Slow Rock",
    "Big Band",
    "Chorus",
    "Easy Listening",
    "Acoustic",
    "Humour",
    "Speech",
    "Chanson",
    "Opera",
    "Chamber Music",
    "Sonata",
    "Symphony",
    "Booty Bass",
    "Primus",
    "Porn Groove",
    "Satire",
    "Slow Jam",
    "Club",
    "Tango",
    "Samba",
    "Folklore",
    "Ballad",
    "Power Ballad",
    "Rhythmic Soul",
    "Freestyle",
    "Duet",
    "Punk Rock",
    "Drum Solo",
    "A capella",
    "Euro-House",
    "Dance Hall",
    "Goa",
    "Drum & Bass",
    "Club-House",
    "Hardcore",
    "Terror",
    "Indie",
    "BritPop",
    "Afro-Punk",
    "Polsk Punk",
    "Beat",
    "Christian Gangsta Rap",
    "Heavy Metal",
    "Black Metal",
    "Crossover",
    "Contemporary Christian",
    "Christian Rock",
    "Merengue",
    "Salsa",
    "Thrash Metal",
    "Anime",
    "Jpop",
    "Synthpop",
];

// ---------------------------------------------------------------------------------------
// APEv2
// ---------------------------------------------------------------------------------------

/// Length of an APEv2 header or footer — the two share one 32-byte layout.
const APE_FOOTER_LEN: usize = 32;
const APE_PREAMBLE: &[u8; 8] = b"APETAGEX";
/// APEv2 limits a key to 255 characters, so this stack buffer can hold any legal key and
/// the uppercasing never needs the heap or an arena.
const APE_KEY_MAX: usize = 255;

/// Total byte length of the APEv2 tail whose **footer** occupies the last 32 bytes of
/// `last32`, or `None` if that is not an APE footer.
///
/// The APEv2 footer (`APEv2_specification`) is `"APETAGEX"`, a `u32` version (1000 for
/// APEv1, 2000 for APEv2), a `u32` tag size, a `u32` item count, a `u32` flags word and 8
/// reserved bytes, all little-endian. **The tag size covers the items plus the footer but
/// never the optional header**, so a header (flags bit 31) adds another 32 bytes in front.
/// Flags bit 29 distinguishes a header from a footer; finding a header at the end means
/// this is not the tag's tail.
///
/// APE tags live at EOF but are conventionally placed *before* a trailing ID3v1 block, so
/// a caller strips ID3v1 first and passes the 32 bytes ending where the audio's trailer
/// ends. The returned length counts back from that same point.
pub fn ape_tail_len(last32: &[u8]) -> Option<usize> {
    let f = last32.get(last32.len().checked_sub(APE_FOOTER_LEN)?..)?;
    if &f[..8] != APE_PREAMBLE {
        return None;
    }
    let version = le32(&f[8..12]);
    if version != 1000 && version != 2000 {
        return None;
    }
    let size = le32(&f[12..16]) as usize;
    // The footer's own 32 bytes are inside `size`; anything smaller is not a tag.
    if size < APE_FOOTER_LEN {
        return None;
    }
    // Preamble 8, version 4, size 4, item count 4, flags 4, reserved 8 — flags at 20.
    let flags = le32(&f[20..24]);
    if flags & (1 << 29) != 0 {
        return None; // this block is a *header*, so we are not at the tag's end
    }
    let header = if flags & (1 << 31) != 0 { APE_FOOTER_LEN } else { 0 };
    Some(size + header)
}

/// Parse an APEv2 tag — `tag` is the whole block [`ape_tail_len`] measured, optional
/// 32-byte header included — into `sink`.
///
/// Items sit between the optional header and the mandatory footer, each one a `u32`
/// little-endian value length, a `u32` little-endian item-flags word, a NUL-terminated
/// ASCII key and the value bytes. Only **text** items are tags: flag bits 1-2 hold the
/// value type (0 = UTF-8 text, 1 = binary, 2 = external locator, 3 = reserved), so binary
/// blobs and URLs are skipped. Text values may hold several NUL-separated values, which
/// emit as repeated keys.
///
/// Keys are ASCII by specification and are uppercased in place on the stack — no arena is
/// needed. The handful of APE key names that differ from the canonical Vorbis-comment
/// vocabulary [`TagSink`] documents are mapped (`Year` → `DATE`, `Track` → `TRACKNUMBER`,
/// `Disc` → `DISCNUMBER`, `Album Artist` → `ALBUMARTIST`); everything else,
/// `replaygain_track_gain` included, is already canonical once uppercased. This is where
/// `mp3gain` writes ReplayGain, which is the main reason to read APE tags on an MP3 at all.
pub fn parse_ape(tag: &[u8], sink: &mut impl TagSink) {
    if tag.len() < APE_FOOTER_LEN {
        return;
    }
    // A leading block with the same preamble and flags bit 29 set is the optional header.
    let start = if tag.len() >= 2 * APE_FOOTER_LEN
        && &tag[..8] == APE_PREAMBLE
        && le32(&tag[20..24]) & (1 << 29) != 0
    {
        APE_FOOTER_LEN
    } else {
        0
    };
    let Some(items) = tag.get(start..tag.len() - APE_FOOTER_LEN) else { return };

    let mut key_buf = [0u8; APE_KEY_MAX];
    let mut p = 0usize;
    while p + 8 <= items.len() {
        let value_len = le32(&items[p..p + 4]) as usize;
        let flags = le32(&items[p + 4..p + 8]);
        let rest = &items[p + 8..];
        let Some(key_len) = rest.iter().position(|&b| b == 0) else { break };
        let key = &rest[..key_len];
        let Some(value) =
            (key_len + 1).checked_add(value_len).and_then(|e| rest.get(key_len + 1..e))
        else {
            break; // value runs past the item area — truncated or corrupt
        };
        // Advances by at least 9 every iteration, so the walk always terminates.
        p += 8 + key_len + 1 + value_len;

        if (flags >> 1) & 0x3 != 0 {
            continue; // not a text item
        }
        // The spec restricts keys to printable ASCII ($20..$7E), 2..255 characters.
        let Some(slot) = key_buf.get_mut(..key_len) else { continue };
        if key.is_empty() || !key.iter().all(|&b| (0x20..0x7F).contains(&b)) {
            continue;
        }
        slot.copy_from_slice(key);
        slot.make_ascii_uppercase();
        let Ok(key) = std::str::from_utf8(slot) else { continue };
        let key = ape_alias(key);
        // APEv2 values are UTF-8 and may carry several values separated by a NUL.
        for value in value.split(|&b| b == 0) {
            if value.is_empty() {
                continue;
            }
            if let Ok(s) = std::str::from_utf8(value) {
                sink.text(key, s);
            }
        }
    }
}

/// Map the few APEv2 key names that differ from the canonical Vorbis-comment vocabulary
/// [`TagSink`] specifies (the APE key list on the Hydrogenaudio wiki is its own dialect).
/// Everything else — `Title`, `Artist`, `Album`, `Genre`, `Composer`, `Comment`, the
/// `replaygain_*` family — already matches once uppercased.
fn ape_alias(key: &str) -> &str {
    match key {
        "YEAR" => DATE,
        "TRACK" => TRACKNUMBER,
        "DISC" => DISCNUMBER,
        "ALBUM ARTIST" => ALBUMARTIST,
        _ => key,
    }
}

/// Little-endian `u32`, 0 on a short slice (total by design — the parse path never panics).
fn le32(b: &[u8]) -> u32 {
    b.get(..4).map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
