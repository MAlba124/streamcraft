//! FLAC metadata-block tag extraction: `VORBIS_COMMENT` (§8.6) and `PICTURE` (§8.8), plus the
//! `STREAMINFO` stream properties (§8.2).
//!
//! The metadata-block chain is byte-aligned, so this parses the raw header bytes directly and is
//! independent of the bit-level frame decoder. Both the one-shot [`FlacDecoder`](crate::FlacDecoder)
//! and the streaming [`StreamDecoder`](crate::StreamDecoder) fill a [`FlacTags`] from it, and the
//! [`FlacDec`](crate::FlacDec) element turns that into a `profluens_core::event::TagList` it emits.
//! Parsing is best-effort: malformed or truncated metadata simply stops the walk, never errors.
//!
//! ## Two faces, one walk
//! - [`parse_into`] is the **borrowing** parser: it pushes each tag into a
//!   [`TagSink`] with values and picture bytes borrowed straight out of `bytes`, allocating
//!   nothing — what a scanner or an element with a pool behind it wants.
//! - [`parse`] is the **owning** convenience: the same walk driven by a sink that copies into a
//!   [`FlacTags`]. It is `parse_into` plus a `Vec`, so the two can never drift apart.
//!
//! [`stream_info`] shares the same block walk to read `STREAMINFO`, and [`seek_table`] /
//! [`audio_start`] read the other two things a *player* wants out of the chain: the `SEEKTABLE`
//! index (§8.5) and where the audio frames it points into actually begin.

use profluens_core::event::TagSink;

/// Tags parsed from a FLAC stream's metadata blocks.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct FlacTags {
    /// Vorbis comments as `(uppercased field name, value)` — e.g. `("TITLE", "Enough")`. A field
    /// may repeat (two `ARTIST`s), and order is preserved.
    pub comments: Vec<(String, String)>,
    /// Attached pictures as `(MIME type, image bytes)`, e.g. `("image/jpeg", <jpeg>)`.
    pub pictures: Vec<(String, Vec<u8>)>,
}

impl FlacTags {
    /// No comments and no pictures.
    pub fn is_empty(&self) -> bool {
        self.comments.is_empty() && self.pictures.is_empty()
    }

    /// Total number of comments plus pictures.
    pub fn len(&self) -> usize {
        self.comments.len() + self.pictures.len()
    }
}

/// The stream properties every consumer needs before the first frame (§8.2, Table 3), as
/// plain integers. Read with [`stream_info`].
///
/// This is the *container-level* view: the four fields that describe the audio. The decoder's
/// [`StreamInfo`](crate::StreamInfo) is the full block — block/frame size bounds and the MD5 as
/// well — because it needs them to decode; a probe reading a file header does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamInfo {
    /// Sample rate in Hz (§8.2: a 20-bit field, so up to 1 048 575 — not just the usual rates).
    /// 0 means "non-audio stream", which §8.2 permits but no decoder can play.
    pub sample_rate: u32,
    /// Channel count, 1–8 (§8.2 stores it as `channels - 1` in 3 bits).
    pub channels: u8,
    /// Bits per sample, 4–32 (§8.2 stores it as `bits_per_sample - 1` in 5 bits).
    pub bits_per_sample: u8,
    /// Total interchannel samples in the stream (§8.2: a **36-bit** field — wider than a `u32`,
    /// which is why this is a `u64`; ~13.5 hours at 44.1 kHz overflows 32 bits). 0 means unknown.
    pub total_samples: u64,
}

/// One entry of a `SEEKTABLE` (§8.5) — the index FLAC hands out so a player can seek a
/// variable-block-size stream without walking it.
///
/// A seek point names an audio frame two ways at once: the sample it starts at (hence *when*
/// it plays) and the byte it starts at (hence *where* to resume reading). That pair is the whole
/// point — the same time↔byte map a Matroska `Cues` element or an MP4 `stss` table provides, at
/// whatever density the encoder chose (the reference encoder writes one per second by default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeekPoint {
    /// Sample number of the first sample in the target frame (§8.5). Divide by the
    /// [`StreamInfo::sample_rate`] for the time it plays at.
    pub sample: u64,
    /// Offset in bytes of the target frame's header, **from the first byte of the first frame
    /// header** (§8.5) — i.e. relative to [`audio_start`], not to the start of the file.
    pub byte_offset: u64,
    /// Number of samples in the target frame (§8.5). A player does not need it to seek; it is
    /// here because dropping a field from a parsed record invites re-parsing later.
    pub frame_samples: u16,
}

/// Byte length of the STREAMINFO block body (§8.2: 16+16+24+24+20+3+5+36+128 bits).
const STREAMINFO_LEN: usize = 34;

/// `SEEKTABLE` block type (§8.5, Table 1).
const SEEKTABLE_TYPE: u8 = 3;

/// Byte length of one seek point (§8.5: a 64-bit sample number, a 64-bit offset and a 16-bit
/// frame length).
const SEEK_POINT_LEN: usize = 18;

/// The sample number that marks a **placeholder** point (§8.5: "0xFFFFFFFFFFFFFFFF for a
/// placeholder point"). An encoder writes these to reserve table space it may fill in later, so
/// they are padding, not seek points, and carry no meaningful offset.
const PLACEHOLDER_SAMPLE: u64 = u64::MAX;

/// Longest field name uppercased on the stack by [`parse_into`]; see [`upper_key`].
const KEY_STACK_MAX: usize = 256;

/// Walk a FLAC stream's metadata-block chain (starting at the `fLaC` marker) and extract its tags.
// One-time: a stream's tags are parsed once, at header time — not the per-frame decode path (spec:
// allocation discipline — header parsing is the sanctioned one-time-allocation exception).
#[allow(clippy::disallowed_methods)]
pub fn parse(bytes: &[u8]) -> FlacTags {
    let mut tags = FlacTags::default();
    parse_into(bytes, &mut OwningSink(&mut tags));
    tags
}

/// The zero-copy twin of [`parse`]: walk the metadata-block chain and push every tag into `sink`.
///
/// Field values and picture bytes are handed over **borrowed from `bytes`** — no copy, no
/// allocation, not even for a 12 MiB cover image. Everything the sink receives is valid for the
/// call only; a sink that retains it copies (that is what [`parse`]'s own sink does).
///
/// Field names arrive uppercased, the canonical [`TagSink`] key vocabulary (§8.6: "the evaluation
/// of the field names MUST be case insensitive"). A name that is already uppercase — nearly all of
/// them — is passed through borrowed; one that is not is uppercased into a stack buffer.
///
/// Best-effort, exactly like [`parse`]: a truncated or malformed block ends the walk, a malformed
/// entry within a block is skipped, and no input can cause a panic.
pub fn parse_into(bytes: &[u8], sink: &mut impl TagSink) {
    walk_blocks(bytes, |block_type, block| {
        match block_type {
            4 => vorbis_comment_into(block, sink),
            6 => picture_into(block, sink),
            _ => {}
        }
        true
    });
}

/// Read `STREAMINFO` (§8.2) from a buffer starting at the `fLaC` marker. `None` if the marker or
/// the block is missing, or the block is short.
///
/// §8.2 requires STREAMINFO to be "the first metadata block in the stream", and it is where the
/// [`FlacDecoder`](crate::FlacDecoder) insists on finding it; this reader is deliberately more
/// forgiving — it takes the first STREAMINFO in the chain wherever it sits — because a probe
/// reporting a rate is not a decoder gating on validity.
pub fn stream_info(bytes: &[u8]) -> Option<StreamInfo> {
    let mut found = None;
    walk_blocks(bytes, |block_type, block| {
        if block_type == 0 {
            found = parse_streaminfo(block);
            return false; // one STREAMINFO per stream (§8.2) — stop at it
        }
        true
    });
    found
}

/// Read the `SEEKTABLE` (§8.5) from a buffer starting at the `fLaC` marker, as an iterator of
/// real [`SeekPoint`]s in the order the encoder wrote them — which §8.5 requires to be ascending
/// by sample number, with no two points sharing one.
///
/// **Placeholder points are skipped.** §8.5 lets an encoder pad the table with points whose
/// sample number is `0xFFFFFFFFFFFFFFFF` and whose other fields are undefined; they must sort
/// last, so dropping them leaves the remaining points ascending and every one of them usable.
///
/// Allocates nothing — the points are decoded straight out of `bytes` as the caller pulls them —
/// and never panics: a table whose length is not a whole number of points yields the whole ones
/// and ignores the remainder, and a stream with no SEEKTABLE (or no `fLaC` marker at all) yields
/// an empty iterator. That is the same best-effort contract as the rest of this module: a missing
/// index costs a player its exact seeking, not its playback.
pub fn seek_table(bytes: &[u8]) -> impl Iterator<Item = SeekPoint> + '_ {
    let mut table: &[u8] = &[];
    walk_blocks(bytes, |block_type, block| {
        if block_type == SEEKTABLE_TYPE {
            table = block;
            return false; // §8.5: "There may be only one SEEKTABLE in a stream"
        }
        true
    });
    // `as_chunks` splits off the whole points and leaves the ragged remainder — a table
    // whose length is not a multiple of 18 (which §8.5 forbids, but untrusted bytes do not
    // care) yields the points that are there and ignores the tail.
    table.as_chunks::<SEEK_POINT_LEN>().0.iter().filter_map(|p| parse_seek_point(p))
}

/// Byte offset of the first audio frame — the end of the metadata-block chain (§8.1: the frames
/// follow the block whose last-block flag is set).
///
/// This is the anchor every [`SeekPoint::byte_offset`] is measured from, so a file-absolute seek
/// target is `audio_start + point.byte_offset`. `None` when `bytes` does not hold the whole
/// chain — a header window that stopped inside a large `PICTURE`, say — because a guess here
/// would silently shift every seek in the file.
pub fn audio_start(bytes: &[u8]) -> Option<usize> {
    walk_blocks(bytes, |_, _| true)
}

/// One 18-byte seek point (§8.5), or `None` for a placeholder or a short slice.
fn parse_seek_point(p: &[u8]) -> Option<SeekPoint> {
    let be64 = |at: usize| -> Option<u64> {
        p.get(at..at + 8).and_then(|s| <[u8; 8]>::try_from(s).ok()).map(u64::from_be_bytes)
    };
    let sample = be64(0)?;
    if sample == PLACEHOLDER_SAMPLE {
        return None;
    }
    let byte_offset = be64(8)?;
    let frame_samples =
        p.get(16..18).and_then(|s| <[u8; 2]>::try_from(s).ok()).map(u16::from_be_bytes)?;
    Some(SeekPoint { sample, byte_offset, frame_samples })
}

/// Walk the metadata-block chain from the `fLaC` marker (§8.1), calling `visit(type, body)` for
/// each block until it returns false, the last-block flag is seen, or the chain runs out.
///
/// Each block is a 1-byte header `[last:1 | type:7]`, a 3-byte big-endian body length, then the
/// body. A length that overruns the buffer ends the walk — the shared "truncated metadata simply
/// stops" rule that keeps every entry point here best-effort.
///
/// Returns the offset just past the final block — where the first audio frame begins (§8.1) — but
/// **only** when the chain terminated properly, at a block whose last-block flag was set. `None`
/// for everything else: no `fLaC` marker, a body running past the end of the buffer, or a `visit`
/// that asked to stop early. That is exactly the distinction [`audio_start`] needs; the callers
/// that stop early ignore the value.
fn walk_blocks<'a>(bytes: &'a [u8], mut visit: impl FnMut(u8, &'a [u8]) -> bool) -> Option<usize> {
    if bytes.len() < 4 || &bytes[0..4] != b"fLaC" {
        return None;
    }
    let mut pos = 4;
    while pos + 4 <= bytes.len() {
        let header = bytes[pos];
        let last = header & 0x80 != 0;
        let block_type = header & 0x7f;
        let len = u32::from_be_bytes([0, bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        let start = pos + 4;
        let end = start.checked_add(len)?;
        let Some(block) = bytes.get(start..end) else { break };
        let go_on = visit(block_type, block);
        pos = end;
        if last {
            return Some(pos);
        }
        if !go_on {
            break;
        }
    }
    None
}

/// `VORBIS_COMMENT` body (§8.6): a little-endian-length-prefixed vendor string (skipped), then a
/// count and that many length-prefixed UTF-8 `FIELD=value` entries. Note the little-endian lengths
/// — §8.6 calls this out as the one place FLAC departs from big-endian.
///
/// The vendor string names the encoder, not the content, and has no field name to key it by, so it
/// is skipped. An entry with no `=`, or one that is not UTF-8, is skipped and the walk continues.
fn vorbis_comment_into(block: &[u8], sink: &mut impl TagSink) {
    let le = |b: &[u8], p: usize| {
        b.get(p..p + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
    };
    let mut p = 0;
    let Some(vendor_len) = le(block, p) else { return };
    p += 4 + vendor_len;
    let Some(count) = le(block, p) else { return };
    p += 4;
    let mut key_buf = [0u8; KEY_STACK_MAX];
    for _ in 0..count {
        let Some(clen) = le(block, p) else { break };
        p += 4;
        let Some(s) = block.get(p..p + clen) else { break };
        p += clen;
        if let Ok(s) = std::str::from_utf8(s) {
            // Only the first '=' separates, so a value may itself contain '=' (§8.6).
            if let Some((key, value)) = s.split_once('=') {
                sink.text(upper_key(key, &mut key_buf), value);
            }
        }
    }
}

/// The field name as its canonical uppercase form, without allocating: borrowed unchanged when it
/// holds no ASCII lowercase (the common case — `TITLE`, `REPLAYGAIN_TRACK_GAIN`), otherwise
/// uppercased into `buf`.
///
/// Uppercasing is ASCII-only, which is what §8.6 defines ("U+0041 through U+005A (A-Z) MUST be
/// considered equivalent to U+0061 through U+007A (a-z)") and what keeps a multi-byte UTF-8
/// sequence — illegal in a field name, but this is untrusted input — byte-for-byte intact and
/// still valid UTF-8.
///
/// A name longer than `buf` is passed through with its original case rather than dropped: no
/// tagger produces a 256-byte field name, and losing the entry would be worse than not
/// normalising it. [`parse`]'s owning sink re-uppercases anyway, so `FlacTags` is unaffected.
fn upper_key<'k>(key: &'k str, buf: &'k mut [u8; KEY_STACK_MAX]) -> &'k str {
    let raw = key.as_bytes();
    if raw.len() > buf.len() || !raw.iter().any(u8::is_ascii_lowercase) {
        return key;
    }
    let out = &mut buf[..raw.len()];
    for (dst, &src) in out.iter_mut().zip(raw) {
        *dst = src.to_ascii_uppercase();
    }
    // Only ASCII bytes changed, so `out` is as valid UTF-8 as `key` was; `from_utf8` keeps this
    // module free of `unsafe` (`#![deny(unsafe_code)]`).
    std::str::from_utf8(out).unwrap_or(key)
}

/// `PICTURE` body (§8.8): picture type (4 BE), MIME (len + bytes), description (len + bytes),
/// width/height/depth/colors (4 BE each), then the image data (len + bytes). Emits the MIME type
/// and image bytes borrowed from `block`.
fn picture_into(block: &[u8], sink: &mut impl TagSink) {
    let be = |b: &[u8], p: usize| {
        b.get(p..p + 4).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize)
    };
    let mut p = 4; // skip picture type
    let Some(mime_len) = be(block, p) else { return };
    p += 4;
    let Some(mime) = block.get(p..p + mime_len).and_then(|s| std::str::from_utf8(s).ok()) else {
        return;
    };
    p += mime_len;
    let Some(desc_len) = be(block, p) else { return };
    p += 4 + desc_len; // skip description
    p += 16; // width, height, colour depth, indexed-colour count (4 × 4 bytes)
    let Some(data_len) = be(block, p) else { return };
    p += 4;
    let Some(data) = block.get(p..p + data_len) else { return };
    sink.picture(mime, data);
}

/// The 34-byte STREAMINFO body (§8.2, Table 3). The fields are **bit**-packed, not byte-aligned:
/// after the two block sizes and two frame sizes, `sample_rate` is 20 bits, `channels - 1` is 3,
/// `bits_per_sample - 1` is 5 and `total_samples` is 36 — so the middle of the block is unpacked by
/// hand rather than sliced. Big-endian throughout, and the MD5 tail (§8.2) is not read here.
fn parse_streaminfo(block: &[u8]) -> Option<StreamInfo> {
    let b = block.get(..STREAMINFO_LEN)?;
    // Bits 80..100: sample rate. Bytes 10, 11 and the high nibble of 12.
    let sample_rate = (b[10] as u32) << 12 | (b[11] as u32) << 4 | (b[12] as u32) >> 4;
    // Bits 100..103: channels - 1. Bits 103..108: bits per sample - 1 (straddling bytes 12/13).
    let channels = ((b[12] >> 1) & 0x07) + 1;
    let bits_per_sample = (((b[12] & 0x01) << 4) | (b[13] >> 4)) + 1;
    // Bits 108..144: total samples — the low nibble of byte 13 then four whole bytes.
    let total_samples = ((b[13] & 0x0F) as u64) << 32
        | u32::from_be_bytes([b[14], b[15], b[16], b[17]]) as u64;
    Some(StreamInfo { sample_rate, channels, bits_per_sample, total_samples })
}

/// The [`TagSink`] behind [`parse`]: copies everything into an owned [`FlacTags`].
struct OwningSink<'t>(&'t mut FlacTags);

// One-time: this is the owning face of the parser, used once per stream at header time (spec:
// allocation discipline — header parsing is the sanctioned one-time-allocation exception). The
// borrowing face, `parse_into`, is the one with no allocation at all.
#[allow(clippy::disallowed_methods)]
impl TagSink for OwningSink<'_> {
    fn text(&mut self, key: &str, value: &str) {
        // `parse_into` already uppercased the key; re-normalising is idempotent and covers the
        // one case it passes through untouched (a field name longer than its stack buffer).
        self.0.comments.push((key.to_ascii_uppercase(), value.to_string()));
    }

    fn picture(&mut self, mime: &str, data: &[u8]) {
        self.0.pictures.push((mime.to_string(), data.to_vec()));
    }
}

#[cfg(test)]
// Tests build header fixtures and record what the sink was handed (spec: allocation discipline —
// tests are the sanctioned exception); `parse_into` itself allocates nothing.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    /// Build a minimal FLAC header: `fLaC`, a (skipped) STREAMINFO, then a VORBIS_COMMENT block.
    fn flac_with_comments(comments: &[&str]) -> Vec<u8> {
        let mut out = b"fLaC".to_vec();
        // STREAMINFO block (type 0, len 34, zeroed body — content irrelevant to tag parsing).
        out.push(0);
        out.extend_from_slice(&[0, 0, 34]);
        out.extend_from_slice(&[0u8; 34]);
        // VORBIS_COMMENT block (type 4), last-block flag set.
        let mut body = Vec::new();
        let vendor = b"ref";
        body.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        body.extend_from_slice(vendor);
        body.extend_from_slice(&(comments.len() as u32).to_le_bytes());
        for c in comments {
            body.extend_from_slice(&(c.len() as u32).to_le_bytes());
            body.extend_from_slice(c.as_bytes());
        }
        out.push(0x80 | 4);
        let len = body.len() as u32;
        out.extend_from_slice(&len.to_be_bytes()[1..]);
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn parses_vorbis_comments() {
        let flac = flac_with_comments(&["TITLE=Set Theory", "ARTIST=Carbon Based Lifeforms", "DATE=2015"]);
        let tags = parse(&flac);
        assert_eq!(
            tags.comments,
            vec![
                ("TITLE".into(), "Set Theory".into()),
                ("ARTIST".into(), "Carbon Based Lifeforms".into()),
                ("DATE".into(), "2015".into()),
            ]
        );
        assert!(tags.pictures.is_empty());
        assert_eq!(tags.len(), 3);
    }

    #[test]
    fn lowercase_keys_are_uppercased_values_kept() {
        let tags = parse(&flac_with_comments(&["title=x", "Comment=hi=there"]));
        // Key uppercased; only the first '=' splits, so values may contain '='.
        assert_eq!(tags.comments[0], ("TITLE".into(), "x".into()));
        assert_eq!(tags.comments[1], ("COMMENT".into(), "hi=there".into()));
    }

    #[test]
    fn non_flac_or_empty_is_empty() {
        assert!(parse(b"not a flac file").is_empty());
        assert!(parse(&[]).is_empty());
        assert!(parse(b"fLaC").is_empty()); // marker only, no blocks
    }

    /// A PICTURE block body (§8.8): type, MIME, description, dimensions, then the image bytes.
    fn picture_block_body(mime: &str, data: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&3u32.to_be_bytes()); // picture type: front cover
        b.extend_from_slice(&(mime.len() as u32).to_be_bytes());
        b.extend_from_slice(mime.as_bytes());
        b.extend_from_slice(&(3u32).to_be_bytes()); // description length
        b.extend_from_slice(b"pic"); // description
        b.extend_from_slice(&[0u8; 16]); // width, height, colour depth, indexed colours
        b.extend_from_slice(&(data.len() as u32).to_be_bytes());
        b.extend_from_slice(data);
        b
    }

    #[test]
    fn parses_pictures() {
        // fLaC + STREAMINFO + PICTURE(last).
        let mut flac = b"fLaC".to_vec();
        flac.push(0);
        flac.extend_from_slice(&[0, 0, 34]);
        flac.extend_from_slice(&[0u8; 34]);
        let body = picture_block_body("image/jpeg", b"\xFF\xD8\xFF\xE0jpeg-ish bytes");
        flac.push(0x80 | 6); // last-block, PICTURE
        flac.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
        flac.extend_from_slice(&body);

        let tags = parse(&flac);
        assert_eq!(tags.pictures.len(), 1);
        assert_eq!(tags.pictures[0].0, "image/jpeg");
        assert_eq!(tags.pictures[0].1, b"\xFF\xD8\xFF\xE0jpeg-ish bytes");
        assert!(tags.comments.is_empty());
    }

    /// fLaC + STREAMINFO(not-last) + VORBIS_COMMENT(not-last) + PICTURE(last) — the fixture the
    /// borrowing/owning equivalence test also runs on.
    fn flac_with_comment_and_picture() -> Vec<u8> {
        let mut flac = b"fLaC".to_vec();
        flac.push(0);
        flac.extend_from_slice(&[0, 0, 34]);
        flac.extend_from_slice(&[0u8; 34]);
        // VORBIS_COMMENT (not last)
        let mut vc = Vec::new();
        vc.extend_from_slice(&3u32.to_le_bytes());
        vc.extend_from_slice(b"ref");
        vc.extend_from_slice(&1u32.to_le_bytes());
        vc.extend_from_slice(&("TITLE=X".len() as u32).to_le_bytes());
        vc.extend_from_slice(b"TITLE=X");
        flac.push(4);
        flac.extend_from_slice(&(vc.len() as u32).to_be_bytes()[1..]);
        flac.extend_from_slice(&vc);
        // PICTURE (last)
        let pic = picture_block_body("image/png", b"\x89PNG...");
        flac.push(0x80 | 6);
        flac.extend_from_slice(&(pic.len() as u32).to_be_bytes()[1..]);
        flac.extend_from_slice(&pic);
        flac
    }

    #[test]
    fn parses_comments_and_a_picture_together() {
        let tags = parse(&flac_with_comment_and_picture());
        assert_eq!(tags.comments, vec![("TITLE".to_string(), "X".to_string())]);
        assert_eq!(tags.pictures.len(), 1);
        assert_eq!(tags.pictures[0].0, "image/png");
        assert_eq!(tags.pictures[0].1, b"\x89PNG...");
    }

    /// A recording sink that also proves the picture bytes were **borrowed**: it notes where in
    /// the input buffer each `&[u8]` it was handed actually lives. (`TagSink`'s arguments are
    /// borrowed for the call only — the trait has no lifetime to keep them past it — so the
    /// evidence is recorded as an address range rather than as a retained slice.)
    struct Rec {
        text: Vec<(String, String)>,
        pictures: Vec<(String, Vec<u8>)>,
        /// `(offset into the input, length)` of every picture's bytes.
        picture_spans: Vec<(usize, usize)>,
        /// Address and length of the buffer `parse_into` was given.
        base: usize,
        len: usize,
    }

    impl Rec {
        fn over(input: &[u8]) -> Self {
            Self {
                text: Vec::new(),
                pictures: Vec::new(),
                picture_spans: Vec::new(),
                base: input.as_ptr() as usize,
                len: input.len(),
            }
        }
    }

    impl TagSink for Rec {
        fn text(&mut self, key: &str, value: &str) {
            self.text.push((key.to_string(), value.to_string()));
        }
        fn picture(&mut self, mime: &str, data: &[u8]) {
            // Borrowed, not copied: the bytes must lie inside the input buffer.
            let off = (data.as_ptr() as usize)
                .checked_sub(self.base)
                .expect("picture bytes must borrow the input, not a copy");
            assert!(off + data.len() <= self.len, "picture bytes must borrow the input");
            self.picture_spans.push((off, data.len()));
            self.pictures.push((mime.to_string(), data.to_vec()));
        }
    }

    #[test]
    fn parse_into_matches_parse_and_borrows_the_picture() {
        let flac = flac_with_comment_and_picture();
        let owned = parse(&flac);

        let mut rec = Rec::over(&flac);
        parse_into(&flac, &mut rec);

        // Identical comments and byte-identical picture data — one borrowed, one owned.
        assert_eq!(rec.text, owned.comments);
        assert_eq!(rec.pictures, owned.pictures);
        assert_eq!(rec.picture_spans.len(), 1);
        let (off, len) = rec.picture_spans[0];
        assert_eq!(&flac[off..off + len], &owned.pictures[0].1[..]);
    }

    #[test]
    fn parse_into_matches_parse_across_fixtures() {
        let fixtures = [
            flac_with_comments(&["TITLE=Set Theory", "artist=x", "ARTIST=y", "DATE=2015"]),
            flac_with_comments(&["title=x", "Comment=hi=there", "MixedCase=Value"]),
            flac_with_comments(&[]),
            flac_with_comment_and_picture(),
            b"not a flac file".to_vec(),
            b"fLaC".to_vec(),
            Vec::new(),
        ];
        for flac in fixtures {
            let owned = parse(&flac);
            let mut rec = Rec::over(&flac);
            parse_into(&flac, &mut rec);
            assert_eq!(rec.text, owned.comments);
            assert_eq!(rec.pictures, owned.pictures);
        }
    }

    #[test]
    fn parse_into_truncations_never_panic() {
        let flac = flac_with_comment_and_picture();
        for n in 0..flac.len() {
            let mut rec = Rec::over(&flac[..n]);
            parse_into(&flac[..n], &mut rec);
            // The owning face must agree on every prefix too.
            assert_eq!(rec.text, parse(&flac[..n]).comments, "prefix of {n} bytes");
        }
    }

    #[test]
    fn oversized_field_name_survives_uncopied() {
        // Longer than the stack buffer: `parse_into` passes it through with its original case,
        // and `parse` still stores it uppercased.
        let key = "k".repeat(KEY_STACK_MAX + 1);
        let entry = format!("{key}=v");
        let tags = parse(&flac_with_comments(&[&entry]));
        assert_eq!(tags.comments, vec![(key.to_ascii_uppercase(), "v".to_string())]);
    }

    /// A STREAMINFO block body (§8.2 Table 3), bit-packed by hand.
    fn streaminfo_body(rate: u32, channels: u8, bits: u8, total: u64) -> Vec<u8> {
        let mut b = vec![0u8; 34];
        b[0..2].copy_from_slice(&4096u16.to_be_bytes()); // min block size
        b[2..4].copy_from_slice(&4096u16.to_be_bytes()); // max block size
        b[10] = (rate >> 12) as u8;
        b[11] = (rate >> 4) as u8;
        b[12] = ((rate << 4) as u8 & 0xF0) | ((channels - 1) << 1) | ((bits - 1) >> 4);
        b[13] = (((bits - 1) & 0x0F) << 4) | ((total >> 32) as u8 & 0x0F);
        b[14..18].copy_from_slice(&(total as u32).to_be_bytes());
        b
    }

    fn flac_with_streaminfo(body: &[u8]) -> Vec<u8> {
        let mut out = b"fLaC".to_vec();
        out.push(0x80); // last block, type 0
        out.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn reads_stream_info() {
        for (rate, channels, bits, total) in [
            (44_100u32, 2u8, 16u8, 9_876_543u64),
            (48_000, 1, 24, 0),
            (8_000, 8, 4, 1),
            // A 20-bit rate that is not a "usual" one, and a 36-bit sample count well past
            // 4 GiB — the two fields whose width is easy to get wrong (§8.2).
            (1_048_575, 5, 32, (1u64 << 36) - 1),
            (37_337, 3, 20, 5_000_000_000),
            (0, 1, 8, 0), // "non-audio stream" (§8.2)
        ] {
            let flac = flac_with_streaminfo(&streaminfo_body(rate, channels, bits, total));
            assert_eq!(
                stream_info(&flac),
                Some(StreamInfo {
                    sample_rate: rate,
                    channels,
                    bits_per_sample: bits,
                    total_samples: total
                }),
                "rate {rate} ch {channels} bits {bits} total {total}"
            );
        }
    }

    #[test]
    fn stream_info_missing_or_truncated() {
        assert_eq!(stream_info(b"not a flac file"), None);
        assert_eq!(stream_info(&[]), None);
        assert_eq!(stream_info(b"fLaC"), None);
        // A comment-only chain has no STREAMINFO.
        let mut no_info = b"fLaC".to_vec();
        no_info.push(0x80 | 4);
        no_info.extend_from_slice(&[0, 0, 4]);
        no_info.extend_from_slice(&[0, 0, 0, 0]);
        assert_eq!(stream_info(&no_info), None);
        // Every truncation of a good header: no panic, and no half-read answer.
        let flac = flac_with_streaminfo(&streaminfo_body(44_100, 2, 16, 1_000));
        for n in 0..flac.len() {
            assert_eq!(stream_info(&flac[..n]), None, "truncated to {n}");
        }
        assert!(stream_info(&flac).is_some());
        // A short (declared 20-byte) STREAMINFO block is not enough.
        let mut short = b"fLaC".to_vec();
        short.push(0x80);
        short.extend_from_slice(&[0, 0, 20]);
        short.extend_from_slice(&[0u8; 20]);
        assert_eq!(stream_info(&short), None);
    }

    /// The cross-check that matters: a real encoder-produced header, read by this byte-level
    /// parser, agrees field for field with the decoder's own bit-level STREAMINFO parse.
    #[test]
    fn stream_info_agrees_with_the_decoder() {
        use crate::{FlacDecoder, FlacEncoder, SampleFormat};

        const SAMPLES: u32 = 4_000; // interchannel samples
        let (mut enc, mut flac) = FlacEncoder::new(44_100, 2, SampleFormat::S16).unwrap();
        let pcm: Vec<u8> = (0..SAMPLES)
            .flat_map(|i| {
                let s = (i as i16).wrapping_mul(97);
                [s.to_le_bytes(), s.wrapping_neg().to_le_bytes()]
            })
            .flatten()
            .collect();
        let mut frames = Vec::new();
        enc.encode_interleaved(&pcm, &mut frames).unwrap();
        let body = enc.finish();
        let at = crate::streaminfo_offset();
        flac[at..at + body.len()].copy_from_slice(&body);
        flac.extend_from_slice(&frames);

        let info = stream_info(&flac).expect("STREAMINFO");
        let decoded = FlacDecoder::decode(&flac).expect("decodes");
        assert_eq!(info.sample_rate, decoded.info.sample_rate);
        assert_eq!(info.channels as u32, decoded.info.channels);
        assert_eq!(info.bits_per_sample as u32, decoded.info.bits_per_sample);
        assert_eq!(info.total_samples, decoded.info.total_samples);
        assert_eq!(
            info,
            StreamInfo {
                sample_rate: 44_100,
                channels: 2,
                bits_per_sample: 16,
                total_samples: SAMPLES as u64,
            }
        );
    }

    // --- SEEKTABLE (§8.5) -------------------------------------------------------------

    /// One 18-byte seek point (§8.5) as an encoder writes it.
    fn seek_point(sample: u64, offset: u64, frame_samples: u16) -> Vec<u8> {
        let mut p = sample.to_be_bytes().to_vec();
        p.extend_from_slice(&offset.to_be_bytes());
        p.extend_from_slice(&frame_samples.to_be_bytes());
        p
    }

    /// The 18 bytes of a placeholder point (§8.5): the all-ones sample number, and whatever
    /// the encoder left in the other fields — here, values that would be obviously wrong if
    /// they were ever believed.
    fn placeholder() -> Vec<u8> {
        seek_point(u64::MAX, 0xDEAD_BEEF, 0xFFFF)
    }

    /// A FLAC header: `fLaC`, a zeroed STREAMINFO, an optional SEEKTABLE, and a last block of
    /// `pad_len` PADDING bytes — so `audio_start` has a definite answer to check.
    fn flac_with_seektable(points: &[Vec<u8>], pad_len: usize) -> Vec<u8> {
        let mut out = b"fLaC".to_vec();
        let mut block = |kind: u8, last: bool, body: &[u8]| {
            out.push(if last { 0x80 | kind } else { kind });
            out.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
            out.extend_from_slice(body);
        };
        block(0, false, &[0u8; STREAMINFO_LEN]);
        if !points.is_empty() {
            block(SEEKTABLE_TYPE, false, &points.concat());
        }
        block(1, true, &vec![0u8; pad_len]); // PADDING (§8.3), last block
        out
    }

    #[test]
    fn seek_table_points_carry_sample_offset_and_frame_length() {
        let points = [
            seek_point(0, 0, 4096),
            seek_point(44_100, 12_345, 4096),
            seek_point(88_200, 25_000, 2048),
        ];
        let flac = flac_with_seektable(&points, 0);
        let got: Vec<SeekPoint> = seek_table(&flac).collect();
        assert_eq!(
            got,
            vec![
                SeekPoint { sample: 0, byte_offset: 0, frame_samples: 4096 },
                SeekPoint { sample: 44_100, byte_offset: 12_345, frame_samples: 4096 },
                SeekPoint { sample: 88_200, byte_offset: 25_000, frame_samples: 2048 },
            ]
        );
        // §8.5 requires ascending, distinct sample numbers, and the parse preserves order.
        assert!(got.windows(2).all(|w| w[0].sample < w[1].sample));
    }

    #[test]
    fn placeholder_points_are_skipped() {
        // §8.5's placeholders sort last and have undefined offsets; a player that used one
        // would seek to byte 0xDEADBEEF. Three real points, then three reserved slots.
        let points = [
            seek_point(0, 0, 4096),
            seek_point(44_100, 12_345, 4096),
            seek_point(88_200, 25_000, 4096),
            placeholder(),
            placeholder(),
            placeholder(),
        ];
        let got: Vec<SeekPoint> = seek_table(&flac_with_seektable(&points, 0)).collect();
        assert_eq!(got.len(), 3, "the three reserved slots are not seek points");
        assert!(got.iter().all(|p| p.byte_offset != 0xDEAD_BEEF));
        assert_eq!(got.last().unwrap().sample, 88_200);

        // A table that is *nothing but* placeholders is an empty index, not three bad points.
        let all_pad = [placeholder(), placeholder()];
        assert_eq!(seek_table(&flac_with_seektable(&all_pad, 0)).count(), 0);
    }

    #[test]
    fn audio_start_is_the_end_of_the_metadata_chain() {
        // fLaC(4) + STREAMINFO(4+34) + SEEKTABLE(4+36) + PADDING(4+64)
        let points = [seek_point(0, 0, 4096), seek_point(44_100, 9_000, 4096)];
        let flac = flac_with_seektable(&points, 64);
        assert_eq!(audio_start(&flac), Some(4 + 38 + 40 + 68));
        assert_eq!(audio_start(&flac), Some(flac.len()), "the frames start right after");

        // Trailing frame bytes do not move it.
        let mut with_frames = flac.clone();
        with_frames.extend_from_slice(&[0xFF, 0xF8, 0x00, 0x00]);
        assert_eq!(audio_start(&with_frames), Some(flac.len()));

        // A window that stops inside the chain has no answer — better than a wrong one.
        assert_eq!(audio_start(&flac[..flac.len() - 1]), None);
        assert_eq!(audio_start(b"fLaC"), None);
        assert_eq!(audio_start(b"not a flac file"), None);
        assert_eq!(audio_start(&[]), None);
    }

    #[test]
    fn a_stream_without_a_seektable_has_an_empty_index() {
        assert_eq!(seek_table(&flac_with_seektable(&[], 16)).count(), 0);
        assert_eq!(seek_table(&flac_with_comments(&["TITLE=x"])).count(), 0);
        assert_eq!(seek_table(b"not a flac file").count(), 0);
        assert_eq!(seek_table(&[]).count(), 0);
    }

    #[test]
    fn a_ragged_or_truncated_table_yields_its_whole_points() {
        // A SEEKTABLE whose length is not a multiple of 18 — §8.5 forbids it, but this is
        // untrusted input. The whole points parse; the ragged remainder is ignored.
        let mut body = [seek_point(0, 0, 4096), seek_point(44_100, 9_000, 4096)].concat();
        body.extend_from_slice(&[0xAB; 7]);
        let mut flac = b"fLaC".to_vec();
        flac.push(0);
        flac.extend_from_slice(&(STREAMINFO_LEN as u32).to_be_bytes()[1..]);
        flac.extend_from_slice(&[0u8; STREAMINFO_LEN]);
        flac.push(0x80 | SEEKTABLE_TYPE);
        flac.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
        flac.extend_from_slice(&body);
        assert_eq!(seek_table(&flac).count(), 2);

        // Every truncation of a real header: no panic, and never a point invented out of
        // bytes that were not there.
        let full = flac_with_seektable(&[seek_point(0, 0, 4096), placeholder()], 8);
        for n in 0..=full.len() {
            let points: Vec<SeekPoint> = seek_table(&full[..n]).collect();
            assert!(points.len() <= 1, "cut at {n} yielded {points:?}");
            assert!(points.iter().all(|p| p.sample == 0));
            let _ = audio_start(&full[..n]);
            let _ = stream_info(&full[..n]);
        }
        // A declared block length far past the buffer ends the walk rather than indexing it.
        let mut lying = b"fLaC".to_vec();
        lying.push(SEEKTABLE_TYPE);
        lying.extend_from_slice(&[0xFF, 0xFF, 0xFF]);
        lying.extend_from_slice(&seek_point(0, 0, 4096));
        assert_eq!(seek_table(&lying).count(), 0);
        assert_eq!(audio_start(&lying), None);
    }
}
