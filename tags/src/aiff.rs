//! AIFF / AIFF-C metadata: the `COMM` chunk for properties, the EA IFF 85 text chunks and a
//! de-facto `ID3 ` chunk for tags.
//!
//! In-tree spec: `spec/AIFF.md` (Apple's *Audio Interchange File Format* 1.3 and the 1991
//! AIFF-C addendum are not redistributable, so that file is the exhaustive field-table
//! variant the `mp4/spec/NOTES.md` convention calls for). Citations below name the section
//! of the Apple paper and, where it matters, the heading in `spec/AIFF.md`.
//!
//! Structurally this is RIFF with the bytes the other way round: a file is
//! `"FORM" <u32 ckSize> <formType> <chunk>*`, each chunk `<ckID:4> <u32 ckSize> <data>`, and
//! **every chunk is word-aligned** — an odd `ckSize` is followed by one pad byte that is not
//! counted in the size (AIFF-1.3, "File Structure"). The one difference from `wav.rs` is the
//! one that matters: **AIFF is big-endian**, because it is a 1988 Motorola 68000 format.
//! Reading `ckSize` little-endian does not fail, it just walks into nonsense — hence the
//! separate module rather than a byte-order flag on the RIFF walk.
//!
//! The walk is *windowed* exactly like `wav.rs`'s: it parses a byte range starting at a known
//! chunk boundary and, if it runs out of window with file left, reports the absolute offset of
//! the next chunk so the engine can read one more extent there. That is what keeps an `ID3 `
//! chunk written *after* a 500 MB `SSND` chunk to one extra positioned read.
//!
//! ## What it costs
//!
//! **One read for the ordinary file**: `COMM` is at the front, and the duration comes from its
//! frame count rather than from the audio, so `SSND` is skipped by size and never touched. A
//! tagger that appended `ID3 ` or the text chunks behind the audio costs one positioned
//! follow-up. Nothing here allocates: text reaches the sink borrowed from the read buffer (or
//! through the shared Latin-1 scratch), and the `ID3 ` payload is handed to [`pf_mp3::id3`]
//! where it lies.

use profluens_core::event::TagSink;
use profluens_core::memory::Arena;

use crate::store::decode_text;
use crate::Props;

/// Scratch for the Latin-1 → UTF-8 fallback in [`decode_text`]. AIFF text chunks are titles,
/// author lines and comments; 1 KiB of transcoded output (≥ 512 source bytes) is well past any
/// real one. Sized and justified exactly as `wav.rs`'s.
const TEXT_SCRATCH: usize = 1024;

/// Bytes of FORM header before the first chunk: `"FORM" <u32 ckSize> <formType>`
/// (AIFF-1.3, "File Structure").
const FORM_HEADER: u64 = 12;

/// Bytes of `COMM` this parser needs: `numChannels`(2) `numSampleFrames`(4) `sampleSize`(2)
/// `sampleRate`(10) (AIFF-1.3, "Common Chunk"). AIFF-C's `compressionType` and
/// `compressionName` follow at octet 18 and steer decoding, not the metadata view — so a
/// plain AIFF `ckSize` of exactly 18 must not be rejected for being short.
const COMM_LEN: usize = 18;

/// Which chunks one pass over the chunk headers emits. See `spec/AIFF.md`, "Precedence: ID3
/// wins": the sink returns the *first* value for a key, so emission order is precedence
/// order, and the richer, explicitly-encoded ID3v2 tag must reach it before the four
/// "pure ASCII" IFF text chunks — whatever order they sit in on disk.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Pass {
    Id3,
    Text,
}

impl Pass {
    /// Both passes, in precedence order. The engine drives them itself because an AIFF's tag
    /// chunks can straddle *two* windows — the prefix and one positioned extent — and running
    /// `Id3` then `Text` inside each window separately would let a `NAME` chunk in the prefix
    /// beat an `ID3 ` chunk in the extent. Precedence has to be the outer loop.
    pub(crate) const ORDER: [Pass; 2] = [Pass::Id3, Pass::Text];
}

/// Walk one window in both passes: the single-window convenience over [`walk_pass`], used by
/// the tests. The engine calls [`walk_pass`] directly, because it may have two windows.
///
/// `window` begins at absolute file offset `base` **on a chunk boundary** (or at the file
/// start, where the 12-byte FORM header is skipped first). Fills `props` and pushes tags into
/// `sink`. Returns the absolute offset of the next chunk when the walk ran past the end of the
/// window with file left to read — the engine's cue to fetch one more extent — and `None` when
/// the walk finished or the file is exhausted.
#[cfg(test)]
pub(crate) fn walk(
    window: &[u8],
    base: u64,
    file_len: u64,
    props: &mut Props,
    arena: &Arena,
    sink: &mut impl TagSink,
) -> Option<u64> {
    let mut resume = None;
    for pass in Pass::ORDER {
        let r = walk_pass(window, base, file_len, props, pass, arena, sink);
        resume = resume.or(r);
    }
    resume
}

/// The head-window plan: where the walk ran out of bytes, without emitting anything.
///
/// A pass with a discarding sink would do — but it would also push an `ID3 ` chunk's frames
/// through [`pf_mp3::id3::parse_v2`], which touches the arena for the frames that need a
/// rewrite. Planning happens before the engine knows it has all the bytes, so it walks headers
/// and nothing else.
pub(crate) fn plan(window: &[u8], base: u64, file_len: u64) -> Option<u64> {
    for_each_chunk(window, base, file_len, |_, _| {})
}

/// One emission pass over one window. Walks the chunk *headers* — a few dozen compares, no
/// payload touched twice — and emits only what `which` selects.
///
/// `props` is filled from `COMM` on both passes: the read is idempotent, and carrying a flag
/// to suppress it would cost more than the eight loads it saves.
pub(crate) fn walk_pass(
    window: &[u8],
    base: u64,
    file_len: u64,
    props: &mut Props,
    which: Pass,
    arena: &Arena,
    sink: &mut impl TagSink,
) -> Option<u64> {
    let mut scratch = [0u8; TEXT_SCRATCH];
    for_each_chunk(window, base, file_len, |id, body| match (id, which) {
        (b"COMM", _) => comm(body, props),
        // Not in either Apple paper: a complete ID3v2 tag as a chunk body, which is what
        // ffmpeg (`-write_id3v2 1`), iTunes and several taggers write. The payload starts at
        // its own `"ID3"` magic, so it is exactly what `parse_v2` expects (`spec/AIFF.md`,
        // "The `ID3 ` chunk").
        (b"ID3 " | b"id3 ", Pass::Id3) => pf_mp3::id3::parse_v2(body, arena, sink),
        (_, Pass::Text) => {
            if let Some(key) = text_key(id) {
                // "text contains pure ASCII characters. It is not a pstring nor a C string"
                // (AIFF-1.3, "Text Chunks") — so the length is `ckSize` and there is no
                // terminator. Taggers nonetheless write one, and files in the wild carry
                // Latin-1 and UTF-8 rather than ASCII, so a trailing NUL is trimmed and the
                // same `decode_text` fallback the RIFF `INFO` reader uses runs here.
                let text = body.split(|b| *b == 0).next().unwrap_or(body);
                let text = decode_text(text, &mut scratch);
                if !text.is_empty() {
                    sink.text(key, text);
                }
            }
        }
        _ => {}
    })
}

/// Walk the chunk headers in `window`, calling `visit(ckID, body)` for every chunk this module
/// parses whose body is fully present.
///
/// Returns the absolute offset of the first chunk that could not be reached, or `None` when
/// the walk finished or the file is exhausted.
fn for_each_chunk(
    window: &[u8],
    base: u64,
    file_len: u64,
    mut visit: impl FnMut(&[u8; 4], &[u8]),
) -> Option<u64> {
    let mut abs = if base == 0 { FORM_HEADER } else { base };
    let mut resume = None;

    while let Some(w) = abs.checked_sub(base).and_then(|d| usize::try_from(d).ok()) {
        let Some(header) = window.get(w..w + 8) else {
            // The next chunk header does not fit the window: resume there.
            resume = Some(abs);
            break;
        };
        let id: [u8; 4] = [header[0], header[1], header[2], header[3]];
        // Big-endian: AIFF is a 68000-era format (`spec/AIFF.md`, "Byte order").
        let size = u64::from(u32::from_be_bytes([header[4], header[5], header[6], header[7]]));
        let body_at = w + 8;

        // The chunks this walk parses need their whole body: half a text chunk is a truncated
        // title, and half an ID3 tag is half its frames. When the window cuts one short and
        // the file really does hold the rest, resume at this chunk — which is how a tag
        // straddling the end of the prefix read is found. `SSND` is exempt: it is skipped by
        // size and never read, which is what keeps a 500 MB file to one op.
        if parsed(&id) {
            if (window.len().saturating_sub(body_at) as u64) < size && abs + 8 + size <= file_len {
                resume = Some(abs);
                break;
            }
            if let Some(body) = window.get(body_at..).map(|t| &t[..t.len().min(size as usize)]) {
                visit(&id, body);
            }
        }

        // Word alignment: an odd chunk size carries a pad byte outside the size (AIFF-1.3,
        // "File Structure": "If the data is an odd number of bytes in length, a zero pad byte
        // must be added at the end. The pad byte is not included in ckSize").
        let advance = 8 + size + (size & 1);
        let Some(next) = abs.checked_add(advance) else { break };
        // `next <= abs` guards a zero-size chunk looping forever; `next >= file_len` is the
        // ordinary end of the walk — either way there is nothing left to resume at.
        if next <= abs || next >= file_len {
            break;
        }
        abs = next;
    }

    resume.filter(|at| *at < file_len)
}

/// Whether this module reads a chunk's body — and therefore whether a body cut short by the
/// window is worth another read.
fn parsed(id: &[u8; 4]) -> bool {
    matches!(id, b"COMM" | b"ID3 " | b"id3 ") || text_key(id).is_some()
}

/// The Common Chunk (AIFF-1.3, "Common Chunk"), required and unique in every FORM AIFF:
///
/// ```text
///   short     numChannels        big-endian
///   ulong     numSampleFrames    big-endian — sample FRAMES, not bytes and not points
///   short     sampleSize         bits per sample point, 1..32
///   extended  sampleRate         80-bit IEEE 754, frames per second
/// ```
///
/// AIFF-C (`FORM` type `AIFC`) appends `compressionType` and a Pascal `compressionName`. The
/// compression type is read for one reason only — to decide whether `numSampleFrames` can be
/// believed. See [`frame_count_is_frames`] and `spec/AIFF.md`, "Duration".
fn comm(body: &[u8], props: &mut Props) {
    let Some(c) = body.get(..COMM_LEN) else { return };
    let channels = u16::from_be_bytes([c[0], c[1]]);
    if channels > 0 {
        props.channels = Some(u32::from(channels));
    }
    let frames = u64::from(u32::from_be_bytes([c[2], c[3], c[4], c[5]]));
    let Some(rate) = extended_rate(&[c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15], c[16], c[17]])
    else {
        // A rate that is zero, negative, infinite, NaN or absurd yields no duration rather
        // than a wrong one.
        return;
    };
    // Both of these are unambiguous whatever the compression is.
    props.sample_rate = Some(rate);
    if !frame_count_is_frames(body.get(18..22)) {
        return;
    }
    // duration = numSampleFrames / sampleRate, in nanoseconds. u128 because `frames * 1e9`
    // overflows u64 past ~18 G frames. Exact: `numSampleFrames` is a declared decoded-frame
    // count, the AIFF equivalent of a FLAC STREAMINFO sample count.
    props.duration_ns =
        u64::try_from(u128::from(frames) * 1_000_000_000 / u128::from(rate)).ok();
    props.duration_exact = true;
}

/// Whether this `COMM`'s `numSampleFrames` can be taken at face value, given its
/// `compressionType` (`None` when the chunk is a plain 18-octet AIFF one, which has no such
/// field and is therefore uncompressed by construction).
///
/// AIFF-C says plainly that "numSampleFrames contains the number of sample frames in the
/// Sound Data Chunk … not the number of bytes nor the number of sample points". **Real files
/// disagree.** An AIFF-C written by ffmpeg with `-c:a adpcm_ima_qt` states
/// `numSampleFrames = 2068` for three seconds of 44.1 kHz audio; the true count is 132352,
/// because QuickTime's `ima4` convention puts the number of *packets* there and each packet
/// holds 64 sample frames per channel. Believing the field would report 47 ms for a 3-second
/// file — wrong by 64x, and wrong in the direction that looks plausible.
///
/// So the field is believed only for the compression types where one unit *is* one sample
/// frame: `NONE` (AIFF-C's own "not compressed") and the QuickTime PCM four-CCs that mean the
/// same thing in another byte order or width. Anything else — an ADPCM, µ-law, or codec
/// four-CC this list does not know — yields **no duration**, which is honest, rather than a
/// number that would need a per-codec frames-per-packet table this crate has no normative
/// source for. Rate and channel count are still reported: those are unambiguous either way.
fn frame_count_is_frames(compression: Option<&[u8]>) -> bool {
    let Some(kind) = compression else {
        return true; // an 18-octet COMM: plain AIFF, no compression field exists
    };
    matches!(
        kind,
        // AIFF-C, "Compression Type IDs": `NONE` — "not compressed".
        b"NONE"
            // Apple QuickTime PCM sound four-CCs, which are the same samples relabelled:
            // big/little-endian 16-bit, offset-binary 8-bit, 24- and 32-bit integer, and
            // IEEE float (QTFF, "Sound Sample Descriptions" — the same source
            // `mp4/spec/NOTES.md` cites for the QuickTime metadata scheme).
            | b"twos"
            | b"sowt"
            | b"raw "
            | b"in24"
            | b"42ni"
            | b"in32"
            | b"23ni"
            | b"fl32"
            | b"FL32"
            | b"fl64"
            | b"FL64"
    )
}

/// An 80-bit IEEE 754 extended ("SANE Extended") big-endian float, as a whole number of hertz.
///
/// AIFF-1.3 types the `COMM` sample rate as "80 bit IEEE Standard 754 floating point number
/// (Standard Apple Numeric Environment [SANE] data type Extended)". Layout, most significant
/// byte first (`spec/AIFF.md`, "The 80-bit IEEE 754 extended sample rate"):
///
/// ```text
///   bit 79      sign
///   bits 78..64 exponent, biased by 16383
///   bits 63..0  mantissa — with an EXPLICIT integer bit at 63, unlike binary32/binary64
/// ```
///
/// so a finite value is `(-1)^sign × mantissa × 2^(exponent - 16383 - 63)`.
///
/// Evaluated in integers rather than through `f64`: the answer wanted is a whole number of
/// hertz, and every rate in use (`40 0E AC 44 …` = 44100, and the table in `spec/AIFF.md`)
/// is exactly representable, so a float round-trip could only introduce error. `None` for
/// anything that is not a plausible rate — zero, negative, denormal, infinity, NaN, or a
/// magnitude past [`u32`] — which is the same "report nothing rather than something wrong"
/// rule the rest of the crate follows.
fn extended_rate(b: &[u8; 10]) -> Option<u32> {
    // A negative sample rate is not one.
    if b[0] & 0x80 != 0 {
        return None;
    }
    let exponent = (u32::from(b[0] & 0x7F) << 8) | u32::from(b[1]);
    let mantissa =
        u64::from_be_bytes([b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9]]);
    // Exponent 0 is zero or a denormal — the largest denormal is about 2^-16382, which is not
    // a sample rate. Exponent 0x7FFF is infinity or NaN.
    if exponent == 0 || exponent == 0x7FFF {
        return None;
    }
    let shift = exponent as i64 - 16383 - 63;
    let value: u128 = if shift >= 0 {
        // Any normalised value with a non-negative shift is at least 2^63 Hz; the bound keeps
        // the shift itself in range for the pathological "unnormal" encodings.
        if shift >= 64 {
            return None;
        }
        u128::from(mantissa) << shift
    } else {
        let s = (-shift) as u32;
        if s > 64 {
            return None; // rounds to zero
        }
        // Round half up, so a rate stored as the nearest representable float still lands on
        // the integer a user would recognise. `s >= 1` here, so the half-ULP shift is valid.
        (u128::from(mantissa) + (1u128 << (s - 1))) >> s
    };
    u32::try_from(value).ok().filter(|&r| r > 0)
}

/// The four EA IFF 85 text chunks (AIFF-1.3, "Text Chunks — Name, Author, Copyright,
/// Annotation") → the canonical uppercase Vorbis-comment keys core's [`TagSink`] documents.
///
/// The copyright id is `'(c) '` — "the 'c' is lowercase and there is a space (0x20) after the
/// close parenthesis" — and the chunk id itself "serves as the copyright characters '©'", so
/// the body is the notice without one.
fn text_key(id: &[u8; 4]) -> Option<&'static str> {
    Some(match id {
        b"NAME" => "TITLE",     // the name of the sampled sound
        b"AUTH" => "ARTIST",    // "the creator of a sampled sound"
        b"(c) " => "COPYRIGHT", // a copyright notice
        b"ANNO" => "COMMENT",   // an annotation; many may exist, the first wins
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;
    use profluens_core::event::{TagList, TagListSink};
    use profluens_core::memory::Pool;

    use crate::fixture::{aiff, aiff_chunk, comm_chunk, extended80, form, id3v2};

    fn run(bytes: &[u8]) -> (Props, TagList, Option<u64>) {
        run_windowed(bytes, bytes.len() as u64)
    }

    fn run_windowed(bytes: &[u8], file_len: u64) -> (Props, TagList, Option<u64>) {
        let pool = Pool::bounded(4096, 8);
        let arena = Arena::default();
        let mut sink = TagListSink::new(&pool);
        let mut props = Props::default();
        let resume = walk(bytes, 0, file_len, &mut props, &arena, &mut sink);
        (props, sink.finish(), resume)
    }

    #[test]
    fn the_four_worked_rates_decode_exactly() {
        for rate in [44_100u32, 48_000, 22_050, 8_000, 96_000, 11_025, 192_000, 1] {
            assert_eq!(extended_rate(&extended80(f64::from(rate))), Some(rate), "rate {rate}");
        }
        // The byte patterns `spec/AIFF.md` tabulates, hand-written rather than round-tripped.
        assert_eq!(
            extended_rate(&[0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]),
            Some(44_100)
        );
        assert_eq!(
            extended_rate(&[0x40, 0x0E, 0xBB, 0x80, 0, 0, 0, 0, 0, 0]),
            Some(48_000)
        );
        assert_eq!(
            extended_rate(&[0x40, 0x0D, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]),
            Some(22_050)
        );
        assert_eq!(
            extended_rate(&[0x40, 0x0B, 0xFA, 0x00, 0, 0, 0, 0, 0, 0]),
            Some(8_000)
        );
    }

    #[test]
    fn extended_edge_cases_yield_no_rate_and_never_panic() {
        let zero = [0u8; 10];
        assert_eq!(extended_rate(&zero), None, "+0 is not a sample rate");
        // A denormal: exponent 0, mantissa non-zero.
        assert_eq!(extended_rate(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), None);
        // +inf and a NaN: exponent all ones.
        assert_eq!(extended_rate(&[0x7F, 0xFF, 0x80, 0, 0, 0, 0, 0, 0, 0]), None);
        assert_eq!(extended_rate(&[0x7F, 0xFF, 0xC0, 0, 0, 0, 0, 0, 0, 0]), None);
        // Negative 44100.
        assert_eq!(extended_rate(&[0xC0, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]), None);
        // 2^64 Hz and 2^100 Hz: past u32.
        assert_eq!(extended_rate(&[0x40, 0x3F, 0x80, 0, 0, 0, 0, 0, 0, 0]), None);
        assert_eq!(extended_rate(&[0x40, 0x63, 0x80, 0, 0, 0, 0, 0, 0, 0]), None);
        // Something far below 1 Hz rounds to zero, which is not a rate.
        assert_eq!(extended_rate(&[0x3F, 0x00, 0x80, 0, 0, 0, 0, 0, 0, 0]), None);
        // Every possible top-two-byte pattern, exhaustively: none may panic.
        for hi in 0..=u16::MAX {
            let [a, b] = hi.to_be_bytes();
            let _ = extended_rate(&[a, b, 0x80, 0, 0, 0, 0, 0, 0, 0]);
            let _ = extended_rate(&[a, b, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        }
    }

    #[test]
    fn props_and_text_chunks() {
        let f = aiff(
            b"AIFF",
            44_100,
            2,
            16,
            44_100, // exactly one second
            &[("NAME", "Enough"), ("AUTH", "Solar Fields"), ("ANNO", "a note"), ("(c) ", "1988 Apple")],
            None,
        );
        let (props, tags, resume) = run(&f);
        assert_eq!(props.sample_rate, Some(44_100));
        assert_eq!(props.channels, Some(2));
        assert_eq!(props.duration_ns, Some(1_000_000_000));
        assert!(props.duration_exact, "a declared frame count is authoritative");
        assert_eq!(tags.get("TITLE"), Some("Enough"));
        assert_eq!(tags.get("ARTIST"), Some("Solar Fields"));
        assert_eq!(tags.get("COMMENT"), Some("a note"));
        assert_eq!(tags.get("COPYRIGHT"), Some("1988 Apple"));
        assert_eq!(resume, None);
    }

    /// AIFF-C with a PCM compression type: the frame count means frames, so the duration is
    /// exact just as for a plain AIFF.
    #[test]
    fn aifc_with_a_pcm_compression_type_states_its_length() {
        for kind in [*b"NONE", *b"sowt", *b"twos", *b"fl32", *b"in24"] {
            let f = aiff(b"AIFC", 22_050, 1, 16, 11_025, &[("NAME", "Compressed")], Some(kind));
            let (props, tags, _) = run(&f);
            assert_eq!(props.sample_rate, Some(22_050));
            assert_eq!(props.channels, Some(1));
            assert_eq!(props.duration_ns, Some(500_000_000), "{:?}", kind);
            assert!(props.duration_exact);
            assert_eq!(tags.get("TITLE"), Some("Compressed"));
        }
    }

    /// …and one that is genuinely compressed reports rate and channels but **no** duration:
    /// real `ima4` files put a packet count in `numSampleFrames`, so believing it would be
    /// wrong by 64x. See [`frame_count_is_frames`].
    #[test]
    fn aifc_with_a_compressed_type_reports_no_duration() {
        for kind in [*b"ima4", *b"ulaw", *b"QDM2", *b"MAC3"] {
            let f = aiff(b"AIFC", 44_100, 2, 16, 2_068, &[("NAME", "Packets")], Some(kind));
            let (props, tags, _) = run(&f);
            assert_eq!(props.sample_rate, Some(44_100), "{:?}", kind);
            assert_eq!(props.channels, Some(2));
            assert_eq!(props.duration_ns, None, "{:?}: a packet count is not a frame count", kind);
            assert!(!props.duration_exact);
            assert_eq!(tags.get("TITLE"), Some("Packets"), "tags are unaffected");
        }
    }

    /// The ID3 chunk must beat the text chunks whichever order they sit in — the whole reason
    /// `walk` makes two passes.
    #[test]
    fn id3_wins_over_the_text_chunks_in_either_order() {
        let tag = id3v2(3, &[("TIT2", "ID3 Title"), ("TALB", "ID3 Album")], None);
        for id3_first in [true, false] {
            let mut chunks = vec![comm_chunk(44_100, 2, 16, 44_100, None)];
            let text = aiff_chunk(b"NAME", b"Text Title");
            let id3 = aiff_chunk(b"ID3 ", &tag);
            if id3_first {
                chunks.push(id3);
                chunks.push(text);
            } else {
                chunks.push(text);
                chunks.push(id3);
            }
            let f = form(b"AIFF", &chunks);
            let (_, tags, _) = run(&f);
            assert_eq!(tags.get("TITLE"), Some("ID3 Title"), "id3_first={id3_first}");
            assert_eq!(tags.get("ALBUM"), Some("ID3 Album"));
        }
    }

    #[test]
    fn odd_sized_chunks_stay_word_aligned() {
        // "abc" is 3 bytes — odd, so a pad byte follows that is not counted in ckSize.
        let f = aiff(b"AIFF", 8_000, 1, 8, 4_000, &[("NAME", "abc"), ("AUTH", "x")], None);
        let (props, tags, _) = run(&f);
        assert_eq!(tags.get("TITLE"), Some("abc"));
        assert_eq!(tags.get("ARTIST"), Some("x"), "the walk stayed aligned past the pad byte");
        assert_eq!(props.duration_ns, Some(500_000_000));
    }

    #[test]
    fn a_tag_after_the_audio_asks_to_resume_at_the_right_offset() {
        let tag = id3v2(3, &[("TIT2", "Tail Tag")], None);
        let id3 = aiff_chunk(b"ID3 ", &tag);
        let f = form(
            b"AIFF",
            &[
                comm_chunk(44_100, 2, 16, 44_100, None),
                aiff_chunk(b"SSND", &vec![0u8; 4096]),
                id3.clone(),
            ],
        );
        let head_len = f.len() - id3.len();
        let (props, tags, resume) = run_windowed(&f[..head_len], f.len() as u64);
        assert!(tags.is_empty(), "the tag is past the window");
        assert_eq!(props.duration_ns, Some(1_000_000_000), "props came from the head");
        assert_eq!(resume, Some(head_len as u64));
        assert_eq!(plan(&f[..head_len], 0, f.len() as u64), Some(head_len as u64));

        // Resuming there finds it.
        let pool = Pool::bounded(4096, 8);
        let arena = Arena::default();
        let mut sink = TagListSink::new(&pool);
        let mut p2 = Props::default();
        let again =
            walk(&f[head_len..], head_len as u64, f.len() as u64, &mut p2, &arena, &mut sink);
        assert_eq!(again, None);
        assert_eq!(sink.finish().get("TITLE"), Some("Tail Tag"));
    }

    /// The window ends *inside* a text chunk, not before it: the header parses, the body does
    /// not. Reported as a resume, or the tag is silently lost.
    #[test]
    fn a_body_straddling_the_window_end_asks_to_resume() {
        let name = aiff_chunk(b"NAME", b"Straddler");
        let f = form(
            b"AIFF",
            &[comm_chunk(48_000, 2, 24, 48_000, None), aiff_chunk(b"SSND", &[0u8; 64]), name.clone()],
        );
        let at = f.len() - name.len();
        let cut = at + 10; // header plus two body bytes
        let (_, tags, resume) = run_windowed(&f[..cut], f.len() as u64);
        assert_eq!(resume, Some(at as u64));
        assert!(tags.is_empty(), "a half-read text chunk yields nothing, not garbage");
    }

    #[test]
    fn a_bad_sample_rate_costs_the_duration_and_nothing_else() {
        // A COMM whose 80-bit rate is +inf: channels still land, the duration does not.
        let mut comm = vec![0u8; COMM_LEN];
        comm[0..2].copy_from_slice(&2u16.to_be_bytes());
        comm[2..6].copy_from_slice(&44_100u32.to_be_bytes());
        comm[6..8].copy_from_slice(&16u16.to_be_bytes());
        comm[8..18].copy_from_slice(&[0x7F, 0xFF, 0x80, 0, 0, 0, 0, 0, 0, 0]);
        let f = form(b"AIFF", &[aiff_chunk(b"COMM", &comm), aiff_chunk(b"NAME", b"Broken")]);
        let (props, tags, _) = run(&f);
        assert_eq!(props.channels, Some(2));
        assert_eq!(props.sample_rate, None);
        assert_eq!(props.duration_ns, None);
        assert!(!props.duration_exact);
        assert_eq!(tags.get("TITLE"), Some("Broken"));
    }

    #[test]
    fn latin1_text_is_transcoded() {
        let f = form(
            b"AIFF",
            &[comm_chunk(44_100, 2, 16, 44_100, None), aiff_chunk(b"AUTH", b"Bj\xf6rk")],
        );
        let (_, tags, _) = run(&f);
        assert_eq!(tags.get("ARTIST"), Some("Björk"));
    }

    #[test]
    fn absurd_chunk_size_does_not_hang_or_overflow() {
        // An `SSND` claiming u32::MAX bytes in a 40-byte file.
        let mut f = b"FORM\x00\x00\x00\x00AIFF".to_vec();
        f.extend_from_slice(b"SSND\xff\xff\xff\xff");
        f.extend_from_slice(&[0u8; 16]);
        let (_, tags, resume) = run(&f);
        assert!(tags.is_empty());
        assert_eq!(resume, None, "the claimed end is past EOF — nothing to resume");
        // A zero-size chunk repeated must terminate, not spin.
        let mut z = b"FORM\x00\x00\x00\x00AIFF".to_vec();
        for _ in 0..64 {
            z.extend_from_slice(b"JUNK\x00\x00\x00\x00");
        }
        let (_, _, r) = run(&z);
        assert_eq!(r, None);
    }

    // The two sweeps below use a deliberately tiny `SSND` (64 frames): they are quadratic in
    // the file length and the audio payload is skipped by size, so a bigger one would buy no
    // coverage at all — every code path lives in the chunk headers and the metadata bodies.
    #[test]
    fn truncation_never_panics() {
        let f = aiff(
            b"AIFF",
            44_100,
            2,
            16,
            64,
            &[("NAME", "T"), ("AUTH", "A"), ("ANNO", "a comment"), ("(c) ", "2020 x")],
            None,
        );
        for n in 0..f.len() {
            let _ = run_windowed(&f[..n], n as u64);
            let _ = plan(&f[..n], 0, n as u64);
        }
    }

    #[test]
    fn single_byte_corruption_never_panics() {
        let base = aiff(b"AIFC", 48_000, 2, 24, 64, &[("NAME", "Corrupt me")], Some(*b"sowt"));
        for i in 0..base.len() {
            for bit in [0x01u8, 0x80, 0xFF] {
                let mut f = base.clone();
                f[i] ^= bit;
                let _ = run(&f);
                let _ = plan(&f, 0, f.len() as u64);
            }
        }
    }

    /// An AIFF whose `ID3 ` chunk carries cover art, plus the same file with the tag behind
    /// the audio — the two shapes the engine plans differently.
    #[test]
    fn id3_chunk_with_a_picture_lands_from_either_position() {
        for tail in [false, true] {
            let f = crate::fixture::aiff_id3(
                44_100,
                2,
                4_410,
                &[("NAME", "Text Title")],
                &[("TIT2", "Tag Title"), ("TPE1", "Tag Artist")],
                tail,
            );
            let (props, tags, _) = run(&f);
            assert_eq!(props.duration_ns, Some(100_000_000), "4410 frames at 44.1 kHz");
            assert_eq!(tags.get("TITLE"), Some("Tag Title"), "tail={tail}");
            assert_eq!(tags.get("ARTIST"), Some("Tag Artist"));
        }
    }
}
