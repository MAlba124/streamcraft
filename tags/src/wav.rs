//! RIFF/WAVE metadata: the `fmt ` and `data` chunks for properties, `LIST`/`INFO` for tags.
//!
//! Structure per the *Multimedia Programming Interface and Data Specifications 1.0*
//! (IBM & Microsoft, August 1991) — the RIFF specification proper. A RIFF file is
//! `"RIFF" <u32 size> <formType> <chunk>*`, each chunk `<ckID:4> <u32 ckSize> <data>`, and
//! **every chunk is word-aligned**: an odd `ckSize` is followed by one pad byte that is not
//! counted in the size (§"RIFF Chunks"). The WAVE form is formType `"WAVE"`.
//!
//! The walk is *windowed*: it parses a byte range starting at a known chunk boundary and,
//! if it runs out of window with file left, reports the absolute offset of the next chunk
//! so the engine can read one more extent there. That is what makes an `INFO` list written
//! *after* a 500 MB `data` chunk — the common layout, since the tagger appends rather than
//! rewrites — cost one extra positioned read instead of a whole-file scan.

use profluens_core::event::TagSink;

use crate::store::decode_text;
use crate::Props;

/// Scratch for the Latin-1 → UTF-8 fallback in [`decode_text`]. INFO fields are titles and
/// comments; 1 KiB of transcoded output (≥ 512 source bytes) is well past any real one.
const TEXT_SCRATCH: usize = 1024;

/// Bytes of RIFF header before the first chunk: `"RIFF" <u32 size> "WAVE"` (§"RIFF Form").
const RIFF_HEADER: u64 = 12;

/// Walk the chunks in `window`, which begins at absolute file offset `base` **on a chunk
/// boundary** (or at the file start, where the 12-byte RIFF header is skipped first).
///
/// Fills `props` and pushes tags into `sink`. Returns the absolute offset of the next chunk
/// when the walk ran past the end of the window with file left to read — the engine's cue
/// to fetch one more extent — and `None` when the walk finished or the file is exhausted.
pub(crate) fn walk(
    window: &[u8],
    base: u64,
    file_len: u64,
    props: &mut Props,
    sink: &mut impl TagSink,
) -> Option<u64> {
    let mut abs = if base == 0 { RIFF_HEADER } else { base };
    // Carried across chunks because `data` may precede `fmt ` in a malformed file, and the
    // duration needs both.
    let mut byte_rate: Option<u32> = None;
    let mut data_len: Option<u64> = None;
    let mut exact = false;
    let mut resume: Option<u64> = None;

    while let Some(w) = abs.checked_sub(base).and_then(|d| usize::try_from(d).ok()) {
        let Some(header) = window.get(w..w + 8) else {
            // The next chunk header does not fit the window: resume there.
            resume = Some(abs);
            break;
        };
        let id: [u8; 4] = [header[0], header[1], header[2], header[3]];
        let size =
            u64::from(u32::from_le_bytes([header[4], header[5], header[6], header[7]]));
        let body_at = w + 8;
        let body = window.get(body_at..).map(|t| &t[..t.len().min(size as usize)]);

        // `data` is skipped by size and never read, but the chunks this walk *parses* need
        // their whole body: half a `LIST` is not half its tags, it is none of them. When the
        // window cuts one short and the file really does hold the rest, resume at this
        // chunk — which is how an `INFO` list straddling the end of the prefix read is found.
        let full = (window.len().saturating_sub(body_at) as u64) >= size;
        if matches!(&id, b"fmt " | b"LIST") && !full && abs + 8 + size <= file_len {
            resume = Some(abs);
            break;
        }

        match &id {
            // §"WAVE Format Chunk": wFormatTag, nChannels, nSamplesPerSec, nAvgBytesPerSec,
            // nBlockAlign, wBitsPerSample — all little-endian.
            b"fmt " => {
                if let Some(f) = body.filter(|b| b.len() >= 16) {
                    let tag = u16::from_le_bytes([f[0], f[1]]);
                    props.channels = Some(u32::from(u16::from_le_bytes([f[2], f[3]])));
                    props.sample_rate = Some(u32::from_le_bytes([f[4], f[5], f[6], f[7]]));
                    byte_rate = Some(u32::from_le_bytes([f[8], f[9], f[10], f[11]]));
                    // `nAvgBytesPerSec` is exact only for a constant-rate encoding. WAVE_FORMAT_PCM
                    // (0x0001), IEEE_FLOAT (0x0003) and EXTENSIBLE (0xFFFE, which wraps one of
                    // those) are; an ADPCM or MP3-in-WAVE payload makes it an average, and the
                    // duration derived from it an estimate.
                    exact = matches!(tag, 0x0001 | 0x0003 | 0xFFFE);
                }
            }
            // §"Data Chunk": the sample data. Its *size* is all the duration needs — the
            // payload itself is never read. Clamped to what the file actually holds, so a
            // truncated (or lying) file reports the duration it really has.
            b"data" => data_len = Some(size.min(file_len.saturating_sub(abs + 8))),
            // §"LIST Chunk" with list type "INFO" (§"INFO List Chunk").
            b"LIST" => {
                if let Some(l) = body.filter(|b| b.len() >= 4 && &b[..4] == b"INFO") {
                    info(&l[4..], sink);
                }
            }
            _ => {}
        }

        // Word alignment: an odd chunk size carries a pad byte outside the size (§"RIFF Chunks").
        let advance = 8 + size + (size & 1);
        let Some(next) = abs.checked_add(advance) else { break };
        // `next <= abs` guards a zero-size chunk looping forever; `next >= file_len` is the
        // ordinary end of the walk — either way there is nothing left to resume at.
        if next <= abs || next >= file_len {
            break;
        }
        abs = next;
    }

    if let (Some(len), Some(rate)) = (data_len, byte_rate.filter(|r| *r > 0)) {
        // duration = data bytes / bytes-per-second, in nanoseconds. u128 because
        // `len * 1e9` overflows u64 for anything past ~18 GB of payload.
        props.duration_ns =
            u64::try_from(u128::from(len) * 1_000_000_000 / u128::from(rate)).ok();
        props.duration_exact = exact;
    }

    resume.filter(|at| *at < file_len)
}

/// The body of a `LIST`/`INFO` chunk: sub-chunks `<ckID:4> <u32 ckSize> <ZSTR>`, word-aligned
/// like any RIFF chunk (§"INFO List Chunk"). `ZSTR` is a NUL-terminated string.
fn info(body: &[u8], sink: &mut impl TagSink) {
    let mut p = 0usize;
    let mut scratch = [0u8; TEXT_SCRATCH];
    while let Some(h) = body.get(p..p + 8) {
        let id: [u8; 4] = [h[0], h[1], h[2], h[3]];
        let size = u32::from_le_bytes([h[4], h[5], h[6], h[7]]) as usize;
        let Some(raw) = body.get(p + 8..p + 8 + size) else { break };
        if let Some(key) = key_for(&id) {
            // ZSTR: everything up to the first NUL (trailing NULs are padding, not content).
            let text = raw.split(|b| *b == 0).next().unwrap_or(raw);
            let text = decode_text(text, &mut scratch);
            if !text.is_empty() {
                sink.text(key, text);
            }
        }
        let Some(next) = p.checked_add(8 + size + (size & 1)) else { break };
        if next <= p {
            break;
        }
        p = next;
    }
}

/// INFO chunk id → the canonical uppercase Vorbis-comment key core's [`TagSink`] documents.
///
/// The ids are the registered ones from the 1991 RIFF specification's INFO list
/// (§"INFO List Chunk"), except `ITRK`, which is not registered but is what every tagger in
/// the wild writes for a track number (the registered `IPRT`, "part of a set", is accepted
/// as its alias).
fn key_for(id: &[u8; 4]) -> Option<&'static str> {
    Some(match id {
        b"INAM" => "TITLE",             // Name
        b"IART" => "ARTIST",            // Artist
        b"IPRD" => "ALBUM",             // Product (the collection the file belongs to)
        b"ITRK" | b"IPRT" => "TRACKNUMBER", // Track / Part
        b"IGNR" => "GENRE",             // Genre
        b"ICRD" => "DATE",              // Creation date
        b"ICMT" => "COMMENT",           // Comments
        b"ISFT" => "ENCODER",           // Software that created the file
        b"ICOP" => "COPYRIGHT",         // Copyright
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;
    use profluens_core::event::{TagList, TagListSink};
    use profluens_core::memory::Pool;

    use crate::fixture::{fmt_chunk, info_chunk, riff, riff_chunk as chunk};

    fn run(bytes: &[u8]) -> (Props, TagList, Option<u64>) {
        let pool = Pool::bounded(4096, 4);
        let mut sink = TagListSink::new(&pool);
        let mut props = Props::default();
        let resume = walk(bytes, 0, bytes.len() as u64, &mut props, &mut sink);
        (props, sink.finish(), resume)
    }

    #[test]
    fn props_and_inline_info() {
        let data = vec![0u8; 44100 * 4]; // exactly 1 s of 44.1 kHz stereo s16
        let f = riff(&[
            fmt_chunk(44_100, 2, 16),
            info_chunk(&[("INAM", "Enough"), ("IART", "Solar Fields")]),
            chunk(b"data", &data),
        ]);
        let (props, tags, resume) = run(&f);
        assert_eq!(props.sample_rate, Some(44_100));
        assert_eq!(props.channels, Some(2));
        assert_eq!(props.duration_ns, Some(1_000_000_000));
        assert!(props.duration_exact);
        assert_eq!(tags.get("TITLE"), Some("Enough"));
        assert_eq!(tags.get("ARTIST"), Some("Solar Fields"));
        assert_eq!(resume, None);
    }

    #[test]
    fn odd_sized_chunks_stay_word_aligned() {
        let f = riff(&[
            fmt_chunk(8_000, 1, 8),
            info_chunk(&[("INAM", "ab"), ("IGNR", "x")]), // "ab\0" is odd → pad byte
            chunk(b"data", &[0u8; 8_000]),
        ]);
        let (props, tags, _) = run(&f);
        assert_eq!(tags.get("TITLE"), Some("ab"));
        assert_eq!(tags.get("GENRE"), Some("x"));
        assert_eq!(props.duration_ns, Some(1_000_000_000));
    }

    #[test]
    fn info_after_data_asks_to_resume_at_the_right_offset() {
        let data = vec![0u8; 4096];
        let info = info_chunk(&[("INAM", "Tail")]);
        let f = riff(&[fmt_chunk(44_100, 2, 16), chunk(b"data", &data), info.clone()]);
        // Hand the walk only the head: it must stop and point at the INFO chunk.
        let head_len = f.len() - info.len();
        let (props, tags, resume) = {
            let pool = Pool::bounded(4096, 4);
            let mut sink = TagListSink::new(&pool);
            let mut props = Props::default();
            let r = walk(&f[..head_len], 0, f.len() as u64, &mut props, &mut sink);
            (props, sink.finish(), r)
        };
        assert!(tags.is_empty(), "tags are past the window");
        assert!(props.duration_ns.is_some(), "props came from the head");
        assert_eq!(resume, Some(head_len as u64));

        // Resuming there finds them.
        let pool = Pool::bounded(4096, 4);
        let mut sink = TagListSink::new(&pool);
        let mut p2 = Props::default();
        let again = walk(&f[head_len..], head_len as u64, f.len() as u64, &mut p2, &mut sink);
        assert_eq!(again, None);
        assert_eq!(sink.finish().get("TITLE"), Some("Tail"));
    }

    /// The window ends *inside* the `LIST` chunk, not before it: the header parses, the body
    /// does not. Reported as a resume at the chunk, or its tags are silently lost — which is
    /// exactly what a 4 KiB prefix against a 4.1 KiB file does.
    #[test]
    fn a_list_body_straddling_the_window_end_asks_to_resume() {
        let info = info_chunk(&[("INAM", "Straddler")]);
        let f = riff(&[fmt_chunk(44_100, 2, 16), chunk(b"data", &[0u8; 64]), info.clone()]);
        let list_at = f.len() - info.len();
        // Keep the LIST's 8-byte header plus 4 bytes of its body inside the window.
        let cut = list_at + 12;
        let pool = Pool::bounded(4096, 4);
        let mut sink = TagListSink::new(&pool);
        let mut props = Props::default();
        let resume = walk(&f[..cut], 0, f.len() as u64, &mut props, &mut sink);
        assert_eq!(resume, Some(list_at as u64));
        assert!(sink.finish().is_empty(), "a half-read LIST yields nothing, not garbage");

        // And the resumed window has them.
        let mut sink = TagListSink::new(&pool);
        let mut p2 = Props::default();
        walk(&f[list_at..], list_at as u64, f.len() as u64, &mut p2, &mut sink);
        assert_eq!(sink.finish().get("TITLE"), Some("Straddler"));
    }

    #[test]
    fn truncation_never_panics() {
        let f = riff(&[
            fmt_chunk(48_000, 2, 24),
            info_chunk(&[("INAM", "T"), ("ICMT", "a comment")]),
            chunk(b"data", &[0u8; 1000]),
        ]);
        for n in 0..f.len() {
            let pool = Pool::bounded(4096, 4);
            let mut sink = TagListSink::new(&pool);
            let mut props = Props::default();
            let _ = walk(&f[..n], 0, n as u64, &mut props, &mut sink);
        }
    }

    #[test]
    fn absurd_chunk_size_does_not_hang_or_overflow() {
        // A `data` chunk claiming u32::MAX bytes in a 40-byte file.
        let mut f = b"RIFF\x00\x00\x00\x00WAVE".to_vec();
        f.extend_from_slice(b"data\xff\xff\xff\xff");
        f.extend_from_slice(&[0u8; 16]);
        let (_, tags, resume) = run(&f);
        assert!(tags.is_empty());
        assert_eq!(resume, None, "the claimed end is past EOF — nothing to resume");
    }

    #[test]
    fn latin1_info_text() {
        let f = riff(&[
            fmt_chunk(44_100, 2, 16),
            {
                // Hand-build an INFO entry with raw Latin-1 bytes.
                let mut b = b"INFO".to_vec();
                b.extend_from_slice(&chunk(b"IART", b"Bj\xf6rk\0"));
                chunk(b"LIST", &b)
            },
            chunk(b"data", &[0u8; 4]),
        ]);
        let (_, tags, _) = run(&f);
        assert_eq!(tags.get("ARTIST"), Some("Björk"));
    }
}
