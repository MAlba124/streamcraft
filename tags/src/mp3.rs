//! MP3 metadata: the side-cars an elementary stream carries because it has no container.
//!
//! MPEG audio is a bare sequence of frames (ISO/IEC 11172-3 §2.4.1) with nowhere to put a
//! title, so every tag lives *outside* the audio, bolted on at one end or the other:
//!
//! ```text
//!   | ID3v2 tag | audio frames … | APEv2 tag | ID3v1 (128 B) | EOF
//! ```
//!
//! All three are optional. ID3v2 is at the very front (id3v2.4.0-structure §3.1); the
//! trailing pair is not MP3's at all — Monkey's Audio, WavPack and Musepack end their files
//! the same way — so measuring and parsing it lives in [`crate::tail`], and this module only
//! adds the ID3v2 head and the duration probe. The parsing itself is [`pf_mp3::id3`]'s and
//! the duration probe is [`pf_mp3::props`]'s.
//!
//! ## What it costs
//!
//! **Two reads for any real file.** The head window covers the ID3v2 tag *and* the first
//! audio frame behind it — one contiguous range, because [`pf_mp3::props::probe_props`]
//! reads the frame immediately following the tag — so a 128 KiB prefix answers both the tags
//! and the duration of every MP3 whose tag fits it. Only an oversized ID3v2 (cover art past
//! the prefix) costs a third: one exactly-sized re-read of tag + probe window. The trailing
//! tags then take the engine's single tail read, which is skipped outright when the prefix
//! already holds the whole file.
//!
//! Nothing here allocates: the tag bytes are parsed where they were read, values reach the
//! sink borrowed from them, and the arena is only touched where the format forces a rewrite
//! (see [`pf_mp3::id3`]'s module docs).

use profluens_core::event::TagSink;
use profluens_core::memory::Arena;

use crate::Props;

/// How much audio past the ID3v2 tag the head window should reach.
///
/// [`pf_mp3::props::probe_props`] needs the whole **first frame** — at most 1441 bytes, the
/// MPEG-1 Layer III maximum at 320 kbit/s and 32 kHz (ISO/IEC 11172-3 §2.4.2.3) — because
/// the Xing/VBRI header that states the exact frame count is buried inside that frame's
/// otherwise-unused payload. 8 KiB leaves room for the junk taggers leave between the tag
/// and the first sync word and still costs nothing: it is part of a read that had to happen.
const PROBE_BYTES: u64 = 8 * 1024;

/// Total length of the ID3v2 tag `window` starts with, or 0 when there is none — via
/// pf-mp3's own header reader (id3v2.4.0-structure §3.1), so the scanner and the parser can
/// never disagree about where the audio begins.
pub(crate) fn v2_len(window: &[u8]) -> u64 {
    pf_mp3::id3::v2_total_len(window).map_or(0, |n| n as u64)
}

/// The least a head window must hold behind the ID3v2 tag to be usable: one maximum-size
/// MPEG-1 Layer III frame (1441 bytes) with room to resync past a byte or two of junk.
/// Below this, `probe_props` can find a header but not the Xing/VBRI payload inside it.
const MIN_AUDIO: u64 = 2 * 1024;

/// The absolute file offset the head window must reach: past the ID3v2 tag *and* past the
/// first frame behind it. Clamped to the file.
pub(crate) fn head_end(window: &[u8], file_len: u64) -> u64 {
    v2_len(window).saturating_add(PROBE_BYTES).min(file_len)
}

/// Whether `window` is too short to answer both questions — i.e. whether the extra read is
/// worth issuing at all.
///
/// Deliberately *not* `window.len() < head_end(..)`: asking for [`PROBE_BYTES`] of audio is
/// the right size for a read that has to happen anyway, but it is the wrong trigger for one.
/// A bare 3 KiB MP3 read into a 4 KiB prefix already holds every byte that matters, and
/// re-reading it to reach an 8 KiB target would double the file's cost for nothing.
pub(crate) fn head_short(window: &[u8], file_len: u64) -> bool {
    v2_len(window).saturating_add(MIN_AUDIO).min(file_len) > window.len() as u64
}

/// Emit the leading ID3v2 tag from a window that starts at file offset 0.
pub(crate) fn emit_v2(head: &[u8], v2: u64, arena: &Arena, sink: &mut impl TagSink) {
    if v2 == 0 {
        return;
    }
    // Clamped to what was actually read: a tag whose declared size runs past the window (or
    // past the file) yields the frames that are there, not a panic.
    let end = usize::try_from(v2).unwrap_or(usize::MAX).min(head.len());
    pf_mp3::id3::parse_v2(&head[..end], arena, sink);
}

/// Properties from the first audio frame: `head` starts at file offset 0, the audio starts
/// `v2` bytes into it, and `audio_len` is the file minus every tag byte at either end (what
/// the constant-bitrate estimate divides).
pub(crate) fn props(head: &[u8], v2: u64, audio_len: u64) -> Props {
    let mut props = Props::default();
    let Some(audio) = usize::try_from(v2).ok().and_then(|at| head.get(at..)) else { return props };
    let Some(a) = pf_mp3::props::probe_props(audio, audio_len) else { return props };
    props.duration_ns = a.duration_ns;
    // `exact` is set only by a Xing/VBRI frame count; the CBR estimate is honestly labelled.
    props.duration_exact = a.exact;
    props.sample_rate = Some(a.sample_rate);
    props.channels = Some(a.channels);
    props
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;
    use profluens_core::event::TagListSink;
    use profluens_core::memory::Pool;

    use crate::fixture::{id3v2, mp3, mp3_frames};

    #[test]
    fn head_end_covers_the_tag_and_the_first_frame() {
        let tag = id3v2(3, &[("TIT2", "T")], None);
        let f = [tag.clone(), mp3_frames(44_100, 128, 4)].concat();
        assert_eq!(v2_len(&f), tag.len() as u64);
        assert_eq!(head_end(&f, f.len() as u64), (tag.len() as u64 + PROBE_BYTES).min(f.len() as u64));
        let bare = mp3_frames(44_100, 128, 4);
        assert_eq!(v2_len(&bare), 0);
    }

    #[test]
    fn only_a_window_missing_the_tag_or_the_first_frame_asks_for_more() {
        // A 20 KiB APIC: no prefix under 20 KiB can hold the tag, so all of them are short.
        let big = mp3(44_100, 128, 6, &[("TIT2", "T")], Some(("image/jpeg", &[7u8; 20_000])), false, None, &[]);
        let len = big.len() as u64;
        assert!(head_short(&big[..4096], len));
        assert!(!head_short(&big, len), "the whole file is never short");
        // A bare stream that fits a 4 KiB prefix must NOT cost a second read just because
        // the probe window is nominally 8 KiB.
        let small = mp3_frames(44_100, 128, 6); // ~2.5 KiB
        assert!(!head_short(&small, small.len() as u64));
        // …but a 1 KiB view of it is genuinely short: no whole first frame is guaranteed.
        assert!(head_short(&small[..1024], small.len() as u64));
    }

    #[test]
    fn tags_and_props_from_a_whole_file() {
        let f = mp3(
            44_100,
            128,
            20,
            &[("TIT2", "Head Tag"), ("TPE1", "An Artist")],
            None,
            false,
            Some(("Tail Title", "Tail Artist", "Tail Album", "1999", 3)),
            &[],
        );
        let pool = Pool::bounded(4096, 8);
        let arena = Arena::default();
        let mut sink = TagListSink::new(&pool);
        let v2 = v2_len(&f);
        emit_v2(&f, v2, &arena, &mut sink);
        let t = crate::tail::measure(&f, 0, f.len() as u64);
        crate::tail::emit(&f, 0, f.len() as u64, t, &arena, &mut sink);
        let tags = sink.finish();

        // ID3v2 wins over the ID3v1 field of the same name; v1-only fields still land.
        assert_eq!(tags.get("TITLE"), Some("Head Tag"));
        assert_eq!(tags.get("ARTIST"), Some("An Artist"));
        assert_eq!(tags.get("ALBUM"), Some("Tail Album"));
        assert_eq!(tags.get("DATE"), Some("1999"));

        let p = props(&f, v2, f.len() as u64 - v2 - t.total());
        assert_eq!(p.sample_rate, Some(44_100));
        assert_eq!(p.channels, Some(2));
        assert!(p.duration_ns.is_some());
    }

    #[test]
    fn a_xing_header_makes_the_duration_exact() {
        let f = mp3(44_100, 128, 20, &[("TIT2", "X")], None, true, None, &[]);
        let v2 = v2_len(&f);
        let p = props(&f, v2, f.len() as u64 - v2);
        assert!(p.duration_exact, "a Xing frame count is authoritative");
        // 20 frames of 1152 samples at 44.1 kHz.
        assert_eq!(p.duration_ns, Some(20 * 1152 * 1_000_000_000 / 44_100));
    }

    #[test]
    fn truncation_never_panics() {
        let f = mp3(
            44_100,
            128,
            6,
            &[("TIT2", "T")],
            Some(("image/png", &[0x11u8; 300])),
            false,
            Some(("t", "a", "r", "2001", 1)),
            &[("REPLAYGAIN_TRACK_GAIN", "0.00 dB")],
        );
        let pool = Pool::bounded(4096, 8);
        for n in 0..f.len() {
            let arena = Arena::default();
            let mut sink = TagListSink::new(&pool);
            let v2 = v2_len(&f[..n]);
            emit_v2(&f[..n], v2, &arena, &mut sink);
            let t = crate::tail::measure(&f[..n], 0, n as u64);
            crate::tail::emit(&f[..n], 0, n as u64, t, &arena, &mut sink);
            let _ = props(&f[..n], v2, (n as u64).saturating_sub(v2));
            let _ = head_end(&f[..n], n as u64);
        }
    }
}
