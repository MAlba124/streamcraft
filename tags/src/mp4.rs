//! MP4 / M4A metadata: finding `moov`, then handing it to [`pf_mp4::ilst`].
//!
//! An ISO base media file (ISO/IEC 14496-12) is a flat sequence of top-level boxes (§4.2),
//! and everything a scanner wants — the iTunes tag list, the duration, the sample rate —
//! lives inside exactly one of them, `moov`. Where that box sits is the whole problem:
//!
//! * after a `faststart` remux it follows `ftyp` immediately, so the prefix read already
//!   holds it and the scan costs **one read**;
//! * straight out of most encoders it *trails* the `mdat`, which may be gigabytes, so the
//!   prefix holds only the box walk that steps over it and the scan costs **two**.
//!
//! [`pf_mp4::ilst::locate_moov`] answers both from a bounded prefix, and its
//! [`Beyond`](pf_mp4::ilst::MoovExtent::Beyond) arm is defined to be re-fed to itself: read
//! the range it names, ask again, and either the answer is the box (`InPrefix { offset: 0 }`)
//! or it is a shorter unread range. Each hop is strictly closer to the answer, so the loop
//! terminates; the engine's extent budget bounds it anyway, because a scanner must not let a
//! crafted file choose how many reads it costs. Two hops is the real ceiling —
//! `ftyp mdat moov` needs one, `faststart` needs none.
//!
//! No tail read: `moov` is found structurally, never by scanning backwards from EOF.

use profluens_core::event::TagSink;
use profluens_core::memory::Arena;

use crate::Props;

/// What the box walk concluded about a window.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Plan {
    /// The complete `moov` box is `window[at..at + len]`.
    Here { at: usize, len: usize },
    /// Read `file[at..at + len]` and walk again (the `Beyond` protocol above).
    Need { at: u64, len: u64 },
    /// No `moov` reachable: not ISO-BMFF, a malformed box length, or a file that has none.
    None,
}

/// Walk `window` — which begins at absolute file offset `base` **on a top-level box
/// boundary** — and report where `moov` is.
///
/// `locate_moov` measures from the start of the slice it is given, so a window that begins
/// mid-file is handed the *remaining* file length and its answers are rebased here. That is
/// exactly the re-feed the `Beyond` protocol asks for.
pub(crate) fn locate(window: &[u8], base: u64, file_len: u64) -> Plan {
    let Some(remaining) = file_len.checked_sub(base) else { return Plan::None };
    match pf_mp4::ilst::locate_moov(window, remaining) {
        pf_mp4::ilst::MoovExtent::InPrefix { offset, len } => Plan::Here { at: offset, len },
        pf_mp4::ilst::MoovExtent::Beyond { offset, len } => Plan::Need { at: base + offset, len },
        pf_mp4::ilst::MoovExtent::NotFound => Plan::None,
    }
}

/// Parse the `moov` at `window[at..at + len]`: props first, then tags into `sink`.
///
/// Cover art (`covr`) reaches the sink borrowed from `window`, which is the buffer the bytes
/// were read into — so [`ArenaSink`](crate::store::ArenaSink) takes a refcounted sub-view of
/// it and an embedded JPEG is never copied.
pub(crate) fn parse(
    window: &[u8],
    at: usize,
    len: usize,
    arena: &Arena,
    sink: &mut impl TagSink,
) -> Props {
    let mut props = Props::default();
    let Some(moov) = at.checked_add(len).and_then(|end| window.get(at..end)) else { return props };
    let p = pf_mp4::ilst::props_from_moov(moov);
    props.duration_ns = p.duration_ns;
    // A duration read out of `mdhd`/`mvhd` is a declared sample count over a declared
    // timescale, not an estimate — the one caveat (`mdhd` still counts the codec priming an
    // edit list would trim) is a definition question, not a guess. See `props_from_moov`.
    props.duration_exact = p.duration_ns.is_some();
    props.sample_rate = p.sample_rate;
    props.channels = p.channels;
    pf_mp4::ilst::parse_moov_tags(moov, arena, sink);
    props
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;
    use profluens_core::event::TagListSink;
    use profluens_core::memory::Pool;

    use crate::fixture::m4a;

    fn run(f: &[u8]) -> (Props, profluens_core::event::TagList) {
        let pool = Pool::bounded(4096, 8);
        let arena = Arena::default();
        let mut sink = TagListSink::new(&pool);
        let props = match locate(f, 0, f.len() as u64) {
            Plan::Here { at, len } => parse(f, at, len, &arena, &mut sink),
            other => panic!("expected the moov in the prefix, got {other:?}"),
        };
        (props, sink.finish())
    }

    #[test]
    fn faststart_moov_is_found_in_the_prefix() {
        let f = m4a(44_100, 2, 44_100 * 3, &[("©nam", "Title"), ("©ART", "Artist")], Some(7), None, true);
        let (props, tags) = run(&f);
        assert_eq!(props.sample_rate, Some(44_100));
        assert_eq!(props.channels, Some(2));
        assert_eq!(props.duration_ns, Some(3_000_000_000));
        assert!(props.duration_exact);
        assert_eq!(tags.get("TITLE"), Some("Title"));
        assert_eq!(tags.get("ARTIST"), Some("Artist"));
        assert_eq!(tags.get("TRACKNUMBER"), Some("7"));
    }

    #[test]
    fn a_trailing_moov_asks_for_the_unread_range() {
        let f = m4a(48_000, 1, 48_000, &[("©nam", "Tail")], None, None, false);
        // Hand the walk only the head: it must ask for the range where `moov` lives.
        let head = 64.min(f.len());
        let Plan::Need { at, len } = locate(&f[..head], 0, f.len() as u64) else {
            panic!("a trailing moov must report a range to read")
        };
        assert!(at >= head as u64, "the range starts past what we already hold");

        // Re-feed exactly that range, as the protocol documents.
        let chunk = &f[at as usize..(at + len) as usize];
        let Plan::Here { at: rel, len } = locate(chunk, at, f.len() as u64) else {
            panic!("the re-read range must contain the moov")
        };
        let pool = Pool::bounded(4096, 8);
        let arena = Arena::default();
        let mut sink = TagListSink::new(&pool);
        let props = parse(chunk, rel, len, &arena, &mut sink);
        assert_eq!(props.duration_ns, Some(1_000_000_000));
        assert_eq!(sink.finish().get("TITLE"), Some("Tail"));
    }

    #[test]
    fn cover_art_is_borrowed_from_the_window() {
        let art = vec![0x5Au8; 2_048];
        let f = m4a(44_100, 2, 44_100, &[("©nam", "Art")], None, Some(("image/jpeg", &art)), true);
        let (_, tags) = run(&f);
        assert_eq!(tags.pictures().len(), 1);
        assert_eq!(&*tags.pictures()[0].mime, "image/jpeg");
        assert_eq!(tags.pictures()[0].data.data(), &art[..]);
    }

    #[test]
    fn truncation_never_panics() {
        let f = m4a(44_100, 2, 44_100, &[("©nam", "T")], Some(2), Some(("image/png", &[3u8; 500])), true);
        let pool = Pool::bounded(4096, 8);
        for n in 0..f.len() {
            let arena = Arena::default();
            let mut sink = TagListSink::new(&pool);
            if let Plan::Here { at, len } = locate(&f[..n], 0, n as u64) {
                let _ = parse(&f[..n], at, len, &arena, &mut sink);
            }
        }
    }
}
