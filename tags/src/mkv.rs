//! Matroska / WebM metadata: locating the Segment's metadata masters, then handing them to
//! [`pf_mkv::tags`].
//!
//! A Matroska Segment (RFC 9559 §4) is a flat run of level-1 masters, and unlike every other
//! format this scanner reads, the ones it wants are not all in one place: `Info` (the duration)
//! and `Tracks` (the rate and channel count) sit at the front, ahead of the first Cluster, while
//! `Tags` and `Attachments` are allowed anywhere and are routinely written *behind* the frames,
//! because a single-pass muxer does not know them until it is finished.
//!
//! What makes that affordable is the `SeekHead` (§5.1.1) — an index at the *start* of the
//! Segment naming each master's Segment Position. [`pf_mkv::tags::index_segment`] reads it out
//! of the prefix this scanner has already paid for, so a tag block at the end of an 80 MB file
//! is one positioned read away instead of a scan. There is no tail read: the metadata is found
//! structurally, never by guessing backwards from EOF.
//!
//! ## What it costs
//!
//! **One read** for every file that keeps its metadata in front of the frames, which is the
//! common layout and was *all 24* of the WebM files in the reference library — their SeekHead,
//! Info, Tracks and (where present) Tags all live in the first 600 bytes. A file that trails its
//! `Tags` or `Attachments` costs one more, and [`plan`] asks for a single range spanning both
//! when they are adjacent, which is how every muxer that writes them writes them.
//!
//! ## Allocation
//!
//! None beyond the read buffer. Tag text reaches the sink borrowed from that buffer, and an
//! attached cover image is handed over as a sub-slice of it — so, as with FLAC and MP4, embedded
//! art is never copied out of the bytes it was read into.

use profluens_core::event::TagSink;

use pf_mkv::tags::{self, Extent, SegmentIndex};

use crate::Props;

/// How much to read at a metadata offset whose element header is not yet in hand. The
/// `SeekHead` gives a position but no length (§5.1.1), so the first read at a trailing `Tags`
/// is a guess; 64 KiB covers any real tag block and most cover art in one op, and the element
/// header it lands on states the true length for the rare second.
const REACH: u64 = 64 * 1024;

/// Ceiling on one metadata range, mirroring the engine's own `MAX_METADATA`. A crafted file can
/// claim a gigabyte of `Attachments`; past this the scan keeps what it already has.
const MAX_META: u64 = 32 << 20;

/// A window the scanner already holds: its bytes, and the file offset of their first byte.
type View<'a> = (&'a [u8], u64);

/// The positioned read this file still wants, as `(offset, len)`, or `None` when everything the
/// scanner can reach is already in hand.
///
/// `views` are the windows held so far — the prefix, and the extent once there is one. Each
/// metadata master is looked up in them; one that is fully present is settled, one that is not
/// contributes its range to the request. The ranges are merged into a single read when they sit
/// within [`MAX_META`] of each other (`Tags` and `Attachments` are written adjacently, so in
/// practice one read fetches both) and otherwise only the earliest is asked for, so a request is
/// never the size of the file.
///
/// The caller must stop asking once a request repeats — see `Scanner::next_extent`, which gates
/// on the request reaching bytes not already held, exactly as the MP4 walk does.
pub(crate) fn plan(views: &[View<'_>], file_len: u64) -> Option<(u64, u64)> {
    let index = index_of(views)?;
    // The base-0 window on its own. A master that is complete *in the prefix* is settled for
    // good, because the prefix is never given back; one that is not must be covered by the
    // single extent this format gets.
    let prefix = &views[..1];

    let mut lo = u64::MAX;
    let mut hi = 0u64;
    let mut incomplete = false;
    for at in [index.info, index.tracks, index.tags, index.attachments].into_iter().flatten() {
        if at >= file_len {
            continue; // a stale or hostile SeekHead position, pointing off the end
        }
        if span(prefix, at).1 {
            continue; // already in hand, and always will be
        }
        // **Anchored at `at`, not at "the earliest thing still missing".** A slot holds one
        // extent, so the next read replaces the current one: a second request that started
        // past a master the first read already fetched would throw that master away. Every
        // out-of-prefix master therefore stays inside the range, whether or not this pass
        // still needs it, which is what makes the second read a superset of the first.
        let (bytes, complete) = span(views, at);
        lo = lo.min(at);
        hi = hi.max(at.saturating_add(bytes));
        incomplete |= !complete;
    }
    if lo == u64::MAX || !incomplete {
        return None; // everything reachable is in hand
    }
    // A span past the ceiling means the file's metadata does not fit one read. Take the
    // earliest master — `Tags` is written before `Attachments` — on the principle `ogg.rs`
    // states: tags without a cover beat no tags at all.
    let len = hi.saturating_sub(lo).min(MAX_META);
    Some((lo, len.min(file_len.saturating_sub(lo))))
}

/// `(bytes needed from `at`, whether a held window already has the element in full)`.
///
/// [`Extent::Invalid`] means the header itself is not readable — either the window stops inside
/// it, or the position is junk. Both answer [`REACH`]: one read settles the first and bounds
/// the second, and the extent budget stops a position that never resolves.
fn span(views: &[View<'_>], at: u64) -> (u64, bool) {
    match view_at(views, at).map(tags::element_extent) {
        Some(Extent::Complete { len }) => (len as u64, true),
        Some(Extent::Short { total }) => (total.min(MAX_META), false),
        Some(Extent::Invalid) | None => (REACH, false),
    }
}

/// The bytes from absolute offset `at` onwards, out of whichever held window contains it.
fn view_at<'a>(views: &[View<'a>], at: u64) -> Option<&'a [u8]> {
    views.iter().find_map(|(bytes, base)| {
        let off = usize::try_from(at.checked_sub(*base)?).ok()?;
        // A window that merely *starts* before `at` is no use unless it reaches it.
        bytes.get(off..).filter(|w| !w.is_empty())
    })
}

/// The Segment index, which only the window starting at file offset 0 can produce — the walk
/// begins at the EBML Header (RFC 8794 §11.2.4).
fn index_of(views: &[View<'_>]) -> Option<SegmentIndex> {
    views.iter().find(|(_, base)| *base == 0).and_then(|(bytes, _)| tags::index_segment(bytes))
}

/// Parse everything the held windows reach: props from `Info`/`Tracks`, tags from `Tags`, cover
/// art from `Attachments`.
///
/// A master the scan never got to simply contributes nothing — a WebM whose trailing tag block
/// overran the extent budget still reports its duration and its format.
pub(crate) fn parse(views: &[View<'_>], sink: &mut impl TagSink) -> Props {
    let mut props = Props::default();
    let Some(index) = index_of(views) else { return props };

    if let Some(info) = index.info.and_then(|at| view_at(views, at)) {
        // Duration is `Info\Duration` (a float, in ticks) times `TimestampScale` (ns per tick),
        // both declared by the file — a stated length, not a bitrate estimate, so `exact`.
        props.duration_ns = tags::info_duration_ns(info);
        props.duration_exact = props.duration_ns.is_some();
    }
    if let Some(tracks) = index.tracks.and_then(|at| view_at(views, at)) {
        if let Some((rate, channels)) = tags::audio_format(tracks) {
            props.sample_rate = (rate != 0).then_some(rate);
            props.channels = (channels != 0).then_some(channels);
        }
    }
    if let Some(t) = index.tags.and_then(|at| view_at(views, at)) {
        tags::parse_tags(t, sink);
    }
    if let Some(a) = index.attachments.and_then(|at| view_at(views, at)) {
        // The sink's `src` is the same buffer, so an attached JPEG reaches the caller as a
        // refcounted sub-view of it rather than a copy.
        tags::parse_attachments(a, sink);
    }
    props
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;
    use profluens_core::event::TagListSink;
    use profluens_core::memory::Pool;

    use crate::fixture::{mkv, mkv_trailing};

    fn run(views: &[View<'_>]) -> (Props, profluens_core::event::TagList) {
        let pool = Pool::bounded(65_536, 8);
        let mut sink = TagListSink::new(&pool);
        let props = parse(views, &mut sink);
        (props, sink.finish())
    }

    /// The common layout — everything ahead of the frames — is one read: the prefix alone
    /// settles the plan and carries every master.
    #[test]
    fn a_front_loaded_file_needs_no_second_read() {
        let f = mkv(48_000.0, 2, 3_000.0, &[("TITLE", "Zonderlig"), ("ARTIST", "Sonderlig")], None);
        let views = [(&f[..], 0u64)];
        assert_eq!(plan(&views, f.len() as u64), None, "the prefix holds everything");

        let (props, tags) = run(&views);
        assert_eq!(props.duration_ns, Some(3_000_000_000));
        assert!(props.duration_exact);
        assert_eq!(props.sample_rate, Some(48_000));
        assert_eq!(props.channels, Some(2));
        assert_eq!(tags.get("TITLE"), Some("Zonderlig"));
        assert_eq!(tags.get("ARTIST"), Some("Sonderlig"));
    }

    /// A file whose `Tags` and `Attachments` trail the frames: the prefix's SeekHead names
    /// them, one read spans both, and the second pass parses them.
    #[test]
    fn trailing_masters_are_reached_in_one_extra_read() {
        let art = vec![0x5Au8; 4_096];
        let f = mkv_trailing(
            44_100.0,
            2,
            1_500.0,
            &[("TITLE", "Far"), ("ALBUM", "Behind the clusters")],
            Some(("image/jpeg", &art)),
            200_000,
        );
        // The scanner holds only its prefix.
        let prefix_len = 128 * 1024;
        let prefix: View = (&f[..prefix_len], 0);

        let (at, len) = plan(&[prefix], f.len() as u64).expect("a trailing read");
        assert!(at >= prefix_len as u64, "the request starts past what we hold");
        assert!(len <= MAX_META);

        // Duration and format already came out of the prefix.
        let (props, tags) = run(&[prefix]);
        assert_eq!(props.duration_ns, Some(1_500_000_000));
        assert_eq!(props.sample_rate, Some(44_100));
        assert_eq!(tags.get("TITLE"), None, "the tags are not in the prefix");

        // Re-feed the range the plan asked for, as the protocol documents.
        let extent: View = (&f[at as usize..(at + len) as usize], at);
        assert_eq!(plan(&[prefix, extent], f.len() as u64), None, "one read settles it");

        let (props, tags) = run(&[prefix, extent]);
        assert_eq!(props.duration_ns, Some(1_500_000_000));
        assert_eq!(tags.get("TITLE"), Some("Far"));
        assert_eq!(tags.get("ALBUM"), Some("Behind the clusters"));
        assert_eq!(tags.pictures().len(), 1);
        assert_eq!(&*tags.pictures()[0].mime, "image/jpeg");
        assert_eq!(tags.pictures()[0].data.data(), &art[..], "cover art is borrowed, not copied");
    }

    /// Cover art far bigger than one blind [`REACH`] costs a second read — and that read is a
    /// **superset** of the first, so the `Tags` the first one fetched are not thrown away when
    /// the slot's single extent is replaced.
    #[test]
    fn a_second_read_is_a_superset_of_the_first() {
        let art = vec![0x11u8; 200_000]; // far past REACH
        let f =
            mkv_trailing(48_000.0, 1, 10.0, &[("TITLE", "Big art")], Some(("image/png", &art)), 4_096);
        let prefix: View = (&f[..2_048], 0);

        // Pass 1: nothing but the SeekHead's positions is known, so the length is a guess.
        let (at, len) = plan(&[prefix], f.len() as u64).expect("a trailing read");
        assert!(len <= REACH + 4_096, "with no headers in hand the first request is REACH-sized");
        let first: View = (&f[at as usize..(at + len) as usize], at);

        // Pass 2: the headers in that window state the real sizes.
        let (at2, len2) = plan(&[prefix, first], f.len() as u64).expect("the art needs more");
        assert_eq!(at2, at, "anchored at the same offset — the second read re-covers the tags");
        assert!(len2 > len && len2 > art.len() as u64, "and reaches past the art");

        let second: View = (&f[at2 as usize..(at2 + len2) as usize], at2);
        assert_eq!(plan(&[prefix, second], f.len() as u64), None, "two reads settle it");
        let (_, tags) = run(&[prefix, second]);
        assert_eq!(tags.get("TITLE"), Some("Big art"), "kept, though only the 2nd extent is held");
        assert_eq!(tags.pictures()[0].data.data().len(), art.len());
    }

    /// A window that stops inside an element names the length that would complete it, so the
    /// engine's top-up asks for the right amount rather than guessing again.
    #[test]
    fn a_window_stopping_inside_an_element_reports_its_true_length() {
        let art = vec![0x22u8; 100_000];
        let f = mkv_trailing(48_000.0, 1, 10.0, &[("TITLE", "T")], Some(("image/png", &art)), 512);
        let prefix: View = (&f[..1_024], 0);
        let (at, len) = plan(&[prefix], f.len() as u64).expect("a trailing read");

        // Hand back only the first 8 bytes of that range — enough for a header, not the data.
        let stub: View = (&f[at as usize..at as usize + 8], at);
        let (next_at, next_len) = plan(&[prefix, stub], f.len() as u64).expect("more is needed");
        assert_eq!(next_at, at, "the same range, measured rather than guessed");
        assert_eq!(next_len, len);
    }

    /// Nothing that is not a Matroska file yields a plan or any tags, and nothing panics.
    #[test]
    fn junk_is_declined() {
        for junk in [&b""[..], b"ID3\x04\x00", b"RIFF\x00\x00\x00\x00WAVE", &[0xFFu8; 64]] {
            let views = [(junk, 0u64)];
            assert_eq!(plan(&views, junk.len() as u64), None);
            let (props, tags) = run(&views);
            assert_eq!(props, Props::default());
            assert_eq!(tags.get("TITLE"), None);
            assert_eq!(tags.pictures().len(), 0);
        }
    }

    /// Every truncation of a full fixture — plan and parse — runs without panicking, which is
    /// exactly what a scanner reading a partial file does.
    #[test]
    fn truncation_never_panics() {
        let f = mkv_trailing(
            48_000.0,
            2,
            42.0,
            &[("TITLE", "T"), ("PART_NUMBER", "3")],
            Some(("image/png", &[7u8; 300])),
            1_024,
        );
        let pool = Pool::bounded(4_096, 8);
        for n in 0..f.len() {
            let views = [(&f[..n], 0u64)];
            let _ = plan(&views, n as u64);
            let mut sink = TagListSink::new(&pool);
            let _ = parse(&views, &mut sink);
        }
    }
}
