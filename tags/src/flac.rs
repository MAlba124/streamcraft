//! Native FLAC metadata: the block chain, STREAMINFO properties, and tags.
//!
//! Structure per RFC 9639 ("Free Lossless Audio Codec"). §8.1: after the `fLaC` marker the
//! stream carries a chain of metadata blocks, each a 4-byte header — one *last-block* flag
//! bit, a 7-bit block type, and a 24-bit big-endian body length — followed by that body.
//! The chain is byte-aligned, so it is walked as plain bytes, independently of the bit-level
//! frame decoder.
//!
//! Only one job is done here: the **extent** of the metadata chain, so a file whose cover art
//! overruns the prefix read costs exactly one more positioned read and no guessing. The
//! contents are pf-flac's business —
//!
//! - [`pf_flac::tags::stream_info`] reads STREAMINFO (§8.2), the only place an exact duration
//!   lives, and
//! - [`pf_flac::tags::parse_into`] streams `VORBIS_COMMENT` (§8.6) and `PICTURE` (§8.8) into a
//!   [`TagSink`], emitting every key, value and image **borrowed from the buffer passed in**.
//!
//! That borrowing is what makes cover art zero-copy end to end: the bytes the sink is handed
//! lie inside this scanner's own read buffer, so [`ArenaSink`](crate::store::ArenaSink) takes
//! a refcounted `Memory` sub-view of them instead of copying (spec: Memory). No second
//! implementation of a format the workspace already owns, and no copy to undo later.

use profluens_core::event::TagSink;

use crate::Props;

/// What the metadata chain needs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Plan {
    /// The whole chain is inside the window; it ends at this absolute file offset.
    Complete { meta_end: u64 },
    /// The window is short: the chain needs bytes up to this absolute offset.
    Need { end: u64 },
}

/// Bytes of a metadata block header: `[last:1|type:7]` + 24-bit big-endian length (§8.1).
const BLOCK_HEADER: u64 = 4;

/// When a block header is itself cut off by the end of the window, ask for this much more
/// rather than one header's worth: a re-read that advances by 4 bytes would need a read per
/// block. One page covers every remaining header in any real file.
const HEADER_LOOKAHEAD: u64 = 4096;

/// How far the metadata chain reaches, given `window` (which starts at file offset 0) with the
/// `fLaC` marker at offset `body` — non-zero only when an ID3v2 tag precedes the stream.
///
/// This walks headers only: it never looks inside a block, so a 5 MB PICTURE costs four bytes
/// of attention here and one read afterwards.
pub(crate) fn plan(window: &[u8], body: u64, file_len: u64) -> Plan {
    let mut p = body + BLOCK_HEADER; // past "fLaC" (§8: the stream begins with the marker)
    loop {
        let Ok(at) = usize::try_from(p) else { return Plan::Complete { meta_end: p } };
        let Some(h) = window.get(at..at + 4) else {
            return Plan::Need { end: (p + BLOCK_HEADER + HEADER_LOOKAHEAD).min(file_len) };
        };
        let last = h[0] & 0x80 != 0;
        let len = u64::from(u32::from_be_bytes([0, h[1], h[2], h[3]]));
        let end = p + BLOCK_HEADER + len;

        // The body must be present for the chain to be walkable past it (and for pf-flac to
        // read it). Clamped to the file: a block claiming more than the file holds is asking
        // for bytes that do not exist.
        if (window.len().saturating_sub(at + 4) as u64) < len {
            return Plan::Need { end: end.min(file_len) };
        }
        p = end;
        if last || end >= file_len {
            return Plan::Complete { meta_end: end };
        }
    }
}

/// STREAMINFO-derived properties, via [`pf_flac::tags::stream_info`] (RFC 9639 §8.2).
///
/// A sample rate of 0 means "non-audio stream" and a total sample count of 0 means "unknown"
/// (§8.2), so neither yields a duration.
pub(crate) fn props(window: &[u8], body: u64) -> Props {
    let mut props = Props::default();
    let Some(from) = window.get(usize::try_from(body).unwrap_or(usize::MAX)..) else {
        return props;
    };
    let Some(si) = pf_flac::tags::stream_info(from) else { return props };
    props.channels = Some(u32::from(si.channels));
    if si.sample_rate > 0 {
        props.sample_rate = Some(si.sample_rate);
        if si.total_samples > 0 {
            // u128: `total_samples` is a 36-bit field, so `× 1e9` overflows u64 outright.
            props.duration_ns =
                u64::try_from(u128::from(si.total_samples) * 1_000_000_000 / u128::from(si.sample_rate))
                    .ok();
            props.duration_exact = true;
        }
    }
    props
}

/// Stream the chain's tags into `sink`, borrowed from `window` (see the module docs).
pub(crate) fn emit_tags(window: &[u8], body: u64, sink: &mut impl TagSink) {
    let Some(from) = window.get(usize::try_from(body).unwrap_or(usize::MAX)..) else { return };
    pf_flac::tags::parse_into(from, sink);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;
    use profluens_core::event::TagListSink;
    use profluens_core::memory::Pool;

    use crate::fixture::{flac_block, streaminfo};

    /// STREAMINFO + VORBIS_COMMENT + PICTURE, and *nothing after* — so the chain's end and
    /// the file's end coincide and `Plan::Complete`'s offset is checkable.
    fn file() -> Vec<u8> {
        let mut f = b"fLaC".to_vec();
        f.extend_from_slice(&flac_block(false, 0, &streaminfo(44_100, 2, 16, 44_100 * 3)));
        f.extend_from_slice(&flac_block(
            false,
            4,
            &crate::fixture::vorbis_comment(&["TITLE=Set Theory", "date=2015"]),
        ));
        f.extend_from_slice(&flac_block(
            true,
            6,
            &crate::fixture::picture_block("image/png", b"\x89PNG-bytes"),
        ));
        f
    }

    #[test]
    fn plan_and_streaminfo() {
        let f = file();
        let props = props(&f, 0);
        assert_eq!(plan(&f, 0, f.len() as u64), Plan::Complete { meta_end: f.len() as u64 });
        assert_eq!(props.sample_rate, Some(44_100));
        assert_eq!(props.channels, Some(2));
        assert_eq!(props.duration_ns, Some(3_000_000_000));
        assert!(props.duration_exact);
    }

    #[test]
    fn a_short_window_asks_for_the_chain_end() {
        let f = file();
        // Cut inside the PICTURE block: the plan must ask for exactly the chain's end.
        let cut = f.len() - 4;
        assert_eq!(plan(&f[..cut], 0, f.len() as u64), Plan::Need { end: f.len() as u64 });
        // STREAMINFO was still in the window, so its props are readable anyway.
        assert_eq!(props(&f[..cut], 0).sample_rate, Some(44_100));
    }

    #[test]
    fn tags_are_forwarded() {
        let f = file();
        let pool = Pool::bounded(4096, 4);
        let mut sink = TagListSink::new(&pool);
        emit_tags(&f, 0, &mut sink);
        let tags = sink.finish();
        assert_eq!(tags.get("TITLE"), Some("Set Theory"));
        assert_eq!(tags.get("DATE"), Some("2015"));
        assert_eq!(tags.pictures().len(), 1);
        assert_eq!(&*tags.pictures()[0].mime, "image/png");
        assert_eq!(tags.pictures()[0].data.data(), b"\x89PNG-bytes");
    }

    #[test]
    fn unknown_total_samples_yields_no_duration() {
        let mut f = b"fLaC".to_vec();
        f.extend_from_slice(&flac_block(true, 0, &streaminfo(48_000, 1, 24, 0)));
        let props = props(&f, 0);
        assert_eq!(props.sample_rate, Some(48_000));
        assert_eq!(props.channels, Some(1));
        assert_eq!(props.duration_ns, None);
        assert!(!props.duration_exact);
    }

    #[test]
    fn truncation_never_panics() {
        let f = file();
        let pool = Pool::bounded(4096, 8);
        for n in 0..f.len() {
            let _ = plan(&f[..n], 0, n as u64);
            let _ = props(&f[..n], 0);
            let mut sink = TagListSink::new(&pool);
            emit_tags(&f[..n], 0, &mut sink);
        }
    }

    #[test]
    fn a_lying_block_length_is_bounded_by_the_file() {
        // A block claiming 16 MiB in a 12-byte file.
        let mut f = b"fLaC".to_vec();
        f.extend_from_slice(&[0x00, 0xFF, 0xFF, 0xFF]);
        f.extend_from_slice(&[0u8; 4]);
        assert_eq!(
            plan(&f, 0, f.len() as u64),
            Plan::Need { end: f.len() as u64 },
            "clamped to EOF, not 16 MiB"
        );
    }
}
