//! pf-avi — a hand-written, dependency-free **AVI (RIFF) container demuxer** for profluens.
//!
//! AVI is a RIFF document (Microsoft *AVI RIFF File Reference*,
//! `learn.microsoft.com/windows/win32/directshow/avi-riff-file-reference`; the
//! `OpenDML AVI File Format Extensions v1.02` specification for the >1 GB extensions). This
//! crate implements the read/demux side entirely by hand — no libavformat, no libav — with
//! the byte grammar and structure cited at point of use in [`riff`]. Layout mirrors the
//! `pf-mkv`/`pf-ogg` plugins: a thin `lib.rs` re-exporting the public surface, a
//! core-independent parser ([`riff`]), a codec→family map ([`codec`]), and the demuxer
//! element ([`element`]).
//!
//! ## What exists
//! - [`riff`] — the hand-written RIFF/AVI primitives: [`probe_header`] discovers streams from
//!   a header prefix, [`MoviWalker`] streams the `movi` list into per-stream [`MediaChunk`]s
//!   (buffering across input boundaries, descending `LIST 'rec '`, skipping `JUNK`, resyncing
//!   past corruption), and [`parse_idx1`] turns the legacy index into a time→byte
//!   [`SeekIndex`]. Bounds-checked throughout; a fuzz-ish bad-input table pins that nothing
//!   panics (this crate's untrusted-input P0).
//! - [`codec`] — the fourcc / WAVEFORMATEX-tag → announce-family map ([`family_for`]) and the
//!   per-stream offer menus. Video keys on the fourcc (`XVID`/`DX50`/`DIVX`/… → `mpeg4/asp`,
//!   case-insensitive; `H264`/`avc1` → `h264/annexb`); audio on the tag (`0x2000`/`0x2001` →
//!   `ac3`, `0x0055` → `mp3`, `0x0001` → `audio/raw`). The families match the profluens
//!   decoders' sink offers so negotiation-driven autoplug selects the right one.
//! - [`AviDemux`] — the profluens demuxer **element** ([`element`]): an AVI byte stream in
//!   on `sink`, one **dynamic src pad per stream** out. Track discovery is
//!   constructor-supplied ([`AviDemux::new`] takes the header prefix — a mid-pipeline element
//!   gets no input during preroll), one demuxed chunk per output buffer stamped with a PTS,
//!   the demuxer backpressure discipline (a demuxer that can't push consumes no input).
//!
//! ## Timing (AVI RIFF Reference — a stream's time base is `dwScale/dwRate`)
//! Video PTS is the frame index × the video `strh` scale/rate (constant-frame-rate XviD, so
//! chunk counting is exact). Audio is usually VBR (AC-3/MP3): one `movi` chunk is one frame,
//! stamped with a monotonically increasing PTS from the accumulated audio time (sample count
//! for CBR, chunk ordinal for VBR). Interleave is by chunk order in `movi`.
//!
//! ## Seeking (spec: flush/seek; AVI RIFF Reference — "AVI Index (idx1)")
//! The `idx1` chunk (at the *end* of the file) maps each `movi` chunk to a byte offset and a
//! keyframe flag. The **element stays reactor-pure** (it never opens the file); the **app**
//! reads the `idx1` range with a sanctioned `#[allow(clippy::disallowed_methods)]` pread —
//! exactly as `sdl3/examples/play_file.rs` builds the MKV Cues index — and calls
//! [`build_seek_index`] to turn it into a [`SeekIndex`]. The demuxer's `FlushStart` handler
//! resyncs the walker to the next chunk id, tolerating a mid-chunk landing.
//!
//! ## OpenDML (>1 GB / >2 GB files)
//! The reference file (`Nord…XviD.AC3.avi`, ~2 GB) carries a `LIST 'odml'` header **and** a
//! standard `idx1` at the end, inside a single RIFF whose 32-bit sizes just fit — so **the
//! legacy `idx1` path handles it; OpenDML `AVIX` extension RIFFs / `indx` super-indexes are
//! NOT needed here**. `AviHeader::has_odml` is reported for diagnostics. A true >4 GB file
//! that overflows the single-RIFF size and chains `RIFF 'AVIX'` segments would need the
//! `indx`/`AVIX` walk — a documented follow-up ([`riff`]'s `MoviWalker` already skips
//! unknown top-level chunks, but does not yet chain a second RIFF).

#![deny(unsafe_code)]

pub mod codec;
pub mod element;
pub mod riff;

pub use codec::{family_for, offers_for};
pub use element::AviDemux;
pub use riff::{
    parse_idx1, probe_header, AviError, AviHeader, Idx1, Idx1Entry, MediaChunk, MoviLocation,
    MoviWalker, Stream, StreamKind,
};

use profluens_core::element::Element;
use profluens_core::pipeline::SeekIndex;
use profluens_core::registry::Registry;

/// Build a time→byte [`SeekIndex`] for an AVI file from its header prefix + its raw `idx1`
/// payload (the bytes of the `idx1` chunk *after* its 8-byte id+size header). This is the
/// **app-facing** helper (spec: flush/seek — mapping time to a byte is the seek issuer's
/// job): the app reads the two byte ranges with a sanctioned `#[allow]` pread — see the
/// module docs and `avi/examples/probe.rs` — and this ties [`probe_header`], [`parse_idx1`],
/// and the video stream's time base together into the index the pipeline installs with
/// `Pipeline::set_seek_index`.
///
/// `header` must reach the `movi` list (so the movi data start is known); `idx1_payload` is
/// the `idx1` chunk body; `file_len` is the total file size (for the proportional fallback
/// when a stream has no `idx1` keyframes). Returns `None` if the header does not parse or
/// reach `movi`, or carries no video stream to key the timeline on.
///
/// The index keys on the **first video stream**'s keyframes (AVI RIFF Reference — a keyframe
/// index entry marks a random-access point); its per-frame duration comes from the video
/// `strh` time base. With no video keyframes the entries are empty and the pipeline uses the
/// proportional `file_len` fallback, which the demuxer's resync tolerates.
pub fn build_seek_index(header: &[u8], idx1_payload: &[u8], file_len: u64) -> Option<SeekIndex> {
    let (avi_header, movi) = probe_header(header).ok()?;
    let movi = movi?;
    // The first video stream keys the timeline (the picture's random-access points).
    let video = avi_header
        .streams
        .iter()
        .find(|s| matches!(s.kind, StreamKind::Video))?;
    let dur = video.sample_duration_ns()?;
    let idx = parse_idx1(idx1_payload, movi.movi_data_start, file_len);
    Some(idx.build_seek_index(video.index, dur))
}

/// Register this crate's elements for name-based lookup (spec: Plugins — `--list` and
/// descriptor introspection). Typed `use` + constructor stays the primary path.
///
/// `avidemux` carries `make_default: None`: it needs the stream **header bytes** at
/// construction (see [`AviDemux::new`]), which a zero-arg default cannot supply until
/// preroll-time header capture / autoplug lands; registration exposes its descriptor for
/// `--list`/help and descriptor queries meanwhile — the same shape as `mkvdemux`.
///
/// The `&'static ElementDesc` is taken from a throwaway instance — only `desc()` is called.
// COLD: one-time registration; the empty header vec seeds a throwaway instance for its desc().
#[allow(clippy::disallowed_methods)]
pub fn register(registry: &mut Registry) {
    registry.register(AviDemux::new(Vec::new()).desc());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_exposes_the_demuxer() {
        let mut reg = Registry::new();
        register(&mut reg);
        assert_eq!(reg.names(), vec!["avidemux"], "the demuxer descriptor is registered");
        assert!(
            reg.get("avidemux").unwrap().make_default.is_none(),
            "avidemux is not name-constructible (needs the stream header bytes)"
        );
    }
}
