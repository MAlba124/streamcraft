//! pf-mkv — a hand-written, dependency-free Matroska (MKV) container **muxer + demuxer** for
//! profluens.
//!
//! Matroska ([RFC 9559](https://www.rfc-editor.org/rfc/rfc9559.txt)) is an
//! [EBML](https://www.rfc-editor.org/rfc/rfc8794.txt) (RFC 8794) document: a tree of typed,
//! ID-keyed elements. This crate implements **both directions** entirely by hand — no
//! libmatroska, no libav — against the spec summary checked into `spec/MATROSKA.md` (which
//! cites RFC 8794 for the EBML byte grammar and RFC 9559 / `matroska.org` for the element ID
//! tree, block/lacing structure, and the `A_FLAC` codec mapping); the two RFCs are checked in
//! at `spec/rfc8794.txt` and `spec/rfc9559.txt`. Code cross-references them as `§ID-tree`,
//! `§sizing`, `§simpleblock`, and `RFC 9559 §10.3`. Review means reading the code against
//! that spec text.
//!
//! Layout mirrors the `pf-ogg` plugin — the nearest analog (a hand-written container with a
//! reader/writer + mux/demux elements + spec in-tree): a thin `lib.rs` re-exporting the
//! public surface, submodules for the work, and the spec in `spec/`.
//!
//! ## What exists
//! - [`ebml`] — hand-written EBML primitives, **both write and read**: pre-encoded Element
//!   IDs, the Element Data Size VINT (shortest-length + a back-patched 8-octet form), the
//!   unknown-size marker, big-endian leaf encoders, and the fallible, bounds-checked reader
//!   ([`ebml::read_element_header`] etc.) the demuxer parses on. Unit-tested both ways.
//! - [`MatroskaWriter`] — a **multi-track** muxer (a `Vec` of [`TrackConfig`]): writes the
//!   header (EBML Header + open Segment + Info + Tracks), a `SimpleBlock` per frame into
//!   Clusters, and a seek-free `finalize`. Segment/Cluster use the unknown-size streamed
//!   form, so muxing is single-pass ([`writer`]).
//! - [`MatroskaReader`] — the **incremental structure reader** ([`reader`]): buffers bytes
//!   across input boundaries and decodes EBML Header / Segment / Info / Tracks then
//!   Cluster / SimpleBlock and BlockGroup+Block — **all three lacing modes** (RFC 9559
//!   §10.3) — into per-track [`Frame`]s with an absolute ns pts. Every field is
//!   bounds-checked; truncated / malformed input errors, never panics.
//! - [`MkvMux`] — the profluens muxer **element** wrapping the writer, with a single static
//!   sink pad (single-track): encoded frames in on `sink`, MKV bytes out on `bytes`.
//! - [`MkvDemux`] — the profluens demuxer **element** ([`element`]): a Matroska byte stream
//!   in on `sink`, one **dynamic src pad per track** out. Track discovery is
//!   constructor-supplied (a mid-pipeline element gets no input during preroll, and the
//!   scheduler freezes topology after preroll), so [`MkvDemux::new`] takes the stream head and
//!   parses it in `preroll` to add the pads; the full stream then streams through `process`.
//! - [`codec`] — the CodecID → announce-family map ([`family_for`]) and the [`Reframer`] that
//!   turns an AVC/HEVC configuration record + length-prefixed Blocks into an Annex B stream for
//!   the H.264/H.265 decoders (spec: video codec mappings). Bounds-checked, golden-vector
//!   tested.
//!
//! ## `A_FLAC` (spec: A_FLAC mapping)
//! The container is codec-agnostic: `CodecID`, `CodecPrivate` (FLAC `fLaC` + STREAMINFO), and
//! the frame bytes are all caller-supplied on the mux side and read back on the demux side.
//! For FLAC the frames are stored natively, one per SimpleBlock, with no transformation. On
//! demux the native FLAC byte stream is **reconstructed** (CodecPrivate = `fLaC` + STREAMINFO,
//! then every frame) so `pf-flac`'s `FlacDec` decodes it directly — exactly the way
//! `oggflacdeframe` reconstructs the native stream from the Ogg mapping. The demux src pad
//! announces family `flac` (rate/channels/sample) via dynamic caps, `bytes` for unknown codec
//! ids.
//!
//! ## Video (spec: video codec mappings; RFC 9559 §12)
//! A video track sets `TrackType = 1` with a `Video` master (PixelWidth/PixelHeight). WebM
//! codecs (`V_VP8`/`V_VP9`/`V_AV1`) store frames raw — one Block per frame, pts already
//! computed — so the demuxer forwards them verbatim and names the pad `vp8`/`vp9`/`av1`. The
//! ISO-BMFF NAL codecs (`V_MPEG4/ISO/AVC` → `h264/annexb`, `V_MPEGH/ISO/HEVC` → `h265/annexb`)
//! carry length-prefixed NALs plus an avcC/hvcC CodecPrivate; the demuxer reframes them to
//! Annex B (parameter-set head + one start-code access unit per buffer) via [`codec`]. The
//! writer emits a video track with [`TrackConfig::video`]/[`TrackConfig::vp8`]; a dedicated
//! video mux **element** is a naming-only follow-up (see "Not yet").
//!
//! ## Not yet
//! - **Remux of av1 / h264 / h265**: [`MkvMux::from_caps`] muxes announced `flac` (CodecPrivate
//!   absorbed from the in-band head), `vp8` and `vp9` tracks today; `av1` needs an `av1C`
//!   CodecPrivate and the NAL codecs need Annex B → length-prefixed reframing + a config
//!   record — both loud errors at the announcement until built (see [`element`] docs).
//! - **Frame-preserving demux emission**: a frame larger than the demuxer's pool slot is
//!   emitted split, which a downstream muxer would write as several blocks; remuxing such
//!   streams needs `Pipeline::set_element_pool` sizing today.
//! ## Seeking (spec: flush/seek; RFC 9559 §5.1.1, §5.1.5)
//! The writer's [`enable_cues`](MatroskaWriter::enable_cues) emits a front SeekHead + a Cues
//! index. The read side turns that into time-based seeking:
//! - [`parse_seek_head`] walks the header bytes for the Cues Segment Position + TimestampScale,
//!   and [`parse_cues`] turns a Cues master into a sorted `(time_ns, absolute_byte)` index —
//!   both pure functions over byte slices (the app does the file IO). Without Cues the app
//!   estimates a byte proportionally.
//! - [`MatroskaReader::resync_streaming`] resets the reader for a mid-Segment resume (keeping
//!   the discovered tracks/scale) and scans forward for the next Cluster ID, tolerating garbage
//!   before it (a proportional seek lands anywhere).
//! - [`MkvDemux`] handles `Event::FlushStart`: it resyncs the reader, drops staged output,
//!   re-emits codec heads (downstream decoders reset on flush), and gates each video track's
//!   post-seek output until its first keyframe (audio passes immediately).

#![deny(unsafe_code)]

pub mod codec;
mod color_map;
pub mod ebml;
pub mod element;
pub mod mux_multi;
pub mod reader;
pub mod writer;

pub use codec::{family_for, nal_head_from_config, Reframer, ReframeError};
pub use element::{MkvDemux, MkvMux};
pub use mux_multi::MkvMuxN;
pub use reader::{parse_cues, parse_seek_head, Frame, MatroskaReader, SeekHeadInfo, Track};
pub use writer::{
    AudioConfig, ColourConfig, MatroskaWriter, MuxOut, MuxPiece, TrackConfig, VideoConfig,
    WriteError, APP_NAME, DEFAULT_TIMESTAMP_SCALE,
};

use profluens_core::element::Element;
use profluens_core::registry::Registry;

/// Register this crate's elements for name-based lookup (spec: Plugins — `--list` and
/// descriptor introspection). Typed `use` + constructor stays the primary path.
///
/// **`mkvmux` is constructible by name** (its `make_default` is the caps-driven
/// [`MkvMux::from_caps`], configured by the upstream announcement — the remux path).
/// `mkvdemux` still carries `make_default: None`: it needs the stream **header bytes** at
/// construction (see [`MkvDemux::new`]), which a zero-arg default cannot supply until
/// preroll-time header capture / autoplug lands; registration exposes its descriptor for
/// `--list`/help and descriptor queries meanwhile.
///
/// The `&'static ElementDesc`s are taken from throwaway instances — only `desc()` is
/// called, so the instances are dropped.
// COLD: one-time registration; empty header for a throwaway descriptor-only instance.
#[allow(clippy::disallowed_methods)]
pub fn register(registry: &mut Registry) {
    registry.register(MkvMux::from_caps().desc());
    registry.register(MkvMuxN::new().desc());
    registry.register(MkvDemux::new(Vec::new()).desc());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `register` exposes the elements under their descriptor names. `mkvmux` and
    /// `mkvmuxn` are name-constructible (caps-driven defaults — the remux paths);
    /// `mkvdemux` is not (it needs the stream header bytes at construction).
    #[test]
    fn register_exposes_both_elements() {
        let mut reg = Registry::new();
        register(&mut reg);
        assert_eq!(
            reg.names(),
            vec!["mkvdemux", "mkvmux", "mkvmuxn"],
            "all descriptors registered"
        );
        assert!(
            reg.get("mkvmux").unwrap().make_default.is_some(),
            "mkvmux is name-constructible (caps-driven default)"
        );
        assert!(
            reg.get("mkvmuxn").unwrap().make_default.is_some(),
            "mkvmuxn is name-constructible (caps-driven default)"
        );
        assert!(reg.get("mkvdemux").unwrap().make_default.is_none(), "mkvdemux not name-constructible");
    }
}
