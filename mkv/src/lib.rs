//! sc-mkv — a hand-written, dependency-free Matroska (MKV) container **muxer + demuxer** for
//! streamcraft.
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
//! Layout mirrors the `sc-ogg` plugin — the nearest analog (a hand-written container with a
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
//! - [`MkvMux`] — the streamcraft muxer **element** wrapping the writer, with a single static
//!   sink pad (single-track): encoded frames in on `sink`, MKV bytes out on `bytes`.
//! - [`MkvDemux`] — the streamcraft demuxer **element** ([`element`]): a Matroska byte stream
//!   in on `sink`, one **dynamic src pad per track** out. Track discovery is
//!   constructor-supplied (a mid-pipeline element gets no input during preroll, and the
//!   scheduler freezes topology after preroll), so [`MkvDemux::new`] takes the stream head and
//!   parses it in `preroll` to add the pads; the full stream then streams through `process`.
//!
//! ## `A_FLAC` (spec: A_FLAC mapping)
//! The container is codec-agnostic: `CodecID`, `CodecPrivate` (FLAC `fLaC` + STREAMINFO), and
//! the frame bytes are all caller-supplied on the mux side and read back on the demux side.
//! For FLAC the frames are stored natively, one per SimpleBlock, with no transformation. On
//! demux the native FLAC byte stream is **reconstructed** (CodecPrivate = `fLaC` + STREAMINFO,
//! then every frame) so `sc-flac`'s `FlacDec` decodes it directly — exactly the way
//! `oggflacdeframe` reconstructs the native stream from the Ogg mapping. The demux src pad
//! announces family `flac` (rate/channels/sample) via dynamic caps, `bytes` for unknown codec
//! ids.
//!
//! ## Not yet
//! - **Multi-track *mux* element**: the writer is N-track, but [`MkvMux`] exposes one sink
//!   pad. Multi-track wants one dynamic sink pad per input plus fan-in on the core side (a
//!   documented follow-up, mirroring `sc-ogg`'s single-stream `OggMux`). The **demux** side
//!   is already multi-track (one dynamic src pad per discovered track).
//! - **Codec-init via caps**: audio params + `CodecPrivate` are constructor arguments for
//!   now. Carrying codec-init-data through a negotiated `FixedFormat` config blob is the
//!   clean future path (see [`element`] docs).
//! - **Seeking metadata**: no Cues/SeekHead — a streaming muxer does not need them, and
//!   `finalize` stays seek-free; the demuxer walks Clusters linearly. A later two-pass /
//!   index-driven mode can add them.

#![deny(unsafe_code)]

pub mod codec;
pub mod ebml;
pub mod element;
pub mod reader;
pub mod writer;

pub use codec::{family_for, nal_head_from_config, Reframer, ReframeError};
pub use element::{MkvDemux, MkvMux};
pub use reader::{Frame, MatroskaReader, Track};
pub use writer::{
    AudioConfig, MatroskaWriter, TrackConfig, VideoConfig, WriteError, APP_NAME,
    DEFAULT_TIMESTAMP_SCALE,
};
