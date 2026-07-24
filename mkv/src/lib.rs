//! sc-mkv — a hand-written, dependency-free Matroska (MKV) container **muxer** for
//! streamcraft.
//!
//! Matroska is an [EBML](https://www.rfc-editor.org/rfc/rfc8794.txt) (RFC 8794) document:
//! a tree of typed, ID-keyed elements. This crate implements the *mux* side entirely by
//! hand — no libmatroska, no libav — against the spec summary checked into
//! `spec/MATROSKA.md` (which cites RFC 8794 for the EBML byte grammar and
//! `matroska.org` for the element ID tree and the `A_FLAC` codec mapping). Code
//! cross-references it as `§ID-tree`, `§sizing`, `§simpleblock`. Review means reading the
//! code against that spec text.
//!
//! Layout mirrors the `sc-ogg` plugin — the nearest analog (a hand-written container with
//! a reader/writer + a mux element + spec in-tree): a thin `lib.rs` re-exporting the
//! public surface, submodules for the work, and the spec in `spec/`.
//!
//! ## What exists
//! - [`ebml`] — hand-written EBML primitives: pre-encoded Element IDs, the Element Data
//!   Size VINT (shortest-length and a back-patched 8-octet form), the unknown-size marker
//!   for streamed masters, and big-endian encoders for uint / `f32` / `f64` / UTF-8
//!   string / binary. VINT round-trip and edge cases are unit-tested.
//! - [`MatroskaWriter`] — a **multi-track** muxer (a `Vec` of [`TrackConfig`]): writes the
//!   header (EBML Header + open Segment + Info + Tracks), a `SimpleBlock` per frame into
//!   Clusters, and a seek-free `finalize`. Segment/Cluster use the unknown-size streamed
//!   form, so muxing is single-pass ([`writer`]).
//! - [`MkvMux`] — the streamcraft **element** wrapping the writer, with a single static
//!   sink pad (single-track for now): encoded frames in on `sink`, MKV bytes out on `bytes`
//!   ([`element`]).
//!
//! ## `A_FLAC` (spec: A_FLAC mapping)
//! The muxer is codec-agnostic: `CodecID`, `CodecPrivate` (FLAC `fLaC` + STREAMINFO), and
//! the frame bytes are all caller-supplied. For FLAC the frames are stored natively, one
//! per SimpleBlock, with no transformation — this is the mux half of a future `ogg→mkv`
//! FLAC remuxer.
//!
//! ## Not yet
//! - **Multi-track element**: the writer is N-track, but [`MkvMux`] exposes one sink pad.
//!   Multi-track wants one dynamic sink pad per input plus fan-in on the core side (a
//!   documented follow-up, mirroring `sc-ogg`'s single-stream `OggMux`).
//! - **Codec-init via caps**: audio params + `CodecPrivate` are constructor arguments for
//!   now. Carrying codec-init-data through a negotiated `FixedFormat` config blob is the
//!   clean future path (see [`element`] docs).
//! - **Seeking metadata**: no Cues/SeekHead — a streaming muxer does not need them, and
//!   `finalize` stays seek-free. A later two-pass mode can add them.

#![deny(unsafe_code)]

pub mod ebml;
pub mod element;
pub mod writer;

pub use element::MkvMux;
pub use writer::{AudioConfig, MatroskaWriter, TrackConfig, WriteError, APP_NAME, DEFAULT_TIMESTAMP_SCALE};
