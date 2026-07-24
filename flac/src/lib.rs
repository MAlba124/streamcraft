//! sc-flac — a hand-written, dependency-free FLAC codec plugin for streamcraft.
//!
//! Milestone 3: `filesrc ! wavparse ! flacenc ! filesink` converts WAV to FLAC
//! (spec: Milestone applications §3; First-party codecs). Everything here is written
//! against the format spec checked into `spec/rfc9639.txt` (RFC 9639, "Free Lossless
//! Audio Codec"), cross-referenced in comments as `§9.2.7` etc. — review means
//! reading the code against the normative text, not a blog post.
//!
//! Layout mirrors the `sc-http` plugin: a thin `lib.rs` re-exporting the public
//! surface, submodules for the actual work, and the spec in `spec/`.
//!
//! ## What exists
//! - [`BitWriter`]/[`BitReader`] — big-endian bit I/O plus FLAC Rice/unary coding and
//!   the two frame CRCs ([`bitstream`]).
//! - [`FlacEncoder`] — a raw interleaved-PCM → FLAC-stream encoder, and [`FlacEnc`],
//!   its streamcraft [`Element`](streamcraft_core::element::Element) wrapper.
//! - [`FlacDecoder`] — decodes what the encoder emits, so round-trip losslessness is
//!   provable in-crate without any external tool.
//! - [`OggFlacDeframe`] — the FLAC-in-Ogg de-framer element: reconstructs a native FLAC
//!   byte stream from an Ogg-mapped one (`filesrc ! oggdemux ! oggflacdeframe ! flacdec`),
//!   per the xiph "Ogg Mapping for FLAC" (`spec/ogg-flac-mapping.md`).
//!
//! ## Encoder scope (correctness first, then speed)
//! - STREAMINFO metadata block (§8.2); frame + subframe headers (§9.1, §9.2).
//! - `CONSTANT`, `VERBATIM`, and `FIXED` (orders 0–4) subframes (§9.2.3–§9.2.5), the
//!   best of the three chosen per subframe by an estimated-bits heuristic.
//! - Rice-coded, partitioned residual with a searched partition order (§9.2.7).
//! - Correct frame sync, CRC-8 header and CRC-16 frame checksums (§9.1.8, §9.3).
//! - Input is raw **interleaved** PCM; audio params (rate/channels/bits) are
//!   constructor arguments, since format negotiation is being built in parallel.
//!
//! ## Not yet
//! - LPC subframes (§9.2.6) — FIXED predictors already make a valid, well-compressing
//!   encoder; LPC is a later compression win.
//! - Stereo decorrelation (left/side, mid/side) — a future compression win; channels
//!   are currently coded independently.
//! - Sample formats beyond signed 8/16/24/32-bit integer PCM.

#![deny(unsafe_code)]

pub mod bitstream;
mod decoder;
mod encoder;
mod flacdec;
mod flacenc;
mod oggflac;

pub use bitstream::{crc16, crc8, BitReader, BitWriter, ReadError};
pub use decoder::{DecodeError, DecodedFrame, FlacDecoder, StreamDecoder, StreamInfo};
pub use encoder::{EncodeError, FlacEncoder, SampleFormat};
pub use flacdec::FlacDec;
pub use flacenc::FlacEnc;
pub use oggflac::OggFlacDeframe;

/// Byte offset of the STREAMINFO block body within a stream produced by
/// [`FlacEncoder::new`] (spec §8.2). A two-pass caller writes the finalised body from
/// [`FlacEncoder::finish`] here to fill in the exact frame sizes and total samples.
pub const fn streaminfo_offset() -> usize {
    encoder::STREAMINFO_BODY_OFFSET
}

