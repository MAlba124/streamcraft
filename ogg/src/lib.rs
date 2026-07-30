//! pf-ogg — a hand-written, dependency-free Ogg container plugin for profluens.
//!
//! Ogg (RFC 3533, "The Ogg Encapsulation Format Version 0") is the streaming container
//! that carries FLAC / Opus / Vorbis packets. This crate implements the reader
//! (demuxer) and writer (muxer) for it entirely by hand — no libogg, no external
//! container library — against the spec text checked into `spec/rfc3533.txt`, with code
//! cross-referencing sections as `§6` etc. Review means reading the code against the
//! normative text.
//!
//! Layout mirrors the `pf-flac` / `pf-http` plugins: a thin `lib.rs` re-exporting the
//! public surface, submodules for the actual work, and the spec (plus interpretation
//! `NOTES.md`) in `spec/`.
//!
//! ## What exists
//! - [`crc32`] / [`Crc32`] — the Ogg page CRC-32 (poly `0x04C11DB7`, init 0, no
//!   reflection, no final XOR; §6, field 7), a 256-entry table, known-answer tested
//!   ([`crc`]).
//! - [`PageHeader`] / [`write_page`] — parse and serialise a single Ogg page: the
//!   `OggS` capture pattern, header-type flags, granule/serial/sequence, segment table
//!   (lacing values), and CRC ([`page`]).
//! - [`OggReader`] / [`demux_all`] — a demuxer that verifies each page's CRC,
//!   reassembles packets from lacing values across page boundaries, tracks one logical
//!   stream **per serial number** (so multiplexed/grouped streams demux correctly), and
//!   **resynchronises past truncation or garbage** without ever panicking ([`reader`]).
//! - [`OggWriter`] / [`mux_packets`] — a muxer that segments packets into correct
//!   lacing values (including the trailing zero for multiples of 255, and nil packets),
//!   splits over-long packets across pages via 255-lacing continuation, and stamps
//!   incrementing sequence numbers, granule positions, and bos/eos flags ([`writer`]).
//!
//! ## Reader/writer support
//! - **Multiplexed streams** (concurrent multiplexing / "grouping", §4): reader keys
//!   reassembly by serial; writer is single-serial, so multiple writers' pages are
//!   interleaved by the caller (whole pages, §4).
//! - **Packet-spanning**: packets larger than one page's 255-segment table are split
//!   (writer) and rejoined (reader) via the CONTINUED flag (§6, flag 0x01).
//! - **Resync / robustness**: any bad capture pattern, version, or CRC makes the reader
//!   skip forward to the next `OggS` (§6). Truncated tails are held (streaming) or
//!   reported as leftover ([`OggReader::finish`]).
//!
//! ## profluens elements
//! - [`OggDemux`] / [`OggMux`] — **single logical bitstream** container elements wrapping
//!   the reader/writer above ([`element`]). `OggDemux` turns an Ogg byte stream into one
//!   codec packet per output buffer (first bos serial only); `OggMux` wraps one input
//!   buffer per packet into an Ogg byte stream, terminating it with an eos page. Both are
//!   passive byte→byte transforms with static `bytes` pads, matching today's pad model.
//!
//! ## Not yet
//! - **Multi-stream elements**: the demuxer emits only the *first* logical bitstream and
//!   the muxer accepts only *one* input stream. The general case wants one dynamic src pad
//!   *per discovered logical stream* (demux) and one dynamic sink pad per input (mux); the
//!   milestone-1 [`Element`](profluens_core::element::Element) pad model is static
//!   (`PadDesc { dynamic: false, .. }`) and the scheduler has no per-stream pad add/remove
//!   yet. Following how `pf-flac` shipped the codec core before element polish, the
//!   single-stream elements ship now; the multi-stream ones land once the core grows
//!   dynamic pads. The reader/writer library already demuxes/muxes every serial, so the
//!   dynamic-pad wrapper is the only missing piece.

#![deny(unsafe_code)]

pub mod crc;
pub mod element;
pub mod page;
pub mod reader;
pub mod writer;

pub use crc::{crc32, Crc32};
pub use element::{OggDemux, OggMux, DEFAULT_SERIAL};

use profluens_core::element::Element;
use profluens_core::registry::Registry;

/// Register this crate's elements for name-based construction (spec: Plugins —
/// `parse("… ! oggdemux ! …")`). Typed `use` + constructor stays primary; both
/// elements are config-free (the muxer uses [`DEFAULT_SERIAL`]; set a specific serial
/// with [`OggMux::with_serial`] in code). Descriptors are `&'static`, taken from a
/// throwaway default instance.
pub fn register(registry: &mut Registry) {
    registry.register(OggDemux::new().desc());
    registry.register(OggMux::new().desc());
}
pub use page::{
    flags as header_flags, page_crc, write_page, PageError, PageHeader, CAPTURE_PATTERN,
    GRANULE_NONE, HEADER_FIXED_LEN, MAX_PAGE_SIZE, MAX_SEGMENTS, STREAM_STRUCTURE_VERSION,
};
pub use reader::{demux_all, OggReader, Packet};
pub use writer::{mux_packets, OggWriter, WriteError};
