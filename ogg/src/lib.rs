//! sc-ogg — a hand-written, dependency-free Ogg container plugin for streamcraft.
//!
//! Ogg (RFC 3533, "The Ogg Encapsulation Format Version 0") is the streaming container
//! that carries FLAC / Opus / Vorbis packets. This crate implements the reader
//! (demuxer) and writer (muxer) for it entirely by hand — no libogg, no external
//! container library — against the spec text checked into `spec/rfc3533.txt`, with code
//! cross-referencing sections as `§6` etc. Review means reading the code against the
//! normative text.
//!
//! Layout mirrors the `sc-flac` / `sc-http` plugins: a thin `lib.rs` re-exporting the
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
//! ## Not yet
//! - **streamcraft elements** (`OggMux` / `OggDemux`): deferred. A demuxer needs one
//!   dynamic src pad *per discovered logical stream* and a muxer one dynamic sink pad
//!   per input stream; the milestone-1 [`Element`](streamcraft_core::element::Element)
//!   pad model is static (`PadDesc { dynamic: false, .. }`) and the scheduler has no
//!   per-stream pad add/remove yet. Following how `sc-flac` shipped the codec core
//!   before element polish, the tested reader/writer **library** is the deliverable;
//!   elements land once the core grows dynamic pads. The library is
//!   framework-independent (depends on `streamcraft-core` only nominally) and drops
//!   straight into an element wrapper when that exists.

#![deny(unsafe_code)]

pub mod crc;
pub mod page;
pub mod reader;
pub mod writer;

pub use crc::{crc32, Crc32};
pub use page::{
    flags as header_flags, page_crc, write_page, PageError, PageHeader, CAPTURE_PATTERN,
    GRANULE_NONE, HEADER_FIXED_LEN, MAX_PAGE_SIZE, MAX_SEGMENTS, STREAM_STRUCTURE_VERSION,
};
pub use reader::{demux_all, OggReader, Packet};
pub use writer::{mux_packets, OggWriter, WriteError};
