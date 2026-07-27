//! `sc-ac3` — a pure-Rust **AC-3 (ATSC A/52)** + **E-AC-3 (Dolby Digital Plus,
//! A/52 Annex E)** audio decoder, hand-written clean-room from the standard.
//!
//! Unlike `sc-mp3` / `sc-aac` (which adopt a vetted third-party pure-Rust codec),
//! this crate implements the decoder itself, from ATSC A/52 (Digital Audio
//! Compression (AC-3, E-AC-3) Standard) — no libav, no liba52, no oxideav. Every
//! non-trivial step cites its A/52 section at the point of use (the repo's
//! standing algorithm-citation rule):
//!
//! - **[`bits`]** — the MSB-first big-endian bit reader (§2.3.3), the single
//!   bounds-enforcing gate (a read past the frame end is an error, never OOB).
//! - **[`parse`]** — the self-syncing framer (§5.1): finds `0B77` sync words,
//!   reads each frame's own length (§5.1.2 / §E.1.2.2), and delimits frames so an
//!   arbitrary byte stream (AVI chunk boundaries, a raw `.ac3`) resyncs cleanly.
//! - **[`frame`]** — the syncframe decode (§5–§7): syncinfo/bsi, differential
//!   exponents (§7.1), the parametric bit allocation (§7.2), grouped/direct
//!   mantissa dequant (§7.3), coupling (§7.4), rematrixing (§7.5), and the E-AC-3
//!   Annex-E branch (bsid 16). Detects vs decodes: AC-3 by bsid ≤ 8, E-AC-3 by
//!   bsid == 16.
//! - **[`imdct`]** — the 512/256-point IMDCT, A/52 window (§7.10) and
//!   overlap-add (§7.9), the synthesis filterbank producing 256 PCM samples per
//!   block per channel.
//! - **[`quant`]** / **[`tables`]** — mantissa reconstruction levels + dither
//!   (§7.3), and the normative bit-allocation / frame-size / hearing-threshold
//!   tables (§5, §7.2).
//!
//! ## Elements ([`element`])
//! - [`Ac3Dec`] (`ac3dec`) — sink family **`ac3`**.
//! - [`Eac3Dec`] (`eac3dec`) — sink family **`eac3`**.
//!
//! Both emit **`audio/raw`** (`rate` Int, `channels` Int, `sample` = `s16`,
//! interleaved), announced via dynamic caps from the first decoded frame. The
//! **channel order is L, R, C, LFE, Ls, Rs** (ITU/SMPTE 5.1(side)) — the order
//! the downstream downmixer assumes. The core decodes A/52's stream order
//! (§5.4.2.2, Table 5.8) and permutes to this ITU order before interleaving.
//!
//! ## Coverage — HONEST STATUS (per the brief: "do not stub-and-claim")
//!
//! This is a **work-in-progress decoder**. The full A/52 decode *structure* is
//! implemented and every stage runs end-to-end (a real AC-3/E-AC-3 file drives
//! all the way to interleaved S16 PCM with the right rate/channels/geometry), but
//! the reconstruction is **not yet bit-accurate**: decoded PCM does not match an
//! ffmpeg reference (measured ~0 dB SNR — see the report). There is an unresolved
//! discrepancy in the **bit-allocation / mantissa bit-accounting** (A/52 §7.2–§7.3):
//! the per-block header parse is bit-exact on one validated stream (Nord AC-3
//! block 0 — SNR-offset set and all six blocks land at the frame boundary), but
//! another stream (an uncoupled 5.1 tone) and later blocks over-read, so the
//! mantissa consumption is subtly wrong for some parameter combinations.
//!
//! What **is** correct and verified:
//! - **Framing** ([`parse`]) — `0B77` sync, frame-length from frmsizecod/frmsiz,
//!   two-frame lock, arbitrary-chunk resync (unit-tested; matches real streams,
//!   1792-byte CBR frames delimited exactly).
//! - **syncinfo / BSI** ([`frame`]) — AC-3 (§5) and E-AC-3 (§E.1) headers parse
//!   to the correct fscod/acmod/lfeon/bsid, validated bit-for-bit against a
//!   hand-decoded reference.
//! - **IMDCT + window + overlap-add** ([`imdct`]) — correct A/52 window (§7.10),
//!   overlap-add wiring (TDAC unit test), and the `2/N` synthesis normalization
//!   (silence decodes to silence; output magnitude is in the right full-scale
//!   range, not the ~256× overshoot an un-normalized transform would give).
//! - **Channel mapping** — A/52 stream order (§5.4.2.2) → ITU L,R,C,LFE,Ls,Rs.
//! - **Robustness (P0)** — every length/index bounds-checked; a malformed or
//!   over-reading frame is a warn-and-drop, never a panic or OOB; the framer
//!   resyncs on the next `0B77`.
//!
//! What is **implemented but not yet correct / verified**:
//! - The parametric **bit allocation** (§7.2) and **mantissa dequant** (§7.3) —
//!   the masking model and grouped/direct mantissa unpacking are coded to the
//!   spec but the exact bit consumption diverges for some blocks (the root cause
//!   of the SNR failure). This is the piece to finish next.
//! - **Coupling** (§7.4) — sub-band structure, per-band coordinates and the
//!   un-couple distribution are coded; a degenerate/zero-width coupling band is
//!   treated as an uncoupled block.
//! - **E-AC-3** (Annex E) — the base 6-block decode runs; **AHT (§E.3.2), SPX
//!   (§E.3.7), enhanced coupling (§E.3.6)** are detected and reported (one-shot
//!   bus warning), then approximated with the base tools; dependent/Atmos
//!   substreams are declined. The E-AC-3 frame-info parse (§E.1.2.3) is
//!   incomplete for exotic layouts — such a frame falls back to silence for the
//!   remaining blocks (flagged), never a crash.

// The A/52 bit-allocation and IMDCT loops walk band/bin/channel *indices* into
// several parallel arrays at once (psd/bndpsd/excite/mask, or coeffs/exp/bap),
// so the index — not an element — is the loop's subject; `needless_range_loop`
// would obscure that. The window table's literals trip the `approx_constant`
// lint against π/4 by coincidence. These are localized, intentional, and clearer
// as written.
#![allow(clippy::needless_range_loop, clippy::approx_constant)]

pub mod bits;
pub mod element;
pub mod frame;
pub mod imdct;
pub mod parse;
pub mod quant;
pub mod tables;

pub use element::{Ac3Dec, Eac3Dec};

use streamcraft_core::element::Element;
use streamcraft_core::registry::Registry;

/// Register both decoder elements for name-based construction (spec: Plugins —
/// `parse("… ! ac3dec ! …")`). Typed `use` + constructor stays primary; this
/// powers `streamcraft launch` and one-liner tests. Both descriptors are
/// `&'static`, taken from throwaway default instances; the decoders are
/// config-free (rate/channels come from the stream, announced at runtime).
pub fn register(registry: &mut Registry) {
    registry.register(Ac3Dec::new().desc());
    registry.register(Eac3Dec::new().desc());
}
