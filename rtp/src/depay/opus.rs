//! Opus depayloading (RFC 7587 §4.2): one RTP payload IS one Opus packet —
//! no fragmentation, no aggregation; the timestamp rate is fixed 48 kHz
//! regardless of the coded bandwidth (§4.1).
//!
//! **STUB — implementation is agent B's scope** (trivial by design; the tests
//! and the DTX/discontinuity notes are the substance).

#![allow(dead_code, unused_variables)]

/// Depacketize one payload: the Opus packet, verbatim. Kept as a function
/// (no state) — the element wrapper handles pts.
pub fn depay(payload: &[u8]) -> Vec<u8> {
    todo!("agent B: RFC 7587 §4.2")
}
