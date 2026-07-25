//! Opus payloading (RFC 7587 §4.2): one Opus packet IS one RTP payload.
//!
//! **STUB — implementation is agent B's scope.**

#![allow(dead_code, unused_variables)]

/// Payload one Opus packet (verbatim; marker semantics per §4.2: set only on
/// the first packet after silence/DTX — plain streams leave it clear).
pub fn pay(opus_packet: &[u8]) -> Vec<u8> {
    todo!("agent B: RFC 7587 §4.2")
}
