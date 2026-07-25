//! H.264 depayloading (RFC 6184): RTP payloads → Annex-B access units.
//!
//! **STUB — implementation is agent B's scope.** Scope: packetization modes 0
//! and 1 — single NAL unit packets (§5.6), STAP-A (§5.7.1), FU-A (§5.8).
//! MTAP/STAP-B/FU-B (mode 2, interleaved) are out of scope and must fail
//! loudly, not silently corrupt. AU boundary: marker bit and/or timestamp
//! change (§5.1). `sprop-parameter-sets` (SDP fmtp, §8.1) arrive out-of-band
//! via the constructor and are prepended as Annex-B SPS/PPS before the first
//! AU.

#![allow(dead_code, unused_variables)]

/// The stateful depacketizer: packets in (in sequence order — the jitter
/// buffer upstream guarantees it), Annex-B access units out.
#[derive(Debug, Default)]
pub struct H264Depay;

impl H264Depay {
    /// `sprop`: the decoded `sprop-parameter-sets` NAL units (already
    /// base64-decoded by the SDP layer), prepended before the first output.
    pub fn new(sprop: Vec<Vec<u8>>) -> H264Depay {
        todo!("agent B")
    }

    /// Feed one payload (marker + timestamp from the RTP header). Returns the
    /// completed access unit ending at this packet, if any.
    pub fn push(&mut self, payload: &[u8], marker: bool, timestamp: u32) -> Option<Vec<u8>> {
        todo!("agent B: RFC 6184 §5.6/§5.7.1/§5.8")
    }

    /// A sequence discontinuity was declared upstream (loss): drop the
    /// partial AU and resynchronize (wait for the next AU start).
    pub fn discontinuity(&mut self) {
        todo!("agent B")
    }
}
