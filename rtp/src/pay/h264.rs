//! H.264 payloading (RFC 6184): Annex-B access units → RTP payloads.
//!
//! **STUB — implementation is agent B's scope.** Single NAL where it fits the
//! MTU, FU-A fragmentation where it does not (§5.8); STAP-A aggregation of
//! small NALs (SPS+PPS) is a nice-to-have. Exists v1 for depay round-trip
//! tests; the send elements come later.

#![allow(dead_code, unused_variables)]

/// Split one Annex-B access unit into RTP payloads of at most `mtu` bytes.
/// Returns `(payload, marker)` pairs — marker set on the AU's last packet
/// (§5.1).
pub fn pay(access_unit: &[u8], mtu: usize) -> Vec<(Vec<u8>, bool)> {
    todo!("agent B: RFC 6184 §5.6/§5.8")
}
