//! RTCP compound packets (RFC 3550 §6): SR/RR/SDES/BYE parse + build, and the
//! sender report's NTP↔RTP timestamp mapping — the anchor inter-stream (A/V)
//! synchronization hangs off.
//!
//! **STUB — implementation is agent A's scope.** API sketch; the one shape the
//! session element relies on is [`SenderInfo`] (parsed from an SR) and a
//! `parse_compound` entry point that yields it. Report *generation* (RRs with
//! the §6.4.1 fields, fixed 5 s tick v1) also lives here.

#![allow(dead_code, unused_variables)]

/// The sender-report media anchor (RFC 3550 §6.4.1): "the RTP timestamp
/// `rtp_ts` was sampled at NTP wall time `ntp`". Two streams' SRs place both
/// media timelines on one wall clock.
#[derive(Clone, Copy, Debug)]
pub struct SenderInfo {
    pub ssrc: u32,
    /// 64-bit NTP timestamp (seconds since 1900 in the top 32 bits).
    pub ntp: u64,
    /// The RTP timestamp corresponding to `ntp`, in the stream's clock rate.
    pub rtp_ts: u32,
    pub packet_count: u32,
    pub octet_count: u32,
}

/// One parsed item of a compound RTCP packet (§6.1: compound = concatenated
/// individual packets, SR/RR first).
#[derive(Clone, Debug)]
pub enum RtcpItem {
    SenderReport(SenderInfo),
    /// Receiver report — parsed but unused on the receive path v1.
    ReceiverReport { ssrc: u32 },
    /// Source description; only CNAME is interesting (§6.5.1).
    Sdes { ssrc: u32, cname: Option<String> },
    Bye { ssrc: u32 },
}

/// Parse a compound RTCP datagram into items (§6.1 framing: each packet's
/// length field is in 32-bit words minus one).
pub fn parse_compound(datagram: &[u8]) -> Result<Vec<RtcpItem>, RtcpError> {
    todo!("agent A")
}

/// Build a receiver-report compound (RR + SDES CNAME) from jitter-buffer
/// stats (§6.4.1 fraction-lost/cumulative-lost/highest-seq/jitter; A.3).
pub fn build_receiver_report(
    reporter_ssrc: u32,
    about_ssrc: u32,
    stats: &crate::jitter::JitterStats,
    cname: &str,
) -> Vec<u8> {
    todo!("agent A")
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RtcpError {
    Truncated,
    Version,
}
