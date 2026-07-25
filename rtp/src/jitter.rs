//! The reorder/dejitter buffer as a pure state machine (RFC 3550 §6.4.1's
//! interarrival-jitter model, appendix A.8 estimator).
//!
//! **STUB — implementation is agent A's scope.** The public API below is
//! frozen (the `RtpSession` element codes against it in parallel); internals,
//! fields, and additional helpers are free.
//!
//! Contract:
//! - No threads, no clock reads — all time is passed in as nanoseconds of the
//!   caller's monotonic running time.
//! - `insert` takes ownership of a depacketized-yet-unparsed datagram (the
//!   session hands whole RTP packets through in arrival order).
//! - `pop_ready(now)` yields packets in *sequence* order once either (a) the
//!   next expected sequence number is present, or (b) the head has waited out
//!   the configured `latency` (a loss — emit what we have; the gap is counted).
//! - Duplicates (same extended seq) are dropped and counted.

#![allow(dead_code, unused_variables)]

/// Receiver statistics the RTCP receiver-report path reads (RFC 3550 §6.4.1
/// fields; A.3 expected/lost arithmetic, A.8 jitter).
#[derive(Clone, Copy, Debug, Default)]
pub struct JitterStats {
    /// Packets received (including late-but-kept).
    pub received: u64,
    /// Duplicates dropped.
    pub duplicates: u64,
    /// Gaps declared lost after waiting out the latency.
    pub lost: u64,
    /// Highest extended sequence number seen (A.1).
    pub ext_highest_seq: u64,
    /// Interarrival jitter estimate in timestamp units (A.8).
    pub jitter: f64,
}

/// One buffered packet: the raw datagram plus its arrival running time.
#[derive(Clone, Debug)]
pub struct Held {
    pub datagram: Vec<u8>,
    pub arrival_ns: u64,
    /// Extended (unwrapped) sequence number, assigned at insert.
    pub ext_seq: u64,
}

/// The per-SSRC jitter buffer.
#[derive(Debug)]
pub struct JitterBuffer {
    latency_ns: u64,
}

impl JitterBuffer {
    /// A buffer that holds out-of-order packets up to `latency_ns` before
    /// declaring the missing ones lost.
    pub fn new(latency_ns: u64) -> JitterBuffer {
        JitterBuffer { latency_ns }
    }

    /// Insert one arrived RTP datagram. `seq`/`timestamp` are the parsed
    /// header fields (the caller already has an [`crate::packet::RtpPacket`]
    /// view); `now_ns` is the arrival running time.
    pub fn insert(&mut self, seq: u16, timestamp: u32, datagram: Vec<u8>, now_ns: u64) {
        todo!("agent A: RFC 3550 A.1 extended-seq tracking + ordered insert")
    }

    /// Pop the next packet that is ready at `now_ns` (in-order head, or a
    /// head that has waited out the latency across a loss). `None` = nothing
    /// ready yet.
    pub fn pop_ready(&mut self, now_ns: u64) -> Option<Held> {
        todo!("agent A: ordered pop with loss declaration")
    }

    /// The earliest running time at which a held packet could become ready
    /// (for the session's next-crank scheduling), if any are held.
    pub fn next_deadline_ns(&self) -> Option<u64> {
        todo!("agent A")
    }

    /// Running receiver statistics (feeds RTCP RRs).
    pub fn stats(&self) -> JitterStats {
        todo!("agent A")
    }
}
