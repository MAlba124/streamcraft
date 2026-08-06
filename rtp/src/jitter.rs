//! The reorder/dejitter buffer as a pure state machine (RFC 3550 §6.4.1's
//! interarrival-jitter model, appendix A.8 estimator).
//!
//! Contract (the `RtpSession` element codes against exactly this):
//! - No threads, no clock reads — all time is passed in as nanoseconds of the
//!   caller's monotonic running time.
//! - `insert` takes ownership of a depacketized-yet-unparsed datagram (the
//!   session hands whole RTP packets through in arrival order).
//! - `pop_ready(now)` yields packets in *sequence* order once either (a) the
//!   next expected sequence number is present, or (b) the head has waited out
//!   the configured `latency` (a loss — emit what we have; the gap is counted
//!   per missing packet, matching §6.4.1's cumulative-loss accounting).
//! - Duplicates (same extended seq) are dropped and counted.
//! - A packet that shows up *after* its gap was declared lost is late-but-
//!   kept: delivered immediately (out of band, below the current expectation)
//!   and un-counted from the losses — §6.4.1: "packets that arrive late are
//!   not counted as lost".
//!
//! Sequence unwrapping and restart detection delegate to [`crate::seq`]
//! (RFC 3550 A.1) with probation disabled — a jitter buffer would rather
//! deliver the first packet than validate the source. The interarrival
//! jitter estimator is A.8's, which measures in *timestamp units*: arrival
//! times must be converted "in the same units" (§6.4.1's `Ri`), hence the
//! payload clock rate is a constructor parameter.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::seq::{ExtendedSeq, SeqUpdate, RTP_SEQ_MOD};

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
    /// The payload's timestamp clock rate in Hz (profile/SDP-provided,
    /// RFC 3550 §5.1) — A.8's arrival-time unit conversion.
    clock_rate: u32,
    /// A.1 unwrapper, probation disabled (module docs).
    seq: ExtendedSeq,
    /// Held packets, ordered by extended sequence: `ext -> (datagram, arrival)`.
    held: BTreeMap<u64, (Vec<u8>, u64)>,
    /// Packets stranded by a source restart (A.1) — the old extension space
    /// is incomparable with the new one, so they drain first, ready now.
    flushing: VecDeque<Held>,
    /// The next extended sequence `pop_ready` expects; `None` until the first
    /// pop anchors the stream (and again after a restart).
    next_expected: Option<u64>,
    /// The extended sequence of the anchor packet — anything below is from
    /// before the base and outside A.3's accounting.
    anchor: Option<u64>,
    /// Positions declared lost, so a late arrival is told apart from a
    /// duplicate of a delivered packet. Pruned to one cycle behind.
    lost_positions: BTreeSet<u64>,
    /// A.8 `s->transit`: relative transit time of the previous packet.
    transit: Option<u32>,
    /// A.8 `s->jitter`, kept in floating point as the appendix suggests.
    jitter: f64,
    received: u64,
    duplicates: u64,
    lost: u64,
}

impl JitterBuffer {
    /// A buffer that holds out-of-order packets up to `latency_ns` before
    /// declaring the missing ones lost. `clock_rate` is the payload's RTP
    /// timestamp rate in Hz, needed by the A.8 jitter estimator (arrival
    /// times enter the estimator "in the same units" as the media timestamp).
    pub fn new(latency_ns: u64, clock_rate: u32) -> JitterBuffer {
        JitterBuffer {
            latency_ns,
            clock_rate,
            seq: ExtendedSeq::with_min_sequential(1),
            held: BTreeMap::new(),
            flushing: VecDeque::new(),
            next_expected: None,
            anchor: None,
            lost_positions: BTreeSet::new(),
            transit: None,
            jitter: 0.0,
            received: 0,
            duplicates: 0,
            lost: 0,
        }
    }

    /// Insert one arrived RTP datagram. `seq`/`timestamp` are the parsed
    /// header fields (the caller already has an [`crate::packet::RtpPacket`]
    /// view); `now_ns` is the arrival running time.
    pub fn insert(&mut self, seq: u16, timestamp: u32, datagram: Vec<u8>, now_ns: u64) {
        let ext = match self.seq.extend(seq) {
            SeqUpdate::Valid(ext) => ext,
            SeqUpdate::Restart(ext) => {
                self.flush_on_restart();
                ext
            }
            // A.1-invalid: the first packet of a large jump. (Probation is
            // disabled here, so that arm is only defensive.)
            SeqUpdate::Bad | SeqUpdate::Probation(_) => return,
        };
        // §6.4.1: jitter "SHOULD be calculated continuously as each data
        // packet i is received ... in order of arrival (not necessarily in
        // sequence)" — so before any duplicate/late classification.
        self.update_jitter(timestamp, now_ns);
        if let Some(nx) = self.next_expected {
            if ext < nx {
                if self.lost_positions.remove(&ext) {
                    // §6.4.1: "packets that arrive late are not counted as
                    // lost" — un-count the declared loss, keep the packet
                    // (it pops immediately, being the smallest held).
                    self.lost -= 1;
                } else if self.anchor.is_some_and(|a| ext >= a) {
                    // Its position was already delivered: a duplicate.
                    self.duplicates += 1;
                    return;
                }
                // Below the anchor: a pre-base straggler — keep it too.
                self.received += 1;
                self.held.insert(ext, (datagram, now_ns));
                return;
            }
        }
        if self.held.contains_key(&ext) {
            self.duplicates += 1;
            return;
        }
        self.received += 1;
        self.held.insert(ext, (datagram, now_ns));
    }

    /// Pop the next packet that is ready at `now_ns` (in-order head, or a
    /// head that has waited out the latency across a loss). `None` = nothing
    /// ready yet.
    pub fn pop_ready(&mut self, now_ns: u64) -> Option<Held> {
        if let Some(h) = self.flushing.pop_front() {
            return Some(h);
        }
        let (&ext, &(_, arrival_ns)) = self.held.first_key_value()?;
        match self.next_expected {
            Some(nx) if ext > nx => {
                // A gap at the head: wait it out up to `latency`, then
                // declare every missing sequence number lost and release.
                if now_ns < arrival_ns.saturating_add(self.latency_ns) {
                    return None;
                }
                self.lost += ext - nx;
                for missing in nx..ext {
                    self.lost_positions.insert(missing);
                }
                self.next_expected = Some(ext + 1);
            }
            Some(nx) if ext == nx => self.next_expected = Some(ext + 1),
            // Late-but-kept head: deliver without moving the expectation.
            Some(_) => {}
            None => {
                // First pop anchors the stream (and re-anchors it after a
                // restart re-based the extension space).
                self.anchor = Some(ext);
                self.next_expected = Some(ext + 1);
            }
        }
        if let Some(nx) = self.next_expected {
            // Keep the declared-lost set to one cycle behind the expectation.
            let cutoff = nx.saturating_sub(RTP_SEQ_MOD);
            self.lost_positions = self.lost_positions.split_off(&cutoff);
        }
        let (datagram, arrival_ns) = self.held.remove(&ext).unwrap();
        Some(Held { datagram, arrival_ns, ext_seq: ext })
    }

    /// The earliest running time at which a held packet could become ready
    /// (for the session's next-crank scheduling), if any are held.
    pub fn next_deadline_ns(&self) -> Option<u64> {
        if let Some(h) = self.flushing.front() {
            return Some(h.arrival_ns);
        }
        let (&ext, &(_, arrival_ns)) = self.held.first_key_value()?;
        match self.next_expected {
            // A gap head becomes ready when its wait expires...
            Some(nx) if ext > nx => Some(arrival_ns.saturating_add(self.latency_ns)),
            // ...an in-order (or late-kept) head was ready on arrival.
            _ => Some(arrival_ns),
        }
    }

    /// Running receiver statistics (feeds RTCP RRs).
    pub fn stats(&self) -> JitterStats {
        JitterStats {
            received: self.received,
            duplicates: self.duplicates,
            lost: self.lost,
            ext_highest_seq: self.seq.ext_highest().unwrap_or(0),
            jitter: self.jitter,
        }
    }

    /// A.8's estimator: `transit = arrival - ts` in timestamp units with
    /// `u_int32` wrap-around, `J += (|d| - J)/16` (§6.4.1's
    /// `J(i) = J(i-1) + (|D(i-1,i)| - J(i-1))/16`).
    fn update_jitter(&mut self, timestamp: u32, now_ns: u64) {
        // "arrival, the current time in the same units" (A.8): running ns →
        // timestamp ticks via the clock rate, truncated to 32 bits exactly
        // like A.8's u_int32 arithmetic.
        let arrival = (now_ns as u128 * self.clock_rate as u128 / 1_000_000_000) as u32;
        let transit = arrival.wrapping_sub(timestamp);
        if let Some(prev) = self.transit {
            // First packet has no D(i-1,i) — §6.4.1 defines D over a *pair*.
            let d = (transit.wrapping_sub(prev) as i32 as i64).abs() as f64;
            self.jitter += (d - self.jitter) / 16.0;
        }
        self.transit = Some(transit);
    }

    /// A.1 restart: "Since multiple complete sequence number cycles may have
    /// been missed", the extension space was re-based — old held packets are
    /// incomparable with new ones, so drain them (in order, ready now) and
    /// re-anchor. Cumulative counters are kept monotone rather than reset
    /// (A.1's `init_seq` zeroes them for A.3's in-place interval arithmetic;
    /// our RR builder differences explicit snapshots and clamps instead —
    /// see `rtcp::build_receiver_report`). The A.8 transit is dropped: a
    /// restarted source may carry a new random timestamp offset (§5.1), and
    /// one bogus |d| spike would pollute the estimate for many packets.
    fn flush_on_restart(&mut self) {
        for (ext_seq, (datagram, arrival_ns)) in std::mem::take(&mut self.held) {
            self.flushing.push_back(Held { datagram, arrival_ns, ext_seq });
        }
        self.lost_positions.clear();
        self.next_expected = None;
        self.anchor = None;
        self.transit = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seq::EXT_EPOCH;

    const MS: u64 = 1_000_000;
    /// 20 ms of latency, video clock (RFC 3551 §6: 90 kHz).
    fn buf() -> JitterBuffer {
        JitterBuffer::new(20 * MS, 90_000)
    }

    fn pkt(seq: u16) -> Vec<u8> {
        vec![seq as u8]
    }

    #[test]
    fn in_order_passthrough_is_ready_immediately() {
        let mut b = buf();
        b.insert(10, 0, pkt(10), 0);
        let h = b.pop_ready(0).expect("in-order head is ready at arrival");
        assert_eq!(h.ext_seq, EXT_EPOCH + 10);
        assert_eq!(h.datagram, pkt(10));
        assert_eq!(h.arrival_ns, 0);
        b.insert(11, 3000, pkt(11), MS);
        assert_eq!(b.pop_ready(MS).unwrap().ext_seq, EXT_EPOCH + 11);
        assert!(b.pop_ready(MS).is_none());
        let s = b.stats();
        assert_eq!((s.received, s.duplicates, s.lost), (2, 0, 0));
    }

    #[test]
    fn reorder_within_latency_pops_in_sequence() {
        let mut b = buf();
        b.insert(1, 0, pkt(1), 0);
        assert_eq!(b.pop_ready(0).unwrap().ext_seq, EXT_EPOCH + 1);
        b.insert(3, 6000, pkt(3), MS);
        // The gap (seq 2) is waited out — nothing ready inside the window.
        assert!(b.pop_ready(2 * MS).is_none());
        b.insert(2, 3000, pkt(2), 2 * MS);
        assert_eq!(b.pop_ready(2 * MS).unwrap().ext_seq, EXT_EPOCH + 2);
        assert_eq!(b.pop_ready(2 * MS).unwrap().ext_seq, EXT_EPOCH + 3);
        assert_eq!(b.stats().lost, 0);
    }

    #[test]
    fn ordering_holds_across_the_16_bit_wrap() {
        let mut b = buf();
        b.insert(65535, 0, pkt(255), 0);
        assert_eq!(b.pop_ready(0).unwrap().ext_seq, EXT_EPOCH + 65535);
        // 1 overtakes 0 across the wrap; both extend into the next cycle.
        b.insert(1, 6000, pkt(1), MS);
        assert!(b.pop_ready(MS).is_none()); // waiting on 0
        b.insert(0, 3000, pkt(0), 2 * MS);
        assert_eq!(b.pop_ready(2 * MS).unwrap().ext_seq, EXT_EPOCH + 65536);
        assert_eq!(b.pop_ready(2 * MS).unwrap().ext_seq, EXT_EPOCH + 65537);
    }

    #[test]
    fn loss_is_declared_after_the_latency_expires() {
        let mut b = buf();
        b.insert(1, 0, pkt(1), 0);
        assert_eq!(b.pop_ready(0).unwrap().ext_seq, EXT_EPOCH + 1);
        b.insert(4, 9000, pkt(4), MS);
        assert!(b.pop_ready(MS).is_none());
        assert_eq!(b.next_deadline_ns(), Some(21 * MS)); // arrival + latency
        assert!(b.pop_ready(20 * MS).is_none());
        // Past the deadline the head is released; seqs 2 and 3 are lost.
        let h = b.pop_ready(21 * MS).expect("head released across the loss");
        assert_eq!(h.ext_seq, EXT_EPOCH + 4);
        assert_eq!(b.stats().lost, 2);
        // The stream continues in order afterwards.
        b.insert(5, 12000, pkt(5), 22 * MS);
        assert_eq!(b.pop_ready(22 * MS).unwrap().ext_seq, EXT_EPOCH + 5);
    }

    #[test]
    fn duplicates_are_dropped_and_counted() {
        let mut b = buf();
        b.insert(7, 0, pkt(7), 0);
        b.insert(7, 0, pkt(7), MS); // duplicate while held
        assert_eq!(b.stats().duplicates, 1);
        assert_eq!(b.pop_ready(MS).unwrap().ext_seq, EXT_EPOCH + 7);
        b.insert(7, 0, pkt(7), 2 * MS); // duplicate of a delivered packet
        assert!(b.pop_ready(2 * MS).is_none());
        let s = b.stats();
        assert_eq!((s.received, s.duplicates, s.lost), (1, 2, 0));
    }

    #[test]
    fn late_arrival_after_declared_loss_is_kept_and_uncounted() {
        let mut b = buf();
        b.insert(1, 0, pkt(1), 0);
        assert_eq!(b.pop_ready(0).unwrap().ext_seq, EXT_EPOCH + 1);
        b.insert(3, 6000, pkt(3), MS);
        assert_eq!(b.pop_ready(21 * MS).unwrap().ext_seq, EXT_EPOCH + 3);
        assert_eq!(b.stats().lost, 1);
        // Seq 2 finally shows up: late-but-kept, and no longer a loss
        // (§6.4.1: late packets are not counted as lost).
        b.insert(2, 3000, pkt(2), 25 * MS);
        let h = b.pop_ready(25 * MS).expect("late packet delivered");
        assert_eq!(h.ext_seq, EXT_EPOCH + 2);
        let s = b.stats();
        assert_eq!((s.received, s.duplicates, s.lost), (3, 0, 0));
        // A second copy of it *is* a duplicate now.
        b.insert(2, 3000, pkt(2), 26 * MS);
        assert_eq!(b.stats().duplicates, 1);
    }

    #[test]
    fn stats_arithmetic_and_highest_seq() {
        let mut b = buf();
        for seq in [5u16, 6, 8, 6] {
            // 6 twice: one duplicate. 7 missing.
            b.insert(seq, seq as u32 * 3000, pkt(seq), seq as u64 * MS);
        }
        let s = b.stats();
        assert_eq!(s.received, 3);
        assert_eq!(s.duplicates, 1);
        assert_eq!(s.ext_highest_seq, EXT_EPOCH + 8);
        assert_eq!(s.lost, 0); // nothing declared until the wait expires
        assert_eq!(b.pop_ready(8 * MS).unwrap().ext_seq, EXT_EPOCH + 5);
        assert_eq!(b.pop_ready(8 * MS).unwrap().ext_seq, EXT_EPOCH + 6);
        assert_eq!(b.pop_ready(8 * MS + 20 * MS).unwrap().ext_seq, EXT_EPOCH + 8);
        assert_eq!(b.stats().lost, 1);
    }

    #[test]
    fn jitter_estimator_stays_zero_on_a_perfectly_paced_stream() {
        // 8 kHz audio (RFC 3551 §6), 20 ms packets: 160 ticks per packet.
        let mut b = JitterBuffer::new(20 * MS, 8000);
        for i in 0u64..50 {
            b.insert(i as u16, (i * 160) as u32, pkt(i as u16), i * 20 * MS);
            b.pop_ready(i * 20 * MS);
        }
        // Constant transit ⇒ every D(i-1,i) = 0 ⇒ J never moves (A.8).
        assert_eq!(b.stats().jitter, 0.0);
    }

    #[test]
    fn jitter_estimator_spikes_then_converges_back_toward_zero() {
        let mut b = JitterBuffer::new(20 * MS, 8000);
        for i in 0u64..10 {
            b.insert(i as u16, (i * 160) as u32, pkt(i as u16), i * 20 * MS);
        }
        assert_eq!(b.stats().jitter, 0.0);
        // One packet 5 ms late: |D| = 40 ticks at 8 kHz ⇒ J jumps by 40/16.
        b.insert(10, 10 * 160, pkt(10), 10 * 20 * MS + 5 * MS);
        let spiked = b.stats().jitter;
        assert!((spiked - 2.5).abs() < 1e-9, "J = {spiked}");
        // Back on the grid: the next |D| is the 40-tick return, then zeros —
        // J decays by 15/16 per packet toward 0.
        let mut last = f64::MAX;
        for i in 11u64..80 {
            b.insert(i as u16, (i * 160) as u32, pkt(i as u16), i * 20 * MS);
            let j = b.stats().jitter;
            assert!(j < last || j == 0.0);
            last = j;
        }
        assert!(b.stats().jitter < 0.1, "J = {}", b.stats().jitter);
    }

    #[test]
    fn restart_flushes_held_packets_and_reanchors() {
        let mut b = buf();
        b.insert(100, 0, pkt(100), 0);
        assert_eq!(b.pop_ready(0).unwrap().ext_seq, EXT_EPOCH + 100);
        b.insert(103, 9000, pkt(103), MS); // held: gap at 101..103
        // A.1 large jump: first occurrence dropped, next-higher confirms a
        // restart — the held packet flushes out, then the new stream flows.
        b.insert(40000, 50_000, pkt(0), 2 * MS);
        b.insert(40001, 50_160, pkt(1), 3 * MS);
        let h = b.pop_ready(3 * MS).expect("stranded packet flushed");
        assert_eq!(h.ext_seq, EXT_EPOCH + 103);
        let h = b.pop_ready(3 * MS).expect("restarted stream anchors");
        assert_eq!(h.ext_seq, EXT_EPOCH + 40001);
        b.insert(40002, 50_320, pkt(2), 4 * MS);
        assert_eq!(b.pop_ready(4 * MS).unwrap().ext_seq, EXT_EPOCH + 40002);
        // The jump's first packet was A.1-invalid: not received, not a dup.
        assert_eq!(b.stats().received, 4);
    }

    #[test]
    fn next_deadline_reports_arrival_for_ready_heads() {
        let mut b = buf();
        assert_eq!(b.next_deadline_ns(), None);
        b.insert(1, 0, pkt(1), 5 * MS);
        assert_eq!(b.next_deadline_ns(), Some(5 * MS));
    }
}
