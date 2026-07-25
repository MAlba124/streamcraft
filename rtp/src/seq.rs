//! Sequence-number arithmetic: 16-bit wrapping comparison and extended
//! (unwrapped) sequence numbers (RFC 3550 appendix A.1).
//!
//! A.1's `update_seq` is two things at once: a **validity gate** (probation
//! for unheard sources, the `bad_seq` two-in-a-row confirmation for restarts)
//! and a **wrap tracker** (the shifted cycle count that extends the 16-bit
//! field into a monotonically comparable value, consumed by A.3's
//! expected/lost arithmetic and the RR's "extended highest sequence number"
//! field, §6.4.1). [`ExtendedSeq`] is that routine as a state machine; every
//! branch mirrors the A.1 code and is cited against it.
//!
//! Extended values live in an epoch offset one full cycle up ([`EXT_EPOCH`]):
//! a packet misordered to *before* the first-received sequence number (A.1's
//! "duplicate or reordered" branch with zero recorded wraps) must map below
//! the base, which an unsigned space starting at zero cannot express. The
//! offset is harmless on the wire: §6.4.1 notes "different receivers within
//! the same session will generate different extensions to the sequence
//! number", and report consumers difference the field between reports
//! (§6.4.4) so a constant offset cancels.

/// One 16-bit sequence-number cycle (A.1's `RTP_SEQ_MOD`).
pub const RTP_SEQ_MOD: u64 = 1 << 16;

/// Extended sequence numbers start one cycle up so pre-base misordered
/// packets extend below the base without underflow (module docs).
pub const EXT_EPOCH: u64 = RTP_SEQ_MOD;

/// A.1: a sequence number is valid "if it is no more than MAX_DROPOUT ahead
/// of s->max_seq" — typical value for a maximum dropout of 1 minute at 50
/// packets/second, kept "a small fraction of the 16-bit sequence number
/// space" so post-restart numbers rarely fall in the pre-restart range.
const MAX_DROPOUT: u32 = 3000;

/// A.1: "nor more than MAX_MISORDER behind" — 2 s misordering at 50 pkt/s.
const MAX_MISORDER: u32 = 100;

/// A.1: "the number of sequential packets required before declaring a source
/// valid (parameter MIN_SEQUENTIAL)".
const MIN_SEQUENTIAL: u32 = 2;

/// The outcome of feeding one 16-bit sequence number through A.1's
/// `update_seq` — its 0/1 return, refined so callers also learn *why*.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SeqUpdate {
    /// A valid packet (A.1 `return 1`) with its extended sequence number.
    Valid(u64),
    /// The source is still on probation (A.1 `return 0` while
    /// `s->probation` is non-zero). Carries a *tentative* extended value so a
    /// buffer may store-and-deliver on validation, which A.1 sanctions
    /// ("they MAY be stored and delivered once validation has been
    /// achieved"); the base is re-anchored when probation completes, so
    /// tentative values are advisory only.
    Probation(u64),
    /// "The sequence number made a very large jump" (A.1): first occurrence —
    /// invalid, `bad_seq` armed awaiting the confirming next-higher number.
    Bad,
    /// Two sequential packets after a jump: "assume that the other side
    /// restarted without telling us so just re-sync (i.e., pretend this was
    /// the first packet)" (A.1). State was re-initialized; the carried
    /// extended value is in the *new* (re-based) extension space.
    Restart(u64),
}

/// Tracks the wrap cycles of a 16-bit sequence to produce a monotonically
/// comparable extended sequence (A.1's `cycles + seq`), including A.1's
/// probation and restart handling.
#[derive(Debug)]
pub struct ExtendedSeq {
    /// A.1 `s->max_seq`: highest sequence number seen.
    max_seq: u16,
    /// A.1 `s->cycles`: shifted count of wraps, plus [`EXT_EPOCH`].
    cycles: u64,
    /// A.1 `s->bad_seq`: the sequence one past the last "very large jump",
    /// `None` standing in for the sentinel `RTP_SEQ_MOD + 1`.
    bad_seq: Option<u16>,
    /// A.1 `s->probation`: sequential packets still required for validity.
    probation: u32,
    /// The configured `MIN_SEQUENTIAL` (see [`ExtendedSeq::with_min_sequential`]).
    min_sequential: u32,
    /// Whether the first packet has been seen (the C code's state-allocation
    /// moment, which runs `init_seq` + `max_seq = seq - 1`).
    started: bool,
}

impl Default for ExtendedSeq {
    fn default() -> ExtendedSeq {
        ExtendedSeq::new()
    }
}

impl ExtendedSeq {
    /// A.1 defaults: sources are on probation until `MIN_SEQUENTIAL` (2)
    /// packets arrive in sequence.
    pub fn new() -> ExtendedSeq {
        ExtendedSeq::with_min_sequential(MIN_SEQUENTIAL)
    }

    /// A.1 makes the run length a parameter ("The validity check can be made
    /// stronger requiring more than two packets in sequence"); `1` trusts the
    /// first packet immediately — what a jitter buffer that would rather
    /// deliver than validate wants. Values below 1 are treated as 1.
    pub fn with_min_sequential(min_sequential: u32) -> ExtendedSeq {
        ExtendedSeq {
            max_seq: 0,
            cycles: EXT_EPOCH,
            bad_seq: None,
            probation: 0,
            min_sequential: min_sequential.max(1),
            started: false,
        }
    }

    /// A.1 `init_seq`: re-anchor on `seq` and clear the restart/wrap state.
    /// (The C code also zeroes the A.3 counters `received`/`*_prior`; those
    /// live with the caller's statistics, not here.)
    fn init_seq(&mut self, seq: u16) {
        self.max_seq = seq;
        self.cycles = EXT_EPOCH;
        self.bad_seq = None;
    }

    /// Feed the next received 16-bit sequence — A.1 `update_seq`, returning
    /// the extended value alongside the validity verdict.
    pub fn extend(&mut self, seq: u16) -> SeqUpdate {
        if !self.started {
            // A.1: "When a new source is heard for the first time ...
            // init_seq(s, seq); s->max_seq = seq - 1;
            // s->probation = MIN_SEQUENTIAL;"
            self.started = true;
            self.init_seq(seq);
            self.max_seq = seq.wrapping_sub(1);
            self.probation = self.min_sequential;
        }
        let udelta = seq.wrapping_sub(self.max_seq) as u32;
        if self.probation > 0 {
            // A.1: "Source is not valid until MIN_SEQUENTIAL packets with
            // sequential sequence numbers have been received."
            if seq == self.max_seq.wrapping_add(1) {
                self.probation -= 1;
                self.max_seq = seq;
                if self.probation == 0 {
                    self.init_seq(seq);
                    return SeqUpdate::Valid(self.cycles + seq as u64);
                }
            } else {
                // A.1: a miss re-arms probation with this packet as anchor.
                self.probation = self.min_sequential - 1;
                self.max_seq = seq;
            }
            SeqUpdate::Probation(self.cycles + seq as u64)
        } else if udelta < MAX_DROPOUT {
            // A.1: "in order, with permissible gap".
            if seq < self.max_seq {
                // A.1: "Sequence number wrapped - count another 64K cycle."
                self.cycles += RTP_SEQ_MOD;
            }
            self.max_seq = seq;
            SeqUpdate::Valid(self.cycles + seq as u64)
        } else if udelta <= (RTP_SEQ_MOD as u32) - MAX_MISORDER {
            // A.1: "the sequence number made a very large jump".
            if Some(seq) == self.bad_seq {
                // A.1: two sequential packets — source restart, re-sync.
                self.init_seq(seq);
                SeqUpdate::Restart(self.cycles + seq as u64)
            } else {
                // A.1: "s->bad_seq = (seq + 1) & (RTP_SEQ_MOD-1); return 0;"
                self.bad_seq = Some(seq.wrapping_add(1));
                SeqUpdate::Bad
            }
        } else {
            // A.1: "duplicate or reordered packet" — valid, no state change.
            // Within MAX_MISORDER behind max_seq; numerically *above* max_seq
            // means it slipped in from before the last wrap.
            let ext = if seq <= self.max_seq {
                self.cycles + seq as u64
            } else {
                self.cycles + seq as u64 - RTP_SEQ_MOD
            };
            SeqUpdate::Valid(ext)
        }
    }

    /// The extended highest sequence number seen — A.3's
    /// `extended_max = s->cycles + s->max_seq`, the low half of the RR's
    /// "extended highest sequence number received" field (§6.4.1). `None`
    /// until the source has passed probation.
    pub fn ext_highest(&self) -> Option<u64> {
        (self.started && self.probation == 0).then(|| self.cycles + self.max_seq as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trusting tracker (no probation) fed one sequence, unwrapped.
    fn trusting(first: u16) -> (ExtendedSeq, u64) {
        let mut s = ExtendedSeq::with_min_sequential(1);
        match s.extend(first) {
            SeqUpdate::Valid(ext) => (s, ext),
            other => panic!("first packet not trusted: {other:?}"),
        }
    }

    #[test]
    fn wrap_around_counts_a_cycle() {
        let (mut s, e) = trusting(65534);
        assert_eq!(e, EXT_EPOCH + 65534);
        assert_eq!(s.extend(65535), SeqUpdate::Valid(e + 1));
        // A.1: seq (0) < max_seq (65535) inside the in-order window ⇒ wrap.
        assert_eq!(s.extend(0), SeqUpdate::Valid(e + 2));
        assert_eq!(s.extend(1), SeqUpdate::Valid(e + 3));
        assert_eq!(s.ext_highest(), Some(e + 3));
    }

    #[test]
    fn reorder_within_misorder_window() {
        let (mut s, e100) = trusting(100);
        assert_eq!(s.extend(102), SeqUpdate::Valid(e100 + 2));
        // 101 is behind max_seq by 1 (< MAX_MISORDER): valid, max unmoved.
        assert_eq!(s.extend(101), SeqUpdate::Valid(e100 + 1));
        assert_eq!(s.ext_highest(), Some(e100 + 2));
        // A duplicate of the highest lands in the in-order branch (udelta 0).
        assert_eq!(s.extend(102), SeqUpdate::Valid(e100 + 2));
    }

    #[test]
    fn straggler_from_before_a_wrap_extends_into_the_previous_cycle() {
        let (mut s, e) = trusting(65535);
        assert_eq!(s.extend(0), SeqUpdate::Valid(e + 1));
        // 65534 is numerically above max_seq (0) but within MAX_MISORDER
        // behind it: it belongs to the cycle before the wrap (A.1 reordered).
        assert_eq!(s.extend(65534), SeqUpdate::Valid(e - 1));
    }

    #[test]
    fn straggler_from_before_the_base_stays_below_it() {
        // First packet 3, then a misordered pre-base 65533: with zero wraps
        // recorded the extension must still order below the base — the
        // EXT_EPOCH offset makes that expressible.
        let (mut s, e3) = trusting(3);
        assert_eq!(s.extend(65533), SeqUpdate::Valid(e3 - 6));
    }

    #[test]
    fn large_jump_is_bad_until_confirmed_then_restarts() {
        let (mut s, e) = trusting(1000);
        assert_eq!(s.extend(1001), SeqUpdate::Valid(e + 1));
        // udelta 38999 is past MAX_DROPOUT: first occurrence is invalid.
        assert_eq!(s.extend(40000), SeqUpdate::Bad);
        // The stream may simply continue — bad_seq stays armed.
        assert_eq!(s.extend(1002), SeqUpdate::Valid(e + 2));
        assert_eq!(s.extend(50000), SeqUpdate::Bad);
        // A.1: "If the next packet received carries the next higher sequence
        // number, it is considered the valid start of a new packet sequence".
        assert_eq!(s.extend(50001), SeqUpdate::Restart(EXT_EPOCH + 50001));
        assert_eq!(s.ext_highest(), Some(EXT_EPOCH + 50001));
        assert_eq!(s.extend(50002), SeqUpdate::Valid(EXT_EPOCH + 50002));
    }

    #[test]
    fn max_dropout_boundary() {
        let (mut s, e) = trusting(100);
        // udelta 2999 < MAX_DROPOUT: an in-order (if gappy) advance.
        assert_eq!(s.extend(3099), SeqUpdate::Valid(e + 2999));
        // udelta exactly MAX_DROPOUT is a jump.
        assert_eq!(s.extend(6099), SeqUpdate::Bad);
    }

    #[test]
    fn probation_requires_min_sequential() {
        let mut s = ExtendedSeq::new(); // MIN_SEQUENTIAL = 2
        assert_eq!(s.extend(500), SeqUpdate::Probation(EXT_EPOCH + 500));
        assert_eq!(s.ext_highest(), None);
        // A miss during probation re-anchors instead of validating.
        assert_eq!(s.extend(700), SeqUpdate::Probation(EXT_EPOCH + 700));
        // The next-in-sequence packet completes probation and is valid.
        assert_eq!(s.extend(701), SeqUpdate::Valid(EXT_EPOCH + 701));
        assert_eq!(s.ext_highest(), Some(EXT_EPOCH + 701));
    }
}
