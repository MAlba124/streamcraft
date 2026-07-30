//! CELT per-band time-frequency change decode — RFC 6716 §4.3.4.5
//! (with the §4.3.1 framing context).
//!
//! [`crate::celt_tf_adjust`] owns the four §4.3.4.5 lookup *tables*
//! (Tables 60–63) and the `tf_select` impact predicate, but it
//! explicitly does **not** read any range-coded symbols. This module
//! owns that read path: the band loop that decodes the per-band
//! `tf_change[b]` choices, the gated `tf_select` flag, and turns the
//! pair into the per-band integer TF adjustment the §4.3.4.5 Hadamard
//! step consumes.
//!
//! ## Decode order (RFC 6716 §4.3.4.5, p. 119)
//!
//! For each coded band `b` in `start..end` (the §4.3 Table-55 band
//! range — `0..21` for CELT-only, `17..21` for Hybrid), in ascending
//! order:
//!
//! 1. **First coded band.** The TF choice is decoded directly — the
//!    decoded bit *is* the absolute TF choice for that band (`0` or
//!    `1`, indexing the Table-60..63 column). Transient frame → PDF
//!    `{3, 1}/4` (`dec_bit_logp(2)`); non-transient frame → PDF
//!    `{15, 1}/16` (`dec_bit_logp(4)`).
//! 2. **Subsequent bands.** The RFC: "the TF choice is coded relative
//!    to the previous TF choice with probability `{15, 1}/16` for
//!    transient frames and `{31, 1}/32` otherwise." A *relative*
//!    binary choice is a difference bit: the decoded symbol toggles
//!    (XORs onto) the running choice carried from the previous band,
//!    so `tf_change[b] = tf_change[b-1] XOR diff`. Transient frame →
//!    PDF `{15, 1}/16` (`dec_bit_logp(4)`); non-transient frame → PDF
//!    `{31, 1}/32` (`dec_bit_logp(5)`).
//!
//! After every band's `tf_change` is known, the §4.3.1 `tf_select`
//! flag is read — but only when it can actually change at least one
//! band's adjustment for the current `(frame_size, transient)` and the
//! decoded set of `tf_change` choices (RFC 6716 §4.3.1: "the tf_select
//! flag … is only decoded if it can have an impact on the result
//! knowing the value of all per-band tf_change flags").
//! [`crate::celt_tf_adjust::celt_tf_select_can_affect`] is that gate.
//! When the gate is closed, `tf_select` is implicitly `0` and no bit is
//! consumed. When open, `tf_select` is decoded with the `{1, 1}/2` PDF
//! (`dec_bit_logp(1)`).
//!
//! Finally each band's adjustment is looked up via
//! [`crate::celt_tf_adjust::celt_tf_adjustment`].
//!
//! ## The "relative" reading
//!
//! The RFC describes subsequent-band coding only as "coded relative to
//! the previous TF choice" with a single binary PDF. For a binary
//! symbol the sole coherent meaning of "relative to the previous
//! choice" is a *difference / toggle*: the high-probability outcome
//! repeats the previous band's choice (the `{15,1}/16` and `{31,1}/32`
//! PDFs put 15/16 and 31/32 of the mass on the "no change" symbol,
//! matching the §4.3 observation that TF resolution is "relatively
//! static from band to band"), and the low-probability outcome flips
//! it. This module documents that interpretation; it is the only
//! reading consistent with both the binary PDF and the word
//! "relative".
//!
//! ## Provenance
//!
//! RFC 6716 §4.3.4.5 + §4.3.1 prose, `docs/audio/opus/rfc6716-opus.txt`
//! pp. 107, 119–120. No external library source consulted; no
//! cross-check against any reference implementation.

use crate::celt_band_layout::CeltFrameSize;
use crate::celt_tf_adjust::{celt_tf_adjustment, celt_tf_select_can_affect, TfAdjustment};
use crate::range_decoder::RangeDecoder;

/// `dec_bit_logp` argument for the `{3, 1}/4` PDF (first band,
/// transient): `p(1) = 1/4 = 2^-2`.
const LOGP_3_1_4: u32 = 2;
/// `dec_bit_logp` argument for the `{15, 1}/16` PDF
/// (first band non-transient / subsequent band transient):
/// `p(1) = 1/16 = 2^-4`.
const LOGP_15_1_16: u32 = 4;
/// `dec_bit_logp` argument for the `{31, 1}/32` PDF (subsequent band,
/// non-transient): `p(1) = 1/32 = 2^-5`.
const LOGP_31_1_32: u32 = 5;
/// `dec_bit_logp` argument for the `{1, 1}/2` PDF (`tf_select`):
/// `p(1) = 1/2 = 2^-1`.
const LOGP_1_1_2: u32 = 1;

/// Result of the §4.3.4.5 TF-change decode for one CELT frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TfDecode {
    /// The per-coded-band absolute TF choice (`false = 0`, `true = 1`),
    /// in band-ascending order over the coded range `start..end`. Index
    /// `i` corresponds to band `start + i`.
    pub tf_change: Vec<bool>,
    /// The decoded `tf_select` flag. `false` when the §4.3.1 gate was
    /// closed and no bit was read.
    pub tf_select: bool,
    /// The per-coded-band TF resolution adjustment (Table 60–63 cell),
    /// in band-ascending order — parallel to [`Self::tf_change`].
    pub adjustments: Vec<TfAdjustment>,
}

impl TfDecode {
    /// Number of coded bands decoded.
    #[inline]
    pub fn len(&self) -> usize {
        self.tf_change.len()
    }

    /// Whether no bands were coded (an empty coded range).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tf_change.is_empty()
    }
}

/// Decode the per-band `tf_change` flags, the gated `tf_select` flag,
/// and the resulting per-band TF adjustments for one CELT frame.
///
/// * `rd` — the shared range decoder, positioned per Table 56 right
///   after coarse-energy decode (the §4.3.1 / §4.3.4.5 `tf_change`
///   symbols are next in bitstream order).
/// * `frame_size` — the CELT frame size (Table-55 column selector).
/// * `transient` — the §4.3.1 global transient flag (from the
///   [`crate::celt_header::CeltHeaderPrefix`]).
/// * `start_band` / `end_band` — the coded band range `[start, end)`
///   from [`crate::celt_band_layout`] (`celt_first_coded_band` /
///   `celt_end_coded_band`). `start = 0` for CELT-only, `17` for
///   Hybrid; `end = 21`.
///
/// Returns the decoded [`TfDecode`]. When `start_band >= end_band`
/// (empty coded range) no symbols are read and the result is empty
/// with `tf_select = false`.
pub fn decode_tf(
    rd: &mut RangeDecoder<'_>,
    frame_size: CeltFrameSize,
    transient: bool,
    start_band: usize,
    end_band: usize,
) -> TfDecode {
    let n = end_band.saturating_sub(start_band);
    let mut tf_change: Vec<bool> = Vec::with_capacity(n);

    // ---- per-band tf_change flags ----------------------------------
    let mut prev = false;
    for i in 0..n {
        let choice = if i == 0 {
            // First coded band: the bit is the absolute choice.
            let logp = if transient { LOGP_3_1_4 } else { LOGP_15_1_16 };
            rd.dec_bit_logp(logp) == 1
        } else {
            // Subsequent bands: a difference bit relative to the
            // previous band's choice.
            let logp = if transient {
                LOGP_15_1_16
            } else {
                LOGP_31_1_32
            };
            let diff = rd.dec_bit_logp(logp) == 1;
            prev ^ diff
        };
        tf_change.push(choice);
        prev = choice;
    }

    // ---- gated tf_select -------------------------------------------
    let tf_select = if celt_tf_select_can_affect(frame_size, transient, &tf_change) {
        rd.dec_bit_logp(LOGP_1_1_2) == 1
    } else {
        false
    };

    // ---- per-band adjustment lookup --------------------------------
    let adjustments = tf_change
        .iter()
        .map(|&c| celt_tf_adjustment(frame_size, transient, tf_select, c))
        .collect();

    TfDecode {
        tf_change,
        tf_select,
        adjustments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_tf_adjust::{
        TF_ADJ_NONTRANSIENT_SELECT1, TF_ADJ_TRANSIENT_SELECT0, TF_ADJ_TRANSIENT_SELECT1,
    };

    // ---- A bit-exact range *encoder* fixture, transcribed from the
    //      normative RFC 6716 §5.1.1 / §5.1.1.1 / §5.1.1.2 / §5.1.5
    //      procedure (the inverse of this crate's §4.1 `RangeDecoder`).
    //      It lets the tests drive a known symbol stream through the
    //      real decoder. (The encoder is a TEST fixture only; the codec
    //      never runs a range encoder — the crate ships a decoder.)
    //
    //      Provenance: `docs/audio/opus/rfc6716-opus.txt` §5.1, pp.
    //      133–135. No external library source consulted.

    /// `val` is the §5.1.1 low end of the current range; `rng` the
    /// range size. `rem` is the buffered output byte (`-1` = none);
    /// `ext` the §5.1.1.2 carry-extension count.
    struct RangeEncoder {
        val: u32,
        rng: u32,
        out: Vec<u8>,
        rem: i32,
        ext: u32,
    }

    impl RangeEncoder {
        fn new() -> Self {
            // §4.1.1 initialization fixes rng = 128 on the decode side;
            // the encoder begins with the full 2**31 range and the same
            // 9-bit-at-a-time output discipline.
            Self {
                val: 0,
                rng: 0x8000_0000,
                out: Vec::new(),
                rem: -1,
                ext: 0,
            }
        }

        /// §5.1.1.2 carry propagation + output buffering. `c` is a
        /// 9-bit value (8 data bits + carry bit).
        fn carry_out(&mut self, c: u32) {
            if c == 0xFF {
                self.ext += 1;
                return;
            }
            let b = (c >> 8) & 1; // carry bit
            if self.rem != -1 {
                self.out.push((self.rem as u32 + b) as u8);
            }
            if self.ext > 0 {
                let fill: u8 = if b == 1 { 0x00 } else { 0xFF };
                for _ in 0..self.ext {
                    self.out.push(fill);
                }
                self.ext = 0;
            }
            self.rem = (c & 0xFF) as i32;
        }

        /// §5.1.1.1 renormalization: while `rng <= 2**23`, ship the top
        /// 9 bits of `val` and shift.
        fn renorm(&mut self) {
            while self.rng <= (1 << 23) {
                self.carry_out(self.val >> 23);
                self.val = (self.val << 8) & 0x7FFF_FFFF;
                self.rng <<= 8;
            }
        }

        /// §5.1.2.2 `ec_enc_bit_logp`: encode a binary symbol with
        /// `p(1) = 2^-logp`. Equivalent to `ec_encode` with the
        /// 3-tuples documented in §5.1.2.2.
        fn bit_logp(&mut self, bit: u32, logp: u32) {
            let r = self.rng;
            let s = r >> logp; // rng/ft with ft = 1<<logp
            if bit == 0 {
                // k=0: (fl=0, fh=ft-1, ft). fl == 0 → val unchanged,
                // rng = rng - (rng/ft)*(ft-fh) = r - s.
                self.rng = r - s;
            } else {
                // k=1: (fl=ft-1, fh=ft, ft). fl > 0 →
                // val += rng - (rng/ft)*(ft-fl) = r - s ; rng = s.
                self.val += r - s;
                self.rng = s;
            }
            self.renorm();
        }

        /// §5.1.5 finalize: emit a value inside `[val, val+rng)` with
        /// the most trailing zero bits, then flush the carry buffer.
        fn finish(mut self) -> Vec<u8> {
            // Choose `end` = smallest multiple of a large power of two
            // within [val, val+rng). We follow the documented intent:
            // pick the value with the most trailing zeros. A simple,
            // correct realisation: start from the full mask and shrink.
            let val = self.val;
            let rng = self.rng;
            let mut end = val;
            // Find the largest b such that (val rounded up to a multiple
            // of 2**b) + (2**b - 1) < val + rng, i.e. a 2**b-aligned
            // value lies in range with room for arbitrary low bits.
            for b in (0..=31u32).rev() {
                let step = 1u32 << b;
                // Round val up to the next multiple of step.
                let aligned = val.wrapping_add(step - 1) & !(step - 1);
                // Need aligned in [val, val+rng) and aligned+step-1 also
                // in range (so the trailing bits are free). Use u64 to
                // avoid overflow at the top of the range.
                let hi = u64::from(val) + u64::from(rng);
                if u64::from(aligned) >= u64::from(val)
                    && u64::from(aligned) + u64::from(step) - 1 < hi
                {
                    end = aligned;
                    break;
                }
            }
            // Ship `end` through the carry buffer, MSB-first, §5.1.5.
            while end != 0 {
                self.carry_out(end >> 23);
                end = (end << 8) & 0x7FFF_FFFF;
            }
            // Flush: if rem holds real data or ext is pending, send 9
            // zero bits to drain the carry buffer.
            if self.rem != -1 || self.ext > 0 {
                self.carry_out(0);
            }
            // Emit the final buffered byte.
            if self.rem != -1 {
                self.out.push(self.rem as u8);
            }
            if self.ext > 0 {
                for _ in 0..self.ext {
                    self.out.push(0xFF);
                }
            }
            self.out
        }
    }

    /// Encode a list of `(bit, logp)` symbols and return the buffer.
    fn encode_bits(symbols: &[(u32, u32)]) -> Vec<u8> {
        let mut enc = RangeEncoder::new();
        for &(bit, logp) in symbols {
            enc.bit_logp(bit, logp);
        }
        enc.finish()
    }

    /// Round-trip sanity: the encoder/decoder pair agrees on a sequence
    /// of mixed-logp bits. This validates the test fixture itself
    /// before it's used to drive TF decode assertions.
    #[test]
    fn fixture_encoder_round_trips_through_real_decoder() {
        let symbols = [
            (1u32, 2u32),
            (0, 4),
            (1, 5),
            (0, 1),
            (1, 4),
            (1, 1),
            (0, 5),
            (0, 2),
        ];
        let buf = encode_bits(&symbols);
        let mut rd = RangeDecoder::new(&buf);
        for &(bit, logp) in &symbols {
            assert_eq!(rd.dec_bit_logp(logp), bit, "logp={logp}");
        }
        assert!(!rd.has_error());
    }

    // ---- decode_tf behavioural tests --------------------------------

    /// Empty coded range reads nothing and yields an empty result.
    #[test]
    fn empty_band_range_reads_nothing() {
        let buf = encode_bits(&[(1, 2), (1, 2)]);
        let mut rd = RangeDecoder::new(&buf);
        let before = rd.tell_frac();
        let res = decode_tf(&mut rd, CeltFrameSize::Ms20, true, 21, 21);
        assert!(res.is_empty());
        assert!(!res.tf_select);
        assert!(res.adjustments.is_empty());
        // No symbols consumed.
        assert_eq!(rd.tell_frac(), before);
    }

    /// 2.5 ms (non-transient): tf_select can never affect the result
    /// (Tables 60 == 61), so no tf_select bit is read regardless of the
    /// tf_change choices. With every band choosing 0 the adjustment is
    /// all-zero.
    #[test]
    fn ms2_5_nontransient_no_tf_select_all_choice0() {
        // First band {15,1}/16 → bit 0; three more relative {31,1}/32
        // diff bits → 0 (no toggle). No tf_select bit follows.
        let buf = encode_bits(&[(0, 4), (0, 5), (0, 5), (0, 5)]);
        let mut rd = RangeDecoder::new(&buf);
        let res = decode_tf(&mut rd, CeltFrameSize::Ms2_5, false, 17, 21);
        assert_eq!(res.tf_change, vec![false, false, false, false]);
        assert!(!res.tf_select);
        assert_eq!(res.adjustments, vec![0, 0, 0, 0]);
    }

    /// First-band absolute choice on a transient frame uses the
    /// `{3,1}/4` PDF; a decoded 1 selects column 1 of the transient
    /// table.
    #[test]
    fn first_band_transient_choice1() {
        // 20 ms transient: choice 1 in Table 62 = 0, but we want a
        // frame where tf_select stays closed. Use a single band so the
        // gate depends only on whether Table 62 / 63 differ for that
        // choice. For 20 ms they DO differ (Table 62 col1 = 0 vs Table
        // 63 col1 = -1), so tf_select WILL be read. Provide it.
        // symbols: first band {3,1}/4 → 1 ; tf_select {1,1}/2 → 0.
        let buf = encode_bits(&[(1, 2), (0, 1)]);
        let mut rd = RangeDecoder::new(&buf);
        let res = decode_tf(&mut rd, CeltFrameSize::Ms20, true, 20, 21);
        assert_eq!(res.tf_change, vec![true]);
        assert!(!res.tf_select);
        // 20 ms transient, tf_select=0, choice 1 → Table 62 [3][1] = 0.
        assert_eq!(res.adjustments, vec![TF_ADJ_TRANSIENT_SELECT0[3][1]]);
    }

    /// The "relative" difference coding toggles the running choice:
    /// first band 0, then a diff-1 flips to 1, then diff-0 keeps 1,
    /// then diff-1 flips back to 0.
    #[test]
    fn subsequent_bands_are_relative_toggles() {
        // 10 ms non-transient: first {15,1}/16, subsequent {31,1}/32.
        // bits: 0 (band0=0), 1 (toggle→1), 0 (keep 1), 1 (toggle→0).
        // After these, Tables 60 vs 61 differ on choice 1 (-2 vs -3),
        // and bands 1,2 chose 1 → tf_select matters → read it (=1).
        let buf = encode_bits(&[(0, 4), (1, 5), (0, 5), (1, 5), (1, 1)]);
        let mut rd = RangeDecoder::new(&buf);
        let res = decode_tf(&mut rd, CeltFrameSize::Ms10, false, 17, 21);
        assert_eq!(res.tf_change, vec![false, true, true, false]);
        assert!(
            res.tf_select,
            "tf_select must be read (Tables 60/61 differ)"
        );
        // tf_select=1 → Table 61. choice0→0, choice1→-3.
        assert_eq!(
            res.adjustments,
            vec![
                TF_ADJ_NONTRANSIENT_SELECT1[2][0], // band0 choice0 = 0
                TF_ADJ_NONTRANSIENT_SELECT1[2][1], // band1 choice1 = -3
                TF_ADJ_NONTRANSIENT_SELECT1[2][1], // band2 choice1 = -3
                TF_ADJ_NONTRANSIENT_SELECT1[2][0], // band3 choice0 = 0
            ]
        );
    }

    /// When all bands pick choice 0 on a frame size where Tables differ
    /// only on choice 1 (10 ms non-transient), tf_select is NOT read.
    #[test]
    fn tf_select_skipped_when_all_choice0_10ms() {
        // first {15,1}/16 → 0, three diff {31,1}/32 → 0. No tf_select.
        let buf = encode_bits(&[(0, 4), (0, 5), (0, 5), (0, 5)]);
        let mut rd = RangeDecoder::new(&buf);
        let res = decode_tf(&mut rd, CeltFrameSize::Ms10, false, 17, 21);
        assert_eq!(res.tf_change, vec![false, false, false, false]);
        assert!(!res.tf_select);
        // Both tf_select tables give 0 for choice 0.
        assert_eq!(res.adjustments, vec![0, 0, 0, 0]);
    }

    /// 20 ms transient: Tables 62 and 63 differ on BOTH choices, so a
    /// single coded band of any choice forces a tf_select read.
    #[test]
    fn ms20_transient_tf_select_always_read_nonempty() {
        // band0 {3,1}/4 → 0, tf_select {1,1}/2 → 1.
        let buf = encode_bits(&[(0, 2), (1, 1)]);
        let mut rd = RangeDecoder::new(&buf);
        let res = decode_tf(&mut rd, CeltFrameSize::Ms20, true, 20, 21);
        assert_eq!(res.tf_change, vec![false]);
        assert!(res.tf_select);
        // tf_select=1, choice0 → Table 63 [3][0] = 1.
        assert_eq!(res.adjustments, vec![TF_ADJ_TRANSIENT_SELECT1[3][0]]);
    }

    /// Full 21-band CELT-only decode (start=0): exercises the longest
    /// band loop with an all-zero choice stream (no tf_select for
    /// 5 ms transient when all choice0? Tables 62/63 5ms: col0 1 vs 1
    /// agree, col1 0 vs -1 differ — all choice0 means agree → no
    /// tf_select). Adjustment all = Table 62 [1][0] = 1.
    #[test]
    fn full_celt_only_5ms_transient_all_choice0() {
        // 21 bands: first {3,1}/4 → 0, then 20 diff {15,1}/16 → 0.
        let mut symbols = vec![(0u32, 2u32)];
        for _ in 0..20 {
            symbols.push((0, 4));
        }
        let buf = encode_bits(&symbols);
        let mut rd = RangeDecoder::new(&buf);
        let res = decode_tf(&mut rd, CeltFrameSize::Ms5, true, 0, 21);
        assert_eq!(res.len(), 21);
        assert!(res.tf_change.iter().all(|&c| !c));
        assert!(
            !res.tf_select,
            "5ms transient all-choice0 agrees → no tf_select"
        );
        // Table 62 [1][0] = 1 for every band.
        assert!(res
            .adjustments
            .iter()
            .all(|&a| a == TF_ADJ_TRANSIENT_SELECT0[1][0]));
    }

    /// tf_select=1 path on a transient frame routes to Table 63.
    #[test]
    fn tf_select_one_routes_transient_to_table_63() {
        // 10 ms transient: first {3,1}/4 → 1, one diff {15,1}/16 → 0
        // (keep 1). Tables 62/63 10ms col1: 0 vs -1 differ → read
        // tf_select = 1.
        let buf = encode_bits(&[(1, 2), (0, 4), (1, 1)]);
        let mut rd = RangeDecoder::new(&buf);
        let res = decode_tf(&mut rd, CeltFrameSize::Ms10, true, 19, 21);
        assert_eq!(res.tf_change, vec![true, true]);
        assert!(res.tf_select);
        assert_eq!(
            res.adjustments,
            vec![
                TF_ADJ_TRANSIENT_SELECT1[2][1],
                TF_ADJ_TRANSIENT_SELECT1[2][1]
            ]
        );
    }

    /// `len` / `is_empty` reflect the coded band count.
    #[test]
    fn len_and_is_empty_track_band_count() {
        let buf = encode_bits(&[(0, 4), (0, 5), (0, 5), (0, 5)]);
        let mut rd = RangeDecoder::new(&buf);
        let res = decode_tf(&mut rd, CeltFrameSize::Ms2_5, false, 17, 21);
        assert_eq!(res.len(), 4);
        assert!(!res.is_empty());

        let buf2 = encode_bits(&[(0, 1)]);
        let mut rd2 = RangeDecoder::new(&buf2);
        let empty = decode_tf(&mut rd2, CeltFrameSize::Ms20, false, 21, 21);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    /// Cross-check: every produced adjustment equals the direct
    /// `celt_tf_adjustment` lookup for the decoded `(tf_select,
    /// tf_change[b])`. Guards against the band loop and the adjustment
    /// table drifting apart.
    #[test]
    fn adjustments_match_direct_table_lookup() {
        let buf = encode_bits(&[(1, 2), (1, 4), (0, 4), (1, 4), (1, 1)]);
        let mut rd = RangeDecoder::new(&buf);
        let res = decode_tf(&mut rd, CeltFrameSize::Ms20, true, 17, 21);
        for (i, &c) in res.tf_change.iter().enumerate() {
            let expected = celt_tf_adjustment(CeltFrameSize::Ms20, true, res.tf_select, c);
            assert_eq!(res.adjustments[i], expected, "band {i}");
        }
    }
}
