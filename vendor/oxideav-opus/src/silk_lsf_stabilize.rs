//! SILK Normalized LSF stabilization — RFC 6716 §4.2.7.5.4.
//!
//! The §4.2.7.5.3 reconstruction ([`crate::silk_lsf_recon`]) produces a
//! candidate `NLSF_Q15[]` vector. Before those coefficients can be
//! interpolated (§4.2.7.5.5) and converted back to LPC coefficients
//! (§4.2.7.5.6), the decoder must guarantee a minimum spacing between
//! consecutive entries — both to keep the synthesis filter stable and to
//! match the encoder's own stabilization so the two stay bit-exact.
//!
//! This module implements `silk_NLSF_stabilize()`:
//!
//!  1. Up to 20 distortion-minimizing adjustment passes. Each pass finds
//!     the coefficient pair with the smallest `NLSF_Q15[i] -
//!     NLSF_Q15[i-1] - NDeltaMin_Q15[i]` (ties broken by lower `i`). If
//!     that value is non-negative the constraints are met and the
//!     procedure stops. Otherwise the offending pair is re-centred per
//!     the §4.2.7.5.4 `min_center` / `max_center` / `center_freq`
//!     formula, with the two boundary cases (`i == 0`, `i == d_LPC`)
//!     handled specially.
//!
//!  2. A single fallback pass that runs after the 20th adjustment: sort
//!     ascending, then a forward `max` sweep and a backward `min` sweep
//!     that mechanically enforce the spacing. Per RFC 8251 §7 the forward
//!     sweep's `NLSF_Q15[i-1] + NDeltaMin_Q15[i]` addition is performed
//!     with 16-bit saturating addition to avoid an integer wrap-around on
//!     adversarial inputs.
//!
//! The boundary conventions from §4.2.7.5.4 are: `NLSF_Q15[-1] = 0` and
//! `NLSF_Q15[d_LPC] = 32768`. `NDeltaMin_Q15[]` (Table 25) has `d_LPC + 1`
//! entries — one per coefficient plus a trailing entry for the spacing
//! against the implicit `32768` upper edge.

use crate::silk_lsf_recon::NlsfReconstructed;
use crate::silk_lsf_stage2::{D_LPC_MAX, D_LPC_NB_MB, D_LPC_WB};
use crate::toc::Bandwidth;
use crate::Error;

/// The implicit upper edge `NLSF_Q15[d_LPC]` from §4.2.7.5.4.
const NLSF_UPPER_Q15: i32 = 32768;

// =====================================================================
// Table 25 — Minimum Spacing for Normalized LSF Coefficients.
//
// `NDeltaMin_Q15[k]` is the minimum allowed value of
// `NLSF_Q15[k] - NLSF_Q15[k-1]`. There are `d_LPC + 1` entries: indices
// `0..d_LPC` cover the real coefficients, and the trailing index
// `d_LPC` covers the spacing against the implicit `NLSF_Q15[d_LPC] =
// 32768` upper edge.
//
// NB and MB share one column (11 entries); WB has its own (17 entries).
// =====================================================================

#[rustfmt::skip]
const NDELTA_MIN_Q15_NB_MB: [i32; D_LPC_NB_MB + 1] = [
    250, 3, 6, 3, 3, 3, 4, 3, 3, 3, 461,
];

#[rustfmt::skip]
const NDELTA_MIN_Q15_WB: [i32; D_LPC_WB + 1] = [
    100, 3, 40, 3, 3, 3, 5, 14, 14, 10, 11, 3, 8, 9, 7, 3, 347,
];

/// Look up the Table 25 minimum-spacing column for the given bandwidth.
///
/// Returns `Error::MalformedPacket` for SWB / FB, which SILK never sees
/// after the §4.2.2 hybrid split.
fn ndelta_min_q15(bandwidth: Bandwidth) -> Result<&'static [i32], Error> {
    match bandwidth {
        Bandwidth::Nb | Bandwidth::Mb => Ok(&NDELTA_MIN_Q15_NB_MB[..]),
        Bandwidth::Wb => Ok(&NDELTA_MIN_Q15_WB[..]),
        _ => Err(Error::MalformedPacket),
    }
}

/// 16-bit saturating addition, per RFC 8251 §7's `silk_ADD_SAT16`.
///
/// Used only in the fallback forward sweep so that an adversarially large
/// `NLSF_Q15[i-1] + NDeltaMin_Q15[i]` cannot wrap around `i16`.
fn add_sat16(a: i32, b: i32) -> i32 {
    (a + b).clamp(i16::MIN as i32, i16::MAX as i32)
}

/// Stabilized normalized LSF coefficients — the §4.2.7.5.4 output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NlsfStabilized {
    len: u8,
    /// Stabilized `NLSF_Q15[k]`. Each consecutive pair now satisfies the
    /// Table 25 minimum-spacing constraint, and every entry is in
    /// `[0, 32767]`.
    nlsf_q15: [i16; D_LPC_MAX],
}

impl NlsfStabilized {
    /// Stabilize the reconstructed `NLSF_Q15[]` from §4.2.7.5.3.
    ///
    /// `bandwidth` selects the Table 25 minimum-spacing column and must
    /// match the bandwidth the `recon` was reconstructed for (NB / MB /
    /// WB). Returns `Error::MalformedPacket` if the bandwidth is SWB / FB
    /// or if the reconstructed length does not match the bandwidth's
    /// `d_LPC`.
    pub fn from_reconstructed(
        bandwidth: Bandwidth,
        recon: &NlsfReconstructed,
    ) -> Result<Self, Error> {
        let ndelta = ndelta_min_q15(bandwidth)?;
        let d_lpc = recon.len();
        // `ndelta` has `d_LPC + 1` entries; the reconstructed length must
        // line up with the same `d_LPC`.
        if d_lpc + 1 != ndelta.len() {
            return Err(Error::MalformedPacket);
        }

        let mut nlsf = [0i32; D_LPC_MAX];
        for (dst, &src) in nlsf.iter_mut().zip(recon.nlsf_q15()) {
            *dst = src as i32;
        }

        stabilize_in_place(&mut nlsf[..d_lpc], ndelta);

        let mut nlsf_q15 = [0i16; D_LPC_MAX];
        for (dst, &src) in nlsf_q15.iter_mut().zip(nlsf.iter()) {
            *dst = src as i16;
        }

        Ok(Self {
            len: d_lpc as u8,
            nlsf_q15,
        })
    }

    /// Number of populated entries (10 for NB / MB, 16 for WB).
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` if no entries were stabilized (impossible after a
    /// successful `from_reconstructed`).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Stabilized `NLSF_Q15[k]` in `[0, 32767]`, each consecutive pair
    /// at least Table 25's minimum spacing apart.
    pub fn nlsf_q15(&self) -> &[i16] {
        &self.nlsf_q15[..self.len()]
    }
}

/// In-place §4.2.7.5.4 stabilization on a `d_LPC`-length `NLSF_Q15[]`
/// slice. `ndelta` is the Table 25 column for the bandwidth and has
/// `d_LPC + 1` entries.
///
/// Split out from [`NlsfStabilized::from_reconstructed`] so it can be
/// exercised directly with hand-built inputs in the unit tests.
fn stabilize_in_place(nlsf: &mut [i32], ndelta: &[i32]) {
    let d_lpc = nlsf.len();
    debug_assert_eq!(ndelta.len(), d_lpc + 1);

    // --- Up to 20 distortion-minimizing passes (§4.2.7.5.4) ----------
    for _ in 0..20 {
        // Find the index `i` in `0..=d_LPC` minimizing the spacing
        // `NLSF[i] - NLSF[i-1] - NDeltaMin[i]`, with the boundary
        // conventions `NLSF[-1] = 0` and `NLSF[d_LPC] = 32768`. Ties
        // break to the lower `i`.
        let mut min_i = 0usize;
        let mut min_spacing = i32::MAX;
        for i in 0..=d_lpc {
            let upper = if i == d_lpc { NLSF_UPPER_Q15 } else { nlsf[i] };
            let lower = if i == 0 { 0 } else { nlsf[i - 1] };
            let spacing = upper - lower - ndelta[i];
            if spacing < min_spacing {
                min_spacing = spacing;
                min_i = i;
            }
        }

        // All constraints satisfied — stop.
        if min_spacing >= 0 {
            return;
        }

        let i = min_i;
        if i == 0 {
            // First coefficient too close to the implicit 0 lower edge.
            nlsf[0] = ndelta[0];
        } else if i == d_lpc {
            // Last coefficient too close to the implicit 32768 upper edge.
            nlsf[d_lpc - 1] = NLSF_UPPER_Q15 - ndelta[d_lpc];
        } else {
            // Re-centre the offending pair (NLSF[i-1], NLSF[i]).
            let half = ndelta[i] >> 1;

            // min_center = (NDeltaMin[i] >> 1) + sum(NDeltaMin[0..i])
            let mut min_center = half;
            for &nd in ndelta.iter().take(i) {
                min_center += nd;
            }

            // max_center = 32768 - (NDeltaMin[i] >> 1)
            //                    - sum(NDeltaMin[i+1..=d_LPC])
            let mut max_center = NLSF_UPPER_Q15 - half;
            for &nd in ndelta.iter().take(d_lpc + 1).skip(i + 1) {
                max_center -= nd;
            }

            // center_freq = clamp(min_center,
            //                     (NLSF[i-1] + NLSF[i] + 1) >> 1,
            //                     max_center)
            // with clamp(lo, x, hi) = max(lo, min(x, hi)) per §1.1.3.
            let midpoint = (nlsf[i - 1] + nlsf[i] + 1) >> 1;
            let center_freq = min_center.max(midpoint.min(max_center));

            nlsf[i - 1] = center_freq - half;
            nlsf[i] = nlsf[i - 1] + ndelta[i];
        }
    }

    // --- Fallback (runs once after the 20th pass) --------------------
    // Sort ascending.
    nlsf.sort_unstable();

    // Forward sweep: enforce the lower spacing bound. The k == 0 case
    // uses NLSF[-1] = 0. RFC 8251 §7 requires the addition to saturate
    // at i16 to avoid a wrap-around.
    for k in 0..d_lpc {
        let prev = if k == 0 { 0 } else { nlsf[k - 1] };
        nlsf[k] = nlsf[k].max(add_sat16(prev, ndelta[k]));
    }

    // Backward sweep: enforce the upper spacing bound. The k == d_LPC-1
    // case uses NLSF[d_LPC] = 32768.
    for k in (0..d_lpc).rev() {
        let next = if k + 1 == d_lpc {
            NLSF_UPPER_Q15
        } else {
            nlsf[k + 1]
        };
        nlsf[k] = nlsf[k].min(next - ndelta[k + 1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;
    use crate::silk_lsf_recon::NlsfReconstructed;
    use crate::silk_lsf_stage2::LsfStage2;

    // --- Table 25 well-formedness ------------------------------------

    #[test]
    fn table25_lengths() {
        assert_eq!(NDELTA_MIN_Q15_NB_MB.len(), D_LPC_NB_MB + 1);
        assert_eq!(NDELTA_MIN_Q15_WB.len(), D_LPC_WB + 1);
    }

    /// Table 25 spot-checks transcribed straight from the RFC text:
    /// NB/MB column index 0 = 250, index 10 = 461; WB column index 0 =
    /// 100, index 2 = 40, index 16 = 347.
    #[test]
    fn table25_spot_checks() {
        assert_eq!(NDELTA_MIN_Q15_NB_MB[0], 250);
        assert_eq!(NDELTA_MIN_Q15_NB_MB[10], 461);
        assert_eq!(NDELTA_MIN_Q15_WB[0], 100);
        assert_eq!(NDELTA_MIN_Q15_WB[2], 40);
        assert_eq!(NDELTA_MIN_Q15_WB[6], 5);
        assert_eq!(NDELTA_MIN_Q15_WB[16], 347);
    }

    #[test]
    fn ndelta_lookup_rejects_swb_fb() {
        assert!(ndelta_min_q15(Bandwidth::Swb).is_err());
        assert!(ndelta_min_q15(Bandwidth::Fb).is_err());
        assert!(ndelta_min_q15(Bandwidth::Nb).is_ok());
        assert!(ndelta_min_q15(Bandwidth::Mb).is_ok());
        assert!(ndelta_min_q15(Bandwidth::Wb).is_ok());
    }

    // --- add_sat16 ----------------------------------------------------

    #[test]
    fn add_sat16_saturates() {
        assert_eq!(add_sat16(100, 200), 300);
        assert_eq!(add_sat16(32000, 1000), 32767);
        assert_eq!(add_sat16(-32000, -2000), -32768);
        assert_eq!(add_sat16(32767, 0), 32767);
    }

    // --- Invariant checker -------------------------------------------

    /// Assert the §4.2.7.5.4 post-condition: every consecutive pair is at
    /// least `NDeltaMin` apart, including the implicit 0 / 32768 edges.
    fn assert_spacing_ok(nlsf: &[i32], ndelta: &[i32]) {
        let d_lpc = nlsf.len();
        for i in 0..=d_lpc {
            let upper = if i == d_lpc { NLSF_UPPER_Q15 } else { nlsf[i] };
            let lower = if i == 0 { 0 } else { nlsf[i - 1] };
            assert!(
                upper - lower >= ndelta[i],
                "spacing constraint violated at i={i}: upper={upper} lower={lower} \
                 needs >= {} got {}",
                ndelta[i],
                upper - lower
            );
        }
    }

    // --- Already-stable inputs are untouched -------------------------

    #[test]
    fn already_stable_is_unchanged_nb() {
        // A comfortably-spaced monotone vector well inside all bounds.
        let mut nlsf = [
            2000, 5000, 8000, 11000, 14000, 17000, 20000, 23000, 26000, 29000,
        ];
        let original = nlsf;
        stabilize_in_place(&mut nlsf, &NDELTA_MIN_Q15_NB_MB);
        assert_eq!(nlsf, original, "stable vector must be left untouched");
        assert_spacing_ok(&nlsf, &NDELTA_MIN_Q15_NB_MB);
    }

    #[test]
    fn already_stable_is_unchanged_wb() {
        let mut nlsf = [
            1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 21000, 23000, 25000,
            27000, 29000, 31000,
        ];
        let original = nlsf;
        stabilize_in_place(&mut nlsf, &NDELTA_MIN_Q15_WB);
        assert_eq!(nlsf, original);
        assert_spacing_ok(&nlsf, &NDELTA_MIN_Q15_WB);
    }

    // --- Boundary cases (i == 0 and i == d_LPC) ----------------------

    #[test]
    fn first_coeff_too_low_is_pushed_up() {
        // NLSF[0] = 10 is far below NDeltaMin[0] = 250, so the i == 0
        // branch sets NLSF[0] = 250.
        let mut nlsf = [
            10, 5000, 8000, 11000, 14000, 17000, 20000, 23000, 26000, 29000,
        ];
        stabilize_in_place(&mut nlsf, &NDELTA_MIN_Q15_NB_MB);
        assert!(nlsf[0] >= 250);
        assert_spacing_ok(&nlsf, &NDELTA_MIN_Q15_NB_MB);
    }

    #[test]
    fn last_coeff_too_high_is_pulled_down() {
        // NLSF[9] = 32760 leaves only 8 below the 32768 edge, far less
        // than NDeltaMin[d_LPC] = 461, so the i == d_LPC branch sets
        // NLSF[9] = 32768 - 461 = 32307.
        let mut nlsf = [
            2000, 5000, 8000, 11000, 14000, 17000, 20000, 23000, 26000, 32760,
        ];
        stabilize_in_place(&mut nlsf, &NDELTA_MIN_Q15_NB_MB);
        assert!(nlsf[9] <= NLSF_UPPER_Q15 - 461);
        assert_spacing_ok(&nlsf, &NDELTA_MIN_Q15_NB_MB);
    }

    // --- Interior re-centring -----------------------------------------

    #[test]
    fn interior_pair_recentred() {
        // NLSF[3] and NLSF[4] are equal (spacing 0 < NDeltaMin[4] = 3),
        // forcing the interior re-centring branch. The midpoint of the
        // pair is (11000 + 11000 + 1) >> 1 = 11000.
        let mut nlsf = [
            2000, 5000, 8000, 11000, 11000, 17000, 20000, 23000, 26000, 29000,
        ];
        stabilize_in_place(&mut nlsf, &NDELTA_MIN_Q15_NB_MB);
        // Pair must now be at least NDeltaMin[4] = 3 apart.
        assert!(nlsf[4] - nlsf[3] >= 3);
        assert_spacing_ok(&nlsf, &NDELTA_MIN_Q15_NB_MB);
    }

    /// Hand-computed interior re-centring: an isolated tight pair where
    /// the midpoint is far from both center bounds, so center_freq equals
    /// the midpoint and the pair straddles it symmetrically.
    #[test]
    fn interior_recentre_exact_values() {
        // Single offending pair at indices 4/5 with spacing 1 (< 3).
        // Everything else is comfortably spaced and inside the
        // min_center / max_center band, so the very first pass picks
        // i = 5, re-centres on the midpoint, and then all constraints
        // are met.
        let mut nlsf = [
            2000, 5000, 8000, 11000, 15000, 15001, 20000, 23000, 26000, 29000,
        ];
        stabilize_in_place(&mut nlsf, &NDELTA_MIN_Q15_NB_MB);
        // midpoint = (15000 + 15001 + 1) >> 1 = 15001; half = 3>>1 = 1.
        // NLSF[4] = 15001 - 1 = 15000; NLSF[5] = 15000 + 3 = 15003.
        assert_eq!(nlsf[4], 15000);
        assert_eq!(nlsf[5], 15003);
        assert_spacing_ok(&nlsf, &NDELTA_MIN_Q15_NB_MB);
    }

    // --- Fallback path (sort + sweeps) -------------------------------

    /// A descending (i.e. badly out-of-order) input cannot be fixed by
    /// the 20 distortion-minimizing passes alone for some shapes; the
    /// fallback's sort + forward/backward sweeps guarantee a valid result
    /// regardless. Verify the post-condition holds on a fully reversed
    /// vector.
    #[test]
    fn fallback_fixes_reversed_input() {
        let mut nlsf = [
            29000, 26000, 23000, 20000, 17000, 14000, 11000, 8000, 5000, 2000,
        ];
        stabilize_in_place(&mut nlsf, &NDELTA_MIN_Q15_NB_MB);
        // Output must be sorted ascending and satisfy spacing.
        for w in nlsf.windows(2) {
            assert!(w[0] <= w[1], "fallback output not ascending: {nlsf:?}");
        }
        assert_spacing_ok(&nlsf, &NDELTA_MIN_Q15_NB_MB);
    }

    /// All-zero input: every coefficient collapsed to 0. The procedure
    /// must spread them out to satisfy spacing without exceeding 32768.
    #[test]
    fn all_zero_input_is_spread() {
        let mut nlsf = [0i32; D_LPC_NB_MB];
        stabilize_in_place(&mut nlsf, &NDELTA_MIN_Q15_NB_MB);
        assert_spacing_ok(&nlsf, &NDELTA_MIN_Q15_NB_MB);
        // Sum of NDeltaMin (incl. the trailing edge term) for NB/MB is
        // 250+3*7+6+4+461 = 742; well under 32768, so no clamping issues.
        for &v in nlsf.iter() {
            assert!((0..=NLSF_UPPER_Q15).contains(&v));
        }
    }

    /// All-32767 input (every coefficient pinned at the top): the
    /// procedure must pull them down below the 32768 edge with valid
    /// spacing.
    #[test]
    fn all_max_input_is_spread() {
        let mut nlsf = [32767i32; D_LPC_WB];
        stabilize_in_place(&mut nlsf, &NDELTA_MIN_Q15_WB);
        assert_spacing_ok(&nlsf, &NDELTA_MIN_Q15_WB);
        for w in nlsf.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    /// RFC 8251 §7 saturation: an adversarial input near i16::MAX must
    /// not wrap around in the fallback forward sweep.
    #[test]
    fn fallback_does_not_wrap_near_i16_max() {
        // Force the fallback (descending so the 20 passes can't settle in
        // a way that skips it) with values pinned at the i16 ceiling.
        let mut nlsf = [
            32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
        ];
        stabilize_in_place(&mut nlsf, &NDELTA_MIN_Q15_NB_MB);
        assert_spacing_ok(&nlsf, &NDELTA_MIN_Q15_NB_MB);
        // No entry exceeds i16::MAX (the wrap-around the erratum guards
        // against would have produced a negative value here).
        for &v in nlsf.iter() {
            assert!((0..=i16::MAX as i32).contains(&v), "wrapped: {nlsf:?}");
        }
    }

    // --- End-to-end against §4.2.7.5.3 reconstruction ----------------

    /// Build a reconstructed NLSF vector for the given bandwidth and
    /// stage-1 index off a synthetic range-decoder buffer, then stabilize
    /// and assert the post-condition holds.
    fn recon_then_stabilize(bandwidth: Bandwidth, i1: u8, buf: &[u8]) -> NlsfStabilized {
        let mut rd = RangeDecoder::new(buf);
        let stage2 = LsfStage2::decode(&mut rd, bandwidth, i1).expect("stage-2 decode");
        let recon =
            NlsfReconstructed::from_stage1_and_stage2(bandwidth, i1, &stage2).expect("recon");
        NlsfStabilized::from_reconstructed(bandwidth, &recon).expect("stabilize")
    }

    #[test]
    fn end_to_end_sweep_all_i1() {
        let buf = [0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6];
        for bandwidth in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            let ndelta = ndelta_min_q15(bandwidth).unwrap();
            for i1 in 0u8..32 {
                let stab = recon_then_stabilize(bandwidth, i1, &buf);
                let nlsf: Vec<i32> = stab.nlsf_q15().iter().map(|&v| v as i32).collect();
                assert_spacing_ok(&nlsf, ndelta);
                // Result stays inside [0, 32767].
                for &v in &nlsf {
                    assert!((0..=i16::MAX as i32).contains(&v));
                }
                // Result is monotone non-decreasing (a corollary of the
                // spacing constraint, since every NDeltaMin >= 3 > 0).
                for w in nlsf.windows(2) {
                    assert!(w[0] < w[1], "not strictly increasing: {nlsf:?}");
                }
            }
        }
    }

    #[test]
    fn end_to_end_length_matches_bandwidth() {
        let buf = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA];
        let nb = recon_then_stabilize(Bandwidth::Nb, 0, &buf);
        assert_eq!(nb.len(), D_LPC_NB_MB);
        let wb = recon_then_stabilize(Bandwidth::Wb, 0, &buf);
        assert_eq!(wb.len(), D_LPC_WB);
    }

    #[test]
    fn from_reconstructed_rejects_swb_fb() {
        let buf = [0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6];
        let mut rd = RangeDecoder::new(&buf);
        let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Nb, 0).unwrap();
        let recon = NlsfReconstructed::from_stage1_and_stage2(Bandwidth::Nb, 0, &stage2).unwrap();
        // The recon is NB (d_LPC = 10) but we ask to stabilize as SWB/FB.
        assert!(NlsfStabilized::from_reconstructed(Bandwidth::Swb, &recon).is_err());
        assert!(NlsfStabilized::from_reconstructed(Bandwidth::Fb, &recon).is_err());
    }

    #[test]
    fn from_reconstructed_rejects_length_mismatch() {
        // Reconstruct as WB (d_LPC = 16) then try to stabilize with the
        // NB/MB Table 25 column (which expects d_LPC = 10).
        let buf = [
            0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6, 0x4C, 0x8E,
        ];
        let mut rd = RangeDecoder::new(&buf);
        let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Wb, 0).unwrap();
        let recon = NlsfReconstructed::from_stage1_and_stage2(Bandwidth::Wb, 0, &stage2).unwrap();
        assert!(NlsfStabilized::from_reconstructed(Bandwidth::Nb, &recon).is_err());
    }

    /// A reconstruction is normally already well-spaced, so stabilization
    /// should leave it untouched. Verify the identity on a real NB recon.
    #[test]
    fn typical_recon_unchanged_by_stabilization() {
        let buf = [0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6];
        let mut rd = RangeDecoder::new(&buf);
        let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Nb, 0).unwrap();
        let recon = NlsfReconstructed::from_stage1_and_stage2(Bandwidth::Nb, 0, &stage2).unwrap();
        let stab = NlsfStabilized::from_reconstructed(Bandwidth::Nb, &recon).unwrap();
        // For this well-conditioned buffer the recon already satisfies
        // Table 25, so the stabilized output equals the input.
        assert_eq!(stab.nlsf_q15(), recon.nlsf_q15());
    }
}
