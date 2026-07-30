//! SILK gain dequantization — RFC 6716 §4.2.7.4 tail-end conversion.
//!
//! Once the [`SubframeGains`](crate::silk_gains::SubframeGains) decoder
//! has produced the integer `log_gain ∈ 0..=63` for every 5 ms subframe
//! of a SILK frame, the §4.2.7.9 LTP / LPC synthesis filters need the
//! corresponding linear Q16 gain. RFC 6716 §4.2.7.4 specifies that
//! mapping in two steps:
//!
//! 1. Re-scale the integer index to a Q7 logarithm in roughly
//!    `0..=4096`:
//!
//!    ```text
//!    inLog_Q7 = ((0x1D1C71 * log_gain) >> 16) + 2090
//!    ```
//!
//!    `0x1D1C71` = 1 907 825; the constant fixes the per-step scaling
//!    so that the smallest `log_gain` (0) gives `inLog_Q7 = 2090` and
//!    the largest (63) gives `inLog_Q7 = 3923`.
//!
//! 2. Convert through the spec's piecewise-linear approximation of
//!    `2^(inLog_Q7 / 128.0)`. Let `i = inLog_Q7 >> 7` be the integer
//!    part and `f = inLog_Q7 & 127` be the fractional part. Then:
//!
//!    ```text
//!    gain_Q16 = (1 << i) + ((-174*f*(128-f) >> 16) + f) * ((1 << i) >> 7)
//!    ```
//!
//!    All multiplies are 32-bit signed. The intermediate `-174*f*(128-f)`
//!    is bounded by `|-174 * 64 * 64| = 712 704`, comfortably inside
//!    `i32`. The outer term `(1 << i) >> 7` is non-negative because
//!    `i <= 30` for every reachable `inLog_Q7`, so the right shift on
//!    `(1 << i)` is exact.
//!
//! Per the spec the final Q16 gain lies between **81 920** (log_gain 0)
//! and **1 686 110 208** (log_gain 63), representing linear scale
//! factors of 1.25 and ~25 728. Both endpoints are pinned exactly by
//! the unit tests in this module.
//!
//! This module owns only [`silk_log2lin`] and [`silk_gains_dequant`]; it
//! has no range-coder dependency and is a pure function over the
//! integer `log_gain`. The matching higher-level
//! `SubframeGains::dequant_q16` wrapper lives on
//! [`SubframeGains`](crate::silk_gains::SubframeGains) in the
//! `silk_gains` module so callers can dequantise an entire frame's
//! gains in one call.
//!
//! # References
//!
//! * RFC 6716, §4.2.7.4 "Subframe Gains" — the formulas above and the
//!   `81920..=1686110208` output-range guarantee.

/// The multiplier `0x1D1C71 = 1 907 825` used in the §4.2.7.4
/// `inLog_Q7 = ((0x1D1C71 * log_gain) >> 16) + 2090` re-scaling.
pub const SILK_LOG_GAIN_MULTIPLIER: u32 = 0x001D_1C71;

/// The bias `2090` used in the §4.2.7.4 re-scaling.
pub const SILK_LOG_GAIN_BIAS: u32 = 2090;

/// The smallest Q16 gain produced by the §4.2.7.4 mapping (at
/// `log_gain = 0`). Equal to 1.25 in linear scale.
pub const SILK_GAIN_Q16_MIN: u32 = 81_920;

/// The largest Q16 gain produced by the §4.2.7.4 mapping (at
/// `log_gain = 63`). Equal to approximately 25 728 in linear scale.
pub const SILK_GAIN_Q16_MAX: u32 = 1_686_110_208;

/// RFC 6716 §4.2.7.4 `silk_log2lin(inLog_Q7)`. Computes the
/// fixed-point approximation of `2^(inLog_Q7 / 128.0)` defined as:
///
/// ```text
/// i  = inLog_Q7 >> 7
/// f  = inLog_Q7 & 127
/// y  = (1 << i) + ((-174*f*(128-f) >> 16) + f) * ((1 << i) >> 7)
/// ```
///
/// `inLog_Q7` is interpreted as a non-negative Q7 value. The caller is
/// responsible for keeping `i = inLog_Q7 >> 7 <= 30` so the inner
/// `1 << i` does not overflow `i32`. Within the §4.2.7.4
/// `silk_gains_dequant` pipeline the maximum reachable `inLog_Q7` is
/// `3923` ⇒ `i = 30` ⇒ `1 << 30 = 1_073_741_824`, well inside `i32`.
///
/// The result is widened to `u32` because every reachable value is
/// non-negative and the §4.2.7.9 LPC synthesis filter divides by it.
///
/// # Spec gap
///
/// RFC 6716 §4.2.7.4 does not spell out the type of the intermediate
/// multiplies. The straightforward `i32` reading suffices for every
/// reachable `inLog_Q7`: `-174 * 64 * 64 = -712_704` and
/// `((1<<30) >> 7) * 127 < 2^30`, both within `i32`.
pub fn silk_log2lin(in_log_q7: u32) -> u32 {
    let in_q7 = in_log_q7 as i32;
    let i = in_q7 >> 7;
    let f = in_q7 & 127;
    let base = 1_i32 << i;
    // `-174*f*(128-f)` fits in i32 (|.| < 712 705).
    let bowed = (-174_i32 * f * (128 - f)) >> 16;
    // `(bowed + f)` is small (<= 128 in magnitude). `base >> 7` is
    // non-negative for any reachable `i`. The whole second term is
    // therefore bounded by `(1<<23) * 128 = 2^30`, inside `i32`.
    let scaled = (bowed + f) * (base >> 7);
    let result = base + scaled;
    // Every reachable result is positive (the smallest, at i=16 / f=42,
    // is already 81920). Widen for the dequantised-gain consumer.
    result as u32
}

/// RFC 6716 §4.2.7.4 `silk_gains_dequant`. Composes the §4.2.7.4
/// `inLog_Q7` re-scaling and [`silk_log2lin`] into a single
/// `log_gain ∈ 0..=63` → `gain_Q16 ∈ [81920, 1_686_110_208]` mapping:
///
/// ```text
/// in_log_q7 = (0x1D1C71 * log_gain >> 16) + 2090
/// gain_Q16  = silk_log2lin(in_log_q7)
/// ```
///
/// The first multiply is widened to `u32` so the bare `(0x1D1C71 * 63)`
/// (= `120 192 975`) does not overflow.
///
/// # Panics
///
/// Panics in debug builds if `log_gain > 63`. The decode layer
/// (`SubframeGains`) constrains `log_gain` to `0..=63` per §4.2.7.4 so
/// the upstream contract makes this unreachable.
pub fn silk_gains_dequant(log_gain: u8) -> u32 {
    debug_assert!(log_gain <= 63, "log_gain must be in 0..=63 per §4.2.7.4");
    let scaled = (SILK_LOG_GAIN_MULTIPLIER * log_gain as u32) >> 16;
    silk_log2lin(scaled + SILK_LOG_GAIN_BIAS)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- §4.2.7.4 endpoint pinning -------------------------------

    #[test]
    fn log_gain_zero_dequantises_to_min() {
        // RFC 6716 §4.2.7.4 final paragraph: "The final Q16 gain values
        // lies between 81920 and 1686110208, inclusive (representing
        // scale factors of 1.25 to 25728, respectively)."
        assert_eq!(silk_gains_dequant(0), SILK_GAIN_Q16_MIN);
        assert_eq!(SILK_GAIN_Q16_MIN, 81_920);
    }

    #[test]
    fn log_gain_max_dequantises_to_max() {
        assert_eq!(silk_gains_dequant(63), SILK_GAIN_Q16_MAX);
        assert_eq!(SILK_GAIN_Q16_MAX, 1_686_110_208);
    }

    // ----- §4.2.7.4 monotonicity ----------------------------------

    #[test]
    fn dequant_is_strictly_monotone_in_log_gain() {
        // 2^x is strictly increasing; both the inLog_Q7 re-scaling and
        // the silk_log2lin piecewise approximation preserve that across
        // the integer log_gain domain.
        let mut prev = 0_u32;
        for g in 0..=63u8 {
            let q16 = silk_gains_dequant(g);
            assert!(
                q16 > prev,
                "log_gain {g} => gain_Q16 {q16} not greater than prev {prev}"
            );
            prev = q16;
        }
    }

    #[test]
    fn dequant_all_in_documented_range() {
        for g in 0..=63u8 {
            let q16 = silk_gains_dequant(g);
            assert!(
                (SILK_GAIN_Q16_MIN..=SILK_GAIN_Q16_MAX).contains(&q16),
                "log_gain {g} => gain_Q16 {q16} outside RFC-documented \
                 [{SILK_GAIN_Q16_MIN}, {SILK_GAIN_Q16_MAX}] range"
            );
        }
    }

    // ----- silk_log2lin algebraic identities ----------------------

    #[test]
    fn log2lin_integer_exponent_is_pure_power_of_two() {
        // f == 0 zeroes the bowed correction and the linear term, so
        // silk_log2lin(128*i) must collapse to (1 << i).
        for i in 0..=30u32 {
            let in_q7 = 128 * i;
            assert_eq!(silk_log2lin(in_q7), 1u32 << i, "i = {i}");
        }
    }

    #[test]
    fn log2lin_zero_is_one() {
        // The smallest representable Q7 logarithm corresponds to 2^0.
        assert_eq!(silk_log2lin(0), 1);
    }

    #[test]
    fn log2lin_one_q7_is_just_above_unity() {
        // At i=0, f=1: base = 1; bowed = (-174*1*127)>>16 = -1
        // (arithmetic shift toward -inf); (bowed + f) = 0;
        // (1 << 0) >> 7 = 0; second term = 0. Result = 1.
        // This pins the documented small-value behaviour of the
        // approximation; the spec's `(1<<i) >> 7` floors at 0 for i<7.
        assert_eq!(silk_log2lin(1), 1);
    }

    #[test]
    fn log2lin_at_q7_seven_doubles() {
        // i=0, f=7: base=1, bowed=(-174*7*121)>>16=-3,
        // (bowed+f)*((1<<0)>>7) = 4*0 = 0 ⇒ 1. Still 1 (the
        // approximation can't resolve sub-128 Q7 below i=7).
        assert_eq!(silk_log2lin(7), 1);
        // i=7, f=0: pure 2^7 = 128.
        assert_eq!(silk_log2lin(7 << 7), 128);
        // i=7, f=64 (halfway between 2^7 and 2^8): exercise both the
        // bowed correction and the linear term simultaneously.
        // base = 128; bowed = (-174*64*64)>>16 (arithmetic shift toward
        // -inf since -712704/65536 = -10.876…) = -11;
        // (bowed + f) = 53; (1<<7)>>7 = 1; second term = 53.
        // total = 128 + 53 = 181. 2^(7+0.5) = 181.019…, so the §4.2.7.4
        // approximation lands one count below the true value.
        assert_eq!(silk_log2lin((7 << 7) | 64), 181);
    }

    // ----- silk_gains_dequant inLog_Q7 algebra ---------------------

    #[test]
    fn rescaled_log_gain_at_zero() {
        // log_gain = 0: in_log_q7 = 0 + 2090 = 2090.
        // Mirror the rescaling locally and feed it to silk_log2lin to
        // pin both halves of the §4.2.7.4 pipeline independently.
        let log_gain: u32 = 0;
        let in_q7 = ((SILK_LOG_GAIN_MULTIPLIER * log_gain) >> 16) + SILK_LOG_GAIN_BIAS;
        assert_eq!(in_q7, 2090);
        assert_eq!(silk_log2lin(in_q7), 81_920);
    }

    #[test]
    fn rescaled_log_gain_at_max() {
        // log_gain = 63: 0x1D1C71 * 63 = 120_192_975.
        // 0x1D1C71 = 1907825. 1907825 * 63 = 120_192_975.
        // 120_192_975 / 65536 = 1833 (remainder 65487; 1833*65536 =
        // 120_127_488 ≤ 120_192_975 < 1834*65536 = 120_193_024).
        // 1833 + 2090 = 3923 ⇒ i = 30, f = 83.
        let in_q7 = (SILK_LOG_GAIN_MULTIPLIER * 63) >> 16;
        let in_q7 = in_q7 + SILK_LOG_GAIN_BIAS;
        assert_eq!(in_q7, 3923);
        assert_eq!(silk_log2lin(in_q7), 1_686_110_208);
    }

    // ----- silk_log2lin spec formula transcription oracle ---------

    /// Independent oracle of the §4.2.7.4 formula, computed in i64 to
    /// rule out 32-bit-arithmetic discrepancies.
    fn log2lin_oracle(in_log_q7: u32) -> u32 {
        let in_q = in_log_q7 as i64;
        let i = in_q >> 7;
        let f = in_q & 127;
        let base: i64 = 1_i64 << i;
        let bowed = (-174_i64 * f * (128 - f)) >> 16;
        let scaled = (bowed + f) * (base >> 7);
        (base + scaled) as u32
    }

    #[test]
    fn log2lin_matches_oracle_on_silk_dequant_domain() {
        // For every reachable inLog_Q7 across the log_gain ∈ 0..=63
        // sweep, the production i32 implementation matches the i64
        // oracle exactly.
        for log_gain in 0..=63u8 {
            let in_q7 = ((SILK_LOG_GAIN_MULTIPLIER * log_gain as u32) >> 16) + SILK_LOG_GAIN_BIAS;
            assert_eq!(
                silk_log2lin(in_q7),
                log2lin_oracle(in_q7),
                "production / oracle disagree at log_gain = {log_gain}, in_log_q7 = {in_q7}"
            );
        }
    }

    #[test]
    fn log2lin_matches_oracle_on_q7_domain_sweep() {
        // A broader sweep over the Q7 domain the §4.2.7.4 dequant
        // pipeline can reach: i ∈ 0..=30 (so 1<<i fits i32) × every
        // possible f ∈ 0..=127.
        for i in 0..=30u32 {
            for f in 0..=127u32 {
                let in_q7 = (i << 7) | f;
                assert_eq!(
                    silk_log2lin(in_q7),
                    log2lin_oracle(in_q7),
                    "production / oracle disagree at in_log_q7 = {in_q7}"
                );
            }
        }
    }

    // ----- silk_gains_dequant against an independent recomposition --

    fn gains_dequant_oracle(log_gain: u8) -> u32 {
        let scaled = (SILK_LOG_GAIN_MULTIPLIER as u64 * log_gain as u64) >> 16;
        log2lin_oracle(scaled as u32 + SILK_LOG_GAIN_BIAS)
    }

    #[test]
    fn dequant_matches_oracle_for_full_log_gain_domain() {
        for g in 0..=63u8 {
            assert_eq!(
                silk_gains_dequant(g),
                gains_dequant_oracle(g),
                "production / oracle disagree at log_gain = {g}"
            );
        }
    }

    // ----- log_gain → linear scale factor sanity --------------------

    #[test]
    fn min_gain_q16_is_one_point_two_five() {
        // 81920 / 65536 = 1.25 exactly.
        assert_eq!(SILK_GAIN_Q16_MIN as f64 / 65536.0, 1.25);
    }

    #[test]
    fn max_gain_q16_is_approximately_twenty_five_thousand() {
        // 1686110208 / 65536 ≈ 25728.x (RFC text: "≈ 25728").
        let linear = SILK_GAIN_Q16_MAX as f64 / 65536.0;
        assert!(
            (25_727.5..25_728.5).contains(&linear),
            "expected ~25728, got {linear}"
        );
    }

    // ----- §4.2.7.4 contract: smallest step is monotone ------------

    #[test]
    fn dequant_step_is_positive_at_each_boundary() {
        // Pin that the smallest reachable Q7 increment between adjacent
        // log_gain values is non-zero — every log_gain bump moves the
        // gain. Required for the LPC/LTP synth code to treat distinct
        // log_gains as distinct gain_Q16 values.
        for g in 0..=62u8 {
            let here = silk_gains_dequant(g);
            let next = silk_gains_dequant(g + 1);
            assert!(
                next > here,
                "log_gain {g} -> {here}; log_gain {} -> {next} not strictly greater",
                g + 1
            );
        }
    }

    // ----- exhaustive pinning of every reachable log_gain ----------

    /// Reference table of `(log_gain, gain_Q16)` derived from the i64
    /// oracle, kept in sync with the production implementation.
    #[test]
    fn dequant_table_pin() {
        // Anchor every log_gain value to the oracle output so a
        // regression in either the rescaling constants or the
        // silk_log2lin formula trips a single pin.
        let mut table: Vec<(u8, u32)> = (0..=63u8).map(|g| (g, gains_dequant_oracle(g))).collect();
        // First and last are the spec-given endpoints.
        assert_eq!(table.first().unwrap().1, SILK_GAIN_Q16_MIN);
        assert_eq!(table.last().unwrap().1, SILK_GAIN_Q16_MAX);
        // Production agreement across every entry.
        for (g, expected) in table.drain(..) {
            assert_eq!(silk_gains_dequant(g), expected, "log_gain = {g}");
        }
    }
}
