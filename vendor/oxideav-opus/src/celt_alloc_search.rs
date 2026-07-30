//! CELT §4.3.3 1/64-step interpolated static-allocation search
//! (RFC 6716 §4.3.3, p. 111).
//!
//! After the round-39 [`crate::celt_static_alloc`] module pinned the
//! 21-band × 11-quality-column Q5 table `alloc[band][q]`, the §4.3.3
//! procedure searches that table for the highest quality column whose
//! summed per-band allocation fits the working budget. The §4.3.3 RFC
//! text (RFC 6716 §4.3.3, p. 111, lines 6223–6230) is explicit:
//!
//! > The "static" bit allocation (in 1/8 bits) for a quality q,
//! > excluding the minimums, maximums, tilt and boosts, is equal to
//! > `channels*N*alloc[band][q]<<LM>>2`, where `alloc[][]` is given in
//! > Table 57 and `LM=log2(frame_size/120)`. The allocation is
//! > obtained by linearly interpolating between two values of `q` (in
//! > steps of 1/64) to find the highest allocation that does not
//! > exceed the number of bits remaining.
//!
//! This module owns the *interpolation + search* half. The orchestrated
//! §4.3.3 allocator that consumes the search output — clamping per-band
//! results between the round-35 minimum threshold and the round-31
//! per-band cap, folding in the round-33 boosts and the round-36 trim
//! offsets, running the §4.3.3 skip / dual-stereo / intensity-stereo
//! reservation reads, and finally splitting the budget into shape /
//! fine-energy — runs at the consumer site once every piece of the
//! §4.3.3 parameter surface is wired together.
//!
//! ## §4.3.3 1/64-step interpolation
//!
//! The §4.3.3 procedure interpolates between two adjacent integer
//! quality columns `q_lo` and `q_lo + 1` in steps of 1/64. Write
//! `q'_fp = q_lo * 64 + frac` (a "fixed-point quality index") with
//! `q_lo ∈ 0..=9` and `frac ∈ 0..=63`, plus the endpoint
//! `q'_fp = 10 * 64 = 640` (the saturation column). Then for each
//! coded band `b` with `N = bins_per_channel(b, frame_size)`:
//!
//! ```text
//! cell_q11(b, q_lo, frac) = alloc[b][q_lo]   * (64 - frac)
//!                         + alloc[b][q_lo+1] *  frac
//! per_band_q3(b)          = (channels * N * cell_q11) << LM >> (2 + 6)
//! ```
//!
//! The Q11 intermediate (= Q5 cell × Q6 step weight) is summed across
//! coded bands and the §4.3.3 `<< LM >> 2` unit fold is composed with
//! the 1/64 step's `>> 6` to land in Q3 (1/8 bits). The `+ 6` shift
//! consumes the Q6 step granularity; the `>> 2` shift folds Q5 to Q3
//! per round-39's [`STATIC_ALLOC_RIGHT_SHIFT`].
//!
//! All arithmetic is performed in `u64`. The largest intermediate
//! product is bounded by `channels (=2) * N_max (=176) * cell_max
//! (=200) * 64 (Q6 step) << 3 (LM=3)` per band, summed across 21
//! coded bands: `2 * 176 * 200 * 64 * 8 * 21 = 757_678_080`, well
//! inside `u64` headroom (and inside `u32` headroom, but we stay in
//! `u64` for the rolling sum so the consumer doesn't have to
//! reason about per-band vs. summed widths).
//!
//! At `q'_fp = 640` the search degenerates to a pure column-10 lookup:
//! `q_lo = 9, frac = 64` re-expresses `cell_q11 = alloc[b][9] * 0 +
//! alloc[b][10] * 64 = alloc[b][10] * 64`, which the `>> 6` step
//! collapses back to `alloc[b][10]`. The module's
//! [`q_fp_to_components`] helper bakes this in.
//!
//! ## What this module does not own
//!
//! * **The orchestrated §4.3.3 allocator.** The per-band cap from
//!   [`crate::celt_cache_caps50::cap_for_band_bits`] (round 31),
//!   the per-band minimum from
//!   [`crate::celt_band_thresh::band_min_thresh`] (round 35),
//!   the per-band trim offsets from
//!   [`crate::celt_trim_offsets::band_trim_offset`] (round 36), and
//!   the per-band boosts from
//!   [`crate::celt_band_boost::decode_band_boosts`] (round 33) all
//!   modify the per-band allocation *after* the search picks a
//!   fractional `q'`. Those modifications happen at the consumer
//!   site, not here.
//! * **The §4.3.3 skip / dual-stereo / intensity-stereo flag reads.**
//!   Those are bitstream-driven and run at the consumer site after
//!   the search converges.
//! * **The §4.3.4 shape allocation / fine-energy split.** Downstream
//!   of the §4.3.3 search.
//! * **Any bitstream read.** This module is a pure function of
//!   `(channels, frame_size, is_hybrid)` plus the working budget; no
//!   range-coder symbol is consumed.
//!
//! ## Units
//!
//! All inputs and outputs are in 1/8 bits ("8th bits" / "Q3" in the
//! §4.3.3 narrative). The fractional quality index `q'_fp` is a
//! pure dimensionless integer in `0..=640`.
//!
//! ## Provenance
//!
//! The §4.3.3 narrative is transcribed from RFC 6716,
//! `docs/audio/opus/rfc6716-opus.txt`, p. 111 (the interpolation
//! description) and p. 112 (Table 57, the source of the cell
//! values reproduced in [`crate::celt_static_alloc::STATIC_ALLOC`]).

use crate::celt_band_layout::{
    celt_band_bins_per_channel, celt_end_coded_band, celt_first_coded_band, CeltFrameSize,
    CELT_NUM_BANDS,
};
use crate::celt_static_alloc::{
    STATIC_ALLOC, STATIC_ALLOC_INTERP_STEPS, STATIC_ALLOC_Q_MAX, STATIC_ALLOC_RIGHT_SHIFT,
};

/// Maximum value of the §4.3.3 fixed-point quality index `q'_fp`.
///
/// `q'_fp` packs `(q_lo, frac)` into a single non-negative integer:
/// `q'_fp = q_lo * STATIC_ALLOC_INTERP_STEPS + frac`. The upper bound
/// is `STATIC_ALLOC_Q_MAX * STATIC_ALLOC_INTERP_STEPS = 10 * 64 = 640`,
/// which represents the §4.3.3 saturation column (`q' = 10.0`). The
/// search returns one value in `0..=640`.
pub const Q_FP_MAX: u32 = STATIC_ALLOC_Q_MAX * STATIC_ALLOC_INTERP_STEPS;

/// Combined right shift the §4.3.3 conversion applies to the Q11
/// per-band cell × step product to land in Q3.
///
/// The 1/64-step interpolation multiplies a Q5 cell by a Q6 step
/// weight (`0..=64`), producing a Q11 per-band product. The §4.3.3
/// `>> 2` shift (see [`STATIC_ALLOC_RIGHT_SHIFT`]) folds Q5 to Q3;
/// composed with the Q11 → Q5 step weight reduction `>> 6` it gives
/// the total right shift `>> 8` consumed by
/// [`per_band_eighth_bits_at_q_fp`].
pub const STATIC_ALLOC_INTERP_RIGHT_SHIFT: u32 = STATIC_ALLOC_RIGHT_SHIFT + 6;

const _: () = {
    // The shift composition is `>> 2` (Q5 → Q3) plus `>> 6` (Q6 step
    // weight reduction). The Q6 piece equals `log2(STATIC_ALLOC_INTERP_STEPS)`;
    // pin the relationship so a future bump to a non-power-of-two step
    // count trips the build instead of silently breaking the arithmetic.
    assert!(STATIC_ALLOC_INTERP_STEPS == 64);
    assert!(STATIC_ALLOC_INTERP_RIGHT_SHIFT == 8);
};

/// Errors returned by the §4.3.3 1/64-step search when caller-side
/// bookkeeping is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocSearchError {
    /// `channels` is outside `1..=2`. RFC 6716 §4.3.3 (p. 111) defines
    /// the unit conversion for `channels ∈ {1, 2}` only.
    ChannelsOutOfRange { channels: u32 },
    /// `q_fp` is outside `0..=Q_FP_MAX`. The fixed-point quality index
    /// must reference an interpolation cell that the §4.3.3 Table 57
    /// grid covers.
    QFpOutOfRange { q_fp: u32 },
    /// `band` is outside `0..21`. The §4.3 Table 55 band count caps
    /// the band index.
    BandOutOfRange { band: u32 },
}

/// Decomposed `(q_lo, frac)` form of a §4.3.3 fixed-point quality
/// index `q'_fp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QFpComponents {
    /// Integer quality column `q_lo ∈ 0..=10`. Indexes [`STATIC_ALLOC`]
    /// rows directly.
    pub q_lo: u32,
    /// Interpolation step `frac ∈ 0..=64`. The `frac = 64` endpoint
    /// is reached only at `q'_fp = 640` (with `q_lo = 9`); otherwise
    /// `frac ∈ 0..=63`.
    pub frac: u32,
}

/// Decompose a fixed-point quality index `q'_fp` into the
/// `(q_lo, frac)` pair the per-band interpolation needs.
///
/// `q'_fp ∈ 0..=Q_FP_MAX = 640`. The decomposition is:
///
/// ```text
/// q_lo = q_fp / 64
/// frac = q_fp % 64
/// ```
///
/// with the corner `q_fp = 640` re-expressed as `(q_lo = 9, frac = 64)`
/// so the interpolation formula `alloc[b][q_lo] * (64 - frac) +
/// alloc[b][q_lo + 1] * frac` lands on `alloc[b][10] * 64` (the pure
/// saturation column at `q' = 10.0`).
///
/// Returns [`AllocSearchError::QFpOutOfRange`] for `q_fp > 640`.
pub const fn q_fp_to_components(q_fp: u32) -> Result<QFpComponents, AllocSearchError> {
    if q_fp > Q_FP_MAX {
        return Err(AllocSearchError::QFpOutOfRange { q_fp });
    }
    if q_fp == Q_FP_MAX {
        return Ok(QFpComponents {
            q_lo: STATIC_ALLOC_Q_MAX - 1,
            frac: STATIC_ALLOC_INTERP_STEPS,
        });
    }
    Ok(QFpComponents {
        q_lo: q_fp / STATIC_ALLOC_INTERP_STEPS,
        frac: q_fp % STATIC_ALLOC_INTERP_STEPS,
    })
}

/// Compose a `(q_lo, frac)` pair back into a fixed-point quality
/// index `q'_fp`.
///
/// Accepts both the natural `frac ∈ 0..=63` range *and* the saturation
/// representation `frac = 64` (when `q_lo = STATIC_ALLOC_Q_MAX - 1 = 9`),
/// since [`q_fp_to_components`] emits the latter at `q_fp = 640`.
///
/// Returns [`AllocSearchError::QFpOutOfRange`] when the inputs sit
/// outside the §4.3.3 grid.
pub const fn q_fp_from_components(q_lo: u32, frac: u32) -> Result<u32, AllocSearchError> {
    if q_lo > STATIC_ALLOC_Q_MAX {
        return Err(AllocSearchError::QFpOutOfRange {
            q_fp: q_lo * STATIC_ALLOC_INTERP_STEPS + frac,
        });
    }
    if frac > STATIC_ALLOC_INTERP_STEPS {
        return Err(AllocSearchError::QFpOutOfRange {
            q_fp: q_lo * STATIC_ALLOC_INTERP_STEPS + frac,
        });
    }
    if frac == STATIC_ALLOC_INTERP_STEPS && q_lo != STATIC_ALLOC_Q_MAX - 1 {
        // The `frac = 64` saturation form is only valid as the canonical
        // representation of `q_fp = Q_FP_MAX` with `q_lo = 9`.
        return Err(AllocSearchError::QFpOutOfRange {
            q_fp: q_lo * STATIC_ALLOC_INTERP_STEPS + frac,
        });
    }
    if q_lo == STATIC_ALLOC_Q_MAX && frac != 0 {
        return Err(AllocSearchError::QFpOutOfRange {
            q_fp: q_lo * STATIC_ALLOC_INTERP_STEPS + frac,
        });
    }
    Ok(q_lo * STATIC_ALLOC_INTERP_STEPS + frac)
}

/// Per-band 1/8-bit allocation under the §4.3.3 1/64-step interpolation.
///
/// Returns the per-band allocation
/// `((channels * N * cell_q11) << LM) >> (2 + 6)` for one band `b` at
/// fractional quality `q'_fp`, where
/// `cell_q11 = alloc[b][q_lo] * (64 - frac) + alloc[b][q_lo + 1] * frac`
/// (in Q11 units).
///
/// Inputs:
///
/// * `band` — §4.3 Table 55 band index in `0..21`.
/// * `q_fp` — §4.3.3 fixed-point quality index in `0..=Q_FP_MAX`.
/// * `channels` — frame channel count in `1..=2`.
/// * `n_bins` — `bins_per_channel(band, frame_size)` from round 24.
/// * `lm` — §4.3 frame-size scale in `0..=3` (= `log2(frame_size /
///   120)`).
///
/// Output: per-band allocation in Q3 (1/8 bits).
///
/// The arithmetic is performed in `u64` to keep all intermediate
/// products well-defined; the §4.3.3 per-band peak fits in `u32`, but
/// the rolling sum in [`total_eighth_bits_at_q_fp`] crosses 21 bands
/// and the wider type makes the boundary explicit.
pub fn per_band_eighth_bits_at_q_fp(
    band: u32,
    q_fp: u32,
    channels: u32,
    n_bins: u32,
    lm: u32,
) -> Result<u64, AllocSearchError> {
    if channels == 0 || channels > 2 {
        return Err(AllocSearchError::ChannelsOutOfRange { channels });
    }
    if band >= CELT_NUM_BANDS as u32 {
        return Err(AllocSearchError::BandOutOfRange { band });
    }
    let comps = q_fp_to_components(q_fp)?;
    let q_lo = comps.q_lo as usize;
    let frac = comps.frac as u64;
    let row = &STATIC_ALLOC[band as usize];
    let cell_lo = row[q_lo] as u64;
    // At `q_lo = 10` the only legal `frac` is `0` (canonicalised by
    // `q_fp_to_components` into `q_lo = 9, frac = 64`), so the `q_lo + 1`
    // index is always in-range when we reach this site.
    let cell_hi = row[q_lo + 1] as u64;
    let cell_q11 = cell_lo * (STATIC_ALLOC_INTERP_STEPS as u64 - frac) + cell_hi * frac;
    // (channels * n_bins * cell_q11) << lm >> 8
    let scaled = (channels as u64) * (n_bins as u64) * cell_q11;
    Ok((scaled << lm) >> STATIC_ALLOC_INTERP_RIGHT_SHIFT)
}

/// Total §4.3.3 static allocation across every coded band at a
/// fractional quality `q'_fp`.
///
/// This is the `sum_{b ∈ coded bands} per_band_eighth_bits_at_q_fp(b)`
/// quantity the §4.3.3 search compares against the working budget.
///
/// Inputs:
///
/// * `q_fp` — §4.3.3 fixed-point quality index in `0..=Q_FP_MAX`.
/// * `channels` — frame channel count in `1..=2`.
/// * `frame_size` — §4.3 CELT frame size; selects the per-band MDCT
///   bin count column of Table 55.
/// * `is_hybrid` — `true` ⇒ the §4.3 Hybrid rule "the first 17 bands
///   are not coded" applies; `false` ⇒ the CELT-only first-coded-band
///   is `0`.
///
/// Output: total allocation in Q3 (1/8 bits).
pub fn total_eighth_bits_at_q_fp(
    q_fp: u32,
    channels: u32,
    frame_size: CeltFrameSize,
    is_hybrid: bool,
) -> Result<u64, AllocSearchError> {
    if channels == 0 || channels > 2 {
        return Err(AllocSearchError::ChannelsOutOfRange { channels });
    }
    if q_fp > Q_FP_MAX {
        return Err(AllocSearchError::QFpOutOfRange { q_fp });
    }
    let lm = frame_size.column_index() as u32;
    let first = celt_first_coded_band(is_hybrid);
    let end = celt_end_coded_band();
    let mut sum: u64 = 0;
    for band in first..end {
        let n_bins = celt_band_bins_per_channel(band, frame_size)
            .expect("first..end is in-range for celt_band_bins_per_channel")
            as u32;
        sum += per_band_eighth_bits_at_q_fp(band as u32, q_fp, channels, n_bins, lm)?;
    }
    Ok(sum)
}

/// Result of the §4.3.3 1/64-step interpolated allocation search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocSearchOutcome {
    /// Chosen fixed-point quality index `q'_fp ∈ 0..=Q_FP_MAX`.
    pub q_fp: u32,
    /// Total Q3 allocation summed across coded bands at `q_fp`. Always
    /// `≤ budget_eighth_bits` (the §4.3.3 "does not exceed the number
    /// of bits remaining" invariant).
    pub total_eighth_bits: u64,
}

/// Run the §4.3.3 1/64-step linear-interpolation search for the
/// highest fractional quality `q'_fp ∈ 0..=Q_FP_MAX` whose summed
/// allocation across coded bands does not exceed `budget_eighth_bits`.
///
/// Inputs:
///
/// * `budget_eighth_bits` — the working budget in 1/8 bits. The §4.3.3
///   procedure derives this from
///   [`crate::celt_reservations::ReservationOutcome::total_remaining_eighth_bits`]
///   after the round-34 reservation block deducts the anti-collapse /
///   skip / intensity-stereo / dual-stereo reservations.
/// * `channels` — frame channel count in `1..=2`.
/// * `frame_size` — §4.3 CELT frame size.
/// * `is_hybrid` — selects the §4.3 first-coded-band rule.
///
/// Output: an [`AllocSearchOutcome`] carrying the chosen `q_fp` plus
/// the corresponding total allocation.
///
/// Algorithm: linear scan from the top `q_fp = Q_FP_MAX` downwards,
/// returning the first candidate whose total fits the budget. The
/// search space is bounded — at most `Q_FP_MAX + 1 = 641` candidates
/// over 21 coded bands (less for Hybrid) — and runs in deterministic
/// constant work, no allocation. The §4.3.3 RFC text doesn't pin a
/// specific search direction; "the highest allocation that does not
/// exceed" admits any equivalent monotone-search formulation. Linear
/// from the top keeps the consumer-site reasoning straightforward and
/// the worst case still small.
pub fn search_q_fp(
    budget_eighth_bits: u64,
    channels: u32,
    frame_size: CeltFrameSize,
    is_hybrid: bool,
) -> Result<AllocSearchOutcome, AllocSearchError> {
    if channels == 0 || channels > 2 {
        return Err(AllocSearchError::ChannelsOutOfRange { channels });
    }
    // The §4.3.3 column-0 row gives a total of 0 for every band, so
    // `q_fp = 0` always fits any non-negative budget. The scan therefore
    // always terminates with a defined answer.
    let mut q_fp = Q_FP_MAX;
    loop {
        let total = total_eighth_bits_at_q_fp(q_fp, channels, frame_size, is_hybrid)?;
        if total <= budget_eighth_bits {
            return Ok(AllocSearchOutcome {
                q_fp,
                total_eighth_bits: total,
            });
        }
        if q_fp == 0 {
            // `q_fp = 0` total is 0 — unreachable in practice, but
            // pinned defensively.
            return Ok(AllocSearchOutcome {
                q_fp: 0,
                total_eighth_bits: 0,
            });
        }
        q_fp -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_static_alloc::static_alloc_eighth_bits;

    // ---- Component / inverse helpers ----

    #[test]
    fn q_fp_max_constant() {
        // Q_FP_MAX = 10 * 64 = 640 — the saturation column.
        assert_eq!(Q_FP_MAX, 640);
        assert_eq!(STATIC_ALLOC_INTERP_RIGHT_SHIFT, 8);
    }

    #[test]
    fn q_fp_zero_decomposes_to_q_lo_zero_frac_zero() {
        let c = q_fp_to_components(0).unwrap();
        assert_eq!(c.q_lo, 0);
        assert_eq!(c.frac, 0);
    }

    #[test]
    fn q_fp_at_integer_column_has_zero_frac() {
        for q in 0..=9 {
            let c = q_fp_to_components(q * 64).unwrap();
            assert_eq!(c.q_lo, q);
            assert_eq!(c.frac, 0);
        }
    }

    #[test]
    fn q_fp_saturation_decomposes_to_q_lo_9_frac_64() {
        let c = q_fp_to_components(Q_FP_MAX).unwrap();
        assert_eq!(c.q_lo, 9);
        assert_eq!(c.frac, 64);
    }

    #[test]
    fn q_fp_mid_step_decomposes() {
        // q_fp = 5 * 64 + 32 = 352 ⇒ q_lo = 5, frac = 32.
        let c = q_fp_to_components(352).unwrap();
        assert_eq!(c.q_lo, 5);
        assert_eq!(c.frac, 32);
    }

    #[test]
    fn q_fp_to_components_rejects_out_of_range() {
        assert_eq!(
            q_fp_to_components(Q_FP_MAX + 1).unwrap_err(),
            AllocSearchError::QFpOutOfRange { q_fp: Q_FP_MAX + 1 },
        );
        assert_eq!(
            q_fp_to_components(u32::MAX).unwrap_err(),
            AllocSearchError::QFpOutOfRange { q_fp: u32::MAX },
        );
    }

    #[test]
    fn q_fp_round_trip_through_components() {
        for q_fp in 0..=Q_FP_MAX {
            let c = q_fp_to_components(q_fp).unwrap();
            let back = q_fp_from_components(c.q_lo, c.frac).unwrap();
            assert_eq!(back, q_fp, "round trip failed at q_fp = {}", q_fp);
        }
    }

    #[test]
    fn q_fp_from_components_rejects_invalid_combinations() {
        // q_lo = 10 with non-zero frac is illegal.
        assert!(q_fp_from_components(10, 1).is_err());
        // frac = 64 is only legal at q_lo = 9.
        assert!(q_fp_from_components(0, 64).is_err());
        assert!(q_fp_from_components(8, 64).is_err());
        // frac > 64 is always illegal.
        assert!(q_fp_from_components(5, 65).is_err());
        // q_lo > 10 is always illegal.
        assert!(q_fp_from_components(11, 0).is_err());
    }

    // ---- Per-band parity with the integer-q conversion ----

    #[test]
    fn per_band_at_integer_q_matches_static_alloc_eighth_bits() {
        // The 1/64 interpolation at `frac = 0` should reduce to the
        // pure column lookup the round-39 `static_alloc_eighth_bits`
        // already produces, modulo the rounding direction. At integer
        // q the cell_q11 = alloc[b][q_lo] * 64 + alloc[b][q_lo + 1] *
        // 0 = alloc[b][q_lo] * 64; the >> 8 then becomes >> 2.
        for band in 0..(CELT_NUM_BANDS as u32) {
            for q in 0..=9u32 {
                for &channels in &[1u32, 2] {
                    for &n_bins in &[1u32, 4, 88, 176] {
                        for lm in 0..4u32 {
                            let got =
                                per_band_eighth_bits_at_q_fp(band, q * 64, channels, n_bins, lm)
                                    .unwrap();
                            let want =
                                static_alloc_eighth_bits(band, q, channels, n_bins, lm).unwrap();
                            assert_eq!(
                                got, want as u64,
                                "parity at band {band} q {q} channels {channels} n_bins {n_bins} lm {lm}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn per_band_at_saturation_matches_column_ten() {
        // Q_FP_MAX represents pure column 10. Decomposes as q_lo = 9,
        // frac = 64 ⇒ cell_q11 = alloc[b][9] * 0 + alloc[b][10] * 64 =
        // alloc[b][10] * 64. After >> 8, matches the q = 10 integer
        // path's >> 2 fold.
        for band in 0..(CELT_NUM_BANDS as u32) {
            for &channels in &[1u32, 2] {
                for &n_bins in &[1u32, 88] {
                    for lm in 0..4u32 {
                        let got =
                            per_band_eighth_bits_at_q_fp(band, Q_FP_MAX, channels, n_bins, lm)
                                .unwrap();
                        let want =
                            static_alloc_eighth_bits(band, 10, channels, n_bins, lm).unwrap();
                        assert_eq!(got, want as u64, "saturation parity at band {band}");
                    }
                }
            }
        }
    }

    #[test]
    fn per_band_at_column_zero_is_zero() {
        for band in 0..(CELT_NUM_BANDS as u32) {
            // q_fp = 0 ⇒ q_lo = 0, frac = 0 ⇒ cell_q11 = alloc[b][0] *
            // 64 + alloc[b][1] * 0 = 0 (column 0 is uniformly zero).
            assert_eq!(per_band_eighth_bits_at_q_fp(band, 0, 2, 88, 3).unwrap(), 0);
        }
    }

    #[test]
    fn per_band_monotone_in_q_fp() {
        // The §4.3.3 search hinges on per-band monotonicity in q_fp:
        // a higher q never reduces the allocation. STATIC_ALLOC is
        // monotone non-decreasing in q per the row-monotonicity test
        // in celt_static_alloc; the linear interpolation preserves
        // that, so per_band_eighth_bits_at_q_fp must also be
        // monotone non-decreasing in q_fp.
        for band in 0..(CELT_NUM_BANDS as u32) {
            let mut prev = 0u64;
            for q_fp in 0..=Q_FP_MAX {
                let cur = per_band_eighth_bits_at_q_fp(band, q_fp, 1, 4, 0).unwrap();
                assert!(
                    cur >= prev,
                    "monotonicity failed at band {band} q_fp {q_fp}: {cur} < {prev}",
                );
                prev = cur;
            }
        }
    }

    #[test]
    fn per_band_rejects_invalid_band() {
        assert_eq!(
            per_band_eighth_bits_at_q_fp(CELT_NUM_BANDS as u32, 0, 1, 1, 0).unwrap_err(),
            AllocSearchError::BandOutOfRange {
                band: CELT_NUM_BANDS as u32
            },
        );
    }

    #[test]
    fn per_band_rejects_invalid_channels() {
        assert_eq!(
            per_band_eighth_bits_at_q_fp(0, 0, 0, 1, 0).unwrap_err(),
            AllocSearchError::ChannelsOutOfRange { channels: 0 },
        );
        assert_eq!(
            per_band_eighth_bits_at_q_fp(0, 0, 3, 1, 0).unwrap_err(),
            AllocSearchError::ChannelsOutOfRange { channels: 3 },
        );
    }

    #[test]
    fn per_band_rejects_invalid_q_fp() {
        assert_eq!(
            per_band_eighth_bits_at_q_fp(0, Q_FP_MAX + 1, 1, 1, 0).unwrap_err(),
            AllocSearchError::QFpOutOfRange { q_fp: Q_FP_MAX + 1 },
        );
    }

    // ---- Total across coded bands ----

    #[test]
    fn total_at_q_fp_zero_is_zero() {
        for &is_hybrid in &[false, true] {
            for fs in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                for &channels in &[1u32, 2] {
                    assert_eq!(
                        total_eighth_bits_at_q_fp(0, channels, fs, is_hybrid).unwrap(),
                        0,
                        "fs {:?} channels {} is_hybrid {}",
                        fs,
                        channels,
                        is_hybrid,
                    );
                }
            }
        }
    }

    #[test]
    fn total_monotone_in_q_fp() {
        // Sum of monotone functions is monotone: total must also be
        // monotone non-decreasing in q_fp. Pin against the four CELT
        // frame sizes × {mono, stereo, CELT-only, Hybrid}.
        for &is_hybrid in &[false, true] {
            for fs in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                for &channels in &[1u32, 2] {
                    let mut prev = 0u64;
                    for q_fp in 0..=Q_FP_MAX {
                        let cur = total_eighth_bits_at_q_fp(q_fp, channels, fs, is_hybrid).unwrap();
                        assert!(
                            cur >= prev,
                            "total non-monotone at fs {:?} channels {} is_hybrid {} q_fp {}: {} < {}",
                            fs, channels, is_hybrid, q_fp, cur, prev,
                        );
                        prev = cur;
                    }
                }
            }
        }
    }

    #[test]
    fn total_celt_only_exceeds_hybrid_for_same_q_fp() {
        // Hybrid skips the first 17 bands, so its summed total is
        // strictly smaller than the CELT-only total for any q' where
        // the first 17 bands have non-zero rows (Table 57: rows 0..=14
        // have non-zero entries in some column 1..=10; rows 15..=20 are
        // mostly zero at low q). At q_fp = 640 (column 10) every row
        // contributes a non-zero cell, so the CELT-only sum exceeds
        // the Hybrid sum.
        let celt_only = total_eighth_bits_at_q_fp(Q_FP_MAX, 1, CeltFrameSize::Ms20, false).unwrap();
        let hybrid = total_eighth_bits_at_q_fp(Q_FP_MAX, 1, CeltFrameSize::Ms20, true).unwrap();
        assert!(
            celt_only > hybrid,
            "expected CELT-only > Hybrid at saturation; got celt = {}, hybrid = {}",
            celt_only,
            hybrid,
        );
    }

    #[test]
    fn total_stereo_at_least_mono_at_saturation() {
        // The per-band conversion is linear in `channels`, but the
        // `>> 8` integer fold can introduce up to a one-unit rounding
        // gap per band between `2 * mono(b)` and `stereo(b)`. Over 21
        // coded bands the stereo total still satisfies
        // `mono * 2 <= stereo <= mono * 2 + 21`.
        for fs in [
            CeltFrameSize::Ms2_5,
            CeltFrameSize::Ms5,
            CeltFrameSize::Ms10,
            CeltFrameSize::Ms20,
        ] {
            let mono = total_eighth_bits_at_q_fp(Q_FP_MAX, 1, fs, false).unwrap();
            let stereo = total_eighth_bits_at_q_fp(Q_FP_MAX, 2, fs, false).unwrap();
            assert!(
                stereo >= mono * 2,
                "stereo < 2 * mono at fs {:?}: mono = {} stereo = {}",
                fs,
                mono,
                stereo,
            );
            assert!(
                stereo <= mono * 2 + CELT_NUM_BANDS as u64,
                "stereo - 2 * mono exceeds per-band slack at fs {:?}: mono = {} stereo = {}",
                fs,
                mono,
                stereo,
            );
        }
    }

    #[test]
    fn total_rejects_invalid_channels() {
        assert!(matches!(
            total_eighth_bits_at_q_fp(0, 0, CeltFrameSize::Ms20, false),
            Err(AllocSearchError::ChannelsOutOfRange { channels: 0 }),
        ));
    }

    #[test]
    fn total_rejects_invalid_q_fp() {
        assert!(matches!(
            total_eighth_bits_at_q_fp(Q_FP_MAX + 1, 1, CeltFrameSize::Ms20, false),
            Err(AllocSearchError::QFpOutOfRange { .. }),
        ));
    }

    // ---- Search behaviour ----

    #[test]
    fn search_zero_budget_returns_q_fp_zero() {
        // A zero budget can't fit anything beyond q_fp = 0 (whose
        // total is 0). The search must converge to q_fp = 0.
        let out = search_q_fp(0, 1, CeltFrameSize::Ms20, false).unwrap();
        assert_eq!(out.q_fp, 0);
        assert_eq!(out.total_eighth_bits, 0);
    }

    #[test]
    fn search_saturation_budget_returns_q_fp_max() {
        // A budget that easily fits the saturation total returns
        // q_fp = Q_FP_MAX. Use u64::MAX as the unambiguously huge
        // budget.
        let out = search_q_fp(u64::MAX, 1, CeltFrameSize::Ms20, false).unwrap();
        assert_eq!(out.q_fp, Q_FP_MAX);
        // The reported total must match a fresh evaluation at the
        // chosen q_fp.
        let expect = total_eighth_bits_at_q_fp(Q_FP_MAX, 1, CeltFrameSize::Ms20, false).unwrap();
        assert_eq!(out.total_eighth_bits, expect);
    }

    #[test]
    fn search_picks_exact_budget() {
        // For every q_fp in a small representative set, ask the search
        // for exactly that q_fp's total bits and confirm the search
        // returns that q_fp.
        let fs = CeltFrameSize::Ms20;
        let channels = 1u32;
        for &q_fp in &[0u32, 64, 128, 320, 384, 640] {
            let exact = total_eighth_bits_at_q_fp(q_fp, channels, fs, false).unwrap();
            let out = search_q_fp(exact, channels, fs, false).unwrap();
            // We're guaranteed `out.q_fp >= q_fp` because the search
            // picks the *highest* fit. It might exceed q_fp if a higher
            // q' happens to have the same total (the monotonicity rules
            // out a *smaller* one).
            assert!(
                out.q_fp >= q_fp,
                "search undercut exact target {}: got {}",
                q_fp,
                out.q_fp,
            );
            assert!(
                out.total_eighth_bits <= exact,
                "search violated budget cap at exact {}: total {} > budget",
                exact,
                out.total_eighth_bits,
            );
        }
    }

    #[test]
    fn search_budget_one_less_picks_lower_q_fp() {
        // Pick a non-degenerate q_fp where the total strictly grows
        // when q_fp advances by one step. Then a budget of `total - 1`
        // must force the search down to a strictly lower q_fp.
        let fs = CeltFrameSize::Ms20;
        let channels = 2u32;
        // Find an interior step where total(q_fp) > total(q_fp - 1).
        let q_fp = 320u32; // q' = 5.0
        let here = total_eighth_bits_at_q_fp(q_fp, channels, fs, false).unwrap();
        let down = total_eighth_bits_at_q_fp(q_fp - 1, channels, fs, false).unwrap();
        assert!(
            here > down,
            "test precondition: total at q_fp 320 should exceed q_fp 319",
        );
        let out = search_q_fp(here - 1, channels, fs, false).unwrap();
        assert!(
            out.q_fp < q_fp,
            "search exceeded budget at q_fp 320 - 1: got {} (total {})",
            out.q_fp,
            out.total_eighth_bits,
        );
        assert!(
            out.total_eighth_bits < here,
            "search violated budget: {} >= {}",
            out.total_eighth_bits,
            here,
        );
    }

    #[test]
    fn search_result_is_self_consistent() {
        // For every CELT frame size × {mono, stereo} × {CELT-only,
        // Hybrid} × budget shape, the search output must satisfy:
        //   1. out.total_eighth_bits == total_eighth_bits_at_q_fp(out.q_fp, ...)
        //   2. out.total_eighth_bits <= budget
        //   3. either out.q_fp == Q_FP_MAX, or
        //      total_eighth_bits_at_q_fp(out.q_fp + 1, ...) > budget.
        let budgets = [0u64, 100, 1_000, 10_000, 100_000];
        for &is_hybrid in &[false, true] {
            for fs in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                for &channels in &[1u32, 2] {
                    for &budget in &budgets {
                        let out = search_q_fp(budget, channels, fs, is_hybrid).unwrap();
                        let recomputed =
                            total_eighth_bits_at_q_fp(out.q_fp, channels, fs, is_hybrid).unwrap();
                        assert_eq!(
                            out.total_eighth_bits, recomputed,
                            "total mismatch at fs {:?} channels {} is_hybrid {} budget {}",
                            fs, channels, is_hybrid, budget,
                        );
                        assert!(
                            out.total_eighth_bits <= budget,
                            "search exceeded budget at fs {:?} budget {}: total {}",
                            fs,
                            budget,
                            out.total_eighth_bits,
                        );
                        if out.q_fp < Q_FP_MAX {
                            let next =
                                total_eighth_bits_at_q_fp(out.q_fp + 1, channels, fs, is_hybrid)
                                    .unwrap();
                            assert!(
                                next > budget,
                                "search undercut at fs {:?} budget {}: next q_fp = {} total {} <= budget",
                                fs, budget, out.q_fp + 1, next,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_rejects_invalid_channels() {
        assert!(matches!(
            search_q_fp(1000, 0, CeltFrameSize::Ms20, false),
            Err(AllocSearchError::ChannelsOutOfRange { channels: 0 }),
        ));
        assert!(matches!(
            search_q_fp(1000, 3, CeltFrameSize::Ms20, false),
            Err(AllocSearchError::ChannelsOutOfRange { channels: 3 }),
        ));
    }
}
