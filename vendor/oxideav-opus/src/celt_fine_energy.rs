//! CELT §4.3.2.2 fine-energy quantization (RFC 6716 §4.3.2.2, p. 109).
//!
//! After the §4.3.2.1 *coarse energy* of each CELT band has been
//! decoded and the 2-D predictor reconstructed the log-energy in 6 dB
//! steps, the §4.3.2.2 *fine energy* layer refines each band's energy
//! with a small uniform correction. RFC 6716 §4.3.2.2 (p. 109) states
//! the mechanism directly:
//!
//! > The number of bits assigned to fine energy quantization in each
//! > band is determined by the bit allocation computation described in
//! > Section 4.3.3. Let `B_i` be the number of fine energy bits for
//! > band `i`; the refinement is an integer `f` in the range
//! > `[0, 2**B_i - 1]`. The mapping between `f` and the correction
//! > applied to the coarse energy is equal to
//! > `(f + 1/2) / 2**B_i - 1/2`.
//!
//! The correction is a *fraction of a 6 dB coarse-energy step*: with
//! `B_i = 0` no bits are read and the correction is the implied `0`
//! (the §4.3.2.2 formula degenerates to "no refinement"); with
//! `B_i = 1` the single bit selects between `-1/4` and `+1/4`; as
//! `B_i` grows the `2**B_i` equally-spaced reconstruction levels tile
//! the open interval `(-1/2, +1/2)` symmetrically about zero (the
//! `+1/2` offset in the numerator centres the levels in their
//! quantization cells, the trailing `-1/2` removes the bias so the
//! correction is zero-mean).
//!
//! ## §4.3.2.2 final fine-energy allocation
//!
//! RFC 6716 §4.3.2.2 (p. 109) continues:
//!
//! > When some bits are left "unused" after all other flags have been
//! > decoded, these bits are assigned to a "final" step of fine
//! > allocation. In effect, these bits are used to add one extra fine
//! > energy bit per band per channel. The allocation process
//! > determines two "priorities" for the final fine bits. Any
//! > remaining bits are first assigned only to bands of priority 0,
//! > starting from band 0 and going up. If all bands of priority 0
//! > have received one bit per channel, then bands of priority 1 are
//! > assigned an extra bit per channel, starting from band 0. If any
//! > bits are left after this, they are left unused.
//!
//! This module owns the *bookkeeping* half of that paragraph: given
//! the per-band priority vector, the channel count, and the number of
//! leftover whole bits, [`plan_final_fine_bits`] decides which bands
//! receive an extra fine-energy bit (in priority-0-then-priority-1,
//! band-0-upward order) and how many bits stay unused. The actual
//! `dec_bits(channels)` reads of those extra bits, and the application
//! of the extra-bit correction to the reconstructed energy, run at the
//! consumer site once the §4.3.3 allocator has produced the per-band
//! `B_i` and priority vectors.
//!
//! ## Fixed-point representation
//!
//! The RFC states the correction as an exact rational
//! `(f + 1/2) / 2**B_i - 1/2`. To stay bit-exact without floating
//! point, this module exposes the correction in two equivalent typed
//! forms:
//!
//! * [`fine_correction_q15`] — the correction scaled by `2**15`,
//!   i.e. `correction * 32768`, as a signed `i32`. Because the
//!   correction lies in the open interval `(-1/2, +1/2)`, the Q15
//!   value lies in `(-16384, +16384)`. The Q15 result is *exact*
//!   whenever `B_i <= 15` (every reachable `B_i` — the §4.3.3
//!   allocator caps fine-energy bits far below 15), because the
//!   denominator `2**(B_i + 1)` then divides `2**16`.
//! * [`fine_correction_ratio`] — the correction as the exact reduced
//!   pair `(numerator, denominator)` with
//!   `numerator = 2*f + 1 - 2**B_i` and `denominator = 2**(B_i + 1)`,
//!   for callers that want to fold the fraction into a different
//!   Q-format without intermediate rounding.
//!
//! The algebra: `(f + 1/2) / 2**B - 1/2 = (2f + 1 - 2**B) / 2**(B+1)`.
//!
//! ## What this module does not own
//!
//! * **The `B_i` allocation itself.** The number of fine-energy bits
//!   per band is the output of the §4.3.3 bit-allocation search and is
//!   supplied to this module as an input.
//! * **The range-decoder reads.** The `dec_bits(B_i)` read that
//!   produces `f`, and the `dec_bits(channels)` reads of the final
//!   extra bits, happen at the consumer site against
//!   [`crate::RangeDecoder`]. This module converts an already-read
//!   `f` into a correction and plans which bands the final bits go to.
//! * **The coarse-energy reconstruction.** The §4.3.2.1 predictor and
//!   the addition of the fine correction onto the reconstructed
//!   log-energy run at the consumer site.
//!
//! ## Provenance
//!
//! Narrative: RFC 6716 §4.3.2.2 (p. 109), reproduced from
//! `docs/audio/opus/rfc6716-opus.txt`; cross-referenced by the
//! clean-room trace `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md`
//! §1 (which notes the fine-energy result feeds back into the
//! §4.3.2.2 quantization and the leftover-bit priorities drive
//! `unquant_energy_finalise()`). No external library source was
//! consulted; the correction formula and the final-allocation
//! priority rule are both stated verbatim in the standards-track text.

/// Maximum number of fine-energy bits per band this module accepts.
///
/// The §4.3.2.2 correction formula is mathematically valid for any
/// `B_i >= 0`, but the §4.3.3 bit-allocation search caps the
/// fine-energy bits per band far below this. The bound is set so the
/// [`fine_correction_q15`] result stays *exact* (the denominator
/// `2**(B_i + 1)` divides `2**16`) and so a `2**B_i` level count fits
/// `u32` with headroom. Inputs above this bound are caller-side
/// bookkeeping bugs.
pub const FINE_ENERGY_MAX_BITS: u32 = 14;

/// The Q15 scale (`2**15`) the [`fine_correction_q15`] correction is
/// expressed in.
pub const FINE_ENERGY_Q15_ONE: i32 = 1 << 15;

/// Half a 6 dB coarse-energy step, in Q15. The §4.3.2.2 correction is
/// strictly bounded by `±1/2` (the open-interval reconstruction
/// levels never reach the cell edges), so every
/// [`fine_correction_q15`] result lies in
/// `(-FINE_ENERGY_HALF_Q15, +FINE_ENERGY_HALF_Q15)`.
pub const FINE_ENERGY_HALF_Q15: i32 = 1 << 14;

/// The number of channels in a CELT frame: mono (1) or stereo (2).
///
/// The §4.3.2.2 final fine-energy allocation assigns "one extra fine
/// energy bit per band per channel", so the per-band cost of a final
/// bit is the channel count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FineEnergyChannels {
    /// Mono — one extra final fine-energy bit per band.
    Mono,
    /// Stereo — two extra final fine-energy bits per band (one per
    /// channel).
    Stereo,
}

impl FineEnergyChannels {
    /// The per-band cost, in whole bits, of granting a band one extra
    /// final fine-energy bit per channel.
    pub const fn cost_per_band(self) -> u32 {
        match self {
            FineEnergyChannels::Mono => 1,
            FineEnergyChannels::Stereo => 2,
        }
    }

    /// The §4.3.2.2 channel count as an integer (`1` or `2`).
    pub const fn count(self) -> u32 {
        self.cost_per_band()
    }
}

/// Errors returnable by the §4.3.2.2 fine-energy helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FineEnergyError {
    /// `B_i` exceeds [`FINE_ENERGY_MAX_BITS`]. Caller-side bookkeeping
    /// bug — the §4.3.3 allocator never assigns this many fine-energy
    /// bits to a band.
    BitsOutOfRange {
        /// The `B_i` the caller passed.
        provided: u32,
        /// The maximum this module accepts ([`FINE_ENERGY_MAX_BITS`]).
        max: u32,
    },
    /// The read refinement value `f` is `>= 2**B_i`, i.e. it does not
    /// fit the `[0, 2**B_i - 1]` range the §4.3.2.2 formula requires.
    /// A conforming `dec_bits(B_i)` read can never produce such a
    /// value; this guards a caller that passed an `f` from the wrong
    /// `B_i`.
    RefinementOutOfRange {
        /// The `f` the caller passed.
        f: u32,
        /// The exclusive upper bound `2**B_i`.
        levels: u32,
    },
}

impl core::fmt::Display for FineEnergyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            FineEnergyError::BitsOutOfRange { provided, max } => write!(
                f,
                "oxideav-opus: fine-energy bits B_i out of range: provided={provided}, max={max}"
            ),
            FineEnergyError::RefinementOutOfRange { f: val, levels } => write!(
                f,
                "oxideav-opus: fine-energy refinement f={val} out of range \
                 (must be < 2**B_i = {levels})"
            ),
        }
    }
}

impl std::error::Error for FineEnergyError {}

/// The number of fine-energy reconstruction levels for `B_i` bits:
/// `2**B_i`.
///
/// # Errors
///
/// [`FineEnergyError::BitsOutOfRange`] if `bits > FINE_ENERGY_MAX_BITS`.
pub fn fine_energy_levels(bits: u32) -> Result<u32, FineEnergyError> {
    if bits > FINE_ENERGY_MAX_BITS {
        return Err(FineEnergyError::BitsOutOfRange {
            provided: bits,
            max: FINE_ENERGY_MAX_BITS,
        });
    }
    Ok(1u32 << bits)
}

/// The §4.3.2.2 fine-energy correction as the exact reduced rational
/// `(numerator, denominator)`.
///
/// RFC 6716 §4.3.2.2: the correction is `(f + 1/2) / 2**B_i - 1/2`,
/// which rearranges to
///
/// ```text
///     (2*f + 1 - 2**B_i) / 2**(B_i + 1)
/// ```
///
/// The returned pair is `(2*f + 1 - 2**B_i, 2**(B_i + 1))`. The
/// numerator is a signed odd integer in
/// `[-(2**B_i - 1), +(2**B_i - 1)]`; the denominator is a positive
/// power of two. The pair is *not* further reduced (the numerator is
/// always odd, so the fraction is already in lowest terms).
///
/// For `B_i = 0` the only refinement is `f = 0`, giving the pair
/// `(0, 2)` — a correction of exactly zero (no refinement).
///
/// # Errors
///
/// * [`FineEnergyError::BitsOutOfRange`] if `bits > FINE_ENERGY_MAX_BITS`.
/// * [`FineEnergyError::RefinementOutOfRange`] if `f >= 2**bits`.
pub fn fine_correction_ratio(bits: u32, f: u32) -> Result<(i32, i32), FineEnergyError> {
    let levels = fine_energy_levels(bits)?;
    if f >= levels {
        return Err(FineEnergyError::RefinementOutOfRange { f, levels });
    }
    // numerator = 2*f + 1 - 2**B_i (signed, odd).
    let numerator = (2 * f as i64) + 1 - levels as i64;
    // denominator = 2**(B_i + 1).
    let denominator = (levels as i64) * 2;
    Ok((numerator as i32, denominator as i32))
}

/// The §4.3.2.2 fine-energy correction in Q15 (scaled by `2**15`).
///
/// Returns `round_to_zero((2*f + 1 - 2**B_i) * 2**15 / 2**(B_i + 1))`,
/// which for every `B_i <= FINE_ENERGY_MAX_BITS <= 15` is *exact*
/// (the denominator `2**(B_i + 1)` divides `2**16`, so no rounding
/// occurs). The result lies in the open interval
/// `(-FINE_ENERGY_HALF_Q15, +FINE_ENERGY_HALF_Q15)` =
/// `(-16384, +16384)`.
///
/// Worked values:
/// * `B_i = 0, f = 0` → `0` (no refinement).
/// * `B_i = 1, f = 0` → `-8192` (`-1/4` in Q15).
/// * `B_i = 1, f = 1` → `+8192` (`+1/4` in Q15).
/// * `B_i = 2, f = 0` → `-12288` (`-3/8`); `f = 3` → `+12288` (`+3/8`).
///
/// # Errors
///
/// * [`FineEnergyError::BitsOutOfRange`] if `bits > FINE_ENERGY_MAX_BITS`.
/// * [`FineEnergyError::RefinementOutOfRange`] if `f >= 2**bits`.
pub fn fine_correction_q15(bits: u32, f: u32) -> Result<i32, FineEnergyError> {
    let (numerator, denominator) = fine_correction_ratio(bits, f)?;
    // correction * 2**15 = numerator * 2**15 / denominator. Since
    // denominator = 2**(B_i + 1) <= 2**15, the division is exact.
    let scaled = (numerator as i64) * (FINE_ENERGY_Q15_ONE as i64) / (denominator as i64);
    Ok(scaled as i32)
}

/// The §4.3.2.2 fine-energy correction in an arbitrary Q-format with
/// `shift` fractional bits (correction scaled by `2**shift`).
///
/// This generalises [`fine_correction_q15`] for callers whose energy
/// pipeline runs in a different Q-format (CELT's internal log-energy
/// representation uses `DB_SHIFT` fractional bits). Returns
/// `(2*f + 1 - 2**B_i) * 2**shift / 2**(B_i + 1)` with truncation
/// toward zero; the result is exact when `shift >= B_i + 1`.
///
/// # Errors
///
/// * [`FineEnergyError::BitsOutOfRange`] if `bits > FINE_ENERGY_MAX_BITS`.
/// * [`FineEnergyError::RefinementOutOfRange`] if `f >= 2**bits`.
pub fn fine_correction_q(bits: u32, f: u32, shift: u32) -> Result<i64, FineEnergyError> {
    let (numerator, denominator) = fine_correction_ratio(bits, f)?;
    let scaled = (numerator as i64) * (1i64 << shift) / (denominator as i64);
    Ok(scaled)
}

/// The §4.3.2.2 final fine-energy allocation priority of a band.
///
/// The §4.3.3 allocator tags every band with one of two priorities;
/// leftover bits are granted first to all priority-0 bands (band 0
/// upward), then to all priority-1 bands (band 0 upward).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalBitPriority {
    /// Priority 0 — receives a final extra bit before any priority-1
    /// band.
    Priority0,
    /// Priority 1 — receives a final extra bit only after every
    /// priority-0 band has one.
    Priority1,
}

/// The plan produced by [`plan_final_fine_bits`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalFineBitPlan {
    /// One flag per band, in band order: `true` if the band receives
    /// an extra final fine-energy bit per channel.
    pub granted: Vec<bool>,
    /// Whole bits granted in total (`sum(granted) * channels`).
    pub bits_used: u32,
    /// Whole bits left unused after the §4.3.2.2 priority sweep.
    pub bits_unused: u32,
}

/// Plans the §4.3.2.2 *final* fine-energy bit allocation.
///
/// Given the per-band priority vector, the channel count, and the
/// number of leftover whole bits, returns the per-band grant flags in
/// the exact §4.3.2.2 order: priority-0 bands first (band 0 upward),
/// then priority-1 bands (band 0 upward), each grant costing
/// `channels` whole bits. The sweep stops as soon as the remaining
/// budget cannot afford another `channels`-bit grant; any leftover is
/// reported in [`FinalFineBitPlan::bits_unused`].
///
/// A band already at its maximum fine-energy precision cannot accept a
/// final bit; callers pass `priorities[b] = None` to exclude such a
/// band from the sweep entirely (it is never granted and never costs
/// budget). This matches the §4.3.3 allocator marking a band
/// ineligible for final refinement.
pub fn plan_final_fine_bits(
    priorities: &[Option<FinalBitPriority>],
    channels: FineEnergyChannels,
    leftover_bits: u32,
) -> FinalFineBitPlan {
    let cost = channels.cost_per_band();
    let mut granted = vec![false; priorities.len()];
    let mut remaining = leftover_bits;
    let mut bits_used = 0u32;

    // Two passes: priority 0 first, then priority 1. Within each pass,
    // band 0 upward.
    for pass in [FinalBitPriority::Priority0, FinalBitPriority::Priority1] {
        for (band, prio) in priorities.iter().enumerate() {
            if *prio != Some(pass) {
                continue;
            }
            if remaining < cost {
                // Not enough budget for another grant — the §4.3.2.2
                // sweep leaves the rest unused.
                break;
            }
            granted[band] = true;
            remaining -= cost;
            bits_used += cost;
        }
        // After priority 0, fall through into priority 1 only if
        // budget remains. (The inner `break` already handled the
        // exhausted case; an explicit guard keeps the intent clear.)
        if remaining < cost {
            break;
        }
    }

    FinalFineBitPlan {
        granted,
        bits_used,
        bits_unused: remaining,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- module constants ---------------------------------------------------

    #[test]
    fn q15_one_is_two_pow_fifteen() {
        assert_eq!(FINE_ENERGY_Q15_ONE, 32768);
    }

    #[test]
    fn half_q15_is_two_pow_fourteen() {
        assert_eq!(FINE_ENERGY_HALF_Q15, 16384);
        assert_eq!(FINE_ENERGY_HALF_Q15 * 2, FINE_ENERGY_Q15_ONE);
    }

    #[test]
    fn max_bits_keeps_q15_exact() {
        // The Q15 correction is exact iff 2**(B_i + 1) divides 2**16,
        // i.e. B_i + 1 <= 16, i.e. B_i <= 15. The bound is set at 14
        // (one below) for u32 level-count headroom; confirm it stays
        // inside the exact regime.
        const { assert!(FINE_ENERGY_MAX_BITS <= 15) };
    }

    // ---- channel helper -----------------------------------------------------

    #[test]
    fn channel_costs() {
        assert_eq!(FineEnergyChannels::Mono.cost_per_band(), 1);
        assert_eq!(FineEnergyChannels::Stereo.cost_per_band(), 2);
        assert_eq!(FineEnergyChannels::Mono.count(), 1);
        assert_eq!(FineEnergyChannels::Stereo.count(), 2);
    }

    // ---- levels -------------------------------------------------------------

    #[test]
    fn levels_is_two_pow_bits() {
        for b in 0..=FINE_ENERGY_MAX_BITS {
            assert_eq!(fine_energy_levels(b), Ok(1u32 << b), "B_i = {b}");
        }
    }

    #[test]
    fn levels_rejects_over_max() {
        assert_eq!(
            fine_energy_levels(FINE_ENERGY_MAX_BITS + 1),
            Err(FineEnergyError::BitsOutOfRange {
                provided: FINE_ENERGY_MAX_BITS + 1,
                max: FINE_ENERGY_MAX_BITS,
            })
        );
    }

    // ---- §4.3.2.2 ratio form ------------------------------------------------

    #[test]
    fn ratio_b0_is_zero() {
        // B_i = 0: only f = 0; correction = 0/2.
        assert_eq!(fine_correction_ratio(0, 0), Ok((0, 2)));
    }

    #[test]
    fn ratio_b1() {
        // B_i = 1: (2f + 1 - 2) / 4.
        // f = 0 → (-1, 4) = -1/4. f = 1 → (+1, 4) = +1/4.
        assert_eq!(fine_correction_ratio(1, 0), Ok((-1, 4)));
        assert_eq!(fine_correction_ratio(1, 1), Ok((1, 4)));
    }

    #[test]
    fn ratio_b2() {
        // B_i = 2: (2f + 1 - 4) / 8.
        // f = 0 → (-3, 8); f = 1 → (-1, 8); f = 2 → (+1, 8); f = 3 → (+3, 8).
        assert_eq!(fine_correction_ratio(2, 0), Ok((-3, 8)));
        assert_eq!(fine_correction_ratio(2, 1), Ok((-1, 8)));
        assert_eq!(fine_correction_ratio(2, 2), Ok((1, 8)));
        assert_eq!(fine_correction_ratio(2, 3), Ok((3, 8)));
    }

    #[test]
    fn ratio_numerator_is_always_odd() {
        // The numerator 2f + 1 - 2**B_i is odd for every B_i >= 1, so
        // the fraction is in lowest terms (denominator is a power of 2).
        for b in 1..=FINE_ENERGY_MAX_BITS {
            let levels = 1u32 << b;
            for f in 0..levels {
                let (num, _den) = fine_correction_ratio(b, f).unwrap();
                assert_eq!(num & 1, 1, "B_i = {b}, f = {f}, num = {num}");
            }
        }
    }

    #[test]
    fn ratio_zero_mean_symmetry() {
        // The reconstruction levels are symmetric about zero: f and
        // (2**B_i - 1 - f) give opposite-sign numerators with the same
        // magnitude.
        for b in 1..=10u32 {
            let levels = 1u32 << b;
            for f in 0..levels {
                let (num, den) = fine_correction_ratio(b, f).unwrap();
                let (num_mirror, den_mirror) = fine_correction_ratio(b, levels - 1 - f).unwrap();
                assert_eq!(den, den_mirror);
                assert_eq!(num, -num_mirror, "B_i = {b}, f = {f}");
            }
        }
    }

    #[test]
    fn ratio_rejects_f_out_of_range() {
        // f = 2**B_i is one past the top level.
        assert_eq!(
            fine_correction_ratio(2, 4),
            Err(FineEnergyError::RefinementOutOfRange { f: 4, levels: 4 })
        );
    }

    #[test]
    fn ratio_rejects_bits_out_of_range() {
        assert_eq!(
            fine_correction_ratio(FINE_ENERGY_MAX_BITS + 1, 0),
            Err(FineEnergyError::BitsOutOfRange {
                provided: FINE_ENERGY_MAX_BITS + 1,
                max: FINE_ENERGY_MAX_BITS,
            })
        );
    }

    // ---- §4.3.2.2 Q15 form --------------------------------------------------

    #[test]
    fn q15_b0_is_zero() {
        assert_eq!(fine_correction_q15(0, 0), Ok(0));
    }

    #[test]
    fn q15_b1_quarters() {
        // -1/4 and +1/4 in Q15 = ±8192.
        assert_eq!(fine_correction_q15(1, 0), Ok(-8192));
        assert_eq!(fine_correction_q15(1, 1), Ok(8192));
    }

    #[test]
    fn q15_b2_eighths() {
        // -3/8, -1/8, +1/8, +3/8 in Q15.
        assert_eq!(fine_correction_q15(2, 0), Ok(-12288)); // -3/8 * 32768
        assert_eq!(fine_correction_q15(2, 1), Ok(-4096)); //  -1/8 * 32768
        assert_eq!(fine_correction_q15(2, 2), Ok(4096)); //   +1/8 * 32768
        assert_eq!(fine_correction_q15(2, 3), Ok(12288)); //  +3/8 * 32768
    }

    #[test]
    fn q15_matches_ratio_exactly() {
        // For every reachable (B_i, f) the Q15 value equals
        // numerator * 32768 / denominator with no rounding.
        for b in 0..=FINE_ENERGY_MAX_BITS {
            let levels = 1u32 << b;
            for f in 0..levels {
                let (num, den) = fine_correction_ratio(b, f).unwrap();
                let expect = (num as i64) * (FINE_ENERGY_Q15_ONE as i64) / (den as i64);
                // Exactness: the remainder must be zero.
                assert_eq!(
                    (num as i64) * (FINE_ENERGY_Q15_ONE as i64) % (den as i64),
                    0,
                    "B_i = {b}, f = {f} not exact in Q15"
                );
                assert_eq!(
                    fine_correction_q15(b, f),
                    Ok(expect as i32),
                    "B_i = {b}, f = {f}"
                );
            }
        }
    }

    #[test]
    fn q15_strictly_within_half() {
        // Every correction lies strictly inside (-1/2, +1/2).
        for b in 0..=FINE_ENERGY_MAX_BITS {
            let levels = 1u32 << b;
            for f in 0..levels {
                let q = fine_correction_q15(b, f).unwrap();
                assert!(
                    q > -FINE_ENERGY_HALF_Q15 && q < FINE_ENERGY_HALF_Q15,
                    "B_i = {b}, f = {f}, q = {q} escapes (-16384, 16384)"
                );
            }
        }
    }

    #[test]
    fn q15_monotone_increasing_in_f() {
        // Larger f → larger (less negative) correction.
        for b in 1..=FINE_ENERGY_MAX_BITS {
            let levels = 1u32 << b;
            let mut prev = i32::MIN;
            for f in 0..levels {
                let q = fine_correction_q15(b, f).unwrap();
                assert!(q > prev, "B_i = {b}, f = {f}: {q} <= {prev}");
                prev = q;
            }
        }
    }

    #[test]
    fn q15_step_is_uniform() {
        // Consecutive reconstruction levels are equally spaced by
        // 1/2**B_i in Q15 = 32768 >> B_i.
        for b in 1..=FINE_ENERGY_MAX_BITS {
            let levels = 1u32 << b;
            let step = (FINE_ENERGY_Q15_ONE as u32 >> b) as i32;
            for f in 1..levels {
                let here = fine_correction_q15(b, f).unwrap();
                let before = fine_correction_q15(b, f - 1).unwrap();
                assert_eq!(here - before, step, "B_i = {b}, f = {f}");
            }
        }
    }

    // ---- generic Q-format ---------------------------------------------------

    #[test]
    fn q_generic_matches_q15_at_shift_15() {
        for b in 0..=FINE_ENERGY_MAX_BITS {
            let levels = 1u32 << b;
            for f in 0..levels {
                assert_eq!(
                    fine_correction_q(b, f, 15).unwrap(),
                    fine_correction_q15(b, f).unwrap() as i64,
                    "B_i = {b}, f = {f}"
                );
            }
        }
    }

    #[test]
    fn q_generic_db_shift_10_exact() {
        // CELT's internal energy Q-format uses DB_SHIFT = 10 fractional
        // bits. For B_i + 1 <= 10 the conversion is exact.
        // B_i = 1: -1/4 in Q10 = -256; +1/4 = +256.
        assert_eq!(fine_correction_q(1, 0, 10).unwrap(), -256);
        assert_eq!(fine_correction_q(1, 1, 10).unwrap(), 256);
        // B_i = 2: -3/8 in Q10 = -384.
        assert_eq!(fine_correction_q(2, 0, 10).unwrap(), -384);
    }

    // ---- §4.3.2.2 final fine-energy allocation ------------------------------

    #[test]
    fn final_priority0_before_priority1() {
        // Two priority-0 bands (0, 2), one priority-1 band (1). With
        // budget for two mono bits, only the priority-0 bands get them.
        let prios = vec![
            Some(FinalBitPriority::Priority0),
            Some(FinalBitPriority::Priority1),
            Some(FinalBitPriority::Priority0),
        ];
        let plan = plan_final_fine_bits(&prios, FineEnergyChannels::Mono, 2);
        assert_eq!(plan.granted, vec![true, false, true]);
        assert_eq!(plan.bits_used, 2);
        assert_eq!(plan.bits_unused, 0);
    }

    #[test]
    fn final_priority1_after_all_priority0() {
        // One priority-0 band (0), two priority-1 bands (1, 2). Budget
        // for three mono bits: band 0 then bands 1, 2.
        let prios = vec![
            Some(FinalBitPriority::Priority0),
            Some(FinalBitPriority::Priority1),
            Some(FinalBitPriority::Priority1),
        ];
        let plan = plan_final_fine_bits(&prios, FineEnergyChannels::Mono, 3);
        assert_eq!(plan.granted, vec![true, true, true]);
        assert_eq!(plan.bits_used, 3);
        assert_eq!(plan.bits_unused, 0);
    }

    #[test]
    fn final_band_order_within_priority() {
        // Three priority-0 bands; only budget for two — bands 0 and 1
        // (lowest indices) win, band 4 does not.
        let prios = vec![
            Some(FinalBitPriority::Priority0),
            Some(FinalBitPriority::Priority0),
            None,
            None,
            Some(FinalBitPriority::Priority0),
        ];
        let plan = plan_final_fine_bits(&prios, FineEnergyChannels::Mono, 2);
        assert_eq!(plan.granted, vec![true, true, false, false, false]);
        assert_eq!(plan.bits_used, 2);
        assert_eq!(plan.bits_unused, 0);
    }

    #[test]
    fn final_stereo_costs_two_per_band() {
        // Stereo: each granted band costs 2 bits. Budget 3 → only one
        // band granted, 1 bit unused.
        let prios = vec![
            Some(FinalBitPriority::Priority0),
            Some(FinalBitPriority::Priority0),
        ];
        let plan = plan_final_fine_bits(&prios, FineEnergyChannels::Stereo, 3);
        assert_eq!(plan.granted, vec![true, false]);
        assert_eq!(plan.bits_used, 2);
        assert_eq!(plan.bits_unused, 1);
    }

    #[test]
    fn final_leftover_unused() {
        // §4.3.2.2: "If any bits are left after this, they are left
        // unused." All bands granted, budget exceeds need.
        let prios = vec![
            Some(FinalBitPriority::Priority0),
            Some(FinalBitPriority::Priority1),
        ];
        let plan = plan_final_fine_bits(&prios, FineEnergyChannels::Mono, 5);
        assert_eq!(plan.granted, vec![true, true]);
        assert_eq!(plan.bits_used, 2);
        assert_eq!(plan.bits_unused, 3);
    }

    #[test]
    fn final_none_bands_excluded() {
        // None-priority bands never receive a final bit and never cost
        // budget, even with surplus.
        let prios = vec![None, None, None];
        let plan = plan_final_fine_bits(&prios, FineEnergyChannels::Mono, 10);
        assert_eq!(plan.granted, vec![false, false, false]);
        assert_eq!(plan.bits_used, 0);
        assert_eq!(plan.bits_unused, 10);
    }

    #[test]
    fn final_zero_budget_grants_nothing() {
        let prios = vec![
            Some(FinalBitPriority::Priority0),
            Some(FinalBitPriority::Priority1),
        ];
        let plan = plan_final_fine_bits(&prios, FineEnergyChannels::Mono, 0);
        assert_eq!(plan.granted, vec![false, false]);
        assert_eq!(plan.bits_used, 0);
        assert_eq!(plan.bits_unused, 0);
    }

    #[test]
    fn final_priority0_exhausts_budget_priority1_gets_none() {
        // Budget for exactly one mono bit: priority-0 band 1 takes it,
        // priority-1 band 0 gets nothing despite being band 0.
        let prios = vec![
            Some(FinalBitPriority::Priority1),
            Some(FinalBitPriority::Priority0),
        ];
        let plan = plan_final_fine_bits(&prios, FineEnergyChannels::Mono, 1);
        assert_eq!(plan.granted, vec![false, true]);
        assert_eq!(plan.bits_used, 1);
        assert_eq!(plan.bits_unused, 0);
    }

    #[test]
    fn final_bits_conserved() {
        // bits_used + bits_unused == leftover_bits, always.
        let prios = vec![
            Some(FinalBitPriority::Priority0),
            None,
            Some(FinalBitPriority::Priority1),
            Some(FinalBitPriority::Priority0),
        ];
        for chan in [FineEnergyChannels::Mono, FineEnergyChannels::Stereo] {
            for budget in 0..=12u32 {
                let plan = plan_final_fine_bits(&prios, chan, budget);
                assert_eq!(
                    plan.bits_used + plan.bits_unused,
                    budget,
                    "chan = {chan:?}, budget = {budget}"
                );
            }
        }
    }

    #[test]
    fn final_empty_priorities() {
        let plan = plan_final_fine_bits(&[], FineEnergyChannels::Mono, 7);
        assert_eq!(plan.granted, Vec::<bool>::new());
        assert_eq!(plan.bits_used, 0);
        assert_eq!(plan.bits_unused, 7);
    }

    // ---- error Display ------------------------------------------------------

    #[test]
    fn display_messages_mention_input() {
        let e1 = FineEnergyError::BitsOutOfRange {
            provided: 99,
            max: FINE_ENERGY_MAX_BITS,
        };
        assert!(format!("{e1}").contains("99"));
        let e2 = FineEnergyError::RefinementOutOfRange { f: 4, levels: 4 };
        let m2 = format!("{e2}");
        assert!(m2.contains("f=4"));
        assert!(m2.contains("4"));
    }
}
