//! CELT §4.3.3 reservation block (RFC 6716 §4.3.3, p. 114).
//!
//! The §4.3.3 *Bit Allocation* procedure begins by skimming a small
//! number of fixed-cost bits off the top of the frame budget so the
//! signalling that follows the Table 57 static-allocation search
//! (anti-collapse flag, skipping flag, intensity-stereo parameter,
//! dual-stereo flag) is guaranteed to fit. These four "reservations"
//! are subtracted from the §4.3.3 working `total` budget in a fixed
//! order; any reservation whose cost would exceed `total` collapses to
//! zero and is *not* signalled, so the §4.3.3 allocator skips that
//! signalling step as well.
//!
//! The §4.3.3 narrative (RFC 6716 §4.3.3, p. 114) reads:
//!
//! > For 10 ms and 20 ms frames using short blocks and that have at
//! > least LM+2 bits left prior to the allocation process, one
//! > anti-collapse bit is reserved in the allocation process so it
//! > can be decoded later. Following the anti-collapse reservation,
//! > one bit is reserved for skip if available.
//! >
//! > For stereo frames, bits are reserved for intensity stereo and
//! > for dual stereo. Intensity stereo requires ilog2(end-start)
//! > bits. Those bits are reserved if there are enough bits left.
//! > Following this, one bit is reserved for dual stereo if
//! > available.
//! >
//! > The allocation computation begins by setting up some initial
//! > conditions. 'total' is set to the remaining available 8th bits,
//! > computed by taking the size of the coded frame times 8 and
//! > subtracting ec_tell_frac(). From this value, one (8th bit) is
//! > subtracted to ensure that the resulting allocation will be
//! > conservative. 'anti_collapse_rsv' is set to 8 (8th bits) if and
//! > only if the frame is a transient, LM is greater than 1, and total
//! > is greater than or equal to (LM+2) * 8. Total is then decremented
//! > by anti_collapse_rsv and clamped to be equal to or greater than
//! > zero. 'skip_rsv' is set to 8 (8th bits) if total is greater than
//! > 8, otherwise it is zero. Total is then decremented by skip_rsv.
//! > This reserves space for the final skipping flag.
//! >
//! > If the current frame is stereo, intensity_rsv is set to the
//! > conservative log2 in 8th bits of the number of coded bands for
//! > this frame (given by the table LOG2_FRAC_TABLE in rate.c). If
//! > intensity_rsv is greater than total, then intensity_rsv is set
//! > to zero. Otherwise, total is decremented by intensity_rsv, and
//! > if total is still greater than 8, dual_stereo_rsv is set to 8
//! > and total is decremented by dual_stereo_rsv.
//!
//! This module owns the §4.3.3 reservation arithmetic and its typed
//! outcome. It does not own:
//!
//! * The §4.3.3 band-boost loop (round 33,
//!   [`crate::celt_band_boost::decode_band_boosts`]) that updates
//!   `ec_tell_frac` and `total_boost` *before* the §4.3.3 reservation
//!   block runs.
//! * The §4.3.3 allocation trim (round 32,
//!   [`crate::celt_alloc_trim::decode_alloc_trim`]) that runs
//!   *between* the band-boost and reservation blocks, biasing the
//!   §4.3.3 Table 57 static-allocation search at the consumer site
//!   that takes the [`ReservationOutcome`] this module emits.
//! * The §4.3.3 *use* of the reserved bits — the actual `dec_bit_logp`
//!   read of the anti-collapse / skip / dual-stereo flags and the
//!   `ec_dec_uint(end - start)` read of the intensity-stereo band — runs
//!   at the §4.3.3 allocator's consumer site after the Table 57 search
//!   produces the per-band shape allocation.
//!
//! ## Units
//!
//! Every input and every reservation cost is in 1/8 bits ("8th bits"
//! in the RFC body, "Q3" elsewhere in CELT). The conversion factor
//! between Opus frame bytes and the budget is `frame_size_bytes * 64`
//! (RFC §3.4 R5: 1275 bytes max ⇒ 81600 1/8 bits worst case, fits in
//! `u32` by a wide margin). The reservation costs:
//!
//! * `anti_collapse_rsv` ∈ {0, 8} — exactly one whole bit at most.
//! * `skip_rsv` ∈ {0, 8} — exactly one whole bit at most.
//! * `intensity_rsv` ∈ {0} ∪ Q3 values from the §4.3.3
//!   `LOG2_FRAC_TABLE` lookup at index `end − start` (see
//!   [`crate::celt_log2_frac_table::LOG2_FRAC_TABLE`]); the §4.3.3
//!   range is `0..=37` in 1/8 bits, i.e. up to ~4.6 whole bits.
//! * `dual_stereo_rsv` ∈ {0, 8} — exactly one whole bit at most.
//!
//! ## §4.3.3 anti-collapse gating
//!
//! The anti-collapse reservation runs only when *all three* of the
//! §4.3.3 conditions hold:
//!
//! 1. The §4.3.1 `transient` flag is set on the frame.
//! 2. `LM > 1`, i.e. CELT frame size ≥ 10 ms ([`CeltFrameSize::Ms10`]
//!    or [`CeltFrameSize::Ms20`]).
//! 3. `total ≥ (LM + 2) * 8` in 1/8 bits.
//!
//! Condition 2 reflects the RFC's "10 ms and 20 ms frames using short
//! blocks": short blocks (the transient case) at 10 ms / 20 ms frame
//! durations decompose into multiple sub-MDCTs and need the
//! anti-collapse safety net; at 2.5 ms and 5 ms the frame *is* a single
//! sub-MDCT and the anti-collapse step has nothing to do.
//!
//! ## §4.3.3 stereo branch gating
//!
//! The intensity-stereo and dual-stereo reservations only run for
//! stereo frames. For mono frames both are zero.
//!
//! Within the stereo branch, the §4.3.3 ordering is strict:
//!
//! 1. Compute `intensity_rsv = LOG2_FRAC_TABLE[coded_bands]` from
//!    [`crate::celt_log2_frac_table::log2_frac`].
//! 2. If `intensity_rsv > total`, set `intensity_rsv = 0` and skip
//!    the dual-stereo branch.
//! 3. Otherwise decrement `total` by `intensity_rsv`, then check
//!    `total > 8`. If so, set `dual_stereo_rsv = 8` and decrement
//!    `total` by 8; otherwise leave `dual_stereo_rsv = 0`.
//!
//! ## Provenance
//!
//! Narrative: RFC 6716 §4.3.3 (p. 114) in
//! `docs/audio/opus/rfc6716-opus.txt`; cross-referenced by
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.5
//! (steps 1–4 of the allocation initial-conditions list). No external
//! numeric table is required for this module: the four reservation
//! costs (8, 8, `LOG2_FRAC_TABLE[…]`, 8) and the §4.3.3 gating
//! predicates are inlined in the RFC body. The intensity-stereo
//! reservation's `LOG2_FRAC_TABLE` lookup is owned by round 30's
//! [`crate::celt_log2_frac_table`] module.

use crate::celt_band_layout::CeltFrameSize;
use crate::celt_log2_frac_table::{log2_frac, Log2FracError};

/// §4.3.3 anti-collapse / skip / dual-stereo reservation cost in 1/8
/// bits (RFC 6716 §4.3.3 p. 114). Each of these flags costs exactly
/// one whole bit, hence 8 1/8-bit units.
pub const ONE_BIT_EIGHTH_BITS: u32 = 8;

/// §4.3.3 conservative-allocation deduction in 1/8 bits (RFC 6716
/// §4.3.3 p. 114: "one (8th bit) is subtracted to ensure that the
/// resulting allocation will be conservative").
pub const CONSERVATIVE_DEDUCTION_EIGHTH_BITS: u32 = 1;

/// §4.3.3 conversion factor: 1 byte = 8 whole bits = 64 1/8-bit units.
/// Matches `EIGHTH_BITS_PER_BYTE` exposed by
/// [`crate::celt_alloc_trim`]; duplicated here so callers don't need
/// to reach into the trim module just to convert byte counts.
pub const EIGHTH_BITS_PER_BYTE: u32 = 64;

/// §4.3.3 anti-collapse LM minimum (RFC 6716 §4.3.3 p. 114: "10 ms and
/// 20 ms frames"; equivalently LM > 1 since LM = 0,1,2,3 maps to
/// 2.5,5,10,20 ms in [`CeltFrameSize`]).
pub const ANTI_COLLAPSE_LM_MIN_EXCLUSIVE: u32 = 1;

/// §4.3.3 anti-collapse total-budget headroom multiplier (RFC 6716
/// §4.3.3 p. 114: "total is greater than or equal to (LM+2) * 8").
/// In 1/8 bits the threshold is `(LM + 2) * 8`.
pub const ANTI_COLLAPSE_HEADROOM_MULT_EIGHTH_BITS: u32 = 8;

/// §4.3.3 anti-collapse LM offset added to LM before multiplying by
/// [`ANTI_COLLAPSE_HEADROOM_MULT_EIGHTH_BITS`].
pub const ANTI_COLLAPSE_HEADROOM_LM_OFFSET: u32 = 2;

/// Errors returned by [`reserve_block`] for inputs that violate the
/// §4.3.3 bookkeeping. None of these come from the range coder; they
/// are caller-side bookkeeping bugs (the range coder's own sticky
/// error flag covers a corrupt stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationError {
    /// `frame_size_bytes * 64` overflows `u32`. The §4.3.3 budget
    /// fits in `u32` by a wide margin for any well-formed Opus packet
    /// (§3.4 R5: 1275-byte max ⇒ 81600 1/8 bits); a `frame_size_bytes`
    /// near `u32::MAX / 64` is a caller-side bug.
    FrameSizeOverflows,
    /// `ec_tell_frac` (in 1/8 bits) already exceeds the frame budget.
    /// The range coder cannot have consumed more 1/8 bits than the
    /// frame contains; a value past the frame budget is a caller-side
    /// bug.
    TellExceedsFrame {
        frame_eighth_bits: u32,
        ec_tell_frac: u32,
    },
    /// `total_boost` already exceeds the §4.3.3 frame budget after
    /// the `ec_tell_frac` deduction. The §4.3.3 boost loop should
    /// have stopped well before the budget is consumed; a larger
    /// value is a caller-side bug.
    TotalBoostExceedsFrame {
        frame_eighth_bits: u32,
        ec_tell_frac: u32,
        total_boost: u32,
    },
    /// `coded_bands` exceeds the §4.3.3
    /// [`crate::celt_log2_frac_table::LOG2_FRAC_TABLE`] coverage
    /// (`0..24`). The §4.3.3 CELT band loop never produces more than
    /// 21 coded bands; a larger value is a caller-side bug.
    LogFracLookupFailed(Log2FracError),
}

impl From<Log2FracError> for ReservationError {
    fn from(value: Log2FracError) -> Self {
        ReservationError::LogFracLookupFailed(value)
    }
}

/// §4.3.3 reservation outcome: the four reservation costs and the
/// post-deduction `total` budget the §4.3.3 Table 57 static-allocation
/// search consumes.
///
/// Every field is in 1/8 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReservationOutcome {
    /// §4.3.3 `anti_collapse_rsv` — 0 (not reserved) or 8 (one whole
    /// bit reserved). The §4.3.3 allocator reads one
    /// `dec_bit_logp(1)` from the stream iff this is `8`.
    pub anti_collapse_rsv: u32,
    /// §4.3.3 `skip_rsv` — 0 or 8.
    pub skip_rsv: u32,
    /// §4.3.3 `intensity_rsv` — 0 or
    /// `LOG2_FRAC_TABLE[end − start]` from
    /// [`crate::celt_log2_frac_table::log2_frac`]. Zero for mono
    /// frames and for stereo frames whose budget can't afford the
    /// reservation.
    pub intensity_rsv: u32,
    /// §4.3.3 `dual_stereo_rsv` — 0 or 8. Always 0 for mono frames
    /// and for stereo frames whose `intensity_rsv` was not reserved.
    pub dual_stereo_rsv: u32,
    /// §4.3.3 working `total` budget after every reservation has been
    /// deducted, in 1/8 bits. This is the value the §4.3.3 Table 57
    /// static-allocation search starts from.
    pub total_remaining_eighth_bits: u32,
}

impl ReservationOutcome {
    /// Sum of every reserved cost in 1/8 bits. Useful for the §4.3.3
    /// invariant
    /// `frame - ec_tell_frac - 1 = total_remaining + reserved_total`.
    pub const fn reserved_total_eighth_bits(&self) -> u32 {
        self.anti_collapse_rsv
            .saturating_add(self.skip_rsv)
            .saturating_add(self.intensity_rsv)
            .saturating_add(self.dual_stereo_rsv)
    }
}

/// Compute the §4.3.3 reservation block (RFC 6716 §4.3.3, p. 114).
///
/// Inputs:
///
/// * `frame_size_bytes` — §3.4 Opus frame size in bytes.
/// * `ec_tell_frac` — 1/8-bit count of bits the range coder has
///   already consumed (i.e. `rd.tell_frac()` at the call site, *after*
///   the §4.3.3 boost loop has finished).
/// * `total_boost` — §4.3.3 `total_boost` from
///   [`crate::celt_band_boost::decode_band_boosts`] (in 1/8 bits).
/// * `lm` — §4.3 frame-size scale (LM ∈ 0..=3, i.e. 2.5/5/10/20 ms),
///   typed as [`CeltFrameSize`].
/// * `is_transient` — §4.3.1 transient flag (already decoded by
///   [`crate::celt_header::CeltHeaderPrefix`]).
/// * `is_stereo` — channel-count flag; `true` for stereo, `false` for
///   mono.
/// * `coded_bands` — `end - start` for the §4.3 coding window (0..=21
///   normally, ≤ 4 in Hybrid mode). Index for the
///   [`crate::celt_log2_frac_table::LOG2_FRAC_TABLE`] lookup.
///
/// Implements the §4.3.3 narrative verbatim:
///
/// 1. `total = frame_size_bytes * 64 − ec_tell_frac − 1`.
/// 2. `anti_collapse_rsv = 8` iff `is_transient && lm > 1 && total ≥
///    (lm + 2) * 8`. Deduct.
/// 3. `skip_rsv = 8` iff `total > 8` after step 2. Deduct.
/// 4. (Stereo only) `intensity_rsv = LOG2_FRAC_TABLE[coded_bands]`.
///    If `intensity_rsv > total`, reset to 0 and skip step 5.
///    Otherwise deduct.
/// 5. (Stereo only, after step 4 deducted) `dual_stereo_rsv = 8` iff
///    `total > 8`. Deduct.
///
/// Returns the [`ReservationOutcome`] holding every reserved cost and
/// the post-deduction `total` budget for the §4.3.3 allocator.
///
/// The §4.3.3 procedure also clamps `total ≥ 0` after the
/// anti-collapse deduction; this implementation enforces that via
/// `saturating_sub` rather than wrapping arithmetic, matching the
/// "clamped to be equal to or greater than zero" wording.
///
/// The `total_boost` input is *not* deducted from the working budget
/// here — the band-boost loop already removed it during the boost
/// decode (the band-boost loop's `total_bits` accumulator already
/// reflects the boost-conditioned remainder). What this function
/// *does* care about is the §4.3.3 invariant that the band-boost loop
/// finished with `tell + total_boost` within the frame budget; the
/// implementation enforces it via the
/// [`ReservationError::TotalBoostExceedsFrame`] check below.
pub fn reserve_block(
    frame_size_bytes: u32,
    ec_tell_frac: u32,
    total_boost: u32,
    lm: CeltFrameSize,
    is_transient: bool,
    is_stereo: bool,
    coded_bands: u32,
) -> Result<ReservationOutcome, ReservationError> {
    // Step 1: total = frame_size * 64 - ec_tell_frac - 1.
    let frame_eighth_bits = frame_size_bytes
        .checked_mul(EIGHTH_BITS_PER_BYTE)
        .ok_or(ReservationError::FrameSizeOverflows)?;
    if ec_tell_frac > frame_eighth_bits {
        return Err(ReservationError::TellExceedsFrame {
            frame_eighth_bits,
            ec_tell_frac,
        });
    }
    // The §4.3.3 invariant is that the band-boost loop's
    // `tell + total_boost ≤ frame_size`. Enforce it as a caller-side
    // contract; an out-of-range value indicates a bookkeeping bug in
    // the band-boost loop's caller, not a malformed bitstream.
    let after_tell = frame_eighth_bits - ec_tell_frac;
    if total_boost > after_tell {
        return Err(ReservationError::TotalBoostExceedsFrame {
            frame_eighth_bits,
            ec_tell_frac,
            total_boost,
        });
    }
    // §4.3.3: "From this value, one (8th bit) is subtracted to ensure
    // that the resulting allocation will be conservative." This is
    // the unconditional 1-unit deduction, independent of any
    // reservation.
    let mut total = after_tell.saturating_sub(CONSERVATIVE_DEDUCTION_EIGHTH_BITS);

    // Step 2: anti-collapse reservation.
    let lm_idx = lm.column_index() as u32;
    let mut anti_collapse_rsv: u32 = 0;
    if is_transient && lm_idx > ANTI_COLLAPSE_LM_MIN_EXCLUSIVE {
        // §4.3.3: total ≥ (LM + 2) * 8.
        let headroom = (lm_idx + ANTI_COLLAPSE_HEADROOM_LM_OFFSET)
            .saturating_mul(ANTI_COLLAPSE_HEADROOM_MULT_EIGHTH_BITS);
        if total >= headroom {
            anti_collapse_rsv = ONE_BIT_EIGHTH_BITS;
        }
    }
    total = total.saturating_sub(anti_collapse_rsv);

    // Step 3: skip reservation.
    let mut skip_rsv: u32 = 0;
    if total > ONE_BIT_EIGHTH_BITS {
        skip_rsv = ONE_BIT_EIGHTH_BITS;
    }
    total = total.saturating_sub(skip_rsv);

    // Steps 4 + 5: stereo intensity + dual-stereo reservations.
    let mut intensity_rsv: u32 = 0;
    let mut dual_stereo_rsv: u32 = 0;
    if is_stereo {
        // §4.3.3 intensity_rsv lookup. The §4.3.3 LOG2_FRAC_TABLE
        // index is `end - start` = `coded_bands` directly.
        let raw_intensity = log2_frac(coded_bands)? as u32;
        if raw_intensity > total {
            // §4.3.3: "If intensity_rsv is greater than total, then
            // intensity_rsv is set to zero." dual_stereo_rsv stays 0
            // because the §4.3.3 procedure requires the intensity
            // reservation to succeed before considering the dual one.
            intensity_rsv = 0;
        } else {
            intensity_rsv = raw_intensity;
            total -= intensity_rsv;
            if total > ONE_BIT_EIGHTH_BITS {
                dual_stereo_rsv = ONE_BIT_EIGHTH_BITS;
                total -= dual_stereo_rsv;
            }
        }
    }

    Ok(ReservationOutcome {
        anti_collapse_rsv,
        skip_rsv,
        intensity_rsv,
        dual_stereo_rsv,
        total_remaining_eighth_bits: total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Shape constants ----

    #[test]
    fn one_bit_constant_matches_rfc() {
        // RFC 6716 §4.3.3 p. 114 names "8 (8th bits)" four times:
        // anti_collapse_rsv, skip_rsv, dual_stereo_rsv, and the
        // "total > 8" gates. The constant is 8 units = 1 whole bit.
        assert_eq!(ONE_BIT_EIGHTH_BITS, 8);
        assert_eq!(ONE_BIT_EIGHTH_BITS / 8, 1);
    }

    #[test]
    fn conservative_deduction_constant_is_one() {
        // RFC 6716 §4.3.3 p. 114: "one (8th bit) is subtracted".
        assert_eq!(CONSERVATIVE_DEDUCTION_EIGHTH_BITS, 1);
    }

    #[test]
    fn eighth_bits_per_byte_matches_trim_module() {
        // The two modules must agree on the byte→1/8-bit scale.
        assert_eq!(EIGHTH_BITS_PER_BYTE, 64);
        assert_eq!(
            EIGHTH_BITS_PER_BYTE,
            crate::celt_alloc_trim::EIGHTH_BITS_PER_BYTE
        );
    }

    #[test]
    fn anti_collapse_lm_threshold_matches_rfc() {
        // RFC 6716 §4.3.3 p. 114: "10 ms and 20 ms frames" — i.e.
        // LM > 1 because the LM column index maps 2.5/5/10/20 ms to
        // 0/1/2/3.
        assert_eq!(ANTI_COLLAPSE_LM_MIN_EXCLUSIVE, 1);
        assert_eq!(CeltFrameSize::Ms2_5.column_index() as u32, 0);
        assert_eq!(CeltFrameSize::Ms5.column_index() as u32, 1);
        assert_eq!(CeltFrameSize::Ms10.column_index() as u32, 2);
        assert_eq!(CeltFrameSize::Ms20.column_index() as u32, 3);
    }

    #[test]
    fn anti_collapse_headroom_constants_match_rfc() {
        // RFC 6716 §4.3.3 p. 114: "total is greater than or equal to
        // (LM+2) * 8".
        assert_eq!(ANTI_COLLAPSE_HEADROOM_LM_OFFSET, 2);
        assert_eq!(ANTI_COLLAPSE_HEADROOM_MULT_EIGHTH_BITS, 8);
    }

    // ---- Smallest-good-input sanity ----

    fn small_mono_inputs() -> (u32, u32, u32, CeltFrameSize, bool, bool, u32) {
        // 100-byte CELT-only 20 ms mono frame, no transient, no
        // outstanding tell or boost. CELT-only window is `0..21`, so
        // coded_bands = 21.
        (100, 0, 0, CeltFrameSize::Ms20, false, false, 21)
    }

    #[test]
    fn mono_nontransient_short_lm_yields_no_anti_collapse_no_stereo_rsv() {
        // 2.5 ms mono — anti-collapse never reserved (LM = 0);
        // stereo branches both off (mono).
        let outcome = reserve_block(
            100,
            0,
            0,
            CeltFrameSize::Ms2_5,
            true, // transient: irrelevant because LM = 0
            false,
            21,
        )
        .unwrap();
        assert_eq!(outcome.anti_collapse_rsv, 0);
        assert_eq!(outcome.skip_rsv, 8);
        assert_eq!(outcome.intensity_rsv, 0);
        assert_eq!(outcome.dual_stereo_rsv, 0);
    }

    #[test]
    fn mono_nontransient_long_lm_yields_no_anti_collapse() {
        let (bytes, tell, boost, lm, _, stereo, cb) = small_mono_inputs();
        let outcome = reserve_block(bytes, tell, boost, lm, false, stereo, cb).unwrap();
        // Non-transient ⇒ anti-collapse never reserved regardless of LM.
        assert_eq!(outcome.anti_collapse_rsv, 0);
        assert_eq!(outcome.skip_rsv, 8);
        assert_eq!(outcome.intensity_rsv, 0);
        assert_eq!(outcome.dual_stereo_rsv, 0);
    }

    #[test]
    fn mono_transient_lm0_yields_no_anti_collapse() {
        // LM = 0 (2.5 ms), transient — RFC explicitly excludes the
        // 2.5 / 5 ms frame sizes ("10 ms and 20 ms frames").
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms2_5, true, false, 21).unwrap();
        assert_eq!(outcome.anti_collapse_rsv, 0);
    }

    #[test]
    fn mono_transient_lm1_yields_no_anti_collapse() {
        // LM = 1 (5 ms) — same exclusion as LM = 0.
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms5, true, false, 21).unwrap();
        assert_eq!(outcome.anti_collapse_rsv, 0);
    }

    #[test]
    fn mono_transient_lm2_with_room_yields_anti_collapse() {
        // LM = 2 (10 ms), transient, plenty of budget.
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms10, true, false, 21).unwrap();
        assert_eq!(outcome.anti_collapse_rsv, 8);
        assert_eq!(outcome.skip_rsv, 8);
    }

    #[test]
    fn mono_transient_lm3_with_room_yields_anti_collapse() {
        // LM = 3 (20 ms), transient, plenty of budget.
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms20, true, false, 21).unwrap();
        assert_eq!(outcome.anti_collapse_rsv, 8);
    }

    #[test]
    fn anti_collapse_threshold_exact_match_passes() {
        // Construct a frame whose `total` after the -1 deduction is
        // exactly `(LM + 2) * 8`. LM = 2 ⇒ threshold = 32. With
        // tell = 0 and frame_eighth = ?, we want frame_eighth − 1 =
        // 32, so frame_eighth = 33. Use a custom non-byte-aligned tell
        // via ec_tell_frac = frame_eighth − 33.
        let frame_size_bytes = 1u32;
        let frame_eighth = frame_size_bytes * 64;
        let want_total = (2 + 2) * 8; // 32
        let ec_tell = frame_eighth - want_total - 1;
        let outcome = reserve_block(
            frame_size_bytes,
            ec_tell,
            0,
            CeltFrameSize::Ms10,
            true,
            false,
            21,
        )
        .unwrap();
        assert_eq!(outcome.anti_collapse_rsv, 8);
    }

    #[test]
    fn anti_collapse_threshold_one_short_fails() {
        // Same setup as above, but with one fewer 1/8 bit so the
        // §4.3.3 inequality `total ≥ (LM + 2) * 8` flips to false.
        let frame_size_bytes = 1u32;
        let frame_eighth = frame_size_bytes * 64;
        let want_total = (2 + 2) * 8; // 32
        let ec_tell = frame_eighth - want_total - 1 + 1;
        let outcome = reserve_block(
            frame_size_bytes,
            ec_tell,
            0,
            CeltFrameSize::Ms10,
            true,
            false,
            21,
        )
        .unwrap();
        assert_eq!(outcome.anti_collapse_rsv, 0);
    }

    #[test]
    fn skip_rsv_set_when_total_above_eight() {
        // 100-byte frame ⇒ frame_eighth = 6400 ⇒ total after -1 =
        // 6399. Plenty above 8, so skip_rsv = 8.
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms20, false, false, 21).unwrap();
        assert_eq!(outcome.skip_rsv, 8);
    }

    #[test]
    fn skip_rsv_threshold_strictly_greater_than_eight() {
        // §4.3.3: "skip_rsv is set to 8 (8th bits) if total is greater
        // than 8". Strictly greater — equality is false. Construct a
        // frame whose `total` after the -1 deduction (and no
        // anti-collapse) equals 8.
        let frame_size_bytes = 1u32;
        let frame_eighth = frame_size_bytes * 64;
        let want_total = 8u32;
        let ec_tell = frame_eighth - want_total - 1;
        let outcome = reserve_block(
            frame_size_bytes,
            ec_tell,
            0,
            CeltFrameSize::Ms5,
            false,
            false,
            21,
        )
        .unwrap();
        assert_eq!(outcome.skip_rsv, 0);
    }

    #[test]
    fn skip_rsv_threshold_one_above_eight() {
        // Same setup as above, but `total` = 9 ⇒ skip_rsv = 8.
        let frame_size_bytes = 1u32;
        let frame_eighth = frame_size_bytes * 64;
        let want_total = 9u32;
        let ec_tell = frame_eighth - want_total - 1;
        let outcome = reserve_block(
            frame_size_bytes,
            ec_tell,
            0,
            CeltFrameSize::Ms5,
            false,
            false,
            21,
        )
        .unwrap();
        assert_eq!(outcome.skip_rsv, 8);
    }

    #[test]
    fn anti_collapse_deducts_from_total_before_skip_gate() {
        // §4.3.3 ordering check: the anti-collapse deduction comes
        // first, so the skip-gate sees `total` *after* the
        // anti-collapse deduction has been applied. Construct a
        // frame whose `total` after the -1 deduction sits at exactly
        // `(LM + 2) * 8 + 8` for LM = 2, i.e. 40 1/8 bits. With
        // anti-collapse reserved that leaves 32 ≥ 8 ⇒ skip_rsv = 8;
        // without anti-collapse the leftover would still be 40 > 8
        // and skip_rsv would still be 8 — so check the post-skip
        // total to confirm anti-collapse really did deduct first.
        let frame_size_bytes = 1u32;
        let frame_eighth = frame_size_bytes * 64;
        // total_after_minus1 = (2 + 2) * 8 + 8 = 40 ⇒ ec_tell = 64 -
        // 40 - 1 = 23.
        let ec_tell = 23u32;
        let outcome = reserve_block(
            frame_size_bytes,
            ec_tell,
            0,
            CeltFrameSize::Ms10,
            true,
            false,
            21,
        )
        .unwrap();
        assert_eq!(outcome.anti_collapse_rsv, 8);
        assert_eq!(outcome.skip_rsv, 8);
        // 40 (post-minus1) − 8 (anti-collapse) − 8 (skip) = 24.
        assert_eq!(outcome.total_remaining_eighth_bits, 24);
        // Reconstructed invariant: frame_eighth − ec_tell − 1 =
        // 64 − 23 − 1 = 40 = reserved (16) + remaining (24).
        assert_eq!(frame_eighth - ec_tell - 1, 16 + 24);
    }

    // ---- Stereo branch ----

    #[test]
    fn stereo_with_budget_sets_intensity_and_dual() {
        // 100-byte stereo frame; coded_bands = 21 ⇒ LOG2_FRAC_TABLE
        // index 21 = 36 (Q3). Plenty of budget.
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms20, false, true, 21).unwrap();
        assert_eq!(outcome.intensity_rsv, 36);
        assert_eq!(outcome.dual_stereo_rsv, 8);
    }

    #[test]
    fn stereo_intensity_above_total_resets_to_zero() {
        // Construct a frame whose `total` is below the intensity_rsv
        // value at `coded_bands = 21` (= 36 in 1/8 bits). Pick
        // coded_bands = 21 (intensity = 36); set `total` = 30 by
        // exhausting most of the budget via ec_tell_frac.
        let frame_size_bytes = 1u32;
        // We want, after the -1 and the skip_rsv = 8 deduction,
        // total = 30. So total_after_minus1 - 8 = 30 ⇒
        // total_after_minus1 = 38 ⇒ ec_tell = 64 - 38 - 1 = 25.
        let ec_tell = 25u32;
        let outcome = reserve_block(
            frame_size_bytes,
            ec_tell,
            0,
            CeltFrameSize::Ms5,
            false,
            true,
            21,
        )
        .unwrap();
        assert_eq!(outcome.skip_rsv, 8);
        assert_eq!(outcome.intensity_rsv, 0);
        // §4.3.3: When intensity_rsv is reset to zero, dual_stereo_rsv
        // is not considered (the §4.3.3 narrative gates the dual-stereo
        // branch on intensity_rsv being deducted).
        assert_eq!(outcome.dual_stereo_rsv, 0);
        // total stays at 30 — no intensity deduction.
        assert_eq!(outcome.total_remaining_eighth_bits, 30);
    }

    #[test]
    fn stereo_intensity_consumes_total_no_dual_stereo() {
        // Construct a frame where intensity_rsv just fits but only
        // leaves <= 8 in total for the dual-stereo gate. coded_bands
        // = 21 ⇒ intensity = 36. Want total_after_skip = 36 + 7 = 43;
        // dual_stereo gate sees total = 7 ≤ 8 ⇒ dual_stereo_rsv = 0.
        let frame_size_bytes = 1u32;
        // total_after_minus1 - 8 = 43 ⇒ total_after_minus1 = 51 ⇒
        // ec_tell = 64 - 51 - 1 = 12.
        let ec_tell = 12u32;
        let outcome = reserve_block(
            frame_size_bytes,
            ec_tell,
            0,
            CeltFrameSize::Ms5,
            false,
            true,
            21,
        )
        .unwrap();
        assert_eq!(outcome.skip_rsv, 8);
        assert_eq!(outcome.intensity_rsv, 36);
        assert_eq!(outcome.dual_stereo_rsv, 0);
        assert_eq!(outcome.total_remaining_eighth_bits, 7);
    }

    #[test]
    fn stereo_intensity_consumes_total_dual_stereo_just_fits() {
        // Same as above but with total = 36 + 9 = 45 after skip;
        // dual_stereo gate sees total = 9 > 8 ⇒ dual_stereo_rsv = 8.
        let frame_size_bytes = 1u32;
        // total_after_minus1 - 8 = 45 ⇒ total_after_minus1 = 53 ⇒
        // ec_tell = 64 - 53 - 1 = 10.
        let ec_tell = 10u32;
        let outcome = reserve_block(
            frame_size_bytes,
            ec_tell,
            0,
            CeltFrameSize::Ms5,
            false,
            true,
            21,
        )
        .unwrap();
        assert_eq!(outcome.skip_rsv, 8);
        assert_eq!(outcome.intensity_rsv, 36);
        assert_eq!(outcome.dual_stereo_rsv, 8);
        assert_eq!(outcome.total_remaining_eighth_bits, 1);
    }

    #[test]
    fn mono_skips_stereo_branches_even_with_budget() {
        // Mono frame with plenty of budget — intensity_rsv and
        // dual_stereo_rsv must both stay at 0 regardless of
        // coded_bands.
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms20, false, false, 21).unwrap();
        assert_eq!(outcome.intensity_rsv, 0);
        assert_eq!(outcome.dual_stereo_rsv, 0);
    }

    #[test]
    fn hybrid_window_intensity_uses_log2_frac_table_at_four() {
        // Hybrid mode: §4.3 coding window is `17..21`, so coded_bands
        // = 4. LOG2_FRAC_TABLE[4] = 19 in Q3.
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms20, false, true, 4).unwrap();
        assert_eq!(outcome.intensity_rsv, 19);
        assert_eq!(outcome.dual_stereo_rsv, 8);
    }

    #[test]
    fn coded_bands_one_intensity_is_eight() {
        // Boundary: coded_bands = 1 ⇒ LOG2_FRAC_TABLE[1] = 8 = 1 whole bit.
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms20, false, true, 1).unwrap();
        assert_eq!(outcome.intensity_rsv, 8);
    }

    #[test]
    fn coded_bands_zero_intensity_is_zero() {
        // LOG2_FRAC_TABLE[0] = 0 ⇒ intensity_rsv = 0 (since the
        // §4.3.3 "greater than total" check fails to trip and the
        // reservation deducts zero). dual_stereo branch still runs.
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms20, false, true, 0).unwrap();
        assert_eq!(outcome.intensity_rsv, 0);
        // Total has room ⇒ dual_stereo_rsv still gets reserved.
        assert_eq!(outcome.dual_stereo_rsv, 8);
    }

    // ---- Total / invariant cross-checks ----

    #[test]
    fn reservation_invariant_holds_for_stereo_with_room() {
        // §4.3.3 invariant: total_remaining + reserved_total =
        // frame_eighth - ec_tell_frac - 1 (the conservative
        // deduction).
        let outcome = reserve_block(50, 0, 0, CeltFrameSize::Ms20, true, true, 21).unwrap();
        let frame_eighth: u32 = 50 * 64;
        let conservative_total = frame_eighth - 1;
        let reserved = outcome.reserved_total_eighth_bits();
        assert_eq!(
            outcome.total_remaining_eighth_bits + reserved,
            conservative_total
        );
    }

    #[test]
    fn reservation_invariant_holds_for_mono_with_no_anti_collapse() {
        let outcome = reserve_block(200, 0, 0, CeltFrameSize::Ms20, false, false, 21).unwrap();
        let frame_eighth: u32 = 200 * 64;
        let conservative_total = frame_eighth - 1;
        let reserved = outcome.reserved_total_eighth_bits();
        assert_eq!(
            outcome.total_remaining_eighth_bits + reserved,
            conservative_total
        );
    }

    #[test]
    fn reservation_invariant_holds_with_nonzero_tell() {
        // Mid-frame: range coder has consumed some 1/8 bits.
        let outcome = reserve_block(100, 137, 24, CeltFrameSize::Ms20, true, true, 21).unwrap();
        let frame_eighth = 100 * 64;
        let conservative_total = frame_eighth - 137 - 1;
        let reserved = outcome.reserved_total_eighth_bits();
        assert_eq!(
            outcome.total_remaining_eighth_bits + reserved,
            conservative_total
        );
    }

    #[test]
    fn reservation_invariant_holds_when_intensity_resets() {
        // Stereo, intensity reset to 0 ⇒ invariant still holds.
        let frame_size_bytes = 1u32;
        let ec_tell = 25u32;
        let outcome = reserve_block(
            frame_size_bytes,
            ec_tell,
            0,
            CeltFrameSize::Ms5,
            false,
            true,
            21,
        )
        .unwrap();
        assert_eq!(outcome.intensity_rsv, 0);
        let frame_eighth = frame_size_bytes * 64;
        let conservative_total = frame_eighth - ec_tell - 1;
        let reserved = outcome.reserved_total_eighth_bits();
        assert_eq!(
            outcome.total_remaining_eighth_bits + reserved,
            conservative_total
        );
    }

    // ---- Error paths ----

    #[test]
    fn frame_size_overflow_rejected() {
        // frame_size_bytes near u32::MAX / 64 ⇒ overflow.
        let err = reserve_block(u32::MAX, 0, 0, CeltFrameSize::Ms20, false, false, 21).unwrap_err();
        assert_eq!(err, ReservationError::FrameSizeOverflows);
    }

    #[test]
    fn tell_exceeds_frame_rejected() {
        let err =
            reserve_block(10, 10 * 64 + 1, 0, CeltFrameSize::Ms20, false, false, 21).unwrap_err();
        assert!(matches!(err, ReservationError::TellExceedsFrame { .. }));
    }

    #[test]
    fn total_boost_exceeds_frame_rejected() {
        let err =
            reserve_block(10, 0, 10 * 64 + 1, CeltFrameSize::Ms20, false, false, 21).unwrap_err();
        assert!(matches!(
            err,
            ReservationError::TotalBoostExceedsFrame { .. }
        ));
    }

    #[test]
    fn coded_bands_above_table_rejected() {
        // LOG2_FRAC_TABLE covers 0..24; ask for 24 ⇒
        // Log2FracError::CodedBandsOutOfRange.
        let err = reserve_block(100, 0, 0, CeltFrameSize::Ms20, false, true, 24).unwrap_err();
        assert!(matches!(
            err,
            ReservationError::LogFracLookupFailed(Log2FracError::CodedBandsOutOfRange { .. })
        ));
    }

    #[test]
    fn mono_with_oversize_coded_bands_does_not_lookup() {
        // Mono ⇒ the intensity branch is skipped entirely; the
        // out-of-range coded_bands input MUST NOT trip an error.
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms20, false, false, 999).unwrap();
        assert_eq!(outcome.intensity_rsv, 0);
        assert_eq!(outcome.dual_stereo_rsv, 0);
    }

    // ---- Edge cases ----

    #[test]
    fn zero_frame_yields_zero_reservations_and_zero_total() {
        let outcome = reserve_block(0, 0, 0, CeltFrameSize::Ms20, true, true, 21).unwrap();
        assert_eq!(outcome.anti_collapse_rsv, 0);
        assert_eq!(outcome.skip_rsv, 0);
        assert_eq!(outcome.intensity_rsv, 0);
        assert_eq!(outcome.dual_stereo_rsv, 0);
        assert_eq!(outcome.total_remaining_eighth_bits, 0);
    }

    #[test]
    fn frame_one_byte_minus_one_yields_no_reservations() {
        // 1-byte frame ⇒ frame_eighth = 64. After -1, total = 63.
        // anti_collapse skipped (LM = 0). skip_rsv = 8 (63 > 8).
        // After skip: total = 55. Plenty for intensity and dual
        // stereo too, when stereo.
        let outcome = reserve_block(1, 0, 0, CeltFrameSize::Ms2_5, false, false, 21).unwrap();
        assert_eq!(outcome.skip_rsv, 8);
        assert_eq!(outcome.total_remaining_eighth_bits, 55);
    }

    #[test]
    fn maximum_frame_yields_correct_remaining() {
        // §3.4 R5 max frame = 1275 bytes ⇒ 81600 1/8 bits. After -1
        // and skip = 8, total = 81591. anti_collapse + intensity +
        // dual_stereo all reserve at their max (8 + 37 + 8 = 53).
        // Final total = 81600 - 1 - 8 - 8 - 37 - 8 = 81538. Use
        // coded_bands = 23 (LOG2_FRAC_TABLE[23] = 37).
        let outcome = reserve_block(1275, 0, 0, CeltFrameSize::Ms20, true, true, 23).unwrap();
        assert_eq!(outcome.anti_collapse_rsv, 8);
        assert_eq!(outcome.skip_rsv, 8);
        assert_eq!(outcome.intensity_rsv, 37);
        assert_eq!(outcome.dual_stereo_rsv, 8);
        assert_eq!(outcome.total_remaining_eighth_bits, 81538);
    }

    #[test]
    fn outcome_default_is_all_zero() {
        // Useful when callers want to short-circuit a CELT-skipped
        // frame.
        let d = ReservationOutcome::default();
        assert_eq!(d.anti_collapse_rsv, 0);
        assert_eq!(d.skip_rsv, 0);
        assert_eq!(d.intensity_rsv, 0);
        assert_eq!(d.dual_stereo_rsv, 0);
        assert_eq!(d.total_remaining_eighth_bits, 0);
        assert_eq!(d.reserved_total_eighth_bits(), 0);
    }

    #[test]
    fn debug_format_renders() {
        // Smoke-test the derived Debug — useful for fixture mismatch
        // diagnostics.
        let outcome = reserve_block(100, 0, 0, CeltFrameSize::Ms20, true, true, 21).unwrap();
        let s = format!("{outcome:?}");
        assert!(s.contains("anti_collapse_rsv"));
        assert!(s.contains("intensity_rsv"));
    }

    #[test]
    fn determinism_across_repeats() {
        // Same inputs ⇒ same outputs.
        let a = reserve_block(137, 91, 24, CeltFrameSize::Ms20, true, true, 17).unwrap();
        let b = reserve_block(137, 91, 24, CeltFrameSize::Ms20, true, true, 17).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn from_log_frac_error_round_trip() {
        let inner = Log2FracError::CodedBandsOutOfRange { coded_bands: 100 };
        let outer: ReservationError = inner.into();
        assert_eq!(outer, ReservationError::LogFracLookupFailed(inner));
    }
}
