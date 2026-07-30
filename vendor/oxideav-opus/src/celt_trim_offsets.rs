//! CELT §4.3.3 per-band allocation-trim offsets (RFC 6716 §4.3.3,
//! p. 115).
//!
//! After round 35's per-band minimum vector
//! ([`crate::celt_band_thresh`]) computes the §4.3.3 hard lower
//! bound, the §4.3.3 procedure derives a per-band *trim-offset*
//! vector `trim_offsets[]` from the previously decoded
//! [`crate::celt_alloc_trim`] index. The trim offsets bias the §4.3.3
//! Table 57 static-allocation search: positive offsets steer the
//! search toward more bits for a band, negative offsets toward fewer.
//!
//! The §4.3.3 narrative (RFC 6716 §4.3.3, p. 115) reads:
//!
//! > The previously decoded allocation trim is used to derive a
//! > vector of per-band adjustments, 'trim_offsets[]'. For each
//! > coded band take the alloc_trim and subtract 5 and LM. Then,
//! > multiply the result by the number of channels, the number of
//! > MDCT bins in the shortest frame size for this mode, the number
//! > of remaining bands, 2**LM, and 8. Next, divide this value by
//! > 64. Finally, if the number of MDCT bins in the band per channel
//! > is only one, 8 times the number of channels is subtracted in
//! > order to diminish the allocation by one bit, because width 1
//! > bands receive greater benefit from the coarse energy coding.
//!
//! ## §4.3.3 formula
//!
//! For each coded band `b`, with `channels ∈ {1, 2}` the channel
//! count, `LM ∈ {0, 1, 2, 3}` the §4.3 frame-size scale (=
//! [`crate::celt_band_layout::CeltFrameSize::column_index`]),
//! `alloc_trim ∈ {0, …, 10}` the round-32 decoded trim index
//! ([`crate::celt_alloc_trim::ALLOC_TRIM_MIN`] /
//! [`crate::celt_alloc_trim::ALLOC_TRIM_MAX`]),
//! `n_shortest = celt_band_bins_per_channel(b, Ms2_5)` the per-band
//! MDCT bin count at the shortest §4.3 frame size,
//! `n_per_channel = celt_band_bins_per_channel(b, frame_size)` the
//! per-band MDCT bin count at the frame's actual frame size, and
//! `remaining_bands` the band-position-dependent "remaining bands"
//! factor (see ["What this module does not own"](#what-this-module-does-not-own)):
//!
//! ```text
//! base = (alloc_trim - 5 - LM)
//!      * channels
//!      * n_shortest
//!      * remaining_bands
//!      * (1 << LM)
//!      * 8
//!      / 64
//! trim_offsets[b] = base - (if n_per_channel == 1 { 8 * channels } else { 0 })
//! ```
//!
//! All arithmetic is signed: `alloc_trim - 5 - LM` ranges from
//! `(0 - 5 - 3) = -8` (lowest trim, largest frame size) to
//! `(10 - 5 - 0) = 5` (highest trim, shortest frame size). The output
//! is in 1/8 bits, the same units as every other §4.3.3 budget
//! quantity (compatible with the round-34
//! [`crate::celt_reservations`] output and the round-35
//! [`crate::celt_band_thresh`] floor at the consumer site).
//!
//! ## What this module does not own
//!
//! * Bitstream parsing. `trim_offsets[]` is a deterministic function
//!   of the round-32 `alloc_trim` (decoded once per frame), the §4.3
//!   band layout, the channel count, and the frame size. The
//!   range-coder reads for `alloc_trim` itself live in
//!   [`crate::celt_alloc_trim::decode_alloc_trim`].
//! * The "number of remaining bands" choice. The §4.3.3 RFC narrative
//!   phrases the factor as "the number of remaining bands" per coded
//!   band; the natural reading is a band-position-dependent quantity
//!   (= bands remaining to be processed at the current band's
//!   position in the §4.3.3 Table 57 search). Because the §4.3.3
//!   search is not yet wired up, this module accepts
//!   `remaining_bands` as an explicit caller-supplied parameter and
//!   defers the choice to the consumer site (the round that lands
//!   the Table 57 static-allocation search). Both readings of the
//!   spec phrasing fit through the same `band_trim_offset()`
//!   signature.
//! * The §4.3.3 Table 57 static-allocation search itself. That search
//!   consumes the working `total` budget (from round 34
//!   [`crate::celt_reservations::ReservationOutcome::total_remaining_eighth_bits`]),
//!   the per-band floor (round 35
//!   [`crate::celt_band_thresh::band_min_thresh`]), the per-band
//!   cap (the round-31
//!   [`crate::celt_cache_caps50::cap_for_band_bits`] surface), the
//!   per-band boosts (round 33
//!   [`crate::celt_band_boost::decode_band_boosts`]), and the
//!   `trim_offsets[]` vector this module produces. The search will
//!   converge on a quality index whose interpolated allocation fits
//!   the budget. The §4.3.3 narrative places `trim_offsets[]`
//!   immediately after the per-band minimum; the search is the
//!   following §4.3.3 paragraph.
//! * The "shortest frame size for this mode" choice. The standard
//!   §4.3 CELT mode covers all four frame sizes (`{2.5, 5, 10, 20}
//!   ms`), so the shortest is `Ms2_5`. The [`shortest_frame_size`]
//!   helper returns that constant; the convenience
//!   [`band_n_shortest`] helper looks up the per-band bin count at
//!   `Ms2_5` via [`crate::celt_band_layout::celt_band_bins_per_channel`].
//!
//! ## Units
//!
//! Every value emitted by this module is in 1/8 bits (the same units
//! the §4.3.3 budget loop works in). The output is `i32` (signed)
//! because the `(alloc_trim - 5 - LM)` factor can be negative; the
//! Table 57 search applies the offset additively to a per-band budget.
//!
//! ## Range
//!
//! With `alloc_trim ∈ {0, …, 10}`, `LM ∈ {0, 1, 2, 3}`,
//! `channels ∈ {1, 2}`, `n_shortest ∈ {1, …, 22}` (Table 55 column 0:
//! the §4.3 standard layer has 21 bands at 2.5 ms; the widest band 20
//! covers 22 MDCT bins / channel at 2.5 ms — see the round-24 band
//! layout pin), and `remaining_bands ≤ 21` (the §4.3 standard layer
//! has at most 21 coded bands per frame), the worst-case product is
//! roughly `|alloc_trim - 5 - LM| × channels × n_shortest ×
//! remaining_bands × (1 << LM) × 8 / 64 ≤ 8 × 2 × 22 × 21 × 8 × 8 /
//! 64 = 7392`, well within `i32` range. The width-1 subtraction
//! removes at most 16 from this. Every `trim_offsets[band]` fits in
//! `i32` by a wide margin.
//!
//! ## Provenance
//!
//! Narrative: RFC 6716 §4.3.3 (p. 115) in
//! `docs/audio/opus/rfc6716-opus.txt`; cross-referenced by
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.6
//! ("Minimums and trim offsets"). The six formula coefficients
//! (`5`, `8`, `64`, and the width-1 trigger value `1`) are inlined
//! in the RFC body — a separate numeric table is unnecessary. The
//! per-band MDCT bin counts come from round 24's
//! [`crate::celt_band_layout::celt_band_bins_per_channel`] (Table 55
//! lookup).

use crate::celt_alloc_trim::{ALLOC_TRIM_MAX, ALLOC_TRIM_MIN};
use crate::celt_band_layout::{celt_band_bins_per_channel, CeltFrameSize, CELT_NUM_BANDS};

/// §4.3.3 constant subtracted from `alloc_trim` in the trim-offsets
/// formula (RFC 6716 §4.3.3 p. 115: "take the alloc_trim and subtract
/// 5 and LM"). Equals the round-32
/// [`crate::celt_alloc_trim::ALLOC_TRIM_DEFAULT`] value: when
/// `alloc_trim == 5` and `LM == 0`, the multiplicative kernel is zero
/// and the trim term cancels out, leaving only the width-1
/// subtraction.
pub const TRIM_OFFSETS_BIAS: i32 = 5;

/// §4.3.3 multiplicative scale applied to the trim-offsets numerator
/// (RFC 6716 §4.3.3 p. 115: the "8" that follows the `2**LM` factor).
/// Combined with the `/64` divisor, the net per-formula scale is
/// `8 / 64 = 1/8`, which keeps the output in 1/8 bits (the same
/// `Q3`-style units as every other §4.3.3 budget value).
pub const TRIM_OFFSETS_NUMERATOR_SCALE: i32 = 8;

/// §4.3.3 divisor applied to the trim-offsets numerator (RFC 6716
/// §4.3.3 p. 115: "divide this value by 64"). The integer division
/// is exact in the spec's wording — it is a truncating divide on the
/// signed numerator, with the sign carried by `(alloc_trim - 5 - LM)`.
pub const TRIM_OFFSETS_DIVISOR: i32 = 64;

/// §4.3.3 trigger value for the width-1 correction (RFC 6716 §4.3.3
/// p. 115: "if the number of MDCT bins in the band per channel is
/// only one"). When the per-band-per-channel MDCT bin count equals
/// this, the formula subtracts [`TRIM_OFFSETS_WIDTH_ONE_PER_CHANNEL_EIGHTH_BITS`]
/// times the channel count.
pub const TRIM_OFFSETS_WIDTH_ONE_BINS_PER_CHANNEL: u16 = 1;

/// §4.3.3 width-1 subtraction per channel in 1/8 bits (RFC 6716
/// §4.3.3 p. 115: "8 times the number of channels is subtracted").
/// Equals one whole bit per channel — the same per-channel unit the
/// round-35 [`crate::celt_band_thresh::BAND_THRESH_PER_CHANNEL_EIGHTH_BITS`]
/// floor uses. The §4.3.3 rationale: "width 1 bands receive greater
/// benefit from the coarse energy coding", so the trim offsets back
/// off one whole bit per channel for them.
pub const TRIM_OFFSETS_WIDTH_ONE_PER_CHANNEL_EIGHTH_BITS: i32 = 8;

/// §4.3.3 mono channel multiplier (1 channel). Matches the round-35
/// [`crate::celt_band_thresh::BAND_THRESH_MONO_CHANNELS`] pin.
pub const TRIM_OFFSETS_MONO_CHANNELS: i32 = 1;

/// §4.3.3 stereo channel multiplier (2 channels). Matches the
/// round-35 [`crate::celt_band_thresh::BAND_THRESH_STEREO_CHANNELS`]
/// pin.
pub const TRIM_OFFSETS_STEREO_CHANNELS: i32 = 2;

/// Errors returned by [`band_trim_offset`] for inputs that violate
/// the §4.3 / §4.3.3 contract. These come from caller-side
/// bookkeeping bugs — the §4.3.3 trim-offsets formula itself is
/// total over the validated input domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimOffsetError {
    /// `alloc_trim` exceeds the round-32 trim signal range
    /// ([`ALLOC_TRIM_MIN`] = 0 .. [`ALLOC_TRIM_MAX`] = 10).
    AllocTrimOutOfRange {
        /// The provided trim value.
        provided: u8,
        /// The inclusive upper bound.
        max: u8,
    },
    /// `band >= CELT_NUM_BANDS` (= 21). The §4.3 standard CELT layer
    /// has exactly 21 bands; Custom mode is out of scope.
    BandOutOfRange {
        /// The provided band index.
        band: usize,
    },
}

/// §4.3.3 shortest frame size for the standard (non-Custom) CELT
/// mode. The §4.3 standard mode covers all four frame sizes
/// (`{2.5, 5, 10, 20} ms`), so the shortest is
/// [`CeltFrameSize::Ms2_5`].
///
/// The §4.3.3 trim-offsets formula reads "the number of MDCT bins in
/// the shortest frame size for this mode" — for the standard mode
/// this is the Table 55 column-0 value (= [`band_n_shortest`]).
pub const fn shortest_frame_size() -> CeltFrameSize {
    CeltFrameSize::Ms2_5
}

/// §4.3.3 helper: returns the per-band MDCT bin count at the
/// [`shortest_frame_size`] (= 2.5 ms column of Table 55).
///
/// `band` is the global band index `0..21` (Table 55 layout).
/// Returns `None` if `band >= 21`.
///
/// This is the `n_shortest` factor in the §4.3.3 trim-offsets
/// formula. The §4.3 standard layer has `n_shortest ∈ {1, …, 22}`
/// across the 21 bands.
pub fn band_n_shortest(band: usize) -> Option<u16> {
    celt_band_bins_per_channel(band, shortest_frame_size())
}

/// §4.3.3 per-band trim-offset for a single coded band (RFC 6716
/// §4.3.3 p. 115).
///
/// Inputs:
///
/// * `alloc_trim`: round-32 decoded trim signal,
///   `[ALLOC_TRIM_MIN, ALLOC_TRIM_MAX] = [0, 10]`. Range violations
///   are reported as
///   [`TrimOffsetError::AllocTrimOutOfRange`].
/// * `lm`: §4.3 frame-size scale, encoded as
///   [`CeltFrameSize`] (=
///   [`CeltFrameSize::column_index`] gives `LM ∈ {0, 1, 2, 3}` for
///   2.5/5/10/20 ms).
/// * `is_stereo`: channel-count selector. Mono ⇒ `channels = 1`;
///   stereo ⇒ `channels = 2`.
/// * `n_shortest`: per-band MDCT bins at the shortest frame size
///   ([`band_n_shortest`] = Table 55 column 0 for the standard
///   §4.3 CELT mode).
/// * `n_per_channel`: per-band MDCT bins at the frame's actual frame
///   size (= [`celt_band_bins_per_channel`] for the current
///   `frame_size`). Used only by the width-1 correction.
/// * `remaining_bands`: band-position-dependent "remaining bands"
///   factor. The RFC phrasing is "the number of remaining bands";
///   the consumer site (the §4.3.3 Table 57 static-allocation
///   search) chooses the per-band value at iteration. This module
///   accepts it verbatim.
///
/// Returns the §4.3.3 trim offset in 1/8 bits (signed `i32`).
///
/// ## Formula
///
/// ```text
/// base = (alloc_trim - 5 - LM) * channels * n_shortest *
///        remaining_bands * (1 << LM) * 8 / 64
/// trim_offsets[b] = base - (if n_per_channel == 1
///                          { 8 * channels } else { 0 })
/// ```
pub fn band_trim_offset(
    alloc_trim: u8,
    lm: CeltFrameSize,
    is_stereo: bool,
    n_shortest: u16,
    n_per_channel: u16,
    remaining_bands: u32,
) -> Result<i32, TrimOffsetError> {
    if alloc_trim > ALLOC_TRIM_MAX {
        return Err(TrimOffsetError::AllocTrimOutOfRange {
            provided: alloc_trim,
            max: ALLOC_TRIM_MAX,
        });
    }
    // ALLOC_TRIM_MIN == 0 == u8::MIN ⇒ the lower bound is
    // structural; every u8 already satisfies `alloc_trim >=
    // ALLOC_TRIM_MIN`. Documented here in a const-eval cross-check
    // rather than a runtime assert (clippy flags the trivially-true
    // comparison).
    const _: () = assert!(ALLOC_TRIM_MIN == 0);

    let lm_idx = lm.column_index() as i32;
    let channels = if is_stereo {
        TRIM_OFFSETS_STEREO_CHANNELS
    } else {
        TRIM_OFFSETS_MONO_CHANNELS
    };

    // (alloc_trim - 5 - LM): can be negative (e.g. alloc_trim=0,
    // LM=3 ⇒ -8). Compute in i32 to preserve sign.
    let trim_term = (alloc_trim as i32) - TRIM_OFFSETS_BIAS - lm_idx;

    // numerator = trim_term * channels * n_shortest * remaining_bands * (1 << LM) * 8
    // Range cross-check (see module-level Range section): the
    // worst-case absolute product is ≤ 8 × 2 × 22 × 21 × 8 × 8 =
    // 473_088, well within i32. Promote each factor to i32 before
    // multiplying.
    let two_pow_lm = 1i32 << lm_idx;
    let numerator = trim_term
        * channels
        * (n_shortest as i32)
        * (remaining_bands as i32)
        * two_pow_lm
        * TRIM_OFFSETS_NUMERATOR_SCALE;
    let base = numerator / TRIM_OFFSETS_DIVISOR;

    // §4.3.3 width-1 correction.
    let offset = if n_per_channel == TRIM_OFFSETS_WIDTH_ONE_BINS_PER_CHANNEL {
        base - TRIM_OFFSETS_WIDTH_ONE_PER_CHANNEL_EIGHTH_BITS * channels
    } else {
        base
    };

    Ok(offset)
}

/// §4.3.3 convenience: derive `n_shortest` and `n_per_channel` for
/// `band` from the Table 55 layout, then call [`band_trim_offset`].
///
/// `band` is the global band index `0..21` (Table 55 layout).
/// `frame_size` is the §4.3 CELT frame size (= LM column).
///
/// Returns [`TrimOffsetError::BandOutOfRange`] if `band >=
/// CELT_NUM_BANDS` (= 21).
pub fn band_trim_offset_for_band(
    band: usize,
    alloc_trim: u8,
    frame_size: CeltFrameSize,
    is_stereo: bool,
    remaining_bands: u32,
) -> Result<i32, TrimOffsetError> {
    if band >= CELT_NUM_BANDS {
        return Err(TrimOffsetError::BandOutOfRange { band });
    }
    let n_shortest = band_n_shortest(band).expect("§4.3 band < CELT_NUM_BANDS by guard above");
    let n_per_channel =
        celt_band_bins_per_channel(band, frame_size).expect("§4.3 band < CELT_NUM_BANDS");
    band_trim_offset(
        alloc_trim,
        frame_size,
        is_stereo,
        n_shortest,
        n_per_channel,
        remaining_bands,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_alloc_trim::ALLOC_TRIM_DEFAULT;

    // ---- Constant pins (each tied to a §4.3.3 RFC narrative phrase) ----

    #[test]
    fn bias_matches_rfc_subtract_5() {
        // RFC 6716 §4.3.3 p. 115: "take the alloc_trim and subtract 5".
        assert_eq!(TRIM_OFFSETS_BIAS, 5);
    }

    #[test]
    fn bias_equals_alloc_trim_default() {
        // The round-32 ALLOC_TRIM_DEFAULT is 5 — the same constant
        // the §4.3.3 trim-offsets formula subtracts. When the trim
        // signal lands at its default, the multiplicative kernel of
        // the formula is zero at LM = 0.
        assert_eq!(TRIM_OFFSETS_BIAS as u8, ALLOC_TRIM_DEFAULT);
    }

    #[test]
    fn numerator_scale_matches_rfc_8() {
        // RFC 6716 §4.3.3 p. 115: the multiplicative `8` that
        // follows the `2**LM` factor.
        assert_eq!(TRIM_OFFSETS_NUMERATOR_SCALE, 8);
    }

    #[test]
    fn divisor_matches_rfc_64() {
        // RFC 6716 §4.3.3 p. 115: "divide this value by 64".
        assert_eq!(TRIM_OFFSETS_DIVISOR, 64);
    }

    #[test]
    fn net_scale_keeps_eighth_bit_units() {
        // RFC's 8 / 64 = 1/8 net scale ⇒ output stays in 1/8 bits,
        // the same units the §4.3.3 budget loop works in.
        assert_eq!(
            TRIM_OFFSETS_NUMERATOR_SCALE * 8,
            TRIM_OFFSETS_DIVISOR,
            "8/64 = 1/8 keeps Q3 units",
        );
    }

    #[test]
    fn width_one_trigger_matches_rfc_one_bin() {
        // RFC 6716 §4.3.3 p. 115: "if the number of MDCT bins in the
        // band per channel is only one".
        assert_eq!(TRIM_OFFSETS_WIDTH_ONE_BINS_PER_CHANNEL, 1);
    }

    #[test]
    fn width_one_per_channel_subtraction_is_one_whole_bit() {
        // RFC 6716 §4.3.3 p. 115: "8 times the number of channels is
        // subtracted [...] to diminish the allocation by one bit".
        // 8 1/8 bits = one whole bit.
        assert_eq!(TRIM_OFFSETS_WIDTH_ONE_PER_CHANNEL_EIGHTH_BITS, 8);
        assert_eq!(TRIM_OFFSETS_WIDTH_ONE_PER_CHANNEL_EIGHTH_BITS / 8, 1);
    }

    #[test]
    fn channel_multipliers_match_audio_layout() {
        assert_eq!(TRIM_OFFSETS_MONO_CHANNELS, 1);
        assert_eq!(TRIM_OFFSETS_STEREO_CHANNELS, 2);
    }

    #[test]
    fn shortest_frame_size_is_2_5_ms() {
        // §4.3 standard mode covers {2.5, 5, 10, 20} ms ⇒ shortest =
        // 2.5 ms ⇒ Table 55 column 0.
        assert_eq!(shortest_frame_size(), CeltFrameSize::Ms2_5);
        assert_eq!(shortest_frame_size().column_index(), 0);
    }

    // ---- band_n_shortest: Table 55 column-0 lookup ----

    #[test]
    fn n_shortest_band0_is_one() {
        // Table 55, band 0, 2.5 ms: N = 1.
        assert_eq!(band_n_shortest(0), Some(1));
    }

    #[test]
    fn n_shortest_band20_is_twenty_two() {
        // Table 55, band 20, 2.5 ms: N = 22 (the largest band 0
        // column value — pinned by the round-24 band_layout tests).
        assert_eq!(band_n_shortest(20), Some(22));
    }

    #[test]
    fn n_shortest_band21_returns_none() {
        // §4.3 standard layer has exactly 21 bands; band 21 has no
        // §4.3 entry.
        assert_eq!(band_n_shortest(21), None);
        assert_eq!(band_n_shortest(100), None);
    }

    #[test]
    fn n_shortest_matches_table55_column0_for_every_band() {
        for band in 0..CELT_NUM_BANDS {
            let want = celt_band_bins_per_channel(band, CeltFrameSize::Ms2_5).unwrap();
            assert_eq!(band_n_shortest(band).unwrap(), want, "band={band}");
        }
    }

    // ---- band_trim_offset: §4.3.3 single-band formula ----

    #[test]
    fn default_trim_lm0_no_width1_yields_zero() {
        // alloc_trim = 5, LM = 0 ⇒ trim_term = 5 - 5 - 0 = 0 ⇒ base
        // = 0. n_per_channel = 2 ⇒ no width-1 subtraction.
        // Result = 0.
        let v = band_trim_offset(
            5,
            CeltFrameSize::Ms2_5,
            /* is_stereo */ false,
            /* n_shortest */ 1,
            /* n_per_channel */ 2,
            /* remaining_bands */ 21,
        )
        .unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn default_trim_lm0_width1_mono_subtracts_eight() {
        // alloc_trim = 5, LM = 0 ⇒ base = 0. n_per_channel = 1 +
        // mono ⇒ subtract 8 × 1 = 8.
        let v = band_trim_offset(5, CeltFrameSize::Ms2_5, false, 1, 1, 21).unwrap();
        assert_eq!(v, -8);
    }

    #[test]
    fn default_trim_lm0_width1_stereo_subtracts_sixteen() {
        // alloc_trim = 5, LM = 0 ⇒ base = 0. n_per_channel = 1 +
        // stereo ⇒ subtract 8 × 2 = 16.
        let v = band_trim_offset(5, CeltFrameSize::Ms2_5, true, 1, 1, 21).unwrap();
        assert_eq!(v, -16);
    }

    #[test]
    fn high_trim_lm0_positive_base_mono() {
        // alloc_trim = 10, LM = 0 ⇒ trim_term = 10 - 5 - 0 = 5.
        // channels = 1, n_shortest = 1, remaining_bands = 1,
        // 1 << 0 = 1, scale = 8, divisor = 64.
        // numerator = 5 * 1 * 1 * 1 * 1 * 8 = 40.
        // base = 40 / 64 = 0 (truncating).
        // n_per_channel = 2 ⇒ no subtraction.
        let v = band_trim_offset(10, CeltFrameSize::Ms2_5, false, 1, 2, 1).unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn high_trim_lm0_positive_base_large_factors() {
        // alloc_trim = 10, LM = 0, stereo, n_shortest = 22 (band 20
        // column 0), remaining_bands = 21 (full coded layer):
        // trim_term = 5. channels = 2.
        // numerator = 5 * 2 * 22 * 21 * 1 * 8 = 36_960.
        // base = 36_960 / 64 = 577 (truncating: 36_960 = 577*64 +
        // 32).
        // n_per_channel = 22 ⇒ no subtraction.
        let v = band_trim_offset(10, CeltFrameSize::Ms2_5, true, 22, 22, 21).unwrap();
        assert_eq!(v, 577);
    }

    #[test]
    fn low_trim_negative_base_mono() {
        // alloc_trim = 0, LM = 3 ⇒ trim_term = 0 - 5 - 3 = -8.
        // channels = 1, n_shortest = 22, remaining_bands = 21,
        // 1 << 3 = 8, scale = 8, divisor = 64.
        // numerator = -8 * 1 * 22 * 21 * 8 * 8 = -236_544.
        // base = -236_544 / 64 = -3_696.
        // n_per_channel > 1 ⇒ no subtraction.
        let v = band_trim_offset(0, CeltFrameSize::Ms20, false, 22, 22, 21).unwrap();
        assert_eq!(v, -3_696);
    }

    #[test]
    fn low_trim_negative_base_stereo_width1() {
        // alloc_trim = 0, LM = 3, stereo, n_shortest = 1
        // (band 0 column 0), remaining_bands = 21,
        // 1 << 3 = 8, scale = 8, divisor = 64.
        // numerator = -8 * 2 * 1 * 21 * 8 * 8 = -21_504.
        // base = -21_504 / 64 = -336.
        // n_per_channel = 1 ⇒ subtract 8 * 2 = 16.
        // Result = -336 - 16 = -352.
        let v = band_trim_offset(0, CeltFrameSize::Ms20, true, 1, 1, 21).unwrap();
        assert_eq!(v, -352);
    }

    #[test]
    fn lm_factor_doubles_with_each_increment() {
        // Fix everything except LM. trim_term shifts with LM, and
        // (1 << LM) doubles. Easier: keep alloc_trim such that the
        // trim_term sign is fixed (e.g. alloc_trim = 10), pick
        // n_shortest = 16, remaining_bands = 4 so the numerator
        // divides evenly.
        // For LM = 0: trim_term = 10 - 5 - 0 = 5.
        //   numerator = 5 * 1 * 16 * 4 * 1 * 8 = 2_560.
        //   base = 40.
        // For LM = 1: trim_term = 10 - 5 - 1 = 4.
        //   numerator = 4 * 1 * 16 * 4 * 2 * 8 = 4_096.
        //   base = 64.
        // For LM = 2: trim_term = 10 - 5 - 2 = 3.
        //   numerator = 3 * 1 * 16 * 4 * 4 * 8 = 6_144.
        //   base = 96.
        // For LM = 3: trim_term = 10 - 5 - 3 = 2.
        //   numerator = 2 * 1 * 16 * 4 * 8 * 8 = 8_192.
        //   base = 128.
        let v0 = band_trim_offset(10, CeltFrameSize::Ms2_5, false, 16, 16, 4).unwrap();
        let v1 = band_trim_offset(10, CeltFrameSize::Ms5, false, 16, 16, 4).unwrap();
        let v2 = band_trim_offset(10, CeltFrameSize::Ms10, false, 16, 16, 4).unwrap();
        let v3 = band_trim_offset(10, CeltFrameSize::Ms20, false, 16, 16, 4).unwrap();
        assert_eq!(v0, 40);
        assert_eq!(v1, 64);
        assert_eq!(v2, 96);
        assert_eq!(v3, 128);
    }

    #[test]
    fn channel_factor_scales_linearly_when_no_width1() {
        // Without the width-1 correction, mono → stereo doubles
        // `channels` ⇒ doubles the formula result.
        let mono = band_trim_offset(10, CeltFrameSize::Ms2_5, false, 16, 16, 4).unwrap();
        let stereo = band_trim_offset(10, CeltFrameSize::Ms2_5, true, 16, 16, 4).unwrap();
        assert_eq!(stereo, 2 * mono);
    }

    #[test]
    fn n_shortest_factor_scales_linearly() {
        // numerator is linear in n_shortest; result scales linearly
        // when the integer-division truncation lines up.
        let v_n16 = band_trim_offset(10, CeltFrameSize::Ms20, false, 16, 32, 1).unwrap();
        let v_n8 = band_trim_offset(10, CeltFrameSize::Ms20, false, 8, 32, 1).unwrap();
        // v_n16 numerator = 2 * 1 * 16 * 1 * 8 * 8 = 2_048; base = 32.
        // v_n8 numerator = 2 * 1 * 8 * 1 * 8 * 8 = 1_024; base = 16.
        assert_eq!(v_n16, 32);
        assert_eq!(v_n8, 16);
        assert_eq!(v_n16, 2 * v_n8);
    }

    #[test]
    fn remaining_bands_factor_scales_linearly() {
        let v_r4 = band_trim_offset(10, CeltFrameSize::Ms20, false, 16, 32, 4).unwrap();
        let v_r2 = band_trim_offset(10, CeltFrameSize::Ms20, false, 16, 32, 2).unwrap();
        // v_r4 numerator = 2 * 1 * 16 * 4 * 8 * 8 = 8_192; base = 128.
        // v_r2 numerator = 2 * 1 * 16 * 2 * 8 * 8 = 4_096; base = 64.
        assert_eq!(v_r4, 128);
        assert_eq!(v_r2, 64);
        assert_eq!(v_r4, 2 * v_r2);
    }

    #[test]
    fn zero_remaining_bands_yields_just_width_correction() {
        // With remaining_bands = 0, numerator = 0 ⇒ base = 0. The
        // width-1 correction (if applicable) is the only nonzero
        // term.
        let v_no_w1 = band_trim_offset(10, CeltFrameSize::Ms2_5, true, 22, 22, 0).unwrap();
        assert_eq!(v_no_w1, 0);

        let v_w1 = band_trim_offset(10, CeltFrameSize::Ms2_5, true, 22, 1, 0).unwrap();
        assert_eq!(v_w1, -16);
    }

    #[test]
    fn trim_minus_lm_zero_zeros_kernel() {
        // alloc_trim = 5, LM = 0 ⇒ trim_term = 0 ⇒ kernel cancels.
        // Test a range of factors — they should all give 0 (no
        // width-1 correction).
        for n_shortest in [1u16, 4, 16, 22] {
            for remaining in [1u32, 4, 21] {
                for stereo in [false, true] {
                    let v = band_trim_offset(
                        5,
                        CeltFrameSize::Ms2_5,
                        stereo,
                        n_shortest,
                        n_shortest, // not width 1 (≥ 1, but skip width-1 case)
                        remaining,
                    )
                    .unwrap();
                    if n_shortest == 1 {
                        let expected = if stereo { -16 } else { -8 };
                        assert_eq!(v, expected, "n_shortest=1 width-1 active");
                    } else {
                        assert_eq!(
                            v, 0,
                            "n_shortest={n_shortest} remaining={remaining} stereo={stereo}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lm_eq_trim_minus_5_kernel_collapses_at_high_trims() {
        // When alloc_trim - 5 == LM, trim_term = 0 ⇒ kernel = 0.
        // E.g. alloc_trim = 8, LM = 3 ⇒ trim_term = 0.
        let v = band_trim_offset(8, CeltFrameSize::Ms20, true, 16, 16, 8).unwrap();
        assert_eq!(v, 0);
    }

    // ---- Width-1 correction interaction ----

    #[test]
    fn width_one_subtraction_is_additive_to_base() {
        // Width-1 subtraction is purely additive: the base term
        // computed with n_per_channel ≥ 2 and again with
        // n_per_channel = 1 differ by exactly 8 * channels.
        let base_no_w1 = band_trim_offset(10, CeltFrameSize::Ms20, true, 8, 8, 4).unwrap();
        let with_w1 = band_trim_offset(10, CeltFrameSize::Ms20, true, 8, 1, 4).unwrap();
        assert_eq!(with_w1, base_no_w1 - 16);
    }

    #[test]
    fn width_one_only_triggers_at_per_channel_eq_one() {
        // The §4.3.3 trigger is `n_per_channel == 1`, not anything
        // else. Verify both inclusion and exclusion edges.
        for n_per_channel in [0u16, 2, 3, 22, 176] {
            let v = band_trim_offset(10, CeltFrameSize::Ms20, true, 8, n_per_channel, 4).unwrap();
            // base (with the same other factors): numerator =
            // 2 * 2 * 8 * 4 * 8 * 8 = 8_192; base = 128.
            assert_eq!(v, 128, "n_per_channel={n_per_channel} should not trigger");
        }
        // n_per_channel = 1 ⇒ subtract 16.
        let triggered = band_trim_offset(10, CeltFrameSize::Ms20, true, 8, 1, 4).unwrap();
        assert_eq!(triggered, 128 - 16);
    }

    // ---- Truncating-division behaviour (toward-zero) ----

    #[test]
    fn negative_division_truncates_toward_zero() {
        // Rust's i32 `/` is toward-zero (= C's truncating div),
        // which matches the §4.3.3 integer-division convention.
        //
        // alloc_trim = 0, LM = 3 ⇒ trim_term = -8.
        // channels = 1, n_shortest = 1, remaining = 1,
        // (1 << 3) = 8, scale = 8.
        // numerator = -8 * 1 * 1 * 1 * 8 * 8 = -512.
        // base = -512 / 64 = -8 (exact).
        let v_exact = band_trim_offset(0, CeltFrameSize::Ms20, false, 1, 2, 1).unwrap();
        assert_eq!(v_exact, -8);

        // Truncating-toward-zero behaviour: alloc_trim = 4, LM = 0
        // ⇒ trim_term = -1. n_shortest = 1, remaining = 1,
        // (1 << 0) = 1, scale = 8.
        // numerator = -1 * 1 * 1 * 1 * 1 * 8 = -8.
        // base = -8 / 64 = 0 (toward zero, since |-8| < 64).
        let v_small = band_trim_offset(4, CeltFrameSize::Ms2_5, false, 1, 2, 1).unwrap();
        assert_eq!(v_small, 0);

        // A case with |numerator| > 64 but not a multiple of 64:
        // alloc_trim = 3, LM = 0 ⇒ trim_term = -2. n_shortest = 5,
        // remaining = 1, (1 << 0) = 1, scale = 8.
        // numerator = -2 * 1 * 5 * 1 * 1 * 8 = -80.
        // base = -80 / 64 = -1 (toward zero; -2 would be away).
        let v_truncating = band_trim_offset(3, CeltFrameSize::Ms2_5, false, 5, 2, 1).unwrap();
        assert_eq!(v_truncating, -1);
    }

    // ---- Error paths ----

    #[test]
    fn alloc_trim_above_max_rejected() {
        let err = band_trim_offset(11, CeltFrameSize::Ms20, false, 16, 16, 4).unwrap_err();
        assert_eq!(
            err,
            TrimOffsetError::AllocTrimOutOfRange {
                provided: 11,
                max: ALLOC_TRIM_MAX,
            }
        );
    }

    #[test]
    fn alloc_trim_far_above_max_rejected() {
        let err = band_trim_offset(255, CeltFrameSize::Ms20, false, 16, 16, 4).unwrap_err();
        assert_eq!(
            err,
            TrimOffsetError::AllocTrimOutOfRange {
                provided: 255,
                max: ALLOC_TRIM_MAX,
            }
        );
    }

    #[test]
    fn alloc_trim_zero_accepted() {
        // ALLOC_TRIM_MIN = 0 ⇒ alloc_trim = 0 is the lower edge.
        band_trim_offset(0, CeltFrameSize::Ms20, false, 16, 16, 4).unwrap();
    }

    #[test]
    fn alloc_trim_max_accepted() {
        band_trim_offset(ALLOC_TRIM_MAX, CeltFrameSize::Ms20, false, 16, 16, 4).unwrap();
    }

    // ---- band_trim_offset_for_band: Table 55 wrapper ----

    #[test]
    fn for_band_rejects_band_at_or_above_celt_num_bands() {
        let err = band_trim_offset_for_band(21, 5, CeltFrameSize::Ms20, false, 21).unwrap_err();
        assert_eq!(err, TrimOffsetError::BandOutOfRange { band: 21 });

        let err = band_trim_offset_for_band(100, 5, CeltFrameSize::Ms20, false, 21).unwrap_err();
        assert_eq!(err, TrimOffsetError::BandOutOfRange { band: 100 });
    }

    #[test]
    fn for_band_matches_explicit_inputs() {
        // Verify the wrapper calls the primitive with the right
        // Table 55 column values.
        for band in 0..CELT_NUM_BANDS {
            for frame_size in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                for is_stereo in [false, true] {
                    let n_shortest = band_n_shortest(band).unwrap();
                    let n_per_channel = celt_band_bins_per_channel(band, frame_size).unwrap();
                    let want =
                        band_trim_offset(5, frame_size, is_stereo, n_shortest, n_per_channel, 7)
                            .unwrap();
                    let got = band_trim_offset_for_band(band, 5, frame_size, is_stereo, 7).unwrap();
                    assert_eq!(
                        got, want,
                        "band={band} frame_size={frame_size:?} stereo={is_stereo}"
                    );
                }
            }
        }
    }

    #[test]
    fn for_band_propagates_alloc_trim_error() {
        let err = band_trim_offset_for_band(0, 11, CeltFrameSize::Ms20, false, 21).unwrap_err();
        assert_eq!(
            err,
            TrimOffsetError::AllocTrimOutOfRange {
                provided: 11,
                max: ALLOC_TRIM_MAX,
            }
        );
    }

    #[test]
    fn for_band_width1_triggers_when_table55_says_so() {
        // Table 55, band 0, 2.5 ms: N = 1 ⇒ width-1 active.
        // Use the kernel-cancelling alloc_trim = 5 + LM = 5 ⇒
        // result = -8 (mono) or -16 (stereo).
        let mono = band_trim_offset_for_band(0, 5, CeltFrameSize::Ms2_5, false, 21).unwrap();
        assert_eq!(mono, -8);
        let stereo = band_trim_offset_for_band(0, 5, CeltFrameSize::Ms2_5, true, 21).unwrap();
        assert_eq!(stereo, -16);
    }

    #[test]
    fn for_band_no_width1_at_higher_bands() {
        // Table 55, band 20, 20 ms: N = 176 ≠ 1 ⇒ width-1 inactive.
        // alloc_trim = 5, LM = 3 ⇒ trim_term = -3.
        // n_shortest = 22 (band 20 col 0). channels = 1.
        // remaining_bands = 21. 1 << 3 = 8.
        // numerator = -3 * 1 * 22 * 21 * 8 * 8 = -88_704.
        // base = -88_704 / 64 = -1_386. width-1 inactive.
        let v = band_trim_offset_for_band(20, 5, CeltFrameSize::Ms20, false, 21).unwrap();
        assert_eq!(v, -1_386);
    }

    // ---- Cross-cutting determinism / invariants ----

    #[test]
    fn determinism_across_repeats() {
        for trim in [0u8, 3, 5, 7, 10] {
            for fs in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                for band in 0..CELT_NUM_BANDS {
                    for stereo in [false, true] {
                        for remaining in [0u32, 1, 4, 21] {
                            let a = band_trim_offset_for_band(band, trim, fs, stereo, remaining)
                                .unwrap();
                            let b = band_trim_offset_for_band(band, trim, fs, stereo, remaining)
                                .unwrap();
                            assert_eq!(a, b);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn output_fits_well_within_i32_for_full_legal_input_space() {
        // Worst-case combination per the Range section: alloc_trim
        // at edges, every frame size, every band, both stereo
        // settings, remaining_bands ∈ {0, …, 21}. Result must fit
        // in i32 (the type guarantees it, but verify no overflow
        // panics).
        for trim in [0u8, ALLOC_TRIM_MAX] {
            for fs in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                for band in 0..CELT_NUM_BANDS {
                    for stereo in [false, true] {
                        for remaining in [0u32, 21] {
                            let v = band_trim_offset_for_band(band, trim, fs, stereo, remaining)
                                .unwrap();
                            // Sanity: the Range section's worst-case
                            // estimate is ~7_392 in absolute value.
                            // Allow significant slack.
                            assert!(v.abs() < 20_000, "v={v} band={band}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn debug_format_renders() {
        let err = TrimOffsetError::AllocTrimOutOfRange {
            provided: 99,
            max: 10,
        };
        let s = format!("{err:?}");
        assert!(s.contains("AllocTrimOutOfRange"));

        let err = TrimOffsetError::BandOutOfRange { band: 22 };
        let s = format!("{err:?}");
        assert!(s.contains("BandOutOfRange"));
    }
}
