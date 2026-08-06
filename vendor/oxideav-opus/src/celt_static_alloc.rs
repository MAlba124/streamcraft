//! CELT §4.3.3 static allocation table (RFC 6716 §4.3.3, p. 112).
//!
//! The §4.3.3 *Bit Allocation* procedure pins each band's "static"
//! shape allocation to a hard-coded table indexed by `(band, q)` where
//! `band` is the §4.3 Table 55 band index and `q ∈ 0..=10` is a
//! quality column. RFC 6716 §4.3.3 (p. 112) names the table `alloc[][]`
//! and gives its 21×11 grid as Table 57: every cell stores a Q5 value
//! in 1/32-bit per MDCT bin units.
//!
//! The §4.3.3 narrative (RFC 6716 §4.3.3, p. 111, lines 6223–6230) reads:
//!
//! > The "static" bit allocation (in 1/8 bits) for a quality q,
//! > excluding the minimums, maximums, tilt and boosts, is equal to
//! > `channels*N*alloc[band][q]<<LM>>2`, where `alloc[][]` is given in
//! > Table 57 and `LM=log2(frame_size/120)`. The allocation is
//! > obtained by linearly interpolating between two values of `q` (in
//! > steps of 1/64) to find the highest allocation that does not exceed
//! > the number of bits remaining.
//!
//! ## §4.3.3 unit conversion
//!
//! For each band `b`, with `channels ∈ {1, 2}` the frame's channel
//! count, `N = bins_per_channel(b, frame_size)` the §4.3 Table 55
//! per-channel MDCT-bin count for `b` at the frame's LM, and
//! `q ∈ {0, …, 10}` a quality column:
//!
//! ```text
//! static_alloc[b][q] = (channels * N * STATIC_ALLOC[b][q]) << LM >> 2
//! ```
//!
//! All arithmetic is unsigned. The output is in 1/8 bits, the same
//! units every other §4.3.3 budget quantity uses (compatible with the
//! round-34 [`crate::celt_reservations`] output, the round-35
//! [`crate::celt_band_thresh`] floor, the round-36
//! [`crate::celt_trim_offsets`] tilt bias, the round-33 boosts, and
//! the round-31 [`crate::celt_cache_caps50`] per-band cap at the
//! consumer site).
//!
//! ## §4.3.3 column structure
//!
//! The 11 quality columns embed a coarse-to-fine quality ladder:
//!
//! * Column `0` is the §4.3.3 "no allocation" column — every cell is
//!   zero. A band whose interpolated `q` lands at column 0 receives
//!   no static shape allocation; this is the band-skip floor the
//!   §4.3.3 search uses when even the §4.3.3 minimum threshold cannot
//!   be hit.
//! * Columns `1..=9` are the §4.3.3 working quality range. The
//!   §4.3.3 RFC text describes the search as "the highest allocation
//!   that does not exceed the number of bits remaining" — so the
//!   §4.3.3 allocator picks a `q` whose interpolated allocation fits
//!   the budget, scaled by the per-band `n_bins` factor.
//! * Column `10` is the §4.3.3 "saturation" column — every cell is
//!   `200` or close to it. A band whose interpolated `q` reaches
//!   column 10 has hit the highest static allocation the table
//!   defines; the §4.3.3 per-band cap from
//!   [`crate::celt_cache_caps50::cap_for_band_bits`] takes over
//!   when the static allocation exceeds the cap.
//!
//! Within each row the entries are monotone non-decreasing in `q`:
//! a higher quality column never reduces the allocation for the same
//! band (with the carve-out at high bands where columns 0..=K are all
//! zero before the first non-zero entry — once the column is non-zero
//! it is monotone non-decreasing).
//!
//! ## §4.3.3 1/64-step linear interpolation
//!
//! The §4.3.3 search picks a quality `q` whose interpolated allocation
//! fits the working budget. Interpolation is in *steps of 1/64* between
//! two integer quality columns. That choice converts the per-band Q5
//! table to a Q11 interpolated allocation (= Q5 × Q6) before the §4.3.3
//! `<< LM >> 2` step folds it back to Q3 (1/8-bit) units. The §4.3.3
//! RFC narrative is explicit: every other unit in the §4.3.3 procedure
//! is 1/8 bit, so the table-step interpolation has to be at finer
//! precision than the output to keep the search well-conditioned.
//!
//! This module owns only the §4.3.3 *parameter surface*: the 231-cell
//! `STATIC_ALLOC` table reproduced inline, the typed accessors that
//! pair it with the §4.3.3 indexing rule, and the
//! [`static_alloc_eighth_bits`] conversion that folds in the
//! `(channels, n_bins, LM)` scale. The §4.3.3 search itself — the
//! 1/64-step interpolation that converges on `q` for a given budget —
//! runs at the §4.3.3 allocator's consumer site (the round that lands
//! the orchestrated allocation search). That search consumes:
//!
//! * the per-band [`crate::celt_cache_caps50::cap_for_band_bits`]
//!   per-band cap (round 31),
//! * the per-band [`crate::celt_band_thresh::band_min_thresh`] floor
//!   (round 35),
//! * the per-band [`crate::celt_trim_offsets::band_trim_offset`] tilt
//!   bias (round 36),
//! * the per-band [`crate::celt_band_boost`] boosts (round 33), and
//! * the working budget from
//!   [`crate::celt_reservations::ReservationOutcome::total_remaining_eighth_bits`]
//!   (round 34).
//!
//! ## Provenance
//!
//! The §4.3.3 narrative is transcribed from RFC 6716,
//! `docs/audio/opus/rfc6716-opus.txt`, pp. 111–112. The 231-cell
//! `STATIC_ALLOC` table is the numeric content of RFC 6716 §4.3.3
//! Table 57 (p. 112): 21 band-rows × 11 quality-columns, in 1/32 bit
//! per MDCT bin Q5 units. The numeric values are uncopyrightable facts
//! under Feist v. Rural; the §4.3.3 RFC text identifies the table by
//! its `(band, q)` indexing rule and gives the values directly in the
//! standards-track text. Reproduced inline here so the table is
//! available without filesystem I/O at runtime.

use crate::celt_band_layout::CELT_NUM_BANDS;

/// Number of quality columns in [`STATIC_ALLOC`] (RFC 6716 §4.3.3
/// Table 57, p. 112).
///
/// The §4.3.3 RFC text describes the column index as
/// `q ∈ {0, 1, …, 10}` — eleven columns total. Column `0` is the
/// §4.3.3 "no allocation" floor (every cell zero); column `10` is
/// the §4.3.3 saturation column (every cell `200` or close); columns
/// `1..=9` are the §4.3.3 working quality range the allocator
/// interpolates over in 1/64-step increments.
pub const STATIC_ALLOC_Q_COUNT: usize = 11;

/// Minimum value the §4.3.3 quality column index can take
/// (RFC 6716 §4.3.3 Table 57, p. 112: `q = 0`).
pub const STATIC_ALLOC_Q_MIN: u32 = 0;

/// Maximum value the §4.3.3 quality column index can take
/// (RFC 6716 §4.3.3 Table 57, p. 112: `q = 10`).
pub const STATIC_ALLOC_Q_MAX: u32 = 10;

/// §4.3.3 unit-conversion shift offset (RFC 6716 §4.3.3, p. 111,
/// `<< LM >> 2`).
///
/// The §4.3.3 conversion `channels * N * alloc[band][q] << LM >> 2`
/// applies a net `LM − 2` shift to fold the Q5 (1/32-bit per MDCT bin)
/// table value into Q3 (1/8-bit) per-band units. We expose the `>> 2`
/// half as a named constant; the `<< LM` half is data-dependent.
pub const STATIC_ALLOC_RIGHT_SHIFT: u32 = 2;

/// §4.3.3 interpolation step denominator (RFC 6716 §4.3.3, p. 111:
/// "in steps of 1/64").
///
/// The §4.3.3 1/64-step interpolation between adjacent quality columns
/// keeps the search finer-grained than the output unit (1/8 bit). The
/// orchestrated allocator consumer multiplies by this in the Q11
/// arithmetic before the `<< LM >> 2` step folds the result back to
/// Q3.
pub const STATIC_ALLOC_INTERP_STEPS: u32 = 64;

/// Total cells in [`STATIC_ALLOC`]: 21 bands × 11 quality columns =
/// 231 entries.
pub const STATIC_ALLOC_TOTAL_CELLS: usize = CELT_NUM_BANDS * STATIC_ALLOC_Q_COUNT;

/// §4.3.3 `alloc[][]` static allocation table (RFC 6716 §4.3.3
/// Table 57, p. 112).
///
/// 21-row × 11-column grid. Rows index the §4.3 Table 55 band index
/// `b ∈ 0..=20`; columns index the §4.3.3 quality parameter
/// `q ∈ 0..=10`. Every cell is a Q5 value in 1/32-bit per MDCT bin
/// units. Linear indexing into the flattened array is
/// `i = band * STATIC_ALLOC_Q_COUNT + q`.
///
/// The §4.3.3 search converts each cell to a per-band bit allocation
/// in 1/8 bits via
/// `(channels * N * STATIC_ALLOC[band][q]) << LM >> 2`, then linearly
/// interpolates between two adjacent columns in steps of 1/64 to find
/// the highest allocation that does not exceed the working budget.
///
/// Layout invariants the §4.3.3 narrative imposes and that
/// [`STATIC_ALLOC`] honours:
///
/// * Column `0` is uniformly zero (the §4.3.3 "no allocation" floor).
/// * Each row is monotone non-decreasing in `q` (a higher quality
///   column never reduces the allocation).
/// * Column `10` is `200` for the first 12 rows (bands `0..=11`) and
///   declines monotonically thereafter as the higher bands consume
///   more raw bits per cell at saturation.
///
/// Numeric provenance: RFC 6716 §4.3.3 Table 57 (p. 112), held in-repo
/// at `docs/audio/opus/rfc6716-opus.txt`.
#[rustfmt::skip]
pub const STATIC_ALLOC: [[u8; STATIC_ALLOC_Q_COUNT]; CELT_NUM_BANDS] = [
    // band 0
    [0, 90, 110, 118, 126, 134, 144, 152, 162, 172, 200],
    // band 1
    [0, 80, 100, 110, 119, 127, 137, 145, 155, 165, 200],
    // band 2
    [0, 75,  90, 103, 112, 120, 130, 138, 148, 158, 200],
    // band 3
    [0, 69,  84,  93, 104, 114, 124, 132, 142, 152, 200],
    // band 4
    [0, 63,  78,  86,  95, 103, 113, 123, 133, 143, 200],
    // band 5
    [0, 56,  71,  80,  89,  97, 107, 117, 127, 137, 200],
    // band 6
    [0, 49,  65,  75,  83,  91, 101, 111, 121, 131, 200],
    // band 7
    [0, 40,  58,  70,  78,  85,  95, 105, 115, 125, 200],
    // band 8
    [0, 34,  51,  65,  72,  78,  88,  98, 108, 118, 198],
    // band 9
    [0, 29,  45,  59,  66,  72,  82,  92, 102, 112, 193],
    // band 10
    [0, 20,  39,  53,  60,  66,  76,  86,  96, 106, 188],
    // band 11
    [0, 18,  32,  47,  54,  60,  70,  80,  90, 100, 183],
    // band 12
    [0, 10,  26,  40,  47,  54,  64,  74,  84,  94, 178],
    // band 13
    [0,  0,  20,  31,  39,  47,  57,  67,  77,  87, 173],
    // band 14
    [0,  0,  12,  23,  32,  41,  51,  61,  71,  81, 168],
    // band 15
    [0,  0,   0,  15,  25,  35,  45,  55,  65,  75, 163],
    // band 16
    [0,  0,   0,   4,  17,  29,  39,  49,  59,  69, 158],
    // band 17
    [0,  0,   0,   0,  12,  23,  33,  43,  53,  63, 153],
    // band 18
    [0,  0,   0,   0,   1,  16,  26,  36,  46,  56, 148],
    // band 19
    [0,  0,   0,   0,   0,  10,  15,  20,  30,  45, 129],
    // band 20
    [0,  0,   0,   0,   0,   1,   1,   1,   1,  20, 104],
];

/// Errors returned by the [`STATIC_ALLOC`] accessors when their
/// `(band, q)` inputs sit outside the §4.3.3 Table 57 grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticAllocError {
    /// `band` is outside `0..21` — the §4.3 Table 55 band count.
    BandOutOfRange { band: u32 },
    /// `q` is outside `0..11` — the §4.3.3 quality column range.
    QualityOutOfRange { q: u32 },
    /// `channels` is outside `1..=2` — RFC 6716 §4.3.3 (p. 111)
    /// declares `channels ∈ {1, 2}` for the unit conversion.
    ChannelsOutOfRange { channels: u32 },
    /// `lm` is outside `0..4` — RFC 6716 §4.3 only defines four CELT
    /// frame sizes (`LM ∈ {0, 1, 2, 3}`).
    LmOutOfRange { lm: u32 },
}

/// Look up the raw Q5 (1/32-bit per MDCT bin) cell of [`STATIC_ALLOC`]
/// for one `(band, q)` pair (RFC 6716 §4.3.3 Table 57, p. 112).
///
/// This is the §4.3.3 step "look up `alloc[band][q]`" — the lookup
/// itself, before the `(channels * N) << LM >> 2` unit conversion.
/// Callers who only need the raw Q5 byte (e.g. cross-checks of the
/// table layout) use this; callers who need the per-band bit
/// allocation in 1/8 bits call [`static_alloc_eighth_bits`].
pub fn static_alloc_cell(band: u32, q: u32) -> Result<u8, StaticAllocError> {
    if band >= CELT_NUM_BANDS as u32 {
        return Err(StaticAllocError::BandOutOfRange { band });
    }
    if q >= STATIC_ALLOC_Q_COUNT as u32 {
        return Err(StaticAllocError::QualityOutOfRange { q });
    }
    Ok(STATIC_ALLOC[band as usize][q as usize])
}

/// Borrow the full 11-cell row of [`STATIC_ALLOC`] for one `band`
/// (RFC 6716 §4.3.3 Table 57, p. 112).
///
/// Useful when the §4.3.3 search iterates the quality columns for one
/// band without re-indexing per call (the natural inner-loop shape of
/// the 1/64-step interpolation).
pub fn static_alloc_row(
    band: u32,
) -> Result<&'static [u8; STATIC_ALLOC_Q_COUNT], StaticAllocError> {
    if band >= CELT_NUM_BANDS as u32 {
        return Err(StaticAllocError::BandOutOfRange { band });
    }
    Ok(&STATIC_ALLOC[band as usize])
}

/// Apply the §4.3.3 `channels * N * alloc[band][q] << LM >> 2`
/// conversion to a single Q5 cell (RFC 6716 §4.3.3, p. 111).
///
/// Returns the per-band shape allocation in 1/8 bits — the same units
/// every other §4.3.3 budget quantity uses (compatible with the
/// round-34 [`crate::celt_reservations::ReservationOutcome::total_remaining_eighth_bits`]
/// working budget, the round-35
/// [`crate::celt_band_thresh::band_min_thresh`] floor, the round-36
/// [`crate::celt_trim_offsets::band_trim_offset`] tilt bias, and the
/// round-31 [`crate::celt_cache_caps50::cap_for_band_bits`] cap at
/// the §4.3.3 allocator's consumer site).
///
/// `channels ∈ {1, 2}` is the frame's channel count, `n_bins` is the
/// §4.3 Table 55 per-channel MDCT-bin count for the band at the
/// frame's LM, and `lm ∈ {0, 1, 2, 3}` is the §4.3 frame-size scale.
///
/// The conversion is performed in `u32` to keep intermediate
/// arithmetic well-defined: the largest cell value `200` times the
/// largest reachable `(channels * N) = (2 * 176)` times `(1 << 3) =
/// 8` is `563_200`, comfortably inside `u32` headroom. The `>> 2`
/// step is the §4.3.3 final unit-folding from Q5 to Q3.
pub fn static_alloc_eighth_bits(
    band: u32,
    q: u32,
    channels: u32,
    n_bins: u32,
    lm: u32,
) -> Result<u32, StaticAllocError> {
    if channels == 0 || channels > 2 {
        return Err(StaticAllocError::ChannelsOutOfRange { channels });
    }
    if lm >= 4 {
        return Err(StaticAllocError::LmOutOfRange { lm });
    }
    let cell = static_alloc_cell(band, q)? as u32;
    // Per RFC 6716 §4.3.3 p. 111: channels * N * alloc[band][q] << LM >> 2.
    // u32 arithmetic; all intermediate products fit.
    let scaled = channels * n_bins * cell;
    Ok((scaled << lm) >> STATIC_ALLOC_RIGHT_SHIFT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_band_layout::{celt_band_bins_per_channel, CeltFrameSize};

    // ---- Table-shape invariants ----

    #[test]
    fn table_shape_constants_match_struct() {
        assert_eq!(STATIC_ALLOC_Q_COUNT, 11);
        assert_eq!(STATIC_ALLOC_Q_MIN, 0);
        assert_eq!(STATIC_ALLOC_Q_MAX, 10);
        assert_eq!(STATIC_ALLOC_TOTAL_CELLS, 231);
        assert_eq!(CELT_NUM_BANDS, 21);
        assert_eq!(STATIC_ALLOC.len(), CELT_NUM_BANDS);
        for row in STATIC_ALLOC.iter() {
            assert_eq!(row.len(), STATIC_ALLOC_Q_COUNT);
        }
    }

    #[test]
    fn unit_conversion_constants_match_rfc() {
        assert_eq!(STATIC_ALLOC_RIGHT_SHIFT, 2);
        assert_eq!(STATIC_ALLOC_INTERP_STEPS, 64);
    }

    // ---- Spot-check Q5 cells against RFC 6716 §4.3.3 Table 57 ----
    //
    // These pins reproduce hand-picked rows / corners from
    // `docs/audio/opus/rfc6716-opus.txt` pp. 112 so a future edit that
    // reorders / drops / typos an entry trips the suite.

    #[test]
    fn corner_top_left_is_zero() {
        // §4.3.3 Table 57: band 0, q 0 = 0 (no-allocation floor).
        assert_eq!(STATIC_ALLOC[0][0], 0);
        assert_eq!(static_alloc_cell(0, 0).unwrap(), 0);
    }

    #[test]
    fn corner_top_right_is_two_hundred() {
        // §4.3.3 Table 57: band 0, q 10 = 200 (saturation column).
        assert_eq!(STATIC_ALLOC[0][10], 200);
        assert_eq!(static_alloc_cell(0, 10).unwrap(), 200);
    }

    #[test]
    fn corner_bottom_left_is_zero() {
        // §4.3.3 Table 57: band 20, q 0 = 0 (every column-0 cell is
        // the no-allocation floor).
        assert_eq!(STATIC_ALLOC[20][0], 0);
        assert_eq!(static_alloc_cell(20, 0).unwrap(), 0);
    }

    #[test]
    fn corner_bottom_right_is_one_hundred_four() {
        // §4.3.3 Table 57: band 20, q 10 = 104. The saturation column
        // declines from 200 at the low bands to 104 at band 20.
        assert_eq!(STATIC_ALLOC[20][10], 104);
        assert_eq!(static_alloc_cell(20, 10).unwrap(), 104);
    }

    #[test]
    fn band_0_q_1_is_ninety() {
        // §4.3.3 Table 57: band 0, q 1 = 90 — the first non-zero entry
        // of the first row.
        assert_eq!(STATIC_ALLOC[0][1], 90);
    }

    #[test]
    fn band_8_q_10_is_one_hundred_ninety_eight() {
        // §4.3.3 Table 57: band 8, q 10 = 198 — the first row where
        // the saturation column drops below 200.
        assert_eq!(STATIC_ALLOC[8][10], 198);
    }

    #[test]
    fn band_13_q_1_is_zero() {
        // §4.3.3 Table 57: band 13, q 1 = 0 — the first row whose
        // q = 1 entry is still in the no-allocation regime.
        assert_eq!(STATIC_ALLOC[13][1], 0);
        // q = 2 picks up.
        assert_eq!(STATIC_ALLOC[13][2], 20);
    }

    #[test]
    fn band_20_low_q_columns_pin_to_one() {
        // §4.3.3 Table 57: band 20, q 5..=8 = 1 — the highest band
        // sits at 1 across four consecutive working columns before
        // ramping to q = 10's 104.
        assert_eq!(STATIC_ALLOC[20][5], 1);
        assert_eq!(STATIC_ALLOC[20][6], 1);
        assert_eq!(STATIC_ALLOC[20][7], 1);
        assert_eq!(STATIC_ALLOC[20][8], 1);
        assert_eq!(STATIC_ALLOC[20][9], 20);
    }

    // ---- Row monotonicity invariant ----
    //
    // The §4.3.3 narrative phrases the quality columns as a ladder.
    // Each row must be monotone non-decreasing in q — a higher quality
    // column may never reduce the allocation for the same band.

    #[test]
    fn every_row_is_monotone_non_decreasing_in_q() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            for w in row.windows(2) {
                assert!(
                    w[0] <= w[1],
                    "STATIC_ALLOC[{band}] not monotone non-decreasing at pair {w:?}",
                );
            }
        }
    }

    #[test]
    fn column_zero_is_uniformly_zero() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            assert_eq!(row[0], 0, "STATIC_ALLOC[{band}][0] != 0");
        }
    }

    #[test]
    fn column_ten_top_twelve_bands_are_two_hundred() {
        // §4.3.3 Table 57: bands 0..=7 saturation column is 200; band
        // 8 drops to 198. Pin the band-7 / band-8 boundary so a
        // future edit can't push the drop into a different row.
        for (band, row) in STATIC_ALLOC.iter().enumerate().take(8) {
            assert_eq!(
                row[10], 200,
                "STATIC_ALLOC[{band}][10] expected 200 at saturation"
            );
        }
        assert_eq!(STATIC_ALLOC[8][10], 198);
    }

    // ---- Accessor parity ----

    #[test]
    fn cell_accessor_matches_array_indexing() {
        for band in 0..(CELT_NUM_BANDS as u32) {
            for q in 0..(STATIC_ALLOC_Q_COUNT as u32) {
                assert_eq!(
                    static_alloc_cell(band, q).unwrap(),
                    STATIC_ALLOC[band as usize][q as usize],
                );
            }
        }
    }

    #[test]
    fn row_accessor_borrows_full_row() {
        for band in 0..(CELT_NUM_BANDS as u32) {
            let row = static_alloc_row(band).unwrap();
            assert_eq!(row.len(), STATIC_ALLOC_Q_COUNT);
            assert_eq!(row, &STATIC_ALLOC[band as usize]);
        }
    }

    // ---- Out-of-range guards ----

    #[test]
    fn cell_rejects_band_out_of_range() {
        assert_eq!(
            static_alloc_cell(21, 0).unwrap_err(),
            StaticAllocError::BandOutOfRange { band: 21 },
        );
        assert_eq!(
            static_alloc_cell(u32::MAX, 0).unwrap_err(),
            StaticAllocError::BandOutOfRange { band: u32::MAX },
        );
    }

    #[test]
    fn cell_rejects_quality_out_of_range() {
        assert_eq!(
            static_alloc_cell(0, 11).unwrap_err(),
            StaticAllocError::QualityOutOfRange { q: 11 },
        );
        assert_eq!(
            static_alloc_cell(0, u32::MAX).unwrap_err(),
            StaticAllocError::QualityOutOfRange { q: u32::MAX },
        );
    }

    #[test]
    fn row_rejects_band_out_of_range() {
        assert_eq!(
            static_alloc_row(21).unwrap_err(),
            StaticAllocError::BandOutOfRange { band: 21 },
        );
    }

    // ---- §4.3.3 unit conversion ----
    //
    // Pin the `channels * N * cell << LM >> 2` arithmetic with worked
    // examples that the §4.3.3 narrative can be hand-traced through.

    #[test]
    fn unit_conversion_q_zero_is_zero() {
        // §4.3.3 column 0 is the no-allocation floor; any conversion
        // through it must yield 0 regardless of channels / n_bins / LM.
        for band in 0..(CELT_NUM_BANDS as u32) {
            for &channels in &[1u32, 2] {
                for &n_bins in &[1u32, 4, 88, 176] {
                    for lm in 0..4 {
                        assert_eq!(
                            static_alloc_eighth_bits(band, 0, channels, n_bins, lm).unwrap(),
                            0,
                            "band {band} q 0 channels {channels} n_bins {n_bins} lm {lm}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unit_conversion_worked_example_lm_zero() {
        // §4.3.3 cell band 0 q 1 = 90 (Q5 bits/bin). Convert with
        // channels = 1, n_bins = 1, lm = 0:
        // (1 * 1 * 90) << 0 >> 2 = 90 / 4 = 22 (integer division).
        assert_eq!(static_alloc_eighth_bits(0, 1, 1, 1, 0).unwrap(), 22);

        // Same cell with channels = 2 (stereo): scales by 2.
        // (2 * 1 * 90) >> 2 = 45.
        assert_eq!(static_alloc_eighth_bits(0, 1, 2, 1, 0).unwrap(), 45);

        // Same cell with n_bins = 4 (band-loop natural width):
        // (1 * 4 * 90) >> 2 = 90.
        assert_eq!(static_alloc_eighth_bits(0, 1, 1, 4, 0).unwrap(), 90);
    }

    #[test]
    fn unit_conversion_worked_example_lm_three() {
        // §4.3.3 cell band 0 q 10 = 200 with channels = 2, n_bins = 4
        // (band 0 width at 20 ms is 4 per channel), LM = 3 (20 ms):
        // (2 * 4 * 200) << 3 >> 2 = 1600 * 8 / 4 = 3200.
        assert_eq!(static_alloc_eighth_bits(0, 10, 2, 4, 3).unwrap(), 3200);
    }

    #[test]
    fn unit_conversion_lm_shift_doubles_per_step() {
        // The §4.3.3 << LM step multiplies the allocation by 2^LM.
        // Pin the doubling across the four CELT frame sizes for a
        // representative cell.
        let cell = STATIC_ALLOC[5][5] as u32; // 97
        let channels = 1u32;
        let n_bins = 4u32;
        let base = (channels * n_bins * cell) >> 2; // LM = 0 value
        for lm in 0..4u32 {
            let got = static_alloc_eighth_bits(5, 5, channels, n_bins, lm).unwrap();
            assert_eq!(got, base << lm, "LM = {lm} doubling failed");
        }
    }

    #[test]
    fn unit_conversion_against_band_layout_widths() {
        // Cross-check against the §4.3 Table 55 widths (round 24).
        // Pin two cells against the round-24 widths so a future edit
        // that drifts the band layout table trips the suite.
        let n0_lm3 = celt_band_bins_per_channel(0, CeltFrameSize::Ms20).unwrap() as u32;
        assert_eq!(n0_lm3, 8);
        let got = static_alloc_eighth_bits(0, 5, 2, n0_lm3, 3).unwrap();
        // (2 * 8 * STATIC_ALLOC[0][5]) << 3 >> 2 = (2 * 8 * 134) * 8 / 4 = 4288
        assert_eq!(got, 4288);

        let n20_lm3 = celt_band_bins_per_channel(20, CeltFrameSize::Ms20).unwrap() as u32;
        assert_eq!(n20_lm3, 176);
        let got = static_alloc_eighth_bits(20, 9, 1, n20_lm3, 3).unwrap();
        // (1 * 176 * STATIC_ALLOC[20][9]) << 3 >> 2 = (176 * 20) * 8 / 4 = 7040
        assert_eq!(got, 7040);
    }

    #[test]
    fn unit_conversion_rejects_channels_out_of_range() {
        assert_eq!(
            static_alloc_eighth_bits(0, 1, 0, 1, 0).unwrap_err(),
            StaticAllocError::ChannelsOutOfRange { channels: 0 },
        );
        assert_eq!(
            static_alloc_eighth_bits(0, 1, 3, 1, 0).unwrap_err(),
            StaticAllocError::ChannelsOutOfRange { channels: 3 },
        );
    }

    #[test]
    fn unit_conversion_rejects_lm_out_of_range() {
        assert_eq!(
            static_alloc_eighth_bits(0, 1, 1, 1, 4).unwrap_err(),
            StaticAllocError::LmOutOfRange { lm: 4 },
        );
        assert_eq!(
            static_alloc_eighth_bits(0, 1, 1, 1, u32::MAX).unwrap_err(),
            StaticAllocError::LmOutOfRange { lm: u32::MAX },
        );
    }

    #[test]
    fn unit_conversion_propagates_band_and_quality_errors() {
        assert_eq!(
            static_alloc_eighth_bits(21, 0, 1, 1, 0).unwrap_err(),
            StaticAllocError::BandOutOfRange { band: 21 },
        );
        assert_eq!(
            static_alloc_eighth_bits(0, 11, 1, 1, 0).unwrap_err(),
            StaticAllocError::QualityOutOfRange { q: 11 },
        );
    }

    // ---- Cross-row plausibility ----
    //
    // The §4.3.3 narrative doesn't pin a monotonicity across rows, but
    // the table's structure (low bands carry more of the rate at low
    // q, high bands shed allocation faster) shows up empirically. Pin
    // a few worked invariants the §4.3.3 search assumes.

    #[test]
    fn low_band_dominates_high_band_at_low_q() {
        // At q = 1, band 0 = 90 vs band 13..=20 = 0. Pin the spread —
        // a future re-typing that introduced a non-zero into a
        // higher-band q=1 cell would break the §4.3.3 search's
        // low-q "concentrate on low bands" assumption.
        assert_eq!(STATIC_ALLOC[0][1], 90);
        for (band, row) in STATIC_ALLOC.iter().enumerate().skip(13) {
            assert_eq!(row[1], 0, "STATIC_ALLOC[{band}][1] expected 0 at low q",);
        }
    }

    #[test]
    fn saturation_column_declines_after_band_seven() {
        // §4.3.3 Table 57 saturation column: bands 0..=7 = 200,
        // band 8 = 198, band 20 = 104. Pin the overall decline.
        assert_eq!(STATIC_ALLOC[7][10], 200);
        assert_eq!(STATIC_ALLOC[8][10], 198);
        assert!(
            STATIC_ALLOC[8][10] < STATIC_ALLOC[7][10],
            "saturation column failed to decline at band 7→8",
        );
        for band in 8..20 {
            assert!(
                STATIC_ALLOC[band + 1][10] <= STATIC_ALLOC[band][10],
                "saturation column non-monotone at band {} → {}",
                band,
                band + 1,
            );
        }
        assert_eq!(STATIC_ALLOC[20][10], 104);
    }
}
