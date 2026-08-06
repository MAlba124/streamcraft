//! CELT §4.3.3 intensity-stereo reservation parameter surface
//! (RFC 6716 §4.3.3, p. 113).
//!
//! The §4.3.3 *Bit Allocation* procedure reserves a handful of fixed
//! 1/8-bit slots before searching the Table 57 static allocation.
//! One of those slots — the *intensity-stereo* reservation — needs
//! the conservative base-2 logarithm of the number of coded bands,
//! expressed in 1/8-bit units, computed as
//! `intensity_rsv = LOG2_FRAC_TABLE[end − start]` per RFC 6716
//! §4.3.3 p. 113. The RFC names the table `LOG2_FRAC_TABLE` (held
//! in `rate.c` per the RFC); the 24 Q3 byte values themselves are
//! uncopyrightable numeric facts under Feist v. Rural and live in
//! `docs/audio/celt/tables/log2_frac_table.csv`. This module owns
//! the *parameter surface*: the table reproduced inline plus the
//! typed accessor that pairs it with the §4.3.3 indexing rule.
//! The §4.3.3 reservation orchestration itself runs at the call
//! site of the lookup.
//!
//! The §4.3.3 narrative is transcribed from RFC 6716,
//! `docs/audio/opus/rfc6716-opus.txt`, pp. 112–114. The 24-byte
//! `LOG2_FRAC_TABLE` data is uncopyrightable numeric facts extracted
//! into `docs/audio/celt/tables/log2_frac_table.csv` (see
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.5
//! for the canonical layout). The values are reproduced inline here
//! so the table is available without filesystem I/O at runtime.
//!
//! ## Layout
//!
//! [`LOG2_FRAC_TABLE`] is a `[u8; 24]`. Index `n ∈ 0..=23` returns the
//! conservative base-2 log of `n` in 1/8-bit units. By construction
//! `LOG2_FRAC_TABLE[0] = 0` (since `log2(0)` is undefined and the
//! §4.3.3 path never reserves intensity bits for a zero-band frame),
//! `LOG2_FRAC_TABLE[1] = 8` (i.e. one bit), `LOG2_FRAC_TABLE[2] = 13`
//! (≈ `log2(2)*8 = 8`, but the §4.3.3 entry is conservative — it
//! rounds up to keep enough room), and so on monotonically up to
//! `LOG2_FRAC_TABLE[23] = 37`.
//!
//! ## §4.3.3 indexing rule
//!
//! Per RFC 6716 §4.3.3 (p. 113), the table is indexed by the number
//! of *coded* bands in the frame, computed as `end − start` over the
//! §4.3 Table 55 band loop. For CELT-only frames the band loop runs
//! `0..=20` so `end − start = 21`; for Hybrid frames the SILK layer
//! covers the first 17 bands so `end − start = 4` (the §4.3 carve-out
//! of bands `17..=20`). The maximum value reachable is 21 in the
//! CELT-only case; the table's 24-entry depth covers that with
//! headroom and keeps the §4.3.3 lookup total over any caller-driven
//! coded-band count up to the 23-element ceiling.
//!
//! ## Provenance
//!
//! Narrative: RFC 6716 §4.3.3 (pp. 112–114) in
//! `docs/audio/opus/rfc6716-opus.txt`. Numeric table: the 24-entry
//! Q3 byte sequence from `docs/audio/celt/tables/log2_frac_table.csv`
//! (see the `.meta` sidecar for the canonical layout). The narrative
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.5
//! cross-references both.

/// Number of entries in [`LOG2_FRAC_TABLE`] (RFC 6716 §4.3.3, p. 113).
///
/// Twenty-four covers every reachable `(end − start)` from the §4.3.3
/// band loop with headroom: CELT-only `(0..=20)` reaches `21`, Hybrid
/// `(17..=20)` reaches `4`.
pub const LOG2_FRAC_TABLE_LEN: usize = 24;

/// §4.3.3 unit denominator: every [`LOG2_FRAC_TABLE`] entry is in
/// 1/8-bit (Q3) units, so multiplying by 8 / dividing by 8 toggles
/// between whole bits and 1/8-bit units.
pub const Q3_BITS_PER_WHOLE_BIT: u32 = 8;

/// §4.3.3 `LOG2_FRAC_TABLE` (RFC 6716 §4.3.3, p. 113).
///
/// `LOG2_FRAC_TABLE[n]` is the conservative base-2 logarithm of `n`
/// in 1/8-bit (Q3) units, used by the §4.3.3 *intensity-stereo*
/// reservation as `intensity_rsv = LOG2_FRAC_TABLE[end − start]`.
///
/// Data provenance: `docs/audio/celt/tables/log2_frac_table.csv` (Q3
/// numeric facts; see the CSV's `.meta` sidecar for the canonical
/// layout). RFC 6716 §4.3.3 names the table `LOG2_FRAC_TABLE` and
/// describes it as held in `rate.c`; only the numeric data is
/// reproduced here.
pub const LOG2_FRAC_TABLE: [u8; LOG2_FRAC_TABLE_LEN] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

/// Errors returned by [`log2_frac`] for indices outside the
/// 24-entry table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Log2FracError {
    /// `coded_bands` is outside `0..24` — the §4.3.3 lookup table only
    /// covers that range. The §4.3.3 band loop never produces more
    /// than `21` coded bands (CELT-only `0..=20`), so an out-of-range
    /// index is a caller-side bug.
    CodedBandsOutOfRange { coded_bands: u32 },
}

/// Look up the conservative `log2` in 1/8-bit units for a single
/// `coded_bands = end − start` count (RFC 6716 §4.3.3, p. 113).
///
/// Returns the §4.3.3 `intensity_rsv` value (in Q3 1/8-bit units)
/// the caller would reserve from `total` before the Table 57 static
/// allocation search runs. The §4.3.3 monotone-non-decreasing
/// invariant guarantees a larger band count never produces a smaller
/// reservation.
pub fn log2_frac(coded_bands: u32) -> Result<u8, Log2FracError> {
    if coded_bands >= LOG2_FRAC_TABLE_LEN as u32 {
        return Err(Log2FracError::CodedBandsOutOfRange { coded_bands });
    }
    Ok(LOG2_FRAC_TABLE[coded_bands as usize])
}

/// Borrow the full 24-entry [`LOG2_FRAC_TABLE`].
///
/// Useful when a downstream sub-decoder wants to iterate the table
/// (or pin its full byte sequence in a regression test) without
/// re-indexing per call.
pub fn log2_frac_row() -> &'static [u8; LOG2_FRAC_TABLE_LEN] {
    &LOG2_FRAC_TABLE
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Table-shape invariants ----

    #[test]
    fn table_shape_constant_matches_struct() {
        assert_eq!(LOG2_FRAC_TABLE_LEN, 24);
        assert_eq!(LOG2_FRAC_TABLE.len(), LOG2_FRAC_TABLE_LEN);
    }

    #[test]
    fn q3_unit_denominator_is_eight() {
        assert_eq!(Q3_BITS_PER_WHOLE_BIT, 8);
    }

    // ---- Spot-check Q3 values against the CSV extract ----
    //
    // These pins reproduce hand-picked rows from
    // `docs/audio/celt/tables/log2_frac_table.csv` so a future edit
    // that reorders the table or drops an entry trips the suite.

    #[test]
    fn csv_index_0_is_zero() {
        // CSV row "0,0" — the §4.3.3 base case where intensity stereo
        // is not reservable.
        assert_eq!(LOG2_FRAC_TABLE[0], 0);
        assert_eq!(log2_frac(0).unwrap(), 0);
    }

    #[test]
    fn csv_index_1_is_eight() {
        // CSV row "1,8" — exactly 1 bit in 1/8-bit units.
        assert_eq!(LOG2_FRAC_TABLE[1], 8);
        assert_eq!(log2_frac(1).unwrap(), 8);
    }

    #[test]
    fn csv_index_2_is_thirteen() {
        // CSV row "2,13" — log2(2)*8 = 8, but §4.3.3 rounds the entry
        // upward to keep the reservation conservative.
        assert_eq!(LOG2_FRAC_TABLE[2], 13);
        assert_eq!(log2_frac(2).unwrap(), 13);
    }

    #[test]
    fn csv_index_4_is_nineteen() {
        // CSV row "4,19" — the Hybrid `end − start = 4` reservation.
        assert_eq!(LOG2_FRAC_TABLE[4], 19);
        assert_eq!(log2_frac(4).unwrap(), 19);
    }

    #[test]
    fn csv_index_15_is_thirty_two() {
        // CSV row "15,32" — pinned because the boundary at 14→15 (the
        // 32-byte plateau) is sensitive to drift.
        assert_eq!(LOG2_FRAC_TABLE[14], 32);
        assert_eq!(LOG2_FRAC_TABLE[15], 32);
        assert_eq!(log2_frac(15).unwrap(), 32);
    }

    #[test]
    fn csv_index_21_is_thirty_six() {
        // CSV row "21,36" — the maximum `end − start` value the
        // §4.3.3 band loop can produce in the CELT-only case
        // (bands 0..=20).
        assert_eq!(LOG2_FRAC_TABLE[21], 36);
        assert_eq!(log2_frac(21).unwrap(), 36);
    }

    #[test]
    fn csv_index_23_is_thirty_seven() {
        // CSV row "23,37" — final entry, marking the table's upper
        // bound.
        assert_eq!(LOG2_FRAC_TABLE[23], 37);
        assert_eq!(log2_frac(23).unwrap(), 37);
    }

    // ---- Monotonicity invariant ----
    //
    // The §4.3.3 `intensity_rsv` reservation is a conservative log2;
    // adding a coded band can never reduce the reservation. The §2.5
    // narrative of the cleanroom spec calls this out implicitly by
    // labeling the table "conservative log2". Pin the property.

    #[test]
    fn table_is_monotone_non_decreasing() {
        for w in LOG2_FRAC_TABLE.windows(2) {
            assert!(
                w[0] <= w[1],
                "LOG2_FRAC_TABLE not monotone non-decreasing at pair {:?}",
                w
            );
        }
    }

    // ---- Conservative-log2 bound ----
    //
    // The §4.3.3 narrative calls the table a *conservative* log2 — the
    // tabulated entry is ≥ `8 * log2(n)` for every n ≥ 1, never less.
    // We check the inequality with a bit-shift-only formulation that
    // avoids floating-point.

    #[test]
    fn entries_are_at_or_above_eight_times_log2() {
        for (n, &entry) in LOG2_FRAC_TABLE.iter().enumerate().skip(1) {
            // `floor(log2(n))` via leading-zero count.
            let floor_log2 = (u32::BITS - 1 - (n as u32).leading_zeros()) as u8;
            let lower_bound = floor_log2.checked_mul(8).expect("log2 fits in u8");
            assert!(
                entry >= lower_bound,
                "LOG2_FRAC_TABLE[{}] = {} should be >= 8 * floor(log2({})) = {}",
                n,
                entry,
                n,
                lower_bound
            );
        }
    }

    // ---- Total-function sweep over the accessor ----

    #[test]
    fn log2_frac_accessor_is_total_over_in_range_inputs() {
        for n in 0..LOG2_FRAC_TABLE_LEN as u32 {
            let v = log2_frac(n).expect("in-range lookup");
            assert_eq!(v, LOG2_FRAC_TABLE[n as usize]);
        }
    }

    // ---- Error-path coverage ----

    #[test]
    fn log2_frac_rejects_coded_bands_out_of_range() {
        let err = log2_frac(LOG2_FRAC_TABLE_LEN as u32).unwrap_err();
        assert_eq!(err, Log2FracError::CodedBandsOutOfRange { coded_bands: 24 });
        let err = log2_frac(u32::MAX).unwrap_err();
        assert_eq!(
            err,
            Log2FracError::CodedBandsOutOfRange {
                coded_bands: u32::MAX
            }
        );
    }

    // ---- Row accessor mirrors raw constant ----

    #[test]
    fn log2_frac_row_returns_full_24_byte_table() {
        let row = log2_frac_row();
        assert_eq!(row.len(), LOG2_FRAC_TABLE_LEN);
        assert_eq!(row, &LOG2_FRAC_TABLE);
        // First entry is the §4.3.3 base case.
        assert_eq!(row[0], 0);
        // Final entry pins the upper bound.
        assert_eq!(row[LOG2_FRAC_TABLE_LEN - 1], 37);
    }

    #[test]
    fn pair_lookup_matches_row_lookup() {
        let row = log2_frac_row();
        for n in 0..LOG2_FRAC_TABLE_LEN as u32 {
            assert_eq!(log2_frac(n).unwrap(), row[n as usize]);
        }
    }

    // ---- §4.3.3 reachable-index sanity ----
    //
    // The §4.3.3 band loop can produce at most 21 coded bands
    // (CELT-only 0..=20). The Hybrid carve-out (bands 17..=20) reaches
    // 4. Pin both edge values so a regression on Hybrid reservation
    // ends up visible here.

    #[test]
    fn hybrid_reachable_index_pins_at_four_coded_bands() {
        // §4.3 Hybrid: end − start = (21) − (17) = 4.
        assert_eq!(log2_frac(4).unwrap(), 19);
    }

    #[test]
    fn celt_only_reachable_index_pins_at_twenty_one_coded_bands() {
        // §4.3 CELT-only: end − start = (21) − (0) = 21.
        assert_eq!(log2_frac(21).unwrap(), 36);
    }
}
