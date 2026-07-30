//! CELT §4.3.4.1 *Bits to Pulses* pulse-cost cache
//! (RFC 6716 §4.3.4 / §4.3.4.1).
//!
//! The §4.3.4 *Shape* decode codes each band's unit-norm spectral
//! shape with Pulse Vector Quantisation (PVQ): a band of `N` MDCT
//! coefficients is represented by `K` signed integer pulse positions
//! whose absolute values sum to `K`. Before the shape can be decoded
//! the allocator must answer the §4.3.4.1 *Bits to Pulses* question:
//! given a per-band bit budget `B`, what is the largest pulse count
//! `K` whose PVQ codeword cost does not exceed `B`?
//!
//! The exact cost is `log2(V(N, K))` plus a small framing overhead,
//! where `V(N, K)` is the number of PVQ codepoints with `K` pulses in
//! `N` slots (the round-32 [`crate::celt_pvq_v`] recurrence). Computing
//! that at decode time for every candidate `K` is expensive, so the
//! §4.3.4.1 procedure precomputes the cost curve for the `(band, LM)`
//! combinations that occur at the codec's normal operating bitrates and
//! stores them as two flat tables:
//!
//! * [`CACHE_INDEX50`] — a 105-entry `i16` table mapping each
//!   `(band, LM)` tuple to a byte offset into [`CACHE_BITS50`], or the
//!   sentinel `-1` for tuples the allocator handles with a closed-form
//!   path.
//! * [`CACHE_BITS50`] — a 392-byte run-packed table; each run holds the
//!   monotone cost curve `qbits[1..=maxK]` for one `(N, max-pulses)`
//!   profile, in the codec's `BITRES = 3` units (1/8 bits / Q3).
//!
//! This module owns only the §4.3.4.1 *cost-cache lookup surface*: the
//! two tables, the `(band, LM)` → offset indexing rule, the run reader,
//! and the bits-to-pulses inversion scan. The §4.3.4 PVQ shape decode
//! that consumes the returned `K` ([`crate::celt_pvq_decode`]) and the
//! §4.3.3 allocator that computes the per-band budget `B` run at their
//! own call sites.
//!
//! ## Indexing
//!
//! The 21 CELT nominal bands are indexed `band ∈ 0..=20`
//! ([`crate::celt_band_layout::CELT_NUM_BANDS`]). Each band's actual
//! coefficient count `N` depends on the frame size through
//! `LM = log2(frame_size / 120)`, with `LM ∈ 0..=4` covering the
//! 2.5 / 5 / 10 / 20 ms frames plus the short-block transient variant.
//! There are `21 × 5 = 105` distinct `(band, LM)` tuples, which is
//! exactly the length of [`CACHE_INDEX50`]. The five LM rows cover the
//! *split levels* `LM ∈ -1..=3`, stored at row `LM + 1` (row 0 is the
//! `LM = -1` level reached by the §4.3.4.4 band splitting; rows 1–4
//! are the 2.5 / 5 / 10 / 20 ms whole-band levels). The tuple maps to
//! the flat index in **LM-major** order (5 rows of 21 bands — the
//! corrected §2.1 mapping of the staged pulse-cache format trace):
//!
//! ```text
//! i      = lm_row * CELT_NUM_BANDS + band      (lm_row = LM + 1)
//! offset = CACHE_INDEX50[i]
//! ```
//!
//! A `-1` offset is a sentinel ([`CACHE_INDEX_SENTINEL`]) meaning the
//! `(band, LM)` has no cached cost curve — its effective `N` is zero
//! (never reached by the allocator). The eight sentinels are bands
//! 0–7 of row 0 (the `LM = -1` split level, where the one-bin bands
//! halve to nothing). Under this mapping each of the 23 distinct runs
//! serves exactly one effective band size `N` (trace §2.3).
//!
//! ## Run format
//!
//! A run at byte `off` in [`CACHE_BITS50`] is:
//!
//! ```text
//! CACHE_BITS50[off]            = maxK       (1 byte: max K this run supports)
//! CACHE_BITS50[off + 1]        = qbits[1]   (cost for K = 1, in 1/8 bits)
//! CACHE_BITS50[off + 2]        = qbits[2]
//! ...
//! CACHE_BITS50[off + maxK]     = qbits[maxK]
//! ```
//!
//! so the run occupies `1 + maxK` bytes. `K = 0` is implicit (cost 0)
//! and never stored. `qbits[1..=maxK]` is monotone non-decreasing.
//! Several `(band, LM)` tuples that share the same `(N, max-pulses)`
//! profile point at the same run, so the 105-entry index resolves to
//! only 23 distinct runs.
//!
//! ## Bits to pulses
//!
//! Given a per-band budget `b_target` in 1/8 bits,
//! [`bits_to_pulses`] performs the §4.3.4.1 inversion: scan the run's
//! `qbits[1..=maxK]` and return the largest `K` whose cost fits the
//! budget. The scan is a linear walk over a constant-bounded run
//! (`maxK ≤ 40`); the monotone property would also admit a binary
//! search. A sentinel `(band, LM)` returns
//! [`PulseCacheError::SentinelTuple`] so the caller can route to its
//! closed-form path rather than guessing.
//!
//! ## Units
//!
//! All cost values are in 1/8 bits (Q3, the `BITRES = 3` convention
//! shared with [`crate::celt_alloc_search`]). `bits_to_pulses` takes
//! the budget `b_target` in the same units and returns a pure pulse
//! count.
//!
//! ## Provenance
//!
//! Narrative: RFC 6716 §4.3.4 / §4.3.4.1 (*Bits to Pulses*) in
//! `docs/audio/opus/rfc6716-opus.txt`, plus the run-format trace
//! `docs/audio/opus/pulse-cache-format-trace.md` (the corrected §2.1
//! LM-major indexing rule, run packing, sentinel pattern, and
//! qbits → bits conversion). Numeric tables: the 105-entry `cache_index50` and
//! 392-byte `cache_bits50` sequences from
//! `docs/audio/opus/tables/cache-index50.csv` and
//! `docs/audio/opus/tables/cache-bits50.csv` (see the `.meta`
//! sidecars for the canonical layout). The values are reproduced
//! inline so the cache is available without filesystem I/O at runtime.

use crate::celt_band_layout::CELT_NUM_BANDS;

/// Number of frame-size (`LM`) columns indexing [`CACHE_INDEX50`].
///
/// `LM ∈ 0..=4` covers the 2.5 / 5 / 10 / 20 ms frames plus the
/// short-block transient variant (RFC 6716 §4.3.4; the run-format
/// trace §2). The cache is keyed on five LM values, unlike the
/// four-value `LM ∈ 0..=3` axis of [`crate::celt_cache_caps50`].
pub const CACHE_LM_COUNT: usize = 5;

/// Total entries in [`CACHE_INDEX50`]: `21 × 5 = 105`.
pub const CACHE_INDEX_LEN: usize = CELT_NUM_BANDS * CACHE_LM_COUNT;

/// Total bytes in [`CACHE_BITS50`].
pub const CACHE_BITS_LEN: usize = 392;

/// Sentinel value in [`CACHE_INDEX50`]: the `(band, LM)` tuple has no
/// cached cost curve and the allocator must use its closed-form path
/// (run-format trace §2 / §5).
pub const CACHE_INDEX_SENTINEL: i16 = -1;

/// Upper bound on a single run's `maxK` (run-format trace §4: the four
/// largest runs cap at 40).
pub const CACHE_MAX_PULSES: u8 = 40;

/// §4.3.4.1 `cache_index50`: maps the LM-major `(lm_row, band)` flat
/// index to a byte offset into [`CACHE_BITS50`], or
/// [`CACHE_INDEX_SENTINEL`].
///
/// Layout: `i = lm_row * CELT_NUM_BANDS + band` with `lm_row = LM + 1`
/// — five rows of 21 bands (the corrected §2.1 mapping of the staged
/// pulse-cache format trace). The first 21 entries are the `LM = -1`
/// split-level row; its eight `-1` sentinels are bands 0–7 (one-bin
/// bands whose `LM = -1` half is empty).
///
/// Numeric facts from `docs/audio/opus/tables/cache-index50.csv`.
pub static CACHE_INDEX50: [i16; CACHE_INDEX_LEN] = [
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 41, 41, 41, 82, 82, 123, 164, 200, 222, 0, 0, 0, 0,
    0, 0, 0, 0, 41, 41, 41, 41, 123, 123, 123, 164, 164, 240, 266, 283, 295, 41, 41, 41, 41, 41,
    41, 41, 41, 123, 123, 123, 123, 240, 240, 240, 266, 266, 305, 318, 328, 336, 123, 123, 123,
    123, 123, 123, 123, 123, 240, 240, 240, 240, 305, 305, 305, 318, 318, 343, 351, 358, 364, 240,
    240, 240, 240, 240, 240, 240, 240, 305, 305, 305, 305, 343, 343, 343, 351, 351, 370, 376, 382,
    387,
];

/// §4.3.4.1 `cache_bits50`: run-packed PVQ cost curves.
///
/// Each run is `[maxK, qbits[1], …, qbits[maxK]]` in 1/8 bits (Q3).
/// Walk a run via the offset stored in [`CACHE_INDEX50`]; see the
/// module docs for the run format. The 23 distinct runs pack into
/// exactly [`CACHE_BITS_LEN`] bytes.
///
/// Numeric facts from `docs/audio/opus/tables/cache-bits50.csv`.
pub static CACHE_BITS50: [u8; CACHE_BITS_LEN] = [
    40, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 40, 15, 23, 28, 31, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47,
    47, 49, 50, 51, 52, 53, 54, 55, 55, 57, 58, 59, 60, 61, 62, 63, 63, 65, 66, 67, 68, 69, 70, 71,
    71, 40, 20, 33, 41, 48, 53, 57, 61, 64, 66, 69, 71, 73, 75, 76, 78, 80, 82, 85, 87, 89, 91, 92,
    94, 96, 98, 101, 103, 105, 107, 108, 110, 112, 114, 117, 119, 121, 123, 124, 126, 128, 40, 23,
    39, 51, 60, 67, 73, 79, 83, 87, 91, 94, 97, 100, 102, 105, 107, 111, 115, 118, 121, 124, 126,
    129, 131, 135, 139, 142, 145, 148, 150, 153, 155, 159, 163, 166, 169, 172, 174, 177, 179, 35,
    28, 49, 65, 78, 89, 99, 107, 114, 120, 126, 132, 136, 141, 145, 149, 153, 159, 165, 171, 176,
    180, 185, 189, 192, 199, 205, 211, 216, 220, 225, 229, 232, 239, 245, 251, 21, 33, 58, 79, 97,
    112, 125, 137, 148, 157, 166, 174, 182, 189, 195, 201, 207, 217, 227, 235, 243, 251, 17, 35,
    63, 86, 106, 123, 139, 152, 165, 177, 187, 197, 206, 214, 222, 230, 237, 250, 25, 31, 55, 75,
    91, 105, 117, 128, 138, 146, 154, 161, 168, 174, 180, 185, 190, 200, 208, 215, 222, 229, 235,
    240, 245, 255, 16, 36, 65, 89, 110, 128, 144, 159, 173, 185, 196, 207, 217, 226, 234, 242, 250,
    11, 41, 74, 103, 128, 151, 172, 191, 209, 225, 241, 255, 9, 43, 79, 110, 138, 163, 186, 207,
    227, 246, 12, 39, 71, 99, 123, 144, 164, 182, 198, 214, 228, 241, 253, 9, 44, 81, 113, 142,
    168, 192, 214, 235, 255, 7, 49, 90, 127, 160, 191, 220, 247, 6, 51, 95, 134, 170, 203, 234, 7,
    47, 87, 123, 155, 184, 212, 237, 6, 52, 97, 137, 174, 208, 240, 5, 57, 106, 151, 192, 231, 5,
    59, 111, 158, 202, 243, 5, 55, 103, 147, 187, 224, 5, 60, 113, 161, 206, 248, 4, 65, 122, 175,
    224, 4, 67, 127, 182, 234,
];

/// Errors returned by the §4.3.4.1 cache lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PulseCacheError {
    /// `band` is outside `0..21` (RFC 6716 §4.3 Table 55 band count).
    BandOutOfRange { band: usize },
    /// `lm` is outside `0..5` (the cache's five-column LM axis).
    LmOutOfRange { lm: usize },
    /// The `(band, LM)` tuple maps to [`CACHE_INDEX_SENTINEL`]: it has
    /// no cached cost curve and the caller must use the §4.3.4.1
    /// closed-form path.
    SentinelTuple { band: usize, lm: usize },
    /// `k` is outside the run's stored `1..=maxK` cost curve (`k = 0`
    /// has implicit cost 0 and is never stored; `k > maxK` exceeds the
    /// run's supported pulse count).
    PulseCountOutOfRange { k: u8, max_k: u8 },
}

/// Resolve the LM-major `(band, lm_row)` flat index into
/// [`CACHE_INDEX50`] (RFC 6716 §4.3.4.1; run-format trace §2.1,
/// corrected mapping).
///
/// `lm` here is the **row index** `LM + 1 ∈ 0..5` (row 0 is the
/// `LM = -1` split level; rows 1–4 the whole-band 2.5/5/10/20 ms
/// levels).
///
/// Returns [`PulseCacheError::BandOutOfRange`] / `LmOutOfRange` for
/// out-of-range inputs.
pub const fn cache_flat_index(band: usize, lm: usize) -> Result<usize, PulseCacheError> {
    if band >= CELT_NUM_BANDS {
        return Err(PulseCacheError::BandOutOfRange { band });
    }
    if lm >= CACHE_LM_COUNT {
        return Err(PulseCacheError::LmOutOfRange { lm });
    }
    Ok(lm * CELT_NUM_BANDS + band)
}

/// Return the [`CACHE_BITS50`] byte offset for a `(band, LM)` tuple, or
/// the sentinel error.
///
/// Returns [`PulseCacheError::SentinelTuple`] when the index entry is
/// [`CACHE_INDEX_SENTINEL`] (the caller must route to its closed-form
/// path), or the range errors from [`cache_flat_index`].
pub const fn cache_run_offset(band: usize, lm: usize) -> Result<usize, PulseCacheError> {
    let i = match cache_flat_index(band, lm) {
        Ok(i) => i,
        Err(e) => return Err(e),
    };
    let off = CACHE_INDEX50[i];
    if off == CACHE_INDEX_SENTINEL {
        return Err(PulseCacheError::SentinelTuple { band, lm });
    }
    Ok(off as usize)
}

/// Return the maximum pulse count `maxK` the run for `(band, LM)`
/// supports (the run's leading byte).
///
/// Returns the same errors as [`cache_run_offset`].
pub const fn cache_max_pulses(band: usize, lm: usize) -> Result<u8, PulseCacheError> {
    let off = match cache_run_offset(band, lm) {
        Ok(off) => off,
        Err(e) => return Err(e),
    };
    Ok(CACHE_BITS50[off])
}

/// Return the §4.3.4.1 cost `qbits[k]` (1/8 bits) of coding exactly `k`
/// pulses for `(band, LM)`.
///
/// `k` must be in `1..=maxK`. `k = 0` has implicit cost 0 (no codeword)
/// and is rejected. Returns [`PulseCacheError::SentinelTuple`] for a
/// sentinel tuple, or [`PulseCacheError::PulseCountOutOfRange`] when
/// `k` is 0 or exceeds the run's `maxK`.
pub const fn cache_pulse_cost(band: usize, lm: usize, k: u8) -> Result<u8, PulseCacheError> {
    let off = match cache_run_offset(band, lm) {
        Ok(off) => off,
        Err(e) => return Err(e),
    };
    let max_k = CACHE_BITS50[off];
    if k == 0 || k > max_k {
        return Err(PulseCacheError::PulseCountOutOfRange { k, max_k });
    }
    Ok(CACHE_BITS50[off + k as usize])
}

/// §4.3.4.1 *Bits to Pulses* inversion: the largest pulse count `K`
/// whose cost fits the per-band budget `b_target` (1/8 bits).
///
/// Scans the run's monotone cost curve `qbits[1..=maxK]` and returns
/// the largest `K` with `qbits[K] <= b_target`, or `0` when not even a
/// single pulse fits. `K = maxK` is allowed (the whole run fits).
///
/// Returns [`PulseCacheError::SentinelTuple`] for a sentinel tuple
/// (the caller uses its closed-form path), or the range errors from
/// [`cache_flat_index`].
pub const fn bits_to_pulses(band: usize, lm: usize, b_target: u8) -> Result<u8, PulseCacheError> {
    let off = match cache_run_offset(band, lm) {
        Ok(off) => off,
        Err(e) => return Err(e),
    };
    let max_k = CACHE_BITS50[off];
    let mut k: u8 = 0;
    let mut probe: u8 = 1;
    while probe <= max_k {
        if CACHE_BITS50[off + probe as usize] > b_target {
            break;
        }
        k = probe;
        probe += 1;
    }
    Ok(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_table_length_matches_band_lm_grid() {
        assert_eq!(CACHE_INDEX50.len(), CACHE_INDEX_LEN);
        assert_eq!(CACHE_INDEX_LEN, CELT_NUM_BANDS * CACHE_LM_COUNT);
        assert_eq!(CACHE_INDEX_LEN, 105);
    }

    #[test]
    fn bits_table_length_is_392() {
        assert_eq!(CACHE_BITS50.len(), CACHE_BITS_LEN);
        assert_eq!(CACHE_BITS_LEN, 392);
    }

    #[test]
    fn exactly_eight_sentinels_in_row_zero_bands_zero_through_seven() {
        let sentinels: Vec<usize> = (0..CACHE_INDEX_LEN)
            .filter(|&i| CACHE_INDEX50[i] == CACHE_INDEX_SENTINEL)
            .collect();
        assert_eq!(sentinels.len(), 8);
        // LM-major layout (trace §2.1): flat indices 0..=7 are row 0
        // (the LM = -1 split level), bands 0..=7 — the one-bin bands
        // whose halved width is empty.
        assert_eq!(sentinels, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        for band in 0..8 {
            assert_eq!(cache_flat_index(band, 0), Ok(band));
        }
    }

    #[test]
    fn sentinel_lookup_returns_sentinel_error() {
        // Row 0 (LM = -1), bands 0..=7 are the sentinels.
        assert_eq!(
            cache_run_offset(0, 0),
            Err(PulseCacheError::SentinelTuple { band: 0, lm: 0 })
        );
        assert_eq!(
            cache_run_offset(7, 0),
            Err(PulseCacheError::SentinelTuple { band: 7, lm: 0 })
        );
        // Band 8 of row 0 (width-2 band halved to N = 1) is the first
        // non-sentinel and points at the flat N = 1 run.
        assert_eq!(cache_run_offset(8, 0), Ok(0));
        // Band 1 at whole-band rows is never a sentinel; at row 3
        // (LM = 2) it uses the run at offset 123.
        assert_eq!(cache_run_offset(1, 3), Ok(123));
    }

    #[test]
    fn each_run_serves_exactly_one_effective_band_size() {
        // Trace §2.3 data-internal check: with the LM-major mapping,
        // every distinct run offset corresponds to exactly one
        // effective size N = (width << lm_row) >> 1, and the sentinel
        // appears exactly when N = 0.
        use crate::celt_band_layout::{celt_band_bins_per_channel, CeltFrameSize};
        use std::collections::HashMap;
        let mut n_for_offset: HashMap<usize, usize> = HashMap::new();
        for lm_row in 0..CACHE_LM_COUNT {
            for band in 0..CELT_NUM_BANDS {
                let width =
                    celt_band_bins_per_channel(band, CeltFrameSize::Ms2_5).unwrap() as usize;
                let n = (width << lm_row) >> 1;
                let flat = cache_flat_index(band, lm_row).unwrap();
                let off = CACHE_INDEX50[flat];
                if n == 0 {
                    assert_eq!(off, CACHE_INDEX_SENTINEL, "band {band} row {lm_row}");
                } else {
                    assert_ne!(off, CACHE_INDEX_SENTINEL, "band {band} row {lm_row}");
                    let prev = n_for_offset.insert(off as usize, n);
                    if let Some(prev_n) = prev {
                        assert_eq!(prev_n, n, "offset {off} serves two sizes");
                    }
                }
            }
        }
        assert_eq!(n_for_offset.len(), 23);
    }

    #[test]
    fn twenty_three_distinct_run_offsets() {
        let mut offsets: Vec<i16> = CACHE_INDEX50
            .iter()
            .copied()
            .filter(|&o| o != CACHE_INDEX_SENTINEL)
            .collect();
        offsets.sort_unstable();
        offsets.dedup();
        assert_eq!(offsets.len(), 23);
        assert_eq!(
            offsets,
            vec![
                0, 41, 82, 123, 164, 200, 222, 240, 266, 283, 295, 305, 318, 328, 336, 343, 351,
                358, 364, 370, 376, 382, 387,
            ]
        );
    }

    #[test]
    fn runs_pack_every_byte_exactly() {
        // Walking each distinct run by its leading maxK byte must cover
        // all 392 bytes with no gaps or overlaps.
        let mut offsets: Vec<usize> = CACHE_INDEX50
            .iter()
            .copied()
            .filter(|&o| o != CACHE_INDEX_SENTINEL)
            .map(|o| o as usize)
            .collect();
        offsets.sort_unstable();
        offsets.dedup();
        let mut total = 0usize;
        for &off in &offsets {
            let max_k = CACHE_BITS50[off] as usize;
            total += 1 + max_k;
        }
        assert_eq!(total, CACHE_BITS_LEN);
    }

    #[test]
    fn each_run_cost_curve_is_monotone_nondecreasing() {
        let mut offsets: Vec<usize> = CACHE_INDEX50
            .iter()
            .copied()
            .filter(|&o| o != CACHE_INDEX_SENTINEL)
            .map(|o| o as usize)
            .collect();
        offsets.sort_unstable();
        offsets.dedup();
        for &off in &offsets {
            let max_k = CACHE_BITS50[off] as usize;
            for k in 2..=max_k {
                assert!(
                    CACHE_BITS50[off + k] >= CACHE_BITS50[off + k - 1],
                    "run at {off} not monotone at k={k}"
                );
            }
        }
    }

    #[test]
    fn max_pulses_caps_at_forty() {
        let mut offsets: Vec<usize> = CACHE_INDEX50
            .iter()
            .copied()
            .filter(|&o| o != CACHE_INDEX_SENTINEL)
            .map(|o| o as usize)
            .collect();
        offsets.sort_unstable();
        offsets.dedup();
        for &off in &offsets {
            assert!(CACHE_BITS50[off] <= CACHE_MAX_PULSES);
        }
    }

    #[test]
    fn first_run_is_flat_seven() {
        // Run at offset 0 (the N = 1 profile; e.g. band 0 at row 1,
        // LM = 0): maxK = 40, qbits[1..=40] = 7.
        assert_eq!(cache_max_pulses(0, 1), Ok(40));
        for k in 1..=40u8 {
            assert_eq!(cache_pulse_cost(0, 1, k), Ok(7));
        }
    }

    #[test]
    fn pulse_cost_rejects_zero_and_overflow_k() {
        // Run at offset 0 has maxK = 40.
        assert_eq!(
            cache_pulse_cost(0, 1, 0),
            Err(PulseCacheError::PulseCountOutOfRange { k: 0, max_k: 40 })
        );
        assert_eq!(
            cache_pulse_cost(0, 1, 41),
            Err(PulseCacheError::PulseCountOutOfRange { k: 41, max_k: 40 })
        );
    }

    #[test]
    fn bits_to_pulses_flat_run_fits_all_at_budget_seven() {
        // Flat run (qbits all 7): a budget of 7 fits all 40 pulses.
        assert_eq!(bits_to_pulses(0, 1, 7), Ok(40));
        // A budget of 6 fits none (every cost is 7 > 6).
        assert_eq!(bits_to_pulses(0, 1, 6), Ok(0));
    }

    #[test]
    fn bits_to_pulses_picks_exact_threshold() {
        // Run at offset 41 (the N = 2 profile; band 2 at row 2, LM =
        // 1): qbits[1]=15, qbits[2]=23,
        // qbits[3]=28, ... A budget of 23 should fit exactly K=2.
        assert_eq!(cache_pulse_cost(2, 2, 1), Ok(15));
        assert_eq!(cache_pulse_cost(2, 2, 2), Ok(23));
        assert_eq!(cache_pulse_cost(2, 2, 3), Ok(28));
        assert_eq!(bits_to_pulses(2, 2, 23), Ok(2));
        // One below the K=2 cost fits only K=1.
        assert_eq!(bits_to_pulses(2, 2, 22), Ok(1));
        // One below the K=1 cost fits nothing.
        assert_eq!(bits_to_pulses(2, 2, 14), Ok(0));
    }

    #[test]
    fn bits_to_pulses_saturating_budget_returns_max_k() {
        // A budget of 255 (the max byte) fits the entire run.
        let max_k = cache_max_pulses(2, 2).unwrap();
        assert_eq!(bits_to_pulses(2, 2, 255), Ok(max_k));
    }

    #[test]
    fn bits_to_pulses_on_sentinel_signals_closed_form() {
        assert_eq!(
            bits_to_pulses(0, 0, 100),
            Err(PulseCacheError::SentinelTuple { band: 0, lm: 0 })
        );
    }

    #[test]
    fn lookup_rejects_out_of_range_band_and_lm() {
        assert_eq!(
            cache_flat_index(CELT_NUM_BANDS, 0),
            Err(PulseCacheError::BandOutOfRange {
                band: CELT_NUM_BANDS
            })
        );
        assert_eq!(
            cache_flat_index(0, CACHE_LM_COUNT),
            Err(PulseCacheError::LmOutOfRange { lm: CACHE_LM_COUNT })
        );
    }

    #[test]
    fn bits_to_pulses_monotone_in_budget() {
        // For a fixed non-sentinel tuple, K is non-decreasing in budget.
        let mut prev = 0u8;
        for b in 0..=255u8 {
            let k = bits_to_pulses(5, 4, b).unwrap();
            assert!(k >= prev, "K decreased at budget {b}");
            prev = k;
        }
        // At saturation it reaches the run's maxK.
        assert_eq!(prev, cache_max_pulses(5, 4).unwrap());
    }

    #[test]
    fn last_run_at_offset_387_has_max_k_four() {
        // Run at offset 387 (band 20 at row 4, LM = 3): maxK = 4,
        // qbits = [67, 127, 182, 234].
        assert_eq!(cache_run_offset(20, 4), Ok(387));
        assert_eq!(cache_max_pulses(20, 4), Ok(4));
        assert_eq!(cache_pulse_cost(20, 4, 1), Ok(67));
        assert_eq!(cache_pulse_cost(20, 4, 4), Ok(234));
    }
}
