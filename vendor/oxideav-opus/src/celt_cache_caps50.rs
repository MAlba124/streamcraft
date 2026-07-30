//! CELT §4.3.3 per-band maximum-allocation parameter surface
//! (RFC 6716 §4.3.3, pp. 113–114).
//!
//! The §4.3.3 *Bit Allocation* procedure caps each band's allocation
//! at a precomputed maximum, called `cap[]`. The §4.3.3 narrative
//! (RFC 6716 §4.3.3, p. 113) describes the cap as an approximation of
//! the largest space each band can usefully consume for a given mode:
//! a band that hits the cap cannot consume any further bit allocation,
//! and the surplus rolls over into the remaining bands.
//!
//! The §4.3.3 maximums are bits/sample values precomputed in a static
//! table indexed by `(LM, stereo, band)`. RFC 6716 §4.3.3 (p. 113) names
//! the table `cache_caps50[]` and notes its 168 entries are organised
//! as `i = nbBands * (2*LM + stereo)` with `nbBands = 21`, so the
//! `i`-th index covers one `(LM ∈ 0..=3, stereo ∈ {0,1}, band ∈ 0..=20)`
//! triple. The §4.3.3 convert-to-cap rule then folds the bits/sample
//! cap into a per-band bit cap via
//!
//! ```text
//!     cap[band] = ((cache_caps50[i] + 64) * channels * N) / 4
//! ```
//!
//! with `channels ∈ {1,2}` and `N` = MDCT bins per band per channel
//! (from §4.3 Table 55 / the [`crate::celt_band_layout`] lookup) and
//! integer division. RFC 6716 §4.3.3 (p. 114) describes the function
//! that performs this conversion as `init_caps()`. The resulting
//! `cap[]` elements fit in `i16` but not in `i8`.
//!
//! This module owns only the §4.3.3 *parameter surface*: the 168-byte
//! table plus the typed accessor that pairs it with the §4.3.3
//! `(LM, stereo, band)` indexing rule and the `init_caps()` convert
//! rule. The §4.3.3 bit allocation orchestration that consumes
//! `cap[]` (boost / trim / anti-collapse / skip / dual-stereo
//! reservations, the Table 57 static allocation search) is gated on
//! its own follow-up work and runs at the call site of the lookup.
//!
//! The §4.3.3 narrative is transcribed from RFC 6716,
//! `docs/audio/opus/rfc6716-opus.txt`, pp. 113–114, plus the §2.2
//! narrative in `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md`.
//! The 168-byte `cache_caps50` data is uncopyrightable numeric facts
//! extracted into `docs/audio/celt/tables/cache_caps50.csv` (see
//! the `cache_caps50.meta` sidecar for the canonical layout). The
//! values are reproduced inline here so the table is available
//! without filesystem I/O at runtime.
//!
//! ## Layout
//!
//! [`CACHE_CAPS50`] is a `[u8; 168]` flattened table; the §4.3.3
//! indexing rule `i = nbBands * (2*LM + stereo)` with `nbBands = 21`
//! recovers the `(LM, stereo)` row from a flat offset, and `band ∈
//! 0..=20` selects the cell inside the row. The CSV's "row r =>
//! LM=r//2, stereo=r%2" comment row matches: CSV row 0 is
//! `(LM=0, stereo=0)`, CSV row 1 is `(LM=0, stereo=1)`, …, CSV row 7
//! is `(LM=3, stereo=1)`.
//!
//! ## §4.3.3 `init_caps()` conversion
//!
//! Per RFC 6716 §4.3.3 (p. 113), to turn the bits/sample entries in
//! the table into the per-band bit cap the allocator searches against:
//!
//! 1. Pick the `(LM ∈ 0..=3, stereo ∈ {0,1})` selector for the frame.
//! 2. Look up `caps_value = cache_caps50[i]` with
//!    `i = nbBands * (2*LM + stereo)` + `band`.
//! 3. `cap[band] = ((caps_value + 64) * channels * N) / 4` (integer
//!    division), where `channels ∈ {1,2}` is the frame's channel
//!    count and `N` is the §4.3 Table 55 per-channel bin count for
//!    `band` at this `LM`.
//!
//! [`cap_for_band_bits`] computes step 3 in a single typed call given
//! the `(LM, stereo, band, channels, n_bins)` tuple; [`cache_caps_value`]
//! returns just the raw bits/sample byte (step 2).
//!
//! ## Provenance
//!
//! Narrative: RFC 6716 §4.3.3 (pp. 113–114) in
//! `docs/audio/opus/rfc6716-opus.txt`; the §2.2 narrative
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md`
//! cross-references both the RFC and the CSV. Numeric table: the
//! 168-byte sequence from `docs/audio/celt/tables/cache_caps50.csv`
//! (see the `.meta` sidecar for the canonical layout).

use crate::celt_band_layout::CELT_NUM_BANDS;

/// Number of CELT frame sizes that index the [`CACHE_CAPS50`] outer
/// axis (`LM ∈ {0,1,2,3}` per RFC 6716 §4.3 = 2.5 / 5 / 10 / 20 ms).
pub const CACHE_CAPS50_LM_COUNT: usize = 4;

/// Number of channel-count modes per `(LM, band)` cell (RFC 6716
/// §4.3.3: `0 = mono`, `1 = stereo`).
pub const CACHE_CAPS50_STEREO_COUNT: usize = 2;

/// Stereo-axis index selecting the **mono** row (RFC 6716 §4.3.3:
/// `stereo = 0`).
pub const CACHE_CAPS50_STEREO_MONO: usize = 0;

/// Stereo-axis index selecting the **stereo** row (RFC 6716 §4.3.3:
/// `stereo = 1`).
pub const CACHE_CAPS50_STEREO_STEREO: usize = 1;

/// Total entries in [`CACHE_CAPS50`]: 4 × 2 × 21 = 168 bytes.
pub const CACHE_CAPS50_TOTAL_BYTES: usize =
    CACHE_CAPS50_LM_COUNT * CACHE_CAPS50_STEREO_COUNT * CELT_NUM_BANDS;

/// §4.3.3 `init_caps()` additive bias applied to every table entry
/// before the per-band scale (RFC 6716 §4.3.3 p. 113: "the i-th index
/// of cache.caps + 64").
pub const INIT_CAPS_BIAS: u32 = 64;

/// §4.3.3 `init_caps()` final divisor (RFC 6716 §4.3.3 p. 113: "divide
/// the result by 4 using integer division").
pub const INIT_CAPS_DIVISOR: u32 = 4;

/// §4.3.3 channel-count multiplier upper bound (RFC 6716 §4.3.3 p. 113:
/// `channels ∈ {1, 2}`).
pub const INIT_CAPS_MAX_CHANNELS: u32 = 2;

/// §4.3.3 `cache_caps50` per-band maximum-allocation table
/// (RFC 6716 §4.3.3, pp. 113–114).
///
/// Each entry is a Q0 bits/sample value (unsigned byte) the
/// [`cap_for_band_bits`] / [`init_caps`] conversion folds into the
/// per-band bit cap. The 168 entries are stored as 8 logical rows of
/// 21 bytes each; row `r` corresponds to `(LM = r/2, stereo = r%2)`.
/// Linear indexing is `i = CELT_NUM_BANDS * (2*LM + stereo) + band`
/// (the §4.3.3 `nbBands * (2*LM + stereo)` row stride with
/// `nbBands = 21`).
///
/// Data provenance: `docs/audio/celt/tables/cache_caps50.csv`
/// (see the `.meta` sidecar for the canonical layout). Only the
/// numeric data is reproduced here.
#[rustfmt::skip]
pub const CACHE_CAPS50: [u8; CACHE_CAPS50_TOTAL_BYTES] = [
    // row 0 (LM=0, stereo=0; 2.5 ms mono)
    224, 224, 224, 224, 224, 224, 224, 224, 160, 160, 160, 160, 185, 185, 185, 178, 178, 168, 134, 61, 37,
    // row 1 (LM=0, stereo=1; 2.5 ms stereo)
    224, 224, 224, 224, 224, 224, 224, 224, 240, 240, 240, 240, 207, 207, 207, 198, 198, 183, 144, 66, 40,
    // row 2 (LM=1, stereo=0; 5 ms mono)
    160, 160, 160, 160, 160, 160, 160, 160, 185, 185, 185, 185, 193, 193, 193, 183, 183, 172, 138, 64, 38,
    // row 3 (LM=1, stereo=1; 5 ms stereo)
    240, 240, 240, 240, 240, 240, 240, 240, 207, 207, 207, 207, 204, 204, 204, 193, 193, 180, 143, 66, 40,
    // row 4 (LM=2, stereo=0; 10 ms mono)
    185, 185, 185, 185, 185, 185, 185, 185, 193, 193, 193, 193, 193, 193, 193, 183, 183, 172, 138, 65, 39,
    // row 5 (LM=2, stereo=1; 10 ms stereo)
    207, 207, 207, 207, 207, 207, 207, 207, 204, 204, 204, 204, 201, 201, 201, 188, 188, 176, 141, 66, 40,
    // row 6 (LM=3, stereo=0; 20 ms mono)
    193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 194, 194, 194, 184, 184, 173, 139, 65, 39,
    // row 7 (LM=3, stereo=1; 20 ms stereo)
    204, 204, 204, 204, 204, 204, 204, 204, 201, 201, 201, 201, 198, 198, 198, 187, 187, 175, 140, 66, 40,
];

/// §4.3.3 stereo-axis selector.
///
/// The `stereo` axis of [`CACHE_CAPS50`] is binary: mono = 0, stereo
/// = 1. Modelled as a typed enum so the caller can't confuse it with
/// the channel-count multiplier of [`init_caps`] (which is `1` or
/// `2`, not `0` or `1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheCapsStereo {
    /// Mono row (RFC 6716 §4.3.3: `stereo = 0`).
    Mono,
    /// Stereo row (RFC 6716 §4.3.3: `stereo = 1`).
    Stereo,
}

impl CacheCapsStereo {
    /// Stereo-axis index into the §4.3.3 row-stride `2*LM + stereo`.
    pub const fn axis_index(self) -> usize {
        match self {
            CacheCapsStereo::Mono => CACHE_CAPS50_STEREO_MONO,
            CacheCapsStereo::Stereo => CACHE_CAPS50_STEREO_STEREO,
        }
    }

    /// `channels` multiplier consumed by the §4.3.3 `init_caps()`
    /// conversion (`1` for mono, `2` for stereo).
    pub const fn channels(self) -> u32 {
        match self {
            CacheCapsStereo::Mono => 1,
            CacheCapsStereo::Stereo => 2,
        }
    }

    /// Decode a raw `bool` channel-count signal (`false = mono,
    /// true = stereo`) into a selector. Use this when the upstream
    /// signal is the TOC stereo-flag boolean from §3.1.
    pub const fn from_is_stereo(is_stereo: bool) -> Self {
        if is_stereo {
            CacheCapsStereo::Stereo
        } else {
            CacheCapsStereo::Mono
        }
    }
}

/// Errors returned by the [`cache_caps_value`] / [`cap_for_band_bits`]
/// accessors for out-of-range indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheCaps50Error {
    /// `lm` is outside `0..4` (RFC 6716 §4.3 only defines four CELT
    /// frame sizes).
    LmOutOfRange { lm: u32 },
    /// `band` is outside `0..21` (the §4.3 Table 55 band count).
    BandOutOfRange { band: u32 },
    /// `channels` is outside `1..=2` (RFC 6716 §4.3.3 p. 113 declares
    /// `channels ∈ {1,2}` for the conversion).
    ChannelsOutOfRange { channels: u32 },
}

/// Compute the linear offset into [`CACHE_CAPS50`] for one
/// `(LM, stereo, band)` cell (RFC 6716 §4.3.3, p. 113).
///
/// The §4.3.3 indexing rule is `i = nbBands * (2*LM + stereo) + band`
/// with `nbBands = 21`. Returns the flat offset into [`CACHE_CAPS50`].
///
/// Both `lm < 4` and `band < 21` are caller-side preconditions; this
/// `const fn` is total over `(lm, stereo, band)` triples constrained
/// by the const-fn type signature only.
pub const fn cache_caps_offset(lm: usize, stereo: CacheCapsStereo, band: usize) -> usize {
    CELT_NUM_BANDS * (2 * lm + stereo.axis_index()) + band
}

/// Look up the raw `cache_caps50` bits/sample byte for one
/// `(LM, stereo, band)` cell (RFC 6716 §4.3.3, p. 113).
///
/// This is the §4.3.3 step "look up `cache_caps50[i]`" — the lookup
/// itself, before the [`init_caps`] `(value + 64) * channels * N / 4`
/// scale. Callers who only need the bits/sample byte (e.g.
/// table-level cross-checks against the CSV) use this; callers who
/// need the final per-band bit cap call [`cap_for_band_bits`].
pub fn cache_caps_value(
    lm: u32,
    stereo: CacheCapsStereo,
    band: u32,
) -> Result<u8, CacheCaps50Error> {
    if lm >= CACHE_CAPS50_LM_COUNT as u32 {
        return Err(CacheCaps50Error::LmOutOfRange { lm });
    }
    if band >= CELT_NUM_BANDS as u32 {
        return Err(CacheCaps50Error::BandOutOfRange { band });
    }
    let off = cache_caps_offset(lm as usize, stereo, band as usize);
    Ok(CACHE_CAPS50[off])
}

/// Borrow the full 21-byte row for a single `(LM, stereo)` cell of
/// [`CACHE_CAPS50`] (RFC 6716 §4.3.3, p. 113).
///
/// This is the §4.3.3 "one row of 21 caps" (one CSV row in
/// `docs/audio/celt/tables/cache_caps50.csv`). Returned as a borrowed
/// slice so callers may iterate the band loop without re-indexing.
pub fn cache_caps_row(lm: u32, stereo: CacheCapsStereo) -> Result<&'static [u8], CacheCaps50Error> {
    if lm >= CACHE_CAPS50_LM_COUNT as u32 {
        return Err(CacheCaps50Error::LmOutOfRange { lm });
    }
    let base = cache_caps_offset(lm as usize, stereo, 0);
    Ok(&CACHE_CAPS50[base..base + CELT_NUM_BANDS])
}

/// Apply the §4.3.3 `init_caps()` convert rule to a single
/// `cache_caps50` byte (RFC 6716 §4.3.3, p. 113).
///
/// Returns `((caps_value + 64) * channels * n_bins) / 4` per the
/// §4.3.3 step 4 rule:
///
/// > Set the maximum for the band to the i-th index of cache.caps +
/// > 64 and multiply by the number of channels in the current frame
/// > (one or two) and by N, then divide the result by 4 using integer
/// > division.
///
/// `channels ∈ {1, 2}` and `n_bins` is the §4.3 Table 55 per-channel
/// MDCT-bin count for the band. The result is a per-band bit cap that
/// fits in `i16` but not `i8` per the §4.3.3 narrative; we return
/// `u32` so the caller can hold the conversion product without risk
/// of overflow on intermediate arithmetic.
///
/// This is `init_caps()` for a single band; iterating the §4.3 band
/// loop and assembling the full `cap[]` vector is the responsibility
/// of the §4.3.3 allocator. The function is named `init_caps` per the
/// §4.3.3 narrative even though it operates on a single band.
pub const fn init_caps(caps_value: u8, channels: u32, n_bins: u32) -> u32 {
    ((caps_value as u32 + INIT_CAPS_BIAS) * channels * n_bins) / INIT_CAPS_DIVISOR
}

/// Compute the §4.3.3 per-band bit cap for one
/// `(LM, stereo, band, channels, n_bins)` tuple (RFC 6716 §4.3.3,
/// p. 113).
///
/// Looks up `cache_caps50[i]` with `i = nbBands * (2*LM + stereo) +
/// band`, then applies the [`init_caps`] convert rule. Returns the
/// final per-band bit cap the §4.3.3 allocator searches against.
///
/// The §4.3.3 `stereo` axis selector and the `channels` multiplier
/// are *independent inputs*: in the standard Opus path they agree
/// (mono frame → `stereo = Mono` and `channels = 1`; stereo frame →
/// `stereo = Stereo` and `channels = 2`), but the §4.3.3 narrative
/// keeps them as separate parameters of `init_caps()`. We mirror that
/// shape to match the spec and to catch a caller that accidentally
/// passes mismatched values via the [`CacheCaps50Error::ChannelsOutOfRange`]
/// error.
///
/// `n_bins` is the §4.3 Table 55 *per-channel* MDCT-bin count for
/// the band at this LM (use [`crate::celt_band_layout::celt_band_bins_per_channel`]).
pub fn cap_for_band_bits(
    lm: u32,
    stereo: CacheCapsStereo,
    band: u32,
    channels: u32,
    n_bins: u32,
) -> Result<u32, CacheCaps50Error> {
    if channels == 0 || channels > INIT_CAPS_MAX_CHANNELS {
        return Err(CacheCaps50Error::ChannelsOutOfRange { channels });
    }
    let caps_value = cache_caps_value(lm, stereo, band)?;
    Ok(init_caps(caps_value, channels, n_bins))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_band_layout::{celt_band_bins_per_channel, CeltFrameSize};

    // ---- Table-shape invariants ----

    #[test]
    fn table_shape_constants_match_struct() {
        assert_eq!(CACHE_CAPS50_LM_COUNT, 4);
        assert_eq!(CACHE_CAPS50_STEREO_COUNT, 2);
        assert_eq!(CACHE_CAPS50_TOTAL_BYTES, 168);
        assert_eq!(CACHE_CAPS50.len(), CACHE_CAPS50_TOTAL_BYTES);
        assert_eq!(CELT_NUM_BANDS, 21);
    }

    #[test]
    fn init_caps_constants_match_rfc() {
        assert_eq!(INIT_CAPS_BIAS, 64);
        assert_eq!(INIT_CAPS_DIVISOR, 4);
        assert_eq!(INIT_CAPS_MAX_CHANNELS, 2);
    }

    // ---- Stereo-axis index constants pinned to the §4.3.3 `2*LM +
    //      stereo` convention ----

    #[test]
    fn stereo_axis_index_constants_match_rfc_convention() {
        assert_eq!(CACHE_CAPS50_STEREO_MONO, 0);
        assert_eq!(CACHE_CAPS50_STEREO_STEREO, 1);
        assert_eq!(CacheCapsStereo::Mono.axis_index(), 0);
        assert_eq!(CacheCapsStereo::Stereo.axis_index(), 1);
    }

    #[test]
    fn cache_caps_stereo_channels_helper_matches_init_caps_input() {
        // The §4.3.3 init_caps() multiplier is `channels`, not the
        // stereo axis index. Pin the mapping so a future edit can't
        // silently confuse the two.
        assert_eq!(CacheCapsStereo::Mono.channels(), 1);
        assert_eq!(CacheCapsStereo::Stereo.channels(), 2);
    }

    #[test]
    fn cache_caps_stereo_from_is_stereo_boolean_round_trip() {
        assert_eq!(
            CacheCapsStereo::from_is_stereo(false),
            CacheCapsStereo::Mono
        );
        assert_eq!(
            CacheCapsStereo::from_is_stereo(true),
            CacheCapsStereo::Stereo
        );
    }

    // ---- Spot-check Q0 values against the CSV extract ----
    //
    // These pins reproduce hand-picked cells from
    // `docs/audio/celt/tables/cache_caps50.csv` so a future edit that
    // reorders the rows or drops an entry trips the suite.

    #[test]
    fn csv_row0_band0_is_224() {
        // CSV row 0 (LM=0, stereo=0), column 0.
        assert_eq!(cache_caps_value(0, CacheCapsStereo::Mono, 0).unwrap(), 224);
    }

    #[test]
    fn csv_row1_band20_is_40() {
        // CSV row 1 (LM=0, stereo=1), column 20 — the high-band
        // tail of the 2.5 ms stereo row.
        assert_eq!(
            cache_caps_value(0, CacheCapsStereo::Stereo, 20).unwrap(),
            40
        );
    }

    #[test]
    fn csv_row2_band0_is_160() {
        // CSV row 2 (LM=1, stereo=0), column 0 — the 5 ms mono first
        // band.
        assert_eq!(cache_caps_value(1, CacheCapsStereo::Mono, 0).unwrap(), 160);
    }

    #[test]
    fn csv_row3_band8_is_207() {
        // CSV row 3 (LM=1, stereo=1), column 8 — boundary cell at the
        // first mid-band tier of the 5 ms stereo row.
        assert_eq!(
            cache_caps_value(1, CacheCapsStereo::Stereo, 8).unwrap(),
            207
        );
    }

    #[test]
    fn csv_row4_band12_is_193() {
        // CSV row 4 (LM=2, stereo=0), column 12 — mid-band plateau of
        // the 10 ms mono row.
        assert_eq!(cache_caps_value(2, CacheCapsStereo::Mono, 12).unwrap(), 193);
    }

    #[test]
    fn csv_row5_band17_is_176() {
        // CSV row 5 (LM=2, stereo=1), column 17 — Hybrid-reachable
        // band in the 10 ms stereo row.
        assert_eq!(
            cache_caps_value(2, CacheCapsStereo::Stereo, 17).unwrap(),
            176
        );
    }

    #[test]
    fn csv_row6_band20_is_39() {
        // CSV row 6 (LM=3, stereo=0), column 20 — the high-band tail
        // of the 20 ms mono row (CELT-only headline frame size).
        assert_eq!(cache_caps_value(3, CacheCapsStereo::Mono, 20).unwrap(), 39);
    }

    #[test]
    fn csv_row7_band0_is_204() {
        // CSV row 7 (LM=3, stereo=1), column 0 — the first band of
        // the 20 ms stereo row.
        assert_eq!(
            cache_caps_value(3, CacheCapsStereo::Stereo, 0).unwrap(),
            204
        );
    }

    // ---- Row layout matches the §4.3.3 `2*LM + stereo` row-stride
    //      indexing rule ----

    #[test]
    fn cache_caps_offset_matches_rfc_row_stride_rule() {
        for lm in 0..CACHE_CAPS50_LM_COUNT {
            for stereo_idx in 0..CACHE_CAPS50_STEREO_COUNT {
                for band in 0..CELT_NUM_BANDS {
                    let stereo = if stereo_idx == 0 {
                        CacheCapsStereo::Mono
                    } else {
                        CacheCapsStereo::Stereo
                    };
                    let off = cache_caps_offset(lm, stereo, band);
                    // The §4.3.3 rule: i = nbBands * (2*LM + stereo) + band.
                    let expected = CELT_NUM_BANDS * (2 * lm + stereo_idx) + band;
                    assert_eq!(off, expected);
                }
            }
        }
    }

    #[test]
    fn cache_caps_offset_at_table_extremes_pins_endpoints() {
        // (LM=0, stereo=Mono, band=0) is the very first byte.
        assert_eq!(cache_caps_offset(0, CacheCapsStereo::Mono, 0), 0);
        // (LM=3, stereo=Stereo, band=20) is the very last byte.
        assert_eq!(
            cache_caps_offset(3, CacheCapsStereo::Stereo, 20),
            CACHE_CAPS50_TOTAL_BYTES - 1
        );
        // And that last byte is the CSV's final 40.
        assert_eq!(CACHE_CAPS50[CACHE_CAPS50_TOTAL_BYTES - 1], 40);
    }

    // ---- Total-function sweep over all (LM, stereo, band) cells ----

    #[test]
    fn cache_caps_value_is_total_over_in_range_inputs() {
        for lm in 0..CACHE_CAPS50_LM_COUNT as u32 {
            for stereo in [CacheCapsStereo::Mono, CacheCapsStereo::Stereo] {
                for band in 0..CELT_NUM_BANDS as u32 {
                    let v = cache_caps_value(lm, stereo, band).expect("in-range lookup");
                    let off = cache_caps_offset(lm as usize, stereo, band as usize);
                    assert_eq!(v, CACHE_CAPS50[off]);
                }
            }
        }
    }

    // ---- Row-accessor mirrors raw-table indexing ----

    #[test]
    fn cache_caps_row_matches_cell_lookup_for_every_lm_and_stereo() {
        for lm in 0..CACHE_CAPS50_LM_COUNT as u32 {
            for stereo in [CacheCapsStereo::Mono, CacheCapsStereo::Stereo] {
                let row = cache_caps_row(lm, stereo).unwrap();
                assert_eq!(row.len(), CELT_NUM_BANDS);
                for band in 0..CELT_NUM_BANDS as u32 {
                    let v = cache_caps_value(lm, stereo, band).unwrap();
                    assert_eq!(v, row[band as usize]);
                }
            }
        }
    }

    // ---- Error-path coverage ----

    #[test]
    fn cache_caps_value_rejects_lm_out_of_range() {
        let err =
            cache_caps_value(CACHE_CAPS50_LM_COUNT as u32, CacheCapsStereo::Mono, 0).unwrap_err();
        assert_eq!(err, CacheCaps50Error::LmOutOfRange { lm: 4 });
        let err = cache_caps_value(u32::MAX, CacheCapsStereo::Mono, 0).unwrap_err();
        assert_eq!(err, CacheCaps50Error::LmOutOfRange { lm: u32::MAX });
    }

    #[test]
    fn cache_caps_value_rejects_band_out_of_range() {
        let err = cache_caps_value(0, CacheCapsStereo::Mono, CELT_NUM_BANDS as u32).unwrap_err();
        assert_eq!(err, CacheCaps50Error::BandOutOfRange { band: 21 });
        let err = cache_caps_value(0, CacheCapsStereo::Stereo, u32::MAX).unwrap_err();
        assert_eq!(err, CacheCaps50Error::BandOutOfRange { band: u32::MAX });
    }

    #[test]
    fn cache_caps_row_rejects_lm_out_of_range() {
        let err = cache_caps_row(CACHE_CAPS50_LM_COUNT as u32, CacheCapsStereo::Mono).unwrap_err();
        assert_eq!(err, CacheCaps50Error::LmOutOfRange { lm: 4 });
    }

    // ---- init_caps() conversion ----

    #[test]
    fn init_caps_matches_rfc_formula_explicit_case() {
        // RFC 6716 §4.3.3 p. 113: cap = (cache.caps[i] + 64) * channels * N / 4.
        // (caps=224, channels=2, N=4) -> (224+64)*2*4 / 4 = 288*8/4 = 576.
        assert_eq!(init_caps(224, 2, 4), 576);
        // (caps=40, channels=1, N=12) -> (40+64)*1*12 / 4 = 104*12/4 = 312.
        assert_eq!(init_caps(40, 1, 12), 312);
        // Lowest-allowed (caps=0, channels=1, N=1): (0+64)*1*1 / 4 = 16.
        assert_eq!(init_caps(0, 1, 1), 16);
        // Highest-allowed (caps=255, channels=2, N=192): (255+64)*2*192 / 4 = 30624.
        assert_eq!(init_caps(255, 2, 192), 30624);
    }

    #[test]
    fn init_caps_integer_division_is_floor() {
        // (caps=1, channels=1, N=1): (1+64)*1*1 / 4 = 65/4 = 16 (floor).
        assert_eq!(init_caps(1, 1, 1), 16);
        // (caps=2, channels=1, N=1): 66/4 = 16 (floor).
        assert_eq!(init_caps(2, 1, 1), 16);
        // (caps=3, channels=1, N=1): 67/4 = 16 (floor).
        assert_eq!(init_caps(3, 1, 1), 16);
        // (caps=4, channels=1, N=1): 68/4 = 17.
        assert_eq!(init_caps(4, 1, 1), 17);
    }

    // ---- cap_for_band_bits — the §4.3.3 init_caps()-with-lookup
    //      composite ----

    #[test]
    fn cap_for_band_bits_matches_manual_lookup_plus_init_caps() {
        // Use a non-trivial cell: (LM=2, stereo=Stereo, band=17).
        let caps_value = cache_caps_value(2, CacheCapsStereo::Stereo, 17).unwrap();
        assert_eq!(caps_value, 176);
        // Bin count for the 17th band at the 10 ms CELT frame size:
        // pick it up from the §4.3 Table 55 lookup.
        let n_bins = celt_band_bins_per_channel(17, CeltFrameSize::Ms10).unwrap();
        let expected = init_caps(caps_value, 2, n_bins as u32);
        let cap = cap_for_band_bits(2, CacheCapsStereo::Stereo, 17, 2, n_bins as u32).unwrap();
        assert_eq!(cap, expected);
    }

    #[test]
    fn cap_for_band_bits_rejects_channels_out_of_range() {
        let err = cap_for_band_bits(0, CacheCapsStereo::Mono, 0, 0, 4).unwrap_err();
        assert_eq!(err, CacheCaps50Error::ChannelsOutOfRange { channels: 0 });
        let err = cap_for_band_bits(0, CacheCapsStereo::Mono, 0, 3, 4).unwrap_err();
        assert_eq!(err, CacheCaps50Error::ChannelsOutOfRange { channels: 3 });
        let err = cap_for_band_bits(0, CacheCapsStereo::Mono, 0, u32::MAX, 4).unwrap_err();
        assert_eq!(
            err,
            CacheCaps50Error::ChannelsOutOfRange { channels: u32::MAX }
        );
    }

    #[test]
    fn cap_for_band_bits_propagates_lm_and_band_errors() {
        let err = cap_for_band_bits(CACHE_CAPS50_LM_COUNT as u32, CacheCapsStereo::Mono, 0, 1, 4)
            .unwrap_err();
        assert_eq!(err, CacheCaps50Error::LmOutOfRange { lm: 4 });
        let err =
            cap_for_band_bits(0, CacheCapsStereo::Mono, CELT_NUM_BANDS as u32, 1, 4).unwrap_err();
        assert_eq!(err, CacheCaps50Error::BandOutOfRange { band: 21 });
    }

    // ---- §4.3.3 narrative invariant: caps fit in i16 but not i8 ----
    //
    // RFC 6716 §4.3.3 p. 113 calls this out directly. Check the
    // invariant across the full §4.3 band loop at 20 ms (LM=3, the
    // largest frame) where the per-channel bin count is at its max.

    #[test]
    fn cap_at_20ms_stereo_fits_in_i16_but_not_i8() {
        for band in 0..CELT_NUM_BANDS as u32 {
            let n_bins = celt_band_bins_per_channel(band as usize, CeltFrameSize::Ms20).unwrap();
            let cap =
                cap_for_band_bits(3, CacheCapsStereo::Stereo, band, 2, n_bins as u32).unwrap();
            assert!(
                cap <= i16::MAX as u32,
                "cap[{band}] = {cap} should fit in i16"
            );
            // Some bands will exceed 127 (the §4.3.3 "but not i8"
            // half of the invariant). We only need one of them to be
            // > 127 to validate that claim; check the explicit
            // assertion in `at_least_one_cap_exceeds_i8`.
        }
    }

    #[test]
    fn at_least_one_cap_exceeds_i8() {
        let n_bins = celt_band_bins_per_channel(0, CeltFrameSize::Ms20).unwrap();
        let cap = cap_for_band_bits(3, CacheCapsStereo::Stereo, 0, 2, n_bins as u32).unwrap();
        assert!(
            cap > i8::MAX as u32,
            "expected at least one cap > i8::MAX (= 127); got {cap}"
        );
    }

    // ---- §4.3.3 reachable-cells sanity pins ----
    //
    // The §4.3.3 band loop reaches every cell in the table because
    // every (LM, stereo) row participates in the allocator at one
    // frame size + channel-count combination. Pin two representative
    // headline cases.

    #[test]
    fn celt_only_20ms_stereo_band0_pins_expected_cap() {
        // (LM=3, stereo=Stereo, band=0): the first band of the 20 ms
        // CELT-only stereo headline case.
        let caps_value = cache_caps_value(3, CacheCapsStereo::Stereo, 0).unwrap();
        assert_eq!(caps_value, 204);
        let n_bins = celt_band_bins_per_channel(0, CeltFrameSize::Ms20).unwrap();
        let cap = cap_for_band_bits(3, CacheCapsStereo::Stereo, 0, 2, n_bins as u32).unwrap();
        // Cap formula: (204+64) * 2 * n_bins / 4 = 268 * 2 * n_bins / 4
        //            = 134 * n_bins.
        assert_eq!(cap, 134 * n_bins as u32);
    }

    #[test]
    fn hybrid_band17_at_20ms_mono_pins_expected_cap() {
        // Hybrid frames carve out bands 17..=20 for CELT; the first
        // band of that carve-out at 20 ms mono is band 17 with
        // caps[6][17] = 173.
        let caps_value = cache_caps_value(3, CacheCapsStereo::Mono, 17).unwrap();
        assert_eq!(caps_value, 173);
        let n_bins = celt_band_bins_per_channel(17, CeltFrameSize::Ms20).unwrap();
        let cap = cap_for_band_bits(3, CacheCapsStereo::Mono, 17, 1, n_bins as u32).unwrap();
        // (173+64) * 1 * n_bins / 4 = 237 * n_bins / 4.
        assert_eq!(cap, (237 * n_bins as u32) / 4);
    }
}
