//! CELT MDCT-band layout — RFC 6716 §4.3, Table 55 + Hybrid offset.
//!
//! The §4.3 CELT decoder partitions the MDCT spectrum into a fixed
//! sequence of 21 bands whose widths roughly follow the Bark critical-
//! band scale (RFC 6716 §4.3, p. 103). Each band carries an integer
//! number of MDCT bins **per channel**, and that count depends only on
//! the CELT frame size (one of `{2.5, 5, 10, 20} ms`). The full
//! `band × frame_size → bins_per_channel` matrix is enumerated as the
//! normative Table 55 (RFC 6716 §4.3, p. 104). Every CELT-layer
//! sub-decoder downstream of this module (§4.3.2 coarse energy, §4.3.3
//! bit allocator, §4.3.4 PVQ shape, §4.3.6 denormalisation, §4.3.7
//! inverse MDCT) needs Table 55 — they all iterate band-by-band and
//! ask "how many MDCT bins in band `b` at this frame size?".
//!
//! For Hybrid frames RFC 6716 §4.3 (p. 103) carves the first 17 bands
//! (`0..=16`, the `0..8 kHz` range covered by the SILK layer) out of
//! the CELT band loop entirely — the §4.3 CELT machinery only iterates
//! `17..=20` in Hybrid mode. That offset is [`HYBRID_FIRST_CODED_BAND`]
//! / [`celt_first_coded_band`]; for CELT-only frames the first coded
//! band is `0`.
//!
//! ## What this module does not own
//!
//! * The §4.3.2.1 coarse-energy Laplace decoder. Table 55 only tells
//!   the band loop how many MDCT bins each band carries; the §4.3.2
//!   energy quantisation runs on the per-band gain, not the bin count.
//! * The §4.3.3 bit allocator. The allocator consumes Table 55 (and a
//!   second table — `cache_caps50` / `LOG2_FRAC_TABLE` — that this
//!   module deliberately does NOT carry; that's the §4.3.3 blocker
//!   tracked in the round-20 prose).
//! * The §4.3.7 inverse MDCT itself. This module exposes the band
//!   layout (where the MDCT bins live); it does not perform the MDCT.
//! * Anything bitstream-level. Table 55 is a pure constant table; no
//!   range-decoder symbols are read here.
//!
//! ## Frame-size index encoding
//!
//! Table 55 has four "Bins:" columns: `2.5 ms`, `5 ms`, `10 ms`,
//! `20 ms`. We expose that as [`CeltFrameSize`], a 4-variant enum
//! whose discriminants give the column index `0..=3`. The §3.1 TOC
//! byte (RFC 6716 §3.1, Table 2) only directly signals `2.5 / 5 / 10 /
//! 20 / 40 / 60 ms` for the *Opus* frame; for `40 / 60 ms` SILK-only
//! frames there is no CELT layer and Table 55 doesn't apply. For
//! Hybrid frames the CELT layer always runs at the Opus frame size
//! (only `10 ms` and `20 ms` Hybrid configurations exist per Table 2:
//! configs 12 / 14 / 13 / 15 are SWB / FB at 10 / 20 ms). For CELT-only
//! frames the Opus frame size maps 1:1 onto [`CeltFrameSize`] across
//! `{2.5, 5, 10, 20} ms`. [`CeltFrameSize::from_frame_tenths_ms`]
//! handles both halves.
//!
//! ## Provenance
//!
//! All cell values, all band-boundary frequencies, and the "first 17
//! bands not coded in Hybrid" rule are transcribed from RFC 6716
//! (`docs/audio/opus/rfc6716-opus.txt`, §4.3 p. 103 prose + Table 55
//! p. 104). No external library source consulted, no cross-check
//! against any CELT reference implementation. The "Custom" mode of
//! §6.2 (which can use a different number of bands) is explicitly out
//! of scope and is rejected by every constructor.

/// Number of bands the standard (non-Custom) CELT layer partitions the
/// MDCT spectrum into. RFC 6716 §4.3, p. 103 ("The normal CELT layer
/// uses 21 of those bands").
pub const CELT_NUM_BANDS: usize = 21;

/// First CELT band index actually coded in Hybrid mode. RFC 6716 §4.3,
/// p. 103: "In Hybrid mode, the first 17 bands (up to 8 kHz) are not
/// coded." So bands `0..=16` are skipped and the §4.3 band loop runs
/// `17..=20`.
pub const HYBRID_FIRST_CODED_BAND: usize = 17;

/// Maximum number of MDCT bins per channel in any single band at any
/// frame size — the 20 ms × band 20 (`15600..=20000 Hz`) cell in
/// Table 55. RFC 6716 §4.3, p. 103 ("as many as 176 bins per channel").
pub const CELT_MAX_BINS_PER_BAND: u16 = 176;

/// CELT frame size, identifying one of the four Table 55 "Bins:"
/// columns. The discriminant equals the column index `0..=3` so it
/// doubles as a Table 55 lookup index.
///
/// Maps onto the Opus frame size carried in the §3.1 TOC byte for
/// CELT-only frames; for Hybrid frames the §3.1 frame size is the same
/// thing (Hybrid only exists at 10 ms and 20 ms — RFC 6716 §3.1,
/// Table 2 configs 12 / 13 / 14 / 15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CeltFrameSize {
    /// 2.5 ms — the first Table 55 "Bins:" column. CELT-only only;
    /// reachable via §3.1 Table 2 configs 16 / 20 / 24 / 28.
    Ms2_5 = 0,
    /// 5 ms — the second Table 55 "Bins:" column. CELT-only only;
    /// reachable via §3.1 Table 2 configs 17 / 21 / 25 / 29.
    Ms5 = 1,
    /// 10 ms — the third Table 55 "Bins:" column. Reachable in
    /// CELT-only (configs 18 / 22 / 26 / 30) and Hybrid (configs 12 /
    /// 14).
    Ms10 = 2,
    /// 20 ms — the fourth Table 55 "Bins:" column. Reachable in
    /// CELT-only (configs 19 / 23 / 27 / 31) and Hybrid (configs 13 /
    /// 15).
    Ms20 = 3,
}

impl CeltFrameSize {
    /// Table 55 column index `0..=3` for `self`.
    #[inline]
    pub const fn column_index(self) -> usize {
        self as usize
    }

    /// Decode an Opus frame size (in tenths-of-a-millisecond, the
    /// integer form [`crate::toc::OpusTocByte::frame_size_tenths_ms`]
    /// returns) into the matching CELT frame size, if the Opus frame
    /// has a CELT layer at all.
    ///
    /// Returns `None` for 40 ms (`tenths == 400`) and 60 ms
    /// (`tenths == 600`) frames — those are SILK-only per §3.1 Table 2
    /// and the CELT layer never runs.
    ///
    /// The caller is responsible for the SILK-only / Hybrid /
    /// CELT-only routing decision (see [`crate::framing::OpusFrameRouting`]);
    /// this entry point reports "no CELT" purely from the frame-size
    /// arithmetic.
    pub const fn from_frame_tenths_ms(tenths: u32) -> Option<Self> {
        match tenths {
            25 => Some(CeltFrameSize::Ms2_5),
            50 => Some(CeltFrameSize::Ms5),
            100 => Some(CeltFrameSize::Ms10),
            200 => Some(CeltFrameSize::Ms20),
            _ => None,
        }
    }

    /// CELT frame size in tenths-of-a-millisecond — the inverse of
    /// [`Self::from_frame_tenths_ms`].
    pub const fn to_frame_tenths_ms(self) -> u32 {
        match self {
            CeltFrameSize::Ms2_5 => 25,
            CeltFrameSize::Ms5 => 50,
            CeltFrameSize::Ms10 => 100,
            CeltFrameSize::Ms20 => 200,
        }
    }
}

/// RFC 6716 Table 55: MDCT bins per channel per band for each frame
/// size. Rows are band index `0..=20`; columns are
/// `[2.5 ms, 5 ms, 10 ms, 20 ms]` in that order — matching
/// [`CeltFrameSize::column_index`].
///
/// Each row doubles by a factor of 2 across columns (a 20 ms frame
/// has 8× the time-domain samples of a 2.5 ms frame, so the MDCT
/// produces 8× as many bins). The doubling is an internal consistency
/// invariant of the table and is pinned by a unit test.
const TABLE_55_BINS_PER_BAND: [[u16; 4]; CELT_NUM_BANDS] = [
    // Band  2.5 ms  5 ms  10 ms  20 ms
    /*  0 */ [1, 2, 4, 8],
    /*  1 */ [1, 2, 4, 8],
    /*  2 */ [1, 2, 4, 8],
    /*  3 */ [1, 2, 4, 8],
    /*  4 */ [1, 2, 4, 8],
    /*  5 */ [1, 2, 4, 8],
    /*  6 */ [1, 2, 4, 8],
    /*  7 */ [1, 2, 4, 8],
    /*  8 */ [2, 4, 8, 16],
    /*  9 */ [2, 4, 8, 16],
    /* 10 */ [2, 4, 8, 16],
    /* 11 */ [2, 4, 8, 16],
    /* 12 */ [4, 8, 16, 32],
    /* 13 */ [4, 8, 16, 32],
    /* 14 */ [4, 8, 16, 32],
    /* 15 */ [6, 12, 24, 48],
    /* 16 */ [6, 12, 24, 48],
    /* 17 */ [8, 16, 32, 64],
    /* 18 */ [12, 24, 48, 96],
    /* 19 */ [18, 36, 72, 144],
    /* 20 */ [22, 44, 88, 176],
];

/// Band-boundary frequencies in Hz from Table 55. Index `b` is the
/// band's start frequency; index `21` (= `CELT_NUM_BANDS`) is the
/// stop frequency of the last band (`20 kHz`). Conveniently, each
/// band's stop frequency is the next band's start frequency, so a
/// `CELT_NUM_BANDS + 1`-entry array captures both columns.
const TABLE_55_BAND_EDGES_HZ: [u16; CELT_NUM_BANDS + 1] = [
    0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 4000, 4800, 5600, 6800,
    8000, 9600, 12000, 15600, 20000,
];

/// MDCT bins per channel in band `band` at frame size `frame_size`,
/// looked up from Table 55. Returns `None` if `band >= CELT_NUM_BANDS`.
pub const fn celt_band_bins_per_channel(band: usize, frame_size: CeltFrameSize) -> Option<u16> {
    if band >= CELT_NUM_BANDS {
        return None;
    }
    Some(TABLE_55_BINS_PER_BAND[band][frame_size.column_index()])
}

/// Start frequency (Hz) of band `band` from Table 55. Returns `None`
/// for `band >= CELT_NUM_BANDS`.
pub const fn celt_band_start_hz(band: usize) -> Option<u16> {
    if band >= CELT_NUM_BANDS {
        return None;
    }
    Some(TABLE_55_BAND_EDGES_HZ[band])
}

/// Stop frequency (Hz) of band `band` from Table 55 — i.e. the start
/// frequency of band `band + 1`, or `20000` for `band == 20`. Returns
/// `None` for `band >= CELT_NUM_BANDS`.
pub const fn celt_band_stop_hz(band: usize) -> Option<u16> {
    if band >= CELT_NUM_BANDS {
        return None;
    }
    Some(TABLE_55_BAND_EDGES_HZ[band + 1])
}

/// First CELT band actually coded for a frame with `is_hybrid` set or
/// unset. RFC 6716 §4.3, p. 103: Hybrid skips bands `0..=16`; CELT-only
/// starts at band `0`.
pub const fn celt_first_coded_band(is_hybrid: bool) -> usize {
    if is_hybrid {
        HYBRID_FIRST_CODED_BAND
    } else {
        0
    }
}

/// One past the last CELT band actually coded for the standard (non-
/// Custom) CELT layer — always `CELT_NUM_BANDS`, regardless of mode.
/// Exposed as a symmetric companion to [`celt_first_coded_band`] so
/// callers can write `for b in celt_first_coded_band(h)..celt_end_coded_band()`.
pub const fn celt_end_coded_band() -> usize {
    CELT_NUM_BANDS
}

/// Total MDCT bins per channel summed across every coded band, for a
/// frame at `frame_size` with `is_hybrid` set or unset. RFC 6716 §4.3
/// Table 55 column sums (pinned exactly by `column_sum_pinned_values`):
///
/// * CELT-only, 2.5 ms — `100` bins / channel (`8×1 + 4×2 + 3×4 + 2×6 +
///   8 + 12 + 18 + 22`). The column doubles per frame-size step, so
///   5 / 10 / 20 ms are `200` / `400` / `800`.
/// * Hybrid (bands `17..=20` only) — `60` / `120` / `240` / `480` for
///   2.5 / 5 / 10 / 20 ms.
///
/// Note these *band-bin* sums are strictly less than the **full MDCT
/// size** `frame_samples` (`120` / `240` / `480` / `960`): Table 55's
/// bands top out at the 20 kHz edge, so the high MDCT bins above the last
/// band carry no coded content (the §4.3.7 inverse MDCT zero-pads them).
/// The synthesis backend ([`crate::celt_synthesis`]) sizes its frequency
/// buffer to the full MDCT size and fills only this coded prefix.
///
/// These sums are not directly stated in the RFC; they are pure
/// arithmetic over Table 55 and the §4.3 first-coded-band rule. They
/// matter to the §4.3.3 bit allocator and the §4.3.4 PVQ shape
/// decoder.
pub const fn celt_total_bins_per_channel(frame_size: CeltFrameSize, is_hybrid: bool) -> u32 {
    let mut sum: u32 = 0;
    let mut b = celt_first_coded_band(is_hybrid);
    let col = frame_size.column_index();
    while b < CELT_NUM_BANDS {
        sum += TABLE_55_BINS_PER_BAND[b][col] as u32;
        b += 1;
    }
    sum
}

/// Returns `Some(band)` where `band` is the smallest CELT band index
/// containing the given frequency, or `None` if `hz` is at or above the
/// last stop frequency (`20000 Hz`). A frequency exactly on a band
/// boundary falls into the higher-indexed band, matching the §4.3
/// half-open `[start, stop)` interval convention.
pub fn celt_band_at_hz(hz: u32) -> Option<usize> {
    if hz >= TABLE_55_BAND_EDGES_HZ[CELT_NUM_BANDS] as u32 {
        return None;
    }
    // Linear scan over 21 entries — bounded constant work, branch-free
    // result usage. Binary search isn't worth the complexity for 21
    // entries.
    let mut b: usize = 0;
    while b < CELT_NUM_BANDS {
        let stop = TABLE_55_BAND_EDGES_HZ[b + 1] as u32;
        if hz < stop {
            return Some(b);
        }
        b += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table 55 self-consistency: row 0 (band 0) starts at 0 Hz and
    /// the last row's stop is 20000 Hz — bracketing the full audible
    /// range that the Opus spec ever assigns a CELT band to.
    #[test]
    fn band_edges_span_full_audible_range() {
        assert_eq!(celt_band_start_hz(0), Some(0));
        assert_eq!(celt_band_stop_hz(CELT_NUM_BANDS - 1), Some(20000));
    }

    /// Adjacent bands tile the spectrum: `stop(b) == start(b+1)` for
    /// every `b ∈ 0..=19`. No gap, no overlap.
    #[test]
    fn adjacent_bands_tile_without_gaps() {
        for b in 0..(CELT_NUM_BANDS - 1) {
            assert_eq!(celt_band_stop_hz(b), celt_band_start_hz(b + 1), "band {b}");
        }
    }

    /// Each band's stop frequency is strictly greater than its start —
    /// no zero-width or inverted bands.
    #[test]
    fn bands_have_positive_width() {
        for b in 0..CELT_NUM_BANDS {
            let start = celt_band_start_hz(b).unwrap();
            let stop = celt_band_stop_hz(b).unwrap();
            assert!(stop > start, "band {b}: {start}..{stop}");
        }
    }

    /// The four Table 55 "Bins:" columns differ only by a power of
    /// two: column `c` is `1 << c` × column 0 (= `2.5 ms`). This is
    /// the consistency invariant that lets the MDCT step double in
    /// size when the frame doubles in duration.
    #[test]
    fn columns_scale_by_power_of_two() {
        for b in 0..CELT_NUM_BANDS {
            let col0 = celt_band_bins_per_channel(b, CeltFrameSize::Ms2_5).unwrap();
            for (c, fs) in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(
                    celt_band_bins_per_channel(b, fs),
                    Some(col0 << c),
                    "band {b}, column {c}"
                );
            }
        }
    }

    /// Every Table 55 bin count is at least 1 (RFC 6716 §4.3, p. 103:
    /// "as little as one MDCT bin per channel") and at most 176 (the
    /// 20 ms × band 20 cell).
    #[test]
    fn bin_count_in_documented_range() {
        for b in 0..CELT_NUM_BANDS {
            for fs in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                let n = celt_band_bins_per_channel(b, fs).unwrap();
                assert!(n >= 1, "band {b}, fs {fs:?}: {n} < 1");
                assert!(
                    n <= CELT_MAX_BINS_PER_BAND,
                    "band {b}, fs {fs:?}: {n} > {CELT_MAX_BINS_PER_BAND}"
                );
            }
        }
    }

    /// Random-access spot pins on a handful of Table 55 cells — the
    /// table is small enough to assert by hand and large enough that
    /// a transcription typo would otherwise sneak through. These pins
    /// guard against a column-shift or row-shift.
    #[test]
    fn table_55_pinned_cells() {
        // Band 0 × every column.
        assert_eq!(celt_band_bins_per_channel(0, CeltFrameSize::Ms2_5), Some(1));
        assert_eq!(celt_band_bins_per_channel(0, CeltFrameSize::Ms5), Some(2));
        assert_eq!(celt_band_bins_per_channel(0, CeltFrameSize::Ms10), Some(4));
        assert_eq!(celt_band_bins_per_channel(0, CeltFrameSize::Ms20), Some(8));

        // Band 8 (the first band where the bin count doubles to 2 at
        // 2.5 ms).
        assert_eq!(celt_band_bins_per_channel(8, CeltFrameSize::Ms2_5), Some(2));
        assert_eq!(celt_band_bins_per_channel(8, CeltFrameSize::Ms20), Some(16));

        // Band 12 (4×).
        assert_eq!(
            celt_band_bins_per_channel(12, CeltFrameSize::Ms2_5),
            Some(4)
        );
        assert_eq!(
            celt_band_bins_per_channel(12, CeltFrameSize::Ms20),
            Some(32)
        );

        // Band 15 (6×).
        assert_eq!(
            celt_band_bins_per_channel(15, CeltFrameSize::Ms2_5),
            Some(6)
        );
        assert_eq!(
            celt_band_bins_per_channel(15, CeltFrameSize::Ms20),
            Some(48)
        );

        // Band 17 (the first Hybrid-coded band).
        assert_eq!(
            celt_band_bins_per_channel(17, CeltFrameSize::Ms2_5),
            Some(8)
        );
        assert_eq!(
            celt_band_bins_per_channel(17, CeltFrameSize::Ms20),
            Some(64)
        );

        // Band 20 (the widest band — the 176 / 22 / 44 / 88 row).
        assert_eq!(
            celt_band_bins_per_channel(20, CeltFrameSize::Ms2_5),
            Some(22)
        );
        assert_eq!(celt_band_bins_per_channel(20, CeltFrameSize::Ms5), Some(44));
        assert_eq!(
            celt_band_bins_per_channel(20, CeltFrameSize::Ms10),
            Some(88)
        );
        assert_eq!(
            celt_band_bins_per_channel(20, CeltFrameSize::Ms20),
            Some(176)
        );
    }

    /// Band-edge spot pins — guards against a shift in
    /// `TABLE_55_BAND_EDGES_HZ`. Three pins at the start, two pins on
    /// the Hybrid boundary, two pins at the tail.
    #[test]
    fn band_edges_pinned() {
        // Start.
        assert_eq!(celt_band_start_hz(0), Some(0));
        assert_eq!(celt_band_start_hz(1), Some(200));
        assert_eq!(celt_band_start_hz(7), Some(1400));
        // Hybrid boundary — band 17 starts at 8 kHz per the §4.3
        // "first 17 bands (up to 8 kHz) are not coded" rule, so band
        // 16 stops at 8 kHz.
        assert_eq!(celt_band_stop_hz(16), Some(8000));
        assert_eq!(celt_band_start_hz(17), Some(8000));
        // Tail.
        assert_eq!(celt_band_start_hz(20), Some(15600));
        assert_eq!(celt_band_stop_hz(20), Some(20000));
    }

    /// Out-of-range band indices return `None`.
    #[test]
    fn out_of_range_band_returns_none() {
        assert_eq!(celt_band_bins_per_channel(21, CeltFrameSize::Ms20), None);
        assert_eq!(
            celt_band_bins_per_channel(usize::MAX, CeltFrameSize::Ms5),
            None
        );
        assert_eq!(celt_band_start_hz(21), None);
        assert_eq!(celt_band_stop_hz(21), None);
    }

    /// `CeltFrameSize::from_frame_tenths_ms` covers the four CELT
    /// durations and rejects the two SILK-only durations (40 / 60 ms)
    /// per the §3.1 Table 2 / §4.2 layout.
    #[test]
    fn frame_size_round_trips_tenths_ms() {
        assert_eq!(
            CeltFrameSize::from_frame_tenths_ms(25),
            Some(CeltFrameSize::Ms2_5)
        );
        assert_eq!(
            CeltFrameSize::from_frame_tenths_ms(50),
            Some(CeltFrameSize::Ms5)
        );
        assert_eq!(
            CeltFrameSize::from_frame_tenths_ms(100),
            Some(CeltFrameSize::Ms10)
        );
        assert_eq!(
            CeltFrameSize::from_frame_tenths_ms(200),
            Some(CeltFrameSize::Ms20)
        );
        // SILK-only — no CELT layer.
        assert_eq!(CeltFrameSize::from_frame_tenths_ms(400), None);
        assert_eq!(CeltFrameSize::from_frame_tenths_ms(600), None);
        // Garbage inputs.
        assert_eq!(CeltFrameSize::from_frame_tenths_ms(0), None);
        assert_eq!(CeltFrameSize::from_frame_tenths_ms(99), None);
        assert_eq!(CeltFrameSize::from_frame_tenths_ms(u32::MAX), None);

        // Round-trip.
        for fs in [
            CeltFrameSize::Ms2_5,
            CeltFrameSize::Ms5,
            CeltFrameSize::Ms10,
            CeltFrameSize::Ms20,
        ] {
            assert_eq!(
                CeltFrameSize::from_frame_tenths_ms(fs.to_frame_tenths_ms()),
                Some(fs)
            );
        }
    }

    /// `column_index` matches the enum's documented `repr(u8)`.
    #[test]
    fn column_index_matches_discriminant() {
        assert_eq!(CeltFrameSize::Ms2_5.column_index(), 0);
        assert_eq!(CeltFrameSize::Ms5.column_index(), 1);
        assert_eq!(CeltFrameSize::Ms10.column_index(), 2);
        assert_eq!(CeltFrameSize::Ms20.column_index(), 3);
    }

    /// Hybrid mode skips bands `0..=16` per RFC 6716 §4.3.
    #[test]
    fn hybrid_first_coded_band_is_17() {
        assert_eq!(celt_first_coded_band(true), 17);
        assert_eq!(HYBRID_FIRST_CODED_BAND, 17);
        // First Hybrid-coded band starts at 8 kHz.
        assert_eq!(celt_band_start_hz(celt_first_coded_band(true)), Some(8000));
    }

    /// CELT-only mode starts at band 0.
    #[test]
    fn celt_only_first_coded_band_is_0() {
        assert_eq!(celt_first_coded_band(false), 0);
        assert_eq!(celt_band_start_hz(celt_first_coded_band(false)), Some(0));
    }

    /// `celt_end_coded_band` is `CELT_NUM_BANDS` regardless of mode.
    #[test]
    fn end_coded_band_is_celt_num_bands() {
        assert_eq!(celt_end_coded_band(), CELT_NUM_BANDS);
    }

    /// Total bins per channel summed across coded bands matches an
    /// independent column sum.
    #[test]
    fn total_bins_per_channel_matches_column_sum() {
        for fs in [
            CeltFrameSize::Ms2_5,
            CeltFrameSize::Ms5,
            CeltFrameSize::Ms10,
            CeltFrameSize::Ms20,
        ] {
            // CELT-only — sum every row of the column.
            let celt_only: u32 = (0..CELT_NUM_BANDS)
                .map(|b| celt_band_bins_per_channel(b, fs).unwrap() as u32)
                .sum();
            assert_eq!(
                celt_total_bins_per_channel(fs, false),
                celt_only,
                "celt-only sum at {fs:?}"
            );

            // Hybrid — sum only `17..=20`.
            let hybrid: u32 = (HYBRID_FIRST_CODED_BAND..CELT_NUM_BANDS)
                .map(|b| celt_band_bins_per_channel(b, fs).unwrap() as u32)
                .sum();
            assert_eq!(
                celt_total_bins_per_channel(fs, true),
                hybrid,
                "hybrid sum at {fs:?}"
            );

            // Hybrid is a strict subset of CELT-only.
            assert!(
                celt_total_bins_per_channel(fs, true) < celt_total_bins_per_channel(fs, false),
                "hybrid total should be < celt-only total at {fs:?}"
            );
        }
    }

    /// Pinned column-sum values — guards against a row-shift in
    /// Table 55. Each column doubles down the table just like the
    /// individual cells do.
    #[test]
    fn column_sum_pinned_values() {
        // Sum across all 21 bands at 2.5 ms:
        // 8 × 1 (bands 0..=7) + 4 × 2 (8..=11) + 3 × 4 (12..=14)
        //   + 2 × 6 (15..=16) + 8 (17) + 12 (18) + 18 (19) + 22 (20)
        //   = 8 + 8 + 12 + 12 + 8 + 12 + 18 + 22
        //   = 100? Let's compute: 8+8=16; +12=28; +12=40; +8=48;
        //   +12=60; +18=78; +22=100. So 100 bins per channel at 2.5 ms.
        assert_eq!(
            celt_total_bins_per_channel(CeltFrameSize::Ms2_5, false),
            100
        );
        // Doubles for each frame-size step (the column scales by 2).
        assert_eq!(celt_total_bins_per_channel(CeltFrameSize::Ms5, false), 200);
        assert_eq!(celt_total_bins_per_channel(CeltFrameSize::Ms10, false), 400);
        assert_eq!(celt_total_bins_per_channel(CeltFrameSize::Ms20, false), 800);

        // Hybrid at 2.5 ms — only bands 17..=20:
        //   8 + 12 + 18 + 22 = 60. (And 2.5 ms Hybrid never actually
        //   occurs per §3.1 Table 2, but the table layout still makes
        //   sense — the column scaling holds.)
        assert_eq!(celt_total_bins_per_channel(CeltFrameSize::Ms2_5, true), 60);
        assert_eq!(celt_total_bins_per_channel(CeltFrameSize::Ms5, true), 120);
        // Real-world Hybrid sizes (10 / 20 ms):
        assert_eq!(celt_total_bins_per_channel(CeltFrameSize::Ms10, true), 240);
        assert_eq!(celt_total_bins_per_channel(CeltFrameSize::Ms20, true), 480);
    }

    /// `celt_band_at_hz` is the inverse of `(celt_band_start_hz,
    /// celt_band_stop_hz)`: every hz strictly inside band `b`'s
    /// `[start, stop)` interval lands on `b`, and the start frequency
    /// itself lands on `b` (the half-open convention).
    #[test]
    fn band_at_hz_round_trips_with_edges() {
        for b in 0..CELT_NUM_BANDS {
            let start = celt_band_start_hz(b).unwrap() as u32;
            let stop = celt_band_stop_hz(b).unwrap() as u32;
            assert_eq!(celt_band_at_hz(start), Some(b), "start of band {b}");
            // Midpoint also lands on b (every band is at least 200 Hz
            // wide so the midpoint is strictly less than stop).
            let mid = (start + stop) / 2;
            assert_eq!(celt_band_at_hz(mid), Some(b), "midpoint of band {b}");
            // stop - 1 (last Hz of the half-open interval) lands on b.
            assert_eq!(celt_band_at_hz(stop - 1), Some(b), "stop-1 of band {b}");
        }
    }

    /// `celt_band_at_hz` rejects anything at or above 20 kHz — the
    /// CELT layout doesn't define a band beyond 20 kHz (the Nyquist
    /// limit for Opus's 48 kHz internal rate).
    #[test]
    fn band_at_hz_rejects_above_20khz() {
        assert_eq!(celt_band_at_hz(20000), None);
        assert_eq!(celt_band_at_hz(20001), None);
        assert_eq!(celt_band_at_hz(48000), None);
        assert_eq!(celt_band_at_hz(u32::MAX), None);
    }

    /// `celt_band_at_hz(8000)` lands on band 17 — i.e. the first
    /// Hybrid-coded band — confirming the §4.3 "first 17 bands (up to
    /// 8 kHz)" rule lines up with the band-at-Hz lookup.
    #[test]
    fn band_at_8khz_is_first_hybrid_coded() {
        assert_eq!(celt_band_at_hz(8000), Some(HYBRID_FIRST_CODED_BAND));
    }

    /// Each band's `(stop - start)` is a whole number of 200 Hz
    /// "fine-resolution" widths — internal consistency check on the
    /// Bark-aligned tiling. Smallest bands (0..=7) are 200 Hz wide;
    /// the widest (band 20) is 4400 Hz wide.
    #[test]
    fn band_widths_multiples_of_200_hz() {
        for b in 0..CELT_NUM_BANDS {
            let width =
                celt_band_stop_hz(b).unwrap() as u32 - celt_band_start_hz(b).unwrap() as u32;
            assert_eq!(width % 200, 0, "band {b} width {width} not multiple of 200");
        }
        // Pinned widths for a few representative bands.
        assert_eq!(
            celt_band_stop_hz(0).unwrap() - celt_band_start_hz(0).unwrap(),
            200
        );
        assert_eq!(
            celt_band_stop_hz(8).unwrap() - celt_band_start_hz(8).unwrap(),
            400
        );
        assert_eq!(
            celt_band_stop_hz(20).unwrap() - celt_band_start_hz(20).unwrap(),
            4400
        );
    }

    /// Re-pin the documented `CELT_MAX_BINS_PER_BAND` constant against
    /// the actual maximum across the whole table.
    #[test]
    fn celt_max_bins_per_band_is_table_max() {
        let mut actual_max: u16 = 0;
        for b in 0..CELT_NUM_BANDS {
            for fs in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                let n = celt_band_bins_per_channel(b, fs).unwrap();
                if n > actual_max {
                    actual_max = n;
                }
            }
        }
        assert_eq!(actual_max, CELT_MAX_BINS_PER_BAND);
    }
}
