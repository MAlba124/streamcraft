//! CELT time-frequency resolution adjustments — RFC 6716 §4.3.4.5,
//! Tables 60–63.
//!
//! The CELT layer can independently change the time-frequency (TF)
//! resolution of each coded band. The mechanism has three pieces:
//!
//! 1. A global per-frame `transient` flag (decoded as part of the
//!    Table 56 prefix; owned by [`crate::celt_header::CeltHeaderPrefix`]).
//! 2. A per-band binary `tf_change[b]` flag (decoded inside the band
//!    loop right after coarse energy; not owned by this module).
//! 3. An optional global `tf_select` flag (1/2 PDF; only present when
//!    it can actually affect the resolution). This flag selects between
//!    two TF-adjustment lookups for the chosen `(frame_size, transient)`
//!    combination.
//!
//! This module owns the four lookup tables (Tables 60–63 RFC 6716
//! §4.3.4.5) that turn `(frame_size, transient, tf_select, tf_change[b])`
//! into a single integer TF adjustment per band. The adjustment is
//! consumed downstream by §4.3.4.5's Hadamard-transform step:
//!
//! * Negative → temporal resolution is *increased*; the decoder
//!   applies `|adj|` levels of the Hadamard transform to each
//!   interleaved MDCT vector.
//! * Positive → frequency resolution is *increased*; the decoder
//!   applies `adj` levels of the Hadamard transform *across* the
//!   interleaved MDCT vector. RFC 6716 §4.3.4.5 notes positive
//!   adjustments only ever appear on transient frames.
//! * Zero → resolution unchanged.
//!
//! ## What this module does **not** own
//!
//! * The §4.3.1 `transient` flag and the §4.3.4.5 `tf_select` flag.
//!   `transient` is in [`crate::celt_header::CeltHeaderPrefix`];
//!   `tf_select` is decoded right after the per-band `tf_change`
//!   flags by the §4.3.4.5 band loop, which is gated on §4.3.2.1
//!   coarse energy + §4.3.3 bit allocation (both still deferred —
//!   see the round-20 prose).
//! * The per-band `tf_change[b]` decoder itself. §4.3.4.5 codes the
//!   first band's choice via a `{3, 1}/4` PDF on transient frames /
//!   `{15, 1}/16` on others, and subsequent bands relative to the
//!   previous band via `{15, 1}/16` (transient) / `{31, 1}/32`
//!   (non-transient). That decoder lives in the band loop ahead.
//! * The Hadamard-transform step that consumes the adjustment.
//!
//! ## Provenance
//!
//! Every cell of every table is transcribed from RFC 6716 §4.3.4.5
//! (`docs/audio/opus/rfc6716-opus.txt`, p. 119–120). No external
//! library source consulted; no cross-check against any reference
//! implementation.

use crate::celt_band_layout::CeltFrameSize;

/// TF adjustment range across all four lookup tables. RFC 6716
/// §4.3.4.5 (Tables 60–63) bounds every documented adjustment to
/// `[-3, 3]`, so `i8` is the right storage type.
pub type TfAdjustment = i8;

/// Maximum positive TF adjustment in any of the four tables (Table 62,
/// transient × 20 ms × `choice = 0` ⇒ +3). Positive adjustments only
/// ever appear on transient frames per RFC 6716 §4.3.4.5.
pub const TF_ADJUSTMENT_MAX: TfAdjustment = 3;

/// Maximum magnitude of any TF adjustment in any of the four tables
/// (Table 61 / 63 each include a -3 cell). Useful for sizing
/// downstream Hadamard-iteration counters.
pub const TF_ADJUSTMENT_ABS_MAX: u8 = 3;

/// Table 60 — TF Adjustments for **non-transient** frames and
/// `tf_select = 0`, indexed by `[frame_size_column][choice]` where
/// `frame_size_column` is the [`CeltFrameSize`] discriminant (Table 55
/// column) and `choice ∈ {0, 1}` is the per-band `tf_change` flag.
///
/// RFC 6716 §4.3.4.5, p. 119.
pub const TF_ADJ_NONTRANSIENT_SELECT0: [[TfAdjustment; 2]; 4] = [
    // 2.5 ms
    [0, -1],
    // 5 ms
    [0, -1],
    // 10 ms
    [0, -2],
    // 20 ms
    [0, -2],
];

/// Table 61 — TF Adjustments for **non-transient** frames and
/// `tf_select = 1`.
///
/// RFC 6716 §4.3.4.5, p. 119.
pub const TF_ADJ_NONTRANSIENT_SELECT1: [[TfAdjustment; 2]; 4] = [
    // 2.5 ms
    [0, -1],
    // 5 ms
    [0, -2],
    // 10 ms
    [0, -3],
    // 20 ms
    [0, -3],
];

/// Table 62 — TF Adjustments for **transient** frames and
/// `tf_select = 0`.
///
/// RFC 6716 §4.3.4.5, p. 119.
pub const TF_ADJ_TRANSIENT_SELECT0: [[TfAdjustment; 2]; 4] = [
    // 2.5 ms
    [0, -1],
    // 5 ms
    [1, 0],
    // 10 ms
    [2, 0],
    // 20 ms
    [3, 0],
];

/// Table 63 — TF Adjustments for **transient** frames and
/// `tf_select = 1`.
///
/// RFC 6716 §4.3.4.5, p. 120.
pub const TF_ADJ_TRANSIENT_SELECT1: [[TfAdjustment; 2]; 4] = [
    // 2.5 ms
    [0, -1],
    // 5 ms
    [1, -1],
    // 10 ms
    [1, -1],
    // 20 ms
    [1, -1],
];

/// Look up the TF resolution adjustment for one band per RFC 6716
/// §4.3.4.5 (Tables 60–63).
///
/// * `frame_size` — the CELT frame size (one of the four Table 55
///   columns).
/// * `transient` — the §4.3.1 global transient flag.
/// * `tf_select` — the §4.3.4.5 `tf_select` flag (only present in
///   the bitstream when it can affect at least one band; absent
///   frames are treated as `tf_select = 0`).
/// * `tf_change` — the per-band `tf_change[b]` choice (one bit; not
///   the prior-band-relative-coded raw bit, the absolute choice
///   `0..=1`).
///
/// Returns a signed adjustment in `[-3, 3]` per Tables 60–63.
#[inline]
pub fn celt_tf_adjustment(
    frame_size: CeltFrameSize,
    transient: bool,
    tf_select: bool,
    tf_change: bool,
) -> TfAdjustment {
    let table = match (transient, tf_select) {
        (false, false) => &TF_ADJ_NONTRANSIENT_SELECT0,
        (false, true) => &TF_ADJ_NONTRANSIENT_SELECT1,
        (true, false) => &TF_ADJ_TRANSIENT_SELECT0,
        (true, true) => &TF_ADJ_TRANSIENT_SELECT1,
    };
    table[frame_size as usize][tf_change as usize]
}

/// Whether `tf_select` can ever affect the per-band TF adjustment for
/// the given `(frame_size, transient)` and the current set of
/// per-band `tf_change` choices.
///
/// RFC 6716 §4.3.1: "The tf_select flag uses a 1/2 probability, but
/// is only decoded if it can have an impact on the result knowing the
/// value of all per-band tf_change flags."
///
/// For a single `tf_change` value the question reduces to: do the
/// `tf_select = 0` and `tf_select = 1` lookups disagree for any choice
/// the band loop has actually selected? This helper takes the full set
/// of decoded `tf_change` bits (one per coded band, in order) and
/// returns `true` iff at least one band's adjustment depends on
/// `tf_select`. The §4.3.4.5 band loop calls this AFTER decoding every
/// `tf_change[b]` to decide whether to read the `tf_select` bit at all.
///
/// An empty slice (e.g. zero coded bands) yields `false` — there is
/// nothing for `tf_select` to affect.
#[inline]
pub fn celt_tf_select_can_affect(
    frame_size: CeltFrameSize,
    transient: bool,
    tf_change: &[bool],
) -> bool {
    let (a, b) = if transient {
        (
            &TF_ADJ_TRANSIENT_SELECT0[frame_size as usize],
            &TF_ADJ_TRANSIENT_SELECT1[frame_size as usize],
        )
    } else {
        (
            &TF_ADJ_NONTRANSIENT_SELECT0[frame_size as usize],
            &TF_ADJ_NONTRANSIENT_SELECT1[frame_size as usize],
        )
    };
    for &c in tf_change {
        let idx = c as usize;
        if a[idx] != b[idx] {
            return true;
        }
    }
    false
}

/// Classification of a TF adjustment in terms of which §4.3.4.5
/// Hadamard-transform branch the decoder takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TfDirection {
    /// `adj == 0` — no Hadamard pass; the band's MDCT vector is used
    /// as-is.
    Unchanged,
    /// `adj < 0` — increase temporal resolution by `|adj|` levels of
    /// the Hadamard transform, applied to each interleaved MDCT vector
    /// (RFC 6716 §4.3.4.5).
    IncreaseTime(u8),
    /// `adj > 0` — increase frequency resolution by `adj` levels of
    /// the Hadamard transform, applied across the interleaved MDCT
    /// vector. Only reachable on transient frames per RFC 6716
    /// §4.3.4.5.
    IncreaseFrequency(u8),
}

impl TfDirection {
    /// Classify a [`TfAdjustment`] into its §4.3.4.5 Hadamard branch.
    #[inline]
    pub const fn from_adjustment(adj: TfAdjustment) -> Self {
        if adj == 0 {
            Self::Unchanged
        } else if adj < 0 {
            // adj.unsigned_abs() is not const stable on this MSRV; do
            // it by hand. -3..=-1 → 1..=3.
            Self::IncreaseTime((-adj) as u8)
        } else {
            Self::IncreaseFrequency(adj as u8)
        }
    }

    /// Number of Hadamard transform levels to apply (either across
    /// the interleaved MDCT vector or per-vector depending on the
    /// direction). Zero for `Unchanged`.
    #[inline]
    pub const fn levels(self) -> u8 {
        match self {
            Self::Unchanged => 0,
            Self::IncreaseTime(n) | Self::IncreaseFrequency(n) => n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- Table-shape and provenance pins ----------

    /// All four tables have the same shape: 4 frame sizes × 2 choices.
    #[test]
    fn tables_have_documented_shape() {
        for table in [
            &TF_ADJ_NONTRANSIENT_SELECT0,
            &TF_ADJ_NONTRANSIENT_SELECT1,
            &TF_ADJ_TRANSIENT_SELECT0,
            &TF_ADJ_TRANSIENT_SELECT1,
        ] {
            assert_eq!(table.len(), 4);
            for row in table.iter() {
                assert_eq!(row.len(), 2);
            }
        }
    }

    /// Every cell of every table fits in `[-3, 3]`, the documented
    /// range. This pins the storage type to `i8` and rules out an
    /// off-by-many transcription typo.
    #[test]
    fn every_cell_in_documented_range() {
        for table in [
            &TF_ADJ_NONTRANSIENT_SELECT0,
            &TF_ADJ_NONTRANSIENT_SELECT1,
            &TF_ADJ_TRANSIENT_SELECT0,
            &TF_ADJ_TRANSIENT_SELECT1,
        ] {
            for row in table.iter() {
                for &cell in row.iter() {
                    assert!((-3..=3).contains(&cell), "cell {cell} out of [-3, 3]");
                }
            }
        }
    }

    // ---------- Per-table hand-pinned spot checks (every cell) ----------

    /// Table 60 (non-transient, tf_select = 0) — every cell pinned.
    /// RFC 6716 §4.3.4.5 p. 119.
    #[test]
    fn table_60_pinned_cells() {
        // [frame_size_column][choice]
        assert_eq!(TF_ADJ_NONTRANSIENT_SELECT0[0], [0, -1]); // 2.5 ms
        assert_eq!(TF_ADJ_NONTRANSIENT_SELECT0[1], [0, -1]); // 5 ms
        assert_eq!(TF_ADJ_NONTRANSIENT_SELECT0[2], [0, -2]); // 10 ms
        assert_eq!(TF_ADJ_NONTRANSIENT_SELECT0[3], [0, -2]); // 20 ms
    }

    /// Table 61 (non-transient, tf_select = 1) — every cell pinned.
    /// RFC 6716 §4.3.4.5 p. 119.
    #[test]
    fn table_61_pinned_cells() {
        assert_eq!(TF_ADJ_NONTRANSIENT_SELECT1[0], [0, -1]); // 2.5 ms
        assert_eq!(TF_ADJ_NONTRANSIENT_SELECT1[1], [0, -2]); // 5 ms
        assert_eq!(TF_ADJ_NONTRANSIENT_SELECT1[2], [0, -3]); // 10 ms
        assert_eq!(TF_ADJ_NONTRANSIENT_SELECT1[3], [0, -3]); // 20 ms
    }

    /// Table 62 (transient, tf_select = 0) — every cell pinned.
    /// RFC 6716 §4.3.4.5 p. 119.
    #[test]
    fn table_62_pinned_cells() {
        assert_eq!(TF_ADJ_TRANSIENT_SELECT0[0], [0, -1]); // 2.5 ms
        assert_eq!(TF_ADJ_TRANSIENT_SELECT0[1], [1, 0]); //  5 ms
        assert_eq!(TF_ADJ_TRANSIENT_SELECT0[2], [2, 0]); // 10 ms
        assert_eq!(TF_ADJ_TRANSIENT_SELECT0[3], [3, 0]); // 20 ms
    }

    /// Table 63 (transient, tf_select = 1) — every cell pinned.
    /// RFC 6716 §4.3.4.5 p. 120.
    #[test]
    fn table_63_pinned_cells() {
        assert_eq!(TF_ADJ_TRANSIENT_SELECT1[0], [0, -1]); // 2.5 ms
        assert_eq!(TF_ADJ_TRANSIENT_SELECT1[1], [1, -1]); // 5 ms
        assert_eq!(TF_ADJ_TRANSIENT_SELECT1[2], [1, -1]); // 10 ms
        assert_eq!(TF_ADJ_TRANSIENT_SELECT1[3], [1, -1]); // 20 ms
    }

    // ---------- Cross-table structural invariants ----------

    /// "Choice 0" never produces a negative adjustment on non-transient
    /// frames: per Tables 60 and 61 every `[*][0]` cell is `0`. This
    /// pins the §4.3.4.5 "no change" semantics of the most-likely
    /// per-band branch on stationary content.
    #[test]
    fn nontransient_choice_zero_is_always_zero() {
        for table in [&TF_ADJ_NONTRANSIENT_SELECT0, &TF_ADJ_NONTRANSIENT_SELECT1] {
            for (fs, row) in table.iter().enumerate() {
                assert_eq!(row[0], 0, "nontransient fs={fs} choice=0 should be 0");
            }
        }
    }

    /// "Choice 1" is always non-positive on non-transient frames
    /// (Tables 60–61) — RFC 6716 §4.3.4.5: "a negative TF adjustment
    /// means that the temporal resolution is increased". Stationary
    /// frames only ever gain temporal resolution, never frequency
    /// resolution.
    #[test]
    fn nontransient_choice_one_is_nonpositive() {
        for table in [&TF_ADJ_NONTRANSIENT_SELECT0, &TF_ADJ_NONTRANSIENT_SELECT1] {
            for row in table.iter() {
                assert!(row[1] <= 0);
            }
        }
    }

    /// Positive adjustments only appear on transient frames. RFC 6716
    /// §4.3.4.5 documents this asymmetry explicitly.
    #[test]
    fn positive_adjustments_only_on_transient_frames() {
        for table in [&TF_ADJ_NONTRANSIENT_SELECT0, &TF_ADJ_NONTRANSIENT_SELECT1] {
            for row in table.iter() {
                for &cell in row.iter() {
                    assert!(cell <= 0, "non-transient cell {cell} should be <= 0");
                }
            }
        }
    }

    /// Table 62's "choice 0" column (transient, tf_select=0) scales
    /// monotonically with frame size: `0, 1, 2, 3` across `2.5, 5, 10,
    /// 20 ms`. This is the §4.3.4.5 "longer transient frames need more
    /// frequency resolution per MDCT" structural property.
    #[test]
    fn table_62_choice0_scales_with_frame_size() {
        assert_eq!(TF_ADJ_TRANSIENT_SELECT0[0][0], 0);
        assert_eq!(TF_ADJ_TRANSIENT_SELECT0[1][0], 1);
        assert_eq!(TF_ADJ_TRANSIENT_SELECT0[2][0], 2);
        assert_eq!(TF_ADJ_TRANSIENT_SELECT0[3][0], 3);
    }

    /// The 2.5 ms row of every table is `[0, -1]`. This is the
    /// shortest CELT frame size and §4.3.4.5 leaves the same option
    /// (gain one level of temporal resolution) available regardless of
    /// transient / tf_select.
    #[test]
    fn ms2_5_row_is_universal() {
        assert_eq!(TF_ADJ_NONTRANSIENT_SELECT0[0], [0, -1]);
        assert_eq!(TF_ADJ_NONTRANSIENT_SELECT1[0], [0, -1]);
        assert_eq!(TF_ADJ_TRANSIENT_SELECT0[0], [0, -1]);
        assert_eq!(TF_ADJ_TRANSIENT_SELECT1[0], [0, -1]);
    }

    /// The documented `TF_ADJUSTMENT_MAX = 3` equals the actual maximum
    /// across every cell of every table.
    #[test]
    fn tf_adjustment_max_matches_tables() {
        let mut observed_max: TfAdjustment = i8::MIN;
        for table in [
            &TF_ADJ_NONTRANSIENT_SELECT0,
            &TF_ADJ_NONTRANSIENT_SELECT1,
            &TF_ADJ_TRANSIENT_SELECT0,
            &TF_ADJ_TRANSIENT_SELECT1,
        ] {
            for row in table.iter() {
                for &cell in row.iter() {
                    if cell > observed_max {
                        observed_max = cell;
                    }
                }
            }
        }
        assert_eq!(observed_max, TF_ADJUSTMENT_MAX);
    }

    /// The documented `TF_ADJUSTMENT_ABS_MAX = 3` equals the actual
    /// maximum magnitude across every cell.
    #[test]
    fn tf_adjustment_abs_max_matches_tables() {
        let mut observed: u8 = 0;
        for table in [
            &TF_ADJ_NONTRANSIENT_SELECT0,
            &TF_ADJ_NONTRANSIENT_SELECT1,
            &TF_ADJ_TRANSIENT_SELECT0,
            &TF_ADJ_TRANSIENT_SELECT1,
        ] {
            for row in table.iter() {
                for &cell in row.iter() {
                    let mag = cell.unsigned_abs();
                    if mag > observed {
                        observed = mag;
                    }
                }
            }
        }
        assert_eq!(observed, TF_ADJUSTMENT_ABS_MAX);
    }

    // ---------- celt_tf_adjustment() entry-point coverage ----------

    /// The entry point routes (non-transient, tf_select=0) to Table 60.
    #[test]
    fn entry_routes_nontransient_select0_to_table_60() {
        for (fs_idx, fs) in [
            CeltFrameSize::Ms2_5,
            CeltFrameSize::Ms5,
            CeltFrameSize::Ms10,
            CeltFrameSize::Ms20,
        ]
        .into_iter()
        .enumerate()
        {
            for choice in [false, true] {
                assert_eq!(
                    celt_tf_adjustment(fs, false, false, choice),
                    TF_ADJ_NONTRANSIENT_SELECT0[fs_idx][choice as usize]
                );
            }
        }
    }

    /// The entry point routes (non-transient, tf_select=1) to Table 61.
    #[test]
    fn entry_routes_nontransient_select1_to_table_61() {
        for (fs_idx, fs) in [
            CeltFrameSize::Ms2_5,
            CeltFrameSize::Ms5,
            CeltFrameSize::Ms10,
            CeltFrameSize::Ms20,
        ]
        .into_iter()
        .enumerate()
        {
            for choice in [false, true] {
                assert_eq!(
                    celt_tf_adjustment(fs, false, true, choice),
                    TF_ADJ_NONTRANSIENT_SELECT1[fs_idx][choice as usize]
                );
            }
        }
    }

    /// The entry point routes (transient, tf_select=0) to Table 62.
    #[test]
    fn entry_routes_transient_select0_to_table_62() {
        for (fs_idx, fs) in [
            CeltFrameSize::Ms2_5,
            CeltFrameSize::Ms5,
            CeltFrameSize::Ms10,
            CeltFrameSize::Ms20,
        ]
        .into_iter()
        .enumerate()
        {
            for choice in [false, true] {
                assert_eq!(
                    celt_tf_adjustment(fs, true, false, choice),
                    TF_ADJ_TRANSIENT_SELECT0[fs_idx][choice as usize]
                );
            }
        }
    }

    /// The entry point routes (transient, tf_select=1) to Table 63.
    #[test]
    fn entry_routes_transient_select1_to_table_63() {
        for (fs_idx, fs) in [
            CeltFrameSize::Ms2_5,
            CeltFrameSize::Ms5,
            CeltFrameSize::Ms10,
            CeltFrameSize::Ms20,
        ]
        .into_iter()
        .enumerate()
        {
            for choice in [false, true] {
                assert_eq!(
                    celt_tf_adjustment(fs, true, true, choice),
                    TF_ADJ_TRANSIENT_SELECT1[fs_idx][choice as usize]
                );
            }
        }
    }

    // ---------- celt_tf_select_can_affect() coverage ----------

    /// Empty `tf_change` slice — `tf_select` cannot affect any band's
    /// adjustment because there are no bands. RFC 6716 §4.3.1: the
    /// flag is "only decoded if it can have an impact".
    #[test]
    fn tf_select_cannot_affect_empty_band_set() {
        for fs in [
            CeltFrameSize::Ms2_5,
            CeltFrameSize::Ms5,
            CeltFrameSize::Ms10,
            CeltFrameSize::Ms20,
        ] {
            for transient in [false, true] {
                assert!(!celt_tf_select_can_affect(fs, transient, &[]));
            }
        }
    }

    /// 2.5 ms non-transient: Tables 60 and 61 are identical
    /// (`[[0, -1]]`), so `tf_select` cannot affect the result no matter
    /// what the band loop chose. RFC 6716 §4.3.1 says the flag is then
    /// not encoded.
    #[test]
    fn ms2_5_nontransient_tf_select_never_matters() {
        for choices in [
            vec![],
            vec![false],
            vec![true],
            vec![false, true],
            vec![true, true, false, false],
        ] {
            assert!(!celt_tf_select_can_affect(
                CeltFrameSize::Ms2_5,
                false,
                &choices
            ));
        }
    }

    /// 2.5 ms transient: Tables 62 and 63 are also identical
    /// (`[[0, -1]]`). `tf_select` has nothing to do; the flag is not
    /// encoded.
    #[test]
    fn ms2_5_transient_tf_select_never_matters() {
        for choices in [
            vec![],
            vec![false],
            vec![true],
            vec![false, true],
            vec![true, true, false, false],
        ] {
            assert!(!celt_tf_select_can_affect(
                CeltFrameSize::Ms2_5,
                true,
                &choices
            ));
        }
    }

    /// 10 ms non-transient: Tables 60 and 61 disagree on `choice = 1`
    /// (-2 vs -3) but agree on `choice = 0` (both 0). If every band
    /// picked `choice = 0`, `tf_select` cannot affect the result and is
    /// not encoded.
    #[test]
    fn ms10_nontransient_tf_select_silent_when_all_choice0() {
        assert!(!celt_tf_select_can_affect(
            CeltFrameSize::Ms10,
            false,
            &[false, false, false, false],
        ));
        // A single `choice = 1` band is enough to make tf_select matter.
        assert!(celt_tf_select_can_affect(
            CeltFrameSize::Ms10,
            false,
            &[false, false, true, false],
        ));
    }

    /// 5 ms transient: Tables 62 and 63 disagree on `choice = 1`
    /// (0 vs -1). A single `choice = 1` band makes `tf_select`
    /// matter.
    #[test]
    fn ms5_transient_tf_select_matters_when_any_choice1() {
        assert!(!celt_tf_select_can_affect(
            CeltFrameSize::Ms5,
            true,
            &[false, false]
        ));
        assert!(celt_tf_select_can_affect(
            CeltFrameSize::Ms5,
            true,
            &[false, true]
        ));
    }

    /// 20 ms transient: Tables 62 ([3, 0]) and 63 ([1, -1]) disagree on
    /// BOTH choices. `tf_select` matters as soon as there's at least
    /// one band of any choice.
    #[test]
    fn ms20_transient_tf_select_matters_for_any_nonempty_band_set() {
        assert!(celt_tf_select_can_affect(
            CeltFrameSize::Ms20,
            true,
            &[false]
        ));
        assert!(celt_tf_select_can_affect(
            CeltFrameSize::Ms20,
            true,
            &[true]
        ));
        assert!(celt_tf_select_can_affect(
            CeltFrameSize::Ms20,
            true,
            &[true, false]
        ));
    }

    // ---------- TfDirection coverage ----------

    /// `TfDirection::from_adjustment` round-trips correctly for every
    /// cell of every documented table.
    #[test]
    fn tf_direction_classifies_every_documented_cell() {
        for table in [
            &TF_ADJ_NONTRANSIENT_SELECT0,
            &TF_ADJ_NONTRANSIENT_SELECT1,
            &TF_ADJ_TRANSIENT_SELECT0,
            &TF_ADJ_TRANSIENT_SELECT1,
        ] {
            for row in table.iter() {
                for &cell in row.iter() {
                    let dir = TfDirection::from_adjustment(cell);
                    match (cell, dir) {
                        (0, TfDirection::Unchanged) => {}
                        (n, TfDirection::IncreaseTime(levels)) if n < 0 => {
                            assert_eq!(levels as i16, -(n as i16));
                        }
                        (n, TfDirection::IncreaseFrequency(levels)) if n > 0 => {
                            assert_eq!(levels as i16, n as i16);
                        }
                        (cell, dir) => {
                            panic!("cell {cell} classified as unexpected {dir:?}");
                        }
                    }
                }
            }
        }
    }

    /// `TfDirection::levels` returns `adj.unsigned_abs()` for every
    /// documented cell.
    #[test]
    fn tf_direction_levels_equals_abs_value() {
        for adj in -3i8..=3i8 {
            let dir = TfDirection::from_adjustment(adj);
            assert_eq!(dir.levels(), adj.unsigned_abs());
        }
    }

    /// `IncreaseFrequency` only ever appears on transient frames: pin
    /// the §4.3.4.5 asymmetry through the classification layer too.
    #[test]
    fn increase_frequency_only_reachable_on_transient_frames() {
        for table in [&TF_ADJ_NONTRANSIENT_SELECT0, &TF_ADJ_NONTRANSIENT_SELECT1] {
            for row in table.iter() {
                for &cell in row.iter() {
                    assert!(
                        !matches!(
                            TfDirection::from_adjustment(cell),
                            TfDirection::IncreaseFrequency(_)
                        ),
                        "non-transient cell {cell} should not request frequency upgrade"
                    );
                }
            }
        }
    }

    /// `IncreaseTime` is reachable on every frame size at `choice = 1`
    /// for non-transient frames. (Every Table 60 / 61 row's
    /// `choice = 1` cell is < 0.)
    #[test]
    fn nontransient_choice1_always_increases_time() {
        for table in [&TF_ADJ_NONTRANSIENT_SELECT0, &TF_ADJ_NONTRANSIENT_SELECT1] {
            for row in table.iter() {
                let dir = TfDirection::from_adjustment(row[1]);
                assert!(matches!(dir, TfDirection::IncreaseTime(_)));
            }
        }
    }
}
