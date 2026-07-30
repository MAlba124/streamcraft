//! CELT §4.3.4.5 time-frequency Hadamard transform
//! (RFC 6716 §4.3.4.5, pp. 119–120).
//!
//! The per-band time-frequency (TF) adjustment decoded by
//! [`crate::celt_tf_decode`] and classified by
//! [`crate::celt_tf_adjust::TfDirection`] selects a change in the
//! time-vs-frequency resolution trade-off for a band. The RFC states
//! the consumption of that adjustment directly (RFC 6716 §4.3.4.5,
//! p. 119):
//!
//! > A negative TF adjustment means that the temporal resolution is
//! > increased, while a positive TF adjustment means that the
//! > frequency resolution is increased. Changes in TF resolution are
//! > implemented using the Hadamard transform. To increase the time
//! > resolution by N, N "levels" of the Hadamard transform are applied
//! > to the decoded vector for each interleaved MDCT vector. To
//! > increase the frequency resolution (assumes a transient frame),
//! > then N levels of the Hadamard transform are applied _across_ the
//! > interleaved MDCT vector. In the case of increased time
//! > resolution, the decoder uses the "sequency order" because the
//! > input vector is sorted in time.
//!
//! ## Band storage convention
//!
//! A transient CELT frame stores each band as `B` short-MDCT blocks of
//! equal length `M = N / B`, laid out **interleaved**: coefficient
//! position `m` of block `b` lives at flat index `m * B + b`
//! (block-minor / bin-major). This is the "interleaved MDCT vector"
//! the RFC refers to: reading the flat vector with stride `B` recovers
//! one MDCT block's coefficients, and reading a contiguous run of `B`
//! recovers the `B` blocks' coefficients for one frequency position.
//!
//! Under this convention the two RFC operations are:
//!
//! * **Increase frequency resolution** (`adj > 0`): apply the
//!   `2**N`-point Hadamard transform *across* the blocks — i.e. to each
//!   contiguous group of `B` samples that shares a frequency position.
//!   `N` is the number of levels; `2**N == B` (the RFC bounds the level
//!   count so the transform exactly spans the available blocks). The
//!   "natural"/Hadamard (Walsh–Hadamard, `H_n = H_1 (x) H_{n-1}`)
//!   ordering is used here because the across-block axis is not a
//!   time-ordered axis.
//!
//! * **Increase time resolution** (`adj < 0`): apply the `2**N`-point
//!   Hadamard transform to "each interleaved MDCT vector" in
//!   **sequency order** ("because the input vector is sorted in time").
//!   Sequency order is the Walsh-ordered Hadamard transform: the rows
//!   of the natural Hadamard matrix permuted into ascending
//!   sign-change (sequency) count. Concretely this implementation
//!   builds the `2**N`-point butterfly and then permutes the result by
//!   the bit-reversal-then-Gray-code index map that takes natural
//!   (Hadamard) order to sequency (Walsh) order.
//!
//! ## Normalisation
//!
//! The Hadamard butterfly used here is the **orthonormal** one: each
//! 2-point stage is `(a + b)/sqrt(2)`, `(a - b)/sqrt(2)`. This keeps
//! the L2 norm of the band's unit-norm shape vector exactly preserved
//! across the TF transform (the §4.3.6 denormalisation multiplies by
//! `sqrt(energy)`, so the shape entering it must stay unit-norm). The
//! transform is its own inverse under this normalisation
//! (`H · H = I`), so the same routine deinterleaves on decode.
//!
//! ## Verification scope
//!
//! The L2-norm preservation and self-inverse properties are pinned by
//! unit tests and are the load-bearing correctness invariants for the
//! §4.3.6 denormalisation that consumes this output. The natural-vs-
//! sequency *ordering* choice changes only the within-band coefficient
//! permutation, not the energy, and cannot be bit-exactly confirmed
//! against a real bitstream until the §4.3.3 allocation orchestration
//! (the `interp_bits2pulses` reallocation / skip / fine-split, absent
//! from the RFC narrative) is unblocked and a transient CELT fixture
//! can be decoded end-to-end. The standard sequency permutation is
//! pinned by `sequency_permutation_level2_is_correct`.
//!
//! ## Provenance
//!
//! Narrative + algorithm: RFC 6716 §4.3.4.5 (pp. 119–120), reproduced
//! from `docs/audio/opus/rfc6716-opus.txt`. The Walsh–Hadamard
//! transform and its sequency (Walsh) ordering are standard
//! mathematical definitions [HADAMARD]. No external library source was
//! consulted.

use crate::celt_tf_adjust::TfDirection;

/// `1/sqrt(2)`, the per-stage orthonormal Hadamard butterfly scale.
const INV_SQRT2: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// Errors returnable by the §4.3.4.5 Hadamard TF transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TfHadamardError {
    /// `nb_blocks` is zero. A band always has at least one MDCT block.
    ZeroBlocks,
    /// `nb_blocks` is not a power of two. The Hadamard transform spans
    /// `2**levels` blocks, so the block count must itself be a power of
    /// two for the across-block transform to be well-defined.
    BlocksNotPowerOfTwo {
        /// The offending block count.
        nb_blocks: usize,
    },
    /// The flat vector length is not a multiple of `nb_blocks`, so it
    /// cannot represent `nb_blocks` equal interleaved blocks.
    BlocksDoNotDivideLength {
        /// The flat vector length.
        len: usize,
        /// The block count it was expected to divide.
        nb_blocks: usize,
    },
    /// The requested level count exceeds `log2(nb_blocks)`: a Hadamard
    /// transform of `2**levels` points cannot be applied across (or
    /// to) only `nb_blocks` blocks when `2**levels > nb_blocks`.
    LevelsExceedBlocks {
        /// Requested Hadamard levels.
        levels: u8,
        /// Available block count.
        nb_blocks: usize,
    },
}

impl core::fmt::Display for TfHadamardError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            TfHadamardError::ZeroBlocks => {
                write!(
                    f,
                    "oxideav-opus: CELT §4.3.4.5 TF transform requires nb_blocks >= 1"
                )
            }
            TfHadamardError::BlocksNotPowerOfTwo { nb_blocks } => write!(
                f,
                "oxideav-opus: CELT §4.3.4.5 TF transform requires a power-of-two block \
                 count, got nb_blocks = {nb_blocks}"
            ),
            TfHadamardError::BlocksDoNotDivideLength { len, nb_blocks } => write!(
                f,
                "oxideav-opus: CELT §4.3.4.5 TF transform: vector length {len} is not a \
                 multiple of nb_blocks = {nb_blocks}"
            ),
            TfHadamardError::LevelsExceedBlocks { levels, nb_blocks } => write!(
                f,
                "oxideav-opus: CELT §4.3.4.5 TF transform: {levels} Hadamard levels exceed \
                 log2(nb_blocks) for nb_blocks = {nb_blocks}"
            ),
        }
    }
}

impl std::error::Error for TfHadamardError {}

/// In-place orthonormal Walsh–Hadamard transform (natural / Hadamard
/// order) of a `2**levels`-point slice.
///
/// `x.len()` must equal `1 << levels`. Each stage applies the 2-point
/// orthonormal butterfly `(a + b)/sqrt(2)`, `(a - b)/sqrt(2)` at the
/// stage's stride. The transform is symmetric and orthogonal, so it is
/// its own inverse and preserves the L2 norm exactly (up to floating
/// rounding).
///
/// This is the natural (Hadamard, `H_n = H_1 (x) H_{n-1}`) ordering.
/// The sequency (Walsh) ordering is obtained by additionally permuting
/// the output via [`sequency_permutation`].
fn fwht_natural_inplace(x: &mut [f64]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut stride = 1;
    while stride < n {
        let mut base = 0;
        while base < n {
            for i in base..base + stride {
                let a = x[i];
                let b = x[i + stride];
                x[i] = (a + b) * INV_SQRT2;
                x[i + stride] = (a - b) * INV_SQRT2;
            }
            base += stride << 1;
        }
        stride <<= 1;
    }
}

/// Number of set bits in `v` interpreted over `bits` low bits, used to
/// derive the sequency (sign-change count) order.
///
/// The sequency of natural-Hadamard row `r` (over `bits` bits) is
/// `gray_to_index(bit_reverse(r))`: bit-reversing the index and then
/// converting from Gray code yields the row's sign-change count. The
/// permutation that sorts natural rows by ascending sequency is exactly
/// this map, and it is an involution composed of a bit-reversal and a
/// Gray-decode.
#[inline]
fn bit_reverse(mut v: usize, bits: u32) -> usize {
    let mut r = 0;
    for _ in 0..bits {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

/// Gray-code → binary (inverse Gray) over `bits` low bits.
#[inline]
fn inverse_gray(mut v: usize) -> usize {
    let mut mask = v >> 1;
    while mask != 0 {
        v ^= mask;
        mask >>= 1;
    }
    v
}

/// Build the permutation that reorders a natural-order Walsh–Hadamard
/// transform of `2**bits` points into sequency (Walsh) order.
///
/// `perm[i]` is the natural-order index whose sequency rank is `i`;
/// i.e. `sequency_out[i] = natural_out[perm[i]]`.
fn sequency_permutation(bits: u32) -> Vec<usize> {
    let n = 1usize << bits;
    // The sequency rank of natural row r is inverse_gray(bit_reverse(r)).
    // We want the inverse map: for each target sequency rank s, find the
    // natural row r. Build rank[r] = s, then invert.
    let mut rank = vec![0usize; n];
    for (r, slot) in rank.iter_mut().enumerate() {
        *slot = inverse_gray(bit_reverse(r, bits));
    }
    let mut perm = vec![0usize; n];
    for (r, &s) in rank.iter().enumerate() {
        perm[s] = r;
    }
    perm
}

/// Apply the §4.3.4.5 Hadamard time-frequency transform to one band's
/// interleaved shape vector, in place.
///
/// * `x` — the band's flat interleaved coefficient vector. Length must
///   be a multiple of `nb_blocks`; coefficient `m` of block `b` lives
///   at flat index `m * nb_blocks + b` (bin-major / block-minor).
/// * `nb_blocks` — the number of interleaved MDCT blocks `B` the band
///   represents (a power of two; `1` for a non-transient band).
/// * `direction` — the [`TfDirection`] classification of the band's TF
///   adjustment.
///
/// For [`TfDirection::Unchanged`] the vector is left untouched. For
/// [`TfDirection::IncreaseFrequency`] the across-block Walsh–Hadamard
/// transform (natural order) is applied to each contiguous group of
/// `B` samples sharing a frequency position. For
/// [`TfDirection::IncreaseTime`] the same butterfly is applied in
/// sequency (Walsh) order.
///
/// The transform spans `2**levels` blocks; the RFC bounds the level
/// count so `2**levels <= nb_blocks`. When `2**levels < nb_blocks` the
/// transform is applied independently to each consecutive run of
/// `2**levels` blocks within the `B`-block group, matching the
/// "N levels of the Hadamard transform" wording (the remaining
/// `log2(B) - levels` axes stay in their pre-transform resolution).
///
/// # Errors
///
/// Returns a [`TfHadamardError`] when `nb_blocks` is zero, not a power
/// of two, does not divide `x.len()`, or when `2**levels` exceeds
/// `nb_blocks`.
pub fn apply_tf_hadamard(
    x: &mut [f64],
    nb_blocks: usize,
    direction: TfDirection,
) -> Result<(), TfHadamardError> {
    if nb_blocks == 0 {
        return Err(TfHadamardError::ZeroBlocks);
    }
    if !nb_blocks.is_power_of_two() {
        return Err(TfHadamardError::BlocksNotPowerOfTwo { nb_blocks });
    }
    let len = x.len();
    if len % nb_blocks != 0 {
        return Err(TfHadamardError::BlocksDoNotDivideLength { len, nb_blocks });
    }

    let levels = direction.levels();
    if levels == 0 {
        return Ok(()); // TfDirection::Unchanged.
    }
    let span = 1usize << levels;
    if span > nb_blocks {
        return Err(TfHadamardError::LevelsExceedBlocks { levels, nb_blocks });
    }

    let bins = len / nb_blocks; // per-block coefficient count M = N / B.
    let sequency = matches!(direction, TfDirection::IncreaseTime(_));
    let perm = if sequency {
        Some(sequency_permutation(levels as u32))
    } else {
        None
    };

    // For each frequency position `m`, the `nb_blocks` coefficients are
    // the contiguous run `x[m*nb_blocks .. m*nb_blocks + nb_blocks]`.
    // Apply the 2**levels-point transform to each consecutive sub-run of
    // `span` blocks within that run.
    let mut scratch = vec![0.0f64; span];
    for m in 0..bins {
        let base = m * nb_blocks;
        let mut off = 0;
        while off + span <= nb_blocks {
            let group = &mut x[base + off..base + off + span];
            scratch.copy_from_slice(group);
            fwht_natural_inplace(&mut scratch);
            if let Some(perm) = &perm {
                for (dst, &src) in group.iter_mut().zip(perm.iter()) {
                    *dst = scratch[src];
                }
            } else {
                group.copy_from_slice(&scratch);
            }
            off += span;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn l2(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    #[test]
    fn unchanged_is_identity() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = x.clone();
        apply_tf_hadamard(&mut x, 4, TfDirection::Unchanged).unwrap();
        assert_eq!(x, orig);
    }

    #[test]
    fn single_block_no_op_even_with_levels() {
        // nb_blocks == 1 ⇒ span 2 exceeds blocks; only levels==0 is
        // legal, and Unchanged leaves it alone. A non-zero direction
        // with nb_blocks==1 must error (span > nb_blocks).
        let mut x = vec![1.0, 2.0, 3.0];
        let r = apply_tf_hadamard(&mut x, 1, TfDirection::IncreaseTime(1));
        assert_eq!(
            r,
            Err(TfHadamardError::LevelsExceedBlocks {
                levels: 1,
                nb_blocks: 1
            })
        );
    }

    #[test]
    fn two_point_frequency_butterfly() {
        // nb_blocks = 2, one bin, IncreaseFrequency(1):
        // [a, b] -> [(a+b)/√2, (a-b)/√2].
        let mut x = vec![3.0, 1.0];
        apply_tf_hadamard(&mut x, 2, TfDirection::IncreaseFrequency(1)).unwrap();
        assert!((x[0] - 4.0 * INV_SQRT2).abs() < TOL);
        assert!((x[1] - 2.0 * INV_SQRT2).abs() < TOL);
    }

    #[test]
    fn orthonormal_preserves_l2_norm() {
        // The transform must preserve the unit-norm shape's energy.
        for dir in [
            TfDirection::IncreaseFrequency(1),
            TfDirection::IncreaseFrequency(2),
            TfDirection::IncreaseTime(1),
            TfDirection::IncreaseTime(2),
        ] {
            let mut x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.37).sin()).collect();
            let before = l2(&x);
            apply_tf_hadamard(&mut x, 4, dir).unwrap();
            let after = l2(&x);
            assert!(
                (before - after).abs() < 1e-9,
                "dir {dir:?}: {before} vs {after}"
            );
        }
    }

    #[test]
    fn transform_is_self_inverse() {
        // H·H = I under the orthonormal normalisation, for both axes.
        for (dir, blocks) in [
            (TfDirection::IncreaseFrequency(2), 4usize),
            (TfDirection::IncreaseTime(2), 4usize),
            (TfDirection::IncreaseFrequency(3), 8usize),
            (TfDirection::IncreaseTime(3), 8usize),
        ] {
            let orig: Vec<f64> = (0..blocks * 3).map(|i| (i as f64 * 0.13).cos()).collect();
            let mut x = orig.clone();
            apply_tf_hadamard(&mut x, blocks, dir).unwrap();
            apply_tf_hadamard(&mut x, blocks, dir).unwrap();
            for (a, b) in x.iter().zip(orig.iter()) {
                assert!((a - b).abs() < 1e-9, "self-inverse {dir:?}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn frequency_and_time_differ_for_multilevel() {
        // For levels >= 2 the natural (frequency) and sequency (time)
        // orderings produce different outputs — the permutation is not
        // the identity. This pins that the sequency path actually
        // reorders.
        let base: Vec<f64> = (0..4).map(|i| (i as f64 + 1.0) * 0.5).collect();
        let mut freq = base.clone();
        let mut time = base.clone();
        apply_tf_hadamard(&mut freq, 4, TfDirection::IncreaseFrequency(2)).unwrap();
        apply_tf_hadamard(&mut time, 4, TfDirection::IncreaseTime(2)).unwrap();
        // Same multiset of magnitudes (a permutation of each other) but
        // not identical ordering.
        assert_ne!(
            freq.iter().map(|v| format!("{v:.9}")).collect::<Vec<_>>(),
            time.iter().map(|v| format!("{v:.9}")).collect::<Vec<_>>()
        );
    }

    #[test]
    fn level1_sequency_equals_natural() {
        // At a single level the 2-point transform has a trivial
        // permutation, so time and frequency agree.
        let base = vec![2.0, 5.0];
        let mut freq = base.clone();
        let mut time = base.clone();
        apply_tf_hadamard(&mut freq, 2, TfDirection::IncreaseFrequency(1)).unwrap();
        apply_tf_hadamard(&mut time, 2, TfDirection::IncreaseTime(1)).unwrap();
        for (a, b) in freq.iter().zip(time.iter()) {
            assert!((a - b).abs() < TOL);
        }
    }

    #[test]
    fn sequency_permutation_level2_is_correct() {
        // Natural Hadamard rows of order 4, by sequency (sign changes):
        //   row 0: +,+,+,+  → 0 changes → sequency 0
        //   row 1: +,-,+,-  → 3 changes → sequency 3
        //   row 2: +,+,-,-  → 1 change  → sequency 1
        //   row 3: +,-,-,+  → 2 changes → sequency 2
        // So sequency order picks natural rows [0, 2, 3, 1].
        let perm = sequency_permutation(2);
        assert_eq!(perm, vec![0, 2, 3, 1]);
    }

    #[test]
    fn partial_levels_transform_subgroups_independently() {
        // nb_blocks = 4 but only 1 level: each consecutive pair of
        // blocks is butterflied independently, leaving the cross-pair
        // structure untouched.
        let mut x = vec![1.0, 3.0, 10.0, 6.0]; // one bin, 4 blocks
        apply_tf_hadamard(&mut x, 4, TfDirection::IncreaseFrequency(1)).unwrap();
        // pair (1,3) -> (4/√2, -2/√2); pair (10,6) -> (16/√2, 4/√2)
        assert!((x[0] - 4.0 * INV_SQRT2).abs() < TOL);
        assert!((x[1] + 2.0 * INV_SQRT2).abs() < TOL);
        assert!((x[2] - 16.0 * INV_SQRT2).abs() < TOL);
        assert!((x[3] - 4.0 * INV_SQRT2).abs() < TOL);
    }

    #[test]
    fn multi_bin_each_bin_independent() {
        // 2 bins, 2 blocks. bin 0 = [a,b] at indices 0,1; bin 1 = [c,d]
        // at indices 2,3 (bin-major / block-minor interleave).
        let mut x = vec![3.0, 1.0, 8.0, 2.0];
        apply_tf_hadamard(&mut x, 2, TfDirection::IncreaseFrequency(1)).unwrap();
        assert!((x[0] - 4.0 * INV_SQRT2).abs() < TOL);
        assert!((x[1] - 2.0 * INV_SQRT2).abs() < TOL);
        assert!((x[2] - 10.0 * INV_SQRT2).abs() < TOL);
        assert!((x[3] - 6.0 * INV_SQRT2).abs() < TOL);
    }

    #[test]
    fn errors_on_non_power_of_two_blocks() {
        let mut x = vec![1.0; 6];
        assert_eq!(
            apply_tf_hadamard(&mut x, 3, TfDirection::IncreaseFrequency(1)),
            Err(TfHadamardError::BlocksNotPowerOfTwo { nb_blocks: 3 })
        );
    }

    #[test]
    fn errors_on_zero_blocks() {
        let mut x = vec![1.0; 4];
        assert_eq!(
            apply_tf_hadamard(&mut x, 0, TfDirection::IncreaseFrequency(1)),
            Err(TfHadamardError::ZeroBlocks)
        );
    }

    #[test]
    fn errors_when_blocks_do_not_divide_length() {
        let mut x = vec![1.0; 5];
        assert_eq!(
            apply_tf_hadamard(&mut x, 2, TfDirection::IncreaseFrequency(1)),
            Err(TfHadamardError::BlocksDoNotDivideLength {
                len: 5,
                nb_blocks: 2
            })
        );
    }

    #[test]
    fn fwht_natural_matches_hand_computed_order4() {
        // Orthonormal natural Hadamard of [1,0,0,0] is the first column
        // of H4/2 = [1,1,1,1]/2.
        let mut x = vec![1.0, 0.0, 0.0, 0.0];
        fwht_natural_inplace(&mut x);
        for v in &x {
            assert!((v - 0.5).abs() < TOL);
        }
    }
}
