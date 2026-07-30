//! CELT §4.3.4.3 spreading (rotation)
//! (RFC 6716 §4.3.4.3, pp. 117–118).
//!
//! The unit-norm shape vector recovered by the §4.3.4.2 PVQ decode
//! ([`crate::celt_pvq_decode`]) is rotated "for the purpose of
//! avoiding tonal artifacts" (RFC 6716 §4.3.4.3, p. 117). The
//! rotation strength is driven by the band's dimension count `N`,
//! its pulse count `K`, and the frame-global "spread" parameter — a
//! four-valued symbol coded once per frame with the Table 56 PDF
//! `{7, 2, 21, 2}/32` and mapped to a rotation factor `f_r` by
//! Table 59.
//!
//! ## §4.3.4.3 rotation procedure
//!
//! RFC 6716 §4.3.4.3 (pp. 117–118) states the procedure directly:
//!
//! > The rotation gain is equal to
//! >
//! > `g_r = N / (N + f_r*K)`
//! >
//! > where N is the number of dimensions, K is the number of pulses,
//! > and f_r depends on the value of the "spread" parameter in the
//! > bitstream. \[Table 59: spread 0 → infinite (no rotation),
//! > 1 → 15, 2 → 10, 3 → 5.\]
//! >
//! > The rotation angle is then calculated as
//! > `theta = pi * g_r^2 / 4`.
//! >
//! > A 2-D rotation R(i,j) between points x_i and x_j is defined as:
//! >
//! > `x_i' =  cos(theta)*x_i + sin(theta)*x_j`
//! > `x_j' = -sin(theta)*x_i + cos(theta)*x_j`
//! >
//! > An N-D rotation is then achieved by applying a series of 2-D
//! > rotations back and forth, in the following order: R(x_1, x_2),
//! > R(x_2, x_3), ..., R(x_N-2, X_N-1), R(x_N-1, X_N),
//! > R(x_N-2, X_N-1), ..., R(x_1, x_2).
//!
//! With 1-based indices `x_1..x_N` the forward leg visits the `N - 1`
//! adjacent pairs in ascending order and the backward leg revisits
//! the first `N - 2` of them in descending order (`N = 2` has an
//! empty backward leg). Every step is an orthogonal 2-D rotation, so
//! the composite is orthogonal and the vector's L2 norm is preserved
//! exactly (up to floating-point rounding).
//!
//! ## §4.3.4.3 multi-block (transient) handling
//!
//! > If the decoded vector represents more than one time block, then
//! > this spreading process is applied separately on each time block.
//! > Also, if each block represents 8 samples or more, then another
//! > N-D rotation, by (pi/2-theta), is applied _before_ the rotation
//! > described above. This extra rotation is applied in an
//! > interleaved manner with a stride equal to
//! > round(sqrt(N/nb_blocks)), i.e., it is applied independently for
//! > each set of sample S_k = {stride*n + k}, n=0..N/stride-1.
//!
//! [`apply_spreading`] composes the pieces under the following
//! reading of that paragraph, documented here because the prose
//! reuses `N` for two roles: the per-block process ("applied
//! separately on each time block") evaluates `g_r` with `N` = the
//! block's dimension count, while the stride formula
//! `round(sqrt(N/nb_blocks))` divides the *whole vector's* length by
//! `nb_blocks` — both quotients equal the block length, so the
//! stride is `round(sqrt(block_len))` either way. The interleaved
//! sets `S_k = {stride*n + k}` are taken over the whole vector (the
//! `n = 0..N/stride-1` bound is stated against the full `N`), and
//! the pre-rotation runs only on the multi-block path (the paragraph
//! is scoped by "If the decoded vector represents more than one time
//! block"). The standalone primitives ([`rotate_in_place`],
//! [`rotate_strided`], [`rotation_gain`], [`rotation_angle`],
//! [`spreading_stride`]) are exact transcriptions, so a consumer
//! site can recompose them if fixture-level verification later pins
//! a different reading.
//!
//! `round()` in the stride formula is not further specified by the
//! RFC; this module uses round-half-away-from-zero (the quotient is
//! non-negative, so this is round-half-up).
//!
//! ## Provenance
//!
//! Narrative + algorithm: RFC 6716 §4.3.4.3 (pp. 117–118), Table 59
//! (p. 117), and the Table 56 "spread" PDF row (p. 107), all
//! reproduced from `docs/audio/opus/rfc6716-opus.txt`. No external
//! library source was consulted.

use crate::range_decoder::RangeDecoder;

/// Number of valid "spread" symbol values (RFC 6716 Table 59:
/// `0..=3`).
pub const SPREAD_VALUE_COUNT: usize = 4;

/// Largest valid "spread" symbol value (RFC 6716 Table 59).
pub const SPREAD_MAX: u8 = 3;

/// §4.3 Table 56 PDF for the per-frame "spread" symbol:
/// `{7, 2, 21, 2}/32` (RFC 6716 p. 107).
pub const SPREAD_PDF: [u8; SPREAD_VALUE_COUNT] = [7, 2, 21, 2];

/// Number of bits spanned by the [`SPREAD_PDF`] denominator
/// (`1 << SPREAD_FTB = 32`).
pub const SPREAD_FTB: u32 = 5;

/// [`SPREAD_PDF`] denominator (`32`).
pub const SPREAD_PDF_DENOMINATOR: u32 = 1 << SPREAD_FTB;

/// Inverse-CDF form of [`SPREAD_PDF`] consumed by
/// [`RangeDecoder::dec_icdf`]: `icdf[k] = 32 - sum(pdf[0..=k])`.
pub const SPREAD_ICDF: [u8; SPREAD_VALUE_COUNT] = [25, 23, 2, 0];

/// RFC 6716 Table 59 rotation factor `f_r` per "spread" value.
/// `None` encodes the Table 59 "infinite (no rotation)" row.
pub const SPREAD_F_R: [Option<u32>; SPREAD_VALUE_COUNT] = [None, Some(15), Some(10), Some(5)];

/// §4.3.4.3 minimum per-block sample count for the multi-block
/// pre-rotation ("if each block represents 8 samples or more").
pub const SPREAD_PRE_ROTATION_MIN_BLOCK_LEN: usize = 8;

/// Errors returnable by the §4.3.4.3 spreading helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpreadingError {
    /// The "spread" symbol is outside the Table 59 range `0..=3`.
    /// `dec_icdf` over [`SPREAD_ICDF`] cannot produce this on a
    /// conforming stream, so it signals a caller-side bookkeeping
    /// bug.
    SpreadOutOfRange {
        /// The value the caller passed.
        spread: u8,
    },
    /// `N = 0` passed to [`rotation_gain`]; the §4.3.4.3 gain
    /// `g_r = N / (N + f_r*K)` is undefined for a zero-dimensional
    /// band.
    ZeroDimensions,
    /// `nb_blocks = 0` passed where a §4.3.4.3 time-block count is
    /// required.
    ZeroBlocks,
    /// `stride = 0` passed to [`rotate_strided`]; the §4.3.4.3
    /// interleaved sets `S_k = {stride*n + k}` are undefined for a
    /// zero stride.
    ZeroStride,
    /// The vector length is not a multiple of `nb_blocks`, so it
    /// cannot "represent" `nb_blocks` equal time blocks
    /// (RFC 6716 §4.3.4.3).
    BlocksDoNotDivideLength {
        /// The vector length the caller passed.
        len: usize,
        /// The block count the caller passed.
        nb_blocks: usize,
    },
}

impl core::fmt::Display for SpreadingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            SpreadingError::SpreadOutOfRange { spread } => write!(
                f,
                "oxideav-opus: CELT spread value {spread} out of range \
                 (RFC 6716 Table 59 allows 0..=3)"
            ),
            SpreadingError::ZeroDimensions => write!(
                f,
                "oxideav-opus: CELT §4.3.4.3 rotation gain requires N >= 1"
            ),
            SpreadingError::ZeroBlocks => write!(
                f,
                "oxideav-opus: CELT §4.3.4.3 spreading requires nb_blocks >= 1"
            ),
            SpreadingError::ZeroStride => write!(
                f,
                "oxideav-opus: CELT §4.3.4.3 strided rotation requires stride >= 1"
            ),
            SpreadingError::BlocksDoNotDivideLength { len, nb_blocks } => write!(
                f,
                "oxideav-opus: CELT §4.3.4.3 vector length {len} is not a \
                 multiple of nb_blocks = {nb_blocks}"
            ),
        }
    }
}

impl std::error::Error for SpreadingError {}

/// Decodes the per-frame §4.3 "spread" symbol with the Table 56 PDF
/// `{7, 2, 21, 2}/32` (RFC 6716 p. 107).
///
/// Returns a value in `0..=3`, directly usable as the [`SPREAD_F_R`]
/// / Table 59 row selector.
pub fn decode_spread(rd: &mut RangeDecoder<'_>) -> u8 {
    let symbol = rd.dec_icdf(&SPREAD_ICDF, SPREAD_FTB);
    debug_assert!(symbol < SPREAD_VALUE_COUNT as u32);
    symbol as u8
}

/// Maps a decoded "spread" symbol to the RFC 6716 Table 59 rotation
/// factor `f_r`; `Ok(None)` is the Table 59 "infinite (no rotation)"
/// row (`spread = 0`).
pub fn spread_f_r(spread: u8) -> Result<Option<u32>, SpreadingError> {
    SPREAD_F_R
        .get(usize::from(spread))
        .copied()
        .ok_or(SpreadingError::SpreadOutOfRange { spread })
}

/// §4.3.4.3 rotation gain `g_r = N / (N + f_r*K)` (RFC 6716 p. 117).
///
/// `n` is the number of dimensions the rotation series runs over,
/// `k` the band's pulse count, `f_r` the Table 59 factor. The result
/// is in `(0, 1]`; `K = 0` yields exactly `1`.
pub fn rotation_gain(n: usize, k: u32, f_r: u32) -> Result<f64, SpreadingError> {
    if n == 0 {
        return Err(SpreadingError::ZeroDimensions);
    }
    let n = n as f64;
    Ok(n / (n + f64::from(f_r) * f64::from(k)))
}

/// §4.3.4.3 rotation angle `theta = pi * g_r^2 / 4` (RFC 6716
/// p. 117). `g_r ∈ (0, 1]` maps to `theta ∈ (0, pi/4]`.
pub fn rotation_angle(g_r: f64) -> f64 {
    core::f64::consts::PI * g_r * g_r / 4.0
}

/// Composes [`spread_f_r`], [`rotation_gain`], and
/// [`rotation_angle`]: the §4.3.4.3 angle for an `(N, K, spread)`
/// triple, or `Ok(None)` when `spread = 0` selects the Table 59
/// "no rotation" row.
pub fn spread_theta(n: usize, k: u32, spread: u8) -> Result<Option<f64>, SpreadingError> {
    match spread_f_r(spread)? {
        None => Ok(None),
        Some(f_r) => Ok(Some(rotation_angle(rotation_gain(n, k, f_r)?))),
    }
}

/// One §4.3.4.3 2-D rotation `R(i, j)` step (RFC 6716 p. 118):
/// `x_i' = cos*x_i + sin*x_j`, `x_j' = -sin*x_i + cos*x_j`.
#[inline]
fn rot2(x: &mut [f64], i: usize, j: usize, cos_t: f64, sin_t: f64) {
    let xi = x[i];
    let xj = x[j];
    x[i] = cos_t * xi + sin_t * xj;
    x[j] = -sin_t * xi + cos_t * xj;
}

/// Applies the §4.3.4.3 N-D rotation by `theta` to `x` in place:
/// the back-and-forth series `R(x_1, x_2), ..., R(x_N-1, x_N),
/// R(x_N-2, x_N-1), ..., R(x_1, x_2)` (RFC 6716 p. 118).
///
/// Vectors shorter than two elements have no adjacent pair and are
/// returned unchanged. The composite is orthogonal: the L2 norm is
/// preserved up to floating-point rounding.
pub fn rotate_in_place(x: &mut [f64], theta: f64) {
    let n = x.len();
    if n < 2 {
        return;
    }
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    // Forward leg: R(x_1, x_2) .. R(x_{N-1}, x_N) — 0-based pairs
    // (0, 1) .. (n-2, n-1).
    for i in 0..n - 1 {
        rot2(x, i, i + 1, cos_t, sin_t);
    }
    // Backward leg: R(x_{N-2}, x_{N-1}) .. R(x_1, x_2) — 0-based
    // pairs (n-3, n-2) .. (0, 1); empty when n = 2.
    for i in (0..n - 2).rev() {
        rot2(x, i, i + 1, cos_t, sin_t);
    }
}

/// §4.3.4.3 interleave stride `round(sqrt(N/nb_blocks))` (RFC 6716
/// p. 118), with `N = len`.
///
/// The RFC does not further specify `round()`; this uses
/// round-half-away-from-zero on the exact real quotient (the
/// quotient is non-negative, so this is round-half-up). A stride of
/// `0` can only arise from `len = 0` and is reported as `1` so the
/// (empty) strided walk stays well defined.
pub fn spreading_stride(len: usize, nb_blocks: usize) -> Result<usize, SpreadingError> {
    if nb_blocks == 0 {
        return Err(SpreadingError::ZeroBlocks);
    }
    let stride = (len as f64 / nb_blocks as f64).sqrt().round() as usize;
    Ok(stride.max(1))
}

/// Applies the §4.3.4.3 N-D rotation by `theta` independently to
/// each interleaved set `S_k = {stride*n + k}` of `x` (RFC 6716
/// p. 118).
///
/// Elements outside the set being rotated are untouched; `stride >=
/// x.len()` makes every set a singleton, so the call is a no-op.
pub fn rotate_strided(x: &mut [f64], stride: usize, theta: f64) -> Result<(), SpreadingError> {
    if stride == 0 {
        return Err(SpreadingError::ZeroStride);
    }
    if stride == 1 {
        rotate_in_place(x, theta);
        return Ok(());
    }
    let len = x.len();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    for k in 0..stride.min(len) {
        // Set S_k = {k, k + stride, k + 2*stride, ...} ∩ [0, len).
        let set_len = (len - k).div_ceil(stride);
        if set_len < 2 {
            continue;
        }
        // Forward leg over adjacent set members, then backward leg
        // excluding the final pair — the same §4.3.4.3 series as
        // `rotate_in_place`, walked at `stride`.
        for m in 0..set_len - 1 {
            rot2(x, k + m * stride, k + (m + 1) * stride, cos_t, sin_t);
        }
        for m in (0..set_len - 2).rev() {
            rot2(x, k + m * stride, k + (m + 1) * stride, cos_t, sin_t);
        }
    }
    Ok(())
}

/// Applies the full §4.3.4.3 spreading process to a decoded shape
/// vector in place (RFC 6716 pp. 117–118).
///
/// `x` is the band's shape vector (all `nb_blocks` time blocks,
/// equal length, concatenated), `k` the band's pulse count, `spread`
/// the decoded Table 59 symbol, and `nb_blocks` the number of time
/// blocks the vector represents.
///
/// Per the module-level reading of the §4.3.4.3 multi-block
/// paragraph:
///
/// 1. `spread = 0` (Table 59 "infinite") performs no rotation.
/// 2. Otherwise `theta` is computed from `g_r = N_b / (N_b + f_r*K)`
///    with `N_b = x.len() / nb_blocks` (the per-block dimension
///    count the rotation series runs over).
/// 3. On the multi-block path (`nb_blocks > 1`) with `N_b >= 8`, the
///    extra rotation by `pi/2 - theta` runs first, interleaved over
///    the whole vector with stride `round(sqrt(len/nb_blocks))`.
/// 4. The back-and-forth rotation by `theta` then runs separately on
///    each time block.
pub fn apply_spreading(
    x: &mut [f64],
    k: u32,
    spread: u8,
    nb_blocks: usize,
) -> Result<(), SpreadingError> {
    let f_r = spread_f_r(spread)?;
    if nb_blocks == 0 {
        return Err(SpreadingError::ZeroBlocks);
    }
    let len = x.len();
    if len % nb_blocks != 0 {
        return Err(SpreadingError::BlocksDoNotDivideLength { len, nb_blocks });
    }
    let Some(f_r) = f_r else {
        return Ok(()); // Table 59 spread = 0: no rotation.
    };
    let block_len = len / nb_blocks;
    if block_len < 2 {
        return Ok(()); // No adjacent pair to rotate.
    }
    let theta = rotation_angle(rotation_gain(block_len, k, f_r)?);
    if nb_blocks > 1 && block_len >= SPREAD_PRE_ROTATION_MIN_BLOCK_LEN {
        let stride = spreading_stride(len, nb_blocks)?;
        rotate_strided(x, stride, core::f64::consts::FRAC_PI_2 - theta)?;
    }
    for block in x.chunks_exact_mut(block_len) {
        rotate_in_place(block, theta);
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

    /// Deterministic LCG so property tests need no external RNG.
    fn pseudo_vector(len: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // RFC 6716 §4.2.7.7-style 32-bit LCG constants are
                // not needed here; any full-period mixer works for
                // test data.
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
            })
            .collect()
    }

    // ---- Table 56 PDF / iCDF ----

    #[test]
    fn spread_pdf_sums_to_denominator() {
        let sum: u32 = SPREAD_PDF.iter().map(|&p| u32::from(p)).sum();
        assert_eq!(sum, SPREAD_PDF_DENOMINATOR);
        assert_eq!(1u32 << SPREAD_FTB, SPREAD_PDF_DENOMINATOR);
    }

    #[test]
    fn spread_icdf_matches_pdf() {
        let mut acc = SPREAD_PDF_DENOMINATOR;
        for (k, &p) in SPREAD_PDF.iter().enumerate() {
            acc -= u32::from(p);
            assert_eq!(u32::from(SPREAD_ICDF[k]), acc, "icdf cell {k}");
        }
        assert_eq!(SPREAD_ICDF[SPREAD_VALUE_COUNT - 1], 0);
    }

    #[test]
    fn decode_spread_is_always_in_table_59_range() {
        // Exhaustive first-byte sweep: whatever the stream contents,
        // the decoded symbol must be a valid Table 59 row.
        for b0 in 0..=u8::MAX {
            let buf = [b0, 0xA5, 0x5A, 0xFF];
            let mut rd = RangeDecoder::new(&buf);
            let spread = decode_spread(&mut rd);
            assert!(spread <= SPREAD_MAX, "byte {b0:#x} -> spread {spread}");
            assert!(spread_f_r(spread).is_ok());
        }
    }

    // ---- Table 59 f_r mapping ----

    #[test]
    fn table_59_f_r_mapping() {
        assert_eq!(spread_f_r(0), Ok(None));
        assert_eq!(spread_f_r(1), Ok(Some(15)));
        assert_eq!(spread_f_r(2), Ok(Some(10)));
        assert_eq!(spread_f_r(3), Ok(Some(5)));
    }

    #[test]
    fn spread_out_of_range_is_rejected() {
        for spread in 4..=u8::MAX {
            assert_eq!(
                spread_f_r(spread),
                Err(SpreadingError::SpreadOutOfRange { spread })
            );
        }
    }

    // ---- g_r / theta ----

    #[test]
    fn rotation_gain_worked_points() {
        // g_r = N / (N + f_r*K): 16 / (16 + 5*4) = 16/36 = 4/9.
        let g = rotation_gain(16, 4, 5).unwrap();
        assert!((g - 4.0 / 9.0).abs() < TOL);
        // K = 0 ⇒ g_r = 1 exactly.
        assert_eq!(rotation_gain(7, 0, 15).unwrap(), 1.0);
        // N = 1, K = 1, f_r = 15 ⇒ 1/16.
        assert!((rotation_gain(1, 1, 15).unwrap() - 1.0 / 16.0).abs() < TOL);
    }

    #[test]
    fn rotation_gain_zero_dimensions_is_rejected() {
        assert_eq!(rotation_gain(0, 1, 5), Err(SpreadingError::ZeroDimensions));
    }

    #[test]
    fn rotation_angle_worked_points() {
        // theta = pi * g_r^2 / 4: g_r = 1 ⇒ pi/4; g_r = 0 ⇒ 0.
        assert!((rotation_angle(1.0) - core::f64::consts::FRAC_PI_4).abs() < TOL);
        assert_eq!(rotation_angle(0.0), 0.0);
        // g_r = 4/9 ⇒ theta = pi * 16/81 / 4 = 4*pi/81.
        let theta = rotation_angle(4.0 / 9.0);
        assert!((theta - 4.0 * core::f64::consts::PI / 81.0).abs() < TOL);
    }

    #[test]
    fn theta_shrinks_with_more_pulses_and_larger_f_r() {
        // More pulses ⇒ smaller g_r ⇒ smaller theta (fixed N, f_r).
        let t1 = spread_theta(16, 1, 3).unwrap().unwrap();
        let t4 = spread_theta(16, 4, 3).unwrap().unwrap();
        let t16 = spread_theta(16, 16, 3).unwrap().unwrap();
        assert!(t1 > t4 && t4 > t16);
        // Larger f_r (lighter spreading) ⇒ smaller theta (fixed N, K):
        // Table 59 spread 1 (f_r = 15) < spread 2 (10) < spread 3 (5).
        let s1 = spread_theta(16, 4, 1).unwrap().unwrap();
        let s2 = spread_theta(16, 4, 2).unwrap().unwrap();
        let s3 = spread_theta(16, 4, 3).unwrap().unwrap();
        assert!(s1 < s2 && s2 < s3);
        // Spread 0: Table 59 "infinite", no angle.
        assert_eq!(spread_theta(16, 4, 0), Ok(None));
    }

    // ---- the 2-D step and the N-D series ----

    #[test]
    fn two_dim_rotation_matches_definition() {
        // N = 2: the series is the single step R(x_1, x_2).
        // x = [1, 0] ⇒ x' = [cos, -sin] per the §4.3.4.3 definition.
        let theta = 0.3;
        let mut x = [1.0, 0.0];
        rotate_in_place(&mut x, theta);
        assert!((x[0] - theta.cos()).abs() < TOL);
        assert!((x[1] + theta.sin()).abs() < TOL);
    }

    #[test]
    fn three_dim_series_matches_matrix_composition() {
        // Independent check: compose the series R(0,1), R(1,2),
        // R(0,1) as explicit 3×3 matrix products and compare.
        let theta = 0.47f64;
        let (c, s) = (theta.cos(), theta.sin());
        let r01 = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]];
        let r12 = [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]];
        let mat_vec = |m: &[[f64; 3]; 3], v: [f64; 3]| {
            [
                m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
            ]
        };
        let v = [0.6, -1.1, 0.35];
        let expected = mat_vec(&r01, mat_vec(&r12, mat_vec(&r01, v)));
        let mut x = v;
        rotate_in_place(&mut x, theta);
        for (got, want) in x.iter().zip(expected.iter()) {
            assert!((got - want).abs() < TOL, "{got} vs {want}");
        }
    }

    #[test]
    fn rotation_preserves_l2_norm() {
        for n in 2..=16 {
            for (i, theta) in [0.0, 0.1, core::f64::consts::FRAC_PI_4, 1.2]
                .iter()
                .enumerate()
            {
                let mut x = pseudo_vector(n, (n * 31 + i) as u64);
                let before = l2(&x);
                rotate_in_place(&mut x, *theta);
                assert!((l2(&x) - before).abs() < 1e-9, "n={n} theta={theta}");
            }
        }
    }

    #[test]
    fn zero_angle_is_identity() {
        let mut x = pseudo_vector(9, 7);
        let orig = x.clone();
        rotate_in_place(&mut x, 0.0);
        for (got, want) in x.iter().zip(orig.iter()) {
            assert!((got - want).abs() < TOL);
        }
    }

    #[test]
    fn rotation_is_linear_in_sign() {
        let theta = 0.9;
        let x = pseudo_vector(11, 99);
        let mut pos = x.clone();
        let mut neg: Vec<f64> = x.iter().map(|v| -v).collect();
        rotate_in_place(&mut pos, theta);
        rotate_in_place(&mut neg, theta);
        for (p, n) in pos.iter().zip(neg.iter()) {
            assert!((p + n).abs() < TOL);
        }
    }

    #[test]
    fn short_vectors_are_untouched() {
        let mut empty: [f64; 0] = [];
        rotate_in_place(&mut empty, 1.0);
        let mut one = [2.5];
        rotate_in_place(&mut one, 1.0);
        assert_eq!(one, [2.5]);
    }

    // ---- stride ----

    #[test]
    fn spreading_stride_worked_points() {
        // round(sqrt(N/nb_blocks)).
        assert_eq!(spreading_stride(16, 1), Ok(4)); // sqrt(16) = 4
        assert_eq!(spreading_stride(15, 1), Ok(4)); // sqrt(15) ≈ 3.873
        assert_eq!(spreading_stride(12, 1), Ok(3)); // sqrt(12) ≈ 3.464
        assert_eq!(spreading_stride(8, 4), Ok(1)); // sqrt(2) ≈ 1.414
        assert_eq!(spreading_stride(32, 2), Ok(4)); // sqrt(16) = 4
        assert_eq!(spreading_stride(25, 4), Ok(3)); // sqrt(6.25) = 2.5 → 3
        assert_eq!(spreading_stride(0, 3), Ok(1)); // floor at 1
        assert_eq!(spreading_stride(16, 0), Err(SpreadingError::ZeroBlocks));
    }

    // ---- strided rotation ----

    #[test]
    fn strided_rotation_stride_one_matches_plain() {
        let theta = 0.6;
        let mut a = pseudo_vector(10, 5);
        let mut b = a.clone();
        rotate_in_place(&mut a, theta);
        rotate_strided(&mut b, 1, theta).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn strided_rotation_matches_gather_rotate_scatter() {
        let theta = 0.8;
        let stride = 3;
        let mut x = pseudo_vector(14, 21);
        let orig = x.clone();
        rotate_strided(&mut x, stride, theta).unwrap();
        for k in 0..stride {
            let mut set: Vec<f64> = orig[k..].iter().step_by(stride).copied().collect();
            rotate_in_place(&mut set, theta);
            for (m, want) in set.iter().enumerate() {
                let got = x[k + m * stride];
                assert!((got - want).abs() < TOL, "set {k} member {m}");
            }
        }
    }

    #[test]
    fn strided_rotation_leaves_other_sets_untouched() {
        let theta = 1.0;
        let stride = 4;
        // Non-zero only in set S_0; every other set must stay zero.
        let mut x = vec![0.0; 16];
        for m in 0..4 {
            x[m * stride] = 1.0 + m as f64;
        }
        rotate_strided(&mut x, stride, theta).unwrap();
        for (i, v) in x.iter().enumerate() {
            if i % stride != 0 {
                assert_eq!(*v, 0.0, "index {i} leaked across sets");
            }
        }
    }

    #[test]
    fn strided_rotation_preserves_l2_norm() {
        for stride in 1..=6 {
            let mut x = pseudo_vector(17, stride as u64);
            let before = l2(&x);
            rotate_strided(&mut x, stride, 0.7).unwrap();
            assert!((l2(&x) - before).abs() < 1e-9, "stride={stride}");
        }
    }

    #[test]
    fn strided_rotation_large_stride_is_identity() {
        let mut x = pseudo_vector(6, 77);
        let orig = x.clone();
        rotate_strided(&mut x, 6, 1.3).unwrap(); // every set a singleton
        assert_eq!(x, orig);
        assert_eq!(
            rotate_strided(&mut x, 0, 1.3),
            Err(SpreadingError::ZeroStride)
        );
    }

    // ---- composed apply_spreading ----

    #[test]
    fn apply_spreading_spread_zero_is_identity() {
        let mut x = pseudo_vector(24, 3);
        let orig = x.clone();
        apply_spreading(&mut x, 5, 0, 2).unwrap();
        assert_eq!(x, orig);
    }

    #[test]
    fn apply_spreading_single_block_matches_rotate_in_place() {
        let n = 16;
        let k = 4;
        let mut x = pseudo_vector(n, 13);
        let mut expected = x.clone();
        let theta = spread_theta(n, k, 3).unwrap().unwrap();
        rotate_in_place(&mut expected, theta);
        apply_spreading(&mut x, k, 3, 1).unwrap();
        assert_eq!(x, expected);
    }

    #[test]
    fn apply_spreading_small_blocks_skip_pre_rotation() {
        // block_len = 4 < 8: only the per-block theta rotation runs.
        let k = 2;
        let mut x = pseudo_vector(8, 41);
        let mut expected = x.clone();
        let theta = spread_theta(4, k, 2).unwrap().unwrap();
        for block in expected.chunks_exact_mut(4) {
            rotate_in_place(block, theta);
        }
        apply_spreading(&mut x, k, 2, 2).unwrap();
        assert_eq!(x, expected);
    }

    #[test]
    fn apply_spreading_multi_block_runs_pre_rotation() {
        // block_len = 8 ≥ 8: the (pi/2 - theta) strided pre-rotation
        // runs first; the result must differ from the no-pre-rotation
        // composition and still preserve the norm.
        let k = 3;
        let mut x = pseudo_vector(16, 8);
        let before = l2(&x);
        let mut no_pre = x.clone();
        let theta = spread_theta(8, k, 3).unwrap().unwrap();
        for block in no_pre.chunks_exact_mut(8) {
            rotate_in_place(block, theta);
        }
        apply_spreading(&mut x, k, 3, 2).unwrap();
        assert!((l2(&x) - before).abs() < 1e-9);
        assert!(
            x.iter()
                .zip(no_pre.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "pre-rotation had no effect"
        );
        // And it matches the documented composition exactly.
        let mut expected = pseudo_vector(16, 8);
        let stride = spreading_stride(16, 2).unwrap();
        rotate_strided(&mut expected, stride, core::f64::consts::FRAC_PI_2 - theta).unwrap();
        for block in expected.chunks_exact_mut(8) {
            rotate_in_place(block, theta);
        }
        assert_eq!(x, expected);
    }

    #[test]
    fn apply_spreading_zero_vector_stays_zero() {
        // K = 0 ⇒ g_r = 1 ⇒ theta = pi/4, but a zero shape vector is
        // a fixed point of any linear map.
        let mut x = vec![0.0; 12];
        apply_spreading(&mut x, 0, 3, 1).unwrap();
        assert!(x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn apply_spreading_error_paths() {
        let mut x = pseudo_vector(10, 1);
        assert_eq!(
            apply_spreading(&mut x, 1, 4, 1),
            Err(SpreadingError::SpreadOutOfRange { spread: 4 })
        );
        assert_eq!(
            apply_spreading(&mut x, 1, 2, 0),
            Err(SpreadingError::ZeroBlocks)
        );
        assert_eq!(
            apply_spreading(&mut x, 1, 2, 3),
            Err(SpreadingError::BlocksDoNotDivideLength {
                len: 10,
                nb_blocks: 3
            })
        );
        // Width-1 blocks: nothing to rotate, but not an error.
        let mut tiny = pseudo_vector(3, 2);
        let orig = tiny.clone();
        apply_spreading(&mut tiny, 1, 3, 3).unwrap();
        assert_eq!(tiny, orig);
    }

    #[test]
    fn spreading_error_display_is_stable() {
        let cases: [(SpreadingError, &str); 5] = [
            (
                SpreadingError::SpreadOutOfRange { spread: 9 },
                "spread value 9",
            ),
            (SpreadingError::ZeroDimensions, "N >= 1"),
            (SpreadingError::ZeroBlocks, "nb_blocks >= 1"),
            (SpreadingError::ZeroStride, "stride >= 1"),
            (
                SpreadingError::BlocksDoNotDivideLength {
                    len: 10,
                    nb_blocks: 3,
                },
                "length 10",
            ),
        ];
        for (err, needle) in cases {
            let msg = err.to_string();
            assert!(msg.contains(needle), "{msg:?} lacks {needle:?}");
        }
    }
}
