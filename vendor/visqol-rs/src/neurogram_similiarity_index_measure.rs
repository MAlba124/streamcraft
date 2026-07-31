use crate::convolution_2d::{binop_into, map_into, perform_valid_2d_conv_with_boundary};
use crate::patch_similarity_comparator::{
    BandValues, PatchSimilarityComparator, PatchSimilarityResult,
};
use ndarray::{arr2, Array2, ArrayBase, ArrayViewMut2, Data, Ix2};
use profluens_core::memory::Arena;

/// Provides a neurogram similarity index measure (NSIM) implementation for a
/// patch similarity comparator. NSIM is a distance metric, adapted from the
/// image processing technique called structural similarity (SSIM) and is here
/// used to compare two patches taken from the reference and degraded
/// spectrograms.
pub struct NeurogramSimiliarityIndexMeasure {
    intensity_range: f64,
    /// The fixed 3×3 NSIM smoothing window (a Gaussian-ish kernel), built once. It was previously
    /// `arr2`-allocated on *every* `compute_sim_map` call — in the hot alignment slide loop that was
    /// the single largest remaining allocator (~1.7M heap `Array2`s per song).
    window: Array2<f64>,
}

/// `map.std_axis(Axis(1), 1.0)` — the per-band standard deviation — without the two heap `Array1`s
/// ndarray allocates for it (one for the variance, one for the `sqrt`ed copy).
///
/// This reproduces ndarray 0.16's [Welford one-pass recurrence](https://www.jstor.org/stable/1266577)
/// *exactly*, element for element: `delta = x - mean`, `mean += delta / (i+1)`,
/// `sum_sq = (x - mean).mul_add(delta, sum_sq)` (an FMA, as there), then `sqrt(sum_sq / (n - ddof))`.
/// Each band's recurrence is independent and runs over ascending columns in both forms, so the
/// result is bit-for-bit what `std_axis` returns.
fn band_std_dev<S: Data<Elem = f64>>(map: &ArrayBase<S, Ix2>) -> BandValues {
    let (nrows, ncols) = map.dim();
    let dof = ncols as f64 - 1.0; // ddof = 1.0
    BandValues::from_fn(nrows, |band| {
        let (mut mean, mut sum_sq) = (0.0f64, 0.0f64);
        for (index, &x) in map.row(band).iter().enumerate() {
            let count = (index + 1) as f64;
            let delta = x - mean;
            mean += delta / count;
            sum_sq = (x - mean).mul_add(delta, sum_sq);
        }
        (sum_sq / dof).sqrt()
    })
}

/// The fixed NSIM 3×3 smoothing window.
fn nsim_window() -> Array2<f64> {
    arr2(&[
        [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
        [0.0838251475442633, 0.619485845753726, 0.0838251475442633],
        [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
    ])
}

#[allow(unused)]
impl NeurogramSimiliarityIndexMeasure {
    pub fn new(intensity_range: f64) -> Self {
        Self { intensity_range, window: nsim_window() }
    }

    /// Shared NSIM core: compute the per-element similarity map into the `arena`. Both entry points
    /// need this — only the final per-band reduction differs. Every intermediate matrix is carved
    /// from the caller-owned bump arena (see [`arena_mat`] et al.) rather than the heap: the per-op
    /// `&a * &b` result arrays become writes into reused arena memory, so this is allocation-free.
    /// The per-element ops and their order are unchanged, so the map is bit-identical.
    ///
    /// [`arena_mat`]: crate::convolution_2d::arena_mat
    fn compute_sim_map<'a, Sr, Sd>(
        &self,
        ref_patch: &ArrayBase<Sr, Ix2>,
        deg_patch: &ArrayBase<Sd, Ix2>,
        arena: &'a Arena,
    ) -> ArrayViewMut2<'a, f64>
    where
        Sr: Data<Elem = f64>,
        Sd: Data<Elem = f64>,
    {
        let window = &self.window;

        let k = [0.01, 0.03];
        let c1 = (k[0] * self.intensity_range).powf(2.0);
        let c3 = (k[1] * self.intensity_range).powf(2.0) / 2.0;

        // Compute mu
        let mu_ref = perform_valid_2d_conv_with_boundary(window, ref_patch, arena);
        let mu_deg = perform_valid_2d_conv_with_boundary(window, deg_patch, arena);

        let ref_mu_squared = binop_into(arena, &mu_ref, &mu_ref, |a, b| a * b);
        let deg_mu_squared = binop_into(arena, &mu_deg, &mu_deg, |a, b| a * b);
        let mu_r_mu_d = binop_into(arena, &mu_ref, &mu_deg, |a, b| a * b);

        let ref_neuro_sq = binop_into(arena, ref_patch, ref_patch, |a, b| a * b);
        let deg_neuro_sq = binop_into(arena, deg_patch, deg_patch, |a, b| a * b);

        // Compute sigmas
        let conv2_ref_neuro_squared =
            perform_valid_2d_conv_with_boundary(window, &ref_neuro_sq, arena);
        let sigma_ref_squared =
            binop_into(arena, &conv2_ref_neuro_squared, &ref_mu_squared, |a, b| a - b);

        let conv2_deg_neuro_squared =
            perform_valid_2d_conv_with_boundary(window, &deg_neuro_sq, arena);
        let sigma_deg_squared =
            binop_into(arena, &conv2_deg_neuro_squared, &deg_mu_squared, |a, b| a - b);

        let ref_neuro_deg = binop_into(arena, ref_patch, deg_patch, |a, b| a * b);
        let conv2_ref_neuro_deg =
            perform_valid_2d_conv_with_boundary(window, &ref_neuro_deg, arena);

        let sigma_r_d = binop_into(arena, &conv2_ref_neuro_deg, &mu_r_mu_d, |a, b| a - b);

        // Compute intensity: `&mu_r_mu_d * 2.0 + c1` and `&ref_mu_squared + &deg_mu_squared + c1`
        let intensity_numerator = map_into(arena, &mu_r_mu_d, |x| x * 2.0 + c1);
        let intensity_denominator =
            binop_into(arena, &ref_mu_squared, &deg_mu_squared, |a, b| a + b + c1);

        let intensity =
            binop_into(arena, &intensity_numerator, &intensity_denominator, |a, b| a / b);

        // Compute structure
        let structure_numerator = map_into(arena, &sigma_r_d, |x| x + c3);
        let mut structure_denominator =
            binop_into(arena, &sigma_ref_squared, &sigma_deg_squared, |a, b| a * b);

        // Avoid nans
        structure_denominator.map_inplace(|element| {
            *element = if *element < 0.0 {
                c3
            } else {
                element.sqrt() + c3
            }
        });

        let structure =
            binop_into(arena, &structure_numerator, &structure_denominator, |a, b| a / b);
        binop_into(arena, &intensity, &structure, |a, b| a * b)
    }

    /// Just the scalar similarity — the only field the O(patches × window) alignment slide loop
    /// consumes. Allocation-free: the `sim_map` scratch is arena, and the per-band `Array1`s + three
    /// result `Vec`s of the full measure are skipped entirely. Generic over both patches' storage so
    /// the slide loop can pass a zero-copy `ArrayView2` slice of the degraded spectrogram rather than
    /// an owned patch copy. The score is the grand mean of `sim_map` computed as
    /// `mean_axis(Axis(1)).mean()` — per-band (row) means then the mean of those — reproduced with
    /// the identical left-to-right sums and identical two divisions, so it is bit-identical to
    /// `measure_patch_similarity(..).similarity` (asserted in the tests).
    pub fn measure_similarity_score<Sr, Sd>(
        &self,
        ref_patch: &ArrayBase<Sr, Ix2>,
        deg_patch: &ArrayBase<Sd, Ix2>,
        arena: &Arena,
    ) -> f64
    where
        Sr: Data<Elem = f64>,
        Sd: Data<Elem = f64>,
    {
        let sim_map = self.compute_sim_map(ref_patch, deg_patch, arena);
        let (nrows, ncols) = sim_map.dim();
        let ncols_f = ncols as f64;
        // `sim_map.row(i).sum()` sums the contiguous row left-to-right, exactly as
        // `sum_axis(Axis(1))[i]` does; dividing by `ncols` matches `mean_axis`. Accumulating those
        // row means in order and dividing by `nrows` matches the subsequent `Array1::mean`.
        let mut acc = 0.0;
        for i in 0..nrows {
            acc += sim_map.row(i).sum() / ncols_f;
        }
        acc / nrows as f64
    }
}

impl Default for NeurogramSimiliarityIndexMeasure {
    fn default() -> Self {
        Self {
            intensity_range: 1.0,
            window: nsim_window(),
        }
    }
}

impl PatchSimilarityComparator for NeurogramSimiliarityIndexMeasure {
    /// Full NSIM: the mean and standard deviation of each frequency band, the energy of the
    /// degraded patch, and the scalar similarity. Called O(patches) times (the backtrace + fine
    /// realignment), so its three per-band result `Vec`s are a negligible allocation; the hot
    /// alignment slide loop uses [`measure_similarity_score`](Self::measure_similarity_score)
    /// instead, which needs none of them.
    fn measure_patch_similarity<Sr, Sd>(
        &self,
        ref_patch: &ArrayBase<Sr, Ix2>,
        deg_patch: &ArrayBase<Sd, Ix2>,
        arena: &Arena,
    ) -> PatchSimilarityResult
    where
        Sr: Data<Elem = f64>,
        Sd: Data<Elem = f64>,
    {
        let sim_map = self.compute_sim_map(ref_patch, deg_patch, arena);

        // Per-band means straight into the inline result storage: same row-sum / ncols as
        // `mean_axis(Axis(1)).to_vec()`, but without the intermediate owned `Array1` *or* the
        // result `Vec`. This full measure runs ~twice per patch (backtrace + fine realignment), so
        // these were the largest remaining allocator in a comparison.
        let dncols = deg_patch.ncols() as f64;
        let freq_band_deg_energy =
            BandValues::from_fn(deg_patch.nrows(), |i| deg_patch.row(i).sum() / dncols);
        let sncols = sim_map.ncols() as f64;
        let freq_band_means =
            BandValues::from_fn(sim_map.nrows(), |i| sim_map.row(i).sum() / sncols);
        let freq_band_std = band_std_dev(&sim_map);
        // `Array1::mean()` is `sum() / n`; `iter().sum()` folds the same row means in the same order.
        let mean_freq_band_means =
            freq_band_means.iter().sum::<f64>() / freq_band_means.len() as f64;

        PatchSimilarityResult::new(
            freq_band_means,
            freq_band_std,
            freq_band_deg_energy,
            mean_freq_band_means,
        )
    }
}

#[cfg(test)]
mod tests {

    use approx::assert_abs_diff_eq;
    use ndarray::Array2;
    use profluens_core::memory::Arena;

    use super::*;

    #[test]
    fn test_neurogram_measure() {
        let arena = Arena::default();
        let ref_patch = vec![1.0, 0.0, 0.0];
        let ref_patch_mat = Array2::from_shape_vec((3, 1), ref_patch).unwrap();
        let deg_patch = vec![0.0, 0.0, 0.0];
        let deg_patch_mat = Array2::from_shape_vec((3, 1), deg_patch).unwrap();
        let expected_result = [0.000125225, 0.00875062, 1.0];

        let sim_comparator = NeurogramSimiliarityIndexMeasure::default();

        let result =
            sim_comparator.measure_patch_similarity(&ref_patch_mat, &deg_patch_mat, &arena);

        assert_abs_diff_eq!(
            result.freq_band_means[0],
            expected_result[0],
            epsilon = 0.0001
        );
        assert_abs_diff_eq!(
            result.freq_band_means[1],
            expected_result[1],
            epsilon = 0.0001
        );
        assert_abs_diff_eq!(
            result.freq_band_means[2],
            expected_result[2],
            epsilon = 0.0001
        );
    }

    /// The zero-alloc scalar path must be *bit-exactly* the full measure's `.similarity`, since the
    /// alignment DP compares these scores against each other and a stored cumulative sum.
    #[test]
    fn score_matches_full_similarity_bit_exact() {
        let arena = Arena::default();
        let sim = NeurogramSimiliarityIndexMeasure::default();
        // A non-trivial multi-band, multi-frame pair (so per-band means genuinely differ).
        let r = Array2::from_shape_vec(
            (4, 5),
            vec![
                0.9, 0.2, 0.5, 0.7, 0.1, 0.4, 0.8, 0.3, 0.6, 0.05, 0.55, 0.15, 0.95, 0.35, 0.75,
                0.25, 0.65, 0.45, 0.85, 0.02,
            ],
        )
        .unwrap();
        let d = Array2::from_shape_vec(
            (4, 5),
            vec![
                0.8, 0.25, 0.55, 0.6, 0.15, 0.35, 0.7, 0.4, 0.5, 0.1, 0.6, 0.2, 0.9, 0.3, 0.8, 0.3,
                0.6, 0.5, 0.8, 0.05,
            ],
        )
        .unwrap();

        let full = sim.measure_patch_similarity(&r, &d, &arena);
        let score = sim.measure_similarity_score(&r, &d, &arena);
        // ...and a zero-copy view of the degraded patch must give the identical score (this is the
        // path the slide loop takes).
        let score_view = sim.measure_similarity_score(&r, &d.slice(ndarray::s![.., ..]), &arena);

        assert_eq!(score, full.similarity, "scalar path must equal full .similarity bit-for-bit");
        assert_eq!(score, score_view, "view and owned degraded patch must score identically");
    }
}
