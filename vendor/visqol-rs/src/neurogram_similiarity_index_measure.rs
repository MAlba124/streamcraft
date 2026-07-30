use crate::convolution_2d::{binop_into, map_into, perform_valid_2d_conv_with_boundary};
use crate::patch_similarity_comparator::{PatchSimilarityComparator, PatchSimilarityResult};
use ndarray::{arr2, Array1, Axis};
use profluens_core::memory::Arena;

/// Provides a neurogram similarity index measure (NSIM) implementation for a
/// patch similarity comparator. NSIM is a distance metric, adapted from the
/// image processing technique called structural similarity (SSIM) and is here
/// used to compare two patches taken from the reference and degraded
/// spectrograms.
pub struct NeurogramSimiliarityIndexMeasure {
    intensity_range: f64,
}

#[allow(unused)]
impl NeurogramSimiliarityIndexMeasure {
    pub fn new(intensity_range: f64) -> Self { Self { intensity_range } }
}

impl Default for NeurogramSimiliarityIndexMeasure {
    fn default() -> Self {
        Self {
            intensity_range: 1.0,
        }
    }
}

impl PatchSimilarityComparator for NeurogramSimiliarityIndexMeasure {
    /// Computes the NSIM between `ref_patch` and `deg_patch` and returns the mean and standard deviation of each frequency band, the energy of the degraded patch and the similarity score.
    ///
    /// Every intermediate matrix is carved from the caller-owned bump `arena` (see [`arena_mat`]
    /// et al.) rather than the heap: the per-op `&a * &b` result arrays of the original — the top
    /// allocator after the flatten fix — become writes into reused arena memory. The caller
    /// [`Arena::reset`]s between patches, so this loop is allocation-free in steady state, and the
    /// per-element ops are unchanged so the score is bit-identical.
    ///
    /// [`arena_mat`]: crate::convolution_2d::arena_mat
    fn measure_patch_similarity(
        &self,
        ref_patch: &mut ndarray::Array2<f64>,
        deg_patch: &mut ndarray::Array2<f64>,
        arena: &Arena,
    ) -> PatchSimilarityResult {
        let window = arr2(&[
            [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
            [0.0838251475442633, 0.619485845753726, 0.0838251475442633],
            [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
        ]);

        let k = [0.01, 0.03];
        let c1 = (k[0] * self.intensity_range).powf(2.0);
        let c3 = (k[1] * self.intensity_range).powf(2.0) / 2.0;

        // Compute mu
        let mu_ref = perform_valid_2d_conv_with_boundary(&window, &*ref_patch, arena);
        let mu_deg = perform_valid_2d_conv_with_boundary(&window, &*deg_patch, arena);

        let ref_mu_squared = binop_into(arena, &mu_ref, &mu_ref, |a, b| a * b);
        let deg_mu_squared = binop_into(arena, &mu_deg, &mu_deg, |a, b| a * b);
        let mu_r_mu_d = binop_into(arena, &mu_ref, &mu_deg, |a, b| a * b);

        let ref_neuro_sq = binop_into(arena, &*ref_patch, &*ref_patch, |a, b| a * b);
        let deg_neuro_sq = binop_into(arena, &*deg_patch, &*deg_patch, |a, b| a * b);

        // Compute sigmas
        let conv2_ref_neuro_squared =
            perform_valid_2d_conv_with_boundary(&window, &ref_neuro_sq, arena);
        let sigma_ref_squared =
            binop_into(arena, &conv2_ref_neuro_squared, &ref_mu_squared, |a, b| a - b);

        let conv2_deg_neuro_squared =
            perform_valid_2d_conv_with_boundary(&window, &deg_neuro_sq, arena);
        let sigma_deg_squared =
            binop_into(arena, &conv2_deg_neuro_squared, &deg_mu_squared, |a, b| a - b);

        let ref_neuro_deg = binop_into(arena, &*ref_patch, &*deg_patch, |a, b| a * b);
        let conv2_ref_neuro_deg =
            perform_valid_2d_conv_with_boundary(&window, &ref_neuro_deg, arena);

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
        let sim_map = binop_into(arena, &intensity, &structure, |a, b| a * b);

        let freq_band_deg_energy: Array1<f64> = deg_patch
            .mean_axis(Axis(1))
            .expect("Failed to compute mean for degraded signal!");
        let freq_band_means: Array1<f64> = sim_map
            .mean_axis(Axis(1))
            .expect("Failed to compute mean for similarity map!");
        let freq_band_std: Array1<f64> = sim_map.std_axis(Axis(1), 1.0);
        let mean_freq_band_means = freq_band_means
            .mean()
            .expect("Failed to compute mean of means for degraded signal!");

        PatchSimilarityResult::new(
            freq_band_means.to_vec(),
            freq_band_std.to_vec(),
            freq_band_deg_energy.to_vec(),
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
        let mut ref_patch_mat = Array2::from_shape_vec((3, 1), ref_patch).unwrap();
        let deg_patch = vec![0.0, 0.0, 0.0];
        let mut deg_patch_mat = Array2::from_shape_vec((3, 1), deg_patch).unwrap();
        let expected_result = [0.000125225, 0.00875062, 1.0];

        let sim_comparator = NeurogramSimiliarityIndexMeasure::default();

        let result = sim_comparator.measure_patch_similarity(
            &mut ref_patch_mat,
            &mut deg_patch_mat,
            &arena,
        );

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
}
