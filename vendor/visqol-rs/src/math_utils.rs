use ndarray::Array1;

/// Maximum of an `f64` iterator. Replaces `ndarray_stats::QuantileExt::max` so the crate can
/// drop `ndarray-stats` *and* `ndarray-linalg` (the latter was a declared-but-unused dep that
/// drags in a BLAS/LAPACK backend visqol never calls). `None` on an empty iterator — matching
/// the `Result::Err` that every call site already `.expect`s. Inputs here are finite dB
/// magnitudes (no NaN), so `f64::max`/`f64::min` are exact.
pub fn max_of<'a>(iter: impl Iterator<Item = &'a f64>) -> Option<f64> {
    iter.copied().reduce(f64::max)
}

/// Minimum of an `f64` iterator — see [`max_of`].
pub fn min_of<'a>(iter: impl Iterator<Item = &'a f64>) -> Option<f64> {
    iter.copied().reduce(f64::min)
}

pub fn normalize_signal(signal: &Array1<f64>) -> Array1<f64> {
    let normalized_mat = signal.clone();
    let max = get_max(signal);
    normalized_mat / max
}

pub fn next_pow_two(input: usize) -> usize {
    let mut next_power_of_two = input - 1;

    next_power_of_two |= next_power_of_two >> 1;
    next_power_of_two |= next_power_of_two >> 2;
    next_power_of_two |= next_power_of_two >> 4;
    next_power_of_two |= next_power_of_two >> 1;
    next_power_of_two |= next_power_of_two >> 16;
    next_power_of_two + 1
}

/// Returns the exponential fit between 2 points
pub fn exponential_from_fit(x: f64, a: f64, b: f64, x_0: f64) -> f64 { a + (b * (x - x_0)).exp() }

/// Normalizes a slice of `i16` to a vector of `f64` values
pub fn normalize_int16_to_double(input: &[i16]) -> Vec<f64> {
    input
        .iter()
        .map(|x| *x as f64 / 32767.0f64)
        .collect::<Vec<f64>>()
}

/// Returns the maximum of an `ndarray::Array1<f64>`
fn get_max(mat: &Array1<f64>) -> f64 {
    max_of(mat.iter()).expect("Failed to compute maximum of matrix!")
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    #[test]
    fn test_next_pow_two() {
        let inputs = [2, 10, 3, 5, 48000, 7, 23, 32];
        let expected = vec![2, 16, 4, 8, 65536, 8, 32, 32];

        let mut results = Vec::new();
        for i in inputs.iter() {
            results.push(next_pow_two(*i));
        }
        assert_eq!(results, expected);
    }

    #[test]
    fn test_exponential_from_fit() {
        assert_abs_diff_eq!(
            1.446_176_4,
            exponential_from_fit(0.5, 1.15, 4.68, 0.76),
            epsilon = 0.0001
        );
        assert_abs_diff_eq!(
            4.224_677_6,
            exponential_from_fit(1.0, 1.15, 4.68, 0.76),
            epsilon = 0.0001
        );
    }
}
