//! FFT cross-correlation for the alignment stage. As in [`crate::envelope`], every buffer comes
//! from the caller's bump [`Arena`]; the transforms and the arg-max rule are unchanged.

use crate::fast_fourier_transform;
use crate::fft_manager::FftManager;
use num::complex::Complex64;
use profluens_core::memory::Arena;

/// Calculate the maximum delay between to signals.
pub fn calculate_best_lag(signal_1: &[f64], signal_2: &[f64], arena: &Arena) -> Option<i64> {
    let max_lag = ((signal_1.len().max(signal_2.len())) - 1) as i64;

    let point_wise_fft_vec = calculate_inverse_fft_pointwise_product(signal_1, signal_2, arena);

    // The search window is the concatenation of the negative-lag tail
    // (`point_wise[len - max_lag ..]`, `max_lag` values) and the positive-lag head
    // (`point_wise[..= max_lag]`) — indexed here rather than materialised as two `to_vec`s plus an
    // `append`. `corr_at(i)` is exactly `corrs[i]` of the concatenated vector.
    let len = point_wise_fft_vec.len();
    let m = max_lag as usize;
    let corr_at = |i: usize| -> f64 {
        if i < m {
            point_wise_fft_vec[len - m + i]
        } else {
            point_wise_fft_vec[i - m]
        }
    };
    let total = 2 * m + 1;

    // Get maximum. `Iterator::max_by` keeps the *later* element on a tie (it replaces the
    // accumulator unless it compares `Greater`), and the `position` that followed then takes the
    // *first* index holding that value — reproduced exactly.
    let mut best_corr = corr_at(0);
    for i in 1..total {
        let candidate = corr_at(i);
        let ordering = best_corr
            .abs()
            .partial_cmp(&candidate.abs())
            .expect("Failed to compute correlation");
        if ordering != std::cmp::Ordering::Greater {
            best_corr = candidate;
        }
    }

    let best_corr_idx = (0..total).position(|i| corr_at(i) == best_corr)?;

    Some(best_corr_idx as i64 - max_lag)
}

/// Calculates the pointwise inverse fft product of 2 signals
///
/// The signals are borrowed shared: the shorter one used to be `resize`d to the longer one's length
/// with zeros, but both are zero-padded to `fft_points` by the forward transform anyway, so the
/// resize (and the caller's defensive `to_vec` copies) were pure overhead.
pub fn calculate_inverse_fft_pointwise_product<'a>(
    signal_1: &[f64],
    signal_2: &[f64],
    arena: &'a Arena,
) -> &'a mut [f64] {
    let biggest_length = signal_1.len().max(signal_2.len());

    let (_, exp) = frexp((biggest_length * 2 - 1) as f64);
    let fft_points = 2usize.pow(exp as u32);
    let mut manager = FftManager::new(fft_points);
    let point_wise_product =
        calculate_fft_pointwise_product(signal_1, signal_2, &mut manager, fft_points, arena);

    fast_fourier_transform::inverse_1d_conj_sym(&mut manager, point_wise_product, arena)
}

/// Calculates the pointwise fft product of 2 signals
pub fn calculate_fft_pointwise_product<'a>(
    signal_1: &[f64],
    signal_2: &[f64],
    manager: &mut FftManager,
    fft_points: usize,
    arena: &'a Arena,
) -> &'a mut [Complex64] {
    let fft_signal_2 =
        fast_fourier_transform::forward_1d_from_points(manager, signal_2, fft_points, arena);
    fft_signal_2
        .iter_mut()
        .for_each(|element| *element = element.conj());

    let fft_signal_1 =
        fast_fourier_transform::forward_1d_from_points(manager, signal_1, fft_points, arena);
    // Multiply **into** signal 1's spectrum rather than a third buffer: it is dead the moment the
    // product is formed, and at a 4 M-point transform each spectrum is 64 MiB — this is the single
    // largest live allocation in a whole-signal alignment, and it dominates the peak. Same operands
    // in the same order, so the products are bit-identical.
    for (a, b) in fft_signal_1.iter_mut().zip(fft_signal_2.iter()) {
        *a = *a * b;
    }
    fft_signal_1
}

///
/// Returns the mantissa and the exponent of a given floating point value.
pub fn frexp(s: f64) -> (f64, i32) {
    if 0.0 == s {
        (s, 0)
    } else {
        let lg = s.abs().log2();
        let x = (lg - lg.floor() - 1.0).exp2();
        let exp = lg.floor() + 1.0;
        (s.signum() * x, exp as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use profluens_core::memory::Arena;

    #[test]
    fn best_lag_signals_have_equal_length() {
        let ref_signal = vec![
            2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0,
        ];
        let deg_signal_lag2 = vec![
            1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8, 2.4, 2.2, 2.2, 2.1,
        ];

        assert_eq!(deg_signal_lag2.len(), 14);
        let ref_signal_mat = Array1::from_vec(ref_signal);
        let deg_signal_lag2_mat = Array1::from_vec(deg_signal_lag2);
        assert_eq!(ref_signal_mat.len(), deg_signal_lag2_mat.len());
        let best_lag = calculate_best_lag(
            ref_signal_mat.as_slice().unwrap(),
            deg_signal_lag2_mat.as_slice().unwrap(),
            &Arena::default(),
        )
        .unwrap();

        let expected_result = 2;
        assert_eq!(best_lag, expected_result);
    }

    #[test]
    fn best_lag_reference_is_shorter_than_degraded() {
        let ref_signal = vec![
            2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0,
        ];
        let deg_signal_lag2 = vec![
            1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8, 2.4, 2.2, 2.2, 2.1, 2.0,
        ];

        assert!(ref_signal.len() < deg_signal_lag2.len());
        let ref_signal_mat = Array1::from_vec(ref_signal);
        let deg_signal_lag2_mat = Array1::from_vec(deg_signal_lag2);
        let best_lag = calculate_best_lag(
            ref_signal_mat.as_slice().unwrap(),
            deg_signal_lag2_mat.as_slice().unwrap(),
            &Arena::default(),
        )
        .unwrap();

        let expected_result = 2;
        assert_eq!(best_lag, expected_result);
    }

    #[test]
    fn best_lag_reference_is_longer_than_degraded() {
        let ref_signal = vec![
            2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0,
        ];
        let deg_signal_lag2 = vec![
            1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8, 2.4, 2.2, 2.2,
        ];
        assert!(ref_signal.len() > deg_signal_lag2.len());

        let ref_signal_mat = Array1::from_vec(ref_signal);
        let deg_signal_lag2_mat = Array1::from_vec(deg_signal_lag2);
        let best_lag = calculate_best_lag(
            ref_signal_mat.as_slice().unwrap(),
            deg_signal_lag2_mat.as_slice().unwrap(),
            &Arena::default(),
        )
        .unwrap();

        let expected_result = 2;
        assert_eq!(best_lag, expected_result);
    }
    #[test]
    fn best_lag_is_negative() {
        let ref_signal = vec![
            2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0,
        ];
        let deg_signal_lag2 = vec![
            2.0, 2.0, 2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0,
        ];

        let ref_signal_mat = Array1::from_vec(ref_signal);
        let deg_signal_lag2_mat = Array1::from_vec(deg_signal_lag2);
        let best_lag = calculate_best_lag(
            ref_signal_mat.as_slice().unwrap(),
            deg_signal_lag2_mat.as_slice().unwrap(),
            &Arena::default(),
        )
        .unwrap();

        let expected_result = -2;
        assert_eq!(best_lag, expected_result);
    }

    #[test]
    fn test_frexp() {
        let (_, result) = frexp(27.0f64);
        let expected_result = 5;

        assert_eq!(result, expected_result);
    }
}
