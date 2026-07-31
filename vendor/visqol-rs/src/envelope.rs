//! Envelope extraction for the alignment stage. Every intermediate — the centred signal, the
//! frequency-domain buffers, the Hilbert scaling vector and the amplitude output — is carved from a
//! bump [`Arena`] instead of the heap; the arithmetic is unchanged.

use crate::convolution_2d::arena_slice;
use crate::fast_fourier_transform;
use crate::fft_manager::FftManager;
use num::complex::Complex64;
use profluens_core::memory::Arena;

/// Calculates the upper envelope for a given time domain signal.
///
/// Two arenas, because a bump allocator cannot reclaim the middle of itself: the envelope goes into
/// `out` (the caller needs it after this returns), while the transform buffers — several times the
/// signal's own size, and dead the moment this returns — go into `scratch`, which the caller resets
/// between calls. Sharing one arena would pile up ~300 MiB of dead FFT scratch across a whole-signal
/// alignment.
pub fn calculate_upper_env<'a>(
    signal: &[f64],
    out: &'a Arena,
    scratch: &Arena,
) -> Option<&'a mut [f64]> {
    // `ArrayBase::mean` is `sum() / n` with ndarray's 8-way unrolled fold — keep it exactly (a plain
    // left-to-right sum would round differently). `aview1` is a zero-copy view of the same
    // contiguous samples, so it folds identically.
    let mean = ndarray::aview1(signal).mean()?;
    let signal_centered = arena_slice::<f64>(scratch, signal.len());
    for (centered, &x) in signal_centered.iter_mut().zip(signal.iter()) {
        *centered = x - mean;
    }
    let hilbert = calculate_hilbert(signal_centered, scratch)?;

    let hilbert_amplitude = arena_slice::<f64>(out, hilbert.len());

    for (amplitude, h) in hilbert_amplitude.iter_mut().zip(hilbert.iter()) {
        *amplitude = h.norm() + mean;
    }
    Some(hilbert_amplitude)
}

/// Calculates the hilbert transform for a given time domain signal.
pub fn calculate_hilbert<'a>(signal: &[f64], arena: &'a Arena) -> Option<&'a mut [Complex64]> {
    let mut fft_manager = FftManager::new(signal.len());
    let freq_domain_signal =
        fast_fourier_transform::forward_1d_from_matrix(&mut fft_manager, signal, arena);

    let is_odd = signal.len() % 2 == 1;
    let is_non_empty = !signal.is_empty();

    // Set up scaling vector
    let hilbert_scaling = arena_slice::<f64>(arena, freq_domain_signal.len());
    hilbert_scaling.fill(0.0); // arena memory is uninitialised; the original was `vec![0.0; n]`
    hilbert_scaling[0] = 1.0;

    if !is_odd && is_non_empty {
        hilbert_scaling[signal.len() / 2] = 1.0;
    } else if is_odd && is_non_empty {
        hilbert_scaling[signal.len() / 2] = 2.0;
    }

    let n = if is_odd {
        freq_domain_signal.len().div_ceil(2)
    } else {
        freq_domain_signal.len() / 2
    };

    hilbert_scaling[1..n].fill(2.0);

    let element_wise_product = arena_slice::<Complex64>(arena, freq_domain_signal.len());

    for i in 0..freq_domain_signal.len() {
        element_wise_product[i] = freq_domain_signal[i] * hilbert_scaling[i];
    }

    let hilbert = fast_fourier_transform::inverse_1d(&mut fft_manager, element_wise_product, arena);
    hilbert
        .iter_mut()
        .for_each(|element| *element = *element * 2.0 - 0.000001);
    Some(hilbert)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        audio_signal::AudioSignal,
        audio_utils::load_as_mono,
        fft_manager,
        xcorr::{
            calculate_best_lag, calculate_fft_pointwise_product,
            calculate_inverse_fft_pointwise_product, frexp,
        },
    };
    use approx::assert_abs_diff_eq;
    use profluens_core::memory::Arena;

    #[test]
    fn hilbert_transform_on_audio_signal() {
        let arena = Arena::default();
        let (signal, _) = load_audio_files();
        let result = calculate_hilbert(signal.data_matrix.as_slice().unwrap(), &arena).unwrap();

        assert_abs_diff_eq!(result[0].re, 0.000_303_661_691_188_833, epsilon = 0.0001);
    }

    #[test]
    fn envelope_on_audio_signal() {
        let (out, scratch) = (Arena::default(), Arena::default());
        let (signal, _) = load_audio_files();
        let result =
            calculate_upper_env(signal.data_matrix.as_slice().unwrap(), &out, &scratch).unwrap();

        assert_abs_diff_eq!(result[0], 0.00030159861338215923, epsilon = 0.0001);
    }

    #[test]
    fn xcorr_pointwise_prod_on_audio_signal() {
        let (ref_signal, deg_signal) = load_audio_files();
        let ref_signal_vec = ref_signal.data_matrix.to_vec();

        let (_, exponent) = frexp((ref_signal_vec.len() * 2 - 1) as f64);
        let fft_points = 2i32.pow(exponent as u32) as usize;
        let mut manager = fft_manager::FftManager::new(fft_points);

        let arena = Arena::default();
        let result = calculate_fft_pointwise_product(
            ref_signal.data_matrix.as_slice().unwrap(),
            deg_signal.data_matrix.as_slice().unwrap(),
            &mut manager,
            fft_points,
            &arena,
        );

        assert_abs_diff_eq!(result[0].re, 0.012231532484292984, epsilon = 0.001);
    }

    #[test]
    fn calculate_inverse_fft_pointwise_product_on_audio_pair() {
        let (ref_signal, deg_signal) = load_audio_files();

        let arena = Arena::default();
        let result = calculate_inverse_fft_pointwise_product(
            ref_signal.data_matrix.as_slice().unwrap(),
            deg_signal.data_matrix.as_slice().unwrap(),
            &arena,
        );

        assert_abs_diff_eq!(result[0], 79.66060597338944, epsilon = 0.0001);
    }

    #[test]
    fn calculate_best_lag_on_audio_signal() {
        let (ref_signal, deg_signal) = load_audio_files();

        let arena = Arena::default();
        let result = calculate_best_lag(
            ref_signal.data_matrix.as_slice().unwrap(),
            deg_signal.data_matrix.as_slice().unwrap(),
            &arena,
        )
        .unwrap();

        assert_abs_diff_eq!(result, 0);
    }

    fn load_audio_files() -> (AudioSignal, AudioSignal) {
        let ref_signal_path = "test_data/clean_speech/CA01_01.wav";
        let deg_signal_path = "test_data/clean_speech/transcoded_CA01_01.wav";
        (
            load_as_mono(ref_signal_path).unwrap(),
            load_as_mono(deg_signal_path).unwrap(),
        )
    }
}
