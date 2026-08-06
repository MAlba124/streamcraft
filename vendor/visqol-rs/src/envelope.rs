//! Envelope extraction for the alignment stage: the feature the cross-correlation aligns on.
//!
//! The upper "envelope" here is **not** an analytic-signal envelope, despite being written as one —
//! see [`calculate_upper_env`]. The Hilbert formulation is kept below as the executable reference
//! the fast path is checked against.

use crate::convolution_2d::arena_slice;
#[cfg(test)]
use crate::fast_fourier_transform;
#[cfg(test)]
use crate::fft_manager::FftManager;
#[cfg(test)]
use num::complex::Complex64;
use profluens_core::memory::Arena;

/// Calculates the upper envelope for a given time domain signal.
///
/// **This is a rectification, not an envelope** — `|2·(x − mean) − 1e-6| + mean`, elementwise.
///
/// It used to be computed with two full-length FFTs: build the analytic signal (zero the negative
/// frequencies, double the positive ones), inverse-transform, take `.norm()`. But [`inverse_1d`]
/// keeps only the **real** part of the inverse transform and leaves the imaginary part at zero, so
/// `.norm()` reduces to `|re|` — and the real part of the analytic signal is the input itself. The
/// imaginary part, the Hilbert transform that would make this an envelope, is discarded. (The
/// upstream source says as much: *"This makes very little sense but oh well…"*.)
///
/// So the FFT round trip computed the identity, to rounding. [`calculate_upper_env_via_hilbert`] is
/// the original formulation, kept as the executable reference; `matches_hilbert_reference` asserts
/// the two agree on real audio (they do, to ~1e-16 absolute — pure FFT round-trip noise) and that
/// the lag the cross-correlation then picks is the same. Removing the transforms cut ~10% of a
/// comparison's wall time and 68 MiB of RSS (the 2 M-point rustfft plan's twiddle tables are never
/// built), with the MOS bit-identical across a 6–96 kbps degradation sweep.
///
/// **If the analytic-signal envelope is ever restored** — i.e. if `inverse_1d` is fixed to carry the
/// imaginary part, which is what reference ViSQOL intends — this fast path becomes wrong and the
/// reference implementation below must take over.
///
/// [`inverse_1d`]: crate::fast_fourier_transform::inverse_1d
pub fn calculate_upper_env<'a>(signal: &[f64], out: &'a Arena) -> Option<&'a mut [f64]> {
    if signal.is_empty() {
        return None;
    }
    // `ArrayBase::mean` is `sum() / n` with ndarray's 8-way unrolled fold — keep it exactly (a plain
    // left-to-right sum would round differently). `aview1` is a zero-copy view of the same
    // contiguous samples, so it folds identically.
    let mean = ndarray::aview1(signal).mean()?;
    let hilbert_amplitude = arena_slice::<f64>(out, signal.len());
    for (amplitude, &x) in hilbert_amplitude.iter_mut().zip(signal.iter()) {
        *amplitude = ((x - mean) * 2.0 - 0.000001).abs() + mean;
    }
    Some(hilbert_amplitude)
}

/// The original two-FFT formulation of [`calculate_upper_env`], kept as its executable reference.
///
/// `out` receives the result; `scratch` takes the transform buffers (several times the signal's own
/// size, and dead on return) — two arenas because a bump allocator cannot reclaim its own middle.
#[cfg(test)]
fn calculate_upper_env_via_hilbert<'a>(
    signal: &[f64],
    out: &'a Arena,
    scratch: &Arena,
) -> Option<&'a mut [f64]> {
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
#[cfg(test)]
fn calculate_hilbert<'a>(signal: &[f64], arena: &'a Arena) -> Option<&'a mut [Complex64]> {
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
        let out = Arena::default();
        let (signal, _) = load_audio_files();
        let result = calculate_upper_env(signal.data_matrix.as_slice().unwrap(), &out).unwrap();

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

    /// The fast path must reproduce the Hilbert formulation it replaced — on real audio, and all the
    /// way through to the lag the cross-correlation picks (the only thing the envelope feeds).
    #[test]
    fn matches_hilbert_reference() {
        let Ok(signal) = crate::audio_utils::load_as_mono("../../fixtures/out/audio.wav") else {
            return; // fixture not present (crate built outside the profluens tree)
        };
        let samples = signal.data_matrix.as_slice().unwrap();
        // A shifted copy, so the cross-correlation has a non-trivial lag to find.
        const SHIFT: usize = 137;
        let shifted = &samples[SHIFT..];

        let (out, scratch) = (Arena::default(), Arena::default());
        let fast_ref = calculate_upper_env(samples, &out).unwrap();
        let fast_deg = calculate_upper_env(shifted, &out).unwrap();
        let hilbert_ref = calculate_upper_env_via_hilbert(samples, &out, &scratch).unwrap();
        let scratch2 = Arena::default();
        let hilbert_deg = calculate_upper_env_via_hilbert(shifted, &out, &scratch2).unwrap();

        let worst = fast_ref
            .iter()
            .zip(hilbert_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-14, "envelope differs from the Hilbert reference by {:e}", worst);

        let arena = Arena::default();
        let fast_lag = crate::xcorr::calculate_best_lag(fast_ref, fast_deg, &arena);
        let arena = Arena::default();
        let hilbert_lag = crate::xcorr::calculate_best_lag(hilbert_ref, hilbert_deg, &arena);
        assert_eq!(fast_lag, hilbert_lag, "the alignment lag must be unchanged");
        assert_eq!(fast_lag, Some(SHIFT as i64), "and it must be the shift we introduced");
    }
}
