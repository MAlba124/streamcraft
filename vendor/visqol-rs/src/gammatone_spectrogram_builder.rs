use crate::analysis_window::AnalysisWindow;
use crate::constants::NUM_BANDS_SPEECH;
use crate::equivalent_rectangular_bandwidth;
use crate::gammatone_filterbank::GammatoneFilterbank;
use crate::spectrogram::Spectrogram;
use crate::spectrogram_builder::SpectrogramBuilder;
use crate::{audio_signal::AudioSignal, visqol_error::VisqolError};
use ndarray::{Array2, Axis};

/// Produces a frequency domain representation from a time domain signal using a gammatone filterbank.
pub struct GammatoneSpectrogramBuilder<const NUM_BANDS: usize> {
    filter_bank: GammatoneFilterbank<NUM_BANDS>,
    /// Sample rate the filterbank coefficients are currently configured for. The gammatone
    /// coefficients depend only on `(sample_rate, NUM_BANDS, min/max freq)`, so they are computed
    /// once and reused — `make_filters` + `set_filter_coefficients` were ~27% of a quality search's
    /// allocations, recomputed on *every* `build` (twice per patch in the fine realignment).
    configured_rate: Option<u32>,
    /// Cached sorted centre frequencies, cloned into each produced `Spectrogram`.
    center_freqs: Vec<f64>,
}

impl<const NUM_BANDS: usize> SpectrogramBuilder for GammatoneSpectrogramBuilder<NUM_BANDS> {
    fn build(
        &mut self,
        signal: &AudioSignal,
        window: &AnalysisWindow,
    ) -> Result<Spectrogram, VisqolError> {
        let time_domain_signal = &signal.data_matrix;
        let sample_rate = signal.sample_rate;
        let max_freq = if NUM_BANDS == NUM_BANDS_SPEECH {
            Self::SPEECH_MODE_MAX_FREQ
        } else {
            sample_rate / 2
        };

        // Gammatone coefficients depend only on the sample rate (plus the const band count and
        // min/max freq), so compute + install them once and reuse across every build; only
        // `reset_filter_conditions` (cheap in-place zeroing) has to run each time.
        if self.configured_rate != Some(sample_rate) {
            let (mut filter_coeffs, mut center_freqs) =
                equivalent_rectangular_bandwidth::make_filters::<NUM_BANDS>(
                    sample_rate as usize,
                    self.filter_bank.min_freq,
                    max_freq as f64,
                );
            filter_coeffs.invert_axis(Axis(0));
            self.filter_bank.set_filter_coefficients(&filter_coeffs);
            center_freqs.as_mut_slice().sort_by(|a, b| {
                a.partial_cmp(b).expect("Failed to sort center frequencies!")
            });
            self.center_freqs = center_freqs;
            self.configured_rate = Some(sample_rate);
        }
        self.filter_bank.reset_filter_conditions();

        let hop_size = (window.size as f64 * window.overlap) as usize;

        if time_domain_signal.len() < window.size {
            return Err(VisqolError::TooFewSamples {
                found: time_domain_signal.len(),
                minimum_required: window.size,
            });
        }

        let num_cols = 1 + ((time_domain_signal.len() - window.size) / hop_size);
        let mut out_matrix = Array2::<f64>::zeros((NUM_BANDS, num_cols));
        let window_size_f = window.size as f64;
        // Per-band filter energy for one frame, filled by the fused (vectorised, AVX2-across-bands)
        // gammatone kernel — no per-frame filtered `Array2`, no per-frame reset.
        let mut energy = [0.0f64; NUM_BANDS];

        for (index, frame) in time_domain_signal
            .windows(window.size)
            .into_iter()
            .step_by(hop_size)
            .enumerate()
        {
            self.filter_bank.filter_frame_energy_into(
                frame
                    .as_slice()
                    .expect("Failed to convert audio frame to slice"),
                &mut energy,
            );

            // RMS straight into the output column: `sqrt(Σ y4² / window)`. Bit-identical to
            // apply_filter → square → mean_axis → sqrt (same left-to-right Σ, same divide, sqrt).
            for band in 0..NUM_BANDS {
                out_matrix[(band, index)] = (energy[band] / window_size_f).sqrt();
            }
        }

        Ok(Spectrogram::new(out_matrix, self.center_freqs.clone()))
    }
}

impl<const NUM_BANDS: usize> GammatoneSpectrogramBuilder<NUM_BANDS> {
    const SPEECH_MODE_MAX_FREQ: u32 = 8000;

    /// Creates a new gammatone spectrogram builder with the given gammatone filterbank.
    /// If `use_speech_mode` is set to `true`, the maximum frequency is determined to be 8000 Hz.
    pub fn new(filter_bank: GammatoneFilterbank<NUM_BANDS>) -> Self {
        Self {
            filter_bank,
            configured_rate: None,
            center_freqs: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis_window::AnalysisWindow;
    use crate::audio_utils;
    use crate::gammatone_filterbank::GammatoneFilterbank;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_spec_builder() {
        // Fixed parameters
        const MINIMUM_FREQ: f64 = 50.0;
        const NUM_BANDS: usize = 32;
        const OVERLAP: f64 = 0.25;

        const REF_SPECTRO_NUM_COLS: usize = 802;

        let signal_ref = audio_utils::load_as_mono(
            "test_data/conformance_testdata_subset/contrabassoon48_stereo.wav",
        )
        .unwrap();
        let filter_bank = GammatoneFilterbank::<{ NUM_BANDS }>::new(MINIMUM_FREQ);
        let window = AnalysisWindow::new(signal_ref.sample_rate, OVERLAP, 0.08);

        let mut spectro_builder: GammatoneSpectrogramBuilder<NUM_BANDS> =
            GammatoneSpectrogramBuilder::new(filter_bank);
        let spectrogram_ref = spectro_builder.build(&signal_ref, &window).unwrap();

        // Check 1st element
        assert_abs_diff_eq!(spectrogram_ref.data[(0, 0)], 9.44161e-05, epsilon = 0.00001);
        // Check dimensions
        assert_eq!(spectrogram_ref.data.ncols(), REF_SPECTRO_NUM_COLS);
    }
}
