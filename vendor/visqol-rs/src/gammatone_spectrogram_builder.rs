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
        let mut out = Spectrogram::empty();
        self.build_into(
            signal.data_matrix.as_slice().expect("audio signal is contiguous"),
            signal.sample_rate,
            window,
            &mut out,
        )?;
        Ok(out)
    }
}

impl<const NUM_BANDS: usize> GammatoneSpectrogramBuilder<NUM_BANDS> {
    /// Build the spectrogram of `samples` **into a caller-owned [`Spectrogram`]**, reusing its
    /// backing buffers.
    ///
    /// The fine-realignment loop builds two spectrograms per patch, so the owned `Array2::zeros` and
    /// the `center_freqs.clone()` of the allocating [`SpectrogramBuilder::build`] were ~300
    /// allocations per comparison. Consecutive patches are near-identical in size, so the reused
    /// `Vec` only ever grows on the first few — allocation-free in steady state, and the values
    /// written are exactly the same.
    pub fn build_into(
        &mut self,
        samples: &[f64],
        sample_rate: u32,
        window: &AnalysisWindow,
        out: &mut Spectrogram,
    ) -> Result<(), VisqolError> {
        let time_domain_signal = samples;
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
        // Reclaim the destination's backing `Vec` and resize it in place — a shrink keeps the
        // capacity, so successive patches of similar length reuse the same allocation. Every
        // element is written below, so the fill value is irrelevant.
        let mut buffer = std::mem::take(&mut out.data).into_raw_vec_and_offset().0;
        buffer.clear();
        buffer.resize(NUM_BANDS * num_cols, 0.0);
        let mut out_matrix = Array2::<f64>::from_shape_vec((NUM_BANDS, num_cols), buffer)
            .expect("spectrogram buffer length matches NUM_BANDS * num_cols");
        let window_size_f = window.size as f64;
        // Per-band filter energy for one frame, filled by the fused (vectorised, AVX2-across-bands)
        // gammatone kernel — no per-frame filtered `Array2`, no per-frame reset.
        let mut energy = [0.0f64; NUM_BANDS];

        for (index, frame) in time_domain_signal.windows(window.size).step_by(hop_size).enumerate() {
            self.filter_bank.filter_frame_energy_into(frame, &mut energy);

            // RMS straight into the output column: `sqrt(Σ y4² / window)`. Bit-identical to
            // apply_filter → square → mean_axis → sqrt (same left-to-right Σ, same divide, sqrt).
            for band in 0..NUM_BANDS {
                out_matrix[(band, index)] = (energy[band] / window_size_f).sqrt();
            }
        }

        out.data = out_matrix;
        // Reuse the destination's `Vec` capacity rather than cloning a fresh one per build.
        out.center_freq_bands.clear();
        out.center_freq_bands.extend_from_slice(&self.center_freqs);
        Ok(())
    }

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
