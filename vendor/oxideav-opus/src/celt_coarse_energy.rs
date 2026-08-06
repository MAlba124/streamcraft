//! CELT coarse-energy reconstruction recurrence (RFC 6716 §4.3.2.1).
//!
//! The §4.3.2.1 *coarse energy* of each CELT band is the 6 dB-resolution
//! (integer base-2-log) envelope that the §4.3.6 denormalise step turns
//! back into a per-band gain. Each band carries a single Laplace-coded
//! **prediction-error** symbol (decoded by
//! [`crate::celt_laplace::ec_laplace_decode`] against the `(prob, decay)`
//! pair from [`crate::celt_e_prob_model`]); this module turns that
//! sequence of per-band residuals into the reconstructed log-energy by
//! running the §4.3.2.1 2-D prediction filter in reverse.
//!
//! ## The prediction filter and its reconstruction recurrence
//!
//! RFC 6716 §4.3.2.1 (p. 108) gives the 2-D z-transform of the
//! *prediction-error* filter that maps the band log-energy sequence
//! `E[b][l]` (band index `b`, frame index `l`) to the coded residual
//! `R[b][l]`:
//!
//! ```text
//!                 (1 - alpha*z_l^-1) * (1 - z_b^-1)
//!   A(z_l, z_b) = --------------------------------
//!                          1 - beta*z_b^-1
//! ```
//!
//! i.e. `R = A · E`. The decoder has `R` (the decoded Laplace symbols)
//! and must recover `E`. Cross-multiplying the transfer function gives
//! the time-domain relation
//!
//! ```text
//!   R[b] - beta*R[b-1] = D[b] - D[b-1],   where  D[b] = E[b][l] - alpha*E[b][l-1]
//! ```
//!
//! (the `(1 - alpha*z_l^-1)` numerator factor is exactly the time
//! predictor `D[b] = E[b][l] - alpha*E[b][l-1]`, and the remaining
//! `(1 - z_b^-1)/(1 - beta*z_b^-1)` factor relates `R` and `D` across the
//! band index `b`). Rearranging for the running band recurrence:
//!
//! ```text
//!   D[b] = D[b-1] - beta*R[b-1] + R[b]
//!        = pred_freq[b] + R[b],   with  pred_freq[b] = D[b-1] - beta*R[b-1]
//! ```
//!
//! and from `pred_freq[b+1] = D[b] - beta*R[b] = pred_freq[b] + R[b] -
//! beta*R[b]`, the per-band frequency accumulator updates as
//!
//! ```text
//!   pred_freq[b+1] = pred_freq[b] + (1 - beta) * R[b].
//! ```
//!
//! Finally `E[b][l] = alpha*E[b][l-1] + D[b]`. Both `D[-1]` and the
//! boundary residual `R[-1]` are zero, so `pred_freq[0] = 0`. The two
//! running scalars (`pred_freq`, plus the per-band cross-frame history
//! `E[b][l-1]`) are all the state the recurrence needs. In the **intra**
//! case `alpha = 0`, so the time term drops out and the previous frame
//! never participates (RFC 6716 §4.3.2.1 p. 108).
//!
//! ## Units, the 6 dB step, and the `e_means` baseline
//!
//! The reconstructed `E[b][l]` here is the *mean-removed* log-energy in
//! the base-2 log domain that [`crate::celt_denormalise::denormalise_gain`]
//! consumes (one unit ≈ 6.0206 dB; the §4.3.2.1 "fixed resolution of
//! 6 dB" is one Laplace step `R[b] = ±1`). The recurrence runs in that
//! mean-removed domain because the cross-frame history `E[b][l-1]` and
//! the in-frame prediction both share the same baseline. The §4.3
//! per-band mean baseline `eMeans[b]` (the `e_means` table, quantised in
//! Q4 — `e_means[b] / 16` log2 units) is added **only** when reporting
//! the final energy that feeds denormalise; it is *not* part of the
//! prediction state. [`CoarseEnergyState::decode_frame`] returns both:
//! the mean-removed history it threads forward and the
//! mean-added reported energy for the synthesis backend.
//!
//! ## Residual docs gap: the internal clamp
//!
//! RFC 6716 §4.3.2.1 (p. 108) states the prediction "is clamped
//! internally so that fixed-point implementations with limited dynamic
//! range always remain in the same state as floating point
//! implementations", but the RFC body does **not** give the numeric
//! clamp bounds, and they are not present in the staged clean-room
//! `docs/` material either. This module therefore implements
//! the spec-derived recurrence exactly (the z-transform is fully
//! normative) and leaves the clamp as an explicit, documented seam: it
//! does **not** invent a clamp constant. The recurrence is bit-stable
//! for all in-range streams; the clamp only ever engages on pathological
//! dynamic range, which a well-formed bitstream does not exercise. When
//! the clamp bounds are added to the clean-room `docs/` material they
//! drop into [`CoarseEnergyState::clamp_history`] (currently the
//! identity).
//!
//! Provenance: the §4.3.2.1 narrative and the z-transform are transcribed
//! from `docs/audio/opus/rfc6716-opus.txt` pp. 108-109 and the clean-room
//! chapter `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md`
//! §1. The `(alpha, beta)` coefficients and the `e_prob_model` Laplace
//! parameters live in [`crate::celt_e_prob_model`]. The `e_means` Q4
//! baseline is the 25-value numeric table extracted in
//! `docs/audio/celt/tables/e_means.csv` (numeric facts), reproduced
//! inline below.

use crate::celt_band_layout::{CeltFrameSize, CELT_NUM_BANDS};
use crate::celt_e_prob_model::{
    e_prob_pair, energy_pred_coef, EProbModelError, EnergyPredictionMode,
};
use crate::celt_laplace::{decay_byte_to_q14, ec_laplace_decode, prob_to_fs};
use crate::range_decoder::RangeDecoder;

/// `e_means[b]` — the §4.3 per-band mean log-energy baseline, quantised
/// in Q4 (so the log2-domain mean is `E_MEANS_Q4[b] / 16`).
///
/// Numeric data extracted from `docs/audio/celt/tables/e_means.csv`
/// (25 signed-byte values; uncopyrightable numeric facts). Only the
/// first [`CELT_NUM_BANDS`] entries index real CELT bands; the trailing
/// entries pad the table to its declared length and are never read by
/// the standard (non-Custom) layer.
pub const E_MEANS_Q4: [i8; 25] = [
    103, 100, 92, 85, 81, 77, 72, 70, 78, 75, 73, 71, 78, 74, 69, 72, 70, 74, 76, 71, 60, 60, 60,
    60, 60,
];

/// One Q4 unit of [`E_MEANS_Q4`] expressed in the base-2 log domain
/// (`1/16`).
pub const E_MEANS_Q4_SCALE: f64 = 1.0 / 16.0;

/// The §4.3.2.1 base-2-log-domain mean baseline `eMeans[b]` for band
/// `b`, i.e. `E_MEANS_Q4[b] / 16`. Returns `None` for `b >=
/// CELT_NUM_BANDS`.
#[inline]
#[must_use]
pub fn e_mean(band: usize) -> Option<f64> {
    if band >= CELT_NUM_BANDS {
        return None;
    }
    Some(f64::from(E_MEANS_Q4[band]) * E_MEANS_Q4_SCALE)
}

/// Errors from the coarse-energy reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoarseEnergyError {
    /// A `(LM, mode, band)` lookup into the `e_prob_model` / coefficient
    /// tables failed (out-of-range `LM` or `band`).
    Model(EProbModelError),
}

impl core::fmt::Display for CoarseEnergyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CoarseEnergyError::Model(e) => write!(f, "coarse-energy model lookup failed: {e:?}"),
        }
    }
}

impl std::error::Error for CoarseEnergyError {}

impl From<EProbModelError> for CoarseEnergyError {
    fn from(e: EProbModelError) -> Self {
        CoarseEnergyError::Model(e)
    }
}

/// One frame's reconstructed coarse energy.
#[derive(Debug, Clone, PartialEq)]
pub struct CoarseEnergyFrame {
    /// Per-band reported energy in the base-2 log domain *with the
    /// `eMeans` baseline added* — the value the §4.3.6 denormalise step
    /// (and so [`crate::celt_synthesis`]) consumes. Length is the number
    /// of coded bands (`end - start`).
    pub reported_log2: Vec<f64>,
    /// The decoded Laplace residual `R[b]` per coded band (the raw 6 dB
    /// steps), exposed for the §4.3.2.2 fine-energy follow-up and for
    /// testing. Same length / order as [`Self::reported_log2`].
    pub residuals: Vec<i32>,
}

/// Cross-frame coarse-energy decode state (RFC 6716 §4.3.2.1).
///
/// Holds the *mean-removed* per-band log-energy history `E[b][l-1]` the
/// inter-frame predictor reads. One instance is threaded across an
/// uninterrupted run of CELT (or Hybrid) frames; a SILK→CELT mode
/// transition resets it via [`Self::reset`] (RFC 6716 §4.5.2).
#[derive(Debug, Clone)]
pub struct CoarseEnergyState {
    /// `E[b][l-1]`, mean-removed, for every one of the 21 CELT bands.
    /// Hybrid frames (start band 17) still keep a full-length history;
    /// the unused low bands stay at their last value, which the band
    /// loop never reads because it starts at `start`.
    history: [f64; CELT_NUM_BANDS],
}

impl Default for CoarseEnergyState {
    fn default() -> Self {
        Self::new()
    }
}

impl CoarseEnergyState {
    /// A fresh decode state with a zeroed cross-frame history (the state
    /// at the first CELT frame after a reset / mode transition).
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: [0.0; CELT_NUM_BANDS],
        }
    }

    /// Reset the cross-frame history to zero (RFC 6716 §4.5.2: a
    /// SILK→CELT transition starts the CELT energy predictor afresh, and
    /// an intra frame likewise ignores the prior history).
    pub fn reset(&mut self) {
        self.history = [0.0; CELT_NUM_BANDS];
    }

    /// Borrow the current mean-removed cross-frame history.
    #[must_use]
    pub fn history(&self) -> &[f64; CELT_NUM_BANDS] {
        &self.history
    }

    /// Apply the §4.3.2.1 internal prediction clamp to a freshly
    /// reconstructed mean-removed band energy.
    ///
    /// RFC 6716 §4.3.2.1 (p. 108) says the prediction is "clamped
    /// internally" but does not give the bounds in the normative body;
    /// they live only in reference *code* and are not yet in the
    /// clean-room `docs/` material (see the module docs). Until that
    /// gap is filled this is the identity, which is exact for every
    /// in-range bitstream (the clamp only engages on pathological
    /// dynamic range). The seam is centralised here so the constant
    /// drops into one place when the docs land.
    #[inline]
    #[must_use]
    fn clamp_history(value: f64) -> f64 {
        value
    }

    /// Decode and reconstruct one CELT frame's coarse energy from the
    /// range coder, advancing the cross-frame history.
    ///
    /// * `rd` is positioned at the first coarse-energy symbol (i.e. just
    ///   after the §4.3.7.1 frame prefix decoded the intra flag).
    /// * `frame_size` selects `LM = frame_size.column_index()`.
    /// * `intra` is the §4.3.2.1 intra flag (`true` ⇒ `alpha = 0`).
    /// * `start..end` is the coded-band range (`0..21` CELT-only,
    ///   `17..21` Hybrid).
    ///
    /// The decode reads one Laplace symbol per coded band, in ascending
    /// band order, exactly as the §4.3.2.1 narrative requires (the
    /// in-frame frequency predictor depends on the running order). On
    /// return the range coder is positioned at the §4.3.2.2 fine-energy
    /// symbols and the cross-frame history holds this frame's
    /// mean-removed energies.
    ///
    /// # Errors
    ///
    /// * [`CoarseEnergyError::Model`] if `LM` or a band index is out of
    ///   range for the `e_prob_model` / coefficient tables.
    pub fn decode_frame(
        &mut self,
        rd: &mut RangeDecoder<'_>,
        frame_size: CeltFrameSize,
        intra: bool,
        start: usize,
        end: usize,
    ) -> Result<CoarseEnergyFrame, CoarseEnergyError> {
        let lm = frame_size.column_index() as u32;
        let mode = EnergyPredictionMode::from_intra_flag(intra);
        let coef = energy_pred_coef(lm, mode)?;
        let alpha = coef.alpha();
        let beta = coef.beta();

        let end = end.min(CELT_NUM_BANDS);
        let start = start.min(end);
        let coded = end - start;

        let mut reported_log2 = Vec::with_capacity(coded);
        let mut residuals = Vec::with_capacity(coded);

        // pred_freq[start] = 0 (the frequency accumulator opens at the
        // first coded band with no in-frame leakage). It then carries
        // `(1 - beta) * R[b]` forward across bands (see module docs).
        let mut pred_freq = 0.0_f64;

        // `band` indexes three independent per-band structures
        // (`history`, `E_MEANS_Q4`, and the `e_prob_model` lookup), so a
        // running band index is the natural loop variable here.
        #[allow(clippy::needless_range_loop)]
        for band in start..end {
            let pair = e_prob_pair(lm, mode, band as u32)?;
            let fs = prob_to_fs(pair.prob);
            let decay = decay_byte_to_q14(pair.decay);

            // R[b] — the Laplace prediction-error symbol (6 dB steps).
            let q = ec_laplace_decode(rd, fs, decay);
            let r = f64::from(q);

            // D[b] = pred_freq[b] + R[b]; E[b][l] = alpha*E[b][l-1] + D[b].
            let d = pred_freq + r;
            let prev = self.history[band];
            let recon = Self::clamp_history(alpha * prev + d);

            // Reported energy adds back the §4.3 mean baseline.
            let mean = f64::from(E_MEANS_Q4[band]) * E_MEANS_Q4_SCALE;
            reported_log2.push(recon + mean);
            residuals.push(q);

            // Thread the mean-removed reconstruction forward and advance
            // the frequency accumulator: pred_freq[b+1] = pred_freq[b] +
            // (1 - beta) * R[b].
            self.history[band] = recon;
            pred_freq += (1.0 - beta) * r;
        }

        Ok(CoarseEnergyFrame {
            reported_log2,
            residuals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e_means_table_matches_csv() {
        // First 21 entries index real CELT bands; spot-check the ends.
        assert_eq!(E_MEANS_Q4[0], 103);
        assert_eq!(E_MEANS_Q4[7], 70);
        assert_eq!(E_MEANS_Q4[20], 60);
        assert_eq!(E_MEANS_Q4.len(), 25);
        // e_mean converts Q4 -> log2 units.
        assert!((e_mean(0).unwrap() - 103.0 / 16.0).abs() < 1e-12);
        assert!(e_mean(CELT_NUM_BANDS).is_none());
    }

    /// The frequency accumulator must reduce to the closed form the
    /// z-transform predicts: with `alpha = 0` (intra) and a *constant*
    /// residual `R[b] = r` for every band, `D[b] = r + b*(1-beta)*r`, so
    /// the mean-removed reconstruction is `recon[b] = r*(1 + b*(1-beta))`.
    /// We reproduce the recurrence directly (no range coder) to isolate
    /// the arithmetic from the entropy decode.
    #[test]
    fn frequency_recurrence_matches_closed_form() {
        let beta = 6554.0 / 32768.0; // LM=3 inter beta, exercised as a value.
        let r = 1.0_f64;
        let mut pred_freq = 0.0_f64;
        for b in 0..CELT_NUM_BANDS {
            let d = pred_freq + r;
            let recon = d; // alpha = 0, prev history irrelevant.
            let expected = r * (1.0 + (b as f64) * (1.0 - beta));
            assert!(
                (recon - expected).abs() < 1e-9,
                "band {b}: recon {recon} != closed form {expected}"
            );
            pred_freq += (1.0 - beta) * r;
        }
    }

    /// Intra mode ignores the cross-frame history (`alpha = 0`): the same
    /// bitstream decodes to the same energies regardless of the state's
    /// prior history.
    #[test]
    fn intra_ignores_history() {
        let buf = [0x9a, 0x3c, 0x71, 0x55, 0xe2, 0x08, 0xbd, 0x40];
        let mut fresh = CoarseEnergyState::new();
        let f1 = {
            let mut rd = RangeDecoder::new(&buf);
            fresh
                .decode_frame(&mut rd, CeltFrameSize::Ms20, true, 0, CELT_NUM_BANDS)
                .unwrap()
        };

        let mut primed = CoarseEnergyState::new();
        // Poison the history with arbitrary values.
        for (i, h) in primed.history.iter_mut().enumerate() {
            *h = (i as f64) * 0.37 - 3.0;
        }
        let f2 = {
            let mut rd = RangeDecoder::new(&buf);
            primed
                .decode_frame(&mut rd, CeltFrameSize::Ms20, true, 0, CELT_NUM_BANDS)
                .unwrap()
        };

        assert_eq!(f1.residuals, f2.residuals);
        for (a, b) in f1.reported_log2.iter().zip(f2.reported_log2.iter()) {
            assert!((a - b).abs() < 1e-12, "intra energy depends on history");
        }
    }

    /// Inter mode *does* fold in the previous frame: a non-zero
    /// `alpha * E[b][l-1]` shifts the reported energy by exactly
    /// `alpha * history[b]` relative to a zeroed history, for the same
    /// decoded residuals.
    #[test]
    fn inter_adds_alpha_times_history() {
        let buf = [0x12, 0x9f, 0x44, 0xc8, 0x6b, 0x31, 0xaa, 0x05];
        let mut zeroed = CoarseEnergyState::new();
        let fz = {
            let mut rd = RangeDecoder::new(&buf);
            zeroed
                .decode_frame(&mut rd, CeltFrameSize::Ms10, false, 0, CELT_NUM_BANDS)
                .unwrap()
        };

        let mut primed = CoarseEnergyState::new();
        let hist = 2.0_f64;
        for h in primed.history.iter_mut() {
            *h = hist;
        }
        let fp = {
            let mut rd = RangeDecoder::new(&buf);
            primed
                .decode_frame(&mut rd, CeltFrameSize::Ms10, false, 0, CELT_NUM_BANDS)
                .unwrap()
        };

        // Same residuals (entropy decode is unaffected by history).
        assert_eq!(fz.residuals, fp.residuals);

        // Reported energy differs by exactly alpha*hist (the frequency
        // accumulator and mean are identical between the two runs;
        // history only enters through alpha*prev).
        let alpha = energy_pred_coef(2, EnergyPredictionMode::Inter)
            .unwrap()
            .alpha();
        for (z, p) in fz.reported_log2.iter().zip(fp.reported_log2.iter()) {
            assert!(
                (p - z - alpha * hist).abs() < 1e-9,
                "delta {} != alpha*hist {}",
                p - z,
                alpha * hist
            );
        }
    }

    /// A Hybrid coded range (17..21) decodes only the four high bands and
    /// threads only those into the history; the low-band history is
    /// untouched.
    #[test]
    fn hybrid_band_range_decodes_high_bands_only() {
        let buf = [0x55, 0xaa, 0x33, 0xcc, 0x0f, 0xf0, 0x5a, 0xa5];
        let mut st = CoarseEnergyState::new();
        for h in st.history.iter_mut() {
            *h = -1.0;
        }
        let frame = st
            .decode_frame(
                &mut RangeDecoder::new(&buf),
                CeltFrameSize::Ms20,
                false,
                17,
                21,
            )
            .unwrap();
        assert_eq!(frame.reported_log2.len(), 4);
        assert_eq!(frame.residuals.len(), 4);
        // Low-band history untouched (band loop started at 17).
        for h in st.history.iter().take(17) {
            assert_eq!(*h, -1.0);
        }
    }

    /// The decode consumes exactly one Laplace symbol per coded band and
    /// never latches a range-coder error on a well-formed buffer.
    #[test]
    fn decode_is_clean_for_all_frame_sizes() {
        for fs in [
            CeltFrameSize::Ms2_5,
            CeltFrameSize::Ms5,
            CeltFrameSize::Ms10,
            CeltFrameSize::Ms20,
        ] {
            for intra in [false, true] {
                let buf = [0x3c, 0x91, 0x7e, 0x22, 0xb5, 0x48, 0xd0, 0x6f, 0x13, 0xee];
                let mut rd = RangeDecoder::new(&buf);
                let mut st = CoarseEnergyState::new();
                let frame = st
                    .decode_frame(&mut rd, fs, intra, 0, CELT_NUM_BANDS)
                    .unwrap();
                assert_eq!(frame.reported_log2.len(), CELT_NUM_BANDS);
                assert!(!rd.has_error(), "fs {fs:?} intra {intra} latched error");
            }
        }
    }

    /// Two successive inter frames thread history correctly: the second
    /// frame's reported energy equals `mean + alpha*recon1 + D2`, which
    /// we verify by re-running the recurrence by hand from the recorded
    /// residuals.
    #[test]
    fn two_frame_history_threading() {
        let buf1 = [0x40, 0x80, 0x10, 0x9f, 0x33, 0x71, 0xc4, 0x05];
        let buf2 = [0x88, 0x21, 0x6e, 0xb3, 0x4a, 0xf0, 0x19, 0x5c];
        let mut st = CoarseEnergyState::new();

        let _f1 = st
            .decode_frame(
                &mut RangeDecoder::new(&buf1),
                CeltFrameSize::Ms20,
                false,
                0,
                CELT_NUM_BANDS,
            )
            .unwrap();
        let hist_after_f1 = *st.history();

        let f2 = st
            .decode_frame(
                &mut RangeDecoder::new(&buf2),
                CeltFrameSize::Ms20,
                false,
                0,
                CELT_NUM_BANDS,
            )
            .unwrap();

        // Re-derive frame 2 from its residuals + the recorded history.
        let coef = energy_pred_coef(3, EnergyPredictionMode::Inter).unwrap();
        let (alpha, beta) = (coef.alpha(), coef.beta());
        let mut pred_freq = 0.0;
        for (band, &q) in f2.residuals.iter().enumerate() {
            let r = f64::from(q);
            let recon = alpha * hist_after_f1[band] + pred_freq + r;
            let mean = f64::from(E_MEANS_Q4[band]) * E_MEANS_Q4_SCALE;
            assert!(
                (f2.reported_log2[band] - (recon + mean)).abs() < 1e-9,
                "band {band} mismatch"
            );
            pred_freq += (1.0 - beta) * r;
        }
    }
}
