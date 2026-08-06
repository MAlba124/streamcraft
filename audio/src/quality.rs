//! Perceptual audio-quality measurement for transcode QA — how audible is the difference between
//! a reference signal and its transcoded version? A **masking-model** metric, not a waveform one:
//! it answers "is the distortion below the ear's masking threshold?" (which is exactly what a
//! perceptual codec like Opus/AAC optimizes for), so — unlike SNR — a transparent lossy transcode
//! scores well even though its samples differ.
//!
//! # What it computes
//!
//! The **front half of PEAQ Basic** (its FFT ear model), implemented from the open spec
//! (ITU-R BS.1387 + Kabal's McGill report) — **without** PEAQ's trained MOV→ODG neural net, which
//! is the un-reproducible part. Per frame: Hann-windowed FFT → power spectrum → critical-band
//! (Bark) grouping → frequency spreading → excitation pattern → masking threshold. From that:
//!
//! - **NMR (Noise-to-Mask Ratio)** — error-energy / masking-threshold per band. **`nmr_db < 0` ≈
//!   the distortion is masked (transparent); `> 0` ≈ audible.** This is the headline signal.
//! - **Bark-band loudness difference** — a compressive specific-loudness distance (Zwicker), a
//!   secondary "distortion loudness" that catches errors NMR alone underweights.
//!
//! Not PEAQ-conformant (no tonality-adaptive masking offset, no forward masking, no calibrated
//! MOS mapping) — a principled, interpretable gate to build on. Calibrate the `nmr_db` threshold
//! against a real reference (ViSQOL / GstPEAQ) offline; see `audio/PERCEPTUAL_QUALITY.md`.
//!
//! # No hot-path allocation
//!
//! [`PerceptualAnalyzer::new`] allocates every scratch buffer once; [`analyze`](
//! PerceptualAnalyzer::analyze) reuses them across every frame — the per-frame loop allocates
//! nothing (spec: performance #1). Reuse one analyzer across many files in a transcode QA sweep.
//!
//! Inputs must be **mono, same length, same sample rate, time-aligned and level-matched** — use
//! [`rms_normalize`] and [`best_offset`] first (a full-reference metric scores a perfect transcode
//! terribly if it is a few ms or 0.5 dB off; Opus adds ~6.5 ms SILK / 2.5 ms CELT look-ahead).
//!
//! # References (clean-room from the primary sources)
//!
//! - **FFT:** Cooley, J. W. & Tukey, J. W. (1965), "An Algorithm for the Machine Calculation of
//!   Complex Fourier Series," *Math. Comp.* 19(90), 297–301 (the radix-2 decimation-in-time).
//! - **Ear model / NMR / masking offset:** ITU-R Rec. BS.1387 (PEAQ); Kabal, P. (2002), "An
//!   Examination and Interpretation of ITU-R BS.1387," McGill University TSP Lab report.
//! - **Bark scale:** Zwicker, E. & Fastl, H., *Psychoacoustics: Facts and Models*; the analytic
//!   critical-band-rate form is Zwicker & Terhardt (1980) / Traunmüller (1990).
//! - **Absolute threshold of hearing:** Terhardt, E. (1979), used in the MPEG-1 (ISO/IEC 11172-3)
//!   psychoacoustic model.
//! - **Spreading function:** Schroeder, M. R., Atal, B. S. & Hall, J. L. (1979), "Optimizing
//!   digital speech coders by exploiting masking properties of the human ear," *JASA* 66(6).
//! - **Specific loudness:** Zwicker's compressive loudness (Stevens' power law, exponent ≈ 0.23).

// Only the one-time table construction (`new`) and the tests allocate; they carry a scoped
// `#[allow]`. `analyze()` and the helpers stay under the allocation ban so clippy *proves* the
// per-frame hot loop is allocation-free (spec: performance #1 — no steady-state heap traffic).

use std::f64::consts::PI;

/// Analysis frame size (a power of two for the radix-2 FFT); ~43 ms at 48 kHz, as PEAQ uses.
const N: usize = 2048;
/// Hop between frames (50 % overlap).
const HOP: usize = N / 2;
/// Bark-band resolution.
const BARK_STEP: f64 = 0.5;
/// Stevens'-law loudness compression exponent (Zwicker specific loudness).
const LOUDNESS_EXP: f64 = 0.23;
/// Floor for the NMR dB conversion when the error energy is ~0 (bit-identical inputs).
const NMR_FLOOR_DB: f64 = -200.0;

/// The result of a perceptual comparison.
#[derive(Clone, Copy, Debug)]
pub struct QualityReport {
    /// Average Noise-to-Mask Ratio in dB. **`< 0` ≈ transparent** (distortion masked); the more
    /// positive, the more audible. Bit-identical inputs report [`NMR_FLOOR_DB`].
    pub nmr_db: f64,
    /// Worst single-frame NMR (dB) — catches localized artifacts an average hides.
    pub peak_nmr_db: f64,
    /// Mean per-frame bark-band loudness difference (a unitless "distortion loudness"); `0` for
    /// identical signals, growing with audible timbre/level error.
    pub noise_loudness: f64,
    /// Frames analyzed.
    pub frames: usize,
}

impl QualityReport {
    /// Is the transcode perceptually transparent at `nmr_threshold_db`? A small positive margin
    /// (e.g. `1.0`) is a reasonable "acceptable" gate; calibrate against a reference metric.
    pub fn is_transparent(&self, nmr_threshold_db: f64) -> bool {
        self.nmr_db < nmr_threshold_db
    }
}

/// Frequency (Hz) → Bark critical-band rate (Zwicker & Terhardt analytic approximation; see the
/// module References).
fn hz_to_bark(f: f64) -> f64 {
    13.0 * (0.00076 * f).atan() + 3.5 * ((f / 7500.0).powi(2)).atan()
}

/// Absolute threshold of hearing (energy, on the FFT power scale) at `f` Hz — the "internal
/// noise" floor so inaudibly-quiet bands don't produce huge NMR. Terhardt's ATH curve (module
/// References), as used by the MPEG-1 psychoacoustic model.
fn ath_energy(f: f64) -> f64 {
    let k = (f / 1000.0).max(0.02);
    let db = 3.64 * k.powf(-0.8) - 6.5 * (-0.6 * (k - 3.3).powi(2)).exp() + 1e-3 * k.powi(4);
    10f64.powf(db / 10.0)
}

/// Schroeder–Atal–Hall level-independent spreading function (dB) at a Bark distance `dz` (module
/// References) — models how a masker's energy spreads across critical bands.
fn spreading_db(dz: f64) -> f64 {
    15.81 + 7.5 * (dz + 0.474) - 17.5 * (1.0 + (dz + 0.474).powi(2)).sqrt()
}

/// Reusable perceptual analyzer: fixed tables + per-frame scratch, all allocated once.
pub struct PerceptualAnalyzer {
    sample_rate: u32,
    num_bands: usize,
    // -- precomputed tables (built once in `new`) --
    window: Vec<f64>,          // Hann window, len N
    band_of_bin: Vec<usize>,   // band index for each bin 0..=N/2
    spread: Vec<f64>,          // num_bands×num_bands spreading matrix (energy-domain, col-normalized)
    ath: Vec<f64>,             // absolute-threshold energy per band
    offset_lin: Vec<f64>,      // masking offset (linear) per band: threshold = excitation / offset
    // -- per-frame scratch (reused; `analyze` never allocates) --
    re_r: Vec<f64>,
    im_r: Vec<f64>,
    re_t: Vec<f64>,
    im_t: Vec<f64>,
    e_ref: Vec<f64>,   // band energy of reference / test / error
    e_test: Vec<f64>,
    e_err: Vec<f64>,
    exc_ref: Vec<f64>, // spread excitation
    exc_test: Vec<f64>,
    mask: Vec<f64>,
}

impl PerceptualAnalyzer {
    /// Build the analyzer for a given sample rate (mono). One-time table construction; after this,
    /// [`analyze`](Self::analyze) allocates nothing.
    #[allow(clippy::disallowed_methods)] // one-time fixed-table + scratch allocation
    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate as f64;
        let half = N / 2;

        // Hann window.
        let mut window = vec![0.0; N];
        for (i, w) in window.iter_mut().enumerate() {
            *w = 0.5 - 0.5 * (2.0 * PI * i as f64 / N as f64).cos();
        }

        // Bin → Bark band, and per-band representative frequency (mean of its bins).
        let mut band_of_bin = vec![0usize; half + 1];
        let bark_max = hz_to_bark(sr / 2.0);
        let num_bands = ((bark_max / BARK_STEP).ceil() as usize).max(1);
        let mut band_freq_sum = vec![0.0f64; num_bands];
        let mut band_bin_count = vec![0u32; num_bands];
        for (k, band) in band_of_bin.iter_mut().enumerate() {
            let f = k as f64 * sr / N as f64;
            let b = ((hz_to_bark(f) / BARK_STEP) as usize).min(num_bands - 1);
            *band = b;
            band_freq_sum[b] += f;
            band_bin_count[b] += 1;
        }

        // Per-band ATH (at the band's centre freq) + PEAQ masking offset.
        let mut ath = vec![0.0f64; num_bands];
        let mut offset_lin = vec![0.0f64; num_bands];
        for b in 0..num_bands {
            let f = if band_bin_count[b] > 0 {
                band_freq_sum[b] / band_bin_count[b] as f64
            } else {
                // Empty band: use its Bark midpoint mapped back to ~Hz for the ATH lookup.
                let z = (b as f64 + 0.5) * BARK_STEP;
                600.0 * (z / 6.0).sinh() // inverse-Bark approximation
            };
            ath[b] = ath_energy(f);
            let z = b as f64 * BARK_STEP;
            let m_db = if z <= 12.0 { 3.0 } else { 0.25 * z }; // BS.1387 masking offset
            offset_lin[b] = 10f64.powf(m_db / 10.0);
        }

        // Spreading matrix, column-normalized so spreading conserves energy.
        let mut spread = vec![0.0f64; num_bands * num_bands];
        for src in 0..num_bands {
            let mut col_sum = 0.0;
            for dst in 0..num_bands {
                let dz = (dst as f64 - src as f64) * BARK_STEP;
                let v = 10f64.powf(spreading_db(dz) / 10.0);
                spread[dst * num_bands + src] = v;
                col_sum += v;
            }
            if col_sum > 0.0 {
                for dst in 0..num_bands {
                    spread[dst * num_bands + src] /= col_sum;
                }
            }
        }

        PerceptualAnalyzer {
            sample_rate,
            num_bands,
            window,
            band_of_bin,
            spread,
            ath,
            offset_lin,
            re_r: vec![0.0; N],
            im_r: vec![0.0; N],
            re_t: vec![0.0; N],
            im_t: vec![0.0; N],
            e_ref: vec![0.0; num_bands],
            e_test: vec![0.0; num_bands],
            e_err: vec![0.0; num_bands],
            exc_ref: vec![0.0; num_bands],
            exc_test: vec![0.0; num_bands],
            mask: vec![0.0; num_bands],
        }
    }

    /// Compare `reference` and `test` (mono, same length, already aligned + level-matched). The
    /// per-frame loop reuses `self`'s buffers — **no allocation**.
    pub fn analyze(&mut self, reference: &[f32], test: &[f32]) -> QualityReport {
        let len = reference.len().min(test.len());
        let num_bands = self.num_bands;
        let mut nmr_sum = 0.0f64; // Σ per-frame mean NMR (linear)
        let mut nmr_peak = 0.0f64;
        let mut loud_sum = 0.0f64;
        let mut frames = 0usize;

        let mut start = 0;
        while start + N <= len {
            // Windowed real frames into the reused FFT buffers.
            for i in 0..N {
                let w = self.window[i];
                self.re_r[i] = reference[start + i] as f64 * w;
                self.re_t[i] = test[start + i] as f64 * w;
            }
            self.im_r.iter_mut().for_each(|x| *x = 0.0);
            self.im_t.iter_mut().for_each(|x| *x = 0.0);
            fft(&mut self.re_r, &mut self.im_r);
            fft(&mut self.re_t, &mut self.im_t);

            // Band energies of reference, test, and the error spectrum (Xref − Xtest).
            self.e_ref.iter_mut().for_each(|x| *x = 0.0);
            self.e_test.iter_mut().for_each(|x| *x = 0.0);
            self.e_err.iter_mut().for_each(|x| *x = 0.0);
            for k in 0..=N / 2 {
                let b = self.band_of_bin[k];
                let pr = self.re_r[k] * self.re_r[k] + self.im_r[k] * self.im_r[k];
                let pt = self.re_t[k] * self.re_t[k] + self.im_t[k] * self.im_t[k];
                let dr = self.re_r[k] - self.re_t[k];
                let di = self.im_r[k] - self.im_t[k];
                self.e_ref[b] += pr;
                self.e_test[b] += pt;
                self.e_err[b] += dr * dr + di * di;
            }

            // Spread → excitation, then masking threshold = excitation / offset, ATH-floored.
            for dst in 0..num_bands {
                let mut er = 0.0;
                let mut et = 0.0;
                let row = dst * num_bands;
                for src in 0..num_bands {
                    let s = self.spread[row + src];
                    er += self.e_ref[src] * s;
                    et += self.e_test[src] * s;
                }
                self.exc_ref[dst] = er + self.ath[dst];
                self.exc_test[dst] = et + self.ath[dst];
                self.mask[dst] = (er / self.offset_lin[dst]).max(self.ath[dst]);
            }

            // NMR + loudness difference for this frame.
            let mut nmr_frame = 0.0;
            let mut loud_frame = 0.0;
            for b in 0..num_bands {
                nmr_frame += self.e_err[b] / self.mask[b];
                let nr = (self.exc_ref[b] / self.ath[b]).powf(LOUDNESS_EXP);
                let nt = (self.exc_test[b] / self.ath[b]).powf(LOUDNESS_EXP);
                loud_frame += (nr - nt).abs();
            }
            nmr_frame /= num_bands as f64;
            loud_frame /= num_bands as f64;

            nmr_sum += nmr_frame;
            nmr_peak = nmr_peak.max(nmr_frame);
            loud_sum += loud_frame;
            frames += 1;
            start += HOP;
        }

        let to_db = |lin: f64| if lin <= 0.0 { NMR_FLOOR_DB } else { (10.0 * lin.log10()).max(NMR_FLOOR_DB) };
        QualityReport {
            nmr_db: if frames == 0 { NMR_FLOOR_DB } else { to_db(nmr_sum / frames as f64) },
            peak_nmr_db: to_db(nmr_peak),
            noise_loudness: if frames == 0 { 0.0 } else { loud_sum / frames as f64 },
            frames,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// One-shot convenience: build an analyzer and compare. For a QA sweep over many files, keep one
/// [`PerceptualAnalyzer`] and call [`analyze`](PerceptualAnalyzer::analyze) instead (this
/// allocates the analyzer per call).
pub fn perceptual_diff(reference: &[f32], test: &[f32], sample_rate: u32) -> QualityReport {
    PerceptualAnalyzer::new(sample_rate).analyze(reference, test)
}

// --- alignment / level helpers (full-reference preprocessing) ------------------------------

/// Scale `test` in place so its RMS matches `reference`'s — a benign gain change must not read as
/// distortion. No-op if either signal is silent.
pub fn rms_normalize(reference: &[f32], test: &mut [f32]) {
    let rms = |s: &[f32]| (s.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / s.len().max(1) as f64).sqrt();
    let (rr, rt) = (rms(reference), rms(test));
    if rt > 1e-12 && rr > 1e-12 {
        let g = (rr / rt) as f32;
        test.iter_mut().for_each(|x| *x *= g);
    }
}

/// Best integer sample shift of `test` relative to `reference` (searching `±max_shift`), by
/// normalized cross-correlation over a leading window. A positive result means `test` lags — drop
/// that many leading samples of `test` to align. Absorbs codec look-ahead / pre-skip.
pub fn best_offset(reference: &[f32], test: &[f32], max_shift: usize) -> isize {
    let win = reference.len().min(test.len()).min(48_000).saturating_sub(max_shift);
    if win == 0 {
        return 0;
    }
    let mut best = (f64::NEG_INFINITY, 0isize);
    let mut shift = -(max_shift as isize);
    while shift <= max_shift as isize {
        let mut acc = 0.0f64;
        for i in 0..win {
            let ti = i as isize + shift;
            if ti >= 0 && (ti as usize) < test.len() {
                acc += reference[i] as f64 * test[ti as usize] as f64;
            }
        }
        if acc > best.0 {
            best = (acc, shift);
        }
        shift += 1;
    }
    best.1
}

// --- from-scratch radix-2 FFT (in place; no allocation) ------------------------------------

/// In-place iterative radix-2 decimation-in-time FFT — Cooley & Tukey (1965), see the module
/// References. `re.len()` must be a power of two. Twiddle factors are generated on the fly by
/// recurrence, so it allocates nothing (bit-reversal is an in-place swap).
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    // Butterflies.
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let (wlr, wli) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let a = i + k;
                let b = i + k + len / 2;
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wlr - ci * wli;
                ci = cr * wli + ci * wlr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests build throwaway signal buffers
mod tests {
    use super::*;

    fn sine(freq: f64, sr: u32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * PI * freq * i as f64 / sr as f64).sin() as f32).collect()
    }

    #[test]
    fn identical_signals_are_transparent() {
        let s = sine(1000.0, 48_000, 48_000);
        let r = perceptual_diff(&s, &s, 48_000);
        assert!(r.frames > 0);
        // No error → NMR at the floor, zero distortion loudness.
        assert!(r.nmr_db < -100.0, "identical NMR should be very low, got {}", r.nmr_db);
        assert!(r.noise_loudness < 1e-6, "identical loudness diff should be ~0, got {}", r.noise_loudness);
        assert!(r.is_transparent(0.0));
    }

    #[test]
    fn nmr_grows_with_added_noise() {
        let clean = sine(1000.0, 48_000, 48_000);
        let mut analyzer = PerceptualAnalyzer::new(48_000);
        // Deterministic pseudo-noise added at increasing levels.
        let noisy = |amp: f32| -> Vec<f32> {
            let mut seed = 0x2545F4914F6CDD1Du64;
            clean
                .iter()
                .map(|&c| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let n = ((seed >> 40) as f32 / (1u32 << 24) as f32) - 0.5;
                    c + amp * n
                })
                .collect()
        };
        let low = analyzer.analyze(&clean, &noisy(0.001)).nmr_db;
        let high = analyzer.analyze(&clean, &noisy(0.05)).nmr_db;
        assert!(high > low, "more noise must raise NMR: {low} -> {high}");
        // Audible noise should read as such (NMR clearly above the masked floor).
        assert!(high > -20.0, "loud noise should be audible, NMR {high}");
    }

    #[test]
    fn rms_normalize_matches_levels() {
        let r = sine(440.0, 48_000, 4096);
        let mut t: Vec<f32> = r.iter().map(|&x| x * 3.0).collect(); // +9.5 dB
        rms_normalize(&r, &mut t);
        let rms = |s: &[f32]| (s.iter().map(|&x| x * x).sum::<f32>() / s.len() as f32).sqrt();
        assert!((rms(&r) - rms(&t)).abs() < 1e-4, "levels should match after normalize");
    }

    #[test]
    fn best_offset_finds_a_known_shift() {
        let r = sine(300.0, 48_000, 20_000);
        let shift = 137usize;
        let mut t = vec![0.0f32; shift];
        t.extend_from_slice(&r); // `test` lags `reference` by `shift`
        assert_eq!(best_offset(&r, &t, 400), shift as isize);
    }

    #[test]
    fn fft_matches_naive_dft() {
        let n = 16;
        let sig: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin() + 0.3 * (i as f64 * 2.1).cos()).collect();
        let mut re = sig.clone();
        let mut im = vec![0.0; n];
        fft(&mut re, &mut im);
        for k in 0..n {
            let (mut dr, mut di) = (0.0, 0.0);
            for (t, &x) in sig.iter().enumerate() {
                let a = -2.0 * PI * k as f64 * t as f64 / n as f64;
                dr += x * a.cos();
                di += x * a.sin();
            }
            assert!((re[k] - dr).abs() < 1e-9 && (im[k] - di).abs() < 1e-9, "bin {k}");
        }
    }
}
