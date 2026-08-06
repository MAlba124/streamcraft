//! CELT encoder analysis front end — §5.3 pre-emphasis, the forward
//! MDCT matching the crate's §4.3.7 synthesis exactly, §4.3.2 band
//! energies / log-energies, band normalisation, and the transient
//! detector (RFC 6716 §5.3, pp. 152–155).
//!
//! ## Transform conventions
//!
//! The crate's inverse MDCT ([`crate::celt_mdct_synthesis`]) is the
//! plain unnormalized cosine sum; the forward transform here carries
//! the `2/N` factor so that analysis → synthesis with the §4.3.7
//! low-overlap window and overlap-add reconstructs the input exactly
//! (verified by the round-trip test below). Each frame analyzes
//! `[overlap-tail of the previous frame | N new samples]`, so the
//! encode→decode chain has a fixed [`CELT_OVERLAP_48K`]-sample
//! (2.5 ms) algorithmic delay — the §4.3.7 MDCT overlap.
//!
//! The signal domain is the decoder's sig-scale (±32768): §5.3
//! pre-emphasis `x'(n) = x(n) - alpha_p * x(n-1)` is the exact inverse
//! of the §4.3.7.2 de-emphasis with the same
//! `alpha_p = 27853/32768` coefficient.
//!
//! ## Provenance
//!
//! RFC 6716 §5.3 narrative + the normative Appendix A reference
//! listing's encoder half (from the staged
//! `docs/audio/opus/rfc6716-opus.txt`, extracted and hash-verified per
//! §A.1). No external library source was consulted.

use crate::celt_band_layout::CELT_NUM_BANDS;
use crate::celt_coarse_energy::e_mean;
use crate::celt_mdct_window::{celt_overlap_window, CELT_OVERLAP_48K};
use crate::celt_rate_alloc::{band_edge, band_width};

/// §5.3 / §4.3.7.2 emphasis coefficient (`27853/32768`).
pub const PREEMPH_COEF: f64 = 27853.0 / 32768.0;

/// The log-energy the listing's encoder assigns to bands above the
/// effective end (never used by standard 48 kHz modes, kept for
/// completeness).
const LOG_E_FLOOR: f64 = -14.0;

/// Streaming pre-emphasis + MDCT input buffering for one CELT encoder
/// (per-channel §4.3.7 overlap history and §5.3 emphasis memory).
#[derive(Debug, Clone)]
pub struct CeltAnalysis {
    channels: usize,
    n: usize,
    /// Per-channel trailing `overlap` pre-emphasized samples of the
    /// previous frame (`in_mem`).
    in_mem: Vec<f64>,
    /// Per-channel §5.3 pre-emphasis memory.
    preemph_mem: [f64; 2],
}

/// One frame's analysis buffers: per-channel planar
/// `[overlap | N new]` pre-emphasized signal.
#[derive(Debug, Clone)]
pub struct AnalysisFrame {
    /// Planar buffers, `channels * (n + overlap)`.
    pub ibuf: Vec<f64>,
    /// Frame length per channel (48 kHz samples).
    pub n: usize,
    /// Overlap length.
    pub overlap: usize,
    /// Channel count.
    pub channels: usize,
    /// True when every pre-emphasized sample of the frame is zero
    /// (the §4.3 silence flag's trigger).
    pub silence: bool,
}

impl CeltAnalysis {
    /// New analysis state for `channels` channels and `n` 48 kHz
    /// samples per frame (120/240/480/960).
    #[must_use]
    pub fn new(channels: usize, n: usize) -> Self {
        let overlap = CELT_OVERLAP_48K.min(n);
        Self {
            channels,
            n,
            in_mem: vec![0.0; channels * overlap],
            preemph_mem: [0.0; 2],
        }
    }

    /// Frame length per channel.
    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.n
    }

    /// Zero the carried state (stream start / §4.5.2 reset).
    pub fn reset(&mut self) {
        self.in_mem.fill(0.0);
        self.preemph_mem = [0.0; 2];
    }

    /// Pre-emphasize one interleaved i16 frame (`channels * n`
    /// samples) into the per-channel `[overlap | N]` analysis layout,
    /// rolling the §4.3.7 overlap history and §5.3 emphasis memory.
    pub fn process_frame(&mut self, pcm: &[i16]) -> AnalysisFrame {
        assert_eq!(pcm.len(), self.channels * self.n);
        let overlap = CELT_OVERLAP_48K.min(self.n);
        let mut ibuf = vec![0.0f64; self.channels * (self.n + overlap)];
        let mut silence = true;
        for c in 0..self.channels {
            let dst = &mut ibuf[c * (self.n + overlap)..(c + 1) * (self.n + overlap)];
            dst[..overlap].copy_from_slice(&self.in_mem[c * overlap..(c + 1) * overlap]);
            let mut mem = self.preemph_mem[c];
            for j in 0..self.n {
                let x = f64::from(pcm[j * self.channels + c]);
                let v = x + mem;
                mem = -PREEMPH_COEF * x;
                dst[overlap + j] = v;
                if v != 0.0 {
                    silence = false;
                }
            }
            self.preemph_mem[c] = mem;
            self.in_mem[c * overlap..(c + 1) * overlap]
                .copy_from_slice(&dst[self.n..self.n + overlap]);
        }
        AnalysisFrame {
            ibuf,
            n: self.n,
            overlap,
            channels: self.channels,
            silence,
        }
    }
}

/// §5.3 transient analysis, transcribed from the reference listing's
/// encoder: high-pass the (channel-summed) signal, take per-half-
/// overlap-block peak magnitudes, and flag a transient when a block
/// dwarfs a run of its neighbours.
#[must_use]
pub fn transient_analysis(frame: &AnalysisFrame) -> bool {
    let len = frame.n + frame.overlap;
    let block = frame.overlap / 2;
    if block == 0 {
        return false;
    }
    let nblocks = len / block;
    let mut tmp = vec![0.0f64; len];
    if frame.channels == 1 {
        tmp.copy_from_slice(&frame.ibuf[..len]);
    } else {
        for (i, v) in tmp.iter_mut().enumerate() {
            *v = frame.ibuf[i] + frame.ibuf[len + i];
        }
    }
    // High-pass filter: (1 - 2 z^-1 + z^-2) / (1 - z^-1 + .5 z^-2).
    let (mut mem0, mut mem1) = (0.0f64, 0.0f64);
    for v in tmp.iter_mut() {
        let x = *v;
        let y = mem0 + x;
        mem0 = mem1 + y - 2.0 * x;
        mem1 = x - 0.5 * y;
        *v = y;
    }
    // The first few samples are bad because we don't propagate the
    // memory.
    for v in tmp.iter_mut().take(12) {
        *v = 0.0;
    }
    let mut bins = vec![0.0f64; nblocks];
    for (i, b) in bins.iter_mut().enumerate() {
        *b = tmp[i * block..(i + 1) * block]
            .iter()
            .fold(0.0f64, |m, &v| m.max(v.abs()));
    }
    let mut is_transient = false;
    for i in 0..nblocks {
        let t1 = 0.15 * bins[i];
        let t2 = 0.4 * bins[i];
        let t3 = 0.15 * bins[i];
        let mut conseq = 0;
        for &b in bins.iter().take(i) {
            if b < t1 {
                conseq += 1;
            }
            if b < t2 {
                conseq += 1;
            } else {
                conseq = 0;
            }
        }
        if conseq >= 3 {
            is_transient = true;
        }
        conseq = 0;
        for &b in bins.iter().skip(i + 1) {
            if b < t3 {
                conseq += 1;
            } else {
                conseq = 0;
            }
        }
        if conseq >= 7 {
            is_transient = true;
        }
    }
    is_transient
}

/// Forward MDCT of one channel's `[overlap | N]` buffer into `N`
/// interleaved frequency bins (`blocks` short MDCTs of `N/blocks`
/// bins each; bin `k` of block `b` lands at `freq[b + k * blocks]`),
/// with the §4.3.7 low-overlap window and the `2/N` scale that makes
/// the crate's synthesis chain the exact inverse.
#[must_use]
pub fn forward_mdct(ibuf: &[f64], n: usize, overlap: usize, blocks: usize) -> Vec<f64> {
    assert_eq!(ibuf.len(), n + overlap);
    let n2 = n / blocks;
    let window = celt_overlap_window();
    let block_overlap = overlap.min(n2);
    let pad = (n2 - block_overlap) / 2;
    let mut freq = vec![0.0f64; n];
    let mut xw = vec![0.0f64; 2 * n2];
    for b in 0..blocks {
        // Windowed block: support t in [pad, pad + n2 + overlap).
        xw.fill(0.0);
        for s in 0..n2 + block_overlap {
            let w = if s < block_overlap {
                window[s]
            } else if s < n2 {
                1.0
            } else {
                window[n2 + block_overlap - 1 - s]
            };
            xw[pad + s] = w * ibuf[b * n2 + s];
        }
        // X(k) = (2/n2) * sum_t xw(t) cos[(pi/n2)(t + 1/2 + n2/2)(k + 1/2)].
        let nf = n2 as f64;
        let scale = 2.0 / nf;
        for k in 0..n2 {
            let base = (std::f64::consts::PI / nf) * (k as f64 + 0.5);
            // phase(t) = base * (t + 1/2 + n2/2); rotation recurrence.
            let (step_c, step_s) = (base.cos(), base.sin());
            let start = base * (0.5 + nf / 2.0);
            let (mut re, mut im) = (start.cos(), start.sin());
            let mut acc = 0.0f64;
            for &xt in xw.iter() {
                acc += xt * re;
                let nre = re * step_c - im * step_s;
                im = re * step_s + im * step_c;
                re = nre;
            }
            freq[b + k * blocks] = scale * acc;
        }
    }
    freq
}

/// §4.3.2 band amplitudes: `bandE[i] = sqrt(1e-27 + Σ freq²)` over the
/// band's `M`-scaled bins.
#[must_use]
pub fn compute_band_energies(freq: &[f64], end: usize, m: usize) -> [f64; CELT_NUM_BANDS] {
    let mut band_e = [0.0f64; CELT_NUM_BANDS];
    for (i, e) in band_e.iter_mut().enumerate().take(end) {
        let off = m * band_edge(i) as usize;
        let len = m * band_width(i) as usize;
        let sum: f64 = 1e-27 + freq[off..off + len].iter().map(|v| v * v).sum::<f64>();
        *e = sum.sqrt();
    }
    band_e
}

/// §4.3.2 log-domain band energies with the means removed
/// (`amp2Log2`): `log2(bandE) - eMeans[i]`, floored at the listing's
/// `-14` above the effective end.
#[must_use]
pub fn amp_to_log2(band_e: &[f64; CELT_NUM_BANDS], end: usize) -> [f64; CELT_NUM_BANDS] {
    let mut out = [LOG_E_FLOOR; CELT_NUM_BANDS];
    for (i, (o, e)) in out.iter_mut().zip(band_e.iter()).enumerate().take(end) {
        *o = e.log2() - e_mean(i).unwrap_or(0.0);
    }
    out
}

/// §4.3.2 band normalisation: unit-norm shapes `X = freq / bandE` over
/// the coded bins (the first `M * 100` of the frame; higher bins are
/// outside every band).
#[must_use]
pub fn normalise_bands(
    freq: &[f64],
    band_e: &[f64; CELT_NUM_BANDS],
    end: usize,
    m: usize,
) -> Vec<f64> {
    let plane = m * band_edge(CELT_NUM_BANDS) as usize;
    let mut x = vec![0.0f64; plane];
    for (i, e) in band_e.iter().enumerate().take(end) {
        let off = m * band_edge(i) as usize;
        let len = m * band_width(i) as usize;
        let g = 1.0 / (1e-27 + e);
        for j in 0..len {
            x[off + j] = freq[off + j] * g;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preemphasis_is_the_exact_inverse_of_deemphasis() {
        // The §4.3.7.2 de-emphasis recurrence with the same
        // 27853/32768 coefficient the decoder's synthesis tail uses
        // (celt_mdct_synthesis) must reconstruct the input exactly.
        let mut an = CeltAnalysis::new(1, 480);
        let mut mem = 0.0f64;
        let pcm: Vec<i16> = (0..960)
            .map(|i| ((i as f64 * 0.1).sin() * 12000.0) as i16)
            .collect();
        let mut recon = Vec::new();
        for chunk in pcm.chunks(480) {
            let frame = an.process_frame(chunk);
            for j in 0..480 {
                let y = frame.ibuf[frame.overlap + j] + mem;
                mem = PREEMPH_COEF * y;
                recon.push(y);
            }
        }
        for (a, b) in pcm.iter().zip(recon.iter()) {
            assert!((f64::from(*a) - b).abs() < 1e-9);
        }
        // And the module-level constant agrees with the decoder's
        // published alpha_p to float precision.
        assert!((PREEMPH_COEF - crate::celt_deemphasis::DEEMPHASIS_ALPHA_P).abs() < 1e-9);
    }

    /// Replicates the synthesis-side windowed overlap-add of
    /// `celt_mdct_synthesis::synthesize_frame` (without post-filter or
    /// de-emphasis) for round-trip verification.
    fn synthesize(freq: &[f64], n: usize, blocks: usize, overlap_mem: &mut [f64]) -> Vec<f64> {
        use crate::celt_imdct::imdct;
        let overlap = overlap_mem.len();
        let window = celt_overlap_window();
        let n2 = n / blocks;
        let block_overlap = overlap.min(n2);
        let pad = (n2 - block_overlap) / 2;
        let mut acc = vec![0.0f64; n + overlap];
        let mut spec = vec![0.0f64; n2];
        for b in 0..blocks {
            for (k, s) in spec.iter_mut().enumerate() {
                *s = freq[b + k * blocks];
            }
            // The crate synthesis kernel is the unnormalized cosine
            // sum; celt_imdct::imdct carries a 1/N scale, so undo it.
            let mut time = imdct(&spec).unwrap();
            for v in time.iter_mut() {
                *v *= n2 as f64;
            }
            for s in 0..n2 + block_overlap {
                let w = if s < block_overlap {
                    window[s]
                } else if s < n2 {
                    1.0
                } else {
                    window[n2 + block_overlap - 1 - s]
                };
                acc[b * n2 + s] += w * time[pad + s];
            }
        }
        let mut out = acc[..n].to_vec();
        for (j, om) in overlap_mem.iter_mut().enumerate() {
            out[j] += *om;
            *om = acc[n + j];
        }
        out
    }

    #[test]
    fn forward_mdct_synthesis_roundtrip_reconstructs_with_overlap_delay() {
        for &(n, blocks) in &[(480usize, 1usize), (960, 1), (960, 8), (240, 2), (120, 1)] {
            let overlap = CELT_OVERLAP_48K.min(n);
            let total = 4 * n;
            let sig: Vec<f64> = (0..total + overlap)
                .map(|i| (i as f64 * 0.037).sin() * 1000.0 + (i as f64 * 0.61).cos() * 300.0)
                .collect();
            let mut om = vec![0.0f64; overlap];
            let mut recon = Vec::new();
            for f in 0..4 {
                // Analysis buffer [overlap history | n new] = stream
                // positions [f*n - overlap, f*n + n).
                let mut ibuf = vec![0.0f64; n + overlap];
                for (j, v) in ibuf.iter_mut().enumerate() {
                    let pos = f as isize * n as isize - overlap as isize + j as isize;
                    if pos >= 0 {
                        *v = sig[pos as usize];
                    }
                }
                let freq = forward_mdct(&ibuf, n, overlap, blocks);
                recon.extend(synthesize(&freq, n, blocks, &mut om));
            }
            // Decoder output position p corresponds to input position
            // p - overlap.
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for p in overlap..recon.len() {
                let x = sig[p - overlap];
                let d = recon[p] - x;
                num += d * d;
                den += x * x;
            }
            let snr = 10.0 * (den / num.max(1e-30)).log10();
            assert!(snr > 120.0, "n={n} blocks={blocks}: snr {snr}");
        }
    }

    #[test]
    fn transient_detector_fires_on_an_attack_not_on_steady_tone() {
        let mut an = CeltAnalysis::new(1, 960);
        // Steady sine: the FIRST frame is itself an attack out of the
        // zero history, so feed two frames and judge the second.
        let tone: Vec<i16> = (0..1920)
            .map(|i| ((i as f64 * 0.3).sin() * 8000.0) as i16)
            .collect();
        let _ = an.process_frame(&tone[..960]);
        let f = an.process_frame(&tone[960..]);
        assert!(!transient_analysis(&f), "steady tone flagged transient");
        // Silence then a hard attack late in the frame.
        let mut atk = vec![0i16; 960];
        for (i, v) in atk.iter_mut().enumerate().skip(700) {
            *v = (((i - 700) as f64 * 0.9).sin() * 20000.0) as i16;
        }
        let mut an2 = CeltAnalysis::new(1, 960);
        let f2 = an2.process_frame(&atk);
        assert!(transient_analysis(&f2), "attack not flagged transient");
    }

    #[test]
    fn band_energy_of_a_pure_tone_lands_in_its_band_and_normalises() {
        // 1 kHz tone at LM=3 (960 bins ~ 25 Hz/bin): bin ~40 → band
        // containing short-bin 5 = band 5 region. Just assert the
        // arg-max band and unit-norm shapes.
        let n = 960;
        let overlap = CELT_OVERLAP_48K;
        let ibuf: Vec<f64> = (0..n + overlap)
            .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48000.0).sin() * 8000.0)
            .collect();
        let freq = forward_mdct(&ibuf, n, overlap, 1);
        let be = compute_band_energies(&freq, 21, 8);
        let peak = (0..21).max_by(|&a, &b| be[a].total_cmp(&be[b])).unwrap();
        let edge_lo = 8 * band_edge(peak);
        let edge_hi = 8 * band_edge(peak + 1);
        // 1 kHz at 25 Hz/bin = bin 40.
        assert!((edge_lo..edge_hi).contains(&40), "peak band {peak}");
        let x = normalise_bands(&freq, &be, 21, 8);
        for i in 0..21 {
            let off = 8 * band_edge(i) as usize;
            let len = 8 * band_width(i) as usize;
            let norm: f64 = x[off..off + len].iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-9, "band {i} norm {norm}");
        }
        let lg = amp_to_log2(&be, 21);
        assert!(lg[peak] > lg[(peak + 5).min(20)]);
    }
}
