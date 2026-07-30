//! CELT §4.3.6–§4.3.7.2 frame synthesis — denormalisation, inverse
//! MDCT (long and short blocks), weighted overlap-add, §4.3.7.1 pitch
//! post-filter, and §4.3.7.2 de-emphasis, with the cross-frame state
//! (RFC 6716 §4.3.6 / §4.3.7, pp. 120–122).
//!
//! This is the signal half of the CELT frame decode: it consumes the
//! normalized spectra and band gains produced by
//! [`crate::celt_frame_decode`] and emits interleaved 16-bit PCM at
//! 48 kHz, carrying four pieces of state across frames:
//!
//! * the §4.3.7 overlap-add memory (120 samples per channel),
//! * the pre-de-emphasis signal history the §4.3.7.1 comb filter
//!   reaches back into (up to 1024 samples per channel),
//! * the post-filter parameter pair (previous / two-frames-back) that
//!   drives the crossfaded filter transition at each frame head, and
//! * the §4.3.7.2 one-pole de-emphasis memory.
//!
//! ## Short blocks
//!
//! A transient frame codes `M = 2^LM` interleaved short MDCTs of 120
//! bins each (coefficient `k` of block `b` at index `b + k*M`). Every
//! block — long or short — contributes `N2 + overlap` windowed
//! samples starting at its block offset (the low-overlap window is
//! zero over the first and last `(N2 - overlap)/2` of the nominal
//! `2*N2` output), so one accumulation buffer of `N + overlap`
//! samples covers both layouts; the trailing `overlap` becomes the
//! next frame's carried memory.
//!
//! ## Post-filter
//!
//! Per §4.3.7.1 the frame's first 120 samples crossfade from the
//! two-frames-back filter to the previous frame's filter, and the
//! remainder runs the previous → current transition (for 2.5 ms
//! frames only the first segment exists and the parameter history
//! shifts accordingly). The comb filter is applied *in place* over
//! the running signal history, so the feedback taps read
//! already-filtered samples — the recursive structure the reference
//! decoder specifies.
//!
//! ## Provenance
//!
//! RFC 6716 §4.3.6–§4.3.7.2 narrative + the normative Appendix A
//! reference decoder, from the staged `docs/audio/opus/rfc6716-opus.txt`.
//! The window is [`crate::celt_mdct_window`]'s §4.3.7 construction;
//! the tapset gains are [`crate::celt_post_filter::POST_FILTER_TAPS`].
//! No external library source was consulted.

use crate::celt_frame_decode::CeltPostFilterOut;
use crate::celt_mdct_window::{celt_overlap_window, CELT_OVERLAP_48K};
use crate::celt_post_filter::POST_FILTER_TAPS;

/// The §4.3.7.1 minimum comb-filter period.
const COMB_MIN_PERIOD: usize = 15;
/// The maximum comb-filter reach (history depth per channel).
const COMB_MAX_PERIOD: usize = 1024;
/// The §4.3.7.2 de-emphasis coefficient (the exact Q15 value
/// `27853/32768`).
const DEEMPH_COEF: f64 = 27853.0 / 32768.0;

/// Cross-frame CELT synthesis state (one stream, 1–2 channels, fixed
/// frame size).
#[derive(Debug, Clone)]
pub struct CeltSynthesis {
    channels: usize,
    /// Frame length per channel (120/240/480/960).
    n: usize,
    /// §4.3.7 overlap memory, `channels * overlap`.
    overlap_mem: Vec<f64>,
    /// Signal history for the comb filter (post-filtered,
    /// pre-de-emphasis), `channels * (COMB_MAX_PERIOD + 2)`.
    hist: Vec<f64>,
    /// §4.3.7.2 de-emphasis memory per channel.
    deemph_mem: Vec<f64>,
    pf_period: usize,
    pf_period_old: usize,
    pf_gain: f64,
    pf_gain_old: f64,
    pf_tapset: usize,
    pf_tapset_old: usize,
}

impl CeltSynthesis {
    /// Fresh zeroed state for `channels` (1 or 2) at frame length `n`
    /// samples per channel.
    #[must_use]
    #[allow(clippy::disallowed_methods)] // one-time construction; reserves the max-channel capacity
    pub fn new(channels: usize, n: usize) -> Self {
        // Profluens patch: reserve the max-channel (2) capacity for the carried synthesis state
        // once here, then set the logical length for `channels`. `set_geometry` rearranges IN PLACE
        // within this reserved capacity (resize/copy_within/truncate — never a fresh Vec), so a
        // §4.5 channel/geometry transition reallocates nothing: the decode is allocation-free even
        // across mode switches while this state correctly stays on the global allocator (a per-packet
        // bump reset would free it — the documented SEGV). Per-channel overlap is a constant 120 for
        // every legal 48 kHz frame, so `2 * CELT_OVERLAP_48K` is the true max.
        let overlap = CELT_OVERLAP_48K.min(n);
        let hist_len = COMB_MAX_PERIOD + 2;
        let mut overlap_mem = Vec::with_capacity(2 * CELT_OVERLAP_48K);
        overlap_mem.resize(channels * overlap, 0.0);
        let mut hist = Vec::with_capacity(2 * hist_len);
        hist.resize(channels * hist_len, 0.0);
        let mut deemph_mem = Vec::with_capacity(2);
        deemph_mem.resize(channels, 0.0);
        Self {
            channels,
            n,
            overlap_mem,
            hist,
            deemph_mem,
            pf_period: 0,
            pf_period_old: 0,
            pf_gain: 0.0,
            pf_gain_old: 0.0,
            pf_tapset: 0,
            pf_tapset_old: 0,
        }
    }

    /// Frame length per channel.
    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.n
    }

    /// Channel count.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Change the frame geometry **without** dropping the carried
    /// state (§4.5.1.4 / §4.5.2: the 5 ms redundant CELT frame and the
    /// frames around a transition share one CELT decoder whose state
    /// is only reset where the §4.5.2 rules say so, even though the
    /// frame sizes differ).
    ///
    /// The §4.3.7 overlap is [`CELT_OVERLAP_48K`] samples for every
    /// legal 48 kHz frame length (120/240/480/960), and the comb-filter
    /// history and de-emphasis memory are length-independent, so a
    /// frame-length change carries everything verbatim. A channel-count
    /// change adapts the per-channel state the way the §4.5 transition
    /// text expects a channel switch to be handled smoothly: mono →
    /// stereo duplicates the mono state into both channels; stereo →
    /// mono averages the two.
    pub fn set_geometry(&mut self, channels: usize, n: usize) {
        if channels != self.channels {
            let overlap = CELT_OVERLAP_48K.min(self.n);
            let hist_len = COMB_MAX_PERIOD + 2;
            // Profluens patch: rearrange the carried state IN PLACE within the reserved capacity
            // (see `new`) so a channel switch reallocates nothing. Bit-exact with the old
            // pick/repeat/average-then-replace: channel 0 ends holding the mono (or averaged) state,
            // then it is spread across the new channel count and the length trimmed.
            if self.channels != 1 {
                // Average the old channels down into channel 0 (read-before-write per position, and
                // only position `j < overlap` of channel 0 is written each step, so the higher
                // channels' data — read via `c * overlap + j`, c ≥ 1 — is never clobbered).
                let old_c = self.channels as f64;
                for j in 0..overlap {
                    let mut acc = 0.0f64;
                    for c in 0..self.channels {
                        acc += self.overlap_mem[c * overlap + j] / old_c;
                    }
                    self.overlap_mem[j] = acc;
                }
                for j in 0..hist_len {
                    let mut acc = 0.0f64;
                    for c in 0..self.channels {
                        acc += self.hist[c * hist_len + j] / old_c;
                    }
                    self.hist[j] = acc;
                }
                let mut de0 = 0.0f64;
                for c in 0..self.channels {
                    de0 += self.deemph_mem[c] / old_c;
                }
                self.deemph_mem[0] = de0;
            }
            // Channel 0 now holds the state to spread. Grow/shrink the logical length within the
            // reserved capacity (no realloc), then duplicate channel 0 into the new channels.
            let de0 = self.deemph_mem[0];
            self.overlap_mem.resize(channels * overlap, 0.0);
            self.hist.resize(channels * hist_len, 0.0);
            self.deemph_mem.resize(channels, de0);
            for c in 1..channels {
                self.overlap_mem.copy_within(0..overlap, c * overlap);
                self.hist.copy_within(0..hist_len, c * hist_len);
                self.deemph_mem[c] = de0;
            }
            self.channels = channels;
        }
        if n != self.n {
            // The overlap length is CELT_OVERLAP_48K for every legal n
            // (all are >= 120); re-slice defensively if it ever
            // differs.
            let old_overlap = CELT_OVERLAP_48K.min(self.n);
            let new_overlap = CELT_OVERLAP_48K.min(n);
            if new_overlap != old_overlap {
                let mut om = vec![0.0f64; self.channels * new_overlap];
                for c in 0..self.channels {
                    let keep = new_overlap.min(old_overlap);
                    om[c * new_overlap..c * new_overlap + keep].copy_from_slice(
                        &self.overlap_mem[c * old_overlap..c * old_overlap + keep],
                    );
                }
                self.overlap_mem = om;
            }
            self.n = n;
        }
    }

    /// Zero all carried state (§4.5.2 CELT reset).
    pub fn reset(&mut self) {
        self.overlap_mem.fill(0.0);
        self.hist.fill(0.0);
        self.deemph_mem.fill(0.0);
        self.pf_period = 0;
        self.pf_period_old = 0;
        self.pf_gain = 0.0;
        self.pf_gain_old = 0.0;
        self.pf_tapset = 0;
        self.pf_tapset_old = 0;
    }

    /// Synthesize one frame: `freq` holds the per-channel planar
    /// *denormalised* spectra (`channels * n` bins; bins the frame
    /// does not code must be zero), `blocks` is the short-block count
    /// (`1` for non-transient, `2^LM` for transient), and `pf` the
    /// frame's decoded post-filter parameters (`None` = not
    /// signalled, gain 0). Returns `channels * n` interleaved i16
    /// samples.
    pub fn synthesize_frame(
        &mut self,
        freq: &[f64],
        blocks: usize,
        pf: Option<CeltPostFilterOut>,
        out: &mut [i16],
    ) {
        debug_assert_eq!(freq.len(), self.channels * self.n);
        debug_assert_eq!(out.len(), self.channels * self.n);
        let n = self.n;
        let overlap = CELT_OVERLAP_48K.min(n);
        let window = celt_overlap_window();
        let hist_len = COMB_MAX_PERIOD + 2;

        let (pf_period_new, pf_gain_new, pf_tapset_new) = match pf {
            Some(p) => (p.period, p.gain, p.tapset),
            None => (0, 0.0, 0),
        };

        // Profluens patch: per-frame synthesis scratch (written to `out`, never carried) — from the
        // per-packet bump arena. The carried overlap/history/de-emphasis state stays on the global
        // allocator (updated in place below).
        let mut planar = Vec::with_capacity_in(self.channels * n, crate::bump::DecodeBump);
        planar.resize(self.channels * n, 0.0f64);
        for c in 0..self.channels {
            // §4.3.7 inverse MDCT + windowed overlap-add.
            let mut acc = Vec::with_capacity_in(n + overlap, crate::bump::DecodeBump);
            acc.resize(n + overlap, 0.0f64);
            let n2 = n / blocks;
            let pad = (n2 - overlap.min(n2)) / 2;
            let block_overlap = overlap.min(n2);
            let mut spec = Vec::with_capacity_in(n2, crate::bump::DecodeBump);
            spec.resize(n2, 0.0f64);
            let mut time = Vec::with_capacity_in(2 * n2, crate::bump::DecodeBump);
            time.resize(2 * n2, 0.0f64);
            for b in 0..blocks {
                for (k, slot) in spec.iter_mut().enumerate() {
                    *slot = freq[c * n + b + k * blocks];
                }
                imdct_via_rotation(&spec, &mut time);
                // The low-overlap window: zero pad, rising taper, flat
                // middle, falling taper, zero pad. Only the nonzero
                // span [pad, 2*n2 - pad) contributes.
                let base = b * n2;
                for s in 0..n2 + block_overlap {
                    let t = pad + s;
                    let w = if s < block_overlap {
                        window[s]
                    } else if s < n2 {
                        1.0
                    } else {
                        window[n2 + block_overlap - 1 - s]
                    };
                    acc[base + s] += w * time[t];
                }
            }

            // Emit N samples, thread the trailing overlap forward.
            let om = &mut self.overlap_mem[c * overlap..(c + 1) * overlap];
            let dst = &mut planar[c * n..(c + 1) * n];
            dst.copy_from_slice(&acc[..n]);
            for (j, o) in om.iter_mut().enumerate() {
                dst[j] += *o;
                *o = acc[n + j];
            }
        }

        // §4.3.7.1 post-filter over [history | frame], in place.
        let period0 = self.pf_period_old.max(COMB_MIN_PERIOD);
        let period1 = self.pf_period.max(COMB_MIN_PERIOD);
        let short_len = 120.min(n);
        for c in 0..self.channels {
            let mut ext = Vec::with_capacity_in(hist_len + n, crate::bump::DecodeBump);
            ext.resize(hist_len + n, 0.0f64);
            ext[..hist_len].copy_from_slice(&self.hist[c * hist_len..(c + 1) * hist_len]);
            ext[hist_len..].copy_from_slice(&planar[c * n..(c + 1) * n]);

            comb_filter(
                &mut ext,
                hist_len,
                period0,
                period1,
                short_len,
                self.pf_gain_old,
                self.pf_gain,
                self.pf_tapset_old,
                self.pf_tapset,
                &window,
                overlap,
            );
            if n > short_len {
                comb_filter(
                    &mut ext,
                    hist_len + short_len,
                    period1,
                    pf_period_new.max(COMB_MIN_PERIOD),
                    n - short_len,
                    self.pf_gain,
                    pf_gain_new,
                    self.pf_tapset,
                    pf_tapset_new,
                    &window,
                    overlap,
                );
            }

            // Roll the history and copy the filtered frame back.
            planar[c * n..(c + 1) * n].copy_from_slice(&ext[hist_len..]);
            let keep = hist_len.min(n);
            let hist = &mut self.hist[c * hist_len..(c + 1) * hist_len];
            if keep < hist_len {
                hist.copy_within(keep.., 0);
            }
            hist[hist_len - keep..].copy_from_slice(&ext[hist_len + n - keep..]);
        }

        // Post-filter parameter roll (two-deep history; short frames
        // collapse both slots onto the new parameters).
        self.pf_period_old = self.pf_period;
        self.pf_gain_old = self.pf_gain;
        self.pf_tapset_old = self.pf_tapset;
        self.pf_period = pf_period_new;
        self.pf_gain = pf_gain_new;
        self.pf_tapset = pf_tapset_new;
        if n > short_len {
            self.pf_period_old = self.pf_period;
            self.pf_gain_old = self.pf_gain;
            self.pf_tapset_old = self.pf_tapset;
        }

        // §4.3.7.2 de-emphasis + interleaved i16 output.
        for c in 0..self.channels {
            let mut mem = self.deemph_mem[c];
            for j in 0..n {
                let tmp = planar[c * n + j] + mem;
                mem = DEEMPH_COEF * tmp;
                out[j * self.channels + c] = sig_to_i16(tmp);
            }
            self.deemph_mem[c] = mem;
        }
    }
}

/// §4.3.7.1 comb filter over `buf[base..base + n]`, in place, with
/// feedback taps reaching back through the (already filtered) earlier
/// samples. The first `overlap` samples crossfade from the
/// `(t0, g0, tapset0)` filter to `(t1, g1, tapset1)`; the remainder
/// runs the target filter alone.
#[allow(clippy::too_many_arguments)]
fn comb_filter(
    buf: &mut [f64],
    base: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f64,
    g1: f64,
    tapset0: usize,
    tapset1: usize,
    window: &[f64],
    overlap: usize,
) {
    let g00 = g0 * POST_FILTER_TAPS[tapset0][0];
    let g01 = g0 * POST_FILTER_TAPS[tapset0][1];
    let g02 = g0 * POST_FILTER_TAPS[tapset0][2];
    let g10 = g1 * POST_FILTER_TAPS[tapset1][0];
    let g11 = g1 * POST_FILTER_TAPS[tapset1][1];
    let g12 = g1 * POST_FILTER_TAPS[tapset1][2];
    let ov = overlap.min(n);
    for (i, &w) in window.iter().enumerate().take(ov) {
        let f = w * w;
        let p = base + i;
        let x0 = |d: isize| buf[(p as isize - t0 as isize + d) as usize];
        let x1 = |d: isize| buf[(p as isize - t1 as isize + d) as usize];
        buf[p] += (1.0 - f) * (g00 * x0(0) + g01 * (x0(-1) + x0(1)) + g02 * (x0(-2) + x0(2)))
            + f * (g10 * x1(0) + g11 * (x1(-1) + x1(1)) + g12 * (x1(-2) + x1(2)));
    }
    for i in ov..n {
        let p = base + i;
        let x1 = |d: isize| buf[(p as isize - t1 as isize + d) as usize];
        buf[p] += g10 * x1(0) + g11 * (x1(-1) + x1(1)) + g12 * (x1(-2) + x1(2));
    }
}

/// §4.3.7 inverse MDCT: `y(t) = Σ_k X(k) cos[(π/N)(t + 1/2 +
/// N/2)(k + 1/2)]` — the §4.3.7 "scaling by 1/2" combined with the
/// doubling the reference transform folds into the window mixing
/// nets to the plain unnormalized cosine sum, with no N-dependent
/// factor (the encoder's matching forward transform carries the 2/N;
/// validated to unit scale against the reference decode of the
/// fixture corpus). The kernel is evaluated with a complex rotation
/// recurrence per output sample instead of one `cos` call per
/// `(t, k)` pair (identical math, drift-free at these lengths in
/// `f64`).
fn imdct_via_rotation(spec: &[f64], out: &mut [f64]) {
    let n = spec.len();
    let nf = n as f64;
    let scale = 1.0;
    for (t, slot) in out.iter_mut().enumerate() {
        let base = (std::f64::consts::PI / nf) * (t as f64 + 0.5 + nf / 2.0);
        // phase(k) = base * (k + 1/2): start at base/2, step by base.
        let (step_c, step_s) = (base.cos(), base.sin());
        let (mut re, mut im) = ((base * 0.5).cos(), (base * 0.5).sin());
        let mut acc = 0.0f64;
        for &xk in spec {
            acc += xk * re;
            let nre = re * step_c - im * step_s;
            im = re * step_s + im * step_c;
            re = nre;
        }
        *slot = scale * acc;
    }
}

/// Sig-scale (±32768) float sample → clamped i16, round-to-nearest.
#[inline]
fn sig_to_i16(v: f64) -> i16 {
    v.round().clamp(-32768.0, 32767.0) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_imdct::imdct;

    #[test]
    fn rotation_imdct_matches_reference_kernel_shape() {
        // Same cosine kernel as celt_imdct::imdct (which scales by
        // 1/N); this synthesis uses the flat unit scale, so the two
        // agree up to the constant N.
        for n in [4usize, 60, 120] {
            let spec: Vec<f64> = (0..n).map(|k| ((k * 37 + 11) % 23) as f64 - 11.0).collect();
            let direct = imdct(&spec).unwrap();
            let mut fast = vec![0.0f64; 2 * n];
            imdct_via_rotation(&spec, &mut fast);
            let factor = n as f64;
            for (a, b) in direct.iter().zip(fast.iter()) {
                assert!((a * factor - b).abs() < 1e-7, "n={n}");
            }
        }
    }

    #[test]
    fn silent_frames_stay_silent_and_state_advances() {
        let mut s = CeltSynthesis::new(1, 480);
        let freq = vec![0.0f64; 480];
        let mut out = vec![0i16; 480];
        for _ in 0..3 {
            s.synthesize_frame(&freq, 1, None, &mut out);
            assert!(out.iter().all(|&v| v == 0));
        }
    }

    #[test]
    fn overlap_carries_energy_into_the_next_frame() {
        let mut s = CeltSynthesis::new(1, 480);
        let mut freq = vec![0.0f64; 480];
        freq[7] = 10000.0;
        let mut out = vec![0i16; 480];
        s.synthesize_frame(&freq, 1, None, &mut out);
        assert!(out.iter().any(|&v| v != 0), "tone frame audible");
        // A following silent frame still carries the overlap tail.
        let silent = vec![0.0f64; 480];
        let mut out2 = vec![0i16; 480];
        s.synthesize_frame(&silent, 1, None, &mut out2);
        assert!(
            out2.iter().take(CELT_OVERLAP_48K).any(|&v| v != 0),
            "overlap tail must bleed into the next frame"
        );
        // After the overlap + de-emphasis decay the rest is near
        // silence.
        assert!(out2[300..].iter().all(|&v| v.abs() < 3));
    }

    #[test]
    fn reset_clears_the_tail() {
        let mut s = CeltSynthesis::new(1, 480);
        let mut freq = vec![0.0f64; 480];
        freq[3] = 20000.0;
        let mut out = vec![0i16; 480];
        s.synthesize_frame(&freq, 1, None, &mut out);
        s.reset();
        let silent = vec![0.0f64; 480];
        s.synthesize_frame(&silent, 1, None, &mut out);
        assert!(out.iter().all(|&v| v == 0), "reset must clear all state");
    }

    #[test]
    fn short_blocks_and_long_blocks_conserve_energy_order() {
        // The same total spectral energy through 1 vs 8 blocks must
        // produce output of the same magnitude order (windowing is
        // power-complementary).
        let energy = |blocks: usize| -> f64 {
            let mut s = CeltSynthesis::new(1, 960);
            let mut freq = vec![0.0f64; 960];
            for (i, v) in freq.iter_mut().enumerate().take(64) {
                *v = if i % 3 == 0 { 5000.0 } else { -3000.0 };
            }
            let mut out = vec![0i16; 960];
            s.synthesize_frame(&freq, blocks, None, &mut out);
            // Second frame flushes the remaining overlap.
            let silent = vec![0.0f64; 960];
            let mut out2 = vec![0i16; 960];
            s.synthesize_frame(&silent, blocks, None, &mut out2);
            out.iter()
                .chain(out2.iter())
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
        };
        let e1 = energy(1);
        let e8 = energy(8);
        assert!(e1 > 0.0 && e8 > 0.0);
        let ratio = e1 / e8;
        assert!(
            (0.2..5.0).contains(&ratio),
            "block layouts should conserve energy order, ratio {ratio}"
        );
    }

    #[test]
    fn post_filter_changes_output_and_is_stable() {
        let mut s = CeltSynthesis::new(1, 480);
        let mut freq = vec![0.0f64; 480];
        freq[5] = 8000.0;
        let mut base_out = vec![0i16; 480];
        let mut s2 = s.clone();
        s.synthesize_frame(&freq, 1, None, &mut base_out);
        let mut pf_out = vec![0i16; 480];
        let pf = CeltPostFilterOut {
            period: 100,
            gain: 0.75,
            tapset: 0,
        };
        s2.synthesize_frame(&freq, 1, Some(pf), &mut pf_out);
        // First frame: the new filter only applies past sample 120,
        // where history is nonzero now.
        assert_ne!(base_out, pf_out, "post-filter must alter the signal");
        // Run several more frames: output must remain bounded (the
        // comb gains are < 1, the filter is stable).
        let mut out = vec![0i16; 480];
        for _ in 0..20 {
            s2.synthesize_frame(&freq, 1, Some(pf), &mut out);
        }
        assert!(out.iter().any(|&v| v != 0));
    }
}
