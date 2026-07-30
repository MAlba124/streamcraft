//! Packet-loss concealment (RFC 6716 §4.4).
//!
//! PLC is "an optional decoder-side feature that SHOULD be included
//! when receiving from an unreliable channel. Because PLC is not part
//! of the bitstream, there are many acceptable ways to implement PLC
//! with different complexity/quality trade-offs" (§4.4). The RFC's
//! guidance is per-mode:
//!
//! * **CELT mode** — "the PLC finds a periodicity in the decoded
//!   signal and repeats the windowed waveform using the pitch offset",
//!   overlapped with the previous and next frames.
//! * **SILK mode** — "the PLC uses LPC extrapolation from the previous
//!   frame".
//!
//! This module implements both flavors over the decoder's 48 kHz
//! output history (the concealment domain is an implementation choice
//! the RFC leaves free):
//!
//! * [`conceal_celt`] — pitch-periodic waveform continuation: a
//!   normalized-autocorrelation pitch search over the §4.3.7.1
//!   post-filter period range (`15..=1022` samples at 48 kHz), then a
//!   cyclic repeat of the final pitch period.
//! * [`conceal_silk`] — LPC extrapolation: a Burg fit
//!   ([`crate::silk_lpc_analysis::burg_lpc`]) over the recent history,
//!   bandwidth-expanded for guaranteed decay, excited by the
//!   pitch-cyclic LPC residual of the final period (the classic
//!   long-term + short-term predictor continuation).
//!
//! Both flavors apply the same **energy-decay envelope** across
//! consecutive losses ([`loss_gain`]): the first concealed frame plays
//! at (near) full energy and each further consecutive loss attenuates
//! multiplicatively until the output reaches the silence floor —
//! concealment must never turn a long loss burst into an infinite
//! tone. Each concealed frame also produces a short extrapolation
//! *tail* beyond the frame boundary; the decoder cross-laps that tail
//! with the head of the next successfully decoded frame using a
//! power-complementary window pair ([`cross_lap`]), the same smoothing
//! shape §4.5's transition machinery recommends for the
//! PLC-to-new-mode seams (Figure 19's `P &` cross-lap), so the
//! concealed-to-real join is as artifact-free as the loss join.

use crate::silk_lpc_analysis::{bandwidth_expand, burg_lpc, lpc_residual};

/// Samples of 48 kHz per-channel history the concealment state keeps
/// (100 ms — enough for the longest §4.3.7.1 pitch period plus the
/// correlation window plus the Burg fit window).
pub const PLC_HISTORY_SAMPLES: usize = 4800;

/// Per-channel samples of extrapolation tail produced past the
/// concealed frame boundary, cross-lapped with the next decoded frame
/// (2.5 ms — the §4.5.1.4 redundancy cross-lap length).
pub const PLC_CROSS_LAP_SAMPLES: usize = 120;

/// Minimum pitch lag searched, in 48 kHz samples (§4.3.7.1 post-filter
/// period floor).
pub const PITCH_MIN_LAG: usize = 15;

/// Maximum pitch lag searched, in 48 kHz samples (§4.3.7.1 post-filter
/// period ceiling).
pub const PITCH_MAX_LAG: usize = 1022;

/// Correlation window for the pitch search (20 ms).
const PITCH_CORR_WINDOW: usize = 960;

/// Burg-fit window for the SILK-flavor LPC extrapolation (30 ms).
const LPC_FIT_WINDOW: usize = 1440;

/// LPC order for the SILK-flavor extrapolation — the §4.2 WB SILK
/// filter order (and [`crate::silk_lpc_analysis::LPC_ANALYSIS_MAX_ORDER`],
/// the Burg fit's ceiling). The short-term spectral envelope is
/// coarser at the 48 kHz concealment rate than a 16 kHz-rate fit of
/// the same order, but the pitch-cyclic excitation carries the
/// harmonic structure, which dominates concealment quality.
const LPC_ORDER: usize = 16;

/// §4.2.7.5.7-style bandwidth expansion chirp applied to the
/// concealment LPC so the zero-input ring always decays.
const LPC_PLC_CHIRP: f64 = 0.99;

/// Multiplicative per-frame attenuation for the second and later
/// consecutive concealed frames (the §4.4 energy decay; the exact
/// schedule is a free implementation choice).
const LOSS_DECAY_PER_FRAME: f32 = 0.8;

/// Gain below which concealment output is treated as the silence floor.
const SILENCE_FLOOR_GAIN: f32 = 1e-3;

/// The §4.4 energy-decay gain applied to the `k`-th *consecutive*
/// concealed frame (`k` counts from 1). The first concealed frame
/// plays at full gain; each further loss multiplies by
/// [`LOSS_DECAY_PER_FRAME`].
#[must_use]
pub fn loss_gain(k: u32) -> f32 {
    if k <= 1 {
        1.0
    } else {
        LOSS_DECAY_PER_FRAME.powi(k as i32 - 1)
    }
}

/// Pitch estimate over `hist`: the lag in
/// [`PITCH_MIN_LAG`]..=[`PITCH_MAX_LAG`] maximizing the normalized
/// cross-correlation between the trailing [`PITCH_CORR_WINDOW`] (or as
/// much as the history allows) and the same window one lag earlier.
///
/// Returns `(lag, normalized_correlation)`, or `None` when the history
/// is too short to correlate even the minimum lag or is (near-)silent.
#[must_use]
pub fn find_pitch(hist: &[f32]) -> Option<(usize, f32)> {
    let max_lag = PITCH_MAX_LAG.min(hist.len().saturating_sub(8));
    if max_lag < PITCH_MIN_LAG {
        return None;
    }
    let mut best: Option<(usize, f32)> = None;
    for lag in PITCH_MIN_LAG..=max_lag {
        let w = PITCH_CORR_WINDOW.min(hist.len() - lag);
        let cur = &hist[hist.len() - w..];
        let prev = &hist[hist.len() - w - lag..hist.len() - lag];
        let mut xy = 0.0f64;
        let mut xx = 0.0f64;
        let mut yy = 0.0f64;
        for (a, b) in cur.iter().zip(prev.iter()) {
            xy += f64::from(*a) * f64::from(*b);
            xx += f64::from(*a) * f64::from(*a);
            yy += f64::from(*b) * f64::from(*b);
        }
        if xx <= 1e-12 || yy <= 1e-12 {
            continue;
        }
        let corr = (xy / (xx * yy).sqrt()) as f32;
        let better = match best {
            None => true,
            Some((_, c)) => corr > c,
        };
        if better {
            best = Some((lag, corr));
        }
    }
    best
}

/// CELT-flavor concealment (§4.4): pitch-periodic continuation of the
/// decoded waveform. Produces `n_out` samples continuing `hist`, with
/// the gain ramping linearly from `gain_start` to `gain_end` across
/// the output (the §4.4 energy decay for this frame).
///
/// The final pitch period of the history is repeated cyclically; the
/// pitch selection makes the period boundary approximately continuous
/// (`x[n] ≈ x[n - lag]` is exactly the property the correlation
/// search maximized). An empty / silent history conceals to silence.
#[must_use]
pub fn conceal_celt(hist: &[f32], n_out: usize, gain_start: f32, gain_end: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n_out];
    if gain_start <= SILENCE_FLOOR_GAIN && gain_end <= SILENCE_FLOOR_GAIN {
        return out;
    }
    let Some((lag, corr)) = find_pitch(hist) else {
        return out;
    };
    // A weakly periodic (noise-like) history still repeats its final
    // period — repeating noise sounds like noise — but the repetition
    // is softened toward the decay floor faster by scaling with the
    // achieved correlation when it is poor.
    let periodicity = corr.clamp(0.0, 1.0);
    let softing = if periodicity > 0.5 {
        1.0
    } else {
        0.5 + periodicity
    };
    let src = &hist[hist.len() - lag..];
    for (n, slot) in out.iter_mut().enumerate() {
        let t = n as f32 / n_out.max(1) as f32;
        let gain = (gain_start + (gain_end - gain_start) * t) * softing;
        *slot = src[n % lag] * gain;
    }
    out
}

/// SILK-flavor concealment (§4.4): LPC extrapolation from the previous
/// frame. A Burg LPC fit over the trailing [`LPC_FIT_WINDOW`] samples
/// (bandwidth-expanded so the recursion always decays) is driven by
/// the pitch-cyclic LPC residual of the final period, seeded with the
/// history tail — the long-term (pitch) + short-term (LPC) predictor
/// continuation the RFC describes. Produces `n_out` samples with the
/// same linear `gain_start → gain_end` decay ramp as
/// [`conceal_celt`]. Falls back to the pitch-periodic flavor when the
/// LPC fit is unavailable (short or degenerate history).
#[must_use]
pub fn conceal_silk(hist: &[f32], n_out: usize, gain_start: f32, gain_end: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n_out];
    if gain_start <= SILENCE_FLOOR_GAIN && gain_end <= SILENCE_FLOOR_GAIN {
        return out;
    }
    let Some((lag, _corr)) = find_pitch(hist) else {
        return out;
    };
    let fit_len = LPC_FIT_WINDOW.min(hist.len());
    if fit_len < LPC_ORDER * 4 || hist.len() < lag + LPC_ORDER {
        return conceal_celt(hist, n_out, gain_start, gain_end);
    }
    let fit: Vec<f64> = hist[hist.len() - fit_len..]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let Ok(mut a) = burg_lpc(&fit, LPC_ORDER) else {
        return conceal_celt(hist, n_out, gain_start, gain_end);
    };
    bandwidth_expand(&mut a, LPC_PLC_CHIRP);

    // LPC residual of the final pitch period: the excitation the
    // extrapolation recycles cyclically (long-term prediction with the
    // found lag).
    let seg: Vec<f64> = hist[hist.len() - lag..]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let res_hist: Vec<f64> = hist[hist.len() - lag - LPC_ORDER..hist.len() - lag]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let residual = lpc_residual(&seg, &res_hist, &a);

    // Synthesis seeded with the history tail: x[n] = e[n] + Σ a_k x[n-k].
    let mut tail: Vec<f64> = hist[hist.len() - LPC_ORDER..]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    for (n, slot) in out.iter_mut().enumerate() {
        let e = residual[n % lag];
        let mut x = e;
        for (k, &ak) in a.iter().enumerate() {
            x += ak * tail[tail.len() - 1 - k];
        }
        let t = n as f32 / n_out.max(1) as f32;
        let gain = gain_start + (gain_end - gain_start) * t;
        *slot = (x as f32) * gain;
        tail.push(x);
        if tail.len() > 2 * LPC_ORDER {
            tail.drain(..LPC_ORDER);
        }
    }
    out
}

/// Cross-lap `incoming[..n]` with `tail[..n]` using the
/// power-complementary `sin`/`cos` window pair: the concealment tail
/// fades out under `cos²`-power weighting while the newly decoded
/// audio fades in under the complementary `sin` — the §4.5.1.4-style
/// smooth seam. Writes the blend into `incoming` in place.
pub fn cross_lap(incoming: &mut [f32], tail: &[f32]) {
    let n = incoming.len().min(tail.len());
    if n == 0 {
        return;
    }
    for i in 0..n {
        let phase = core::f32::consts::FRAC_PI_2 * (i as f32 + 0.5) / n as f32;
        let w_in = phase.sin();
        let w_out = phase.cos();
        incoming[i] = incoming[i] * w_in * w_in + tail[i] * w_out * w_out;
    }
}

/// Concealment flavor, selected from the operating mode of the last
/// successfully decoded frame (§4.4 "depends on the mode of last
/// packet received").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlcFlavor {
    /// SILK-only / Hybrid predecessor: LPC extrapolation.
    Silk,
    /// CELT-only predecessor: pitch-periodic repetition.
    Celt,
}

/// Decoder-side concealment state: the per-channel 48 kHz output
/// history, the consecutive-loss counter driving the §4.4 energy
/// decay, and the pending cross-lap tail for the next decoded frame.
#[derive(Debug, Default)]
pub struct PlcState {
    /// Per-channel trailing output history (up to
    /// [`PLC_HISTORY_SAMPLES`] each), in the `i16 / 32768` float
    /// domain.
    hist: Vec<Vec<f32>>,
    /// Number of consecutive concealed frames since the last real
    /// decode (0 = the last frame was real).
    consecutive_losses: u32,
    /// Per-channel extrapolation tail to cross-lap into the head of
    /// the next decoded frame.
    tail: Option<Vec<Vec<f32>>>,
}

impl PlcState {
    /// Fresh state (no history, no pending tail).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all state (decoder reset).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Number of consecutive concealed frames since the last real
    /// decode.
    #[must_use]
    pub fn consecutive_losses(&self) -> u32 {
        self.consecutive_losses
    }

    /// Whether a concealment tail is pending for the next decoded
    /// frame.
    #[must_use]
    pub fn has_pending_tail(&self) -> bool {
        self.tail.is_some()
    }

    /// Record one packet's real decoded output (interleaved i16,
    /// `channels`-way) into the history and clear the loss counter.
    /// A channel-count change drops the old history.
    pub fn feed_decoded(&mut self, pcm: &[i16], channels: usize) {
        self.ensure_channels(channels);
        // Profluens patch: per-channel sample count this frame appends to the
        // *persistent* PLC history. `step_by`/`skip` iterators give `extend` a
        // loose lower size hint, so it reallocs in chunks every packet even
        // once len is capped by `drain`. Reserving the exact final capacity
        // once (a persistent, one-time alloc — the buffer stays global, Rule 2)
        // removes the steady-state realloc without changing a single sample.
        let per_channel = if channels == 0 { 0 } else { pcm.len().div_ceil(channels) };
        for (c, hist) in self.hist.iter_mut().enumerate() {
            let target_cap = (hist.len() + per_channel).max(PLC_HISTORY_SAMPLES);
            if hist.capacity() < target_cap {
                hist.reserve(target_cap - hist.len());
            }
            hist.extend(
                pcm.iter()
                    .skip(c)
                    .step_by(channels)
                    .map(|&s| f32::from(s) / 32768.0),
            );
            if hist.len() > PLC_HISTORY_SAMPLES {
                hist.drain(..hist.len() - PLC_HISTORY_SAMPLES);
            }
        }
        self.consecutive_losses = 0;
    }

    /// Cross-lap the pending concealment tail (if any) into the head
    /// of a newly decoded interleaved buffer. Call before
    /// [`Self::feed_decoded`]. A channel-count mismatch drops the tail
    /// instead (the §4.2.7.1-style transition already resets the
    /// layers).
    pub fn apply_tail(&mut self, pcm: &mut [i16], channels: usize) {
        let Some(tails) = self.tail.take() else {
            return;
        };
        if tails.len() != channels || channels == 0 {
            return;
        }
        let per_channel = pcm.len() / channels;
        for (c, tail) in tails.iter().enumerate() {
            let n = tail.len().min(per_channel);
            let mut head: Vec<f32> = (0..n)
                .map(|i| f32::from(pcm[i * channels + c]) / 32768.0)
                .collect();
            cross_lap(&mut head, tail);
            for (i, &v) in head.iter().enumerate() {
                pcm[i * channels + c] = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            }
        }
    }

    /// Conceal one lost frame of `per_channel` samples (§4.4),
    /// returning interleaved i16 PCM. `flavor` selects the per-mode
    /// algorithm; the §4.4 energy decay is driven by the
    /// consecutive-loss counter, which this call increments. The
    /// extrapolation continues [`PLC_CROSS_LAP_SAMPLES`] past the
    /// frame boundary; the excess is stored as the cross-lap tail for
    /// the next real frame.
    pub fn conceal(&mut self, per_channel: usize, channels: usize, flavor: PlcFlavor) -> Vec<i16> {
        self.ensure_channels(channels);
        self.consecutive_losses += 1;
        let k = self.consecutive_losses;
        let gain_start = loss_gain(k);
        let gain_end = loss_gain(k + 1);

        let total = per_channel + PLC_CROSS_LAP_SAMPLES;
        let mut pcm = vec![0i16; per_channel * channels];
        let mut tails: Vec<Vec<f32>> = Vec::with_capacity(channels);
        for c in 0..channels {
            let hist = &self.hist[c];
            let ext = match flavor {
                PlcFlavor::Silk => conceal_silk(hist, total, gain_start, gain_end),
                PlcFlavor::Celt => conceal_celt(hist, total, gain_start, gain_end),
            };
            for (i, &v) in ext[..per_channel].iter().enumerate() {
                pcm[i * channels + c] = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            }
            tails.push(ext[per_channel..].to_vec());
            let hist = &mut self.hist[c];
            hist.extend_from_slice(&ext[..per_channel]);
            if hist.len() > PLC_HISTORY_SAMPLES {
                hist.drain(..hist.len() - PLC_HISTORY_SAMPLES);
            }
        }
        self.tail = Some(tails);
        pcm
    }

    fn ensure_channels(&mut self, channels: usize) {
        if self.hist.len() != channels {
            self.hist = vec![Vec::new(); channels];
            self.tail = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(n: usize, period: f32, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (core::f32::consts::TAU * i as f32 / period).sin())
            .collect()
    }

    #[test]
    fn pitch_search_finds_planted_period() {
        // 160-sample period (300 Hz at 48 kHz) — well inside the range.
        let hist = sine(4800, 160.0, 0.5);
        let (lag, corr) = find_pitch(&hist).expect("pitch found");
        // The best lag is the fundamental or a multiple of it.
        assert_eq!(lag % 160, 0, "lag {lag} is not a multiple of 160");
        assert!(corr > 0.99, "corr {corr}");
    }

    #[test]
    fn pitch_search_none_on_short_or_silent_history() {
        assert!(find_pitch(&[]).is_none());
        assert!(find_pitch(&[0.0; 10]).is_none());
        // Long but silent: correlation undefined everywhere.
        assert!(find_pitch(&vec![0.0; 4800]).is_none());
    }

    #[test]
    fn celt_flavor_continues_a_periodic_signal() {
        let hist = sine(4800, 160.0, 0.5);
        let out = conceal_celt(&hist, 960, 1.0, 1.0);
        // The continuation must track the mathematical extension of the
        // sine: compare against the true next 960 samples.
        let truth: Vec<f32> = (0..960)
            .map(|i| 0.5 * (core::f32::consts::TAU * (4800 + i) as f32 / 160.0).sin())
            .collect();
        let mut err = 0.0f64;
        let mut sig = 0.0f64;
        for (o, t) in out.iter().zip(truth.iter()) {
            err += f64::from(o - t) * f64::from(o - t);
            sig += f64::from(*t) * f64::from(*t);
        }
        let snr = 10.0 * (sig / err.max(1e-30)).log10();
        assert!(snr > 20.0, "continuation SNR {snr:.1} dB");
    }

    #[test]
    fn silk_flavor_continues_a_periodic_signal() {
        let hist = sine(4800, 160.0, 0.5);
        let out = conceal_silk(&hist, 960, 1.0, 1.0);
        let truth: Vec<f32> = (0..960)
            .map(|i| 0.5 * (core::f32::consts::TAU * (4800 + i) as f32 / 160.0).sin())
            .collect();
        let mut err = 0.0f64;
        let mut sig = 0.0f64;
        for (o, t) in out.iter().zip(truth.iter()) {
            err += f64::from(o - t) * f64::from(o - t);
            sig += f64::from(*t) * f64::from(*t);
        }
        let snr = 10.0 * (sig / err.max(1e-30)).log10();
        assert!(snr > 20.0, "LPC continuation SNR {snr:.1} dB");
    }

    #[test]
    fn silk_flavor_join_is_continuous() {
        let hist = sine(4800, 160.0, 0.5);
        let out = conceal_silk(&hist, 960, 1.0, 1.0);
        // The first concealed sample continues the history: the jump at
        // the join must be comparable to the intra-signal sample delta
        // (the sine's max step is amp * 2π/period ≈ 0.0196).
        let jump = (out[0] - hist[hist.len() - 1]).abs();
        assert!(jump < 0.05, "join jump {jump}");
    }

    #[test]
    fn decay_envelope_is_monotone_and_reaches_floor() {
        let mut prev = f32::INFINITY;
        for k in 1..=40 {
            let g = loss_gain(k);
            assert!(g <= prev, "gain not monotone at k={k}");
            prev = g;
        }
        assert!(loss_gain(1) == 1.0);
        assert!(loss_gain(40) < SILENCE_FLOOR_GAIN);
    }

    #[test]
    fn consecutive_conceals_decay_to_silence() {
        let mut plc = PlcState::new();
        let hist = sine(4800, 160.0, 0.5);
        let pcm_i16: Vec<i16> = hist.iter().map(|&v| (v * 32767.0) as i16).collect();
        plc.feed_decoded(&pcm_i16, 1);
        let mut energies = Vec::new();
        for _ in 0..30 {
            let out = plc.conceal(960, 1, PlcFlavor::Celt);
            let e: f64 = out.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
            energies.push(e);
        }
        // Decay: later frames carry (weakly) less energy, and the final
        // frames are silence-floor quiet.
        assert!(energies[0] > 0.0, "first concealed frame is silent");
        for w in energies.windows(2).skip(1) {
            assert!(
                w[1] <= w[0] * 1.05,
                "energy grew across consecutive losses: {energies:?}"
            );
        }
        let first = energies[0];
        let last = energies[energies.len() - 1];
        assert!(
            last < first * 1e-4,
            "no decay to silence floor: first {first}, last {last}"
        );
    }

    #[test]
    fn cross_lap_blends_endpoints() {
        let mut incoming = vec![1.0f32; 120];
        let tail = vec![-1.0f32; 120];
        cross_lap(&mut incoming, &tail);
        // Start ≈ tail, end ≈ incoming, monotone in between.
        assert!(incoming[0] < -0.9);
        assert!(incoming[119] > 0.9);
        for w in incoming.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn apply_tail_smooths_the_next_frame_head() {
        let mut plc = PlcState::new();
        let hist = sine(4800, 160.0, 0.5);
        let pcm_i16: Vec<i16> = hist.iter().map(|&v| (v * 32767.0) as i16).collect();
        plc.feed_decoded(&pcm_i16, 1);
        let _ = plc.conceal(960, 1, PlcFlavor::Silk);
        assert!(plc.has_pending_tail());
        // Next real frame continues the sine — the blend must leave it
        // essentially intact (tail ≈ the same extension).
        let mut next: Vec<i16> = (0..960)
            .map(|i| {
                let v = 0.5 * (core::f32::consts::TAU * (5760 + i) as f32 / 160.0).sin();
                (v * 32767.0) as i16
            })
            .collect();
        let orig = next.clone();
        plc.apply_tail(&mut next, 1);
        assert!(!plc.has_pending_tail());
        // Only the first PLC_CROSS_LAP_SAMPLES may differ, and modestly.
        for i in PLC_CROSS_LAP_SAMPLES..960 {
            assert_eq!(next[i], orig[i], "sample {i} beyond the lap changed");
        }
        let max_dev = (0..PLC_CROSS_LAP_SAMPLES)
            .map(|i| (i32::from(next[i]) - i32::from(orig[i])).abs())
            .max()
            .unwrap();
        assert!(
            max_dev < 3300,
            "cross-lap deviated {max_dev} (> 10% full scale) on a continuous signal"
        );
    }

    #[test]
    fn channel_count_change_drops_history_and_tail() {
        let mut plc = PlcState::new();
        plc.feed_decoded(&[100i16; 960], 1);
        let _ = plc.conceal(960, 1, PlcFlavor::Celt);
        assert!(plc.has_pending_tail());
        // Stereo feed drops the mono tail and history.
        let mut stereo = vec![50i16; 1920];
        plc.apply_tail(&mut stereo, 2); // mismatch → tail dropped, pcm intact
        assert!(stereo.iter().all(|&s| s == 50));
        plc.feed_decoded(&stereo, 2);
        assert_eq!(plc.consecutive_losses(), 0);
    }

    #[test]
    fn conceal_with_no_history_is_silence() {
        let mut plc = PlcState::new();
        let out = plc.conceal(960, 1, PlcFlavor::Silk);
        assert!(out.iter().all(|&s| s == 0));
        let out = plc.conceal(960, 2, PlcFlavor::Celt);
        assert!(out.iter().all(|&s| s == 0));
    }

    #[test]
    fn stereo_conceal_keeps_channels_independent() {
        let mut plc = PlcState::new();
        // L = 160-period sine, R = silence.
        let mut inter = vec![0i16; 9600];
        for i in 0..4800 {
            let v = 0.5 * (core::f32::consts::TAU * i as f32 / 160.0).sin();
            inter[2 * i] = (v * 32767.0) as i16;
        }
        plc.feed_decoded(&inter, 2);
        let out = plc.conceal(960, 2, PlcFlavor::Celt);
        let l_energy: f64 = out
            .iter()
            .step_by(2)
            .map(|&s| f64::from(s) * f64::from(s))
            .sum();
        let r_energy: f64 = out
            .iter()
            .skip(1)
            .step_by(2)
            .map(|&s| f64::from(s) * f64::from(s))
            .sum();
        assert!(l_energy > 1e6, "left concealment silent");
        assert!(r_energy < 1e-6, "silent right channel grew energy");
    }
}
