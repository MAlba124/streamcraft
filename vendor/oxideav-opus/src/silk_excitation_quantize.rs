//! Closed-loop excitation quantisation for the SILK encoder — the
//! §5.2.3.8 noise-shaping-quantizer role, RFC 6716.
//!
//! Given the frame's already-quantized parameters (gains, LPC, LTP)
//! and the target signal at the internal rate, choose the §4.2.7.8
//! excitation `e_raw[]` so that the DECODER's reconstruction tracks
//! the target. §5.2.3.8's spectral noise shaping and delayed-decision
//! search are encoder freedoms this implementation trades away; what
//! it keeps is the essential closed-loop structure: each sample's
//! pulse is chosen against the prediction the decoder will actually
//! form from the previously chosen (quantized) samples, so
//! quantisation error does not accumulate through the synthesis
//! recursion.
//!
//! Per subframe:
//!
//!  1. Probe the real §4.2.7.9.1 LTP synthesis with an all-zero
//!     excitation. Because the voiced residual recursion is linear in
//!     the excitation, the probe yields the "zero-input response"
//!     `res_zero[i]` (rewhitened lookback + LTP ringing), and the
//!     actual residual is `res[i] = res_zero[i] + D[i]` with
//!     `D[i] = e_Q23[i]/2^23 + sum_k b_Q7[k]/128 * D[i-lag+2-k]`
//!     (`D = 0` in the lookback region) — no duplicate rewhitening.
//!  2. Walk the subframe sample by sample: form the decoder's LPC
//!     prediction from the (quantized) local history seeded by the
//!     real [`LpcSynthState`], derive the excitation value that would
//!     land the output on the target, and round it to the nearest
//!     representable `e_raw` — accounting for the §4.2.7.8.6 LCG sign
//!     inversion, whose flip decision is a deterministic function of
//!     the seed state at that sample.
//!  3. Feed the chosen excitation through the real
//!     [`ltp_synthesis_subframe`] / [`lpc_synthesis_subframe`] /
//!     [`ltp_synth_commit_subframe`] chain so the carried decoder
//!     state is authoritative (bit-identical to a decoder walking the
//!     produced bitstream).
//!
//! Magnitudes are capped at 1024 per sample, which guarantees every
//! 16-sample shell block satisfies the Table 46 pre-LSB pulse cap
//! with at most 10 extra LSBs (`16 * 1024 >> 10 = 16`); the per-block
//! LSB counts are derived afterwards as the smallest depth that fits,
//! and the §4.2.7.8.1 rate level is chosen by scratch-encoding the
//! excitation at all nine levels and keeping the smallest.
//!
//! All truth is taken from RFC 6716 §4.2.7.8 / §4.2.7.9 / §5.2.3.8.
//! No external library source is consulted.

use crate::range_encoder::RangeEncoder;
use crate::silk_excitation::{
    quantization_offset_q23, shell_block_count, Excitation, ExcitationConfig, ExcitationSymbols,
    SilkFrameSize, SHELL_BLOCK_SAMPLES,
};
use crate::silk_frame::{QuantizationOffsetType, SignalType};
use crate::silk_lpc_synth::{lpc_synthesis_subframe, subframe_samples, LpcSynthState};
use crate::silk_ltp::{LTP_FILTER_TAPS, LTP_MAX_SUBFRAMES};
use crate::silk_ltp_synth::{
    ltp_synth_commit_subframe, ltp_synthesis_subframe, LtpSynthState, LtpSynthSubframe,
};
use crate::toc::Bandwidth;
use crate::Error;

/// Per-sample excitation magnitude cap: keeps every shell block
/// encodable (`16 * 1024 >> 10 = 16` pre-LSB pulses max).
pub const MAX_PULSE_MAGNITUDE: i32 = 1024;

/// The voiced-frame LTP parameters the quantiser needs (all already
/// quantized to decoder-identical values).
#[derive(Debug, Clone, Copy)]
pub struct LtpFrameParams {
    /// Decoded per-subframe pitch lags.
    pub pitch_lags: [i32; LTP_MAX_SUBFRAMES],
    /// Decoder-identical Q7 taps per subframe.
    pub taps_q7: [[i8; LTP_FILTER_TAPS]; LTP_MAX_SUBFRAMES],
    /// §4.2.7.6.3 Q14 LTP scaling in force for this frame.
    pub ltp_scaling_q14: u16,
}

/// Result of quantizing one SILK frame's excitation.
#[derive(Debug, Clone, PartialEq)]
pub struct ExcitationQuantized {
    /// Signed excitation values, one per coded sample (padded with
    /// zeros to `16 * shell_blocks` — the 10 ms MB case codes 128
    /// samples for a 120-sample frame).
    pub e_raw: Vec<i32>,
    /// Per-shell-block extra-LSB counts (smallest depth that fits the
    /// Table 46 pulse cap).
    pub lsb_counts: Vec<u8>,
    /// §4.2.7.8.1 rate level chosen by measured size.
    pub rate_level: u8,
    /// The reconstruction the decoder will produce for this frame
    /// (from the real §4.2.7.9 synthesis chain).
    pub reconstructed: Vec<f32>,
}

/// Quantize one SILK frame's excitation closed-loop against `target`.
///
/// * `gains_q16` — per-subframe DEQUANTIZED §4.2.7.4 gains (from
///   [`crate::silk_gains::SubframeGains::dequant_q16`]).
/// * `a_q12` — the §4.2.7.5.8 stabilised Q12 predictor used for every
///   subframe (this encoder always signals `w_Q2 = 4`, so there is no
///   first-half filter).
/// * `ltp` — `Some` iff `signal_type == Voiced`.
/// * `target` — the frame's target signal at the internal rate
///   (`num_subframes * subframe_samples(bandwidth)` samples).
/// * `ltp_state` / `lpc_state` — the carried §4.2.7.9 histories;
///   updated through the real synthesis functions so they end the
///   frame exactly as a decoder's would.
#[allow(clippy::too_many_arguments)]
pub fn quantize_excitation_frame(
    bandwidth: Bandwidth,
    frame_size: SilkFrameSize,
    signal_type: SignalType,
    qoff_type: QuantizationOffsetType,
    lcg_seed: u8,
    gains_q16: &[u32],
    a_q12: &[i32],
    ltp: Option<&LtpFrameParams>,
    target: &[f32],
    ltp_state: &mut LtpSynthState,
    lpc_state: &mut LpcSynthState,
) -> Result<ExcitationQuantized, Error> {
    let n = subframe_samples(bandwidth)?;
    let num_subframes = match frame_size {
        SilkFrameSize::TenMs => 2usize,
        SilkFrameSize::TwentyMs => 4usize,
    };
    let frame_len = n * num_subframes;
    if lcg_seed > 3
        || gains_q16.len() != num_subframes
        || target.len() != frame_len
        || (signal_type == SignalType::Voiced) != ltp.is_some()
    {
        return Err(Error::MalformedPacket);
    }
    let d_lpc = lpc_state.d_lpc();
    if a_q12.len() != d_lpc {
        return Err(Error::MalformedPacket);
    }
    let a_i16: Vec<i16> = a_q12
        .iter()
        .map(|&c| c.clamp(i16::MIN as i32, i16::MAX as i32) as i16)
        .collect();
    let a_f: Vec<f32> = a_i16.iter().map(|&c| c as f32 / 4096.0).collect();

    let offset_q23 = quantization_offset_q23(signal_type, qoff_type);
    let mut seed: u32 = lcg_seed as u32;
    let mut e_raw_all: Vec<i32> = Vec::with_capacity(frame_len);
    let mut e_q23_all: Vec<i32> = Vec::with_capacity(frame_len);
    let mut reconstructed = vec![0.0f32; frame_len];

    ltp_state.start_frame();

    for s in 0..num_subframes {
        let gain_q16 = gains_q16[s];
        let gain_f = gain_q16 as f32 / 65536.0;
        let inv_gain = 65536.0 / gain_q16 as f32;
        let (pitch_lag, b_q7) = match ltp {
            Some(p) => (p.pitch_lags[s], p.taps_q7[s]),
            None => (1i32, [0i8; LTP_FILTER_TAPS]),
        };
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type,
            frame_size,
            subframe_index: s as u8,
            gain_q16,
            pitch_lag,
            b_q7,
            ltp_scaling_q14: ltp.map(|p| p.ltp_scaling_q14).unwrap_or(0),
            a_q12: &a_i16,
            lsf_interp_used: false,
        };

        // Step 1: zero-input response of the LTP stage (res_zero).
        let zeros = vec![0i32; n];
        let mut res_zero = vec![0.0f32; n];
        ltp_synthesis_subframe(ltp_state, cfg, &zeros, &mut res_zero)?;

        // Step 2: per-sample closed-loop pulse decisions.
        let b_f: [f32; LTP_FILTER_TAPS] = core::array::from_fn(|k| b_q7[k] as f32 / 128.0);
        let voiced = signal_type == SignalType::Voiced;
        // Local unclamped-lpc history: seeded from the carried state
        // (oldest first), extended as we produce samples.
        let mut lpc_local: Vec<f32> = lpc_state.history().to_vec();
        let hist_len = lpc_local.len();
        // Excitation-delta history for the LTP recursion (local
        // subframe coordinates; lookback is zero by definition).
        let mut delta = vec![0.0f32; n];
        let mut e_sub = vec![0i32; n];

        for i in 0..n {
            // LTP part: res_base = res_zero + delta ringing.
            let mut res_base = res_zero[i];
            if voiced {
                for (k, &bf) in b_f.iter().enumerate() {
                    let src = i as i32 - pitch_lag + 2 - k as i32;
                    if src >= 0 {
                        res_base += delta[src as usize] * bf;
                    }
                }
            }
            // LPC prediction from the quantized local history.
            let mut lpc_pred = 0.0f32;
            for (k, &af) in a_f.iter().enumerate() {
                // lpc[i - k - 1]: local index hist_len + i - k - 1.
                let idx = hist_len + i - k - 1;
                lpc_pred += lpc_local[idx] * af;
            }

            // Desired residual and excitation target (Q23).
            let desired_res = (target[s * n + i] - lpc_pred) * inv_gain;
            let e_target_q23 = (desired_res - res_base) * 8_388_608.0;

            // §4.2.7.8.6 LCG: the flip decision precedes the e_raw
            // choice; the seed update after depends on it.
            seed = seed.wrapping_mul(196_314_165).wrapping_add(907_633_515);
            let flip = (seed & 0x8000_0000) != 0;
            let want = if flip { -e_target_q23 } else { e_target_q23 };
            let e_raw = choose_pulse(want, offset_q23);
            seed = seed.wrapping_add(e_raw as u32);

            // Decoder-identical reconstruction of this sample's e_Q23.
            let sign_e = match e_raw.cmp(&0) {
                core::cmp::Ordering::Less => -1,
                core::cmp::Ordering::Greater => 1,
                core::cmp::Ordering::Equal => 0,
            };
            let mut e_q23 = (e_raw << 8) - sign_e * 20 + offset_q23;
            if flip {
                e_q23 = -e_q23;
            }

            // Advance the local decoder mirror.
            let e_f = e_q23 as f32 / 8_388_608.0;
            delta[i] = e_f + (res_base - res_zero[i]);
            let res_i = res_base + e_f;
            let lpc_i = gain_f * res_i + lpc_pred;
            lpc_local.push(lpc_i);
            e_sub[i] = e_raw;
            e_q23_all.push(e_q23);
        }

        // Step 3: authoritative state update through the real chain.
        let mut res_actual = vec![0.0f32; n];
        ltp_synthesis_subframe(
            ltp_state,
            cfg,
            &e_q23_all[s * n..(s + 1) * n],
            &mut res_actual,
        )?;
        let mut out_sub = vec![0.0f32; n];
        let lpc_unclamped = lpc_synthesis_subframe(
            bandwidth,
            lpc_state,
            &res_actual,
            gain_q16,
            &a_i16,
            &mut out_sub,
        )?;
        ltp_synth_commit_subframe(ltp_state, &out_sub, &lpc_unclamped)?;
        reconstructed[s * n..(s + 1) * n].copy_from_slice(&out_sub);
        e_raw_all.extend_from_slice(&e_sub);
    }

    // Pad to whole shell blocks (10 ms MB codes 128 samples for a
    // 120-sample frame; the §4.2.7.8 tail is discarded by synthesis).
    let shell_blocks = shell_block_count(bandwidth, frame_size)?;
    e_raw_all.resize(shell_blocks * SHELL_BLOCK_SAMPLES, 0);

    // Per-block LSB depth: smallest L with sum(mag >> L) <= 16.
    let mut lsb_counts = vec![0u8; shell_blocks];
    for (block, slot) in lsb_counts.iter_mut().enumerate() {
        let base = block * SHELL_BLOCK_SAMPLES;
        let mut l = 0u8;
        loop {
            let sum: u32 = e_raw_all[base..base + SHELL_BLOCK_SAMPLES]
                .iter()
                .map(|&v| v.unsigned_abs() >> l)
                .sum();
            if sum <= 16 {
                break;
            }
            l += 1;
            if l > 10 {
                // Unreachable with the magnitude cap; defend anyway.
                return Err(Error::MalformedPacket);
            }
        }
        *slot = l;
    }

    // Rate level by measured size over the nine §4.2.7.8.1 choices.
    let ex_cfg = ExcitationConfig {
        bandwidth,
        frame_size,
        signal_type,
        qoff_type,
        lcg_seed,
    };
    let mut best = (0u8, usize::MAX);
    for rl in 0..=8u8 {
        let mut re = RangeEncoder::new();
        let symbols = ExcitationSymbols {
            rate_level: rl,
            lsb_counts: &lsb_counts,
            e_raw: &e_raw_all,
        };
        Excitation::encode(&mut re, ex_cfg, &symbols)?;
        let len = re.finish().len();
        if len < best.1 {
            best = (rl, len);
        }
    }

    Ok(ExcitationQuantized {
        e_raw: e_raw_all,
        lsb_counts,
        rate_level: best.0,
        reconstructed,
    })
}

/// Pick the `e_raw` whose §4.2.7.8.6 pre-flip reconstruction
/// `(e_raw << 8) - sign(e_raw)*20 + offset` lands closest to `want`
/// (Q23). The reconstruction is monotone in `e_raw`, so the floor and
/// ceiling of the linear estimate, plus zero, cover the optimum.
fn choose_pulse(want: f32, offset_q23: i32) -> i32 {
    let est = (want - offset_q23 as f32) / 256.0;
    let lo = est.floor() as i32;
    let mut best = (0i32, recon_err(0, want, offset_q23));
    for cand in [lo, lo + 1] {
        let c = cand.clamp(-MAX_PULSE_MAGNITUDE, MAX_PULSE_MAGNITUDE);
        let err = recon_err(c, want, offset_q23);
        if err < best.1 {
            best = (c, err);
        }
    }
    best.0
}

#[inline]
fn recon_err(e_raw: i32, want: f32, offset_q23: i32) -> f32 {
    let sign_e = match e_raw.cmp(&0) {
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Greater => 1,
        core::cmp::Ordering::Equal => 0,
    };
    let r = (e_raw << 8) - sign_e * 20 + offset_q23;
    (r as f32 - want).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;
    use crate::silk_lsf_recon::cb1_q8;
    use crate::silk_lsf_to_lpc::LpcQ17;
    use crate::silk_ltp::ltp_filter_taps_q7;

    fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
        let mut sig = 0.0f64;
        let mut err = 0.0f64;
        for (&r, &t) in reference.iter().zip(test.iter()) {
            sig += (r as f64) * (r as f64);
            err += ((r - t) as f64) * ((r - t) as f64);
        }
        if err == 0.0 {
            return 120.0;
        }
        10.0 * (sig / err).log10()
    }

    fn wb_codebook_lpc(i1: u8) -> Vec<i32> {
        let cb = cb1_q8(Bandwidth::Wb, i1).unwrap();
        let nlsf: Vec<i16> = cb.iter().map(|&v| (v as i16) << 7).collect();
        LpcQ17::from_nlsf(Bandwidth::Wb, &nlsf)
            .unwrap()
            .range_limited()
            .prediction_gain_limited()
            .a_q12()
            .to_vec()
    }

    /// Unvoiced closed-loop tracking: a lowpass-ish target through a
    /// real codebook LPC filter reconstructs with a healthy SNR, and
    /// the produced e_raw survives the real §4.2.7.8 wire encode with
    /// a decoder-identical e_Q23.
    #[test]
    fn unvoiced_tracks_target_and_roundtrips_wire() {
        let bw = Bandwidth::Wb;
        let n = subframe_samples(bw).unwrap();
        let a_q12 = wb_codebook_lpc(7);
        let mut ltp_state = LtpSynthState::new(bw).unwrap();
        let mut lpc_state = LpcSynthState::new(bw).unwrap();

        // Smooth deterministic target, |x| < 0.2.
        let target: Vec<f32> = (0..4 * n)
            .map(|i| {
                let t = i as f32;
                0.12 * (t * 0.05).sin() + 0.06 * (t * 0.013).sin()
            })
            .collect();
        // Gain sized so pulses land in the ±2..3 range.
        let gains = [40_000_000u32; 4];

        let q = quantize_excitation_frame(
            bw,
            SilkFrameSize::TwentyMs,
            SignalType::Unvoiced,
            QuantizationOffsetType::Low,
            0,
            &gains,
            &a_q12,
            None,
            &target,
            &mut ltp_state,
            &mut lpc_state,
        )
        .unwrap();

        let snr = snr_db(&target, &q.reconstructed);
        assert!(snr > 12.0, "unvoiced SNR too low: {snr} dB");

        // Wire roundtrip: encode + decode the excitation and compare
        // e_Q23 with a from-scratch synthesis over the decoded values.
        let ex_cfg = ExcitationConfig {
            bandwidth: bw,
            frame_size: SilkFrameSize::TwentyMs,
            signal_type: SignalType::Unvoiced,
            qoff_type: QuantizationOffsetType::Low,
            lcg_seed: 0,
        };
        let mut re = RangeEncoder::new();
        let symbols = ExcitationSymbols {
            rate_level: q.rate_level,
            lsb_counts: &q.lsb_counts,
            e_raw: &q.e_raw,
        };
        let enc = Excitation::encode(&mut re, ex_cfg, &symbols).unwrap();
        let bytes = re.finish();
        let mut rd = RangeDecoder::new(&bytes);
        let dec = Excitation::decode(&mut rd, ex_cfg).unwrap();
        assert_eq!(enc.e_q23(), dec.e_q23());

        // Re-synthesize from the decoded excitation on fresh states:
        // must match the quantiser's predicted reconstruction.
        let mut ltp2 = LtpSynthState::new(bw).unwrap();
        let mut lpc2 = LpcSynthState::new(bw).unwrap();
        let a_i16: Vec<i16> = a_q12.iter().map(|&c| c as i16).collect();
        ltp2.start_frame();
        let mut out_all = Vec::new();
        for (s, &gain) in gains.iter().enumerate() {
            let cfg = LtpSynthSubframe {
                bandwidth: bw,
                signal_type: SignalType::Unvoiced,
                frame_size: SilkFrameSize::TwentyMs,
                subframe_index: s as u8,
                gain_q16: gain,
                pitch_lag: 1,
                b_q7: [0; LTP_FILTER_TAPS],
                ltp_scaling_q14: 0,
                a_q12: &a_i16,
                lsf_interp_used: false,
            };
            let mut res = vec![0.0f32; n];
            ltp_synthesis_subframe(&ltp2, cfg, &dec.e_q23()[s * n..(s + 1) * n], &mut res).unwrap();
            let mut out = vec![0.0f32; n];
            let lpc_unclamped =
                lpc_synthesis_subframe(bw, &mut lpc2, &res, gain, &a_i16, &mut out).unwrap();
            ltp_synth_commit_subframe(&mut ltp2, &out, &lpc_unclamped).unwrap();
            out_all.extend_from_slice(&out);
        }
        for (a, b) in out_all.iter().zip(q.reconstructed.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    /// Voiced closed-loop: a periodic target with real LTP taps
    /// reconstructs with a healthy SNR, across all four LCG seeds
    /// (the flip compensation must be seed-agnostic).
    #[test]
    fn voiced_tracks_target_for_all_seeds() {
        let bw = Bandwidth::Wb;
        let n = subframe_samples(bw).unwrap();
        let a_q12 = wb_codebook_lpc(12);
        let taps = ltp_filter_taps_q7(1, 3).unwrap();
        let ltp = LtpFrameParams {
            pitch_lags: [80; LTP_MAX_SUBFRAMES],
            taps_q7: [taps; LTP_MAX_SUBFRAMES],
            ltp_scaling_q14: 15565,
        };
        let target: Vec<f32> = (0..4 * n)
            .map(|i| {
                let t = i as f32;
                0.25 * (t * core::f32::consts::TAU / 80.0).sin()
            })
            .collect();
        let gains = [30_000_000u32; 4];

        for seed in 0..=3u8 {
            let mut ltp_state = LtpSynthState::new(bw).unwrap();
            let mut lpc_state = LpcSynthState::new(bw).unwrap();
            let q = quantize_excitation_frame(
                bw,
                SilkFrameSize::TwentyMs,
                SignalType::Voiced,
                QuantizationOffsetType::Low,
                seed,
                &gains,
                &a_q12,
                Some(&ltp),
                &target,
                &mut ltp_state,
                &mut lpc_state,
            )
            .unwrap();
            let snr = snr_db(&target, &q.reconstructed);
            assert!(snr > 12.0, "seed {seed}: voiced SNR too low: {snr} dB");
        }
    }

    /// Shell-block encodability: even a pathological full-scale step
    /// target produces blocks that Excitation::encode accepts.
    #[test]
    fn pathological_target_stays_encodable() {
        let bw = Bandwidth::Nb;
        let n = subframe_samples(bw).unwrap();
        let a_q12 = vec![0i32; 10];
        let mut ltp_state = LtpSynthState::new(bw).unwrap();
        let mut lpc_state = LpcSynthState::new(bw).unwrap();
        // Full-scale alternating target with the smallest gain: wants
        // enormous pulses; the cap + LSB depths must keep it legal.
        let target: Vec<f32> = (0..4 * n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let gains = [81_920u32; 4]; // minimum §4.2.7.4 gain
        let q = quantize_excitation_frame(
            bw,
            SilkFrameSize::TwentyMs,
            SignalType::Unvoiced,
            QuantizationOffsetType::Low,
            2,
            &gains,
            &a_q12,
            None,
            &target,
            &mut ltp_state,
            &mut lpc_state,
        )
        .unwrap();
        assert!(q.e_raw.iter().all(|&v| v.abs() <= MAX_PULSE_MAGNITUDE));
        let ex_cfg = ExcitationConfig {
            bandwidth: bw,
            frame_size: SilkFrameSize::TwentyMs,
            signal_type: SignalType::Unvoiced,
            qoff_type: QuantizationOffsetType::Low,
            lcg_seed: 2,
        };
        let mut re = RangeEncoder::new();
        let symbols = ExcitationSymbols {
            rate_level: q.rate_level,
            lsb_counts: &q.lsb_counts,
            e_raw: &q.e_raw,
        };
        assert!(Excitation::encode(&mut re, ex_cfg, &symbols).is_ok());
    }

    /// 10 ms MB: 120-sample frame, 128-sample (8-block) excitation —
    /// the pad must appear and stay zero.
    #[test]
    fn ten_ms_mb_pads_to_shell_blocks() {
        let bw = Bandwidth::Mb;
        let n = subframe_samples(bw).unwrap();
        let a_q12 = vec![0i32; 10];
        let mut ltp_state = LtpSynthState::new(bw).unwrap();
        let mut lpc_state = LpcSynthState::new(bw).unwrap();
        let target: Vec<f32> = (0..2 * n).map(|i| 0.05 * (i as f32 * 0.07).sin()).collect();
        let gains = [20_000_000u32; 2];
        let q = quantize_excitation_frame(
            bw,
            SilkFrameSize::TenMs,
            SignalType::Unvoiced,
            QuantizationOffsetType::Low,
            0,
            &gains,
            &a_q12,
            None,
            &target,
            &mut ltp_state,
            &mut lpc_state,
        )
        .unwrap();
        assert_eq!(q.e_raw.len(), 128);
        assert!(q.e_raw[120..].iter().all(|&v| v == 0));
    }

    #[test]
    fn rejects_mismatched_args() {
        let bw = Bandwidth::Nb;
        let n = subframe_samples(bw).unwrap();
        let a_q12 = vec![0i32; 10];
        let mut ltp_state = LtpSynthState::new(bw).unwrap();
        let mut lpc_state = LpcSynthState::new(bw).unwrap();
        let target = vec![0.0f32; 4 * n];
        // Voiced without LTP params.
        assert!(quantize_excitation_frame(
            bw,
            SilkFrameSize::TwentyMs,
            SignalType::Voiced,
            QuantizationOffsetType::Low,
            0,
            &[100_000; 4],
            &a_q12,
            None,
            &target,
            &mut ltp_state,
            &mut lpc_state,
        )
        .is_err());
        // Bad seed.
        assert!(quantize_excitation_frame(
            bw,
            SilkFrameSize::TwentyMs,
            SignalType::Unvoiced,
            QuantizationOffsetType::Low,
            4,
            &[100_000; 4],
            &a_q12,
            None,
            &target,
            &mut ltp_state,
            &mut lpc_state,
        )
        .is_err());
    }
}
