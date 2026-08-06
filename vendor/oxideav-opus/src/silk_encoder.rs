//! Analysed SILK encoding: PCM → SILK-only Opus packets — RFC 6716
//! §5.2 front half driving the §4.2.7 write-side wire mirrors.
//!
//! This module is the round-388 integration of the encoder
//! signal-analysis stack: unlike
//! [`crate::silk_packet_encode::encode_silk_only_packet_mono`], which
//! consumes caller-supplied symbol scripts, these encoders derive
//! every Table-5 symbol from the input audio itself. The per-channel
//! analysis chain ([`ChannelAnalyzer`]):
//!
//!  1. **Short-term prediction** (§5.2.3.4): Burg LPC on the frame
//!     ([`crate::silk_lpc_analysis`]), bandwidth-expanded, converted
//!     to NLSF ([`crate::silk_lpc_to_nlsf`]) and quantized to the
//!     §4.2.7.5 `(I1, I2[])` wire indices
//!     ([`crate::silk_nlsf_quantize`]). The frame always signals
//!     `w_Q2 = 4` (no interpolation split).
//!  2. **Pitch + voicing** (§5.2.3.2): whitened-domain open-loop lag
//!     search ([`crate::silk_pitch`]); voiced frames quantize the
//!     primary lag (absolutely on a packet's first SILK frame, relative
//!     to the previous frame's decoded lag when §4.2.7.6.1 allows it)
//!     and pitch contour, and run the §5.2.3.6 LTP codebook
//!     quantisation ([`crate::silk_ltp_analysis`]) against the DECODED
//!     lags. A frame whose input RMS sits below the activity floor is
//!     coded INACTIVE (§4.2.3 VAD clear) and skips the search.
//!  3. **Gains** (§5.2.3.3's role): per-subframe residual energy →
//!     the §4.2.7.4 log-gain index whose dequantised value scales the
//!     excitation pulses to a target RMS; the first subframe's index
//!     is floored at `previous_log_gain - 16` so the decoder-side
//!     §4.2.7.4 clamp (which carries ACROSS packets) can never bind
//!     and the fresh-decoder packet writer stays stream-exact.
//!  4. **Excitation** (§5.2.3.8's role): closed-loop pulse rounding
//!     against the carried decoder state
//!     ([`crate::silk_excitation_quantize`]).
//!
//! [`SilkEncoderMono`] wraps one analyzer into code-0 mono packets.
//! [`SilkEncoderStereo`] adds the §5.2.2 stereo mixing front end: the
//! least-squares §4.2.7.1 weight estimate
//! ([`crate::silk_stereo::estimate_stereo_weights`]) quantized through
//! [`StereoWeightSymbols::quantize`], the exact §4.2.8 algebraic
//! downmix ([`crate::silk_stereo::stereo_lr_to_ms`]) run with the
//! QUANTIZED weights, per-channel analysis of the mid and side
//! signals, and the §4.2.7.2 mid-only escape when the residual side
//! energy is negligible (mirroring the decoder's side-state reset on
//! an uncoded side frame).
//!
//! The produced packets are ordinary code-0 SILK-only packets that a
//! fresh or streaming [`crate::decoder::OpusDecoder`] decodes to
//! audio tracking the input.
//!
//! Input is PCM at the SILK **internal** rate (8 kHz NB / 12 kHz MB /
//! 16 kHz WB), nominal range `[-1.0, 1.0]`. Packets carry 20, 40, or
//! 60 ms (one to three analysed 20 ms SILK frames per §4.2.2, with the
//! intra-packet §4.2.7.4 / §4.2.7.6.1 carried state threaded across
//! them exactly the way the decoder's regular-frame walk threads it).
//!
//! All truth is taken from RFC 6716 §4.2.7 / §4.2.8 / §5.2. No
//! external library source is consulted.

use crate::silk_decode::SilkFrameSymbols;
use crate::silk_excitation::{ExcitationSymbols, SilkFrameSize};
use crate::silk_excitation_quantize::{
    quantize_excitation_frame, ExcitationQuantized, LtpFrameParams,
};
use crate::silk_frame::{
    QuantizationOffsetType, SignalType, SilkHeaderSymbols, StereoPredictionWeights,
    StereoWeightSymbols,
};
use crate::silk_gains::{GainSymbol, SubframeGains, SubframeGainsConfig};
use crate::silk_log2lin::silk_gains_dequant;
use crate::silk_lpc_analysis::{bandwidth_expand, burg_lpc, lpc_residual};
use crate::silk_lpc_synth::{subframe_samples, LpcSynthState};
use crate::silk_lpc_to_nlsf::lpc_to_nlsf_q15;
use crate::silk_lsf_stage2::{D_LPC_NB_MB, D_LPC_WB};
use crate::silk_lsf_to_lpc::LpcQ17;
use crate::silk_ltp::{contour_offsets, lag_range, LtpSymbols, LTP_MAX_SUBFRAMES};
use crate::silk_ltp_analysis::ltp_analysis;
use crate::silk_ltp_synth::LtpSynthState;
use crate::silk_nlsf_quantize::quantize_nlsf;
use crate::silk_packet_encode::{
    encode_silk_only_packet_mono_with_lbrr, encode_silk_only_packet_stereo_with_lbrr,
    StereoIntervalLbrr, StereoIntervalScripts,
};
use crate::silk_pitch::{pitch_analysis, quantize_lag};
use crate::silk_stereo::{stereo_lr_to_ms, StereoDownmixState, StereoWeightsQ13};
use crate::toc::Bandwidth;
use crate::Error;

/// Target RMS of the excitation pulses (`e_raw`) the gain selection
/// aims for — the bitrate/precision knob of this encoder.
const TARGET_PULSE_RMS: f64 = 2.0;

/// Reduced pulse target for §4.2.5 LBRR (in-band FEC) re-encodes: the
/// redundant copy spends roughly half the pulse budget of the regular
/// frames (§2.1.7 codes the previous frame "at a lower bitrate").
const LBRR_PULSE_RMS: f64 = 1.0;

/// Initial bandwidth-expansion chirp applied to the Burg predictor
/// before LSF conversion.
const ANALYSIS_CHIRP: f64 = 0.996;

/// Side-channel RMS below which a stereo interval is coded mid-only
/// (§4.2.7.2).
const MID_ONLY_SIDE_RMS: f64 = 1.0e-4;

/// Frame RMS below which a frame is coded INACTIVE (§4.2.7.3 frame
/// type 0, VAD flag clear) — the signal-derived §4.2.3 VAD decision.
const ACTIVITY_RMS: f64 = 1.0e-3;

/// One channel's fully analysed SILK frame: every Table-5 symbol
/// (owned) plus the encoder's decode-mirror monitoring.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedFrame {
    /// Steps 1-3 header symbols (stereo parts filled by the caller).
    pub header: SilkHeaderSymbols,
    /// §4.2.7.4 per-subframe gain symbols.
    pub gains: Vec<GainSymbol>,
    /// §4.2.7.5.1 stage-1 index.
    pub lsf_stage1: u8,
    /// §4.2.7.5.2 stage-2 indices.
    pub i2: Vec<i8>,
    /// §4.2.7.6 symbols (voiced frames only).
    pub ltp: Option<LtpSymbols>,
    /// §4.2.7.7 LCG seed.
    pub lcg_seed: u8,
    /// §4.2.7.8.1 rate level.
    pub rate_level: u8,
    /// §4.2.7.8.2 per-block extra-LSB counts.
    pub lsb_counts: Vec<u8>,
    /// §4.2.7.8 signed excitation values.
    pub e_raw: Vec<i32>,
    /// The internal-rate signal the decoder will reconstruct.
    pub reconstructed: Vec<f32>,
    /// Whether the frame was coded voiced.
    pub voiced: bool,
    /// §4.2.7.5.5 factor: `Some(4)` on a 20 ms frame (no interpolation
    /// split), `None` on a 10 ms frame (the factor is not stored).
    pub lsf_interp_w_q2: Option<u8>,
}

impl AnalyzedFrame {
    /// Borrow the owned symbols as the [`SilkFrameSymbols`] view the
    /// §4.2 packet writers consume.
    pub fn symbols(&self) -> SilkFrameSymbols<'_> {
        SilkFrameSymbols {
            header: self.header,
            gains: &self.gains,
            lsf_stage1: self.lsf_stage1,
            lsf_stage2_i2: &self.i2,
            lsf_interp_w_q2: self.lsf_interp_w_q2,
            ltp: self.ltp,
            lcg_seed: self.lcg_seed,
            excitation: ExcitationSymbols {
                rate_level: self.rate_level,
                lsb_counts: &self.lsb_counts,
                e_raw: &self.e_raw,
            },
        }
    }
}

/// The per-channel §5.2.3 analysis chain with its carried state
/// (input history, §4.2.7.9 synthesis histories, cross-packet gain
/// clamp base). One instance per coded channel.
#[derive(Debug, Clone)]
pub struct ChannelAnalyzer {
    bandwidth: Bandwidth,
    d_lpc: usize,
    /// Input history (internal rate) for whitening + pitch lookback.
    hist: Vec<f64>,
    /// Carried §4.2.7.9 synthesis histories (decoder-authoritative).
    ltp_state: LtpSynthState,
    lpc_state: LpcSynthState,
    /// Cross-packet §4.2.7.4 clamp base.
    prev_log_gain: Option<u8>,
    /// Intra-packet §4.2.7.6.1 relative-lag base: the previous frame's
    /// decoded primary lag when that frame was voiced. Cleared at every
    /// packet boundary (the first frame of an Opus frame always codes
    /// its lag absolutely).
    prev_lag: Option<i32>,
    /// Target excitation-pulse RMS of the gain selection
    /// ([`TARGET_PULSE_RMS`] normally, [`LBRR_PULSE_RMS`] for a
    /// re-armed LBRR re-encoder).
    pulse_rms: f64,
    /// When set (LBRR re-encoders), active frames are coded UNVOICED —
    /// the LBRR sequence synthesizes from a fresh §4.2.7.9 state on
    /// both sides, so an LTP filter would predict from an all-zero
    /// history: no prediction gain, and the closed-loop pulses would
    /// have to carry the entire un-predicted signal at enormous rate.
    force_unvoiced: bool,
}

impl ChannelAnalyzer {
    /// Create an analyzer for one SILK internal bandwidth (NB / MB /
    /// WB; SWB / FB are rejected — SILK never codes them).
    pub fn new(bandwidth: Bandwidth) -> Result<Self, Error> {
        let d_lpc = match bandwidth {
            Bandwidth::Nb | Bandwidth::Mb => D_LPC_NB_MB,
            Bandwidth::Wb => D_LPC_WB,
            _ => return Err(Error::MalformedPacket),
        };
        let (_, lag_max, _) = lag_range(bandwidth)?;
        let hist_len = lag_max as usize + 2 + d_lpc;
        Ok(Self {
            bandwidth,
            d_lpc,
            hist: vec![0.0; hist_len],
            ltp_state: LtpSynthState::new(bandwidth)?,
            lpc_state: LpcSynthState::new(bandwidth)?,
            prev_log_gain: None,
            prev_lag: None,
            pulse_rms: TARGET_PULSE_RMS,
            force_unvoiced: false,
        })
    }

    /// Reset all carried state (matches a §4.5.2 decoder reset, or
    /// the decoder's side-channel state clear after an uncoded side
    /// frame).
    pub fn reset(&mut self) {
        for v in self.hist.iter_mut() {
            *v = 0.0;
        }
        self.ltp_state.reset();
        self.lpc_state.reset();
        self.prev_log_gain = None;
        self.prev_lag = None;
    }

    /// Re-arm a CLONE of a channel analyzer as the §4.2.5 LBRR
    /// re-encoder for the packet the clone was snapshotted BEFORE: the
    /// FEC decoder synthesizes LBRR frames from a fresh §4.2.7.9 state
    /// and the LBRR sequence codes its first gain independently with no
    /// clamp base, so the closed-loop mirror starts fresh too. The
    /// input history is kept — it only conditions the analysis — and
    /// the pulse target drops to the reduced LBRR rate.
    pub fn rearm_for_lbrr(&mut self) {
        self.ltp_state.reset();
        self.lpc_state.reset();
        self.prev_log_gain = None;
        self.prev_lag = None;
        self.pulse_rms = LBRR_PULSE_RMS;
        self.force_unvoiced = true;
    }

    /// Mark a §4.2.4 LBRR-flag gap (an interval with no LBRR frame):
    /// the next coded LBRR frame codes its gain independently and its
    /// lag absolutely again, mirroring the packet writer's and the
    /// decoder's gap handling.
    pub fn mark_lbrr_gap(&mut self) {
        self.prev_log_gain = None;
        self.prev_lag = None;
    }

    /// Analyse one 20 ms internal-rate frame into a complete Table-5
    /// symbol script (stereo header parts left `None`; the stereo
    /// wrapper fills them). Advances every carried state. Equivalent to
    /// [`Self::analyze_frame_at`] with `first_in_packet = true` (the
    /// one-frame-per-packet case).
    pub fn analyze_frame(&mut self, pcm: &[f32]) -> Result<AnalyzedFrame, Error> {
        self.analyze_frame_at(pcm, true)
    }

    /// Analyse one 20 ms internal-rate frame as SILK frame
    /// `first_in_packet ? 0 : k>0` of a (possibly multi-frame) Opus
    /// frame, threading the intra-packet carried state exactly the way
    /// the §4.2.6 regular-frame decode walk threads it:
    ///
    /// * **Gains** (§4.2.7.4): the packet's first frame codes its first
    ///   subframe gain independently (floored against the cross-packet
    ///   clamp base); later frames delta-code it against the previous
    ///   frame's last subframe gain.
    /// * **Pitch lag** (§4.2.7.6.1): the packet's first frame codes its
    ///   primary lag absolutely; a later frame following a VOICED frame
    ///   codes relative to that frame's decoded lag.
    /// * **LTP scaling** (§4.2.7.6.3): present only on the packet's
    ///   first frame.
    /// * **VAD** (§4.2.3 / §4.2.7.3): derived from the signal — a frame
    ///   whose RMS is below the activity floor is coded INACTIVE (frame
    ///   type 0, VAD flag clear), skipping the pitch/LTP search.
    pub fn analyze_frame_at(
        &mut self,
        pcm: &[f32],
        first_in_packet: bool,
    ) -> Result<AnalyzedFrame, Error> {
        self.analyze_frame_sized(pcm, first_in_packet, SilkFrameSize::TwentyMs)
    }

    /// [`Self::analyze_frame_at`] for an explicit SILK frame size: a
    /// 20 ms frame (4 subframes) or the §4.2.2 single 10 ms frame
    /// (2 subframes, whose §4.2.7.5.5 factor is not stored).
    pub fn analyze_frame_sized(
        &mut self,
        pcm: &[f32],
        first_in_packet: bool,
        frame_size: SilkFrameSize,
    ) -> Result<AnalyzedFrame, Error> {
        let n = subframe_samples(self.bandwidth)?;
        let num_subframes = match frame_size {
            SilkFrameSize::TenMs => 2usize,
            SilkFrameSize::TwentyMs => 4usize,
        };
        let frame_len = n * num_subframes;
        if pcm.len() != frame_len {
            return Err(Error::MalformedPacket);
        }
        let (lag_min, lag_max, _) = lag_range(self.bandwidth)?;
        let hist_len = self.hist.len();

        // §4.2.7.6.1: the packet's first frame always codes its lag
        // absolutely — the relative base never crosses a packet.
        if first_in_packet {
            self.prev_lag = None;
        }

        // ---- 0. Signal-activity (VAD) decision (§4.2.3). ----
        let input_energy: f64 = pcm.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let active = (input_energy / frame_len as f64).sqrt() >= ACTIVITY_RMS;

        // Contiguous analysis buffer: [history | frame], f64.
        let mut buf = Vec::with_capacity(hist_len + frame_len);
        buf.extend_from_slice(&self.hist);
        buf.extend(pcm.iter().map(|&v| v as f64));

        // ---- 1. Short-term prediction analysis (§5.2.3.4). ----
        // Burg over the frame plus one subframe of history for
        // conditioning; chirp; convert to NLSF with re-conditioning
        // retries; fall back to the uniform spectrum (A(z) = 1).
        let window = &buf[buf.len() - (frame_len + n)..];
        let mut a = burg_lpc(window, self.d_lpc)?;
        bandwidth_expand(&mut a, ANALYSIS_CHIRP);
        let nlsf_target = {
            let mut attempt = a.clone();
            let mut result = lpc_to_nlsf_q15(&attempt);
            let mut tries = 0;
            while result.is_err() && tries < 4 {
                bandwidth_expand(&mut attempt, 0.92);
                result = lpc_to_nlsf_q15(&attempt);
                tries += 1;
            }
            result.unwrap_or_else(|_| {
                (0..self.d_lpc)
                    .map(|k| (32768 * (k + 1) / (self.d_lpc + 1)) as i16)
                    .collect()
            })
        };

        // ---- 2. NLSF quantisation (§4.2.7.5) + decoder LPC. ----
        let nq = quantize_nlsf(self.bandwidth, &nlsf_target)?;
        let lpc_hat = LpcQ17::from_nlsf(self.bandwidth, nq.nlsf_q15())?
            .range_limited()
            .prediction_gain_limited();
        let a_q12 = lpc_hat.a_q12().to_vec();
        let a_hat: Vec<f64> = a_q12.iter().map(|&c| c as f64 / 4096.0).collect();

        // Whitened residual over the whole buffer with the QUANTIZED
        // filter (what the decoder's synthesis will invert).
        let r = lpc_residual(&buf, &[], &a_hat);

        // ---- 3. Pitch analysis + LTP (§5.2.3.2 / §5.2.3.6). ----
        // Skipped for an inactive frame (no LTP search on silence) and
        // for LBRR re-encoders (`force_unvoiced` — LTP has no history
        // to predict from in the fresh-state LBRR sequence).
        let pa = if active && !self.force_unvoiced {
            Some(pitch_analysis(self.bandwidth, &r, hist_len, num_subframes)?)
        } else {
            None
        };
        let voiced = pa.as_ref().is_some_and(|p| p.voiced);
        let (ltp_symbols, ltp_params, decoded_lags, decoded_primary) = if voiced {
            let pa = pa.as_ref().expect("voiced implies analysis ran");
            // Quantize the primary lag first — relative to the previous
            // frame's decoded lag when §4.2.7.6.1 allows it (a voiced
            // predecessor in the same packet), absolutely otherwise —
            // then re-derive the DECODED subframe lags from the decoded
            // primary so the LTP analysis and the excitation loop see
            // exactly what the decoder will.
            let (lag_sym, decoded_primary) =
                quantize_lag(self.bandwidth, pa.primary_lag, self.prev_lag)?;
            let offs = contour_offsets(self.bandwidth, num_subframes, pa.contour_index)?;
            let mut lags = [0i32; LTP_MAX_SUBFRAMES];
            for (s, slot) in lags.iter_mut().enumerate().take(num_subframes) {
                *slot = (decoded_primary + offs[s] as i32).clamp(lag_min, lag_max);
            }
            let lq = ltp_analysis(
                self.bandwidth,
                &r,
                hist_len,
                num_subframes,
                &lags[..num_subframes],
            )?;
            let symbols = LtpSymbols {
                lag: lag_sym,
                contour_index: pa.contour_index,
                periodicity_index: lq.periodicity_index,
                filter_indices: lq.filter_indices,
                // §4.2.7.6.3: the scaling field rides only the packet's
                // first frame; index 0 → the default 15565 (which is
                // also what an absent field reconstructs to).
                ltp_scaling_index: first_in_packet.then_some(0),
            };
            let params = LtpFrameParams {
                pitch_lags: lags,
                taps_q7: lq.taps_q7,
                ltp_scaling_q14: 15565,
            };
            (Some(symbols), Some(params), lags, Some(decoded_primary))
        } else {
            (None, None, [0i32; LTP_MAX_SUBFRAMES], None)
        };
        // §4.2.7.6.1: only a VOICED frame arms relative lag coding for
        // the next frame in the same packet.
        self.prev_lag = decoded_primary;

        // ---- 4. Gain selection (§4.2.7.4 quantize). ----
        // Per-subframe LTP-filtered residual energy → the log gain
        // whose dequantised value puts the pulses at TARGET_PULSE_RMS.
        let signal_type = if voiced {
            SignalType::Voiced
        } else if active {
            SignalType::Unvoiced
        } else {
            SignalType::Inactive
        };
        let mut desired = [0u8; LTP_MAX_SUBFRAMES];
        for s in 0..num_subframes {
            let base = hist_len + s * n;
            let mut energy = 0.0f64;
            for i in 0..n {
                let mut v = r[base + i];
                if let Some(p) = &ltp_params {
                    let lag = decoded_lags[s] as usize;
                    for (k, &t) in p.taps_q7[s].iter().enumerate() {
                        v -= t as f64 / 128.0 * r[base + i + 2 - lag - k];
                    }
                }
                energy += v * v;
            }
            let rms = (energy / n as f64).sqrt();
            // e_raw ≈ res * 2^31 / gain_Q16 (see the module docs of
            // silk_excitation_quantize): gain that lands the pulses on
            // the target RMS.
            let want_gain = rms * (1u64 << 31) as f64 / self.pulse_rms;
            desired[s] = quantize_log_gain(want_gain);
        }
        // §4.2.7.4 threading. Packet-first frame: independent coding
        // with the cross-packet clamp kept inert by flooring the index
        // (the decoder computes log_gain = max(gain_index, prev - 16)
        // with prev carried ACROSS packets). Later frames in the same
        // packet: delta-coded against the previous frame's last
        // subframe gain, exactly as the decoder's regular walk does.
        let first_subframe_is_independent = first_in_packet || self.prev_log_gain.is_none();
        if first_subframe_is_independent {
            if let Some(prev) = self.prev_log_gain {
                desired[0] = desired[0].max(prev.saturating_sub(16));
            }
        }
        let gains_cfg = SubframeGainsConfig {
            signal_type,
            num_subframes: num_subframes as u8,
            first_subframe_is_independent,
            previous_log_gain: if first_in_packet {
                None
            } else {
                self.prev_log_gain
            },
        };
        let (gain_symbols, gains) = SubframeGains::quantize(gains_cfg, &desired[..num_subframes])?;
        let gains_q16_arr = gains.dequant_q16();
        // The fresh-decoder packet writer will reproduce these exact
        // gains; carry the last for the next packet's floor.
        self.prev_log_gain = Some(gains.last_log_gain());

        // ---- 5. Closed-loop excitation (§5.2.3.8 role). ----
        let lcg_seed = 0u8;
        let ExcitationQuantized {
            e_raw,
            lsb_counts,
            rate_level,
            reconstructed,
        } = quantize_excitation_frame(
            self.bandwidth,
            frame_size,
            signal_type,
            QuantizationOffsetType::Low,
            lcg_seed,
            &gains_q16_arr[..num_subframes],
            &a_q12,
            ltp_params.as_ref(),
            pcm,
            &mut self.ltp_state,
            &mut self.lpc_state,
        )?;

        // Roll the input history.
        self.hist.extend(pcm.iter().map(|&v| v as f64));
        let cut = self.hist.len() - hist_len;
        self.hist.drain(..cut);

        Ok(AnalyzedFrame {
            header: SilkHeaderSymbols {
                stereo: None,
                mid_only_flag: None,
                // Table 10 (Low offset column): Inactive = 0,
                // Unvoiced = 2, Voiced = 4. Types >= 2 set the §4.2.3
                // VAD flag; the signal-derived activity decision above
                // selects Inactive for silent frames.
                frame_type: if voiced {
                    4
                } else if active {
                    2
                } else {
                    0
                },
            },
            gains: gain_symbols,
            lsf_stage1: nq.lsf_stage1,
            i2: nq.i2().to_vec(),
            ltp: ltp_symbols,
            lcg_seed,
            rate_level,
            lsb_counts,
            e_raw,
            reconstructed,
            voiced,
            // §4.2.7.5.5: stored (as the no-split value 4) only on
            // 20 ms frames.
            lsf_interp_w_q2: (frame_size == SilkFrameSize::TwentyMs).then_some(4),
        })
    }
}

/// One encoded packet plus the encoder's decode-mirror monitoring.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedSilkPacket {
    /// The complete code-0 SILK-only Opus packet (TOC + payload).
    pub packet: Vec<u8>,
    /// The internal-rate signal the decoder will reconstruct for this
    /// packet (mono: the channel; stereo: the MID channel).
    pub reconstructed: Vec<f32>,
    /// Whether the (mid) frame was coded voiced.
    pub voiced: bool,
}

/// Map an analysed-encoder packet duration to its §4.2.2 layout: a
/// 10 ms packet carries one 10 ms (2-subframe) SILK frame; 20 / 40 /
/// 60 ms packets carry one to three 20 ms SILK frames.
fn packet_layout(packet_tenths_ms: u16) -> Result<(usize, SilkFrameSize), Error> {
    match packet_tenths_ms {
        100 => Ok((1, SilkFrameSize::TenMs)),
        200 => Ok((1, SilkFrameSize::TwentyMs)),
        400 => Ok((2, SilkFrameSize::TwentyMs)),
        600 => Ok((3, SilkFrameSize::TwentyMs)),
        _ => Err(Error::MalformedPacket),
    }
}

/// Samples in one SILK frame of `frame_size` at `bandwidth`'s
/// internal rate.
fn frame_size_samples(bandwidth: Bandwidth, frame_size: SilkFrameSize) -> Result<usize, Error> {
    let n = subframe_samples(bandwidth)?;
    Ok(match frame_size {
        SilkFrameSize::TenMs => n * 2,
        SilkFrameSize::TwentyMs => n * 4,
    })
}

/// The previous packet's material a FEC-enabled encoder keeps so the
/// NEXT packet can carry its §4.2.5 LBRR re-encode: the input PCM and
/// a clone of the channel analyzer as it stood BEFORE that packet was
/// analysed (its history conditions the re-analysis).
#[derive(Debug, Clone)]
struct PendingFecMono {
    pcm: Vec<f32>,
    analyzer: ChannelAnalyzer,
}

/// Streaming mono SILK encoder: 20 / 40 / 60 ms of internal-rate PCM
/// in, one SILK-only Opus packet out (one §4.2.2 SILK frame per 20 ms
/// interval, with the intra-packet §4.2.7.4 / §4.2.7.6.1 carried state
/// threaded across them and per-frame §4.2.3 VAD flags derived from
/// the signal). With [`Self::set_fec`] enabled, each packet also
/// carries the §4.2.5 LBRR (in-band FEC) re-encode of the previous
/// packet's active intervals at the reduced LBRR rate.
#[derive(Debug, Clone)]
pub struct SilkEncoderMono {
    channel: ChannelAnalyzer,
    packet_tenths_ms: u16,
    frames_per_packet: usize,
    silk_frame_size: SilkFrameSize,
    fec: bool,
    pending_fec: Option<PendingFecMono>,
}

impl SilkEncoderMono {
    /// Create a 20 ms-packet encoder for one SILK internal bandwidth
    /// (NB / MB / WB; SWB / FB are rejected — SILK never codes them).
    pub fn new(bandwidth: Bandwidth) -> Result<Self, Error> {
        Self::with_packet_duration(bandwidth, 200)
    }

    /// Create an encoder emitting `packet_tenths_ms` (100 / 200 / 400 /
    /// 600 — i.e. 10 / 20 / 40 / 60 ms) SILK-only packets: one analysed
    /// SILK frame per §4.2.2 time interval (a single 2-subframe frame
    /// for a 10 ms packet), packed into a single code-0 Opus packet.
    pub fn with_packet_duration(
        bandwidth: Bandwidth,
        packet_tenths_ms: u16,
    ) -> Result<Self, Error> {
        let (frames_per_packet, silk_frame_size) = packet_layout(packet_tenths_ms)?;
        Ok(Self {
            channel: ChannelAnalyzer::new(bandwidth)?,
            packet_tenths_ms,
            frames_per_packet,
            silk_frame_size,
            fec: false,
            pending_fec: None,
        })
    }

    /// Enable / disable §4.2.5 LBRR (in-band FEC) emission: when on,
    /// every packet after the first carries a reduced-rate re-encode of
    /// the previous packet's ACTIVE intervals, which a receiver can
    /// recover with [`crate::decoder::OpusDecoder::decode_packet_fec`]
    /// when the previous packet is lost. Disabling also drops any
    /// pending redundancy.
    pub fn set_fec(&mut self, enabled: bool) {
        self.fec = enabled;
        if !enabled {
            self.pending_fec = None;
        }
    }

    /// Per-packet input length: the packet duration at the internal
    /// rate (20 ms = 160 NB / 240 MB / 320 WB samples, times the 1-3
    /// SILK frames per packet; half that for a 10 ms packet).
    pub fn frame_samples(&self) -> usize {
        frame_size_samples(self.channel.bandwidth, self.silk_frame_size).unwrap_or(0)
            * self.frames_per_packet
    }

    /// Reset all carried state (matches a §4.5.2 decoder reset).
    pub fn reset(&mut self) {
        self.channel.reset();
        self.pending_fec = None;
    }

    /// Encode one packet's worth (20 / 40 / 60 ms) of mono
    /// internal-rate PCM into a SILK-only Opus packet.
    ///
    /// `pcm.len()` must equal [`Self::frame_samples`]; samples are
    /// nominally in `[-1.0, 1.0]`.
    pub fn encode_packet(&mut self, pcm: &[f32]) -> Result<EncodedSilkPacket, Error> {
        let bandwidth = self.channel.bandwidth;
        if pcm.len() != self.frame_samples() {
            return Err(Error::MalformedPacket);
        }
        let flen = frame_size_samples(bandwidth, self.silk_frame_size)?;

        // §4.2.5 / §2.1.7: the LBRR frames riding in THIS packet are a
        // reduced-rate re-encode of the PREVIOUS packet's intervals,
        // analysed from the pre-packet analyzer snapshot with a fresh
        // closed-loop state (mirroring the FEC decoder's fresh
        // synthesis). Inactive intervals carry no LBRR (§4.2.7.3 codes
        // every LBRR frame with the active PDFs) and re-arm the gap
        // rules.
        let lbrr_frames: Vec<Option<AnalyzedFrame>> = match self.pending_fec.take() {
            Some(pending) => {
                let mut la = pending.analyzer;
                la.rearm_for_lbrr();
                let mut lbrr_first = true;
                let mut out = Vec::with_capacity(self.frames_per_packet);
                for chunk in pending.pcm.chunks_exact(flen) {
                    let f = la.analyze_frame_sized(chunk, lbrr_first, self.silk_frame_size)?;
                    if f.header.frame_type >= 2 {
                        lbrr_first = false;
                        out.push(Some(f));
                    } else {
                        la.mark_lbrr_gap();
                        lbrr_first = true;
                        out.push(None);
                    }
                }
                out
            }
            None => vec![None; self.frames_per_packet],
        };
        if self.fec {
            self.pending_fec = Some(PendingFecMono {
                pcm: pcm.to_vec(),
                analyzer: self.channel.clone(),
            });
        }

        let mut frames = Vec::with_capacity(self.frames_per_packet);
        for (k, chunk) in pcm.chunks_exact(flen).enumerate() {
            frames.push(
                self.channel
                    .analyze_frame_sized(chunk, k == 0, self.silk_frame_size)?,
            );
        }
        let symbols: Vec<_> = frames.iter().map(AnalyzedFrame::symbols).collect();
        let lbrr_symbols: Vec<Option<SilkFrameSymbols<'_>>> = lbrr_frames
            .iter()
            .map(|o| o.as_ref().map(AnalyzedFrame::symbols))
            .collect();
        let (packet, _, _) = encode_silk_only_packet_mono_with_lbrr(
            bandwidth,
            self.packet_tenths_ms,
            &symbols,
            &lbrr_symbols,
        )?;
        let mut reconstructed = Vec::with_capacity(pcm.len());
        let mut voiced = false;
        for f in &frames {
            reconstructed.extend_from_slice(&f.reconstructed);
            voiced |= f.voiced;
        }
        Ok(EncodedSilkPacket {
            packet,
            reconstructed,
            voiced,
        })
    }

    /// [`Self::encode_packet`] with §3.2.5 CBR transport shaping: the
    /// packet is re-framed as a code-3 packet padded to **exactly**
    /// `target_bytes` (constant packet size on the wire; the decode is
    /// identical). Errors when the compressed packet exceeds the
    /// target — the analysis state has then already advanced, so pick
    /// a target with headroom for the configured rate.
    pub fn encode_packet_cbr(
        &mut self,
        pcm: &[f32],
        target_bytes: usize,
    ) -> Result<EncodedSilkPacket, Error> {
        let mut out = self.encode_packet(pcm)?;
        out.packet = crate::packet_compose::pad_packet_to(&out.packet, target_bytes)?;
        Ok(out)
    }
}

/// One interval of the previous stereo packet kept for the next
/// packet's §4.2.5 LBRR re-encode: the DOWNMIXED mid/side signals and
/// coded weight quintuple from the regular pass (so the redundant copy
/// codes the identical §4.2.8 mix), plus whether the regular pass
/// coded an ACTIVE side frame (only then does the interval carry a
/// side LBRR frame).
#[derive(Debug, Clone)]
struct PendingFecStereoInterval {
    mid_pcm: Vec<f32>,
    side_pcm: Vec<f32>,
    side_active: bool,
    weights: StereoWeightSymbols,
}

/// The previous stereo packet's material a FEC-enabled encoder keeps
/// (see [`PendingFecMono`]): per-interval mix products plus clones of
/// both channel analyzers as they stood BEFORE that packet.
#[derive(Debug, Clone)]
struct PendingFecStereo {
    intervals: Vec<PendingFecStereoInterval>,
    mid_analyzer: ChannelAnalyzer,
    side_analyzer: ChannelAnalyzer,
}

/// Streaming stereo SILK encoder: 20 / 40 / 60 ms of L/R internal-rate
/// PCM in, one stereo SILK-only Opus packet out (§5.2.2 stereo mixing +
/// §4.2.7.1 weight coding + §4.2.7.2 mid-only escape, run **per 20 ms
/// interval** exactly like the decoder's §4.2.2 interleaved walk). With
/// [`Self::set_fec`] enabled, each packet also carries the §4.2.5 LBRR
/// re-encode of the previous packet's active intervals.
#[derive(Debug, Clone)]
pub struct SilkEncoderStereo {
    bandwidth: Bandwidth,
    mid: ChannelAnalyzer,
    side: ChannelAnalyzer,
    downmix: StereoDownmixState,
    /// Previous interval's trailing raw-mid sample (the §4.2.8 `p0`
    /// boundary term for the weight estimate).
    prev_mid: f32,
    packet_tenths_ms: u16,
    frames_per_packet: usize,
    silk_frame_size: SilkFrameSize,
    fec: bool,
    pending_fec: Option<PendingFecStereo>,
}

impl SilkEncoderStereo {
    /// Create a 20 ms-packet stereo encoder for one SILK internal
    /// bandwidth.
    pub fn new(bandwidth: Bandwidth) -> Result<Self, Error> {
        Self::with_packet_duration(bandwidth, 200)
    }

    /// Create a stereo encoder emitting `packet_tenths_ms` (100 / 200 /
    /// 400 / 600 — i.e. 10 / 20 / 40 / 60 ms) SILK-only packets: per
    /// §4.2.2 interval, the §4.2.7.1 weights are re-estimated, the
    /// §4.2.8 downmix re-run, and the §4.2.7.2 mid-only decision
    /// re-taken, exactly matching the decoder's per-interval unmix.
    pub fn with_packet_duration(
        bandwidth: Bandwidth,
        packet_tenths_ms: u16,
    ) -> Result<Self, Error> {
        let (frames_per_packet, silk_frame_size) = packet_layout(packet_tenths_ms)?;
        Ok(Self {
            bandwidth,
            mid: ChannelAnalyzer::new(bandwidth)?,
            side: ChannelAnalyzer::new(bandwidth)?,
            downmix: StereoDownmixState::new(),
            prev_mid: 0.0,
            packet_tenths_ms,
            frames_per_packet,
            silk_frame_size,
            fec: false,
            pending_fec: None,
        })
    }

    /// Enable / disable §4.2.5 LBRR (in-band FEC) emission (see
    /// [`SilkEncoderMono::set_fec`]).
    pub fn set_fec(&mut self, enabled: bool) {
        self.fec = enabled;
        if !enabled {
            self.pending_fec = None;
        }
    }

    /// Per-packet input length per channel (the packet duration at the
    /// internal rate).
    pub fn frame_samples(&self) -> usize {
        frame_size_samples(self.bandwidth, self.silk_frame_size).unwrap_or(0)
            * self.frames_per_packet
    }

    /// Reset all carried state.
    pub fn reset(&mut self) {
        self.mid.reset();
        self.side.reset();
        self.downmix.reset();
        self.prev_mid = 0.0;
        self.pending_fec = None;
    }

    /// Encode one packet's worth (20 / 40 / 60 ms) of stereo
    /// internal-rate PCM.
    ///
    /// `left.len()` and `right.len()` must equal
    /// [`Self::frame_samples`]. `next_lr` is the first left/right
    /// sample pair AFTER this packet when known (the §4.2.8 one-sample
    /// lookahead of the exact downmix); pass `None` at stream end.
    pub fn encode_packet(
        &mut self,
        left: &[f32],
        right: &[f32],
        next_lr: Option<(f32, f32)>,
    ) -> Result<EncodedSilkPacket, Error> {
        let total_len = self.frame_samples();
        if left.len() != total_len || right.len() != total_len {
            return Err(Error::MalformedPacket);
        }
        let flen = frame_size_samples(self.bandwidth, self.silk_frame_size)?;

        // §4.2.5 / §2.1.7: re-encode the PREVIOUS packet's intervals as
        // this packet's LBRR frames, from the pre-packet analyzer
        // snapshots at the reduced rate. The stored downmix products +
        // weight quintuples make the redundant copy code the identical
        // §4.2.8 mix. Per channel, an interval without an LBRR frame
        // re-arms the §4.2.4 gap rules.
        let (lbrr_mid, lbrr_side): (Vec<Option<AnalyzedFrame>>, Vec<Option<AnalyzedFrame>>) =
            match self.pending_fec.take() {
                Some(pending) => {
                    let mut lm = pending.mid_analyzer;
                    let mut ls = pending.side_analyzer;
                    lm.rearm_for_lbrr();
                    ls.rearm_for_lbrr();
                    let mut mid_first = true;
                    let mut side_first = true;
                    let mut mids = Vec::with_capacity(self.frames_per_packet);
                    let mut sides = Vec::with_capacity(self.frames_per_packet);
                    for iv in &pending.intervals {
                        // Side first: the mid LBRR frame's §4.2.7.2
                        // flag depends on whether a side LBRR frame
                        // actually rides this interval.
                        let side_frame = if iv.side_active {
                            let sf = ls.analyze_frame_sized(
                                &iv.side_pcm,
                                side_first,
                                self.silk_frame_size,
                            )?;
                            if sf.header.frame_type >= 2 {
                                side_first = false;
                                Some(sf)
                            } else {
                                ls.mark_lbrr_gap();
                                side_first = true;
                                None
                            }
                        } else {
                            ls.mark_lbrr_gap();
                            side_first = true;
                            None
                        };
                        let mf =
                            lm.analyze_frame_sized(&iv.mid_pcm, mid_first, self.silk_frame_size)?;
                        if mf.header.frame_type >= 2 {
                            let mut mf = mf;
                            mf.header.stereo = Some(iv.weights);
                            // §4.2.7.2 on an LBRR mid frame: the flag
                            // is present (and must be SET) iff no side
                            // LBRR frame follows in this interval.
                            mf.header.mid_only_flag = if side_frame.is_some() {
                                None
                            } else {
                                Some(true)
                            };
                            mid_first = false;
                            mids.push(Some(mf));
                        } else {
                            lm.mark_lbrr_gap();
                            mid_first = true;
                            mids.push(None);
                        }
                        sides.push(side_frame);
                    }
                    (mids, sides)
                }
                None => (
                    vec![None; self.frames_per_packet],
                    vec![None; self.frames_per_packet],
                ),
            };
        let fec_snapshot = self.fec.then(|| (self.mid.clone(), self.side.clone()));
        let mut fec_intervals: Vec<PendingFecStereoInterval> =
            Vec::with_capacity(self.frames_per_packet);

        let mut mid_frames: Vec<AnalyzedFrame> = Vec::with_capacity(self.frames_per_packet);
        let mut side_frames: Vec<Option<AnalyzedFrame>> =
            Vec::with_capacity(self.frames_per_packet);

        for k in 0..self.frames_per_packet {
            let l = &left[k * flen..(k + 1) * flen];
            let r = &right[k * flen..(k + 1) * flen];
            // The §4.2.8 one-sample lookahead: the next interval's
            // first pair inside the packet, the caller's `next_lr` on
            // the last interval.
            let interval_next = if k + 1 < self.frames_per_packet {
                Some((left[(k + 1) * flen], right[(k + 1) * flen]))
            } else {
                next_lr
            };

            // ---- §5.2.3.4 stereo mixing: estimate + quantize weights,
            // then run the exact §4.2.8 inverse with the QUANTIZED pair
            // (what the decoder will apply), per interval — the decoder
            // reads one §4.2.7.1 quintuple per mid frame and restarts
            // its 8 ms interpolation ramp each interval. ----
            let mid_raw: Vec<f32> = l.iter().zip(r).map(|(&a, &b)| (a + b) / 2.0).collect();
            let side_raw: Vec<f32> = l.iter().zip(r).map(|(&a, &b)| (a - b) / 2.0).collect();
            let mid_next = match interval_next {
                Some((a, b)) => (a + b) / 2.0,
                None => mid_raw[flen - 1],
            };
            let target = crate::silk_stereo::estimate_stereo_weights(
                &mid_raw,
                &side_raw,
                self.prev_mid,
                mid_next,
            )?;
            let weight_symbols = StereoWeightSymbols::quantize(StereoPredictionWeights {
                w0_q13: target.w0_q13,
                w1_q13: target.w1_q13,
            });
            let decoded_w = weight_symbols.weights();
            let ms = stereo_lr_to_ms(
                self.bandwidth,
                l,
                r,
                StereoWeightsQ13 {
                    w0_q13: decoded_w.w0_q13,
                    w1_q13: decoded_w.w1_q13,
                },
                interval_next,
                &mut self.downmix,
            )?;
            self.prev_mid = mid_raw[flen - 1];

            // ---- Mid-only decision (§4.2.7.2), per interval. ----
            let side_energy: f64 = ms.side.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let side_rms = (side_energy / flen as f64).sqrt();
            let code_side = side_rms > MID_ONLY_SIDE_RMS;

            // ---- Per-channel analysis. ----
            let mut mid_frame =
                self.mid
                    .analyze_frame_sized(&ms.mid, k == 0, self.silk_frame_size)?;
            mid_frame.header.stereo = Some(weight_symbols);

            let side_active;
            if code_side {
                // The side channel's "first frame" is the packet's
                // first interval; a side frame after a mid-only
                // interval keeps `first_in_packet = false` (the
                // decoder's side sequence cleared its bases but not its
                // §4.2.7.6.3 first-frame flag — mirroring
                // `mark_interval_uncoded(false)`).
                let side_frame =
                    self.side
                        .analyze_frame_sized(&ms.side, k == 0, self.silk_frame_size)?;
                // §4.2.7.2: the mid-only flag is present iff the side
                // VAD for the interval is CLEAR. A coded ACTIVE side
                // frame (type >= 2) sets the VAD, so the flag is
                // absent; a coded INACTIVE side frame leaves the VAD
                // clear and the flag rides as `Some(false)`.
                side_active = side_frame.header.frame_type >= 2;
                mid_frame.header.mid_only_flag = if side_active { None } else { Some(false) };
                mid_frames.push(mid_frame);
                side_frames.push(Some(side_frame));
            } else {
                // Mid-only: flag present and set; the decoder clears
                // the side channel's synthesis state and carried bases
                // after the uncoded side frame — mirror it.
                side_active = false;
                mid_frame.header.mid_only_flag = Some(true);
                self.side.reset();
                mid_frames.push(mid_frame);
                side_frames.push(None);
            }
            if fec_snapshot.is_some() {
                fec_intervals.push(PendingFecStereoInterval {
                    mid_pcm: ms.mid.clone(),
                    side_pcm: ms.side.clone(),
                    side_active,
                    weights: weight_symbols,
                });
            }
        }
        if let Some((mid_analyzer, side_analyzer)) = fec_snapshot {
            self.pending_fec = Some(PendingFecStereo {
                intervals: fec_intervals,
                mid_analyzer,
                side_analyzer,
            });
        }

        let intervals: Vec<StereoIntervalScripts<'_>> = mid_frames
            .iter()
            .zip(side_frames.iter())
            .map(|(m, s)| StereoIntervalScripts {
                mid: m.symbols(),
                side: s.as_ref().map(AnalyzedFrame::symbols),
            })
            .collect();
        let lbrr: Vec<StereoIntervalLbrr<'_>> = lbrr_mid
            .iter()
            .zip(lbrr_side.iter())
            .map(|(m, s)| StereoIntervalLbrr {
                mid: m.as_ref().map(AnalyzedFrame::symbols),
                side: s.as_ref().map(AnalyzedFrame::symbols),
            })
            .collect();
        let (packet, _, _) = encode_silk_only_packet_stereo_with_lbrr(
            self.bandwidth,
            self.packet_tenths_ms,
            &intervals,
            &lbrr,
        )?;

        let mut reconstructed = Vec::with_capacity(total_len);
        let mut voiced = false;
        for m in &mid_frames {
            reconstructed.extend_from_slice(&m.reconstructed);
            voiced |= m.voiced;
        }
        Ok(EncodedSilkPacket {
            packet,
            reconstructed,
            voiced,
        })
    }

    /// [`Self::encode_packet`] with §3.2.5 CBR transport shaping (see
    /// [`SilkEncoderMono::encode_packet_cbr`]).
    pub fn encode_packet_cbr(
        &mut self,
        left: &[f32],
        right: &[f32],
        next_lr: Option<(f32, f32)>,
        target_bytes: usize,
    ) -> Result<EncodedSilkPacket, Error> {
        let mut out = self.encode_packet(left, right, next_lr)?;
        out.packet = crate::packet_compose::pad_packet_to(&out.packet, target_bytes)?;
        Ok(out)
    }
}

/// Map a desired linear Q16 gain to the §4.2.7.4 `log_gain` index
/// whose dequantised value is nearest in the log domain.
fn quantize_log_gain(want_gain_q16: f64) -> u8 {
    let want = want_gain_q16.max(1.0).ln();
    let mut best = (0u8, f64::MAX);
    for lg in 0..=63u8 {
        let g = silk_gains_dequant(lg) as f64;
        let err = (g.ln() - want).abs();
        if err < best.1 {
            best = (lg, err);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::OpusDecoder;

    fn rms(pcm: &[i16]) -> f64 {
        let e: f64 = pcm.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (e / pcm.len().max(1) as f64).sqrt()
    }

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

    /// Least-squares sinusoid projection at frequency `f_hz`: returns
    /// (amplitude, SNR dB of the fit). Absorbs the decoder
    /// resampler's delay/phase/gain, making it a pure "is this the
    /// input tone" metric.
    fn sine_projection(x: &[f64], f_hz: f64, rate: f64) -> (f64, f64) {
        let (mut ss, mut cc, mut sc, mut xs, mut xc) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (i, &v) in x.iter().enumerate() {
            let w = core::f64::consts::TAU * f_hz * i as f64 / rate;
            let (s, c) = w.sin_cos();
            ss += s * s;
            cc += c * c;
            sc += s * c;
            xs += v * s;
            xc += v * c;
        }
        let det = ss * cc - sc * sc;
        if det.abs() < 1e-9 {
            return (0.0, 0.0);
        }
        let a = (xs * cc - xc * sc) / det;
        let b = (xc * ss - xs * sc) / det;
        let mut sig = 0.0;
        let mut err = 0.0;
        for (i, &v) in x.iter().enumerate() {
            let w = core::f64::consts::TAU * f_hz * i as f64 / rate;
            let fit = a * w.sin() + b * w.cos();
            sig += fit * fit;
            err += (v - fit) * (v - fit);
        }
        let amp = (a * a + b * b).sqrt();
        if err == 0.0 {
            return (amp, 120.0);
        }
        (amp, 10.0 * (sig / err).log10())
    }

    /// End-to-end: encode a 400 Hz sine at WB from PCM alone, decode
    /// the packets with the real streaming OpusDecoder, and require
    /// the 48 kHz output to BE that sine (projection SNR), plus a
    /// healthy internal-rate closed-loop SNR.
    #[test]
    fn wb_sine_roundtrips_through_real_decoder() {
        let bw = Bandwidth::Wb;
        let mut enc = SilkEncoderMono::new(bw).unwrap();
        let flen = enc.frame_samples();
        assert_eq!(flen, 320);
        let fs = 16_000.0f64;
        let f = 400.0f64;

        let mut dec = OpusDecoder::new();
        let mut decoded_48k: Vec<f64> = Vec::new();
        let mut internal_snrs = Vec::new();
        for pkt_idx in 0..12 {
            let pcm: Vec<f32> = (0..flen)
                .map(|i| {
                    let t = (pkt_idx * flen + i) as f64 / fs;
                    (0.3 * (core::f64::consts::TAU * f * t).sin()) as f32
                })
                .collect();
            let out = enc.encode_packet(&pcm).unwrap();
            internal_snrs.push(snr_db(&pcm, &out.reconstructed));

            let audio = dec.decode_packet(&out.packet).unwrap();
            assert_eq!(audio.channels, 1);
            assert_eq!(audio.sample_rate_hz, 48_000);
            decoded_48k.extend(audio.pcm.iter().map(|&v| v as f64 / 32768.0));
        }

        // Steady-state packets (skip 2 warmup frames) must track the
        // input closely at the internal rate.
        let steady = &internal_snrs[2..];
        let avg = steady.iter().sum::<f64>() / steady.len() as f64;
        assert!(avg > 12.0, "internal-rate SNR too low: {avg:.1} dB");

        // The 48 kHz decode must BE the 400 Hz tone.
        let tail = &decoded_48k[decoded_48k.len() / 3..];
        let (_, proj) = sine_projection(tail, f, 48_000.0);
        assert!(proj > 10.0, "48 kHz sine projection SNR: {proj:.1} dB");
    }

    /// NB sine: same end-to-end path at 8 kHz internal rate.
    #[test]
    fn nb_sine_roundtrips_through_real_decoder() {
        let bw = Bandwidth::Nb;
        let mut enc = SilkEncoderMono::new(bw).unwrap();
        let flen = enc.frame_samples();
        assert_eq!(flen, 160);
        let fs = 8_000.0f64;
        let f = 220.0f64;

        let mut dec = OpusDecoder::new();
        let mut decoded_48k: Vec<f64> = Vec::new();
        for pkt_idx in 0..10 {
            let pcm: Vec<f32> = (0..flen)
                .map(|i| {
                    let t = (pkt_idx * flen + i) as f64 / fs;
                    (0.25 * (core::f64::consts::TAU * f * t).sin()) as f32
                })
                .collect();
            let out = enc.encode_packet(&pcm).unwrap();
            let audio = dec.decode_packet(&out.packet).unwrap();
            decoded_48k.extend(audio.pcm.iter().map(|&v| v as f64 / 32768.0));
        }
        let tail = &decoded_48k[decoded_48k.len() / 3..];
        let (_, proj) = sine_projection(tail, f, 48_000.0);
        assert!(proj > 10.0, "48 kHz sine projection SNR: {proj:.1} dB");
    }

    /// Speech-like signal — a glottal-style pulse train through a
    /// one-pole resonator (rich harmonics an order-16 LPC cannot
    /// fully whiten, unlike a few pure sinusoids) — must classify
    /// voiced, track well, and decode cleanly.
    #[test]
    fn pulse_train_encodes_voiced_and_decodes() {
        let bw = Bandwidth::Wb;
        let mut enc = SilkEncoderMono::new(bw).unwrap();
        let flen = enc.frame_samples();
        let period = 100usize; // 160 Hz at 16 kHz

        let mut dec = OpusDecoder::new();
        let mut any_voiced = false;
        let mut snrs = Vec::new();
        let mut lp = 0.0f64; // one-pole resonator state
        let mut sample_idx = 0usize;
        for _pkt in 0..8 {
            let pcm: Vec<f32> = (0..flen)
                .map(|_| {
                    let pulse = if sample_idx % period == 0 { 1.0 } else { 0.0 };
                    lp = 0.75 * lp + 0.25 * pulse;
                    sample_idx += 1;
                    (0.5 * lp) as f32
                })
                .collect();
            let out = enc.encode_packet(&pcm).unwrap();
            any_voiced |= out.voiced;
            snrs.push(snr_db(&pcm, &out.reconstructed));
            dec.decode_packet(&out.packet).unwrap();
        }
        assert!(any_voiced, "pulse train never classified voiced");
        let steady = &snrs[2..];
        let avg = steady.iter().sum::<f64>() / steady.len() as f64;
        assert!(avg > 10.0, "pulse-train tracking SNR too low: {avg:.1} dB");
    }

    /// 10 ms packets (one 2-subframe SILK frame, §4.2.7.5.5 factor not
    /// stored): NB sine end-to-end through the streaming decoder with
    /// the §3 sample count and the tone preserved.
    #[test]
    fn nb_sine_10ms_packets_roundtrip() {
        let bw = Bandwidth::Nb;
        let mut enc = SilkEncoderMono::with_packet_duration(bw, 100).unwrap();
        let flen = enc.frame_samples();
        assert_eq!(flen, 80); // 10 ms at 8 kHz
        let fs = 8_000.0f64;
        let f = 220.0f64;

        let mut dec = OpusDecoder::new();
        let mut decoded_48k: Vec<f64> = Vec::new();
        for pkt_idx in 0..20 {
            let pcm: Vec<f32> = (0..flen)
                .map(|i| {
                    let t = (pkt_idx * flen + i) as f64 / fs;
                    (0.3 * (core::f64::consts::TAU * f * t).sin()) as f32
                })
                .collect();
            let out = enc.encode_packet(&pcm).unwrap();
            let audio = dec.decode_packet(&out.packet).unwrap();
            assert_eq!(audio.channels, 1);
            // §3.1: a 10 ms packet is 480 samples at 48 kHz.
            assert_eq!(audio.pcm.len(), 480);
            decoded_48k.extend(audio.pcm.iter().map(|&v| v as f64 / 32768.0));
        }
        let tail = &decoded_48k[decoded_48k.len() / 3..];
        let (_, proj) = sine_projection(tail, f, 48_000.0);
        assert!(proj > 8.0, "48 kHz sine projection SNR: {proj:.1} dB");
    }

    /// 10 ms stereo packets (with FEC on, exercising the 10 ms LBRR
    /// path too) decode with the right shape on the real decoder.
    #[test]
    fn stereo_10ms_packets_with_fec_decode() {
        let bw = Bandwidth::Wb;
        let mut enc = SilkEncoderStereo::with_packet_duration(bw, 100).unwrap();
        enc.set_fec(true);
        let flen = enc.frame_samples();
        assert_eq!(flen, 160); // 10 ms at 16 kHz, per channel
        let fs = 16_000.0f64;

        let gen = |i: usize| -> (f32, f32) {
            let t = i as f64 / fs;
            let s = (core::f64::consts::TAU * 500.0 * t).sin();
            ((0.35 * s) as f32, (0.15 * s) as f32)
        };
        let mut dec = OpusDecoder::new();
        for pkt_idx in 0..8 {
            let mut left = Vec::with_capacity(flen);
            let mut right = Vec::with_capacity(flen);
            for i in 0..flen {
                let (l, r) = gen(pkt_idx * flen + i);
                left.push(l);
                right.push(r);
            }
            let next = gen((pkt_idx + 1) * flen);
            let out = enc.encode_packet(&left, &right, Some(next)).unwrap();
            let audio = dec.decode_packet(&out.packet).unwrap();
            assert_eq!(audio.channels, 2);
            assert_eq!(audio.pcm.len(), 2 * 480);
        }
    }

    /// 40 ms multi-frame packets (two §4.2.2 SILK frames per packet)
    /// from PCM: the intra-packet delta-gain / relative-lag threading
    /// must decode end-to-end on the real streaming decoder with the
    /// §3 sample count and the input tone preserved.
    #[test]
    fn wb_sine_40ms_multiframe_roundtrips() {
        let bw = Bandwidth::Wb;
        let mut enc = SilkEncoderMono::with_packet_duration(bw, 400).unwrap();
        let flen = enc.frame_samples();
        assert_eq!(flen, 640); // 40 ms at 16 kHz
        let fs = 16_000.0f64;
        let f = 400.0f64;

        let mut dec = OpusDecoder::new();
        let mut decoded_48k: Vec<f64> = Vec::new();
        for pkt_idx in 0..6 {
            let pcm: Vec<f32> = (0..flen)
                .map(|i| {
                    let t = (pkt_idx * flen + i) as f64 / fs;
                    (0.3 * (core::f64::consts::TAU * f * t).sin()) as f32
                })
                .collect();
            let out = enc.encode_packet(&pcm).unwrap();
            let audio = dec.decode_packet(&out.packet).unwrap();
            assert_eq!(audio.channels, 1);
            // §3.1: a 40 ms packet is 1920 samples at 48 kHz.
            assert_eq!(audio.pcm.len(), 1920);
            decoded_48k.extend(audio.pcm.iter().map(|&v| v as f64 / 32768.0));
        }
        let tail = &decoded_48k[decoded_48k.len() / 3..];
        let (_, proj) = sine_projection(tail, f, 48_000.0);
        assert!(proj > 10.0, "48 kHz sine projection SNR: {proj:.1} dB");
    }

    /// 60 ms multi-frame packets on a voiced pulse train: successive
    /// voiced frames inside one packet exercise the §4.2.7.6.1
    /// relative-lag coding; the packets must decode and track.
    #[test]
    fn mb_60ms_multiframe_voiced_pulse_train() {
        let bw = Bandwidth::Mb;
        let mut enc = SilkEncoderMono::with_packet_duration(bw, 600).unwrap();
        let flen = enc.frame_samples();
        assert_eq!(flen, 720); // 60 ms at 12 kHz
        let period = 80usize; // 150 Hz at 12 kHz

        let mut dec = OpusDecoder::new();
        let mut any_voiced = false;
        let mut snrs = Vec::new();
        let mut lp = 0.0f64;
        let mut sample_idx = 0usize;
        for _pkt in 0..4 {
            let pcm: Vec<f32> = (0..flen)
                .map(|_| {
                    let pulse = if sample_idx % period == 0 { 1.0 } else { 0.0 };
                    lp = 0.75 * lp + 0.25 * pulse;
                    sample_idx += 1;
                    (0.5 * lp) as f32
                })
                .collect();
            let out = enc.encode_packet(&pcm).unwrap();
            any_voiced |= out.voiced;
            snrs.push(snr_db(&pcm, &out.reconstructed));
            let audio = dec.decode_packet(&out.packet).unwrap();
            // §3.1: a 60 ms packet is 2880 samples at 48 kHz.
            assert_eq!(audio.pcm.len(), 2880);
        }
        assert!(any_voiced, "pulse train never classified voiced");
        let steady = &snrs[1..];
        let avg = steady.iter().sum::<f64>() / steady.len() as f64;
        assert!(avg > 8.0, "multi-frame tracking SNR too low: {avg:.1} dB");
    }

    /// The per-frame §4.2.3 VAD flags come from the signal: a 60 ms
    /// packet whose middle 20 ms interval is silent must code that
    /// frame INACTIVE (VAD bit clear) while the flanking tone frames
    /// stay active — verified by decoding the packet's §4.2.3 header
    /// bits directly.
    #[test]
    fn silent_interval_codes_inactive_vad_flag() {
        use crate::range_decoder::RangeDecoder;
        use crate::silk_header::SilkHeaderBits;

        let bw = Bandwidth::Nb;
        let mut enc = SilkEncoderMono::with_packet_duration(bw, 600).unwrap();
        let flen20 = 160usize;
        let mut pcm = vec![0.0f32; 3 * flen20];
        for (k, slot) in pcm.iter_mut().enumerate() {
            if k / flen20 != 1 {
                *slot = 0.3 * ((k as f32) * 0.5).sin();
            }
        }
        let out = enc.encode_packet(&pcm).unwrap();

        // Parse the packet: TOC byte + one code-0 frame; the §4.2.3
        // header VAD bits are the first symbols of the SILK payload.
        let body = &out.packet[1..];
        let mut rd = RangeDecoder::new(body);
        let header = SilkHeaderBits::decode(&mut rd, 3, false).unwrap();
        assert!(header.mid_vad(0), "tone interval 0 must be active");
        assert!(!header.mid_vad(1), "silent interval 1 must be inactive");
        assert!(header.mid_vad(2), "tone interval 2 must be active");

        // And the packet still decodes to real PCM.
        let mut dec = OpusDecoder::new();
        let audio = dec.decode_packet(&out.packet).unwrap();
        assert_eq!(audio.pcm.len(), 2880);
    }

    /// Loud → silence → loud transitions: the cross-packet §4.2.7.4
    /// gain-clamp floor must keep every packet decodable with no
    /// encoder/decoder divergence blowup afterwards.
    #[test]
    fn gain_transitions_stay_stream_consistent() {
        let bw = Bandwidth::Mb;
        let mut enc = SilkEncoderMono::new(bw).unwrap();
        let flen = enc.frame_samples();
        let fs = 12_000.0f64;

        let mut dec = OpusDecoder::new();
        let mut last_snr = 0.0f64;
        for pkt_idx in 0..12 {
            let amp = match pkt_idx % 4 {
                0 | 1 => 0.4f64,
                2 => 0.0,
                _ => 0.4,
            };
            let pcm: Vec<f32> = (0..flen)
                .map(|i| {
                    let t = (pkt_idx * flen + i) as f64 / fs;
                    (amp * (core::f64::consts::TAU * 300.0 * t).sin()) as f32
                })
                .collect();
            let out = enc.encode_packet(&pcm).unwrap();
            dec.decode_packet(&out.packet).unwrap();
            if pkt_idx == 11 {
                last_snr = snr_db(&pcm, &out.reconstructed);
            }
        }
        assert!(
            last_snr > 8.0,
            "post-transition tracking degraded: {last_snr:.1} dB"
        );
    }

    /// Stereo end-to-end: an amplitude-panned 350 Hz sine (L = 2R)
    /// through SilkEncoderStereo decodes on the real OpusDecoder to a
    /// stereo 48 kHz signal that is that tone on both channels with
    /// the panning preserved.
    #[test]
    fn stereo_panned_sine_roundtrips_through_real_decoder() {
        let bw = Bandwidth::Wb;
        let mut enc = SilkEncoderStereo::new(bw).unwrap();
        let flen = enc.frame_samples();
        let fs = 16_000.0f64;
        let f = 350.0f64;

        let gen = |i: usize| -> (f32, f32) {
            let t = i as f64 / fs;
            let s = (core::f64::consts::TAU * f * t).sin();
            ((0.4 * s) as f32, (0.2 * s) as f32)
        };

        let mut dec = OpusDecoder::new();
        let mut l48: Vec<f64> = Vec::new();
        let mut r48: Vec<f64> = Vec::new();
        for pkt_idx in 0..12 {
            let mut left = Vec::with_capacity(flen);
            let mut right = Vec::with_capacity(flen);
            for i in 0..flen {
                let (l, r) = gen(pkt_idx * flen + i);
                left.push(l);
                right.push(r);
            }
            let next = gen((pkt_idx + 1) * flen);
            let out = enc.encode_packet(&left, &right, Some(next)).unwrap();
            let audio = dec.decode_packet(&out.packet).unwrap();
            assert_eq!(audio.channels, 2);
            for ch in audio.pcm.chunks_exact(2) {
                l48.push(ch[0] as f64 / 32768.0);
                r48.push(ch[1] as f64 / 32768.0);
            }
        }

        let tail_l = &l48[l48.len() / 3..];
        let tail_r = &r48[r48.len() / 3..];
        let (amp_l, proj_l) = sine_projection(tail_l, f, 48_000.0);
        let (amp_r, proj_r) = sine_projection(tail_r, f, 48_000.0);
        assert!(proj_l > 8.0, "left projection SNR: {proj_l:.1} dB");
        assert!(proj_r > 8.0, "right projection SNR: {proj_r:.1} dB");
        // Panning preserved: L ≈ 2R within 25%.
        let ratio = amp_l / amp_r.max(1e-9);
        assert!(
            (ratio - 2.0).abs() < 0.5,
            "panning ratio {ratio:.2} (amps {amp_l:.3}/{amp_r:.3})"
        );
    }

    /// LBRR from PCM (mono): with FEC on, every packet after the first
    /// carries the previous packet's re-encode; dropping a packet and
    /// recovering it via `decode_packet_fec` on the NEXT packet yields
    /// real (tone-tracking) audio, and the stream continues cleanly.
    #[test]
    fn mono_fec_recovers_lost_packet() {
        use crate::decoder::FecDecodeStatus;
        use crate::range_decoder::RangeDecoder;
        use crate::silk_header::SilkHeaderBits;

        let bw = Bandwidth::Wb;
        let mut enc = SilkEncoderMono::new(bw).unwrap();
        enc.set_fec(true);
        let flen = enc.frame_samples();
        let fs = 16_000.0f64;
        let f = 400.0f64;

        let mut packets = Vec::new();
        for pkt_idx in 0..8 {
            let pcm: Vec<f32> = (0..flen)
                .map(|i| {
                    let t = (pkt_idx * flen + i) as f64 / fs;
                    (0.3 * (core::f64::consts::TAU * f * t).sin()) as f32
                })
                .collect();
            packets.push(enc.encode_packet(&pcm).unwrap().packet);
        }

        // Every packet after the first must carry the §4.2.4 LBRR flag.
        for (idx, pkt) in packets.iter().enumerate() {
            let mut rd = RangeDecoder::new(&pkt[1..]);
            let header = SilkHeaderBits::decode(&mut rd, 1, false).unwrap();
            assert_eq!(
                header.mid_has_lbrr(0),
                idx > 0,
                "packet {idx} LBRR flag wrong"
            );
        }

        // Lose packet 3; recover it from packet 4's LBRR.
        let mut dec = OpusDecoder::new();
        let mut out48: Vec<f64> = Vec::new();
        for idx in 0..8 {
            if idx == 3 {
                continue;
            }
            if idx == 4 {
                let rec = dec.decode_packet_fec(&packets[4]).unwrap();
                assert_eq!(rec.status, FecDecodeStatus::Recovered);
                assert_eq!(rec.pcm.len(), 960);
                // The recovered interval must be real signal, not the
                // silence fallback.
                assert!(rms(&rec.pcm) > 500.0, "recovered rms {}", rms(&rec.pcm));
                out48.extend(rec.pcm.iter().map(|&v| v as f64 / 32768.0));
            }
            let audio = dec.decode_packet(&packets[idx]).unwrap();
            out48.extend(audio.pcm.iter().map(|&v| v as f64 / 32768.0));
        }
        // The whole healed stream is still the 400 Hz tone.
        let tail = &out48[out48.len() / 3..];
        let (_, proj) = sine_projection(tail, f, 48_000.0);
        assert!(proj > 8.0, "healed-stream projection SNR: {proj:.1} dB");
    }

    /// LBRR from PCM (stereo, 40 ms packets): the redundant copy spans
    /// both channels and every interval; recovery succeeds on the real
    /// decoder.
    #[test]
    fn stereo_fec_recovers_lost_packet() {
        use crate::decoder::FecDecodeStatus;

        let bw = Bandwidth::Nb;
        let mut enc = SilkEncoderStereo::with_packet_duration(bw, 400).unwrap();
        enc.set_fec(true);
        let flen = enc.frame_samples();
        let fs = 8_000.0f64;
        let f = 300.0f64;

        let gen = |i: usize| -> (f32, f32) {
            let t = i as f64 / fs;
            let s = (core::f64::consts::TAU * f * t).sin();
            ((0.4 * s) as f32, (0.2 * s) as f32)
        };
        let mut packets = Vec::new();
        for pkt_idx in 0..6 {
            let mut left = Vec::with_capacity(flen);
            let mut right = Vec::with_capacity(flen);
            for i in 0..flen {
                let (l, r) = gen(pkt_idx * flen + i);
                left.push(l);
                right.push(r);
            }
            let next = gen((pkt_idx + 1) * flen);
            packets.push(enc.encode_packet(&left, &right, Some(next)).unwrap().packet);
        }

        let mut dec = OpusDecoder::new();
        for idx in 0..6 {
            if idx == 2 {
                continue; // lost
            }
            if idx == 3 {
                let rec = dec.decode_packet_fec(&packets[3]).unwrap();
                assert_eq!(rec.status, FecDecodeStatus::Recovered);
                assert_eq!(rec.channels, 2);
                assert_eq!(rec.pcm.len(), 2 * 1920);
                assert!(rms(&rec.pcm) > 300.0, "recovered rms {}", rms(&rec.pcm));
            }
            let audio = dec.decode_packet(&packets[idx]).unwrap();
            assert_eq!(audio.channels, 2);
        }
    }

    /// CBR shaping: every packet lands at exactly the target size and
    /// decodes to the same PCM as the unpadded VBR stream.
    #[test]
    fn cbr_packets_have_exact_size_and_identical_decode() {
        let bw = Bandwidth::Nb;
        let mut enc_vbr = SilkEncoderMono::new(bw).unwrap();
        let mut enc_cbr = SilkEncoderMono::new(bw).unwrap();
        let flen = enc_vbr.frame_samples();
        let target = 400usize;

        let mut dec_vbr = OpusDecoder::new();
        let mut dec_cbr = OpusDecoder::new();
        for pkt_idx in 0..5 {
            let pcm: Vec<f32> = (0..flen)
                .map(|i| {
                    let t = (pkt_idx * flen + i) as f64 / 8_000.0;
                    (0.3 * (core::f64::consts::TAU * 250.0 * t).sin()) as f32
                })
                .collect();
            let v = enc_vbr.encode_packet(&pcm).unwrap();
            let c = enc_cbr.encode_packet_cbr(&pcm, target).unwrap();
            assert_eq!(c.packet.len(), target, "packet {pkt_idx}");
            assert!(v.packet.len() < target, "target must leave headroom");
            let out_v = dec_vbr.decode_packet(&v.packet).unwrap();
            let out_c = dec_cbr.decode_packet(&c.packet).unwrap();
            assert_eq!(out_v.pcm, out_c.pcm, "packet {pkt_idx}");
        }

        // A target below the compressed size is rejected.
        let pcm: Vec<f32> = (0..flen).map(|i| 0.4 * (i as f32 * 0.7).sin()).collect();
        assert!(enc_cbr.encode_packet_cbr(&pcm, 4).is_err());
    }

    /// FEC off ⇒ byte-identical to the FEC-on encoder's FIRST packet
    /// (which has no previous packet to protect), and no LBRR flag on
    /// later packets.
    #[test]
    fn fec_off_packets_carry_no_lbrr() {
        use crate::range_decoder::RangeDecoder;
        use crate::silk_header::SilkHeaderBits;

        let bw = Bandwidth::Nb;
        let mut enc_plain = SilkEncoderMono::new(bw).unwrap();
        let mut enc_fec = SilkEncoderMono::new(bw).unwrap();
        enc_fec.set_fec(true);
        let flen = enc_plain.frame_samples();
        let pcm: Vec<f32> = (0..flen).map(|i| 0.2 * (i as f32 * 0.3).sin()).collect();

        let p_plain = enc_plain.encode_packet(&pcm).unwrap().packet;
        let p_fec = enc_fec.encode_packet(&pcm).unwrap().packet;
        assert_eq!(p_plain, p_fec, "first FEC packet has nothing to protect");

        let p2 = enc_plain.encode_packet(&pcm).unwrap().packet;
        let mut rd = RangeDecoder::new(&p2[1..]);
        let header = SilkHeaderBits::decode(&mut rd, 1, false).unwrap();
        assert!(!header.mid_has_lbrr(0));
    }

    /// Stereo 40 ms multi-frame packets: two §4.2.2 intervals per
    /// packet, each with its own §4.2.7.1 weights and §4.2.7.2
    /// decision, decoding on the real streaming decoder with panning
    /// preserved.
    #[test]
    fn stereo_40ms_multiframe_roundtrips() {
        let bw = Bandwidth::Wb;
        let mut enc = SilkEncoderStereo::with_packet_duration(bw, 400).unwrap();
        let flen = enc.frame_samples();
        assert_eq!(flen, 640); // 40 ms at 16 kHz, per channel
        let fs = 16_000.0f64;
        let f = 350.0f64;

        let gen = |i: usize| -> (f32, f32) {
            let t = i as f64 / fs;
            let s = (core::f64::consts::TAU * f * t).sin();
            ((0.4 * s) as f32, (0.2 * s) as f32)
        };

        let mut dec = OpusDecoder::new();
        let mut l48: Vec<f64> = Vec::new();
        let mut r48: Vec<f64> = Vec::new();
        for pkt_idx in 0..6 {
            let mut left = Vec::with_capacity(flen);
            let mut right = Vec::with_capacity(flen);
            for i in 0..flen {
                let (l, r) = gen(pkt_idx * flen + i);
                left.push(l);
                right.push(r);
            }
            let next = gen((pkt_idx + 1) * flen);
            let out = enc.encode_packet(&left, &right, Some(next)).unwrap();
            let audio = dec.decode_packet(&out.packet).unwrap();
            assert_eq!(audio.channels, 2);
            // §3.1: 40 ms stereo = 1920 samples per channel at 48 kHz.
            assert_eq!(audio.pcm.len(), 2 * 1920);
            for ch in audio.pcm.chunks_exact(2) {
                l48.push(ch[0] as f64 / 32768.0);
                r48.push(ch[1] as f64 / 32768.0);
            }
        }

        let tail_l = &l48[l48.len() / 3..];
        let tail_r = &r48[r48.len() / 3..];
        let (amp_l, proj_l) = sine_projection(tail_l, f, 48_000.0);
        let (amp_r, proj_r) = sine_projection(tail_r, f, 48_000.0);
        assert!(proj_l > 8.0, "left projection SNR: {proj_l:.1} dB");
        assert!(proj_r > 8.0, "right projection SNR: {proj_r:.1} dB");
        let ratio = amp_l / amp_r.max(1e-9);
        assert!(
            (ratio - 2.0).abs() < 0.5,
            "panning ratio {ratio:.2} (amps {amp_l:.3}/{amp_r:.3})"
        );
    }

    /// Stereo 60 ms packets whose side channel dies mid-packet: the
    /// per-interval §4.2.7.2 decision must mix coded-side and mid-only
    /// intervals inside ONE packet and still decode.
    #[test]
    fn stereo_60ms_mixed_midonly_intervals_decode() {
        let bw = Bandwidth::Nb;
        let mut enc = SilkEncoderStereo::with_packet_duration(bw, 600).unwrap();
        let flen = enc.frame_samples();
        let flen20 = flen / 3;

        let mut dec = OpusDecoder::new();
        for pkt_idx in 0..3 {
            let mut left = Vec::with_capacity(flen);
            let mut right = Vec::with_capacity(flen);
            for i in 0..flen {
                let s = 0.3 * (((pkt_idx * flen + i) as f32) * 0.4).sin();
                // Interval 1 of each packet: identical channels (zero
                // side); intervals 0 and 2: panned (live side).
                if i / flen20 == 1 {
                    left.push(s);
                    right.push(s);
                } else {
                    left.push(s);
                    right.push(0.5 * s);
                }
            }
            let out = enc.encode_packet(&left, &right, None).unwrap();
            let audio = dec.decode_packet(&out.packet).unwrap();
            assert_eq!(audio.channels, 2);
            assert_eq!(audio.pcm.len(), 2 * 2880);
        }
    }

    /// Identical L and R (zero side signal): the encoder must take the
    /// §4.2.7.2 mid-only path and still decode to both channels.
    #[test]
    fn stereo_identical_channels_use_mid_only() {
        let bw = Bandwidth::Nb;
        let mut enc = SilkEncoderStereo::new(bw).unwrap();
        let flen = enc.frame_samples();

        let mut dec = OpusDecoder::new();
        for pkt_idx in 0..4 {
            let pcm: Vec<f32> = (0..flen)
                .map(|i| 0.3 * (((pkt_idx * flen + i) as f32) * 0.2).sin())
                .collect();
            let out = enc.encode_packet(&pcm, &pcm, None).unwrap();
            let audio = dec.decode_packet(&out.packet).unwrap();
            assert_eq!(audio.channels, 2);
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert!(SilkEncoderMono::new(Bandwidth::Swb).is_err());
        assert!(SilkEncoderStereo::new(Bandwidth::Fb).is_err());
        let mut enc = SilkEncoderMono::new(Bandwidth::Wb).unwrap();
        assert!(enc.encode_packet(&[0.0; 100]).is_err());
        let mut senc = SilkEncoderStereo::new(Bandwidth::Wb).unwrap();
        assert!(senc.encode_packet(&[0.0; 100], &[0.0; 100], None).is_err());
    }

    /// reset() returns the encoder to a decoder-reset-compatible
    /// state: a fresh decoder accepts the first packet after reset.
    #[test]
    fn reset_realigns_with_fresh_decoder() {
        let bw = Bandwidth::Nb;
        let mut enc = SilkEncoderMono::new(bw).unwrap();
        let flen = enc.frame_samples();
        let pcm: Vec<f32> = (0..flen).map(|i| 0.2 * (i as f32 * 0.3).sin()).collect();
        for _ in 0..3 {
            enc.encode_packet(&pcm).unwrap();
        }
        enc.reset();
        let out = enc.encode_packet(&pcm).unwrap();
        let mut dec = OpusDecoder::new();
        assert!(dec.decode_packet(&out.packet).is_ok());
    }
}
