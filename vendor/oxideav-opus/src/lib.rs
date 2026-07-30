//! # oxideav-opus
//!
//! **Status:** orphan-rebuild scaffold (post 2026-05-20 audit).
//!
//! The prior implementation was retired under the workspace clean-room
//! policy. The crate is being re-implemented from scratch against
//! RFC 6716 + RFC 8251 + RFC 7587 + RFC 7845 using only material under
//! `docs/` and black-box validator binaries (`opusdec` / `opusenc`).
//!
//! ## Current surface
//!
//! * Round 1 lands the [`OpusTocByte`] parser per RFC 6716 §3.1
//!   (Table 2, Table 3, Table 4 — the 32-config × stereo-flag ×
//!   frame-count-code triple that prefixes every well-formed Opus
//!   packet).
//! * Round 2 lands the [`OpusPacket`] §3.2 frame-packing parser for
//!   all four `c` codes (code 0 single frame; code 1 two equal-size;
//!   code 2 two unequal with §3.2.1 length encoding; code 3 signalled
//!   frame count with optional VBR per-frame lengths and Opus
//!   padding). The returned slices borrow from the input packet, so
//!   the SILK / CELT decoders can be hooked up against them in a
//!   subsequent round without copying.
//! * Round 3 lands the [`RangeDecoder`] RFC 6716 §4.1 range coder —
//!   the shared entropy primitive consumed by both the SILK and CELT
//!   layers. The sibling `oxideav-celt` crate owns an independent
//!   clean-room copy of the same primitive; both crates carry their
//!   own copy until a shared low-level primitives crate exists.
//! * Round 4 lands the [`SilkFrameHeader`] decoder for RFC 6716
//!   §4.2.7.1 (stereo prediction weights), §4.2.7.2 (mid-only flag),
//!   §4.2.7.3 (frame type / quantization-offset type), and §4.2.7.5.1
//!   (normalized LSF stage-1 codebook index `I1`). These are the four
//!   structural decisions that gate every subsequent SILK stage
//!   (gains, LSF stage-2, LTP, excitation). Implemented as
//!   inverse-CDF reads against the range decoder, with the PDFs
//!   transcribed from Tables 6, 8, 9, and 14.
//! * Round 5 lands the [`SubframeGains`] decoder for RFC 6716
//!   §4.2.7.4 — per-subframe quantization gains for the two- or
//!   four-subframe SILK frame. The first subframe is **independently**
//!   coded (Table 11 signal-type-conditioned MSB PDF + Table 12
//!   uniform LSB PDF + the `max(gain_index, previous_log_gain - 16)`
//!   clamp from §4.2.7.4) when the §4.2.7.4 enumeration triggers;
//!   otherwise it's coded as a 41-symbol delta (Table 13) against
//!   the previous coded subframe gain via the `clamp(0,
//!   max(2*delta - 16, prev + delta - 4), 63)` rule. All subsequent
//!   subframes in the frame use the delta path. Output is integer
//!   `log_gain` in `0..=63`; the §4.2.7.4 tail-end `gain_Q16`
//!   conversion (`silk_log2lin`) is part of the excitation stage
//!   and not wired up yet.
//!
//! * Round 6 lands the [`LsfStage2`] decoder for RFC 6716 §4.2.7.5.2 —
//!   the per-coefficient stage-2 residual indices `I2[k] ∈ [-10, 10]`
//!   plus the backwards-prediction-undone `res_Q10[k]`. Tables 15
//!   (NB/MB) and 16 (WB) are the eight signal-shape codebooks; Tables
//!   17 (NB/MB) and 18 (WB) map `(I1, k)` → codebook letter; Table 19
//!   is the 7-cell extension PDF for the `|I2| == 4` saturation case;
//!   Table 20 holds the four prediction-weight lists (A/B for NB/MB,
//!   C/D for WB); Tables 21 (NB/MB) and 22 (WB) map `(I1, k)` →
//!   weight-list. Output stops at `res_Q10[]`.
//!
//! * Round 7 lands the [`NlsfReconstructed`] decoder for RFC 6716
//!   §4.2.7.5.3 — the stage-1 codebook lookup (Tables 23 NB/MB and
//!   24 WB carrying `cb1_Q8[]` for each `I1 ∈ 0..32`), the
//!   low-complexity Inverse Harmonic Mean Weighting (IHMW) derivation
//!   of `w_Q9[k]` from `cb1_Q8[]` via
//!   `w2_Q18[k] = (1024/(cb1_Q8[k]-cb1_Q8[k-1]) + 1024/(cb1_Q8[k+1]-cb1_Q8[k])) << 16`
//!   reduced through the spec's square-root approximation, and the
//!   final reconstructed
//!   `NLSF_Q15[k] = clamp(0, (cb1_Q8[k]<<7) + (res_Q10[k]<<14)/w_Q9[k], 32767)`.
//!   The §4.2.7.5.5 interpolation step that consumes the stabilized
//!   `NLSF_Q15[]` is deferred to a later round.
//!
//! * Round 8 lands the [`NlsfStabilized`] decoder for RFC 6716
//!   §4.2.7.5.4 — the normalized-LSF stabilization that enforces the
//!   Table 25 minimum spacing between consecutive `NLSF_Q15[]` entries.
//!   Up to 20 distortion-minimizing re-centring passes run first
//!   (finding the smallest-spacing pair, then the `min_center` /
//!   `max_center` / `center_freq` re-centring, with special handling
//!   for the implicit `NLSF_Q15[-1] = 0` and `NLSF_Q15[d_LPC] = 32768`
//!   edges), falling back after the 20th pass to a guaranteed sort +
//!   forward-`max` + backward-`min` sweep. The fallback's forward sweep
//!   uses 16-bit saturating addition per the RFC 8251 §7 erratum.
//!
//! * Round 9 lands the [`LsfInterpolated`] decoder for RFC 6716
//!   §4.2.7.5.5 — the normalized-LSF interpolation that produces the
//!   first-half coefficients of a 20 ms SILK frame. A Q2 factor
//!   `w_Q2 ∈ 0..=4` is decoded from the Table 26 PDF and
//!   `n1_Q15[k] = n0_Q15[k] + (w_Q2*(n2_Q15[k] - n0_Q15[k]) >> 2)` blends
//!   the prior coded frame's NLSF vector (`n0`) with the current
//!   stabilized one (`n2`). After a decoder reset or an uncoded regular
//!   side-channel SILK frame the factor is still decoded (to keep the
//!   range coder in sync) but discarded and `4` is used instead; for a
//!   10 ms SILK frame no factor is present at all.
//!
//! * Round 10 lands the [`LpcQ17`] core converter for RFC 6716
//!   §4.2.7.5.6 — the NLSF → LPC reconstruction (`silk_NLSF2A`). The
//!   Table 28 Q12 cosine table with linear interpolation produces the
//!   re-ordered Q17 cosine vector `c_Q17[]` per Table 27, the
//!   `silk_NLSF2A_find_poly` P/Q recurrence runs in i64 to absorb the
//!   "up to 48 bits of intermediate precision" the spec calls out, and
//!   the last-row sum/difference assembly produces the 32-bit
//!   `a32_Q17[]`.
//!
//! * Round 11 lands the §4.2.7.5.7 range-limiting bandwidth expansion
//!   ([`LpcQ17::range_limited`]) — up to 10 rounds of `silk_bwexpander_32`
//!   chirping (`maxabs_Q12 = min((maxabs_Q17 + 16) >> 5, 163838)`, chirp
//!   factor `sc_Q16[0] = 65470 - ((maxabs_Q12 - 32767) << 14) /
//!   ((maxabs_Q12 * (k+1)) >> 2)`) that shrink the raw `a32_Q17[]` until
//!   it fits a signed 16-bit Q12 value, followed by the documented
//!   post-loop Q12 saturation `clamp(-32768, (a + 16) >> 5, 32767) << 5`.
//!   The result is held in the Q17 domain for the §4.2.7.5.8
//!   prediction-gain limiting that follows.
//!
//! * Round 12 lands the §4.2.7.5.8 prediction-gain limiting
//!   ([`LpcQ17::prediction_gain_limited`] → [`LpcQ12`]) — the
//!   `silk_LPC_inverse_pred_gain_QA()` stability test (DC-response check
//!   plus the fixed-point Levinson recurrence on the Q24-widened Q12
//!   coefficients, with the `abs(a32_Q24[k][k]) > 16773022` and
//!   `inv_gain_Q30[k] < 107374` instability bounds) driving up to 16
//!   rounds of bandwidth expansion with `sc_Q16[0] = 65536 - (2<<i)`.
//!   The result is the final stable Q12 filter `a_Q12[k]` consumed by the
//!   §4.2.7.9.2 LPC synthesis.
//!
//! * Round 13 lands the §4.2.7.6 Long-Term Prediction parameters
//!   ([`LtpParameters`]) — the primary pitch lag (§4.2.7.6.1; absolute via
//!   Table 29 high part + Table 30 bandwidth-conditioned low part, or
//!   relative via the Table 31 delta with a zero-delta fallback to
//!   absolute), the pitch-contour VQ index (Table 32 PDF; Tables 33–36
//!   codebooks) that refines the primary lag into per-subframe pitch lags
//!   clamped to `[lag_min, lag_max]`, the §4.2.7.6.2 periodicity index
//!   (Table 37) and per-subframe 5-tap Q7 LTP filter taps (Table 38 PDFs;
//!   Tables 39–41 codebooks), and the §4.2.7.6.3 optional Q14 LTP scaling
//!   factor (Table 42 → `{15565, 12288, 8192}`; default `15565` when not
//!   coded). Non-voiced frames consume no LTP bits.
//!
//! * Round 14 lands the §4.2.7.7 LCG seed ([`decode_lcg_seed`]) and the
//!   §4.2.7.8 SILK excitation decoder ([`Excitation`] / [`ExcitationConfig`]).
//!   The excitation is decoded in six substeps: §4.2.7.8.1 rate level
//!   (Table 45 PDFs, one symbol per SILK frame), §4.2.7.8.2 per-shell-block
//!   pulse count (Table 46 PDFs at one of 11 rate levels; the "extra LSB"
//!   value 17 chains into rate level 9, then 10), §4.2.7.8.3 recursive
//!   pulse-location partition (16 → 8 → 4 → 2 → 1; Tables 47–50 select
//!   the split PDF by partition size + remaining pulse count),
//!   §4.2.7.8.4 per-coefficient LSB decoding (Table 51), §4.2.7.8.5
//!   sign decoding (Table 52, picked by signal type × quantization
//!   offset type × pulse count bin with 6+ saturating), and §4.2.7.8.6
//!   reconstruction with the LCG `seed' = 196314165*seed + 907633515
//!   mod 2^32` plus the Table 53 Q23 quantization offset. The result is
//!   the final Q23 excitation `e_Q23[]` consumed by the §4.2.7.9 LTP
//!   and LPC synthesis filters.
//!
//! * Round 15 lands the §4.2.7.9.2 SILK LPC synthesis filter
//!   ([`lpc_synthesis_subframe`] / [`lpc_synthesis_frame`] /
//!   [`LpcSynthState`]). The short-term predictor combines the §4.2.7.4
//!   Q16 gain, the §4.2.7.9.1 residual `res[i]`, and the §4.2.7.5.8 Q12
//!   stabilised filter `a_Q12[k]` into the unclamped `lpc[i]` and its
//!   clamped output `out[i] = clamp(-1.0, lpc[i], 1.0)`; the per-subframe
//!   `d_LPC` unclamped history is carried across subframes via the
//!   stateful [`LpcSynthState`] (cleared to zero on a decoder reset).
//!
//! * Round 16 lands the §4.2.7.9.1 SILK LTP synthesis filter
//!   ([`ltp_synthesis_subframe`] / [`ltp_synth_commit_subframe`] /
//!   [`LtpSynthState`]). Unvoiced subframes produce `res[i] = e_Q23[i] /
//!   2^23` (a normalised excitation copy). Voiced subframes go through the
//!   §4.2.7.6 5-tap Q7 LTP convolution `res[i] = e_Q23[i]/2^23 + Σ
//!   res[i - pitch_lag + 2 - k] * b_Q7[k]/128`, with the prior-subframe
//!   `out[]` history rewhitened via `4*LTP_scale_Q14/gain_Q16 *
//!   clamp(out[i] - Σ out[i-k-1] * a_Q12[k]/4096, -1, 1)` (region A) and
//!   the prior-subframe unclamped `lpc[]` rewhitened via `65536/gain_Q16 *
//!   (lpc[i] - Σ lpc[i-k-1] * a_Q12[k]/4096)` (region B). `out_end` and
//!   the effective `LTP_scale_Q14` (= 16384 fresh-LPC override) follow the
//!   §4.2.7.9.1 third/fourth-subframe LSF-interpolation-split branch. The
//!   stateful [`LtpSynthState`] carries 306 samples of out[] and 256
//!   samples of lpc[] history (the spec-stated WB worst cases) across
//!   subframes and across SILK frame boundaries, cleared to zero on a
//!   decoder reset per §4.5.2.
//!
//! * Round 17 lands the §4.2.8 SILK stereo unmixing
//!   ([`stereo_ms_to_lr`] / [`StereoUnmixState`] / [`StereoWeightsQ13`] /
//!   [`StereoFrame`]) — the `silk_stereo_MS_to_LR` conversion that turns
//!   the decoded mid/side `out[]` signals into left/right. The side
//!   channel is predicted from a low-passed mid term
//!   (`p0 = (mid[i-2] + 2*mid[i-1] + mid[i]) / 4`) and the unfiltered
//!   one-sample-delayed mid (`mid[i-1]`) via the §4.2.7.1 Q13 weights:
//!   `left[i] = clamp(-1, (1+w1)*mid[i-1] + side[i-1] + w0*p0, 1)` and
//!   `right[i] = clamp(-1, (1-w1)*mid[i-1] - side[i-1] - w0*p0, 1)`. The
//!   first `n1` samples (64 NB / 96 MB / 128 WB) interpolate the weights
//!   from the previous frame's `(prev_w0_Q13, prev_w1_Q13)` to the
//!   current frame's; the remainder use the current weights. An uncoded
//!   side channel (§4.2.7.2) is treated as all-zero. The two trailing
//!   mid samples, one trailing side sample, and previous-frame weights
//!   carry across the frame boundary via [`StereoUnmixState`], cleared
//!   to zero on a decoder reset per §4.2.8.
//!
//! * Round 19 lands the §4.2.9 SILK resampler delay budget and the
//!   internal-vs-output sample-rate accounting ([`silk_resampler_delay_ms`] /
//!   [`silk_resampler_delay_samples_at`] / [`silk_internal_rate_hz`] /
//!   [`silk_frame_samples_internal`] / [`silk_frame_samples_at_output`] /
//!   [`is_supported_output_rate`] / [`SUPPORTED_OUTPUT_RATES_HZ`]).
//!   The §4.2.9 resampler itself is non-normative ("a decoder can use
//!   any method it wants"); what IS normative is the Table 54 maximum
//!   delay allocation (NB = 0.538 ms, MB = 0.692 ms, WB = 0.706 ms) so
//!   the encoder can apply a matching pre-delay to keep SILK and CELT
//!   aligned across a §4.5 mode switch. This module owns Table 54 plus
//!   the implied SILK internal rates (NB = 8000 Hz, MB = 12000 Hz,
//!   WB = 16000 Hz) and the §4.2.9 supported output rates (8 / 12 / 16 /
//!   24 / 48 kHz). SWB and FB never reach the §4.2.9 SILK stage and are
//!   rejected with `None`. The module also carries the actual §4.2.9
//!   conversion, [`SilkUpsampler`]: the reference decoder's fixed-point
//!   resampler to 48 kHz (per-rate delay compensation, 2× allpass
//!   upsampling, fractional-phase 8-tap FIR interpolation, with the
//!   RFC 8251 §5 correction), transcribed from the RFC 6716 §A
//!   reference listing so decoded SILK lands sample-exact on the
//!   reference timeline.
//!
//! * Round 18 lands the §4.2.3 SILK packet-level header bits and the
//!   §4.2.4 per-frame LBRR flags ([`SilkHeaderBits`] / [`silk_frame_count`]).
//!   For each channel (mono: 1; stereo: 2), the decoder reads N uniform
//!   `dec_bit_logp(1)` VAD bits (N = SILK-frame count from §4.2.2: 1 for
//!   10/20 ms Opus frames, 2 for 40 ms, 3 for 60 ms) followed by a single
//!   global LBRR flag. For Opus frames longer than 20 ms, each channel
//!   whose global LBRR flag is set then contributes one Table 4 symbol
//!   (`{0, 53, 53, 150}/256` for 40 ms / `{0, 41, 20, 29, 41, 15, 28,
//!   82}/256` for 60 ms) carrying a per-SILK-frame LBRR bitmap, packed
//!   LSB-to-MSB. For 10/20 ms Opus frames the global LBRR flag itself
//!   implies a single LBRR frame. Output is a [`SilkHeaderBits`]
//!   carrying the per-channel VAD bitmap, global LBRR flag, and the
//!   fully expanded per-channel × per-SILK-frame [`PerFrameLbrr`]
//!   bitmap consumed by the downstream §4.2.5 LBRR / §4.2.6 regular
//!   SILK frame loop.
//!
//! * Round 20 lands the first CELT-layer fragment ([`CeltHeaderPrefix`] /
//!   [`CeltPostFilter`]) — the §4.3, Table 56 pre-band header symbols
//!   that every CELT-bearing Opus frame opens with: `silence`
//!   (`{32767, 1}/32768`), the §4.3.7.1 pitch post-filter parameter
//!   group (logp=1 enable bit, then `octave` uniform[0,6), `period =
//!   (16<<octave) + fine_pitch - 1` from `4+octave` raw bits bounded
//!   to `15..=1022`, `gain` 3 raw bits ⇒ `G = 3*(gain_index+1)/32`,
//!   `tapset` `{2,1,1}/4`), the §4.3.1 `transient` (`{7,1}/8`), and
//!   the §4.3.2.1 `intra` (`{7,1}/8`) flag. When `silence` is set,
//!   the rest of the prefix is force-defaulted per the §4.3
//!   shortcut. This is the only Table-56 segment that fits between
//!   the SILK pipeline already wired up and the §4.3.2.1 coarse
//!   energy (#936, blocked on the Laplace decoder + `e_prob_model`
//!   table) / §4.3.3 bit allocation (#943, blocked on `cache_caps50`
//!   + `LOG2_FRAC_TABLE`) sub-pieces.
//!
//! * Round 21 lands the §3.1 / §4.2 framing dispatch ([`OpusFrameRouting`]
//!   / [`OperatingMode`] / [`SilkBandwidth`]) — the single
//!   pure-function lookup that turns an [`OpusTocByte`] into the
//!   per-Opus-frame routing decision a §4 decoder needs *before* it
//!   touches the range coder: which layer(s) are present (SILK-only /
//!   Hybrid / CELT-only), the SILK internal bandwidth (pinned to WB
//!   for Hybrid per §4.2 even when the TOC bandwidth is SWB / FB), the
//!   §4.2.2 SILK-frame count per channel (1 for 10/20 ms, 2 for 40 ms,
//!   3 for 60 ms), the §4.2.4 per-frame LBRR-flag presence gate
//!   (duration > 20 ms), and the channel-count multiplier for stereo.
//!   Codifies the dispatch decision so downstream decoders consume one
//!   `OpusFrameRouting` instead of open-coding the
//!   `(mode, bandwidth, frame_size)` switch each time.
//!
//! * Round 23 lands the §4.2.7.4 SILK gain dequantization tail
//!   ([`silk_log2lin`] / [`silk_gains_dequant`] /
//!   [`SubframeGains::dequant_q16`](crate::silk_gains::SubframeGains::dequant_q16))
//!   — the piecewise-linear approximation of `2^(inLog_Q7/128)` and the
//!   composed `log_gain ∈ 0..=63 → gain_Q16 ∈ [81920, 1_686_110_208]`
//!   mapping that the §4.2.7.9.1 LTP and §4.2.7.9.2 LPC synthesis
//!   filters consume. The two §4.2.7.4 endpoints (`log_gain = 0`
//!   ⇒ `81920` = 1.25× linear; `log_gain = 63` ⇒ `1_686_110_208` ≈
//!   25 728× linear) are pinned to the RFC text. The §4.2.7.5 NLSF
//!   stages had been deferred since round 5; this round closes that gap.
//!
//! * Round 24 lands the §4.3 CELT MDCT-band layout
//!   ([`celt_band_layout`]: [`CeltFrameSize`] + Table 55
//!   `bins_per_channel` lookups via [`celt_band_bins_per_channel`] +
//!   [`celt_band_start_hz`] / [`celt_band_stop_hz`] band-edge
//!   accessors + [`celt_band_at_hz`] reverse lookup + the §4.3
//!   "first 17 bands not coded in Hybrid mode" rule baked into
//!   [`celt_first_coded_band`] / [`HYBRID_FIRST_CODED_BAND`] + the
//!   [`celt_total_bins_per_channel`] column-sum helper). The standard
//!   non-Custom CELT layer's [`CELT_NUM_BANDS`] = 21 bands and the
//!   per-band MDCT bin counts at the four CELT frame sizes (2.5 / 5
//!   / 10 / 20 ms) are the lookup every §4.3.2 coarse-energy decoder,
//!   §4.3.3 bit allocator, §4.3.4 PVQ shape decoder, §4.3.6
//!   denormaliser, and §4.3.7 inverse-MDCT pass needs before any
//!   band-loop iteration can start.
//!
//! * Round 25 lands the §4.3.4.5 CELT TF-resolution adjustment lookup
//!   ([`celt_tf_adjust`]: Tables 60–63 [`TF_ADJ_NONTRANSIENT_SELECT0`]
//!   / [`TF_ADJ_NONTRANSIENT_SELECT1`] / [`TF_ADJ_TRANSIENT_SELECT0`]
//!   / [`TF_ADJ_TRANSIENT_SELECT1`] +
//!   [`celt_tf_adjustment`](crate::celt_tf_adjust::celt_tf_adjustment)
//!   `(frame_size, transient, tf_select, tf_change) -> i8` entry +
//!   the §4.3.1
//!   [`celt_tf_select_can_affect`](crate::celt_tf_adjust::celt_tf_select_can_affect)
//!   "tf_select is only decoded if it can have an impact on the
//!   result knowing the value of all per-band tf_change flags" gate +
//!   [`TfDirection`] classification (`Unchanged` / `IncreaseTime(N)` /
//!   `IncreaseFrequency(N)`) carrying the §4.3.4.5 Hadamard-transform
//!   level count). The §4.3.4.5 band loop downstream — gated on
//!   §4.3.2.1 coarse energy + §4.3.3 bit allocation, both still
//!   deferred — turns each per-band `tf_change[b]` bit into one of
//!   these adjustments before the §4.3.4.2 PVQ shape decoder runs.
//!
//! * Round 26 lands the §4.5.1 CELT redundancy / mode-transition side
//!   information ([`decode_redundancy`] / [`RedundancyDecision`] /
//!   [`RedundancyPosition`]) — the three-step procedure that decides
//!   whether an Opus frame embeds an extra 5 ms redundant CELT frame
//!   for a clean mode transition. §4.5.1.1 implicit signalling for
//!   SILK-only Opus frames (the 17-bit remaining-budget gate),
//!   §4.5.1.1 explicit signalling for Hybrid Opus frames (the 37-bit
//!   gate + Table 64 `{4095, 1}/4096` flag), §4.5.1.2 redundancy
//!   position (Table 65 `{1, 1}/2` uniform symbol: 0 = end-of-frame
//!   / first-frame-in-transition, 1 = start-of-frame / second-frame-
//!   in-transition), and §4.5.1.3 redundancy size (SILK-only =
//!   remaining whole bytes; Hybrid = `2 + dec_uint(256)` with the
//!   "claimed > remaining" branch routed to [`RedundancyDecision::Invalid`]
//!   per the §4.5.1.3 "stop decoding and discard" recommendation).
//!   CELT-only Opus frames bypass the §4.5.1 path entirely. This
//!   round does NOT decode the redundant CELT frame itself — that
//!   requires the §4.3.2.1 / §4.3.3 blockers (#936 / #943) — only
//!   the boundary metadata that tells the caller WHERE the redundant
//!   CELT bytes start and HOW MANY of them there are.
//!
//! * Round 28 lands the §4.5.1.4 redundant-CELT-frame decode
//!   parameters and the §4.5.1.4 cross-lap placement
//!   ([`redundant_frame_params`] / [`RedundantFrameParams`] /
//!   [`CrossLapPlacement`] / [`apply_mb_to_wb_override`] /
//!   [`REDUNDANT_FRAME_TENTHS_MS`] / [`REDUNDANT_CROSS_LAP_TENTHS_MS`])
//!   — the pure-function lookup that turns an [`OpusFrameRouting`]
//!   plus a [`RedundancyDecision`] into the four normative
//!   §4.5.1.4 facts a §4.3 CELT decoder needs to actually decode
//!   the redundant frame: "no TOC byte" (just feed the redundant
//!   bytes into the CELT decoder), 5 ms fixed duration
//!   ([`REDUNDANT_FRAME_TENTHS_MS`] = 50 tenths-ms), channel count
//!   inherited from the carrier Opus frame, and audio bandwidth
//!   inherited from the carrier with the §4.5.1.4 "MB SILK frames
//!   → WB" exception ([`apply_mb_to_wb_override`]). Also lands the
//!   §4.5.1.4 cross-lap placement decision
//!   ([`CrossLapPlacement::FirstHalfAsIs`] for [`RedundancyPosition::Beginning`]
//!   — CELT→SILK/Hybrid transitions, where the redundant CELT
//!   frame's first 2.5 ms replace the SILK/Hybrid leading 2.5 ms
//!   and the second 2.5 ms cross-lap; [`CrossLapPlacement::SecondHalfAsIs`]
//!   for [`RedundancyPosition::End`] — SILK/Hybrid→CELT
//!   transitions, where only the redundant frame's second 2.5 ms
//!   is used and that half cross-laps with the SILK/Hybrid
//!   trailing edge). The §4.3.7 power-complementary MDCT window
//!   that actually performs the cross-lap mix is gated on the
//!   undelivered §4.3.2 / §4.3.3 / §4.3.4 chain; this round owns
//!   only the placement metadata (which 2.5 ms region cross-laps,
//!   where in the carrier's sample buffer it sits).
//!
//! * Round 29 lands the §4.3.2.1 CELT coarse-energy Laplace-model
//!   parameter surface ([`celt_e_prob_model`]: [`E_PROB_MODEL`] —
//!   the 336-byte `[LM ∈ 0..4][mode ∈ {inter, intra}][band × 2]` Q8
//!   `{prob, decay}` table feeding `ec_laplace_decode` +
//!   [`EnergyPredictionMode::{Inter, Intra}`] selector driven by the
//!   §4.3.2.1 CELT header `intra` flag + [`e_prob_pair`] / [`e_prob_row`]
//!   accessors returning [`EProbPair`] / `&[u8; 42]` + the
//!   [`INTRA_PRED_ALPHA_Q15`] / [`INTRA_PRED_BETA_Q15`] / [`Q15_ONE`]
//!   intra-mode prediction-coefficient constants (`alpha = 0`,
//!   `beta = 4915 / 32768` per RFC 6716 §4.3.2.1 p. 108)). This is the
//!   parameter-surface fragment needed before the §4.3.2.1
//!   Laplace decoder + 2-D `(time, frequency)` predictor can run;
//!   the decoder itself and the per-LM inter-mode `(alpha, beta)`
//!   pair were deferred (the latter landed in round 45 —
//!   [`INTER_PRED_ALPHA_Q15`] / [`INTER_PRED_BETA_Q15`] /
//!   [`energy_pred_coef`] returning [`EnergyPredCoef`], the Q15
//!   numerators fixed by the RFC 6716 Appendix A normative
//!   reference code).
//!
//! * Round 30 lands the §4.3.3 *intensity-stereo reservation*
//!   parameter surface ([`celt_log2_frac_table`]:
//!   [`LOG2_FRAC_TABLE`] — the 24-byte Q3 (1/8-bit) conservative
//!   `log2` table feeding the §4.3.3 `intensity_rsv =
//!   LOG2_FRAC_TABLE[end − start]` reservation + [`log2_frac`] typed
//!   accessor + [`log2_frac_row`] full-row borrow + the
//!   [`Q3_BITS_PER_WHOLE_BIT`] = 8 unit-denominator constant). This
//!   is a parameter-surface piece of the §4.3.3 bit-allocation
//!   procedure; the boost / trim / anti-collapse / skip / dual-stereo
//!   reservations, the Table 57 static allocation search, the
//!   `cache_caps50` per-band maximum, and the rest of the §4.3.3
//!   allocation loop are all out of scope for this round.
//!
//! * Round 31 lands the §4.3.3 *per-band maximum-allocation* parameter
//!   surface ([`celt_cache_caps50`]: [`CACHE_CAPS50`] — the 168-byte
//!   `[LM ∈ 0..4][stereo ∈ {mono, stereo}][band ∈ 0..21]` Q0
//!   bits/sample table feeding the §4.3.3 per-band bit cap +
//!   [`CacheCapsStereo::{Mono, Stereo}`] selector + [`cache_caps_value`]
//!   / [`cache_caps_row`] accessors + [`init_caps`] /
//!   [`cap_for_band_bits`] convert-to-bits rule
//!   `cap[band] = ((cache_caps50[i] + 64) * channels * N) / 4` per
//!   RFC 6716 §4.3.3 p. 113 + [`INIT_CAPS_BIAS`] / [`INIT_CAPS_DIVISOR`]
//!   / [`INIT_CAPS_MAX_CHANNELS`] convert-rule constants). Closes the
//!   second of the two table dependencies round 24 noted for the
//!   §4.3.3 allocator (round 30 landed [`LOG2_FRAC_TABLE`]; this round
//!   lands [`CACHE_CAPS50`]). The §4.3.3 bit allocation orchestration
//!   that consumes `cap[]` (boost / trim / anti-collapse / skip /
//!   dual-stereo reservations, the Table 57 static allocation search)
//!   is still out of scope.
//!
//! * Round 32 lands the §4.3.3 *allocation trim* parameter surface
//!   ([`celt_alloc_trim`]: [`ALLOC_TRIM_PDF`] — the Table-58 PDF
//!   `{2, 2, 5, 10, 22, 46, 22, 10, 5, 2, 2}/128` and its derived
//!   [`ALLOC_TRIM_ICDF`] for [`RangeDecoder::dec_icdf`] consumption +
//!   [`ALLOC_TRIM_DEFAULT`] = 5 / [`ALLOC_TRIM_MIN`] = 0 /
//!   [`ALLOC_TRIM_MAX`] = 10 trim-integer range + the §4.3.3
//!   signalling gate `(ec_tell_frac + 48) ≤ (frame_bytes * 8 −
//!   total_boost)` in [`alloc_trim_is_signalled`] + the
//!   [`decode_alloc_trim`] wrapper that fuses the gate, the
//!   gate-fail-returns-default rule, and the [`RangeDecoder::dec_icdf`]
//!   read into one typed call). The §4.3.3 use of the trim — the
//!   per-band `trim_offsets[]` derivation that shifts the Table 57
//!   static allocation search — is still out of scope and runs at the
//!   call site of [`decode_alloc_trim`].
//!
//! * Round 33 lands the §4.3.3 *band-boost* decoder
//!   ([`celt_band_boost`]: [`decode_band_boosts`] driver +
//!   [`band_boost_quanta`] §4.3.3 `min(8*N, max(48, N))` helper +
//!   [`BandBoost`] / [`BandBoostOutcome`] per-band and full-driver
//!   outcomes carrying the §4.3.3 `total_boost` accumulator consumed
//!   by [`decode_alloc_trim`] downstream + [`DYNALLOC_LOGP_INIT`] = 6
//!   / [`DYNALLOC_LOGP_MIN`] = 2 / [`DYNALLOC_LOOP_LOGP_AFTER_FIRST`]
//!   = 1 cost constants + [`BAND_BOOST_QUANTA_FLOOR_EIGHTH_BITS`] = 48
//!   / [`BAND_BOOST_QUANTA_CEIL_MULT`] = 8 quanta-rule constants +
//!   [`BandBoostError`] caller-side bookkeeping bugs). Bridges round
//!   31's [`crate::celt_cache_caps50::cap_for_band_bits`] per-band
//!   upper bound and round 32's [`decode_alloc_trim`] gate's
//!   `total_boost` input. The §4.3.3 *use* of the per-band boost
//!   values — the §4.3.3 Table 57 static-allocation search +
//!   anti-collapse / skip / dual-stereo reservations — is the
//!   responsibility of the §4.3.3 allocator and runs at the call
//!   site of [`decode_band_boosts`].
//!
//! * Round 34 lands the §4.3.3 *reservation block*
//!   ([`celt_reservations`]: [`reserve_block`] /
//!   [`ReservationOutcome`] / [`ReservationError`] +
//!   [`ONE_BIT_EIGHTH_BITS`] = 8 /
//!   [`CONSERVATIVE_DEDUCTION_EIGHTH_BITS`] = 1 /
//!   [`ANTI_COLLAPSE_LM_MIN_EXCLUSIVE`] = 1 /
//!   [`ANTI_COLLAPSE_HEADROOM_MULT_EIGHTH_BITS`] = 8 /
//!   [`ANTI_COLLAPSE_HEADROOM_LM_OFFSET`] = 2 reservation-cost +
//!   gating constants). The §4.3.3 procedure (RFC 6716 §4.3.3, p. 114)
//!   skims four fixed-cost reservations off the top of the working
//!   `total` budget before the Table 57 static-allocation search:
//!   `anti_collapse_rsv` (8 1/8 bits iff transient && LM > 1 &&
//!   total ≥ (LM + 2) * 8), `skip_rsv` (8 1/8 bits iff total > 8 after
//!   anti-collapse), `intensity_rsv = LOG2_FRAC_TABLE[end − start]`
//!   (stereo only; reset to 0 if > total), and `dual_stereo_rsv`
//!   (8 1/8 bits iff total > 8 after intensity). The initial `total`
//!   is `frame_size_bytes * 64 − ec_tell_frac − 1` (the §4.3.3
//!   conservative `-1` deduction). Bridges round 33's `total_boost`
//!   accumulator (validated as `≤ frame_eighth − ec_tell_frac`) and
//!   round 30's [`crate::celt_log2_frac_table::log2_frac`] lookup with
//!   the §4.3.3 Table 57 static-allocation search at the consumer
//!   site. The §4.3.3 *use* of the reservations — the actual
//!   `dec_bit_logp(1)` reads of the anti-collapse / skip /
//!   dual-stereo flags and the `ec_dec_uint(end − start)` read of the
//!   intensity-stereo band — runs at the §4.3.3 allocator's consumer
//!   site once the Table 57 search produces the per-band shape
//!   allocation.
//!
//! * Round 35 lands the §4.3.3 *per-band minimum-allocation vector*
//!   ([`celt_band_thresh`]: [`band_min_thresh`] /
//!   [`compute_band_min_thresh`] / [`band_min_thresh_vec`] /
//!   [`standard_band_window`] / [`BandThreshError`] +
//!   [`BAND_THRESH_BINS_MULTIPLIER`] = 24 /
//!   [`BAND_THRESH_BINS_DIVISOR`] = 16 /
//!   [`BAND_THRESH_PER_CHANNEL_EIGHTH_BITS`] = 8 /
//!   [`BAND_THRESH_MONO_CHANNELS`] = 1 /
//!   [`BAND_THRESH_STEREO_CHANNELS`] = 2 formula constants). The
//!   §4.3.3 narrative (RFC 6716 §4.3.3, p. 115) computes a hard
//!   per-band lower bound on the shape allocation: bands whose
//!   allocation would drop below `thresh[band]` are dropped rather
//!   than coded sparsely. For each coded band `b`, with
//!   `N = celt_band_bins_per_channel(b, frame_size)` and
//!   `channels ∈ {1, 2}`, the per-band minimum is
//!   `thresh[b] = max((24 * N) / 16, 8 * channels)` in 1/8 bits — one
//!   whole bit per channel or 48 128th-bits per MDCT bin, whichever is
//!   greater. The §4.3.3 narrative is explicit that the band-size
//!   dependent term `(24 * N) / 16` is *not* scaled by the channel
//!   count (at the very low rates where this floor binds, the
//!   §4.3.3 allocator concentrates the budget on the mid channel).
//!   Bridges round 24's Table 55 band layout with the §4.3.3 Table 57
//!   static-allocation search at the consumer site (where the
//!   per-band minimum competes with the round-31 `cap[]` per-band
//!   maximum, the round-33 boosts, and the upcoming
//!   `trim_offsets[]`).
//!
//! * Round 36 lands the §4.3.3 *per-band allocation-trim offsets*
//!   ([`celt_trim_offsets`]: [`band_trim_offset`] /
//!   [`band_trim_offset_for_band`] / [`band_n_shortest`] /
//!   [`shortest_frame_size`] / [`TrimOffsetError`] +
//!   [`TRIM_OFFSETS_BIAS`] = 5 /
//!   [`TRIM_OFFSETS_NUMERATOR_SCALE`] = 8 /
//!   [`TRIM_OFFSETS_DIVISOR`] = 64 /
//!   [`TRIM_OFFSETS_WIDTH_ONE_BINS_PER_CHANNEL`] = 1 /
//!   [`TRIM_OFFSETS_WIDTH_ONE_PER_CHANNEL_EIGHTH_BITS`] = 8 /
//!   [`TRIM_OFFSETS_MONO_CHANNELS`] = 1 /
//!   [`TRIM_OFFSETS_STEREO_CHANNELS`] = 2 formula constants). The
//!   §4.3.3 narrative (RFC 6716 §4.3.3, p. 115) derives a per-band
//!   *trim-offset* vector from the round-32 `alloc_trim` index; the
//!   §4.3.3 Table 57 static-allocation search will add these offsets
//!   to the per-band budget when ranking quality columns. For each
//!   coded band `b`, with `channels ∈ {1, 2}`, `LM ∈ {0, 1, 2, 3}`,
//!   `n_shortest = celt_band_bins_per_channel(b, Ms2_5)`,
//!   `n_per_channel = celt_band_bins_per_channel(b, frame_size)`,
//!   and `remaining_bands` the band-position-dependent factor:
//!   `base = (alloc_trim - 5 - LM) * channels * n_shortest *
//!   remaining_bands * (1 << LM) * 8 / 64`, then
//!   `trim_offsets[b] = base - (8 * channels)` when
//!   `n_per_channel == 1` (width-1 bands receive greater benefit
//!   from the coarse-energy coding; the §4.3.3 narrative backs the
//!   trim off by one whole bit per channel). All arithmetic is
//!   signed; the output is in 1/8 bits. Bridges round 32's
//!   [`decode_alloc_trim`] gate, round 24's Table 55 layout, and
//!   round 35's [`band_min_thresh`] floor with the upcoming §4.3.3
//!   Table 57 static-allocation search.
//!
//! * Round 38 lands the §4.5.3 *Summary of Transitions* (Figure 18
//!   plus Figure 19) ([`celt_transitions`]: [`NormativeTransition`]
//!   with one variant per row of Figure 18 +
//!   [`RecommendedNonNormativeTransition`] with one variant per row
//!   of Figure 19 + [`BoundaryOp`] lifting the §4.5.3 figure-key
//!   markers `;` / `|` / `!` / `&` / `+` / `c` / `P` / `>` to a
//!   typed list +
//!   [`classify_normative_transition`](crate::celt_transitions::classify_normative_transition)
//!   `(prev_mode, prev_silk_bw, next_mode, next_silk_bw,
//!   redundancy_present) -> Option<NormativeTransition>` for the
//!   Figure-18 lookup +
//!   [`recommended_non_normative`](crate::celt_transitions::recommended_non_normative)
//!   `(prev_mode, prev_silk_bw, next_mode, next_silk_bw) ->
//!   Option<RecommendedNonNormativeTransition>` for the Figure-19
//!   lookup + the
//!   [`NormativeTransition::seam_operations`](crate::celt_transitions::NormativeTransition::seam_operations)
//!   and
//!   [`RecommendedNonNormativeTransition::seam_operations`](crate::celt_transitions::RecommendedNonNormativeTransition::seam_operations)
//!   accessors returning the ordered marker list at each
//!   transition seam, transcribed from the §4.5.3 figures). Closes
//!   the §4.5 chain after the round-26 §4.5.1 redundancy side
//!   information, the round-28 §4.5.1.4 cross-lap placement, and
//!   the round-27 §4.5.2 state-reset policy. The §4.5.3
//!   classifier's SILK-bandwidth split between Figure-18 rows 2
//!   (NB/MB SILK to Hybrid with R) and row 3 (WB SILK to Hybrid, no
//!   R), the symmetric Hybrid to SILK split (rows 5 and 6), and the
//!   §4.5 "audio-bandwidth change is the glitch source" reading
//!   that rules out same-bandwidth SILK to SILK from row 1 are all
//!   baked in.
//!
//! * Round 39 lands the §4.3.3 *static allocation table*
//!   ([`celt_static_alloc`]:
//!   [`STATIC_ALLOC`](crate::celt_static_alloc::STATIC_ALLOC) —
//!   the 21×11 Q5 grid `alloc[band][q]` in 1/32-bit per MDCT bin
//!   units transcribed from RFC 6716 §4.3.3 Table 57 (p. 112) +
//!   [`STATIC_ALLOC_Q_COUNT`](crate::celt_static_alloc::STATIC_ALLOC_Q_COUNT)
//!   = 11 / [`STATIC_ALLOC_Q_MIN`](crate::celt_static_alloc::STATIC_ALLOC_Q_MIN)
//!   = 0 / [`STATIC_ALLOC_Q_MAX`](crate::celt_static_alloc::STATIC_ALLOC_Q_MAX)
//!   = 10 / [`STATIC_ALLOC_TOTAL_CELLS`](crate::celt_static_alloc::STATIC_ALLOC_TOTAL_CELLS)
//!   = 231 / [`STATIC_ALLOC_RIGHT_SHIFT`](crate::celt_static_alloc::STATIC_ALLOC_RIGHT_SHIFT)
//!   = 2 / [`STATIC_ALLOC_INTERP_STEPS`](crate::celt_static_alloc::STATIC_ALLOC_INTERP_STEPS)
//!   = 64 layout / conversion constants +
//!   [`static_alloc_cell`](crate::celt_static_alloc::static_alloc_cell)
//!   `(band, q) -> u8` raw-cell lookup +
//!   [`static_alloc_row`](crate::celt_static_alloc::static_alloc_row)
//!   `(band) -> &[u8; 11]` row borrow for the §4.3.3 search's
//!   per-band quality inner loop +
//!   [`static_alloc_eighth_bits`](crate::celt_static_alloc::static_alloc_eighth_bits)
//!   `(band, q, channels, n_bins, lm) -> u32` applying the §4.3.3
//!   `channels * N * alloc[band][q] << LM >> 2` unit conversion
//!   from Q5 to Q3 (1/8-bit) per-band units +
//!   [`StaticAllocError`](crate::celt_static_alloc::StaticAllocError)).
//!   Pins the §4.3.3 invariants the allocator relies on: column 0
//!   is uniformly zero (the no-allocation floor), each row is
//!   monotone non-decreasing in `q`, and the saturation column
//!   (col 10) is `200` for bands 0..=7 and declines to `104` at
//!   band 20. Bridges the round-31 cap surface, the round-33
//!   boosts, the round-34 reservations, the round-35 minimum
//!   threshold, and the round-36 trim offsets with the §4.3.3
//!   1/64-step interpolated search the next round will land.
//!
//! * Round 40 lands the §4.3.3 *1/64-step interpolated static-allocation
//!   search* ([`celt_alloc_search`]:
//!   [`Q_FP_MAX`](crate::celt_alloc_search::Q_FP_MAX) = 640
//!   fixed-point-quality bound +
//!   [`STATIC_ALLOC_INTERP_RIGHT_SHIFT`](crate::celt_alloc_search::STATIC_ALLOC_INTERP_RIGHT_SHIFT)
//!   = 8 combined shift constant +
//!   [`QFpComponents`](crate::celt_alloc_search::QFpComponents)
//!   `(q_lo, frac)` decomposition +
//!   [`q_fp_to_components`](crate::celt_alloc_search::q_fp_to_components)
//!   `/ q_fp_from_components` invertible accessors +
//!   [`per_band_eighth_bits_at_q_fp`](crate::celt_alloc_search::per_band_eighth_bits_at_q_fp)
//!   `(band, q_fp, channels, n_bins, lm) -> u64` per-band Q3 lookup
//!   under the §4.3.3 1/64-step linear interpolation
//!   `cell_q11 = alloc[b][q_lo] * (64 - frac) + alloc[b][q_lo + 1] *
//!   frac` followed by the `(channels * N * cell_q11) << LM >> 8`
//!   unit conversion that folds the round-39 `>> 2` (Q5 → Q3) with
//!   the 1/64-step `>> 6` (Q11 → Q5) in one step +
//!   [`total_eighth_bits_at_q_fp`](crate::celt_alloc_search::total_eighth_bits_at_q_fp)
//!   `(q_fp, channels, frame_size, is_hybrid) -> u64` summing across
//!   coded bands respecting the §4.3 first-coded-band rule (`0` for
//!   CELT-only / `17` for Hybrid) +
//!   [`search_q_fp`](crate::celt_alloc_search::search_q_fp)
//!   `(budget, channels, frame_size, is_hybrid) -> AllocSearchOutcome`
//!   the §4.3.3 "highest allocation that does not exceed the number
//!   of bits remaining" linear scan returning
//!   [`AllocSearchOutcome`](crate::celt_alloc_search::AllocSearchOutcome)
//!   `{ q_fp, total_eighth_bits }` +
//!   [`AllocSearchError`](crate::celt_alloc_search::AllocSearchError)).
//!   Closes the §4.3.3 1/64-step interpolation gap round 39 noted as
//!   the next step. The orchestrated §4.3.3 allocator that consumes
//!   the search output (folding in the round-33 boosts, the round-35
//!   per-band minimum threshold, the round-31 per-band cap, and the
//!   round-36 trim offsets, then running the skip / dual-stereo /
//!   intensity-stereo flag reads) runs at the consumer site once the
//!   round-34 reservation block + this round's search are composed.
//!
//! * Round 41 lands the §4.3.4.2 *PVQ codebook-size function*
//!   ([`celt_pvq_v`]: [`pvq_codebook_size`]`(n, k) -> Result<u32,
//!   PvqVError>` evaluating the RFC 6716 §4.3.4.2 bivariate
//!   recurrence `V(N, K) = V(N - 1, K) + V(N, K - 1) + V(N - 1,
//!   K - 1)` with base cases `V(N, 0) = 1` / `V(0, K) = 0 (K != 0)`
//!   over two rolling rows + [`PVQ_V_N_MAX`] = 352 / [`PVQ_V_K_MAX`]
//!   = 4096 caller-side bookkeeping bounds + [`PVQ_V_MAX`] =
//!   `2**32 − 1` overflow guard inherited from RFC 6716 §4.1.5's
//!   `ec_dec_uint(ft)` upper bound + [`PvqVError::{NOutOfRange,
//!   KOutOfRange, OverflowsDecUintRange}`] error reporting). The
//!   §4.3.4.2 PVQ index decode (`ec_dec_uint(V(N, K))` followed by
//!   the §4.3.4.2 conversion of the index to a sign-magnitude
//!   lattice point) and the §4.3.4.1 *Bits-to-Pulses* search both
//!   consume this primitive; both run at the consumer site.
//!
//! * The §4.3.4.1 *Bits-to-Pulses* pulse-cost cache
//!   ([`celt_pulse_cache`]: the 105-entry [`CACHE_INDEX50`] LM-major
//!   `(band, LM)` → offset map + the 392-byte run-packed
//!   [`CACHE_BITS50`] cost curves, [`bits_to_pulses`]`(band, lm,
//!   b_target) -> Result<u8, PulseCacheError>` returning the largest
//!   pulse count whose 1/8-bit cost fits the per-band budget +
//!   [`cache_run_offset`] / [`cache_max_pulses`] / [`cache_pulse_cost`]
//!   run accessors + [`PulseCacheError::{BandOutOfRange, LmOutOfRange,
//!   SentinelTuple, PulseCountOutOfRange}`]). Closes the cache half of
//!   the §4.3.4 PVQ allocator: given the §4.3.3 per-band budget, the
//!   §4.3.4.2 [`celt_pvq_v`] codebook-size function and this round's
//!   bits-to-pulses inversion together select `K` before the
//!   [`celt_pvq_decode`] shape decode. The closed-form path for the
//!   eight sentinel `(band, LM)` tuples runs at the consumer site.
//!
//! The rest of the CELT layer is not yet wired up; the [`Decoder`]
//! / [`Encoder`] entry points still return [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]
// Profluens patch (PROFLUENS-PATCHES.md): the decoder's transient buffers draw from a per-packet
// bump arena (`bump::DecodeBump`) via the stable-in-nightly allocator API, so a steady decode makes
// zero system-heap allocations. Requires the nightly workspace this crate is vendored into.
#![feature(allocator_api)]

use oxideav_core::RuntimeContext;

/// Profluens patch: per-thread bump arena for the decoder's transient allocations. See the module.
pub mod bump;

/// Crate-local error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The caller passed a zero-length packet. RFC 6716 §3.1 requires
    /// every well-formed Opus packet to contain at least one byte (R1).
    EmptyPacket,
    /// The packet violates one of the §3.2 frame-packing
    /// requirements (R2..R7). Examples: a code-1 packet with an odd
    /// payload length; a code-2 packet whose declared first-frame
    /// length runs off the end of the buffer; a code-3 packet with
    /// `M = 0` or whose CBR per-frame size is not an integer divisor
    /// of the remaining payload.
    MalformedPacket,
    /// The clean-room rebuild has not yet wired up a working
    /// SILK / CELT pipeline; the higher-level decode / encode paths
    /// return this until that work lands.
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::EmptyPacket => write!(
                f,
                "oxideav-opus: packet is empty; RFC 6716 §3.1 R1 requires at least one byte"
            ),
            Error::MalformedPacket => write!(
                f,
                "oxideav-opus: packet violates an RFC 6716 §3.2 frame-packing requirement"
            ),
            Error::NotImplemented => write!(
                f,
                "oxideav-opus: orphan-rebuild scaffold — SILK/CELT pipeline not wired up yet"
            ),
        }
    }
}

impl std::error::Error for Error {}

// Stable public surface: `Error`, the registry entry point (`register`),
// and the packet/container-level modules: [`decoder`] (`OpusDecoder` +
// frame/FEC/PLC outcome types), [`toc`], [`frames`], [`framing`],
// [`framing_self_delim`], [`packet_compose`], [`multistream`], and
// [`opus_head`]. Every other module is SILK/CELT stage plumbing (range
// coder, band shapes, PVQ, resampler internals, transition machinery)
// that is `pub` only so the integration-test and fuzz targets can reach
// it; those modules are `#[doc(hidden)]` and NOT part of the stable API
// surface for semver purposes.
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_alloc_encode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_alloc_search;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_alloc_trim;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_analysis;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_band_boost;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_band_decode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_band_encode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_band_layout;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_band_shape;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_band_thresh;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_cache_caps50;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_coarse_energy;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_deemphasis;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_denormalise;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_e_prob_model;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_energy_encode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_fine_energy;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_frame_decode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_frame_encode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_frame_prefix;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_header;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_imdct;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_laplace;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_log2_frac_table;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_mdct_synthesis;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_mdct_window;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_overlap_add;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_packet_encode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_post_filter;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_pulse_cache;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_pvq_decode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_pvq_encode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_pvq_v;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_rate_alloc;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_redundancy;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_reservations;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_spreading;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_static_alloc;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_synthesis;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_tf_adjust;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_tf_decode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_tf_hadamard;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_transitions;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod celt_trim_offsets;
pub mod decoder;
pub mod frames;
pub mod framing;
pub mod framing_self_delim;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod hybrid_packet_encode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod mode_transition_reset;
pub mod multistream;
pub mod opus_head;
pub mod packet_compose;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod plc;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod range_decoder;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod range_encoder;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod redundancy_decode_params;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_decode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_decode_core;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_encoder;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_excitation;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_excitation_quantize;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_frame;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_gains;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_header;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_lcg_seed;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_log2lin;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_lpc_analysis;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_lpc_synth;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_lpc_to_nlsf;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_lsf_interp;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_lsf_recon;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_lsf_stabilize;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_lsf_stage2;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_lsf_to_lpc;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_ltp;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_ltp_analysis;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_ltp_synth;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_nlsf_quantize;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_packet_encode;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_pitch;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_resampler;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_stereo;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod silk_synthesis;
pub mod toc;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod vbr;

#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_alloc_search::{
    per_band_eighth_bits_at_q_fp, q_fp_from_components, q_fp_to_components, search_q_fp,
    total_eighth_bits_at_q_fp, AllocSearchError, AllocSearchOutcome, QFpComponents, Q_FP_MAX,
    STATIC_ALLOC_INTERP_RIGHT_SHIFT,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_alloc_trim::{
    alloc_trim_icdf, alloc_trim_is_signalled, alloc_trim_pdf, decode_alloc_trim, frame_eighth_bits,
    AllocTrimError, ALLOC_TRIM_DEFAULT, ALLOC_TRIM_FTB, ALLOC_TRIM_ICDF, ALLOC_TRIM_MAX,
    ALLOC_TRIM_MIN, ALLOC_TRIM_PDF, ALLOC_TRIM_PDF_DENOMINATOR, ALLOC_TRIM_PDF_LEN,
    ALLOC_TRIM_SIGNAL_COST_EIGHTH_BITS, EIGHTH_BITS_PER_BYTE,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_band_boost::{
    band_boost_quanta, decode_band_boosts, BandBoost, BandBoostError, BandBoostOutcome,
    BAND_BOOST_QUANTA_CEIL_MULT, BAND_BOOST_QUANTA_FLOOR_EIGHTH_BITS, DYNALLOC_LOGP_INIT,
    DYNALLOC_LOGP_MIN, DYNALLOC_LOOP_LOGP_AFTER_FIRST,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_band_layout::{
    celt_band_at_hz, celt_band_bins_per_channel, celt_band_start_hz, celt_band_stop_hz,
    celt_end_coded_band, celt_first_coded_band, celt_total_bins_per_channel, CeltFrameSize,
    CELT_MAX_BINS_PER_BAND, CELT_NUM_BANDS, HYBRID_FIRST_CODED_BAND,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_band_thresh::{
    band_min_thresh, band_min_thresh_vec, compute_band_min_thresh, standard_band_window,
    BandThreshError, BAND_THRESH_BINS_DIVISOR, BAND_THRESH_BINS_MULTIPLIER,
    BAND_THRESH_MONO_CHANNELS, BAND_THRESH_PER_CHANNEL_EIGHTH_BITS, BAND_THRESH_STEREO_CHANNELS,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_cache_caps50::{
    cache_caps_offset, cache_caps_row, cache_caps_value, cap_for_band_bits, init_caps,
    CacheCaps50Error, CacheCapsStereo, CACHE_CAPS50, CACHE_CAPS50_LM_COUNT,
    CACHE_CAPS50_STEREO_COUNT, CACHE_CAPS50_STEREO_MONO, CACHE_CAPS50_STEREO_STEREO,
    CACHE_CAPS50_TOTAL_BYTES, INIT_CAPS_BIAS, INIT_CAPS_DIVISOR, INIT_CAPS_MAX_CHANNELS,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_deemphasis::{DeemphasisError, DeemphasisFilter, DEEMPHASIS_ALPHA_P};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_denormalise::{
    denormalise_band, denormalise_bands, denormalise_gain, DenormaliseError,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_e_prob_model::{
    e_prob_pair, e_prob_row, energy_pred_coef, EProbModelError, EProbPair, EnergyPredCoef,
    EnergyPredictionMode, E_PROB_MODEL, E_PROB_MODEL_BYTES_PER_BAND, E_PROB_MODEL_BYTES_PER_ROW,
    E_PROB_MODEL_LM_COUNT, E_PROB_MODEL_MODE_COUNT, E_PROB_MODEL_MODE_INTER,
    E_PROB_MODEL_MODE_INTRA, E_PROB_MODEL_TOTAL_BYTES, INTER_PRED_ALPHA_Q15, INTER_PRED_BETA_Q15,
    INTRA_PRED_ALPHA_Q15, INTRA_PRED_BETA_Q15, Q15_ONE,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_frame_prefix::{decode_celt_frame_prefix, CeltFramePrefix, CeltPostFilterParams};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_header::{CeltHeaderPrefix, CeltPostFilter};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_laplace::{
    ec_laplace_decode, LAPLACE_DECAY_UNIT, LAPLACE_LOG_MINP, LAPLACE_MINP, LAPLACE_NMIN,
    LAPLACE_TOTAL,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_log2_frac_table::{
    log2_frac, log2_frac_row, Log2FracError, LOG2_FRAC_TABLE, LOG2_FRAC_TABLE_LEN,
    Q3_BITS_PER_WHOLE_BIT,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_mdct_window::{
    basic_window, celt_overlap_window, mdct_window, window_tap, MdctWindowError, BASIC_WINDOW_LEN,
    CELT_OVERLAP_48K,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_overlap_add::{apply_synthesis_window, OverlapAddError, WeightedOverlapAdd};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_pulse_cache::{
    bits_to_pulses, cache_flat_index, cache_max_pulses, cache_pulse_cost, cache_run_offset,
    PulseCacheError, CACHE_BITS50, CACHE_BITS_LEN, CACHE_INDEX50, CACHE_INDEX_LEN,
    CACHE_INDEX_SENTINEL, CACHE_LM_COUNT, CACHE_MAX_PULSES,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_pvq_decode::{
    decode_pvq_shape, decode_pvq_shape_into, decode_pvq_vector, decode_pvq_vector_into,
    pvq_l1_norm, pvq_l2_norm_squared, pvq_unit_normalize, PvqDecodeError, PvqShapeError,
    PVQ_DECODE_K_MAX, PVQ_DECODE_N_MAX,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_pvq_v::{pvq_codebook_size, PvqVError, PVQ_V_K_MAX, PVQ_V_MAX, PVQ_V_N_MAX};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_redundancy::{
    decode_redundancy, remaining_bits, whole_bytes_remaining, RedundancyDecision,
    RedundancyPosition, HYBRID_REDUNDANCY_MIN_REMAINING_BITS,
    HYBRID_REDUNDANCY_SIZE_BASELINE_BYTES, HYBRID_REDUNDANCY_SIZE_DEC_UINT_FT,
    REDUNDANCY_FLAG_ICDF, REDUNDANCY_FLAG_ICDF_FTB, REDUNDANCY_MIN_SIZE_BYTES,
    REDUNDANCY_POSITION_ICDF, REDUNDANCY_POSITION_ICDF_FTB,
    SILK_ONLY_REDUNDANCY_MIN_REMAINING_BITS,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_reservations::{
    reserve_block, ReservationError, ReservationOutcome, ANTI_COLLAPSE_HEADROOM_LM_OFFSET,
    ANTI_COLLAPSE_HEADROOM_MULT_EIGHTH_BITS, ANTI_COLLAPSE_LM_MIN_EXCLUSIVE,
    CONSERVATIVE_DEDUCTION_EIGHTH_BITS, ONE_BIT_EIGHTH_BITS,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_spreading::{
    apply_spreading, decode_spread, rotate_in_place, rotate_strided, rotation_angle, rotation_gain,
    spread_f_r, spread_theta, spreading_stride, SpreadingError, SPREAD_FTB, SPREAD_F_R,
    SPREAD_ICDF, SPREAD_MAX, SPREAD_PDF, SPREAD_PDF_DENOMINATOR, SPREAD_PRE_ROTATION_MIN_BLOCK_LEN,
    SPREAD_VALUE_COUNT,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_static_alloc::{
    static_alloc_cell, static_alloc_eighth_bits, static_alloc_row, StaticAllocError, STATIC_ALLOC,
    STATIC_ALLOC_INTERP_STEPS, STATIC_ALLOC_Q_COUNT, STATIC_ALLOC_Q_MAX, STATIC_ALLOC_Q_MIN,
    STATIC_ALLOC_RIGHT_SHIFT, STATIC_ALLOC_TOTAL_CELLS,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_tf_adjust::{
    celt_tf_adjustment, celt_tf_select_can_affect, TfAdjustment, TfDirection,
    TF_ADJUSTMENT_ABS_MAX, TF_ADJUSTMENT_MAX, TF_ADJ_NONTRANSIENT_SELECT0,
    TF_ADJ_NONTRANSIENT_SELECT1, TF_ADJ_TRANSIENT_SELECT0, TF_ADJ_TRANSIENT_SELECT1,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_tf_decode::{decode_tf, TfDecode};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_transitions::{
    classify_normative_transition, recommended_non_normative, BoundaryOp, NormativeTransition,
    RecommendedNonNormativeTransition,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use celt_trim_offsets::{
    band_n_shortest, band_trim_offset, band_trim_offset_for_band, shortest_frame_size,
    TrimOffsetError, TRIM_OFFSETS_BIAS, TRIM_OFFSETS_DIVISOR, TRIM_OFFSETS_MONO_CHANNELS,
    TRIM_OFFSETS_NUMERATOR_SCALE, TRIM_OFFSETS_STEREO_CHANNELS,
    TRIM_OFFSETS_WIDTH_ONE_BINS_PER_CHANNEL, TRIM_OFFSETS_WIDTH_ONE_PER_CHANNEL_EIGHTH_BITS,
};
pub use decoder::{
    channel_count, output_samples_per_channel, DecodedAudio, FecDecodeStatus, FecRecovered,
    FrameDecodeStatus, FrameOutcome, OpusDecoder, OUTPUT_SAMPLES_PER_MS, OUTPUT_SAMPLE_RATE_HZ,
};
pub use frames::{OpusPacket, MAX_FRAMES_PER_PACKET, MAX_FRAME_BYTES};
pub use framing::{OperatingMode, OpusFrameRouting, SilkBandwidth};
pub use framing_self_delim::{parse_self_delimited, SelfDelimitedParse};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use mode_transition_reset::{decide_state_resets, CeltResetPlacement, StateReset};
pub use multistream::{
    assemble_multistream_packet, split_multistream_packet, MultistreamAudio, MultistreamDecoder,
    StreamPacket,
};
pub use opus_head::{
    apply_output_gain, ChannelMappingTable, OpusHead, OpusHeadError, PreSkip, OPUS_HEAD_MAGIC,
    OPUS_HEAD_MAX_VERSION, OPUS_HEAD_MIN_LEN,
};
pub use packet_compose::{
    compose_packet, compose_packet_code3, compose_self_delimited, encode_length,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use plc::{
    conceal_celt, conceal_silk, cross_lap, find_pitch, loss_gain, PlcFlavor, PlcState,
    PITCH_MAX_LAG, PITCH_MIN_LAG, PLC_CROSS_LAP_SAMPLES, PLC_HISTORY_SAMPLES,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use range_decoder::RangeDecoder;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use range_encoder::RangeEncoder;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use redundancy_decode_params::{
    apply_mb_to_wb_override, redundant_frame_params, CrossLapPlacement, RedundantFrameParams,
    REDUNDANT_CROSS_LAP_TENTHS_MS, REDUNDANT_FRAME_TENTHS_MS,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_decode::{
    decode_silk_frame, encode_silk_frame, SilkFrameConfig, SilkFrameDecoded, SilkFrameSymbols,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_excitation::{
    quantization_offset_q23, shell_block_count, Excitation, ExcitationConfig, ExcitationSymbols,
    SilkFrameSize, MAX_EXCITATION_SAMPLES, MAX_SHELL_BLOCKS, SHELL_BLOCK_SAMPLES,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_frame::{
    FrameKind, QuantizationOffsetType, SignalType, SilkFrameHeader, SilkFrameHeaderConfig,
    SilkHeaderPreGains, SilkHeaderSymbols, StereoPredictionWeights, StereoWeightSymbols,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_gains::{
    GainSymbol, SubframeGain, SubframeGains, SubframeGainsConfig, SILK_MAX_SUBFRAMES,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_header::{
    per_frame_lbrr_pdf, silk_frame_count, PerFrameLbrr, SilkChannelHeader, SilkHeaderBits,
    SILK_MAX_FRAMES_PER_CHANNEL,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_lcg_seed::{decode_lcg_seed, encode_lcg_seed};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_log2lin::{
    silk_gains_dequant, silk_log2lin, SILK_GAIN_Q16_MAX, SILK_GAIN_Q16_MIN, SILK_LOG_GAIN_BIAS,
    SILK_LOG_GAIN_MULTIPLIER,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_lpc_synth::{
    lpc_synthesis_frame, lpc_synthesis_subframe, subframe_samples, LpcSynthState,
    LPC_SYNTH_MAX_ORDER, LPC_SYNTH_MAX_SUBFRAME_SAMPLES,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_lsf_interp::{LsfInterpContext, LsfInterpolated};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_lsf_recon::{cb1_q8, NlsfReconstructed};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_lsf_stabilize::NlsfStabilized;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_lsf_stage2::{
    LsfStage2, D_LPC_MAX, D_LPC_NB_MB, D_LPC_WB, QSTEP_NB_MB_Q16, QSTEP_WB_Q16,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_lsf_to_lpc::{nlsf_to_c_q17, ordering, LpcQ12, LpcQ17};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_ltp::{
    LagCoding, LagSymbols, LtpConfig, LtpParameters, LtpSymbols, LTP_FILTER_TAPS,
    LTP_MAX_SUBFRAMES, LTP_SCALING_DEFAULT_Q14,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_ltp_synth::{
    ltp_synth_commit_subframe, ltp_synthesis_subframe, LtpSynthState, LtpSynthSubframe,
    LTP_LPC_HISTORY_MAX, LTP_MAX_PITCH_LAG, LTP_OUT_HISTORY_MAX, LTP_SCALE_FRESH_Q14,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_packet_encode::{
    encode_silk_only_packet_mono, encode_silk_only_packet_mono_with_lbrr,
    encode_silk_only_packet_stereo, encode_silk_only_packet_stereo_with_lbrr, StereoIntervalLbrr,
    StereoIntervalScripts, StereoLbrrPredictions, StereoPacketPredictions,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_resampler::{
    is_supported_output_rate, silk_frame_samples_at_output, silk_frame_samples_internal,
    silk_internal_rate_hz, silk_resampler_delay_ms, silk_resampler_delay_samples_at,
    SilkChannelPath, SilkUpsampler, REFERENCE_RATE_HZ, SILK_RESAMPLER_DELAY_MS_MB,
    SILK_RESAMPLER_DELAY_MS_NB, SILK_RESAMPLER_DELAY_MS_WB, SUPPORTED_OUTPUT_RATES_HZ,
};
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub use silk_stereo::{
    estimate_stereo_weights, interp_phase_samples, stereo_lr_to_ms, stereo_ms_to_lr, MidSideFrame,
    StereoDownmixState, StereoFrame, StereoUnmixState, StereoWeightsQ13,
};
pub use toc::{Bandwidth, ChannelMapping, FrameCountCode, Mode, OpusTocByte};

/// Codec registration: declares the `opus` codec id together with its
/// payload-magic claim.
///
/// Some carriage formats announce the codec not with a container-level
/// tag but by the leading bytes of the payload itself — for Opus that
/// is the fixed 8-octet `OpusHead` signature opening the RFC 7845 §5.1
/// identification header (the first packet of an Ogg logical stream).
/// Declaring [`opus_head::OPUS_HEAD_MAGIC`] via
/// [`oxideav_core::registry::CodecInfo::payload_magic`] lets container
/// layers resolve such a stream to `opus` through
/// `CodecRegistry::resolve_payload_magic_ref` / the
/// [`oxideav_core::CodecResolver`] trait. The prefix is 8 bytes long,
/// so truncations (`"Opus"`, `"OpusHea"`) and the sibling comment
/// header signature (`"OpusTags"`, RFC 7845 §5.2) do not resolve.
///
/// Registry-level decoder/encoder factories are not wired yet — the
/// crate's working entry points are the direct APIs
/// ([`decoder::OpusDecoder`], [`multistream::MultistreamDecoder`], and
/// the packet encoders) — so this registration carries capability
/// metadata and the magic claim only (a tag-only registration in
/// registry terms).
pub fn register(ctx: &mut RuntimeContext) {
    let mut caps = oxideav_core::CodecCapabilities::audio("opus_sw");
    caps.lossy = true;
    caps.max_sample_rate = Some(48_000);
    // RFC 7845 §5.1.1 channel mapping: up to 255 output channels.
    caps.max_channels = Some(255);
    ctx.codecs.register(
        oxideav_core::CodecInfo::new(oxideav_core::CodecId::new("opus"))
            .capabilities(caps)
            .payload_magic(opus_head::OPUS_HEAD_MAGIC.as_slice()),
    );
}

oxideav_core::register!("opus", register);
