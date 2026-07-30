//! SILK core signal reconstruction in exact fixed point — RFC 6716
//! §4.2.7.9, realized with the integer arithmetic of the RFC 6716 §A
//! reference listing (the normative implementation embedded in the RFC
//! itself).
//!
//! The RFC's §4.2.7.9.1 / §4.2.7.9.2 narrative describes the LTP and LPC
//! synthesis filters with real-number formulas, but the decoder the RFC
//! ships in Appendix A reconstructs the signal in Q-domain integer
//! arithmetic whose rounding recirculates through the LTP feedback and
//! the per-subframe gain rescaling. Matching the reference decode
//! waveform therefore requires reproducing that integer arithmetic
//! exactly; this module does so:
//!
//! * the excitation is taken in the Q14 domain (`e_Q23 >> …` — our
//!   §4.2.7.8.6 `e_Q23[]` equals the reference `exc_Q14[]` scaled by
//!   `2^6`, verified sample-exact against the instrumented listing),
//! * §4.2.7.9.1 LTP synthesis runs in Q13/Q15 with the `silk_SMLAWB`
//!   truncating 32×16 multiplies and the re-whitening
//!   §4.2.7.5.8-filter pass over the carried output history,
//! * §4.2.7.9.2 LPC synthesis runs in Q14 with the state rescaling on
//!   every gain change (`gain_adj_Q16`), and
//! * the output samples are `SAT16(round(sLPC_Q14 × Gain_Q10 >> 24))`
//!   — signed 16-bit PCM at the internal SILK rate.
//!
//! All cross-frame state (the `outBuf` output history used by the LTP
//! re-whitening, the Q14 LPC filter state, the previous subframe gain,
//! the previous frame's final pitch lag and signal type) lives in
//! [`SilkCoreState`], one per SILK channel.
//!
//! Clean-room note: every formula here is from RFC 6716 (narrative
//! sections cited per item) and its §A embedded reference listing, which
//! the project treats as staged spec material. No external source is
//! consulted.

use crate::silk_decode::SilkFrameDecoded;
use crate::silk_excitation::SilkFrameSize;
use crate::silk_frame::SignalType;
use crate::toc::Bandwidth;
use crate::Error;

/// §4.2.7.6 LTP filter order (5 taps).
const LTP_ORDER: usize = 5;
/// Maximum LPC order (WB).
const MAX_LPC_ORDER: usize = 16;
/// Maximum SILK frame length in samples (20 ms at 16 kHz).
const MAX_FRAME_LENGTH: usize = 320;
/// Maximum subframe length in samples (5 ms at 16 kHz).
const MAX_SUB_FRAME_LENGTH: usize = 80;
// ---------------------------------------------------------------------
// Fixed-point helpers (§A listing arithmetic conventions; all i32 with
// two's-complement wrapping where the listing tolerates overflow).
// ---------------------------------------------------------------------

/// `(a32 * b16) >> 16` with the multiplier taken from the *low* 16 bits
/// of `b` (sign-extended), truncating toward −∞.
#[inline]
pub(crate) fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

/// `a + ((b32 * c16) >> 16)` (wrapping add).
#[inline]
pub(crate) fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulwb(b, c))
}

/// 16×16 → 32 multiply of the low halves.
#[inline]
pub(crate) fn smulbb(a: i32, b: i32) -> i32 {
    (i32::from(a as i16)).wrapping_mul(i32::from(b as i16))
}

/// `a + b16*c16` (wrapping).
#[inline]
pub(crate) fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulbb(b, c))
}

/// Full 32×32 → upper-32 multiply (`(a*b) >> 32`).
#[inline]
pub(crate) fn smmul(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

/// Rounding right shift (`shift ≥ 1`).
#[inline]
pub(crate) fn rshift_round(a: i32, shift: u32) -> i32 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}

/// Saturate to the signed 16-bit range.
#[inline]
pub(crate) fn sat16(a: i32) -> i16 {
    a.clamp(-32768, 32767) as i16
}

/// `a32 * b32 >> 16` composed as the listing's `silk_SMULWW`:
/// `SMULWB(a, b) + a * round(b / 2^16)` with wrapping adds/multiplies.
#[inline]
pub(crate) fn smulww(a: i32, b: i32) -> i32 {
    smulwb(a, b).wrapping_add(a.wrapping_mul(rshift_round(b, 16)))
}

/// Saturating left shift.
#[inline]
fn lshift_sat32(a: i32, shift: u32) -> i32 {
    let min = i32::MIN >> shift;
    let max = i32::MAX >> shift;
    a.clamp(min, max) << shift
}

/// `(1 << qres) / b` approximation — the listing's `silk_INVERSE32_varQ`
/// (used with `Qres = 47` for the LTP-state inverse gain).
pub(crate) fn inverse32_varq(b32: i32, qres: i32) -> i32 {
    debug_assert!(b32 != 0);
    let b_headrm = (b32.unsigned_abs().leading_zeros() as i32) - 1;
    let b32_nrm = b32 << b_headrm;
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    let mut result = b32_inv << 16;
    let err_q32 = ((1i32 << 29).wrapping_sub(smulwb(b32_nrm, b32_inv))) << 3;
    // Refinement: result += err_Q32 * b32_inv >> 16 (SMLAWW).
    result = result
        .wrapping_add(smulwb(err_q32, b32_inv))
        .wrapping_add(err_q32.wrapping_mul(rshift_round(b32_inv, 16)));
    let lshift = 61 - b_headrm - qres;
    if lshift <= 0 {
        lshift_sat32(result, (-lshift) as u32)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// `(a << qres) / b` approximation — the listing's `silk_DIV32_varQ`
/// (used with `Qres = 16` for the gain-adjustment factor).
pub(crate) fn div32_varq(a32: i32, b32: i32, qres: i32) -> i32 {
    debug_assert!(b32 != 0);
    let a_headrm = (a32.unsigned_abs().leading_zeros() as i32) - 1;
    let a32_nrm = a32 << a_headrm;
    let b_headrm = (b32.unsigned_abs().leading_zeros() as i32) - 1;
    let b32_nrm = b32 << b_headrm;
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    let mut result = smulwb(a32_nrm, b32_inv);
    let a32_res = a32_nrm.wrapping_sub(((smmul(b32_nrm, result) as u32) << 3) as i32);
    result = smlawb(result, a32_res, b32_inv);
    let lshift = 29 + a_headrm - b_headrm - qres;
    if lshift < 0 {
        lshift_sat32(result, (-lshift) as u32)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// §4.2.7.5.8-form whitening filter over the output history (the
/// listing's `silk_LPC_analysis_filter`): `out[i] = SAT16(round(in[i] −
/// Σ a_Q12[j]·in[i−1−j] scaled Q12))`, first `order` outputs zeroed.
fn lpc_analysis_filter(out: &mut [i16], input: &[i16], a_q12: &[i16], order: usize) {
    let len = out.len();
    debug_assert_eq!(input.len(), len);
    debug_assert!(order >= 6 && order % 2 == 0 && order <= len);
    for ix in order..len {
        // in_ptr points one behind the predicted sample.
        let base = ix - 1;
        let mut out32_q12: i32 = 0;
        for (j, &c) in a_q12.iter().enumerate().take(order) {
            out32_q12 = smlabb(out32_q12, i32::from(input[base - j]), i32::from(c));
        }
        let out32_q12 = (i32::from(input[ix]) << 12).wrapping_sub(out32_q12);
        out[ix] = sat16(rshift_round(out32_q12, 12));
    }
    for o in out.iter_mut().take(order) {
        *o = 0;
    }
}

// ---------------------------------------------------------------------
// Cross-frame state
// ---------------------------------------------------------------------

/// Cross-frame fixed-point SILK reconstruction state for one channel —
/// the integer counterpart of the §4.2.7.9 histories.
#[derive(Debug, Clone)]
pub struct SilkCoreState {
    bandwidth: Bandwidth,
    fs_khz: usize,
    lpc_order: usize,
    /// 20 ms of output history at the internal rate plus 10 ms of
    /// scratch (the §4.2.7.9.1 re-whitening window).
    out_buf: [i16; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
    /// §4.2.7.9.2 LPC filter state, Q14.
    s_lpc_q14: [i32; MAX_LPC_ORDER],
    /// Previous subframe's dequantized gain, Q16 (65536 after reset).
    prev_gain_q16: i32,
    /// Previous frame's final pitch lag (100 after reset).
    lag_prev: i32,
    /// Previous frame's signal type (Inactive after reset).
    prev_signal_type: SignalType,
}

impl SilkCoreState {
    /// Fresh zeroed state for a SILK bandwidth (NB / MB / WB only).
    pub fn new(bandwidth: Bandwidth) -> Result<Self, Error> {
        let (fs_khz, lpc_order) = match bandwidth {
            Bandwidth::Nb => (8, 10),
            Bandwidth::Mb => (12, 10),
            Bandwidth::Wb => (16, 16),
            _ => return Err(Error::MalformedPacket),
        };
        Ok(Self {
            bandwidth,
            fs_khz,
            lpc_order,
            out_buf: [0; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
            s_lpc_q14: [0; MAX_LPC_ORDER],
            prev_gain_q16: 65536,
            lag_prev: 100,
            prev_signal_type: SignalType::Inactive,
        })
    }

    /// Bandwidth this state was created for.
    pub fn bandwidth(&self) -> Bandwidth {
        self.bandwidth
    }

    /// Clear every history (the §4.5.2 full SILK decoder reset).
    pub fn reset(&mut self) {
        self.reset_prediction_memory();
        self.prev_gain_q16 = 65536;
    }

    /// Clear the prediction memories only — the output history, the
    /// Q14 LPC state, the previous pitch lag and signal type — while
    /// **keeping** the previous-subframe gain. This is the reference
    /// decoder's side-channel reset (armed when a mid-only interval run
    /// ends) and its internal-rate-change reset; only a full §4.5.2
    /// decoder reset re-arms the gain to unity.
    pub fn reset_prediction_memory(&mut self) {
        self.out_buf = [0; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH];
        self.s_lpc_q14 = [0; MAX_LPC_ORDER];
        self.lag_prev = 100;
        self.prev_signal_type = SignalType::Inactive;
    }

    /// 20 ms of output history in samples (`ltp_mem_length`).
    fn ltp_mem_length(&self) -> usize {
        20 * self.fs_khz
    }

    /// 5 ms subframe length in samples.
    fn subfr_length(&self) -> usize {
        5 * self.fs_khz
    }
}

/// Reconstruct one decoded regular SILK frame into signed 16-bit samples
/// at the internal SILK rate (8/12/16 kHz), in the exact fixed-point
/// arithmetic of the §A reference listing, threading `state` across
/// frames.
///
/// Returns `subfr_length × num_subframes` samples.
pub fn decode_core(
    state: &mut SilkCoreState,
    frame_size: SilkFrameSize,
    decoded: &SilkFrameDecoded,
) -> Result<Vec<i16, crate::bump::DecodeBump>, Error> {
    let nb_subfr = match frame_size {
        SilkFrameSize::TenMs => 2usize,
        SilkFrameSize::TwentyMs => 4usize,
    };
    let subfr_length = state.subfr_length();
    let frame_length = nb_subfr * subfr_length;
    let ltp_mem_length = state.ltp_mem_length();
    let lpc_order = state.lpc_order;

    if decoded.gains.len() != nb_subfr {
        return Err(Error::MalformedPacket);
    }

    // §4.2.7.8.6: the Q23 excitation equals the listing's exc_Q14 × 2^6.
    let e_q23 = decoded.excitation.e_q23();
    if e_q23.len() < frame_length {
        return Err(Error::MalformedPacket);
    }
    // Profluens patch: per-packet transient excitation, consumed only as a
    // slice below — drawn from the bump arena, not the global heap.
    let mut exc_q14: Vec<i32, crate::bump::DecodeBump> =
        Vec::with_capacity_in(frame_length, crate::bump::DecodeBump);
    exc_q14.extend(e_q23[..frame_length].iter().map(|&e| e << 6));

    let gains_q16 = decoded.gains.dequant_q16();
    let is_voiced = decoded.signal_type == SignalType::Voiced;
    let pitch_lags = decoded.ltp.pitch_lags();
    let taps_q7 = decoded.ltp.filter_taps_q7();
    if is_voiced && (pitch_lags.len() != nb_subfr || taps_q7.len() != nb_subfr) {
        return Err(Error::MalformedPacket);
    }

    // Per-half LPC filters as i16 Q12 (§4.2.7.5.8 bounds them to i16).
    let mut a_q12 = [[0i16; MAX_LPC_ORDER]; 2];
    let second = decoded.lpc_second_half.a_q12();
    if second.len() != lpc_order {
        return Err(Error::MalformedPacket);
    }
    for (dst, &src) in a_q12[1].iter_mut().zip(second) {
        *dst = src.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
    match decoded.lpc_first_half.as_ref() {
        Some(first) => {
            if first.a_q12().len() != lpc_order {
                return Err(Error::MalformedPacket);
            }
            for (dst, &src) in a_q12[0].iter_mut().zip(first.a_q12()) {
                *dst = src.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
            }
        }
        None => a_q12[0] = a_q12[1],
    }

    // §4.2.7.5.5: the interpolation split is armed when w_Q2 < 4.
    let nlsf_interp_flag = decoded.lsf_interp_q2.is_some_and(|w| w < 4);

    // Profluens patch: the frame's decoded PCM. Returned up through synthesize_silk_frame_i16 and
    // slice-copied into the caller's per-channel accumulator — from the per-packet bump arena.
    let mut xq = Vec::with_capacity_in(frame_length, crate::bump::DecodeBump);
    xq.resize(frame_length, 0i16);
    let mut s_ltp = [0i16; MAX_FRAME_LENGTH];
    let mut s_ltp_q15 = [0i32; 2 * MAX_FRAME_LENGTH];
    let mut res_q14 = [0i32; MAX_SUB_FRAME_LENGTH];
    let mut s_lpc_q14 = [0i32; MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER];
    s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&state.s_lpc_q14);

    let mut s_ltp_buf_idx = ltp_mem_length;
    let mut lag = 0i32;

    for k in 0..nb_subfr {
        let a = &a_q12[k >> 1][..lpc_order];
        let b_q14: [i32; LTP_ORDER] = if is_voiced {
            let taps = taps_q7[k];
            [
                i32::from(taps[0]) << 7,
                i32::from(taps[1]) << 7,
                i32::from(taps[2]) << 7,
                i32::from(taps[3]) << 7,
                i32::from(taps[4]) << 7,
            ]
        } else {
            [0; LTP_ORDER]
        };
        let gain_q16 = gains_q16[k] as i32;
        let gain_q10 = gain_q16 >> 6;
        let mut inv_gain_q31 = inverse32_varq(gain_q16, 47);

        // Gain-change state rescale.
        let gain_adj_q16 = if gain_q16 != state.prev_gain_q16 {
            let adj = div32_varq(state.prev_gain_q16, gain_q16, 16);
            for s in s_lpc_q14.iter_mut().take(MAX_LPC_ORDER) {
                *s = smulww(adj, *s);
            }
            adj
        } else {
            1 << 16
        };
        state.prev_gain_q16 = gain_q16;

        if is_voiced {
            lag = pitch_lags[k];

            // §4.2.7.9.1 re-whitening of the output history.
            if k == 0 || (k == 2 && nlsf_interp_flag) {
                let start_idx =
                    ltp_mem_length as i32 - lag - lpc_order as i32 - (LTP_ORDER as i32) / 2;
                if start_idx <= 0 {
                    return Err(Error::MalformedPacket);
                }
                let start_idx = start_idx as usize;

                if k == 2 {
                    state.out_buf[ltp_mem_length..ltp_mem_length + 2 * subfr_length]
                        .copy_from_slice(&xq[..2 * subfr_length]);
                }

                let region_len = ltp_mem_length - start_idx;
                let in_start = start_idx + k * subfr_length;
                // Profluens patch: per-subframe re-whitening scratch (copied into s_ltp) — bump.
                let mut whitened = Vec::with_capacity_in(region_len, crate::bump::DecodeBump);
                whitened.resize(region_len, 0i16);
                lpc_analysis_filter(
                    &mut whitened,
                    &state.out_buf[in_start..in_start + region_len],
                    a,
                    lpc_order,
                );
                s_ltp[start_idx..ltp_mem_length].copy_from_slice(&whitened);

                // §4.2.7.9.1: after re-whitening the LTP state is
                // unscaled; at k == 0 the LTP scaling reduces
                // inter-packet dependency.
                if k == 0 {
                    inv_gain_q31 = smulwb(
                        inv_gain_q31,
                        i32::from(decoded.ltp.ltp_scaling_q14() as i16),
                    ) << 2;
                }
                for i in 0..(lag as usize + LTP_ORDER / 2) {
                    s_ltp_q15[s_ltp_buf_idx - i - 1] =
                        smulwb(inv_gain_q31, i32::from(s_ltp[ltp_mem_length - i - 1]));
                }
            } else if gain_adj_q16 != 1 << 16 {
                // Rescale the carried LTP state on gain change.
                for i in 0..(lag as usize + LTP_ORDER / 2) {
                    s_ltp_q15[s_ltp_buf_idx - i - 1] =
                        smulww(gain_adj_q16, s_ltp_q15[s_ltp_buf_idx - i - 1]);
                }
            }
        }

        // §4.2.7.9.1 long-term prediction.
        let exc_base = k * subfr_length;
        if is_voiced {
            let pred_lag_base = s_ltp_buf_idx - lag as usize + LTP_ORDER / 2;
            for (i, pred_lag_idx) in (pred_lag_base..pred_lag_base + subfr_length).enumerate() {
                // The +2 bias offsets SMLAWB's truncation toward −∞.
                let mut ltp_pred_q13 = 2i32;
                ltp_pred_q13 = smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_idx], b_q14[0]);
                ltp_pred_q13 = smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_idx - 1], b_q14[1]);
                ltp_pred_q13 = smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_idx - 2], b_q14[2]);
                ltp_pred_q13 = smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_idx - 3], b_q14[3]);
                ltp_pred_q13 = smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_idx - 4], b_q14[4]);

                res_q14[i] = exc_q14[exc_base + i].wrapping_add(ltp_pred_q13 << 1);

                s_ltp_q15[s_ltp_buf_idx] = res_q14[i] << 1;
                s_ltp_buf_idx += 1;
            }
        } else {
            res_q14[..subfr_length].copy_from_slice(&exc_q14[exc_base..exc_base + subfr_length]);
        }

        // §4.2.7.9.2 short-term prediction + gain scaling.
        for i in 0..subfr_length {
            // The bias `order/2` offsets SMLAWB's truncation toward −∞.
            let mut lpc_pred_q10 = (lpc_order as i32) >> 1;
            for (j, &c) in a.iter().enumerate() {
                lpc_pred_q10 = smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + i - 1 - j],
                    i32::from(c),
                );
            }

            s_lpc_q14[MAX_LPC_ORDER + i] = res_q14[i].wrapping_add(lpc_pred_q10 << 4);

            xq[k * subfr_length + i] = sat16(rshift_round(
                smulww(s_lpc_q14[MAX_LPC_ORDER + i], gain_q10),
                8,
            ));
        }

        // Carry the LPC filter state into the next subframe.
        s_lpc_q14.copy_within(subfr_length..subfr_length + MAX_LPC_ORDER, 0);
    }

    state.s_lpc_q14.copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);

    // Frame-level output-history update (the listing's decode_frame
    // tail): keep the most recent `ltp_mem_length` samples.
    let mv_len = ltp_mem_length - frame_length;
    state
        .out_buf
        .copy_within(frame_length..frame_length + mv_len, 0);
    state.out_buf[mv_len..ltp_mem_length].copy_from_slice(&xq);

    state.lag_prev = if is_voiced {
        pitch_lags[nb_subfr - 1]
    } else {
        0
    };
    state.prev_signal_type = decoded.signal_type;

    Ok(xq)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixed-point primitives match their defining formulas on
    /// hand-computed values.
    #[test]
    fn primitive_arithmetic() {
        // smulwb takes the low 16 bits of b, sign-extended.
        assert_eq!(smulwb(1 << 16, 3), 3);
        assert_eq!(smulwb(1 << 20, -5), -80);
        // b's high bits are ignored.
        assert_eq!(smulwb(65536, 65536 + 7), 7);
        // Truncation toward −∞.
        assert_eq!(smulwb(-1, 1), -1);
        // smulww includes the high half of b.
        assert_eq!(smulww(65536, 65536 + 7), 65536 + 7);
        // rshift_round rounds to nearest (half away from zero on the
        // positive side of the two's-complement shift).
        assert_eq!(rshift_round(5, 1), 3);
        assert_eq!(rshift_round(4, 1), 2);
        assert_eq!(rshift_round(-5, 1), -2);
        assert_eq!(rshift_round(11, 2), 3);
        // sat16 clamps.
        assert_eq!(sat16(40000), 32767);
        assert_eq!(sat16(-40000), -32768);
    }

    /// `inverse32_varq(b, 47)` approximates `2^47 / b`.
    #[test]
    fn inverse32_varq_approximates() {
        for b in [65536i32, 100_000, 1 << 20, 123_456, 3_000_000] {
            let got = i64::from(inverse32_varq(b, 47));
            let want = (1i64 << 47) / i64::from(b);
            let err = (got - want).abs();
            assert!(
                err <= want / 1000 + 2,
                "b={b}: got {got}, want ≈{want}, err {err}"
            );
        }
    }

    /// `div32_varq(a, b, 16)` approximates `(a << 16) / b`.
    #[test]
    fn div32_varq_approximates() {
        for (a, b) in [(65536i32, 65536i32), (100_000, 50_000), (77_777, 111_111)] {
            let got = i64::from(div32_varq(a, b, 16));
            let want = (i64::from(a) << 16) / i64::from(b);
            let err = (got - want).abs();
            assert!(
                err <= want.abs() / 1000 + 2,
                "a={a} b={b}: got {got}, want ≈{want}, err {err}"
            );
        }
    }

    /// The whitening filter zeroes its first `order` outputs and computes
    /// the Q12 residual with rounding.
    #[test]
    fn lpc_analysis_filter_basic() {
        let order = 6;
        let input: Vec<i16> = (0..32).map(|i| (i * 100 - 800) as i16).collect();
        let a: Vec<i16> = vec![4096, 0, 0, 0, 0, 0]; // predict = previous sample
        let mut out = vec![0i16; 32];
        lpc_analysis_filter(&mut out, &input, &a, order);
        for &o in out.iter().take(order) {
            assert_eq!(o, 0);
        }
        // With a = [1, 0, …] the residual is the first difference.
        for i in order..32 {
            assert_eq!(out[i], input[i] - input[i - 1], "at {i}");
        }
    }

    /// State construction rejects SWB/FB and resets clean.
    #[test]
    fn state_lifecycle() {
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            let mut st = SilkCoreState::new(bw).unwrap();
            assert_eq!(st.bandwidth(), bw);
            st.prev_gain_q16 = 5;
            st.out_buf[3] = 7;
            st.reset();
            assert_eq!(st.prev_gain_q16, 65536);
            assert_eq!(st.out_buf[3], 0);
            assert_eq!(st.lag_prev, 100);
        }
        assert!(SilkCoreState::new(Bandwidth::Swb).is_err());
        assert!(SilkCoreState::new(Bandwidth::Fb).is_err());
    }
}
