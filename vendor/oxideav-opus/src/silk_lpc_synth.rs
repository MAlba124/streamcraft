//! SILK §4.2.7.9.2 LPC synthesis filter — RFC 6716.
//!
//! The final stage of SILK reconstruction takes the residual signal
//! `res[i]` produced by §4.2.7.9.1 (LTP synthesis for voiced frames; a
//! normalised excitation copy for unvoiced frames), the §4.2.7.4
//! quantization gain `gain_Q16[s]` for the current subframe, and the
//! §4.2.7.5.8 stabilised Q12 short-term predictor `a_Q12[k]`, and
//! produces the subframe output signal:
//!
//! ```text
//!                                   d_LPC-1
//!                  gain_Q16[s]        __              a_Q12[k]
//!         lpc[i] = ----------- * res[i] + \  lpc[i-k-1] * --------
//!                    65536.0              /_               4096.0
//!                                         k=0
//!
//!         out[i] = clamp(-1.0, lpc[i], 1.0)
//! ```
//!
//! The §4.2.7.9 preamble of RFC 6716 explicitly notes that "the
//! remainder of the reconstruction process for the frame does not need
//! to be bit-exact, as small errors should only introduce
//! proportionally small distortions". We therefore implement this stage
//! in `f32`, using the spec's formula verbatim with a left-to-right
//! accumulator. The d_LPC unclamped `lpc[i]` history samples are
//! retained between subframes (the spec mandates this — the LPC
//! synthesis is recursive across the SILK frame boundary).
//!
//! Two unclamped values are saved per the §4.2.7.9.2 wording:
//!
//!   * `lpc[i]` (unclamped) feeds the next subframe's LPC synthesis
//!     filter.
//!   * `out[i]` (clamped to `[-1.0, 1.0]`) is the rendered audio sample
//!     and also feeds the §4.2.7.9.1 LTP rewhitening path for the next
//!     subframe of a voiced frame.
//!
//! All truth is taken from RFC 6716 §4.2.7.9.2. No external library
//! source is consulted.

use crate::silk_excitation::SilkFrameSize;
use crate::silk_lsf_stage2::{D_LPC_NB_MB, D_LPC_WB};
use crate::toc::Bandwidth;
use crate::Error;

/// Largest LPC order across SILK bandwidths (WB → 16).
pub const LPC_SYNTH_MAX_ORDER: usize = D_LPC_WB;

/// Largest subframe sample count across SILK bandwidths
/// (`subframe_samples(Wb) = 80`).
pub const LPC_SYNTH_MAX_SUBFRAME_SAMPLES: usize = 80;

/// Number of audio samples in a single SILK subframe per RFC 6716
/// §4.2.7.9: 40 for NB, 60 for MB, 80 for WB.
///
/// Rejects SWB / FB — the SILK layer never sees them after the §4.2.2
/// hybrid split.
pub fn subframe_samples(bandwidth: Bandwidth) -> Result<usize, Error> {
    Ok(match bandwidth {
        Bandwidth::Nb => 40,
        Bandwidth::Mb => 60,
        Bandwidth::Wb => 80,
        _ => return Err(Error::MalformedPacket),
    })
}

/// LPC-synthesis history buffer (size `d_LPC`, holding the unclamped
/// `lpc[i]` values from the previous subframe).
///
/// Initially cleared to zeros on a decoder reset or after an uncoded
/// regular SILK frame for this channel (RFC 6716 §4.2.7.9.2 first
/// bullet list).
#[derive(Debug, Clone)]
pub struct LpcSynthState {
    d_lpc: usize,
    /// History samples from the end of the previous subframe. The
    /// newest sample (`lpc[j-1]`) is at index `d_lpc - 1`; the oldest
    /// (`lpc[j-d_LPC]`) is at index `0`.
    history: [f32; LPC_SYNTH_MAX_ORDER],
}

impl LpcSynthState {
    /// Construct a zero-initialised history buffer for `bandwidth`.
    ///
    /// Rejects SWB / FB.
    pub fn new(bandwidth: Bandwidth) -> Result<Self, Error> {
        let d_lpc = match bandwidth {
            Bandwidth::Nb | Bandwidth::Mb => D_LPC_NB_MB,
            Bandwidth::Wb => D_LPC_WB,
            _ => return Err(Error::MalformedPacket),
        };
        Ok(Self {
            d_lpc,
            history: [0.0; LPC_SYNTH_MAX_ORDER],
        })
    }

    /// d_LPC for the bandwidth this state was created with.
    pub fn d_lpc(&self) -> usize {
        self.d_lpc
    }

    /// Read-only access to the `d_LPC` retained unclamped `lpc[]` history
    /// samples in source-order (oldest at index `0`, newest at index
    /// `d_lpc - 1`).
    pub fn history(&self) -> &[f32] {
        &self.history[..self.d_lpc]
    }

    /// Reset to all-zero history (RFC 6716 §4.2.7.9.2 "decoder reset"
    /// path).
    pub fn reset(&mut self) {
        self.history = [0.0; LPC_SYNTH_MAX_ORDER];
    }
}

/// Run one subframe of LPC synthesis per RFC 6716 §4.2.7.9.2.
///
/// * `state` — `d_LPC` history samples from the previous subframe. On
///   return, holds the last `d_LPC` *unclamped* `lpc[i]` values for the
///   next subframe.
/// * `res` — the LPC residual for this subframe (length must equal
///   `subframe_samples(bandwidth)`).
/// * `gain_q16` — the §4.2.7.4 dequantised Q16 gain. Per §4.2.7.4 the
///   value is in `[81920, 1686110208]`; this routine does not enforce
///   the bound (the caller's gain dequantizer does).
/// * `a_q12` — the §4.2.7.5.8 stabilised Q12 short-term LPC predictor
///   for this subframe. Length must equal `state.d_lpc()`.
/// * `out_clamped` — output buffer, same length as `res`. Receives the
///   `out[i] = clamp(-1.0, lpc[i], 1.0)` samples.
///
/// Returns the unclamped `lpc[i]` vector for this subframe (length =
/// `res.len()`); callers that need to re-whiten through the §4.2.7.9.1
/// LTP path on the next subframe consume `out_clamped`, but the §4.2.7.9.2
/// "decoder saves the unclamped values lpc[i] to feed into the LPC
/// filter for the next subframe" wording is implemented by updating
/// `state` from the unclamped buffer before this function returns.
///
/// Errors:
///
/// * `Error::MalformedPacket` if `res.len() != out_clamped.len()`, or
///   `a_q12.len() != state.d_lpc()`, or `bandwidth` is SWB / FB.
pub fn lpc_synthesis_subframe(
    bandwidth: Bandwidth,
    state: &mut LpcSynthState,
    res: &[f32],
    gain_q16: u32,
    a_q12: &[i16],
    out_clamped: &mut [f32],
) -> Result<Vec<f32>, Error> {
    let n = subframe_samples(bandwidth)?;
    if res.len() != n || out_clamped.len() != n {
        return Err(Error::MalformedPacket);
    }
    if a_q12.len() != state.d_lpc {
        return Err(Error::MalformedPacket);
    }

    let d_lpc = state.d_lpc;
    // Working buffer holds `d_LPC` history samples followed by the
    // current subframe's `n` unclamped `lpc[i]` values. Indexing into
    // `work` uses the SILK convention of `j = d_LPC` (the first sample
    // of the current subframe).
    let mut work = vec![0.0f32; d_lpc + n];
    work[..d_lpc].copy_from_slice(&state.history[..d_lpc]);

    let gain_scale = (gain_q16 as f32) / 65536.0;
    // Pre-scale Q12 coefficients to floating point once.
    let mut a_f = [0.0f32; LPC_SYNTH_MAX_ORDER];
    for k in 0..d_lpc {
        a_f[k] = (a_q12[k] as f32) / 4096.0;
    }

    for i in 0..n {
        // sum = Σ_{k=0..d_LPC-1} lpc[i-k-1] * a_Q12[k] / 4096.0
        let mut sum = 0.0f32;
        for k in 0..d_lpc {
            // lpc[i - k - 1] lives at work[d_lpc + i - k - 1] under the
            // d_LPC-shift indexing convention.
            // (d_lpc + i).wrapping_sub(k + 1) is always >= 0 because
            // k+1 ≤ d_lpc and d_lpc + i - k - 1 ≥ i ≥ 0.
            sum += work[d_lpc + i - k - 1] * a_f[k];
        }
        let value = gain_scale * res[i] + sum;
        work[d_lpc + i] = value;
        // out[i] is the clamped version; lpc[i] is the unclamped one
        // saved into state below.
        out_clamped[i] = value.clamp(-1.0, 1.0);
    }

    // Save the final d_LPC *unclamped* values into state for the next
    // subframe.
    for k in 0..d_lpc {
        state.history[k] = work[d_lpc + n - d_lpc + k];
    }

    // Return the per-subframe unclamped lpc[i] vector for callers that
    // want it (e.g. tests, or the §4.2.7.9.1 rewhitening path on the
    // next subframe which actually consumes out[] but may want lpc[]
    // for self-checks).
    Ok(work[d_lpc..].to_vec())
}

/// Convenience wrapper that runs LPC synthesis across every subframe
/// of a SILK frame in source order. The history is carried across
/// subframes inside `state` per RFC 6716 §4.2.7.9.2.
///
/// Returns the concatenated clamped `out[]` vector (length =
/// `subframe_samples(bandwidth) * num_subframes`).
///
/// `gains_q16` and `a_q12_per_subframe` must each be `num_subframes`
/// long. `num_subframes` is the SILK subframe count: 2 for 10 ms, 4
/// for 20 ms.
pub fn lpc_synthesis_frame(
    bandwidth: Bandwidth,
    frame_size: SilkFrameSize,
    state: &mut LpcSynthState,
    res: &[f32],
    gains_q16: &[u32],
    a_q12_per_subframe: &[Vec<i16>],
) -> Result<Vec<f32>, Error> {
    let n = subframe_samples(bandwidth)?;
    let num_subframes = match frame_size {
        SilkFrameSize::TenMs => 2,
        SilkFrameSize::TwentyMs => 4,
    };
    if res.len() != n * num_subframes
        || gains_q16.len() != num_subframes
        || a_q12_per_subframe.len() != num_subframes
    {
        return Err(Error::MalformedPacket);
    }

    let mut out = vec![0.0f32; n * num_subframes];
    for s in 0..num_subframes {
        let j = s * n;
        let res_sub = &res[j..j + n];
        let a_sub = &a_q12_per_subframe[s];
        // Borrow a mutable slice for this subframe's output.
        let (head, tail) = out.split_at_mut(j);
        let _ = head; // silence unused warning when s == 0
        let out_sub = &mut tail[..n];
        lpc_synthesis_subframe(bandwidth, state, res_sub, gains_q16[s], a_sub, out_sub)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- subframe_samples / state plumbing -----------------------------

    #[test]
    fn subframe_samples_table() {
        assert_eq!(subframe_samples(Bandwidth::Nb).unwrap(), 40);
        assert_eq!(subframe_samples(Bandwidth::Mb).unwrap(), 60);
        assert_eq!(subframe_samples(Bandwidth::Wb).unwrap(), 80);
        // SWB / FB are SILK-illegal at this stage.
        assert!(subframe_samples(Bandwidth::Swb).is_err());
        assert!(subframe_samples(Bandwidth::Fb).is_err());
    }

    #[test]
    fn state_dlpc_routing() {
        assert_eq!(LpcSynthState::new(Bandwidth::Nb).unwrap().d_lpc(), 10);
        assert_eq!(LpcSynthState::new(Bandwidth::Mb).unwrap().d_lpc(), 10);
        assert_eq!(LpcSynthState::new(Bandwidth::Wb).unwrap().d_lpc(), 16);
        assert!(LpcSynthState::new(Bandwidth::Swb).is_err());
        assert!(LpcSynthState::new(Bandwidth::Fb).is_err());
    }

    #[test]
    fn state_starts_zero_and_resets_zero() {
        let mut s = LpcSynthState::new(Bandwidth::Wb).unwrap();
        assert!(s.history().iter().all(|&x| x == 0.0));
        s.history[3] = 0.5;
        s.reset();
        assert!(s.history().iter().all(|&x| x == 0.0));
    }

    // --- input validation ----------------------------------------------

    #[test]
    fn subframe_rejects_mismatched_res_len() {
        let mut s = LpcSynthState::new(Bandwidth::Nb).unwrap();
        let res = vec![0.0f32; 39]; // wrong: NB needs 40
        let a = vec![0i16; 10];
        let mut out = vec![0.0f32; 39];
        assert!(lpc_synthesis_subframe(Bandwidth::Nb, &mut s, &res, 65536, &a, &mut out).is_err());
    }

    #[test]
    fn subframe_rejects_mismatched_out_len() {
        let mut s = LpcSynthState::new(Bandwidth::Nb).unwrap();
        let res = vec![0.0f32; 40];
        let a = vec![0i16; 10];
        let mut out = vec![0.0f32; 39]; // wrong
        assert!(lpc_synthesis_subframe(Bandwidth::Nb, &mut s, &res, 65536, &a, &mut out).is_err());
    }

    #[test]
    fn subframe_rejects_mismatched_a_len() {
        let mut s = LpcSynthState::new(Bandwidth::Nb).unwrap();
        let res = vec![0.0f32; 40];
        let a = vec![0i16; 9]; // wrong: needs 10
        let mut out = vec![0.0f32; 40];
        assert!(lpc_synthesis_subframe(Bandwidth::Nb, &mut s, &res, 65536, &a, &mut out).is_err());
    }

    // --- algebraic identities ------------------------------------------

    #[test]
    fn all_zero_filter_passes_scaled_residual() {
        // a_Q12[k] = 0 ⇒ no feedback; lpc[i] == (gain_Q16 / 65536) * res[i].
        let mut s = LpcSynthState::new(Bandwidth::Nb).unwrap();
        let mut res = vec![0.0f32; 40];
        for (i, r) in res.iter_mut().enumerate() {
            // arbitrary residual: small floats in [-0.5, 0.5].
            *r = ((i as f32) - 20.0) * 0.01;
        }
        let a = vec![0i16; 10];
        let mut out = vec![0.0f32; 40];
        let gain_q16: u32 = 131072; // 2.0 in Q16
        let lpc =
            lpc_synthesis_subframe(Bandwidth::Nb, &mut s, &res, gain_q16, &a, &mut out).unwrap();
        for i in 0..40 {
            let expect = 2.0 * res[i];
            assert!(
                (lpc[i] - expect).abs() < 1e-6,
                "i={i}: lpc={} expect={}",
                lpc[i],
                expect
            );
            // out is clamp(-1, lpc, 1); all our scaled residuals fit
            // comfortably in [-1, 1].
            assert!((out[i] - expect.clamp(-1.0, 1.0)).abs() < 1e-6);
        }
        // History after this subframe equals the last d_LPC unclamped
        // values.
        let hist = s.history().to_vec();
        for k in 0..10 {
            assert!((hist[k] - lpc[40 - 10 + k]).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_residual_with_zero_history_yields_zero_output() {
        // With history = 0 and residual = 0, the output is identically
        // zero regardless of a_Q12 or gain.
        let mut s = LpcSynthState::new(Bandwidth::Wb).unwrap();
        let res = vec![0.0f32; 80];
        // Random-looking but well-bounded coefficients.
        let mut a = vec![0i16; 16];
        for (k, v) in a.iter_mut().enumerate() {
            *v = ((k as i16) * 37 - 200).clamp(-2048, 2047);
        }
        let mut out = vec![0.0f32; 80];
        let lpc =
            lpc_synthesis_subframe(Bandwidth::Wb, &mut s, &res, 1234567, &a, &mut out).unwrap();
        assert!(lpc.iter().all(|&x| x == 0.0));
        assert!(out.iter().all(|&x| x == 0.0));
        assert!(s.history().iter().all(|&x| x == 0.0));
    }

    // --- hand-pinned single-tap filter ---------------------------------

    #[test]
    fn single_tap_filter_pin_nb_q16_unity() {
        // d_LPC=10, a_Q12 = [4096, 0, 0, ..., 0] ⇒ a[0] = 1.0.
        // gain_Q16 = 65536 ⇒ gain_scale = 1.0.
        // res = [1.0, 0.0, 0.0, ...].
        // history all zero, so:
        //   lpc[0] = 1.0 * 1.0 + (0 * 1.0) = 1.0
        //   lpc[1] = 0.0 + lpc[0] * 1.0  = 1.0
        //   lpc[2] = 0.0 + lpc[1] * 1.0  = 1.0
        //   ...
        // Every sample after the impulse should be exactly 1.0; out[]
        // saturates to 1.0 too (no overshoot).
        let mut s = LpcSynthState::new(Bandwidth::Nb).unwrap();
        let mut res = vec![0.0f32; 40];
        res[0] = 1.0;
        let mut a = vec![0i16; 10];
        a[0] = 4096;
        let mut out = vec![0.0f32; 40];
        let lpc = lpc_synthesis_subframe(Bandwidth::Nb, &mut s, &res, 65536, &a, &mut out).unwrap();
        for (i, v) in lpc.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "i={i}: lpc={v}");
        }
        for (i, v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "i={i}: out={v}");
        }
    }

    #[test]
    fn single_tap_filter_pin_wb_half_gain() {
        // d_LPC=16, a_Q12[0] = 2048 ⇒ a[0] = 0.5. Others zero.
        // gain_Q16 = 32768 ⇒ gain_scale = 0.5.
        // res = [1.0, 0.0, 0.0, ...].
        // History zero:
        //   lpc[0] = 0.5 * 1.0 + 0 = 0.5
        //   lpc[1] = 0.0       + 0.5 * 0.5 = 0.25
        //   lpc[2] = 0.0       + 0.25 * 0.5 = 0.125
        //   lpc[i] = 0.5^(i+1)
        let mut s = LpcSynthState::new(Bandwidth::Wb).unwrap();
        let mut res = vec![0.0f32; 80];
        res[0] = 1.0;
        let mut a = vec![0i16; 16];
        a[0] = 2048;
        let mut out = vec![0.0f32; 80];
        let lpc = lpc_synthesis_subframe(Bandwidth::Wb, &mut s, &res, 32768, &a, &mut out).unwrap();
        for (i, v) in lpc.iter().enumerate() {
            let expect = 0.5f32.powi((i + 1) as i32);
            assert!((v - expect).abs() < 1e-6, "i={i}: lpc={v} expect={expect}");
        }
        // State preserves last 16 unclamped lpc[] samples.
        let hist = s.history().to_vec();
        for (k, h) in hist.iter().enumerate() {
            let i_global = 80 - 16 + k;
            let expect = 0.5f32.powi((i_global + 1) as i32);
            assert!((h - expect).abs() < 1e-9);
        }
    }

    // --- multi-tap filter pin ------------------------------------------

    #[test]
    fn two_tap_filter_pin_nb_hand_traced() {
        // d_LPC=10, a_Q12 = [4096/2, 4096/4, 0, ...] = a = [0.5, 0.25].
        // gain_Q16 = 65536 (scale = 1.0). res = [1.0, 2.0, 3.0, 0,...].
        // history = [0; 10].
        // lpc[0] = 1*1 + lpc[-1]*0.5 + lpc[-2]*0.25 = 1.0
        // lpc[1] = 1*2 + lpc[ 0]*0.5 + lpc[-1]*0.25 = 2.5
        // lpc[2] = 1*3 + lpc[ 1]*0.5 + lpc[ 0]*0.25 = 4.5
        // lpc[3] = 0   + lpc[ 2]*0.5 + lpc[ 1]*0.25 = 2.875
        // lpc[4] = 0   + lpc[ 3]*0.5 + lpc[ 2]*0.25 = 2.5625
        // lpc[5] = 0   + lpc[ 4]*0.5 + lpc[ 3]*0.25 = 1.99...
        let mut s = LpcSynthState::new(Bandwidth::Nb).unwrap();
        let mut res = vec![0.0f32; 40];
        res[0] = 1.0;
        res[1] = 2.0;
        res[2] = 3.0;
        let mut a = vec![0i16; 10];
        a[0] = 2048; // 0.5
        a[1] = 1024; // 0.25
        let mut out = vec![0.0f32; 40];
        let lpc = lpc_synthesis_subframe(Bandwidth::Nb, &mut s, &res, 65536, &a, &mut out).unwrap();

        let expected = [1.0f32, 2.5, 4.5, 2.875, 2.5625];
        for (i, ex) in expected.iter().enumerate() {
            assert!(
                (lpc[i] - ex).abs() < 1e-5,
                "i={i}: lpc={} expect={}",
                lpc[i],
                ex
            );
        }
        // out[] for lpc[2] = 4.5 clamps to 1.0; lpc[3] = 2.875 clamps
        // to 1.0; lpc[4] = 2.5625 clamps to 1.0.
        assert!((out[2] - 1.0).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
        assert!((out[4] - 1.0).abs() < 1e-6);
        // out[0] = 1.0 (boundary). out[1] = 1.0 (clamp).
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    // --- history carry-over --------------------------------------------

    #[test]
    fn history_carries_between_subframes() {
        // Set up a single-tap unity feedback filter and feed a single
        // impulse in subframe 0; subframe 1 must continue emitting 1.0
        // forever because the history kept lpc[39] = 1.0.
        let mut s = LpcSynthState::new(Bandwidth::Nb).unwrap();
        let mut res = vec![0.0f32; 40];
        res[0] = 1.0;
        let mut a = vec![0i16; 10];
        a[0] = 4096;
        let mut out0 = vec![0.0f32; 40];
        lpc_synthesis_subframe(Bandwidth::Nb, &mut s, &res, 65536, &a, &mut out0).unwrap();
        // After subframe 0, history[k] should all be 1.0 for k in 0..10.
        for v in s.history() {
            assert!((v - 1.0).abs() < 1e-6);
        }
        // Now subframe 1 with zero residual: lpc[i] = 0 + lpc[i-1] * 1.0 = 1.0.
        let res1 = vec![0.0f32; 40];
        let mut out1 = vec![0.0f32; 40];
        let lpc1 =
            lpc_synthesis_subframe(Bandwidth::Nb, &mut s, &res1, 65536, &a, &mut out1).unwrap();
        for (i, v) in lpc1.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "i={i}: lpc1={v}");
        }
    }

    #[test]
    fn reset_zeroes_history_for_decoder_reset_path() {
        let mut s = LpcSynthState::new(Bandwidth::Nb).unwrap();
        // Inject non-zero history by running a quick subframe.
        let mut res = vec![0.0f32; 40];
        res[0] = 1.0;
        let mut a = vec![0i16; 10];
        a[0] = 4096;
        let mut out = vec![0.0f32; 40];
        lpc_synthesis_subframe(Bandwidth::Nb, &mut s, &res, 65536, &a, &mut out).unwrap();
        assert!(s.history().iter().any(|&x| x != 0.0));
        s.reset();
        assert!(s.history().iter().all(|&x| x == 0.0));
    }

    // --- clamping post-condition ---------------------------------------

    #[test]
    fn out_always_in_minus_one_one() {
        // Drive the filter hard so lpc[] overshoots; out[] must clamp.
        let mut s = LpcSynthState::new(Bandwidth::Wb).unwrap();
        let mut res = vec![0.0f32; 80];
        for (i, r) in res.iter_mut().enumerate() {
            *r = if i % 2 == 0 { 10.0 } else { -10.0 };
        }
        // a[0] near unity, others zero.
        let mut a = vec![0i16; 16];
        a[0] = 4000;
        let mut out = vec![0.0f32; 80];
        let _lpc =
            lpc_synthesis_subframe(Bandwidth::Wb, &mut s, &res, 65536, &a, &mut out).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert!((-1.0..=1.0).contains(v), "i={i}: out={v} outside [-1, 1]");
        }
    }

    // --- §4.2.7.9.2 wording cross-check: history holds unclamped, not
    //     clamped values --------------------------------------------------

    #[test]
    fn history_stores_unclamped_lpc_not_out() {
        // Use a filter that drives lpc[i] > 1.0 in the last d_LPC
        // samples and confirm history[] holds the unclamped values
        // (which exceed 1.0), not the saturated out[] (= 1.0).
        let mut s = LpcSynthState::new(Bandwidth::Nb).unwrap();
        let res = vec![5.0f32; 40];
        let mut a = vec![0i16; 10];
        a[0] = 0; // no feedback, lpc[i] = 5.0 for all i.
        let mut out = vec![0.0f32; 40];
        let _ = lpc_synthesis_subframe(Bandwidth::Nb, &mut s, &res, 65536, &a, &mut out).unwrap();
        for v in s.history() {
            assert!(
                (v - 5.0).abs() < 1e-6,
                "history must be unclamped 5.0, got {v}"
            );
        }
        for v in &out {
            assert!((v - 1.0).abs() < 1e-6, "out must clamp to 1.0, got {v}");
        }
    }

    // --- lpc_synthesis_frame wrapper -----------------------------------

    #[test]
    fn frame_wrapper_matches_per_subframe() {
        // Run the frame wrapper and an explicit per-subframe loop with
        // the same inputs; outputs must agree bit-for-bit.
        let bandwidth = Bandwidth::Wb;
        let frame_size = SilkFrameSize::TwentyMs;
        let n = subframe_samples(bandwidth).unwrap();
        let num_subframes = 4;
        let mut res = vec![0.0f32; n * num_subframes];
        for (i, r) in res.iter_mut().enumerate() {
            *r = ((i as f32) * 0.013).sin() * 0.5;
        }
        let gains_q16 = vec![65536u32, 49152, 32768, 24576];
        let mut a_per: Vec<Vec<i16>> = Vec::with_capacity(num_subframes);
        for s in 0..num_subframes {
            let mut row = vec![0i16; 16];
            row[0] = (1000 + (s as i16) * 200).min(2047);
            row[1] = -((s as i16) * 100);
            a_per.push(row);
        }

        let mut state_a = LpcSynthState::new(bandwidth).unwrap();
        let out_wrapper = lpc_synthesis_frame(
            bandwidth,
            frame_size,
            &mut state_a,
            &res,
            &gains_q16,
            &a_per,
        )
        .unwrap();

        let mut state_b = LpcSynthState::new(bandwidth).unwrap();
        let mut out_manual = vec![0.0f32; n * num_subframes];
        for s in 0..num_subframes {
            let j = s * n;
            let res_sub = &res[j..j + n];
            let (head, tail) = out_manual.split_at_mut(j);
            let _ = head;
            let out_sub = &mut tail[..n];
            lpc_synthesis_subframe(
                bandwidth,
                &mut state_b,
                res_sub,
                gains_q16[s],
                &a_per[s],
                out_sub,
            )
            .unwrap();
        }
        assert_eq!(out_wrapper, out_manual);
        assert_eq!(state_a.history(), state_b.history());
    }

    #[test]
    fn frame_wrapper_rejects_bad_lengths() {
        let mut s = LpcSynthState::new(Bandwidth::Nb).unwrap();
        let res = vec![0.0f32; 79]; // 40*2 = 80 expected
        let gains = vec![65536u32, 65536];
        let a: Vec<Vec<i16>> = vec![vec![0i16; 10]; 2];
        assert!(lpc_synthesis_frame(
            Bandwidth::Nb,
            SilkFrameSize::TenMs,
            &mut s,
            &res,
            &gains,
            &a
        )
        .is_err());

        // gain length wrong.
        let res = vec![0.0f32; 80];
        let gains = vec![65536u32];
        assert!(lpc_synthesis_frame(
            Bandwidth::Nb,
            SilkFrameSize::TenMs,
            &mut s,
            &res,
            &gains,
            &a
        )
        .is_err());
    }

    // --- a sweep that exercises every bandwidth × frame size -----------

    #[test]
    fn no_panic_sweep_all_bandwidths_frame_sizes() {
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            for fs in [SilkFrameSize::TenMs, SilkFrameSize::TwentyMs] {
                let n = subframe_samples(bw).unwrap();
                let num_subframes = match fs {
                    SilkFrameSize::TenMs => 2,
                    SilkFrameSize::TwentyMs => 4,
                };
                let total = n * num_subframes;
                let mut res = vec![0.0f32; total];
                for (i, r) in res.iter_mut().enumerate() {
                    *r = ((i as f32) * 0.0173).cos() * 0.3;
                }
                let gains: Vec<u32> = (0..num_subframes)
                    .map(|s| 50000 + (s as u32) * 30000)
                    .collect();
                let d_lpc = if matches!(bw, Bandwidth::Wb) { 16 } else { 10 };
                let a_per: Vec<Vec<i16>> = (0..num_subframes)
                    .map(|s| {
                        let mut row = vec![0i16; d_lpc];
                        // gentle coefficients that won't blow up.
                        row[0] = 1000 + (s as i16) * 50;
                        if d_lpc > 1 {
                            row[1] = -200;
                        }
                        row
                    })
                    .collect();
                let mut state = LpcSynthState::new(bw).unwrap();
                let out = lpc_synthesis_frame(bw, fs, &mut state, &res, &gains, &a_per).unwrap();
                assert_eq!(out.len(), total);
                // Clamping post-condition holds everywhere.
                for v in &out {
                    assert!(
                        (-1.0..=1.0).contains(v),
                        "clamping post-condition violated: out={v}"
                    );
                }
                assert_eq!(state.history().len(), d_lpc);
            }
        }
    }
}
