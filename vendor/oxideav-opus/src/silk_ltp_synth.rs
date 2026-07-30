//! SILK §4.2.7.9.1 LTP synthesis filter — RFC 6716.
//!
//! Per-subframe production of the LPC residual `res[i]` that feeds the
//! §4.2.7.9.2 LPC synthesis filter. Two regimes:
//!
//!  * **Unvoiced** (`signal_type != Voiced`). The residual is just a
//!    normalised copy of the §4.2.7.8 excitation:
//!
//!    ```text
//!                 e_Q23[i]
//!       res[i] = ---------
//!                 2.0**23
//!    ```
//!
//!  * **Voiced**. The §4.2.7.6 LTP filter is applied. Before the filter
//!    convolution runs, the prior subframes' output samples are
//!    "rewhitened" back into LPC-residual values using the current
//!    subframe's LPC coefficients (because the LPC coefficients may have
//!    changed between subframes). Two rewhitening regions:
//!
//!     * **Out-rewhitening region.** Indices
//!       `(j - pitch_lags[s] - 2) <= i < out_end` rewhiten from the
//!       *clamped* `out[]` history per
//!
//!       ```text
//!                   4.0 * LTP_scale_Q14
//!         res[i] = -------------------- * clamp(-1.0,
//!                       gain_Q16[s]
//!                                                 d_LPC-1
//!                                                    __                a_Q12[k]
//!                                          out[i] -  \  out[i-k-1] * --------, 1.0)
//!                                                    /_                4096.0
//!                                                    k=0
//!       ```
//!
//!     * **Lpc-rewhitening region.** Indices `out_end <= i < j` rewhiten
//!       from the *unclamped* `lpc[]` history per
//!
//!       ```text
//!                       65536.0                d_LPC-1
//!                                                  __              a_Q12[k]
//!         res[i] = ----------- * (lpc[i] - \  lpc[i-k-1] * --------)
//!                  gain_Q16[s]             /_               4096.0
//!                                          k=0
//!       ```
//!
//!  * **Convolution.** Finally, for `j <= i < (j + n)`:
//!
//!    ```text
//!                                4
//!                  e_Q23[i]      __                                    b_Q7[k]
//!         res[i] = --------- + \  res[i - pitch_lags[s] + 2 - k] * -------
//!                  2.0**23     /_                                    128.0
//!                              k=0
//!    ```
//!
//! `out_end` and `LTP_scale_Q14` follow the §4.2.7.9.1 split:
//!
//!  * Third or fourth subframe of a 20 ms SILK frame with `w_Q2 < 4`:
//!    `out_end = j - (s - 2) * n` and `LTP_scale_Q14 = 16384`. This is the
//!    "LSF interp split" case: the second half of a 20 ms frame uses fresh
//!    (uninterpolated) LPC coefficients, and the rewhitening only goes back
//!    over the half-frame boundary.
//!  * Otherwise: `out_end = j - s * n` (rewhitens over every prior subframe
//!    in the current SILK frame) and `LTP_scale_Q14` is the §4.2.7.6.3
//!    decoded scaling factor.
//!
//! Buffer sizes come straight from §4.2.7.9.1:
//!
//!  * `out[]` requires up to 306 samples (WB max pitch 288 + d_LPC 16 + 2).
//!  * `lpc[]` requires up to 256 samples (240 from the current SILK frame
//!    plus 16 from the previous frame).
//!
//! Both buffers are cleared to zero on a decoder reset or after an uncoded
//! regular SILK frame for the side channel (RFC 6716 §4.2.7.9.1 + §4.5.2).
//!
//! All truth is taken from RFC 6716 §4.2.7.9.1. No external library source
//! is consulted.

use crate::silk_excitation::SilkFrameSize;
use crate::silk_frame::SignalType;
use crate::silk_lpc_synth::{subframe_samples, LPC_SYNTH_MAX_ORDER};
use crate::silk_lsf_stage2::{D_LPC_NB_MB, D_LPC_WB};
use crate::silk_ltp::LTP_FILTER_TAPS;
use crate::toc::Bandwidth;
use crate::Error;

/// Maximum pitch lag across SILK bandwidths (WB `lag_max`, §4.2.7.6.1).
pub const LTP_MAX_PITCH_LAG: usize = 288;

/// Maximum `out[]` history needed for §4.2.7.9.1: `lag_max + d_LPC + 2`
/// = 288 + 16 + 2 = 306 samples (WB worst case).
pub const LTP_OUT_HISTORY_MAX: usize = LTP_MAX_PITCH_LAG + LPC_SYNTH_MAX_ORDER + 2;

/// Maximum `lpc[]` history needed for §4.2.7.9.1: 3 prior WB subframes
/// (3 × 80) + d_LPC (16) = 256 samples (WB worst case).
pub const LTP_LPC_HISTORY_MAX: usize = 3 * 80 + LPC_SYNTH_MAX_ORDER;

/// Q14 "fresh-LPC" LTP scaling override used in the third/fourth subframe
/// of a 20 ms SILK frame with `w_Q2 < 4` per RFC 6716 §4.2.7.9.1.
pub const LTP_SCALE_FRESH_Q14: u16 = 16384;

/// SILK §4.2.7.9.1 state buffer for the LTP synthesis filter.
///
/// Holds the prior-subframe `out[]` (clamped) and `lpc[]` (unclamped)
/// histories needed for rewhitening. Both are cleared to zero on a decoder
/// reset (§4.5.2) or after an uncoded regular SILK frame for the side
/// channel.
///
/// The buffers track samples from the most recent ones (highest index) to
/// the oldest, in source order — i.e. `out_history[N-1]` is the most
/// recently produced output sample, `out_history[0]` is the oldest.
#[derive(Debug, Clone)]
pub struct LtpSynthState {
    bandwidth: Bandwidth,
    d_lpc: usize,
    /// `out[]` history, MSB-recent. Always carries `LTP_OUT_HISTORY_MAX`
    /// samples; the oldest are discarded as new subframes push samples
    /// in.
    out_history: Vec<f32>,
    /// `lpc[]` history (unclamped values from previous subframes within
    /// the current SILK frame, plus `d_LPC` carry-over from the prior
    /// SILK frame). MSB-recent. Always carries `LTP_LPC_HISTORY_MAX`
    /// samples.
    lpc_history: Vec<f32>,
    /// Position within the current SILK frame of the next subframe to
    /// be processed (0..num_subframes). Used to drive `out_end` and the
    /// LSF-interpolation-split decision in §4.2.7.9.1.
    subframe_index: u8,
}

impl LtpSynthState {
    /// Construct a fresh zero-initialised state for `bandwidth`.
    ///
    /// Rejects SWB / FB (the SILK layer never sees them after the §4.2.2
    /// hybrid split).
    pub fn new(bandwidth: Bandwidth) -> Result<Self, Error> {
        let d_lpc = match bandwidth {
            Bandwidth::Nb | Bandwidth::Mb => D_LPC_NB_MB,
            Bandwidth::Wb => D_LPC_WB,
            _ => return Err(Error::MalformedPacket),
        };
        Ok(Self {
            bandwidth,
            d_lpc,
            out_history: vec![0.0; LTP_OUT_HISTORY_MAX],
            lpc_history: vec![0.0; LTP_LPC_HISTORY_MAX],
            subframe_index: 0,
        })
    }

    /// Bandwidth this state was created for.
    pub fn bandwidth(&self) -> Bandwidth {
        self.bandwidth
    }

    /// `d_LPC` for the bandwidth.
    pub fn d_lpc(&self) -> usize {
        self.d_lpc
    }

    /// Position within the current SILK frame of the next subframe to be
    /// processed (0..=num_subframes).
    pub fn subframe_index(&self) -> u8 {
        self.subframe_index
    }

    /// Read-only access to the `out[]` history (most-recent at index
    /// `len()-1`, oldest at index `0`).
    pub fn out_history(&self) -> &[f32] {
        &self.out_history
    }

    /// Read-only access to the `lpc[]` history (most-recent at index
    /// `len()-1`, oldest at index `0`).
    pub fn lpc_history(&self) -> &[f32] {
        &self.lpc_history
    }

    /// Reset to zero histories and start-of-frame position (RFC 6716
    /// §4.5.2 decoder-reset path, or after an uncoded regular SILK frame
    /// for the side channel).
    pub fn reset(&mut self) {
        for v in self.out_history.iter_mut() {
            *v = 0.0;
        }
        for v in self.lpc_history.iter_mut() {
            *v = 0.0;
        }
        self.subframe_index = 0;
    }

    /// Mark the start of a new SILK frame — clears the in-frame subframe
    /// counter without touching the cross-frame `out[]` / `lpc[]` histories
    /// (those persist across SILK frames per §4.2.7.9.1).
    pub fn start_frame(&mut self) {
        self.subframe_index = 0;
    }

    /// Push `n` newly produced subframe samples (out[], unclamped lpc[])
    /// into the history buffers, shifting the oldest out. `out_clamped`
    /// and `lpc_unclamped` are the §4.2.7.9.2 outputs for the just-
    /// completed subframe.
    fn push_subframe(&mut self, out_clamped: &[f32], lpc_unclamped: &[f32]) {
        let n = out_clamped.len();
        debug_assert_eq!(n, lpc_unclamped.len());
        // Shift older samples down.
        if n < self.out_history.len() {
            self.out_history.copy_within(n.., 0);
        }
        if n < self.lpc_history.len() {
            self.lpc_history.copy_within(n.., 0);
        }
        // Place the new samples at the tail.
        let out_tail = self.out_history.len() - n;
        self.out_history[out_tail..].copy_from_slice(out_clamped);
        let lpc_tail = self.lpc_history.len() - n;
        self.lpc_history[lpc_tail..].copy_from_slice(lpc_unclamped);
        self.subframe_index = self.subframe_index.saturating_add(1);
    }
}

/// Caller-supplied parameters for one subframe of LTP synthesis.
///
/// Fields are sourced as follows:
///
///  * `signal_type` — §4.2.7.3.
///  * `frame_size` — driven from the SILK frame's 10/20 ms decision.
///  * `subframe_index` — 0..num_subframes for the subframe being produced.
///  * `gain_q16` — §4.2.7.4 dequantised Q16 gain for this subframe.
///  * `pitch_lag` — §4.2.7.6.1 per-subframe pitch lag (only used for
///    voiced frames).
///  * `b_q7` — §4.2.7.6.2 5-tap Q7 LTP filter coefficients (only used for
///    voiced frames).
///  * `ltp_scaling_q14` — §4.2.7.6.3 Q14 LTP scaling factor for the
///    current SILK frame.
///  * `a_q12` — §4.2.7.5.8 stabilised Q12 LPC predictor for this subframe.
///  * `lsf_interp_used` — true if the **first half** of the current 20 ms
///    SILK frame used a `w_Q2 < 4` LSF interpolation (i.e. fresh LPC
///    coefficients for the second half). Drives the §4.2.7.9.1 "third/
///    fourth subframe" branch.
#[derive(Debug, Clone, Copy)]
pub struct LtpSynthSubframe<'a> {
    pub bandwidth: Bandwidth,
    pub signal_type: SignalType,
    pub frame_size: SilkFrameSize,
    pub subframe_index: u8,
    pub gain_q16: u32,
    pub pitch_lag: i32,
    pub b_q7: [i8; LTP_FILTER_TAPS],
    pub ltp_scaling_q14: u16,
    pub a_q12: &'a [i16],
    pub lsf_interp_used: bool,
}

/// Run one subframe of §4.2.7.9.1 LTP synthesis, producing the LPC
/// residual `res[i]` for the LPC synthesis filter that follows.
///
/// `e_q23` is the §4.2.7.8 excitation for the subframe (length =
/// `subframe_samples(bandwidth)`). `res_out` is the destination
/// residual buffer (same length).
///
/// For voiced frames `state` must already carry the prior-subframe
/// histories. For unvoiced frames the histories are not consulted. Either
/// way the caller must invoke [`LtpSynthState::push_subframe`] (via the
/// companion [`ltp_synth_commit_subframe`] helper) once it has run the
/// §4.2.7.9.2 LPC synthesis and has both the clamped `out[]` and
/// unclamped `lpc[]` for the subframe in hand.
///
/// Errors:
///
///  * `Error::MalformedPacket` if `e_q23.len()` / `res_out.len()` /
///    `a_q12.len()` disagree with the expected per-subframe sizes for
///    `bandwidth`, or if the frame_size / num_subframes / subframe_index
///    triple is internally inconsistent.
pub fn ltp_synthesis_subframe(
    state: &LtpSynthState,
    cfg: LtpSynthSubframe<'_>,
    e_q23: &[i32],
    res_out: &mut [f32],
) -> Result<(), Error> {
    let n = subframe_samples(cfg.bandwidth)?;
    if e_q23.len() != n || res_out.len() != n {
        return Err(Error::MalformedPacket);
    }
    if cfg.a_q12.len() != state.d_lpc {
        return Err(Error::MalformedPacket);
    }
    if state.bandwidth != cfg.bandwidth {
        return Err(Error::MalformedPacket);
    }
    let num_subframes = match cfg.frame_size {
        SilkFrameSize::TenMs => 2u8,
        SilkFrameSize::TwentyMs => 4u8,
    };
    if cfg.subframe_index >= num_subframes {
        return Err(Error::MalformedPacket);
    }

    // --- unvoiced: trivial copy ---------------------------------------
    if cfg.signal_type != SignalType::Voiced {
        for i in 0..n {
            res_out[i] = (e_q23[i] as f32) / 8_388_608.0; // 2^23
        }
        return Ok(());
    }

    // --- voiced: full rewhiten + LTP convolution ----------------------
    // The subframe occupies indices [j, j+n) in the SILK-frame coordinate
    // system. The histories carry "lookback" samples — what the spec
    // calls out[i] / lpc[i] for i < j.
    //
    // We need two rewhitening regions:
    //
    //   region A:  (j - pitch_lag - 2) <= i <  out_end    (from out[])
    //   region B:           out_end    <= i <  j          (from lpc[])
    //
    // After rewhitening, run the 5-tap LTP convolution for j <= i < j+n.
    //
    // Working coordinate: use a single buffer `res_buf` whose index 0 is
    // the leftmost-needed residual sample. j is `pitch_lag + 2` in this
    // local frame.
    let s = cfg.subframe_index as usize;
    let pitch_lag = cfg.pitch_lag;
    if pitch_lag <= 0 {
        return Err(Error::MalformedPacket);
    }
    let d_lpc = state.d_lpc;

    // out_end and effective LTP_scale_Q14 per §4.2.7.9.1.
    let (out_end_offset_from_j, ltp_scale_q14): (i32, u16) = {
        let n_i = n as i32;
        let s_i = s as i32;
        let split_branch =
            matches!(cfg.frame_size, SilkFrameSize::TwentyMs) && (s >= 2) && cfg.lsf_interp_used;
        if split_branch {
            (-(s_i - 2) * n_i, LTP_SCALE_FRESH_Q14)
        } else {
            (-s_i * n_i, cfg.ltp_scaling_q14)
        }
    };
    // out_end and the start of region A in the SILK-frame coordinate system
    // expressed as offsets from j (always <= 0).
    let region_a_start_off = -(pitch_lag + 2); // = j - pitch_lag - 2 from j
    let out_end_off = out_end_offset_from_j; // <= 0

    // The "rewhitening lookback" is the union of region A
    // [region_a_start_off, out_end_off) and region B [out_end_off, 0).
    // The earliest offset we may need to address is min(region_a_start_off,
    // out_end_off). Note that when LSF-interp-split fires for s >= 2 and
    // s*n > pitch_lag + 2 (e.g. subframe 3, s=3, s*n=240 > pitch_lag+2),
    // out_end_off can be more negative than region_a_start_off — region A
    // becomes effectively empty (out_end <= region_A_start), but region B
    // still extends back to out_end_off.
    //
    // In the split branch (out_end = j - (s-2)*n), out_end_off may be
    // closer to zero than region_a_start_off (region B small, region A
    // large), or it may be farther from zero (region A empty, region B
    // large). Either way the lookback must cover both regions.
    let earliest_off = region_a_start_off.min(out_end_off);
    let lookback = (-earliest_off) as usize;
    // res_buf indices: 0..lookback for the rewhitened region; lookback..lookback+n
    // for the produced subframe. The offset-to-index map is
    //   index(off) = off - earliest_off.
    let total = lookback + n;
    let mut res_buf = vec![0.0f32; total];

    // Pre-scale Q12 coefficients to floating point once.
    let mut a_f = [0.0f32; LPC_SYNTH_MAX_ORDER];
    for (slot, &q12) in a_f.iter_mut().zip(cfg.a_q12.iter()).take(d_lpc) {
        *slot = (q12 as f32) / 4096.0;
    }
    let gain_q16 = cfg.gain_q16 as f32;
    let inv_gain = 65536.0 / gain_q16;
    let ltp_scale_factor = 4.0 * (ltp_scale_q14 as f32) / gain_q16;

    // Convenience: fetch out[j + off] from the state history when off < 0.
    // out_history is MSB-recent: out_history[len-1] = out[j-1]
    // (j-1 corresponds to off = -1).
    let out_hist = state.out_history();
    let out_hist_len = out_hist.len();
    let fetch_out = |off: i32| -> f32 {
        // off must be <= -1 (else we'd be reading current/future samples).
        // out_history[len - 1 + off + 1] = out_history[len + off].
        let idx = (out_hist_len as i32) + off;
        if (0..out_hist_len as i32).contains(&idx) {
            out_hist[idx as usize]
        } else {
            // Past the start of stored history → zero (decoder reset
            // default).
            0.0
        }
    };

    let lpc_hist = state.lpc_history();
    let lpc_hist_len = lpc_hist.len();
    let fetch_lpc = |off: i32| -> f32 {
        let idx = (lpc_hist_len as i32) + off;
        if (0..lpc_hist_len as i32).contains(&idx) {
            lpc_hist[idx as usize]
        } else {
            0.0
        }
    };

    // --- region A: rewhiten out[] -------------------------------------
    // For i such that (j - pitch_lag - 2) <= i < out_end:
    //   res[i] = (4*LTP_scale / gain) * clamp(-1.0, out[i] - Σ out[i-k-1] * a_Q12[k]/4096, 1.0)
    //
    // Loop is empty when region_a_start_off >= out_end_off (region A
    // collapses; only region B is active).
    for off in region_a_start_off..out_end_off {
        // out[i] at offset `off` (< 0): from out_history.
        let out_i = fetch_out(off);
        let mut sum = 0.0f32;
        for (k, &af) in a_f.iter().enumerate().take(d_lpc) {
            // out[i - k - 1] at offset `off - k - 1`.
            sum += fetch_out(off - (k as i32) - 1) * af;
        }
        let predicted = (out_i - sum).clamp(-1.0, 1.0);
        let res_idx = (off - earliest_off) as usize;
        res_buf[res_idx] = ltp_scale_factor * predicted;
    }

    // --- region B: rewhiten lpc[] -------------------------------------
    // For i such that out_end <= i < j:
    //   res[i] = (65536 / gain) * (lpc[i] - Σ lpc[i-k-1] * a_Q12[k]/4096)
    for off in out_end_off..0 {
        let lpc_i = fetch_lpc(off);
        let mut sum = 0.0f32;
        for (k, &af) in a_f.iter().enumerate().take(d_lpc) {
            sum += fetch_lpc(off - (k as i32) - 1) * af;
        }
        let res_idx = (off - earliest_off) as usize;
        res_buf[res_idx] = inv_gain * (lpc_i - sum);
    }

    // --- main convolution: produce j..j+n -----------------------------
    // res[i] = e_Q23[i]/2^23 + Σ_{k=0..4} res[i - pitch_lag + 2 - k] * b_Q7[k]/128
    //
    // In local-buffer indices, i corresponds to res_buf[lookback + (i - j)],
    // and i - pitch_lag + 2 - k corresponds to
    //   res_buf[lookback + (i - j) - pitch_lag + 2 - k]
    //   = res_buf[lookback - pitch_lag + 2 - k + local_i].
    //
    // Note lookback = pitch_lag + 2, so lookback - pitch_lag + 2 - k = 4 - k.
    // That means at the very first sample (local_i = 0, k = 0..4) we read
    // res_buf[4 - k], i.e. res_buf[4], [3], [2], [1], [0] — all in the
    // rewhitening region.
    let mut b_f = [0.0f32; LTP_FILTER_TAPS];
    for (slot, &q7) in b_f.iter_mut().zip(cfg.b_q7.iter()) {
        *slot = (q7 as f32) / 128.0;
    }
    for (local_i, (e_sample, out_sample)) in e_q23.iter().zip(res_out.iter_mut()).enumerate() {
        let buf_i = lookback + local_i;
        let mut sum = (*e_sample as f32) / 8_388_608.0;
        for (k, &bf) in b_f.iter().enumerate() {
            // res_buf index = buf_i - pitch_lag + 2 - k. The spec's
            // §4.2.7.6.1 lag_min ≥ 16 guarantees pitch_lag - 2 + k ≥ 14 > 0
            // for k ≤ 4, so `src < buf_i` always; and earliest_off chosen
            // above guarantees `src >= 0` whenever the reference falls in
            // the SILK-frame's stored lookback.
            let src = (buf_i as i32) - pitch_lag + 2 - (k as i32);
            if src >= 0 && (src as usize) < total {
                sum += res_buf[src as usize] * bf;
            }
            // src < 0 (only possible if pitch_lag < k - 2 + buf_i; in
            // practice never hit since lag_min ≥ 16 and pitch_lag - 2 + k
            // < lookback always). Treat as zero — matches §4.2.7.9.1's
            // implicit "prior res[] absent → zero".
        }
        res_buf[buf_i] = sum;
        *out_sample = sum;
    }

    Ok(())
}

/// Commit the §4.2.7.9.2 outputs back into `state` so the next subframe's
/// rewhitening sees them. Must be called once per subframe after the LPC
/// synthesis filter has run.
///
/// `out_clamped` is the clamped output signal for the just-produced
/// subframe (§4.2.7.9.2 `out[i]`); `lpc_unclamped` is the unclamped
/// equivalent (§4.2.7.9.2 `lpc[i]`). Both must have length equal to
/// `subframe_samples(state.bandwidth())`.
pub fn ltp_synth_commit_subframe(
    state: &mut LtpSynthState,
    out_clamped: &[f32],
    lpc_unclamped: &[f32],
) -> Result<(), Error> {
    let n = subframe_samples(state.bandwidth)?;
    if out_clamped.len() != n || lpc_unclamped.len() != n {
        return Err(Error::MalformedPacket);
    }
    state.push_subframe(out_clamped, lpc_unclamped);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_a_zero(d_lpc: usize) -> Vec<i16> {
        vec![0i16; d_lpc]
    }

    fn make_b_zero() -> [i8; LTP_FILTER_TAPS] {
        [0; LTP_FILTER_TAPS]
    }

    // ----- constants and constructor ---------------------------------------

    #[test]
    fn constants_match_spec_paragraphs() {
        // §4.2.7.9.1 explicitly states the buffer sizes.
        assert_eq!(LTP_MAX_PITCH_LAG, 288);
        assert_eq!(LTP_OUT_HISTORY_MAX, 306);
        assert_eq!(LTP_LPC_HISTORY_MAX, 256);
        assert_eq!(LTP_SCALE_FRESH_Q14, 16384);
    }

    #[test]
    fn state_new_routes_d_lpc() {
        assert_eq!(LtpSynthState::new(Bandwidth::Nb).unwrap().d_lpc(), 10);
        assert_eq!(LtpSynthState::new(Bandwidth::Mb).unwrap().d_lpc(), 10);
        assert_eq!(LtpSynthState::new(Bandwidth::Wb).unwrap().d_lpc(), 16);
        assert!(LtpSynthState::new(Bandwidth::Swb).is_err());
        assert!(LtpSynthState::new(Bandwidth::Fb).is_err());
    }

    #[test]
    fn state_starts_zero_resets_zero() {
        let mut s = LtpSynthState::new(Bandwidth::Wb).unwrap();
        assert!(s.out_history().iter().all(|&x| x == 0.0));
        assert!(s.lpc_history().iter().all(|&x| x == 0.0));
        assert_eq!(s.subframe_index(), 0);
        // Inject some samples and confirm reset clears.
        let dummy_out = vec![0.5f32; 80];
        let dummy_lpc = vec![0.7f32; 80];
        ltp_synth_commit_subframe(&mut s, &dummy_out, &dummy_lpc).unwrap();
        assert!(s.out_history().iter().any(|&x| x != 0.0));
        assert!(s.lpc_history().iter().any(|&x| x != 0.0));
        assert_eq!(s.subframe_index(), 1);
        s.reset();
        assert!(s.out_history().iter().all(|&x| x == 0.0));
        assert!(s.lpc_history().iter().all(|&x| x == 0.0));
        assert_eq!(s.subframe_index(), 0);
    }

    #[test]
    fn start_frame_clears_subframe_index_but_keeps_history() {
        let mut s = LtpSynthState::new(Bandwidth::Nb).unwrap();
        let dummy_out = vec![0.25f32; 40];
        let dummy_lpc = vec![0.125f32; 40];
        ltp_synth_commit_subframe(&mut s, &dummy_out, &dummy_lpc).unwrap();
        assert_eq!(s.subframe_index(), 1);
        // Histories are non-zero.
        assert!(s.out_history().iter().any(|&x| x != 0.0));
        s.start_frame();
        assert_eq!(s.subframe_index(), 0);
        // History persists across frames.
        assert!(s.out_history().iter().any(|&x| x != 0.0));
        assert!(s.lpc_history().iter().any(|&x| x != 0.0));
    }

    #[test]
    fn push_subframe_keeps_most_recent_at_tail() {
        // After pushing a known pattern, the last n samples should be at
        // the very end of the out_history slice.
        let mut s = LtpSynthState::new(Bandwidth::Nb).unwrap();
        let n = 40;
        let mut out_a = vec![0.0f32; n];
        let mut lpc_a = vec![0.0f32; n];
        for i in 0..n {
            out_a[i] = (i as f32) * 0.01;
            lpc_a[i] = (i as f32) * 0.02;
        }
        ltp_synth_commit_subframe(&mut s, &out_a, &lpc_a).unwrap();
        let oh = s.out_history();
        let lh = s.lpc_history();
        for i in 0..n {
            assert!((oh[oh.len() - n + i] - out_a[i]).abs() < 1e-7);
            assert!((lh[lh.len() - n + i] - lpc_a[i]).abs() < 1e-7);
        }
        // Push a second batch; the first batch shifts down.
        let mut out_b = vec![0.0f32; n];
        let mut lpc_b = vec![0.0f32; n];
        for i in 0..n {
            out_b[i] = 1.0 + (i as f32) * 0.01;
            lpc_b[i] = 2.0 + (i as f32) * 0.02;
        }
        ltp_synth_commit_subframe(&mut s, &out_b, &lpc_b).unwrap();
        let oh = s.out_history();
        let lh = s.lpc_history();
        // The second batch is now at the tail.
        for i in 0..n {
            assert!((oh[oh.len() - n + i] - out_b[i]).abs() < 1e-7);
            assert!((lh[lh.len() - n + i] - lpc_b[i]).abs() < 1e-7);
        }
        // The first batch is the next n samples back.
        for i in 0..n {
            assert!((oh[oh.len() - 2 * n + i] - out_a[i]).abs() < 1e-7);
            assert!((lh[lh.len() - 2 * n + i] - lpc_a[i]).abs() < 1e-7);
        }
    }

    // ----- unvoiced: trivial e_Q23/2^23 -----------------------------------

    #[test]
    fn unvoiced_residual_is_excitation_scaled() {
        let bandwidth = Bandwidth::Wb;
        let n = subframe_samples(bandwidth).unwrap();
        let state = LtpSynthState::new(bandwidth).unwrap();
        let mut e_q23 = vec![0i32; n];
        for (i, slot) in e_q23.iter_mut().enumerate() {
            *slot = (i as i32 - 40) * 100_000;
        }
        let a = make_a_zero(state.d_lpc());
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Unvoiced,
            frame_size: SilkFrameSize::TwentyMs,
            subframe_index: 0,
            gain_q16: 65536,
            pitch_lag: 100,
            b_q7: make_b_zero(),
            ltp_scaling_q14: 15565,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let mut res = vec![0.0f32; n];
        ltp_synthesis_subframe(&state, cfg, &e_q23, &mut res).unwrap();
        for i in 0..n {
            let expect = (e_q23[i] as f32) / 8_388_608.0;
            assert!(
                (res[i] - expect).abs() < 1e-9,
                "i={i}: res={} expect={expect}",
                res[i]
            );
        }
    }

    #[test]
    fn inactive_treated_as_unvoiced_for_residual_path() {
        let bandwidth = Bandwidth::Nb;
        let n = subframe_samples(bandwidth).unwrap();
        let state = LtpSynthState::new(bandwidth).unwrap();
        let mut e_q23 = vec![0i32; n];
        e_q23[0] = 8_388_608; // 1.0
        e_q23[1] = -8_388_608; // -1.0
        let a = make_a_zero(state.d_lpc());
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Inactive,
            frame_size: SilkFrameSize::TenMs,
            subframe_index: 0,
            gain_q16: 65536,
            pitch_lag: 50,
            b_q7: make_b_zero(),
            ltp_scaling_q14: 15565,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let mut res = vec![0.0f32; n];
        ltp_synthesis_subframe(&state, cfg, &e_q23, &mut res).unwrap();
        assert!((res[0] - 1.0).abs() < 1e-7);
        assert!((res[1] - -1.0).abs() < 1e-7);
        for &v in &res[2..] {
            assert_eq!(v, 0.0);
        }
    }

    // ----- input validation -----------------------------------------------

    #[test]
    fn rejects_mismatched_lengths() {
        let bandwidth = Bandwidth::Nb;
        let state = LtpSynthState::new(bandwidth).unwrap();
        let a = make_a_zero(state.d_lpc());
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Unvoiced,
            frame_size: SilkFrameSize::TwentyMs,
            subframe_index: 0,
            gain_q16: 65536,
            pitch_lag: 50,
            b_q7: make_b_zero(),
            ltp_scaling_q14: 15565,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let e = vec![0i32; 39];
        let mut r = vec![0.0f32; 40];
        assert!(ltp_synthesis_subframe(&state, cfg, &e, &mut r).is_err());
        let e = vec![0i32; 40];
        let mut r = vec![0.0f32; 39];
        assert!(ltp_synthesis_subframe(&state, cfg, &e, &mut r).is_err());
    }

    #[test]
    fn rejects_bad_a_q12_len() {
        let bandwidth = Bandwidth::Wb;
        let n = subframe_samples(bandwidth).unwrap();
        let state = LtpSynthState::new(bandwidth).unwrap();
        // WB needs d_LPC = 16, but we provide 10.
        let a = make_a_zero(10);
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Voiced,
            frame_size: SilkFrameSize::TwentyMs,
            subframe_index: 0,
            gain_q16: 65536,
            pitch_lag: 100,
            b_q7: make_b_zero(),
            ltp_scaling_q14: 15565,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let e = vec![0i32; n];
        let mut r = vec![0.0f32; n];
        assert!(ltp_synthesis_subframe(&state, cfg, &e, &mut r).is_err());
    }

    #[test]
    fn rejects_mismatched_state_bandwidth() {
        let n = subframe_samples(Bandwidth::Nb).unwrap();
        let state = LtpSynthState::new(Bandwidth::Wb).unwrap();
        let a = make_a_zero(16);
        let cfg = LtpSynthSubframe {
            bandwidth: Bandwidth::Nb,
            signal_type: SignalType::Voiced,
            frame_size: SilkFrameSize::TwentyMs,
            subframe_index: 0,
            gain_q16: 65536,
            pitch_lag: 50,
            b_q7: make_b_zero(),
            ltp_scaling_q14: 15565,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let e = vec![0i32; n];
        let mut r = vec![0.0f32; n];
        assert!(ltp_synthesis_subframe(&state, cfg, &e, &mut r).is_err());
    }

    #[test]
    fn rejects_swb_fb_bandwidth() {
        // The bandwidth check rejects SWB/FB inside subframe_samples().
        for bw in [Bandwidth::Swb, Bandwidth::Fb] {
            assert!(LtpSynthState::new(bw).is_err());
        }
    }

    #[test]
    fn rejects_bad_subframe_index() {
        let bandwidth = Bandwidth::Nb;
        let n = subframe_samples(bandwidth).unwrap();
        let state = LtpSynthState::new(bandwidth).unwrap();
        let a = make_a_zero(state.d_lpc());
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Voiced,
            frame_size: SilkFrameSize::TenMs,
            subframe_index: 2, // out of range for 10 ms (only 0,1)
            gain_q16: 65536,
            pitch_lag: 50,
            b_q7: make_b_zero(),
            ltp_scaling_q14: 15565,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let e = vec![0i32; n];
        let mut r = vec![0.0f32; n];
        assert!(ltp_synthesis_subframe(&state, cfg, &e, &mut r).is_err());
    }

    #[test]
    fn rejects_nonpositive_pitch_lag_voiced() {
        let bandwidth = Bandwidth::Wb;
        let n = subframe_samples(bandwidth).unwrap();
        let state = LtpSynthState::new(bandwidth).unwrap();
        let a = make_a_zero(state.d_lpc());
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Voiced,
            frame_size: SilkFrameSize::TwentyMs,
            subframe_index: 0,
            gain_q16: 65536,
            pitch_lag: 0,
            b_q7: make_b_zero(),
            ltp_scaling_q14: 15565,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let e = vec![0i32; n];
        let mut r = vec![0.0f32; n];
        assert!(ltp_synthesis_subframe(&state, cfg, &e, &mut r).is_err());
    }

    // ----- voiced: structural / algebraic identities ----------------------

    #[test]
    fn voiced_zero_history_zero_excitation_zero_b_yields_zero() {
        // Empty histories + zero excitation + zero LTP taps + zero a_Q12
        // produces an all-zero residual.
        let bandwidth = Bandwidth::Mb;
        let n = subframe_samples(bandwidth).unwrap();
        let state = LtpSynthState::new(bandwidth).unwrap();
        let a = make_a_zero(state.d_lpc());
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Voiced,
            frame_size: SilkFrameSize::TwentyMs,
            subframe_index: 0,
            gain_q16: 65536,
            pitch_lag: 100,
            b_q7: make_b_zero(),
            ltp_scaling_q14: 15565,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let e = vec![0i32; n];
        let mut r = vec![0.0f32; n];
        ltp_synthesis_subframe(&state, cfg, &e, &mut r).unwrap();
        for (i, v) in r.iter().enumerate() {
            assert_eq!(*v, 0.0, "i={i}: nonzero {v}");
        }
    }

    #[test]
    fn voiced_zero_b_passes_scaled_excitation() {
        // With b_Q7 = [0;5] the LTP convolution drops out: res[i] =
        // e_Q23[i]/2^23. The rewhiten regions are independent of res_out
        // because the LTP filter does not read from them (b == 0).
        let bandwidth = Bandwidth::Wb;
        let n = subframe_samples(bandwidth).unwrap();
        let mut state = LtpSynthState::new(bandwidth).unwrap();
        // Populate non-trivial history.
        let mut out_h = vec![0.0f32; n];
        let mut lpc_h = vec![0.0f32; n];
        for i in 0..n {
            out_h[i] = ((i as f32) - 40.0) * 0.013;
            lpc_h[i] = ((i as f32) - 40.0) * 0.011;
        }
        ltp_synth_commit_subframe(&mut state, &out_h, &lpc_h).unwrap();
        let mut a = vec![0i16; state.d_lpc()];
        a[0] = 1234;
        a[3] = -456;
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Voiced,
            frame_size: SilkFrameSize::TwentyMs,
            subframe_index: 1, // not subframe 0 — there's prior history
            gain_q16: 100_000,
            pitch_lag: 50,
            b_q7: [0; LTP_FILTER_TAPS],
            ltp_scaling_q14: 15565,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let mut e = vec![0i32; n];
        for (i, slot) in e.iter_mut().enumerate() {
            *slot = ((i as i32) - 40) * 50_000;
        }
        let mut r = vec![0.0f32; n];
        ltp_synthesis_subframe(&state, cfg, &e, &mut r).unwrap();
        for (i, (&ri, &ei)) in r.iter().zip(e.iter()).enumerate() {
            let expect = (ei as f32) / 8_388_608.0;
            assert!(
                (ri - expect).abs() < 1e-7,
                "i={i}: r={} expect={expect}",
                ri
            );
        }
    }

    #[test]
    fn voiced_b0_only_uses_pitch_lag_lookback() {
        // With b_Q7 = [64, 0, 0, 0, 0] (= 0.5 on first tap, others zero),
        // the LTP convolution becomes:
        //   res[i] = e/2^23 + 0.5 * res[i - pitch_lag + 2]
        // i.e. it pulls 0.5x a value from the rewhitened buffer at offset
        // (-pitch_lag + 2) relative to the current sample.
        //
        // If a_Q12 = 0 and gain = 65536 then the lpc/out rewhitening
        // reduces to:
        //   region A:  res[i] = 4*LTP_scale/65536 * clamp(out[i], -1, 1)
        //                     = (4*15565/65536) * out[i] (small)
        //   region B:  res[i] = 65536/65536 * lpc[i] = lpc[i]
        //
        // We test the first sample of the current subframe (local_i = 0).
        // For this sample, the pitch-lag reference is i = j - pitch_lag + 2,
        // which is in region A (since j - pitch_lag + 2 < out_end when
        // out_end >= j) OR region B depending on out_end and pitch_lag.
        //
        // Easier test: set pitch_lag = 16 (small) and subframe_index = 0.
        // Then out_end = j (region B is empty for subframe 0). The pitch
        // reference for local_i = 0 is buf_i - pitch_lag + 2 = (pitch_lag+2)
        // - pitch_lag + 2 = 4 (offset 4 in the rewhitened buffer).
        // That index 4 corresponds to out[j - pitch_lag - 2 + 4]
        //   = out[j - pitch_lag + 2] = out[j - 14] (with pitch_lag = 16)
        //
        // Set out_history so that out[j - 14] is exactly some known
        // value V and a_Q12 = 0 (so the rewhiten = LTP_scale_factor * V).
        let bandwidth = Bandwidth::Nb;
        let n = subframe_samples(bandwidth).unwrap();
        let mut state = LtpSynthState::new(bandwidth).unwrap();
        // Populate the out_history: the most recent sample (idx = max)
        // corresponds to out[j-1]; out[j-14] is at idx = max - 13.
        let v = 0.42f32;
        // Inject by committing fake subframes — easier: directly fill the
        // exposed history via push.
        let mut out_h = vec![0.0f32; n];
        out_h[n - 14] = v;
        let lpc_h = vec![0.0f32; n];
        ltp_synth_commit_subframe(&mut state, &out_h, &lpc_h).unwrap();
        let a = make_a_zero(state.d_lpc());
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Voiced,
            frame_size: SilkFrameSize::TwentyMs,
            subframe_index: 0,
            gain_q16: 65536,
            pitch_lag: 16,
            // b_Q7[0] = 64 => b[0] = 0.5
            b_q7: [64, 0, 0, 0, 0],
            ltp_scaling_q14: 15565,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let e = vec![0i32; n];
        let mut r = vec![0.0f32; n];
        ltp_synthesis_subframe(&state, cfg, &e, &mut r).unwrap();
        // Expected res[0] = 0 + 0.5 * (4*15565/65536) * v.
        // Region A rewhitening factor = 4 * 15565 / 65536 = 0.9500...
        let scale = 4.0 * 15565.0 / 65536.0;
        let expect = 0.5 * scale * v;
        assert!(
            (r[0] - expect).abs() < 1e-6,
            "r[0]={} expect={expect}",
            r[0]
        );
    }

    #[test]
    fn voiced_region_b_lpc_path_no_out_history() {
        // For subframe_index >= 1 (and no LSF-interp split), out_end =
        // j - s*n: region A may be small or absent depending on
        // pitch_lag; region B covers s*n samples back. Set pitch_lag
        // small (= 2) so that region A is exactly 4 samples and region B
        // is s*n samples.
        //
        // With b_Q7 = [128, 0, 0, 0, 0] and a_Q12 = 0, gain_q16 = 65536,
        // and an injected lpc_history pattern that has a known value at
        // the offset we'll read, we can verify region B rewhitening.
        //
        // For local_i = 0, the pitch reference is buf_i - pitch_lag + 2
        // = (pitch_lag+2) - pitch_lag + 2 = 4 = lookback. But lookback
        // is the *current subframe's* first sample's local index — let's
        // double-check.
        //
        // lookback = pitch_lag + 2 = 4. buf_i = lookback + local_i = 4
        // for local_i = 0. So src = 4 - 2 + 2 - 0 = 4 = lookback, which
        // is the current subframe's local_i = 0 slot. That's res[j],
        // which is the sample we're computing. The formula intends
        // res[i - pitch_lag + 2 - k] which for i = j and k = 0 is
        // res[j - pitch_lag + 2] = res[j - 0]; with pitch_lag = 2,
        // that's res[j], reading the sample we're writing. The spec
        // permits this (b[0] can be 1.0 only if the filter is stable,
        // which it would not be in practice). To avoid this self-
        // reference, use pitch_lag = 16 (typical) and b_Q7[2] = 128
        // (k=2 ⇒ res[j - 16 + 2 - 2] = res[j - 16]).
        let bandwidth = Bandwidth::Wb;
        let n = subframe_samples(bandwidth).unwrap();
        let mut state = LtpSynthState::new(bandwidth).unwrap();
        // Set a known lpc_history value at position corresponding to
        // out[j - 16]. The history layout is MSB-recent: the last
        // sample is at idx = max - 1 (offset = -1 from j). So
        // out[j - 16] is at idx = max - 16.
        // Inject by committing a subframe with that pattern.
        let mut lpc_h = vec![0.0f32; n];
        let v = 0.33f32;
        lpc_h[n - 16] = v;
        let out_h = vec![0.0f32; n];
        ltp_synth_commit_subframe(&mut state, &out_h, &lpc_h).unwrap();
        let a = make_a_zero(state.d_lpc());
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Voiced,
            frame_size: SilkFrameSize::TwentyMs,
            subframe_index: 1, // out_end = j - n, so region B covers n samples
            gain_q16: 65536,
            pitch_lag: 16,
            // b_Q7[2] = 64 -> b[2] = 0.5
            b_q7: [0, 0, 64, 0, 0],
            ltp_scaling_q14: 15565,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let e = vec![0i32; n];
        let mut r = vec![0.0f32; n];
        ltp_synthesis_subframe(&state, cfg, &e, &mut r).unwrap();
        // For local_i = 0: src index = buf_i - pitch_lag + 2 - k
        //   = lookback - 16 + 2 - 2 = (16+2) - 16 = 2.
        // res_buf[2] corresponds to offset = 2 - lookback = -16. That's
        // out_end_off = -n = -80, region B covers -80..0. Index 2 is at
        // offset -16, well inside region B (read from lpc_history).
        // With a_Q12 = 0 and gain = 65536, region B rewhitening:
        //   res[i] = 1.0 * (lpc[i] - 0) = lpc[i]
        // So res_buf[2] = v.
        // Then res[0] = 0 + b[2] * res_buf[2] = 0.5 * v.
        let expect = 0.5 * v;
        assert!(
            (r[0] - expect).abs() < 1e-6,
            "r[0]={} expect={expect}",
            r[0]
        );
    }

    #[test]
    fn voiced_lsf_interp_split_uses_fresh_scale_and_shorter_lookback() {
        // For subframe_index = 2 with lsf_interp_used = true and 20 ms
        // frame, the §4.2.7.9.1 split branch fires:
        //   out_end = j - (s - 2) * n = j  (since s = 2)
        //   LTP_scale_Q14 = 16384
        // So region B is empty (out_end == j) and the entire lookback is
        // region A.
        //
        // Test: with a_Q12 = 0 and a single non-zero out_history sample
        // at the right offset, and b_Q7 = [64,0,0,0,0] (b[0] = 0.5), the
        // rewhiten factor is 4*16384/65536 = exactly 1.0; the final res
        // value at sample 0 is 0.5 * 1.0 * v = 0.5 * v.
        let bandwidth = Bandwidth::Nb;
        let n = subframe_samples(bandwidth).unwrap();
        let mut state = LtpSynthState::new(bandwidth).unwrap();
        // Push three "previous" subframes worth of out_history so that
        // out[j - 14] is well-defined and not at the boundary.
        for _ in 0..3 {
            let mut out_h = vec![0.0f32; n];
            // Mark the most-recent position with a unique signature.
            out_h[n - 14] = 0.0;
            ltp_synth_commit_subframe(&mut state, &out_h, &vec![0.0f32; n]).unwrap();
        }
        // Place v at the position that will be read by local_i = 0.
        // Most-recent push: the most recent out_history slot is offset
        // -1 from j. out[j - 14] = idx = max - 13.
        let v = 0.5f32;
        // Replace the most-recent subframe with one carrying v.
        let mut out_h = vec![0.0f32; n];
        out_h[n - 14] = v;
        // We need to push this as a *fresh* subframe so out_h covers the
        // last n samples; but this will be the *4th* push, and
        // subframe_index will be 4. We want to test subframe_index = 2,
        // so use start_frame() to reset that counter without touching
        // the history.
        ltp_synth_commit_subframe(&mut state, &out_h, &vec![0.0f32; n]).unwrap();
        state.start_frame();
        // Now manually advance subframe_index so the state thinks we're
        // about to produce subframe 2 in a 20 ms frame. We have no
        // public setter for subframe_index — that's deliberate. Instead,
        // we test the cfg's subframe_index field directly (the function
        // uses cfg.subframe_index, not state.subframe_index, for the
        // split-branch decision).
        let a = make_a_zero(state.d_lpc());
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Voiced,
            frame_size: SilkFrameSize::TwentyMs,
            subframe_index: 2,
            gain_q16: 65536,
            pitch_lag: 16,
            b_q7: [64, 0, 0, 0, 0],
            ltp_scaling_q14: 15565, // ignored: split branch overrides to 16384
            a_q12: &a,
            lsf_interp_used: true,
        };
        let e = vec![0i32; n];
        let mut r = vec![0.0f32; n];
        ltp_synthesis_subframe(&state, cfg, &e, &mut r).unwrap();
        // 4 * 16384 / 65536 = 1.0 exactly; then 0.5 * v from b[0].
        let expect = 0.5 * v;
        assert!(
            (r[0] - expect).abs() < 1e-6,
            "r[0]={} expect={expect}",
            r[0]
        );
    }

    // ----- determinism / sweep --------------------------------------------

    #[test]
    fn voiced_decode_is_deterministic_for_same_inputs() {
        let bandwidth = Bandwidth::Wb;
        let n = subframe_samples(bandwidth).unwrap();
        let state0 = {
            let mut s = LtpSynthState::new(bandwidth).unwrap();
            let mut out_h = vec![0.0f32; n];
            let mut lpc_h = vec![0.0f32; n];
            for i in 0..n {
                out_h[i] = ((i as f32) * 0.017).sin() * 0.4;
                lpc_h[i] = ((i as f32) * 0.013).cos() * 0.3;
            }
            ltp_synth_commit_subframe(&mut s, &out_h, &lpc_h).unwrap();
            s
        };
        let mut a = vec![0i16; 16];
        for (k, slot) in a.iter_mut().enumerate() {
            *slot = (200 - (k as i16) * 25).clamp(-2047, 2047);
        }
        let mut e = vec![0i32; n];
        for (i, slot) in e.iter_mut().enumerate() {
            *slot = ((i as i32) - 40) * 12_345;
        }
        let cfg = LtpSynthSubframe {
            bandwidth,
            signal_type: SignalType::Voiced,
            frame_size: SilkFrameSize::TwentyMs,
            subframe_index: 1,
            gain_q16: 200_000,
            pitch_lag: 90,
            b_q7: [13, 22, 39, 23, 12],
            ltp_scaling_q14: 12288,
            a_q12: &a,
            lsf_interp_used: false,
        };
        let mut r1 = vec![0.0f32; n];
        let mut r2 = vec![0.0f32; n];
        ltp_synthesis_subframe(&state0, cfg, &e, &mut r1).unwrap();
        ltp_synthesis_subframe(&state0, cfg, &e, &mut r2).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn voiced_sweep_no_panic_finite() {
        let buffers: [&[u8]; 3] = [
            &[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77],
            &[0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88],
            &[0x5A, 0xA5, 0x3C, 0xC3, 0x0F, 0xF0, 0x69, 0x96],
        ];
        for buf_seed in buffers {
            for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
                let n = subframe_samples(bw).unwrap();
                let d_lpc = if matches!(bw, Bandwidth::Wb) { 16 } else { 10 };
                for fs in [SilkFrameSize::TenMs, SilkFrameSize::TwentyMs] {
                    let num_subframes = match fs {
                        SilkFrameSize::TenMs => 2u8,
                        SilkFrameSize::TwentyMs => 4u8,
                    };
                    let mut state = LtpSynthState::new(bw).unwrap();
                    // Seed history with deterministic samples.
                    let mut out_h = vec![0.0f32; n];
                    let mut lpc_h = vec![0.0f32; n];
                    for i in 0..n {
                        let s = buf_seed[i % buf_seed.len()] as f32;
                        out_h[i] = (s - 128.0) / 256.0;
                        lpc_h[i] = (s - 128.0) / 200.0;
                    }
                    ltp_synth_commit_subframe(&mut state, &out_h, &lpc_h).unwrap();
                    let mut a = vec![0i16; d_lpc];
                    for (k, slot) in a.iter_mut().enumerate() {
                        *slot = (k as i16) * 50 - 200;
                    }
                    let mut e = vec![0i32; n];
                    for (i, slot) in e.iter_mut().enumerate() {
                        *slot = ((buf_seed[i % buf_seed.len()] as i32) - 128) * 60_000;
                    }
                    for s in 0..num_subframes {
                        let cfg = LtpSynthSubframe {
                            bandwidth: bw,
                            signal_type: SignalType::Voiced,
                            frame_size: fs,
                            subframe_index: s,
                            gain_q16: 100_000 + (s as u32) * 20_000,
                            pitch_lag: 32 + (s as i32) * 8,
                            b_q7: [4, 6, 24, 7, 5],
                            ltp_scaling_q14: 15565,
                            a_q12: &a,
                            lsf_interp_used: s >= 2 && matches!(fs, SilkFrameSize::TwentyMs),
                        };
                        let mut r = vec![0.0f32; n];
                        ltp_synthesis_subframe(&state, cfg, &e, &mut r).unwrap();
                        for (i, v) in r.iter().enumerate() {
                            assert!(v.is_finite(), "non-finite res at i={i}: {v}");
                        }
                        // Pretend the LPC synthesis produced some output.
                        let out_clamped: Vec<f32> = r.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
                        let lpc_unclamped: Vec<f32> = r.clone();
                        ltp_synth_commit_subframe(&mut state, &out_clamped, &lpc_unclamped)
                            .unwrap();
                    }
                }
            }
        }
    }

    // ----- commit-subframe length validation ------------------------------

    #[test]
    fn commit_rejects_bad_lengths() {
        let mut s = LtpSynthState::new(Bandwidth::Nb).unwrap();
        let out = vec![0.0f32; 39]; // wrong: NB expects 40
        let lpc = vec![0.0f32; 40];
        assert!(ltp_synth_commit_subframe(&mut s, &out, &lpc).is_err());
        let out = vec![0.0f32; 40];
        let lpc = vec![0.0f32; 39];
        assert!(ltp_synth_commit_subframe(&mut s, &out, &lpc).is_err());
    }
}
