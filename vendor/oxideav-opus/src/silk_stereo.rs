//! SILK §4.2.8 stereo unmixing (mid/side → left/right) — RFC 6716.
//!
//! For stereo SILK streams, the two channels are decoded as a
//! *mid* (M) channel and a *side* (S) channel. After both channels
//! finish their §4.2.7.9 reconstruction (LTP + LPC synthesis,
//! producing the per-channel `out[]` signal), the decoder converts the
//! mid/side representation back into the left/right (LR) representation
//! the application expects. RFC 6716 calls this the
//! `silk_stereo_MS_to_LR` step.
//!
//! The side channel is predicted from two things:
//!
//!   * a simple low-passed version of the mid channel
//!     (`p0 = (mid[i-2] + 2*mid[i-1] + mid[i]) / 4`), and
//!   * the unfiltered mid channel,
//!
//! using the two Q13 prediction weights `(w0_Q13, w1_Q13)` decoded for
//! the *mid* channel in §4.2.7.1. The low-pass filter imposes a
//! one-sample delay, and the unfiltered mid term is also delayed by one
//! sample (it reads `mid[i-1]`, not `mid[i]`), so the reconstruction
//! reaches back two samples into the mid channel and one sample into the
//! side channel relative to the first frame sample.
//!
//! The unmixing runs in two phases (§4.2.8):
//!
//!   1. **Interpolation phase** — for the first `n1` samples
//!      (`64` NB, `96` MB, `128` WB ≈ 8 ms) the weights ramp linearly
//!      from the *previous* frame's weights `(prev_w0_Q13, prev_w1_Q13)`
//!      to the current frame's `(w0_Q13, w1_Q13)`:
//!
//!      ```text
//!            prev_w0_Q13                    (w0_Q13 - prev_w0_Q13)
//!      w0 =  ----------- + min(i - j, n1) * ----------------------
//!              8192.0                             8192.0*n1
//!      ```
//!
//!      (and likewise for `w1`).
//!   2. **Steady phase** — for the remaining `n2 - n1` samples,
//!      `min(i - j, n1) == n1`, so the weights are simply the current
//!      frame's values.
//!
//! The per-sample reconstruction is then:
//!
//! ```text
//!              p0 = (mid[i-2] + 2*mid[i-1] + mid[i]) / 4.0
//!         left[i] = clamp(-1.0, (1 + w1)*mid[i-1] + side[i-1] + w0*p0, 1.0)
//!        right[i] = clamp(-1.0, (1 - w1)*mid[i-1] - side[i-1] - w0*p0, 1.0)
//! ```
//!
//! When the side channel is not coded for this frame (§4.2.7.2 mid-only
//! flag), `side[i]` is taken to be zero everywhere — including the
//! `side[i-1]` history term.
//!
//! The two prior mid samples and one prior side sample carried across
//! the frame boundary live in [`StereoUnmixState`]; on a decoder reset
//! (or for the first frame) they are zero, per the §4.2.8 closing
//! paragraph.
//!
//! Per the §4.2.7.9 preamble, this stage "does not need to be
//! bit-exact"; we follow the spec's floating-point formulation in `f32`.
//!
//! All truth is taken from RFC 6716 §4.2.8 (and §4.2.7.1 for the weight
//! decode).

use crate::toc::Bandwidth;
use crate::Error;

/// Number of samples in the §4.2.8 interpolation phase (`n1`): 64 for
/// NB, 96 for MB, 128 for WB. Roughly 8 ms at each SILK internal rate.
///
/// SWB / FB are SILK-illegal at this stage (the §4.2.2 hybrid split
/// hands SILK only NB/MB/WB), so they are rejected.
pub fn interp_phase_samples(bandwidth: Bandwidth) -> Result<usize, Error> {
    Ok(match bandwidth {
        Bandwidth::Nb => 64,
        Bandwidth::Mb => 96,
        Bandwidth::Wb => 128,
        _ => return Err(Error::MalformedPacket),
    })
}

/// One channel-pair's stereo prediction weights, in Q13 fixed-point, as
/// produced by §4.2.7.1 ([`crate::StereoPredictionWeights`]).
///
/// `silk_stereo_MS_to_LR` consumes the *current* frame's weights and the
/// *previous* frame's weights together (the first `n1` samples
/// interpolate between them). The previous-frame weights are zero on a
/// decoder reset / first frame, mirroring the cleared sample history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StereoWeightsQ13 {
    /// Low-pass prediction weight `w0_Q13`.
    pub w0_q13: i32,
    /// Direct (unfiltered) prediction weight `w1_Q13`.
    pub w1_q13: i32,
}

/// Cross-frame history needed by §4.2.8: the two trailing mid samples
/// and the one trailing side sample from the previous frame, plus the
/// previous frame's prediction weights for the phase-1 interpolation.
///
/// All fields are zero after a decoder reset or for the first frame
/// after one (RFC 6716 §4.2.8: "For the first frame after a decoder
/// reset, zeros are used instead.").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoUnmixState {
    /// `mid[j-2]` and `mid[j-1]` for the upcoming frame (oldest first).
    mid_hist: [f32; 2],
    /// `side[j-1]` for the upcoming frame.
    side_hist: f32,
    /// The previous frame's `(w0_Q13, w1_Q13)` — `prev_w0_Q13` /
    /// `prev_w1_Q13` in the §4.2.8 formulas.
    prev_weights: StereoWeightsQ13,
}

impl Default for StereoUnmixState {
    fn default() -> Self {
        Self::new()
    }
}

impl StereoUnmixState {
    /// A freshly-reset state: cleared mid/side history and zero previous
    /// weights, as required for the first frame after a decoder reset.
    pub fn new() -> Self {
        StereoUnmixState {
            mid_hist: [0.0; 2],
            side_hist: 0.0,
            prev_weights: StereoWeightsQ13::default(),
        }
    }

    /// Reset the state to its post-decoder-reset values (RFC 6716
    /// §4.2.8 / §4.5.2): all sample history and previous weights zeroed.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// The previous-frame weights currently held (exposed for tests /
    /// introspection).
    pub fn prev_weights(&self) -> StereoWeightsQ13 {
        self.prev_weights
    }
}

/// One frame of stereo output, two equal-length channel signals nominally
/// in `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoFrame {
    /// Left channel, length `n2` (the frame sample count).
    pub left: Vec<f32>,
    /// Right channel, same length as [`StereoFrame::left`].
    pub right: Vec<f32>,
}

/// Apply RFC 6716 §4.2.8 stereo unmixing to one frame, converting the
/// decoded mid/side channels into left/right.
///
/// * `bandwidth` selects the §4.2.8 interpolation length `n1`.
/// * `mid` is the mid channel's §4.2.7.9.2 `out[]` for this frame
///   (length `n2`, the total frame sample count).
/// * `side` is the side channel's `out[]`, or `None` when the side
///   channel is not coded for this frame (§4.2.7.2 mid-only flag), in
///   which case `side[i]` is treated as zero everywhere.
/// * `weights` are this frame's §4.2.7.1 weights (`w0_Q13`, `w1_Q13`).
/// * `state` carries the two prior mid samples, one prior side sample,
///   and the previous frame's weights across the frame boundary; it is
///   updated in place ready for the next frame.
///
/// Returns the reconstructed [`StereoFrame`]. Errors if `side` is
/// present but its length differs from `mid`, if `mid` is empty, or if
/// `bandwidth` is SILK-illegal.
pub fn stereo_ms_to_lr(
    bandwidth: Bandwidth,
    mid: &[f32],
    side: Option<&[f32]>,
    weights: StereoWeightsQ13,
    state: &mut StereoUnmixState,
) -> Result<StereoFrame, Error> {
    let n2 = mid.len();
    if n2 == 0 {
        return Err(Error::MalformedPacket);
    }
    if let Some(s) = side {
        if s.len() != n2 {
            return Err(Error::MalformedPacket);
        }
    }
    // n1 is the interpolation-phase length; it must not exceed n2 (the
    // §4.2.8 `min(i - j, n1)` term clamps the ramp regardless, but a
    // very short frame would otherwise never reach the steady phase —
    // which is still correct, just fully-interpolated).
    let n1 = interp_phase_samples(bandwidth)?;

    let prev = state.prev_weights;
    let w0_q13 = weights.w0_q13 as f32;
    let w1_q13 = weights.w1_q13 as f32;
    let prev_w0_q13 = prev.w0_q13 as f32;
    let prev_w1_q13 = prev.w1_q13 as f32;

    // Precompute the per-sample interpolation increment denominators.
    // w0 = prev_w0_Q13/8192 + min(i-j,n1) * (w0_Q13 - prev_w0_Q13)/(8192*n1)
    let n1_f = n1 as f32;
    let w0_base = prev_w0_q13 / 8192.0;
    let w1_base = prev_w1_q13 / 8192.0;
    let w0_step = (w0_q13 - prev_w0_q13) / (8192.0 * n1_f);
    let w1_step = (w1_q13 - prev_w1_q13) / (8192.0 * n1_f);

    let mut left = vec![0.0f32; n2];
    let mut right = vec![0.0f32; n2];

    // History accessors. mid[j-2], mid[j-1] come from the carried state;
    // side[j-1] likewise. With j == 0 (each frame's local index space),
    // "i-2" / "i-1" for i in 0..2 reach into the history.
    let mid_m2 = state.mid_hist[0]; // mid[j-2]
    let mid_m1 = state.mid_hist[1]; // mid[j-1]
    let side_m1 = if side.is_some() {
        state.side_hist
    } else {
        // Side not coded → side[i] = 0 everywhere, including history.
        0.0
    };

    for i in 0..n2 {
        // Interpolated weights for this sample. min(i - j, n1) with j=0.
        let ramp = (i.min(n1)) as f32;
        let w0 = w0_base + ramp * w0_step;
        let w1 = w1_base + ramp * w1_step;

        // mid[i], mid[i-1], mid[i-2] with i-1 / i-2 dipping into history.
        let m_i = mid[i];
        let m_i1 = if i >= 1 { mid[i - 1] } else { mid_m1 };
        let m_i2 = match i {
            0 => mid_m2,
            1 => mid_m1,
            _ => mid[i - 2],
        };

        // side[i-1] with i-1 dipping into history (or 0 if uncoded).
        let s_i1 = match side {
            Some(s) if i >= 1 => s[i - 1],
            Some(_) => side_m1,
            None => 0.0,
        };

        // p0 = (mid[i-2] + 2*mid[i-1] + mid[i]) / 4.0
        let p0 = (m_i2 + 2.0 * m_i1 + m_i) / 4.0;

        // left[i]  = clamp(-1, (1 + w1)*mid[i-1] + side[i-1] + w0*p0, 1)
        // right[i] = clamp(-1, (1 - w1)*mid[i-1] - side[i-1] - w0*p0, 1)
        let l = (1.0 + w1) * m_i1 + s_i1 + w0 * p0;
        let r = (1.0 - w1) * m_i1 - s_i1 - w0 * p0;
        left[i] = l.clamp(-1.0, 1.0);
        right[i] = r.clamp(-1.0, 1.0);
    }

    // Carry the trailing samples + current weights into the next frame.
    // mid history: the last two samples of this frame.
    state.mid_hist = if n2 >= 2 {
        [mid[n2 - 2], mid[n2 - 1]]
    } else {
        // n2 == 1: shift the old most-recent into the older slot.
        [mid_m1, mid[n2 - 1]]
    };
    // side history: last sample of this frame, or 0 if uncoded.
    state.side_hist = match side {
        Some(s) => s[n2 - 1],
        None => 0.0,
    };
    state.prev_weights = weights;

    Ok(StereoFrame { left, right })
}

/// Cross-frame history for the encode-side [`stereo_lr_to_ms`] downmix:
/// the one trailing mid sample (feeding the next frame's first `p0`)
/// and the previous frame's weights (anchoring the §4.2.8 ramp).
///
/// Zero after an encoder reset, mirroring the decoder's
/// [`StereoUnmixState`] reset semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoDownmixState {
    /// `mid[j-1]` for the upcoming frame.
    prev_mid: f32,
    /// The previous frame's `(w0_Q13, w1_Q13)`.
    prev_weights: StereoWeightsQ13,
}

impl Default for StereoDownmixState {
    fn default() -> Self {
        Self::new()
    }
}

impl StereoDownmixState {
    /// A freshly-reset state: zero mid history and zero previous
    /// weights.
    pub fn new() -> Self {
        StereoDownmixState {
            prev_mid: 0.0,
            prev_weights: StereoWeightsQ13::default(),
        }
    }

    /// Reset to the post-reset values (all zero).
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// One frame of encode-side mid/side output produced by
/// [`stereo_lr_to_ms`].
#[derive(Debug, Clone, PartialEq)]
pub struct MidSideFrame {
    /// Mid channel, length `n2`.
    pub mid: Vec<f32>,
    /// Side channel, same length.
    pub side: Vec<f32>,
}

/// Convert one frame of left/right input into the mid/side pair the
/// §4.2.8 unmixer will reconstruct it from — the exact algebraic
/// inverse of [`stereo_ms_to_lr`], derived by solving the §4.2.8
/// reconstruction for `mid` and `side`.
///
/// Adding and subtracting the two §4.2.8 output equations gives
///
/// ```text
///   left[i] + right[i] = 2 * mid[i-1]
///   left[i] - right[i] = 2 * (w1*mid[i-1] + side[i-1] + w0*p0)
/// ```
///
/// so with the decoder's inherent one-sample delay embraced (the
/// reconstruction reads `mid[i-1]` / `side[i-1]`, never `mid[i]`), the
/// frame-aligned inverse is
///
/// ```text
///        mid[k] = (left[k] + right[k]) / 2
///       side[k] = (left[k] - right[k])/2 - w1(k+1)*mid[k] - w0(k+1)*p0(k+1)
///   p0(k+1)     = (mid[k-1] + 2*mid[k] + mid[k+1]) / 4
/// ```
///
/// where `w0(i)` / `w1(i)` follow the same §4.2.8 interpolation ramp
/// the decoder applies (previous-frame weights → `weights` over the
/// first `n1` samples). Feeding the result to [`stereo_ms_to_lr`] with
/// the same weight sequence reproduces the input delayed by exactly
/// one sample (the §4.2.8 delay), apart from the `[-1, 1]` output
/// clamp, which is unreachable for in-range audio.
///
/// The final side sample's `p0` needs `mid[n2]` — the *next* frame's
/// first mid sample, `(left[n2] + right[n2]) / 2`. Pass it via
/// `next_lr` (the next frame's first left/right pair); at the end of
/// the stream pass `None` and the last mid sample is held instead
/// (the one-sided difference only perturbs the final side sample, and
/// only when a next frame exists after all).
///
/// A SILK frame is always longer than the §4.2.8 interpolation phase
/// (`n2 > n1` for every legal bandwidth / duration), which this
/// inverse relies on: the last side sample is consumed by the *next*
/// frame's first reconstruction, whose ramp starts at this frame's
/// final weights. Errors on an empty frame or a SILK-illegal
/// bandwidth, mirroring [`stereo_ms_to_lr`].
pub fn stereo_lr_to_ms(
    bandwidth: Bandwidth,
    left: &[f32],
    right: &[f32],
    weights: StereoWeightsQ13,
    next_lr: Option<(f32, f32)>,
    state: &mut StereoDownmixState,
) -> Result<MidSideFrame, Error> {
    let n2 = left.len();
    if n2 == 0 || right.len() != n2 {
        return Err(Error::MalformedPacket);
    }
    let n1 = interp_phase_samples(bandwidth)?;

    let prev = state.prev_weights;
    let n1_f = n1 as f32;
    let w0_base = prev.w0_q13 as f32 / 8192.0;
    let w1_base = prev.w1_q13 as f32 / 8192.0;
    let w0_step = (weights.w0_q13 - prev.w0_q13) as f32 / (8192.0 * n1_f);
    let w1_step = (weights.w1_q13 - prev.w1_q13) as f32 / (8192.0 * n1_f);

    // mid[k] = (left[k] + right[k]) / 2, frame-aligned.
    let mid: Vec<f32> = left
        .iter()
        .zip(right)
        .map(|(&l, &r)| (l + r) / 2.0)
        .collect();
    // mid[n2]: the next frame's first mid sample, or a hold at the
    // stream end.
    let mid_next = match next_lr {
        Some((l, r)) => (l + r) / 2.0,
        None => mid[n2 - 1],
    };

    let mut side = vec![0.0f32; n2];
    for k in 0..n2 {
        // side[k] is consumed by the reconstruction of output sample
        // k+1 (this frame's ramp for k+1 < n2; the next frame's ramp
        // *start* — which equals this frame's final weights — for the
        // last sample, both covered by min(k+1, n1)).
        let ramp = ((k + 1).min(n1)) as f32;
        let w0 = w0_base + ramp * w0_step;
        let w1 = w1_base + ramp * w1_step;

        let m_km1 = if k >= 1 { mid[k - 1] } else { state.prev_mid };
        let m_kp1 = if k + 1 < n2 { mid[k + 1] } else { mid_next };
        let p0 = (m_km1 + 2.0 * mid[k] + m_kp1) / 4.0;
        side[k] = (left[k] - right[k]) / 2.0 - w1 * mid[k] - w0 * p0;
    }

    state.prev_mid = mid[n2 - 1];
    state.prev_weights = weights;

    Ok(MidSideFrame { mid, side })
}

/// Estimate the §4.2.7.1 prediction-weight pair that minimizes the
/// coded side-channel energy for one frame — the encoder's *analysis*
/// choice feeding [`crate::silk_frame::StereoWeightSymbols::quantize`]
/// and [`stereo_lr_to_ms`].
///
/// The §4.2.8 reconstruction predicts the raw side signal
/// `s[k] = (left[k] - right[k]) / 2` from the low-passed mid `p0` and
/// the mid itself, so the residual the bitstream must carry is
/// `s[k] - w0*p0(k+1) - w1*mid[k]` (the same terms
/// [`stereo_lr_to_ms`] subtracts). RFC 6716 leaves the encoder's
/// weight choice free (only the decode of the coded quintuple is
/// normative); this estimator picks the classical least-squares
/// optimum, solving the 2×2 normal equations
///
/// ```text
///   | Σ p0²    Σ p0·m |   | w0 |   | Σ p0·s |
///   |                 | · |    | = |        |
///   | Σ p0·m   Σ m²   |   | w1 |   | Σ m·s  |
/// ```
///
/// over the frame in f64, then converts to Q13. A singular /
/// near-singular system (e.g. a silent mid channel) returns zero
/// weights. The result is a *target* pair — feed it through
/// [`crate::silk_frame::StereoWeightSymbols::quantize`] to obtain the
/// coded quintuple, and use the quantized pair (what the decoder will
/// reconstruct) in [`stereo_lr_to_ms`].
///
/// `prev_mid` is the previous frame's trailing mid sample (0.0 after a
/// reset) and `mid_next` the next frame's first mid sample (or a hold
/// at stream end) — the same boundary terms the downmix uses for
/// `p0`. Errors on empty or length-mismatched inputs.
pub fn estimate_stereo_weights(
    mid: &[f32],
    side_raw: &[f32],
    prev_mid: f32,
    mid_next: f32,
) -> Result<StereoWeightsQ13, Error> {
    let n2 = mid.len();
    if n2 == 0 || side_raw.len() != n2 {
        return Err(Error::MalformedPacket);
    }
    let mut sum_pp = 0f64;
    let mut sum_pm = 0f64;
    let mut sum_mm = 0f64;
    let mut sum_ps = 0f64;
    let mut sum_ms = 0f64;
    for k in 0..n2 {
        let m_km1 = if k >= 1 { mid[k - 1] } else { prev_mid } as f64;
        let m_kp1 = if k + 1 < n2 { mid[k + 1] } else { mid_next } as f64;
        let m = mid[k] as f64;
        let p0 = (m_km1 + 2.0 * m + m_kp1) / 4.0;
        let s = side_raw[k] as f64;
        sum_pp += p0 * p0;
        sum_pm += p0 * m;
        sum_mm += m * m;
        sum_ps += p0 * s;
        sum_ms += m * s;
    }
    let det = sum_pp * sum_mm - sum_pm * sum_pm;
    // Regularization floor: treat a (near-)singular system as "no
    // usable predictor" and code zero weights, which the §4.2.8
    // reconstruction handles exactly (pure sum/difference stereo).
    if det.abs() < 1e-12 {
        return Ok(StereoWeightsQ13::default());
    }
    let w0 = (sum_ps * sum_mm - sum_ms * sum_pm) / det;
    let w1 = (sum_ms * sum_pp - sum_ps * sum_pm) / det;
    let to_q13 = |w: f64| -> i32 {
        (w * 8192.0)
            .round()
            .clamp(-(1 << 30) as f64, (1 << 30) as f64) as i32
    };
    Ok(StereoWeightsQ13 {
        w0_q13: to_q13(w0),
        w1_q13: to_q13(w1),
    })
}

// ---------------------------------------------------------------------
// §4.2.8 unmix in the exact fixed point of the §A reference listing.
// ---------------------------------------------------------------------

/// Cross-frame history for the fixed-point §4.2.8 unmix: the two
/// trailing mid samples, two trailing side samples, and the previous
/// frame's Q13 prediction weights — the integer counterpart of
/// [`StereoUnmixState`], matching the RFC 6716 §A reference listing's
/// stereo decode state. All zero after a decoder reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StereoUnmixStateI16 {
    /// `mid[j-2]`, `mid[j-1]` for the upcoming frame (oldest first).
    s_mid: [i16; 2],
    /// `side[j-2]`, `side[j-1]` for the upcoming frame.
    s_side: [i16; 2],
    /// The previous frame's `(w0_Q13, w1_Q13)`.
    pred_prev_q13: [i32; 2],
}

impl StereoUnmixStateI16 {
    /// A freshly-reset state (all-zero history and weights).
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset to the post-decoder-reset values (§4.2.8 / §4.5.2).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// The previous-frame weights currently held.
    pub fn prev_weights(&self) -> StereoWeightsQ13 {
        StereoWeightsQ13 {
            w0_q13: self.pred_prev_q13[0],
            w1_q13: self.pred_prev_q13[1],
        }
    }
}

/// One frame of fixed-point stereo output (left, right), each `n`
/// signed 16-bit samples at the internal SILK rate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StereoFrameI16 {
    /// Left channel.
    pub left: Vec<i16, crate::bump::DecodeBump>,
    /// Right channel.
    pub right: Vec<i16, crate::bump::DecodeBump>,
}

/// RFC 6716 §4.2.8 stereo unmixing in the exact fixed-point arithmetic
/// of the §A reference listing: convert one frame of decoded mid/side
/// i16 signals into left/right.
///
/// The 8 ms weight-interpolation ramp, the 3-tap low-passed mid
/// predictor, and the final sum/difference all run in the listing's
/// Q8/Q11/Q13 integer forms, so the output is bit-exact against the
/// reference decoder. The frame's output is delayed by one sample
/// relative to its input (the listing's two-sample buffering with the
/// resampler reading from offset 1), exactly like the mono path's
/// §4.2.8 one-sample delay.
///
/// * `mid` — the mid channel's §4.2.7.9 reconstruction (`n` samples).
/// * `side` — the side channel's reconstruction, or `None` when the
///   side channel is not coded this frame (§4.2.7.2 mid-only flag);
///   zeros are substituted but the carried side history still applies,
///   matching the reference decoder.
/// * `weights` — this frame's §4.2.7.1 Q13 weights.
/// * `fs_khz` — the internal rate in kHz (8/12/16), fixing the 8 ms
///   interpolation length.
///
/// Errors if `mid` is empty, shorter than the 8 ms ramp, or if `side`
/// has a different length.
pub fn stereo_ms_to_lr_i16(
    fs_khz: usize,
    mid: &[i16],
    side: Option<&[i16]>,
    weights: StereoWeightsQ13,
    state: &mut StereoUnmixStateI16,
) -> Result<StereoFrameI16, Error> {
    use crate::silk_decode_core::{rshift_round, sat16, smlawb, smulbb};

    let n = mid.len();
    let interp_len = 8 * fs_khz; // STEREO_INTERP_LEN_MS = 8
    if n < interp_len || n < 2 {
        return Err(Error::MalformedPacket);
    }
    if let Some(s) = side {
        if s.len() != n {
            return Err(Error::MalformedPacket);
        }
    }

    // x1/x2 with the two-sample carried history at the front.
    // Profluens patch: unmixing scratch with the 2-sample carried history — bump arena.
    let mut x1 = Vec::with_capacity_in(n + 2, crate::bump::DecodeBump);
    x1.extend_from_slice(&state.s_mid);
    x1.extend_from_slice(mid);
    let mut x2 = Vec::with_capacity_in(n + 2, crate::bump::DecodeBump);
    x2.extend_from_slice(&state.s_side);
    match side {
        Some(s) => x2.extend_from_slice(s),
        None => x2.resize(n + 2, 0),
    }
    state.s_mid = [x1[n], x1[n + 1]];
    state.s_side = [x2[n], x2[n + 1]];

    // Interpolate predictors over the first 8 ms and add the prediction
    // to the side channel.
    let mut pred0_q13 = state.pred_prev_q13[0];
    let mut pred1_q13 = state.pred_prev_q13[1];
    let denom_q16 = (1 << 16) / (interp_len as i32);
    let delta0_q13 = rshift_round(
        smulbb(weights.w0_q13 - state.pred_prev_q13[0], denom_q16),
        16,
    );
    let delta1_q13 = rshift_round(
        smulbb(weights.w1_q13 - state.pred_prev_q13[1], denom_q16),
        16,
    );
    let unmix_at = |k: usize, p0: i32, p1: i32, x2k1: i16| -> i16 {
        // sum = ((x1[k] + x1[k+2] + 2·x1[k+1]) << 9)  — Q11.
        let sum = (i32::from(x1[k]) + i32::from(x1[k + 2]) + (i32::from(x1[k + 1]) << 1)) << 9;
        let sum = smlawb(i32::from(x2k1) << 8, sum, p0); // Q8
        let sum = smlawb(sum, i32::from(x1[k + 1]) << 11, p1); // Q8
        sat16(rshift_round(sum, 8))
    };
    for k in 0..interp_len {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        x2[k + 1] = unmix_at(k, pred0_q13, pred1_q13, x2[k + 1]);
    }
    for k in interp_len..n {
        x2[k + 1] = unmix_at(k, weights.w0_q13, weights.w1_q13, x2[k + 1]);
    }
    state.pred_prev_q13 = [weights.w0_q13, weights.w1_q13];

    // Convert to left/right (sum / difference), n samples from offset 1
    // (the built-in one-sample delay).
    // Profluens patch: L/R output, consumed via slice by the caller's accumulator — bump arena.
    let mut left = Vec::with_capacity_in(n, crate::bump::DecodeBump);
    left.resize(n, 0i16);
    let mut right = Vec::with_capacity_in(n, crate::bump::DecodeBump);
    right.resize(n, 0i16);
    for k in 0..n {
        let sum = i32::from(x1[k + 1]) + i32::from(x2[k + 1]);
        let diff = i32::from(x1[k + 1]) - i32::from(x2[k + 1]);
        left[k] = sat16(sum);
        right[k] = sat16(diff);
    }
    Ok(StereoFrameI16 { left, right })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!(
            (a - b).abs() < 1e-5,
            "expected {b}, got {a} (delta {})",
            (a - b).abs()
        );
    }

    #[test]
    fn interp_phase_table() {
        assert_eq!(interp_phase_samples(Bandwidth::Nb).unwrap(), 64);
        assert_eq!(interp_phase_samples(Bandwidth::Mb).unwrap(), 96);
        assert_eq!(interp_phase_samples(Bandwidth::Wb).unwrap(), 128);
        assert!(interp_phase_samples(Bandwidth::Swb).is_err());
        assert!(interp_phase_samples(Bandwidth::Fb).is_err());
    }

    #[test]
    fn state_starts_and_resets_zero() {
        let mut s = StereoUnmixState::new();
        assert_eq!(s.mid_hist, [0.0, 0.0]);
        assert_eq!(s.side_hist, 0.0);
        assert_eq!(s.prev_weights, StereoWeightsQ13::default());
        s.mid_hist = [0.3, -0.2];
        s.side_hist = 0.1;
        s.prev_weights = StereoWeightsQ13 {
            w0_q13: 5,
            w1_q13: 7,
        };
        s.reset();
        assert_eq!(s, StereoUnmixState::new());
    }

    #[test]
    fn rejects_empty_and_mismatched() {
        let mut s = StereoUnmixState::new();
        assert!(stereo_ms_to_lr(
            Bandwidth::Wb,
            &[],
            None,
            StereoWeightsQ13::default(),
            &mut s
        )
        .is_err());
        let mid = vec![0.0f32; 80];
        let side = vec![0.0f32; 79];
        assert!(stereo_ms_to_lr(
            Bandwidth::Wb,
            &mid,
            Some(&side),
            StereoWeightsQ13::default(),
            &mut s
        )
        .is_err());
    }

    /// With zero weights and no side channel, the unmixer collapses to
    /// `left[i] = right[i] = mid[i-1]` (a one-sample delay), and the
    /// first sample reads the zeroed mid history → 0.
    #[test]
    fn zero_weights_no_side_is_delayed_mono() {
        let mut s = StereoUnmixState::new();
        let mid: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let out = stereo_ms_to_lr(
            Bandwidth::Wb,
            &mid,
            None,
            StereoWeightsQ13::default(),
            &mut s,
        )
        .unwrap();
        // left[i] = right[i] = mid[i-1]; mid[-1] = 0.
        let expect = [0.0, 0.1, 0.2, 0.3];
        for (i, &e) in expect.iter().enumerate() {
            approx(out.left[i], e);
            approx(out.right[i], e);
        }
        // L == R when there's no side / weights.
        assert_eq!(out.left, out.right);
    }

    /// Hand-computed reference: WB frame, constant non-zero weights so
    /// the phase-1 ramp is flat (prev == current), a coded side channel,
    /// and a handful of mid samples. We reproduce the §4.2.8 formulas by
    /// hand and check the unmixer matches.
    #[test]
    fn known_midside_reconstruction_constant_weights() {
        // Use prev == current so min(i,n1)*step term vanishes and the
        // weights are constant w0 = w0_Q13/8192, w1 = w1_Q13/8192.
        let w = StereoWeightsQ13 {
            w0_q13: 4096, // -> w0 = 0.5
            w1_q13: 8192, // -> w1 = 1.0
        };
        let mut s = StereoUnmixState::new();
        s.prev_weights = w; // flat ramp

        let mid = vec![0.4f32, -0.2, 0.1, 0.3, -0.1];
        let side = vec![0.05f32, 0.0, -0.1, 0.2, 0.1];

        let out = stereo_ms_to_lr(Bandwidth::Wb, &mid, Some(&side), w, &mut s).unwrap();

        let w0 = 0.5f32;
        let w1 = 1.0f32;
        // History is all zero (fresh state for mid[-1], mid[-2], side[-1]).
        let mut mhist = [0.0f32, 0.0]; // [mid[i-2], mid[i-1]] sliding
        let mut shist = 0.0f32; // side[i-1]
        for i in 0..mid.len() {
            let m_i = mid[i];
            let m_i1 = mhist[1];
            let m_i2 = mhist[0];
            let s_i1 = shist;
            let p0 = (m_i2 + 2.0 * m_i1 + m_i) / 4.0;
            let l = ((1.0 + w1) * m_i1 + s_i1 + w0 * p0).clamp(-1.0, 1.0);
            let r = ((1.0 - w1) * m_i1 - s_i1 - w0 * p0).clamp(-1.0, 1.0);
            approx(out.left[i], l);
            approx(out.right[i], r);
            // slide
            mhist = [m_i1, m_i];
            shist = side[i];
        }
    }

    /// The phase-1 weight ramp: with prev != current weights, the first
    /// `n1` samples interpolate linearly. We verify the effective weight
    /// at sample 0 equals prev/8192 and at sample >= n1 equals cur/8192.
    /// We isolate w1 by using a unit mid impulse delayed one sample and
    /// zero w0 / side.
    #[test]
    fn phase1_ramp_endpoints() {
        // NB so n1 = 64; make a frame long enough to reach the steady
        // phase. w0 = 0 to drop the p0 term. Side uncoded.
        let n1 = 64usize;
        let n2 = n1 + 4;
        let w_cur = StereoWeightsQ13 {
            w0_q13: 0,
            w1_q13: 8192,
        }; // cur w1 = 1.0
        let mut s = StereoUnmixState::new();
        s.prev_weights = StereoWeightsQ13 {
            w0_q13: 0,
            w1_q13: 0,
        }; // prev w1 = 0.0

        // mid constant 0.4 so mid[i-1] = 0.4 for i >= 1 (small enough
        // that the ramped left term stays inside the clamp range).
        let m = 0.4f32;
        let mid = vec![m; n2];
        let out = stereo_ms_to_lr(Bandwidth::Nb, &mid, None, w_cur, &mut s).unwrap();

        // left[i] = (1 + w1(i)) * mid[i-1]; with mid[i-1]=0.4 for i>=1.
        // w1(i) = 0 + min(i, n1) * (1.0 - 0)/n1 = min(i,n1)/n1.
        // i = 1 → w1 = 1/64 → left = (1 + 1/64)*0.4.
        approx(out.left[1], (1.0 + 1.0 / 64.0) * m);
        // i = n1 → w1 = 1.0 → left = 2.0*0.4 = 0.8 (still in range).
        approx(out.left[n1], 2.0 * m);
        // steady region (i = n1 + 1) → w1 = 1.0 → 0.8.
        approx(out.left[n1 + 1], 2.0 * m);
        // right[i] = (1 - w1) * mid[i-1]; at i=1 → (1 - 1/64)*0.4.
        approx(out.right[1], (1.0 - 1.0 / 64.0) * m);
        // right at steady → (1 - 1.0)*0.4 = 0.
        approx(out.right[n1 + 1], 0.0);
    }

    /// History carries across frame boundaries: the second frame's
    /// first sample must read the previous frame's trailing mid / side
    /// samples and weights, not zero.
    #[test]
    fn history_carries_across_frames() {
        let w = StereoWeightsQ13 {
            w0_q13: 0,
            w1_q13: 0,
        }; // pure delay, L == R == mid[i-1]
        let mut s = StereoUnmixState::new();
        s.prev_weights = w;

        let frame1 = vec![0.1f32, 0.2, 0.3, 0.4];
        let _ = stereo_ms_to_lr(Bandwidth::Wb, &frame1, None, w, &mut s).unwrap();
        // After frame1 the mid history holds [0.3, 0.4].
        assert_eq!(s.mid_hist, [0.3, 0.4]);

        let frame2 = vec![0.5f32, 0.6, 0.7, 0.8];
        let out2 = stereo_ms_to_lr(Bandwidth::Wb, &frame2, None, w, &mut s).unwrap();
        // frame2 left[0] = mid[-1] = last sample of frame1 = 0.4.
        approx(out2.left[0], 0.4);
        approx(out2.left[1], 0.5);
    }

    /// Side history carries across frames too (for a coded side channel),
    /// and adds/subtracts symmetrically into L/R.
    #[test]
    fn side_history_carries_across_frames() {
        let w = StereoWeightsQ13 {
            w0_q13: 0,
            w1_q13: 0,
        };
        let mut s = StereoUnmixState::new();
        s.prev_weights = w;

        let mid1 = vec![0.0f32; 4];
        let side1 = vec![0.1f32, 0.2, 0.3, 0.4];
        let _ = stereo_ms_to_lr(Bandwidth::Wb, &mid1, Some(&side1), w, &mut s).unwrap();
        assert_eq!(s.side_hist, 0.4);

        let mid2 = vec![0.0f32; 4];
        let side2 = vec![0.5f32, 0.6, 0.7, 0.8];
        let out2 = stereo_ms_to_lr(Bandwidth::Wb, &mid2, Some(&side2), w, &mut s).unwrap();
        // mid all zero, w=0 → left[i] = side[i-1], right[i] = -side[i-1].
        // left[0] = side[-1] = last of side1 = 0.4.
        approx(out2.left[0], 0.4);
        approx(out2.right[0], -0.4);
        approx(out2.left[1], 0.5);
        approx(out2.right[1], -0.5);
    }

    /// A tiny deterministic LCG for the downmix roundtrips.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Uniform in [-0.4, 0.4]: safely inside the §4.2.8 clamp
            // for any codebook weight pair.
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.8
        }
    }

    /// The full-stream roundtrip: several frames of random L/R are
    /// downmixed by `stereo_lr_to_ms` (with one-sample lookahead across
    /// frame boundaries) and unmixed back by `stereo_ms_to_lr` with the
    /// same per-frame weights; the reconstruction equals the input
    /// delayed by exactly one sample (the §4.2.8 delay), the first
    /// output sample reading the zeroed histories.
    #[test]
    fn lr_to_ms_roundtrips_through_ms_to_lr() {
        for (bandwidth, n2) in [
            (Bandwidth::Nb, 160usize),
            (Bandwidth::Mb, 240),
            (Bandwidth::Wb, 320),
        ] {
            let mut rng = Lcg(0xD0_1985 ^ n2 as u64);
            let frames = 3usize;
            let left: Vec<f32> = (0..frames * n2).map(|_| rng.next_f32()).collect();
            let right: Vec<f32> = (0..frames * n2).map(|_| rng.next_f32()).collect();
            // Distinct per-frame weights exercise the §4.2.8 ramp.
            let frame_weights = [
                StereoWeightsQ13 {
                    w0_q13: -2950,
                    w1_q13: 820,
                },
                StereoWeightsQ13 {
                    w0_q13: 5000,
                    w1_q13: -6500,
                },
                StereoWeightsQ13 {
                    w0_q13: 820,
                    w1_q13: 10050,
                },
            ];

            let mut enc = StereoDownmixState::new();
            let mut dec = StereoUnmixState::new();
            let mut out_left = Vec::new();
            let mut out_right = Vec::new();
            for f in 0..frames {
                let l = &left[f * n2..(f + 1) * n2];
                let r = &right[f * n2..(f + 1) * n2];
                let next_lr = if f + 1 < frames {
                    Some((left[(f + 1) * n2], right[(f + 1) * n2]))
                } else {
                    None
                };
                let ms =
                    stereo_lr_to_ms(bandwidth, l, r, frame_weights[f], next_lr, &mut enc).unwrap();
                // Mid is exactly the frame-aligned (L + R) / 2.
                for k in 0..n2 {
                    approx(ms.mid[k], (l[k] + r[k]) / 2.0);
                }
                let rec = stereo_ms_to_lr(
                    bandwidth,
                    &ms.mid,
                    Some(&ms.side),
                    frame_weights[f],
                    &mut dec,
                )
                .unwrap();
                out_left.extend_from_slice(&rec.left);
                out_right.extend_from_slice(&rec.right);
            }

            // out[i] == in[i-1] globally; out[0] reads zeroed histories.
            assert!(out_left[0].abs() < 1e-5, "{bandwidth:?}");
            assert!(out_right[0].abs() < 1e-5, "{bandwidth:?}");
            for i in 1..frames * n2 {
                assert!(
                    (out_left[i] - left[i - 1]).abs() < 1e-4,
                    "{bandwidth:?} left sample {i}: {} vs {}",
                    out_left[i],
                    left[i - 1]
                );
                assert!(
                    (out_right[i] - right[i - 1]).abs() < 1e-4,
                    "{bandwidth:?} right sample {i}: {} vs {}",
                    out_right[i],
                    right[i - 1]
                );
            }
        }
    }

    /// Within a single decoded frame the last side sample is never
    /// consumed (it feeds the *next* frame's first reconstruction), so
    /// a single-frame roundtrip is exact regardless of the `next_lr`
    /// lookahead / hold choice.
    #[test]
    fn lr_to_ms_single_frame_roundtrip_ignores_lookahead() {
        let n2 = 160usize;
        let mut rng = Lcg(0x0AC4);
        let left: Vec<f32> = (0..n2).map(|_| rng.next_f32()).collect();
        let right: Vec<f32> = (0..n2).map(|_| rng.next_f32()).collect();
        let w = StereoWeightsQ13 {
            w0_q13: 6500,
            w1_q13: -820,
        };
        for next in [None, Some((0.35f32, -0.2f32))] {
            let mut enc = StereoDownmixState::new();
            let mut dec = StereoUnmixState::new();
            let ms = stereo_lr_to_ms(Bandwidth::Nb, &left, &right, w, next, &mut enc).unwrap();
            let rec = stereo_ms_to_lr(Bandwidth::Nb, &ms.mid, Some(&ms.side), w, &mut dec).unwrap();
            for i in 1..n2 {
                approx(rec.left[i], left[i - 1]);
                approx(rec.right[i], right[i - 1]);
            }
        }
    }

    /// Planted-weight recovery: a side channel constructed as exactly
    /// `w0*p0 + w1*mid` (no residual) estimates back to the planted
    /// Q13 pair within rounding.
    #[test]
    fn estimate_recovers_planted_weights() {
        let mut rng = Lcg(0xE571_0001);
        let n2 = 320usize;
        let mid: Vec<f32> = (0..n2).map(|_| rng.next_f32()).collect();
        let prev_mid = rng.next_f32();
        let mid_next = rng.next_f32();
        let (w0, w1) = (0.37f64, -0.61f64);
        let mut side_raw = vec![0.0f32; n2];
        for k in 0..n2 {
            let m_km1 = if k >= 1 { mid[k - 1] } else { prev_mid } as f64;
            let m_kp1 = if k + 1 < n2 { mid[k + 1] } else { mid_next } as f64;
            let p0 = (m_km1 + 2.0 * mid[k] as f64 + m_kp1) / 4.0;
            side_raw[k] = (w0 * p0 + w1 * mid[k] as f64) as f32;
        }
        let est = estimate_stereo_weights(&mid, &side_raw, prev_mid, mid_next).unwrap();
        assert!(
            (est.w0_q13 - (w0 * 8192.0).round() as i32).abs() <= 1,
            "w0 {} vs planted {}",
            est.w0_q13,
            (w0 * 8192.0).round()
        );
        assert!(
            (est.w1_q13 - (w1 * 8192.0).round() as i32).abs() <= 1,
            "w1 {} vs planted {}",
            est.w1_q13,
            (w1 * 8192.0).round()
        );
    }

    /// A silent (or constant-zero) mid channel has no usable predictor:
    /// the estimator returns zero weights instead of a singular solve,
    /// and bad shapes are rejected.
    #[test]
    fn estimate_zero_mid_and_bad_input() {
        let side = vec![0.25f32; 160];
        let mid = vec![0.0f32; 160];
        let est = estimate_stereo_weights(&mid, &side, 0.0, 0.0).unwrap();
        assert_eq!(est, StereoWeightsQ13::default());
        assert!(estimate_stereo_weights(&[], &[], 0.0, 0.0).is_err());
        assert!(estimate_stereo_weights(&mid, &side[..159], 0.0, 0.0).is_err());
    }

    /// The full encoder analysis chain — raw L/R → estimate → quantize
    /// → downmix — codes a side channel with strictly less energy than
    /// the unpredicted `(L - R)/2`, and the roundtrip through the
    /// §4.2.8 unmixer still reproduces the input at the one-sample
    /// delay.
    #[test]
    fn estimate_quantize_downmix_reduces_side_energy() {
        use crate::silk_frame::StereoWeightSymbols;
        let mut rng = Lcg(0xE571_C0DE);
        let n2 = 320usize;
        // Correlated stereo: right = mostly-scaled left plus a small
        // independent component, so the mid channel predicts the side.
        let left: Vec<f32> = (0..n2).map(|_| rng.next_f32()).collect();
        let right: Vec<f32> = left
            .iter()
            .map(|&l| 0.4 * l + 0.12 * rng.next_f32())
            .collect();

        let mid: Vec<f32> = left
            .iter()
            .zip(&right)
            .map(|(&l, &r)| (l + r) / 2.0)
            .collect();
        let side_raw: Vec<f32> = left
            .iter()
            .zip(&right)
            .map(|(&l, &r)| (l - r) / 2.0)
            .collect();
        let target = estimate_stereo_weights(&mid, &side_raw, 0.0, mid[n2 - 1]).unwrap();
        let quintuple = StereoWeightSymbols::quantize(crate::silk_frame::StereoPredictionWeights {
            w0_q13: target.w0_q13,
            w1_q13: target.w1_q13,
        });
        let coded = quintuple.weights();
        let coded_w = StereoWeightsQ13 {
            w0_q13: coded.w0_q13,
            w1_q13: coded.w1_q13,
        };

        // Downmix from a state whose previous weights equal the coded
        // pair (flat ramp — the steady-state condition).
        let mut enc = StereoDownmixState::new();
        enc.prev_weights = coded_w;
        let ms = stereo_lr_to_ms(Bandwidth::Wb, &left, &right, coded_w, None, &mut enc).unwrap();

        let energy = |v: &[f32]| -> f64 { v.iter().map(|&x| (x as f64) * (x as f64)).sum() };
        assert!(
            energy(&ms.side) < 0.7 * energy(&side_raw),
            "predicted side energy {} not below unpredicted {}",
            energy(&ms.side),
            energy(&side_raw)
        );

        // The roundtrip guarantee is weight-independent: decode back
        // and confirm the delayed identity still holds.
        let mut dec = StereoUnmixState::new();
        dec.prev_weights = coded_w;
        let rec =
            stereo_ms_to_lr(Bandwidth::Wb, &ms.mid, Some(&ms.side), coded_w, &mut dec).unwrap();
        for i in 1..n2 {
            assert!((rec.left[i] - left[i - 1]).abs() < 1e-4, "left {i}");
            assert!((rec.right[i] - right[i - 1]).abs() < 1e-4, "right {i}");
        }
    }

    /// Downmix input validation mirrors the unmixer: empty frames,
    /// length mismatches, and SILK-illegal bandwidths are rejected, and
    /// the state resets to zero.
    #[test]
    fn lr_to_ms_rejects_bad_input_and_resets() {
        let mut s = StereoDownmixState::new();
        assert!(stereo_lr_to_ms(
            Bandwidth::Wb,
            &[],
            &[],
            StereoWeightsQ13::default(),
            None,
            &mut s
        )
        .is_err());
        assert!(stereo_lr_to_ms(
            Bandwidth::Wb,
            &[0.0; 4],
            &[0.0; 3],
            StereoWeightsQ13::default(),
            None,
            &mut s
        )
        .is_err());
        assert!(stereo_lr_to_ms(
            Bandwidth::Swb,
            &[0.0; 4],
            &[0.0; 4],
            StereoWeightsQ13::default(),
            None,
            &mut s
        )
        .is_err());
        let _ = stereo_lr_to_ms(
            Bandwidth::Wb,
            &[0.1; 320],
            &[0.2; 320],
            StereoWeightsQ13 {
                w0_q13: 820,
                w1_q13: 820,
            },
            None,
            &mut s,
        )
        .unwrap();
        assert_ne!(s, StereoDownmixState::new());
        s.reset();
        assert_eq!(s, StereoDownmixState::new());
    }

    /// Clamping: drive both L and R out of range and confirm the
    /// `[-1.0, 1.0]` clamp from §4.2.8 is applied.
    #[test]
    fn output_is_clamped() {
        let w = StereoWeightsQ13 {
            w0_q13: 8192 * 4, // huge w0
            w1_q13: 8192 * 4, // huge w1
        };
        let mut s = StereoUnmixState::new();
        s.prev_weights = w;
        let mid = vec![1.0f32; 8];
        let side = vec![1.0f32; 8];
        let out = stereo_ms_to_lr(Bandwidth::Wb, &mid, Some(&side), w, &mut s).unwrap();
        for i in 0..8 {
            assert!((-1.0..=1.0).contains(&out.left[i]), "left {}", out.left[i]);
            assert!(
                (-1.0..=1.0).contains(&out.right[i]),
                "right {}",
                out.right[i]
            );
        }
    }
}
