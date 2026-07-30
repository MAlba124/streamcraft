//! CELT §4.3.7.1 pitch post-filter *response*
//! (RFC 6716 §4.3.7.1, pp. 120–121).
//!
//! The post-filter is the second-to-last stage of the CELT decode
//! pipeline: it runs on the output of the inverse MDCT (after weighted
//! overlap-add, §4.3.7) and feeds the §4.3.7.2 de-emphasis filter
//! ([`crate::celt_deemphasis`]). Its *parameters* — the enable bit, the
//! octave, the fine pitch, the gain index, and the tapset — are decoded
//! near the BEGINNING of the CELT frame (immediately after the silence
//! flag) so the §4.3.3 bit allocator can account for the bits they
//! consume; that decode lives in [`crate::celt_header::CeltPostFilter`].
//! This module owns the *application* of those parameters: the
//! time-domain comb filter the RFC states as the "post-filter response".
//!
//! ## The comb filter
//!
//! RFC 6716 §4.3.7.1 (p. 121) gives the response as
//!
//! ```text
//!   y(n) = x(n) + G*( g0*y(n-T)
//!                   + g1*(y(n-T+1) + y(n-T-1))
//!                   + g2*(y(n-T+2) + y(n-T-2)) )
//! ```
//!
//! where `x(n)` is the IMDCT/overlap-add output, `T` is the pitch
//! period, `G` is the post-filter gain, and `(g0, g1, g2)` are the
//! tapset coefficients. This is a five-tap symmetric comb filter
//! centred on the delayed *output* sample `y(n-T)`: a centre tap `g0`,
//! a first symmetric pair `g1` at `y(n-T±1)`, and a second symmetric
//! pair `g2` at `y(n-T±2)`. The filter is recursive — it feeds back
//! past *output* samples `y`, not past input samples — so its state is
//! the trailing `T + 2` output values, which carry across frame
//! boundaries (reset only on a §4.5.2 CELT state reset).
//!
//! ### A note on the printed pair terms
//!
//! The standards-track text prints the two pair terms as
//! `g1*(y(n-T+1)+y(n-T+1))` and `g2*(y(n-T+2)+y(n-T+2))` — each pair
//! repeats the `+1` / `+2` index. A comb filter whose two summed
//! samples are identical is not a filter at all (it would be
//! `2*g1*y(n-T+1)`, an asymmetric single tap), which contradicts the
//! surrounding prose describing a symmetric tap *set* "g0, g1, g2"
//! arranged around the pitch period: the only reading consistent with a
//! symmetric three-coefficient tap set is the pair
//! `y(n-T+1) + y(n-T-1)` for `g1` and `y(n-T+2) + y(n-T-2)` for `g2`.
//! The module uses the symmetric form; the asymmetric printed form is
//! treated as a transcription slip in the ASCII rendering of the
//! equation. (Reading the printed form literally would make the filter
//! non-symmetric and break the comb structure the same paragraph
//! describes.)
//!
//! ## Tapsets and gain
//!
//! The three tapsets (§4.3.7.1) and the gain map are exact decimals /
//! fractions stated in the text:
//!
//! * tapset 0 → `(0.3066406250, 0.2170410156, 0.1296386719)`
//! * tapset 1 → `(0.4638671875, 0.2680664062, 0.0)`
//! * tapset 2 → `(0.7998046875, 0.1000976562, 0.0)`
//! * gain `G = 3 * (gain_index + 1) / 32` for `gain_index ∈ 0..=7`.
//!
//! ## Gain transitions
//!
//! RFC 6716 §4.3.7.1 (p. 121): "During a transition between different
//! gains, a smooth transition is calculated using the square of the
//! MDCT window. It is important that values of y(n) be interpolated one
//! at a time such that the past value of y(n) used is interpolated."
//! [`crossfade_transition`] implements that crossfade across the
//! [`crate::celt_mdct_window`] overlap region: at each output position
//! the old-parameter filter output and the new-parameter filter output
//! are mixed with the squared window weights `1 - W(n)^2` / `W(n)^2`,
//! and — per the "interpolated one at a time" requirement — the mixed
//! value is the one written back to the shared output history both
//! filters read from on the next step. The transition length is the
//! §1 fixed decoder overlap (`OVERLAP` samples at 48 kHz).
//!
//! ## Provenance
//!
//! Comb-filter response, tapset coefficients, gain map, pitch-period
//! bound, and the squared-window transition rule: RFC 6716 §4.3.7.1
//! (pp. 120–121), held in-repo at `docs/audio/opus/rfc6716-opus.txt`.
//! The squared window is the §4.3.7 window already implemented in
//! [`crate::celt_mdct_window`]. No external library source was
//! consulted; the response equation, the coefficient set, and the
//! transition rule are stated directly in the standards-track text.

use crate::celt_header::CeltPostFilter;
use crate::celt_mdct_window::{self, MdctWindowError};

/// Minimum post-filter pitch period `T` (§4.3.7.1: "bounded between 15
/// and 1022, inclusively").
pub const POST_FILTER_MIN_PERIOD: u16 = 15;

/// Maximum post-filter pitch period `T` (§4.3.7.1).
pub const POST_FILTER_MAX_PERIOD: u16 = 1022;

/// Number of distinct tapsets (§4.3.7.1 `{2, 1, 1}/4` symbol → 0/1/2).
pub const POST_FILTER_TAPSET_COUNT: usize = 3;

/// Gain numerator in `G = 3*(gain_index + 1)/32` (§4.3.7.1).
pub const POST_FILTER_GAIN_NUMERATOR: u32 = 3;

/// Gain denominator in `G = 3*(gain_index + 1)/32` (§4.3.7.1).
pub const POST_FILTER_GAIN_DENOMINATOR: u32 = 32;

/// The largest raw 3-bit gain index (§4.3.7.1: "the gain is decoded as
/// three raw bits").
pub const POST_FILTER_GAIN_INDEX_MAX: u8 = 7;

/// The three §4.3.7.1 tapsets `(g0, g1, g2)`, indexed by tapset.
///
/// These are the exact decimals printed in the standards-track text;
/// each is an exact binary fraction (a multiple of `1/8192`).
pub const POST_FILTER_TAPS: [[f64; 3]; POST_FILTER_TAPSET_COUNT] = [
    [0.306_640_625_0, 0.217_041_015_6, 0.129_638_671_9],
    [0.463_867_187_5, 0.268_066_406_2, 0.0],
    [0.799_804_687_5, 0.100_097_656_2, 0.0],
];

/// Errors returnable by the §4.3.7.1 post-filter helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostFilterError {
    /// The tapset index is `>= POST_FILTER_TAPSET_COUNT`.
    TapsetOutOfRange {
        /// The supplied tapset index.
        tapset: u8,
    },
    /// The pitch period `T` is outside `15..=1022`.
    PeriodOutOfRange {
        /// The supplied period.
        period: u16,
    },
    /// The gain index is `> POST_FILTER_GAIN_INDEX_MAX`.
    GainIndexOutOfRange {
        /// The supplied gain index.
        gain_index: u8,
    },
    /// The output buffer passed to a block helper is shorter than the
    /// input.
    OutputBufferTooSmall {
        /// Number of input samples.
        input_len: usize,
        /// Length of the supplied output buffer.
        output_len: usize,
    },
    /// The crossfade transition length does not match the two output
    /// slices' lengths.
    TransitionLengthMismatch {
        /// Length expected (the overlap / window length).
        expected: usize,
        /// Length actually supplied.
        provided: usize,
    },
    /// The §4.3.7 window lookup feeding the transition crossfade failed.
    Window(MdctWindowError),
}

impl core::fmt::Display for PostFilterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PostFilterError::TapsetOutOfRange { tapset } => {
                write!(f, "post-filter tapset out of range: {tapset} (max 2)")
            }
            PostFilterError::PeriodOutOfRange { period } => write!(
                f,
                "post-filter period out of range: {period} (must be 15..=1022)"
            ),
            PostFilterError::GainIndexOutOfRange { gain_index } => {
                write!(
                    f,
                    "post-filter gain index out of range: {gain_index} (max 7)"
                )
            }
            PostFilterError::OutputBufferTooSmall {
                input_len,
                output_len,
            } => write!(
                f,
                "post-filter output buffer too small: need {input_len} samples, got {output_len}"
            ),
            PostFilterError::TransitionLengthMismatch { expected, provided } => write!(
                f,
                "post-filter transition length mismatch: expected {expected}, got {provided}"
            ),
            PostFilterError::Window(e) => write!(f, "post-filter window error: {e}"),
        }
    }
}

impl std::error::Error for PostFilterError {}

impl From<MdctWindowError> for PostFilterError {
    fn from(e: MdctWindowError) -> Self {
        PostFilterError::Window(e)
    }
}

/// Resolve the §4.3.7.1 tapset index to its `(g0, g1, g2)` coefficients.
pub fn tapset_coefficients(tapset: u8) -> Result<[f64; 3], PostFilterError> {
    POST_FILTER_TAPS
        .get(tapset as usize)
        .copied()
        .ok_or(PostFilterError::TapsetOutOfRange { tapset })
}

/// Compute the §4.3.7.1 post-filter gain `G = 3*(gain_index + 1)/32`.
///
/// `gain_index` is the raw 3-bit field (`0..=7`); `G` ranges from
/// `3/32 = 0.09375` (index 0) to `24/32 = 0.75` (index 7).
pub fn post_filter_gain(gain_index: u8) -> Result<f64, PostFilterError> {
    if gain_index > POST_FILTER_GAIN_INDEX_MAX {
        return Err(PostFilterError::GainIndexOutOfRange { gain_index });
    }
    let g = f64::from(POST_FILTER_GAIN_NUMERATOR * (u32::from(gain_index) + 1))
        / f64::from(POST_FILTER_GAIN_DENOMINATOR);
    Ok(g)
}

/// A single set of §4.3.7.1 post-filter coefficients, ready to apply.
///
/// Built from a decoded [`CeltPostFilter`] (or directly), this folds the
/// gain into the three taps so the per-sample recurrence is a plain
/// multiply-accumulate against the output history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostFilterCoeffs {
    /// Pitch period `T` (`15..=1022`).
    pub period: u16,
    /// `G * g0` — the gain-scaled centre tap.
    pub a0: f64,
    /// `G * g1` — the gain-scaled first symmetric-pair tap.
    pub a1: f64,
    /// `G * g2` — the gain-scaled second symmetric-pair tap.
    pub a2: f64,
}

impl PostFilterCoeffs {
    /// Build the gain-scaled coefficients from a period, gain index, and
    /// tapset, validating each against §4.3.7.1.
    pub fn new(period: u16, gain_index: u8, tapset: u8) -> Result<Self, PostFilterError> {
        if !(POST_FILTER_MIN_PERIOD..=POST_FILTER_MAX_PERIOD).contains(&period) {
            return Err(PostFilterError::PeriodOutOfRange { period });
        }
        let g = post_filter_gain(gain_index)?;
        let [g0, g1, g2] = tapset_coefficients(tapset)?;
        Ok(Self {
            period,
            a0: g * g0,
            a1: g * g1,
            a2: g * g2,
        })
    }

    /// Build the gain-scaled coefficients from a decoded
    /// [`CeltPostFilter`] header struct.
    pub fn from_header(pf: &CeltPostFilter) -> Result<Self, PostFilterError> {
        Self::new(pf.period, pf.gain_index, pf.tapset)
    }

    /// The number of past output samples this filter reads (`T + 2`).
    #[must_use]
    pub fn history_len(&self) -> usize {
        usize::from(self.period) + 2
    }
}

/// A single-channel §4.3.7.1 post-filter carrying its output history
/// across frame boundaries.
///
/// The comb filter is recursive over past *output* samples `y`, so the
/// state is the trailing `T + 2` outputs. Construct one filter per
/// channel and reuse it across frames; the history carries continuously
/// (reset only on a §4.5.2 CELT state reset, via [`Self::reset`]).
///
/// The history buffer is sized for the maximum period
/// ([`POST_FILTER_MAX_PERIOD`] `+ 2`) so a within-stream period change
/// never needs reallocation.
#[derive(Debug, Clone)]
pub struct PostFilter {
    /// Ring of the most recent output samples; `hist[head]` is the
    /// oldest, advancing forward in time. Length is fixed at
    /// `POST_FILTER_MAX_PERIOD + 2`.
    hist: Vec<f64>,
    /// Index just past the most-recently-written sample.
    head: usize,
}

impl Default for PostFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl PostFilter {
    /// History ring capacity: the longest reach (`T = 1022`) plus the
    /// two extra symmetric-pair taps.
    const HIST_CAP: usize = POST_FILTER_MAX_PERIOD as usize + 2;

    /// Create a fresh filter with zeroed history (the state the CELT
    /// decoder starts each stream / post-reset run with).
    #[must_use]
    pub fn new() -> Self {
        Self {
            hist: vec![0.0; Self::HIST_CAP],
            head: 0,
        }
    }

    /// Reset the output history to all zeros, as on a §4.5.2 CELT state
    /// reset.
    pub fn reset(&mut self) {
        for v in self.hist.iter_mut() {
            *v = 0.0;
        }
        self.head = 0;
    }

    /// Fetch the output sample `delay` steps in the past (`y(n-delay)`).
    ///
    /// `delay` of 0 would be the sample about to be written and is never
    /// requested by the recurrence; the smallest delay the filter uses
    /// is `T - 2` (with `T >= 15`).
    fn past(&self, delay: usize) -> f64 {
        debug_assert!((1..=Self::HIST_CAP).contains(&delay));
        // head points one past the newest written sample.
        let idx = (self.head + Self::HIST_CAP - delay) % Self::HIST_CAP;
        self.hist[idx]
    }

    /// Push a newly-produced output sample into the history ring.
    fn push(&mut self, y: f64) {
        self.hist[self.head] = y;
        self.head = (self.head + 1) % Self::HIST_CAP;
    }

    /// Apply the §4.3.7.1 comb recurrence to one input sample and
    /// advance the state:
    ///
    /// ```text
    ///   y(n) = x(n) + a0*y(n-T) + a1*(y(n-T+1)+y(n-T-1))
    ///                           + a2*(y(n-T+2)+y(n-T-2))
    /// ```
    ///
    /// where `(a0, a1, a2)` are the gain-scaled taps. The produced `y(n)`
    /// is pushed into the history so subsequent samples in the same
    /// stream see it.
    #[must_use]
    pub fn step(&mut self, x: f64, c: &PostFilterCoeffs) -> f64 {
        let t = usize::from(c.period);
        let y = x
            + c.a0 * self.past(t)
            + c.a1 * (self.past(t - 1) + self.past(t + 1))
            + c.a2 * (self.past(t - 2) + self.past(t + 2));
        self.push(y);
        y
    }

    /// Apply the §4.3.7.1 post-filter to a block of samples in place
    /// with one fixed coefficient set, updating the carried history.
    pub fn process_in_place(&mut self, samples: &mut [f64], c: &PostFilterCoeffs) {
        for x in samples.iter_mut() {
            *x = self.step(*x, c);
        }
    }

    /// Apply the §4.3.7.1 post-filter to `input`, writing to `output`,
    /// and update the carried history.
    ///
    /// Returns the number of samples written (`input.len()`), or
    /// [`PostFilterError::OutputBufferTooSmall`] if `output` is shorter
    /// than `input`. On error the filter state is left unchanged.
    pub fn process(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        c: &PostFilterCoeffs,
    ) -> Result<usize, PostFilterError> {
        if output.len() < input.len() {
            return Err(PostFilterError::OutputBufferTooSmall {
                input_len: input.len(),
                output_len: output.len(),
            });
        }
        for (i, &x) in input.iter().enumerate() {
            output[i] = self.step(x, c);
        }
        Ok(input.len())
    }

    /// Apply a §4.3.7.1 *gain transition*: a per-sample crossfade from
    /// the old coefficient set to the new one across the §4.3.7 overlap
    /// region, using the square of the MDCT window.
    ///
    /// At output position `n ∈ 0..overlap` the window square
    /// `w2 = W(n)^2` rises from 0 to 1, and the produced sample is
    ///
    /// ```text
    ///   y(n) = (1 - w2) * y_old(n) + w2 * y_new(n)
    /// ```
    ///
    /// where `y_old` / `y_new` are the two filters' comb responses at the
    /// same input sample. Per the §4.3.7.1 requirement that "values of
    /// y(n) be interpolated one at a time such that the past value of
    /// y(n) used is interpolated", the *crossfaded* `y(n)` is the value
    /// pushed into the shared output history both branches read from on
    /// the next step — so each branch's feedback already sees the mixed
    /// past output, not its own un-mixed one.
    ///
    /// `overlap` must equal `input.len()` and `output.len()`.
    pub fn process_gain_transition(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        old: &PostFilterCoeffs,
        new: &PostFilterCoeffs,
        overlap: usize,
    ) -> Result<usize, PostFilterError> {
        if input.len() != overlap {
            return Err(PostFilterError::TransitionLengthMismatch {
                expected: overlap,
                provided: input.len(),
            });
        }
        if output.len() < overlap {
            return Err(PostFilterError::OutputBufferTooSmall {
                input_len: overlap,
                output_len: output.len(),
            });
        }
        for (n, &x) in input.iter().enumerate() {
            // Both filters read the same (shared) past output. Compute
            // each branch's comb response WITHOUT pushing — we push the
            // crossfaded result instead, so the feedback the RFC says
            // must be "the interpolated past value" is exactly what the
            // next sample reads.
            let y_old = self.comb_response(x, old);
            let y_new = self.comb_response(x, new);
            let w2 = {
                let w = celt_mdct_window::window_tap(n, overlap)?;
                w * w
            };
            let y = (1.0 - w2) * y_old + w2 * y_new;
            self.push(y);
            output[n] = y;
        }
        Ok(overlap)
    }

    /// The §4.3.7.1 comb response at `x` using `c`, WITHOUT advancing the
    /// history (used by the gain-transition crossfade, which pushes the
    /// mixed value once both branches are evaluated).
    fn comb_response(&self, x: f64, c: &PostFilterCoeffs) -> f64 {
        let t = usize::from(c.period);
        x + c.a0 * self.past(t)
            + c.a1 * (self.past(t - 1) + self.past(t + 1))
            + c.a2 * (self.past(t - 2) + self.past(t + 2))
    }
}

/// Crossfade two already-computed post-filter outputs across the
/// §4.3.7 overlap region using the squared MDCT window (§4.3.7.1 gain
/// transition), as a standalone helper for callers that produced the
/// two branches separately.
///
/// `old_out` and `new_out` must both have length `overlap`. The result
/// `out[n] = (1 - W(n)^2) * old_out[n] + W(n)^2 * new_out[n]` is written
/// to `out`. This is the non-recursive view of the transition — it does
/// not carry feedback state; use [`PostFilter::process_gain_transition`]
/// for the recursive, feedback-aware path the decoder runs.
pub fn crossfade_transition(
    old_out: &[f64],
    new_out: &[f64],
    out: &mut [f64],
    overlap: usize,
) -> Result<(), PostFilterError> {
    if old_out.len() != overlap || new_out.len() != overlap {
        return Err(PostFilterError::TransitionLengthMismatch {
            expected: overlap,
            provided: old_out.len().min(new_out.len()),
        });
    }
    if out.len() < overlap {
        return Err(PostFilterError::OutputBufferTooSmall {
            input_len: overlap,
            output_len: out.len(),
        });
    }
    for n in 0..overlap {
        let w = celt_mdct_window::window_tap(n, overlap)?;
        let w2 = w * w;
        out[n] = (1.0 - w2) * old_out[n] + w2 * new_out[n];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three tapsets match the §4.3.7.1 printed decimals exactly.
    #[test]
    fn tapsets_match_rfc() {
        assert_eq!(
            tapset_coefficients(0).unwrap(),
            [0.306_640_625_0, 0.217_041_015_6, 0.129_638_671_9]
        );
        assert_eq!(
            tapset_coefficients(1).unwrap(),
            [0.463_867_187_5, 0.268_066_406_2, 0.0]
        );
        assert_eq!(
            tapset_coefficients(2).unwrap(),
            [0.799_804_687_5, 0.100_097_656_2, 0.0]
        );
    }

    /// Tapsets 1 and 2 zero out `g2` (§4.3.7.1).
    #[test]
    fn tapsets_one_and_two_drop_g2() {
        assert_eq!(tapset_coefficients(1).unwrap()[2], 0.0);
        assert_eq!(tapset_coefficients(2).unwrap()[2], 0.0);
    }

    /// Out-of-range tapset is rejected.
    #[test]
    fn tapset_out_of_range_rejected() {
        assert_eq!(
            tapset_coefficients(3),
            Err(PostFilterError::TapsetOutOfRange { tapset: 3 })
        );
    }

    /// `G = 3*(gain_index+1)/32` worked at the endpoints and a midpoint.
    #[test]
    fn gain_formula() {
        assert_eq!(post_filter_gain(0).unwrap(), 3.0 / 32.0);
        assert_eq!(post_filter_gain(7).unwrap(), 24.0 / 32.0);
        assert_eq!(post_filter_gain(3).unwrap(), 12.0 / 32.0);
    }

    /// Gain is monotone increasing in the index.
    #[test]
    fn gain_monotone() {
        let mut prev = -1.0;
        for gi in 0..=POST_FILTER_GAIN_INDEX_MAX {
            let g = post_filter_gain(gi).unwrap();
            assert!(g > prev, "gain not monotone at {gi}");
            prev = g;
        }
        // Max gain is 0.75 (well below 1.0, so the recursive comb stays
        // bounded for the tap sets used).
        assert_eq!(post_filter_gain(POST_FILTER_GAIN_INDEX_MAX).unwrap(), 0.75);
    }

    /// Out-of-range gain index is rejected.
    #[test]
    fn gain_index_out_of_range_rejected() {
        assert_eq!(
            post_filter_gain(8),
            Err(PostFilterError::GainIndexOutOfRange { gain_index: 8 })
        );
    }

    /// `PostFilterCoeffs::new` folds the gain into each tap.
    #[test]
    fn coeffs_fold_gain() {
        let c = PostFilterCoeffs::new(100, 7, 0).unwrap();
        let g = 0.75;
        assert_eq!(c.a0, g * 0.306_640_625_0);
        assert_eq!(c.a1, g * 0.217_041_015_6);
        assert_eq!(c.a2, g * 0.129_638_671_9);
        assert_eq!(c.period, 100);
        assert_eq!(c.history_len(), 102);
    }

    /// Period bounds enforced (§4.3.7.1: 15..=1022).
    #[test]
    fn period_bounds_enforced() {
        assert!(PostFilterCoeffs::new(15, 0, 0).is_ok());
        assert!(PostFilterCoeffs::new(1022, 0, 0).is_ok());
        assert_eq!(
            PostFilterCoeffs::new(14, 0, 0),
            Err(PostFilterError::PeriodOutOfRange { period: 14 })
        );
        assert_eq!(
            PostFilterCoeffs::new(1023, 0, 0),
            Err(PostFilterError::PeriodOutOfRange { period: 1023 })
        );
    }

    /// Building from a header struct routes through the same validation.
    #[test]
    fn from_header_round_trips() {
        let pf = CeltPostFilter {
            octave: 2,
            period: 64,
            gain_index: 4,
            tapset: 1,
        };
        let c = PostFilterCoeffs::from_header(&pf).unwrap();
        assert_eq!(c, PostFilterCoeffs::new(64, 4, 1).unwrap());
    }

    /// A fresh filter has zeroed history: with all-zero past output the
    /// first samples of a stream pass through unchanged (the comb taps
    /// all read zeros).
    #[test]
    fn fresh_filter_passes_through_until_period() {
        let c = PostFilterCoeffs::new(15, 7, 0).unwrap();
        let mut f = PostFilter::new();
        // For the first T-2 = 13 samples the deepest tap (T+2 = 17) and
        // all shallower taps read the still-zero history, so y == x.
        for k in 0..13 {
            let x = (k as f64) + 1.0;
            assert_eq!(f.step(x, &c), x, "sample {k} should pass through");
        }
    }

    /// The comb recurrence matches a hand expansion once the history has
    /// filled. Use a period of 15 and a known input so we can predict
    /// which past outputs feed back.
    #[test]
    fn comb_recurrence_matches_hand_expansion() {
        let c = PostFilterCoeffs::new(15, 7, 0).unwrap();
        let mut f = PostFilter::new();
        // Feed an impulse then zeros; track outputs explicitly.
        let mut ys = Vec::new();
        // Impulse at n=0.
        ys.push(f.step(1.0, &c));
        // Zeros for the next 20 samples.
        for _ in 0..20 {
            ys.push(f.step(0.0, &c));
        }
        // y(0) = 1 (all past zero). y(1..12) = 0. At n = T-2 = 13 the
        // g2 tap reaches y(13-17)=y(-4)->0 still; the first non-zero
        // feedback is when a tap reaches y(0)=1. y(n-T+2)=y(0) => n=13;
        // so y(13) = a2*y(0) (the y(n-T+2) half of the g2 pair).
        assert_eq!(ys[0], 1.0);
        for (n, y) in ys.iter().enumerate().take(13).skip(1) {
            assert_eq!(*y, 0.0, "y({n}) should be zero before feedback");
        }
        // n=13: y(n-T+2) = y(0) = 1 contributes a2; its mirror y(n-T-2)
        // = y(-4) = 0.
        assert_eq!(ys[13], c.a2 * 1.0);
        // n=14: y(n-T+1) = y(0) contributes a1.
        assert_eq!(ys[14], c.a1 * 1.0);
        // n=15: y(n-T) = y(0) contributes a0.
        assert_eq!(ys[15], c.a0 * 1.0);
        // n=16: y(n-T-1) = y(0) contributes a1 (the mirror half).
        assert_eq!(ys[16], c.a1 * 1.0);
        // n=17: y(n-T-2) = y(0) contributes a2 (the mirror half).
        assert_eq!(ys[17], c.a2 * 1.0);
        // n=18: no tap reaches y(0) any more, and y(1..) were zero, so 0.
        assert_eq!(ys[18], 0.0);
    }

    /// The g1/g2 pairs are symmetric: an impulse produces a response
    /// that is mirror-symmetric about the centre tap at lag T.
    #[test]
    fn impulse_response_is_symmetric_about_period() {
        let c = PostFilterCoeffs::new(20, 5, 0).unwrap();
        let mut f = PostFilter::new();
        let mut ys = vec![f.step(1.0, &c)];
        for _ in 0..30 {
            ys.push(f.step(0.0, &c));
        }
        // Centre is at n = T = 20. Pairs at ±1, ±2 must match.
        assert_eq!(ys[19], ys[21], "g1 pair not symmetric");
        assert_eq!(ys[18], ys[22], "g2 pair not symmetric");
    }

    /// History carries across blocks: filtering a split stream equals
    /// filtering it whole.
    #[test]
    fn history_carries_across_blocks() {
        let c = PostFilterCoeffs::new(15, 6, 0).unwrap();
        let stream: Vec<f64> = (0..40).map(|k| ((k * 7) % 11) as f64 - 5.0).collect();

        let mut whole = PostFilter::new();
        let mut ref_out = stream.clone();
        whole.process_in_place(&mut ref_out, &c);

        let mut split = PostFilter::new();
        let mut a = stream[..18].to_vec();
        let mut b = stream[18..].to_vec();
        split.process_in_place(&mut a, &c);
        split.process_in_place(&mut b, &c);
        for (i, v) in a.iter().chain(b.iter()).enumerate() {
            assert_eq!(*v, ref_out[i], "block-split mismatch at {i}");
        }
    }

    /// `process` writes the output buffer and matches in-place.
    #[test]
    fn process_writes_output_buffer() {
        let c = PostFilterCoeffs::new(15, 3, 1).unwrap();
        let input: Vec<f64> = (0..30).map(|k| (k as f64).sin()).collect();
        let mut out = vec![0.0; 30];
        let mut f = PostFilter::new();
        let n = f.process(&input, &mut out, &c).unwrap();
        assert_eq!(n, 30);

        let mut g = PostFilter::new();
        let mut ref_out = input.clone();
        g.process_in_place(&mut ref_out, &c);
        assert_eq!(out, ref_out);
    }

    /// `process` accepts an over-long output buffer.
    #[test]
    fn process_accepts_longer_output() {
        let c = PostFilterCoeffs::new(15, 0, 0).unwrap();
        let input = [1.0_f64, 2.0, 3.0];
        let mut out = [9.0_f64; 6];
        let mut f = PostFilter::new();
        let n = f.process(&input, &mut out, &c).unwrap();
        assert_eq!(n, 3);
        // First 3 pass through (period 15, fresh history), trailing kept.
        assert_eq!(&out[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&out[3..], &[9.0, 9.0, 9.0]);
    }

    /// `process` rejects a short output buffer and leaves state intact.
    #[test]
    fn process_rejects_short_output() {
        let c = PostFilterCoeffs::new(15, 0, 0).unwrap();
        let input = [1.0_f64, 2.0, 3.0];
        let mut out = [0.0_f64; 2];
        let mut f = PostFilter::new();
        let err = f.process(&input, &mut out, &c).unwrap_err();
        assert_eq!(
            err,
            PostFilterError::OutputBufferTooSmall {
                input_len: 3,
                output_len: 2,
            }
        );
        // History untouched: a fresh re-run still passes the first
        // sample through unchanged.
        let mut g = PostFilter::new();
        assert_eq!(g.step(1.0, &c), f.step(1.0, &c));
    }

    /// `reset` zeroes the history.
    #[test]
    fn reset_zeroes_history() {
        let c = PostFilterCoeffs::new(15, 7, 0).unwrap();
        let mut f = PostFilter::new();
        for k in 0..40 {
            let _ = f.step((k as f64) + 1.0, &c);
        }
        f.reset();
        // After reset the first sample passes through (all past zero).
        assert_eq!(f.step(2.5, &c), 2.5);
    }

    /// The gain-transition crossfade starts at the old output (W(0)^2≈0)
    /// and ends at the new output (W(overlap-1)^2≈1).
    #[test]
    fn gain_transition_endpoints() {
        let overlap = 16usize;
        // Distinct gains so old/new branches differ.
        let old = PostFilterCoeffs::new(15, 1, 0).unwrap();
        let new = PostFilterCoeffs::new(15, 7, 0).unwrap();

        // Prime a shared history with some non-zero output so the comb
        // feedback is exercised at the transition.
        let mut prime = PostFilter::new();
        for k in 0..40 {
            let _ = prime.step(((k % 5) as f64) - 2.0, &old);
        }

        let input: Vec<f64> = (0..overlap).map(|k| (k as f64).cos()).collect();

        // Reference: pure old / pure new outputs from copies of the
        // primed filter.
        let mut f_old = prime.clone();
        let mut old_out = input.clone();
        f_old.process_in_place(&mut old_out, &old);
        let mut f_new = prime.clone();
        let mut new_out = input.clone();
        f_new.process_in_place(&mut new_out, &new);

        let mut f = prime.clone();
        let mut out = vec![0.0; overlap];
        f.process_gain_transition(&input, &mut out, &old, &new, overlap)
            .unwrap();

        // At n=0 the window square is W(0)^2; the crossfade weight on the
        // new branch equals W(0)^2 and on the old branch 1-W(0)^2. Since
        // the first sample reads the same (primed) history in all three
        // runs, the crossfade is an exact convex mix of old_out[0] and
        // new_out[0].
        let w0 = {
            let w = celt_mdct_window::window_tap(0, overlap).unwrap();
            w * w
        };
        let expect0 = (1.0 - w0) * old_out[0] + w0 * new_out[0];
        assert!((out[0] - expect0).abs() < 1e-12, "n=0 mix wrong");
        // The transition weight rises monotonically to ~1 at the end.
        let wlast = {
            let w = celt_mdct_window::window_tap(overlap - 1, overlap).unwrap();
            w * w
        };
        assert!(wlast > 0.99, "window square should approach 1 at the end");
    }

    /// `process_gain_transition` with old == new equals a plain
    /// fixed-coefficient run (the crossfade collapses to identity).
    #[test]
    fn gain_transition_identity_when_same_params() {
        let overlap = 16usize;
        let c = PostFilterCoeffs::new(15, 4, 0).unwrap();
        let input: Vec<f64> = (0..overlap).map(|k| (k as f64) * 0.1 - 0.5).collect();

        let mut a = PostFilter::new();
        let mut out_a = vec![0.0; overlap];
        a.process_gain_transition(&input, &mut out_a, &c, &c, overlap)
            .unwrap();

        let mut b = PostFilter::new();
        let mut out_b = input.clone();
        b.process_in_place(&mut out_b, &c);

        for (i, (x, y)) in out_a.iter().zip(out_b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-12, "transition!=plain at {i}");
        }
    }

    /// Length-mismatched transition is rejected.
    #[test]
    fn gain_transition_length_mismatch_rejected() {
        let c = PostFilterCoeffs::new(15, 0, 0).unwrap();
        let mut f = PostFilter::new();
        let input = [1.0; 8];
        let mut out = [0.0; 8];
        let err = f
            .process_gain_transition(&input, &mut out, &c, &c, 16)
            .unwrap_err();
        assert_eq!(
            err,
            PostFilterError::TransitionLengthMismatch {
                expected: 16,
                provided: 8,
            }
        );
    }

    /// The standalone `crossfade_transition` mixes two output slices.
    #[test]
    fn crossfade_helper_mixes() {
        let overlap = 8usize;
        let old_out = vec![1.0; overlap];
        let new_out = vec![3.0; overlap];
        let mut out = vec![0.0; overlap];
        crossfade_transition(&old_out, &new_out, &mut out, overlap).unwrap();
        for (n, v) in out.iter().enumerate() {
            let w = celt_mdct_window::window_tap(n, overlap).unwrap();
            let w2 = w * w;
            let expect = (1.0 - w2) * 1.0 + w2 * 3.0;
            assert!((v - expect).abs() < 1e-12, "mix wrong at {n}");
            // Convex combination stays within [1, 3].
            assert!(*v >= 1.0 - 1e-12 && *v <= 3.0 + 1e-12);
        }
    }

    /// `crossfade_transition` rejects mismatched slice lengths.
    #[test]
    fn crossfade_helper_rejects_mismatch() {
        let mut out = vec![0.0; 8];
        let err = crossfade_transition(&[1.0; 8], &[2.0; 4], &mut out, 8).unwrap_err();
        assert_eq!(
            err,
            PostFilterError::TransitionLengthMismatch {
                expected: 8,
                provided: 4,
            }
        );
    }

    /// `crossfade_transition` rejects a short output buffer.
    #[test]
    fn crossfade_helper_rejects_short_output() {
        let mut out = vec![0.0; 4];
        let err = crossfade_transition(&[1.0; 8], &[2.0; 8], &mut out, 8).unwrap_err();
        assert_eq!(
            err,
            PostFilterError::OutputBufferTooSmall {
                input_len: 8,
                output_len: 4,
            }
        );
    }

    /// Constants match the §4.3.7.1 narrative.
    #[test]
    fn constants_match_rfc() {
        assert_eq!(POST_FILTER_MIN_PERIOD, 15);
        assert_eq!(POST_FILTER_MAX_PERIOD, 1022);
        assert_eq!(POST_FILTER_TAPSET_COUNT, 3);
        assert_eq!(POST_FILTER_GAIN_NUMERATOR, 3);
        assert_eq!(POST_FILTER_GAIN_DENOMINATOR, 32);
        assert_eq!(POST_FILTER_GAIN_INDEX_MAX, 7);
    }

    /// Every error variant renders an informative `Display`.
    #[test]
    fn error_display() {
        assert!(PostFilterError::TapsetOutOfRange { tapset: 9 }
            .to_string()
            .contains("tapset"));
        assert!(PostFilterError::PeriodOutOfRange { period: 5 }
            .to_string()
            .contains("period"));
        assert!(PostFilterError::GainIndexOutOfRange { gain_index: 9 }
            .to_string()
            .contains("gain"));
        assert!(PostFilterError::OutputBufferTooSmall {
            input_len: 4,
            output_len: 1
        }
        .to_string()
        .contains("need 4"));
        assert!(PostFilterError::TransitionLengthMismatch {
            expected: 16,
            provided: 8
        }
        .to_string()
        .contains("expected 16"));
        let w = PostFilterError::Window(MdctWindowError::ZeroLength);
        assert!(w.to_string().contains("window"));
    }

    /// The deepest history index the filter ever reads (`T + 2` at the
    /// max period) fits inside the ring capacity.
    #[test]
    fn max_reach_fits_history() {
        let c = PostFilterCoeffs::new(POST_FILTER_MAX_PERIOD, 7, 0).unwrap();
        assert_eq!(c.history_len(), PostFilter::HIST_CAP);
        // Exercise a step at the max period without panicking.
        let mut f = PostFilter::new();
        let _ = f.step(1.0, &c);
    }
}
