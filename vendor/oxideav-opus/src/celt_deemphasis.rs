//! CELT §4.3.7.2 de-emphasis filter
//! (RFC 6716 §4.3.7.2, p. 122).
//!
//! The final stage of the CELT decode pipeline, applied after the
//! inverse MDCT (with weighted overlap-add, §4.3.7) and the §4.3.7.1
//! pitch post-filter, is the *de-emphasis* filter. RFC 6716 §4.3.7.2
//! (p. 122) states it as the inverse of the pre-emphasis filter the
//! encoder applies:
//!
//! ```text
//!     1            1
//!    ---- = ---------------
//!    A(z)                -1
//!           1 - alpha_p*z
//! ```
//!
//! with `alpha_p = 0.8500061035`.
//!
//! ## Time-domain recurrence
//!
//! The pre-emphasis the encoder applied is the FIR
//! `A(z) = 1 - alpha_p * z^-1`, i.e. `x(n) = s(n) - alpha_p*s(n-1)`
//! where `s` is the original signal and `x` is the pre-emphasised
//! signal that (after the MDCT round-trip and the rest of the codec)
//! arrives at this stage. De-emphasis inverts that single pole. With
//! `S(z) = X(z) / A(z)` and `A(z) = 1 - alpha_p*z^-1`:
//!
//! ```text
//!   S(z) * (1 - alpha_p*z^-1) = X(z)
//!   s(n) - alpha_p*s(n-1)      = x(n)
//!   s(n)                       = x(n) + alpha_p * s(n-1)
//! ```
//!
//! so the de-emphasis output is the one-pole IIR recurrence
//!
//! ```text
//!   y(n) = x(n) + alpha_p * y(n-1)
//! ```
//!
//! The pole `alpha_p ≈ 0.85` is just inside the unit circle, so the
//! filter is stable; its single state element `y(n-1)` (the "memory")
//! carries across frame boundaries — the recurrence is continuous over
//! the whole decoded stream, not reset per frame. Per-channel decoders
//! each carry their own independent memory.
//!
//! This module owns only the de-emphasis recurrence and its state. The
//! inverse MDCT (§4.3.7), the overlap-add window (§4.3.7,
//! [`crate::celt_mdct_window`]), and the §4.3.7.1 pitch post-filter that
//! feed this stage run at their own consumer sites; this filter is the
//! last thing applied before the time-domain samples leave the CELT
//! decoder.
//!
//! ## Provenance
//!
//! De-emphasis transfer function + `alpha_p` constant: RFC 6716
//! §4.3.7.2 (p. 122), reproduced from
//! `docs/audio/opus/rfc6716-opus.txt`. The recurrence is the
//! textbook inverse of the stated one-pole transfer function. No
//! external library source was consulted; the filter and its single
//! coefficient are stated directly in the standards-track text.

/// The §4.3.7.2 de-emphasis (= inverse pre-emphasis) coefficient
/// `alpha_p = 0.8500061035` (RFC 6716 §4.3.7.2, p. 122).
///
/// The pole of the one-pole IIR `y(n) = x(n) + alpha_p*y(n-1)`. Stated
/// to 10 decimal places in the standards-track text; this is the exact
/// decimal the RFC gives, not a rounding of a binary fraction.
pub const DEEMPHASIS_ALPHA_P: f64 = 0.850_006_103_5;

/// A single-channel §4.3.7.2 de-emphasis filter carrying its one-pole
/// state across frame boundaries.
///
/// The CELT decoder applies de-emphasis continuously over the whole
/// decoded stream: the filter memory `y(n-1)` is *not* reset at frame
/// boundaries (only at a full decoder reset / mode transition that
/// resets the CELT state, per RFC 6716 §4.5.2). Construct one
/// [`DeemphasisFilter`] per channel and reuse it across frames.
///
/// ```
/// use oxideav_opus::celt_deemphasis::DeemphasisFilter;
/// let mut f = DeemphasisFilter::new();
/// // y(0) = x(0) + alpha*0 = x(0)
/// assert_eq!(f.step(1.0), 1.0);
/// // y(1) = x(1) + alpha*y(0) = 0 + 0.8500061035
/// assert_eq!(f.step(0.0), 0.850_006_103_5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeemphasisFilter {
    /// The single one-pole memory element `y(n-1)`.
    mem: f64,
}

impl Default for DeemphasisFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl DeemphasisFilter {
    /// Create a fresh filter with zeroed memory (`y(-1) = 0`), the
    /// state the CELT decoder starts each stream / post-reset run with.
    #[must_use]
    pub fn new() -> Self {
        Self { mem: 0.0 }
    }

    /// Create a filter seeded with a given memory value `y(n-1)`.
    ///
    /// Used when resuming a stream whose de-emphasis state was carried
    /// over from a prior decode (e.g. across a redundant-frame splice
    /// that preserves the CELT state).
    #[must_use]
    pub fn with_memory(mem: f64) -> Self {
        Self { mem }
    }

    /// The current one-pole memory `y(n-1)`.
    #[must_use]
    pub fn memory(&self) -> f64 {
        self.mem
    }

    /// Reset the filter memory to zero (`y(-1) = 0`), as on a §4.5.2
    /// CELT state reset.
    pub fn reset(&mut self) {
        self.mem = 0.0;
    }

    /// Apply the §4.3.7.2 recurrence to one input sample and advance the
    /// state: returns `y(n) = x(n) + alpha_p * y(n-1)` and stores it as
    /// the new `y(n-1)`.
    #[must_use]
    pub fn step(&mut self, x: f64) -> f64 {
        let y = x + DEEMPHASIS_ALPHA_P * self.mem;
        self.mem = y;
        y
    }

    /// Apply the §4.3.7.2 de-emphasis to a block of samples in place,
    /// updating the carried memory so the next call continues the same
    /// recurrence. Equivalent to calling [`Self::step`] on each sample
    /// left-to-right.
    pub fn process_in_place(&mut self, samples: &mut [f64]) {
        for x in samples.iter_mut() {
            *x = self.step(*x);
        }
    }

    /// Apply the §4.3.7.2 de-emphasis to `input`, writing the result to
    /// `output`, and update the carried memory.
    ///
    /// Returns the number of samples written (`input.len()`), or
    /// [`DeemphasisError::OutputBufferTooSmall`] if `output` is shorter
    /// than `input`. On error the filter state is left unchanged.
    pub fn process(&mut self, input: &[f64], output: &mut [f64]) -> Result<usize, DeemphasisError> {
        if output.len() < input.len() {
            return Err(DeemphasisError::OutputBufferTooSmall {
                input_len: input.len(),
                output_len: output.len(),
            });
        }
        for (i, &x) in input.iter().enumerate() {
            output[i] = self.step(x);
        }
        Ok(input.len())
    }
}

/// Errors returnable by the §4.3.7.2 de-emphasis helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeemphasisError {
    /// The output buffer passed to [`DeemphasisFilter::process`] is
    /// shorter than the input.
    OutputBufferTooSmall {
        /// Number of input samples to de-emphasise.
        input_len: usize,
        /// Length of the supplied output buffer.
        output_len: usize,
    },
}

impl core::fmt::Display for DeemphasisError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DeemphasisError::OutputBufferTooSmall {
                input_len,
                output_len,
            } => write!(
                f,
                "de-emphasis output buffer too small: need {input_len} samples, got {output_len}"
            ),
        }
    }
}

impl std::error::Error for DeemphasisError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The §4.3.7.2 coefficient is the exact decimal the RFC prints.
    #[test]
    fn alpha_p_constant_matches_rfc() {
        assert_eq!(DEEMPHASIS_ALPHA_P, 0.850_006_103_5);
        // Just inside the unit circle ⇒ stable one-pole IIR.
        const { assert!(DEEMPHASIS_ALPHA_P > 0.0 && DEEMPHASIS_ALPHA_P < 1.0) };
    }

    /// Fresh filter starts with zeroed memory.
    #[test]
    fn new_has_zero_memory() {
        let f = DeemphasisFilter::new();
        assert_eq!(f.memory(), 0.0);
        assert_eq!(DeemphasisFilter::default(), f);
    }

    /// `y(0) = x(0) + alpha*0 = x(0)` for a fresh filter.
    #[test]
    fn first_sample_passes_through() {
        let mut f = DeemphasisFilter::new();
        assert_eq!(f.step(3.5), 3.5);
        assert_eq!(f.memory(), 3.5);
    }

    /// `y(n) = x(n) + alpha_p*y(n-1)` worked out by hand for a short run.
    #[test]
    fn recurrence_matches_hand_computation() {
        let a = DEEMPHASIS_ALPHA_P;
        let mut f = DeemphasisFilter::new();
        let y0 = f.step(1.0);
        let y1 = f.step(2.0);
        let y2 = f.step(-1.0);
        assert_eq!(y0, 1.0);
        assert_eq!(y1, 2.0 + a * 1.0);
        assert_eq!(y2, -1.0 + a * y1);
        assert_eq!(f.memory(), y2);
    }

    /// A constant input converges toward the DC gain `1/(1 - alpha_p)`.
    #[test]
    fn constant_input_converges_to_dc_gain() {
        let a = DEEMPHASIS_ALPHA_P;
        let dc_gain = 1.0 / (1.0 - a);
        let mut f = DeemphasisFilter::new();
        let mut y = 0.0;
        for _ in 0..10_000 {
            y = f.step(1.0);
        }
        assert!((y - dc_gain).abs() < 1e-6, "y={y} dc_gain={dc_gain}");
    }

    /// De-emphasis exactly inverts the encoder's pre-emphasis
    /// `x(n) = s(n) - alpha_p*s(n-1)` (a round-trip property), provided
    /// both filters start from the same zero state.
    #[test]
    fn inverts_pre_emphasis() {
        let a = DEEMPHASIS_ALPHA_P;
        let s = [0.3_f64, -0.7, 1.2, 0.0, -0.4, 0.9, 0.05, -1.1];
        // Pre-emphasise (encoder side, FIR A(z) = 1 - alpha*z^-1).
        let mut prev = 0.0_f64;
        let x: Vec<f64> = s
            .iter()
            .map(|&sn| {
                let xn = sn - a * prev;
                prev = sn;
                xn
            })
            .collect();
        // De-emphasise (decoder side, this module).
        let mut f = DeemphasisFilter::new();
        for (i, &xn) in x.iter().enumerate() {
            let yn = f.step(xn);
            assert!(
                (yn - s[i]).abs() < 1e-12,
                "round-trip mismatch at {i}: got {yn}, want {}",
                s[i]
            );
        }
    }

    /// The memory carries across calls: splitting a stream into two
    /// blocks and filtering each with the same filter equals filtering
    /// the whole stream at once.
    #[test]
    fn memory_carries_across_blocks() {
        let stream = [0.5_f64, -0.2, 0.9, 1.0, -0.3, 0.1, 0.7, -0.8];
        // Whole-stream reference.
        let mut whole = DeemphasisFilter::new();
        let mut ref_out = stream;
        whole.process_in_place(&mut ref_out);
        // Split into two blocks sharing one filter.
        let mut split = DeemphasisFilter::new();
        let mut a = stream[..3].to_vec();
        let mut b = stream[3..].to_vec();
        split.process_in_place(&mut a);
        split.process_in_place(&mut b);
        for (i, v) in a.iter().chain(b.iter()).enumerate() {
            assert_eq!(*v, ref_out[i], "block-split mismatch at {i}");
        }
    }

    /// `with_memory` seeds the recurrence: `y(0) = x(0) + alpha*seed`.
    #[test]
    fn with_memory_seeds_recurrence() {
        let a = DEEMPHASIS_ALPHA_P;
        let mut f = DeemphasisFilter::with_memory(2.0);
        assert_eq!(f.memory(), 2.0);
        assert_eq!(f.step(0.0), a * 2.0);
    }

    /// `reset` zeroes the memory.
    #[test]
    fn reset_zeroes_memory() {
        let mut f = DeemphasisFilter::new();
        let _ = f.step(5.0);
        assert_ne!(f.memory(), 0.0);
        f.reset();
        assert_eq!(f.memory(), 0.0);
    }

    /// `process_in_place` equals stepping each sample.
    #[test]
    fn process_in_place_equals_stepping() {
        let input = [1.0_f64, 2.0, 3.0, -1.0, 0.5];
        let mut by_step = DeemphasisFilter::new();
        let stepped: Vec<f64> = input.iter().map(|&x| by_step.step(x)).collect();
        let mut by_block = DeemphasisFilter::new();
        let mut block = input;
        by_block.process_in_place(&mut block);
        assert_eq!(block.to_vec(), stepped);
        assert_eq!(by_step.memory(), by_block.memory());
    }

    /// `process` writes into the output buffer and matches in-place.
    #[test]
    fn process_writes_output_buffer() {
        let input = [0.1_f64, -0.2, 0.3, 0.4];
        let mut out = [0.0_f64; 4];
        let mut f = DeemphasisFilter::new();
        let n = f.process(&input, &mut out).unwrap();
        assert_eq!(n, 4);

        let mut g = DeemphasisFilter::new();
        let mut ref_out = input;
        g.process_in_place(&mut ref_out);
        assert_eq!(out, ref_out);
        assert_eq!(f.memory(), g.memory());
    }

    /// `process` accepts an over-long output buffer (only the leading
    /// `input.len()` slots are written).
    #[test]
    fn process_accepts_longer_output() {
        let input = [1.0_f64, 2.0];
        let mut out = [9.0_f64; 5];
        let mut f = DeemphasisFilter::new();
        let n = f.process(&input, &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0 + DEEMPHASIS_ALPHA_P);
        // Trailing slots untouched.
        assert_eq!(&out[2..], &[9.0, 9.0, 9.0]);
    }

    /// A short output buffer is rejected and the filter state is
    /// left unchanged.
    #[test]
    fn process_rejects_short_output() {
        let input = [1.0_f64, 2.0, 3.0];
        let mut out = [0.0_f64; 2];
        let mut f = DeemphasisFilter::new();
        let before = f.memory();
        let err = f.process(&input, &mut out).unwrap_err();
        assert_eq!(
            err,
            DeemphasisError::OutputBufferTooSmall {
                input_len: 3,
                output_len: 2,
            }
        );
        assert_eq!(f.memory(), before, "state must be unchanged on error");
    }

    /// An empty input is a no-op that leaves the state unchanged.
    #[test]
    fn empty_input_is_noop() {
        let mut f = DeemphasisFilter::with_memory(1.5);
        let mut empty: [f64; 0] = [];
        f.process_in_place(&mut empty);
        assert_eq!(f.memory(), 1.5);
        let mut out: [f64; 0] = [];
        assert_eq!(f.process(&[], &mut out).unwrap(), 0);
        assert_eq!(f.memory(), 1.5);
    }

    /// Error `Display` is informative.
    #[test]
    fn error_display() {
        let s = DeemphasisError::OutputBufferTooSmall {
            input_len: 4,
            output_len: 1,
        }
        .to_string();
        assert!(s.contains("need 4"));
        assert!(s.contains("got 1"));
    }
}
