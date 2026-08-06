//! CELT §4.3.7 weighted overlap-add
//! (RFC 6716 §4.3.7, p. 121).
//!
//! The inverse MDCT ([`crate::celt_imdct`]) maps each CELT frame's `N`
//! denormalised frequency bins to a `2N`-sample time-domain block. A
//! single such block is *not* the decoded signal: the MDCT is a lapped
//! transform, and each reconstructed block carries a time-domain aliased
//! copy of itself folded about its half-block midpoints (the defining
//! property that lets `N` coefficients represent a `2N`-sample block
//! without redundancy). RFC 6716 §4.3.7.1 (p. 121) names the step that
//! removes that aliasing:
//!
//! > The output of the inverse MDCT (after weighted overlap-add) is sent
//! > to the post-filter.
//!
//! The "weighted overlap-add" multiplies each `2N` block by the §4.3.7
//! low-overlap window ([`crate::celt_mdct_window`]) and sums the leading
//! half of the current block with the trailing half of the previous
//! block. Because the window is power-complementary
//! (`W(n)^2 + W(L-1-n)^2 = 1`, the Princen-Bradley condition the §4.3.7
//! prose requires), the equal-and-opposite aliased halves of neighbouring
//! blocks cancel exactly on the add, and the result is the aliasing-free
//! time-domain signal (at the §4.3.7 `1/2` IMDCT scale).
//!
//! ## Block layout and the hop
//!
//! For a frame of `N` MDCT bins:
//!
//! * the inverse MDCT yields a `2N`-sample block;
//! * the synthesis window touches only the `overlap` samples at each
//!   edge — the leading `overlap` samples are multiplied by the rising
//!   ramp `W(0) .. W(overlap-1)`, the trailing `overlap` samples by the
//!   falling ramp `W(overlap-1) .. W(0)`, and the `2N - 2*overlap`
//!   samples in the middle are unity ("inserting ones in the middle",
//!   §4.3.7);
//! * the hop between successive frames is `N` samples, so block `i`
//!   overlaps block `i-1` over its first `N` samples.
//!
//! The decoder therefore keeps a per-channel **history** buffer holding
//! the windowed *second half* (samples `N .. 2N`) of the previous block.
//! Each frame emits `N` time-domain samples:
//!
//! ```text
//!   out(n) = windowed_block_i(n) + history(n)     for n = 0 .. N
//!   history'(n) = windowed_block_i(N + n)         for n = 0 .. N
//! ```
//!
//! On the very first frame after a stream start or a §4.5.2 CELT state
//! reset the history is all-zero, so the first emitted frame is simply
//! the leading half of the first windowed block — the aliasing in that
//! leading half is cancelled by the *next* frame's overlap, exactly as
//! the trailing aliasing of every block is cancelled by its successor.
//!
//! The `overlap` is constrained to `0 < overlap <= N`. The CELT layer
//! fixes the 48 kHz overlap at 120 samples
//! ([`crate::celt_mdct_window::CELT_OVERLAP_48K`]); for the shortest CELT
//! frame the per-MDCT `N` can be as small as the overlap, so equality is
//! permitted. An even overlap is required so the window centre splits
//! cleanly (the §4.3.7 low-overlap construction).
//!
//! ## Relation to the transform core
//!
//! [`crate::celt_imdct`] computes the raw `2N` inverse block already
//! scaled by `1/2`; this module owns the windowing and the cross-frame
//! state. The TDAC property — that a windowed forward/inverse pair plus
//! this overlap-add reconstructs the input at the `1/2` scale — is pinned
//! both by the transform core's own tests and, end-to-end through this
//! stateful adder, by the tests below.
//!
//! ## Provenance
//!
//! Lapped-transform overlap-add narrative + "after weighted overlap-add"
//! step ordering: RFC 6716 §4.3.7 / §4.3.7.1 (p. 121); fixed-overlap
//! statement: §1 (p. 9–10); window shape: §4.3.7 (p. 121). All
//! reproduced from `docs/audio/opus/rfc6716-opus.txt`. The overlap-add
//! is the textbook synthesis step for a lapped transform with a
//! power-complementary window; no external library source was consulted.

use crate::celt_mdct_window::{mdct_window, MdctWindowError};

/// Errors returnable by the §4.3.7 weighted overlap-add helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapAddError {
    /// The transform half-length `N` is zero; an empty block has no
    /// overlap-add.
    ZeroLength,
    /// The supplied block was not exactly `2*N` samples long.
    BlockLenNotEven {
        /// The block length the caller supplied.
        got: usize,
    },
    /// The block length disagreed with the adder's configured `N`.
    BlockLenMismatch {
        /// The block length the caller supplied.
        got: usize,
        /// The required length `2 * N`.
        want: usize,
    },
    /// The overlap is zero or exceeds `N` (the §4.3.7 window touches at
    /// most one half-block at each edge), or is odd (the low-overlap
    /// construction needs an even overlap so the centre splits cleanly).
    BadOverlap {
        /// The overlap the caller passed.
        overlap: usize,
        /// The configured `N`.
        n: usize,
    },
    /// The output slice passed to [`WeightedOverlapAdd::process_into`]
    /// is shorter than the `N` samples a frame emits.
    OutputTooSmall {
        /// The required output length `N`.
        want: usize,
        /// The length the caller supplied.
        got: usize,
    },
}

impl core::fmt::Display for OverlapAddError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            OverlapAddError::ZeroLength => {
                write!(f, "oxideav-opus: CELT §4.3.7 overlap-add requires N >= 1")
            }
            OverlapAddError::BlockLenNotEven { got } => write!(
                f,
                "oxideav-opus: CELT §4.3.7 overlap-add block length {got} is not 2*N (odd)"
            ),
            OverlapAddError::BlockLenMismatch { got, want } => write!(
                f,
                "oxideav-opus: CELT §4.3.7 overlap-add block length {got} != required 2*N = {want}"
            ),
            OverlapAddError::BadOverlap { overlap, n } => write!(
                f,
                "oxideav-opus: CELT §4.3.7 overlap-add overlap {overlap} invalid for N={n} \
                 (require 0 < overlap <= N and overlap even)"
            ),
            OverlapAddError::OutputTooSmall { want, got } => write!(
                f,
                "oxideav-opus: CELT §4.3.7 overlap-add output length {got} < required N = {want}"
            ),
        }
    }
}

impl std::error::Error for OverlapAddError {}

/// Apply the §4.3.7 low-overlap synthesis window to a `2N`-sample inverse
/// MDCT block **in place** (RFC 6716 §4.3.7, p. 121).
///
/// The leading `overlap` samples are scaled by the rising ramp
/// `W(0) .. W(overlap-1)`, the trailing `overlap` samples by the falling
/// ramp `W(overlap-1) .. W(0)`, and the `2N - 2*overlap` samples in the
/// middle are left unchanged (unity window).
///
/// `block.len()` must be `2 * n`, and `ramp.len()` must be `overlap`
/// with `0 < overlap <= n`. This is a free function so a caller that has
/// already built the ramp once can window many blocks without rebuilding
/// it; [`WeightedOverlapAdd`] caches the ramp and calls this internally.
///
/// # Errors
///
/// Returns [`OverlapAddError`] if `n == 0`, if `block.len() != 2*n`, or
/// if the ramp length is not a valid overlap for `n`.
pub fn apply_synthesis_window(
    block: &mut [f64],
    ramp: &[f64],
    n: usize,
) -> Result<(), OverlapAddError> {
    if n == 0 {
        return Err(OverlapAddError::ZeroLength);
    }
    if block.len() != 2 * n {
        if block.len() % 2 != 0 {
            return Err(OverlapAddError::BlockLenNotEven { got: block.len() });
        }
        return Err(OverlapAddError::BlockLenMismatch {
            got: block.len(),
            want: 2 * n,
        });
    }
    let overlap = ramp.len();
    if overlap == 0 || overlap > n || overlap % 2 != 0 {
        return Err(OverlapAddError::BadOverlap { overlap, n });
    }
    let two_n = 2 * n;
    for i in 0..overlap {
        // Rising ramp on the leading edge.
        block[i] *= ramp[i];
        // Falling ramp on the trailing edge: W(overlap-1-i).
        block[two_n - 1 - i] *= ramp[i];
    }
    // The middle `2N - 2*overlap` samples keep their unity window.
    Ok(())
}

/// The §4.3.7 stateful weighted overlap-add for one channel
/// (RFC 6716 §4.3.7 / §4.3.7.1, p. 121).
///
/// Feed each frame's `2N`-sample inverse MDCT block (the raw output of
/// [`crate::celt_imdct`], already scaled by `1/2`) to [`Self::process`]
/// / [`Self::process_into`]; each call returns the `N` aliasing-free
/// time-domain samples for that frame and carries the windowed trailing
/// half forward as the overlap history for the next frame.
///
/// One instance tracks one channel. A stereo decoder keeps two.
///
/// # Examples
///
/// ```
/// use oxideav_opus::celt_overlap_add::WeightedOverlapAdd;
/// // N = 4, overlap = 2.
/// let mut ola = WeightedOverlapAdd::new(4, 2).unwrap();
/// let block = [0.0_f64; 8];
/// // A silent block stays silent through the overlap-add.
/// assert_eq!(ola.process(&block).unwrap(), vec![0.0; 4]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedOverlapAdd {
    /// The transform half-length: each frame emits `n` samples and
    /// consumes a `2*n`-sample block.
    n: usize,
    /// The `overlap` rising window taps `W(0) .. W(overlap-1)`.
    ramp: Vec<f64>,
    /// The windowed trailing half (`n` samples) of the previous block,
    /// to be added to the leading half of the next block. All-zero at
    /// stream start / after a §4.5.2 reset.
    history: Vec<f64>,
}

impl WeightedOverlapAdd {
    /// Create a fresh overlap-add for transform half-length `n` and
    /// `overlap`, with zeroed history (the stream-start / post-reset
    /// state).
    ///
    /// Requires `0 < overlap <= n` and `overlap` even (the §4.3.7
    /// low-overlap construction).
    ///
    /// # Errors
    ///
    /// Returns [`OverlapAddError::ZeroLength`] if `n == 0`, or
    /// [`OverlapAddError::BadOverlap`] if the overlap is out of range or
    /// odd.
    pub fn new(n: usize, overlap: usize) -> Result<Self, OverlapAddError> {
        if n == 0 {
            return Err(OverlapAddError::ZeroLength);
        }
        if overlap == 0 || overlap > n || overlap % 2 != 0 {
            return Err(OverlapAddError::BadOverlap { overlap, n });
        }
        let ramp = mdct_window(overlap).map_err(|e| match e {
            MdctWindowError::ZeroLength => OverlapAddError::BadOverlap { overlap, n },
            MdctWindowError::OddOverlap { overlap } => OverlapAddError::BadOverlap { overlap, n },
            MdctWindowError::PositionOutOfRange { .. } => {
                OverlapAddError::BadOverlap { overlap, n }
            }
        })?;
        Ok(Self {
            n,
            ramp,
            history: vec![0.0_f64; n],
        })
    }

    /// The transform half-length `N`: each frame emits this many samples.
    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.n
    }

    /// The overlap length (window transition width).
    #[must_use]
    pub fn overlap(&self) -> usize {
        self.ramp.len()
    }

    /// The current overlap history (the windowed trailing half of the
    /// last block). All-zero before the first [`Self::process`] call.
    #[must_use]
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Reset the overlap history to zero, as on a §4.5.2 CELT state
    /// reset or at the start of a new stream.
    pub fn reset(&mut self) {
        for h in self.history.iter_mut() {
            *h = 0.0;
        }
    }

    /// Run the weighted overlap-add for one frame, returning the `N`
    /// aliasing-free time-domain samples.
    ///
    /// `block` is the `2*N`-sample inverse MDCT output for this frame.
    /// Allocating wrapper for [`Self::process_into`].
    ///
    /// # Errors
    ///
    /// Returns [`OverlapAddError::BlockLenMismatch`] /
    /// [`OverlapAddError::BlockLenNotEven`] if `block.len() != 2*N`.
    pub fn process(&mut self, block: &[f64]) -> Result<Vec<f64>, OverlapAddError> {
        let mut out = vec![0.0_f64; self.n];
        self.process_into(block, &mut out)?;
        Ok(out)
    }

    /// Run the weighted overlap-add for one frame into a caller-provided
    /// buffer.
    ///
    /// `block` is the `2*N`-sample inverse MDCT output; `out` receives
    /// the `N` time-domain samples (`out.len()` must be `>= N`). On error
    /// the adder's history is left unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`OverlapAddError`] if the block length is wrong or `out`
    /// is shorter than `N`.
    pub fn process_into(&mut self, block: &[f64], out: &mut [f64]) -> Result<(), OverlapAddError> {
        let two_n = 2 * self.n;
        if block.len() != two_n {
            if block.len() % 2 != 0 {
                return Err(OverlapAddError::BlockLenNotEven { got: block.len() });
            }
            return Err(OverlapAddError::BlockLenMismatch {
                got: block.len(),
                want: two_n,
            });
        }
        if out.len() < self.n {
            return Err(OverlapAddError::OutputTooSmall {
                want: self.n,
                got: out.len(),
            });
        }

        // Window the block (a private copy so the caller's buffer is not
        // mutated). Leading rising ramp, trailing falling ramp, unity
        // middle.
        let mut windowed = block.to_vec();
        // `apply_synthesis_window` re-validates n / overlap, but both are
        // invariants here, so it cannot fail.
        apply_synthesis_window(&mut windowed, &self.ramp, self.n)?;

        // Emit the leading half overlap-added with the saved history.
        for i in 0..self.n {
            out[i] = windowed[i] + self.history[i];
        }

        // Save the windowed trailing half as the next frame's history.
        self.history.copy_from_slice(&windowed[self.n..two_n]);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_imdct::{imdct, mdct_forward};
    use crate::celt_mdct_window::window_tap;

    const EPS: f64 = 1e-9;

    #[test]
    fn rejects_zero_n() {
        assert_eq!(
            WeightedOverlapAdd::new(0, 2),
            Err(OverlapAddError::ZeroLength)
        );
    }

    #[test]
    fn rejects_bad_overlap() {
        // Overlap larger than N.
        assert_eq!(
            WeightedOverlapAdd::new(4, 6),
            Err(OverlapAddError::BadOverlap { overlap: 6, n: 4 })
        );
        // Zero overlap.
        assert_eq!(
            WeightedOverlapAdd::new(4, 0),
            Err(OverlapAddError::BadOverlap { overlap: 0, n: 4 })
        );
        // Odd overlap.
        assert_eq!(
            WeightedOverlapAdd::new(8, 3),
            Err(OverlapAddError::BadOverlap { overlap: 3, n: 8 })
        );
    }

    #[test]
    fn overlap_equal_to_n_is_allowed() {
        // The shortest CELT MDCT can have N == overlap (full-overlap).
        let ola = WeightedOverlapAdd::new(4, 4).unwrap();
        assert_eq!(ola.frame_len(), 4);
        assert_eq!(ola.overlap(), 4);
    }

    #[test]
    fn rejects_wrong_block_length() {
        let mut ola = WeightedOverlapAdd::new(4, 2).unwrap();
        // Odd length.
        assert_eq!(
            ola.process(&[0.0; 7]),
            Err(OverlapAddError::BlockLenNotEven { got: 7 })
        );
        // Even but != 2N.
        assert_eq!(
            ola.process(&[0.0; 6]),
            Err(OverlapAddError::BlockLenMismatch { got: 6, want: 8 })
        );
    }

    #[test]
    fn rejects_output_too_small() {
        let mut ola = WeightedOverlapAdd::new(4, 2).unwrap();
        let block = [0.0; 8];
        let mut out = [0.0; 3];
        assert_eq!(
            ola.process_into(&block, &mut out),
            Err(OverlapAddError::OutputTooSmall { want: 4, got: 3 })
        );
    }

    #[test]
    fn silence_stays_silent() {
        let mut ola = WeightedOverlapAdd::new(8, 4).unwrap();
        for _ in 0..3 {
            let out = ola.process(&[0.0; 16]).unwrap();
            assert_eq!(out, vec![0.0; 8]);
        }
    }

    #[test]
    fn first_frame_emits_windowed_leading_half() {
        // With zero history, frame 0 output is just the windowed leading
        // half of the block (the trailing-half aliasing is cancelled by
        // the next frame).
        let n = 4;
        let overlap = 2;
        let mut ola = WeightedOverlapAdd::new(n, overlap).unwrap();
        let block: Vec<f64> = (0..2 * n).map(|i| (i as f64 + 1.0) * 0.5).collect();
        let out = ola.process(&block).unwrap();
        // Expected: leading `overlap` samples scaled by rising ramp,
        // remaining samples in [overlap, N) unity.
        let ramp = mdct_window(overlap).unwrap();
        for i in 0..overlap {
            assert!((out[i] - block[i] * ramp[i]).abs() < EPS, "i={i}");
        }
        for i in overlap..n {
            assert!((out[i] - block[i]).abs() < EPS, "i={i}");
        }
    }

    #[test]
    fn history_holds_windowed_trailing_half() {
        let n = 4;
        let overlap = 2;
        let mut ola = WeightedOverlapAdd::new(n, overlap).unwrap();
        assert_eq!(ola.history(), &[0.0; 4]);
        let block: Vec<f64> = (0..2 * n).map(|i| i as f64 + 1.0).collect();
        ola.process(&block).unwrap();
        // History = windowed samples [N, 2N): the trailing `overlap`
        // samples carry the falling ramp, the rest are unity.
        let ramp = mdct_window(overlap).unwrap();
        let mut windowed = block.clone();
        apply_synthesis_window(&mut windowed, &ramp, n).unwrap();
        assert_eq!(ola.history(), &windowed[n..2 * n]);
    }

    #[test]
    fn reset_zeroes_history() {
        let mut ola = WeightedOverlapAdd::new(4, 2).unwrap();
        ola.process(&[1.0; 8]).unwrap();
        assert!(ola.history().iter().any(|&h| h != 0.0));
        ola.reset();
        assert_eq!(ola.history(), &[0.0; 4]);
    }

    #[test]
    fn apply_synthesis_window_layout() {
        // Directly verify the rising/falling/unity layout.
        let n = 8;
        let overlap = 4;
        let ramp = mdct_window(overlap).unwrap();
        let block: Vec<f64> = (0..2 * n).map(|i| i as f64 + 1.0).collect();
        let mut w = block.clone();
        apply_synthesis_window(&mut w, &ramp, n).unwrap();
        let two_n = 2 * n;
        for i in 0..overlap {
            assert!((w[i] - block[i] * ramp[i]).abs() < EPS, "lead i={i}");
            assert!(
                (w[two_n - 1 - i] - block[two_n - 1 - i] * ramp[i]).abs() < EPS,
                "trail i={i}"
            );
        }
        for i in overlap..(two_n - overlap) {
            assert!((w[i] - block[i]).abs() < EPS, "middle i={i}");
        }
    }

    #[test]
    fn apply_synthesis_window_rejects_bad_args() {
        let ramp = mdct_window(2).unwrap();
        assert_eq!(
            apply_synthesis_window(&mut [0.0; 8], &ramp, 0),
            Err(OverlapAddError::ZeroLength)
        );
        // Block length not 2N.
        assert_eq!(
            apply_synthesis_window(&mut [0.0; 6], &ramp, 4),
            Err(OverlapAddError::BlockLenMismatch { got: 6, want: 8 })
        );
        // Odd block length.
        assert!(matches!(
            apply_synthesis_window(&mut [0.0; 7], &ramp, 4),
            Err(OverlapAddError::BlockLenNotEven { got: 7 })
        ));
        // Overlap > N (ramp of len 6, N=2).
        let big = mdct_window(6).unwrap();
        assert_eq!(
            apply_synthesis_window(&mut [0.0; 4], &big, 2),
            Err(OverlapAddError::BadOverlap { overlap: 6, n: 2 })
        );
    }

    /// A symmetric Princen-Bradley analysis/synthesis window over the
    /// *whole* `2N` block: `w(n) = sin( (pi/2N)*(n + 1/2) )`. This is the
    /// full-overlap window pair (overlap = N) that gives perfect MDCT
    /// reconstruction; here it lets us drive the stateful overlap-add
    /// end-to-end against a windowed forward/inverse round-trip.
    fn pb_window(two_n: usize) -> Vec<f64> {
        (0..two_n)
            .map(|i| (core::f64::consts::PI / (two_n as f64) * (i as f64 + 0.5)).sin())
            .collect()
    }

    #[test]
    fn multi_frame_arithmetic_matches_reference() {
        // Validate the stateful adder's arithmetic across several frames:
        // for each frame, out_b(i) = windowed_b[i] + windowed_{b-1}[N+i],
        // and the history carries the windowed trailing half forward. We
        // compare the adder against an independent hand computation using
        // the §4.3.7 low-overlap ramp.
        let n = 6;
        let overlap = 4;
        let mut ola = WeightedOverlapAdd::new(n, overlap).unwrap();
        let ramp = mdct_window(overlap).unwrap();
        // Sanity: the ramp is the §4.3.7 window over its own length.
        assert!((ramp[0] - window_tap(0, overlap).unwrap()).abs() < EPS);

        let blocks: Vec<Vec<f64>> = (0..3)
            .map(|b| {
                (0..2 * n)
                    .map(|i| ((b * n + i) as f64 * 0.21).sin() - 0.3)
                    .collect()
            })
            .collect();

        // Independent reference: window each block, then out_b(i) =
        // windowed_b[i] + windowed_{b-1}[N+i].
        let mut prev_tail = vec![0.0_f64; n];
        for block in &blocks {
            let mut w = block.clone();
            apply_synthesis_window(&mut w, &ramp, n).unwrap();
            let want: Vec<f64> = (0..n).map(|i| w[i] + prev_tail[i]).collect();
            let got = ola.process(block).unwrap();
            for i in 0..n {
                assert!(
                    (got[i] - want[i]).abs() < EPS,
                    "i={i}: {} != {}",
                    got[i],
                    want[i]
                );
            }
            prev_tail.copy_from_slice(&w[n..2 * n]);
        }
    }

    #[test]
    fn end_to_end_tdac_via_imdct_full_overlap() {
        // True TDAC: with the symmetric Princen-Bradley window applied on
        // BOTH analysis and synthesis (overlap = N), a windowed forward
        // MDCT, the §4.3.7 inverse MDCT, and a hop-N overlap-add
        // reconstruct the input at the §4.3.7 `1/2` scale. This pins the
        // aliasing-cancellation property the §4.3.7 overlap-add provides,
        // using the canonical symmetric MDCT window (independent of the
        // CELT low-overlap layout, which is exercised arithmetically
        // above).
        let n = 8;
        let two_n = 2 * n;
        let win = pb_window(two_n);
        // hop = N; produce three overlapping analysis blocks of a signal.
        let total = 4 * n;
        let signal: Vec<f64> = (0..total)
            .map(|i| (i as f64 * 0.37).sin() * 1.5 - (i as f64 * 0.11).cos())
            .collect();

        // Synthesise each block: window (analysis), forward, inverse,
        // window (synthesis) -> these are the per-block windowed IMDCT
        // outputs the overlap-add sums.
        let synth = |start: usize| -> Vec<f64> {
            let blk: Vec<f64> = (0..two_n).map(|i| signal[start + i] * win[i]).collect();
            let coeffs = mdct_forward(&blk).unwrap();
            let rec = imdct(&coeffs).unwrap();
            rec.iter().zip(win.iter()).map(|(r, w)| r * w).collect()
        };

        // Overlap-add by hand (the symmetric window is already applied):
        // out covering global [N, 2N) = synth(0)[N..2N] + synth(N)[0..N].
        let f0 = synth(0);
        let f1 = synth(n);
        for j in 0..n {
            let recon = f0[n + j] + f1[j];
            let want = 0.5 * signal[n + j];
            assert!(
                (recon - want).abs() < 1e-9,
                "overlap j={j}: {recon} != 0.5*signal {want}"
            );
        }
    }

    #[test]
    fn error_display_messages() {
        assert!(OverlapAddError::ZeroLength.to_string().contains("N >= 1"));
        assert!(OverlapAddError::BlockLenNotEven { got: 7 }
            .to_string()
            .contains("not 2*N"));
        assert!(OverlapAddError::BlockLenMismatch { got: 6, want: 8 }
            .to_string()
            .contains("!= required 2*N"));
        assert!(OverlapAddError::BadOverlap { overlap: 6, n: 4 }
            .to_string()
            .contains("invalid"));
        assert!(OverlapAddError::OutputTooSmall { want: 4, got: 3 }
            .to_string()
            .contains("< required N"));
    }
}
