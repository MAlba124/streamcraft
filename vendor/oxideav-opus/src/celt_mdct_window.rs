//! CELT §4.3.7 inverse-MDCT overlap window
//! (RFC 6716 §4.3.7, p. 121).
//!
//! The last stage of the CELT decoder before time-domain output is the
//! inverse MDCT followed by a weighted overlap-add. The overlap-add
//! uses a "low-overlap" window derived from the window used by the
//! Vorbis codec. RFC 6716 §4.3.7 (p. 121) states the *basic*
//! (full-overlap) 240-sample window directly:
//!
//! ```text
//!                                                  2
//!                   /   /pi      /pi   n + 1/2\ \ \
//!            W(n) = |sin|-- * sin|-- * -------| | |
//!                   \   \2       \2       L   / / /
//! ```
//!
//! The squared-double-sine form. The `2` superscript squares the
//! *inner* sine (the Vorbis window this is "derived from" — and the
//! power-complementarity requirement below — fix the nesting): the
//! inner term `s = sin( (pi/2) * (n + 1/2) / L )` is squared, then
//! multiplied by `pi/2` and passed through `sin`:
//!
//! ```text
//!   W(n) = sin( (pi/2) * sin( (pi/2) * (n + 1/2) / L )^2 )
//! ```
//!
//! For the basic full-overlap window `L = 240` and `n` runs
//! `0 ..= 239`. `W(n)` is the *amplitude* window tap (in `[0, 1]`),
//! not the squared/power tap.
//!
//! ## Power complementarity
//!
//! The §4.3.7 prose requires the window to satisfy *power
//! complementarity* (also called the Princen-Bradley condition,
//! `[PRINCEN86]`): `W(n)^2 + W(L - 1 - n)^2 = 1` for every `n`. With
//! the inner-squared form above this holds exactly. Write
//! `t = (n + 1/2)/L` and `s(n) = sin( (pi/2) * t )^2`. The reflected
//! index `L - 1 - n` gives `t' = 1 - t`, and
//! `sin( (pi/2)*(1 - t) ) = cos( (pi/2)*t )`, so
//! `s(L - 1 - n) = cos( (pi/2)*t )^2 = 1 - s(n)`. Then
//! `W(n) = sin( (pi/2)*s(n) )` and
//! `W(L-1-n) = sin( (pi/2)*(1 - s(n)) ) = cos( (pi/2)*s(n) )`, so
//! `W(n)^2 + W(L-1-n)^2 = sin( (pi/2)*s )^2 + cos( (pi/2)*s )^2 = 1`.
//! This is exactly why the inner sine carries the square: the *whole*
//! expression squared would break complementarity. The unit tests pin
//! `W(n)^2 + W(L-1-n)^2 = 1` to floating-point tolerance.
//!
//! ## Low-overlap construction
//!
//! RFC 6716 §4.3.7 (p. 121):
//!
//! > The low-overlap window is created by zero-padding the basic
//! > window and inserting ones in the middle, such that the resulting
//! > window still satisfies power complementarity.
//!
//! The MDCT overlap region for an `2N`-point inverse transform spans
//! `overlap` samples (the CELT layer fixes the overlap at the 2.5 ms
//! look-ahead — 120 samples at 48 kHz — per RFC 6716 §1, "the MDCT
//! overlap, whose size is fixed by the decoder"). [`mdct_window`]
//! builds the windowed overlap taps for an arbitrary even `overlap`
//! by evaluating the same squared-double-sine shape with `L = overlap`
//! over `n = 0 ..= overlap - 1`; with `L = overlap` this is exactly
//! the "zero-padding + ones-in-the-middle" construction expressed as a
//! per-overlap window (the zero-padded tail and the inserted ones are
//! the parts of the `2N`-point frame *outside* the `overlap`-sample
//! transition that this function returns; the consumer applies the
//! returned ramp to the leading and trailing `overlap` samples and
//! treats the centre as unity). The returned ramp is monotonically
//! increasing from `W(0)` to `W(overlap-1)` and remains power-
//! complementary about its own centre.
//!
//! This module owns only the *window shape*; the inverse MDCT itself
//! ("no special characteristics. The input is N frequency-domain
//! samples and the output is 2*N time-domain samples, while scaling by
//! 1/2") and the weighted overlap-add that consumes this ramp run at
//! the §4.3.7 consumer site.
//!
//! ## Provenance
//!
//! Window formula + low-overlap narrative + power-complementarity
//! requirement: RFC 6716 §4.3.7 (p. 121) and the §1 fixed-overlap
//! statement (p. 10), reproduced from
//! `docs/audio/opus/rfc6716-opus.txt`. No external library source was
//! consulted; the window equation is stated directly in the
//! standards-track text.

use core::f64::consts::FRAC_PI_2;

/// Length of the §4.3.7 *basic* (full-overlap) Vorbis-derived window:
/// 240 samples (RFC 6716 §4.3.7, p. 121).
pub const BASIC_WINDOW_LEN: usize = 240;

/// CELT MDCT overlap at 48 kHz: 120 samples (the 2.5 ms look-ahead
/// "fixed by the decoder", RFC 6716 §1, p. 10).
pub const CELT_OVERLAP_48K: usize = 120;

/// Errors returnable by the §4.3.7 window helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MdctWindowError {
    /// The window position `n` is outside `0 ..= len - 1`.
    PositionOutOfRange {
        /// The position the caller passed.
        n: usize,
        /// The window length `L` the caller passed.
        len: usize,
    },
    /// The window length `L` is zero; the §4.3.7
    /// `(n + 1/2) / L` term is undefined for a zero-length window.
    ZeroLength,
    /// The overlap length passed to [`mdct_window`] is odd; the
    /// §4.3.7 "inserting ones in the middle" construction requires an
    /// even overlap so the centre splits cleanly.
    OddOverlap {
        /// The overlap the caller passed.
        overlap: usize,
    },
}

impl core::fmt::Display for MdctWindowError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            MdctWindowError::PositionOutOfRange { n, len } => write!(
                f,
                "oxideav-opus: CELT §4.3.7 window position {n} out of range \
                 (window length {len} allows 0..={})",
                len.saturating_sub(1)
            ),
            MdctWindowError::ZeroLength => {
                write!(f, "oxideav-opus: CELT §4.3.7 window requires length L >= 1")
            }
            MdctWindowError::OddOverlap { overlap } => write!(
                f,
                "oxideav-opus: CELT §4.3.7 low-overlap window requires an even \
                 overlap (got {overlap})"
            ),
        }
    }
}

impl std::error::Error for MdctWindowError {}

/// Evaluate the §4.3.7 amplitude window tap `W(n)` for window length
/// `L = len`.
///
/// Computes
/// `W(n) = sin( (pi/2) * sin( (pi/2) * (n + 1/2) / L )^2 )`
/// (RFC 6716 §4.3.7, p. 121). For the basic full-overlap window pass
/// `len = `[`BASIC_WINDOW_LEN`]` (240)`; for a low-overlap window pass
/// the overlap length. The returned tap is the amplitude window in
/// `[0, 1]`; consecutive frames overlap-add with
/// `W(n)^2 + W(L-1-n)^2 = 1`.
///
/// # Errors
///
/// Returns [`MdctWindowError::ZeroLength`] if `len == 0`, or
/// [`MdctWindowError::PositionOutOfRange`] if `n >= len`.
pub fn window_tap(n: usize, len: usize) -> Result<f64, MdctWindowError> {
    if len == 0 {
        return Err(MdctWindowError::ZeroLength);
    }
    if n >= len {
        return Err(MdctWindowError::PositionOutOfRange { n, len });
    }
    let t = (n as f64 + 0.5) / len as f64;
    let inner = (FRAC_PI_2 * t).sin();
    let inner_sq = inner * inner;
    Ok((FRAC_PI_2 * inner_sq).sin())
}

/// Build the full basic (full-overlap) 240-sample window
/// (RFC 6716 §4.3.7, p. 121).
///
/// Element `n` is [`window_tap`]`(n, `[`BASIC_WINDOW_LEN`]`)`.
pub fn basic_window() -> [f64; BASIC_WINDOW_LEN] {
    let mut w = [0.0_f64; BASIC_WINDOW_LEN];
    for (n, slot) in w.iter_mut().enumerate() {
        // `n < BASIC_WINDOW_LEN` and `BASIC_WINDOW_LEN != 0`, so this
        // never errors.
        *slot = window_tap(n, BASIC_WINDOW_LEN).expect("in-range basic-window tap");
    }
    w
}

/// Build the §4.3.7 low-overlap window ramp for an arbitrary even
/// `overlap` (RFC 6716 §4.3.7, p. 121).
///
/// Returns the `overlap` rising taps `W(0) ..= W(overlap - 1)` with
/// window length `L = overlap`. The caller applies this ramp to the
/// leading `overlap` samples of the `2N`-sample inverse-MDCT output
/// (and its time-reverse to the trailing `overlap` samples); the
/// `2N - 2*overlap` samples in the middle are unity ("inserting ones
/// in the middle"). With `L = overlap` the ramp is power-complementary
/// about its own centre, so the overlap-add of two adjacent frames
/// preserves energy.
///
/// # Errors
///
/// Returns [`MdctWindowError::ZeroLength`] if `overlap == 0`, or
/// [`MdctWindowError::OddOverlap`] if `overlap` is odd.
pub fn mdct_window(overlap: usize) -> Result<Vec<f64>, MdctWindowError> {
    if overlap == 0 {
        return Err(MdctWindowError::ZeroLength);
    }
    if overlap % 2 != 0 {
        return Err(MdctWindowError::OddOverlap { overlap });
    }
    let mut w = Vec::with_capacity(overlap);
    for n in 0..overlap {
        w.push(window_tap(n, overlap)?);
    }
    Ok(w)
}

/// The CELT 48 kHz overlap window ramp: [`mdct_window`]`(120)`.
///
/// Convenience wrapper for the fixed CELT overlap
/// ([`CELT_OVERLAP_48K`]).
pub fn celt_overlap_window() -> [f64; CELT_OVERLAP_48K] {
    let mut w = [0.0_f64; CELT_OVERLAP_48K];
    for (n, slot) in w.iter_mut().enumerate() {
        *slot = window_tap(n, CELT_OVERLAP_48K).expect("in-range CELT overlap tap");
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for the double-precision trigonometric identities.
    const EPS: f64 = 1e-12;

    #[test]
    fn basic_window_len_is_240() {
        assert_eq!(BASIC_WINDOW_LEN, 240);
        assert_eq!(basic_window().len(), 240);
    }

    #[test]
    fn celt_overlap_is_120() {
        assert_eq!(CELT_OVERLAP_48K, 120);
        assert_eq!(celt_overlap_window().len(), 120);
    }

    #[test]
    fn window_tap_matches_formula() {
        // Spot-check the inner-squared double-sine form directly:
        // W(n) = sin( (pi/2) * sin( (pi/2)*(n+1/2)/L )^2 ).
        let len = 240;
        for &n in &[0_usize, 1, 59, 119, 120, 179, 239] {
            let t = (n as f64 + 0.5) / len as f64;
            let want = (FRAC_PI_2 * (FRAC_PI_2 * t).sin().powi(2)).sin();
            let got = window_tap(n, len).unwrap();
            assert!((got - want).abs() < EPS, "n={n}: {got} != {want}");
        }
    }

    #[test]
    fn window_taps_are_within_unit_interval() {
        for len in [2_usize, 16, 120, 240] {
            for n in 0..len {
                let w = window_tap(n, len).unwrap();
                assert!((0.0..=1.0).contains(&w), "len={len} n={n} w={w}");
            }
        }
    }

    #[test]
    fn window_is_monotonically_increasing() {
        // The rising overlap ramp grows from W(0) toward W(L-1).
        for len in [2_usize, 16, 120, 240] {
            for n in 1..len {
                let prev = window_tap(n - 1, len).unwrap();
                let cur = window_tap(n, len).unwrap();
                assert!(cur >= prev, "len={len} n={n}: {cur} < {prev}");
            }
        }
    }

    #[test]
    fn power_complementarity_holds_basic() {
        // W(n)^2 + W(L-1-n)^2 == 1 for the amplitude window
        // (RFC 6716 §4.3.7 Princen-Bradley requirement).
        let len = BASIC_WINDOW_LEN;
        for n in 0..len {
            let a = window_tap(n, len).unwrap();
            let b = window_tap(len - 1 - n, len).unwrap();
            assert!((a * a + b * b - 1.0).abs() < EPS, "n={n}: {a}^2+{b}^2 != 1");
        }
    }

    #[test]
    fn power_complementarity_holds_overlap() {
        for len in [2_usize, 4, 16, 120, 240] {
            for n in 0..len {
                let a = window_tap(n, len).unwrap();
                let b = window_tap(len - 1 - n, len).unwrap();
                assert!(
                    (a * a + b * b - 1.0).abs() < EPS,
                    "len={len} n={n}: {a}^2+{b}^2 != 1"
                );
            }
        }
    }

    #[test]
    fn window_centre_is_one_over_sqrt_two() {
        // The two central amplitude taps straddle the half-power point
        // 1/sqrt(2); their squares sum to one and they are symmetric
        // about 1/sqrt(2)^2 in power terms.
        let half_power = core::f64::consts::FRAC_1_SQRT_2;
        for len in [2_usize, 16, 120, 240] {
            let lo = window_tap(len / 2 - 1, len).unwrap();
            let hi = window_tap(len / 2, len).unwrap();
            assert!((lo * lo + hi * hi - 1.0).abs() < EPS);
            // The centre pair straddles the half-power amplitude.
            assert!(
                lo <= half_power + EPS && hi >= half_power - EPS,
                "len={len}"
            );
        }
    }

    #[test]
    fn first_and_last_tap_endpoints() {
        // W(0) is small and positive; W(L-1) is close to one; their
        // squares are complementary.
        let len = BASIC_WINDOW_LEN;
        let first = window_tap(0, len).unwrap();
        let last = window_tap(len - 1, len).unwrap();
        assert!(first > 0.0 && first < 0.01, "first={first}");
        assert!(last > 0.99 && last < 1.0, "last={last}");
        assert!((first * first + last * last - 1.0).abs() < EPS);
    }

    #[test]
    fn mdct_window_matches_window_tap() {
        let w = mdct_window(120).unwrap();
        assert_eq!(w.len(), 120);
        for (n, &got) in w.iter().enumerate() {
            assert_eq!(got, window_tap(n, 120).unwrap());
        }
    }

    #[test]
    fn mdct_window_is_power_complementary() {
        let w = mdct_window(120).unwrap();
        let len = w.len();
        for n in 0..len {
            assert!((w[n] * w[n] + w[len - 1 - n] * w[len - 1 - n] - 1.0).abs() < EPS);
        }
    }

    #[test]
    fn celt_overlap_window_matches_mdct_window() {
        let arr = celt_overlap_window();
        let vec = mdct_window(CELT_OVERLAP_48K).unwrap();
        assert_eq!(arr.len(), vec.len());
        for (a, b) in arr.iter().zip(vec.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn basic_window_matches_window_tap() {
        let w = basic_window();
        for (n, &got) in w.iter().enumerate() {
            assert_eq!(got, window_tap(n, BASIC_WINDOW_LEN).unwrap());
        }
    }

    #[test]
    fn window_tap_rejects_out_of_range_position() {
        assert_eq!(
            window_tap(240, 240),
            Err(MdctWindowError::PositionOutOfRange { n: 240, len: 240 })
        );
        assert_eq!(
            window_tap(5, 5),
            Err(MdctWindowError::PositionOutOfRange { n: 5, len: 5 })
        );
    }

    #[test]
    fn window_tap_rejects_zero_length() {
        assert_eq!(window_tap(0, 0), Err(MdctWindowError::ZeroLength));
    }

    #[test]
    fn mdct_window_rejects_zero_overlap() {
        assert_eq!(mdct_window(0), Err(MdctWindowError::ZeroLength));
    }

    #[test]
    fn mdct_window_rejects_odd_overlap() {
        assert_eq!(
            mdct_window(121),
            Err(MdctWindowError::OddOverlap { overlap: 121 })
        );
    }

    #[test]
    fn error_display_messages() {
        assert!(MdctWindowError::PositionOutOfRange { n: 7, len: 4 }
            .to_string()
            .contains("position 7"));
        assert!(MdctWindowError::ZeroLength.to_string().contains("L >= 1"));
        assert!(MdctWindowError::OddOverlap { overlap: 3 }
            .to_string()
            .contains("even"));
    }
}
