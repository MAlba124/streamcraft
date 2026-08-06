//! CELT §4.3.7 inverse MDCT transform core
//! (RFC 6716 §4.3.7, p. 121).
//!
//! The inverse MDCT is the CELT stage between band denormalisation
//! (§4.3.6, [`crate::celt_denormalise`]) and the weighted overlap-add /
//! post-filter (§4.3.7.1). RFC 6716 §4.3.7 (p. 121) states the
//! transform completely in one sentence:
//!
//! > The inverse MDCT implementation has no special characteristics.
//! > The input is N frequency-domain samples and the output is 2*N
//! > time-domain samples, while scaling by 1/2.
//!
//! "No special characteristics" means the transform is the textbook
//! inverse modified-discrete-cosine transform — a mathematical
//! definition, not an implementation-specific algorithm. This module
//! implements that definition directly from the standards-track text;
//! the only RFC-supplied parameters are the `N -> 2N` length expansion
//! and the `1/2` output scale.
//!
//! ## The transform
//!
//! The MDCT is a lapped transform built on the type-IV DCT. The forward
//! MDCT maps a `2N`-sample windowed time block `x(0..2N)` to `N`
//! frequency coefficients:
//!
//! ```text
//!            2N-1
//!   X(k) =    Σ   x(n) * cos[ (pi/N) * (n + 1/2 + N/2) * (k + 1/2) ]
//!            n=0
//! ```
//!
//! for `k = 0 .. N`. Its inverse — the transform this module computes —
//! reconstructs `2N` time-domain samples from the `N` coefficients:
//!
//! ```text
//!             N-1
//!   y(n) = c * Σ   X(k) * cos[ (pi/N) * (n + 1/2 + N/2) * (k + 1/2) ]
//!             k=0
//! ```
//!
//! for `n = 0 .. 2N`. The cosine kernel of the inverse is the transpose
//! of the forward kernel (the MDCT matrix is — up to scale — its own
//! transpose), so the two share the single phase term
//! `(n + 1/2 + N/2)*(k + 1/2)`. The constant `c` is fixed by the RFC's
//! "scaling by 1/2" statement: `c = 1/N`, which for a unitary-up-to-
//! scale type-IV kernel makes the output exactly half the value that the
//! self-adjoint round-trip would otherwise produce, i.e. the IMDCT
//! "scales by 1/2" relative to the unnormalised forward/inverse pair.
//! See the round-trip / TDAC discussion below for why `1/N` is the value
//! that makes overlap-add reconstruct the input.
//!
//! ## Time-domain aliasing cancellation (TDAC)
//!
//! A single `2N -> N -> 2N` MDCT round-trip is *not* the identity: each
//! reconstructed block carries a time-domain aliased copy of itself
//! folded about its two half-block midpoints. This is the defining
//! property of the MDCT (it is why `N` coefficients can represent a
//! `2N`-sample block without redundancy). The aliasing is cancelled
//! only when adjacent blocks — overlapped by `N` samples and each
//! multiplied by a power-complementary window (§4.3.7,
//! [`crate::celt_mdct_window`]) — are summed. The two aliased halves of
//! neighbouring blocks are equal-and-opposite in the overlap region and
//! cancel on the add. This module computes the raw `2N`-sample inverse
//! block (already scaled by `1/2`); the windowing and the overlap-add
//! that complete the cancellation run at the §4.3.7 consumer site.
//!
//! The structure of the aliasing is exact and is pinned by the unit
//! tests: for a forward/inverse pair built from the *same* kernel, the
//! reconstructed block `y` satisfies
//!
//! ```text
//!   y(n)        = -y(N - 1 - n)              for 0      <= n < N/2
//!   y(N/2 + n)  =  y(N - 1 - (N/2 + n))      (folds about 3N/2 too)
//! ```
//!
//! i.e. the first quarter is the negated mirror of the second quarter
//! and the last quarter is the mirror of the third — the canonical MDCT
//! aliasing pattern. The tests verify this directly and verify that a
//! windowed forward/inverse round-trip with overlap-add of two adjacent
//! blocks reconstructs an arbitrary input — at the RFC's documented
//! `1/2` IMDCT scaling — confirming the aliasing fully cancels.
//!
//! ## Provenance
//!
//! Transform length (`N -> 2N`), output scale (`1/2`), and the
//! "no special characteristics" definition: RFC 6716 §4.3.7 (p. 121),
//! reproduced from `docs/audio/opus/rfc6716-opus.txt`. The type-IV /
//! MDCT cosine kernel and the TDAC property are textbook mathematical
//! facts about the inverse MDCT the RFC names; no external library
//! source was consulted, and the RFC explicitly states the transform
//! "has no special characteristics" beyond the stated length and scale.

use core::f64::consts::PI;

/// Errors returnable by the §4.3.7 inverse-MDCT helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImdctError {
    /// The transform half-length `N` is zero; an empty spectrum has no
    /// inverse.
    ZeroLength,
    /// The output slice passed to [`imdct_into`] is not exactly `2*N`
    /// samples long.
    OutputLenMismatch {
        /// The output length the caller supplied.
        got: usize,
        /// The required length `2 * spectrum.len()`.
        want: usize,
    },
}

impl core::fmt::Display for ImdctError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            ImdctError::ZeroLength => {
                write!(f, "oxideav-opus: CELT §4.3.7 inverse MDCT requires N >= 1")
            }
            ImdctError::OutputLenMismatch { got, want } => write!(
                f,
                "oxideav-opus: CELT §4.3.7 inverse MDCT output length {got} \
                 != required 2*N = {want}"
            ),
        }
    }
}

impl std::error::Error for ImdctError {}

/// Compute the §4.3.7 inverse MDCT of an `N`-sample spectrum into a
/// caller-provided `2*N`-sample time-domain buffer (RFC 6716 §4.3.7,
/// p. 121).
///
/// `spectrum` holds the `N` denormalised MDCT bins; `out` receives the
/// `2*N` time-domain samples, already scaled by the §4.3.7 `1/2` factor.
/// The output is the *raw* inverse block before windowing and
/// overlap-add — the consumer multiplies the leading/trailing `overlap`
/// samples by the [`crate::celt_mdct_window`] ramp and overlap-adds with
/// the neighbouring block to cancel the time-domain aliasing.
///
/// Each output sample is
/// `y(n) = (1/N) * Σ_k X(k) * cos[ (pi/N)*(n + 1/2 + N/2)*(k + 1/2) ]`.
///
/// # Errors
///
/// Returns [`ImdctError::ZeroLength`] if `spectrum` is empty, or
/// [`ImdctError::OutputLenMismatch`] if `out.len() != 2 * spectrum.len()`.
pub fn imdct_into(spectrum: &[f64], out: &mut [f64]) -> Result<(), ImdctError> {
    let n = spectrum.len();
    if n == 0 {
        return Err(ImdctError::ZeroLength);
    }
    let want = 2 * n;
    if out.len() != want {
        return Err(ImdctError::OutputLenMismatch {
            got: out.len(),
            want,
        });
    }

    let nf = n as f64;
    // The §4.3.7 "scaling by 1/2": with the symmetric (self-adjoint)
    // cosine kernel below, the forward/inverse round-trip without
    // normalisation scales the signal by `N`, so dividing by `N` here
    // both inverts that and applies the RFC's `1/2` factor relative to
    // the unnormalised pair (the windowed overlap-add of two blocks then
    // reconstructs the input at the documented 1/2 gain — see TDAC tests).
    let scale = 1.0 / nf;
    let half_n = nf / 2.0;

    for (idx, slot) in out.iter_mut().enumerate() {
        let n_term = idx as f64 + 0.5 + half_n;
        let mut acc = 0.0_f64;
        for (k, &xk) in spectrum.iter().enumerate() {
            let phase = (PI / nf) * n_term * (k as f64 + 0.5);
            acc += xk * phase.cos();
        }
        *slot = scale * acc;
    }
    Ok(())
}

/// Allocating wrapper for [`imdct_into`]: returns a freshly allocated
/// `2*N`-sample time-domain block (RFC 6716 §4.3.7, p. 121).
///
/// # Errors
///
/// Returns [`ImdctError::ZeroLength`] if `spectrum` is empty.
pub fn imdct(spectrum: &[f64]) -> Result<Vec<f64>, ImdctError> {
    let mut out = vec![0.0_f64; 2 * spectrum.len()];
    imdct_into(spectrum, &mut out)?;
    Ok(out)
}

/// Compute the forward MDCT of a `2*N`-sample windowed time block into
/// `N` coefficients, sharing the §4.3.7 inverse kernel.
///
/// This is the transpose-kernel partner of [`imdct_into`]; it exists so
/// the round-trip / TDAC behaviour of the inverse can be exercised
/// against a matched forward transform built from the *identical*
/// cosine kernel (the MDCT matrix is its own transpose up to scale).
/// It is **not** part of the CELT *decode* path — the decoder never runs
/// a forward MDCT — but the property it pins (perfect reconstruction
/// after windowed overlap-add) is exactly the §4.3.7 contract the
/// inverse must satisfy.
///
/// `X(k) = Σ_n x(n) * cos[ (pi/N)*(n + 1/2 + N/2)*(k + 1/2) ]`.
///
/// # Errors
///
/// Returns [`ImdctError::ZeroLength`] if `time` is empty, or
/// [`ImdctError::OutputLenMismatch`] if `time.len()` is odd.
pub fn mdct_forward(time: &[f64]) -> Result<Vec<f64>, ImdctError> {
    if time.is_empty() {
        return Err(ImdctError::ZeroLength);
    }
    if time.len() % 2 != 0 {
        return Err(ImdctError::OutputLenMismatch {
            got: time.len(),
            want: time.len() + 1,
        });
    }
    let n = time.len() / 2;
    let nf = n as f64;
    let half_n = nf / 2.0;
    let mut out = vec![0.0_f64; n];
    for (k, coeff) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (idx, &xn) in time.iter().enumerate() {
            let n_term = idx as f64 + 0.5 + half_n;
            let phase = (PI / nf) * n_term * (k as f64 + 0.5);
            acc += xn * phase.cos();
        }
        *coeff = acc;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_mdct_window::window_tap;

    const EPS: f64 = 1e-9;

    #[test]
    fn empty_spectrum_rejected() {
        assert_eq!(imdct(&[]), Err(ImdctError::ZeroLength));
        let mut out = [];
        assert_eq!(imdct_into(&[], &mut out), Err(ImdctError::ZeroLength));
    }

    #[test]
    fn output_length_is_twice_n() {
        let spec = [1.0, 2.0, 3.0, 4.0];
        let y = imdct(&spec).unwrap();
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn output_len_mismatch_rejected() {
        let spec = [1.0, 2.0];
        let mut out = [0.0; 3];
        assert_eq!(
            imdct_into(&spec, &mut out),
            Err(ImdctError::OutputLenMismatch { got: 3, want: 4 })
        );
    }

    #[test]
    fn matches_direct_formula() {
        // Independent recomputation of the §4.3.7 inverse kernel.
        let spec = [0.5, -1.0, 2.0, 0.25];
        let n = spec.len();
        let nf = n as f64;
        let y = imdct(&spec).unwrap();
        for (idx, &got) in y.iter().enumerate() {
            let n_term = idx as f64 + 0.5 + nf / 2.0;
            let mut want = 0.0;
            for (k, &xk) in spec.iter().enumerate() {
                want += xk * ((PI / nf) * n_term * (k as f64 + 0.5)).cos();
            }
            want /= nf;
            assert!((got - want).abs() < EPS, "n={idx}: {got} != {want}");
        }
    }

    #[test]
    fn imdct_into_matches_imdct() {
        let spec = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let owned = imdct(&spec).unwrap();
        let mut buf = vec![0.0; 2 * spec.len()];
        imdct_into(&spec, &mut buf).unwrap();
        assert_eq!(owned, buf);
    }

    #[test]
    fn linearity() {
        // The transform is linear: IMDCT(a*X + b*Y) = a*IMDCT(X) + b*IMDCT(Y).
        let x = [1.0, 2.0, -1.0, 0.5];
        let yv = [0.3, -0.7, 2.1, 1.0];
        let a = 1.5;
        let b = -2.0;
        let comb: Vec<f64> = x
            .iter()
            .zip(yv.iter())
            .map(|(p, q)| a * p + b * q)
            .collect();
        let lhs = imdct(&comb).unwrap();
        let ix = imdct(&x).unwrap();
        let iy = imdct(&yv).unwrap();
        for i in 0..lhs.len() {
            let rhs = a * ix[i] + b * iy[i];
            assert!((lhs[i] - rhs).abs() < EPS, "i={i}");
        }
    }

    #[test]
    fn aliasing_symmetry_pattern() {
        // The canonical MDCT time-domain aliasing structure: the inverse
        // block of length 2N folds about N/2 (first quarter = negated
        // mirror of second quarter) and about 3N/2 (last quarter =
        // mirror of third quarter). This is what the windowed
        // overlap-add later cancels.
        let spec = [0.7, -1.3, 2.2, 0.1, -0.9, 1.1, 0.4, -2.5];
        let n = spec.len();
        let y = imdct(&spec).unwrap();
        // Fold about N/2: y(n) = -y(N-1-n) for 0 <= n < N/2.
        for nn in 0..n / 2 {
            assert!(
                (y[nn] + y[n - 1 - nn]).abs() < EPS,
                "lower fold n={nn}: {} vs {}",
                y[nn],
                y[n - 1 - nn]
            );
        }
        // Fold about 3N/2: y(N + m) = y(2N - 1 - m) for 0 <= m < N/2.
        for m in 0..n / 2 {
            assert!(
                (y[n + m] - y[2 * n - 1 - m]).abs() < EPS,
                "upper fold m={m}: {} vs {}",
                y[n + m],
                y[2 * n - 1 - m]
            );
        }
    }

    #[test]
    fn forward_then_inverse_block_alias() {
        // A single 2N -> N -> 2N round-trip reproduces the aliased block,
        // not the identity. With the matched (self-transpose) kernel the
        // aliased block is, in the first half,
        //   y(n) = (x(n) - x(N-1-n)) / 2   for 0 <= n < N,
        // i.e. the time-domain aliasing folds x about N/2. We verify the
        // exact aliased relation rather than identity.
        let n = 4;
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = mdct_forward(&x).unwrap();
        assert_eq!(coeffs.len(), n);
        let y = imdct(&coeffs).unwrap();
        assert_eq!(y.len(), 2 * n);
        // Lower half: y(n) = (x(n) - x(N-1-n)) / 2.
        for nn in 0..n {
            let want = (x[nn] - x[n - 1 - nn]) / 2.0;
            assert!(
                (y[nn] - want).abs() < EPS,
                "lower n={nn}: {} != {}",
                y[nn],
                want
            );
        }
        // Upper half: y(N+m) = (x(N+m) + x(2N-1-m)) / 2.
        for m in 0..n {
            let want = (x[n + m] + x[2 * n - 1 - m]) / 2.0;
            assert!(
                (y[n + m] - want).abs() < EPS,
                "upper m={m}: {} != {}",
                y[n + m],
                want
            );
        }
    }

    /// A symmetric Princen-Bradley analysis/synthesis window
    /// `w(n) = sin( (pi/2N) * (n + 1/2) )`, `n = 0 .. 2N`. It is
    /// symmetric (`w(n) = w(2N-1-n)`) and satisfies the Princen-Bradley
    /// condition `w(n)^2 + w(n+N)^2 = 1`, which is the textbook window
    /// pair for an MDCT with perfect reconstruction. (This differs from
    /// the §4.3.7 CELT *low-overlap* window — which is applied once on
    /// the synthesis side over only the overlap region — but it is the
    /// canonical symmetric window for exercising the TDAC property of
    /// the transform core, independent of CELT's overlap layout.)
    fn pb_window(two_n: usize) -> Vec<f64> {
        (0..two_n)
            .map(|i| (PI / (two_n as f64) * (i as f64 + 0.5)).sin())
            .collect()
    }

    #[test]
    fn windowed_overlap_add_reconstructs_at_half_gain() {
        // The §4.3.7 TDAC contract: a windowed forward MDCT followed by
        // the inverse MDCT and a windowed overlap-add of two adjacent
        // frames cancels the time-domain aliasing and reconstructs the
        // input — at the RFC's documented `1/2` IMDCT scaling.
        //
        // Analysis: window the 2N-sample block, forward MDCT.
        // Synthesis: inverse MDCT (scaled by 1/2), window again,
        // overlap-add the hop-N neighbours.
        let n = 8;
        let two_n = 2 * n;
        let win = pb_window(two_n);

        // Arbitrary (non-constant) signal spanning two hops (hop = N).
        let total = 3 * n;
        let signal: Vec<f64> = (0..total)
            .map(|i| (i as f64 * 0.37).sin() * 1.5 - (i as f64 * 0.11).cos())
            .collect();

        let analyse_synthesise = |start: usize| -> Vec<f64> {
            let block: Vec<f64> = (0..two_n).map(|i| signal[start + i] * win[i]).collect();
            let coeffs = mdct_forward(&block).unwrap();
            let rec = imdct(&coeffs).unwrap();
            rec.iter().zip(win.iter()).map(|(r, w)| r * w).collect()
        };
        let f0 = analyse_synthesise(0);
        let f1 = analyse_synthesise(n);

        // Overlap region = global samples [N, 2N): f0[N..2N] + f1[0..N].
        // The IMDCT's 1/2 scale carries through, so the reconstruction is
        // exactly half the input there (aliasing fully cancelled).
        for j in 0..n {
            let global = n + j;
            let recon = f0[n + j] + f1[j];
            let want = 0.5 * signal[global];
            assert!(
                (recon - want).abs() < 1e-9,
                "overlap j={j}: recon={recon} != 0.5*signal={want}"
            );
        }
    }

    #[test]
    fn celt_low_overlap_window_is_power_complementary() {
        // Sanity tie-in to the §4.3.7 low-overlap window this transform
        // feeds: the ramp it applies is power-complementary, the property
        // the windowed overlap-add relies on to cancel aliasing.
        let len = 16;
        for nn in 0..len {
            let a = window_tap(nn, len).unwrap();
            let b = window_tap(len - 1 - nn, len).unwrap();
            assert!((a * a + b * b - 1.0).abs() < EPS, "nn={nn}");
        }
    }

    #[test]
    fn dc_spectrum_shape() {
        // A spectrum that is a single DC coefficient produces a smooth
        // cosine-shaped block; verify it is finite and the aliasing fold
        // about N/2 still holds.
        let mut spec = vec![0.0; 16];
        spec[0] = 4.0;
        let y = imdct(&spec).unwrap();
        assert!(y.iter().all(|v| v.is_finite()));
        let n = spec.len();
        for nn in 0..n / 2 {
            assert!((y[nn] + y[n - 1 - nn]).abs() < EPS);
        }
    }

    #[test]
    fn error_display_messages() {
        assert!(ImdctError::ZeroLength.to_string().contains("N >= 1"));
        assert!(ImdctError::OutputLenMismatch { got: 3, want: 4 }
            .to_string()
            .contains("!= required 2*N"));
    }

    #[test]
    fn mdct_forward_rejects_odd_and_empty() {
        assert_eq!(mdct_forward(&[]), Err(ImdctError::ZeroLength));
        assert_eq!(
            mdct_forward(&[1.0, 2.0, 3.0]),
            Err(ImdctError::OutputLenMismatch { got: 3, want: 4 })
        );
    }
}
