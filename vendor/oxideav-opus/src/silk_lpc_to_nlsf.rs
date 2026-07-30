//! LPC → normalized-LSF conversion for the SILK encoder — RFC 6716
//! §5.2.3.4 ("The estimated LPC coefficients are converted to a Line
//! Spectral Frequency (LSF) vector and quantized").
//!
//! This is the analysis-direction inverse of the normative §4.2.7.5.6
//! NLSF → LPC reconstruction implemented in
//! [`crate::silk_lsf_to_lpc`]. Given a prediction filter `a[]` in the
//! §4.2.7.9.2 convention (`x[i] ≈ sum_k a[k]*x[i-k-1]`, so the error
//! filter is `A(z) = 1 - sum_k a[k] z^-(k+1)`), form the symmetric /
//! antisymmetric line-spectral polynomials
//!
//! ```text
//!   P(z) = A(z) + z^-(d+1) * A(1/z)        (root at z = -1)
//!   Q(z) = A(z) - z^-(d+1) * A(1/z)        (root at z = +1)
//! ```
//!
//! deflate the trivial roots (`P' = P/(1+z^-1)`, `Q' = Q/(1-z^-1)`),
//! and locate the remaining `d/2 + d/2` roots, all of which lie on the
//! unit circle at angles `0 < ω_1 < ω_2 < ... < ω_d < π` when `A(z)`
//! is minimum phase. A symmetric deflated polynomial of even degree
//! `d` evaluates on the circle as the real cosine series
//!
//! ```text
//!   G(ω) = p'[d/2] + 2 * sum_{k=1}^{d/2} p'[d/2 - k] * cos(k ω)
//! ```
//!
//! whose sign changes over a fine ω grid bracket each root; bisection
//! refines them. The decode-side Table 27 ordering assigns the
//! even-indexed `c_Q17[]` slots (the P polynomial) to the
//! even-numbered positions of the sorted NLSF vector and the
//! odd-indexed slots (Q) to the odd positions, so the roots of P' and
//! Q' must strictly interleave with a P' root first — this module
//! verifies that interleaving and errors out (letting the caller fall
//! back) if numerical trouble breaks it.
//!
//! The output is the Q15 normalized LSF vector `NLSF_Q15[k] =
//! round(ω_k / π * 32768)` that feeds the §4.2.7.5 quantiser.
//!
//! All truth is taken from RFC 6716 §4.2.7.5.6 (conventions) and
//! standard line-spectral-pair theory. No external library source is
//! consulted.

use crate::silk_lsf_stage2::{D_LPC_MAX, D_LPC_NB_MB, D_LPC_WB};
use crate::Error;
use core::f64::consts::PI;

/// Number of initial grid cells used to bracket the roots of each
/// cosine-series polynomial over `(0, π)`.
const ROOT_GRID: usize = 512;

/// Bisection refinement steps per bracketed root (halves the bracket
/// each step; 46 steps ≪ f64 precision but cheap).
const ROOT_BISECTIONS: usize = 46;

/// Convert a prediction filter `a[]` (length 10 or 16, §4.2.7.9.2
/// convention) into the sorted Q15 normalized-LSF vector.
///
/// Returns [`Error::MalformedPacket`] when the order is unsupported,
/// or when the line-spectral root structure of the filter cannot be
/// resolved (wrong root count or broken P/Q interleaving — both only
/// happen when `A(z)` is not minimum phase or roots collide beyond
/// f64 resolution). Callers treat that as "re-condition the filter
/// and retry" (e.g. more bandwidth expansion).
pub fn lpc_to_nlsf_q15(a: &[f64]) -> Result<Vec<i16>, Error> {
    let d = a.len();
    if d != D_LPC_NB_MB && d != D_LPC_WB {
        return Err(Error::MalformedPacket);
    }
    let d2 = d / 2;

    // Error-filter coefficients A[0..=d]: A(z) = 1 - sum a[k] z^-(k+1).
    // One extra zero cell so the A[d+1-i] reflection below can address
    // the (always zero) degree-(d+1) term.
    let mut ac = [0.0f64; D_LPC_MAX + 2];
    ac[0] = 1.0;
    for (k, &ak) in a.iter().enumerate() {
        ac[k + 1] = -ak;
    }

    // P[i] = A[i] + A[d+1-i], Q[i] = A[i] - A[d+1-i], degree d+1.
    // Deflate: P' = P / (1 + z^-1), Q' = Q / (1 - z^-1), degree d,
    // both symmetric. Only the first d/2 + 1 coefficients matter.
    let mut p = [0.0f64; D_LPC_MAX + 1];
    let mut q = [0.0f64; D_LPC_MAX + 1];
    {
        let mut p_full = [0.0f64; D_LPC_MAX + 2];
        let mut q_full = [0.0f64; D_LPC_MAX + 2];
        for i in 0..=(d + 1) {
            p_full[i] = ac[i] + ac[d + 1 - i];
            q_full[i] = ac[i] - ac[d + 1 - i];
        }
        // Synthetic division by (1 + z^-1) and (1 - z^-1).
        p[0] = p_full[0];
        q[0] = q_full[0];
        for i in 1..=d {
            p[i] = p_full[i] - p[i - 1];
            q[i] = q_full[i] + q[i - 1];
        }
    }

    // Locate the d/2 roots of each deflated polynomial's cosine series.
    let p_roots = cosine_series_roots(&p[..=d2], d2)?;
    let q_roots = cosine_series_roots(&q[..=d2], d2)?;

    // Interleave: P root first, then Q, alternating (Table 27 parity).
    let mut nlsf = Vec::with_capacity(d);
    let mut prev = 0.0f64;
    for k in 0..d2 {
        for &w in &[p_roots[k], q_roots[k]] {
            if w <= prev {
                // Broken interleaving — the filter's line spectrum is
                // not resolvable; let the caller re-condition.
                return Err(Error::MalformedPacket);
            }
            prev = w;
            let v = (w / PI * 32768.0).round();
            nlsf.push(v.clamp(1.0, 32767.0) as i16);
        }
    }
    // Rounding to Q15 may merge ultra-close neighbours; enforce strict
    // monotonicity in the integer domain (the §4.2.7.5.4 stabilizer
    // will re-space them anyway, but the quantiser expects sorted
    // input).
    for k in 1..nlsf.len() {
        if nlsf[k] <= nlsf[k - 1] {
            nlsf[k] = (nlsf[k - 1] as i32 + 1).min(32767) as i16;
        }
    }
    Ok(nlsf)
}

/// Evaluate `G(ω) = c[d2] + 2 * sum_{k=1}^{d2} c[d2-k] cos(kω)` — the
/// unit-circle restriction of a symmetric polynomial with first-half
/// coefficients `c[0..=d2]`.
fn eval_cosine_series(c: &[f64], d2: usize, w: f64) -> f64 {
    let mut acc = c[d2];
    for k in 1..=d2 {
        acc += 2.0 * c[d2 - k] * (k as f64 * w).cos();
    }
    acc
}

/// Find exactly `d2` roots of the cosine series over `(0, π)` by grid
/// bracketing + bisection. Errors when the sign-change count is not
/// `d2` (non-minimum-phase source filter or colliding roots).
fn cosine_series_roots(c: &[f64], d2: usize) -> Result<Vec<f64>, Error> {
    let mut roots = Vec::with_capacity(d2);
    let step = PI / ROOT_GRID as f64;
    let mut w0 = 1e-9f64;
    let mut g0 = eval_cosine_series(c, d2, w0);
    for i in 1..=ROOT_GRID {
        let w1 = if i == ROOT_GRID {
            PI - 1e-9
        } else {
            i as f64 * step
        };
        let g1 = eval_cosine_series(c, d2, w1);
        if g0 == 0.0 {
            roots.push(w0);
        } else if g0 * g1 < 0.0 {
            // Bisect the bracket.
            let (mut lo, mut hi, mut glo) = (w0, w1, g0);
            for _ in 0..ROOT_BISECTIONS {
                let mid = 0.5 * (lo + hi);
                let gm = eval_cosine_series(c, d2, mid);
                if glo * gm <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    glo = gm;
                }
            }
            roots.push(0.5 * (lo + hi));
        }
        if roots.len() > d2 {
            return Err(Error::MalformedPacket);
        }
        w0 = w1;
        g0 = g1;
    }
    if roots.len() != d2 {
        return Err(Error::MalformedPacket);
    }
    Ok(roots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk_lsf_recon::cb1_q8;
    use crate::silk_lsf_to_lpc::LpcQ17;
    use crate::toc::Bandwidth;

    /// A(z) = 1 (zero predictor): the line spectrum is analytically
    /// uniform, ω_k = (k+1) π / (d+1).
    #[test]
    fn zero_predictor_gives_uniform_spectrum() {
        for d in [10usize, 16] {
            let a = vec![0.0f64; d];
            let nlsf = lpc_to_nlsf_q15(&a).unwrap();
            for (k, &v) in nlsf.iter().enumerate() {
                let expect = 32768.0 * (k + 1) as f64 / (d + 1) as f64;
                assert!(
                    (v as f64 - expect).abs() < 2.0,
                    "d={d} k={k}: {v} vs {expect}"
                );
            }
        }
    }

    /// Roundtrip through the normative decode chain: every stage-1
    /// codebook vector → Q12 LPC (§4.2.7.5.6-.8) → back to NLSF. The
    /// Q12 rounding and the Q15 grid perturb the roots slightly; the
    /// recovered vector must stay close and sorted. (Measured maxima:
    /// 9 Q15 units NB, 14 WB.)
    #[test]
    fn roundtrips_stage1_codebook_vectors() {
        for bw in [Bandwidth::Nb, Bandwidth::Wb] {
            for i1 in 0..32u8 {
                let cb = cb1_q8(bw, i1).unwrap();
                let nlsf0: Vec<i16> = cb.iter().map(|&v| (v as i16) << 7).collect();
                let lpc = LpcQ17::from_nlsf(bw, &nlsf0)
                    .unwrap()
                    .range_limited()
                    .prediction_gain_limited();
                let a: Vec<f64> = lpc.a_q12().iter().map(|&v| v as f64 / 4096.0).collect();
                let nlsf = lpc_to_nlsf_q15(&a).unwrap();
                assert_eq!(nlsf.len(), nlsf0.len());
                for (k, (&r, &o)) in nlsf.iter().zip(nlsf0.iter()).enumerate() {
                    assert!(
                        (r as i32 - o as i32).abs() < 40,
                        "bw={bw:?} I1={i1} k={k}: {r} vs {o}"
                    );
                }
                for w in nlsf.windows(2) {
                    assert!(w[0] < w[1]);
                }
            }
        }
    }

    /// The conversion output feeds the decode chain directly: convert,
    /// reconstruct Q12 LPC from the converted NLSF, and confirm the
    /// filter response barely moved (coefficient-domain check).
    #[test]
    fn converted_nlsf_reconstructs_similar_filter() {
        // A gentle low-pass-ish predictor, then decode-side sanitise.
        let mut a = vec![0.0f64; 16];
        a[0] = 1.1;
        a[1] = -0.4;
        a[2] = 0.12;
        a[3] = -0.05;
        let nlsf = lpc_to_nlsf_q15(&a).unwrap();
        let lpc = LpcQ17::from_nlsf(Bandwidth::Wb, &nlsf)
            .unwrap()
            .range_limited()
            .prediction_gain_limited();
        for (k, &q) in lpc.a_q12().iter().enumerate() {
            let back = q as f64 / 4096.0;
            assert!(
                (back - a[k]).abs() < 0.02,
                "k={k}: {back} vs {}, nlsf={nlsf:?}",
                a[k]
            );
        }
    }

    #[test]
    fn rejects_unsupported_order() {
        assert!(lpc_to_nlsf_q15(&[0.0; 8]).is_err());
        assert!(lpc_to_nlsf_q15(&[0.0; 12]).is_err());
    }

    /// A strongly non-minimum-phase filter must error (root structure
    /// unresolvable), never panic.
    #[test]
    fn non_minimum_phase_errors_cleanly() {
        let mut a = vec![0.0f64; 10];
        a[0] = 3.5; // pole far outside the unit circle
        let r = lpc_to_nlsf_q15(&a);
        assert!(r.is_err());
    }
}
