//! SILK encoder short-term prediction analysis — RFC 6716 §5.2.3.4.
//!
//! The signal-analysis front half of the SILK encoder estimates a
//! short-term predictor for each SILK frame and whitens the input with
//! it. RFC 6716 §5.2.3.4.2.1 prescribes Burg's method for the LPC
//! estimation because "it provides higher prediction gain than the
//! autocorrelation method and, unlike the covariance method, produces
//! stable filters". This module implements the classic Burg lattice
//! recursion in `f64`:
//!
//! ```text
//!   f_0[i] = b_0[i] = x[i]
//!   for m = 0 .. order-1:
//!             -2 * sum_i f_m[i] * b_m[i-1]
//!     k_m = ---------------------------------      (i over m+1 .. n)
//!            sum_i (f_m[i]^2 + b_m[i-1]^2)
//!     c_{m+1}[j] = c_m[j] + k_m * c_m[m+1-j]       (error filter update)
//!     f_{m+1}[i] = f_m[i]   + k_m * b_m[i-1]
//!     b_{m+1}[i] = b_m[i-1] + k_m * f_m[i]
//! ```
//!
//! The minimisation of the summed forward + backward error energies
//! keeps every reflection coefficient `|k_m| < 1`, so the resulting
//! error filter `A(z) = 1 + c[1] z^-1 + ... + c[d] z^-d` is minimum
//! phase (stable synthesis filter) up to numerical round-off. The
//! prediction coefficients returned follow the §4.2.7.9.2 synthesis
//! convention: `x[i] ≈ sum_k a[k] * x[i-k-1]`, i.e. `a[k] = -c[k+1]`.
//!
//! The exact analysis strategy is an encoder-side freedom (§5.2 is
//! informative); the values produced here only ever reach the
//! bitstream through the normative §4.2.7.5 LSF quantisation.
//!
//! All truth is taken from RFC 6716. No external library source is
//! consulted.

use crate::Error;

/// Hard cap on the analysis order (WB `d_LPC` = 16 is the largest the
/// SILK layer uses; the §5.2.3.2 whitening filter also uses 16).
pub const LPC_ANALYSIS_MAX_ORDER: usize = 16;

/// Estimate an order-`order` short-term predictor from `x` using
/// Burg's method (RFC 6716 §5.2.3.4.2.1).
///
/// Returns `a[0..order]` in the §4.2.7.9.2 prediction convention
/// (`x[i] ≈ sum_k a[k] * x[i-k-1]`). The reflection coefficients are
/// clamped to `(-0.999, 0.999)` before each update so the returned
/// predictor is stable even for pathological inputs (pure sinusoids
/// drive `|k|` arbitrarily close to 1).
///
/// Errors with [`Error::MalformedPacket`] when `order` is zero or
/// above [`LPC_ANALYSIS_MAX_ORDER`], or when `x` is shorter than
/// `2 * order` samples (not enough lag products to form the
/// recursion).
pub fn burg_lpc(x: &[f64], order: usize) -> Result<Vec<f64>, Error> {
    if order == 0 || order > LPC_ANALYSIS_MAX_ORDER || x.len() < 2 * order {
        return Err(Error::MalformedPacket);
    }
    let n = x.len();
    let mut f = x.to_vec();
    let mut b = x.to_vec();
    // Error-filter coefficients c[0..=order], c[0] = 1.
    let mut c = [0.0f64; LPC_ANALYSIS_MAX_ORDER + 1];
    c[0] = 1.0;

    for m in 0..order {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in (m + 1)..n {
            num += f[i] * b[i - 1];
            den += f[i] * f[i] + b[i - 1] * b[i - 1];
        }
        let mut k = if den > 0.0 { -2.0 * num / den } else { 0.0 };
        // Burg guarantees |k| <= 1; clamp defensively away from the
        // unit circle so the filter stays strictly minimum phase.
        k = k.clamp(-0.999, 0.999);

        // Error-filter update (Levinson-style, symmetric).
        let mut c_next = c;
        for (j, slot) in c_next.iter_mut().enumerate().take(m + 2).skip(1) {
            *slot = c[j] + k * c[m + 1 - j];
        }
        c = c_next;

        // Lattice error update, in place, right to left so b[i-1] is
        // still the order-m value when f[i] is updated.
        for i in ((m + 1)..n).rev() {
            let fi = f[i];
            let bi = b[i - 1];
            f[i] = fi + k * bi;
            b[i] = bi + k * fi;
        }
    }

    Ok((1..=order).map(|k| -c[k]).collect())
}

/// Apply the whitening (analysis) filter `e[i] = x[i] - sum_k a[k] *
/// x[i-k-1]` over `x`, with `hist` supplying the samples before
/// `x[0]` (most recent last). Missing history is taken as zero.
///
/// This is the exact inverse of the §4.2.7.9.2 synthesis recursion
/// for the same coefficient convention, used by the §5.2.3.2 pitch
/// front end (whitened signal) and the §5.2.3.4 residual-energy
/// measurement.
pub fn lpc_residual(x: &[f64], hist: &[f64], a: &[f64]) -> Vec<f64> {
    let mut e = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let mut pred = 0.0f64;
        for (k, &ak) in a.iter().enumerate() {
            let idx = i as isize - k as isize - 1;
            let s = if idx >= 0 {
                x[idx as usize]
            } else {
                let h = hist.len() as isize + idx;
                if h >= 0 {
                    hist[h as usize]
                } else {
                    0.0
                }
            };
            pred += ak * s;
        }
        e.push(x[i] - pred);
    }
    e
}

/// Bandwidth-expand a predictor in place: `a[k] *= chirp^(k+1)`.
///
/// Standard analysis conditioning — moves every pole radially toward
/// the origin by `chirp`, guaranteeing margin before the normative
/// §4.2.7.5 quantisation reshapes the filter anyway.
pub fn bandwidth_expand(a: &mut [f64], chirp: f64) {
    let mut c = chirp;
    for ak in a.iter_mut() {
        *ak *= c;
        c *= chirp;
    }
}

/// Sum of squares helper for the per-subframe residual-energy
/// measurements (§5.2.3.4: "filter the input signal and measure
/// residual energy for each of the four subframes").
pub fn energy(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random generator (unit-variance-ish white
    /// noise substitute) so the tests need no external crates.
    struct Lcg(u32);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            // Same LCG constants as §4.2.7.7/§4.2.7.8.6 (a documented
            // full-period 32-bit LCG), used here only as a test noise
            // source.
            self.0 = self.0.wrapping_mul(196_314_165).wrapping_add(907_633_515);
            (self.0 >> 8) as f64 / (1u32 << 24) as f64 - 0.5
        }
    }

    /// Drive a known AR(2) synthesis filter with noise and check Burg
    /// recovers the coefficients.
    #[test]
    fn burg_recovers_ar2() {
        let a_true = [1.6, -0.81]; // poles at 0.9 e^{±j pi/6}-ish, stable
        let mut rng = Lcg(1);
        let mut x = vec![0.0f64; 2048];
        for i in 2..x.len() {
            x[i] = a_true[0] * x[i - 1] + a_true[1] * x[i - 2] + rng.next_f64();
        }
        let a = burg_lpc(&x[512..], 2).unwrap();
        assert!((a[0] - a_true[0]).abs() < 0.05, "a0 = {}", a[0]);
        assert!((a[1] - a_true[1]).abs() < 0.05, "a1 = {}", a[1]);
    }

    /// Higher-order fit of the same AR(2): extra taps go ~0 and the
    /// residual is ~the driving noise (prediction gain check).
    #[test]
    fn burg_prediction_gain_on_ar_signal() {
        let mut rng = Lcg(7);
        let mut x = vec![0.0f64; 4096];
        for i in 2..x.len() {
            x[i] = 1.6 * x[i - 1] - 0.81 * x[i - 2] + rng.next_f64();
        }
        let seg = &x[1024..];
        let a = burg_lpc(seg, 10).unwrap();
        let res = lpc_residual(seg, &[], &a);
        let gain = energy(seg) / energy(&res).max(1e-30);
        // AR(2) with these poles has ~13 dB prediction gain; require a
        // healthy chunk of it.
        assert!(gain > 10.0, "prediction gain {gain}");
    }

    /// A pure sinusoid must not blow up the recursion (|k| clamp) and
    /// must still yield a large prediction gain.
    #[test]
    fn burg_on_pure_sine_is_stable() {
        let x: Vec<f64> = (0..640).map(|i| (i as f64 * 0.19).sin() * 0.7).collect();
        let a = burg_lpc(&x, 16).unwrap();
        assert!(a.iter().all(|v| v.is_finite()));
        let res = lpc_residual(&x, &[], &a);
        let gain = energy(&x) / energy(&res).max(1e-30);
        // The |k| < 0.999 stability clamp deliberately caps the
        // achievable gain on a pure sinusoid; ~67x with this clamp.
        assert!(gain > 50.0, "sine prediction gain {gain}");
    }

    /// Whitening an AR signal recovers the driving noise: residual of
    /// the true filter equals the injected excitation exactly.
    #[test]
    fn lpc_residual_inverts_synthesis() {
        let a = [0.9f64, -0.2, 0.05];
        let mut rng = Lcg(3);
        let noise: Vec<f64> = (0..256).map(|_| rng.next_f64()).collect();
        let mut x = vec![0.0f64; 256];
        for i in 0..x.len() {
            let mut v = noise[i];
            for (k, &ak) in a.iter().enumerate() {
                if i > k {
                    v += ak * x[i - k - 1];
                }
            }
            x[i] = v;
        }
        let res = lpc_residual(&x, &[], &a);
        for (r, n) in res.iter().zip(noise.iter()) {
            assert!((r - n).abs() < 1e-12);
        }
    }

    /// History samples are honoured: splitting a signal in two and
    /// passing the first half as history gives the same residual as
    /// one whole-signal call.
    #[test]
    fn lpc_residual_history_seam() {
        let a = [1.2f64, -0.5, 0.1, -0.02];
        let mut rng = Lcg(9);
        let x: Vec<f64> = (0..200).map(|_| rng.next_f64()).collect();
        let whole = lpc_residual(&x, &[], &a);
        let head = lpc_residual(&x[..120], &[], &a);
        let tail = lpc_residual(&x[120..], &x[..120], &a);
        assert_eq!(&whole[..120], &head[..]);
        for (w, t) in whole[120..].iter().zip(tail.iter()) {
            assert!((w - t).abs() < 1e-12);
        }
    }

    #[test]
    fn bandwidth_expand_shrinks_taps() {
        let mut a = [1.0f64, 1.0, 1.0];
        bandwidth_expand(&mut a, 0.5);
        assert!((a[0] - 0.5).abs() < 1e-15);
        assert!((a[1] - 0.25).abs() < 1e-15);
        assert!((a[2] - 0.125).abs() < 1e-15);
    }

    #[test]
    fn burg_rejects_bad_args() {
        let x = vec![0.0f64; 64];
        assert!(burg_lpc(&x, 0).is_err());
        assert!(burg_lpc(&x, LPC_ANALYSIS_MAX_ORDER + 1).is_err());
        assert!(burg_lpc(&x[..8], 16).is_err());
    }
}
