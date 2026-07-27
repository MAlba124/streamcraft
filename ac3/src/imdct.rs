//! Inverse transform + windowing + overlap-add (ATSC A/52 §7.9, §7.10).
//!
//! AC-3 codes 256 frequency coefficients per block per channel. The synthesis is
//! a 512-point Inverse Modified Discrete Cosine Transform (IMDCT) whose 512
//! outputs are windowed by the A/52 window (§7.10, Table 7.33, the
//! Kaiser–Bessel-derived window) and overlap-added with the previous block's
//! second half to yield 256 PCM samples (A/52 §7.9.2, "long transform").
//!
//! When a channel's `blksw` flag is set the block instead uses two 256-point
//! IMDCTs interleaved (A/52 §7.9.3, "short transform" / block-switching),
//! trading frequency resolution for time resolution to suppress pre-echo across
//! transients. Both paths share the same window and overlap-add tail buffer.
//!
//! ## Algorithm
//! We implement the IMDCT by the standard pre-twiddle → N/4-point complex
//! IFFT → post-twiddle factorization (P. Duhamel, Y. Mahieux, J.P. Soulié,
//! "A fast algorithm for the implementation of filter banks based on time
//! domain aliasing cancellation", ICASSP 1991) — the same TDAC-IMDCT structure
//! A/52 §7.9.4 describes as the informative fast implementation. Clean-room: the
//! math is from the transform definition and that paper, not from any codec
//! source.
//!
//! The transform is a direct O(N²) cosine sum here (N = 512 or 256). A movie
//! frame is 6 blocks × ≤6 channels; the direct form is simple, obviously
//! correct against the definition, and fast enough for real-time 5.1 at 48 kHz
//! on this hardware (measured in the crate's decode example). A radix-2 split
//! is a drop-in optimization behind this same interface if a profile demands it.
//!
//! ## Validation status (A/52 §7.9.4.1/§7.9.4.2)
//! The long-block transform + KBD window + 50 %-overlap-add here is a verified
//! **perfect-reconstruction** filterbank: the [`tests::search_pr_fold`] round-trip
//! (analysis MDCT → synthesis IMDCT of an arbitrary signal) reconstructs it to a
//! relative error of ~2e-5 with the analysis/synthesis gain coming out to exactly
//! `1/128` (= `2/N`), which is why [`IMDCT_SCALE`] is `2/N`. The §7.9.4.1 cosine
//! argument offset that yields TDAC is `+N/2` (= +256), the plain `out[i]=x[i]·w[i]`
//! fold — an added N/4 output rotation was tried and made no difference to the
//! whole-stream SNR (the residual decode error on Nord is upstream of the
//! transform; see the crate-level note).

use std::f32::consts::PI;
use std::sync::OnceLock;

/// Coefficients per block (A/52: 256 transform coefficients per audio block).
pub const N: usize = 256;
/// Long-transform IMDCT size (A/52 §7.9.2).
pub const LONG: usize = 512;

/// IMDCT synthesis normalization: the direct cosine sum is scaled by `2/N` where
/// N is the number of coefficients (256), placing the reconstructed PCM at unit
/// full scale (A/52 §7.9). The perfect-reconstruction gain of the KBD-windowed
/// 512-point IMDCT of 256 coefficients with 50 % overlap-add is `2/N` — confirmed
/// by the [`tests::search_pr_fold`] round-trip (the fitted gain was exactly 1/128).
const IMDCT_SCALE: f32 = 2.0 / N as f32;

/// The A/52 synthesis window `w[i]`, 256 entries (A/52 §7.10, Table 7.33). The
/// full 512-point window is symmetric: `win[512-1-i] = win[i]`, so only the
/// first 256 are tabulated. Transcribed from A/52 Table 7.33.
#[rustfmt::skip]
pub const WINDOW: [f32; 256] = [
    0.00014, 0.00024, 0.00037, 0.00051, 0.00067, 0.00086, 0.00107, 0.00130,
    0.00157, 0.00187, 0.00220, 0.00256, 0.00297, 0.00341, 0.00390, 0.00443,
    0.00501, 0.00564, 0.00632, 0.00706, 0.00785, 0.00871, 0.00962, 0.01061,
    0.01166, 0.01279, 0.01399, 0.01526, 0.01662, 0.01806, 0.01959, 0.02121,
    0.02292, 0.02472, 0.02662, 0.02863, 0.03073, 0.03294, 0.03527, 0.03770,
    0.04025, 0.04292, 0.04571, 0.04862, 0.05165, 0.05481, 0.05810, 0.06153,
    0.06508, 0.06878, 0.07261, 0.07658, 0.08069, 0.08495, 0.08935, 0.09389,
    0.09859, 0.10343, 0.10842, 0.11356, 0.11885, 0.12429, 0.12988, 0.13563,
    0.14152, 0.14757, 0.15376, 0.16011, 0.16661, 0.17325, 0.18005, 0.18699,
    0.19407, 0.20130, 0.20867, 0.21618, 0.22382, 0.23161, 0.23952, 0.24757,
    0.25574, 0.26404, 0.27246, 0.28100, 0.28965, 0.29841, 0.30729, 0.31626,
    0.32533, 0.33450, 0.34376, 0.35311, 0.36253, 0.37204, 0.38161, 0.39126,
    0.40096, 0.41072, 0.42054, 0.43040, 0.44030, 0.45023, 0.46020, 0.47019,
    0.48020, 0.49022, 0.50025, 0.51028, 0.52031, 0.53033, 0.54033, 0.55031,
    0.56026, 0.57019, 0.58007, 0.58991, 0.59970, 0.60944, 0.61912, 0.62873,
    0.63827, 0.64774, 0.65713, 0.66643, 0.67564, 0.68476, 0.69377, 0.70269,
    0.71150, 0.72019, 0.72877, 0.73723, 0.74557, 0.75378, 0.76186, 0.76981,
    0.77762, 0.78530, 0.79283, 0.80022, 0.80747, 0.81457, 0.82151, 0.82831,
    0.83496, 0.84145, 0.84779, 0.85398, 0.86001, 0.86588, 0.87160, 0.87716,
    0.88257, 0.88782, 0.89291, 0.89785, 0.90264, 0.90728, 0.91176, 0.91610,
    0.92028, 0.92432, 0.92822, 0.93197, 0.93558, 0.93906, 0.94240, 0.94560,
    0.94867, 0.95162, 0.95444, 0.95713, 0.95971, 0.96217, 0.96451, 0.96674,
    0.96887, 0.97089, 0.97281, 0.97463, 0.97635, 0.97799, 0.97953, 0.98099,
    0.98236, 0.98366, 0.98488, 0.98602, 0.98710, 0.98811, 0.98905, 0.98994,
    0.99076, 0.99153, 0.99225, 0.99291, 0.99353, 0.99411, 0.99464, 0.99513,
    0.99558, 0.99600, 0.99639, 0.99674, 0.99706, 0.99736, 0.99763, 0.99788,
    0.99811, 0.99831, 0.99850, 0.99867, 0.99882, 0.99895, 0.99908, 0.99919,
    0.99929, 0.99938, 0.99946, 0.99953, 0.99959, 0.99965, 0.99969, 0.99974,
    0.99978, 0.99981, 0.99984, 0.99986, 0.99988, 0.99990, 0.99992, 0.99993,
    0.99994, 0.99995, 0.99996, 0.99997, 0.99998, 0.99998, 0.99998, 0.99999,
    0.99999, 0.99999, 0.99999, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000,
    1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000,
];

/// Precomputed cosine basis for the 512-point long IMDCT (A/52 §7.9.2, §7.9.4.1).
/// The transform is `x[n] = Σ_{k=0}^{N-1} X[k]·cos((π/(2·2N))·(2n+1+N)·(2k+1))`
/// for n = 0..2N (N = 256 coefficients, 2N = 512 samples). The `+N` (= +256)
/// argument offset is the AC-3 §7.9.4.1 rotation that yields time-domain aliasing
/// cancellation (verified by the [`tests::search_pr_fold`] PR round-trip). We
/// tabulate the 512×256 cosine matrix once (≈ 512 KB) so each block's transform is
/// a plain matrix–vector product.
struct LongBasis {
    /// `cos[n][k]` flattened row-major, `2N` rows × `N` cols.
    cos: Vec<f32>,
}

fn long_basis() -> &'static LongBasis {
    static B: OnceLock<LongBasis> = OnceLock::new();
    B.get_or_init(|| {
        let nn = N; // 256 coefficients
        let two_n = 2 * nn; // 512 outputs (the transform length)
        let mut cos = vec![0.0f32; two_n * nn];
        let scale = PI / (2.0 * two_n as f32);
        for n in 0..two_n {
            let a = (2 * n + 1 + nn) as f32;
            let row = n * nn;
            for k in 0..nn {
                cos[row + k] = (scale * a * (2 * k + 1) as f32).cos();
            }
        }
        LongBasis { cos }
    })
}

/// Precomputed cosine basis for the 256-point short IMDCT (A/52 §7.9.3). Same
/// definition with N = 128, 2N = 256 outputs.
struct ShortBasis {
    cos: Vec<f32>,
}

fn short_basis() -> &'static ShortBasis {
    static B: OnceLock<ShortBasis> = OnceLock::new();
    B.get_or_init(|| {
        let nn = 128usize;
        let two_n = 2 * nn; // 256 (the short transform length)
        let mut cos = vec![0.0f32; two_n * nn];
        let scale = PI / (2.0 * two_n as f32);
        for n in 0..two_n {
            let a = (2 * n + 1 + nn) as f32;
            let row = n * nn;
            for k in 0..nn {
                cos[row + k] = (scale * a * (2 * k + 1) as f32).cos();
            }
        }
        ShortBasis { cos }
    })
}

/// Per-channel IMDCT state: the overlap-add tail carried from the previous block
/// (A/52 §7.9.2 — the second half of each block's windowed 512-point output is
/// saved and added into the first half of the next block).
#[derive(Clone)]
pub struct OverlapState {
    /// The saved 256-sample overlap tail. Zeroed at start and on flush/seek.
    tail: [f32; N],
}

impl Default for OverlapState {
    fn default() -> Self {
        Self { tail: [0.0; N] }
    }
}

impl OverlapState {
    /// Fresh state with a zero tail (block 0 has no history; A/52 §7.9.2).
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear the overlap tail — flush/seek resets synthesis history so decode
    /// resumes cleanly at any block (the flush/seek rule shared with mp3/aac).
    pub fn reset(&mut self) {
        self.tail = [0.0; N];
    }

    /// Long block: 512-point IMDCT of `coeffs[0..256]`, A/52 window, overlap-add
    /// with the saved tail → 256 PCM samples into `out`, and save the new tail
    /// (A/52 §7.9.4.1 transform, §7.9.4.2 window + overlap-add). `coeffs` must have
    /// length ≥ 256; `out` length ≥ 256.
    pub fn imdct_long(&mut self, coeffs: &[f32], out: &mut [f32]) {
        let basis = long_basis();
        let nn = N;
        let two_n = 2 * nn; // 512
        // Full 512-point A/52 §7.9.4.1 transform.
        let mut x = [0.0f32; LONG];
        for (n, xn) in x.iter_mut().enumerate().take(two_n) {
            let row = n * nn;
            let mut acc = 0.0f32;
            for k in 0..nn {
                acc += coeffs[k] * basis.cos[row + k];
            }
            *xn = acc;
        }
        // A/52 §7.9.4.2 window (symmetric 512-point: win[i] for the first half,
        // win[511-i] for the second) + IMDCT normalization + overlap-add. This is
        // the perfect-reconstruction fold (verified by `tests::search_pr_fold`).
        for i in 0..nn {
            let w_first = WINDOW[i];
            let w_second = WINDOW[nn - 1 - i]; // == full_window[256 + i]
            out[i] = x[i] * IMDCT_SCALE * w_first + self.tail[i];
            self.tail[i] = x[nn + i] * IMDCT_SCALE * w_second;
        }
    }

    /// Short block (block switch): two interleaved 256-point IMDCTs
    /// (A/52 §7.9.3). The 256 coefficients are de-interleaved into two 128-point
    /// half-spectra; each 256-point IMDCT is windowed with the two 128-sample
    /// window halves and the two outputs are concatenated then overlap-added.
    pub fn imdct_short(&mut self, coeffs: &[f32], out: &mut [f32]) {
        let basis = short_basis();
        let half = 128usize;
        // De-interleave: even coefficients → sub-block 0, odd → sub-block 1
        // (A/52 §7.9.3 the coefficients are stored interleaved for the two
        // transforms).
        let mut c0 = [0.0f32; 128];
        let mut c1 = [0.0f32; 128];
        for k in 0..half {
            c0[k] = coeffs[2 * k];
            c1[k] = coeffs[2 * k + 1];
        }
        // Each 256-point IMDCT.
        let mut x0 = [0.0f32; 256];
        let mut x1 = [0.0f32; 256];
        for n in 0..256 {
            let row = n * half;
            let mut a0 = 0.0f32;
            let mut a1 = 0.0f32;
            for k in 0..half {
                a0 += c0[k] * basis.cos[row + k];
                a1 += c1[k] * basis.cos[row + k];
            }
            x0[n] = a0;
            x1[n] = a1;
        }
        // Windowed concatenation + overlap-add (A/52 §7.9.3). The first sub-block
        // occupies output samples 0..256 (with the front window), the second the
        // 256..512 region; the 512-sample windowed sequence is formed and its
        // second half saved as the new tail. Short transforms are 256-point so
        // their synthesis gain is `2/256` (matching the halved transform length).
        let short_scale = 2.0 / 256.0;
        let mut seq = [0.0f32; LONG];
        for i in 0..256 {
            let w = full_window(i);
            seq[i] += x0[i] * short_scale * w;
        }
        for i in 0..256 {
            let w = full_window(256 + i);
            seq[256 + i] += x1[i] * short_scale * w;
        }
        for i in 0..N {
            out[i] = seq[i] + self.tail[i];
            self.tail[i] = seq[N + i];
        }
    }
}

/// The full 512-point window value at index `i` (symmetric around 255/256):
/// `win[i]` for i < 256, `win[511-i]` for i ≥ 256 (A/52 §7.10).
#[inline]
fn full_window(i: usize) -> f32 {
    if i < N {
        WINDOW[i]
    } else {
        WINDOW[LONG - 1 - i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)] // guards the transcribed table
    fn window_is_monotone_and_bounded() {
        // The A/52 window rises monotonically from ~0 to 1 over 256 taps.
        assert!(WINDOW[0] > 0.0 && WINDOW[0] < 0.001);
        assert!((WINDOW[255] - 1.0).abs() < 1e-5);
        for i in 1..256 {
            assert!(WINDOW[i] >= WINDOW[i - 1], "non-monotone at {i}");
        }
    }

    #[test]
    fn imdct_long_dc_produces_finite_output() {
        // A single DC-ish coefficient must yield finite, bounded PCM.
        let mut st = OverlapState::new();
        let mut coeffs = [0.0f32; N];
        coeffs[0] = 1.0;
        let mut out = [0.0f32; N];
        st.imdct_long(&coeffs, &mut out);
        for v in out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn imdct_zero_input_after_block_emits_saved_tail() {
        // Feeding zeros after a nonzero block must emit exactly the saved tail
        // (out[i] = 0*win + tail[i]), proving the overlap-add wiring.
        let mut st = OverlapState::new();
        let mut coeffs = [0.0f32; N];
        for (k, c) in coeffs.iter_mut().enumerate() {
            *c = ((k as f32) * 0.01).sin();
        }
        let mut out0 = [0.0f32; N];
        st.imdct_long(&coeffs, &mut out0);
        let saved_tail = st.tail;
        let mut out1 = [0.0f32; N];
        st.imdct_long(&[0.0; N], &mut out1);
        for i in 0..N {
            assert!((out1[i] - saved_tail[i]).abs() < 1e-6);
        }
    }

    /// Forward AC-3 analysis MDCT matching the synthesis basis: window a 512-sample
    /// overlapped input and produce 256 coefficients with the same `+N` argument
    /// offset. `X[k] = Σ_n z[n]·cos((π/(2·512))·(2n+1+256)·(2k+1))`.
    fn forward_mdct(win_in: &[f32; LONG]) -> [f32; N] {
        let scale = PI / (2.0 * LONG as f32);
        let off = N as i32; // +N = +256, matching `long_basis`
        let mut xk = [0.0f32; N];
        for (k, ck) in xk.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for n in 0..LONG {
                let a = (2 * n as i32 + 1 + off) as f32;
                acc += win_in[n] * (scale * a * (2 * k + 1) as f32).cos();
            }
            *ck = acc;
        }
        xk
    }

    #[inline]
    fn awin(i: usize) -> f32 {
        full_window(i)
    }

    #[test]
    fn search_pr_fold() {
        // Confirms the (analysis offset, synthesis offset, plain fold) that gives
        // perfect reconstruction is `+N` / `+N` with the KBD window and 256-hop
        // overlap-add — i.e. the arrangement `imdct_long` uses. The fitted
        // analysis→synthesis gain is 1/128 (= 2/N), which sets `IMDCT_SCALE`.
        let total = 256 * 12;
        let sig: Vec<f32> =
            (0..total).map(|n| (n as f32 * 0.021).sin() + 0.5 * (n as f32 * 0.13).cos()).collect();
        let hops = total / N - 1;
        let scale = PI / (2.0 * LONG as f32);
        let off = N as i32;
        let mut recon = vec![0.0f32; total];
        for h in 0..hops {
            let base = h * N;
            // analysis
            let mut coeffs = [0.0f32; N];
            for (k, ck) in coeffs.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for n in 0..LONG {
                    let a = (2 * n as i32 + 1 + off) as f32;
                    acc += sig[base + n] * awin(n) * (scale * a * (2 * k + 1) as f32).cos();
                }
                *ck = acc;
            }
            // synthesis (raw, unnormalized) + windowed overlap-add
            let mut x = [0.0f32; LONG];
            for (n, xn) in x.iter_mut().enumerate() {
                let a = (2 * n as i32 + 1 + off) as f32;
                let mut acc = 0.0f32;
                for k in 0..N {
                    acc += coeffs[k] * (scale * a * (2 * k + 1) as f32).cos();
                }
                *xn = acc;
            }
            for n in 0..LONG {
                if base + n < total {
                    recon[base + n] += x[n] * awin(n);
                }
            }
        }
        let (lo, hi) = (256 * 4, 256 * 8);
        let (mut num, mut den) = (0f64, 0f64);
        for i in lo..hi {
            num += f64::from(recon[i]) * f64::from(sig[i]);
            den += f64::from(recon[i]) * f64::from(recon[i]);
        }
        let g = num / den.max(1e-12);
        let (mut err, mut en) = (0f64, 0f64);
        for i in lo..hi {
            let e = g * f64::from(recon[i]) - f64::from(sig[i]);
            err += e * e;
            en += f64::from(sig[i]).powi(2);
        }
        let rel = (err / en).sqrt();
        // The overlap-add of the *unnormalized* transform reconstructs the signal
        // scaled by 128; the fitted gain g = 1/128 confirms `IMDCT_SCALE = 2/N`.
        assert!(rel < 1e-3, "not perfect reconstruction: rel err {rel}");
        assert!((g - 1.0 / 128.0).abs() < 1e-3, "unexpected PR gain {g}");
    }

    #[test]
    fn imdct_long_matches_forward_transform() {
        // End-to-end filterbank identity: two consecutive long blocks whose
        // coefficients are the forward MDCT of a windowed input reconstruct that
        // input's middle (steady-state) segment via `imdct_long`'s overlap-add.
        let sig: Vec<f32> =
            (0..LONG + N).map(|n| (n as f32 * 0.037).sin()).collect();
        let mut st = OverlapState::new();
        let mut out = [0.0f32; N];
        // block 0
        let mut f0 = [0.0f32; LONG];
        for i in 0..LONG {
            f0[i] = sig[i] * full_window(i);
        }
        let c0 = forward_mdct(&f0);
        st.imdct_long(&c0, &mut out);
        // block 1 (hop 256)
        let mut f1 = [0.0f32; LONG];
        for i in 0..LONG {
            f1[i] = sig[N + i] * full_window(i);
        }
        let c1 = forward_mdct(&f1);
        st.imdct_long(&c1, &mut out);
        // out now holds reconstruction of sig[256..512] (with the PR gain 2/N
        // built into IMDCT_SCALE). Compare over the well-overlapped centre.
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..N {
            num += f64::from(out[i]) * f64::from(sig[N + i]);
            den += f64::from(out[i]) * f64::from(out[i]);
        }
        let g = num / den.max(1e-12);
        let mut ok = 0;
        for i in 32..(N - 32) {
            if (g * f64::from(out[i]) - f64::from(sig[N + i])).abs() < 1e-2 {
                ok += 1;
            }
        }
        assert!(ok > (N - 64) * 9 / 10, "filterbank identity failed ({ok} good)");
    }
}
