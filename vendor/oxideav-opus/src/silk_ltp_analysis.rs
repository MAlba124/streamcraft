//! LTP filter analysis + quantisation for the SILK encoder — RFC 6716
//! §5.2.3.4.1 / §5.2.3.6.
//!
//! For a voiced frame, §4.2.7.9.1 predicts each LPC-residual sample
//! from five delayed residual taps around the subframe's pitch lag:
//!
//! ```text
//!                         4
//!                         __                                  b_Q7[k]
//!   pred[i] =             \   res[i - pitch_lag + 2 - k]  *  -------
//!                         /_                                   128
//!                         k=0
//! ```
//!
//! The encoder must pick, per §4.2.7.6.2, ONE periodicity index
//! (selecting one of the three Table 39-41 codebooks) for the whole
//! frame and one 5-tap codebook vector per subframe. §5.2.3.6
//! describes the reference's weighted rate-distortion search; the
//! strategy is an encoder freedom, and this implementation minimizes
//! the exact squared prediction error directly over the discrete
//! codebooks:
//!
//! ```text
//!   E(cb) = sum_i (r[i] - pred_cb[i])^2
//!         = c  -  2/128 * g·cb  +  1/128^2 * cb'·R·cb
//! ```
//!
//! with `g[k] = sum_i r[i] * r[i-lag+2-k]` and
//! `R[k][j] = sum_i r[i-lag+2-k] * r[i-lag+2-j]` accumulated once per
//! subframe — every codebook vector is then scored in closed form,
//! per-subframe minima are taken within each codebook, and the
//! codebook whose summed distortion over all subframes is lowest
//! wins (matching §5.2.3.6's "codebook search for the subframe LTP
//! vectors is constrained to only allow codebook vectors to be
//! chosen from the same codebook").
//!
//! All truth is taken from RFC 6716 §4.2.7.9.1 (prediction geometry),
//! §4.2.7.6.2 (codebooks), §5.2.3.6 (search outline). No external
//! library source is consulted.

use crate::silk_lpc_synth::subframe_samples;
use crate::silk_ltp::{
    lag_range, ltp_filter_codebook_len, ltp_filter_taps_q7, LTP_FILTER_TAPS, LTP_MAX_SUBFRAMES,
};
use crate::toc::Bandwidth;
use crate::Error;

/// Result of the §5.2.3.6 LTP quantisation for one voiced SILK frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtpQuantized {
    /// §4.2.7.6.2 periodicity index (`0..=2`), one per frame.
    pub periodicity_index: u8,
    /// Per-subframe indices into the chosen Table 39-41 codebook.
    /// Only `num_subframes` entries are valid.
    pub filter_indices: [u8; LTP_MAX_SUBFRAMES],
    /// The chosen Q7 taps (decoder-identical). Only `num_subframes`
    /// rows are valid.
    pub taps_q7: [[i8; LTP_FILTER_TAPS]; LTP_MAX_SUBFRAMES],
}

/// Analyse and quantize the per-subframe 5-tap LTP filters.
///
/// * `res` — LPC-residual-domain signal: `frame_start` history
///   samples (at least `lag_max + 2`) followed by the frame
///   (`num_subframes * subframe_samples(bandwidth)` samples).
/// * `lags` — the DECODED per-subframe pitch lags (from
///   [`crate::silk_pitch::PitchAnalysis::subframe_lags`]), one per
///   subframe.
///
/// Returns the frame's periodicity index, per-subframe filter
/// indices, and the exact Q7 taps the decoder will apply.
pub fn ltp_analysis(
    bandwidth: Bandwidth,
    res: &[f64],
    frame_start: usize,
    num_subframes: usize,
    lags: &[i32],
) -> Result<LtpQuantized, Error> {
    if num_subframes != 2 && num_subframes != 4 || lags.len() != num_subframes {
        return Err(Error::MalformedPacket);
    }
    let (lag_min, lag_max, _) = lag_range(bandwidth)?;
    let n = subframe_samples(bandwidth)?;
    if frame_start < (lag_max + 2) as usize || res.len() != frame_start + n * num_subframes {
        return Err(Error::MalformedPacket);
    }
    if lags.iter().any(|&l| l < lag_min || l > lag_max) {
        return Err(Error::MalformedPacket);
    }

    // Per-subframe correlation accumulators.
    let mut g = [[0.0f64; LTP_FILTER_TAPS]; LTP_MAX_SUBFRAMES];
    let mut r_mat = [[[0.0f64; LTP_FILTER_TAPS]; LTP_FILTER_TAPS]; LTP_MAX_SUBFRAMES];
    for s in 0..num_subframes {
        let lag = lags[s] as usize;
        let base = frame_start + s * n;
        for i in 0..n {
            let target = res[base + i];
            // Tap k reads res[i - lag + 2 - k]; base+i - lag + 2 - k
            // is >= 0 because frame_start >= lag_max + 2.
            let mut taps = [0.0f64; LTP_FILTER_TAPS];
            for (k, slot) in taps.iter_mut().enumerate() {
                *slot = res[base + i + 2 - lag - k];
            }
            for k in 0..LTP_FILTER_TAPS {
                g[s][k] += target * taps[k];
                for j in k..LTP_FILTER_TAPS {
                    r_mat[s][k][j] += taps[k] * taps[j];
                }
            }
        }
        // Mirror the symmetric half.
        #[allow(clippy::needless_range_loop)]
        for k in 1..LTP_FILTER_TAPS {
            for j in 0..k {
                let v = r_mat[s][j][k];
                r_mat[s][k][j] = v;
            }
        }
    }

    // Score every codebook: per-subframe minimum within the codebook,
    // then the codebook with the least summed distortion.
    let mut best: Option<(u8, [u8; LTP_MAX_SUBFRAMES], f64)> = None;
    for p in 0..3u8 {
        let cb_len = ltp_filter_codebook_len(p)?;
        let mut indices = [0u8; LTP_MAX_SUBFRAMES];
        let mut total = 0.0f64;
        for s in 0..num_subframes {
            let mut sub_best = (0u8, f64::MAX);
            for idx in 0..cb_len as u8 {
                let taps = ltp_filter_taps_q7(p, idx)?;
                let b: [f64; LTP_FILTER_TAPS] = core::array::from_fn(|k| taps[k] as f64 / 128.0);
                // E - c = -2 g·b + b' R b.
                let mut e = 0.0f64;
                for k in 0..LTP_FILTER_TAPS {
                    e -= 2.0 * g[s][k] * b[k];
                    for j in 0..LTP_FILTER_TAPS {
                        e += b[k] * r_mat[s][k][j] * b[j];
                    }
                }
                if e < sub_best.1 {
                    sub_best = (idx, e);
                }
            }
            indices[s] = sub_best.0;
            total += sub_best.1;
        }
        let better = match &best {
            None => true,
            Some((_, _, t)) => total < *t,
        };
        if better {
            best = Some((p, indices, total));
        }
    }

    let (periodicity_index, filter_indices, _) = best.ok_or(Error::MalformedPacket)?;
    let mut taps_q7 = [[0i8; LTP_FILTER_TAPS]; LTP_MAX_SUBFRAMES];
    for s in 0..num_subframes {
        taps_q7[s] = ltp_filter_taps_q7(periodicity_index, filter_indices[s])?;
    }
    Ok(LtpQuantized {
        periodicity_index,
        filter_indices,
        taps_q7,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize, mut seed: u32) -> Vec<f64> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(196_314_165).wrapping_add(907_633_515);
                (seed >> 8) as f64 / (1u32 << 24) as f64 - 0.5
            })
            .collect()
    }

    /// A strongly periodic residual (r[i] = 0.9 r[i-80] + small noise)
    /// must yield quantized taps with a real prediction gain.
    #[test]
    fn periodic_residual_gets_prediction_gain() {
        let bw = Bandwidth::Wb;
        let (_, lag_max, _) = lag_range(bw).unwrap();
        let n = subframe_samples(bw).unwrap();
        let frame_start = (lag_max + 2) as usize;
        let total = frame_start + 4 * n;
        let drive = noise(total, 0x1234);
        let mut r = vec![0.0f64; total];
        for i in 0..total {
            let prev = if i >= 80 { r[i - 80] } else { 0.0 };
            r[i] = 0.9 * prev + 0.1 * drive[i];
        }
        let lags = [80i32; 4];
        let q = ltp_analysis(bw, &r, frame_start, 4, &lags).unwrap();

        // Measure the actual §4.2.7.9.1 prediction error with the
        // chosen taps vs no LTP at all.
        let mut e_none = 0.0f64;
        let mut e_ltp = 0.0f64;
        for s in 0..4 {
            let base = frame_start + s * n;
            for i in 0..n {
                let mut pred = 0.0f64;
                for k in 0..LTP_FILTER_TAPS {
                    pred += q.taps_q7[s][k] as f64 / 128.0 * r[base + i + 2 - 80 - k];
                }
                e_none += r[base + i] * r[base + i];
                e_ltp += (r[base + i] - pred) * (r[base + i] - pred);
            }
        }
        assert!(
            e_ltp < e_none / 3.0,
            "LTP gain too small: {e_none} -> {e_ltp}"
        );
        // Taps must be the decoder's exact codebook rows.
        for s in 0..4 {
            let expect = ltp_filter_taps_q7(q.periodicity_index, q.filter_indices[s]).unwrap();
            assert_eq!(q.taps_q7[s], expect);
        }
    }

    /// The chosen codebook must beat (or tie) the other two on the
    /// same exact-distortion metric, re-measured independently.
    #[test]
    fn chosen_codebook_is_optimal() {
        let bw = Bandwidth::Nb;
        let (_, lag_max, _) = lag_range(bw).unwrap();
        let n = subframe_samples(bw).unwrap();
        let frame_start = (lag_max + 2) as usize;
        let total = frame_start + 2 * n;
        let drive = noise(total, 77);
        let mut r = vec![0.0f64; total];
        for i in 0..total {
            let prev = if i >= 40 { r[i - 40] } else { 0.0 };
            r[i] = 0.6 * prev + 0.4 * drive[i];
        }
        let lags = [40i32; 2];
        let q = ltp_analysis(bw, &r, frame_start, 2, &lags).unwrap();

        let measure = |p: u8, indices: &[u8]| -> f64 {
            let mut e = 0.0f64;
            for (s, &fi) in indices.iter().enumerate().take(2) {
                let taps = ltp_filter_taps_q7(p, fi).unwrap();
                let base = frame_start + s * n;
                for i in 0..n {
                    let mut pred = 0.0f64;
                    for k in 0..LTP_FILTER_TAPS {
                        pred += taps[k] as f64 / 128.0 * r[base + i + 2 - 40 - k];
                    }
                    e += (r[base + i] - pred) * (r[base + i] - pred);
                }
            }
            e
        };
        let chosen = measure(q.periodicity_index, &q.filter_indices[..2]);
        for p in 0..3u8 {
            // Best indices for codebook p by exhaustive re-search.
            let cb_len = ltp_filter_codebook_len(p).unwrap();
            let mut best_e = 0.0;
            #[allow(clippy::needless_range_loop)]
            for s in 0..2usize {
                let mut sub = f64::MAX;
                for idx in 0..cb_len as u8 {
                    // Score subframe s alone.
                    let taps = ltp_filter_taps_q7(p, idx).unwrap();
                    let base = frame_start + s * n;
                    let mut e = 0.0;
                    for i in 0..n {
                        let mut pred = 0.0f64;
                        for k in 0..LTP_FILTER_TAPS {
                            pred += taps[k] as f64 / 128.0 * r[base + i + 2 - 40 - k];
                        }
                        e += (r[base + i] - pred) * (r[base + i] - pred);
                    }
                    if e < sub {
                        sub = e;
                    }
                }
                best_e += sub;
            }
            assert!(
                chosen <= best_e + 1e-9,
                "codebook {p} would beat the winner: {best_e} < {chosen}"
            );
        }
    }

    /// All-zero residual: no panics, valid indices.
    #[test]
    fn zero_residual_is_safe() {
        let bw = Bandwidth::Mb;
        let (_, lag_max, _) = lag_range(bw).unwrap();
        let n = subframe_samples(bw).unwrap();
        let frame_start = (lag_max + 2) as usize;
        let r = vec![0.0f64; frame_start + 4 * n];
        let lags = [60i32; 4];
        let q = ltp_analysis(bw, &r, frame_start, 4, &lags).unwrap();
        assert!(q.periodicity_index <= 2);
        let cb_len = ltp_filter_codebook_len(q.periodicity_index).unwrap();
        for s in 0..4 {
            assert!((q.filter_indices[s] as usize) < cb_len);
        }
    }

    #[test]
    fn rejects_bad_geometry() {
        let bw = Bandwidth::Nb;
        let (lag_min, lag_max, _) = lag_range(bw).unwrap();
        let n = subframe_samples(bw).unwrap();
        let frame_start = (lag_max + 2) as usize;
        let r = vec![0.0f64; frame_start + 4 * n];
        // Bad subframe count.
        assert!(ltp_analysis(bw, &r, frame_start, 3, &[40; 3]).is_err());
        // Lag out of range.
        assert!(ltp_analysis(bw, &r, frame_start, 4, &[lag_min - 1; 4]).is_err());
        // Not enough history.
        let short = vec![0.0f64; 10 + 4 * n];
        assert!(ltp_analysis(bw, &short, 10, 4, &[40; 4]).is_err());
    }
}
