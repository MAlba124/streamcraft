//! Open-loop pitch analysis for the SILK encoder — RFC 6716 §5.2.3.2.
//!
//! Finds the voiced/unvoiced classification, the per-subframe pitch
//! lags, and their quantisation to the §4.2.7.6.1 wire indices
//! (primary lag high/low or relative delta + contour codebook index).
//!
//! §5.2.3.2 describes the reference's three-stage
//! downsample-and-refine correlator; the analysis strategy is an
//! encoder freedom, and this implementation runs a single-stage
//! normalized time correlation at the SILK internal rate:
//!
//!  1. Whole-frame normalized autocorrelation over every lag in the
//!     bandwidth's `[lag_min, lag_max]` (Table 30), with the
//!     §5.2.3.2 "small bias towards short lags to avoid ending up
//!     with a multiple of the true pitch lag".
//!  2. Voiced iff the best biased correlation exceeds a threshold.
//!  3. Per-subframe refinement in a narrow window around the winner.
//!  4. Joint quantisation: over primary-lag candidates near the
//!     winner and every Table 33-36 contour codebook entry, minimize
//!     the squared distance between the DECODED per-subframe lags
//!     (primary + offset, clamped exactly as §4.2.7.6.1 does) and the
//!     measured per-subframe lags.
//!
//! The caller hands the analysis the whitened signal (§5.2.3.2 runs
//! the correlator on the LPC-whitened input; see
//! [`crate::silk_lpc_analysis::lpc_residual`]) as one contiguous
//! buffer with at least `lag_max` history samples ahead of the frame.
//!
//! All truth is taken from RFC 6716 §4.2.7.6.1 (wire semantics) and
//! §5.2.3.2 (analysis outline). No external library source is
//! consulted.

use crate::silk_lpc_synth::subframe_samples;
use crate::silk_ltp::{contour_codebook_len, contour_offsets, lag_range, LagSymbols};
use crate::toc::Bandwidth;
use crate::Error;

/// Voicing decision threshold on the biased normalized correlation
/// (encoder freedom; §5.2.3.2's reference threshold adapts to speech
/// activity and spectral tilt, which this front end does not model).
pub const PITCH_CORR_THRESHOLD: f64 = 0.45;

/// Half-width of the per-subframe refinement window around the
/// whole-frame winner lag (samples at the internal rate).
const SUBFRAME_SEARCH_HALF_WIDTH: i32 = 8;

/// Half-width of the primary-lag candidate sweep in the joint
/// (primary, contour) quantisation step.
const PRIMARY_SEARCH_HALF_WIDTH: i32 = 4;

/// Strength of the short-lag bias: the correlation is scaled by
/// `1 - BIAS * (lag - lag_min) / (lag_max - lag_min)`.
const SHORT_LAG_BIAS: f64 = 0.02;

/// Result of the §5.2.3.2 pitch analysis for one SILK frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchAnalysis {
    /// Voicing decision (biased correlation above
    /// [`PITCH_CORR_THRESHOLD`]).
    pub voiced: bool,
    /// Best whole-frame normalized correlation (unbiased, for
    /// diagnostics / mode decisions).
    pub correlation: f64,
    /// The quantized primary lag the decoder will reconstruct.
    pub primary_lag: i32,
    /// Chosen Table 33-36 contour codebook index.
    pub contour_index: u8,
    /// The DECODED per-subframe lags (primary + contour offset,
    /// clamped) — exactly what §4.2.7.6.1 reconstructs. Only
    /// `num_subframes` entries are valid.
    pub subframe_lags: [i32; 4],
}

/// Run the open-loop pitch analysis over one SILK frame.
///
/// * `x` — whitened signal, containing `frame_start` history samples
///   followed by the frame itself. `frame_start` must be at least the
///   bandwidth's `lag_max` and `x.len() - frame_start` must equal
///   `num_subframes * subframe_samples(bandwidth)`.
/// * `num_subframes` — 2 (10 ms) or 4 (20 ms).
///
/// Returns the classification plus the jointly quantized
/// (primary lag, contour) pair when voiced; for an unvoiced frame the
/// quantisation fields hold `lag_min` / index 0 placeholders and
/// should not be coded.
pub fn pitch_analysis(
    bandwidth: Bandwidth,
    x: &[f64],
    frame_start: usize,
    num_subframes: usize,
) -> Result<PitchAnalysis, Error> {
    if num_subframes != 2 && num_subframes != 4 {
        return Err(Error::MalformedPacket);
    }
    let (lag_min, lag_max, _) = lag_range(bandwidth)?;
    let n = subframe_samples(bandwidth)?;
    let frame_len = n * num_subframes;
    if frame_start < lag_max as usize || x.len() != frame_start + frame_len {
        return Err(Error::MalformedPacket);
    }

    // ---- Stage 1: whole-frame lag sweep with short-lag bias. ----
    let frame = &x[frame_start..];
    let mut best_lag = lag_min;
    let mut best_biased = f64::MIN;
    let mut best_unbiased = 0.0f64;
    for lag in lag_min..=lag_max {
        let c = normalized_corr(x, frame_start, 0, frame_len, lag);
        let bias = 1.0 - SHORT_LAG_BIAS * (lag - lag_min) as f64 / (lag_max - lag_min) as f64;
        let biased = c * bias;
        if biased > best_biased {
            best_biased = biased;
            best_unbiased = c;
            best_lag = lag;
        }
    }

    let energy: f64 = frame.iter().map(|&v| v * v).sum();
    let voiced = energy > 0.0 && best_biased > PITCH_CORR_THRESHOLD;
    if !voiced {
        return Ok(PitchAnalysis {
            voiced: false,
            correlation: best_unbiased.max(0.0),
            primary_lag: lag_min,
            contour_index: 0,
            subframe_lags: [lag_min; 4],
        });
    }

    // ---- Stage 2: per-subframe refinement near the winner. ----
    let mut measured = [best_lag; 4];
    for (s, slot) in measured.iter_mut().enumerate().take(num_subframes) {
        let lo = (best_lag - SUBFRAME_SEARCH_HALF_WIDTH).max(lag_min);
        let hi = (best_lag + SUBFRAME_SEARCH_HALF_WIDTH).min(lag_max);
        let mut best = (best_lag, f64::MIN);
        for lag in lo..=hi {
            let c = normalized_corr(x, frame_start, s * n, n, lag);
            if c > best.1 {
                best = (lag, c);
            }
        }
        *slot = best.0;
    }

    // ---- Stage 3: joint (primary, contour) quantisation. ----
    let cb_len = contour_codebook_len(bandwidth, num_subframes)?;
    let mut best_pick: Option<(i32, u8, i64)> = None;
    let p_lo = (best_lag - PRIMARY_SEARCH_HALF_WIDTH).max(lag_min);
    let p_hi = (best_lag + PRIMARY_SEARCH_HALF_WIDTH).min(lag_max);
    for primary in p_lo..=p_hi {
        for idx in 0..cb_len as u8 {
            let offs = contour_offsets(bandwidth, num_subframes, idx)?;
            let mut cost: i64 = 0;
            for (s, &m) in measured.iter().enumerate().take(num_subframes) {
                let decoded = (primary + offs[s] as i32).clamp(lag_min, lag_max);
                let d = (decoded - m) as i64;
                cost += d * d;
            }
            let better = match best_pick {
                None => true,
                Some((_, _, c)) => cost < c,
            };
            if better {
                best_pick = Some((primary, idx, cost));
            }
        }
    }
    let (primary, contour_index, _) = best_pick.ok_or(Error::MalformedPacket)?;
    let offs = contour_offsets(bandwidth, num_subframes, contour_index)?;
    let mut subframe_lags = [lag_min; 4];
    for (s, slot) in subframe_lags.iter_mut().enumerate().take(num_subframes) {
        *slot = (primary + offs[s] as i32).clamp(lag_min, lag_max);
    }

    Ok(PitchAnalysis {
        voiced: true,
        correlation: best_unbiased,
        primary_lag: primary,
        contour_index,
        subframe_lags,
    })
}

/// Normalized cross-correlation between the window
/// `x[frame_start+off .. +len]` and the same window delayed by `lag`
/// samples (reaching into the history prefix).
fn normalized_corr(x: &[f64], frame_start: usize, off: usize, len: usize, lag: i32) -> f64 {
    let a0 = frame_start + off;
    let b0 = a0 - lag as usize;
    let mut xy = 0.0f64;
    let mut xx = 0.0f64;
    let mut yy = 0.0f64;
    for i in 0..len {
        let xa = x[a0 + i];
        let xb = x[b0 + i];
        xy += xa * xb;
        xx += xa * xa;
        yy += xb * xb;
    }
    if xx <= 0.0 || yy <= 0.0 {
        return 0.0;
    }
    xy / (xx * yy).sqrt()
}

/// Quantize a primary pitch lag to the §4.2.7.6.1 [`LagSymbols`],
/// matching the frame's lag-coding context.
///
/// * `previous_lag == None` → absolute coding (Table 29 high part +
///   Table 30 low part). The requested lag is clamped to
///   `[lag_min, lag_max]` and reproduces exactly.
/// * `previous_lag == Some(prev)` → relative coding when the delta
///   fits the non-zero Table 31 support (`delta_index = delta + 9 ∈
///   1..=20`, i.e. `delta ∈ -8..=11`); otherwise the zero-delta
///   fallback carrying the absolute pair.
///
/// Returns the symbols and the lag the decoder will reconstruct.
pub fn quantize_lag(
    bandwidth: Bandwidth,
    lag: i32,
    previous_lag: Option<i32>,
) -> Result<(LagSymbols, i32), Error> {
    let (lag_min, lag_max, scale) = lag_range(bandwidth)?;
    let clamped = lag.clamp(lag_min, lag_max);
    // The absolute pair spans lag_high ∈ 0..=31, lag_low ∈ 0..scale,
    // i.e. lag_min ..= lag_min + 32*scale - 1 = lag_max - 1: the top
    // lag itself is only reachable through relative coding (whose
    // §4.2.7.6.1 reconstruction is unclamped).
    let abs_parts = |lag: i32| {
        let capped = lag.min(lag_max - 1);
        let rel = capped - lag_min;
        ((rel / scale) as u8, (rel % scale) as u8, capped)
    };
    match previous_lag {
        None => {
            let (lag_high, lag_low, decoded) = abs_parts(clamped);
            Ok((LagSymbols::Absolute { lag_high, lag_low }, decoded))
        }
        Some(prev) => {
            // §4.2.7.6.1: decoded = prev + (delta_index - 9), unclamped.
            let delta = clamped - prev;
            let idx = delta + 9;
            if (1..=20).contains(&idx) {
                Ok((
                    LagSymbols::RelativeDelta {
                        delta_index: idx as u8,
                    },
                    clamped,
                ))
            } else {
                let (lag_high, lag_low, decoded) = abs_parts(clamped);
                Ok((LagSymbols::RelativeFallback { lag_high, lag_low }, decoded))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;
    use crate::range_encoder::RangeEncoder;
    use crate::silk_frame::SignalType;
    use crate::silk_ltp::{LagCoding, LtpConfig, LtpParameters, LtpSymbols};

    /// Deterministic noise for the unvoiced test.
    fn noise(len: usize, mut seed: u32) -> Vec<f64> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(196_314_165).wrapping_add(907_633_515);
                (seed >> 8) as f64 / (1u32 << 24) as f64 - 0.5
            })
            .collect()
    }

    /// A decaying pulse train with period 80 at WB must classify
    /// voiced with lags ~80 on every subframe.
    #[test]
    fn pulse_train_recovers_lag() {
        let bw = Bandwidth::Wb;
        let (_, lag_max, _) = lag_range(bw).unwrap();
        let n = subframe_samples(bw).unwrap();
        let frame_len = 4 * n;
        let total = lag_max as usize + frame_len;
        let mut x = vec![0.0f64; total];
        for (i, slot) in x.iter_mut().enumerate() {
            if i % 80 == 3 {
                *slot = 1.0;
            }
            if i % 80 == 4 {
                *slot = 0.5;
            }
        }
        let pa = pitch_analysis(bw, &x, lag_max as usize, 4).unwrap();
        assert!(pa.voiced, "corr = {}", pa.correlation);
        assert!(pa.correlation > 0.9);
        for s in 0..4 {
            assert!(
                (pa.subframe_lags[s] - 80).abs() <= 2,
                "subframe {s}: {}",
                pa.subframe_lags[s]
            );
        }
    }

    /// White noise must classify unvoiced.
    #[test]
    fn noise_is_unvoiced() {
        let bw = Bandwidth::Nb;
        let (_, lag_max, _) = lag_range(bw).unwrap();
        let n = subframe_samples(bw).unwrap();
        let total = lag_max as usize + 4 * n;
        let x = noise(total, 0xBEEF);
        let pa = pitch_analysis(bw, &x, lag_max as usize, 4).unwrap();
        assert!(!pa.voiced, "corr = {}", pa.correlation);
    }

    /// Silence must classify unvoiced (no division blowups).
    #[test]
    fn silence_is_unvoiced() {
        let bw = Bandwidth::Mb;
        let (_, lag_max, _) = lag_range(bw).unwrap();
        let n = subframe_samples(bw).unwrap();
        let x = vec![0.0f64; lag_max as usize + 2 * n];
        let pa = pitch_analysis(bw, &x, lag_max as usize, 2).unwrap();
        assert!(!pa.voiced);
    }

    /// A sine's fundamental period wins over its multiples thanks to
    /// the short-lag bias.
    #[test]
    fn sine_prefers_fundamental() {
        let bw = Bandwidth::Nb;
        let (_, lag_max, _) = lag_range(bw).unwrap();
        let n = subframe_samples(bw).unwrap();
        let total = lag_max as usize + 4 * n;
        let period = 50.0f64;
        let x: Vec<f64> = (0..total)
            .map(|i| (i as f64 * std::f64::consts::TAU / period).sin())
            .collect();
        let pa = pitch_analysis(bw, &x, lag_max as usize, 4).unwrap();
        assert!(pa.voiced);
        assert!(
            (pa.primary_lag - 50).abs() <= 2,
            "primary = {}",
            pa.primary_lag
        );
    }

    /// quantize_lag absolute path reproduces every representable lag
    /// (`lag_min ..= lag_max - 1`) exactly across all three
    /// bandwidths; `lag_max` itself caps to `lag_max - 1` (it is only
    /// reachable through relative coding).
    #[test]
    fn absolute_lag_quantisation_is_exact() {
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            let (lag_min, lag_max, scale) = lag_range(bw).unwrap();
            for lag in lag_min..=lag_max {
                let (sym, decoded) = quantize_lag(bw, lag, None).unwrap();
                assert_eq!(decoded, lag.min(lag_max - 1));
                match sym {
                    LagSymbols::Absolute { lag_high, lag_low } => {
                        assert!(lag_high <= 31);
                        assert!((lag_low as i32) < scale);
                        assert_eq!(lag_high as i32 * scale + lag_low as i32 + lag_min, decoded);
                    }
                    _ => panic!("expected absolute"),
                }
            }
        }
    }

    /// Relative deltas in [-8, 11] use the delta path; larger jumps
    /// fall back — and both survive the real §4.2.7.6 wire encode.
    #[test]
    fn relative_lag_quantisation_roundtrips_wire() {
        let bw = Bandwidth::Wb;
        for (prev, lag) in [(100, 105), (100, 92), (100, 111), (100, 140), (100, 60)] {
            let (sym, decoded) = quantize_lag(bw, lag, Some(prev)).unwrap();
            assert_eq!(decoded, lag);
            match sym {
                LagSymbols::RelativeDelta { delta_index } => {
                    assert!((1..=20).contains(&delta_index));
                    assert_eq!(prev + delta_index as i32 - 9, lag);
                }
                LagSymbols::RelativeFallback { .. } => {
                    assert!(!(-8..=11).contains(&(lag - prev)));
                }
                LagSymbols::Absolute { .. } => panic!("expected relative"),
            }

            // Wire roundtrip through the real §4.2.7.6 coder.
            let cfg = LtpConfig {
                bandwidth: bw,
                signal_type: SignalType::Voiced,
                num_subframes: 4,
                lag_coding: LagCoding::Relative { previous_lag: prev },
                ltp_scaling_present: false,
            };
            let symbols = LtpSymbols {
                lag: sym,
                contour_index: 0,
                periodicity_index: 0,
                filter_indices: [0; 4],
                ltp_scaling_index: None,
            };
            let mut re = RangeEncoder::new();
            let enc = LtpParameters::encode(&mut re, cfg, Some(&symbols)).unwrap();
            assert_eq!(enc.primary_lag(), lag);
            let bytes = re.finish();
            let mut rd = RangeDecoder::new(&bytes);
            let dec = LtpParameters::decode(&mut rd, cfg).unwrap();
            assert_eq!(dec.primary_lag(), lag);
        }
    }

    /// The joint quantiser's decoded subframe lags match the
    /// §4.2.7.6.1 reconstruction for the chosen (primary, contour):
    /// verified through the real wire coder.
    #[test]
    fn contour_choice_matches_decoder_reconstruction() {
        let bw = Bandwidth::Wb;
        let (_, lag_max, _) = lag_range(bw).unwrap();
        let n = subframe_samples(bw).unwrap();
        let total = lag_max as usize + 4 * n;
        // Slightly drifting period so a non-flat contour can win.
        let mut x = vec![0.0f64; total];
        let mut next = 3usize;
        let mut period = 78usize;
        while next < total {
            x[next] = 1.0;
            next += period;
            if next % 3 == 0 && period < 84 {
                period += 1;
            }
        }
        let pa = pitch_analysis(bw, &x, lag_max as usize, 4).unwrap();
        assert!(pa.voiced);

        let (sym, _) = quantize_lag(bw, pa.primary_lag, None).unwrap();
        let cfg = LtpConfig {
            bandwidth: bw,
            signal_type: SignalType::Voiced,
            num_subframes: 4,
            lag_coding: LagCoding::Absolute,
            ltp_scaling_present: false,
        };
        let symbols = LtpSymbols {
            lag: sym,
            contour_index: pa.contour_index,
            periodicity_index: 0,
            filter_indices: [0; 4],
            ltp_scaling_index: None,
        };
        let mut re = RangeEncoder::new();
        let enc = LtpParameters::encode(&mut re, cfg, Some(&symbols)).unwrap();
        assert_eq!(enc.pitch_lags(), &pa.subframe_lags[..4]);
    }

    #[test]
    fn rejects_bad_geometry() {
        let bw = Bandwidth::Nb;
        let (_, lag_max, _) = lag_range(bw).unwrap();
        let n = subframe_samples(bw).unwrap();
        // Too little history.
        let x = vec![0.0f64; 10 + 4 * n];
        assert!(pitch_analysis(bw, &x, 10, 4).is_err());
        // Wrong frame length.
        let x = vec![0.0f64; lag_max as usize + 4 * n - 1];
        assert!(pitch_analysis(bw, &x, lag_max as usize, 4).is_err());
        // Bad subframe count.
        let x = vec![0.0f64; lag_max as usize + 3 * n];
        assert!(pitch_analysis(bw, &x, lag_max as usize, 3).is_err());
    }
}
