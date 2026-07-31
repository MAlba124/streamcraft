use crate::audio_signal::AudioSignal;
use crate::envelope;
use crate::xcorr;
use ndarray::Array1;
use profluens_core::memory::Arena;
use std::mem::align_of;

/// Time-aligns `deg_samples` to `ref_samples` — zero-padding the beginning and truncating the end,
/// or truncating the beginning — and writes the aligned pair into `out_ref`/`out_deg`, returning the
/// delay between the signals in seconds.
///
/// The destinations are caller-owned and only cleared (never reallocated once grown), so the
/// fine-realignment loop can reuse one pair across every patch; `out`/`scratch` back the envelope +
/// cross-correlation buffers (see [`globally_align_into`]).
pub fn align_and_truncate_into(
    ref_samples: &[f64],
    deg_samples: &[f64],
    sample_rate: u32,
    out_ref: &mut Vec<f64>,
    out_deg: &mut Vec<f64>,
    out: &Arena,
    scratch: &mut Arena,
) -> Option<f64> {
    let lag =
        globally_align_into(ref_samples, deg_samples, sample_rate, out_deg, out, scratch)?;

    out_ref.clear();
    match ref_samples.len().cmp(&out_deg.len()) {
        std::cmp::Ordering::Less => {
            // For positive lag, the beginning of ref is now padded with zeros, so
            // that amount should be truncated.
            let start = (lag * sample_rate as f64) as usize;
            out_ref.extend_from_slice(&ref_samples[start..ref_samples.len()]);
            // `deg[start..ref_len]` — truncate the tail first, then shift the head off.
            out_deg.truncate(ref_samples.len());
            out_deg.drain(..start);
        }
        std::cmp::Ordering::Greater => {
            out_ref.extend_from_slice(&ref_samples[..out_deg.len()]);
        }
        std::cmp::Ordering::Equal => out_ref.extend_from_slice(ref_samples),
    }
    Some(lag)
}

/// Aligns a degraded signal to the reference signal, truncating them to
/// be the same length.
///
/// The two envelopes must both be live for the cross-correlation, so they are carved from `out`;
/// everything else is transform scratch, carved from `scratch` and reclaimed between the three
/// phases. (`scratch` is reset here, not by the caller, so the reset discipline lives next to the
/// allocations it reclaims.)
pub fn globally_align_into(
    ref_samples: &[f64],
    deg_samples: &[f64],
    sample_rate: u32,
    out_deg: &mut Vec<f64>,
    out: &Arena,
    scratch: &mut Arena,
) -> Option<f64> {
    // A bump arena reuses chunks across `reset` only while the *next* phase's allocations still fit
    // the chunks it already holds — and the cross-correlation's buffers are about twice the
    // envelope's, so the three phases below would otherwise each grow their own set (~360 MiB of
    // chunks for a 30 s clip, where the largest single phase needs ~230). Force one chunk sized for
    // that largest phase up front by carving and immediately dropping it; every phase then fits in
    // it and the resets genuinely recycle. Purely a sizing hint: if it is short the arena just grows.
    //
    // The dominating phase is the cross-correlation — two `fft_points`-long complex spectra (the
    // product overwrites one of them) plus one real output, i.e. `2*16 + 8` bytes per point.
    let longest = ref_samples.len().max(deg_samples.len()).max(1);
    let hint = 40 * crate::math_utils::next_pow_two(2 * longest - 1);
    scratch.reset();
    let _ = scratch.alloc(hint, align_of::<num::complex::Complex64>());

    scratch.reset();
    let ref_upper_env = envelope::calculate_upper_env(ref_samples, out, &*scratch)?;
    scratch.reset();
    let deg_upper_env = envelope::calculate_upper_env(deg_samples, out, &*scratch)?;
    scratch.reset();

    let best_lag = xcorr::calculate_best_lag(ref_upper_env, deg_upper_env, &*scratch)?;

    out_deg.clear();
    if best_lag == 0 || best_lag.abs() > (ref_samples.len() / 2) as i64 {
        // If signals are correlated already, return deg signal and 0.
        out_deg.extend_from_slice(deg_samples);
        Some(0.0f64)
    } else if best_lag < 0 {
        out_deg.extend_from_slice(&deg_samples[best_lag.unsigned_abs() as usize..]);
        Some(best_lag as f64 / sample_rate as f64)
    } else {
        // Zero-pad the degraded signal's start by the lag.
        out_deg.resize(best_lag as usize, 0.0);
        out_deg.extend_from_slice(deg_samples);
        Some(best_lag as f64 / sample_rate as f64)
    }
}

/// [`globally_align_into`] returning an owned [`AudioSignal`] — for the one whole-signal alignment,
/// whose result has to outlive every arena in play.
pub fn globally_align(
    ref_signal: &AudioSignal,
    deg_signal: &AudioSignal,
    out: &Arena,
    scratch: &mut Arena,
) -> Option<(AudioSignal, f64)> {
    let mut aligned = Vec::new();
    let lag = globally_align_into(
        ref_signal.data_matrix.as_slice()?,
        deg_signal.data_matrix.as_slice()?,
        deg_signal.sample_rate,
        &mut aligned,
        out,
        scratch,
    )?;
    Some((
        AudioSignal { data_matrix: Array1::from_vec(aligned), sample_rate: deg_signal.sample_rate },
        lag,
    ))
}
