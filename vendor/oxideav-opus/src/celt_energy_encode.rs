//! CELT §5.3.2 energy quantisation — coarse (two-pass intra/inter with
//! the Laplace model), fine, and the final-bit backfill; the exact
//! write-side mirrors of the §4.3.2 decoders (RFC 6716 §5.3.2 /
//! §4.3.2, float-build arithmetic).
//!
//! The encoder carries the same quantized `oldBandE` state the decoder
//! reconstructs, so the §4.3.2.1 2-D prediction stays in lockstep
//! frame over frame.
//!
//! ## Provenance
//!
//! RFC 6716 §5.3.2 narrative + the normative Appendix A reference
//! listing's encoder (staged `docs/audio/opus/rfc6716-opus.txt`,
//! extracted and hash-verified per §A.1). No external library source
//! was consulted.

use crate::celt_band_layout::CELT_NUM_BANDS;
use crate::celt_e_prob_model::{e_prob_pair, EnergyPredictionMode};
use crate::celt_laplace::ec_laplace_encode;
use crate::celt_rate_alloc::MAX_FINE_BITS;
use crate::range_encoder::RangeEncoder;

/// §4.3.2.1 inter-frame prediction coefficients, per LM.
const PRED_COEF: [f64; 4] = [
    29440.0 / 32768.0,
    26112.0 / 32768.0,
    21248.0 / 32768.0,
    16384.0 / 32768.0,
];
/// §4.3.2.1 in-frame (band) feedback, per LM.
const BETA_COEF: [f64; 4] = [
    30147.0 / 32768.0,
    22282.0 / 32768.0,
    12124.0 / 32768.0,
    6554.0 / 32768.0,
];
/// §4.3.2.1 intra-frame feedback coefficient.
const BETA_INTRA: f64 = 4915.0 / 32768.0;

/// ICDF for the low-budget coarse-energy fallback.
const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

/// Per-channel × per-band `f64` energy grid.
pub type EnergyGrid = [[f64; CELT_NUM_BANDS]; 2];

/// The listing's `loss_distortion`: a capped distance between this
/// frame's energies and the carried state, feeding the delayed-intra
/// decision.
fn loss_distortion(
    band_log_e: &EnergyGrid,
    old_band_e: &EnergyGrid,
    start: usize,
    end: usize,
    channels: usize,
) -> f64 {
    let mut dist = 0.0f64;
    for c in 0..channels {
        for i in start..end {
            let d = band_log_e[c][i] - old_band_e[c][i];
            dist += d * d;
        }
    }
    dist.min(200.0)
}

/// One coarse pass at fixed intra/inter mode. Returns the badness (the
/// total |qi0 - qi| clamping distance).
#[allow(clippy::too_many_arguments)]
fn quant_coarse_energy_impl(
    enc: &mut RangeEncoder,
    start: usize,
    end: usize,
    band_log_e: &EnergyGrid,
    old_band_e: &mut EnergyGrid,
    budget: i64,
    mut tell: i64,
    error: &mut EnergyGrid,
    channels: usize,
    lm: usize,
    intra: bool,
    max_decay: f64,
) -> i32 {
    let mut badness = 0i32;
    let mut prev = [0.0f64; 2];
    if tell + 3 <= budget {
        enc.enc_bit_logp(intra, 3);
    }
    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm], BETA_COEF[lm])
    };
    let mode = EnergyPredictionMode::from_intra_flag(intra);

    for i in start..end {
        for c in 0..channels {
            let x = band_log_e[c][i];
            let old_e = old_band_e[c][i].max(-9.0);
            let f = x - coef * old_e - prev[c];
            // Rounding to nearest integer here is really important.
            let mut qi = (0.5 + f).floor() as i32;
            let decay_bound = old_band_e[c][i].max(-28.0) - max_decay;
            // Prevent the energy from going down too quickly.
            if qi < 0 && x < decay_bound {
                qi += (decay_bound - x) as i32;
                if qi > 0 {
                    qi = 0;
                }
            }
            let qi0 = qi;
            // If we don't have enough bits to encode all the energy,
            // just assume something safe.
            tell = i64::from(enc.tell());
            let bits_left = budget - tell - 3 * (channels as i64) * ((end - i) as i64);
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = qi.min(1);
                }
                if bits_left < 16 {
                    qi = qi.max(-1);
                }
            }
            if budget - tell >= 15 {
                let pair = e_prob_pair(lm as u32, mode, i.min(20) as u32)
                    .expect("band/lm ranges are valid");
                ec_laplace_encode(
                    enc,
                    &mut qi,
                    u32::from(pair.prob) << 7,
                    u32::from(pair.decay) << 6,
                );
            } else if budget - tell >= 2 {
                qi = qi.clamp(-1, 1);
                let sym = (2 * qi) ^ -i32::from(qi < 0);
                enc.enc_icdf(sym as usize, &SMALL_ENERGY_ICDF, 2);
            } else if budget - tell >= 1 {
                qi = qi.min(0);
                enc.enc_bit_logp(qi == -1, 1);
            } else {
                qi = -1;
            }
            error[c][i] = f - f64::from(qi);
            badness += (qi0 - qi).abs();
            let q = f64::from(qi);
            old_band_e[c][i] = coef * old_e + prev[c] + q;
            prev[c] += q - beta * q;
        }
    }
    badness
}

/// §5.3.2 coarse energy quantisation with the listing's two-pass
/// intra/inter selection. Returns the coded intra flag.
#[allow(clippy::too_many_arguments)]
pub fn quant_coarse_energy(
    enc: &mut RangeEncoder,
    start: usize,
    end: usize,
    band_log_e: &EnergyGrid,
    old_band_e: &mut EnergyGrid,
    budget: u32,
    error: &mut EnergyGrid,
    channels: usize,
    lm: usize,
    nb_available_bytes: i32,
    force_intra: bool,
    delayed_intra: &mut f64,
    mut two_pass: bool,
) -> bool {
    let c = channels as f64;
    let span = (end - start) as f64;
    let mut intra = force_intra
        || (!two_pass
            && *delayed_intra > 2.0 * c * span
            && f64::from(nb_available_bytes) > span * c);
    let new_distortion = loss_distortion(band_log_e, old_band_e, start, end, channels);
    let budget = i64::from(budget);
    let tell = i64::from(enc.tell());
    if tell + 3 > budget {
        two_pass = false;
        intra = false;
    }
    let max_decay = 16.0f64.min(0.125 * f64::from(nb_available_bytes));

    let enc_start_state = enc.clone();
    let mut old_intra = *old_band_e;
    let mut error_intra = *error;

    let badness1 = if two_pass || intra {
        quant_coarse_energy_impl(
            enc,
            start,
            end,
            band_log_e,
            &mut old_intra,
            budget,
            tell,
            &mut error_intra,
            channels,
            lm,
            true,
            max_decay,
        )
    } else {
        0
    };

    if !intra {
        let tell_intra = i64::from(enc.tell_frac());
        let enc_intra_state = enc.clone();
        *enc = enc_start_state;
        let badness2 = quant_coarse_energy_impl(
            enc, start, end, band_log_e, old_band_e, budget, tell, error, channels, lm, false,
            max_decay,
        );
        // No packet-loss bias (loss_rate = 0 → intra_bias = 0).
        if two_pass
            && (badness1 < badness2
                || (badness1 == badness2 && i64::from(enc.tell_frac()) > tell_intra))
        {
            *enc = enc_intra_state;
            *old_band_e = old_intra;
            *error = error_intra;
            intra = true;
        }
    } else {
        *old_band_e = old_intra;
        *error = error_intra;
    }

    if intra {
        *delayed_intra = new_distortion;
    } else {
        *delayed_intra = PRED_COEF[lm] * PRED_COEF[lm] * *delayed_intra + new_distortion;
    }
    intra
}

/// §5.3.2 fine energy quantisation (`quant_fine_energy`).
pub fn quant_fine_energy(
    enc: &mut RangeEncoder,
    start: usize,
    end: usize,
    old_band_e: &mut EnergyGrid,
    error: &mut EnergyGrid,
    fine_quant: &[i32; CELT_NUM_BANDS],
    channels: usize,
) {
    for i in start..end {
        if fine_quant[i] <= 0 {
            continue;
        }
        let frac = 1i32 << fine_quant[i];
        for c in 0..channels {
            let mut q2 = ((error[c][i] + 0.5) * f64::from(frac)).floor() as i32;
            q2 = q2.clamp(0, frac - 1);
            enc.enc_bits(q2 as u32, fine_quant[i] as u32);
            let offset = (f64::from(q2) + 0.5) / f64::from(frac) - 0.5;
            old_band_e[c][i] += offset;
            error[c][i] -= offset;
        }
    }
}

/// §5.3.2 final fine-energy bit backfill (`quant_energy_finalise`).
#[allow(clippy::too_many_arguments)]
pub fn quant_energy_finalise(
    enc: &mut RangeEncoder,
    start: usize,
    end: usize,
    old_band_e: &mut EnergyGrid,
    error: &EnergyGrid,
    fine_quant: &[i32; CELT_NUM_BANDS],
    fine_priority: &[bool; CELT_NUM_BANDS],
    mut bits_left: i32,
    channels: usize,
) {
    for prio in [false, true] {
        let mut i = start;
        while i < end && bits_left >= channels as i32 {
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                i += 1;
                continue;
            }
            for c in 0..channels {
                let q2 = i32::from(error[c][i] >= 0.0);
                enc.enc_bits(q2 as u32, 1);
                let offset = (f64::from(q2) - 0.5) / f64::from(1i32 << (fine_quant[i] + 1));
                old_band_e[c][i] += offset;
                bits_left -= 1;
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;

    /// Decode-side §4.3.2.1 mirror (transcribing the decoder's
    /// unquant walk) to verify state lockstep.
    fn unquant_coarse(
        rd: &mut RangeDecoder<'_>,
        start: usize,
        end: usize,
        budget: u32,
        lm: usize,
        channels: usize,
        old: &mut EnergyGrid,
    ) {
        let intra = if i64::from(rd.tell()) + 3 <= i64::from(budget) {
            rd.dec_bit_logp(3) == 1
        } else {
            false
        };
        let mode = EnergyPredictionMode::from_intra_flag(intra);
        let (coef, beta) = if intra {
            (0.0, BETA_INTRA)
        } else {
            (PRED_COEF[lm], BETA_COEF[lm])
        };
        let mut prev = [0.0f64; 2];
        #[allow(clippy::needless_range_loop)]
        for i in start..end {
            for (ch, prev_c) in prev.iter_mut().enumerate().take(channels) {
                let tell = rd.tell();
                let qi: i32 = if budget.saturating_sub(tell) >= 15 {
                    let pair = e_prob_pair(lm as u32, mode, i.min(20) as u32).unwrap();
                    crate::celt_laplace::ec_laplace_decode(
                        rd,
                        u32::from(pair.prob) << 7,
                        u32::from(pair.decay) << 6,
                    )
                } else if budget.saturating_sub(tell) >= 2 {
                    let q = rd.dec_icdf(&SMALL_ENERGY_ICDF, 2) as i32;
                    (q >> 1) ^ -(q & 1)
                } else if budget.saturating_sub(tell) >= 1 {
                    -(rd.dec_bit_logp(1) as i32)
                } else {
                    -1
                };
                let q = f64::from(qi);
                let old_e = old[ch][i].max(-9.0);
                old[ch][i] = coef * old_e + *prev_c + q;
                *prev_c += q - beta * q;
            }
        }
    }

    #[test]
    fn coarse_encode_decode_stay_in_lockstep_across_frames() {
        // Three frames of drifting energies: after coarse encode, the
        // decoder's reconstruction must equal the encoder's carried
        // state exactly (same f64 recurrence on both sides).
        let mut old_enc: EnergyGrid = [[0.0; CELT_NUM_BANDS]; 2];
        let mut old_dec: EnergyGrid = [[0.0; CELT_NUM_BANDS]; 2];
        let mut delayed = 0.0f64;
        for frame in 0..3 {
            let mut band_log_e: EnergyGrid = [[0.0; CELT_NUM_BANDS]; 2];
            for (i, v) in band_log_e[0].iter_mut().enumerate() {
                *v = 3.0 * ((i as f64) * 0.7 + frame as f64).sin() + 2.0;
            }
            let mut error: EnergyGrid = [[0.0; CELT_NUM_BANDS]; 2];
            let mut enc = RangeEncoder::new();
            let intra = quant_coarse_energy(
                &mut enc,
                0,
                21,
                &band_log_e,
                &mut old_enc,
                200 * 8,
                &mut error,
                1,
                3,
                200,
                frame == 0,
                &mut delayed,
                true,
            );
            let buf = enc.finish();
            let mut rd = RangeDecoder::new(&buf);
            unquant_coarse(&mut rd, 0, 21, 200 * 8, 3, 1, &mut old_dec);
            let _ = intra;
            for i in 0..21 {
                assert!(
                    (old_enc[0][i] - old_dec[0][i]).abs() < 1e-12,
                    "frame {frame} band {i}: enc {} dec {}",
                    old_enc[0][i],
                    old_dec[0][i]
                );
            }
            // The prediction error the fine stage will refine is
            // bounded by half a coarse step (plus the low-budget
            // clamps, absent at this budget).
            for (i, e) in error[0].iter().enumerate() {
                assert!(e.abs() <= 0.5 + 1e-9, "band {i}: {e}");
            }
        }
    }

    #[test]
    fn fine_and_finalise_refine_toward_the_target() {
        let mut old: EnergyGrid = [[0.0; CELT_NUM_BANDS]; 2];
        let mut error: EnergyGrid = [[0.0; CELT_NUM_BANDS]; 2];
        for (i, e) in error[0].iter_mut().enumerate() {
            *e = 0.37 - 0.03 * i as f64;
        }
        let mut fine = [0i32; CELT_NUM_BANDS];
        for (i, f) in fine.iter_mut().enumerate() {
            *f = (i % 4) as i32;
        }
        let prio = [false; CELT_NUM_BANDS];
        let mut enc = RangeEncoder::new();
        quant_fine_energy(&mut enc, 0, 21, &mut old, &mut error, &fine, 1);
        for i in 0..21 {
            if fine[i] > 0 {
                assert!(
                    error[0][i].abs() <= 0.5 / f64::from(1 << fine[i]) + 1e-9,
                    "band {i}"
                );
            }
        }
        quant_energy_finalise(&mut enc, 0, 21, &mut old, &error, &fine, &prio, 21, 1);
        // Each band consumed one extra bit worth of refinement.
        let buf = enc.finish();
        assert!(!buf.is_empty());
    }
}
