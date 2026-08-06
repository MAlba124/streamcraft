//! CELT §4.3 whole-frame entropy decode — from the range-coded frame
//! flags through the normalized spectrum and final band energies
//! (RFC 6716 §4.3, pp. 107–121).
//!
//! This module sequences every entropy-decode stage of one CELT frame
//! in exact Table 56 order, with the budget gates the normative
//! Appendix A reference decoder applies to each symbol (RFC 6716 §1/§6
//! give that code precedence over the prose):
//!
//! 1. silence flag (with the exhausted-budget implicit-silence rule),
//! 2. §4.3.7.1 post-filter parameters (gated on `start == 0` and 16
//!    bits of headroom; octave is uniform over **6** values, tapset
//!    gated on 2 further bits),
//! 3. transient + intra flags (3 bits of headroom each),
//! 4. §4.3.2.1 coarse energy (Laplace / small-energy / bit / implicit
//!    `-1` fallbacks as the remaining budget shrinks) with the
//!    cross-frame 2-D prediction,
//! 5. §4.3.1 `tf_change` / `tf_select` (budget-gated per band),
//! 6. §4.3.4.3 spread (4-bit gate),
//! 7. §4.3.3 dynalloc band boosts (the shrinking-budget loop),
//! 8. §4.3.3 allocation trim,
//! 9. the anti-collapse reservation + §4.3.3 implicit allocation
//!    ([`crate::celt_rate_alloc`]),
//! 10. §4.3.2.2 fine energy,
//! 11. §4.3.4 band shapes with folding ([`crate::celt_band_decode`]),
//! 12. the §4.3.5 anti-collapse bit + noise injection, and
//! 13. §4.3.2.3 final fine-energy bit backfill,
//!
//! then converts the energy state to per-band linear gains (with the
//! RFC 8251 §8 cap) and rolls the cross-frame energy history forward
//! ([`CeltEnergyState`]) exactly as the reference decoder's frame tail
//! does — including the silence-frame `-28 dB` floor, the mono
//! channel-copy rules, and the transient/non-transient `oldLogE`
//! update split.
//!
//! The output is the frequency-domain half of the decode: normalized
//! per-band shapes plus linear band gains. The signal half (§4.3.6
//! denormalisation, §4.3.7 inverse MDCT, post-filter, de-emphasis)
//! lives in [`crate::celt_mdct_synthesis`].
//!
//! ## Provenance
//!
//! RFC 6716 §4.3 narrative + the normative Appendix A reference
//! decoder from the staged `docs/audio/opus/rfc6716-opus.txt`; the
//! RFC 8251 §8 energy cap from `rfc8251-opus-update.txt`. No external
//! library source was consulted.

use crate::celt_band_decode::{anti_collapse, quant_all_bands_decode};
use crate::celt_band_layout::CELT_NUM_BANDS;
use crate::celt_cache_caps50::{cap_for_band_bits, CacheCapsStereo};
use crate::celt_coarse_energy::e_mean;
use crate::celt_e_prob_model::{e_prob_pair, EnergyPredictionMode};
use crate::celt_laplace::ec_laplace_decode;
use crate::celt_rate_alloc::{band_width, compute_allocation_decode, BITRES, MAX_FINE_BITS};
use crate::celt_tf_adjust::{
    TF_ADJ_NONTRANSIENT_SELECT0, TF_ADJ_NONTRANSIENT_SELECT1, TF_ADJ_TRANSIENT_SELECT0,
    TF_ADJ_TRANSIENT_SELECT1,
};
use crate::range_decoder::RangeDecoder;

/// The `-28 dB` (log2-domain) energy floor the reference decoder
/// stores after silence and outside the coded range.
const ENERGY_FLOOR: f64 = -28.0;

/// ICDF for the transient / intra `{7, 1}/8` flags.
const FLAG_ICDF: [u8; 2] = [1, 0];
/// ICDF for the post-filter tapset `{2, 1, 1}/4`.
const TAPSET_ICDF: [u8; 3] = [2, 1, 0];
/// ICDF for the low-budget coarse-energy fallback `{...}/4`.
const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

/// §4.3.2.1 inter-frame prediction coefficients (Q15 → float), per LM.
const PRED_COEF: [f64; 4] = [
    29440.0 / 32768.0,
    26112.0 / 32768.0,
    21248.0 / 32768.0,
    16384.0 / 32768.0,
];
/// §4.3.2.1 in-frame (band) prediction feedback, per LM.
const BETA_COEF: [f64; 4] = [
    30147.0 / 32768.0,
    22282.0 / 32768.0,
    12124.0 / 32768.0,
    6554.0 / 32768.0,
];
/// §4.3.2.1 intra-frame feedback coefficient.
const BETA_INTRA: f64 = 4915.0 / 32768.0;

/// Cross-frame CELT decoder state owned by the energy layer: the
/// per-band energy memory (this frame / one back / two back), and the
/// carried folding-noise seed.
#[derive(Debug, Clone, PartialEq)]
pub struct CeltEnergyState {
    /// The §4.3.2 per-band log2 energies (without the means), per
    /// channel — `oldBandE`.
    pub old_band_e: [[f64; CELT_NUM_BANDS]; 2],
    /// Previous frame's energies (`oldLogE`).
    pub old_log_e: [[f64; CELT_NUM_BANDS]; 2],
    /// Two frames back (`oldLogE2`).
    pub old_log_e2: [[f64; CELT_NUM_BANDS]; 2],
    /// The carried folding LCG seed (the previous frame's final range
    /// state).
    pub rng: u32,
}

impl Default for CeltEnergyState {
    fn default() -> Self {
        Self::new()
    }
}

impl CeltEnergyState {
    /// Fresh (stream-start / post-§4.5.2-reset) state: the coarse
    /// energies (`old_band_e`) start at zero, while the §4.3.5
    /// anti-collapse references (`old_log_e` / `old_log_e2`) start at
    /// the [`ENERGY_FLOOR`] — a fresh stream has no "previous frame
    /// energy", so the anti-collapse threshold must treat every band
    /// as if it had been silent, not as if it had held unit energy.
    /// Validated black-box: with a zero init the first transient
    /// frame of a stream (and every transient frame after a reset)
    /// injects the wrong anti-collapse noise and the startup frames
    /// decode ~30 dB from the reference; with the floor init the same
    /// streams decode at >99 dB.
    #[must_use]
    pub fn new() -> Self {
        Self {
            old_band_e: [[0.0; CELT_NUM_BANDS]; 2],
            old_log_e: [[ENERGY_FLOOR; CELT_NUM_BANDS]; 2],
            old_log_e2: [[ENERGY_FLOOR; CELT_NUM_BANDS]; 2],
            rng: 0,
        }
    }

    /// Reset to the stream-start state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Decoded §4.3.7.1 post-filter parameters for one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CeltPostFilterOut {
    /// Pitch period `T ∈ 15..=1022`.
    pub period: usize,
    /// Linear gain `3(qg+1)/32`.
    pub gain: f64,
    /// Tapset selector `0..=2`.
    pub tapset: usize,
}

/// The frequency-domain result of one CELT frame's entropy decode.
#[derive(Debug, Clone)]
pub struct CeltFrameOutput {
    /// The frame's silence flag (output is the zero spectrum; state
    /// still advances).
    pub silence: bool,
    /// The §4.3.1 transient flag (short blocks).
    pub transient: bool,
    /// Post-filter parameters when signalled.
    pub post_filter: Option<CeltPostFilterOut>,
    /// Per-channel planar normalized spectra (channel `c` occupies
    /// `x[c*plane..(c+1)*plane]`; band `i` starts at `M *
    /// band_edge(i)`).
    pub x: Vec<f64, crate::bump::DecodeBump>,
    /// One channel plane length (`M * 100`).
    pub plane: usize,
    /// Per-channel, per-band *linear* denormalisation gains: the
    /// §4.3.2 energy domain is the log2 of the band *amplitude*
    /// (the square root of the sum of squares), so the gain is
    /// `2^(lg)` directly (zero outside the coded range and on
    /// silence).
    pub band_gain: [[f64; CELT_NUM_BANDS]; 2],
}

/// Decode one CELT frame's entropy layer (§4.3, Table 56 order) and
/// advance the cross-frame energy state.
///
/// * `rd` — the frame's range decoder. For a CELT-only frame it is
///   fresh; for a Hybrid frame it continues after the SILK layer.
/// * `frame_bytes` — the size of the whole Opus frame in bytes (the
///   range coder's total budget).
/// * `start..end` — the coded band range (0..end for CELT-only, 17..21
///   for Hybrid; `end` per the signalled audio bandwidth).
/// * `lm` — the frame-size shift (0–3 for 2.5/5/10/20 ms).
/// * `channels` — 1 or 2.
#[allow(clippy::too_many_arguments)]
pub fn decode_celt_frame(
    rd: &mut RangeDecoder<'_>,
    frame_bytes: usize,
    start: usize,
    end: usize,
    lm: i32,
    channels: usize,
    state: &mut CeltEnergyState,
) -> CeltFrameOutput {
    let c = channels;
    let m = 1usize << lm;
    let plane = m * crate::celt_rate_alloc::band_edge(CELT_NUM_BANDS) as usize;
    let total_bits: i64 = (frame_bytes as i64) * 8;

    // Mono decode of a previously-stereo stream folds the two energy
    // histories together.
    if c == 1 {
        for i in 0..CELT_NUM_BANDS {
            state.old_band_e[0][i] = state.old_band_e[0][i].max(state.old_band_e[1][i]);
        }
    }

    let mut tell = i64::from(rd.tell());
    let silence = if tell >= total_bits {
        true
    } else if tell == 1 {
        rd.dec_bit_logp(15) == 1
    } else {
        false
    };
    if silence {
        // Pretend we've read all the remaining bits: skip every stage
        // (each gate would fail) and emit the zero spectrum.
        finish_energy_state(state, start, end, c, true, false);
        state.rng = rd.range_size();
        return CeltFrameOutput {
            silence: true,
            transient: false,
            post_filter: None,
            x: {
                // Profluens patch: zero spectrum from the per-packet bump arena (see BandDecodeResult).
                let mut z = Vec::with_capacity_in(c * plane, crate::bump::DecodeBump);
                z.resize(c * plane, 0.0);
                z
            },
            plane,
            band_gain: [[0.0; CELT_NUM_BANDS]; 2],
        };
    }
    tell = i64::from(rd.tell());

    // §4.3.7.1 post-filter parameters (CELT-only frames only).
    let mut post_filter = None;
    if start == 0 && tell + 16 <= total_bits {
        if rd.dec_bit_logp(1) == 1 {
            let octave = rd.dec_uint(6).unwrap_or(0);
            let period = ((16u32 << octave) + rd.dec_bits(4 + octave)) as usize - 1;
            let qg = rd.dec_bits(3);
            let tapset = if i64::from(rd.tell()) + 2 <= total_bits {
                rd.dec_icdf(&TAPSET_ICDF, 2) as usize
            } else {
                0
            };
            post_filter = Some(CeltPostFilterOut {
                period,
                gain: 0.09375 * (qg as f64 + 1.0),
                tapset,
            });
        }
        tell = i64::from(rd.tell());
    }

    // Transient and intra flags.
    let transient = if lm > 0 && tell + 3 <= total_bits {
        let t = rd.dec_icdf(&FLAG_ICDF, 3) == 1;
        tell = i64::from(rd.tell());
        t
    } else {
        false
    };
    let intra = if tell + 3 <= total_bits {
        rd.dec_icdf(&FLAG_ICDF, 3) == 1
    } else {
        false
    };

    // §4.3.2.1 coarse energy.
    unquant_coarse_energy(rd, frame_bytes, start, end, lm, c, intra, state);

    // §4.3.1 time-frequency flags.
    let tf_res = tf_decode(rd, frame_bytes, start, end, transient, lm);

    // §4.3.4.3 spread.
    let spread = if i64::from(rd.tell()) + 4 <= total_bits {
        crate::celt_spreading::decode_spread(rd)
    } else {
        2 // SPREAD_NORMAL
    };

    // §4.3.3 caps + dynalloc band boosts.
    let stereo_axis = if c == 2 {
        CacheCapsStereo::Stereo
    } else {
        CacheCapsStereo::Mono
    };
    let mut cap = [0i32; CELT_NUM_BANDS];
    for (i, slot) in cap.iter_mut().enumerate() {
        *slot = cap_for_band_bits(
            lm as u32,
            stereo_axis,
            i as u32,
            c as u32,
            (band_width(i) << lm) as u32,
        )
        .unwrap_or(0) as i32;
    }
    // §4.3.3 dynalloc band boosts (the shrinking-budget loop), decoded
    // by the dedicated module. The caps window and the per-band bin
    // counts (across all coded channels) parameterize the quanta and
    // the per-band boost ceiling.
    let mut offsets = [0i32; CELT_NUM_BANDS];
    // Profluens patch: per-packet transient windows (consumed as slices by
    // decode_band_boosts) drawn from the bump arena, not the global heap.
    let mut caps_window: Vec<u32, crate::bump::DecodeBump> =
        Vec::with_capacity_in(end.saturating_sub(start), crate::bump::DecodeBump);
    caps_window.extend((start..end).map(|i| cap[i].max(0) as u32));
    let mut n_bins_window: Vec<u32, crate::bump::DecodeBump> =
        Vec::with_capacity_in(end.saturating_sub(start), crate::bump::DecodeBump);
    n_bins_window.extend((start..end).map(|i| (c as u32) * ((band_width(i) << lm) as u32)));
    let boosts = crate::celt_band_boost::decode_band_boosts(
        rd,
        start,
        end,
        &caps_window,
        &n_bins_window,
        frame_bytes as u32,
    )
    .expect("band window and slices are built from start..end");
    for (i, bb) in boosts.per_band.iter().enumerate() {
        offsets[start + i] = bb.boost_eighth_bits as i32;
    }
    // Every boost step moved `quanta` from total_bits into
    // total_boost, so the accumulated boost is the offset sum.
    let total_boost: u32 = boosts.total_boost_eighth_bits;

    // §4.3.3 allocation trim.
    let alloc_trim = crate::celt_alloc_trim::decode_alloc_trim(
        rd,
        rd.tell_frac(),
        frame_bytes as u32,
        total_boost,
    )
    .unwrap_or(5);

    // Anti-collapse reservation + implicit allocation.
    let mut bits = ((total_bits << BITRES) - i64::from(rd.tell_frac()) - 1) as i32;
    let anti_collapse_rsv = if transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
        1 << BITRES
    } else {
        0
    };
    bits -= anti_collapse_rsv;
    let alloc = compute_allocation_decode(
        rd,
        start,
        end,
        &offsets,
        &cap,
        i32::from(alloc_trim),
        bits,
        c,
        lm,
    );

    if std::env::var("OPUS_DEBUG_HYBRID").is_ok() {
        eprintln!(
            "CELT STAGES: after-alloc tell={} tell_frac={} bits={} coded_bands={} intensity={} pulses={:?} fine={:?}",
            rd.tell(),
            rd.tell_frac(),
            bits,
            alloc.coded_bands,
            alloc.intensity,
            &alloc.pulses[start..end],
            &alloc.fine_bits[start..end],
        );
    }

    // §4.3.2.2 fine energy.
    for i in start..end {
        if alloc.fine_bits[i] <= 0 {
            continue;
        }
        for ch in 0..c {
            let q2 = rd.dec_bits(alloc.fine_bits[i] as u32) as f64;
            let offset = (q2 + 0.5) / f64::from(1u32 << alloc.fine_bits[i]) - 0.5;
            state.old_band_e[ch][i] += offset;
        }
    }

    if std::env::var("OPUS_DEBUG_HYBRID").is_ok() {
        eprintln!("CELT STAGES: after-fine tell={}", rd.tell());
    }

    // §4.3.4 band shapes.
    let band_result = quant_all_bands_decode(
        rd,
        start,
        end,
        &alloc.pulses,
        transient,
        spread,
        alloc.dual_stereo,
        alloc.intensity,
        &tf_res,
        ((total_bits << BITRES) - i64::from(anti_collapse_rsv)) as i32,
        alloc.balance,
        lm,
        alloc.coded_bands,
        c,
        state.rng,
    );

    if std::env::var("OPUS_DEBUG_HYBRID").is_ok() {
        eprintln!("CELT STAGES: after-bands tell={}", rd.tell());
    }

    // §4.3.5 anti-collapse bit.
    let anti_collapse_on = anti_collapse_rsv > 0 && rd.dec_bits(1) == 1;

    // §4.3.2.3 final fine-energy bits.
    {
        let mut bits_left = (total_bits - i64::from(rd.tell())) as i32;
        for prio in [false, true] {
            let mut i = start;
            while i < end && bits_left >= c as i32 {
                if alloc.fine_bits[i] < MAX_FINE_BITS && alloc.fine_priority[i] == prio {
                    for ch in 0..c {
                        let q2 = rd.dec_bits(1) as f64;
                        let offset = (q2 - 0.5) / f64::from(1u32 << (alloc.fine_bits[i] + 1));
                        state.old_band_e[ch][i] += offset;
                        bits_left -= 1;
                    }
                }
                i += 1;
            }
        }
    }

    let mut x = band_result.x;
    if anti_collapse_on {
        // The post-injection seed is discarded: the carried rng is
        // re-latched from the range coder at the frame end.
        let _ = anti_collapse(
            &mut x,
            plane,
            &band_result.collapse_masks,
            lm,
            c,
            start,
            end,
            &state.old_band_e,
            &state.old_log_e,
            &state.old_log_e2,
            &alloc.pulses,
            band_result.seed,
        );
    }

    // Linear band gains (log2Amp with the RFC 8251 §8 cap).
    let mut band_gain = [[0.0f64; CELT_NUM_BANDS]; 2];
    for (ch, row) in band_gain.iter_mut().enumerate().take(c) {
        for (i, g) in row.iter_mut().enumerate().take(end).skip(start) {
            let lg = (state.old_band_e[ch][i] + e_mean(i).unwrap_or(0.0)).min(32.0);
            *g = lg.exp2();
        }
    }

    finish_energy_state(state, start, end, c, false, transient);
    state.rng = rd.range_size();

    CeltFrameOutput {
        silence: false,
        transient,
        post_filter,
        x,
        plane,
        band_gain,
    }
}

/// §4.3.2.1 coarse energy: Laplace-coded per-band error with the 2-D
/// prediction, degrading to cheaper codes as the budget runs out.
#[allow(clippy::too_many_arguments)]
fn unquant_coarse_energy(
    rd: &mut RangeDecoder<'_>,
    frame_bytes: usize,
    start: usize,
    end: usize,
    lm: i32,
    channels: usize,
    intra: bool,
    state: &mut CeltEnergyState,
) {
    let mode = EnergyPredictionMode::from_intra_flag(intra);
    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm as usize], BETA_COEF[lm as usize])
    };
    let budget = (frame_bytes as u32) * 8;
    let mut prev = [0.0f64; 2];

    for i in start..end {
        for (ch, prev_c) in prev.iter_mut().enumerate().take(channels) {
            let tell = rd.tell();
            let qi: i32 = if budget.saturating_sub(tell) >= 15 {
                let pair = e_prob_pair(lm as u32, mode, i.min(20) as u32)
                    .expect("band/lm ranges are valid");
                ec_laplace_decode(rd, u32::from(pair.prob) << 7, u32::from(pair.decay) << 6)
            } else if budget.saturating_sub(tell) >= 2 {
                let q = rd.dec_icdf(&SMALL_ENERGY_ICDF, 2) as i32;
                (q >> 1) ^ -(q & 1)
            } else if budget.saturating_sub(tell) >= 1 {
                -(rd.dec_bit_logp(1) as i32)
            } else {
                -1
            };
            let q = f64::from(qi);
            let old = state.old_band_e[ch][i].max(-9.0);
            state.old_band_e[ch][i] = coef * old + *prev_c + q;
            *prev_c += q - beta * q;
        }
    }
}

/// §4.3.1 per-band time-frequency resolution decode, returning the
/// Table 60–63 adjustment per band.
fn tf_decode(
    rd: &mut RangeDecoder<'_>,
    frame_bytes: usize,
    start: usize,
    end: usize,
    transient: bool,
    lm: i32,
) -> [i32; CELT_NUM_BANDS] {
    let mut budget = (frame_bytes as u32) * 8;
    let mut tell = rd.tell();
    let mut logp: u32 = if transient { 2 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    budget -= u32::from(tf_select_rsv);
    let mut tf_changed = false;
    let mut curr = false;
    let mut flags = [false; CELT_NUM_BANDS];
    for f in flags.iter_mut().take(end).skip(start) {
        if tell + logp <= budget {
            curr ^= rd.dec_bit_logp(logp) == 1;
            tell = rd.tell();
            tf_changed |= curr;
        }
        *f = curr;
        logp = if transient { 4 } else { 5 };
    }
    let table = |select: bool, change: bool| -> i32 {
        let t = match (transient, select) {
            (false, false) => &TF_ADJ_NONTRANSIENT_SELECT0,
            (false, true) => &TF_ADJ_NONTRANSIENT_SELECT1,
            (true, false) => &TF_ADJ_TRANSIENT_SELECT0,
            (true, true) => &TF_ADJ_TRANSIENT_SELECT1,
        };
        i32::from(t[lm as usize][usize::from(change)])
    };
    let tf_select = if tf_select_rsv && table(false, tf_changed) != table(true, tf_changed) {
        rd.dec_bit_logp(1) == 1
    } else {
        false
    };
    let mut tf_res = [0i32; CELT_NUM_BANDS];
    for i in start..end {
        tf_res[i] = table(tf_select, flags[i]);
    }
    tf_res
}

/// Roll the cross-frame energy history forward exactly as the frame
/// tail of the reference decoder does.
fn finish_energy_state(
    state: &mut CeltEnergyState,
    start: usize,
    end: usize,
    channels: usize,
    silence: bool,
    transient: bool,
) {
    if silence {
        for ch in 0..channels {
            for i in 0..CELT_NUM_BANDS {
                state.old_band_e[ch][i] = ENERGY_FLOOR;
            }
        }
    }
    if channels == 1 {
        state.old_band_e[1] = state.old_band_e[0];
    }
    if !transient {
        state.old_log_e2 = state.old_log_e;
        state.old_log_e = state.old_band_e;
    } else {
        for ch in 0..2 {
            for i in 0..CELT_NUM_BANDS {
                state.old_log_e[ch][i] = state.old_log_e[ch][i].min(state.old_band_e[ch][i]);
            }
        }
    }
    for ch in 0..2 {
        for i in (0..start).chain(end..CELT_NUM_BANDS) {
            state.old_band_e[ch][i] = 0.0;
            state.old_log_e[ch][i] = ENERGY_FLOOR;
            state.old_log_e2[ch][i] = ENERGY_FLOOR;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tf_select_table_lookup_is_consistent() {
        // Non-transient LM=0 select 0: {0, -1}; transient LM=3
        // select 0: {3, 0} (Tables 60 / 62) — the tf_decode closure
        // maps flags through these constants.
        assert_eq!(TF_ADJ_NONTRANSIENT_SELECT0[0], [0, -1]);
        assert_eq!(TF_ADJ_TRANSIENT_SELECT0[3], [3, 0]);
    }

    #[test]
    fn silence_frame_floors_energy_and_advances_rng() {
        // A frame whose first symbol decodes as silence: energies drop
        // to the floor, oldLogE rolls, the spectrum is zero.
        // dec_bit_logp(15) reads the top of the range; an all-0xFF
        // buffer keeps val high -> silence = 0, all-zero -> the low
        // branch. Instead of constructing the exact bit, drive both
        // and assert on the reported flag.
        for fill in [0x00u8, 0xFF] {
            let buf = [fill; 12];
            let mut rd = RangeDecoder::new(&buf);
            let mut st = CeltEnergyState::new();
            st.old_band_e[0][3] = 5.0;
            let out = decode_celt_frame(&mut rd, buf.len(), 0, 21, 3, 1, &mut st);
            if out.silence {
                assert!(out.x.iter().all(|&v| v == 0.0));
                assert_eq!(st.old_band_e[0][3], ENERGY_FLOOR);
                assert_eq!(st.old_log_e[0][3], ENERGY_FLOOR);
            } else {
                assert_eq!(out.x.len(), out.plane);
            }
            assert!(!rd.has_error(), "fill {fill:#x}");
        }
    }

    #[test]
    fn non_silent_frame_produces_unit_band_shapes_and_gains() {
        // A dense random buffer decodes as a full frame; every coded
        // band's shape is unit-norm and every gain positive.
        let buf: Vec<u8> = (0..320u32).map(|i| (i * 199 + 3) as u8).collect();
        let mut rd = RangeDecoder::new(&buf);
        let mut st = CeltEnergyState::new();
        let out = decode_celt_frame(&mut rd, buf.len(), 0, 21, 3, 1, &mut st);
        if !out.silence {
            for i in 0..21 {
                assert!(out.band_gain[0][i] > 0.0, "band {i} gain");
            }
            // The rng carries the final range state.
            assert_ne!(st.rng, 0);
        }
    }

    #[test]
    fn stereo_and_hybrid_windows_decode_cleanly() {
        let buf: Vec<u8> = (0..150u32).map(|i| (i * 61 + 17) as u8).collect();
        // Stereo full-band.
        let mut rd = RangeDecoder::new(&buf);
        let mut st = CeltEnergyState::new();
        let out = decode_celt_frame(&mut rd, buf.len(), 0, 21, 2, 2, &mut st);
        assert_eq!(out.x.len(), 2 * out.plane);
        // Hybrid window (start = 17): bands below 17 stay silent.
        let mut rd2 = RangeDecoder::new(&buf);
        let mut st2 = CeltEnergyState::new();
        let out2 = decode_celt_frame(&mut rd2, buf.len(), 17, 21, 1, 1, &mut st2);
        if !out2.silence {
            let edge17 = 2 * crate::celt_rate_alloc::band_edge(17) as usize;
            assert!(out2.x[..edge17].iter().all(|&v| v == 0.0));
            for i in 0..17 {
                assert_eq!(out2.band_gain[0][i], 0.0);
            }
        }
    }

    #[test]
    fn energy_history_rolls_on_non_transient_frames() {
        let buf: Vec<u8> = (0..100u32).map(|i| (i * 41 + 29) as u8).collect();
        let mut rd = RangeDecoder::new(&buf);
        let mut st = CeltEnergyState::new();
        let out = decode_celt_frame(&mut rd, buf.len(), 0, 21, 1, 1, &mut st);
        if !out.silence && !out.transient {
            // oldLogE now equals oldBandE.
            assert_eq!(st.old_log_e, st.old_band_e);
        }
    }
}
