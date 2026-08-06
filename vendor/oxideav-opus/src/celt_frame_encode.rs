//! CELT whole-frame **encode** — the §5.3 stage sequencing that
//! mirrors [`crate::celt_frame_decode`] symbol for symbol: silence,
//! post-filter (off), transient, intra + §5.3.2 coarse energy, tf
//! flags, spread, dynalloc boosts, trim, the §4.3.3 allocation with
//! the coded skip/intensity/dual decisions, fine energy, §4.3.4 band
//! shapes, the anti-collapse bit, and the final fine backfill.
//!
//! Encoder *decisions* (transient detection, dynalloc offsets, trim
//! analysis, the stereo intensity/dual heuristics) are the reference
//! listing's; every coded symbol goes through the exact write-side
//! mirrors so the decoder walks the identical recursion.
//!
//! ## Provenance
//!
//! RFC 6716 §5.3 + the normative Appendix A reference listing (staged
//! `docs/audio/opus/rfc6716-opus.txt`, hash-verified per §A.1). No
//! external library source was consulted.

use crate::celt_alloc_encode::compute_allocation_encode;
use crate::celt_analysis::{
    amp_to_log2, compute_band_energies, forward_mdct, normalise_bands, transient_analysis,
    CeltAnalysis,
};
use crate::celt_band_encode::quant_all_bands_encode;
use crate::celt_band_layout::CELT_NUM_BANDS;
use crate::celt_cache_caps50::{cap_for_band_bits, CacheCapsStereo};
use crate::celt_energy_encode::{
    quant_coarse_energy, quant_energy_finalise, quant_fine_energy, EnergyGrid,
};
use crate::celt_rate_alloc::{band_edge, band_width, BITRES};
use crate::celt_spreading::SPREAD_ICDF;
use crate::celt_tf_adjust::{
    TF_ADJ_NONTRANSIENT_SELECT0, TF_ADJ_NONTRANSIENT_SELECT1, TF_ADJ_TRANSIENT_SELECT0,
    TF_ADJ_TRANSIENT_SELECT1,
};
use crate::range_encoder::RangeEncoder;

/// The `-28 dB` energy floor (mirrors the decoder's).
const ENERGY_FLOOR: f64 = -28.0;

/// Trim ICDF (Table 58).
const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];

/// Cross-frame CELT encoder state.
#[derive(Debug, Clone)]
pub struct CeltEncoderState {
    /// Pre-emphasis + MDCT input buffering.
    pub analysis: CeltAnalysis,
    /// The quantized energy state the decoder reconstructs
    /// (`oldBandE`), kept in lockstep.
    pub old_band_e: EnergyGrid,
    /// §5.3.2 delayed-intra accumulator.
    pub delayed_intra: f64,
    /// Consecutive-transient counter (anti-collapse decision).
    pub consec_transient: u32,
    /// Previous frame's coded-band count (`lastCodedBands`).
    pub last_coded_bands: usize,
    /// Force an intra frame next (stream start / reset).
    pub force_intra: bool,
    /// §5.3.4 spreading-decision recursive tonality average.
    pub tonal_average: i32,
    /// Previous frame's coded spread decision (hysteresis input).
    pub spread_decision: u8,
    channels: usize,
    n: usize,
}

/// Reported outcome of one frame encode (diagnostics for tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CeltFrameEncodeInfo {
    /// The coded silence flag.
    pub silence: bool,
    /// The coded transient flag.
    pub transient: bool,
    /// The coded intra flag.
    pub intra: bool,
    /// Coded band count from the allocation.
    pub coded_bands: usize,
}

impl CeltEncoderState {
    /// New encoder state for `channels` channels and `n` 48 kHz
    /// samples per frame (120/240/480/960).
    #[must_use]
    pub fn new(channels: usize, n: usize) -> Self {
        Self {
            analysis: CeltAnalysis::new(channels, n),
            old_band_e: [[0.0; CELT_NUM_BANDS]; 2],
            delayed_intra: 1.0,
            consec_transient: 0,
            last_coded_bands: 0,
            force_intra: true,
            tonal_average: 256,
            spread_decision: 2,
            channels,
            n,
        }
    }

    /// Frame length per channel.
    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.n
    }

    /// Channel count.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Reset to stream-start state (§4.5.2).
    pub fn reset(&mut self) {
        self.analysis.reset();
        self.old_band_e = [[0.0; CELT_NUM_BANDS]; 2];
        self.delayed_intra = 1.0;
        self.consec_transient = 0;
        self.last_coded_bands = 0;
        self.force_intra = true;
        self.tonal_average = 256;
        self.spread_decision = 2;
    }
}

/// Encode one CELT frame into `enc` (already carrying any earlier
/// layers, e.g. the Hybrid SILK layer), against a total budget of
/// `frame_bytes` for the whole Opus frame.
///
/// * `pcm` — `channels * n` interleaved i16 samples at 48 kHz.
/// * `start..end` — the coded band range (0 for CELT-only; 17 for
///   Hybrid), `end` per the audio bandwidth (13/17/19/21).
/// * `lm` — the frame-size shift (0–3).
pub fn encode_celt_frame(
    state: &mut CeltEncoderState,
    enc: &mut RangeEncoder,
    pcm: &[i16],
    frame_bytes: usize,
    start: usize,
    end: usize,
    lm: i32,
) -> CeltFrameEncodeInfo {
    let channels = state.channels;
    let n = state.n;
    let m = 1usize << lm;
    let plane = m * band_edge(CELT_NUM_BANDS) as usize;
    let total_bits: i64 = frame_bytes as i64 * 8;

    let mut tell = i64::from(enc.tell());
    let nb_filled_bytes = (tell + 4) >> 3;
    let nb_available_bytes = frame_bytes as i64 - nb_filled_bytes;
    let effective_bytes = nb_available_bytes;

    // Pre-emphasis + input buffering (also detects digital silence).
    let frame = state.analysis.process_frame(pcm);

    let silence = frame.silence && tell == 1;
    if tell == 1 {
        enc.enc_bit_logp(silence, 15);
    }
    if silence {
        // Every remaining gate fails on the decoder side; mirror its
        // energy-state roll (finish_energy_state with silence).
        for ch in 0..2 {
            for i in 0..CELT_NUM_BANDS {
                state.old_band_e[ch][i] = ENERGY_FLOOR;
            }
        }
        state.consec_transient = 0;
        state.force_intra = false;
        return CeltFrameEncodeInfo {
            silence: true,
            transient: false,
            intra: false,
            coded_bands: 0,
        };
    }
    tell = i64::from(enc.tell());

    // §4.3.7.1 post-filter: signalled off.
    if start == 0 && tell + 16 <= total_bits {
        enc.enc_bit_logp(false, 1);
        tell = i64::from(enc.tell());
    }

    // Transient analysis + flag.
    let mut transient = false;
    if lm > 0 && tell + 3 <= total_bits {
        transient = transient_analysis(&frame);
        enc.enc_bit_logp(transient, 3);
    }
    let blocks = if transient { m } else { 1 };

    // Forward MDCTs + energy analysis.
    let mut band_e: EnergyGrid = [[0.0; CELT_NUM_BANDS]; 2];
    let mut band_log_e: EnergyGrid = [[0.0; CELT_NUM_BANDS]; 2];
    let mut x = vec![0.0f64; channels * plane];
    for c in 0..channels {
        let ibuf = &frame.ibuf[c * (n + frame.overlap)..(c + 1) * (n + frame.overlap)];
        let freq = forward_mdct(ibuf, n, frame.overlap, blocks);
        let be = compute_band_energies(&freq, end, m);
        let ble = amp_to_log2(&be, end);
        let xs = normalise_bands(&freq, &be, end, m);
        band_e[c] = be;
        band_log_e[c] = ble;
        x[c * plane..(c + 1) * plane].copy_from_slice(&xs);
        let _ = freq;
    }

    // tf_res: no per-band time-frequency changes (a valid encoder
    // choice; the flags still consume their budgeted bits).
    let mut tf_res = [0i32; CELT_NUM_BANDS];

    // §5.3.2 coarse energy (two-pass intra/inter).
    let mut error: EnergyGrid = [[0.0; CELT_NUM_BANDS]; 2];
    let intra = quant_coarse_energy(
        enc,
        start,
        end,
        &band_log_e,
        &mut state.old_band_e,
        total_bits as u32,
        &mut error,
        channels,
        lm as usize,
        nb_available_bytes as i32,
        state.force_intra,
        &mut state.delayed_intra,
        true,
    );

    // §4.3.1 tf encode (all-zero changes, tf_select = 0).
    tf_encode(
        enc,
        start,
        end,
        transient,
        &mut tf_res,
        lm,
        total_bits as u32,
    );

    // §4.3.4.3 spread decision (the listing's tonality analysis; at
    // full complexity the analysed decision is used except on
    // transients, which keep SPREAD_NORMAL).
    let mut spread: u8 = 2; // SPREAD_NORMAL
    state.spread_decision = 2;
    if i64::from(enc.tell()) + 4 <= total_bits {
        if !transient {
            spread = spreading_decision(
                &x,
                &mut state.tonal_average,
                state.spread_decision,
                end,
                channels,
                m,
                plane,
            );
        }
        state.spread_decision = spread;
        enc.enc_icdf(usize::from(spread), &SPREAD_ICDF, 5);
    }

    // Caps + dynalloc offsets (the listing's band-peak rule).
    let stereo_axis = if channels == 2 {
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
            channels as u32,
            (band_width(i) << lm) as u32,
        )
        .unwrap_or(0) as i32;
    }
    let mut offsets = [0i32; CELT_NUM_BANDS];
    if effective_bytes > 50 && lm >= 1 {
        let (t1, t2) = if lm <= 1 { (3.0, 5.0) } else { (2.0, 4.0) };
        for i in (start + 1)..end.saturating_sub(1) {
            let mut d2 = 2.0 * band_log_e[0][i] - band_log_e[0][i - 1] - band_log_e[0][i + 1];
            if channels == 2 {
                d2 = 0.5
                    * (d2 + 2.0 * band_log_e[1][i] - band_log_e[1][i - 1] - band_log_e[1][i + 1]);
            }
            if d2 > t1 {
                offsets[i] += 1;
            }
            if d2 > t2 {
                offsets[i] += 1;
            }
        }
    }

    // Dynalloc boost encode (1/8-bit units from here on).
    let total_bits_q3 = (total_bits << BITRES) as i32;
    let mut dynalloc_logp: i32 = 6;
    let mut total_boost: i32 = 0;
    let mut tell_q3 = enc.tell_frac() as i32;
    for i in start..end {
        let width = (channels as i32) * (band_width(i) << lm);
        // quanta of 6 bits, but no more than 1 bit/sample and no less
        // than 1/8 bit/sample.
        let quanta = (width << BITRES).min((6 << BITRES).max(width));
        let mut dynalloc_loop_logp = dynalloc_logp;
        let mut boost = 0i32;
        let mut j = 0i32;
        while tell_q3 + (dynalloc_loop_logp << BITRES) < total_bits_q3 - total_boost
            && boost < cap[i]
        {
            let flag = j < offsets[i];
            enc.enc_bit_logp(flag, dynalloc_loop_logp as u32);
            tell_q3 = enc.tell_frac() as i32;
            if !flag {
                break;
            }
            boost += quanta;
            total_boost += quanta;
            dynalloc_loop_logp = 1;
            j += 1;
        }
        if j > 0 {
            dynalloc_logp = 2.max(dynalloc_logp - 1);
        }
        offsets[i] = boost;
    }

    // Allocation trim.
    let mut alloc_trim = 5i32;
    if tell_q3 + (6 << BITRES) <= total_bits_q3 - total_boost {
        alloc_trim = alloc_trim_analysis(&x, &band_log_e, end, lm, channels, plane);
        enc.enc_icdf(alloc_trim as usize, &TRIM_ICDF, 7);
        tell_q3 = enc.tell_frac() as i32;
    }
    let _ = tell_q3;

    // Stereo decisions (listing heuristics).
    let mut intensity = 0usize;
    let mut dual_stereo = false;
    if channels == 2 {
        // Always use MS for 2.5 ms frames.
        if lm != 0 {
            dual_stereo = stereo_analysis(&x, lm, plane);
        }
        let mut effective_rate = ((8 * effective_bytes - 80) >> lm) as i32;
        effective_rate = 2 * effective_rate / 5;
        let it = if effective_rate < 35 {
            8
        } else if effective_rate < 50 {
            12
        } else if effective_rate < 68 {
            16
        } else if effective_rate < 84 {
            18
        } else if effective_rate < 102 {
            19
        } else if effective_rate < 130 {
            20
        } else {
            100
        };
        intensity = it.clamp(start, end);
    }

    // §4.3.3 allocation.
    let mut bits = ((total_bits << BITRES) - i64::from(enc.tell_frac()) - 1) as i32;
    let anti_collapse_rsv = if transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
        1 << BITRES
    } else {
        0
    };
    bits -= anti_collapse_rsv;
    let alloc = compute_allocation_encode(
        enc,
        start,
        end,
        &offsets,
        &cap,
        alloc_trim,
        bits,
        channels,
        lm,
        intensity,
        dual_stereo,
        state.last_coded_bands,
    );
    state.last_coded_bands = alloc.coded_bands;

    // §4.3.2.2 fine energy.
    quant_fine_energy(
        enc,
        start,
        end,
        &mut state.old_band_e,
        &mut error,
        &alloc.fine_bits,
        channels,
    );

    // §4.3.4 band shapes.
    quant_all_bands_encode(
        enc,
        start,
        end,
        &mut x,
        channels,
        &band_e,
        &alloc.pulses,
        transient,
        spread,
        alloc.dual_stereo,
        alloc.intensity,
        &tf_res,
        (frame_bytes as i32) * (8 << BITRES) - anti_collapse_rsv,
        alloc.balance,
        lm,
        alloc.coded_bands,
    );

    // §4.3.5 anti-collapse bit.
    if anti_collapse_rsv > 0 {
        let on = state.consec_transient < 2;
        enc.enc_bits(u32::from(on), 1);
    }

    // §4.3.2.3 final fine bits.
    let bits_left = (total_bits - i64::from(enc.tell())) as i32;
    quant_energy_finalise(
        enc,
        start,
        end,
        &mut state.old_band_e,
        &error,
        &alloc.fine_bits,
        &alloc.fine_priority,
        bits_left,
        channels,
    );

    // Frame-tail state roll, mirroring the decoder's
    // finish_energy_state so predictions stay in lockstep.
    if channels == 1 {
        state.old_band_e[1] = state.old_band_e[0];
    }
    for ch in 0..2 {
        for i in (0..start).chain(end..CELT_NUM_BANDS) {
            state.old_band_e[ch][i] = 0.0;
        }
    }
    state.consec_transient = if transient {
        state.consec_transient + 1
    } else {
        0
    };
    state.force_intra = false;

    CeltFrameEncodeInfo {
        silence: false,
        transient,
        intra,
        coded_bands: alloc.coded_bands,
    }
}

/// §5.3.1 tf encode (the listing's `tf_encode`): difference-coded
/// per-band flags with the budget gates, plus the gated `tf_select`
/// bit. `tf_res` enters as the raw 0/1 change flags and leaves as the
/// Table 60–63 adjustments.
fn tf_encode(
    enc: &mut RangeEncoder,
    start: usize,
    end: usize,
    transient: bool,
    tf_res: &mut [i32; CELT_NUM_BANDS],
    lm: i32,
    total_bits: u32,
) {
    let mut budget = total_bits;
    let mut tell = enc.tell();
    let mut logp: u32 = if transient { 2 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    budget -= u32::from(tf_select_rsv);
    let mut curr = 0i32;
    let mut tf_changed = 0i32;
    for tf in tf_res.iter_mut().take(end).skip(start) {
        if tell + logp <= budget {
            enc.enc_bit_logp((*tf ^ curr) == 1, logp);
            tell = enc.tell();
            curr = *tf;
            tf_changed |= curr;
        } else {
            *tf = curr;
        }
        logp = if transient { 4 } else { 5 };
    }
    let table = |select: bool, change: i32| -> i32 {
        let t = match (transient, select) {
            (false, false) => &TF_ADJ_NONTRANSIENT_SELECT0,
            (false, true) => &TF_ADJ_NONTRANSIENT_SELECT1,
            (true, false) => &TF_ADJ_TRANSIENT_SELECT0,
            (true, true) => &TF_ADJ_TRANSIENT_SELECT1,
        };
        i32::from(t[lm as usize][usize::from(change != 0)])
    };
    // tf_select = 0; only code it when it would make a difference.
    let tf_select = false;
    if tf_select_rsv && table(false, tf_changed) != table(true, tf_changed) {
        enc.enc_bit_logp(tf_select, 1);
    }
    for tf in tf_res.iter_mut().take(end).skip(start) {
        *tf = table(tf_select, *tf);
    }
}

/// The listing's `spreading_decision`: a rough per-band CDF of the
/// normalized coefficient magnitudes measures tonality; a recursive
/// average plus hysteresis against the previous decision maps it to
/// one of the four Table 59 spread values.
fn spreading_decision(
    x: &[f64],
    average: &mut i32,
    last_decision: u8,
    end: usize,
    channels: usize,
    m: usize,
    plane: usize,
) -> u8 {
    if m * (band_edge(end) - band_edge(end - 1)) as usize <= 8 {
        return 0; // SPREAD_NONE
    }
    let mut sum = 0i32;
    let mut nb_bands = 0i32;
    for c in 0..channels {
        for i in 0..end {
            let off = c * plane + m * band_edge(i) as usize;
            let n = m * band_width(i) as usize;
            if n <= 8 {
                continue;
            }
            let mut tcount = [0i32; 3];
            for &v in &x[off..off + n] {
                let x2n = v * v * n as f64;
                if x2n < 0.25 {
                    tcount[0] += 1;
                }
                if x2n < 0.0625 {
                    tcount[1] += 1;
                }
                if x2n < 0.015625 {
                    tcount[2] += 1;
                }
            }
            let tmp = i32::from(2 * tcount[2] >= n as i32)
                + i32::from(2 * tcount[1] >= n as i32)
                + i32::from(2 * tcount[0] >= n as i32);
            sum += tmp * 256;
            nb_bands += 1;
        }
    }
    debug_assert!(nb_bands > 0);
    sum /= nb_bands.max(1);
    // Recursive averaging + hysteresis against the last decision.
    sum = (sum + *average) >> 1;
    *average = sum;
    sum = (3 * sum + (((3 - i32::from(last_decision)) << 7) + 64) + 2) >> 2;
    if sum < 80 {
        3 // SPREAD_AGGRESSIVE
    } else if sum < 256 {
        2 // SPREAD_NORMAL
    } else if sum < 384 {
        1 // SPREAD_LIGHT
    } else {
        0 // SPREAD_NONE
    }
}

/// The listing's `alloc_trim_analysis`: stereo low-band correlation
/// plus a spectral-tilt estimate over the log energies.
fn alloc_trim_analysis(
    x: &[f64],
    band_log_e: &EnergyGrid,
    end: usize,
    lm: i32,
    channels: usize,
    plane: usize,
) -> i32 {
    let m = 1usize << lm;
    let mut trim_index = 5i32;
    if channels == 2 {
        // Inter-channel correlation over the first 8 bands.
        let mut sum = 0.0f64;
        for i in 0..8 {
            let lo = m * band_edge(i) as usize;
            let hi = m * band_edge(i + 1) as usize;
            let mut partial = 0.0f64;
            for j in lo..hi {
                partial += x[j] * x[plane + j];
            }
            sum += partial;
        }
        sum *= 1.0 / 8.0;
        if sum > 0.995 {
            trim_index -= 4;
        } else if sum > 0.92 {
            trim_index -= 3;
        } else if sum > 0.85 {
            trim_index -= 2;
        } else if sum > 0.8 {
            trim_index -= 1;
        }
    }
    // Spectral tilt.
    let mut diff = 0.0f64;
    for row in band_log_e.iter().take(channels) {
        for (i, v) in row.iter().enumerate().take(end - 1) {
            diff += v * f64::from(2 + 2 * i as i32 - CELT_NUM_BANDS as i32);
        }
    }
    diff /= f64::from(2 * channels as i32 * (end as i32 - 1));
    if diff > 2.0 {
        trim_index -= 1;
    }
    if diff > 8.0 {
        trim_index -= 1;
    }
    if diff < -4.0 {
        trim_index += 1;
    }
    if diff < -10.0 {
        trim_index += 1;
    }
    trim_index.clamp(0, 10)
}

/// The listing's `stereo_analysis`: L1-norm entropy model of L/R vs
/// M/S over the first 13 bands.
fn stereo_analysis(x: &[f64], lm: i32, plane: usize) -> bool {
    let m = 1usize << lm;
    let mut sum_lr = 1e-15f64;
    let mut sum_ms = 1e-15f64;
    for i in 0..13 {
        let lo = m * band_edge(i) as usize;
        let hi = m * band_edge(i + 1) as usize;
        for j in lo..hi {
            let l = x[j];
            let r = x[plane + j];
            sum_lr += l.abs() + r.abs();
            sum_ms += (l + r).abs() + (l - r).abs();
        }
    }
    sum_ms *= std::f64::consts::FRAC_1_SQRT_2;
    let mut thetas = 13i32;
    if lm <= 1 {
        thetas -= 8;
    }
    let edge13 = f64::from(band_edge(13) << (lm + 1));
    (edge13 + f64::from(thetas)) * sum_ms > edge13 * sum_lr
}
