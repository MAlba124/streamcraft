//! CELT §4.3.4 band decode — PVQ shapes, band splitting, stereo
//! merging, folding, and the §4.3.5 anti-collapse (RFC 6716 §4.3.4 /
//! §4.3.5, pp. 116–121).
//!
//! This is the decode-side port of the recursive band decoder the
//! normative Appendix A reference implementation specifies
//! (`quant_all_bands` / `quant_band`, bands.c, and `alg_unquant`,
//! vq.c; RFC 6716 §1/§6 give that code precedence over the prose).
//! For each coded band it:
//!
//! 1. computes the band's bit budget from the §4.3.3 allocation plus
//!    the running balance,
//! 2. recursively splits the band (§4.3.4.4) when its codebook would
//!    exceed 32 bits — decoding the entropy-coded split angle `itheta`
//!    (triangular / uniform / step PDF as the geometry dictates) and
//!    dividing the budget between the halves,
//! 3. decodes the PVQ codeword (§4.3.4.2) at the leaves and applies
//!    the spreading rotation (§4.3.4.3),
//! 4. reconstructs stereo bands by mid/side merging (or intensity /
//!    dual-stereo paths as signalled),
//! 5. folds previously-decoded spectrum (or LCG noise) into bands that
//!    received no pulses, tracking per-short-block collapse masks, and
//! 6. applies the §4.3.5 anti-collapse noise injection for transient
//!    frames when the reserved bit is set.
//!
//! The RFC 8251 §9 hybrid-folding update is applied: the first coded
//! band's folding data is duplicated so the second band never falls
//! back to noise merely because the first band is narrower.
//!
//! All bitstream-facing decisions are exact integer arithmetic
//! (`FRAC_MUL16`, `bitexact_cos`, `bitexact_log2tan`, the budget
//! bookkeeping in 1/8-bit units); only the signal values themselves are
//! computed in `f64`.
//!
//! ## Provenance
//!
//! RFC 6716 §4.3.4 / §4.3.5 narrative + the normative Appendix A
//! reference decoder, both from the staged `docs/audio/opus/`
//! documents (`rfc6716-opus.txt`; RFC 8251 §9 from
//! `rfc8251-opus-update.txt`). Tables: `docs/audio/opus/tables/`
//! (`ordery-table.csv`, `bit-interleave-table.csv`,
//! `bit-deinterleave-table.csv`, `exp2-table8.csv`). No external
//! library source was consulted.

use crate::celt_band_layout::CELT_NUM_BANDS;
use crate::celt_pvq_decode::decode_pvq_vector_into;
use crate::celt_pvq_v::pvq_codebook_size;
use crate::celt_rate_alloc::{
    band_edge, band_width, bits2pulses, cache_run, get_pulses, pulses2bits, BITRES, LOG_N_400,
};
use crate::range_decoder::RangeDecoder;

/// Spread values (Table 59 symbols).
pub const SPREAD_NONE: u8 = 0;
/// Aggressive spreading (Table 59 symbol 3).
pub const SPREAD_AGGRESSIVE: u8 = 3;

/// `QTHETA_OFFSET` — split-angle resolution offset.
const QTHETA_OFFSET: i32 = 4;
/// `QTHETA_OFFSET_TWOPHASE` — the stereo `N = 2` variant.
const QTHETA_OFFSET_TWOPHASE: i32 = 16;

/// `exp2` table used by the split-angle resolution computation
/// (Q14 `2^(i/8)` for `i = 0..8`; `docs/audio/opus/tables/exp2-table8.csv`).
const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];

/// Hadamard reordering per stride (2, 4, 8, 16), packed back-to-back;
/// the stride-`s` row starts at offset `s - 2`
/// (`docs/audio/opus/tables/ordery-table.csv`).
const ORDERY_TABLE: [usize; 30] = [
    1, 0, // stride 2
    3, 0, 2, 1, // stride 4
    7, 0, 4, 3, 6, 1, 5, 2, // stride 8
    15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5, // stride 16
];

/// Collapse-mask bit interleaving used when recombining bands
/// (`docs/audio/opus/tables/bit-interleave-table.csv`).
const BIT_INTERLEAVE_TABLE: [u32; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];

/// Inverse of [`BIT_INTERLEAVE_TABLE`] on whole masks
/// (`docs/audio/opus/tables/bit-deinterleave-table.csv`).
const BIT_DEINTERLEAVE_TABLE: [u32; 16] = [
    0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
];

/// The §4.3.4 / §4.3.5 folding-noise linear congruential generator.
#[inline]
#[must_use]
pub fn celt_lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

/// `FRAC_MUL16`: Q15 multiply with rounding, exact 16-bit semantics.
#[inline]
fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + a * b) >> 15
}

/// `EC_ILOG`: position of the most significant set bit, counting from
/// 1 (`EC_ILOG(0) = 0`).
#[inline]
fn ec_ilog(x: u32) -> i32 {
    (32 - x.leading_zeros()) as i32
}

/// Bit-exact cosine approximation over the open interval
/// `x ∈ (0, 16384)` (Q14 angle → Q15 cosine).
#[must_use]
fn bitexact_cos(x: i32) -> i32 {
    let tmp = (4096 + x * x) >> 13;
    debug_assert!(tmp <= 32767);
    let mut x2 = tmp;
    x2 = (32767 - x2) + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2)));
    debug_assert!(x2 <= 32766);
    1 + x2
}

/// Bit-exact `log2(tan)` approximation used for the mid/side bit
/// split.
#[must_use]
fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ec_ilog(icos as u32);
    let ls = ec_ilog(isin as u32);
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// Integer square root (floor).
#[inline]
fn isqrt32(v: u32) -> u32 {
    (v as f64).sqrt() as u32
}

/// The Haar transform used for time/frequency resolution changes
/// (orthonormal butterfly with `1/sqrt(2)` scaling).
fn haar1(x: &mut [f64], n0: usize, stride: usize) {
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let n0 = n0 >> 1;
    for i in 0..stride {
        for j in 0..n0 {
            let a = INV_SQRT2 * x[stride * 2 * j + i];
            let b = INV_SQRT2 * x[stride * (2 * j + 1) + i];
            x[stride * 2 * j + i] = a + b;
            x[stride * (2 * j + 1) + i] = a - b;
        }
    }
}

/// De-interleave the samples of `nb` interleaved short blocks into
/// time order, with the Hadamard reordering on the long-block path.
fn deinterleave_hadamard(x: &mut [f64], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    // Profluens patch: function-local scratch (copied back into `x`), from the per-packet bump.
    let mut tmp = Vec::with_capacity_in(n, crate::bump::DecodeBump);
    tmp.resize(n, 0.0f64);
    debug_assert!(stride > 0);
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[ordery[i] * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

/// Inverse of [`deinterleave_hadamard`].
fn interleave_hadamard(x: &mut [f64], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    // Profluens patch: function-local scratch (copied back into `x`), from the per-packet bump.
    let mut tmp = Vec::with_capacity_in(n, crate::bump::DecodeBump);
    tmp.resize(n, 0.0f64);
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ordery[i] * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

/// Split-angle resolution: the number of quantization levels for
/// `itheta` given the band budget.
fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    // The upper limit keeps enough bits for at least one side pulse in
    // a stereo split with itheta = 16384.
    let mut qb = (b + n2 * offset) / n2;
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let qn = EXP2_TABLE8[(qb & 0x7) as usize] >> (14 - (qb >> BITRES));
        let qn = ((qn + 1) >> 1) << 1;
        debug_assert!(qn <= 256);
        qn
    }
}

/// §4.3.4.3 spreading rotation (and its inverse), matching the
/// normative rotation exactly: a pre-rotation at the derived second
/// stride for wide bands, then the main lattice rotation.
fn exp_rotation1(x: &mut [f64], stride: usize, c: f64, s: f64) {
    let len = x.len();
    if len < 2 * stride {
        return;
    }
    for i in 0..len - stride {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
    let mut i = len as i32 - 2 * stride as i32 - 1;
    while i >= 0 {
        let iu = i as usize;
        let x1 = x[iu];
        let x2 = x[iu + stride];
        x[iu + stride] = c * x2 + s * x1;
        x[iu] = c * x1 - s * x2;
        i -= 1;
    }
}

/// The full rotation: `dir = -1` is the decode direction.
pub(crate) fn exp_rotation(x: &mut [f64], dir: i32, stride: usize, k: i32, spread: u8) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    let len = x.len();
    if 2 * (k as usize) >= len || spread == SPREAD_NONE {
        return;
    }
    let factor = SPREAD_FACTOR[(spread - 1) as usize];
    let gain = len as f64 / (len as f64 + (factor * k) as f64);
    let theta = 0.5 * gain * gain;
    let c = (0.5 * std::f64::consts::PI * theta).cos();
    let s = (0.5 * std::f64::consts::PI * (1.0 - theta)).cos();

    let mut stride2 = 0usize;
    if len >= 8 * stride {
        stride2 = 1;
        // sqrt(len / stride) with rounding, incrementally.
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }
    let sub = len / stride;
    for i in 0..stride {
        let seg = &mut x[i * sub..(i + 1) * sub];
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(seg, stride2, s, c);
            }
            exp_rotation1(seg, 1, c, s);
        } else {
            exp_rotation1(seg, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(seg, stride2, s, -c);
            }
        }
    }
}

/// Per-short-block non-zero mask of the integer PVQ vector.
pub(crate) fn extract_collapse_mask(iy: &[i32], b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = iy.len() / b;
    let mut mask = 0u32;
    for (i, chunk) in iy.chunks_exact(n0).enumerate() {
        if chunk.iter().any(|&v| v != 0) {
            mask |= 1 << i;
        }
    }
    mask
}

/// Normalize `x` to L2 norm `gain` (§4.3.4.2 tail + folding renorm).
fn renormalise_vector(x: &mut [f64], gain: f64) {
    let e: f64 = 1e-15 + x.iter().map(|v| v * v).sum::<f64>();
    let g = gain / e.sqrt();
    for v in x.iter_mut() {
        *v *= g;
    }
}

/// §4.3.4.2 PVQ decode of one leaf band: read the codeword, scale the
/// integer vector to L2 norm `gain`, undo the spreading rotation, and
/// report the collapse mask.
fn alg_unquant(
    rd: &mut RangeDecoder<'_>,
    x: &mut [f64],
    k: i32,
    spread: u8,
    b: usize,
    gain: f64,
) -> u32 {
    debug_assert!(k > 0, "alg_unquant needs at least one pulse");
    debug_assert!(x.len() > 1, "alg_unquant needs at least two dimensions");
    let n = x.len() as u32;
    // Profluens patch: local pulse vector (decoded, scales `x`, feeds the collapse mask), from the
    // per-packet bump arena — no per-band heap allocation.
    let mut iy = Vec::with_capacity_in(x.len(), crate::bump::DecodeBump);
    iy.resize(x.len(), 0i32);
    let ft = pvq_codebook_size(n, k as u32).unwrap_or(1);
    let index = rd.dec_uint(ft).unwrap_or(0);
    let _ = decode_pvq_vector_into(n, k as u32, index, &mut iy);
    let ryy: f64 = iy.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let g = gain / ryy.max(1e-15).sqrt();
    for (dst, &src) in x.iter_mut().zip(iy.iter()) {
        *dst = g * src as f64;
    }
    exp_rotation(x, -1, b, k, spread);
    extract_collapse_mask(&iy, b)
}

/// Stereo mid/side reconstruction (the decode-side merge).
fn stereo_merge(x: &mut [f64], y: &mut [f64], mid: f64) {
    let mut xp = 0.0f64;
    let mut side = 0.0f64;
    for (xv, yv) in x.iter().zip(y.iter()) {
        xp += xv * yv;
        side += yv * yv;
    }
    // Compensate for the mid normalization.
    xp *= mid;
    let el = mid * mid + side - 2.0 * xp;
    let er = mid * mid + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y.copy_from_slice(x);
        return;
    }
    let lgain = 1.0 / el.sqrt();
    let rgain = 1.0 / er.sqrt();
    for (xv, yv) in x.iter_mut().zip(y.iter_mut()) {
        let l = mid * *xv;
        let r = *yv;
        *xv = lgain * (l - r);
        *yv = rgain * (l + r);
    }
}

/// Frame-constant parameters and running state shared by the band
/// recursion.
struct BandCtx<'a, 'b> {
    rd: &'a mut RangeDecoder<'b>,
    remaining_bits: i32,
    intensity: usize,
    spread: u8,
    seed: u32,
}

/// Decoded output of [`quant_all_bands_decode`].
#[derive(Debug, Clone)]
pub struct BandDecodeResult {
    /// Per-channel normalized spectra, planar: channel `c`'s coded
    /// region occupies `x[c * plane .. c * plane + plane]` with
    /// `plane = M * 100` short-scaled bins (band `i` at
    /// `M * band_edge(i)`).
    pub x: Vec<f64, crate::bump::DecodeBump>,
    /// Length of one channel plane (`M * 100`).
    pub plane: usize,
    /// Per-band, per-channel collapse masks
    /// (`collapse_masks[band * channels + c]`).
    pub collapse_masks: Vec<u8, crate::bump::DecodeBump>,
    /// The folding LCG seed after the frame (carried decoder state).
    pub seed: u32,
}

/// §4.3.4 band decode for a whole frame (normative `quant_all_bands`,
/// decode side, with the RFC 8251 §9 folding update).
///
/// * `pulses` — per-band PVQ budgets from the §4.3.3 allocation.
/// * `tf_res` — per-band TF adjustment (Table 60–63 cell values).
/// * `total_bits_q3` — frame bits in 1/8 bits minus the anti-collapse
///   reservation.
/// * `balance` — the allocation's left-over 1/8 bits.
/// * `seed` — the carried folding LCG seed (previous frame's final
///   range state).
#[allow(clippy::too_many_arguments)]
pub fn quant_all_bands_decode(
    rd: &mut RangeDecoder<'_>,
    start: usize,
    end: usize,
    pulses: &[i32; CELT_NUM_BANDS],
    short_blocks: bool,
    spread: u8,
    mut dual_stereo: bool,
    intensity: usize,
    tf_res: &[i32; CELT_NUM_BANDS],
    total_bits_q3: i32,
    mut balance: i32,
    lm: i32,
    coded_bands: usize,
    channels: usize,
    seed: u32,
) -> BandDecodeResult {
    let m = 1usize << lm;
    let b_blocks: usize = if short_blocks { m } else { 1 };
    let plane = m * band_edge(CELT_NUM_BANDS) as usize;
    // Profluens patch: the frame's decoded spectra + collapse masks flow to synthesis and are
    // dropped at the end of the packet's CELT decode (never carried in `self`), so allocate them
    // from the per-packet bump arena. `BandDecodeResult`'s fields are `DecodeBump`-parameterized to
    // match — the type system then forbids storing them in any global-allocator (persistent) field.
    let mut x_out = Vec::with_capacity_in(channels * plane, crate::bump::DecodeBump);
    x_out.resize(channels * plane, 0.0f64);
    let mut collapse_masks = Vec::with_capacity_in(channels * CELT_NUM_BANDS, crate::bump::DecodeBump);
    collapse_masks.resize(channels * CELT_NUM_BANDS, 0u8);
    // Folding buffers: one per channel (norm / norm2). Profluens patch: per-frame folding scratch,
    // used only within this call — from the per-packet bump arena.
    let mut norm = Vec::with_capacity_in(channels * plane, crate::bump::DecodeBump);
    norm.resize(channels * plane, 0.0f64);

    let mut ctx = BandCtx {
        rd,
        remaining_bits: 0,
        intensity,
        spread,
        seed,
    };

    let mut lowband_offset = 0usize;
    let mut update_lowband = true;

    for i in start..end {
        let band_off = m * band_edge(i) as usize;
        let n = m * band_width(i) as usize;
        let tell = ctx.rd.tell_frac() as i32;

        // Compute how many bits to give to this band.
        if i != start {
            balance -= tell;
        }
        let remaining_bits = total_bits_q3 - tell - 1;
        ctx.remaining_bits = remaining_bits;
        let b = if i < coded_bands {
            let curr_balance = balance / (3.min(coded_bands - i) as i32);
            0.max(16383.min((remaining_bits + 1).min(pulses[i] + curr_balance)))
        } else {
            0
        };

        // RFC 8251 §9: also latch the first band after start so the
        // second band can fold.
        if (band_off as i32 - n as i32 >= m as i32 * band_edge(start) || i == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }

        // RFC 8251 §9: duplicate enough of the first band's folding
        // data to be able to fold the second band. (Copies no data in
        // CELT-only mode where the first two bands are equal width.)
        if i == start + 1 {
            let n1 = m * band_width(start) as usize;
            let n2 = m * band_width(start + 1) as usize;
            let off = m * band_edge(start) as usize;
            if n2 > n1 {
                for c in 0..channels {
                    let pl = &mut norm[c * plane..(c + 1) * plane];
                    for j in 0..n2 - n1 {
                        pl[off + n1 + j] = pl[off + 2 * n1 - n2 + j];
                    }
                }
            }
        }

        let tf_change = tf_res[i];
        let mut effective_lowband: Option<usize> = None;
        let (mut x_cm, mut y_cm);
        // Conservative collapse estimate for the folding source.
        if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || b_blocks > 1 || tf_change < 0) {
            let eff = (m as i32 * band_edge(start))
                .max(m as i32 * band_edge(lowband_offset) - n as i32)
                as usize;
            effective_lowband = Some(eff);
            let mut fold_start = lowband_offset;
            while m * band_edge(fold_start - 1) as usize > eff {
                fold_start -= 1;
            }
            fold_start -= 1;
            // RFC 8251 §9 fold_end bound: never run past the current
            // band.
            let mut fold_end = lowband_offset - 1;
            while fold_end + 1 < i && (m * band_edge(fold_end + 1) as usize) < eff + n {
                fold_end += 1;
            }
            fold_end += 1;
            x_cm = 0u32;
            y_cm = 0u32;
            let mut fold_i = fold_start;
            loop {
                x_cm |= collapse_masks[fold_i * channels] as u32;
                y_cm |= collapse_masks[fold_i * channels + channels - 1] as u32;
                fold_i += 1;
                if fold_i >= fold_end {
                    break;
                }
            }
        } else {
            x_cm = (1u32 << b_blocks) - 1;
            y_cm = x_cm;
        }

        if dual_stereo && i == intensity {
            // Switch off dual stereo to do intensity.
            dual_stereo = false;
            let from = m * band_edge(start) as usize;
            for j in from..band_off {
                let a = norm[j];
                let bqq = norm[plane + j];
                norm[j] = 0.5 * (a + bqq);
            }
        }

        if dual_stereo && channels == 2 {
            let (x_plane, y_plane) = x_out.split_at_mut(plane);
            let (norm0, norm1) = norm.split_at_mut(plane);
            let lowband0 = effective_lowband.map(|e| norm0[e..e + n].to_vec_in(crate::bump::DecodeBump));
            let lowband1 = effective_lowband.map(|e| norm1[e..e + n].to_vec_in(crate::bump::DecodeBump));
            x_cm = quant_band(
                &mut ctx,
                i,
                &mut x_plane[band_off..band_off + n],
                None,
                b / 2,
                b_blocks,
                tf_change,
                lowband0.as_deref(),
                lm,
                Some(&mut norm0[band_off..band_off + n]),
                0,
                1.0,
                x_cm,
            );
            y_cm = quant_band(
                &mut ctx,
                i,
                &mut y_plane[band_off..band_off + n],
                None,
                b / 2,
                b_blocks,
                tf_change,
                lowband1.as_deref(),
                lm,
                Some(&mut norm1[band_off..band_off + n]),
                0,
                1.0,
                y_cm,
            );
        } else if channels == 2 {
            let (x_plane, y_plane) = x_out.split_at_mut(plane);
            let lowband = effective_lowband.map(|e| norm[e..e + n].to_vec_in(crate::bump::DecodeBump));
            let cm = quant_band(
                &mut ctx,
                i,
                &mut x_plane[band_off..band_off + n],
                Some(&mut y_plane[band_off..band_off + n]),
                b,
                b_blocks,
                tf_change,
                lowband.as_deref(),
                lm,
                Some(&mut norm[band_off..band_off + n]),
                0,
                1.0,
                x_cm | y_cm,
            );
            x_cm = cm;
            y_cm = cm;
        } else {
            let lowband = effective_lowband.map(|e| norm[e..e + n].to_vec_in(crate::bump::DecodeBump));
            x_cm = quant_band(
                &mut ctx,
                i,
                &mut x_out[band_off..band_off + n],
                None,
                b,
                b_blocks,
                tf_change,
                lowband.as_deref(),
                lm,
                Some(&mut norm[band_off..band_off + n]),
                0,
                1.0,
                x_cm | y_cm,
            );
            y_cm = x_cm;
        }
        collapse_masks[i * channels] = x_cm as u8;
        collapse_masks[i * channels + channels - 1] = y_cm as u8;
        balance += pulses[i] + tell;

        // Update the folding position only while at 1 bit/sample.
        update_lowband = b > (n as i32) << BITRES;
    }

    BandDecodeResult {
        x: x_out,
        plane,
        collapse_masks,
        seed: ctx.seed,
    }
}

/// Decode one band (normative `quant_band`, decode side): recursive
/// mid/side or half-band splitting down to PVQ leaves, with folding on
/// pulse-less leaves and the level-0 inverse time/frequency
/// reorganization. Returns the band's collapse mask.
#[allow(clippy::too_many_arguments)]
fn quant_band(
    ctx: &mut BandCtx<'_, '_>,
    band: usize,
    x: &mut [f64],
    mut y: Option<&mut [f64]>,
    mut b: i32,
    mut blocks: usize,
    mut tf_change: i32,
    lowband: Option<&[f64]>,
    mut lm: i32,
    mut lowband_out: Option<&mut [f64]>,
    level: i32,
    gain: f64,
    mut fill: u32,
) -> u32 {
    let n0 = x.len();
    let mut n = n0;
    let mut n_b = n / blocks;
    // Entry-time block count decides the Hadamard ordering.
    let long_blocks = blocks == 1;
    let stereo = y.is_some();
    let mut b0 = blocks;
    let mut time_divide = 0usize;
    let mut recombine = 0usize;
    let mut inv = false;
    let mut cm: u32;

    // Special case for one sample.
    if n == 1 {
        for x1 in [Some(&mut *x), y.as_deref_mut()].into_iter().flatten() {
            let mut sign = 0u32;
            if ctx.remaining_bits >= 1 << BITRES {
                sign = ctx.rd.dec_bits(1);
                ctx.remaining_bits -= 1 << BITRES;
            }
            x1[0] = if sign == 1 { -1.0 } else { 1.0 };
        }
        if let Some(out) = lowband_out {
            out[0] = x[0];
        }
        return 1;
    }

    // Local copy of the folding source when transforms must run on it. Profluens patch: from the
    // per-packet bump arena (transformed in place per recursion level, never escapes the packet).
    let mut lowband_local: Option<Vec<f64, crate::bump::DecodeBump>> =
        lowband.map(|l| l.to_vec_in(crate::bump::DecodeBump));

    if !stereo && level == 0 {
        if tf_change > 0 {
            recombine = tf_change as usize;
        }

        // Band recombining to increase frequency resolution (the
        // folding source is a local copy, so transforming it in place
        // is safe).
        for k in 0..recombine {
            if let Some(lb) = lowband_local.as_deref_mut() {
                haar1(lb, n >> k, 1 << k);
            }
            fill = BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize]
                | BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] << 2;
        }
        blocks >>= recombine;
        n_b <<= recombine;

        // Increasing the time resolution.
        while (n_b & 1) == 0 && tf_change < 0 {
            if let Some(lb) = lowband_local.as_deref_mut() {
                haar1(lb, n_b, blocks);
            }
            fill |= fill << blocks;
            blocks <<= 1;
            n_b >>= 1;
            time_divide += 1;
            tf_change += 1;
        }
        b0 = blocks;
        let n_b0 = n_b;

        // Reorganize samples in time order.
        if b0 > 1 {
            if let Some(lb) = lowband_local.as_deref_mut() {
                deinterleave_hadamard(lb, n_b >> recombine, b0 << recombine, long_blocks);
            }
        }
        let _ = n_b0;
    }

    // If we need 1.5 more bits than the codebook can produce, split.
    let run = cache_run(band, lm);
    let mut split = stereo;
    let mut mid_split_y: Option<(usize, usize)> = None; // (n_half, marker)
    if !stereo && lm != -1 && !run.is_empty() && b > run[run[0] as usize] as i32 + 12 && n > 2 {
        n >>= 1;
        split = true;
        lm -= 1;
        if blocks == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        blocks = (blocks + 1) >> 1;
        mid_split_y = Some((n, 0));
    }

    if split {
        let mut itheta: i32 = 0;
        let pulse_cap = LOG_N_400[band] + lm * (1 << BITRES);
        let offset = (pulse_cap >> 1)
            - if stereo && n == 2 {
                QTHETA_OFFSET_TWOPHASE
            } else {
                QTHETA_OFFSET
            };
        let mut qn = compute_qn(n as i32, b, offset, pulse_cap, stereo);
        if stereo && band >= ctx.intensity {
            qn = 1;
        }
        let tell = ctx.rd.tell_frac() as i32;
        if qn != 1 {
            // Entropy decoding of the angle: step PDF for stereo,
            // uniform for the time split, triangular otherwise.
            if stereo && n > 2 {
                let p0 = 3i32;
                let x0 = qn / 2;
                let ft = (p0 * (x0 + 1) + x0) as u32;
                let fs = ctx.rd.ec_decode(ft) as i32;
                let xv = if fs < (x0 + 1) * p0 {
                    fs / p0
                } else {
                    x0 + 1 + (fs - (x0 + 1) * p0)
                };
                let (fl, fh) = if xv <= x0 {
                    (p0 * xv, p0 * (xv + 1))
                } else {
                    ((xv - 1 - x0) + (x0 + 1) * p0, (xv - x0) + (x0 + 1) * p0)
                };
                ctx.rd.ec_dec_update(fl as u32, fh as u32, ft);
                itheta = xv;
            } else if b0 > 1 || stereo {
                itheta = ctx.rd.dec_uint(qn as u32 + 1).unwrap_or(0) as i32;
            } else {
                let half = qn >> 1;
                let ft = ((half + 1) * (half + 1)) as u32;
                let fm = ctx.rd.ec_decode(ft) as i32;
                let (fl, fs);
                if fm < (half * (half + 1)) >> 1 {
                    itheta = ((isqrt32((8 * fm + 1) as u32) as i32) - 1) >> 1;
                    fs = itheta + 1;
                    fl = (itheta * (itheta + 1)) >> 1;
                } else {
                    itheta =
                        (2 * (qn + 1) - isqrt32((8 * (ft as i32 - fm - 1) + 1) as u32) as i32) >> 1;
                    fs = qn + 1 - itheta;
                    fl = ft as i32 - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
                }
                ctx.rd.ec_dec_update(fl as u32, (fl + fs) as u32, ft);
            }
            debug_assert!(itheta >= 0);
            itheta = itheta * 16384 / qn;
        } else if stereo {
            // Intensity band: only an inversion flag when affordable.
            inv = if b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
                ctx.rd.dec_bit_logp(2) == 1
            } else {
                false
            };
            itheta = 0;
        }
        let qalloc = ctx.rd.tell_frac() as i32 - tell;
        b -= qalloc;

        let orig_fill = fill;
        let (imid, iside, mut delta);
        if itheta == 0 {
            imid = 32767;
            iside = 0;
            fill &= (1u32 << blocks) - 1;
            delta = -16384;
        } else if itheta == 16384 {
            imid = 0;
            iside = 32767;
            fill &= ((1u32 << blocks) - 1) << blocks;
            delta = 16384;
        } else {
            imid = bitexact_cos(itheta);
            iside = bitexact_cos(16384 - itheta);
            // Mid/side allocation minimizing squared error in the band.
            delta = frac_mul16((n as i32 - 1) << 7, bitexact_log2tan(iside, imid));
        }
        let mid = imid as f64 / 32768.0;
        let side = iside as f64 / 32768.0;

        if n == 2 && stereo {
            // Special stereo N=2 case: one sign bit for the side.
            let y_ref = y.as_deref_mut().expect("stereo split has y");
            let mut sbits = 0i32;
            if itheta != 0 && itheta != 16384 {
                sbits = 1 << BITRES;
            }
            let mbits = b - sbits;
            let c_swap = itheta > 8192;
            ctx.remaining_bits -= qalloc + sbits;

            let mut sign = 0i32;
            if sbits != 0 {
                sign = ctx.rd.dec_bits(1) as i32;
            }
            let sign = 1 - 2 * sign;
            // Decode the dominant channel; reconstruct the other by
            // the orthogonality trick.
            {
                let x2: &mut [f64] = if c_swap { &mut *y_ref } else { &mut *x };
                cm = quant_band(
                    ctx,
                    band,
                    x2,
                    None,
                    mbits,
                    blocks,
                    tf_change,
                    lowband_local.as_deref(),
                    lm,
                    lowband_out.take(),
                    level,
                    gain,
                    orig_fill,
                );
            }
            let (x2_0, x2_1) = if c_swap {
                (y_ref[0], y_ref[1])
            } else {
                (x[0], x[1])
            };
            {
                let y2: &mut [f64] = if c_swap { &mut *x } else { &mut *y_ref };
                y2[0] = -(sign as f64) * x2_1;
                y2[1] = (sign as f64) * x2_0;
            }
            // Resynthesis: scale and rotate mid/side into left/right.
            x[0] *= mid;
            x[1] *= mid;
            y_ref[0] *= side;
            y_ref[1] *= side;
            let tmp = x[0];
            x[0] = tmp - y_ref[0];
            y_ref[0] += tmp;
            let tmp = x[1];
            x[1] = tmp - y_ref[1];
            y_ref[1] += tmp;
        } else {
            // "Normal" split code.
            if b0 > 1 && !stereo && (itheta & 0x3fff) != 0 {
                if itheta > 8192 {
                    // Rough approximation for pre-echo masking.
                    delta -= delta >> (4 - lm);
                } else {
                    // Corresponds to a forward-masking slope of
                    // 1.5 dB per 10 ms.
                    delta = 0.min(delta + ((n as i32) << BITRES >> (5 - lm)));
                }
            }
            let mut mbits = 0.max(b.min((b - delta) / 2));
            let mut sbits = b - mbits;
            ctx.remaining_bits -= qalloc;

            let next_level = if stereo { level } else { level + 1 };
            let side_shift: u32 = if stereo { 0 } else { (b0 as u32) >> 1 };

            if let Some((n_half, _)) = mid_split_y {
                // Mono split: x's halves are mid and side.
                let (x_mid, x_side) = x.split_at_mut(n_half);
                let (lb_mid, lb_side) = match lowband_local.as_deref() {
                    Some(lb) => (Some(&lb[..n_half]), Some(&lb[n_half..])),
                    None => (None, None),
                };
                let rebalance = ctx.remaining_bits;
                if mbits >= sbits {
                    cm = quant_band(
                        ctx,
                        band,
                        x_mid,
                        None,
                        mbits,
                        blocks,
                        tf_change,
                        lb_mid,
                        lm,
                        None,
                        next_level,
                        gain * mid,
                        fill,
                    );
                    let rebalance = mbits - (rebalance - ctx.remaining_bits);
                    if rebalance > 3 << BITRES && itheta != 0 {
                        sbits += rebalance - (3 << BITRES);
                    }
                    cm |= quant_band(
                        ctx,
                        band,
                        x_side,
                        None,
                        sbits,
                        blocks,
                        tf_change,
                        lb_side,
                        lm,
                        None,
                        next_level,
                        gain * side,
                        fill >> blocks,
                    ) << side_shift;
                } else {
                    cm = quant_band(
                        ctx,
                        band,
                        x_side,
                        None,
                        sbits,
                        blocks,
                        tf_change,
                        lb_side,
                        lm,
                        None,
                        next_level,
                        gain * side,
                        fill >> blocks,
                    ) << side_shift;
                    let rebalance = sbits - (rebalance - ctx.remaining_bits);
                    if rebalance > 3 << BITRES && itheta != 16384 {
                        mbits += rebalance - (3 << BITRES);
                    }
                    cm |= quant_band(
                        ctx,
                        band,
                        x_mid,
                        None,
                        mbits,
                        blocks,
                        tf_change,
                        lb_mid,
                        lm,
                        None,
                        next_level,
                        gain * mid,
                        fill,
                    );
                }
            } else {
                // Stereo split: x is mid, y is side.
                let y_ref = y.as_deref_mut().expect("stereo split has y");
                let rebalance = ctx.remaining_bits;
                if mbits >= sbits {
                    cm = quant_band(
                        ctx,
                        band,
                        x,
                        None,
                        mbits,
                        blocks,
                        tf_change,
                        lowband_local.as_deref(),
                        lm,
                        lowband_out.take(),
                        next_level,
                        1.0,
                        fill,
                    );
                    let rebalance = mbits - (rebalance - ctx.remaining_bits);
                    if rebalance > 3 << BITRES && itheta != 0 {
                        sbits += rebalance - (3 << BITRES);
                    }
                    cm |= quant_band(
                        ctx,
                        band,
                        y_ref,
                        None,
                        sbits,
                        blocks,
                        tf_change,
                        None,
                        lm,
                        None,
                        next_level,
                        gain * side,
                        fill >> blocks,
                    ) << side_shift;
                } else {
                    cm = quant_band(
                        ctx,
                        band,
                        y_ref,
                        None,
                        sbits,
                        blocks,
                        tf_change,
                        None,
                        lm,
                        None,
                        next_level,
                        gain * side,
                        fill >> blocks,
                    ) << side_shift;
                    let rebalance = sbits - (rebalance - ctx.remaining_bits);
                    if rebalance > 3 << BITRES && itheta != 16384 {
                        mbits += rebalance - (3 << BITRES);
                    }
                    cm |= quant_band(
                        ctx,
                        band,
                        x,
                        None,
                        mbits,
                        blocks,
                        tf_change,
                        lowband_local.as_deref(),
                        lm,
                        lowband_out.take(),
                        next_level,
                        1.0,
                        fill,
                    );
                }
            }
        }

        // Stereo resynthesis: merge mid/side (except N=2, already
        // rotated) and apply the inversion flag.
        if stereo {
            let y_ref = y.as_mut().expect("stereo has y");
            if n != 2 {
                stereo_merge(x, y_ref, mid);
            }
            if inv {
                for v in y_ref.iter_mut() {
                    *v = -*v;
                }
            }
        }
    } else {
        // The basic no-split case.
        let q = bits2pulses(band, lm, b);
        let mut curr_bits = pulses2bits(band, lm, q);
        let mut q = q;
        ctx.remaining_bits -= curr_bits;
        // Never bust the budget.
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(band, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            let k = get_pulses(q);
            cm = alg_unquant(ctx.rd, x, k, ctx.spread, blocks, gain);
        } else {
            // No pulses: fill the band anyway.
            let cm_mask = ((1u64 << blocks) - 1) as u32;
            fill &= cm_mask;
            if fill == 0 {
                x.fill(0.0);
                cm = 0;
            } else {
                match lowband_local.as_deref() {
                    None => {
                        // Noise.
                        for v in x.iter_mut() {
                            ctx.seed = celt_lcg_rand(ctx.seed);
                            *v = ((ctx.seed as i32) >> 20) as f64;
                        }
                        cm = cm_mask;
                    }
                    Some(lb) => {
                        // Folded spectrum, dithered ~48 dB down.
                        for (v, &l) in x.iter_mut().zip(lb.iter()) {
                            ctx.seed = celt_lcg_rand(ctx.seed);
                            let tmp = if ctx.seed & 0x8000 != 0 {
                                1.0 / 256.0
                            } else {
                                -1.0 / 256.0
                            };
                            *v = l + tmp;
                        }
                        cm = fill;
                    }
                }
                renormalise_vector(x, gain);
            }
        }
    }

    // Level-0 mono inverse reorganization (shared decode/resynthesis
    // tail).
    if !stereo && level == 0 {
        if b0 > 1 {
            interleave_hadamard(x, n_b >> recombine, b0 << recombine, long_blocks);
        }

        // Undo time-freq changes.
        let mut blocks_back = b0;
        let mut n_b_back = n_b;
        for _ in 0..time_divide {
            blocks_back >>= 1;
            n_b_back <<= 1;
            cm |= cm >> blocks_back;
            haar1(x, n_b_back, blocks_back);
        }

        for k in 0..recombine {
            cm = BIT_DEINTERLEAVE_TABLE[(cm & 0xF) as usize];
            haar1(x, n0 >> k, 1 << k);
        }
        let blocks_final = blocks_back << recombine;

        // Scale output for later folding.
        if let Some(out) = lowband_out {
            let g = (n0 as f64).sqrt();
            for (o, &v) in out.iter_mut().zip(x.iter()) {
                *o = g * v;
            }
        }
        cm &= (1u32 << blocks_final) - 1;
    }
    cm
}

/// §4.3.5 anti-collapse: inject renormalized noise into collapsed
/// short-block lines of a transient frame (normative `anti_collapse`).
///
/// * `x` — per-channel planar spectra (as produced by
///   [`quant_all_bands_decode`]).
/// * `log_e` — this frame's per-band log2 energies (sans means),
///   `[channel][band]`.
/// * `prev1` / `prev2` — the previous two frames' energies (the
///   carried `oldLogE` / `oldLogE2`).
/// * `pulses` — the §4.3.3 per-band PVQ budgets; `seed` — the running
///   folding LCG seed. Returns the updated seed.
#[allow(clippy::too_many_arguments)]
pub fn anti_collapse(
    x: &mut [f64],
    plane: usize,
    collapse_masks: &[u8],
    lm: i32,
    channels: usize,
    start: usize,
    end: usize,
    log_e: &[[f64; CELT_NUM_BANDS]; 2],
    prev1: &[[f64; CELT_NUM_BANDS]; 2],
    prev2: &[[f64; CELT_NUM_BANDS]; 2],
    pulses: &[i32; CELT_NUM_BANDS],
    mut seed: u32,
) -> u32 {
    let m = 1usize << lm;
    for i in start..end {
        let n0 = band_width(i) as usize;
        // Depth in 1/8 bits.
        let depth = ((1 + pulses[i]) / (band_width(i) << lm)) as f64;
        let thresh = 0.5 * (-0.125 * depth).exp2();
        let sqrt_1 = 1.0 / ((n0 << lm) as f64).sqrt();

        for c in 0..channels {
            let mut prev1v = prev1[c][i];
            let mut prev2v = prev2[c][i];
            if channels == 1 {
                prev1v = prev1v.max(prev1[1][i]);
                prev2v = prev2v.max(prev2[1][i]);
            }
            let ediff = (log_e[c][i] - prev1v.min(prev2v)).max(0.0);
            // r is doubled (or ×2·√2 at LM=3) because short blocks
            // don't have the same energy as long ones.
            let mut r = 2.0 * (-ediff).exp2();
            if lm == 3 {
                r *= std::f64::consts::SQRT_2;
            }
            r = r.min(thresh) * sqrt_1;
            let band_off = c * plane + m * band_edge(i) as usize;
            let mut renormalize = false;
            for k in 0..(1usize << lm) {
                // Detect collapse.
                if collapse_masks[i * channels + c] & (1 << k) == 0 {
                    // Fill with noise.
                    for j in 0..n0 {
                        seed = celt_lcg_rand(seed);
                        x[band_off + (j << lm) + k] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            // We just added energy: renormalize.
            if renormalize {
                renormalise_vector(&mut x[band_off..band_off + (n0 << lm)], 1.0);
            }
        }
    }
    seed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    #[test]
    fn lcg_matches_the_normative_constants() {
        assert_eq!(celt_lcg_rand(0), 1_013_904_223);
        assert_eq!(celt_lcg_rand(1), 1_664_525 + 1_013_904_223);
    }

    #[test]
    fn bitexact_cos_is_decreasing_and_on_unit_circle() {
        let mut prev = i32::MAX;
        for x in (256..16384).step_by(256) {
            let c = bitexact_cos(x);
            assert!(c < prev, "cos must decrease");
            prev = c;
            let s = bitexact_cos(16384 - x);
            // imid^2 + iside^2 ~ 32768^2 within ~1%.
            let r = (c as f64).hypot(s as f64) / 32768.0;
            assert!((r - 1.0).abs() < 0.01, "x={x}: |.|={r}");
        }
    }

    #[test]
    fn log2tan_is_antisymmetric_and_zero_on_diagonal() {
        assert_eq!(bitexact_log2tan(16384, 16384), 0);
        let a = bitexact_log2tan(20000, 10000);
        let b = bitexact_log2tan(10000, 20000);
        assert_eq!(a, -b);
        // log2(tan) with a 2:1 ratio is ~1.0 in Q11.
        assert!((a - 2048).abs() < 8, "a={a}");
    }

    #[test]
    fn haar1_is_an_involution_up_to_scale() {
        // haar1 applied twice restores the input (it is orthonormal
        // and symmetric).
        let orig: Vec<f64> = (0..16).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut x = orig.clone();
        haar1(&mut x, 16, 1);
        assert!((l2(&x) - l2(&orig)).abs() < 1e-12);
        haar1(&mut x, 16, 1);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn hadamard_interleave_roundtrips() {
        for &(n0, stride, hadamard) in &[
            (4usize, 2usize, true),
            (4, 4, true),
            (3, 8, true),
            (5, 2, false),
        ] {
            let orig: Vec<f64> = (0..n0 * stride).map(|i| i as f64).collect();
            let mut x = orig.clone();
            deinterleave_hadamard(&mut x, n0, stride, hadamard);
            interleave_hadamard(&mut x, n0, stride, hadamard);
            assert_eq!(x, orig, "n0={n0} stride={stride} h={hadamard}");
        }
    }

    #[test]
    fn exp_rotation_roundtrips_and_preserves_norm() {
        for spread in [1u8, 2, 3] {
            let mut x: Vec<f64> = (0..24).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
            renormalise_vector(&mut x, 1.0);
            let orig = x.clone();
            exp_rotation(&mut x, 1, 1, 4, spread);
            assert!((l2(&x) - 1.0).abs() < 1e-9, "forward norm");
            exp_rotation(&mut x, -1, 1, 4, spread);
            for (a, b) in x.iter().zip(orig.iter()) {
                assert!((a - b).abs() < 1e-9, "spread {spread}");
            }
        }
    }

    #[test]
    fn extract_collapse_mask_flags_nonzero_blocks() {
        // 2 blocks of 3: block 0 nonzero, block 1 zero.
        assert_eq!(extract_collapse_mask(&[1, 0, 0, 0, 0, 0], 2), 0b01);
        assert_eq!(extract_collapse_mask(&[0, 0, 0, 0, 2, 0], 2), 0b10);
        assert_eq!(extract_collapse_mask(&[0, 1, 0, 0, 2, 0], 2), 0b11);
        // B <= 1 is always 1.
        assert_eq!(extract_collapse_mask(&[0, 0], 1), 1);
    }

    #[test]
    fn alg_unquant_yields_gain_norm_and_mask() {
        let buf = [0x9Eu8, 0x11, 0x5C, 0x33, 0x7A, 0x42, 0x99, 0x21];
        let mut rd = RangeDecoder::new(&buf);
        let mut x = vec![0.0f64; 8];
        let cm = alg_unquant(&mut rd, &mut x, 3, 2, 1, 0.75);
        assert!((l2(&x) - 0.75).abs() < 1e-9);
        assert_eq!(cm, 1);
    }

    #[test]
    fn quant_all_bands_mono_produces_unit_norm_bands() {
        // A generous budget on a random bitstream: every coded band
        // with pulses must come out with L2 norm 1 (gain = 1 at level
        // 0).
        let buf: Vec<u8> = (0..400u32).map(|i| (i * 89 + 41) as u8).collect();
        let mut rd = RangeDecoder::new(&buf);
        let mut pulses = [0i32; CELT_NUM_BANDS];
        for (i, p) in pulses.iter_mut().enumerate() {
            *p = 64 + 8 * i as i32;
        }
        let tf = [0i32; CELT_NUM_BANDS];
        let res = quant_all_bands_decode(
            &mut rd,
            0,
            21,
            &pulses,
            false,
            2,
            false,
            0,
            &tf,
            400 * 64,
            0,
            2, // LM=2: 10 ms
            21,
            1,
            0,
        );
        assert_eq!(res.plane, 400);
        for i in 0..21 {
            let off = 4 * band_edge(i) as usize;
            let n = 4 * band_width(i) as usize;
            let norm = l2(&res.x[off..off + n]);
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "band {i} norm {norm} (should be unit)"
            );
        }
    }

    #[test]
    fn quant_all_bands_is_deterministic() {
        let buf: Vec<u8> = (0..200u32).map(|i| (i * 57 + 13) as u8).collect();
        let mut pulses = [40i32; CELT_NUM_BANDS];
        pulses[20] = 200;
        let tf = [0i32; CELT_NUM_BANDS];
        let run = |seed: u32| {
            let mut rd = RangeDecoder::new(&buf);
            quant_all_bands_decode(
                &mut rd,
                0,
                21,
                &pulses,
                false,
                2,
                false,
                0,
                &tf,
                200 * 64,
                0,
                3,
                21,
                1,
                seed,
            )
        };
        let a = run(7);
        let b = run(7);
        assert_eq!(a.x, b.x);
        assert_eq!(a.collapse_masks, b.collapse_masks);
        assert_eq!(a.seed, b.seed);
    }

    #[test]
    fn zero_budget_folds_or_noises_with_unit_norm() {
        // No pulses anywhere: every band is noise-filled (no earlier
        // band to fold from) but still unit norm.
        let buf = [0u8; 8];
        let mut rd = RangeDecoder::new(&buf);
        let pulses = [0i32; CELT_NUM_BANDS];
        let tf = [0i32; CELT_NUM_BANDS];
        let res = quant_all_bands_decode(
            &mut rd,
            0,
            21,
            &pulses,
            false,
            2,
            false,
            0,
            &tf,
            8 * 64,
            0,
            0,
            1,
            1,
            42,
        );
        for i in 0..21 {
            let off = band_edge(i) as usize;
            let n = band_width(i) as usize;
            if n > 1 {
                let norm = l2(&res.x[off..off + n]);
                assert!((norm - 1.0).abs() < 1e-9, "band {i} norm {norm}");
            }
        }
        // The seed advanced (noise was drawn).
        assert_ne!(res.seed, 42);
    }

    #[test]
    fn stereo_bands_produce_two_planes() {
        let buf: Vec<u8> = (0..300u32).map(|i| (i * 131 + 7) as u8).collect();
        let mut rd = RangeDecoder::new(&buf);
        let pulses = [80i32; CELT_NUM_BANDS];
        let tf = [0i32; CELT_NUM_BANDS];
        let res = quant_all_bands_decode(
            &mut rd,
            0,
            21,
            &pulses,
            false,
            2,
            false,
            21, // intensity off (past the last band)
            &tf,
            300 * 64,
            0,
            3,
            21,
            2,
            0,
        );
        assert_eq!(res.x.len(), 2 * res.plane);
        assert_eq!(res.collapse_masks.len(), 2 * CELT_NUM_BANDS);
        // Both channels carry signal.
        assert!(l2(&res.x[..res.plane]) > 0.0);
        assert!(l2(&res.x[res.plane..]) > 0.0);
    }

    #[test]
    fn anti_collapse_fills_collapsed_blocks() {
        // One band, LM=1 (2 blocks), collapse mask says block 1
        // collapsed. Energy history close to current → strong r.
        let mut x = vec![0.0f64; 2 * 100];
        // Band 0 (width 1, 2 bins at LM=1): put signal in line 0 only.
        x[0] = 1.0;
        let mut masks = vec![0u8; CELT_NUM_BANDS];
        masks[0] = 0b01; // block 1 collapsed
        let log_e = [[0.0f64; CELT_NUM_BANDS]; 2];
        let prev = [[0.0f64; CELT_NUM_BANDS]; 2];
        let pulses = [8i32; CELT_NUM_BANDS];
        let seed = anti_collapse(
            &mut x, 200, &masks, 1, 1, 0, 1, &log_e, &prev, &prev, &pulses, 99,
        );
        assert_ne!(seed, 99, "noise must be drawn");
        assert!(x[1] != 0.0, "collapsed line must be filled");
        // Renormalized to unit norm.
        assert!((l2(&x[0..2]) - 1.0).abs() < 1e-9);
    }
}
