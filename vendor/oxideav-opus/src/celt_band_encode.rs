//! CELT §4.3.4 band **encode** — the write-side of the recursive band
//! coder (the reference listing's `quant_all_bands` / `quant_band`
//! with `encode = 1`, no resynthesis): mid/side and time splits with
//! the entropy-coded angle derived from the actual band energies,
//! stereo intensity/dual paths, and PVQ leaves through
//! [`crate::celt_pvq_encode::alg_quant`].
//!
//! The budget bookkeeping (1/8-bit units, the running balance, the
//! rebalance passes, the `remaining_bits` gates) is line-for-line the
//! decode module's ([`crate::celt_band_decode`]); only the symbol
//! operations differ, so any stream written here walks the decoder
//! down the identical recursion.
//!
//! Without resynthesis the encoder never maintains folding buffers:
//! the folding decisions are decoder-side signal reconstruction and
//! consume no symbols (the non-resynthesis encoder in the listing
//! skips them the same way).
//!
//! ## Provenance
//!
//! RFC 6716 §4.3.4 / §5.3.4 + the normative Appendix A reference
//! listing (staged `docs/audio/opus/rfc6716-opus.txt`, hash-verified
//! per §A.1). No external library source was consulted.

use crate::celt_band_layout::CELT_NUM_BANDS;
use crate::celt_pvq_encode::alg_quant;
use crate::celt_rate_alloc::{
    band_edge, band_width, bits2pulses, cache_run, get_pulses, pulses2bits, BITRES, LOG_N_400,
};
use crate::range_encoder::RangeEncoder;

const QTHETA_OFFSET: i32 = 4;
const QTHETA_OFFSET_TWOPHASE: i32 = 16;

/// `exp2` table for the split-angle resolution (Q14 `2^(i/8)`).
const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];

/// Hadamard reordering per stride (2, 4, 8, 16).
const ORDERY_TABLE: [usize; 30] = [
    1, 0, //
    3, 0, 2, 1, //
    7, 0, 4, 3, 6, 1, 5, 2, //
    15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

const BIT_INTERLEAVE_TABLE: [u32; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];

#[inline]
fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + a * b) >> 15
}

#[inline]
fn ec_ilog(x: u32) -> i32 {
    (32 - x.leading_zeros()) as i32
}

fn bitexact_cos(x: i32) -> i32 {
    let tmp = (4096 + x * x) >> 13;
    let mut x2 = tmp;
    x2 = (32767 - x2) + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2)));
    1 + x2
}

fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ec_ilog(icos as u32);
    let ls = ec_ilog(isin as u32);
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// Orthonormal Haar butterfly (shared with the decode side).
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

fn deinterleave_hadamard(x: &mut [f64], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f64; n];
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

/// Split-angle resolution (identical to the decode side).
fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let mut qb = (b + n2 * offset) / n2;
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let qn = EXP2_TABLE8[(qb & 0x7) as usize] >> (14 - (qb >> BITRES));
        ((qn + 1) >> 1) << 1
    }
}

/// The §5.3.4 split-angle measurement: `atan2(‖side‖, ‖mid‖)` scaled
/// to the Q14 quarter-circle (the listing's `stereo_itheta`).
fn stereo_itheta(x: &[f64], y: &[f64], stereo: bool, n: usize) -> i32 {
    let mut emid = 1e-15f64;
    let mut eside = 1e-15f64;
    if stereo {
        for i in 0..n {
            let m = 0.5 * (x[i] + y[i]);
            let s = 0.5 * (x[i] - y[i]);
            emid += m * m;
            eside += s * s;
        }
    } else {
        for i in 0..n {
            emid += x[i] * x[i];
            eside += y[i] * y[i];
        }
    }
    let mid = emid.sqrt();
    let side = eside.sqrt();
    (0.5 + 16384.0 * 0.636_619_772_367_581_3 * side.atan2(mid)).floor() as i32
}

/// The listing's `intensity_stereo`: collapse the pair onto the mid
/// with amplitude weights from the per-band energies.
fn intensity_stereo(
    x: &mut [f64],
    y: &[f64],
    band_e: &[[f64; CELT_NUM_BANDS]; 2],
    band: usize,
    n: usize,
) {
    let left = band_e[0][band];
    let right = band_e[1][band];
    let norm = 1e-15 + (1e-15 + left * left + right * right).sqrt();
    let a1 = left / norm;
    let a2 = right / norm;
    for j in 0..n {
        x[j] = a1 * x[j] + a2 * y[j];
    }
}

/// The listing's `stereo_split`: L/R → normalized mid/side.
fn stereo_split(x: &mut [f64], y: &mut [f64], n: usize) {
    const C: f64 = std::f64::consts::FRAC_1_SQRT_2;
    for j in 0..n {
        let l = C * x[j];
        let r = C * y[j];
        x[j] = l + r;
        y[j] = r - l;
    }
}

/// Frame-constant parameters and running state for the band encode
/// recursion.
struct EncBandCtx<'a, 'b> {
    enc: &'a mut RangeEncoder,
    remaining_bits: i32,
    intensity: usize,
    spread: u8,
    band_e: &'b [[f64; CELT_NUM_BANDS]; 2],
}

/// §4.3.4 band encode for a whole frame (normative `quant_all_bands`,
/// encode side, no resynthesis).
///
/// `x` holds the per-channel planar normalized spectra (channel `c` at
/// `x[c * plane ..]`, `plane = M * 100`), consumed as scratch.
#[allow(clippy::too_many_arguments)]
pub fn quant_all_bands_encode(
    enc: &mut RangeEncoder,
    start: usize,
    end: usize,
    x: &mut [f64],
    channels: usize,
    band_e: &[[f64; CELT_NUM_BANDS]; 2],
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
) {
    let m = 1usize << lm;
    let b_blocks: usize = if short_blocks { m } else { 1 };
    let plane = m * band_edge(CELT_NUM_BANDS) as usize;
    debug_assert_eq!(x.len(), channels * plane);

    let mut ctx = EncBandCtx {
        enc,
        remaining_bits: 0,
        intensity,
        spread,
        band_e,
    };

    for i in start..end {
        let band_off = m * band_edge(i) as usize;
        let n = m * band_width(i) as usize;
        let tell = ctx.enc.tell_frac() as i32;

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

        let tf_change = tf_res[i];
        // No resynthesis: the folding source estimate is the all-ones
        // mask (the listing's non-resynth encoder path).
        let fill = ((1u64 << b_blocks) - 1) as u32;

        if dual_stereo && i == intensity {
            dual_stereo = false;
        }

        if dual_stereo && channels == 2 {
            let (x_plane, y_plane) = x.split_at_mut(plane);
            quant_band_encode(
                &mut ctx,
                i,
                &mut x_plane[band_off..band_off + n],
                None,
                b / 2,
                b_blocks,
                tf_change,
                lm,
                0,
                fill,
            );
            quant_band_encode(
                &mut ctx,
                i,
                &mut y_plane[band_off..band_off + n],
                None,
                b / 2,
                b_blocks,
                tf_change,
                lm,
                0,
                fill,
            );
        } else if channels == 2 {
            let (x_plane, y_plane) = x.split_at_mut(plane);
            let (xb, yb) = (
                &mut x_plane[band_off..band_off + n],
                &mut y_plane[band_off..band_off + n],
            );
            quant_band_encode(
                &mut ctx,
                i,
                xb,
                Some(yb),
                b,
                b_blocks,
                tf_change,
                lm,
                0,
                fill,
            );
        } else {
            quant_band_encode(
                &mut ctx,
                i,
                &mut x[band_off..band_off + n],
                None,
                b,
                b_blocks,
                tf_change,
                lm,
                0,
                fill,
            );
        }
        balance += pulses[i] + tell;
    }
}

/// Encode one band (normative `quant_band`, encode side, no
/// resynthesis).
#[allow(clippy::too_many_arguments)]
fn quant_band_encode(
    ctx: &mut EncBandCtx<'_, '_>,
    band: usize,
    x: &mut [f64],
    mut y: Option<&mut [f64]>,
    mut b: i32,
    mut blocks: usize,
    mut tf_change: i32,
    mut lm: i32,
    level: i32,
    mut fill: u32,
) {
    let n0 = x.len();
    let mut n = n0;
    let mut n_b = n / blocks;
    let long_blocks = blocks == 1;
    let stereo = y.is_some();
    let mut b0 = blocks;
    let mut recombine = 0usize;

    // Special case for one sample.
    if n == 1 {
        for x1 in [Some(&mut *x), y.as_deref_mut()].into_iter().flatten() {
            if ctx.remaining_bits >= 1 << BITRES {
                let sign = x1[0] < 0.0;
                ctx.enc.enc_bits(u32::from(sign), 1);
                ctx.remaining_bits -= 1 << BITRES;
                b -= 1 << BITRES;
            }
        }
        let _ = b;
        return;
    }

    if !stereo && level == 0 {
        if tf_change > 0 {
            recombine = tf_change as usize;
        }
        for k in 0..recombine {
            haar1(x, n >> k, 1 << k);
            fill = BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize]
                | BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] << 2;
        }
        blocks >>= recombine;
        n_b <<= recombine;

        while (n_b & 1) == 0 && tf_change < 0 {
            haar1(x, n_b, blocks);
            fill |= fill << blocks;
            blocks <<= 1;
            n_b >>= 1;
            tf_change += 1;
        }
        b0 = blocks;

        if b0 > 1 {
            deinterleave_hadamard(x, n_b >> recombine, b0 << recombine, long_blocks);
        }
    }

    // If we need 1.5 more bits than the codebook can produce, split.
    let run = cache_run(band, lm);
    let mut split = stereo;
    let mut mid_split: Option<usize> = None;
    if !stereo && lm != -1 && !run.is_empty() && b > run[run[0] as usize] as i32 + 12 && n > 2 {
        n >>= 1;
        split = true;
        lm -= 1;
        if blocks == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        blocks = (blocks + 1) >> 1;
        mid_split = Some(n);
    }

    if split {
        let mut itheta: i32;
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
        // Measure theta from the actual (transformed) signal halves.
        itheta = if stereo {
            let y_ref = y.as_deref().expect("stereo has y");
            stereo_itheta(x, y_ref, true, n)
        } else {
            let n_half = mid_split.expect("mono split length");
            let (xm, xs) = x.split_at(n_half);
            stereo_itheta(xm, xs, false, n)
        };
        let tell = ctx.enc.tell_frac() as i32;
        let mut inv = false;
        if qn != 1 {
            itheta = (itheta * qn + 8192) >> 14;

            // Entropy coding of the angle: step PDF for stereo N>2,
            // uniform for the time split, triangular otherwise.
            if stereo && n > 2 {
                let p0 = 3i32;
                let xv = itheta;
                let x0 = qn / 2;
                let ft = (p0 * (x0 + 1) + x0) as u32;
                let (fl, fh) = if xv <= x0 {
                    (p0 * xv, p0 * (xv + 1))
                } else {
                    ((xv - 1 - x0) + (x0 + 1) * p0, (xv - x0) + (x0 + 1) * p0)
                };
                ctx.enc.encode(fl as u32, fh as u32, ft);
            } else if b0 > 1 || stereo {
                ctx.enc.enc_uint(itheta as u32, qn as u32 + 1);
            } else {
                let half = qn >> 1;
                let ft = ((half + 1) * (half + 1)) as u32;
                let (fl, fs) = if itheta <= half {
                    ((itheta * (itheta + 1)) >> 1, itheta + 1)
                } else {
                    (
                        ft as i32 - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1),
                        qn + 1 - itheta,
                    )
                };
                ctx.enc.encode(fl as u32, (fl + fs) as u32, ft);
            }
            itheta = itheta * 16384 / qn;
            if stereo {
                let y_ref = y.as_deref_mut().expect("stereo has y");
                if itheta == 0 {
                    intensity_stereo(x, y_ref, ctx.band_e, band, n);
                } else {
                    stereo_split(x, y_ref, n);
                }
            }
        } else if stereo {
            // Intensity band: inversion flag + intensity collapse.
            let y_ref = y.as_deref_mut().expect("stereo has y");
            inv = itheta > 8192;
            if inv {
                for v in y_ref.iter_mut() {
                    *v = -*v;
                }
            }
            intensity_stereo(x, y_ref, ctx.band_e, band, n);
            if b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
                ctx.enc.enc_bit_logp(inv, 2);
            }
            itheta = 0;
        }
        let qalloc = ctx.enc.tell_frac() as i32 - tell;
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
            delta = frac_mul16((n as i32 - 1) << 7, bitexact_log2tan(iside, imid));
        }
        let _ = (imid, iside);

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

            if sbits != 0 {
                let (x2, y2): (&[f64], &[f64]) = if c_swap {
                    (&*y_ref, &*x)
                } else {
                    (&*x, &*y_ref)
                };
                let sign = x2[0] * y2[1] - x2[1] * y2[0] < 0.0;
                ctx.enc.enc_bits(u32::from(sign), 1);
            }
            {
                let x2: &mut [f64] = if c_swap { y_ref } else { &mut *x };
                quant_band_encode(
                    ctx, band, x2, None, mbits, blocks, tf_change, lm, level, orig_fill,
                );
            }
        } else {
            // "Normal" split code.
            if b0 > 1 && !stereo && (itheta & 0x3fff) != 0 {
                if itheta > 8192 {
                    delta -= delta >> (4 - lm);
                } else {
                    delta = 0.min(delta + ((n as i32) << BITRES >> (5 - lm)));
                }
            }
            let mut mbits = 0.max(b.min((b - delta) / 2));
            let mut sbits = b - mbits;
            ctx.remaining_bits -= qalloc;

            let next_level = if stereo { level } else { level + 1 };

            if let Some(n_half) = mid_split {
                // Mono split: halves are mid and side.
                let (x_mid, x_side) = x.split_at_mut(n_half);
                let rebalance = ctx.remaining_bits;
                if mbits >= sbits {
                    quant_band_encode(
                        ctx, band, x_mid, None, mbits, blocks, tf_change, lm, next_level, fill,
                    );
                    let rebalance = mbits - (rebalance - ctx.remaining_bits);
                    if rebalance > 3 << BITRES && itheta != 0 {
                        sbits += rebalance - (3 << BITRES);
                    }
                    quant_band_encode(
                        ctx,
                        band,
                        x_side,
                        None,
                        sbits,
                        blocks,
                        tf_change,
                        lm,
                        next_level,
                        fill >> blocks,
                    );
                } else {
                    quant_band_encode(
                        ctx,
                        band,
                        x_side,
                        None,
                        sbits,
                        blocks,
                        tf_change,
                        lm,
                        next_level,
                        fill >> blocks,
                    );
                    let rebalance = sbits - (rebalance - ctx.remaining_bits);
                    if rebalance > 3 << BITRES && itheta != 16384 {
                        mbits += rebalance - (3 << BITRES);
                    }
                    quant_band_encode(
                        ctx, band, x_mid, None, mbits, blocks, tf_change, lm, next_level, fill,
                    );
                }
            } else {
                // Stereo split: x is mid, y is side.
                let y_ref = y.expect("stereo split has y");
                let rebalance = ctx.remaining_bits;
                if mbits >= sbits {
                    quant_band_encode(
                        ctx, band, x, None, mbits, blocks, tf_change, lm, next_level, fill,
                    );
                    let rebalance = mbits - (rebalance - ctx.remaining_bits);
                    if rebalance > 3 << BITRES && itheta != 0 {
                        sbits += rebalance - (3 << BITRES);
                    }
                    quant_band_encode(
                        ctx,
                        band,
                        y_ref,
                        None,
                        sbits,
                        blocks,
                        tf_change,
                        lm,
                        next_level,
                        fill >> blocks,
                    );
                } else {
                    quant_band_encode(
                        ctx,
                        band,
                        y_ref,
                        None,
                        sbits,
                        blocks,
                        tf_change,
                        lm,
                        next_level,
                        fill >> blocks,
                    );
                    let rebalance = sbits - (rebalance - ctx.remaining_bits);
                    if rebalance > 3 << BITRES && itheta != 16384 {
                        mbits += rebalance - (3 << BITRES);
                    }
                    quant_band_encode(
                        ctx, band, x, None, mbits, blocks, tf_change, lm, next_level, fill,
                    );
                }
            }
        }
        let _ = inv;
    } else {
        // The basic no-split case.
        let q = bits2pulses(band, lm, b);
        let mut curr_bits = pulses2bits(band, lm, q);
        let mut q = q;
        ctx.remaining_bits -= curr_bits;
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(band, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            let k = get_pulses(q);
            let _ = alg_quant(ctx.enc, x, k, ctx.spread, blocks);
        }
        // q == 0: no pulses, nothing coded (folding is decoder-side).
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_band_decode::quant_all_bands_decode;
    use crate::range_decoder::RangeDecoder;

    fn l2(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Build a deterministic set of unit-norm band shapes.
    fn unit_shapes(plane: usize, channels: usize, m: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        let mut x = vec![0.0f64; channels * plane];
        for c in 0..channels {
            for i in 0..CELT_NUM_BANDS {
                let off = c * plane + m * band_edge(i) as usize;
                let len = m * band_width(i) as usize;
                for j in 0..len {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    x[off + j] = ((s >> 33) as f64 / 2.0_f64.powi(30)) - 1.0;
                }
                let seg = &mut x[off..off + len];
                let norm = l2(seg).max(1e-12);
                for v in seg.iter_mut() {
                    *v /= norm;
                }
            }
        }
        x
    }

    #[test]
    fn band_encode_then_decode_reproduces_the_shapes() {
        // Encode unit-norm shapes at a generous budget, decode with the
        // production §4.3.4 decoder: the recursion must consume the
        // same symbols (identical tell) and reproduce each coded band
        // shape with high correlation.
        for &(channels, lm, frame_bytes) in &[
            (1usize, 0i32, 80i32),
            (1, 2, 160),
            (1, 3, 240),
            (2, 3, 320),
            (2, 1, 120),
        ] {
            let m = 1usize << lm;
            let plane = m * band_edge(CELT_NUM_BANDS) as usize;
            let mut x = unit_shapes(plane, channels, m, 0xABCD + lm as u64);
            let x_orig = x.clone();
            let band_e = [[1.0f64; CELT_NUM_BANDS]; 2];
            let mut pulses = [0i32; CELT_NUM_BANDS];
            // A plausible declining allocation.
            let per = frame_bytes * 64 / 30;
            for (i, p) in pulses.iter_mut().enumerate() {
                *p = (per - (i as i32) * per / 40).max(8);
            }
            let tf = [0i32; CELT_NUM_BANDS];
            let total_q3 = frame_bytes * 8 * 8;
            let intensity = if channels == 2 { 21 } else { 0 };

            let mut enc = RangeEncoder::new();
            quant_all_bands_encode(
                &mut enc, 0, 21, &mut x, channels, &band_e, &pulses, false, 2, false, intensity,
                &tf, total_q3, 0, lm, 21,
            );
            let enc_tell = enc.tell_frac();
            let buf = enc.finish_fixed(frame_bytes as usize).expect("fits");
            let mut rd = RangeDecoder::new(&buf);
            let res = quant_all_bands_decode(
                &mut rd, 0, 21, &pulses, false, 2, false, intensity, &tf, total_q3, 0, lm, 21,
                channels, 0,
            );
            assert_eq!(
                rd.tell_frac(),
                enc_tell,
                "coder desync: channels={channels} lm={lm}"
            );
            // Every band with a real budget must decode to a shape
            // correlated with the input (PVQ quantization keeps the
            // direction; at these budgets the match is strong).
            let mut weak = 0usize;
            for c in 0..channels {
                for i in 0..21 {
                    let off = c * plane + m * band_edge(i) as usize;
                    let len = m * band_width(i) as usize;
                    let a = &x_orig[off..off + len];
                    let d = &res.x[off..off + len];
                    let corr: f64 = a.iter().zip(d.iter()).map(|(p, q)| p * q).sum();
                    if corr <= 0.5 {
                        weak += 1;
                    }
                }
            }
            // Mono at these budgets tracks tightly; independent random
            // L/R shapes lose side precision in the widest bands, so
            // allow a few weak stereo bands.
            assert!(
                weak <= 4 * channels,
                "channels={channels} lm={lm}: {weak} weak bands"
            );
        }
    }

    #[test]
    fn stereo_itheta_measures_the_energy_angle() {
        // Pure mid → 0; pure side → 16384; equal → ~8192.
        let x = vec![1.0f64; 8];
        let y = vec![1.0f64; 8];
        assert!(stereo_itheta(&x, &y, true, 8) < 200);
        let y_neg: Vec<f64> = y.iter().map(|v| -v).collect();
        assert!(stereo_itheta(&x, &y_neg, true, 8) > 16200);
        let zero = vec![0.0f64; 8];
        let th = stereo_itheta(&x, &zero, false, 8);
        assert!(th < 200, "mono all-mid: {th}");
    }
}
