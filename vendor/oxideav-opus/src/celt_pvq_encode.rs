//! CELT §4.3.4.2 PVQ **encode** — codeword index construction (the
//! exact inverse of [`crate::celt_pvq_decode`]) and the pyramid
//! projection + greedy pulse search the reference listing's encoder
//! uses to pick the vector (RFC 6716 §5.3.4).
//!
//! The index construction inverts the §4.3.4.2 five-step recovery
//! walk: at each position the same `V(N, K)` counts partition the
//! index space, so the index of a given pulse vector is the sum of the
//! sub-interval offsets the decoder would have subtracted while
//! recovering it. The search is the listing encoder's two-phase
//! procedure: an L1 ("pyramid") projection places most of the `K`
//! pulses in one pass, then a greedy loop adds the remainder one pulse
//! at a time, maximizing `Rxy² / Ryy` (the correlation against the
//! band shape per unit energy).
//!
//! ## Provenance
//!
//! RFC 6716 §4.3.4.2 / §5.3.4 narrative + the normative Appendix A
//! reference listing (both from the staged
//! `docs/audio/opus/rfc6716-opus.txt`, extracted and hash-verified per
//! §A.1). No external library source was consulted.

use crate::celt_pvq_v::{pvq_codebook_size, PvqVError};
use crate::range_encoder::RangeEncoder;

/// Errors returnable by [`encode_pvq_vector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PvqEncodeError {
    /// A `V(N, K)` evaluation rejected its arguments (dimension or
    /// pulse count out of the §4.1.5 range).
    CodebookSize(PvqVError),
    /// The vector is empty.
    EmptyVector,
}

impl core::fmt::Display for PvqEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            PvqEncodeError::CodebookSize(e) => {
                write!(
                    f,
                    "oxideav-opus: PVQ vector encode codebook-size error: {e}"
                )
            }
            PvqEncodeError::EmptyVector => {
                write!(
                    f,
                    "oxideav-opus: PVQ vector encode needs at least one coordinate"
                )
            }
        }
    }
}

impl std::error::Error for PvqEncodeError {}

impl From<PvqVError> for PvqEncodeError {
    fn from(e: PvqVError) -> Self {
        PvqEncodeError::CodebookSize(e)
    }
}

/// Computes the §4.3.4.2 codeword index of the integer pulse vector
/// `x` (whose L1 norm is the codeword's `K`). Returns `(index, k)`.
///
/// The index is exactly the value [`crate::celt_pvq_decode::decode_pvq_vector`]
/// maps back to `x`: at position `j` with `k` pulses left, the decoder
/// subtracts `p_full = (V(N-j-1, k) + V(N-j, k)) / 2` when the sign is
/// negative, then walks `p` down by `V(N-j-1, k')` for `k' = k, k-1,
/// …, k-m` (magnitude `m`); the encoder adds those same counts.
///
/// # Errors
///
/// * [`PvqEncodeError::EmptyVector`] on an empty slice.
/// * [`PvqEncodeError::CodebookSize`] if `V(N, K)` cannot be evaluated.
pub fn encode_pvq_vector(x: &[i32]) -> Result<(u32, u32), PvqEncodeError> {
    let n = u32::try_from(x.len()).map_err(|_| PvqEncodeError::EmptyVector)?;
    if n == 0 {
        return Err(PvqEncodeError::EmptyVector);
    }
    let k: u32 = x.iter().map(|&v| v.unsigned_abs()).sum();
    // Validate (N, K) once; the intermediate lookups then stay in range.
    let _ = pvq_codebook_size(n, k)?;

    let mut index: u64 = 0;
    let mut k_cur = k;
    for (j, &xj) in x.iter().enumerate() {
        let j = j as u32;
        let v_lower = u64::from(pvq_codebook_size(n - j - 1, k_cur)?);
        let v_upper = u64::from(pvq_codebook_size(n - j, k_cur)?);
        let p_full = (v_lower + v_upper) / 2;
        if xj < 0 {
            index += p_full;
        }
        let m = xj.unsigned_abs();
        // p_final = p_full - Σ_{k' = k_cur-m ..= k_cur} V(N-j-1, k').
        let mut p = p_full - v_lower;
        for step in 1..=m {
            p -= u64::from(pvq_codebook_size(n - j - 1, k_cur - step)?);
        }
        index += p;
        k_cur -= m;
    }
    debug_assert_eq!(k_cur, 0);
    debug_assert!(index < u64::from(pvq_codebook_size(n, k)?));
    Ok((index as u32, k))
}

/// The §5.3.4 PVQ pulse search: distribute `k` pulses over the
/// (already sign-stripped, non-negative) shape `x`, maximizing the
/// correlation per unit energy. Returns the unsigned pulse vector.
///
/// This is the listing encoder's float search: an L1 projection
/// pre-places pulses when `k > n/2`, then a greedy loop places the
/// remainder one at a time on the position maximizing
/// `(xy + x[j])² / (yy + y[j])`.
fn op_pvq_search(x: &mut [f64], k: i32) -> Vec<i32> {
    let n = x.len();
    let mut iy = vec![0i32; n];
    let mut y = vec![0.0f64; n];
    let mut pulses_left = k;
    let mut xy = 0.0f64;
    let mut yy = 0.0f64;

    // Pre-search by projecting on the pyramid.
    if k > (n as i32) >> 1 {
        let mut sum: f64 = x.iter().sum();
        // Prevents infinities and NaNs from causing too many pulses to
        // be allocated; 64 approximates infinity here.
        if !(sum > 1e-15 && sum < 64.0) {
            x[0] = 1.0;
            for v in x.iter_mut().skip(1) {
                *v = 0.0;
            }
            sum = 1.0;
        }
        let rcp = (k as f64 - 1.0) / sum;
        for j in 0..n {
            iy[j] = (rcp * x[j]).floor() as i32;
            y[j] = iy[j] as f64;
            yy += y[j] * y[j];
            xy += x[j] * y[j];
            y[j] *= 2.0;
            pulses_left -= iy[j];
        }
    }
    debug_assert!(
        pulses_left >= 1,
        "allocated too many pulses in the quick pass"
    );

    // On pathological input, dump the remainder into the first bin.
    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as f64;
        yy += tmp * tmp;
        yy += tmp * y[0];
        iy[0] += pulses_left;
        pulses_left = 0;
    }

    for i in 0..pulses_left {
        let mut best_id = 0usize;
        let mut best_num = -1e30f64;
        let mut best_den = 0.0f64;
        let _ = i;
        // The squared-magnitude term is added outside the loop.
        yy += 1.0;
        for j in 0..n {
            let rxy = xy + x[j];
            let ryy = yy + y[j];
            let rxy2 = rxy * rxy;
            if best_den * rxy2 > ryy * best_num {
                best_den = ryy;
                best_num = rxy2;
                best_id = j;
            }
        }
        xy += x[best_id];
        yy += y[best_id];
        y[best_id] += 2.0;
        iy[best_id] += 1;
    }
    iy
}

/// §4.3.4.2 PVQ encode of one leaf band (the listing's `alg_quant`
/// without resynthesis): apply the §4.3.4.3 spreading rotation, strip
/// signs, run the pulse search, restore signs, and write the codeword
/// with `ec_enc_uint(V(N, K))`. Returns the §4.3.4 collapse mask.
///
/// `x` is consumed as scratch (rotated and sign-stripped in place).
pub fn alg_quant(enc: &mut RangeEncoder, x: &mut [f64], k: i32, spread: u8, b: usize) -> u32 {
    debug_assert!(k > 0, "alg_quant needs at least one pulse");
    debug_assert!(x.len() > 1, "alg_quant needs at least two dimensions");
    crate::celt_band_decode::exp_rotation(x, 1, b, k, spread);

    // Strip the signs (a zero coordinate takes the negative sign, as
    // in the listing; it never affects the codeword unless the search
    // places a pulse there, and then either sign is a valid choice).
    let mut signs = vec![false; x.len()];
    for (v, s) in x.iter_mut().zip(signs.iter_mut()) {
        if *v > 0.0 {
            *s = false;
        } else {
            *s = true;
            *v = -*v;
        }
    }

    let mut iy = op_pvq_search(x, k);
    for (v, &s) in iy.iter_mut().zip(signs.iter()) {
        if s {
            *v = -*v;
        }
    }
    let (index, kk) = encode_pvq_vector(&iy).expect("search preserves the L1 budget");
    debug_assert_eq!(kk, k as u32);
    let ft = pvq_codebook_size(x.len() as u32, kk).expect("validated by encode_pvq_vector");
    enc.enc_uint(index, ft);
    crate::celt_band_decode::extract_collapse_mask(&iy, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_pvq_decode::{decode_pvq_vector, pvq_l1_norm};
    use crate::range_decoder::RangeDecoder;

    #[test]
    fn index_encode_inverts_decode_exhaustively_for_small_codebooks() {
        for n in 1..=6u32 {
            for k in 1..=5u32 {
                let size = pvq_codebook_size(n, k).unwrap();
                for index in 0..size {
                    let v = decode_pvq_vector(n, k, index).unwrap();
                    assert_eq!(pvq_l1_norm(&v), u64::from(k));
                    let (back, kk) = encode_pvq_vector(&v).unwrap();
                    assert_eq!((back, kk), (index, k), "n={n} k={k} v={v:?}");
                }
            }
        }
    }

    #[test]
    fn index_encode_inverts_decode_on_large_random_codewords() {
        // Large legal (N, K) pairs (V(N, K) < 2^32, the §4.3.4.4 split
        // rule's guarantee for every leaf): the largest K each N
        // admits, plus the extreme thin shapes.
        let pairs: Vec<(u32, u32)> = [176u32, 100, 48, 24, 8, 2]
            .iter()
            .map(|&n| {
                let mut k = 1u32;
                while pvq_codebook_size(n, k + 1).is_ok() && k < 4096 {
                    k += 1;
                }
                (n, k)
            })
            .collect();
        for &(n, k) in &pairs {
            let size = pvq_codebook_size(n, k).unwrap();
            let mut idx: u64 = 0x1234_5678;
            for _ in 0..40 {
                idx = idx
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let index = (idx >> 32) as u32 % size;
                let v = decode_pvq_vector(n, k, index).unwrap();
                let (back, kk) = encode_pvq_vector(&v).unwrap();
                assert_eq!((back, kk), (index, k), "n={n} k={k}");
            }
        }
    }

    #[test]
    fn pvq_search_meets_the_pulse_budget_and_tracks_the_shape() {
        // A smooth shape: the search must place exactly K pulses and
        // put the most pulses on the largest coordinate.
        let mut x: Vec<f64> = (0..16).map(|i| (1.0 + i as f64).recip()).collect();
        let iy = op_pvq_search(&mut x, 20);
        assert_eq!(iy.iter().map(|&v| v.unsigned_abs()).sum::<u32>(), 20);
        let maxpos = (0..16).max_by_key(|&i| iy[i]).unwrap();
        assert_eq!(maxpos, 0);
    }

    #[test]
    fn alg_quant_roundtrips_through_the_range_coder() {
        // Encode a band with alg_quant, decode the index back with the
        // same V(N, K): the codeword must reproduce the search's iy.
        let mut enc = RangeEncoder::new();
        let mut x: Vec<f64> = (0..24)
            .map(|i| ((i as f64 * 0.7).sin() * 0.4) + if i == 5 { 1.0 } else { 0.0 })
            .collect();
        let mut x2 = x.clone();
        let cm = alg_quant(&mut enc, &mut x, 9, 2, 1);
        assert_eq!(cm, 1);
        let buf = enc.finish();
        let mut rd = RangeDecoder::new(&buf);
        let ft = pvq_codebook_size(24, 9).unwrap();
        let index = rd.dec_uint(ft).unwrap();
        let v = decode_pvq_vector(24, 9, index).unwrap();
        assert_eq!(pvq_l1_norm(&v), 9);
        // The decoded vector correlates with the (rotated) input shape.
        crate::celt_band_decode::exp_rotation(&mut x2, 1, 1, 9, 2);
        let corr: f64 = v.iter().zip(x2.iter()).map(|(&a, &b)| a as f64 * b).sum();
        assert!(corr > 0.0);
    }
}
