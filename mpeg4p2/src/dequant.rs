//! Inverse quantisation for MPEG-4 Visual (ISO/IEC 14496-2 §7.4.4). Two methods
//! coexist and a stream selects one in the VOL header via `quant_type`:
//!
//! - **H.263 quantisation** (`quant_type == 0`, §7.4.4 "Method 1"): a flat
//!   nonlinear rule `|F''| = quant·(2·|level|+1)` (minus `quant` when even), no
//!   weighting matrix. Simple and fast.
//! - **MPEG quantisation** (`quant_type == 1`, §7.4.4 "Method 2"): a weighting
//!   matrix (default or custom) scales each coefficient, intra and inter using
//!   different formulas. XviD ASP files commonly use this.
//!
//! Both paths clip the reconstructed coefficients to the §7.4.4 range
//! `[-2048, 2047]` and (for MPEG-quant) apply the "mismatch control" oddification
//! of the coefficient sum. The intra DC coefficient has its own scaler
//! (`dc_scaler`, §7.4.4).

/// The intra DC scaler as a function of the quantiser (§7.4.4, Table 7-a). Luma
/// and chroma differ. `is_luma` selects the component.
pub fn dc_scaler(quant: u32, is_luma: bool) -> i32 {
    let q = quant as i32;
    if is_luma {
        if q < 5 {
            8
        } else if q < 9 {
            2 * q
        } else if q < 25 {
            q + 8
        } else {
            2 * q - 16
        }
    } else if q < 5 {
        8
    } else if q < 25 {
        (q + 13) / 2
    } else {
        q - 6
    }
}

/// Clip a reconstructed coefficient to the §7.4.4 range `[-2048, 2047]`.
#[inline]
fn clip_coeff(v: i32) -> i32 {
    v.clamp(-2048, 2047)
}

/// H.263 (Method 1) inverse quantisation of one block's AC coefficients in place
/// (§7.4.4). `block` is in natural order; `quant` is the macroblock quantiser.
/// The DC of an intra block is handled by the caller with `dc_scaler`; here we
/// dequantise `start..64` (start == 1 for intra to skip DC, 0 for inter).
pub fn dequant_h263(block: &mut [i32; 64], quant: u32, start: usize) {
    let q = quant as i32;
    let add = if q & 1 == 1 { 0 } else { -1 }; // even quant: subtract one (§7.4.4)
    for c in block.iter_mut().skip(start) {
        let level = *c;
        if level == 0 {
            continue;
        }
        let mut v = if level > 0 {
            q * (2 * level + 1) + add
        } else {
            q * (2 * level - 1) - add
        };
        v = clip_coeff(v);
        *c = v;
    }
}

/// MPEG (Method 2) inverse quantisation of one block in place (§7.4.4). Uses the
/// weighting `matrix` (natural order). `intra` selects the intra vs inter
/// formula; `start` skips the intra DC (handled separately). Applies mismatch
/// control (§7.4.4): after dequant, sum all coefficients and, if even, toggle the
/// LSB of the last coefficient.
pub fn dequant_mpeg(
    block: &mut [i32; 64],
    quant: u32,
    matrix: &[u8; 64],
    intra: bool,
    start: usize,
) {
    let q = quant as i32;
    let mut sum: i64 = 0;
    if intra {
        // Intra: F'' = (2·level·W·quant) / 16 ; DC done by caller.
        for i in 0..64 {
            if i < start {
                sum += block[i] as i64;
                continue;
            }
            let level = block[i];
            if level == 0 {
                continue;
            }
            let w = matrix[i] as i32;
            let mut v = (2 * level * w * q) / 16;
            v = clip_coeff(v);
            block[i] = v;
            sum += v as i64;
        }
    } else {
        // Inter: F'' = ((2·level + sign(level))·W·quant) / 16.
        for i in 0..64 {
            let level = block[i];
            if level == 0 {
                continue;
            }
            let w = matrix[i] as i32;
            let s = if level > 0 { 1 } else { -1 };
            let mut v = ((2 * level + s) * w * q) / 16;
            v = clip_coeff(v);
            block[i] = v;
            sum += v as i64;
        }
    }
    // Mismatch control (§7.4.4): if the sum of all reconstructed coefficients is
    // even, toggle the LSB of coefficient 63.
    if sum & 1 == 0 {
        block[63] ^= 1;
        block[63] = clip_coeff(block[63]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_scaler_luma_ranges() {
        assert_eq!(dc_scaler(1, true), 8);
        assert_eq!(dc_scaler(4, true), 8);
        assert_eq!(dc_scaler(5, true), 10);
        assert_eq!(dc_scaler(8, true), 16);
        assert_eq!(dc_scaler(9, true), 17);
        assert_eq!(dc_scaler(24, true), 32);
        assert_eq!(dc_scaler(25, true), 34);
    }

    #[test]
    fn h263_odd_quant_no_offset() {
        let mut b = [0i32; 64];
        b[1] = 3; // level 3
        dequant_h263(&mut b, 5, 1); // quant 5 (odd) → 5*(2*3+1)=35
        assert_eq!(b[1], 35);
    }

    #[test]
    fn h263_even_quant_offsets() {
        let mut b = [0i32; 64];
        b[1] = 3;
        dequant_h263(&mut b, 4, 1); // even: 4*(2*3+1) - 1 = 27
        assert_eq!(b[1], 27);
    }

    #[test]
    fn h263_zero_stays_zero() {
        let mut b = [0i32; 64];
        dequant_h263(&mut b, 6, 0);
        assert!(b.iter().all(|&c| c == 0));
    }
}
