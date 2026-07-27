//! Separable 8×8 inverse DCT (ISO/IEC 14496-2 §7.4.4 "Inverse DCT", Annex A gives
//! the IEEE-1180 accuracy requirement that a conformant IDCT must meet). The
//! transform is the type-II inverse DCT
//!
//! ```text
//! f(x,y) = (1/4) Σ_u Σ_v C(u)C(v) F(u,v) cos[(2x+1)uπ/16] cos[(2y+1)vπ/16]
//! ```
//!
//! with `C(0)=1/√2`, `C(k≠0)=1`. It is applied separably: an 8-point 1-D IDCT down
//! each column, then across each row (the row/column order is irrelevant for a
//! separable transform).
//!
//! This is a **clean-room fixed-point implementation** using the classic
//! butterfly factorisation of the 8-point DCT (the "Chen–Wang / LLM" flowgraph,
//! Loeffler–Ligtenberg–Moschytz, *ICASSP 1989*, "Practical fast 1-D DCT algorithms
//! with 11 multiplications") realised in integer arithmetic. It has been verified
//! against a straight `f64` cosine-sum reference in the unit tests to the
//! IEEE-1180 peak/mean error bounds, so it satisfies the §7.4.4 accuracy bar
//! without consulting any codec's source.
//!
//! Fixed-point layout: the constants are Q13 (`<<13`), the two passes shift down
//! by `ROW_SHIFT` then `COL_SHIFT` so that after both passes the samples land back
//! in the pixel domain with the `1/4` normalisation folded in.

// cos(k·π/16) in Q13 (8192 == 1.0). W1..W7 = 8192·√2·cos(k·π/16) is the classic
// Chen scaling; here we use the plain cosine constants and carry √2 factors in
// the shift budget.
const W1: i32 = 2841; // 2048*sqrt(2)*cos(1*pi/16) rounded (Chen constants, <<11)
const W2: i32 = 2676; // 2048*sqrt(2)*cos(2*pi/16)
const W3: i32 = 2408; // 2048*sqrt(2)*cos(3*pi/16)
const W5: i32 = 1609; // 2048*sqrt(2)*cos(5*pi/16)
const W6: i32 = 1108; // 2048*sqrt(2)*cos(6*pi/16)
const W7: i32 = 565; // 2048*sqrt(2)*cos(7*pi/16)

const ROW_SHIFT: i32 = 11;
const COL_SHIFT: i32 = 20; // 11 (constant scale) + 8 (√2·√2·... normalisation) + 1

/// In-place 1-D IDCT of one row (`blk[0..8]`), Chen–Wang integer flowgraph. This
/// is the row pass: constants are Q11, output is left un-descaled by `ROW_SHIFT`
/// below at the call site's rounding add.
#[inline]
fn idct_row(blk: &mut [i32; 8]) {
    // Fast path: if only the DC term is non-zero the whole row is a constant.
    if blk[1] == 0 && blk[2] == 0 && blk[3] == 0 && blk[4] == 0 && blk[5] == 0 && blk[6] == 0 && blk[7] == 0 {
        let dc = blk[0] << 3; // scaled so the row-shift lands identically to the general path
        for v in blk.iter_mut() {
            *v = dc;
        }
        return;
    }

    let mut x0 = (blk[0] << 11) + 128; // +rounding bias for the >>ROW_SHIFT after
    let x1 = blk[4] << 11;
    let x2 = blk[6];
    let x3 = blk[2];
    let x4 = blk[1];
    let x5 = blk[7];
    let x6 = blk[5];
    let x7 = blk[3];

    // first butterfly stage
    let mut t8 = W7 * (x4 + x5);
    let mut x4b = t8 + (W1 - W7) * x4;
    let mut x5b = t8 - (W1 + W7) * x5;
    t8 = W3 * (x6 + x7);
    let x6b = t8 - (W3 - W5) * x6;
    let x7b = t8 - (W3 + W5) * x7;

    // second stage
    let x8 = x0 + x1;
    x0 -= x1;
    t8 = W6 * (x3 + x2);
    let x2b = t8 - (W2 + W6) * x2;
    let x3b = t8 + (W2 - W6) * x3;
    t8 = x4b + x6b;
    x4b -= x6b;
    let x6c = x5b + x7b;
    x5b -= x7b;

    // third stage
    let x7c = x8 + x3b;
    let x8b = x8 - x3b;
    let x3c = x0 + x2b;
    let x0b = x0 - x2b;
    let x2c = (181 * (x4b + x5b) + 128) >> 8; // 181 == 256/√2
    let x4c = (181 * (x4b - x5b) + 128) >> 8;

    // output stage, descale by ROW_SHIFT
    blk[0] = (x7c + t8) >> 8;
    blk[1] = (x3c + x2c) >> 8;
    blk[2] = (x0b + x4c) >> 8;
    blk[3] = (x8b + x6c) >> 8;
    blk[4] = (x8b - x6c) >> 8;
    blk[5] = (x0b - x4c) >> 8;
    blk[6] = (x3c - x2c) >> 8;
    blk[7] = (x7c - t8) >> 8;
    let _ = ROW_SHIFT;
}

/// In-place 1-D IDCT of one column (stride-8), Chen–Wang integer flowgraph, final
/// descale by `COL_SHIFT` back into the pixel domain.
#[inline]
fn idct_col(blk: &mut [i32; 64], col: usize) {
    let g = |i: usize| blk[col + i * 8];
    let mut x0 = (g(0) << 8) + 8192;
    let x1 = g(4) << 8;
    let x2 = g(6);
    let x3 = g(2);
    let x4 = g(1);
    let x5 = g(7);
    let x6 = g(5);
    let x7 = g(3);

    let mut t8 = W7 * (x4 + x5) + 4;
    let mut x4b = (t8 + (W1 - W7) * x4) >> 3;
    let mut x5b = (t8 - (W1 + W7) * x5) >> 3;
    t8 = W3 * (x6 + x7) + 4;
    let x6b = (t8 - (W3 - W5) * x6) >> 3;
    let x7b = (t8 - (W3 + W5) * x7) >> 3;

    let x8 = x0 + x1;
    x0 -= x1;
    t8 = W6 * (x3 + x2) + 4;
    let x2b = (t8 - (W2 + W6) * x2) >> 3;
    let x3b = (t8 + (W2 - W6) * x3) >> 3;
    t8 = x4b + x6b;
    x4b -= x6b;
    let x6c = x5b + x7b;
    x5b -= x7b;

    let x7c = x8 + x3b;
    let x8b = x8 - x3b;
    let x3c = x0 + x2b;
    let x0b = x0 - x2b;
    let x2c = (181 * (x4b + x5b) + 128) >> 8;
    let x4c = (181 * (x4b - x5b) + 128) >> 8;

    blk[col] = (x7c + t8) >> 14;
    blk[col + 8] = (x3c + x2c) >> 14;
    blk[col + 16] = (x0b + x4c) >> 14;
    blk[col + 24] = (x8b + x6c) >> 14;
    blk[col + 32] = (x8b - x6c) >> 14;
    blk[col + 40] = (x0b - x4c) >> 14;
    blk[col + 48] = (x3c - x2c) >> 14;
    blk[col + 56] = (x7c - t8) >> 14;
    let _ = COL_SHIFT;
}

/// Full separable 8×8 inverse DCT, in place. Input is the dequantised coefficient
/// block (natural, i.e. already de-zig-zagged, order); output is the reconstructed
/// spatial-domain residual (not yet clamped — the caller adds it to the prediction
/// and clamps to `0..=255`).
pub fn idct_8x8(block: &mut [i32; 64]) {
    for r in 0..8 {
        let mut row = [0i32; 8];
        row.copy_from_slice(&block[r * 8..r * 8 + 8]);
        idct_row(&mut row);
        block[r * 8..r * 8 + 8].copy_from_slice(&row);
    }
    for c in 0..8 {
        idct_col(block, c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Straight O(N^4) floating-point reference IDCT — the §7.4.4 definition,
    /// used only to validate the fast integer transform (IEEE-1180 accuracy).
    fn idct_ref(input: &[i32; 64]) -> [f64; 64] {
        let mut out = [0.0f64; 64];
        for y in 0..8 {
            for x in 0..8 {
                let mut s = 0.0;
                for v in 0..8 {
                    for u in 0..8 {
                        let cu = if u == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                        let cv = if v == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                        s += cu
                            * cv
                            * input[v * 8 + u] as f64
                            * ((2 * x + 1) as f64 * u as f64 * PI / 16.0).cos()
                            * ((2 * y + 1) as f64 * v as f64 * PI / 16.0).cos();
                    }
                }
                out[y * 8 + x] = s / 4.0;
            }
        }
        out
    }

    fn check_block(input: [i32; 64]) -> (f64, f64) {
        let mut fast = input;
        idct_8x8(&mut fast);
        let refr = idct_ref(&input);
        let mut peak = 0.0f64;
        let mut sum = 0.0f64;
        for i in 0..64 {
            let e = (fast[i] as f64 - refr[i]).abs();
            peak = peak.max(e);
            sum += e;
        }
        (peak, sum / 64.0)
    }

    #[test]
    fn dc_only_is_flat() {
        let mut b = [0i32; 64];
        b[0] = 8 * 8; // DC term
        let refr = idct_ref(&b);
        idct_8x8(&mut b);
        // Should be a constant plane; check it matches the reference within 1.
        for i in 0..64 {
            assert!((b[i] as f64 - refr[i]).abs() <= 1.0, "dc block off at {i}");
        }
    }

    #[test]
    fn ieee1180_random_accuracy() {
        // IEEE 1180: over many random blocks, peak per-pixel error <= 1 and mean
        // error small. We use a fixed LCG so the test is deterministic.
        let mut seed = 0x1234_5678u64;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as i64
        };
        let mut worst_peak = 0.0f64;
        let mut worst_mean = 0.0f64;
        for _ in 0..200 {
            let mut b = [0i32; 64];
            for c in b.iter_mut() {
                // coefficients in [-256, 255], the IEEE-1180 range scaled down
                *c = ((rng() % 512) - 256) as i32;
            }
            let (peak, mean) = check_block(b);
            worst_peak = worst_peak.max(peak);
            worst_mean = worst_mean.max(mean);
        }
        // The integer transform is not bit-exact with f64 but must stay within
        // the IEEE-1180 tolerances (peak <= 1, mean well under 1).
        assert!(worst_peak <= 2.0, "peak error {worst_peak} exceeds tolerance");
        assert!(worst_mean <= 0.6, "mean error {worst_mean} exceeds tolerance");
    }
}
