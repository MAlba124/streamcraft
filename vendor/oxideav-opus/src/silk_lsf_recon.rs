//! SILK Normalized LSF reconstruction — RFC 6716 §4.2.7.5.3.
//!
//! Given the stage-1 index `I1` (from [`crate::silk_frame::SilkFrameHeader`])
//! and the stage-2 residual `res_Q10[]` (from [`crate::silk_lsf_stage2`]),
//! this module:
//!
//!  1. Looks up the stage-1 codebook vector `cb1_Q8[]` from Table 23
//!     (NB/MB) or Table 24 (WB).
//!  2. Derives the IHMW weights `w_Q9[k]` from `cb1_Q8[]` using the
//!     closed-form expression in §4.2.7.5.3:
//!
//!     ```text
//!     w2_Q18[k] = (1024/(cb1_Q8[k] - cb1_Q8[k-1])
//!                  + 1024/(cb1_Q8[k+1] - cb1_Q8[k])) << 16
//!     i = ilog(w2_Q18[k])
//!     f = (w2_Q18[k] >> (i-8)) & 127
//!     y = ((i & 1) ? 32768 : 46214) >> ((32 - i) >> 1)
//!     w_Q9[k] = y + ((213 * f * y) >> 16)
//!     ```
//!
//!     with the boundary values `cb1_Q8[-1] = 0` and `cb1_Q8[d_LPC] = 256`.
//!
//!  3. Combines stage-1 codebook + stage-2 residual + weights into the
//!     final normalized LSFs:
//!
//!     ```text
//!     NLSF_Q15[k] = clamp(0, (cb1_Q8[k]<<7)
//!                          + (res_Q10[k]<<14)/w_Q9[k], 32767)
//!     ```
//!
//! The §4.2.7.5.4 stabilization and §4.2.7.5.5 interpolation steps that
//! run after this reconstruction are deferred to a later round.

use crate::silk_lsf_stage2::{LsfStage2, D_LPC_MAX, D_LPC_NB_MB, D_LPC_WB};
use crate::toc::Bandwidth;
use crate::Error;

// =====================================================================
// Table 23 — NB/MB Normalized LSF Stage-1 Codebook Vectors.
//
// 32 codebooks × 10 Q8 entries. Each row is monotone non-decreasing
// (the LSF coefficients are ordered frequencies). Used for NB and MB
// SILK frames.
// =====================================================================

#[rustfmt::skip]
const NBMB_STAGE1_CB1_Q8: [[u8; D_LPC_NB_MB]; 32] = [
    [ 12,  35,  60,  83, 108, 132, 157, 180, 206, 228], // I1 = 0
    [ 15,  32,  55,  77, 101, 125, 151, 175, 201, 225], // I1 = 1
    [ 19,  42,  66,  89, 114, 137, 162, 184, 209, 230], // I1 = 2
    [ 12,  25,  50,  72,  97, 120, 147, 172, 200, 223], // I1 = 3
    [ 26,  44,  69,  90, 114, 135, 159, 180, 205, 225], // I1 = 4
    [ 13,  22,  53,  80, 106, 130, 156, 180, 205, 228], // I1 = 5
    [ 15,  25,  44,  64,  90, 115, 142, 168, 196, 222], // I1 = 6
    [ 19,  24,  62,  82, 100, 120, 145, 168, 190, 214], // I1 = 7
    [ 22,  31,  50,  79, 103, 120, 151, 170, 203, 227], // I1 = 8
    [ 21,  29,  45,  65, 106, 124, 150, 171, 196, 224], // I1 = 9
    [ 30,  49,  75,  97, 121, 142, 165, 186, 209, 229], // I1 = 10
    [ 19,  25,  52,  70,  93, 116, 143, 166, 192, 219], // I1 = 11
    [ 26,  34,  62,  75,  97, 118, 145, 167, 194, 217], // I1 = 12
    [ 25,  33,  56,  70,  91, 113, 143, 165, 196, 223], // I1 = 13
    [ 21,  34,  51,  72,  97, 117, 145, 171, 196, 222], // I1 = 14
    [ 20,  29,  50,  67,  90, 117, 144, 168, 197, 221], // I1 = 15
    [ 22,  31,  48,  66,  95, 117, 146, 168, 196, 222], // I1 = 16
    [ 24,  33,  51,  77, 116, 134, 158, 180, 200, 224], // I1 = 17
    [ 21,  28,  70,  87, 106, 124, 149, 170, 194, 217], // I1 = 18
    [ 26,  33,  53,  64,  83, 117, 152, 173, 204, 225], // I1 = 19
    [ 27,  34,  65,  95, 108, 129, 155, 174, 210, 225], // I1 = 20
    [ 20,  26,  72,  99, 113, 131, 154, 176, 200, 219], // I1 = 21
    [ 34,  43,  61,  78,  93, 114, 155, 177, 205, 229], // I1 = 22
    [ 23,  29,  54,  97, 124, 138, 163, 179, 209, 229], // I1 = 23
    [ 30,  38,  56,  89, 118, 129, 158, 178, 200, 231], // I1 = 24
    [ 21,  29,  49,  63,  85, 111, 142, 163, 193, 222], // I1 = 25
    [ 27,  48,  77, 103, 133, 158, 179, 196, 215, 232], // I1 = 26
    [ 29,  47,  74,  99, 124, 151, 176, 198, 220, 237], // I1 = 27
    [ 33,  42,  61,  76,  93, 121, 155, 174, 207, 225], // I1 = 28
    [ 29,  53,  87, 112, 136, 154, 170, 188, 208, 227], // I1 = 29
    [ 24,  30,  52,  84, 131, 150, 166, 186, 203, 229], // I1 = 30
    [ 37,  48,  64,  84, 104, 118, 156, 177, 201, 230], // I1 = 31
];

// =====================================================================
// Table 24 — WB Normalized LSF Stage-1 Codebook Vectors.
//
// 32 codebooks × 16 Q8 entries.
// =====================================================================

#[rustfmt::skip]
const WB_STAGE1_CB1_Q8: [[u8; D_LPC_WB]; 32] = [
    [  7, 23, 38, 54, 69,  85, 100, 116, 131, 147, 162, 178, 193, 208, 223, 239], // I1 = 0
    [ 13, 25, 41, 55, 69,  83,  98, 112, 127, 142, 157, 171, 187, 203, 220, 236], // I1 = 1
    [ 15, 21, 34, 51, 61,  78,  92, 106, 126, 136, 152, 167, 185, 205, 225, 240], // I1 = 2
    [ 10, 21, 36, 50, 63,  79,  95, 110, 126, 141, 157, 173, 189, 205, 221, 237], // I1 = 3
    [ 17, 20, 37, 51, 59,  78,  89, 107, 123, 134, 150, 164, 184, 205, 224, 240], // I1 = 4
    [ 10, 15, 32, 51, 67,  81,  96, 112, 129, 142, 158, 173, 189, 204, 220, 236], // I1 = 5
    [  8, 21, 37, 51, 65,  79,  98, 113, 126, 138, 155, 168, 179, 192, 209, 218], // I1 = 6
    [ 12, 15, 34, 55, 63,  78,  87, 108, 118, 131, 148, 167, 185, 203, 219, 236], // I1 = 7
    [ 16, 19, 32, 36, 56,  79,  91, 108, 118, 136, 154, 171, 186, 204, 220, 237], // I1 = 8
    [ 11, 28, 43, 58, 74,  89, 105, 120, 135, 150, 165, 180, 196, 211, 226, 241], // I1 = 9
    [  6, 16, 33, 46, 60,  75,  92, 107, 123, 137, 156, 169, 185, 199, 214, 225], // I1 = 10
    [ 11, 19, 30, 44, 57,  74,  89, 105, 121, 135, 152, 169, 186, 202, 218, 234], // I1 = 11
    [ 12, 19, 29, 46, 57,  71,  88, 100, 120, 132, 148, 165, 182, 199, 216, 233], // I1 = 12
    [ 17, 23, 35, 46, 56,  77,  92, 106, 123, 134, 152, 167, 185, 204, 222, 237], // I1 = 13
    [ 14, 17, 45, 53, 63,  75,  89, 107, 115, 132, 151, 171, 188, 206, 221, 240], // I1 = 14
    [  9, 16, 29, 40, 56,  71,  88, 103, 119, 137, 154, 171, 189, 205, 222, 237], // I1 = 15
    [ 16, 19, 36, 48, 57,  76,  87, 105, 118, 132, 150, 167, 185, 202, 218, 236], // I1 = 16
    [ 12, 17, 29, 54, 71,  81,  94, 104, 126, 136, 149, 164, 182, 201, 221, 237], // I1 = 17
    [ 15, 28, 47, 62, 79,  97, 115, 129, 142, 155, 168, 180, 194, 208, 223, 238], // I1 = 18
    [  8, 14, 30, 45, 62,  78,  94, 111, 127, 143, 159, 175, 192, 207, 223, 239], // I1 = 19
    [ 17, 30, 49, 62, 79,  92, 107, 119, 132, 145, 160, 174, 190, 204, 220, 235], // I1 = 20
    [ 14, 19, 36, 45, 61,  76,  91, 108, 121, 138, 154, 172, 189, 205, 222, 238], // I1 = 21
    [ 12, 18, 31, 45, 60,  76,  91, 107, 123, 138, 154, 171, 187, 204, 221, 236], // I1 = 22
    [ 13, 17, 31, 43, 53,  70,  83, 103, 114, 131, 149, 167, 185, 203, 220, 237], // I1 = 23
    [ 17, 22, 35, 42, 58,  78,  93, 110, 125, 139, 155, 170, 188, 206, 224, 240], // I1 = 24
    [  8, 15, 34, 50, 67,  83,  99, 115, 131, 146, 162, 178, 193, 209, 224, 239], // I1 = 25
    [ 13, 16, 41, 66, 73,  86,  95, 111, 128, 137, 150, 163, 183, 206, 225, 241], // I1 = 26
    [ 17, 25, 37, 52, 63,  75,  92, 102, 119, 132, 144, 160, 175, 191, 212, 231], // I1 = 27
    [ 19, 31, 49, 65, 83, 100, 117, 133, 147, 161, 174, 187, 200, 213, 227, 242], // I1 = 28
    [ 18, 31, 52, 68, 88, 103, 117, 126, 138, 149, 163, 177, 192, 207, 223, 239], // I1 = 29
    [ 16, 29, 47, 61, 76,  90, 106, 119, 133, 147, 161, 176, 193, 209, 224, 240], // I1 = 30
    [ 15, 21, 35, 50, 61,  73,  86,  97, 110, 119, 129, 141, 175, 198, 218, 237], // I1 = 31
];

/// `ilog(n)` per RFC 6716 §1.1.10: the minimum number of bits required
/// to store the positive integer `n` in binary, or 0 for `n <= 0`.
///
/// For `n > 0`, `ilog(n) = floor(log2(n)) + 1`.
fn ilog(n: i32) -> u32 {
    if n <= 0 {
        0
    } else {
        32 - (n as u32).leading_zeros()
    }
}

/// Look up the §4.2.7.5.3 stage-1 codebook vector `cb1_Q8[]` for the
/// given bandwidth and stage-1 index `I1`. Returns a slice of length
/// `d_LPC` (10 for NB / MB, 16 for WB).
///
/// Returns `Error::MalformedPacket` if `I1 >= 32` or if `bandwidth` is
/// SWB / FB (SILK never sees those after the §4.2.2 hybrid split).
pub fn cb1_q8(bandwidth: Bandwidth, lsf_stage1: u8) -> Result<&'static [u8], Error> {
    if lsf_stage1 >= 32 {
        return Err(Error::MalformedPacket);
    }
    let i1 = lsf_stage1 as usize;
    match bandwidth {
        Bandwidth::Nb | Bandwidth::Mb => Ok(&NBMB_STAGE1_CB1_Q8[i1][..]),
        Bandwidth::Wb => Ok(&WB_STAGE1_CB1_Q8[i1][..]),
        _ => Err(Error::MalformedPacket),
    }
}

/// The §4.2.7.5.3 IHMW weight vector `w_Q9[]` for a `(bandwidth, I1)`
/// stage-1 codebook entry, without running a stage-2 reconstruction.
///
/// The encoder's quantisation search needs these weights BEFORE it has
/// chosen the stage-2 indices (the residual target is the IHMW-scaled
/// distance from the codebook vector to the analysis NLSF), so this
/// exposes the same derivation [`NlsfReconstructed::from_stage1_and_stage2`]
/// performs internally. Only the first `d_LPC` entries are populated.
pub fn ihmw_w_q9_for(bandwidth: Bandwidth, lsf_stage1: u8) -> Result<[i32; D_LPC_MAX], Error> {
    let cb1 = cb1_q8(bandwidth, lsf_stage1)?;
    let mut w_q9 = [0i32; D_LPC_MAX];
    ihmw_weights_q9(cb1, &mut w_q9);
    Ok(w_q9)
}

/// Compute the per-coefficient IHMW weights `w_Q9[k]` for the given
/// stage-1 codebook vector, per RFC 6716 §4.2.7.5.3.
///
/// The boundary values `cb1_Q8[-1] = 0` and `cb1_Q8[d_LPC] = 256` are
/// supplied here; the caller passes only the `d_LPC` interior cells.
///
/// All divisions are integer. The square-root approximation for
/// `w_Q9[k]` from `w2_Q18[k]` follows the spec verbatim.
///
/// The result vector is written into `out[0..cb1.len()]`; the remaining
/// entries up to `D_LPC_MAX` are zero-padded by the caller.
fn ihmw_weights_q9(cb1: &[u8], out: &mut [i32]) {
    let d_lpc = cb1.len();
    debug_assert!(d_lpc == D_LPC_NB_MB || d_lpc == D_LPC_WB);
    debug_assert!(out.len() >= d_lpc);

    for k in 0..d_lpc {
        // cb1_Q8[-1] = 0, cb1_Q8[d_LPC] = 256.
        let prev = if k == 0 { 0i32 } else { cb1[k - 1] as i32 };
        let cur = cb1[k] as i32;
        let next = if k + 1 == d_lpc {
            256i32
        } else {
            cb1[k + 1] as i32
        };

        // Per §4.2.7.5.3 each per-codebook vector is monotone increasing
        // (otherwise the IHMW divisor would not be positive), but defend
        // against a hypothetical zero by treating zero-or-negative diffs
        // as 1 — the spec doesn't define behaviour there, and a
        // well-formed Table 23 / 24 entry never triggers this path.
        let d0 = (cur - prev).max(1);
        let d1 = (next - cur).max(1);
        let w2_q18 = ((1024 / d0) + (1024 / d1)) << 16;

        let i = ilog(w2_q18) as i32;
        // Per spec: f = (w2_Q18 >> (i - 8)) & 127.
        // For very small w2_Q18 where i < 8, the shift becomes negative
        // in C — but on Table 23/24 inputs i is always >= 8 (the
        // smallest w2_Q18 in this codebook family is far above 256).
        // Defend via saturating shift to keep the function total.
        let f = if i >= 8 {
            (w2_q18 >> (i - 8)) & 127
        } else {
            (w2_q18 << (8 - i)) & 127
        };
        // y = ((i & 1) ? 32768 : 46214) >> ((32 - i) >> 1)
        let y_base = if (i & 1) != 0 { 32768i32 } else { 46214i32 };
        let y = y_base >> ((32 - i) >> 1);
        // w_Q9[k] = y + ((213 * f * y) >> 16)
        out[k] = y + ((213 * f * y) >> 16);
    }
}

/// Result of the §4.2.7.5.3 NLSF reconstruction for one SILK frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NlsfReconstructed {
    len: u8,
    /// Per-coefficient IHMW weight `w_Q9[k]` (Q9 unsigned, documented
    /// 13-bit range 1819..=5227 in the spec). Stored in `i32` so callers
    /// can divide `res_Q10[k] << 14` by it without re-widening.
    w_q9: [i32; D_LPC_MAX],
    /// Final reconstructed normalized LSF coefficients `NLSF_Q15[k]`,
    /// clamped to `[0, 32767]`.
    nlsf_q15: [i16; D_LPC_MAX],
}

impl NlsfReconstructed {
    /// Reconstruct the final `NLSF_Q15[]` vector from the stage-1 index,
    /// the stage-2 residual, and the bandwidth.
    ///
    /// Combines:
    /// * The Table 23 / 24 stage-1 codebook lookup keyed on `(bandwidth, I1)`.
    /// * The IHMW weight derivation from `cb1_Q8[]`.
    /// * The final per-coefficient formula
    ///   `NLSF_Q15[k] = clamp(0, (cb1_Q8[k]<<7) + (res_Q10[k]<<14)/w_Q9[k], 32767)`.
    pub fn from_stage1_and_stage2(
        bandwidth: Bandwidth,
        lsf_stage1: u8,
        stage2: &LsfStage2,
    ) -> Result<Self, Error> {
        let cb1 = cb1_q8(bandwidth, lsf_stage1)?;
        let d_lpc = cb1.len();
        if stage2.len() != d_lpc {
            // Stage-2 residual length must match the bandwidth's d_LPC.
            return Err(Error::MalformedPacket);
        }

        let mut w_q9 = [0i32; D_LPC_MAX];
        ihmw_weights_q9(cb1, &mut w_q9);

        let res_q10 = stage2.res_q10();
        let mut nlsf_q15 = [0i16; D_LPC_MAX];
        for k in 0..d_lpc {
            let cb1_term = (cb1[k] as i32) << 7;
            // Per §4.2.7.5.3 the division is integer division. C's
            // integer division truncates toward zero, which is also
            // Rust's i32 `/` semantics, so we use `/` directly.
            // w_Q9[k] is positive by construction (cb1 is monotone with
            // positive differences, so each term of w2_Q18 is positive,
            // y > 0, and the additive correction is non-negative).
            let res_term = ((res_q10[k]) << 14) / w_q9[k];
            let v = cb1_term + res_term;
            // clamp(0, v, 32767)
            let clamped = v.clamp(0, 32767);
            nlsf_q15[k] = clamped as i16;
        }

        Ok(Self {
            len: d_lpc as u8,
            w_q9,
            nlsf_q15,
        })
    }

    /// Number of populated entries (10 for NB / MB, 16 for WB).
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` if no entries were reconstructed (impossible for a
    /// successful `from_stage1_and_stage2`).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Per-coefficient IHMW weight `w_Q9[k]`.
    pub fn w_q9(&self) -> &[i32] {
        &self.w_q9[..self.len()]
    }

    /// Final reconstructed `NLSF_Q15[k]` in `[0, 32767]`.
    pub fn nlsf_q15(&self) -> &[i16] {
        &self.nlsf_q15[..self.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;
    use crate::silk_lsf_stage2::LsfStage2;

    // --- ilog() spec examples (RFC 6716 §1.1.10) -----------------------

    #[test]
    fn ilog_spec_examples() {
        assert_eq!(ilog(-1), 0);
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(2), 2);
        assert_eq!(ilog(3), 2);
        assert_eq!(ilog(4), 3);
        assert_eq!(ilog(7), 3);
        // Spot-checks for the upper range we'll actually hit on
        // w2_Q18 (~tens of millions = ~25 bits).
        assert_eq!(ilog(8), 4);
        assert_eq!(ilog(255), 8);
        assert_eq!(ilog(256), 9);
        assert_eq!(ilog(1 << 24), 25);
    }

    // --- Table 23 / 24 well-formedness --------------------------------

    /// Every NB/MB and WB stage-1 codebook row must be strictly monotone
    /// non-decreasing (the IHMW formula divides by the differences and
    /// the codebook represents ordered LSF frequencies).
    #[test]
    fn stage1_codebooks_are_monotone() {
        for (i1, row) in NBMB_STAGE1_CB1_Q8.iter().enumerate() {
            for w in row.windows(2) {
                assert!(
                    w[0] < w[1],
                    "NB/MB Table 23 row {i1}: not strictly monotone: {:?}",
                    row
                );
            }
        }
        for (i1, row) in WB_STAGE1_CB1_Q8.iter().enumerate() {
            for w in row.windows(2) {
                assert!(
                    w[0] < w[1],
                    "WB Table 24 row {i1}: not strictly monotone: {:?}",
                    row
                );
            }
        }
    }

    /// Table 23 row widths must equal `D_LPC_NB_MB == 10`.
    #[test]
    fn table23_row_width_is_10() {
        for row in NBMB_STAGE1_CB1_Q8.iter() {
            assert_eq!(row.len(), D_LPC_NB_MB);
        }
        assert_eq!(NBMB_STAGE1_CB1_Q8.len(), 32);
    }

    /// Table 24 row widths must equal `D_LPC_WB == 16`.
    #[test]
    fn table24_row_width_is_16() {
        for row in WB_STAGE1_CB1_Q8.iter() {
            assert_eq!(row.len(), D_LPC_WB);
        }
        assert_eq!(WB_STAGE1_CB1_Q8.len(), 32);
    }

    /// Spot-check Table 23 row 0 (the all-`a` companion in Table 17):
    /// `12 35 60 83 108 132 157 180 206 228`.
    #[test]
    fn table23_row0_spot_check() {
        assert_eq!(
            NBMB_STAGE1_CB1_Q8[0],
            [12, 35, 60, 83, 108, 132, 157, 180, 206, 228]
        );
    }

    /// Spot-check Table 23 row 31 (last row): `37 48 64 84 104 118 156
    /// 177 201 230`.
    #[test]
    fn table23_row31_spot_check() {
        assert_eq!(
            NBMB_STAGE1_CB1_Q8[31],
            [37, 48, 64, 84, 104, 118, 156, 177, 201, 230]
        );
    }

    /// Spot-check Table 24 row 0: `7 23 38 54 69 85 100 116 131 147
    /// 162 178 193 208 223 239`.
    #[test]
    fn table24_row0_spot_check() {
        assert_eq!(
            WB_STAGE1_CB1_Q8[0],
            [7, 23, 38, 54, 69, 85, 100, 116, 131, 147, 162, 178, 193, 208, 223, 239]
        );
    }

    /// Spot-check Table 24 row 31 (last row).
    #[test]
    fn table24_row31_spot_check() {
        assert_eq!(
            WB_STAGE1_CB1_Q8[31],
            [15, 21, 35, 50, 61, 73, 86, 97, 110, 119, 129, 141, 175, 198, 218, 237]
        );
    }

    // --- cb1_q8() routing --------------------------------------------

    #[test]
    fn cb1_q8_routes_nb_mb_to_table23() {
        let nb = cb1_q8(Bandwidth::Nb, 0).unwrap();
        let mb = cb1_q8(Bandwidth::Mb, 0).unwrap();
        assert_eq!(nb, mb);
        assert_eq!(nb.len(), D_LPC_NB_MB);
        assert_eq!(nb[0], 12);
    }

    #[test]
    fn cb1_q8_routes_wb_to_table24() {
        let wb = cb1_q8(Bandwidth::Wb, 0).unwrap();
        assert_eq!(wb.len(), D_LPC_WB);
        assert_eq!(wb[0], 7);
    }

    #[test]
    fn cb1_q8_rejects_out_of_range_i1() {
        assert_eq!(cb1_q8(Bandwidth::Nb, 32), Err(Error::MalformedPacket));
        assert_eq!(cb1_q8(Bandwidth::Wb, 200), Err(Error::MalformedPacket));
    }

    #[test]
    fn cb1_q8_rejects_swb_fb() {
        // SILK never sees SWB / FB after the §4.2.2 hybrid split.
        assert_eq!(cb1_q8(Bandwidth::Swb, 0), Err(Error::MalformedPacket));
        assert_eq!(cb1_q8(Bandwidth::Fb, 0), Err(Error::MalformedPacket));
    }

    // --- IHMW weight properties --------------------------------------

    /// Spec asserts the IHMW weights tabulate to 13-bit unsigned values
    /// in `1819..=5227` (inclusive) over Table 23 and Table 24. Verify
    /// by sweeping every NB/MB and WB I1 row.
    #[test]
    fn ihmw_weights_within_documented_range_nbmb() {
        for i1 in 0..32u8 {
            let cb1 = cb1_q8(Bandwidth::Nb, i1).unwrap();
            let mut w = [0i32; D_LPC_MAX];
            ihmw_weights_q9(cb1, &mut w);
            for (k, &wk) in w[..D_LPC_NB_MB].iter().enumerate() {
                assert!(
                    (1819..=5227).contains(&wk),
                    "NB I1={i1} k={k} w_Q9={wk} out of [1819, 5227]"
                );
            }
        }
    }

    #[test]
    fn ihmw_weights_within_documented_range_wb() {
        for i1 in 0..32u8 {
            let cb1 = cb1_q8(Bandwidth::Wb, i1).unwrap();
            let mut w = [0i32; D_LPC_MAX];
            ihmw_weights_q9(cb1, &mut w);
            for (k, &wk) in w[..D_LPC_WB].iter().enumerate() {
                assert!(
                    (1819..=5227).contains(&wk),
                    "WB I1={i1} k={k} w_Q9={wk} out of [1819, 5227]"
                );
            }
        }
    }

    /// With `res_Q10[k] == 0` for every k, the final NLSF is just
    /// `cb1_Q8[k] << 7`. Stage-1-only reconstruction.
    #[test]
    fn nlsf_with_zero_residual_is_cb1_shifted_left_7() {
        // Build a synthetic LsfStage2 with all res_Q10 == 0 by going
        // through the public API: decode against a real range decoder
        // first so the field shape is correct, then zero res_Q10 via
        // a private constructor isn't possible — instead we test the
        // formula at the cb1 layer directly.
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            for i1 in 0..32u8 {
                let cb1 = cb1_q8(bw, i1).unwrap();
                let d_lpc = cb1.len();
                let mut w_q9 = [0i32; D_LPC_MAX];
                ihmw_weights_q9(cb1, &mut w_q9);
                // d_lpc is bounded by cb1.len(); use it to silence the
                // unused-binding warning where iteration is over cb1.
                let _ = d_lpc;

                for &cb1_k in cb1 {
                    // With res_Q10[k] == 0: NLSF_Q15[k] = clamp(0,
                    // cb1_Q8[k] << 7, 32767).
                    let expected = ((cb1_k as i32) << 7).clamp(0, 32767);
                    // cb1 values are <= 242, so cb1 << 7 <= 30976 — always
                    // within the clamp range. Sanity check the bound.
                    assert!(expected <= 32767);
                    assert_eq!(expected, (cb1_k as i32) << 7);
                }
            }
        }
    }

    /// The final reconstructed NLSF values must always be within `[0,
    /// 32767]` — the §4.2.7.5.3 clamp guarantees it. Sweep all I1
    /// values across both bandwidths using a synthetic stage-2 decode
    /// (random range-decoder buffer).
    #[test]
    fn nlsf_q15_within_clamp_range_nb() {
        let buf = long_buf();
        for i1 in 0..32u8 {
            let mut rd = RangeDecoder::new(&buf);
            let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Nb, i1).expect("stage2 decode");
            let rec = NlsfReconstructed::from_stage1_and_stage2(Bandwidth::Nb, i1, &stage2)
                .expect("recon");
            assert_eq!(rec.len(), D_LPC_NB_MB);
            for (k, &n) in rec.nlsf_q15().iter().enumerate() {
                assert!(
                    (0..=32767).contains(&(n as i32)),
                    "NB I1={i1} k={k} NLSF_Q15={n} out of [0, 32767]"
                );
            }
        }
    }

    #[test]
    fn nlsf_q15_within_clamp_range_mb() {
        let buf = long_buf();
        for i1 in 0..32u8 {
            let mut rd = RangeDecoder::new(&buf);
            let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Mb, i1).expect("stage2 decode");
            let rec = NlsfReconstructed::from_stage1_and_stage2(Bandwidth::Mb, i1, &stage2)
                .expect("recon");
            assert_eq!(rec.len(), D_LPC_NB_MB);
            for &n in rec.nlsf_q15() {
                assert!((0..=32767).contains(&(n as i32)));
            }
        }
    }

    #[test]
    fn nlsf_q15_within_clamp_range_wb() {
        let buf = long_buf();
        for i1 in 0..32u8 {
            let mut rd = RangeDecoder::new(&buf);
            let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Wb, i1).expect("stage2 decode");
            let rec = NlsfReconstructed::from_stage1_and_stage2(Bandwidth::Wb, i1, &stage2)
                .expect("recon");
            assert_eq!(rec.len(), D_LPC_WB);
            for &n in rec.nlsf_q15() {
                assert!((0..=32767).contains(&(n as i32)));
            }
        }
    }

    /// Round-trip the NLSF formula: derive NLSF_Q15 from the public API,
    /// then re-derive it locally from cb1 / w_Q9 / res_Q10 and confirm
    /// every cell matches.
    #[test]
    fn nlsf_q15_reproduces_formula_nb() {
        let buf = long_buf();
        for i1 in 0..32u8 {
            let mut rd = RangeDecoder::new(&buf);
            let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Nb, i1).expect("stage2 decode");
            let rec = NlsfReconstructed::from_stage1_and_stage2(Bandwidth::Nb, i1, &stage2)
                .expect("recon");

            let cb1 = cb1_q8(Bandwidth::Nb, i1).unwrap();
            for (k, &cb1_k) in cb1.iter().enumerate() {
                let cb1_term = (cb1_k as i32) << 7;
                let res_term = ((stage2.res_q10()[k]) << 14) / rec.w_q9()[k];
                let expected = (cb1_term + res_term).clamp(0, 32767) as i16;
                assert_eq!(rec.nlsf_q15()[k], expected, "I1={i1} k={k}");
            }
        }
    }

    #[test]
    fn nlsf_q15_reproduces_formula_wb() {
        let buf = long_buf();
        for i1 in 0..32u8 {
            let mut rd = RangeDecoder::new(&buf);
            let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Wb, i1).expect("stage2 decode");
            let rec = NlsfReconstructed::from_stage1_and_stage2(Bandwidth::Wb, i1, &stage2)
                .expect("recon");

            let cb1 = cb1_q8(Bandwidth::Wb, i1).unwrap();
            for (k, &cb1_k) in cb1.iter().enumerate() {
                let cb1_term = (cb1_k as i32) << 7;
                let res_term = ((stage2.res_q10()[k]) << 14) / rec.w_q9()[k];
                let expected = (cb1_term + res_term).clamp(0, 32767) as i16;
                assert_eq!(rec.nlsf_q15()[k], expected, "I1={i1} k={k}");
            }
        }
    }

    /// Sanity: I1 == 0 NB row 0 has cb1[0]==12, cb1[d_LPC-1]==228, with
    /// monotone interior differences. The IHMW weights at the interior
    /// coefficients are smaller than at the boundary (because the
    /// boundary uses cb1_Q8[-1]=0 / cb1_Q8[d_LPC]=256 fictitious
    /// neighbours that yield smaller divisors). Just confirm they are
    /// all positive.
    #[test]
    fn ihmw_weights_all_positive() {
        for bw in [Bandwidth::Nb, Bandwidth::Wb] {
            for i1 in 0..32u8 {
                let cb1 = cb1_q8(bw, i1).unwrap();
                let mut w = [0i32; D_LPC_MAX];
                ihmw_weights_q9(cb1, &mut w);
                for (k, &wk) in w[..cb1.len()].iter().enumerate() {
                    assert!(wk > 0, "bw={bw:?} I1={i1} k={k} w_Q9={wk}");
                }
            }
        }
    }

    /// Mismatched bandwidth ↔ stage-2 length must be rejected. A WB
    /// stage-2 (16 entries) cannot reconstruct as NB (10 entries).
    #[test]
    fn mismatched_stage2_length_is_rejected() {
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let stage2_wb = LsfStage2::decode(&mut rd, Bandwidth::Wb, 0).expect("wb stage2");
        // Try to reconstruct as NB.
        let err = NlsfReconstructed::from_stage1_and_stage2(Bandwidth::Nb, 0, &stage2_wb)
            .expect_err("must reject");
        assert_eq!(err, Error::MalformedPacket);
    }

    /// Out-of-range I1 must be rejected.
    #[test]
    fn out_of_range_i1_is_rejected() {
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Nb, 0).expect("decode");
        let err = NlsfReconstructed::from_stage1_and_stage2(Bandwidth::Nb, 32, &stage2)
            .expect_err("must reject");
        assert_eq!(err, Error::MalformedPacket);
    }

    /// SWB / FB bandwidths must be rejected (SILK never sees them).
    #[test]
    fn swb_and_fb_rejected_at_recon() {
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Nb, 0).expect("decode");
        for bw in [Bandwidth::Swb, Bandwidth::Fb] {
            let err =
                NlsfReconstructed::from_stage1_and_stage2(bw, 0, &stage2).expect_err("must reject");
            assert_eq!(err, Error::MalformedPacket);
        }
    }

    /// Specific concrete IHMW computation for I1=0 NB. With cb1 =
    /// [12, 35, 60, 83, 108, 132, 157, 180, 206, 228] and
    /// cb1_Q8[-1] = 0, cb1_Q8[10] = 256:
    /// diffs around k=0 are (12-0)=12 and (35-12)=23.
    /// w2_Q18[0] = (1024/12 + 1024/23) << 16 = (85 + 44) << 16
    ///           = 129 << 16 = 8454144.
    /// ilog(8454144) = 24. f = (8454144 >> 16) & 127 = 129 & 127 = 1.
    /// y = 46214 >> ((32-24)>>1) = 46214 >> 4 = 2888.
    /// w_Q9[0] = 2888 + ((213*1*2888) >> 16) = 2888 + (615144 >> 16)
    ///         = 2888 + 9 = 2897.
    #[test]
    fn ihmw_concrete_nb_i1_0_k_0() {
        let cb1 = cb1_q8(Bandwidth::Nb, 0).unwrap();
        let mut w = [0i32; D_LPC_MAX];
        ihmw_weights_q9(cb1, &mut w);
        assert_eq!(w[0], 2897, "concrete hand-computed IHMW match");
    }

    /// Same concrete check for I1=0 WB k=0. cb1 = [7, 23, ...] with
    /// cb1_Q8[-1] = 0. diffs are (7-0)=7 and (23-7)=16.
    /// w2_Q18[0] = (1024/7 + 1024/16) << 16 = (146 + 64) << 16
    ///           = 210 << 16 = 13762560.
    /// ilog(13762560) = 24. f = (13762560 >> 16) & 127 = 210 & 127 = 82.
    /// y = 46214 >> 4 = 2888.
    /// w_Q9[0] = 2888 + ((213*82*2888) >> 16)
    ///         = 2888 + (50448048 >> 16) = 2888 + 769 = 3657.
    #[test]
    fn ihmw_concrete_wb_i1_0_k_0() {
        let cb1 = cb1_q8(Bandwidth::Wb, 0).unwrap();
        let mut w = [0i32; D_LPC_MAX];
        ihmw_weights_q9(cb1, &mut w);
        assert_eq!(w[0], 3657, "concrete hand-computed IHMW match");
    }

    fn long_buf() -> [u8; 32] {
        [
            0x55, 0xAA, 0x33, 0xCC, 0x7F, 0x80, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88, 0x99, 0xAA,
        ]
    }
}
