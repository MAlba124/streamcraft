//! SILK Normalized LSF Stage-2 decoding — RFC 6716 §4.2.7.5.2.
//!
//! After the §4.2.7.5.1 stage-1 codebook index `I1` is decoded, the SILK
//! decoder reads `d_LPC` stage-2 residual indices `I2[k]` (10 for NB / MB,
//! 16 for WB) using one of 16 PDFs:
//!
//! * 8 PDFs `a..h` for NB / MB (Table 15).
//! * 8 PDFs `i..p` for WB (Table 16).
//!
//! Which PDF is used for which coefficient is driven by `I1` via the
//! Table 17 (NB/MB) and Table 18 (WB) selection maps.
//!
//! Each raw symbol is in `0..=8`; subtract 4 to get a signed index in
//! `[-4, 4]`. If `|idx| == 4`, a second symbol is read from the Table 19
//! extension PDF (7 cells, in `0..=6`) and added to the index magnitude
//! (preserving sign), yielding `I2[k] ∈ [-10, 10]`.
//!
//! After all `d_LPC` `I2[k]` are read, the decoder performs the
//! backwards-prediction inverse:
//!
//! ```text
//! for k from d_LPC-1 down to 0:
//!     res_Q10[k] = (k+1 < d_LPC ? (res_Q10[k+1]*pred_Q8[k])>>8 : 0)
//!                + ((((I2[k]<<10) - sign(I2[k])*102) * qstep) >> 16)
//! ```
//!
//! where `qstep = 11796` (Q16) for NB / MB and `9830` for WB, and
//! `pred_Q8[k]` for `0 <= k < d_LPC - 1` is the weight selected from
//! Table 20 by Table 21 (NB/MB) or Table 22 (WB), indexed by `I1`.
//!
//! The output `res_Q10[]` is the Q10 stage-2 residual that the
//! §4.2.7.5.3 reconstruction step feeds into the final NLSF formula.
//! Round 6 lands ONLY through `res_Q10[]`; §4.2.7.5.3 codebook lookup,
//! the IHMW weights, the §4.2.7.5.4 stabilization, and §4.2.7.5.5
//! interpolation are deferred to round 7+.

use crate::range_decoder::RangeDecoder;
use crate::range_encoder::RangeEncoder;
use crate::toc::Bandwidth;
use crate::Error;

/// Order of the SILK LPC for NB and MB (Table 15 / 17 / 21 width).
pub const D_LPC_NB_MB: usize = 10;

/// Order of the SILK LPC for WB (Table 16 / 18 / 22 width).
pub const D_LPC_WB: usize = 16;

/// Upper bound on `d_LPC` (the WB value); sizes the fixed-size arrays.
pub const D_LPC_MAX: usize = D_LPC_WB;

/// Q16 quantization step for NB / MB stage-2 residual.
pub const QSTEP_NB_MB_Q16: i32 = 11796;

/// Q16 quantization step for WB stage-2 residual.
pub const QSTEP_WB_Q16: i32 = 9830;

// =====================================================================
// Table 15: NB / MB stage-2 PDFs (codebooks a..h). Each row is 9 cells
// summing to 256. Stored as iCDF with a terminating 0 (10 bytes).
// =====================================================================

/// Table 15 codebook `a`: PDF {1, 1, 1, 15, 224, 11, 1, 1, 1}/256.
/// fh = [1, 2, 3, 18, 242, 253, 254, 255, 256]; iCDF = 256 - fh.
const NBMB_STAGE2_ICDF_A: &[u8] = &[255, 254, 253, 238, 14, 3, 2, 1, 0];
/// Table 15 codebook `b`: PDF {1, 1, 2, 34, 183, 32, 1, 1, 1}/256.
const NBMB_STAGE2_ICDF_B: &[u8] = &[255, 254, 252, 218, 35, 3, 2, 1, 0];
/// Table 15 codebook `c`: PDF {1, 1, 4, 42, 149, 55, 2, 1, 1}/256.
const NBMB_STAGE2_ICDF_C: &[u8] = &[255, 254, 250, 208, 59, 4, 2, 1, 0];
/// Table 15 codebook `d`: PDF {1, 1, 8, 52, 123, 61, 8, 1, 1}/256.
const NBMB_STAGE2_ICDF_D: &[u8] = &[255, 254, 246, 194, 71, 10, 2, 1, 0];
/// Table 15 codebook `e`: PDF {1, 3, 16, 53, 101, 74, 6, 1, 1}/256.
const NBMB_STAGE2_ICDF_E: &[u8] = &[255, 252, 236, 183, 82, 8, 2, 1, 0];
/// Table 15 codebook `f`: PDF {1, 3, 17, 55, 90, 73, 15, 1, 1}/256.
const NBMB_STAGE2_ICDF_F: &[u8] = &[255, 252, 235, 180, 90, 17, 2, 1, 0];
/// Table 15 codebook `g`: PDF {1, 7, 24, 53, 74, 67, 26, 3, 1}/256.
const NBMB_STAGE2_ICDF_G: &[u8] = &[255, 248, 224, 171, 97, 30, 4, 1, 0];
/// Table 15 codebook `h`: PDF {1, 1, 18, 63, 78, 58, 30, 6, 1}/256.
const NBMB_STAGE2_ICDF_H: &[u8] = &[255, 254, 236, 173, 95, 37, 7, 1, 0];

// =====================================================================
// Table 16: WB stage-2 PDFs (codebooks i..p). Each row is 9 cells
// summing to 256. Stored as iCDF with terminating 0.
// =====================================================================

/// Table 16 codebook `i`: PDF {1, 1, 1, 9, 232, 9, 1, 1, 1}/256.
const WB_STAGE2_ICDF_I: &[u8] = &[255, 254, 253, 244, 12, 3, 2, 1, 0];
/// Table 16 codebook `j`: PDF {1, 1, 2, 28, 186, 35, 1, 1, 1}/256.
const WB_STAGE2_ICDF_J: &[u8] = &[255, 254, 252, 224, 38, 3, 2, 1, 0];
/// Table 16 codebook `k`: PDF {1, 1, 3, 42, 152, 53, 2, 1, 1}/256.
const WB_STAGE2_ICDF_K: &[u8] = &[255, 254, 251, 209, 57, 4, 2, 1, 0];
/// Table 16 codebook `l`: PDF {1, 1, 10, 49, 126, 65, 2, 1, 1}/256.
const WB_STAGE2_ICDF_L: &[u8] = &[255, 254, 244, 195, 69, 4, 2, 1, 0];
/// Table 16 codebook `m`: PDF {1, 4, 19, 48, 100, 77, 5, 1, 1}/256.
const WB_STAGE2_ICDF_M: &[u8] = &[255, 251, 232, 184, 84, 7, 2, 1, 0];
/// Table 16 codebook `n`: PDF {1, 1, 14, 54, 100, 72, 12, 1, 1}/256.
const WB_STAGE2_ICDF_N: &[u8] = &[255, 254, 240, 186, 86, 14, 2, 1, 0];
/// Table 16 codebook `o`: PDF {1, 1, 15, 61, 87, 61, 25, 4, 1}/256.
const WB_STAGE2_ICDF_O: &[u8] = &[255, 254, 239, 178, 91, 30, 5, 1, 0];
/// Table 16 codebook `p`: PDF {1, 7, 21, 50, 77, 81, 17, 1, 1}/256.
const WB_STAGE2_ICDF_P: &[u8] = &[255, 248, 227, 177, 100, 19, 2, 1, 0];

/// Table 19 — extension PDF for `|I2[k]| == 4`: {156, 60, 24, 9, 4, 2, 1}/256.
/// fh = [156, 216, 240, 249, 253, 255, 256]; iCDF = 256 - fh.
const STAGE2_EXTENSION_ICDF: &[u8] = &[100, 40, 16, 7, 3, 1, 0];

/// Table 17 — Codebook Selection for NB/MB Normalized LSF Stage-2 Index
/// Decoding. Rows are I1=0..=31; columns are coefficient 0..=9.
///
/// Letters `a..h` are stored as `0..=7`. The row labelled `g` at I1=6 in
/// the RFC text is a typographical error in the source document; the
/// neighbouring cells are valid codebook letters. We restore the row at
/// I1=6 here (its contents `a c c c c c c c c b` are the unaltered
/// table cells from that row).
#[rustfmt::skip]
const NBMB_STAGE2_SELECT: [[u8; D_LPC_NB_MB]; 32] = [
    // I1 = 0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // a a a a a a a a a a
    // I1 = 1
    [1, 3, 1, 2, 2, 1, 2, 1, 1, 1], // b d b c c b c b b b
    // I1 = 2
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1], // c b b b b b b b b b
    // I1 = 3
    [1, 2, 2, 2, 2, 1, 2, 1, 1, 1], // b c c c c b c b b b
    // I1 = 4
    [2, 3, 3, 3, 3, 2, 2, 2, 2, 2], // c d d d d c c c c c
    // I1 = 5
    [0, 5, 3, 3, 2, 2, 2, 2, 1, 1], // a f d d c c c c b b
    // I1 = 6  (RFC row-label typo "g" corrected to "6")
    [0, 2, 2, 2, 2, 2, 2, 2, 2, 1], // a c c c c c c c c b
    // I1 = 7
    [2, 3, 6, 4, 4, 4, 5, 4, 5, 5], // c d g e e e f e f f
    // I1 = 8
    [2, 4, 5, 5, 4, 5, 4, 6, 4, 4], // c e f f e f e g e e
    // I1 = 9
    [2, 4, 4, 7, 4, 5, 4, 5, 5, 4], // c e e h e f e f f e
    // I1 = 10
    [4, 3, 3, 3, 2, 3, 2, 2, 2, 2], // e d d d c d c c c c
    // I1 = 11
    [1, 5, 5, 6, 4, 5, 4, 5, 5, 5], // b f f g e f e f f f
    // I1 = 12
    [2, 7, 4, 6, 5, 5, 5, 5, 5, 5], // c h e g f f f f f f
    // I1 = 13
    [2, 7, 5, 5, 5, 5, 5, 6, 5, 4], // c h f f f f f g f e
    // I1 = 14
    [3, 3, 5, 4, 4, 5, 4, 5, 4, 4], // d d f e e f e f e e
    // I1 = 15
    [2, 3, 3, 5, 5, 4, 4, 4, 4, 4], // c d d f f e e e e e
    // I1 = 16
    [2, 4, 4, 6, 4, 5, 4, 5, 5, 5], // c e e g e f e f f f
    // I1 = 17
    [2, 5, 4, 6, 5, 5, 5, 4, 5, 4], // c f e g f f f e f e
    // I1 = 18
    [2, 7, 4, 5, 4, 5, 4, 5, 5, 5], // c h e f e f e f f f
    // I1 = 19
    [2, 5, 4, 6, 7, 6, 5, 6, 5, 4], // c f e g h g f g f e
    // I1 = 20
    [3, 6, 7, 4, 6, 5, 5, 6, 4, 5], // d g h e g f f g e f
    // I1 = 21
    [2, 7, 6, 4, 4, 4, 5, 4, 5, 5], // c h g e e e f e f f
    // I1 = 22
    [4, 5, 5, 4, 6, 6, 5, 6, 5, 4], // e f f e g g f g f e
    // I1 = 23
    [2, 5, 5, 6, 5, 6, 4, 6, 4, 4], // c f f g f g e g e e
    // I1 = 24
    [4, 5, 5, 5, 3, 7, 4, 5, 5, 4], // e f f f d h e f f e
    // I1 = 25
    [2, 3, 4, 5, 5, 6, 4, 5, 5, 4], // c d e f f g e f f e
    // I1 = 26
    [2, 3, 2, 3, 3, 4, 2, 3, 3, 3], // c d c d d e c d d d
    // I1 = 27
    [1, 1, 2, 2, 2, 2, 2, 3, 2, 2], // b b c c c c c d c c
    // I1 = 28
    [4, 5, 5, 6, 6, 6, 5, 6, 4, 5], // e f f g g g f g e f
    // I1 = 29
    [3, 5, 5, 4, 4, 4, 4, 3, 3, 2], // d f f e e e e d d c
    // I1 = 30
    [2, 5, 3, 7, 5, 5, 4, 4, 5, 4], // c f d h f f e e f e
    // I1 = 31
    [4, 4, 5, 4, 5, 6, 5, 6, 5, 4], // e e f e f g f g f e
];

/// Table 18 — Codebook Selection for WB Normalized LSF Stage-2 Index
/// Decoding. Rows are I1=0..=31; columns are coefficient 0..=15.
/// Letters `i..p` are stored as `0..=7`.
#[rustfmt::skip]
const WB_STAGE2_SELECT: [[u8; D_LPC_WB]; 32] = [
    // I1 = 0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    // I1 = 1
    [2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 0, 3],
    // I1 = 2
    [2, 5, 5, 3, 7, 4, 4, 5, 2, 5, 4, 5, 5, 4, 3, 3],
    // I1 = 3
    [0, 2, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
    // I1 = 4
    [0, 6, 5, 4, 6, 4, 7, 5, 4, 4, 4, 5, 5, 4, 4, 3],
    // I1 = 5
    [0, 3, 5, 5, 4, 3, 3, 5, 3, 3, 3, 3, 3, 3, 2, 4],
    // I1 = 6
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    // I1 = 7
    [0, 2, 6, 3, 7, 2, 5, 3, 4, 5, 5, 4, 3, 3, 2, 3],
    // I1 = 8
    [0, 6, 2, 6, 6, 4, 5, 4, 6, 5, 4, 4, 5, 3, 3, 3],
    // I1 = 9
    [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    // I1 = 10
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    // I1 = 11
    [2, 2, 3, 4, 5, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 3],
    // I1 = 12
    [2, 2, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 3],
    // I1 = 13
    [3, 4, 4, 4, 6, 4, 4, 5, 3, 5, 4, 4, 5, 4, 3, 4],
    // I1 = 14
    [0, 6, 4, 5, 4, 7, 5, 2, 6, 5, 7, 4, 4, 3, 5, 3],
    // I1 = 15
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    // I1 = 16
    [1, 6, 5, 7, 5, 4, 5, 3, 4, 5, 4, 4, 4, 3, 3, 4],
    // I1 = 17
    [1, 3, 3, 4, 4, 3, 3, 5, 2, 3, 3, 5, 5, 5, 3, 4],
    // I1 = 18
    [2, 3, 3, 2, 2, 2, 3, 2, 1, 2, 1, 2, 1, 1, 1, 4],
    // I1 = 19
    [0, 2, 3, 5, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0],
    // I1 = 20
    [3, 4, 3, 5, 3, 3, 2, 2, 1, 1, 1, 1, 1, 2, 2, 4],
    // I1 = 21
    [2, 6, 3, 7, 7, 4, 5, 4, 5, 3, 5, 3, 3, 2, 3, 3],
    // I1 = 22
    [2, 3, 5, 6, 6, 3, 5, 3, 4, 4, 3, 3, 3, 3, 2, 4],
    // I1 = 23
    [1, 3, 3, 4, 4, 4, 4, 3, 5, 5, 5, 3, 1, 1, 1, 1],
    // I1 = 24
    [2, 5, 3, 6, 6, 4, 7, 4, 4, 5, 3, 4, 4, 3, 3, 3],
    // I1 = 25
    [0, 6, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    // I1 = 26
    [0, 6, 6, 3, 5, 2, 5, 5, 3, 4, 4, 7, 7, 4, 4, 4],
    // I1 = 27
    [3, 3, 7, 3, 5, 4, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3],
    // I1 = 28
    [0, 0, 1, 0, 0, 0, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1],
    // I1 = 29
    [0, 3, 2, 5, 3, 3, 2, 3, 2, 1, 0, 0, 1, 0, 0, 1],
    // I1 = 30
    [3, 5, 5, 4, 7, 5, 3, 3, 2, 3, 2, 2, 1, 0, 1, 0],
    // I1 = 31
    [2, 3, 5, 3, 4, 3, 3, 3, 2, 1, 2, 6, 4, 0, 0, 0],
];

/// Look up the iCDF for an NB/MB stage-2 codebook letter (`0..=7` =>
/// `a..h`).
const fn nbmb_stage2_icdf(letter: u8) -> &'static [u8] {
    match letter {
        0 => NBMB_STAGE2_ICDF_A,
        1 => NBMB_STAGE2_ICDF_B,
        2 => NBMB_STAGE2_ICDF_C,
        3 => NBMB_STAGE2_ICDF_D,
        4 => NBMB_STAGE2_ICDF_E,
        5 => NBMB_STAGE2_ICDF_F,
        6 => NBMB_STAGE2_ICDF_G,
        7 => NBMB_STAGE2_ICDF_H,
        _ => NBMB_STAGE2_ICDF_A,
    }
}

/// Look up the iCDF for a WB stage-2 codebook letter (`0..=7` =>
/// `i..p`).
const fn wb_stage2_icdf(letter: u8) -> &'static [u8] {
    match letter {
        0 => WB_STAGE2_ICDF_I,
        1 => WB_STAGE2_ICDF_J,
        2 => WB_STAGE2_ICDF_K,
        3 => WB_STAGE2_ICDF_L,
        4 => WB_STAGE2_ICDF_M,
        5 => WB_STAGE2_ICDF_N,
        6 => WB_STAGE2_ICDF_O,
        7 => WB_STAGE2_ICDF_P,
        _ => WB_STAGE2_ICDF_I,
    }
}

// =====================================================================
// Table 20: Prediction Weights for Normalized LSF Decoding (Q8).
//
// Lists A / B are NB/MB (indexed 0..=8). Lists C / D are WB (indexed
// 0..=14). The Q8 weight is read with `pred_Q8[k] = list[k]`.
// =====================================================================

const NBMB_PRED_WEIGHT_A: [u8; 9] = [179, 138, 140, 148, 151, 149, 153, 151, 163];
const NBMB_PRED_WEIGHT_B: [u8; 9] = [116, 67, 82, 59, 92, 72, 100, 89, 92];

const WB_PRED_WEIGHT_C: [u8; 15] = [
    175, 148, 160, 176, 178, 173, 174, 164, 177, 174, 196, 182, 198, 192, 182,
];
const WB_PRED_WEIGHT_D: [u8; 15] = [
    68, 62, 66, 60, 72, 117, 85, 90, 118, 136, 151, 142, 160, 142, 155,
];

// =====================================================================
// Table 21: NB/MB Prediction Weight Selection.
//
// 0 = list A, 1 = list B. Indexed by I1 (0..=31) then coefficient
// (0..=8). The table covers `d_LPC - 1 == 9` coefficients (the final
// coefficient has no successor, so no pred_Q8 entry).
// =====================================================================

#[rustfmt::skip]
const NBMB_PRED_WEIGHT_SELECT: [[u8; D_LPC_NB_MB - 1]; 32] = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0], // 0
    [1, 0, 0, 0, 0, 0, 0, 0, 0], // 1
    [0, 0, 0, 0, 0, 0, 0, 0, 0], // 2
    [1, 1, 1, 0, 0, 0, 0, 1, 0], // 3
    [0, 1, 0, 0, 0, 0, 0, 0, 0], // 4
    [0, 1, 0, 0, 0, 0, 0, 0, 0], // 5
    [1, 0, 1, 1, 0, 0, 0, 1, 0], // 6
    [0, 1, 1, 0, 0, 1, 1, 0, 0], // 7
    [0, 0, 1, 1, 0, 1, 0, 1, 1], // 8
    [0, 0, 1, 1, 0, 0, 1, 1, 1], // 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0], // 10
    [0, 1, 0, 1, 1, 1, 1, 1, 0], // 11
    [0, 1, 0, 1, 1, 1, 1, 1, 0], // 12
    [0, 1, 1, 1, 1, 1, 1, 1, 0], // 13
    [1, 0, 1, 1, 0, 1, 1, 1, 1], // 14
    [0, 1, 1, 1, 1, 1, 0, 1, 0], // 15
    [0, 0, 1, 1, 0, 1, 0, 1, 0], // 16
    [0, 0, 1, 1, 1, 0, 1, 1, 1], // 17
    [0, 1, 1, 0, 0, 1, 1, 1, 0], // 18
    [0, 0, 0, 1, 1, 1, 0, 1, 0], // 19
    [0, 1, 1, 0, 0, 1, 0, 1, 0], // 20
    [0, 1, 1, 0, 0, 0, 1, 1, 0], // 21
    [0, 0, 0, 0, 0, 1, 1, 1, 1], // 22
    [0, 0, 1, 1, 0, 0, 0, 1, 1], // 23
    [0, 0, 0, 1, 0, 1, 1, 1, 1], // 24
    [0, 1, 1, 1, 1, 1, 1, 1, 0], // 25
    [0, 0, 0, 0, 0, 0, 0, 0, 0], // 26
    [0, 0, 0, 0, 0, 0, 0, 0, 0], // 27
    [0, 0, 1, 0, 1, 1, 0, 1, 0], // 28
    [1, 0, 0, 1, 0, 0, 0, 0, 0], // 29
    [0, 0, 0, 1, 1, 0, 1, 0, 1], // 30
    [1, 0, 1, 1, 0, 1, 1, 1, 1], // 31
];

// =====================================================================
// Table 22: WB Prediction Weight Selection.
//
// 0 = list C, 1 = list D. Indexed by I1 (0..=31) then coefficient
// (0..=14). The table covers `d_LPC - 1 == 15` coefficients.
// =====================================================================

#[rustfmt::skip]
const WB_PRED_WEIGHT_SELECT: [[u8; D_LPC_WB - 1]; 32] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], // 0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 1
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0], // 2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], // 3
    [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0], // 4
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 5
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], // 6
    [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1], // 7
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1], // 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], // 9
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 10
    [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0], // 11
    [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0], // 12
    [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0], // 13
    [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1], // 14
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], // 15
    [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0], // 16
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0], // 17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], // 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], // 19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 20
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], // 21
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0], // 22
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], // 23
    [0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], // 24
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], // 25
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1], // 26
    [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], // 27
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], // 28
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], // 29
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], // 30
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0], // 31
];

/// Per-bandwidth bundle of references into the static tables, picked
/// once per `LsfStage2::decode` call by `bandwidth`. Lifts the
/// otherwise-noisy `(usize, i32, &[u8], fn, &[u8], &[u8], &[u8])`
/// tuple out of the function body.
struct StageTables {
    d_lpc: usize,
    qstep: i32,
    select_row: &'static [u8],
    icdf_lookup: fn(u8) -> &'static [u8],
    pred_weight_select_row: &'static [u8],
    weight_list_0: &'static [u8],
    weight_list_1: &'static [u8],
}

impl StageTables {
    /// Pick the per-bandwidth tables for stage-1 index `I1 < 32`.
    /// Hybrid is handled upstream by splitting the signal; the SILK
    /// layer always sees Nb / Mb / Wb — SWB / FB are rejected.
    fn for_bandwidth(bandwidth: Bandwidth, lsf_stage1: u8) -> Result<Self, Error> {
        if lsf_stage1 >= 32 {
            return Err(Error::MalformedPacket);
        }
        let i1 = lsf_stage1 as usize;
        match bandwidth {
            Bandwidth::Nb | Bandwidth::Mb => Ok(StageTables {
                d_lpc: D_LPC_NB_MB,
                qstep: QSTEP_NB_MB_Q16,
                select_row: &NBMB_STAGE2_SELECT[i1][..],
                icdf_lookup: nbmb_stage2_icdf,
                pred_weight_select_row: &NBMB_PRED_WEIGHT_SELECT[i1][..],
                weight_list_0: &NBMB_PRED_WEIGHT_A[..],
                weight_list_1: &NBMB_PRED_WEIGHT_B[..],
            }),
            Bandwidth::Wb => Ok(StageTables {
                d_lpc: D_LPC_WB,
                qstep: QSTEP_WB_Q16,
                select_row: &WB_STAGE2_SELECT[i1][..],
                icdf_lookup: wb_stage2_icdf,
                pred_weight_select_row: &WB_PRED_WEIGHT_SELECT[i1][..],
                weight_list_0: &WB_PRED_WEIGHT_C[..],
                weight_list_1: &WB_PRED_WEIGHT_D[..],
            }),
            _ => Err(Error::MalformedPacket),
        }
    }
}

/// The §4.2.7.5.2 backwards-prediction inverse, shared by the decode
/// and encode paths:
///
/// ```text
/// res_Q10[k] = (k+1 < d_LPC ? (res_Q10[k+1]*pred_Q8[k])>>8 : 0)
///            + ((((I2[k]<<10) - sign(I2[k])*102) * qstep) >> 16)
/// ```
///
/// The recursion runs `k = d_LPC-1` down to `0`. `pred_Q8[k]` is only
/// defined for `0 <= k < d_LPC - 1`; for `k == d_LPC-1` the first term
/// is zero (the `(k+1 < d_LPC)` guard).
fn compute_res_q10(
    i2: &[i8; D_LPC_MAX],
    d_lpc: usize,
    qstep: i32,
    weight_lists: (&[u8], &[u8]),
    pred_weight_select_row: &[u8],
) -> [i32; D_LPC_MAX] {
    let mut res_q10 = [0i32; D_LPC_MAX];
    for k in (0..d_lpc).rev() {
        // Coefficient-quantisation contribution.
        let i2k = i2[k] as i32;
        let sign = match i2k.cmp(&0) {
            core::cmp::Ordering::Less => -1,
            core::cmp::Ordering::Greater => 1,
            core::cmp::Ordering::Equal => 0,
        };
        let q_contrib = (((i2k << 10) - sign * 102) * qstep) >> 16;

        // Backwards-prediction contribution.
        let pred_contrib = if k + 1 < d_lpc {
            let pred_q8 = pred_weight(weight_lists, pred_weight_select_row, k);
            (res_q10[k + 1] * pred_q8) >> 8
        } else {
            0
        };

        res_q10[k] = pred_contrib + q_contrib;
    }
    res_q10
}

/// Decoded stage-2 result for one SILK frame: the `d_LPC` signed indices
/// `I2[k] ∈ [-10, 10]`, plus the Q10 backwards-prediction-undone residual
/// `res_Q10[k]`. Round 6 stops here; round 7+ will feed `res_Q10[]` and
/// the stage-1 codebook `cb1_Q8[]` into the §4.2.7.5.3 reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LsfStage2 {
    /// Number of populated entries in [`Self::i2`] and [`Self::res_q10`].
    /// 10 for NB / MB, 16 for WB.
    len: u8,
    /// Raw signed stage-2 indices `I2[k] ∈ [-10, 10]`. Indices
    /// `[0..len]` are populated; trailing entries are zero-padded.
    i2: [i8; D_LPC_MAX],
    /// Q10 stage-2 residual `res_Q10[k]` after the §4.2.7.5.2
    /// backwards-prediction inverse. Indices `[0..len]` are populated.
    res_q10: [i32; D_LPC_MAX],
}

impl LsfStage2 {
    /// Decode all `d_LPC` stage-2 indices and the resulting `res_Q10[]`
    /// for the given SILK frame.
    ///
    /// `bandwidth` must be one of `Nb`, `Mb`, `Wb`; SWB / FB are
    /// rejected as `Error::MalformedPacket` (SILK never sees SWB / FB
    /// after the §4.2.2 split).
    ///
    /// `lsf_stage1` is the `I1 ∈ 0..32` value returned by
    /// [`crate::silk_frame::SilkFrameHeader::decode`].
    pub fn decode(
        rd: &mut RangeDecoder<'_>,
        bandwidth: Bandwidth,
        lsf_stage1: u8,
    ) -> Result<Self, Error> {
        // signal_type is unused for stage-2; it only mattered at stage-1.
        // Kept off the signature so the caller doesn't need to thread it.

        let tables = StageTables::for_bandwidth(bandwidth, lsf_stage1)?;
        let StageTables {
            d_lpc,
            qstep,
            select_row,
            icdf_lookup,
            pred_weight_select_row,
            weight_list_0,
            weight_list_1,
        } = tables;

        // Pass 1 — read the raw `I2[k]` indices, applying the Table 19
        // extension whenever the base symbol saturates at the ±4 edge.
        let mut i2 = [0i8; D_LPC_MAX];
        for k in 0..d_lpc {
            let letter = select_row[k];
            let icdf = icdf_lookup(letter);
            // dec_icdf returns 0..=8 from the 9-cell table.
            let raw = rd.dec_icdf(icdf, 8) as i32;
            // §4.2.7.5.2: "subtract 4 ... an index in the range -4 to 4".
            let mut idx = raw - 4;
            if idx == 4 || idx == -4 {
                // Extension: read second symbol from Table 19 (7 cells,
                // values 0..=6) and add with the same sign as `idx`.
                let ext = rd.dec_icdf(STAGE2_EXTENSION_ICDF, 8) as i32;
                if idx > 0 {
                    idx += ext;
                } else {
                    idx -= ext;
                }
            }
            // §4.2.7.5.2: total range is -10..=10 after the optional
            // extension (4 + max-ext-6 = 10). Defend regardless.
            if !(-10..=10).contains(&idx) {
                return Err(Error::MalformedPacket);
            }
            i2[k] = idx as i8;
        }

        if rd.has_error() {
            return Err(Error::MalformedPacket);
        }

        // Pass 2 — backwards-prediction inverse over the `I2[k]`.
        let res_q10 = compute_res_q10(
            &i2,
            d_lpc,
            qstep,
            (weight_list_0, weight_list_1),
            pred_weight_select_row,
        );

        Ok(Self {
            len: d_lpc as u8,
            i2,
            res_q10,
        })
    }

    /// Encode `d_LPC` stage-2 indices `I2[k] ∈ [-10, 10]` into `re` —
    /// the exact write-side mirror of [`Self::decode`].
    ///
    /// Each index with `|I2[k]| <= 3` is written as the single base
    /// symbol `I2[k] + 4`; each index with `|I2[k]| >= 4` is written as
    /// the saturated base symbol (`0` or `8`) followed by the Table 19
    /// extension symbol `|I2[k]| - 4` — the §4.2.7.5.2 coding is
    /// prefix-unique, so this choice is forced. Returns the
    /// [`LsfStage2`] the decoder will reconstruct (identical `i2` plus
    /// the derived `res_Q10[]`).
    ///
    /// `i2.len()` must equal the bandwidth's `d_LPC` (10 for NB / MB,
    /// 16 for WB), every index must lie in `[-10, 10]`, and
    /// `lsf_stage1` must be the same `I1 < 32` the decoder will use
    /// (it selects the per-coefficient codebooks).
    pub fn encode(
        re: &mut RangeEncoder,
        bandwidth: Bandwidth,
        lsf_stage1: u8,
        i2_in: &[i8],
    ) -> Result<Self, Error> {
        let tables = StageTables::for_bandwidth(bandwidth, lsf_stage1)?;
        let StageTables {
            d_lpc,
            qstep,
            select_row,
            icdf_lookup,
            pred_weight_select_row,
            weight_list_0,
            weight_list_1,
        } = tables;

        if i2_in.len() != d_lpc || i2_in.iter().any(|&v| !(-10..=10).contains(&v)) {
            return Err(Error::MalformedPacket);
        }

        let mut i2 = [0i8; D_LPC_MAX];
        i2[..d_lpc].copy_from_slice(i2_in);

        for (k, &idx) in i2_in.iter().enumerate() {
            let icdf = icdf_lookup(select_row[k]);
            let idx = idx as i32;
            if idx.abs() >= 4 {
                // Saturated base symbol + Table 19 extension.
                let base = if idx > 0 { 8usize } else { 0 };
                re.enc_icdf(base, icdf, 8);
                re.enc_icdf((idx.abs() - 4) as usize, STAGE2_EXTENSION_ICDF, 8);
            } else {
                re.enc_icdf((idx + 4) as usize, icdf, 8);
            }
        }

        let res_q10 = compute_res_q10(
            &i2,
            d_lpc,
            qstep,
            (weight_list_0, weight_list_1),
            pred_weight_select_row,
        );

        Ok(Self {
            len: d_lpc as u8,
            i2,
            res_q10,
        })
    }

    /// Build an [`LsfStage2`] directly from `d_LPC` signed indices
    /// `I2[k] ∈ [-10, 10]` without touching a bitstream — the same
    /// non-bitstream tail [`Self::decode`] and [`Self::encode`] share
    /// (Table selection + the §4.2.7.5.2 backwards-prediction
    /// inverse). Used by the encoder's quantisation search, which
    /// needs the `res_Q10[]` of a candidate before deciding whether
    /// to write it.
    pub fn from_indices(bandwidth: Bandwidth, lsf_stage1: u8, i2_in: &[i8]) -> Result<Self, Error> {
        let tables = StageTables::for_bandwidth(bandwidth, lsf_stage1)?;
        if i2_in.len() != tables.d_lpc || i2_in.iter().any(|&v| !(-10..=10).contains(&v)) {
            return Err(Error::MalformedPacket);
        }
        let mut i2 = [0i8; D_LPC_MAX];
        i2[..tables.d_lpc].copy_from_slice(i2_in);
        let res_q10 = compute_res_q10(
            &i2,
            tables.d_lpc,
            tables.qstep,
            (tables.weight_list_0, tables.weight_list_1),
            tables.pred_weight_select_row,
        );
        Ok(Self {
            len: tables.d_lpc as u8,
            i2,
            res_q10,
        })
    }

    /// Quantize a desired stage-2 residual `res_target_q10[]` (the
    /// IHMW-scaled distance from the stage-1 codebook vector to the
    /// analysis NLSF target) into the indices `I2[k]` whose decoded
    /// `res_Q10[]` best approaches it — the encode-side inverse of
    /// the §4.2.7.5.2 reconstruction.
    ///
    /// The decode recursion runs `k = d_LPC-1` down to `0`, each
    /// step adding a backwards prediction from the already-decoded
    /// `res_Q10[k+1]`; this quantiser walks the same direction,
    /// subtracts the prediction that the decoder WILL apply (computed
    /// from the indices already chosen), and picks the index whose
    /// quantisation contribution lands closest to the remainder
    /// (ties toward the smaller |I2|, i.e. cheaper symbols — the
    /// full delayed-decision search of §5.2.3.5 is an encoder-side
    /// freedom this deterministic greedy trades away).
    ///
    /// Returns the [`LsfStage2`] the decoder will reconstruct.
    pub fn quantize(
        bandwidth: Bandwidth,
        lsf_stage1: u8,
        res_target_q10: &[i32],
    ) -> Result<Self, Error> {
        let tables = StageTables::for_bandwidth(bandwidth, lsf_stage1)?;
        let d_lpc = tables.d_lpc;
        if res_target_q10.len() != d_lpc {
            return Err(Error::MalformedPacket);
        }
        let qstep = tables.qstep;
        let q_contrib = |i2: i32| -> i32 {
            let sign = match i2.cmp(&0) {
                core::cmp::Ordering::Less => -1,
                core::cmp::Ordering::Greater => 1,
                core::cmp::Ordering::Equal => 0,
            };
            (((i2 << 10) - sign * 102) * qstep) >> 16
        };

        let mut i2 = [0i8; D_LPC_MAX];
        let mut res_q10 = [0i32; D_LPC_MAX];
        for k in (0..d_lpc).rev() {
            let pred_contrib = if k + 1 < d_lpc {
                let pred_q8 = pred_weight(
                    (tables.weight_list_0, tables.weight_list_1),
                    tables.pred_weight_select_row,
                    k,
                );
                (res_q10[k + 1] * pred_q8) >> 8
            } else {
                0
            };
            let want = res_target_q10[k] - pred_contrib;
            // One I2 step contributes ~qstep<<10>>16 res_Q10 units;
            // seed near the ratio then refine over ±1 neighbours.
            let seed = (((want as i64) << 16) / ((qstep as i64) << 10)) as i32;
            let mut best = (0i32, i32::MAX);
            for cand in (seed - 1)..=(seed + 1) {
                let c = cand.clamp(-10, 10);
                let err = (q_contrib(c) - want).abs();
                if err < best.1 || (err == best.1 && c.abs() < best.0.abs()) {
                    best = (c, err);
                }
            }
            // Zero is always a candidate (cheap symbol; also covers a
            // seed pushed away by the ±102 offset).
            if (q_contrib(0) - want).abs() < best.1 {
                best = (0, (q_contrib(0) - want).abs());
            }
            i2[k] = best.0 as i8;
            res_q10[k] = pred_contrib + q_contrib(best.0);
        }

        Ok(Self {
            len: d_lpc as u8,
            i2,
            res_q10,
        })
    }

    /// Number of populated entries (== `d_LPC`: 10 for NB / MB, 16 for
    /// WB).
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` if no entries were decoded — impossible for a successful
    /// `decode()`.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw `I2[k]` signed indices in `[-10, 10]`.
    pub fn i2(&self) -> &[i8] {
        &self.i2[..self.len()]
    }

    /// Q10 backwards-prediction-undone residual `res_Q10[k]`. Feeds
    /// into §4.2.7.5.3 (deferred to round 7+).
    pub fn res_q10(&self) -> &[i32] {
        &self.res_q10[..self.len()]
    }
}

/// Resolve the Q8 prediction weight for coefficient `k` (`0..d_LPC-1`)
/// from the per-I1 weight-selection row and the (list-0, list-1) pair.
///
/// `weight_lists.0` is "list A" (NB/MB) or "list C" (WB); `.1` is
/// "list B" / "list D".
fn pred_weight(weight_lists: (&[u8], &[u8]), select_row: &[u8], k: usize) -> i32 {
    let list = if select_row[k] == 0 {
        weight_lists.0
    } else {
        weight_lists.1
    };
    list[k] as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Table 15 / 16 PDF→iCDF transcription self-checks --------

    /// Each Table 15 / 16 PDF must sum to 256.
    #[test]
    fn nbmb_stage2_pdfs_sum_to_256() {
        // (label, pdf as written in Table 15)
        let pdfs: &[(&str, [u8; 9])] = &[
            ("a", [1, 1, 1, 15, 224, 11, 1, 1, 1]),
            ("b", [1, 1, 2, 34, 183, 32, 1, 1, 1]),
            ("c", [1, 1, 4, 42, 149, 55, 2, 1, 1]),
            ("d", [1, 1, 8, 52, 123, 61, 8, 1, 1]),
            ("e", [1, 3, 16, 53, 101, 74, 6, 1, 1]),
            ("f", [1, 3, 17, 55, 90, 73, 15, 1, 1]),
            ("g", [1, 7, 24, 53, 74, 67, 26, 3, 1]),
            ("h", [1, 1, 18, 63, 78, 58, 30, 6, 1]),
        ];
        for (label, pdf) in pdfs {
            let s: u32 = pdf.iter().map(|&x| x as u32).sum();
            assert_eq!(s, 256, "Table 15 codebook {label} PDF sum");
        }
    }

    #[test]
    fn wb_stage2_pdfs_sum_to_256() {
        let pdfs: &[(&str, [u8; 9])] = &[
            ("i", [1, 1, 1, 9, 232, 9, 1, 1, 1]),
            ("j", [1, 1, 2, 28, 186, 35, 1, 1, 1]),
            ("k", [1, 1, 3, 42, 152, 53, 2, 1, 1]),
            ("l", [1, 1, 10, 49, 126, 65, 2, 1, 1]),
            ("m", [1, 4, 19, 48, 100, 77, 5, 1, 1]),
            ("n", [1, 1, 14, 54, 100, 72, 12, 1, 1]),
            ("o", [1, 1, 15, 61, 87, 61, 25, 4, 1]),
            ("p", [1, 7, 21, 50, 77, 81, 17, 1, 1]),
        ];
        for (label, pdf) in pdfs {
            let s: u32 = pdf.iter().map(|&x| x as u32).sum();
            assert_eq!(s, 256, "Table 16 codebook {label} PDF sum");
        }
    }

    /// Each codebook iCDF must be monotone non-increasing and end in 0.
    #[test]
    fn stage2_icdfs_well_formed() {
        let tables: &[(&str, &[u8])] = &[
            ("a", NBMB_STAGE2_ICDF_A),
            ("b", NBMB_STAGE2_ICDF_B),
            ("c", NBMB_STAGE2_ICDF_C),
            ("d", NBMB_STAGE2_ICDF_D),
            ("e", NBMB_STAGE2_ICDF_E),
            ("f", NBMB_STAGE2_ICDF_F),
            ("g", NBMB_STAGE2_ICDF_G),
            ("h", NBMB_STAGE2_ICDF_H),
            ("i", WB_STAGE2_ICDF_I),
            ("j", WB_STAGE2_ICDF_J),
            ("k", WB_STAGE2_ICDF_K),
            ("l", WB_STAGE2_ICDF_L),
            ("m", WB_STAGE2_ICDF_M),
            ("n", WB_STAGE2_ICDF_N),
            ("o", WB_STAGE2_ICDF_O),
            ("p", WB_STAGE2_ICDF_P),
        ];
        for (label, icdf) in tables {
            assert_eq!(icdf.len(), 9, "{label} iCDF length");
            assert_eq!(*icdf.last().unwrap(), 0, "{label} iCDF terminator");
            for w in icdf.windows(2) {
                assert!(w[0] >= w[1], "{label} iCDF must be non-increasing");
            }
            // §4.1.3.3 wants the first cell to equal `ft - PDF[0]`; for
            // a 256-sum PDF with PDF[0]==1, that's 255.
            assert_eq!(icdf[0], 255, "{label} first iCDF cell == 256-PDF[0]");
        }
    }

    /// Table 19 extension PDF sums to 256 and matches the transcribed
    /// iCDF.
    #[test]
    fn stage2_extension_pdf_self_check() {
        let pdf = [156, 60, 24, 9, 4, 2, 1];
        let s: u32 = pdf.iter().sum();
        assert_eq!(s, 256);
        assert_eq!(STAGE2_EXTENSION_ICDF.len(), pdf.len());
        let mut acc = 0u32;
        for (k, p) in pdf.iter().enumerate() {
            acc += *p;
            assert_eq!(STAGE2_EXTENSION_ICDF[k] as u32, 256u32.saturating_sub(acc));
        }
    }

    /// Table 17 entries must all be in 0..=7 and the row width must be 10.
    #[test]
    fn nbmb_stage2_select_table_well_formed() {
        for (i1, row) in NBMB_STAGE2_SELECT.iter().enumerate() {
            assert_eq!(row.len(), D_LPC_NB_MB, "Table 17 row {i1} width");
            for (k, letter) in row.iter().enumerate() {
                assert!(*letter <= 7, "Table 17[{i1}][{k}] = {letter} must be 0..=7");
            }
        }
    }

    /// Table 18 entries must all be in 0..=7 and the row width must be 16.
    #[test]
    fn wb_stage2_select_table_well_formed() {
        for (i1, row) in WB_STAGE2_SELECT.iter().enumerate() {
            assert_eq!(row.len(), D_LPC_WB, "Table 18 row {i1} width");
            for (k, letter) in row.iter().enumerate() {
                assert!(*letter <= 7, "Table 18[{i1}][{k}] = {letter} must be 0..=7");
            }
        }
    }

    /// Spot-check the I1=0 row of Table 17: all-`a` (10 cells of 0).
    #[test]
    fn nbmb_table17_i1_0_is_all_a() {
        assert_eq!(NBMB_STAGE2_SELECT[0], [0u8; D_LPC_NB_MB]);
    }

    /// Spot-check the I1=2 row of Table 17: "c b b b b b b b b b".
    #[test]
    fn nbmb_table17_i1_2_spot_check() {
        assert_eq!(NBMB_STAGE2_SELECT[2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    }

    /// Spot-check the I1=6 row (the RFC's row-label-typo "g" row).
    /// Cell contents are `a c c c c c c c c b`.
    #[test]
    fn nbmb_table17_i1_6_typo_row() {
        assert_eq!(NBMB_STAGE2_SELECT[6], [0, 2, 2, 2, 2, 2, 2, 2, 2, 1]);
    }

    /// Spot-check the I1=0 row of Table 18: all-`i` (16 cells of 0).
    #[test]
    fn wb_table18_i1_0_is_all_i() {
        assert_eq!(WB_STAGE2_SELECT[0], [0u8; D_LPC_WB]);
    }

    /// Spot-check the I1=6 row of Table 18: also all-`i` (16 cells of
    /// 0), per the WB table.
    #[test]
    fn wb_table18_i1_6_is_all_i() {
        assert_eq!(WB_STAGE2_SELECT[6], [0u8; D_LPC_WB]);
    }

    /// Spot-check the I1=9 row of Table 18: `k j i i ... i` (2,1,0...0).
    #[test]
    fn wb_table18_i1_9_spot_check() {
        assert_eq!(
            WB_STAGE2_SELECT[9],
            [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    // --- Table 20 / 21 / 22 self-checks --------

    /// Table 20 weight A / B / C / D values must be in 0..=255 (u8) and
    /// match the transcribed row lengths.
    #[test]
    fn pred_weight_tables_self_check() {
        assert_eq!(NBMB_PRED_WEIGHT_A.len(), 9);
        assert_eq!(NBMB_PRED_WEIGHT_B.len(), 9);
        assert_eq!(WB_PRED_WEIGHT_C.len(), 15);
        assert_eq!(WB_PRED_WEIGHT_D.len(), 15);

        // A few known-good cell values from Table 20.
        assert_eq!(NBMB_PRED_WEIGHT_A[0], 179);
        assert_eq!(NBMB_PRED_WEIGHT_B[0], 116);
        assert_eq!(NBMB_PRED_WEIGHT_A[8], 163);
        assert_eq!(NBMB_PRED_WEIGHT_B[8], 92);
        assert_eq!(WB_PRED_WEIGHT_C[0], 175);
        assert_eq!(WB_PRED_WEIGHT_D[0], 68);
        assert_eq!(WB_PRED_WEIGHT_C[14], 182);
        assert_eq!(WB_PRED_WEIGHT_D[14], 155);
    }

    /// Table 21 / 22 must only contain 0 / 1 and match the
    /// `d_LPC - 1` row width.
    #[test]
    fn pred_weight_selection_tables_well_formed() {
        for (i1, row) in NBMB_PRED_WEIGHT_SELECT.iter().enumerate() {
            assert_eq!(row.len(), D_LPC_NB_MB - 1, "Table 21 row {i1} width");
            for sel in row {
                assert!(*sel <= 1, "Table 21 entry must be 0 or 1");
            }
        }
        for (i1, row) in WB_PRED_WEIGHT_SELECT.iter().enumerate() {
            assert_eq!(row.len(), D_LPC_WB - 1, "Table 22 row {i1} width");
            for sel in row {
                assert!(*sel <= 1, "Table 22 entry must be 0 or 1");
            }
        }
    }

    /// Spot-check Table 21 row 0: `A B A A A A A A A` => `0 1 0 0 0 0 0 0 0`.
    #[test]
    fn table21_row0_spot_check() {
        assert_eq!(NBMB_PRED_WEIGHT_SELECT[0], [0, 1, 0, 0, 0, 0, 0, 0, 0]);
    }

    /// Spot-check Table 22 row 0: `C C C C C C C C C C C C C C D` =>
    /// `0 0 ... 0 1`.
    #[test]
    fn table22_row0_spot_check() {
        assert_eq!(
            WB_PRED_WEIGHT_SELECT[0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        );
    }

    /// Spot-check Table 22 row 2: `C C D C C D D D C D D D D C C` =>
    /// `0 0 1 0 0 1 1 1 0 1 1 1 1 0 0`.
    #[test]
    fn table22_row2_spot_check() {
        assert_eq!(
            WB_PRED_WEIGHT_SELECT[2],
            [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
        );
    }

    // --- pred_weight() resolution --------

    #[test]
    fn pred_weight_resolves_a_vs_b() {
        // NB/MB I1=0 row uses A for k=0, B for k=1, then A again.
        let lists = (&NBMB_PRED_WEIGHT_A[..], &NBMB_PRED_WEIGHT_B[..]);
        let select = &NBMB_PRED_WEIGHT_SELECT[0][..];
        assert_eq!(pred_weight(lists, select, 0), 179); // A[0]
        assert_eq!(pred_weight(lists, select, 1), 67); //  B[1]
        assert_eq!(pred_weight(lists, select, 2), 140); // A[2]
        assert_eq!(pred_weight(lists, select, 8), 163); // A[8]
    }

    #[test]
    fn pred_weight_resolves_c_vs_d() {
        // WB I1=0 row uses C for k=0..=13, D only at k=14.
        let lists = (&WB_PRED_WEIGHT_C[..], &WB_PRED_WEIGHT_D[..]);
        let select = &WB_PRED_WEIGHT_SELECT[0][..];
        assert_eq!(pred_weight(lists, select, 0), 175); //  C[0]
        assert_eq!(pred_weight(lists, select, 13), 192); // C[13]
        assert_eq!(pred_weight(lists, select, 14), 155); // D[14]
    }

    // --- End-to-end decode against a hand-crafted RangeDecoder.
    //
    // We can't encode a specific I2 pattern without an encoder, but we
    // CAN check round-trip behaviour: every decoded I2[k] is in [-10, 10],
    // res_Q10[] is finite (i32 arithmetic doesn't overflow on legal
    // inputs), and decode_basic does not latch the corrupt-frame flag.

    fn long_buf() -> [u8; 32] {
        [
            0x55, 0xAA, 0x33, 0xCC, 0x7F, 0x80, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88, 0x99, 0xAA,
        ]
    }

    #[test]
    fn nb_i1_0_decode_basic() {
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Nb, 0).expect("decode");
        assert_eq!(stage2.len(), D_LPC_NB_MB);
        assert_eq!(stage2.i2().len(), D_LPC_NB_MB);
        assert_eq!(stage2.res_q10().len(), D_LPC_NB_MB);
        for (k, idx) in stage2.i2().iter().enumerate() {
            assert!((-10..=10).contains(idx), "i2[{k}]={idx}");
        }
    }

    #[test]
    fn mb_i1_5_decode_basic() {
        // MB shares the NB/MB tables; I1=5 picks codebooks
        // [a, f, d, d, c, c, c, c, b, b].
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Mb, 5).expect("decode");
        assert_eq!(stage2.len(), D_LPC_NB_MB);
        for (k, idx) in stage2.i2().iter().enumerate() {
            assert!((-10..=10).contains(idx), "i2[{k}]={idx}");
        }
    }

    #[test]
    fn wb_i1_0_decode_basic() {
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Wb, 0).expect("decode");
        assert_eq!(stage2.len(), D_LPC_WB);
        for (k, idx) in stage2.i2().iter().enumerate() {
            assert!((-10..=10).contains(idx), "i2[{k}]={idx}");
        }
    }

    #[test]
    fn wb_i1_9_decode_basic() {
        // I1=9 row picks `k j i i ... i` — exercise both `j` and the
        // long tail of `i` cells.
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Wb, 9).expect("decode");
        assert_eq!(stage2.len(), D_LPC_WB);
    }

    #[test]
    fn invalid_i1_is_rejected() {
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let err = LsfStage2::decode(&mut rd, Bandwidth::Nb, 32).unwrap_err();
        assert_eq!(err, Error::MalformedPacket);
    }

    #[test]
    fn swb_bandwidth_is_rejected() {
        // SILK does not operate on SWB; the §4.2.2 hybrid split is
        // upstream. Reject defensively.
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let err = LsfStage2::decode(&mut rd, Bandwidth::Swb, 0).unwrap_err();
        assert_eq!(err, Error::MalformedPacket);
    }

    #[test]
    fn fb_bandwidth_is_rejected() {
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let err = LsfStage2::decode(&mut rd, Bandwidth::Fb, 0).unwrap_err();
        assert_eq!(err, Error::MalformedPacket);
    }

    /// `res_Q10[k]` for the all-zero `I2[]` case must be all zeros: the
    /// `q_contrib` is zero and the backwards recursion only propagates
    /// zeros. We synthesize this via the all-zero codepath by passing a
    /// buffer that decodes `raw == 4` for every coefficient (i.e.
    /// `idx == 0` after the -4 subtraction). We can't directly force
    /// that without an encoder, so instead test the algorithmic
    /// behaviour by re-deriving `res_Q10[]` via the same formula
    /// against the decoded `I2[]`.
    #[test]
    fn res_q10_reproduces_formula_nbmb() {
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Nb, 7).expect("decode");

        let d_lpc = D_LPC_NB_MB;
        let qstep = QSTEP_NB_MB_Q16;
        let select_row = &NBMB_PRED_WEIGHT_SELECT[7][..];
        let lists = (&NBMB_PRED_WEIGHT_A[..], &NBMB_PRED_WEIGHT_B[..]);

        let mut expect = [0i32; D_LPC_MAX];
        for k in (0..d_lpc).rev() {
            let i2k = stage2.i2()[k] as i32;
            let sign = match i2k.cmp(&0) {
                core::cmp::Ordering::Less => -1,
                core::cmp::Ordering::Greater => 1,
                core::cmp::Ordering::Equal => 0,
            };
            let q_contrib = (((i2k << 10) - sign * 102) * qstep) >> 16;
            let pred_contrib = if k + 1 < d_lpc {
                let p = pred_weight(lists, select_row, k);
                (expect[k + 1] * p) >> 8
            } else {
                0
            };
            expect[k] = pred_contrib + q_contrib;
        }
        assert_eq!(stage2.res_q10(), &expect[..d_lpc]);
    }

    #[test]
    fn res_q10_reproduces_formula_wb() {
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let stage2 = LsfStage2::decode(&mut rd, Bandwidth::Wb, 13).expect("decode");

        let d_lpc = D_LPC_WB;
        let qstep = QSTEP_WB_Q16;
        let select_row = &WB_PRED_WEIGHT_SELECT[13][..];
        let lists = (&WB_PRED_WEIGHT_C[..], &WB_PRED_WEIGHT_D[..]);

        let mut expect = [0i32; D_LPC_MAX];
        for k in (0..d_lpc).rev() {
            let i2k = stage2.i2()[k] as i32;
            let sign = match i2k.cmp(&0) {
                core::cmp::Ordering::Less => -1,
                core::cmp::Ordering::Greater => 1,
                core::cmp::Ordering::Equal => 0,
            };
            let q_contrib = (((i2k << 10) - sign * 102) * qstep) >> 16;
            let pred_contrib = if k + 1 < d_lpc {
                let p = pred_weight(lists, select_row, k);
                (expect[k + 1] * p) >> 8
            } else {
                0
            };
            expect[k] = pred_contrib + q_contrib;
        }
        assert_eq!(stage2.res_q10(), &expect[..d_lpc]);
    }

    /// Sweep all 32 I1 values across a few buffers for both NB and WB;
    /// every successful decode must yield `I2[k] ∈ [-10, 10]` for every
    /// coefficient and the corrupt-frame flag must not latch on
    /// non-corrupt input.
    #[test]
    fn sweep_all_i1_nb_wb_decodes() {
        let buf = long_buf();
        for i1 in 0..32u8 {
            for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
                let mut rd = RangeDecoder::new(&buf);
                let stage2 = LsfStage2::decode(&mut rd, bw, i1).expect("decode");
                for (k, idx) in stage2.i2().iter().enumerate() {
                    assert!(
                        (-10..=10).contains(idx),
                        "bw={bw:?} i1={i1} k={k} idx={idx}"
                    );
                }
                let expect_len = match bw {
                    Bandwidth::Wb => D_LPC_WB,
                    _ => D_LPC_NB_MB,
                };
                assert_eq!(stage2.len(), expect_len);
            }
        }
    }

    /// The cumulative bit count consumed by a stage-2 decode must be
    /// monotone non-decreasing with respect to the d_LPC growth (each
    /// coefficient consumes ≥ 0 fractional bits per §4.1.6).
    #[test]
    fn tell_monotone_through_stage2_decode() {
        let buf = long_buf();
        let mut rd = RangeDecoder::new(&buf);
        let tell0 = rd.tell();
        let _ = LsfStage2::decode(&mut rd, Bandwidth::Wb, 4).expect("decode");
        let tell1 = rd.tell();
        assert!(
            tell1 >= tell0,
            "tell must not decrease across stage-2 decode: {tell0} -> {tell1}"
        );
        // 16 stage-2 reads plus (occasionally) some extension reads
        // must consume some bits.
        assert!(tell1 - tell0 > 0, "stage-2 decode should consume bits");
    }

    // ----- §4.2.7.5.2 encode-side mirror ----------------------------

    /// A tiny deterministic LCG for the encode/decode roundtrip sweeps.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n
        }
    }

    /// encode → decode roundtrip over random index vectors across all
    /// bandwidths and stage-1 selections, including the ±4..±10
    /// extension range: the decoder must reconstruct exactly the
    /// encoder's predicted `LsfStage2` (both `i2` and `res_Q10`).
    #[test]
    fn stage2_encode_decode_roundtrip_random() {
        use crate::range_encoder::RangeEncoder;
        let mut rng = Lcg(0x57A6_E2E2);
        for _ in 0..600 {
            let bandwidth = match rng.below(3) {
                0 => Bandwidth::Nb,
                1 => Bandwidth::Mb,
                _ => Bandwidth::Wb,
            };
            let d_lpc = if bandwidth == Bandwidth::Wb {
                D_LPC_WB
            } else {
                D_LPC_NB_MB
            };
            let i1 = rng.below(32) as u8;
            let i2: Vec<i8> = (0..d_lpc).map(|_| rng.below(21) as i8 - 10).collect();

            let mut re = RangeEncoder::new();
            let predicted = LsfStage2::encode(&mut re, bandwidth, i1, &i2).expect("encode");
            assert_eq!(predicted.i2(), &i2[..]);
            let bytes = re.finish();

            let mut rd = RangeDecoder::new(&bytes);
            let decoded = LsfStage2::decode(&mut rd, bandwidth, i1).expect("decode");
            assert!(!rd.has_error());
            assert_eq!(decoded, predicted, "bw={bandwidth:?} i1={i1} i2={i2:?}");
        }
    }

    /// The encode path rejects wrong lengths, out-of-range indices,
    /// out-of-range stage-1, and SWB/FB bandwidths.
    #[test]
    fn stage2_encode_rejects_bad_inputs() {
        use crate::range_encoder::RangeEncoder;
        let ok10 = [0i8; 10];
        // Wrong length for the bandwidth.
        let mut re = RangeEncoder::new();
        assert!(LsfStage2::encode(&mut re, Bandwidth::Wb, 0, &ok10).is_err());
        // Out-of-range index.
        let mut re = RangeEncoder::new();
        let mut bad = [0i8; 10];
        bad[3] = 11;
        assert!(LsfStage2::encode(&mut re, Bandwidth::Nb, 0, &bad).is_err());
        // Out-of-range stage-1 index.
        let mut re = RangeEncoder::new();
        assert!(LsfStage2::encode(&mut re, Bandwidth::Nb, 32, &ok10).is_err());
        // SWB / FB rejected.
        let mut re = RangeEncoder::new();
        assert!(LsfStage2::encode(&mut re, Bandwidth::Swb, 0, &ok10).is_err());
    }
}
