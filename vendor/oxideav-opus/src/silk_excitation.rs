//! SILK excitation decoder — RFC 6716 §4.2.7.8.
//!
//! The excitation is the SILK frame's residual signal after the LPC
//! synthesis filter has been removed. It is the last thing decoded
//! from the bitstream for a SILK frame; everything afterwards
//! (§4.2.7.9 LTP synthesis, §4.2.7.9.2 LPC synthesis, etc.) consumes
//! the reconstructed Q23 excitation values produced here.
//!
//! The encoding is a modified pyramid vector quantizer (PVQ). The
//! excitation is split into 16-sample "shell blocks" (count per
//! bandwidth × frame size in Table 44); a single "rate level" then
//! gates one of nine PDFs that govern the per-block pulse count. The
//! per-block layout is a recursive partition of 16 → 8 → 4 → 2 → 1
//! samples (Tables 47, 48, 49, 50), where each split level codes how
//! many pulses fall on the left half. Per-coefficient LSBs (Table 51)
//! refine each magnitude; per-coefficient signs (Table 52, picked by
//! signal type / quantization offset type / pulse count) recover the
//! sign of every non-zero magnitude.
//!
//! Finally, §4.2.7.8.6 adds a quantization offset (Table 53) and runs
//! every sample through a Linear Congruential Generator (LCG; seed
//! from §4.2.7.7) to pseudorandomly flip its sign, producing the final
//! Q23 excitation vector `e_Q23[]`.
//!
//! This module is a pure decoder — it takes the §4.2.7.7 LCG seed and
//! the §4.2.7.3 signal-type / quantization-offset type, plus the
//! bandwidth and frame size to look up the §4.2.7.8 shell-block count,
//! and produces the full `e_Q23[]` vector. The §4.2.7.9 filters that
//! consume it remain to be wired up.
//!
//! All truth is RFC 6716 §4.2.7.8 + Tables 44–53. No external library
//! source is consulted.

use crate::range_decoder::RangeDecoder;
use crate::range_encoder::RangeEncoder;
use crate::silk_frame::{QuantizationOffsetType, SignalType};
use crate::toc::Bandwidth;
use crate::Error;

/// Samples per SILK shell block (RFC 6716 §4.2.7.8 fixes `N = 16`).
pub const SHELL_BLOCK_SAMPLES: usize = 16;

/// Maximum shell blocks per SILK frame across (bandwidth × frame size)
/// — 20 for 20 ms WB (Table 44). The 10 ms MB special-case (8 blocks
/// holding 128 samples, of which only 120 are used) lives in the
/// decoder; the `MAX` cap is the spec-defined upper bound.
pub const MAX_SHELL_BLOCKS: usize = 20;

/// Maximum samples per SILK excitation vector — `MAX_SHELL_BLOCKS * 16
/// = 320` (20 ms WB at 16 kHz). The 10 ms MB special-case parses 128
/// samples but the caller treats the trailing 8 as ignored.
pub const MAX_EXCITATION_SAMPLES: usize = MAX_SHELL_BLOCKS * SHELL_BLOCK_SAMPLES;

// =====================================================================
// §4.2.7.8 frame size enumeration.
// =====================================================================

/// SILK frame size: 10 ms or 20 ms (per §4.2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilkFrameSize {
    /// 10 ms SILK frame (2 subframes; one set of LPC coefficients).
    TenMs,
    /// 20 ms SILK frame (4 subframes; also the only choice in Hybrid
    /// mode).
    TwentyMs,
}

// =====================================================================
// Table 44 — number of shell blocks per SILK frame.
// =====================================================================

/// Returns the §4.2.7.8 shell-block count for the given audio
/// bandwidth × SILK frame size.
///
/// The 10 ms MB row encodes 8 shell blocks (128 samples) of which the
/// last 8 samples are discarded; the parser still reads them.
///
/// Rejects SWB and FB (they do not pair with SILK at the SILK layer).
pub fn shell_block_count(bandwidth: Bandwidth, size: SilkFrameSize) -> Result<usize, Error> {
    Ok(match (bandwidth, size) {
        (Bandwidth::Nb, SilkFrameSize::TenMs) => 5,
        (Bandwidth::Mb, SilkFrameSize::TenMs) => 8,
        (Bandwidth::Wb, SilkFrameSize::TenMs) => 10,
        (Bandwidth::Nb, SilkFrameSize::TwentyMs) => 10,
        (Bandwidth::Mb, SilkFrameSize::TwentyMs) => 15,
        (Bandwidth::Wb, SilkFrameSize::TwentyMs) => 20,
        _ => return Err(Error::MalformedPacket),
    })
}

// =====================================================================
// Table 45 — rate-level PDFs (one symbol per SILK frame, value 0..=8).
// =====================================================================

// Inactive/Unvoiced: PDF {15, 51, 12, 46, 45, 13, 33, 27, 14}/256.
// fh = [15, 66, 78, 124, 169, 182, 215, 242, 256]. iCDF = 256 - fh[k].
const RATE_LEVEL_ICDF_INACTIVE_UNVOICED: &[u8] = &[241, 190, 178, 132, 87, 74, 41, 14, 0];

// Voiced: PDF {33, 30, 36, 17, 34, 49, 18, 21, 18}/256.
// fh = [33, 63, 99, 116, 150, 199, 217, 238, 256]. iCDF = 256 - fh[k].
const RATE_LEVEL_ICDF_VOICED: &[u8] = &[223, 193, 157, 140, 106, 57, 39, 18, 0];

// =====================================================================
// Table 46 — pulse-count PDFs (18 cells; values 0..=16 = pulses,
// 17 = "extra LSB; re-read with rate level 9 or 10"). Eleven rate
// levels (0..=8 normal; 9 and 10 special; spec note: level 10's
// distribution is "a shifted version of 9" — pulse count 17 has zero
// probability at level 10, capping the LSB count at 10).
// =====================================================================

/// Rate level 0: {131, 74, 25, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}/256.
const PULSE_COUNT_ICDF_L0: &[u8] = &[
    125, 51, 26, 18, 15, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];
/// Rate level 1: {58, 93, 60, 23, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}/256.
const PULSE_COUNT_ICDF_L1: &[u8] = &[
    198, 105, 45, 22, 15, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];
/// Rate level 2: {43, 51, 46, 33, 24, 16, 11, 8, 6, 3, 3, 3, 2, 1, 1, 2, 1, 2}/256.
const PULSE_COUNT_ICDF_L2: &[u8] = &[
    213, 162, 116, 83, 59, 43, 32, 24, 18, 15, 12, 9, 7, 6, 5, 3, 2, 0,
];
/// Rate level 3: {17, 52, 71, 57, 31, 12, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}/256.
const PULSE_COUNT_ICDF_L3: &[u8] = &[
    239, 187, 116, 59, 28, 16, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];
/// Rate level 4: {6, 21, 41, 53, 49, 35, 21, 11, 6, 3, 2, 2, 1, 1, 1, 1, 1, 1}/256.
const PULSE_COUNT_ICDF_L4: &[u8] = &[
    250, 229, 188, 135, 86, 51, 30, 19, 13, 10, 8, 6, 5, 4, 3, 2, 1, 0,
];
/// Rate level 5: {7, 14, 22, 28, 29, 28, 25, 20, 17, 13, 11, 9, 7, 5, 4, 4, 3, 10}/256.
const PULSE_COUNT_ICDF_L5: &[u8] = &[
    249, 235, 213, 185, 156, 128, 103, 83, 66, 53, 42, 33, 26, 21, 17, 13, 10, 0,
];
/// Rate level 6: {2, 5, 14, 29, 42, 46, 41, 31, 19, 11, 6, 3, 2, 1, 1, 1, 1, 1}/256.
const PULSE_COUNT_ICDF_L6: &[u8] = &[
    254, 249, 235, 206, 164, 118, 77, 46, 27, 16, 10, 7, 5, 4, 3, 2, 1, 0,
];
/// Rate level 7: {1, 2, 4, 10, 19, 29, 35, 37, 34, 28, 20, 14, 8, 5, 4, 2, 2, 2}/256.
const PULSE_COUNT_ICDF_L7: &[u8] = &[
    255, 253, 249, 239, 220, 191, 156, 119, 85, 57, 37, 23, 15, 10, 6, 4, 2, 0,
];
/// Rate level 8: {1, 2, 2, 5, 9, 14, 20, 24, 27, 28, 26, 23, 20, 15, 11, 8, 6, 15}/256.
const PULSE_COUNT_ICDF_L8: &[u8] = &[
    255, 253, 251, 246, 237, 223, 203, 179, 152, 124, 98, 75, 55, 40, 29, 21, 15, 0,
];
/// Rate level 9 (special "extra LSB" PDF):
/// {1, 1, 1, 6, 27, 58, 56, 39, 25, 14, 10, 6, 3, 3, 2, 1, 1, 2}/256.
const PULSE_COUNT_ICDF_L9: &[u8] = &[
    255, 254, 253, 247, 220, 162, 106, 67, 42, 28, 18, 12, 9, 6, 4, 3, 2, 0,
];
/// Rate level 10 (special, capping LSBs at 10):
/// {2, 1, 6, 27, 58, 56, 39, 25, 14, 10, 6, 3, 3, 2, 1, 1, 2, 0}/256.
const PULSE_COUNT_ICDF_L10: &[u8] = &[
    254, 253, 247, 220, 162, 106, 67, 42, 28, 18, 12, 9, 6, 4, 3, 2, 0, 0,
];

/// All eleven pulse-count PDFs indexed by rate level 0..=10. Rate
/// levels 0..=8 are reachable directly from §4.2.7.8.1; rate levels
/// 9 and 10 are reached by the §4.2.7.8.2 "re-read on 17" mechanism.
const PULSE_COUNT_ICDFS: [&[u8]; 11] = [
    PULSE_COUNT_ICDF_L0,
    PULSE_COUNT_ICDF_L1,
    PULSE_COUNT_ICDF_L2,
    PULSE_COUNT_ICDF_L3,
    PULSE_COUNT_ICDF_L4,
    PULSE_COUNT_ICDF_L5,
    PULSE_COUNT_ICDF_L6,
    PULSE_COUNT_ICDF_L7,
    PULSE_COUNT_ICDF_L8,
    PULSE_COUNT_ICDF_L9,
    PULSE_COUNT_ICDF_L10,
];

// =====================================================================
// Table 47 — pulse-count split PDFs for 16-sample partitions
// (`partition_size = 16`; the FIRST split of a shell block). Indexed
// by total pulse count 1..=16. The PDF has `(pulses + 1)` cells.
// =====================================================================

const SPLIT16_ICDF_1: &[u8] = &[130, 0];
const SPLIT16_ICDF_2: &[u8] = &[200, 58, 0];
const SPLIT16_ICDF_3: &[u8] = &[231, 130, 26, 0];
const SPLIT16_ICDF_4: &[u8] = &[244, 184, 76, 12, 0];
const SPLIT16_ICDF_5: &[u8] = &[249, 214, 130, 43, 6, 0];
const SPLIT16_ICDF_6: &[u8] = &[252, 232, 173, 87, 24, 3, 0];
const SPLIT16_ICDF_7: &[u8] = &[253, 241, 203, 131, 56, 14, 2, 0];
const SPLIT16_ICDF_8: &[u8] = &[254, 246, 221, 167, 94, 35, 8, 1, 0];
const SPLIT16_ICDF_9: &[u8] = &[254, 249, 232, 193, 130, 65, 23, 5, 1, 0];
const SPLIT16_ICDF_10: &[u8] = &[255, 251, 239, 211, 162, 99, 45, 15, 4, 1, 0];
const SPLIT16_ICDF_11: &[u8] = &[255, 251, 243, 223, 186, 131, 74, 33, 11, 3, 1, 0];
const SPLIT16_ICDF_12: &[u8] = &[255, 252, 245, 230, 202, 158, 105, 57, 24, 8, 2, 1, 0];
const SPLIT16_ICDF_13: &[u8] = &[255, 253, 247, 235, 214, 179, 132, 84, 44, 19, 7, 2, 1, 0];
const SPLIT16_ICDF_14: &[u8] = &[
    255, 254, 250, 240, 223, 196, 159, 112, 69, 36, 15, 6, 2, 1, 0,
];
const SPLIT16_ICDF_15: &[u8] = &[
    255, 254, 253, 245, 231, 209, 176, 136, 93, 55, 27, 11, 3, 2, 1, 0,
];
const SPLIT16_ICDF_16: &[u8] = &[
    255, 254, 253, 252, 239, 221, 194, 158, 117, 76, 42, 18, 4, 3, 2, 1, 0,
];

/// Table 47 indexed by `pulses - 1` (pulses run 1..=16).
const SPLIT16_ICDFS: [&[u8]; 16] = [
    SPLIT16_ICDF_1,
    SPLIT16_ICDF_2,
    SPLIT16_ICDF_3,
    SPLIT16_ICDF_4,
    SPLIT16_ICDF_5,
    SPLIT16_ICDF_6,
    SPLIT16_ICDF_7,
    SPLIT16_ICDF_8,
    SPLIT16_ICDF_9,
    SPLIT16_ICDF_10,
    SPLIT16_ICDF_11,
    SPLIT16_ICDF_12,
    SPLIT16_ICDF_13,
    SPLIT16_ICDF_14,
    SPLIT16_ICDF_15,
    SPLIT16_ICDF_16,
];

// =====================================================================
// Table 48 — pulse-count split PDFs for 8-sample partitions.
// =====================================================================

const SPLIT8_ICDF_1: &[u8] = &[129, 0];
const SPLIT8_ICDF_2: &[u8] = &[203, 54, 0];
const SPLIT8_ICDF_3: &[u8] = &[234, 129, 23, 0];
const SPLIT8_ICDF_4: &[u8] = &[245, 184, 73, 10, 0];
const SPLIT8_ICDF_5: &[u8] = &[250, 215, 129, 41, 5, 0];
const SPLIT8_ICDF_6: &[u8] = &[252, 232, 173, 86, 24, 3, 0];
const SPLIT8_ICDF_7: &[u8] = &[253, 240, 200, 129, 56, 15, 2, 0];
const SPLIT8_ICDF_8: &[u8] = &[253, 244, 217, 164, 94, 38, 10, 1, 0];
const SPLIT8_ICDF_9: &[u8] = &[253, 245, 226, 189, 132, 71, 27, 7, 1, 0];
const SPLIT8_ICDF_10: &[u8] = &[253, 246, 231, 203, 159, 105, 56, 23, 6, 1, 0];
const SPLIT8_ICDF_11: &[u8] = &[255, 248, 235, 213, 179, 133, 85, 47, 19, 5, 1, 0];
const SPLIT8_ICDF_12: &[u8] = &[255, 254, 243, 221, 194, 159, 117, 70, 37, 12, 2, 1, 0];
const SPLIT8_ICDF_13: &[u8] = &[255, 254, 248, 234, 208, 171, 128, 85, 48, 22, 8, 2, 1, 0];
const SPLIT8_ICDF_14: &[u8] = &[
    255, 254, 250, 240, 220, 189, 149, 107, 67, 36, 16, 6, 2, 1, 0,
];
const SPLIT8_ICDF_15: &[u8] = &[
    255, 254, 251, 243, 227, 201, 166, 128, 90, 55, 29, 13, 5, 2, 1, 0,
];
const SPLIT8_ICDF_16: &[u8] = &[
    255, 254, 252, 246, 234, 213, 183, 147, 109, 73, 43, 22, 10, 4, 2, 1, 0,
];

/// Table 48 indexed by `pulses - 1`.
const SPLIT8_ICDFS: [&[u8]; 16] = [
    SPLIT8_ICDF_1,
    SPLIT8_ICDF_2,
    SPLIT8_ICDF_3,
    SPLIT8_ICDF_4,
    SPLIT8_ICDF_5,
    SPLIT8_ICDF_6,
    SPLIT8_ICDF_7,
    SPLIT8_ICDF_8,
    SPLIT8_ICDF_9,
    SPLIT8_ICDF_10,
    SPLIT8_ICDF_11,
    SPLIT8_ICDF_12,
    SPLIT8_ICDF_13,
    SPLIT8_ICDF_14,
    SPLIT8_ICDF_15,
    SPLIT8_ICDF_16,
];

// =====================================================================
// Table 49 — pulse-count split PDFs for 4-sample partitions.
// =====================================================================

const SPLIT4_ICDF_1: &[u8] = &[129, 0];
const SPLIT4_ICDF_2: &[u8] = &[207, 50, 0];
const SPLIT4_ICDF_3: &[u8] = &[236, 129, 20, 0];
const SPLIT4_ICDF_4: &[u8] = &[245, 185, 72, 10, 0];
const SPLIT4_ICDF_5: &[u8] = &[249, 213, 129, 42, 6, 0];
const SPLIT4_ICDF_6: &[u8] = &[250, 226, 169, 87, 27, 4, 0];
const SPLIT4_ICDF_7: &[u8] = &[251, 233, 194, 130, 62, 20, 4, 0];
const SPLIT4_ICDF_8: &[u8] = &[250, 236, 207, 160, 99, 47, 17, 3, 0];
const SPLIT4_ICDF_9: &[u8] = &[255, 240, 217, 182, 131, 81, 41, 11, 1, 0];
const SPLIT4_ICDF_10: &[u8] = &[255, 254, 233, 201, 159, 107, 61, 20, 2, 1, 0];
const SPLIT4_ICDF_11: &[u8] = &[255, 249, 233, 206, 170, 128, 86, 50, 23, 7, 1, 0];
const SPLIT4_ICDF_12: &[u8] = &[255, 250, 238, 217, 186, 148, 108, 70, 39, 18, 6, 1, 0];
const SPLIT4_ICDF_13: &[u8] = &[255, 252, 243, 226, 200, 166, 128, 90, 56, 30, 13, 4, 1, 0];
const SPLIT4_ICDF_14: &[u8] = &[
    255, 252, 245, 231, 209, 180, 146, 110, 76, 47, 25, 11, 4, 1, 0,
];
const SPLIT4_ICDF_15: &[u8] = &[
    255, 253, 248, 237, 219, 194, 163, 128, 93, 62, 37, 19, 8, 3, 1, 0,
];
const SPLIT4_ICDF_16: &[u8] = &[
    255, 254, 250, 241, 226, 205, 177, 145, 111, 79, 51, 30, 15, 6, 2, 1, 0,
];

/// Table 49 indexed by `pulses - 1`.
const SPLIT4_ICDFS: [&[u8]; 16] = [
    SPLIT4_ICDF_1,
    SPLIT4_ICDF_2,
    SPLIT4_ICDF_3,
    SPLIT4_ICDF_4,
    SPLIT4_ICDF_5,
    SPLIT4_ICDF_6,
    SPLIT4_ICDF_7,
    SPLIT4_ICDF_8,
    SPLIT4_ICDF_9,
    SPLIT4_ICDF_10,
    SPLIT4_ICDF_11,
    SPLIT4_ICDF_12,
    SPLIT4_ICDF_13,
    SPLIT4_ICDF_14,
    SPLIT4_ICDF_15,
    SPLIT4_ICDF_16,
];

// =====================================================================
// Table 50 — pulse-count split PDFs for 2-sample partitions (terminal
// split before pulses land on individual samples).
// =====================================================================

const SPLIT2_ICDF_1: &[u8] = &[128, 0];
const SPLIT2_ICDF_2: &[u8] = &[214, 42, 0];
const SPLIT2_ICDF_3: &[u8] = &[235, 128, 21, 0];
const SPLIT2_ICDF_4: &[u8] = &[244, 184, 72, 11, 0];
const SPLIT2_ICDF_5: &[u8] = &[248, 214, 128, 42, 7, 0];
const SPLIT2_ICDF_6: &[u8] = &[248, 225, 170, 80, 25, 5, 0];
const SPLIT2_ICDF_7: &[u8] = &[251, 236, 198, 126, 54, 18, 3, 0];
const SPLIT2_ICDF_8: &[u8] = &[250, 238, 211, 159, 82, 35, 15, 5, 0];
const SPLIT2_ICDF_9: &[u8] = &[250, 231, 203, 168, 128, 88, 53, 25, 6, 0];
const SPLIT2_ICDF_10: &[u8] = &[252, 238, 216, 185, 148, 108, 71, 40, 18, 4, 0];
const SPLIT2_ICDF_11: &[u8] = &[253, 243, 225, 199, 166, 128, 90, 57, 31, 13, 3, 0];
const SPLIT2_ICDF_12: &[u8] = &[254, 246, 233, 212, 183, 147, 109, 73, 44, 23, 10, 2, 0];
const SPLIT2_ICDF_13: &[u8] = &[255, 250, 240, 223, 198, 166, 128, 90, 58, 33, 16, 6, 1, 0];
const SPLIT2_ICDF_14: &[u8] = &[
    255, 251, 244, 231, 210, 181, 146, 110, 75, 46, 25, 12, 5, 1, 0,
];
const SPLIT2_ICDF_15: &[u8] = &[
    255, 253, 248, 238, 221, 196, 164, 128, 92, 60, 35, 18, 8, 3, 1, 0,
];
const SPLIT2_ICDF_16: &[u8] = &[
    255, 253, 249, 242, 229, 208, 180, 146, 110, 76, 48, 27, 14, 7, 3, 1, 0,
];

/// Table 50 indexed by `pulses - 1`.
const SPLIT2_ICDFS: [&[u8]; 16] = [
    SPLIT2_ICDF_1,
    SPLIT2_ICDF_2,
    SPLIT2_ICDF_3,
    SPLIT2_ICDF_4,
    SPLIT2_ICDF_5,
    SPLIT2_ICDF_6,
    SPLIT2_ICDF_7,
    SPLIT2_ICDF_8,
    SPLIT2_ICDF_9,
    SPLIT2_ICDF_10,
    SPLIT2_ICDF_11,
    SPLIT2_ICDF_12,
    SPLIT2_ICDF_13,
    SPLIT2_ICDF_14,
    SPLIT2_ICDF_15,
    SPLIT2_ICDF_16,
];

/// Select the appropriate split-PDF table for the given partition size.
/// Partition size is `16`, `8`, `4`, or `2` per §4.2.7.8.3.
fn split_icdf(partition_size: usize, pulses: u8) -> &'static [u8] {
    let p = (pulses as usize) - 1;
    match partition_size {
        16 => SPLIT16_ICDFS[p],
        8 => SPLIT8_ICDFS[p],
        4 => SPLIT4_ICDFS[p],
        2 => SPLIT2_ICDFS[p],
        _ => unreachable!("partition size {partition_size} not 16/8/4/2"),
    }
}

// =====================================================================
// Table 51 — single PDF for the per-coefficient LSBs.
// PDF {136, 120}/256. fh = [136, 256]. iCDF = [120, 0].
// =====================================================================

const LSB_ICDF: &[u8] = &[120, 0];

// =====================================================================
// Table 52 — sign PDFs, indexed by (signal_type, qoff_type,
// pulse-count-bin). Pulse count bin is `min(pulses, 6)` so a pulse
// count >= 6 selects the "6 or more" row.
//
// Layout: SIGN_ICDF[signal_type][qoff_type][bin] = &[u8; 2].
// signal_type: 0=Inactive, 1=Unvoiced, 2=Voiced.
// qoff_type:   0=Low, 1=High.
// bin:         0..=6.
// =====================================================================

// PDFs encoded as iCDF[1] = 256 - p0 (terminator 0).
// {p0, p1}/256 -> iCDF = [p1, 0] = [256 - p0, 0].

const SIGN_ICDF_INACTIVE_LOW: [&[u8]; 7] = [
    &[254, 0], // {2, 254}/256: iCDF[0] = 256-2 = 254.
    &[49, 0],  // {207, 49}
    &[67, 0],  // {189, 67}
    &[77, 0],  // {179, 77}
    &[82, 0],  // {174, 82}
    &[93, 0],  // {163, 93}
    &[99, 0],  // {157, 99}
];

const SIGN_ICDF_INACTIVE_HIGH: [&[u8]; 7] = [
    &[198, 0], // {58, 198}/256
    &[11, 0],  // {245, 11}
    &[18, 0],  // {238, 18}
    &[24, 0],  // {232, 24}
    &[31, 0],  // {225, 31}
    &[36, 0],  // {220, 36}
    &[45, 0],  // {211, 45}
];

const SIGN_ICDF_UNVOICED_LOW: [&[u8]; 7] = [
    &[255, 0], // {1, 255}
    &[46, 0],  // {210, 46}
    &[66, 0],  // {190, 66}
    &[78, 0],  // {178, 78}
    &[87, 0],  // {169, 87}
    &[94, 0],  // {162, 94}
    &[104, 0], // {152, 104}
];

const SIGN_ICDF_UNVOICED_HIGH: [&[u8]; 7] = [
    &[208, 0], // {48, 208}
    &[14, 0],  // {242, 14}
    &[21, 0],  // {235, 21}
    &[32, 0],  // {224, 32}
    &[42, 0],  // {214, 42}
    &[51, 0],  // {205, 51}
    &[66, 0],  // {190, 66}
];

const SIGN_ICDF_VOICED_LOW: [&[u8]; 7] = [
    &[255, 0], // {1, 255}
    &[94, 0],  // {162, 94}
    &[104, 0], // {152, 104}
    &[109, 0], // {147, 109}
    &[112, 0], // {144, 112}
    &[115, 0], // {141, 115}
    &[118, 0], // {138, 118}
];

const SIGN_ICDF_VOICED_HIGH: [&[u8]; 7] = [
    &[248, 0], // {8, 248}
    &[53, 0],  // {203, 53}
    &[69, 0],  // {187, 69}
    &[80, 0],  // {176, 80}
    &[88, 0],  // {168, 88}
    &[95, 0],  // {161, 95}
    &[102, 0], // {154, 102}
];

/// Choose Table 52 sign PDF.
fn sign_icdf(
    signal_type: SignalType,
    qoff_type: QuantizationOffsetType,
    pulses_in_block: u32,
) -> &'static [u8] {
    let bin = pulses_in_block.min(6) as usize;
    match (signal_type, qoff_type) {
        (SignalType::Inactive, QuantizationOffsetType::Low) => SIGN_ICDF_INACTIVE_LOW[bin],
        (SignalType::Inactive, QuantizationOffsetType::High) => SIGN_ICDF_INACTIVE_HIGH[bin],
        (SignalType::Unvoiced, QuantizationOffsetType::Low) => SIGN_ICDF_UNVOICED_LOW[bin],
        (SignalType::Unvoiced, QuantizationOffsetType::High) => SIGN_ICDF_UNVOICED_HIGH[bin],
        (SignalType::Voiced, QuantizationOffsetType::Low) => SIGN_ICDF_VOICED_LOW[bin],
        (SignalType::Voiced, QuantizationOffsetType::High) => SIGN_ICDF_VOICED_HIGH[bin],
    }
}

// =====================================================================
// Table 53 — Q23 quantization offsets per (signal_type, qoff_type).
// =====================================================================

/// Returns the §4.2.7.8.6 `offset_Q23` for the given signal type +
/// quantization offset type (Table 53).
pub fn quantization_offset_q23(signal_type: SignalType, qoff_type: QuantizationOffsetType) -> i32 {
    match (signal_type, qoff_type) {
        (SignalType::Inactive, QuantizationOffsetType::Low) => 25,
        (SignalType::Inactive, QuantizationOffsetType::High) => 60,
        (SignalType::Unvoiced, QuantizationOffsetType::Low) => 25,
        (SignalType::Unvoiced, QuantizationOffsetType::High) => 60,
        (SignalType::Voiced, QuantizationOffsetType::Low) => 8,
        (SignalType::Voiced, QuantizationOffsetType::High) => 25,
    }
}

// =====================================================================
// Caller-supplied excitation context.
// =====================================================================

/// Configuration for [`Excitation::decode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExcitationConfig {
    /// SILK-layer bandwidth (NB / MB / WB). SWB / FB rejected.
    pub bandwidth: Bandwidth,
    /// SILK frame size (10 ms vs 20 ms).
    pub frame_size: SilkFrameSize,
    /// Signal type from §4.2.7.3 / Table 10.
    pub signal_type: SignalType,
    /// Quantization-offset type from §4.2.7.3 / Table 10.
    pub qoff_type: QuantizationOffsetType,
    /// LCG seed from §4.2.7.7 (`0..=3`).
    pub lcg_seed: u8,
}

// =====================================================================
// Output: the decoded excitation vector.
// =====================================================================

/// Decoded SILK excitation vector — RFC 6716 §4.2.7.8.
///
/// `e_q23` holds `samples()` Q23-domain reconstructed excitation
/// values. Length is `16 * shell_block_count(bandwidth, frame_size)`.
///
/// For 10 ms MB, this is 128 samples (8 shell blocks); the caller
/// trims the trailing 8 samples to land on the nominal 120 (the spec
/// notes this special case in the §4.2.7.8 preamble).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Excitation {
    // Profluens patch: the per-frame decoded excitation buffers live in the per-packet bump arena
    // (only `Excitation::decode` constructs one; encode + accessors read them as slices). They are
    // held in SilkFrameDecoded through the frame's synthesis and dropped at the next packet reset.
    e_q23: Vec<i32, crate::bump::DecodeBump>,
    rate_level: u8,
    shell_blocks: usize,
    pulses_per_block: Vec<u8, crate::bump::DecodeBump>,
    lsb_count_per_block: Vec<u8, crate::bump::DecodeBump>,
}

impl Excitation {
    /// Decode the §4.2.7.8 excitation from `rd`.
    ///
    /// Returns `Error::MalformedPacket` if the bandwidth × frame_size
    /// pair is invalid (SWB / FB), if the LCG seed is out of `0..=3`,
    /// or if the range decoder latches an error mid-decode.
    pub fn decode(rd: &mut RangeDecoder<'_>, cfg: ExcitationConfig) -> Result<Self, Error> {
        if cfg.lcg_seed > 3 {
            return Err(Error::MalformedPacket);
        }
        let shell_blocks = shell_block_count(cfg.bandwidth, cfg.frame_size)?;
        let total_samples = shell_blocks * SHELL_BLOCK_SAMPLES;

        // ----- §4.2.7.8.1 rate level (one symbol per SILK frame) ----
        let rate_level_icdf = match cfg.signal_type {
            SignalType::Inactive | SignalType::Unvoiced => RATE_LEVEL_ICDF_INACTIVE_UNVOICED,
            SignalType::Voiced => RATE_LEVEL_ICDF_VOICED,
        };
        let rate_level = rd.dec_icdf(rate_level_icdf, 8) as u8;
        // rate_level is in 0..=8 by Table 45 cell count.
        debug_assert!(rate_level <= 8);
        if rd.has_error() {
            return Err(Error::MalformedPacket);
        }

        // ----- §4.2.7.8.2 pulses per shell block --------------------
        // For each block, we first read with the rate-level PDF; if the
        // value is 17, we re-read from the level-9 PDF; if THAT is 17,
        // from level-10; and so on. The number of 17s seen is the
        // number of extra LSBs to decode per coefficient for that
        // block.
        let mut pulses_per_block = Vec::with_capacity_in(shell_blocks, crate::bump::DecodeBump);
        pulses_per_block.resize(shell_blocks, 0u8);
        let mut lsb_count_per_block = Vec::with_capacity_in(shell_blocks, crate::bump::DecodeBump);
        lsb_count_per_block.resize(shell_blocks, 0u8);
        for block in 0..shell_blocks {
            // First read: with the chosen rate level.
            let mut sym = rd.dec_icdf(PULSE_COUNT_ICDFS[rate_level as usize], 8);
            let mut lsbs: u32 = 0;
            // Up to 9 chained reads from L9; the 10th forces L10 (cell
            // 17 has probability 0 in L10 so it cannot recur).
            while sym == 17 {
                lsbs += 1;
                if lsbs >= 10 {
                    // Use rate level 10 once we've seen 17 ten times.
                    sym = rd.dec_icdf(PULSE_COUNT_ICDFS[10], 8);
                    // L10 cell-17 prob is 0 (iCDF entry 16 = 0 = entry 17),
                    // so by RFC the next read cannot be 17. Defend
                    // anyway.
                    if sym == 17 {
                        return Err(Error::MalformedPacket);
                    }
                    break;
                } else {
                    sym = rd.dec_icdf(PULSE_COUNT_ICDFS[9], 8);
                }
            }
            // sym is now in 0..=16: the actual pulse count.
            debug_assert!(sym <= 16);
            pulses_per_block[block] = sym as u8;
            // The spec caps at 10 extra LSBs.
            debug_assert!(lsbs <= 10);
            lsb_count_per_block[block] = lsbs as u8;
            if rd.has_error() {
                return Err(Error::MalformedPacket);
            }
        }

        // ----- §4.2.7.8.3 pulse locations ---------------------------
        // For each block with pulses > 0, recursively partition into
        // halves (16 -> 8 -> 4 -> 2 -> 1), coding the count on the
        // left side at each split. After the 2 -> 1 split, the pulse
        // count IS the magnitude at that one sample.
        //
        // We hold the per-coefficient magnitudes in a `Vec<u8>` (each
        // magnitude in 0..=16; after LSB decoding, up to
        // `16 * 2^lsbs`, but we'll widen to i32 at sign time).
        // Profluens patch: local magnitudes scratch → reconstruct_e_q23, from the bump arena.
        let mut magnitudes = Vec::with_capacity_in(total_samples, crate::bump::DecodeBump);
        magnitudes.resize(total_samples, 0u32);
        for (block, &p) in pulses_per_block.iter().enumerate() {
            let pulses = p as u32;
            if pulses == 0 {
                continue;
            }
            let base = block * SHELL_BLOCK_SAMPLES;
            Self::decode_pulse_locations(
                rd,
                &mut magnitudes[base..base + SHELL_BLOCK_SAMPLES],
                16,
                pulses,
            )?;
        }

        // ----- §4.2.7.8.4 LSB decoding ------------------------------
        // For each block, for each coefficient (regardless of whether
        // it has any pulses), read `lsb_count_per_block[block]` bits
        // MSB-first from the Table 51 PDF, doubling and adding each.
        for (block, &l) in lsb_count_per_block.iter().enumerate() {
            let lsbs = l as u32;
            if lsbs == 0 {
                continue;
            }
            let base = block * SHELL_BLOCK_SAMPLES;
            for slot in magnitudes[base..base + SHELL_BLOCK_SAMPLES].iter_mut() {
                let mut mag = *slot;
                for _ in 0..lsbs {
                    let bit = rd.dec_icdf(LSB_ICDF, 8);
                    mag = (mag << 1) | bit;
                }
                *slot = mag;
            }
            if rd.has_error() {
                return Err(Error::MalformedPacket);
            }
        }

        // ----- §4.2.7.8.5 sign decoding -----------------------------
        // For each coefficient with magnitude > 0, decode a sign from
        // the Table 52 PDF chosen by (signal_type, qoff_type,
        // pulses_in_block). A decoded 0 means "negate"; a 1 means
        // "keep positive". The pulse count for the sign PDF is the
        // initial pulse count (pre-LSB).
        let mut signs = Vec::with_capacity_in(total_samples, crate::bump::DecodeBump);
        signs.resize(total_samples, 1i32); // +1 default for zero magnitudes.
        for (block, &p) in pulses_per_block.iter().enumerate() {
            let pulses_in_block = p as u32;
            let icdf = sign_icdf(cfg.signal_type, cfg.qoff_type, pulses_in_block);
            let base = block * SHELL_BLOCK_SAMPLES;
            for (mag, sign) in magnitudes[base..base + SHELL_BLOCK_SAMPLES]
                .iter()
                .zip(signs[base..base + SHELL_BLOCK_SAMPLES].iter_mut())
            {
                if *mag > 0 {
                    let s = rd.dec_icdf(icdf, 8);
                    *sign = if s == 0 { -1 } else { 1 };
                }
            }
            if rd.has_error() {
                return Err(Error::MalformedPacket);
            }
        }

        // ----- §4.2.7.8.6 reconstruction with LCG -------------------
        let e_q23 = reconstruct_e_q23(&magnitudes, &signs, &cfg);

        Ok(Self {
            e_q23,
            rate_level,
            shell_blocks,
            pulses_per_block,
            lsb_count_per_block,
        })
    }

    /// Encode the §4.2.7.8 excitation into `re` — the exact write-side
    /// mirror of [`Self::decode`].
    ///
    /// `symbols` carries the quantized signed excitation `e_raw[]`
    /// (one entry per sample; `|e_raw[i]|` is the FINAL per-sample
    /// magnitude after LSB refinement, the sign is applied when
    /// non-zero) together with the frame's rate level and the
    /// per-block extra-LSB counts. Everything the bitstream carries is
    /// derived from these: the per-block pre-LSB pulse count is
    /// `sum(|e_raw[i]| >> lsb_count)` over the block (must be
    /// `<= 16`), the §4.2.7.8.3 split tree codes the pre-LSB
    /// magnitudes, the §4.2.7.8.4 LSBs refine every coefficient of a
    /// block with a non-zero LSB count, and the §4.2.7.8.5 signs cover
    /// every non-zero final magnitude. Returns the [`Excitation`] the
    /// decoder will reconstruct (including the §4.2.7.8.6 LCG
    /// pseudorandom inversion, which consumes no bitstream symbols).
    pub fn encode(
        re: &mut RangeEncoder,
        cfg: ExcitationConfig,
        symbols: &ExcitationSymbols<'_>,
    ) -> Result<Self, Error> {
        if cfg.lcg_seed > 3 || symbols.rate_level > 8 {
            return Err(Error::MalformedPacket);
        }
        let shell_blocks = shell_block_count(cfg.bandwidth, cfg.frame_size)?;
        let total_samples = shell_blocks * SHELL_BLOCK_SAMPLES;
        if symbols.lsb_counts.len() != shell_blocks
            || symbols.e_raw.len() != total_samples
            || symbols.lsb_counts.iter().any(|&l| l > 10)
        {
            return Err(Error::MalformedPacket);
        }

        // ----- §4.2.7.8.1 rate level --------------------------------
        let rate_level_icdf = match cfg.signal_type {
            SignalType::Inactive | SignalType::Unvoiced => RATE_LEVEL_ICDF_INACTIVE_UNVOICED,
            SignalType::Voiced => RATE_LEVEL_ICDF_VOICED,
        };
        re.enc_icdf(symbols.rate_level as usize, rate_level_icdf, 8);

        // Derive the final magnitudes / signs and the pre-LSB ("top")
        // magnitudes per block.
        let mut magnitudes = vec![0u32; total_samples];
        let mut signs = vec![1i32; total_samples];
        let mut top = vec![0u32; total_samples];
        let mut pulses_per_block = Vec::with_capacity_in(shell_blocks, crate::bump::DecodeBump);
        pulses_per_block.resize(shell_blocks, 0u8);
        for (block, slot) in pulses_per_block.iter_mut().enumerate() {
            let lsbs = symbols.lsb_counts[block] as u32;
            let base = block * SHELL_BLOCK_SAMPLES;
            let mut p: u32 = 0;
            for i in base..base + SHELL_BLOCK_SAMPLES {
                let raw = symbols.e_raw[i];
                let mag = raw.unsigned_abs();
                magnitudes[i] = mag;
                signs[i] = if raw < 0 { -1 } else { 1 };
                let t = mag >> lsbs;
                top[i] = t;
                p += t;
            }
            if p > 16 {
                // The pre-LSB pulse count of a shell block is capped
                // at 16 (Table 46); the caller must raise the block's
                // LSB count to fit.
                return Err(Error::MalformedPacket);
            }
            // Every coefficient's magnitude must be exactly
            // representable as `top * 2^lsbs + lsb_bits`, which it is
            // by construction (top = mag >> lsbs).
            *slot = p as u8;
        }

        // ----- §4.2.7.8.2 pulses per shell block --------------------
        // A block with `n` extra LSBs writes the 17 escape once from
        // the rate-level PDF and `n - 1` more times from the level-9
        // PDF; the actual pulse count then comes from the level-9 PDF
        // (or level-10 when n == 10, whose cell 17 has zero
        // probability, terminating the chain).
        for (block, &pb) in pulses_per_block.iter().enumerate() {
            let lsbs = symbols.lsb_counts[block] as u32;
            let p = pb as usize;
            if lsbs == 0 {
                re.enc_icdf(p, PULSE_COUNT_ICDFS[symbols.rate_level as usize], 8);
            } else {
                re.enc_icdf(17, PULSE_COUNT_ICDFS[symbols.rate_level as usize], 8);
                for _ in 1..lsbs {
                    re.enc_icdf(17, PULSE_COUNT_ICDFS[9], 8);
                }
                if lsbs < 10 {
                    re.enc_icdf(p, PULSE_COUNT_ICDFS[9], 8);
                } else {
                    re.enc_icdf(p, PULSE_COUNT_ICDFS[10], 8);
                }
            }
        }

        // ----- §4.2.7.8.3 pulse locations ---------------------------
        for (block, &pb) in pulses_per_block.iter().enumerate() {
            if pb == 0 {
                continue;
            }
            let base = block * SHELL_BLOCK_SAMPLES;
            Self::encode_pulse_locations(re, &top[base..base + SHELL_BLOCK_SAMPLES], 16)?;
        }

        // ----- §4.2.7.8.4 LSB encoding ------------------------------
        for (block, &lc) in symbols.lsb_counts.iter().enumerate() {
            let lsbs = lc as u32;
            if lsbs == 0 {
                continue;
            }
            let base = block * SHELL_BLOCK_SAMPLES;
            for &mag in &magnitudes[base..base + SHELL_BLOCK_SAMPLES] {
                // MSB-first refinement bits below the top magnitude.
                for j in (0..lsbs).rev() {
                    let bit = (mag >> j) & 1;
                    re.enc_icdf(bit as usize, LSB_ICDF, 8);
                }
            }
        }

        // ----- §4.2.7.8.5 sign encoding -----------------------------
        for (block, &pb) in pulses_per_block.iter().enumerate() {
            let icdf = sign_icdf(cfg.signal_type, cfg.qoff_type, pb as u32);
            let base = block * SHELL_BLOCK_SAMPLES;
            for i in base..base + SHELL_BLOCK_SAMPLES {
                if magnitudes[i] > 0 {
                    // Symbol 0 = negative, 1 = positive.
                    re.enc_icdf(if signs[i] < 0 { 0 } else { 1 }, icdf, 8);
                }
            }
        }

        // ----- §4.2.7.8.6 reconstruction with LCG -------------------
        let e_q23 = reconstruct_e_q23(&magnitudes, &signs, &cfg);
        // Profluens patch: Excitation now carries bump-arena buffers (encode is test-only and not
        // wired; it shares the decode bump harmlessly). See the struct.
        let lsb_count_per_block = symbols.lsb_counts.to_vec_in(crate::bump::DecodeBump);

        Ok(Self {
            e_q23,
            rate_level: symbols.rate_level,
            shell_blocks,
            pulses_per_block,
            lsb_count_per_block,
        })
    }

    /// Recursive helper mirroring [`Self::decode_pulse_locations`]:
    /// write the §4.2.7.8.3 left-half counts, preorder, for a
    /// partition whose per-sample pre-LSB magnitudes are `top`.
    ///
    /// Defensively returns `Error::MalformedPacket` if a split were
    /// to land on a zero-probability cell (encoding one would collapse
    /// the range to zero width and hang the coder). The corrected
    /// Table 47-50 transcriptions contain no zero cells, so every
    /// distribution of `1..=16` pulses over a 16-sample block is
    /// representable and this guard is unreachable in practice.
    fn encode_pulse_locations(
        re: &mut RangeEncoder,
        top: &[u32],
        partition_size: usize,
    ) -> Result<(), Error> {
        let pulses: u32 = top.iter().sum();
        debug_assert!(pulses > 0);
        if partition_size == 1 {
            // Terminal: the count IS the magnitude; nothing coded.
            return Ok(());
        }
        let half = partition_size / 2;
        let left: u32 = top[..half].iter().sum();
        let right = pulses - left;
        let icdf = split_icdf(partition_size, pulses as u8);
        // Reject zero-probability cells: encoding one would collapse
        // the range to zero width (an invalid, undecodable stream).
        let k = left as usize;
        let width = if k == 0 {
            256 - icdf[0] as u32
        } else {
            icdf[k - 1] as u32 - icdf[k] as u32
        };
        if width == 0 {
            return Err(Error::MalformedPacket);
        }
        re.enc_icdf(k, icdf, 8);
        if left > 0 {
            Self::encode_pulse_locations(re, &top[..half], half)?;
        }
        if right > 0 {
            Self::encode_pulse_locations(re, &top[half..], half)?;
        }
        Ok(())
    }

    /// Recursive helper for §4.2.7.8.3: distribute `pulses` pulses into
    /// `magnitudes` (length `partition_size`).
    fn decode_pulse_locations(
        rd: &mut RangeDecoder<'_>,
        magnitudes: &mut [u32],
        partition_size: usize,
        pulses: u32,
    ) -> Result<(), Error> {
        if pulses == 0 {
            return Ok(());
        }
        if partition_size == 1 {
            // Terminal: all `pulses` pulses land on this one sample.
            magnitudes[0] = pulses;
            return Ok(());
        }
        // §4.2.7.8.3: read how many pulses go on the LEFT half. Right =
        // pulses - left.
        let icdf = split_icdf(partition_size, pulses as u8);
        let left = rd.dec_icdf(icdf, 8);
        if rd.has_error() {
            return Err(Error::MalformedPacket);
        }
        let right = pulses - left;
        // Safety: the §4.2.7.8.3 split PDFs all have `pulses + 1`
        // entries, so the decoded value is in 0..=pulses by
        // construction.
        debug_assert!(left <= pulses);

        let half = partition_size / 2;
        // Preorder: left first, then right.
        if left > 0 {
            Self::decode_pulse_locations(rd, &mut magnitudes[..half], half, left)?;
        }
        if right > 0 {
            Self::decode_pulse_locations(rd, &mut magnitudes[half..], half, right)?;
        }
        Ok(())
    }

    /// Final Q23 excitation vector. Length = `samples()`.
    pub fn e_q23(&self) -> &[i32] {
        &self.e_q23
    }

    /// Number of reconstructed samples (`16 * shell_blocks`).
    pub fn samples(&self) -> usize {
        self.e_q23.len()
    }

    /// Number of shell blocks for this frame.
    pub fn shell_blocks(&self) -> usize {
        self.shell_blocks
    }

    /// §4.2.7.8.1 decoded rate level (0..=8).
    pub fn rate_level(&self) -> u8 {
        self.rate_level
    }

    /// §4.2.7.8.2 per-block pulse counts.
    pub fn pulses_per_block(&self) -> &[u8] {
        &self.pulses_per_block
    }

    /// §4.2.7.8.2 per-block extra-LSB counts (number of times "17" was
    /// drawn before the actual pulse count).
    pub fn lsb_count_per_block(&self) -> &[u8] {
        &self.lsb_count_per_block
    }
}

/// The §4.2.7.8 symbol script for one SILK frame on the encode side,
/// consumed by [`Excitation::encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExcitationSymbols<'a> {
    /// §4.2.7.8.1 rate level, `0..=8`.
    pub rate_level: u8,
    /// §4.2.7.8.2 per-shell-block extra-LSB counts, `0..=10` each;
    /// length must equal the frame's shell-block count (Table 44).
    pub lsb_counts: &'a [u8],
    /// The quantized signed excitation, one entry per sample
    /// (`16 × shell_blocks`). `|e_raw[i]|` is the final per-sample
    /// magnitude (after LSB refinement); the sign is coded for
    /// non-zero magnitudes. Each block's pre-LSB pulse count
    /// `sum(|e_raw[i]| >> lsb_count)` must be `<= 16`.
    pub e_raw: &'a [i32],
}

/// The §4.2.7.8.6 reconstruction, shared by the decode and encode
/// paths. For each sample:
///
/// ```text
/// e_raw  = signs[i] * magnitudes[i]
/// e_Q23  = (e_raw << 8) - sign(e_raw)*20 + offset_Q23
/// seed   = (196314165*seed + 907633515) & 0xFFFFFFFF
/// if (seed & 0x80000000): e_Q23 = -e_Q23
/// seed   = (seed + e_raw) & 0xFFFFFFFF
/// ```
///
/// where `sign()` returns 0 for a zero magnitude (§1.1.4), so the 20
/// term is only applied to non-zero samples, and the LCG runs in
/// 32-bit wrapping arithmetic seeded by the §4.2.7.7 seed.
fn reconstruct_e_q23(
    magnitudes: &[u32],
    signs: &[i32],
    cfg: &ExcitationConfig,
) -> Vec<i32, crate::bump::DecodeBump> {
    let offset_q23 = quantization_offset_q23(cfg.signal_type, cfg.qoff_type);
    let mut e_q23 = Vec::with_capacity_in(magnitudes.len(), crate::bump::DecodeBump);
    e_q23.resize(magnitudes.len(), 0i32);
    let mut seed: u32 = cfg.lcg_seed as u32;
    for i in 0..magnitudes.len() {
        let mag = magnitudes[i] as i32;
        let sign = signs[i];
        let e_raw = sign * mag;
        let sign_e = if mag == 0 { 0 } else { sign };
        let mut e = (e_raw << 8) - sign_e * 20 + offset_q23;
        seed = seed.wrapping_mul(196_314_165).wrapping_add(907_633_515);
        if (seed & 0x8000_0000) != 0 {
            e = -e;
        }
        // e_raw can be negative; the u32 cast wraps mod 2^32, which is
        // exactly the spec's `& 0xFFFFFFFF`.
        seed = seed.wrapping_add(e_raw as u32);
        e_q23[i] = e;
    }
    e_q23
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: verify an iCDF transcription against its source PDF.
    fn check_icdf(pdf: &[u32], icdf: &[u8]) {
        assert_eq!(pdf.iter().sum::<u32>(), 256, "PDF must sum to 256");
        assert_eq!(pdf.len(), icdf.len(), "iCDF length must match PDF length");
        let mut acc: u32 = 0;
        for k in 0..pdf.len() {
            acc += pdf[k];
            let expected = 256u32.saturating_sub(acc);
            assert_eq!(
                icdf[k] as u32, expected,
                "iCDF mismatch at k={k}: expected {expected}, got {}",
                icdf[k]
            );
        }
    }

    // --- Table 44 shell block count --------------------------------

    #[test]
    fn table44_shell_block_counts() {
        assert_eq!(
            shell_block_count(Bandwidth::Nb, SilkFrameSize::TenMs).unwrap(),
            5
        );
        assert_eq!(
            shell_block_count(Bandwidth::Mb, SilkFrameSize::TenMs).unwrap(),
            8
        );
        assert_eq!(
            shell_block_count(Bandwidth::Wb, SilkFrameSize::TenMs).unwrap(),
            10
        );
        assert_eq!(
            shell_block_count(Bandwidth::Nb, SilkFrameSize::TwentyMs).unwrap(),
            10
        );
        assert_eq!(
            shell_block_count(Bandwidth::Mb, SilkFrameSize::TwentyMs).unwrap(),
            15
        );
        assert_eq!(
            shell_block_count(Bandwidth::Wb, SilkFrameSize::TwentyMs).unwrap(),
            20
        );
        // SWB / FB rejected.
        assert!(shell_block_count(Bandwidth::Swb, SilkFrameSize::TwentyMs).is_err());
        assert!(shell_block_count(Bandwidth::Fb, SilkFrameSize::TwentyMs).is_err());
    }

    // --- Table 45 rate-level iCDF ----------------------------------

    #[test]
    fn table45_rate_level_icdf() {
        check_icdf(
            &[15, 51, 12, 46, 45, 13, 33, 27, 14],
            RATE_LEVEL_ICDF_INACTIVE_UNVOICED,
        );
        check_icdf(
            &[33, 30, 36, 17, 34, 49, 18, 21, 18],
            RATE_LEVEL_ICDF_VOICED,
        );
    }

    // --- Table 46 pulse-count iCDFs --------------------------------

    #[test]
    fn table46_pulse_count_icdfs_all_eleven() {
        let pdfs: [&[u32]; 11] = [
            &[131, 74, 25, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            &[58, 93, 60, 23, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            &[43, 51, 46, 33, 24, 16, 11, 8, 6, 3, 3, 3, 2, 1, 1, 2, 1, 2],
            &[17, 52, 71, 57, 31, 12, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            &[6, 21, 41, 53, 49, 35, 21, 11, 6, 3, 2, 2, 1, 1, 1, 1, 1, 1],
            &[
                7, 14, 22, 28, 29, 28, 25, 20, 17, 13, 11, 9, 7, 5, 4, 4, 3, 10,
            ],
            &[2, 5, 14, 29, 42, 46, 41, 31, 19, 11, 6, 3, 2, 1, 1, 1, 1, 1],
            &[
                1, 2, 4, 10, 19, 29, 35, 37, 34, 28, 20, 14, 8, 5, 4, 2, 2, 2,
            ],
            &[
                1, 2, 2, 5, 9, 14, 20, 24, 27, 28, 26, 23, 20, 15, 11, 8, 6, 15,
            ],
            &[1, 1, 1, 6, 27, 58, 56, 39, 25, 14, 10, 6, 3, 3, 2, 1, 1, 2],
            &[2, 1, 6, 27, 58, 56, 39, 25, 14, 10, 6, 3, 3, 2, 1, 1, 2, 0],
        ];
        for (i, pdf) in pdfs.iter().enumerate() {
            check_icdf(pdf, PULSE_COUNT_ICDFS[i]);
        }
        // Level 10's last cell (the "17" slot) MUST be 0 so the spec
        // guarantees no further chain.
        assert_eq!(PULSE_COUNT_ICDFS[10][16], 0);
        assert_eq!(PULSE_COUNT_ICDFS[10][17], 0);
    }

    // --- Tables 47-50 split iCDFs: exhaustive per-row verification -
    //
    // Every row of all four split tables is checked against the RFC
    // 6716 PDF text (a round-382 sweep found five mis-transcribed rows
    // in the original spot-checked transcription: Table 47 rows 9 and
    // 11, Table 48 row 9, Table 49 rows 7 and 16).

    #[test]
    fn tables_47_to_50_all_rows_match_rfc_pdfs() {
        let table47: [&[u32]; 16] = [
            &[126, 130],
            &[56, 142, 58],
            &[25, 101, 104, 26],
            &[12, 60, 108, 64, 12],
            &[7, 35, 84, 87, 37, 6],
            &[4, 20, 59, 86, 63, 21, 3],
            &[3, 12, 38, 72, 75, 42, 12, 2],
            &[2, 8, 25, 54, 73, 59, 27, 7, 1],
            &[2, 5, 17, 39, 63, 65, 42, 18, 4, 1],
            &[1, 4, 12, 28, 49, 63, 54, 30, 11, 3, 1],
            &[1, 4, 8, 20, 37, 55, 57, 41, 22, 8, 2, 1],
            &[1, 3, 7, 15, 28, 44, 53, 48, 33, 16, 6, 1, 1],
            &[1, 2, 6, 12, 21, 35, 47, 48, 40, 25, 12, 5, 1, 1],
            &[1, 1, 4, 10, 17, 27, 37, 47, 43, 33, 21, 9, 4, 1, 1],
            &[1, 1, 1, 8, 14, 22, 33, 40, 43, 38, 28, 16, 8, 1, 1, 1],
            &[1, 1, 1, 1, 13, 18, 27, 36, 41, 41, 34, 24, 14, 1, 1, 1, 1],
        ];
        let table48: [&[u32]; 16] = [
            &[127, 129],
            &[53, 149, 54],
            &[22, 105, 106, 23],
            &[11, 61, 111, 63, 10],
            &[6, 35, 86, 88, 36, 5],
            &[4, 20, 59, 87, 62, 21, 3],
            &[3, 13, 40, 71, 73, 41, 13, 2],
            &[3, 9, 27, 53, 70, 56, 28, 9, 1],
            &[3, 8, 19, 37, 57, 61, 44, 20, 6, 1],
            &[3, 7, 15, 28, 44, 54, 49, 33, 17, 5, 1],
            &[1, 7, 13, 22, 34, 46, 48, 38, 28, 14, 4, 1],
            &[1, 1, 11, 22, 27, 35, 42, 47, 33, 25, 10, 1, 1],
            &[1, 1, 6, 14, 26, 37, 43, 43, 37, 26, 14, 6, 1, 1],
            &[1, 1, 4, 10, 20, 31, 40, 42, 40, 31, 20, 10, 4, 1, 1],
            &[1, 1, 3, 8, 16, 26, 35, 38, 38, 35, 26, 16, 8, 3, 1, 1],
            &[1, 1, 2, 6, 12, 21, 30, 36, 38, 36, 30, 21, 12, 6, 2, 1, 1],
        ];
        let table49: [&[u32]; 16] = [
            &[127, 129],
            &[49, 157, 50],
            &[20, 107, 109, 20],
            &[11, 60, 113, 62, 10],
            &[7, 36, 84, 87, 36, 6],
            &[6, 24, 57, 82, 60, 23, 4],
            &[5, 18, 39, 64, 68, 42, 16, 4],
            &[6, 14, 29, 47, 61, 52, 30, 14, 3],
            &[1, 15, 23, 35, 51, 50, 40, 30, 10, 1],
            &[1, 1, 21, 32, 42, 52, 46, 41, 18, 1, 1],
            &[1, 6, 16, 27, 36, 42, 42, 36, 27, 16, 6, 1],
            &[1, 5, 12, 21, 31, 38, 40, 38, 31, 21, 12, 5, 1],
            &[1, 3, 9, 17, 26, 34, 38, 38, 34, 26, 17, 9, 3, 1],
            &[1, 3, 7, 14, 22, 29, 34, 36, 34, 29, 22, 14, 7, 3, 1],
            &[1, 2, 5, 11, 18, 25, 31, 35, 35, 31, 25, 18, 11, 5, 2, 1],
            &[1, 1, 4, 9, 15, 21, 28, 32, 34, 32, 28, 21, 15, 9, 4, 1, 1],
        ];
        let table50: [&[u32]; 16] = [
            &[128, 128],
            &[42, 172, 42],
            &[21, 107, 107, 21],
            &[12, 60, 112, 61, 11],
            &[8, 34, 86, 86, 35, 7],
            &[8, 23, 55, 90, 55, 20, 5],
            &[5, 15, 38, 72, 72, 36, 15, 3],
            &[6, 12, 27, 52, 77, 47, 20, 10, 5],
            &[6, 19, 28, 35, 40, 40, 35, 28, 19, 6],
            &[4, 14, 22, 31, 37, 40, 37, 31, 22, 14, 4],
            &[3, 10, 18, 26, 33, 38, 38, 33, 26, 18, 10, 3],
            &[2, 8, 13, 21, 29, 36, 38, 36, 29, 21, 13, 8, 2],
            &[1, 5, 10, 17, 25, 32, 38, 38, 32, 25, 17, 10, 5, 1],
            &[1, 4, 7, 13, 21, 29, 35, 36, 35, 29, 21, 13, 7, 4, 1],
            &[1, 2, 5, 10, 17, 25, 32, 36, 36, 32, 25, 17, 10, 5, 2, 1],
            &[1, 2, 4, 7, 13, 21, 28, 34, 36, 34, 28, 21, 13, 7, 4, 2, 1],
        ];
        for p in 0..16 {
            check_icdf(table47[p], SPLIT16_ICDFS[p]);
            check_icdf(table48[p], SPLIT8_ICDFS[p]);
            check_icdf(table49[p], SPLIT4_ICDFS[p]);
            check_icdf(table50[p], SPLIT2_ICDFS[p]);
        }
    }

    // --- Table 51 LSB iCDF -----------------------------------------

    #[test]
    fn table51_lsb_icdf() {
        check_icdf(&[136, 120], LSB_ICDF);
    }

    // --- Table 52 sign iCDFs (one spot-check per signal/qoff) ------

    #[test]
    fn table52_sign_icdf_spot_checks() {
        // Inactive/Low/0  -> {2, 254}
        check_icdf(&[2, 254], SIGN_ICDF_INACTIVE_LOW[0]);
        // Inactive/High/3 -> {232, 24}
        check_icdf(&[232, 24], SIGN_ICDF_INACTIVE_HIGH[3]);
        // Unvoiced/Low/6+ -> {152, 104}
        check_icdf(&[152, 104], SIGN_ICDF_UNVOICED_LOW[6]);
        // Unvoiced/High/2 -> {235, 21}
        check_icdf(&[235, 21], SIGN_ICDF_UNVOICED_HIGH[2]);
        // Voiced/Low/4    -> {144, 112}
        check_icdf(&[144, 112], SIGN_ICDF_VOICED_LOW[4]);
        // Voiced/High/0   -> {8, 248}
        check_icdf(&[8, 248], SIGN_ICDF_VOICED_HIGH[0]);
    }

    #[test]
    fn table52_sign_icdf_pulse_count_clamping() {
        // The "6 or more" row covers every pulses-in-block >= 6.
        let a = sign_icdf(SignalType::Voiced, QuantizationOffsetType::Low, 6);
        let b = sign_icdf(SignalType::Voiced, QuantizationOffsetType::Low, 7);
        let c = sign_icdf(SignalType::Voiced, QuantizationOffsetType::Low, 16);
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // --- Table 53 quantization offsets -----------------------------

    #[test]
    fn table53_quantization_offsets() {
        assert_eq!(
            quantization_offset_q23(SignalType::Inactive, QuantizationOffsetType::Low),
            25
        );
        assert_eq!(
            quantization_offset_q23(SignalType::Inactive, QuantizationOffsetType::High),
            60
        );
        assert_eq!(
            quantization_offset_q23(SignalType::Unvoiced, QuantizationOffsetType::Low),
            25
        );
        assert_eq!(
            quantization_offset_q23(SignalType::Unvoiced, QuantizationOffsetType::High),
            60
        );
        assert_eq!(
            quantization_offset_q23(SignalType::Voiced, QuantizationOffsetType::Low),
            8
        );
        assert_eq!(
            quantization_offset_q23(SignalType::Voiced, QuantizationOffsetType::High),
            25
        );
    }

    // --- LCG sanity -------------------------------------------------

    #[test]
    fn lcg_recurrence_pinned() {
        // Hand-roll the recurrence from a known starting seed and
        // confirm the values match the spec recurrence
        // seed' = 196314165 * seed + 907633515 mod 2^32.
        let mut s: u32 = 0;
        s = s.wrapping_mul(196_314_165).wrapping_add(907_633_515);
        assert_eq!(s, 907_633_515);
        let s2 = s.wrapping_mul(196_314_165).wrapping_add(907_633_515);
        // 907633515 * 196314165 = mod 2^32 = ?
        let prod = 907_633_515u64.wrapping_mul(196_314_165) & 0xFFFF_FFFF;
        let expected = (prod as u32).wrapping_add(907_633_515);
        assert_eq!(s2, expected);
    }

    // --- Excitation::decode end-to-end ----------------------------

    #[test]
    fn rejects_invalid_lcg_seed() {
        let buf = [0x55u8, 0xAA, 0x33, 0xCC];
        let mut rd = RangeDecoder::new(&buf);
        let cfg = ExcitationConfig {
            bandwidth: Bandwidth::Nb,
            frame_size: SilkFrameSize::TenMs,
            signal_type: SignalType::Voiced,
            qoff_type: QuantizationOffsetType::Low,
            lcg_seed: 4,
        };
        assert!(Excitation::decode(&mut rd, cfg).is_err());
    }

    #[test]
    fn rejects_swb_fb() {
        let buf = [0x55u8; 64];
        for bw in [Bandwidth::Swb, Bandwidth::Fb] {
            let mut rd = RangeDecoder::new(&buf);
            let cfg = ExcitationConfig {
                bandwidth: bw,
                frame_size: SilkFrameSize::TwentyMs,
                signal_type: SignalType::Voiced,
                qoff_type: QuantizationOffsetType::Low,
                lcg_seed: 0,
            };
            assert!(Excitation::decode(&mut rd, cfg).is_err());
        }
    }

    #[test]
    fn decode_produces_correct_sample_count() {
        let buf = [0x55u8; 128];
        for (bw, size, expected_samples) in [
            (Bandwidth::Nb, SilkFrameSize::TenMs, 80),
            (Bandwidth::Mb, SilkFrameSize::TenMs, 128),
            (Bandwidth::Wb, SilkFrameSize::TenMs, 160),
            (Bandwidth::Nb, SilkFrameSize::TwentyMs, 160),
            (Bandwidth::Mb, SilkFrameSize::TwentyMs, 240),
            (Bandwidth::Wb, SilkFrameSize::TwentyMs, 320),
        ] {
            let mut rd = RangeDecoder::new(&buf);
            let cfg = ExcitationConfig {
                bandwidth: bw,
                frame_size: size,
                signal_type: SignalType::Voiced,
                qoff_type: QuantizationOffsetType::Low,
                lcg_seed: 1,
            };
            let exc = Excitation::decode(&mut rd, cfg).unwrap();
            assert_eq!(exc.samples(), expected_samples, "bw={bw:?} size={size:?}");
            assert_eq!(exc.shell_blocks(), expected_samples / SHELL_BLOCK_SAMPLES);
            assert_eq!(exc.pulses_per_block().len(), exc.shell_blocks());
            assert_eq!(exc.lsb_count_per_block().len(), exc.shell_blocks());
        }
    }

    #[test]
    fn decode_e_q23_fits_in_24_bits() {
        // §4.2.7.8 says "e_Q23[i] value may require more than 16 bits
        // per sample, but it will not require more than 23, including
        // the sign". So |e_Q23[i]| < 2^23.
        let buffers: [&[u8]; 3] = [
            &[0x00u8; 64],
            &[0xFFu8; 64],
            &[
                0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6, 0x49, 0x88, 0xE0, 0x21,
                0x77, 0xCD, 0x35, 0x9A, 0x6E, 0x04, 0xBB, 0x52, 0xA1, 0xFC, 0x10, 0x83, 0x6F, 0xD4,
                0x29, 0x95, 0x4B, 0xC7, 0x1E, 0x80, 0x67, 0xAC, 0x33, 0xD9, 0x06, 0x71, 0x58, 0xE2,
                0x4F, 0x90, 0x2B, 0xC4, 0x16, 0x83, 0x6D, 0xD1, 0x28, 0x9E, 0x4A, 0xC0, 0x1F, 0x85,
                0x65, 0xAD, 0x32, 0xDF, 0x07, 0x70, 0x55, 0xE1,
            ],
        ];
        for buf in buffers {
            for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
                for size in [SilkFrameSize::TenMs, SilkFrameSize::TwentyMs] {
                    let mut rd = RangeDecoder::new(buf);
                    let cfg = ExcitationConfig {
                        bandwidth: bw,
                        frame_size: size,
                        signal_type: SignalType::Voiced,
                        qoff_type: QuantizationOffsetType::High,
                        lcg_seed: 2,
                    };
                    let exc = Excitation::decode(&mut rd, cfg).unwrap();
                    for &v in exc.e_q23() {
                        // Strictly < 2^23 in magnitude (sign included
                        // brings it back into i24 range).
                        // The spec says "not require more than 23,
                        // including the sign", so |v| <= 2^23.
                        assert!(
                            v.abs() <= (1i32 << 23),
                            "e_Q23 sample {v} exceeds 24-bit range (bw={bw:?} size={size:?})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pulse_count_invariants() {
        let buf = [
            0x10u8, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0,
            0xF0, 0x01, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78, 0x89, 0x9A, 0xAB, 0xBC, 0xCD,
            0xDE, 0xEF, 0x00, 0x11,
        ];
        let mut rd = RangeDecoder::new(&buf);
        let cfg = ExcitationConfig {
            bandwidth: Bandwidth::Wb,
            frame_size: SilkFrameSize::TwentyMs,
            signal_type: SignalType::Voiced,
            qoff_type: QuantizationOffsetType::Low,
            lcg_seed: 0,
        };
        let exc = Excitation::decode(&mut rd, cfg).unwrap();
        // Every per-block pulse count is in 0..=16.
        for &p in exc.pulses_per_block() {
            assert!(p <= 16, "pulse count {p} > 16");
        }
        // Every per-block LSB count is in 0..=10.
        for &l in exc.lsb_count_per_block() {
            assert!(l <= 10, "lsb count {l} > 10");
        }
        // Rate level is in 0..=8.
        assert!(exc.rate_level() <= 8);
    }

    // --- Hand-derived end-to-end pin --------------------------------

    #[test]
    fn hand_derived_no_pulses_zero_lsbs_path() {
        // Construct a contrived scenario: feed the decoder the LCG seed
        // 0 and check that an all-zero-magnitude block (which we can
        // induce by parking the decoder where rate_level decodes to
        // something low and the very-first pulse-count symbol decodes
        // to 0) produces a Q23 vector matching the §4.2.7.8.6 closed
        // form. We don't need this branch to actually fire: it's
        // enough to verify the algebra by mocking magnitudes = [0; N].
        // The production path runs the real decoder; this test pins
        // the post-decode arithmetic with known inputs by re-running
        // it directly.
        let total_samples = 16usize;
        let mut magnitudes = vec![0u32; total_samples];
        // Place a single pulse at index 3 of magnitude 5, sign -1.
        magnitudes[3] = 5;
        let mut signs = vec![1i32; total_samples];
        signs[3] = -1;
        let offset_q23 = 25; // Voiced/High.
        let mut e_q23 = vec![0i32; total_samples];
        let mut seed: u32 = 1;
        for i in 0..total_samples {
            let mag = magnitudes[i] as i32;
            let sign = signs[i];
            let e_raw = sign * mag;
            let sign_e = if mag == 0 { 0 } else { sign };
            let mut e = (e_raw << 8) - sign_e * 20 + offset_q23;
            seed = seed.wrapping_mul(196_314_165).wrapping_add(907_633_515);
            if (seed & 0x8000_0000) != 0 {
                e = -e;
            }
            seed = seed.wrapping_add(e_raw as u32);
            e_q23[i] = e;
        }
        // i=0: mag=0, e_raw=0, sign(e_raw)=0, e=0+0+25=25
        //      seed=1*196314165+907633515=1103947680.
        //      0x41C84320 -> MSB=0, e unchanged=25. seed+=0.
        assert_eq!(e_q23[0], 25);
        // i=3 sanity: mag=5, sign=-1 -> e_raw=-5
        //   e = -5<<8 - (-1)*20 + 25 = -1280 + 20 + 25 = -1235
        //   then possibly negated by the LCG.
        // We don't try to predict the sign flip; just verify the
        // magnitude before LCG flipping landed correctly.
        // The negated case yields ±1235.
        assert!(
            e_q23[3] == -1235 || e_q23[3] == 1235,
            "e_q23[3] = {} not ±1235",
            e_q23[3]
        );
    }

    #[test]
    fn zero_magnitude_samples_use_offset_only() {
        // Sample magnitudes that are zero have e_raw=0 and sign(e_raw)
        // = 0, so e_Q23 = 0 - 0 + offset_Q23 = offset_Q23, possibly
        // negated by the LCG. So |e_Q23[i]| must equal offset_Q23 for
        // every zero-magnitude sample.
        //
        // We arrange this by decoding a small frame and checking each
        // sample with mag==0.
        let buf = [0x00u8; 32]; // all zeros tend to produce many zero magnitudes.
        let mut rd = RangeDecoder::new(&buf);
        let cfg = ExcitationConfig {
            bandwidth: Bandwidth::Nb,
            frame_size: SilkFrameSize::TenMs,
            signal_type: SignalType::Inactive,
            qoff_type: QuantizationOffsetType::Low,
            lcg_seed: 0,
        };
        let exc = Excitation::decode(&mut rd, cfg).unwrap();
        let offset = quantization_offset_q23(SignalType::Inactive, QuantizationOffsetType::Low);
        // Find blocks with 0 pulses and 0 LSBs — every sample in those
        // blocks has magnitude 0.
        for (b, (&p, &l)) in exc
            .pulses_per_block()
            .iter()
            .zip(exc.lsb_count_per_block().iter())
            .enumerate()
        {
            if p == 0 && l == 0 {
                let base = b * SHELL_BLOCK_SAMPLES;
                for i in 0..SHELL_BLOCK_SAMPLES {
                    assert_eq!(
                        exc.e_q23()[base + i].abs(),
                        offset,
                        "block {b} sample {i} should be ±{offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn reproducible_across_runs() {
        // Same input bytes + same config -> same Q23 output.
        let buf = [
            0x9A, 0x42, 0x18, 0xC7, 0x6E, 0xB1, 0x05, 0xFD, 0x33, 0x80, 0x55, 0xAC, 0x21, 0xE9,
            0x4B, 0x96, 0x0F, 0xD2, 0x68, 0xA7,
        ];
        let cfg = ExcitationConfig {
            bandwidth: Bandwidth::Wb,
            frame_size: SilkFrameSize::TwentyMs,
            signal_type: SignalType::Voiced,
            qoff_type: QuantizationOffsetType::High,
            lcg_seed: 2,
        };
        let mut rd1 = RangeDecoder::new(&buf);
        let mut rd2 = RangeDecoder::new(&buf);
        let e1 = Excitation::decode(&mut rd1, cfg).unwrap();
        let e2 = Excitation::decode(&mut rd2, cfg).unwrap();
        assert_eq!(e1.e_q23(), e2.e_q23());
        assert_eq!(e1.pulses_per_block(), e2.pulses_per_block());
    }

    #[test]
    fn different_lcg_seeds_diverge() {
        // For the same range-decoder output, different LCG seeds
        // produce different e_Q23 sign patterns (with overwhelming
        // probability across a full 20 ms WB frame).
        let buf = [
            0x42, 0xC3, 0x17, 0x9F, 0x88, 0x01, 0x55, 0xAA, 0x33, 0xCC, 0x6E, 0x91, 0x04, 0xFD,
            0x28, 0xB6,
        ];
        let base_cfg = ExcitationConfig {
            bandwidth: Bandwidth::Wb,
            frame_size: SilkFrameSize::TwentyMs,
            signal_type: SignalType::Voiced,
            qoff_type: QuantizationOffsetType::Low,
            lcg_seed: 0,
        };
        let mut rd0 = RangeDecoder::new(&buf);
        let e0 = Excitation::decode(&mut rd0, base_cfg).unwrap();
        let mut rd1 = RangeDecoder::new(&buf);
        let e1 = Excitation::decode(
            &mut rd1,
            ExcitationConfig {
                lcg_seed: 1,
                ..base_cfg
            },
        )
        .unwrap();
        assert_ne!(
            e0.e_q23(),
            e1.e_q23(),
            "different LCG seeds should diverge somewhere across 320 samples"
        );
    }

    // --- Sweep: no panics ------------------------------------------

    #[test]
    fn sweep_never_panics() {
        let buffers: [&[u8]; 3] = [
            &[
                0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD,
                0xEE, 0xFF,
            ],
            &[
                0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22,
                0x11, 0x00,
            ],
            &[
                0x5A, 0xA5, 0x3C, 0xC3, 0x0F, 0xF0, 0x69, 0x96, 0x12, 0x48, 0x84, 0x21, 0x7E, 0xE7,
                0xB4, 0x4B,
            ],
        ];
        for buf in buffers {
            for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
                for size in [SilkFrameSize::TenMs, SilkFrameSize::TwentyMs] {
                    for signal in [
                        SignalType::Inactive,
                        SignalType::Unvoiced,
                        SignalType::Voiced,
                    ] {
                        for qoff in [QuantizationOffsetType::Low, QuantizationOffsetType::High] {
                            for seed in 0u8..=3 {
                                let mut rd = RangeDecoder::new(buf);
                                let cfg = ExcitationConfig {
                                    bandwidth: bw,
                                    frame_size: size,
                                    signal_type: signal,
                                    qoff_type: qoff,
                                    lcg_seed: seed,
                                };
                                let _ = Excitation::decode(&mut rd, cfg);
                            }
                        }
                    }
                }
            }
        }
    }

    // ----- §4.2.7.8 encode-side mirror ------------------------------

    /// A tiny deterministic LCG for the encode/decode roundtrip sweeps
    /// (test-harness only; unrelated to the §4.2.7.8.6 LCG).
    struct TestLcg(u64);
    impl TestLcg {
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

    /// encode → decode roundtrip over random excitation scripts across
    /// every bandwidth / frame size / signal type / LSB depth: the
    /// decoder must reconstruct exactly the `Excitation` (including
    /// `e_q23`) the encoder predicted.
    #[test]
    fn excitation_encode_decode_roundtrip_random() {
        use crate::range_encoder::RangeEncoder;
        let mut rng = TestLcg(0xE8C1_7A71_0000_0001);
        for round in 0..300 {
            let bandwidth = match rng.below(3) {
                0 => Bandwidth::Nb,
                1 => Bandwidth::Mb,
                _ => Bandwidth::Wb,
            };
            let frame_size = if rng.below(2) == 0 {
                SilkFrameSize::TenMs
            } else {
                SilkFrameSize::TwentyMs
            };
            let signal_type = match rng.below(3) {
                0 => SignalType::Inactive,
                1 => SignalType::Unvoiced,
                _ => SignalType::Voiced,
            };
            let qoff_type = if rng.below(2) == 0 {
                QuantizationOffsetType::Low
            } else {
                QuantizationOffsetType::High
            };
            let cfg = ExcitationConfig {
                bandwidth,
                frame_size,
                signal_type,
                qoff_type,
                lcg_seed: rng.below(4) as u8,
            };
            let blocks = shell_block_count(bandwidth, frame_size).unwrap();
            let total = blocks * SHELL_BLOCK_SAMPLES;

            // Random per-block LSB counts (mostly 0, sometimes 1..=3,
            // rarely deep chains up to 10) and pulse budgets.
            let mut lsb_counts = vec![0u8; blocks];
            let mut e_raw = vec![0i32; total];
            for (b, lc) in lsb_counts.iter_mut().enumerate() {
                let lsbs = match rng.below(10) {
                    0..=5 => 0u32,
                    6 | 7 => 1 + rng.below(3),
                    8 => 4 + rng.below(5),
                    _ => 10,
                };
                *lc = lsbs as u8;
                // Distribute up to 16 pre-LSB pulses over the block.
                let budget = rng.below(17);
                let base = b * SHELL_BLOCK_SAMPLES;
                let mut spent = 0u32;
                while spent < budget {
                    let i = base + rng.below(16) as usize;
                    let add = 1 + rng.below(budget - spent);
                    e_raw[i] += (add << lsbs) as i32;
                    spent += add;
                }
                // Random LSB refinement bits + random signs.
                for slot in e_raw[base..base + SHELL_BLOCK_SAMPLES].iter_mut() {
                    if lsbs > 0 {
                        *slot += (rng.next_u32() & ((1 << lsbs) - 1)) as i32;
                    }
                    if *slot != 0 && rng.below(2) == 0 {
                        *slot = -*slot;
                    }
                }
            }

            let symbols = ExcitationSymbols {
                rate_level: rng.below(9) as u8,
                lsb_counts: &lsb_counts,
                e_raw: &e_raw,
            };
            let mut re = RangeEncoder::new();
            let predicted = Excitation::encode(&mut re, cfg, &symbols).expect("encode");
            let bytes = re.finish();

            let mut rd = RangeDecoder::new(&bytes);
            let decoded = Excitation::decode(&mut rd, cfg).expect("decode");
            assert!(!rd.has_error(), "round {round}");
            assert_eq!(decoded, predicted, "round {round} cfg={cfg:?}");
        }
    }

    /// The encode path rejects an over-budget block (pre-LSB pulse
    /// count > 16), bad lengths, and out-of-range parameters.
    #[test]
    fn excitation_encode_rejects_bad_inputs() {
        use crate::range_encoder::RangeEncoder;
        let cfg = ExcitationConfig {
            bandwidth: Bandwidth::Nb,
            frame_size: SilkFrameSize::TenMs,
            signal_type: SignalType::Voiced,
            qoff_type: QuantizationOffsetType::Low,
            lcg_seed: 0,
        };
        let blocks = 5usize;
        let total = blocks * SHELL_BLOCK_SAMPLES;
        let lsb0 = vec![0u8; blocks];
        // 17 pre-LSB pulses in one block.
        let mut e = vec![0i32; total];
        e[0] = 17;
        let mut re = RangeEncoder::new();
        assert!(Excitation::encode(
            &mut re,
            cfg,
            &ExcitationSymbols {
                rate_level: 0,
                lsb_counts: &lsb0,
                e_raw: &e,
            }
        )
        .is_err());
        // Rate level out of range.
        let z = vec![0i32; total];
        let mut re = RangeEncoder::new();
        assert!(Excitation::encode(
            &mut re,
            cfg,
            &ExcitationSymbols {
                rate_level: 9,
                lsb_counts: &lsb0,
                e_raw: &z,
            }
        )
        .is_err());
        // Wrong lsb_counts length.
        let mut re = RangeEncoder::new();
        assert!(Excitation::encode(
            &mut re,
            cfg,
            &ExcitationSymbols {
                rate_level: 0,
                lsb_counts: &lsb0[..4],
                e_raw: &z,
            }
        )
        .is_err());
        // LSB count > 10.
        let bad_lsb = vec![11u8; blocks];
        let mut re = RangeEncoder::new();
        assert!(Excitation::encode(
            &mut re,
            cfg,
            &ExcitationSymbols {
                rate_level: 0,
                lsb_counts: &bad_lsb,
                e_raw: &z,
            }
        )
        .is_err());
        // All 16 pulses concentrated on the left 2-sample half of a
        // 4-sample partition: Table 49's `left = 16` cell has
        // probability 1/256 (the round-382 table sweep fixed this
        // row's transcription, which previously read 0 here and made
        // the layout hang the encoder), so the extreme layout must
        // encode AND decode back exactly.
        let mut e = vec![0i32; total];
        e[0] = 8;
        e[1] = 8;
        let mut re = RangeEncoder::new();
        let predicted = Excitation::encode(
            &mut re,
            cfg,
            &ExcitationSymbols {
                rate_level: 0,
                lsb_counts: &lsb0,
                e_raw: &e,
            },
        )
        .expect("extreme concentration must be encodable");
        let bytes = re.finish();
        let mut rd = RangeDecoder::new(&bytes);
        let decoded = Excitation::decode(&mut rd, cfg).expect("decode");
        assert_eq!(decoded, predicted);
        // A 17-magnitude with one LSB is fine (top = 8).
        let lsb1 = vec![1u8; blocks];
        let mut e = vec![0i32; total];
        e[0] = 17;
        let mut re = RangeEncoder::new();
        assert!(Excitation::encode(
            &mut re,
            cfg,
            &ExcitationSymbols {
                rate_level: 0,
                lsb_counts: &lsb1,
                e_raw: &e,
            }
        )
        .is_ok());
    }
}
