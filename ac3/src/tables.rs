//! Constant tables from ATSC A/52 — the normative lookup tables the AC-3 /
//! E-AC-3 decode chain reads. Every table cites the A/52 section (or table
//! number) it transcribes. Clean-room from the standard (no liba52/ffmpeg).
//!
//! Table numbers here refer to ATSC A/52:2018 (the consolidated AC-3 + E-AC-3
//! document). Where AC-3 and E-AC-3 share a table (bit allocation, mantissa
//! dequant, windows) the AC-3 chapter is cited; E-AC-3-only tables are marked.

// ---------------------------------------------------------------------------
// Sample rate / frame size (A/52 §5.3.2, Table 5.18 "Sample rate codes"; the
// frame-size table maps (frmsizecod, fscod) → words-per-syncframe for AC-3).
// ---------------------------------------------------------------------------

/// Sample rate in Hz keyed by `fscod` (A/52 Table 5.18). `fscod == 3` is reserved.
pub const SAMPLE_RATE: [u32; 3] = [48_000, 44_100, 32_000];

/// AC-3 frame size in 16-bit words keyed by `[fscod][frmsizecod]`
/// (A/52 Table 5.18 "Frame size code table"). 38 legal `frmsizecod` values
/// (0..=37), i.e. 19 nominal bit rates × 2 (the pad step). The frame length in
/// bytes is `2 * words`. For 44.1 kHz the words include the extra pad word
/// baked in here (A/52 §5.3.2 note). Values are transcribed directly from the
/// standard's frame-size table.
pub const FRAME_SIZE_WORDS: [[u16; 38]; 3] = [
    // fscod = 0 (48 kHz)
    [
        64, 64, 80, 80, 96, 96, 112, 112, 128, 128, 160, 160, 192, 192, 224, 224, 256, 256, 320,
        320, 384, 384, 448, 448, 512, 512, 640, 640, 768, 768, 896, 896, 1024, 1024, 1152, 1152,
        1280, 1280,
    ],
    // fscod = 1 (44.1 kHz)
    [
        69, 70, 87, 88, 104, 105, 121, 122, 139, 140, 174, 175, 208, 209, 243, 244, 278, 279, 348,
        349, 417, 418, 487, 488, 557, 558, 696, 697, 835, 836, 975, 976, 1114, 1115, 1253, 1254,
        1393, 1394,
    ],
    // fscod = 2 (32 kHz)
    [
        96, 96, 120, 120, 144, 144, 168, 168, 192, 192, 240, 240, 288, 288, 336, 336, 384, 384,
        480, 480, 576, 576, 672, 672, 768, 768, 960, 960, 1152, 1152, 1344, 1344, 1536, 1536, 1728,
        1728, 1920, 1920,
    ],
];

/// Number of full-bandwidth channels keyed by `acmod` (A/52 Table 5.8 "Audio
/// coding mode"). LFE is separate (`lfeon`). acmod 0 = 1+1 (Ch1,Ch2 dual mono).
pub const NFCHANS: [usize; 8] = [2, 1, 2, 3, 3, 4, 4, 5];

// ---------------------------------------------------------------------------
// Exponent decode (A/52 §7.1). Exponents are differentially coded within a
// channel; the "grouped" mantissa exponents come in packets of 3 (D15/D25/D45
// strategies) each encoding 3 five-level absolute-exponent deltas.
// ---------------------------------------------------------------------------

/// Exponent-strategy group size in mantissas per exponent
/// (A/52 §7.1.3, Table: D15→1, D25→2, D45→4). Index by `expstr` minus 1
/// (REUSE=0 is handled separately). Here index 0..=2 = D15,D25,D45.
pub const EXP_GROUP_SIZE: [usize; 3] = [1, 2, 4];

// ---------------------------------------------------------------------------
// Bit allocation (A/52 §7.2). The parametric bit allocation derives a masking
// curve and a bit-allocation pointer per bin. These are the normative tables.
// ---------------------------------------------------------------------------

/// Slow-decay table `slowdec[sdcycod]` (A/52 §7.2.2.5, Table). dB/band step
/// applied while descending the masking curve.
pub const SLOW_DECAY: [u8; 4] = [0x0F, 0x11, 0x13, 0x15];

/// Fast-decay table `fastdec[fdcycod]` (A/52 §7.2.2.5).
pub const FAST_DECAY: [u8; 4] = [0x3F, 0x53, 0x67, 0x7B];

/// Slow-gain table `slowgain[sgaincod]` (A/52 §7.2.2.5).
pub const SLOW_GAIN: [u16; 4] = [0x540, 0x4D8, 0x478, 0x410];

/// dB/bit-per-mantissa knee `dbknee[dbpbcod]` (A/52 §7.2.2.5).
pub const DB_PER_BIT: [u16; 4] = [0x000, 0x700, 0x900, 0xB00];

/// Floor table `floortab[floorcod]` (A/52 §7.2.2.5, Table 7.20), added back after
/// the `& 0x1FE0` quantization in §7.2.2.10. The masking floor is monotonically
/// **decreasing** with floorcod (a lower floor ⇒ more bits allocated); the last
/// entry `0x0800` is the two's-complement value **−2048**, the lowest floor
/// (most bits). Reading it as +2048 would make floorcod 7 the *highest* floor,
/// inverting the table and collapsing the allocation to near-silence.
pub const FLOOR: [i32; 8] = [
    0x2F0, 0x2B0, 0x270, 0x230, 0x1F0, 0x170, 0x0F0, -2048,
];

/// Fast gain table `fgaintab[fgaincod]` (A/52 §7.2.2.5). Indexed by the 3-bit
/// per-channel `fgaincod` (fine-grain SNR offset).
pub const FAST_GAIN: [u16; 8] = [
    0x080, 0x100, 0x180, 0x200, 0x280, 0x300, 0x380, 0x400,
];

/// Banding structure: `bndtab[band]` = starting bin of each of the 50 bit
/// allocation bands (A/52 §7.2.2.3, Table 7.5 "Banding structure"). The
/// companion `bndsz[band]` (band width) is derived below.
pub const BAND_START: [u8; 50] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 31, 34, 37, 40, 43, 46, 49, 55, 61, 67, 73, 79, 85, 97, 109, 121, 133, 157, 181,
    205, 229,
];

/// Band width `bndsz[band]` = number of bins in each of the 50 bands
/// (A/52 §7.2.2.3, Table 7.5). This MUST equal the consecutive difference of
/// [`BAND_START`] (the last band runs up to the 253-bin transform limit), or the
/// §7.2.2 excitation/mask is integrated over the wrong bins and the whole
/// bit-allocation shifts. The width breaks fall on the *first* band of each new
/// width: band **28** (start bin 28) is the first width-3 band, band **35** (bin
/// 49) the first width-6, band **41** (bin 85) the first width-12, and band **45**
/// (bin 133) the first of five width-24 bands (…229–252). So the run is 28×1,
/// 7×3, 6×6, 4×12, 5×24 = 253 bins.
pub const BAND_SIZE: [u8; 50] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3,
    3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 12, 12, 12, 12, 24, 24, 24, 24,
];

/// `bin_to_band[bin]` — the band each of the 256 possible bins belongs to,
/// precomputed from [`BAND_START`] / [`BAND_SIZE`] at first use (A/52 §7.2.2.3).
/// Built lazily since Rust const-fn table generation would be noisier than the
/// `OnceLock` here.
pub fn bin_to_band() -> &'static [u8; 256] {
    use std::sync::OnceLock;
    static T: OnceLock<[u8; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u8; 256];
        for band in 0..50 {
            let start = BAND_START[band] as usize;
            let size = BAND_SIZE[band] as usize;
            for b in start..(start + size).min(256) {
                t[b] = band as u8;
            }
        }
        // Bins past the last band map to the last band (only used defensively).
        t
    })
}

/// Hearing threshold table `hth[fscod][band]` (A/52 §7.2.2.4, Table 7.6, "Hearing
/// threshold"). 50 bands × 3 sample rates. Values in the standard's fixed-point
/// scale. Transcribed from A/52 Table 7.6.
#[rustfmt::skip]
pub const HEARING_THRESHOLD: [[i16; 50]; 3] = [
    // fscod = 0 (48 kHz)
    [
        0x04D0, 0x04D0, 0x0440, 0x0400, 0x03E0, 0x03C0, 0x03B0, 0x03B0, 0x03A0, 0x03A0,
        0x03A0, 0x03A0, 0x03A0, 0x0390, 0x0390, 0x0390, 0x0380, 0x0380, 0x0370, 0x0370,
        0x0360, 0x0360, 0x0350, 0x0350, 0x0340, 0x0340, 0x0330, 0x0320, 0x0310, 0x0300,
        0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0,
        0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x0300, 0x0310, 0x0340, 0x0390, 0x03E0, 0x0420,
    ],
    // fscod = 1 (44.1 kHz)
    [
        0x04F0, 0x04F0, 0x0460, 0x0410, 0x03E0, 0x03D0, 0x03C0, 0x03B0, 0x03B0, 0x03A0,
        0x03A0, 0x03A0, 0x03A0, 0x03A0, 0x0390, 0x0390, 0x0390, 0x0380, 0x0380, 0x0370,
        0x0370, 0x0360, 0x0360, 0x0350, 0x0350, 0x0340, 0x0340, 0x0330, 0x0320, 0x0310,
        0x0300, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0,
        0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x0300, 0x0310, 0x0330, 0x0350, 0x03C0,
    ],
    // fscod = 2 (32 kHz)
    [
        0x0580, 0x0580, 0x04B0, 0x0450, 0x0420, 0x03F0, 0x03E0, 0x03D0, 0x03C0, 0x03B0,
        0x03B0, 0x03B0, 0x03A0, 0x03A0, 0x03A0, 0x03A0, 0x03A0, 0x03A0, 0x03A0, 0x03A0,
        0x0390, 0x0390, 0x0390, 0x0390, 0x0380, 0x0380, 0x0380, 0x0370, 0x0370, 0x0360,
        0x0360, 0x0350, 0x0350, 0x0340, 0x0340, 0x0330, 0x0320, 0x0310, 0x0300, 0x02F0,
        0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0, 0x02F0,
    ],
];

/// Bit allocation pointer → mask conversion "log-add" table `latab[]`
/// (A/52 §7.2.2.6, Table 7.7). 260 entries. Used to combine two dB values in
/// the log domain. Transcribed from A/52 Table 7.7.
#[rustfmt::skip]
pub const LATAB: [u16; 260] = [
    0x0040, 0x003f, 0x003e, 0x003d, 0x003c, 0x003b, 0x003a, 0x0039, 0x0038, 0x0037,
    0x0036, 0x0035, 0x0034, 0x0034, 0x0033, 0x0032, 0x0031, 0x0030, 0x002f, 0x002f,
    0x002e, 0x002d, 0x002c, 0x002c, 0x002b, 0x002a, 0x0029, 0x0029, 0x0028, 0x0027,
    0x0026, 0x0026, 0x0025, 0x0024, 0x0024, 0x0023, 0x0023, 0x0022, 0x0021, 0x0021,
    0x0020, 0x0020, 0x001f, 0x001e, 0x001e, 0x001d, 0x001d, 0x001c, 0x001c, 0x001b,
    0x001b, 0x001a, 0x001a, 0x0019, 0x0019, 0x0018, 0x0018, 0x0017, 0x0017, 0x0016,
    0x0016, 0x0015, 0x0015, 0x0015, 0x0014, 0x0014, 0x0013, 0x0013, 0x0013, 0x0012,
    0x0012, 0x0012, 0x0011, 0x0011, 0x0011, 0x0010, 0x0010, 0x0010, 0x000f, 0x000f,
    0x000f, 0x000e, 0x000e, 0x000e, 0x000d, 0x000d, 0x000d, 0x000d, 0x000c, 0x000c,
    0x000c, 0x000c, 0x000b, 0x000b, 0x000b, 0x000b, 0x000a, 0x000a, 0x000a, 0x000a,
    0x000a, 0x0009, 0x0009, 0x0009, 0x0009, 0x0009, 0x0008, 0x0008, 0x0008, 0x0008,
    0x0008, 0x0008, 0x0007, 0x0007, 0x0007, 0x0007, 0x0007, 0x0007, 0x0006, 0x0006,
    0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0005, 0x0005, 0x0005, 0x0005,
    0x0005, 0x0005, 0x0005, 0x0005, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004,
    0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003,
    0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0002,
    0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002,
    0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0001,
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001,
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001,
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001,
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
];

/// Bit-allocation-pointer → bits table `baptab[]` (A/52 §7.2.2.10, Table 7.8).
/// 64 entries, the address into which the SNR-offset-adjusted mask index maps to
/// a bit allocation pointer (0..=15). Transcribed from A/52 Table 7.8.
#[rustfmt::skip]
pub const BAPTAB: [u8; 64] = [
    0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6,
    6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10,
    10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14,
    14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15,
];

/// Number of quantization levels per bit-allocation-pointer, `bap` → levels
/// (A/52 Table 7.10 "Quantization" derived; bap 1,2,4 are grouped 3-in-5,
/// 3-in-7, 3-in-15). 0 = allocated 0 bits (dither/zero). Indexed by bap 0..=15.
pub const BAP_QUANT_LEVELS: [u16; 16] = [
    0, 3, 5, 7, 11, 15, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65535,
];

/// Number of mantissa bits per bit-allocation-pointer, `bap` → bits
/// (A/52 Table 7.10). For grouped baps (1,2,4) the *group* is coded (3 values in
/// one word); this table gives bits for the ungrouped ones and 0 for grouped
/// (whose bit cost is handled by the group extraction). Index by bap.
pub const BAP_BITS: [u8; 16] = [
    0, 0, 0, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16,
];
