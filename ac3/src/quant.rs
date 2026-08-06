//! Mantissa dequantization (ATSC A/52 §7.3.2). Given a bit-allocation pointer
//! (`bap`) and a quantized mantissa code, produce the normalized coefficient in
//! `[-1, 1)`. baps 1, 2, and 4 pack three mantissas into a single grouped code
//! (3-level, 5-level and 11-level respectively); baps 3, 5..=15 code each
//! mantissa directly as a two's-complement fraction (A/52 §7.3.2, Table 7.10 and
//! the symmetric/asymmetric quantizer definitions).
//!
//! Dither (A/52 §7.3.1): a `bap == 0` bin is either zero or replaced with a
//! pseudo-random dither value when that channel's `dithflag` is set. The dither
//! generator is the 16-bit Galois LFSR of A/52 §7.3.1.

/// The 3 reconstruction levels for a bap==1 (3-level) mantissa, as f32 fractions
/// (A/52 §7.3.2: symmetric quantizer, `(2*code - 2)/3` scaled). Values −2/3, 0,
/// 2/3.
pub const QUANT_3: [f32; 3] = [-2.0 / 3.0, 0.0, 2.0 / 3.0];

/// The 5 reconstruction levels for bap==2 (5-level) (A/52 §7.3.2): −4/5 … 4/5.
pub const QUANT_5: [f32; 5] = [-4.0 / 5.0, -2.0 / 5.0, 0.0, 2.0 / 5.0, 4.0 / 5.0];

/// The 7 levels for bap==3 (A/52 §7.3.2): −6/7 … 6/7, in steps of 2/7.
pub const QUANT_7: [f32; 7] =
    [-6.0 / 7.0, -4.0 / 7.0, -2.0 / 7.0, 0.0, 2.0 / 7.0, 4.0 / 7.0, 6.0 / 7.0];

/// The 11 levels for bap==4 (A/52 §7.3.2): −10/11 … 10/11.
pub const QUANT_11: [f32; 11] = [
    -10.0 / 11.0,
    -8.0 / 11.0,
    -6.0 / 11.0,
    -4.0 / 11.0,
    -2.0 / 11.0,
    0.0,
    2.0 / 11.0,
    4.0 / 11.0,
    6.0 / 11.0,
    8.0 / 11.0,
    10.0 / 11.0,
];

/// The 15 levels for bap==5 (A/52 §7.3.2): −14/15 … 14/15.
pub const QUANT_15: [f32; 15] = [
    -14.0 / 15.0,
    -12.0 / 15.0,
    -10.0 / 15.0,
    -8.0 / 15.0,
    -6.0 / 15.0,
    -4.0 / 15.0,
    -2.0 / 15.0,
    0.0,
    2.0 / 15.0,
    4.0 / 15.0,
    6.0 / 15.0,
    8.0 / 15.0,
    10.0 / 15.0,
    12.0 / 15.0,
    14.0 / 15.0,
];

/// Bits per grouped code word (A/52 §7.3.5, Table 7.18 "Mapping of bap to
/// group/mantissa bits"): bap 1 packs **3** mantissas in `3^3 = 27` values over
/// **5** bits; bap 2 packs **3** in `5^3 = 125` over **7** bits; bap 4 packs
/// **2** in `11^2 = 121` over **7** bits. Index by bap (only 1,2,4 are grouped).
pub const GROUP_BITS: [u8; 5] = [0, 5, 7, 0, 7];

/// Number of mantissas packed into one grouped code word, by bap (1→3, 2→3, 4→2)
/// (A/52 §7.3.5). This is the count of coefficients consumed before the next word
/// must be read — bap 4 is **2 per group**, not 3 (the classic ungroup trap:
/// 11^3 = 1331 would not fit in 7 bits, but 11^2 = 121 does).
pub const GROUP_COUNT: [usize; 5] = [0, 3, 3, 0, 2];

/// Number of levels for a grouped bap (base of the mixed-radix code). Index by
/// bap (1→3, 2→5, 4→11); other entries unused.
pub const GROUP_LEVELS: [u32; 5] = [0, 3, 5, 0, 11];

/// Dequantize a *direct* (ungrouped) mantissa code for bap 3, 5..=15.
///
/// For bap 3 the 7-level table is used. For bap 5 the 15-level table. For
/// bap 6..=15 the code is a `qntztab[bap]`-bit two's-complement fraction scaled
/// to `[-1, 1)` (A/52 §7.3.2: `mantissa = code * 2^-(qbits-1)` sign-corrected).
///
/// `code` is the raw unsigned field read from the stream; `bits` its width.
#[inline]
pub fn dequant_direct(bap: u8, code: u32, bits: u8) -> f32 {
    match bap {
        3 => QUANT_7[(code as usize) % 7],
        5 => QUANT_15[(code as usize) % 15],
        _ => {
            // bap 6..=15: `bits`-wide two's-complement fraction in [-1, 1).
            // Sign-extend `code` from `bits` to i32, then divide by 2^(bits-1).
            let shift = 32 - u32::from(bits);
            let signed = ((code << shift) as i32) >> shift;
            let scale = 1.0f32 / (1u32 << (bits - 1)) as f32;
            signed as f32 * scale
        }
    }
}

/// Unpack a grouped mantissa code (bap 1, 2, 4) into its constituent mantissas
/// (A/52 §7.3.5 grouped-mantissa mixed-radix decode). The **first** mantissa is
/// the most-significant digit of the base-`levels` code (A/52 §7.3.5: the group
/// value is `levels^(k-1)*m[0] + … + m[k-1]`), so it is extracted by dividing by
/// the highest power first. Returns 3 f32 fractions; for bap 4 (2 per group) only
/// the first two are meaningful and the caller consumes `GROUP_COUNT[bap]`.
#[inline]
pub fn dequant_group(bap: u8, code: u32) -> [f32; 3] {
    let levels = GROUP_LEVELS[bap as usize];
    let count = GROUP_COUNT[bap as usize];
    let tab: &[f32] = match bap {
        1 => &QUANT_3,
        2 => &QUANT_5,
        4 => &QUANT_11,
        _ => &QUANT_3, // unreachable in practice; keep total
    };
    // MSD-first mixed-radix extraction: mant[0] = code / levels^(count-1), etc.
    let mut out = [0.0f32; 3];
    let mut rem = code;
    let mut place = levels.pow((count - 1) as u32);
    for slot in out.iter_mut().take(count) {
        let digit = (rem / place) as usize;
        *slot = tab[digit % tab.len()];
        rem %= place;
        if place > 1 {
            place /= levels;
        }
    }
    out
}

/// The 16-bit Galois LFSR dither generator (A/52 §7.3.1). Seeded once per
/// decoder; advanced one step per dithered (`bap == 0`, `dithflag == 1`) bin.
/// Produces a value in `[-1, 1)` for substitution into a zero-bit coefficient.
pub struct Dither {
    state: u32,
}

impl Default for Dither {
    fn default() -> Self {
        Self::new()
    }
}

impl Dither {
    /// Fresh generator. A/52 §7.3.1 does not mandate a specific seed for
    /// bit-exactness (dither is decoder-defined pseudo-random content beneath the
    /// masking floor); we use the conventional 0 initial state and the standard
    /// polynomial.
    pub fn new() -> Self {
        Self { state: 0 }
    }

    /// Advance the LFSR and return the next dither sample in `[-1, 1)`
    /// (A/52 §7.3.1: `x^16 + x^15 + x^13 + x^4 + 1` taps, symmetric two's
    /// complement 16-bit output). Named `sample` (not `next`) to avoid confusion
    /// with the `Iterator` trait — this is a stateful generator, not an iterator.
    #[inline]
    pub fn sample(&mut self) -> f32 {
        // Two 8-bit advances give a fresh 16-bit word (matching the A/52
        // dither_gen 2-nibble step). Taps per the standard's polynomial.
        let mut s = self.state;
        for _ in 0..16 {
            let bit = ((s >> 15) ^ (s >> 14) ^ (s >> 12) ^ (s >> 3)) & 1;
            s = ((s << 1) | bit) & 0xFFFF;
        }
        self.state = s;
        // Map the 16-bit LFSR word to a signed fraction in [-1, 1).
        let signed = s as i16;
        f32::from(signed) / 32768.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_bap_symmetric_tables() {
        assert!((dequant_direct(3, 0, 3) - (-6.0 / 7.0)).abs() < 1e-6);
        assert!((dequant_direct(3, 3, 3) - 0.0).abs() < 1e-6);
        assert!((dequant_direct(5, 7, 4) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn direct_bap_twos_complement() {
        // bap 15 = 16-bit signed fraction. code 0 → 0. Max positive.
        assert!((dequant_direct(15, 0, 16) - 0.0).abs() < 1e-9);
        // 0x8000 = -1.0 (most negative).
        assert!((dequant_direct(15, 0x8000, 16) - (-1.0)).abs() < 1e-6);
        // 0x4000 = +0.5
        assert!((dequant_direct(15, 0x4000, 16) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn grouped_mixed_radix() {
        // bap 1 (3 levels), MSD-first: code = 9*m0 + 3*m1 + m2.
        // code 0 → all -2/3. code 26 = 9*2+3*2+2 → all +2/3.
        let g0 = dequant_group(1, 0);
        assert!((g0[0] - (-2.0 / 3.0)).abs() < 1e-6);
        let g26 = dequant_group(1, 26);
        for v in g26 {
            assert!((v - 2.0 / 3.0).abs() < 1e-6);
        }
        // code 13 = 9*1 + 3*1 + 1 → all 0 (level index 1 = 0.0 in each slot).
        let g13 = dequant_group(1, 13);
        for v in g13 {
            assert!(v.abs() < 1e-6);
        }
        // Ordering: code 9 = 9*1 + 3*0 + 0 → (0.0, -2/3, -2/3) with m0 MSD.
        let g9 = dequant_group(1, 9);
        assert!((g9[0] - 0.0).abs() < 1e-6);
        assert!((g9[1] - (-2.0 / 3.0)).abs() < 1e-6);
        assert!((g9[2] - (-2.0 / 3.0)).abs() < 1e-6);
        // bap 4 packs 2 mantissas base 11: code = 11*m0 + m1. code 11 → (idx1, idx0).
        let g = dequant_group(4, 11);
        assert!((g[0] - QUANT_11[1]).abs() < 1e-6);
        assert!((g[1] - QUANT_11[0]).abs() < 1e-6);
    }

    #[test]
    fn dither_bounded() {
        let mut d = Dither::new();
        for _ in 0..1000 {
            let v = d.sample();
            assert!((-1.0..1.0).contains(&v), "dither out of range: {v}");
        }
    }
}
