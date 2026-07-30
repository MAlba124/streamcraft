//! SILK Linear Congruential Generator (LCG) seed — RFC 6716 §4.2.7.7.
//!
//! Each SILK frame carries a 2-bit pseudorandom seed (after the LTP
//! parameters, if any). The seed initialises the LCG that drives the
//! pseudorandom sign-flip in the §4.2.7.8.6 excitation-reconstruction
//! step. The seed is decoded from the uniform 4-entry PDF in Table 43.
//!
//! This module is intentionally tiny — the LCG seed is one symbol per
//! SILK frame and is consumed by the [`crate::silk_excitation`] module.

use crate::range_decoder::RangeDecoder;
use crate::Error;

/// Table 43 — uniform 4-entry PDF for the LCG seed.
///
/// PDF `{64, 64, 64, 64}/256`. Cumulative `fh = [64, 128, 192, 256]`.
/// iCDF = `256 - fh[k]` terminated by 0: `[192, 128, 64, 0]`.
pub(crate) const LCG_SEED_ICDF: &[u8] = &[192, 128, 64, 0];

/// Decode the §4.2.7.7 LCG seed from `rd`.
///
/// Returns the seed in `0..=3`. The §4.2.7.8.6 reconstruction widens
/// this to a `u32` (the LCG runs in 32-bit unsigned arithmetic) on its
/// first use.
pub fn decode_lcg_seed(rd: &mut RangeDecoder<'_>) -> u8 {
    rd.dec_icdf(LCG_SEED_ICDF, 8) as u8
}

/// Encode the §4.2.7.7 LCG seed into `re` — the write-side mirror of
/// [`decode_lcg_seed`]. `seed` must be in `0..=3`.
pub fn encode_lcg_seed(re: &mut crate::range_encoder::RangeEncoder, seed: u8) -> Result<(), Error> {
    if seed > 3 {
        return Err(Error::MalformedPacket);
    }
    re.enc_icdf(seed as usize, LCG_SEED_ICDF, 8);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table43_pdf_sums_to_256() {
        let pdf = [64u32, 64, 64, 64];
        assert_eq!(pdf.iter().sum::<u32>(), 256);
        // Self-check transcription.
        assert_eq!(LCG_SEED_ICDF, &[192u8, 128, 64, 0]);
        // Terminator zero, strictly monotone decreasing.
        for w in LCG_SEED_ICDF.windows(2) {
            assert!(w[0] > w[1]);
        }
        assert_eq!(*LCG_SEED_ICDF.last().unwrap(), 0);
    }

    #[test]
    fn decode_lcg_seed_returns_0_through_3() {
        // Sweep a handful of buffers and confirm the decoded value is
        // always in 0..=3.
        let buffers: [&[u8]; 4] = [
            &[0x00u8, 0x00, 0x00, 0x00],
            &[0xFFu8, 0xFF, 0xFF, 0xFF],
            &[0x55u8, 0xAA, 0x55, 0xAA],
            &[0xC3u8, 0x3C, 0x96, 0x69],
        ];
        for buf in buffers {
            let mut rd = RangeDecoder::new(buf);
            let seed = decode_lcg_seed(&mut rd);
            assert!(seed <= 3, "seed {seed} out of range");
        }
    }

    #[test]
    fn decode_lcg_seed_distribution_partition() {
        // Walk through the four `val` partitions {[0, 64), [64, 128),
        // [128, 192), [192, 256)} that a 2-bit uniform iCDF must induce
        // (per the §4.1.3.3 lookup against ft=256). We construct
        // synthetic decoder states by reading raw bits first to bias
        // `val`, then assert the seed lands in each quadrant at least
        // once across a sweep.
        let mut seen = [false; 4];
        // 256 candidate buffers covers the symbol distribution well.
        for byte in 0..=255u8 {
            let buf = [byte, byte ^ 0x5A, byte.wrapping_add(0x33), 0xA5];
            let mut rd = RangeDecoder::new(&buf);
            let seed = decode_lcg_seed(&mut rd);
            seen[seed as usize] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "uniform PDF should produce every symbol across 256 buffers: seen={seen:?}"
        );
    }

    #[test]
    fn decode_lcg_seed_consumes_one_symbol() {
        // Two consecutive seed decodes from the same buffer must
        // advance `tell()` (the symbol is non-trivial) and both must be
        // in 0..=3.
        let buf = [0x57u8, 0xC4, 0x9E, 0x1B, 0x86];
        let mut rd = RangeDecoder::new(&buf);
        let tell0 = rd.tell();
        let s0 = decode_lcg_seed(&mut rd);
        let tell1 = rd.tell();
        let s1 = decode_lcg_seed(&mut rd);
        let tell2 = rd.tell();
        assert!(s0 <= 3 && s1 <= 3);
        assert!(tell1 > tell0);
        assert!(tell2 > tell1);
        // tell() returns whole bits. A 4-entry uniform PDF carries
        // log2(4) = 2 bits per symbol.
        assert_eq!(tell1 - tell0, 2);
        assert_eq!(tell2 - tell1, 2);
    }
}
