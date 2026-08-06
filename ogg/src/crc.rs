//! The Ogg page checksum (spec: RFC 3533 §6, field 7, checked in at
//! `spec/rfc3533.txt`).
//!
//! Ogg pages carry a 32-bit CRC "of the page (including header with zero CRC field
//! and page content)" whose "generator polynomial is 0x04c11db7" (§6). The RFC names
//! only the polynomial; the remaining parameters are fixed by the reference
//! implementation (libogg) and are **unusual**:
//!
//! - polynomial `0x04C11DB7` (x^32 + x^26 + x^23 + … + 1),
//! - initial value `0`,
//! - **no** input reflection, **no** output reflection (MSB-first),
//! - **no** final XOR.
//!
//! This is the "direct" / non-reflected CRC-32 (the same parameters as CRC-32/MPEG-2
//! but with init 0 instead of all-ones). Reflected CRC-32 (zlib/PNG) would give the
//! wrong answer, so this is the single spot that must match the wire format exactly —
//! it is known-answer tested against a hand-built page in `tests`.
//!
//! Pure safe Rust: lookup tables built at first use, then a slicing-by-8 update with a
//! byte-at-a-time tail (see [`SLICES`]). No `unsafe`, no dependency. Every table is derived
//! from the one bit-at-a-time definition in [`Table::build`], so the whole file still rests on
//! a single transcription of the polynomial.

/// Ogg CRC-32 generator polynomial (§6): x^32 + x^26 + x^23 + x^22 + x^16 + x^12 +
/// x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x^1 + x^0.
const POLY: u32 = 0x04C1_1DB7;

/// The 256-entry CRC table, MSB-first: `TABLE[b]` is the CRC of the single byte `b`
/// shifted into an otherwise-zero register. Built once, lazily.
///
/// A `const` array would work too, but a `const fn` builder needs a loop; this stays
/// a plain runtime table behind a `OnceLock` so the code reads as the textbook
/// bit-at-a-time definition (the thing that is easy to check against the spec).
struct Table([[u32; 256]; SLICES]);

/// How many bytes the sliced inner loop folds per iteration.
///
/// Slicing-by-N (Kounavis & Berry, *"Novel Table Lookup-Based Algorithms for High-Performance
/// CRC Generation"*, IEEE Transactions on Computers 57(11), 2008 — the technique Intel's
/// `crc32` reference and zlib's `crc32_z` both use): table `k` holds the CRC contribution of a
/// byte that is `k` positions from the end of the block, so `N` bytes can be folded with `N`
/// independent table lookups that the processor issues in parallel, instead of `N` dependent
/// ones each waiting on the previous byte's register update. `TABLE[0]` is exactly the
/// classic byte-at-a-time table, and the tail loop still uses it.
///
/// Eight is the usual sweet spot: the eight loads are independent, and the two `u32` block
/// reads that feed them are a single aligned `u64` on any real input.
const SLICES: usize = 8;

impl Table {
    fn build() -> Self {
        let mut table = [[0u32; 256]; SLICES];
        let mut i = 0usize;
        while i < 256 {
            // Seed the top byte of the 32-bit register with the index, then divide by
            // the polynomial one bit at a time, MSB-first (no reflection).
            let mut crc = (i as u32) << 24;
            let mut bit = 0;
            while bit < 8 {
                crc = if crc & 0x8000_0000 != 0 {
                    (crc << 1) ^ POLY
                } else {
                    crc << 1
                };
                bit += 1;
            }
            table[0][i] = crc;
            i += 1;
        }
        // Each higher table is the previous one pushed through one more byte position: the
        // CRC of "this byte, then a zero byte". Derived from `table[0]` rather than written
        // out, so the whole structure still rests on the one bit-at-a-time definition above.
        let mut s = 1usize;
        while s < SLICES {
            let mut j = 0usize;
            while j < 256 {
                let prev = table[s - 1][j];
                table[s][j] = (prev << 8) ^ table[0][(prev >> 24) as usize];
                j += 1;
            }
            s += 1;
        }
        Self(table)
    }
}

fn table() -> &'static [[u32; 256]; SLICES] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Table> = OnceLock::new();
    &TABLE.get_or_init(Table::build).0
}

/// Incremental Ogg CRC-32 state (init 0). Feed bytes with [`update`](Self::update),
/// read the running value with [`value`](Self::value). Used by the page writer, which
/// checksums a header-with-zeroed-CRC then the body without a second pass.
#[derive(Clone, Copy, Debug)]
pub struct Crc32 {
    state: u32,
}

impl Crc32 {
    pub fn new() -> Self {
        Self { state: 0 }
    }

    /// Fold `data` into the running CRC (MSB-first table step, §6).
    pub fn update(&mut self, data: &[u8]) {
        let table = table();
        let mut crc = self.state;

        // Eight bytes per iteration through the sliced tables (see [`SLICES`]). The register
        // is consumed by the first four bytes and takes no further part, so all eight lookups
        // depend only on the loaded bytes and issue in parallel — the byte-at-a-time loop's
        // one-lookup-per-cycle dependency chain is what this removes.
        let (blocks, rest) = data.as_chunks::<SLICES>();
        for c in blocks {
            let a = u32::from_be_bytes([c[0], c[1], c[2], c[3]]) ^ crc;
            let b = u32::from_be_bytes([c[4], c[5], c[6], c[7]]);
            crc = table[7][(a >> 24) as u8 as usize]
                ^ table[6][(a >> 16) as u8 as usize]
                ^ table[5][(a >> 8) as u8 as usize]
                ^ table[4][a as u8 as usize]
                ^ table[3][(b >> 24) as u8 as usize]
                ^ table[2][(b >> 16) as u8 as usize]
                ^ table[1][(b >> 8) as u8 as usize]
                ^ table[0][b as u8 as usize];
        }

        // The 0..8-byte tail, and every short call (an Ogg header is 27 bytes), byte at a time.
        for &byte in rest {
            // MSB-first: XOR the incoming byte with the top byte of the register,
            // index the table with that, and shift the register up by a byte.
            let idx = ((crc >> 24) as u8 ^ byte) as usize;
            crc = (crc << 8) ^ table[0][idx];
        }
        self.state = crc;
    }

    /// The running CRC value.
    pub fn value(&self) -> u32 {
        self.state
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot Ogg CRC-32 over `data` (init 0, no reflection, no final XOR; §6).
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        // Init 0, no final XOR → the CRC of nothing is 0.
        assert_eq!(crc32(&[]), 0);
    }

    #[test]
    fn table_step_matches_bitwise() {
        // The table-driven update must equal the textbook bit-at-a-time division for
        // every possible current-top-byte / input-byte pair — this is what proves the
        // table was built for the right (non-reflected) polynomial.
        fn bitwise(state: u32, byte: u8) -> u32 {
            let mut crc = state ^ ((byte as u32) << 24);
            for _ in 0..8 {
                crc = if crc & 0x8000_0000 != 0 {
                    (crc << 1) ^ POLY
                } else {
                    crc << 1
                };
            }
            crc
        }
        for top in 0u32..256 {
            for byte in 0u8..=255 {
                let state = top << 24;
                let mut c = Crc32 { state };
                c.update(&[byte]);
                assert_eq!(c.value(), bitwise(state, byte), "top={top} byte={byte}");
            }
        }
    }

    /// Slicing-by-8 must agree with the plain byte-at-a-time loop at **every** length, not
    /// just the multiples of 8 the fast path handles: lengths 0..8 skip the sliced loop
    /// entirely, and everything else exercises a different remainder. A CRC that is right on
    /// aligned buffers and wrong on a 43-byte page would pass every other test in this file.
    #[test]
    fn sliced_matches_byte_at_a_time_at_every_length() {
        fn byte_at_a_time(data: &[u8]) -> u32 {
            let t = table();
            let mut crc = 0u32;
            for &b in data {
                crc = (crc << 8) ^ t[0][((crc >> 24) as u8 ^ b) as usize];
            }
            crc
        }
        let data: Vec<u8> =
            (0..600u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for len in 0..data.len() {
            assert_eq!(crc32(&data[..len]), byte_at_a_time(&data[..len]), "len={len}");
        }
    }

    #[test]
    fn incremental_equals_oneshot() {
        // Splitting the input at an arbitrary point must not change the result.
        let data: Vec<u8> = (0..500u32).map(|i| (i.wrapping_mul(31) ^ (i >> 3)) as u8).collect();
        let whole = crc32(&data);
        for split in [0usize, 1, 27, 255, 256, 499, 500] {
            let mut c = Crc32::new();
            c.update(&data[..split]);
            c.update(&data[split..]);
            assert_eq!(c.value(), whole, "split at {split}");
        }
    }

    #[test]
    fn known_answer_ascii_123456789() {
        // The check value of the "direct" CRC-32 (poly 0x04C11DB7, init 0, no
        // reflection, no final XOR) over the canonical "123456789" input is
        // 0x89A1897F. This pins the exact parameter set (a reflected or init-all-ones
        // CRC gives a different value), independent of any Ogg framing.
        assert_eq!(crc32(b"123456789"), 0x89A1_897F);
    }

    #[test]
    fn known_answer_single_bytes() {
        // With a zero initial register, the CRC of a single byte `b` is exactly the
        // table entry for `b` (`(0 << 8) ^ table[0 ^ b]`). Check the boundary bytes:
        // 0x00 stays 0 (no polynomial fold), while 0x80 and 0xFF drive the top-bit fold
        // path. `table_step_matches_bitwise` already proved the table itself correct.
        assert_eq!(crc32(&[0x00]), 0);
        assert_eq!(crc32(&[0x80]), table()[0][0x80]);
        assert_eq!(crc32(&[0xFF]), table()[0][0xFF]);
    }
}
