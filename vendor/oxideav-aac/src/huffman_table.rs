//! Canonical-prefix-code decode tables (streamcraft patch — see `STREAMCRAFT-PATCHES.md`).
//!
//! The §4.A spectrum Huffman codebooks (`hcod1..hcod11`) and the §4.A.1 scalefactor codebook
//! (`hcod_sf`) were decoded by a **linear scan**: for each of up to `max_len` bit-reads, walk
//! the whole codebook (81–121 entries) looking for a `(length, codeword)` match. That is
//! `O(entries · max_len)` per symbol — for `hcod_sf` up to `121 · 19 ≈ 2300` comparisons per
//! scalefactor — and the spectral/scalefactor Huffman decode dominated audio-decode CPU in
//! profiling once the video path was zero-copy and the allocation churn was gone.
//!
//! This replaces that with the standard **peek-and-lookup** decode. A prefix code has the
//! property that the next `max_len` bits uniquely identify the codeword (its length-`L`
//! codeword sits in the high `L` bits; the remaining bits are "don't care"). So a table of
//! `2^max_len` entries — each holding the `(length, symbol index)` for that prefix — turns a
//! decode into: peek `max_len` bits, index the table, consume the codeword's length. `O(1)` per
//! symbol. Tables are built once per codebook (lazily, `OnceLock`) from the *same* codebook
//! data the linear scan used, so every decoded symbol is identical — the `hcodN_is_complete`
//! and `pcm_byte_exact` conformance tests gate it, plus a per-book equivalence test.

use oxideav_core::bits::BitReader;

use crate::{Error, Result};

/// A decode table for one canonical prefix code, indexed by the next `max_len` bits.
#[derive(Debug)]
pub struct PrefixTable {
    /// `entries[key]` packs the codeword length and symbol index for the `max_len`-bit prefix
    /// `key` as `(length << 16) | index`. A `length` of `0` marks a prefix that begins no
    /// codeword (only possible for an *incomplete* code; a complete code fills every slot).
    entries: Vec<u32>,
    /// The longest codeword in the book — the table's index width.
    max_len: u32,
}

impl PrefixTable {
    /// Build the table from a codebook's `(length, codeword)` rows. `codeword` is right-aligned
    /// (the value the bit-walk accumulates after reading `length` bits MSB-first), exactly as
    /// the `HCOD*` statics store it. `L`/`C` are the row's integer types (`u8`/`u16`/`u32`).
    ///
    /// Each row claims the contiguous block of `max_len`-bit prefixes whose high `length` bits
    /// equal its codeword — `2^(max_len − length)` slots starting at `codeword << (max_len −
    /// length)`. For a complete prefix code these blocks tile `0..2^max_len` exactly.
    pub fn build<L, C>(codebook: &[(L, C)], max_len: u32) -> PrefixTable
    where
        L: Copy + Into<u32>,
        C: Copy + Into<u32>,
    {
        debug_assert!(max_len >= 1 && max_len <= 24, "prefix table width out of range");
        let mut entries = vec![0u32; 1usize << max_len];
        for (idx, &(len, cw)) in codebook.iter().enumerate() {
            let len: u32 = len.into();
            let cw: u32 = cw.into();
            debug_assert!(len >= 1 && len <= max_len);
            let shift = max_len - len;
            let base = (cw << shift) as usize;
            let packed = (len << 16) | (idx as u32);
            for slot in &mut entries[base..base + (1usize << shift)] {
                *slot = packed;
            }
        }
        PrefixTable { entries, max_len }
    }

    /// Decode one symbol, returning its codebook index. Consumes exactly the codeword's bits.
    ///
    /// Near the end of the bitstream fewer than `max_len` bits may remain; we peek what is
    /// available and left-justify it into the `max_len`-wide key (zero-padding the missing low
    /// bits). Because a valid codeword occupies only the high bits, this resolves correctly as
    /// long as the codeword fits in the available bits — otherwise (`length > available`, or an
    /// unfilled slot) the stream is truncated/invalid and we report [`Error::UnexpectedEnd`],
    /// matching the linear scan's failure on a short/rogue tail.
    #[inline]
    pub fn decode(&self, reader: &mut BitReader<'_>) -> Result<u32> {
        let avail = reader.bits_remaining().min(u64::from(self.max_len)) as u32;
        if avail == 0 {
            return Err(Error::UnexpectedEnd);
        }
        let peeked = reader.peek_u32(avail).map_err(|_| Error::UnexpectedEnd)?;
        let key = (peeked << (self.max_len - avail)) as usize;
        let packed = self.entries[key];
        let len = packed >> 16;
        if len == 0 || len > avail {
            return Err(Error::UnexpectedEnd);
        }
        reader.consume(len).map_err(|_| Error::UnexpectedEnd)?;
        Ok(packed & 0xFFFF)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    // A tiny complete prefix code (max_len = 3): a canonical Huffman set.
    //   idx 0 -> "0"      (len 1, cw 0b0)
    //   idx 1 -> "10"     (len 2, cw 0b10)
    //   idx 2 -> "110"    (len 3, cw 0b110)
    //   idx 3 -> "111"    (len 3, cw 0b111)
    const BOOK: [(u8, u16); 4] = [(1, 0b0), (2, 0b10), (3, 0b110), (3, 0b111)];
    const MAX: u32 = 3;

    fn encode(indices: &[usize]) -> Vec<u8> {
        let mut w = BitWriter::new();
        for &i in indices {
            let (len, cw) = BOOK[i];
            w.write_u32(u32::from(cw), u32::from(len));
        }
        w.into_bytes() // pads to a byte boundary with zero bits
    }

    #[test]
    fn decodes_each_codeword_and_advances_correctly() {
        let table = PrefixTable::build(&BOOK, MAX);
        let seq = [0usize, 3, 1, 2, 0, 0, 1];
        let bytes = encode(&seq);
        let mut r = BitReader::new(&bytes);
        for &want in &seq {
            assert_eq!(table.decode(&mut r).unwrap(), want as u32);
        }
    }

    #[test]
    fn short_codewords_at_end_of_stream_use_the_left_justify_path_then_eof() {
        // A byte of all-zero bits is eight one-bit "0" codewords (idx 0). Decoding the last few
        // exercises the `avail < max_len` path (fewer than 3 bits remain, zero-padded into the
        // key); once the byte is drained the next decode must report end-of-stream, not a bogus
        // symbol.
        let table = PrefixTable::build(&BOOK, MAX);
        let bytes = [0b0000_0000u8];
        let mut r = BitReader::new(&bytes);
        for _ in 0..8 {
            assert_eq!(table.decode(&mut r).unwrap(), 0);
        }
        assert!(matches!(table.decode(&mut r), Err(Error::UnexpectedEnd)));
    }
}
