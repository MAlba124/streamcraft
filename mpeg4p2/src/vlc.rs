//! Variable-length code decoding for MPEG-4 Visual (ISO/IEC 14496-2 Annex B). The
//! bitstream codes macroblock types (MCBPC), coded-block patterns (CBPY), motion
//! vectors, DC size classes, and DCT coefficients (TCOEF, the "last/run/level"
//! events) with the Huffman tables in Annex B; this module holds those tables and
//! a small matcher.
//!
//! The matcher is a straight longest-prefix scan over `(code, len, symbol)`
//! entries: for each table we try lengths shortest-first and compare the peeked
//! bits. Annex-B codes are prefix-free, so the first length that matches is the
//! symbol. It is O(entries) per code — not the fastest possible, but the tables
//! are tiny and the hot cost is the IDCT/MC, not VLC (noted as a possible future
//! table-driven speedup). Every lookup is bounded and reports "no match" rather
//! than reading past the buffer.
//!
//! Lint note: the tables use `x | 0` (documenting a zero-valued sub-field, e.g.
//! `(mb_type << 2) | cbpc` with `cbpc == 0`) and irregular binary-digit groupings
//! (VLC codes are odd-length by nature), so `identity_op` and
//! `unusual_byte_groupings` are allowed for this table-heavy module.
#![allow(clippy::identity_op, clippy::unusual_byte_groupings, clippy::eq_op)]

use crate::bits::BitReader;

/// One Huffman entry: the code bit-pattern (right-justified in `code`), its bit
/// length, and the decoded symbol payload.
#[derive(Clone, Copy)]
pub struct VlcEntry {
    pub code: u32,
    pub len: u8,
    pub sym: i32,
}

/// Match the next bits against `table`, consuming the matched code. Returns the
/// symbol, or `None` if no entry matches (a malformed stream — the caller
/// warns/drops). Codes are tried in the table's order; put shorter codes first is
/// not required because we compare against the exact peeked `len` bits.
pub fn decode(r: &mut BitReader, table: &[VlcEntry]) -> Option<i32> {
    // Group by length: peek the maximum length once, then test each entry by
    // masking to its length. Prefix-freeness guarantees at most one match.
    let maxlen = table.iter().map(|e| e.len).max().unwrap_or(0) as u32;
    if maxlen == 0 {
        return None;
    }
    let peek = r.peek_bits(maxlen);
    for e in table {
        let shift = maxlen - e.len as u32;
        if (peek >> shift) == e.code {
            r.skip(e.len as usize);
            return Some(e.sym);
        }
    }
    None
}

/// MCBPC for I-VOP macroblocks (§Annex B, Table B-6). Symbol encodes
/// `(mb_type<<4) | cbpc` where cbpc is the two chroma coded-block-pattern bits.
/// For I-VOP, mb_type ∈ {3 (Intra), 4 (Intra+Q)}; here we pack cbpc in the low 2
/// bits and mb_type in bits 2..4.
#[rustfmt::skip]
pub static MCBPC_INTRA: [VlcEntry; 9] = [
    // code, len, sym = (mb_type<<2)|cbpc  ; mb_type: 3=intra 4=intra_q
    VlcEntry { code: 0b1,       len: 1, sym: (3 << 2) | 0 },
    VlcEntry { code: 0b001,     len: 3, sym: (3 << 2) | 1 },
    VlcEntry { code: 0b010,     len: 3, sym: (3 << 2) | 2 },
    VlcEntry { code: 0b011,     len: 3, sym: (3 << 2) | 3 },
    VlcEntry { code: 0b0001,    len: 4, sym: (4 << 2) | 0 },
    VlcEntry { code: 0b0000_01, len: 6, sym: (4 << 2) | 1 },
    VlcEntry { code: 0b0000_10, len: 6, sym: (4 << 2) | 2 },
    VlcEntry { code: 0b0000_11, len: 6, sym: (4 << 2) | 3 },
    // stuffing code 000000 1 (9 bits) is handled by the caller; not a MB type.
    VlcEntry { code: 0b0000_0000_1, len: 9, sym: -1 }, // stuffing marker
];

/// MCBPC for P-VOP macroblocks (§Annex B, Table B-7). Symbol packs
/// `(mb_type<<2)|cbpc`. mb_type: 0=inter,1=inter_q,2=inter4v,3=intra,4=intra_q.
#[rustfmt::skip]
pub static MCBPC_INTER: [VlcEntry; 21] = [
    VlcEntry { code: 0b1,            len: 1,  sym: (0 << 2) | 0 },
    VlcEntry { code: 0b0011,         len: 4,  sym: (0 << 2) | 1 },
    VlcEntry { code: 0b0010,         len: 4,  sym: (0 << 2) | 2 },
    VlcEntry { code: 0b0001_01,      len: 6,  sym: (0 << 2) | 3 },
    VlcEntry { code: 0b011,          len: 3,  sym: (1 << 2) | 0 },
    VlcEntry { code: 0b0000_0011_1,  len: 9,  sym: (1 << 2) | 1 },
    VlcEntry { code: 0b0000_0011_0,  len: 9,  sym: (1 << 2) | 2 },
    VlcEntry { code: 0b0000_0001_00, len: 10, sym: (1 << 2) | 3 },
    VlcEntry { code: 0b010,          len: 3,  sym: (2 << 2) | 0 },
    VlcEntry { code: 0b0000_101,     len: 7,  sym: (2 << 2) | 1 },
    VlcEntry { code: 0b0000_100,     len: 7,  sym: (2 << 2) | 2 },
    VlcEntry { code: 0b0001_00,      len: 6,  sym: (2 << 2) | 3 },
    VlcEntry { code: 0b0001_1,       len: 5,  sym: (3 << 2) | 0 },
    VlcEntry { code: 0b0000_0010_1,  len: 9,  sym: (3 << 2) | 1 },
    VlcEntry { code: 0b0000_0010_0,  len: 9,  sym: (3 << 2) | 2 },
    VlcEntry { code: 0b0000_0001_1,  len: 9,  sym: (3 << 2) | 3 },
    VlcEntry { code: 0b0000_1,       len: 5,  sym: (4 << 2) | 0 },
    VlcEntry { code: 0b0000_0000_101,len: 11, sym: (4 << 2) | 1 },
    VlcEntry { code: 0b0000_0000_100,len: 11, sym: (4 << 2) | 2 },
    VlcEntry { code: 0b0000_0001_01, len: 10, sym: (4 << 2) | 3 },
    VlcEntry { code: 0b0000_0000_1,  len: 9,  sym: -1 }, // stuffing
];

/// CBPY (§Annex B, Table B-8) — the luma coded-block-pattern for the 4 luma
/// blocks. For intra MBs the symbol is the 4-bit pattern directly; for inter MBs
/// the table's value is bit-inverted (handled by the caller). Symbol == cbpy
/// (0..15) as decoded for INTRA. Entries here are the intra ordering.
#[rustfmt::skip]
pub static CBPY: [VlcEntry; 16] = [
    VlcEntry { code: 0b0011,       len: 4, sym: 0  },
    VlcEntry { code: 0b0010_1,     len: 5, sym: 1  },
    VlcEntry { code: 0b0010_0,     len: 5, sym: 2  },
    VlcEntry { code: 0b1001,       len: 4, sym: 3  },
    VlcEntry { code: 0b0001_1,     len: 5, sym: 4  },
    VlcEntry { code: 0b0111,       len: 4, sym: 5  },
    VlcEntry { code: 0b0000_10,    len: 6, sym: 6  },
    VlcEntry { code: 0b1011,       len: 4, sym: 7  },
    VlcEntry { code: 0b0001_0,     len: 5, sym: 8  },
    VlcEntry { code: 0b0000_11,    len: 6, sym: 9  },
    VlcEntry { code: 0b0101,       len: 4, sym: 10 },
    VlcEntry { code: 0b1010,       len: 4, sym: 11 },
    VlcEntry { code: 0b0100,       len: 4, sym: 12 },
    VlcEntry { code: 0b1000,       len: 4, sym: 13 },
    VlcEntry { code: 0b0110,       len: 4, sym: 14 },
    VlcEntry { code: 0b11,         len: 2, sym: 15 },
];

/// Motion-vector residual VLC (§Annex B, Table B-12). Maps a code to the *index*
/// into the ±magnitude table; the caller converts the index to a signed delta and
/// applies the sign bit for non-zero magnitudes. Symbol == the index 0..=32 where
/// 0 corresponds to MV difference 0 and higher indices to larger magnitudes; the
/// caller maps index→(magnitude, needs_sign). We store the *signed* code value
/// directly (the table is symmetric): sym is the vector residual before residual
/// scaling, in the range −16..=16 with the sign already resolved by the table
/// half. To keep this compact we store the unsigned magnitude index and a sign
/// bit is read separately by the caller for magnitude != 0.
#[rustfmt::skip]
pub static MV: [VlcEntry; 33] = [
    VlcEntry { code: 0b1,                  len: 1,  sym: 0 },
    VlcEntry { code: 0b010,                len: 3,  sym: 1 },
    VlcEntry { code: 0b0010,               len: 4,  sym: 2 },
    VlcEntry { code: 0b0001_0,             len: 5,  sym: 3 },
    VlcEntry { code: 0b0000_110,           len: 7,  sym: 4 },
    VlcEntry { code: 0b0000_1010,          len: 8,  sym: 5 },
    VlcEntry { code: 0b0000_1000,          len: 8,  sym: 6 },
    VlcEntry { code: 0b0000_0110,          len: 8,  sym: 7 },
    VlcEntry { code: 0b0000_0101_10,       len: 10, sym: 8 },
    VlcEntry { code: 0b0000_0101_00,       len: 10, sym: 9 },
    VlcEntry { code: 0b0000_0100_10,       len: 10, sym: 10 },
    VlcEntry { code: 0b0000_0100_010,      len: 11, sym: 11 },
    VlcEntry { code: 0b0000_0100_000,      len: 11, sym: 12 },
    VlcEntry { code: 0b0000_0011_110,      len: 11, sym: 13 },
    VlcEntry { code: 0b0000_0011_100,      len: 11, sym: 14 },
    VlcEntry { code: 0b0000_0011_010,      len: 11, sym: 15 },
    VlcEntry { code: 0b0000_0011_000,      len: 11, sym: 16 },
    VlcEntry { code: 0b0000_0010_110,      len: 11, sym: 17 },
    VlcEntry { code: 0b0000_0010_100,      len: 11, sym: 18 },
    VlcEntry { code: 0b0000_0010_010,      len: 11, sym: 19 },
    VlcEntry { code: 0b0000_0010_000,      len: 11, sym: 20 },
    VlcEntry { code: 0b0000_0001_1110,     len: 12, sym: 21 },
    VlcEntry { code: 0b0000_0001_1100,     len: 12, sym: 22 },
    VlcEntry { code: 0b0000_0001_1010,     len: 12, sym: 23 },
    VlcEntry { code: 0b0000_0001_1000,     len: 12, sym: 24 },
    VlcEntry { code: 0b0000_0001_0110,     len: 12, sym: 25 },
    VlcEntry { code: 0b0000_0001_0100,     len: 12, sym: 26 },
    VlcEntry { code: 0b0000_0001_0010,     len: 12, sym: 27 },
    VlcEntry { code: 0b0000_0001_0000,     len: 12, sym: 28 },
    VlcEntry { code: 0b0000_0000_1110,     len: 12, sym: 29 },
    VlcEntry { code: 0b0000_0000_1100,     len: 12, sym: 30 },
    VlcEntry { code: 0b0000_0000_1010,     len: 12, sym: 31 },
    VlcEntry { code: 0b0000_0000_1000,     len: 12, sym: 32 },
];

/// DC size VLC for the luma component (§Annex B, Table B-13): symbol == the number
/// of additional bits to read for the intra DC differential magnitude.
#[rustfmt::skip]
pub static DC_LUM: [VlcEntry; 13] = [
    VlcEntry { code: 0b011,          len: 3, sym: 0 },
    VlcEntry { code: 0b11,           len: 2, sym: 1 },
    VlcEntry { code: 0b10,           len: 2, sym: 2 },
    VlcEntry { code: 0b010,          len: 3, sym: 3 },
    VlcEntry { code: 0b001,          len: 3, sym: 4 },
    VlcEntry { code: 0b0001,         len: 4, sym: 5 },
    VlcEntry { code: 0b0000_1,       len: 5, sym: 6 },
    VlcEntry { code: 0b0000_01,      len: 6, sym: 7 },
    VlcEntry { code: 0b0000_001,     len: 7, sym: 8 },
    VlcEntry { code: 0b0000_0001,    len: 8, sym: 9 },
    VlcEntry { code: 0b0000_0000_1,  len: 9, sym: 10 },
    VlcEntry { code: 0b0000_0000_01, len: 10, sym: 11 },
    VlcEntry { code: 0b0000_0000_001,len: 11, sym: 12 },
];

/// DC size VLC for the chroma components (§Annex B, Table B-14).
#[rustfmt::skip]
pub static DC_CHROM: [VlcEntry; 13] = [
    VlcEntry { code: 0b11,            len: 2, sym: 0 },
    VlcEntry { code: 0b10,            len: 2, sym: 1 },
    VlcEntry { code: 0b01,            len: 2, sym: 2 },
    VlcEntry { code: 0b001,          len: 3, sym: 3 },
    VlcEntry { code: 0b0001,         len: 4, sym: 4 },
    VlcEntry { code: 0b0000_1,       len: 5, sym: 5 },
    VlcEntry { code: 0b0000_01,      len: 6, sym: 6 },
    VlcEntry { code: 0b0000_001,     len: 7, sym: 7 },
    VlcEntry { code: 0b0000_0001,    len: 8, sym: 8 },
    VlcEntry { code: 0b0000_0000_1,  len: 9, sym: 9 },
    VlcEntry { code: 0b0000_0000_01, len: 10, sym: 10 },
    VlcEntry { code: 0b0000_0000_001,len: 11, sym: 11 },
    VlcEntry { code: 0b0000_0000_0001,len: 12, sym: 12 },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_bytes(bytes: &[u8], table: &[VlcEntry]) -> Option<i32> {
        let mut r = BitReader::new(bytes);
        decode(&mut r, table)
    }

    #[test]
    fn mcbpc_intra_shortest_code() {
        // "1" → mb_type 3 (intra), cbpc 0
        assert_eq!(decode_bytes(&[0b1000_0000], &MCBPC_INTRA), Some(3 << 2));
    }

    #[test]
    fn cbpy_two_bit_code() {
        // "11" → 15
        assert_eq!(decode_bytes(&[0b1100_0000], &CBPY), Some(15));
    }

    #[test]
    fn mv_zero_code() {
        assert_eq!(decode_bytes(&[0b1000_0000], &MV), Some(0));
    }

    #[test]
    fn no_match_returns_none() {
        // A code that is not a prefix of any DC_LUM entry within maxlen.
        // All-zeros of maxlen bits: DC_LUM max len is 11, and 0b0000_0000_000
        // (11 zeros) is not a listed code (the shortest all-zero prefix is
        // len-11 "0000_0000_001").
        let mut r = BitReader::new(&[0x00, 0x00]);
        assert_eq!(decode(&mut r, &DC_LUM), None);
    }
}
