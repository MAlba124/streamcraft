//! Big-endian bit I/O and the FLAC checksums (spec: RFC 9639, checked in at
//! `spec/rfc9639.txt`).
//!
//! FLAC is a bit-packed format: metadata fields, frame/subframe headers, and the
//! Rice-coded residual all straddle byte boundaries, MSB first (§6, §9.2). This
//! module is the single place that touches individual bits, so the codec above it
//! reads as "write a `u(5)` here, a unary quotient there" against the normative
//! text. It is pure `std`, no `unsafe` — all the twiddling is safe shifts and masks.
//!
//! Two checksums live here as well, because they are computed over the *byte* stream
//! the writer emits and the reader consumes:
//! - **CRC-8** over the frame header, polynomial x^8 + x^2 + x^1 + x^0 (§9.1.8).
//! - **CRC-16** over the whole frame, polynomial x^16 + x^15 + x^2 + x^0 (§9.3).
//! Both are initialised with 0 and are plain MSB-first CRCs (no reflection, no final
//! XOR), matching the worked examples in Appendix D.

/// Big-endian bit writer: bits accumulate MSB-first into a byte buffer.
///
/// A partial trailing byte is held in `bit_buf` / `bit_count`; [`align`](Self::align)
/// or [`into_bytes`](Self::into_bytes) flushes it, zero-padding to the next byte
/// boundary as FLAC frames require (§9.3 "zero bits are added until byte alignment").
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Up to 7 not-yet-emitted bits, left-justified is *not* used; instead the low
    /// `bit_count` bits of `bit_buf` hold the pending bits in order (MSB = first).
    bit_buf: u64,
    bit_count: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    /// Total bits written so far (emitted bytes plus the partial byte).
    pub fn bit_len(&self) -> u64 {
        self.bytes.len() as u64 * 8 + self.bit_count as u64
    }

    /// True when the next write starts on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_count == 0
    }

    /// Write the low `n` bits of `value`, MSB first. `n` in `0..=64`.
    pub fn write_bits(&mut self, value: u64, n: u32) {
        debug_assert!(n <= 64);
        if n == 0 {
            return;
        }
        // Mask to the requested width so stray high bits never corrupt the stream.
        let value = if n == 64 { value } else { value & ((1u64 << n) - 1) };
        let mut remaining = n;
        while remaining > 0 {
            let take = remaining.min(8 - self.bit_count);
            // The next `take` bits, counting from the top of the still-unwritten field.
            let shift = remaining - take;
            let chunk = (value >> shift) & ((1u64 << take) - 1);
            self.bit_buf = (self.bit_buf << take) | chunk;
            self.bit_count += take;
            remaining -= take;
            if self.bit_count == 8 {
                self.bytes.push(self.bit_buf as u8);
                self.bit_buf = 0;
                self.bit_count = 0;
            }
        }
    }

    /// Write the low `n` bits of a two's-complement signed value (§9.2.3 "signed two's
    /// complement"). Sign bits above `n` are dropped by the width mask in `write_bits`.
    pub fn write_signed(&mut self, value: i64, n: u32) {
        self.write_bits(value as u64, n);
    }

    /// Write a unary-coded number: `q` zero bits followed by a single one bit
    /// (§9.2.7.2 "the most-significant part [is] coded as unary"). This is the FLAC
    /// convention (terminating 1), as used for Rice quotients and wasted-bit counts.
    pub fn write_unary(&mut self, q: u32) {
        let mut left = q;
        // Emit zeros in chunks so a large quotient does not loop bit-by-bit.
        while left >= 32 {
            self.write_bits(0, 32);
            left -= 32;
        }
        // `left` zeros then the terminating 1: a field of width `left + 1` whose only
        // set bit is the least-significant one.
        self.write_bits(1, left + 1);
    }

    /// Pad with zero bits up to the next byte boundary (§9.3 frame footer alignment).
    pub fn align(&mut self) {
        if self.bit_count != 0 {
            let pad = 8 - self.bit_count;
            self.write_bits(0, pad);
        }
    }

    /// Append whole bytes. Only valid on a byte boundary (used for the frame footer
    /// CRC-16, which is written after `align`).
    pub fn write_aligned_bytes(&mut self, data: &[u8]) {
        debug_assert!(self.is_byte_aligned());
        self.bytes.extend_from_slice(data);
    }

    /// The bytes written so far, up to the last byte boundary. Panics if a partial
    /// byte is pending — call [`align`](Self::align) first.
    pub fn as_aligned_bytes(&self) -> &[u8] {
        debug_assert!(self.is_byte_aligned());
        &self.bytes
    }

    /// Finish: flush any partial byte (zero-padded) and return the buffer.
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.align();
        self.bytes
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Big-endian bit reader over a borrowed byte slice, MSB first. The mirror of
/// [`BitWriter`]; every method fails with [`Eof`](ReadError::Eof) rather than
/// panicking, because a decoder parses untrusted input (spec: decoders parse
/// untrusted input — a crash on any bitstream is a P0).
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Bit position from the start of `data` (byte = pos/8, bit-in-byte = pos%8).
    pos: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReadError {
    /// Ran off the end of the input.
    Eof,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Bits consumed so far.
    pub fn bit_pos(&self) -> u64 {
        self.pos
    }

    /// Bits still available.
    pub fn bits_left(&self) -> u64 {
        (self.data.len() as u64 * 8).saturating_sub(self.pos)
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.pos % 8 == 0
    }

    /// Read `n` bits (`0..=64`), MSB first, into the low bits of the result.
    pub fn read_bits(&mut self, n: u32) -> Result<u64, ReadError> {
        debug_assert!(n <= 64);
        if n == 0 {
            return Ok(0);
        }
        if self.bits_left() < n as u64 {
            return Err(ReadError::Eof);
        }
        let mut result: u64 = 0;
        let mut remaining = n;
        while remaining > 0 {
            let byte_idx = (self.pos / 8) as usize;
            let bit_in_byte = (self.pos % 8) as u32;
            let avail = 8 - bit_in_byte;
            let take = remaining.min(avail);
            // Extract `take` bits starting `bit_in_byte` from the top of this byte.
            let byte = self.data[byte_idx] as u64;
            let shift = avail - take;
            let chunk = (byte >> shift) & ((1u64 << take) - 1);
            result = (result << take) | chunk;
            self.pos += take as u64;
            remaining -= take;
        }
        Ok(result)
    }

    /// Read `n` bits as a two's-complement signed value, sign-extending from bit `n`.
    pub fn read_signed(&mut self, n: u32) -> Result<i64, ReadError> {
        let raw = self.read_bits(n)?;
        if n == 0 {
            return Ok(0);
        }
        // Sign-extend: if the top (bit n-1) is set, fill the higher bits with ones.
        let sign_bit = 1u64 << (n - 1);
        let value = if raw & sign_bit != 0 {
            (raw as i64) - (1i64 << n)
        } else {
            raw as i64
        };
        Ok(value)
    }

    /// Read a FLAC unary code: count zero bits up to (and consuming) the terminating
    /// one bit (§9.2.7.2). Returns the count of zeros (the quotient).
    pub fn read_unary(&mut self) -> Result<u32, ReadError> {
        let mut count = 0u32;
        loop {
            if self.bits_left() == 0 {
                return Err(ReadError::Eof);
            }
            let bit = self.read_bits(1)?;
            if bit == 1 {
                return Ok(count);
            }
            count += 1;
        }
    }

    /// Advance to the next byte boundary, discarding up to 7 padding bits.
    pub fn align(&mut self) {
        let rem = self.pos % 8;
        if rem != 0 {
            self.pos += 8 - rem;
        }
    }

    /// Read `len` whole bytes. Only valid on a byte boundary.
    pub fn read_aligned_bytes(&mut self, len: usize) -> Result<&'a [u8], ReadError> {
        debug_assert!(self.is_byte_aligned());
        let start = (self.pos / 8) as usize;
        let end = start.checked_add(len).ok_or(ReadError::Eof)?;
        if end > self.data.len() {
            return Err(ReadError::Eof);
        }
        self.pos += len as u64 * 8;
        Ok(&self.data[start..end])
    }
}

/// CRC-8 with polynomial x^8 + x^2 + x^1 + x^0 (== 0x07), init 0, no reflection
/// (§9.1.8, frame header CRC). MSB-first byte-wise update.
pub fn crc8(data: &[u8]) -> u8 {
    let mut crc: u8 = 0;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// CRC-16 with polynomial x^16 + x^15 + x^2 + x^0 (== 0x8005), init 0, no reflection
/// (§9.3, frame footer CRC). MSB-first byte-wise update over the whole frame up to
/// (but excluding) the CRC field itself.
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x8005;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_fixed_widths() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_bits(0xABCD, 16);
        w.write_bits(0, 1);
        w.write_bits(0x1F, 5);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert_eq!(r.read_bits(16).unwrap(), 0xABCD);
        assert_eq!(r.read_bits(1).unwrap(), 0);
        assert_eq!(r.read_bits(5).unwrap(), 0x1F);
    }

    #[test]
    fn unary_roundtrip() {
        for q in [0u32, 1, 2, 7, 8, 31, 32, 33, 100, 1000] {
            let mut w = BitWriter::new();
            w.write_unary(q);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_unary().unwrap(), q, "unary q={q}");
        }
    }

    #[test]
    fn signed_roundtrip_all_widths() {
        for n in 1..=32u32 {
            let lo = -(1i64 << (n - 1));
            let hi = (1i64 << (n - 1)) - 1;
            // Candidate values, filtered to those representable in `n`-bit two's
            // complement (e.g. +1 is invalid at n=1, whose range is only {-1, 0}).
            for v in [lo, lo + 1, -1, 0, 1, hi - 1, hi] {
                if v < lo || v > hi {
                    continue;
                }
                let mut w = BitWriter::new();
                w.write_signed(v, n);
                let bytes = w.into_bytes();
                let mut r = BitReader::new(&bytes);
                assert_eq!(r.read_signed(n).unwrap(), v, "signed v={v} n={n}");
            }
        }
    }

    #[test]
    fn random_mixed_stream_roundtrip() {
        // A deterministic LCG drives a stream of mixed writes; the reader must
        // recover every field exactly. Covers all widths 1..=64 and unary codes.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state
        };
        #[derive(Clone, Copy)]
        enum Op {
            Bits(u64, u32),
            Unary(u32),
            Signed(i64, u32),
        }
        let mut ops = Vec::new();
        let mut w = BitWriter::new();
        for _ in 0..5000 {
            match next() % 3 {
                0 => {
                    let n = (next() % 64 + 1) as u32;
                    let v = next() & if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
                    ops.push(Op::Bits(v, n));
                    w.write_bits(v, n);
                }
                1 => {
                    let q = (next() % 200) as u32;
                    ops.push(Op::Unary(q));
                    w.write_unary(q);
                }
                _ => {
                    let n = (next() % 32 + 1) as u32;
                    let range = 1i64 << (n - 1);
                    let v = (next() as i64).rem_euclid(2 * range) - range;
                    ops.push(Op::Signed(v, n));
                    w.write_signed(v, n);
                }
            }
        }
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        for op in ops {
            match op {
                Op::Bits(v, n) => assert_eq!(r.read_bits(n).unwrap(), v),
                Op::Unary(q) => assert_eq!(r.read_unary().unwrap(), q),
                Op::Signed(v, n) => assert_eq!(r.read_signed(n).unwrap(), v),
            }
        }
    }

    #[test]
    fn eof_is_reported_not_panicked() {
        let data = [0xFFu8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(8).unwrap(), 0xFF);
        assert_eq!(r.read_bits(1), Err(ReadError::Eof));
        assert_eq!(r.read_unary(), Err(ReadError::Eof));
    }

    #[test]
    fn align_pads_with_zeros() {
        let mut w = BitWriter::new();
        w.write_bits(0b111, 3);
        w.align();
        let bytes = w.into_bytes();
        assert_eq!(bytes, vec![0b1110_0000]);
    }

    // --- CRC known-answer tests derived from RFC 9639 Appendix D ---

    #[test]
    fn crc8_known_answer_from_appendix_d1() {
        // §D.1.4 Table 29: the frame header at 0x2a spans 6 bytes and its CRC-8 is
        // 0xbf. Header bytes from the binary dump (§D.1.2): the frame starts at 0x2a.
        // Bytes 0x2a..0x30: ff f8 69 18 00 00  (sync..blocksize), CRC at 0x30 = 0xbf.
        let header = [0xff, 0xf8, 0x69, 0x18, 0x00, 0x00];
        assert_eq!(crc8(&header), 0xbf);
    }

    #[test]
    fn crc8_known_answer_from_appendix_d2() {
        // §D.2.7 Table 36: frame header at 0x88, CRC-8 = 0x99.
        // Bytes 0x88..0x8e: ff f8 69 98 00 0f.
        let header = [0xff, 0xf8, 0x69, 0x98, 0x00, 0x0f];
        assert_eq!(crc8(&header), 0x99);
    }

    #[test]
    fn crc16_known_answer_from_appendix_d1() {
        // §D.1.4: the frame runs 0x2a..=0x38; its last two bytes (0x37=0xaa, 0x38=0x9a)
        // are the CRC-16, so the CRC covers 0x2a..=0x36 — 13 bytes ending in 0x8b — and
        // must equal 0xaa9a. (The §D.1.2 binary dump gives the exact bytes.)
        let frame = [
            0xff, 0xf8, 0x69, 0x18, 0x00, 0x00, 0xbf, 0x03, 0x58, 0xfd, 0x03, 0x12, 0x8b,
        ];
        assert_eq!(crc16(&frame), 0xaa9a);
    }

    #[test]
    fn crc_empty_is_zero() {
        assert_eq!(crc8(&[]), 0);
        assert_eq!(crc16(&[]), 0);
    }
}
