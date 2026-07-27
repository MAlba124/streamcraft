//! Big-endian MSB-first bit reader for the AC-3 / E-AC-3 syncframe.
//!
//! ATSC A/52 (Digital Audio Compression (AC-3, E-AC-3) Standard) §2.3.3 defines
//! the bit stream as read **most-significant-bit-first** within each byte, bytes
//! in stream order. Every field-read in [`frame`](crate::frame) goes through this
//! reader, which is the single place bounds are enforced: a read past the end of
//! the buffer returns [`Err`] rather than panicking (robustness is P0 — the input
//! is untrusted; a malformed frame must warn-and-drop, never OOB).

/// A read past the end of the frame buffer. Carried up to the frame parser, which
/// turns it into a warn-and-drop (never a panic).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EndOfData;

/// MSB-first big-endian bit cursor over a byte slice (A/52 §2.3.3 bit order).
///
/// Holds a byte position and an in-byte bit offset. All reads are checked; the
/// caller propagates [`EndOfData`] as a frame-level error.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit position from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// A reader over `data`, positioned at bit 0.
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Total number of bits in the underlying buffer.
    #[inline]
    pub fn total_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Bits consumed so far (from the start of the buffer).
    #[inline]
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    /// Bits remaining before the end of the buffer.
    #[inline]
    pub fn bits_left(&self) -> usize {
        self.total_bits().saturating_sub(self.pos)
    }

    /// Read a single bit (A/52 uses `bit()` throughout). Returns 0/1.
    #[inline]
    pub fn bit(&mut self) -> Result<u32, EndOfData> {
        let byte = self.pos >> 3;
        if byte >= self.data.len() {
            return Err(EndOfData);
        }
        let shift = 7 - (self.pos & 7);
        self.pos += 1;
        Ok(((self.data[byte] >> shift) & 1) as u32)
    }

    /// Read `n` bits (0..=32) MSB-first into the low bits of a `u32`
    /// (A/52 §2.3.3: fields are unsigned, MSB-first). `n == 0` yields 0.
    #[inline]
    pub fn bits(&mut self, n: u32) -> Result<u32, EndOfData> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.pos + n as usize > self.total_bits() {
            return Err(EndOfData);
        }
        let mut v: u32 = 0;
        // Read bit-by-bit: clarity over speed. The whole frame is a few KB and the
        // hot path is the IMDCT, not the header parse.
        for _ in 0..n {
            let byte = self.pos >> 3;
            let shift = 7 - (self.pos & 7);
            v = (v << 1) | ((self.data[byte] >> shift) & 1) as u32;
            self.pos += 1;
        }
        Ok(v)
    }

    /// Read `n` bits (0..=64) into a `u64` (for the rare wide field).
    #[inline]
    pub fn bits64(&mut self, n: u32) -> Result<u64, EndOfData> {
        debug_assert!(n <= 64);
        if n <= 32 {
            return Ok(u64::from(self.bits(n)?));
        }
        let hi = self.bits(n - 32)?;
        let lo = self.bits(32)?;
        Ok((u64::from(hi) << 32) | u64::from(lo))
    }

    /// Skip `n` bits (bounds-checked).
    #[inline]
    pub fn skip(&mut self, n: usize) -> Result<(), EndOfData> {
        if self.pos + n > self.total_bits() {
            return Err(EndOfData);
        }
        self.pos += n;
        Ok(())
    }

    /// Move the absolute bit position (used to re-align after the fixed-length BSI
    /// when the frame carries optional trailing fields). Bounds-checked.
    #[inline]
    pub fn seek_bits(&mut self, pos: usize) -> Result<(), EndOfData> {
        if pos > self.total_bits() {
            return Err(EndOfData);
        }
        self.pos = pos;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msb_first_bit_order() {
        // 0b1010_0110 = 0xA6
        let mut r = BitReader::new(&[0xA6]);
        assert_eq!(r.bit().unwrap(), 1);
        assert_eq!(r.bit().unwrap(), 0);
        assert_eq!(r.bit().unwrap(), 1);
        assert_eq!(r.bit().unwrap(), 0);
        assert_eq!(r.bits(4).unwrap(), 0b0110);
    }

    #[test]
    fn multibyte_field() {
        // 0x0B77 sync word read as one 16-bit field.
        let mut r = BitReader::new(&[0x0B, 0x77]);
        assert_eq!(r.bits(16).unwrap(), 0x0B77);
    }

    #[test]
    fn past_end_is_error_not_panic() {
        let mut r = BitReader::new(&[0xFF]);
        assert_eq!(r.bits(8).unwrap(), 0xFF);
        assert_eq!(r.bit(), Err(EndOfData));
        assert_eq!(r.bits(1), Err(EndOfData));
    }

    #[test]
    fn wide_field() {
        let mut r = BitReader::new(&[0x12, 0x34, 0x56, 0x78, 0x9A]);
        assert_eq!(r.bits64(40).unwrap(), 0x0000_0012_3456_789A);
    }
}
