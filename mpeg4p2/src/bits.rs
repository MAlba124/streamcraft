//! Big-endian bit reader for the MPEG-4 Visual bitstream (ISO/IEC 14496-2 §5.2.1:
//! bits are read most-significant first; the stream is byte-aligned only at
//! start codes and after `next_start_code()` stuffing). Every read is
//! bounds-checked — this reader is the trust boundary between untrusted input and
//! the decode core, so it never indexes past `data` and reports exhaustion rather
//! than panicking (spec robustness P0: a malformed VOP must warn-and-drop, never
//! OOB).

/// A most-significant-bit-first reader over a borrowed byte slice.
///
/// Position is tracked in bits from the start of `data`. All accessors saturate
/// at the end of the buffer: reading past the end returns zero-extended values
/// and leaves the cursor at `data.len() * 8` so callers can detect exhaustion via
/// [`BitReader::overrun`].
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit position (0 == the MSB of `data[0]`).
    pos: usize,
    /// Set once a read tried to go past the end of the buffer.
    overrun: bool,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, overrun: false }
    }

    /// A reader positioned at `bit_pos` bits into `data`.
    pub fn at(data: &'a [u8], bit_pos: usize) -> Self {
        Self { data, pos: bit_pos.min(data.len() * 8), overrun: false }
    }

    /// Total length of the underlying buffer, in bits.
    #[inline]
    pub fn total_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// The current absolute bit position.
    #[inline]
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    /// True once any read attempted to consume bits past the end of the buffer.
    #[inline]
    pub fn overrun(&self) -> bool {
        self.overrun
    }

    /// Bits remaining before the end of the buffer (0 once exhausted).
    #[inline]
    pub fn bits_left(&self) -> usize {
        self.total_bits().saturating_sub(self.pos)
    }

    /// Peek a single bit without advancing, 0 if past the end.
    #[inline]
    fn peek_bit_at(&self, pos: usize) -> u32 {
        if pos >= self.total_bits() {
            return 0;
        }
        let byte = self.data[pos >> 3];
        ((byte >> (7 - (pos & 7))) & 1) as u32
    }

    /// Read a single bit (§5.2.1, `bslbf`/`uimsbf` element read one bit wide).
    #[inline]
    pub fn read_bit(&mut self) -> u32 {
        if self.pos >= self.total_bits() {
            self.overrun = true;
            return 0;
        }
        let b = self.peek_bit_at(self.pos);
        self.pos += 1;
        b
    }

    /// Read `n` bits (0..=32) as an unsigned big-endian integer (`uimsbf`).
    #[inline]
    pub fn read_bits(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit();
        }
        v
    }

    /// Peek `n` bits (0..=32) without advancing the cursor.
    #[inline]
    pub fn peek_bits(&self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        let mut v = 0u32;
        for i in 0..n {
            v = (v << 1) | self.peek_bit_at(self.pos + i as usize);
        }
        v
    }

    /// Skip `n` bits (saturating at the buffer end).
    #[inline]
    pub fn skip(&mut self, n: usize) {
        if self.pos + n > self.total_bits() {
            self.overrun = true;
        }
        self.pos = (self.pos + n).min(self.total_bits());
    }

    /// Read a `1`-terminated marker bit (§6.3.1 `marker_bit`). Returns the bit
    /// value; a well-formed stream always has this as 1, but we do not reject on
    /// it (some muxers are sloppy) — the caller may check.
    #[inline]
    pub fn marker_bit(&mut self) -> u32 {
        self.read_bit()
    }

    /// Byte-align the cursor to the next byte boundary (§5.2.1: `next_start_code`
    /// aligns with a `0` bit then `1`-bits stuffing, but for scanning we only need
    /// the alignment).
    #[inline]
    pub fn align_to_byte(&mut self) {
        let rem = self.pos & 7;
        if rem != 0 {
            self.pos += 8 - rem;
        }
    }

    /// True if the cursor sits on a byte boundary.
    #[inline]
    pub fn is_byte_aligned(&self) -> bool {
        self.pos & 7 == 0
    }

    /// Peek whether the next bits (after byte alignment) are a start-code prefix
    /// `0000 0000 0000 0000 0000 0001` (§6.2.1). Does not advance.
    pub fn next_bits_are_start_code(&self) -> bool {
        // A start code appears only byte-aligned in a legal stream. Look at the
        // next aligned 24 bits.
        let aligned = (self.pos + 7) & !7;
        if aligned + 24 > self.total_bits() {
            return false;
        }
        let b0 = self.data[aligned >> 3];
        let b1 = self.data[(aligned >> 3) + 1];
        let b2 = self.data[(aligned >> 3) + 2];
        b0 == 0 && b1 == 0 && b2 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_msb_first() {
        // 0b1010_0110 = 0xA6
        let mut r = BitReader::new(&[0xA6, 0x00]);
        assert_eq!(r.read_bit(), 1);
        assert_eq!(r.read_bit(), 0);
        assert_eq!(r.read_bits(2), 0b10);
        assert_eq!(r.read_bits(4), 0b0110);
    }

    #[test]
    fn multibyte_and_peek() {
        let mut r = BitReader::new(&[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(r.peek_bits(16), 0xDEAD);
        assert_eq!(r.read_bits(16), 0xDEAD);
        assert_eq!(r.read_bits(16), 0xBEEF);
        assert!(!r.overrun());
    }

    #[test]
    fn overrun_is_flagged_not_panicked() {
        let mut r = BitReader::new(&[0xFF]);
        let _ = r.read_bits(16); // asks for 16, only 8 available
        assert!(r.overrun());
        assert_eq!(r.bits_left(), 0);
    }

    #[test]
    fn start_code_detection_is_byte_aligned() {
        let data = [0x00, 0x00, 0x01, 0xB6];
        let r = BitReader::new(&data);
        assert!(r.next_bits_are_start_code());
        let data2 = [0xFF, 0x00, 0x00, 0x01];
        let r2 = BitReader::new(&data2);
        assert!(!r2.next_bits_are_start_code(), "not at offset 0");
    }
}
