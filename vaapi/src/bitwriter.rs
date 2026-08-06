//! MSB-first bitstream writer for authoring H.264/H.265 parameter sets and slice
//! headers (the packed headers VA-API encode expects from the application).
//!
//! Three layers, matching the specs' own vocabulary:
//! - [`BitWriter`] — raw descriptor writing: `u(n)` fixed-width, `ue(v)`/`se(v)`
//!   Exp-Golomb (ITU-T H.264 §9.1 / H.265 §9.2 — identical codes), and
//!   `rbsp_trailing_bits()` (H.264 §7.3.2.11: stop bit + zero-fill to a byte).
//! - [`to_ebsp`] — emulation prevention: within a NAL payload every
//!   `0x000000/1/2/3` pattern gets `emulation_prevention_three_byte` (0x03)
//!   inserted after the two zeros (H.264 §7.4.1.1, H.265 §7.4.2).
//! - [`annexb_nal`] — a complete Annex-B NAL: 4-byte start code, the raw NAL
//!   header byte(s) (never subject to emulation prevention), then the EBSP.

/// MSB-first bit accumulator. Bits are appended into bytes high-bit-first, the
/// bitstream order both H.26x specs read (§7.2: most significant bit first).
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Free bits remaining in the tail byte (0..=7). 0 means byte-aligned.
    free: u32,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter { bytes: Vec::with_capacity(64), free: 0 }
    }

    /// Total bits written so far (the packed-header `bit_length`).
    pub fn bit_len(&self) -> u32 {
        self.bytes.len() as u32 * 8 - self.free
    }

    /// `u(n)`: `n` bits of `val`, most significant first (n ≤ 32).
    pub fn u(&mut self, val: u32, n: u32) {
        debug_assert!(n <= 32);
        debug_assert!(n == 32 || val < (1u64 << n) as u32, "u({val},{n}) overflows the field");
        for i in (0..n).rev() {
            let bit = (val >> i) & 1;
            if self.free == 0 {
                self.bytes.push(0);
                self.free = 8;
            }
            let last = self.bytes.last_mut().expect("tail byte exists");
            self.free -= 1;
            *last |= (bit as u8) << self.free;
        }
    }

    /// A single flag bit.
    pub fn flag(&mut self, b: bool) {
        self.u(b as u32, 1);
    }

    /// `ue(v)`: unsigned Exp-Golomb (H.264 §9.1): `leadingZeroBits` zeros, then
    /// `v+1` in `leadingZeroBits+1` bits, where `leadingZeroBits = floor(log2(v+1))`.
    pub fn ue(&mut self, v: u32) {
        let code = v as u64 + 1;
        let bits = 64 - code.leading_zeros(); // floor(log2(code)) + 1
        self.u(0, bits - 1);
        // code < 2^bits with bits ≤ 33; write high half then low if needed.
        if bits > 32 {
            self.u((code >> 32) as u32, bits - 32);
            self.u(code as u32, 32);
        } else {
            self.u(code as u32, bits);
        }
    }

    /// `se(v)`: signed Exp-Golomb (H.264 §9.1.1): codeNum = 2|v|−1 for v>0, 2|v|
    /// for v≤0 (positive values map to odd codes).
    pub fn se(&mut self, v: i32) {
        let code = if v > 0 { (v as u32) * 2 - 1 } else { (-(v as i64) as u32) * 2 };
        self.ue(code);
    }

    /// `rbsp_trailing_bits()` (H.264 §7.3.2.11 / H.265 §7.3.2.11): the
    /// `rbsp_stop_one_bit`, then zeros to the byte boundary.
    pub fn rbsp_trailing_bits(&mut self) {
        self.flag(true);
        while self.free != 0 {
            self.flag(false);
        }
    }

    /// The written bytes. Any trailing partial byte is zero-padded low (callers
    /// that need exact bit lengths read [`bit_len`](Self::bit_len) first).
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Insert `emulation_prevention_three_byte`s: any `0x00 0x00` followed by a byte
/// ≤ 0x03 gets 0x03 interposed (H.264 §7.4.1.1) so no NAL payload byte sequence
/// mimics a start code.
pub fn to_ebsp(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 8);
    let mut zeros = 0u32;
    for &b in rbsp {
        if zeros >= 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// A complete Annex-B NAL unit: `00 00 00 01` start code, the raw `header` bytes
/// (the NAL header is defined to never need emulation prevention), then the
/// emulation-guarded payload.
pub fn annexb_nal(header: &[u8], rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + header.len() + rbsp.len() + 8);
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(header);
    out.extend_from_slice(&to_ebsp(rbsp));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ue_matches_spec_table() {
        // H.264 Table 9-1: codeNum 0..8 → bit strings.
        let cases: [(u32, &[u8], u32); 6] = [
            (0, &[0b1000_0000], 1),          // "1"
            (1, &[0b0100_0000], 3),          // "010"
            (2, &[0b0110_0000], 3),          // "011"
            (3, &[0b0010_0000], 5),          // "00100"
            (4, &[0b0010_1000], 5),          // "00101"
            (8, &[0b0001_0010], 7),          // "0001001"
        ];
        for (v, bytes, bits) in cases {
            let mut w = BitWriter::new();
            w.ue(v);
            assert_eq!(w.bit_len(), bits, "ue({v}) bit length");
            assert_eq!(w.as_bytes(), bytes, "ue({v}) bits");
        }
    }

    #[test]
    fn se_maps_signed_per_9_1_1() {
        // H.264 Table 9-3: v: 0→0, 1→1, −1→2, 2→3, −2→4 (codeNum).
        for (v, code) in [(0i32, 0u32), (1, 1), (-1, 2), (2, 3), (-2, 4), (3, 5), (-3, 6)] {
            let mut a = BitWriter::new();
            a.se(v);
            let mut b = BitWriter::new();
            b.ue(code);
            assert_eq!(a.as_bytes(), b.as_bytes(), "se({v}) == ue({code})");
        }
    }

    #[test]
    fn trailing_bits_align() {
        let mut w = BitWriter::new();
        w.u(0b101, 3);
        w.rbsp_trailing_bits();
        assert_eq!(w.as_bytes(), &[0b1011_0000]);
        assert_eq!(w.bit_len(), 8);

        // Already-aligned: stop bit opens a fresh byte.
        let mut w = BitWriter::new();
        w.u(0xAB, 8);
        w.rbsp_trailing_bits();
        assert_eq!(w.as_bytes(), &[0xAB, 0x80]);
    }

    #[test]
    fn ebsp_guards_start_codes() {
        assert_eq!(to_ebsp(&[0, 0, 0]), &[0, 0, 3, 0]);
        assert_eq!(to_ebsp(&[0, 0, 1]), &[0, 0, 3, 1]);
        assert_eq!(to_ebsp(&[0, 0, 2]), &[0, 0, 3, 2]);
        assert_eq!(to_ebsp(&[0, 0, 3]), &[0, 0, 3, 3]);
        assert_eq!(to_ebsp(&[0, 0, 4]), &[0, 0, 4]); // > 3: legal, untouched
        assert_eq!(to_ebsp(&[1, 0, 0]), &[1, 0, 0]); // trailing zeros: no third byte
        // Long zero runs re-arm after each insertion.
        assert_eq!(to_ebsp(&[0, 0, 0, 0, 1]), &[0, 0, 3, 0, 0, 3, 1]);
    }

    #[test]
    fn annexb_frames_header_and_payload() {
        let nal = annexb_nal(&[0x67], &[0x42, 0x00, 0x00, 0x00]);
        assert_eq!(nal, &[0, 0, 0, 1, 0x67, 0x42, 0x00, 0x00, 0x03, 0x00]);
    }
}
