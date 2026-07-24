//! Hand-written EBML primitives (spec: `spec/MATROSKA.md`; RFC 8794). The byte grammar
//! every Matroska element is built from: Element IDs, Element Data Sizes, the unknown-size
//! marker, and the typed leaf encoders (uint, float, string, binary).
//!
//! All multi-byte scalars in EBML are **big-endian** (RFC 8794 §7.1: "stored … with the
//! most significant octet first"), which is the opposite of Ogg — do not copy the
//! `to_le_bytes` habit from `sc-ogg` here.
//!
//! Everything appends into a caller-owned `Vec<u8>` (no per-call allocation), matching the
//! `sc-ogg` writer style so the muxer can build a whole header/cluster in one buffer.

/// Canonical Matroska Element IDs, **pre-encoded** as their on-the-wire big-endian bytes
/// (spec `§ID-tree`). A Matroska ID *is* a VINT including its length-descriptor bits (RFC
/// 8794 §5), so these are written verbatim by [`write_id`] — the width is not re-derived.
/// Grouped in element-tree order.
pub mod id {
    // EBML Header (RFC 8794 §11.2.4) and its children.
    pub const EBML: &[u8] = &[0x1A, 0x45, 0xDF, 0xA3];
    pub const EBML_VERSION: &[u8] = &[0x42, 0x86];
    pub const EBML_READ_VERSION: &[u8] = &[0x42, 0xF7];
    pub const EBML_MAX_ID_LENGTH: &[u8] = &[0x42, 0xF2];
    pub const EBML_MAX_SIZE_LENGTH: &[u8] = &[0x42, 0xF3];
    pub const DOC_TYPE: &[u8] = &[0x42, 0x82];
    pub const DOC_TYPE_VERSION: &[u8] = &[0x42, 0x87];
    pub const DOC_TYPE_READ_VERSION: &[u8] = &[0x42, 0x85];

    // Segment and Info.
    pub const SEGMENT: &[u8] = &[0x18, 0x53, 0x80, 0x67];
    pub const INFO: &[u8] = &[0x15, 0x49, 0xA9, 0x66];
    pub const TIMESTAMP_SCALE: &[u8] = &[0x2A, 0xD7, 0xB1];
    pub const MUXING_APP: &[u8] = &[0x4D, 0x80];
    pub const WRITING_APP: &[u8] = &[0x57, 0x41];

    // Tracks.
    pub const TRACKS: &[u8] = &[0x16, 0x54, 0xAE, 0x6B];
    pub const TRACK_ENTRY: &[u8] = &[0xAE];
    pub const TRACK_NUMBER: &[u8] = &[0xD7];
    pub const TRACK_UID: &[u8] = &[0x73, 0xC5];
    pub const TRACK_TYPE: &[u8] = &[0x83];
    pub const CODEC_ID: &[u8] = &[0x86];
    pub const CODEC_PRIVATE: &[u8] = &[0x63, 0xA2];
    pub const FLAG_LACING: &[u8] = &[0x9C];
    pub const AUDIO: &[u8] = &[0xE1];
    pub const SAMPLING_FREQUENCY: &[u8] = &[0xB5];
    pub const CHANNELS: &[u8] = &[0x9F];
    pub const BIT_DEPTH: &[u8] = &[0x62, 0x64];

    // Cluster.
    pub const CLUSTER: &[u8] = &[0x1F, 0x43, 0xB6, 0x75];
    pub const TIMESTAMP: &[u8] = &[0xE7];
    pub const SIMPLE_BLOCK: &[u8] = &[0xA3];
}

/// The fixed width, in octets, of a back-patched size (spec `§sizing`). An 8-octet size
/// holds any length up to `2^56-2` and — being fixed width — avoids the shortest-length
/// `2^(7n)-1` edge case (RFC 8794 §6.3). The first byte is the 8-octet VINT marker
/// (`0x01`); the remaining seven carry the 56-bit big-endian length.
pub const BACKPATCH_SIZE_LEN: usize = 8;

/// The largest length a size VINT can carry: `2^56 - 1` unsigned, but `2^56 - 1` is itself
/// the all-ones "unknown" pattern (RFC 8794 §6.2/§6.3), so the largest *definite* length is
/// `2^56 - 2`. Used only in a `debug_assert` — real elements are nowhere near this.
pub const MAX_VINT_DATA: u64 = (1u64 << 56) - 2;

/// Write a pre-encoded Element ID verbatim (spec `§ID-tree`; RFC 8794 §5). The IDs in
/// [`id`] already include their VINT length-descriptor bits, so this is a raw append.
#[inline]
pub fn write_id(out: &mut Vec<u8>, element_id: &[u8]) {
    out.extend_from_slice(element_id);
}

/// The minimum number of octets a VINT needs to carry `value` as its `VINT_DATA` (RFC 8794
/// §4). Width `n` holds `0 ..= 2^(7n)-1`; this returns the smallest such `n` in `1..=8`.
///
/// Note the boundary rule (RFC 8794 §6.3): a value equal to `2^(7n)-1` would encode as
/// all-ones at width `n` (the "unknown" marker), so a *size* that large must widen to
/// `n+1`. This helper is used for element **data sizes** whose values are known and small
/// (leaf lengths), so it applies that rule: it returns the smallest width whose all-ones
/// value is strictly greater than `value`.
pub fn vint_size_len(value: u64) -> usize {
    debug_assert!(value <= MAX_VINT_DATA, "value too large for a VINT size");
    let mut len = 1usize;
    // `all_ones(len) = 2^(7*len) - 1`. Widen while `value` would hit or exceed it, so the
    // encoded VINT_DATA is never the reserved all-ones pattern (RFC 8794 §6.3).
    while len < 8 && value >= (1u64 << (7 * len)) - 1 {
        len += 1;
    }
    len
}

/// Write an Element Data Size as a shortest-valid VINT (RFC 8794 §6.1) carrying `value`
/// octets. The top bits are the length descriptor: for width `len`, bit `7*len` (counting
/// from the LSB of the whole VINT) is the marker `1`, and the low `7*len` bits are the
/// big-endian data. Used for leaves whose length is known when written.
pub fn write_size(out: &mut Vec<u8>, value: u64) {
    let len = vint_size_len(value);
    write_size_fixed(out, value, len);
}

/// Write an Element Data Size as a VINT of an explicit `len` octets (RFC 8794 §6.1 — sizes
/// "not mandated to be … shortest"). Used for the 8-octet back-patch slot; also the
/// primitive behind [`write_size`]. Sets the marker bit for `len` and packs `value`
/// big-endian into the low `7*len` bits.
pub fn write_size_fixed(out: &mut Vec<u8>, value: u64, len: usize) {
    debug_assert!((1..=8).contains(&len));
    debug_assert!(value < (1u64 << (7 * len)) || len == 8, "value overflows a {len}-octet size");
    // The VINT is `len` big-endian octets. The marker `1` sits at bit index `7*len` from
    // the whole-VINT LSB — i.e. the top bit of the first octet for len 1, next bit down for
    // len 2, etc. OR it into the assembled value's most significant used byte.
    let marker = 1u64 << (7 * len);
    let v = marker | value;
    // Emit the low `len` octets of `v`, big-endian.
    for i in (0..len).rev() {
        out.push((v >> (8 * i)) as u8);
    }
}

/// Reserve an 8-octet size slot to be back-patched later (spec `§sizing`). Writes the
/// canonical 8-octet-width VINT prefix (`0x01` marker + seven zero data octets) and returns
/// the offset of the slot's first byte, for [`patch_size`]. The seven zero octets are the
/// placeholder `VINT_DATA`.
pub fn reserve_size(out: &mut Vec<u8>) -> usize {
    let at = out.len();
    out.push(0x01); // 8-octet marker, data bits all zero for now
    out.extend_from_slice(&[0u8; 7]);
    at
}

/// Patch a slot reserved by [`reserve_size`] with the now-known data length (spec
/// `§sizing`). `at` is the slot offset; `data_len` is the number of octets of element data
/// that follow the slot. Rewrites the seven data octets (the marker byte at `at` stays
/// `0x01`).
pub fn patch_size(out: &mut [u8], at: usize, data_len: u64) {
    debug_assert!(data_len <= MAX_VINT_DATA, "element too large for an 8-octet size");
    debug_assert_eq!(out[at], 0x01, "patching a slot that is not an 8-octet size marker");
    // Seven big-endian data octets follow the marker byte.
    for i in 0..7 {
        out[at + 1 + i] = (data_len >> (8 * (6 - i))) as u8;
    }
}

/// Write the 1-octet **unknown data size** marker for a streamed master element (spec
/// `§sizing`; RFC 8794 §6.2: "all VINT_DATA bits set to one … the size … is unknown"). At
/// width 1 that is the single byte `0xFF`. Legal only for master elements (Segment,
/// Cluster) whose end is found implicitly.
#[inline]
pub fn write_unknown_size(out: &mut Vec<u8>) {
    out.push(0xFF);
}

/// Write a full unsigned-integer element: ID, size, then the value as a big-endian uint of
/// its minimal byte length (RFC 8794 §7.1). EBML uints are 0–8 octets; a value of 0 encodes
/// as a single `0x00` octet (length 1), not empty, so the element is unambiguous.
pub fn write_uint(out: &mut Vec<u8>, element_id: &[u8], value: u64) {
    write_id(out, element_id);
    let n = uint_len(value);
    write_size(out, n as u64);
    for i in (0..n).rev() {
        out.push((value >> (8 * i)) as u8);
    }
}

/// Minimal big-endian octet length of `value` as an EBML uint (RFC 8794 §7.1): the number
/// of bytes needed to represent it, minimum 1 (so 0 → 1 byte).
pub fn uint_len(value: u64) -> usize {
    let mut n = 1usize;
    while n < 8 && (value >> (8 * n)) != 0 {
        n += 1;
    }
    n
}

/// Write a full 32-bit float element: ID, size (4), value big-endian (RFC 8794 §7.3 —
/// EBML floats are IEEE 754 big-endian, 4 or 8 octets).
pub fn write_f32(out: &mut Vec<u8>, element_id: &[u8], value: f32) {
    write_id(out, element_id);
    write_size(out, 4);
    out.extend_from_slice(&value.to_bits().to_be_bytes());
}

/// Write a full 64-bit float element: ID, size (8), value big-endian (RFC 8794 §7.3).
pub fn write_f64(out: &mut Vec<u8>, element_id: &[u8], value: f64) {
    write_id(out, element_id);
    write_size(out, 8);
    out.extend_from_slice(&value.to_bits().to_be_bytes());
}

/// Write a full UTF-8 string element: ID, byte length, the bytes verbatim (RFC 8794 §7.4 —
/// EBML strings are unterminated; the size gives the length). No NUL is appended.
pub fn write_string(out: &mut Vec<u8>, element_id: &[u8], value: &str) {
    write_id(out, element_id);
    write_size(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

/// Write a full binary element: ID, byte length, the bytes verbatim (RFC 8794 §7.5). Used
/// for CodecPrivate and SimpleBlock.
pub fn write_binary(out: &mut Vec<u8>, element_id: &[u8], value: &[u8]) {
    write_id(out, element_id);
    write_size(out, value.len() as u64);
    out.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The width helper matches the RFC 8794 §4 length ladder at each boundary, and the
    /// §6.3 rule that a value hitting the all-ones pattern for width `n` widens to `n+1`.
    #[test]
    fn vint_size_len_boundaries() {
        assert_eq!(vint_size_len(0), 1);
        assert_eq!(vint_size_len(1), 1);
        // 2^7-1 = 127 is all-ones at width 1 → must widen to 2 (§6.3).
        assert_eq!(vint_size_len(126), 1);
        assert_eq!(vint_size_len(127), 2);
        assert_eq!(vint_size_len((1 << 14) - 2), 2);
        assert_eq!(vint_size_len((1 << 14) - 1), 3); // 2^14-1 all-ones at width 2
        assert_eq!(vint_size_len((1 << 21) - 1), 4);
        assert_eq!(vint_size_len((1 << 28) - 1), 5);
        assert_eq!(vint_size_len(MAX_VINT_DATA), 8);
    }

    /// Decode a size VINT the way a reader does: the first set bit (from the MSB of the
    /// first byte) is the marker; its position gives the length; the remaining bits are the
    /// big-endian value. Returns `(value, len)`.
    fn decode_size(bytes: &[u8]) -> (u64, usize) {
        let first = bytes[0];
        let len = first.leading_zeros() as usize + 1; // marker is the first 1 bit
        // Clear the marker bit, then accumulate the remaining data octets big-endian.
        // `0xFF >> len` in u16 avoids the u8 overflow when len == 8 (mask is then 0).
        let mut value = (first as u64) & ((0xFFu16 >> len) as u64);
        for &b in &bytes[1..len] {
            value = (value << 8) | b as u64;
        }
        (value, len)
    }

    /// Round-trip: `write_size` then decode recovers the value at the expected width, for a
    /// spread of edge cases across every VINT width (RFC 8794 §4, §6).
    #[test]
    fn write_size_roundtrip_and_width() {
        let cases: &[(u64, usize)] = &[
            (0, 1),
            (1, 1),
            (126, 1),
            (127, 2),
            (128, 2),
            (0x3FFE, 2),
            (0x3FFF, 3),
            (100_000, 3),
            ((1 << 21) - 2, 3),
            ((1 << 21) - 1, 4),
            (1_000_000, 3),
            (16_000_000, 4),
        ];
        for &(value, want_len) in cases {
            let mut out = Vec::new();
            write_size(&mut out, value);
            assert_eq!(out.len(), want_len, "width for {value}");
            let (got, len) = decode_size(&out);
            assert_eq!(got, value, "value round-trip for {value}");
            assert_eq!(len, want_len);
        }
    }

    /// The reserved 8-octet slot decodes to the patched length, and the marker byte is the
    /// canonical `0x01` (spec `§sizing`).
    #[test]
    fn reserve_then_patch_roundtrips() {
        for &len in &[0u64, 1, 255, 256, 65_535, 1_000_000, MAX_VINT_DATA] {
            let mut out = Vec::new();
            let at = reserve_size(&mut out);
            assert_eq!(out.len(), BACKPATCH_SIZE_LEN);
            assert_eq!(out[at], 0x01, "8-octet marker");
            patch_size(&mut out, at, len);
            let (got, width) = decode_size(&out);
            assert_eq!(width, 8, "back-patch slot stays 8 octets");
            assert_eq!(got, len, "patched length");
        }
    }

    #[test]
    fn unknown_size_is_ff() {
        let mut out = Vec::new();
        write_unknown_size(&mut out);
        assert_eq!(out, vec![0xFF]);
        // 0xFF decodes as width 1, value all-ones (2^7-1) — the reserved marker.
        let (value, len) = decode_size(&out);
        assert_eq!(len, 1);
        assert_eq!(value, (1 << 7) - 1);
    }

    #[test]
    fn uint_len_minimal() {
        assert_eq!(uint_len(0), 1);
        assert_eq!(uint_len(255), 1);
        assert_eq!(uint_len(256), 2);
        assert_eq!(uint_len(65_535), 2);
        assert_eq!(uint_len(65_536), 3);
        assert_eq!(uint_len(u64::MAX), 8);
    }

    /// A uint element is ID · size · minimal-big-endian value. Check the exact bytes for a
    /// small and a multi-byte value, including the mandatory single `0x00` for zero.
    #[test]
    fn write_uint_bytes() {
        let mut out = Vec::new();
        write_uint(&mut out, id::TRACK_TYPE, 2); // 0x83, size 1, value 0x02
        assert_eq!(out, vec![0x83, 0x81, 0x02]);

        let mut out = Vec::new();
        write_uint(&mut out, id::CHANNELS, 0); // zero → one 0x00 octet, not empty
        assert_eq!(out, vec![0x9F, 0x81, 0x00]);

        let mut out = Vec::new();
        write_uint(&mut out, id::TIMESTAMP_SCALE, 1_000_000); // 0x0F4240, 3 octets
        assert_eq!(out, vec![0x2A, 0xD7, 0xB1, 0x83, 0x0F, 0x42, 0x40]);
    }

    /// Floats are big-endian IEEE 754 (RFC 8794 §7.3), the opposite of Ogg's LE fields.
    #[test]
    fn write_float_bytes_big_endian() {
        // SAMPLING_FREQUENCY id 0xB5 is one byte, so size is at index 1, data from index 2.
        let mut out = Vec::new();
        write_f64(&mut out, id::SAMPLING_FREQUENCY, 48000.0);
        assert_eq!(&out[..1], id::SAMPLING_FREQUENCY);
        assert_eq!(out[1], 0x88); // size = 8
        assert_eq!(&out[2..], &48000.0f64.to_bits().to_be_bytes());

        let mut out = Vec::new();
        write_f32(&mut out, id::SAMPLING_FREQUENCY, 44100.0);
        assert_eq!(out[1], 0x84); // size = 4
        assert_eq!(&out[2..], &44100.0f32.to_bits().to_be_bytes());
    }

    #[test]
    fn write_string_and_binary_bytes() {
        let mut out = Vec::new();
        write_string(&mut out, id::CODEC_ID, "A_FLAC");
        // 0x86, size 6, "A_FLAC" — no NUL terminator.
        assert_eq!(out, vec![0x86, 0x86, b'A', b'_', b'F', b'L', b'A', b'C']);

        let mut out = Vec::new();
        write_binary(&mut out, id::CODEC_PRIVATE, &[0xDE, 0xAD]);
        assert_eq!(out, vec![0x63, 0xA2, 0x82, 0xDE, 0xAD]);
    }
}
