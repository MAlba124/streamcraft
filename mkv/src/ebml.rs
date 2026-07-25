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
/// `Info\Duration` (RFC 9559 §5.1.2): a float, in TimestampScale ticks. Without it a
/// player treats an unknown-size Segment as a live/duration-less stream.
pub const DURATION: &[u8] = &[0x44, 0x89];
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

    // Video master (RFC 9559 §5.1.4.1.28) — the video counterpart of the Audio master. The
    // demuxer reads PixelWidth/PixelHeight to seed a video track's pad announcement; the
    // writer emits them for a video (V_VP8, …) track.
    pub const VIDEO: &[u8] = &[0xE0];
    pub const PIXEL_WIDTH: &[u8] = &[0xB0];
    pub const PIXEL_HEIGHT: &[u8] = &[0xBA];

    // Cluster.
    pub const CLUSTER: &[u8] = &[0x1F, 0x43, 0xB6, 0x75];
    pub const TIMESTAMP: &[u8] = &[0xE7];
    pub const SIMPLE_BLOCK: &[u8] = &[0xA3];

    // BlockGroup + Block — the demuxer reads these; the muxer emits only SimpleBlock, but a
    // conforming Matroska file may store frames in a BlockGroup wrapping a plain Block (RFC
    // 9559 §5.1.3.5). The demuxer accepts both forms.
    pub const BLOCK_GROUP: &[u8] = &[0xA0];
    pub const BLOCK: &[u8] = &[0xA1];
    pub const BLOCK_DURATION: &[u8] = &[0x9B];
    pub const REFERENCE_BLOCK: &[u8] = &[0xFB];

    // SeekHead (RFC 9559 §5.1.1) — the index at the Segment's start telling a player
    // where the other level-1 masters live; how end-of-file Cues are discovered
    // without scanning the whole Segment.
    pub const SEEK_HEAD: &[u8] = &[0x11, 0x4D, 0x9B, 0x74];
    pub const SEEK: &[u8] = &[0x4D, 0xBB];
    pub const SEEK_ID: &[u8] = &[0x53, 0xAB];
    pub const SEEK_POSITION: &[u8] = &[0x53, 0xAC];

    // Cues (RFC 9559 §5.1.5) — the seek index: CueTime → (CueTrack,
    // CueClusterPosition), one CuePoint per seekable Cluster. Positions are Segment
    // Positions (relative to the first byte of the Segment's data — RFC 9559 §4).
    pub const CUES: &[u8] = &[0x1C, 0x53, 0xBB, 0x6B];
    pub const CUE_POINT: &[u8] = &[0xBB];
    pub const CUE_TIME: &[u8] = &[0xB3];
    pub const CUE_TRACK_POSITIONS: &[u8] = &[0xB7];
    pub const CUE_TRACK: &[u8] = &[0xF7];
    pub const CUE_CLUSTER_POSITION: &[u8] = &[0xF1];

    // Void (RFC 8794 §11.3.2) — reserved dead space. The writer emits one where the
    // SeekHead will go and overwrites it at finalize (the single back-patch).
    pub const VOID: &[u8] = &[0xEC];
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

// =====================================================================================
// Reading — the parse side (spec: `spec/MATROSKA.md`; RFC 8794 §4–§6)
//
// The demuxer (`crate::element::MkvDemux`) parses untrusted input, so every reader here is
// **fallible and bounds-checked** — a truncated or malformed stream returns a
// [`ReadError`], never panics or slices out of range (spec: "a crash on bad input is a
// P0"). The readers operate on a borrowed `&[u8]` slice with a caller-held cursor; buffering
// bytes across input boundaries is the demuxer's job (it accumulates a `Vec<u8>` and re-runs
// the parser once more bytes arrive), so these primitives only ever see a contiguous window.
// =====================================================================================

/// A parse failure. The demuxer maps this to a `streamcraft` [`Error`](streamcraft_core::error::Error);
/// [`Incomplete`](ReadError::Incomplete) is special — it means "need more bytes", not
/// "corrupt", so the demuxer buffers and retries rather than erroring out.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ReadError {
    /// The window ends before the value is complete — the demuxer needs to buffer more input
    /// and retry (NOT a corruption; RFC 8794 elements can straddle a read boundary).
    Incomplete,
    /// A VINT's first octet is `0x00`, which would encode a width > 8 (RFC 8794 §4.4: the
    /// maximum VINT width is 8 octets).
    VintTooLong,
    /// A structural violation: a size or field that cannot be valid (e.g. a child claiming to
    /// run past its parent, a leaf whose declared length overflows `usize`).
    Malformed(&'static str),
}

/// Read one VINT starting at `data[at]`, returning `(value, len, all_ones)` and *not*
/// advancing any cursor (RFC 8794 §4). `all_ones` flags the reserved value whose VINT_DATA is
/// entirely `1` bits — the unknown-size marker for a data size (§6.2). Bounds-checked: a
/// window too short for the marker-indicated width is [`Incomplete`](ReadError::Incomplete).
///
/// This is the shared decoder behind both [`read_id`] (which keeps the marker bits) and
/// [`read_size`] (which strips them): a size's *value* is the VINT_DATA, whereas an ID *is*
/// the whole VINT octets — see [`read_id`].
pub fn read_vint(data: &[u8], at: usize) -> Result<(u64, usize, bool), ReadError> {
    let first = *data.get(at).ok_or(ReadError::Incomplete)?;
    if first == 0x00 {
        return Err(ReadError::VintTooLong); // width would exceed 8 octets (RFC 8794 §4.4)
    }
    let len = first.leading_zeros() as usize + 1; // marker is the first 1 bit
    if at + len > data.len() {
        return Err(ReadError::Incomplete);
    }
    // Data bits of the first octet: clear the marker. `0xFF >> len` in u16 avoids the u8
    // overflow at len == 8 (the mask is then 0 — an 8-octet VINT has no data in octet 0).
    let mask = (0xFFu16 >> len) as u64;
    let mut value = (first as u64) & mask;
    let mut all_ones = value == mask;
    for &b in &data[at + 1..at + len] {
        value = (value << 8) | b as u64;
        all_ones &= b == 0xFF;
    }
    Ok((value, len, all_ones))
}

/// Read an Element **ID** at `data[at]` as its raw on-the-wire octets (spec `§ID-tree`; RFC
/// 8794 §5). Unlike a size, a Matroska ID *includes* its VINT length-descriptor bits, so the
/// bytes are returned verbatim to compare against the [`id`] constants. Returns `(id_bytes,
/// len)`. Bounds-checked.
pub fn read_id(data: &[u8], at: usize) -> Result<(&[u8], usize), ReadError> {
    let (_v, len, _ao) = read_vint(data, at)?;
    Ok((&data[at..at + len], len))
}

/// Read an Element **Data Size** at `data[at]` (RFC 8794 §6.1), returning `(size, len)` where
/// `size` is `None` for the unknown-size marker (all VINT_DATA bits `1`, §6.2) and
/// `Some(octets)` otherwise. Bounds-checked.
pub fn read_size(data: &[u8], at: usize) -> Result<(Option<u64>, usize), ReadError> {
    let (value, len, all_ones) = read_vint(data, at)?;
    Ok((if all_ones { None } else { Some(value) }, len))
}

/// One parsed element header: its ID (raw bytes), the byte offset where its *data* begins,
/// and the data length (`None` = unknown-size streamed master, §6.2). Returned by
/// [`read_element_header`].
#[derive(Clone, Copy, Debug)]
pub struct ElementHeader<'a> {
    /// The element's ID as its raw on-the-wire octets (compare against [`id`] constants).
    pub id: &'a [u8],
    /// Byte offset (in the same slice) where the element's data starts — i.e. just past the
    /// ID and size VINTs.
    pub data_start: usize,
    /// Data length in octets, or `None` for an unknown-size master (Segment/Cluster).
    pub size: Option<u64>,
}

impl ElementHeader<'_> {
    /// Byte offset just past this element's data — where the next sibling begins — for a
    /// definite-size element. `None` for an unknown-size master (its end is implicit). Errors
    /// [`Malformed`](ReadError::Malformed) if the length overflows a `usize` offset.
    pub fn data_end(&self) -> Result<Option<usize>, ReadError> {
        match self.size {
            None => Ok(None),
            Some(sz) => {
                let end = (self.data_start as u64)
                    .checked_add(sz)
                    .ok_or(ReadError::Malformed("element size overflows the address space"))?;
                Ok(Some(end as usize))
            }
        }
    }
}

/// Read one element header (ID + size) at `data[at]`, returning the parsed header (the data
/// is *not* read — the caller descends into a master or skips a leaf). Bounds-checked:
/// [`Incomplete`](ReadError::Incomplete) when the window is too short for the ID+size.
pub fn read_element_header(data: &[u8], at: usize) -> Result<ElementHeader<'_>, ReadError> {
    let (id, id_len) = read_id(data, at)?;
    let (size, size_len) = read_size(data, at + id_len)?;
    Ok(ElementHeader {
        id,
        data_start: at + id_len + size_len,
        size,
    })
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

    // --- Reader primitives (RFC 8794 §4–§6) ---

    /// Every size the writer emits reads back to the same value at the same width — the
    /// reader is the exact inverse of `write_size` (RFC 8794 §6.1).
    #[test]
    fn read_size_inverts_write_size() {
        for &v in &[0u64, 1, 126, 127, 128, 0x3FFF, 100_000, 1_000_000, MAX_VINT_DATA] {
            let mut buf = Vec::new();
            write_size(&mut buf, v);
            let (got, len) = read_size(&buf, 0).expect("read");
            assert_eq!(got, Some(v), "value round-trip for {v}");
            assert_eq!(len, buf.len(), "width round-trip for {v}");
        }
    }

    /// The `0xFF` unknown-size marker reads as `None` (streamed master, §6.2); an ID reads
    /// back as its verbatim octets (§5).
    #[test]
    fn read_unknown_size_and_id() {
        let (size, len) = read_size(&[0xFF], 0).expect("read");
        assert_eq!((size, len), (None, 1), "0xFF is the unknown-size marker");

        let (id, len) = read_id(id::EBML, 0).expect("read id");
        assert_eq!(id, id::EBML, "ID octets returned verbatim");
        assert_eq!(len, 4);
    }

    /// A window too short for the marker-indicated width is `Incomplete` (need more bytes),
    /// distinct from a corrupt `0x00` first octet which is `VintTooLong` — the demuxer
    /// buffers on the former and errors on the latter.
    #[test]
    fn read_vint_incomplete_vs_too_long() {
        // A 4-octet VINT (marker 0x1A) with only 2 octets present → Incomplete.
        assert_eq!(read_vint(&[0x1A, 0x45], 0), Err(ReadError::Incomplete));
        // Empty window → Incomplete.
        assert_eq!(read_vint(&[], 0), Err(ReadError::Incomplete));
        // First octet 0x00 would mean width > 8 → VintTooLong.
        assert_eq!(read_vint(&[0x00, 0x01], 0), Err(ReadError::VintTooLong));
    }

    /// `read_element_header` parses ID + size and points `data_start`/`data_end` at the right
    /// offsets; an unknown-size master reports `size == None` and `data_end == None`.
    #[test]
    fn read_element_header_offsets() {
        // A definite-size uint element: TIMESTAMP_SCALE = 1_000_000 (from write_uint_bytes).
        let mut buf = Vec::new();
        write_uint(&mut buf, id::TIMESTAMP_SCALE, 1_000_000);
        let h = read_element_header(&buf, 0).expect("header");
        assert_eq!(h.id, id::TIMESTAMP_SCALE);
        assert_eq!(h.size, Some(3), "value 0x0F4240 is 3 octets");
        assert_eq!(h.data_end().unwrap(), Some(buf.len()));

        // An open Segment: ID then 0xFF → unknown size.
        let mut seg = Vec::new();
        write_id(&mut seg, id::SEGMENT);
        write_unknown_size(&mut seg);
        let h = read_element_header(&seg, 0).expect("header");
        assert_eq!(h.id, id::SEGMENT);
        assert_eq!(h.size, None, "unknown-size streamed master");
        assert_eq!(h.data_end().unwrap(), None);
    }
}
