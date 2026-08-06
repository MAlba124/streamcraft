//! The Ogg page model: parse and serialise a single page (spec: RFC 3533 §6,
//! `spec/rfc3533.txt`).
//!
//! A physical Ogg bitstream is "a sequence of concatenated pages" (§6); this module
//! owns exactly one of them. The [`OggReader`](crate::OggReader) drives
//! [`PageHeader::parse`] over a byte stream and the [`OggWriter`](crate::OggWriter)
//! drives [`write_page`]; packet reassembly and framing live in their modules.
//!
//! ## On-the-wire layout (§6)
//! The 27-byte fixed header, then `page_segments` lacing bytes, then the payload:
//! ```text
//!  0   capture_pattern "OggS"            (4 bytes)
//!  4   stream_structure_version = 0      (1 byte)
//!  5   header_type flags                 (1 byte)   0x01 cont, 0x02 bos, 0x04 eos
//!  6   granule_position                  (8 bytes, little-endian)
//! 14   bitstream_serial_number           (4 bytes, little-endian)
//! 18   page_sequence_number              (4 bytes, little-endian)
//! 22   CRC_checksum                      (4 bytes, little-endian)
//! 26   page_segments                     (1 byte)
//! 27   segment_table                     (page_segments bytes)
//! ```
//! "Fields with more than one byte length are encoded LSB (least significant byte)
//! first" (§6) — i.e. little-endian, which is why the multi-byte getters below use
//! `from_le_bytes`.

/// `"OggS"` — the 4-byte capture pattern that "signifies the beginning of a page"
/// (§6, field 1). `0x4f 0x67 0x67 0x53`.
pub const CAPTURE_PATTERN: [u8; 4] = *b"OggS";

/// This document specifies stream_structure_version 0 (§6, field 2).
pub const STREAM_STRUCTURE_VERSION: u8 = 0;

/// Fixed part of the page header before the segment table (§6: "header_size =
/// number_page_segments + 27").
pub const HEADER_FIXED_LEN: usize = 27;

/// A page carries at most 255 segments (`page_segments` is one byte, §6, field 8),
/// each at most 255 bytes, so the payload is at most 255 * 255 = 65025 bytes and the
/// whole page at most 27 + 255 + 65025 = 65307 bytes (§6: "maximum 65307 bytes").
pub const MAX_SEGMENTS: usize = 255;

/// Largest possible payload of a single page (255 segments × 255 bytes).
pub const MAX_PAGE_PAYLOAD: usize = MAX_SEGMENTS * 255;

/// Largest possible on-the-wire page: fixed header + full segment table + max payload.
pub const MAX_PAGE_SIZE: usize = HEADER_FIXED_LEN + MAX_SEGMENTS + MAX_PAGE_PAYLOAD;

/// header_type flag bits (§6, field 3).
pub mod flags {
    /// bit 0x01 set: "page contains data of a packet continued from the previous page".
    pub const CONTINUED: u8 = 0x01;
    /// bit 0x02 set: "this is the first page of a logical bitstream (bos)".
    pub const BOS: u8 = 0x02;
    /// bit 0x04 set: "this is the last page of a logical bitstream (eos)".
    pub const EOS: u8 = 0x04;
}

/// The granule_position value that "indicates that no packets finish on this page"
/// (§6, field 4: "A special value of -1 (in two's complement)"). Stored as an unsigned
/// 64-bit field on the wire, so the sentinel is `u64::MAX`.
pub const GRANULE_NONE: u64 = u64::MAX;

/// A parsed, integrity-checked view over one page borrowed from the input stream. The
/// header fields are decoded lazily from the borrowed bytes; [`payload`](Self::payload)
/// is the concatenated segment data after the header.
#[derive(Clone, Copy, Debug)]
pub struct PageHeader<'a> {
    /// The whole page: fixed header + segment table + payload. `len()` is exactly the
    /// page size, so the reader advances by `page.len()`.
    bytes: &'a [u8],
    /// Number of lacing values (== `bytes[26]`), cached for slicing.
    n_segments: usize,
}

/// Why a candidate page failed to parse. The reader treats every one of these as
/// "not a valid page here" and resynchronises to the next capture pattern (§6: the
/// capture pattern lets a decoder "regain synchronisation after parsing a corrupted
/// stream"); nothing here ever panics.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PageError {
    /// Fewer than 27 bytes available — cannot even read the fixed header.
    NeedMoreHeader,
    /// The first four bytes are not `OggS`.
    BadCapturePattern,
    /// stream_structure_version was not 0 (§6, field 2).
    BadVersion(u8),
    /// The declared page size exceeds the bytes available (truncated page).
    NeedMoreBody,
    /// The page's stored CRC did not match the computed one (corrupt page).
    BadCrc { stored: u32, computed: u32 },
}

impl<'a> PageHeader<'a> {
    /// Try to parse a page starting at `data[0]`. On success returns the page (whose
    /// [`len`](Self::len) is the exact on-the-wire size) and validates the CRC. Any
    /// failure is reported, never panicked — the caller resyncs on error.
    ///
    /// A `NeedMore*` error means "insufficient bytes *so far*": with more data buffered
    /// the same offset may yet parse, so a streaming reader must not discard the byte.
    pub fn parse(data: &'a [u8]) -> Result<PageHeader<'a>, PageError> {
        if data.len() < HEADER_FIXED_LEN {
            return Err(PageError::NeedMoreHeader);
        }
        if data[0..4] != CAPTURE_PATTERN {
            return Err(PageError::BadCapturePattern);
        }
        let version = data[4];
        if version != STREAM_STRUCTURE_VERSION {
            return Err(PageError::BadVersion(version));
        }
        let n_segments = data[26] as usize;
        let header_len = HEADER_FIXED_LEN + n_segments;
        if data.len() < header_len {
            return Err(PageError::NeedMoreHeader);
        }
        // Sum the lacing values to get the payload length (§6: page_size = header_size
        // + sum(lacing_values)). Each lacing value is a single byte, so the sum of up
        // to 255 of them fits easily in usize.
        let segment_table = &data[HEADER_FIXED_LEN..header_len];
        let payload_len: usize = segment_table.iter().map(|&b| b as usize).sum();
        let page_len = header_len + payload_len;
        if data.len() < page_len {
            return Err(PageError::NeedMoreBody);
        }
        let bytes = &data[..page_len];

        // Verify the CRC over the whole page with the CRC field (bytes 22..26) treated
        // as zero (§6, field 7). We checksum the three spans around the CRC field
        // without copying the page.
        let computed = page_crc(bytes);
        let stored = u32::from_le_bytes([bytes[22], bytes[23], bytes[24], bytes[25]]);
        if stored != computed {
            return Err(PageError::BadCrc { stored, computed });
        }

        Ok(PageHeader { bytes, n_segments })
    }

    /// The full on-the-wire size of this page in bytes (header + table + payload).
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        // A page always has a 27-byte header, so it is never truly empty; provided for
        // clippy/`len`-symmetry.
        self.bytes.is_empty()
    }

    /// header_type flags byte (§6, field 3).
    pub fn header_type(&self) -> u8 {
        self.bytes[5]
    }

    /// True if this page continues a packet from the previous page (§6: flag 0x01).
    pub fn is_continued(&self) -> bool {
        self.header_type() & flags::CONTINUED != 0
    }

    /// True if this is the first page of a logical bitstream (§6: flag 0x02, bos).
    pub fn is_bos(&self) -> bool {
        self.header_type() & flags::BOS != 0
    }

    /// True if this is the last page of a logical bitstream (§6: flag 0x04, eos).
    pub fn is_eos(&self) -> bool {
        self.header_type() & flags::EOS != 0
    }

    /// granule_position (§6, field 4), little-endian. `u64::MAX` == "no packets finish
    /// on this page" (the −1 sentinel, [`GRANULE_NONE`]).
    pub fn granule_position(&self) -> u64 {
        u64::from_le_bytes([
            self.bytes[6], self.bytes[7], self.bytes[8], self.bytes[9], self.bytes[10],
            self.bytes[11], self.bytes[12], self.bytes[13],
        ])
    }

    /// bitstream_serial_number (§6, field 5), little-endian.
    pub fn serial(&self) -> u32 {
        u32::from_le_bytes([self.bytes[14], self.bytes[15], self.bytes[16], self.bytes[17]])
    }

    /// page_sequence_number (§6, field 6), little-endian.
    pub fn sequence(&self) -> u32 {
        u32::from_le_bytes([self.bytes[18], self.bytes[19], self.bytes[20], self.bytes[21]])
    }

    /// The stored CRC field (§6, field 7), little-endian. Always equals the computed
    /// CRC because [`parse`](Self::parse) verified it.
    pub fn crc(&self) -> u32 {
        u32::from_le_bytes([self.bytes[22], self.bytes[23], self.bytes[24], self.bytes[25]])
    }

    /// Number of lacing values in the segment table (§6, field 8).
    pub fn n_segments(&self) -> usize {
        self.n_segments
    }

    /// The lacing values (§6, field 9), one byte per segment.
    pub fn segment_table(&self) -> &'a [u8] {
        &self.bytes[HEADER_FIXED_LEN..HEADER_FIXED_LEN + self.n_segments]
    }

    /// The concatenated segment data (the page payload), after the header.
    pub fn payload(&self) -> &'a [u8] {
        &self.bytes[HEADER_FIXED_LEN + self.n_segments..]
    }

    /// The raw page bytes (for cross-validation / passthrough).
    pub fn raw(&self) -> &'a [u8] {
        self.bytes
    }
}

/// Compute the Ogg page CRC over `page`, treating the 4-byte CRC field at offset 22 as
/// zero (§6, field 7). Feeds the pre-CRC span, four zero bytes, then the post-CRC span
/// into one running [`Crc32`](crate::crc::Crc32) — no copy of the page.
pub fn page_crc(page: &[u8]) -> u32 {
    debug_assert!(page.len() >= HEADER_FIXED_LEN);
    let mut crc = crate::crc::Crc32::new();
    crc.update(&page[..22]);
    crc.update(&[0, 0, 0, 0]);
    crc.update(&page[26..]);
    crc.value()
}

/// Serialise one page into `out`, computing and stamping its CRC (§6). Low-level: the
/// caller supplies already-segmented payload plus its exact lacing values and the
/// header fields. [`crate::writer`] builds these from packets.
///
/// Preconditions (checked with `debug_assert`, and by construction in the writer):
/// `segment_table.len() <= 255`, and `payload.len()` equals the sum of the lacing
/// values.
#[allow(clippy::too_many_arguments)]
pub fn write_page(
    out: &mut Vec<u8>,
    header_type: u8,
    granule_position: u64,
    serial: u32,
    sequence: u32,
    segment_table: &[u8],
    payload: &[u8],
) {
    debug_assert!(segment_table.len() <= MAX_SEGMENTS);
    debug_assert_eq!(
        payload.len(),
        segment_table.iter().map(|&b| b as usize).sum::<usize>(),
        "payload length must equal the sum of lacing values"
    );

    let start = out.len();
    out.extend_from_slice(&CAPTURE_PATTERN);
    out.push(STREAM_STRUCTURE_VERSION);
    out.push(header_type);
    out.extend_from_slice(&granule_position.to_le_bytes());
    out.extend_from_slice(&serial.to_le_bytes());
    out.extend_from_slice(&sequence.to_le_bytes());
    // CRC field: write zeros for now, patch after the whole page is laid out (§6: the
    // CRC is computed "including header with zero CRC field and page content").
    let crc_at = out.len();
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.push(segment_table.len() as u8);
    out.extend_from_slice(segment_table);
    out.extend_from_slice(payload);

    let crc = page_crc(&out[start..]);
    out[crc_at..crc_at + 4].copy_from_slice(&crc.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid page (one 4-byte packet, bos) and round-trip it through
    /// the parser, checking every field and the CRC.
    #[test]
    fn write_then_parse_roundtrips_all_fields() {
        let payload = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut out = Vec::new();
        write_page(&mut out, flags::BOS, 0x0102_0304_0506_0708, 0xAABB_CCDD, 42, &[4], &payload);

        let page = PageHeader::parse(&out).expect("valid page");
        assert_eq!(page.len(), out.len());
        assert!(page.is_bos());
        assert!(!page.is_eos());
        assert!(!page.is_continued());
        assert_eq!(page.granule_position(), 0x0102_0304_0506_0708);
        assert_eq!(page.serial(), 0xAABB_CCDD);
        assert_eq!(page.sequence(), 42);
        assert_eq!(page.n_segments(), 1);
        assert_eq!(page.segment_table(), &[4]);
        assert_eq!(page.payload(), &payload);
        // Stored CRC equals the recomputed one.
        assert_eq!(page.crc(), page_crc(&out));
    }

    #[test]
    fn capture_pattern_is_oggs() {
        assert_eq!(&CAPTURE_PATTERN, b"OggS");
        assert_eq!(CAPTURE_PATTERN, [0x4f, 0x67, 0x67, 0x53]);
    }

    #[test]
    fn rejects_bad_capture_pattern() {
        let mut out = Vec::new();
        write_page(&mut out, 0, 0, 1, 0, &[0], &[]);
        out[1] = b'X'; // corrupt "OggS" → "OXgS"
        assert!(matches!(PageHeader::parse(&out), Err(PageError::BadCapturePattern)));
    }

    #[test]
    fn rejects_bad_version() {
        let mut out = Vec::new();
        write_page(&mut out, 0, 0, 1, 0, &[0], &[]);
        out[4] = 1; // version 1, not 0 — but this also breaks the CRC; version is
                    // checked before the CRC, so we still get BadVersion.
        assert!(matches!(PageHeader::parse(&out), Err(PageError::BadVersion(1))));
    }

    #[test]
    fn detects_corrupt_payload_via_crc() {
        let mut out = Vec::new();
        write_page(&mut out, 0, 7, 1, 0, &[3], &[1, 2, 3]);
        let last = out.len() - 1;
        out[last] ^= 0xFF; // flip a payload byte
        match PageHeader::parse(&out) {
            Err(PageError::BadCrc { .. }) => {}
            other => panic!("expected BadCrc, got {other:?}"),
        }
    }

    #[test]
    fn truncated_header_and_body_report_need_more() {
        let mut out = Vec::new();
        write_page(&mut out, 0, 0, 1, 0, &[5], &[1, 2, 3, 4, 5]);
        // Less than the fixed header.
        assert!(matches!(PageHeader::parse(&out[..10]), Err(PageError::NeedMoreHeader)));
        // Header + partial payload (n_segments known but the declared payload is short).
        let short = out.len() - 2;
        assert!(matches!(PageHeader::parse(&out[..short]), Err(PageError::NeedMoreBody)));
    }

    #[test]
    fn zero_length_packet_page() {
        // A nil packet is "nothing more than a lacing value of zero" (§5).
        let mut out = Vec::new();
        write_page(&mut out, 0, 0, 1, 0, &[0], &[]);
        let page = PageHeader::parse(&out).unwrap();
        assert_eq!(page.n_segments(), 1);
        assert_eq!(page.segment_table(), &[0]);
        assert!(page.payload().is_empty());
    }

    #[test]
    fn granule_none_sentinel() {
        let mut out = Vec::new();
        write_page(&mut out, 0, GRANULE_NONE, 1, 0, &[0], &[]);
        let page = PageHeader::parse(&out).unwrap();
        assert_eq!(page.granule_position(), GRANULE_NONE);
    }

    /// The strongest known-answer test: an actual bos page produced by libVorbis /
    /// libogg (`oggenc`). If our parser reads its fields and — crucially — our CRC
    /// recomputes the exact value libogg stamped, then our page format and CRC-32 match
    /// the real wire format bit-for-bit, independent of any tool at test time.
    ///
    /// This is the first (bos) page of an Ogg Vorbis file: it starts the Vorbis
    /// identification header, so its payload begins `0x01 "vorbis"` (§4). Captured once
    /// from `oggenc` output; the round-trip against `ogginfo` (which validates CRCs)
    /// confirmed it was well-formed when captured.
    #[test]
    fn real_libogg_vorbis_bos_page_crc_and_fields() {
        // 58-byte page: 27-byte header + 1 lacing value (0x1e = 30) + 30-byte payload.
        const PAGE: [u8; 58] = [
            0x4f, 0x67, 0x67, 0x53, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x27, 0x70, 0xb3, 0x23, 0x00, 0x00, 0x00, 0x00, 0xce, 0x7f,
            0x3c, 0x45, 0x01, 0x1e, 0x01, 0x76, 0x6f, 0x72, 0x62, 0x69, 0x73, 0x00,
            0x00, 0x00, 0x00, 0x02, 0x44, 0xac, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x80, 0xb5, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb8, 0x01,
        ];
        let page = PageHeader::parse(&PAGE).expect("real libogg page must parse");
        // Header fields libogg wrote.
        assert_eq!(page.len(), 58);
        assert!(page.is_bos());
        assert!(!page.is_eos());
        assert!(!page.is_continued());
        assert_eq!(page.serial(), 0x23b3_7027);
        assert_eq!(page.sequence(), 0);
        assert_eq!(page.granule_position(), 0); // no audio samples on the id-header page
        assert_eq!(page.n_segments(), 1);
        assert_eq!(page.segment_table(), &[30]);
        // The Vorbis identification packet: 0x01 then "vorbis" (§4).
        assert_eq!(&page.payload()[..7], &[0x01, b'v', b'o', b'r', b'b', b'i', b's']);
        // The load-bearing assertion: our CRC == the CRC libogg stamped (0x453c7fce).
        assert_eq!(page.crc(), 0x453c_7fce);
        assert_eq!(page_crc(&PAGE), 0x453c_7fce);
    }
}
