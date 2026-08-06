//! The RTP fixed header and payload as a zero-copy view (RFC 3550 §5.1).
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |V=2|P|X|  CC   |M|     PT      |       sequence number         |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                           timestamp                           |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |           synchronization source (SSRC) identifier            |
//! +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
//! |            contributing source (CSRC) identifiers             |
//! |                             ....                              |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```
//!
//! [`RtpPacket::parse`] borrows the datagram: header fields are decoded, the
//! optional header extension (§5.3.1) is located, and padding (§5.1 `P` bit:
//! the last octet counts the pad bytes, itself included) is stripped from the
//! reported payload. One datagram = one packet (RTP itself has no framing).

/// Why a datagram was rejected as RTP.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParseError {
    /// Shorter than the 12-byte fixed header, or truncated CSRC/extension.
    Truncated,
    /// `V != 2` (§5.1 — the only version on the wire).
    Version,
    /// The padding count is zero or exceeds the bytes after the header.
    Padding,
}

/// A parsed, borrowed RTP packet (RFC 3550 §5.1).
#[derive(Clone, Copy, Debug)]
pub struct RtpPacket<'a> {
    marker: bool,
    payload_type: u8,
    seq: u16,
    timestamp: u32,
    ssrc: u32,
    /// CSRC list bytes (4 per entry), directly after the fixed header.
    csrc: &'a [u8],
    /// Header extension `(defined-by-profile id, data)` when `X` is set (§5.3.1).
    extension: Option<(u16, &'a [u8])>,
    /// The payload with padding already stripped.
    payload: &'a [u8],
}

impl<'a> RtpPacket<'a> {
    /// Parse one datagram (§5.1). Returns a borrowed view; no bytes are copied.
    pub fn parse(d: &'a [u8]) -> Result<RtpPacket<'a>, ParseError> {
        if d.len() < 12 {
            return Err(ParseError::Truncated);
        }
        let b0 = d[0];
        if b0 >> 6 != 2 {
            return Err(ParseError::Version);
        }
        let padding = b0 & 0x20 != 0;
        let has_ext = b0 & 0x10 != 0;
        let cc = (b0 & 0x0F) as usize;
        let b1 = d[1];

        let mut at = 12 + 4 * cc;
        if d.len() < at {
            return Err(ParseError::Truncated);
        }
        let csrc = &d[12..at];

        let extension = if has_ext {
            // §5.3.1: 16-bit profile-defined id, 16-bit length in 32-bit words
            // (excluding this 4-byte extension header).
            if d.len() < at + 4 {
                return Err(ParseError::Truncated);
            }
            let id = u16::from_be_bytes([d[at], d[at + 1]]);
            let words = u16::from_be_bytes([d[at + 2], d[at + 3]]) as usize;
            let data_start = at + 4;
            let data_end = data_start + 4 * words;
            if d.len() < data_end {
                return Err(ParseError::Truncated);
            }
            at = data_end;
            Some((id, &d[data_start..data_end]))
        } else {
            None
        };

        let mut end = d.len();
        if padding {
            // §5.1: the last octet of the packet is the pad count, including
            // itself; it must fit between the header end and the packet end.
            let pad = d[end - 1] as usize;
            if pad == 0 || at + pad > end {
                return Err(ParseError::Padding);
            }
            end -= pad;
        }

        Ok(RtpPacket {
            marker: b1 & 0x80 != 0,
            payload_type: b1 & 0x7F,
            seq: u16::from_be_bytes([d[2], d[3]]),
            timestamp: u32::from_be_bytes([d[4], d[5], d[6], d[7]]),
            ssrc: u32::from_be_bytes([d[8], d[9], d[10], d[11]]),
            csrc,
            extension,
            payload: &d[at..end],
        })
    }

    /// The marker bit `M` — payload-format-defined (§5.1; e.g. RFC 6184: set
    /// on the last packet of an access unit).
    pub fn marker(&self) -> bool {
        self.marker
    }

    /// The 7-bit payload type `PT` (static types: RFC 3551 §6; dynamic 96–127
    /// bound by the SDP `rtpmap`).
    pub fn payload_type(&self) -> u8 {
        self.payload_type
    }

    /// The 16-bit sequence number (wraps; see [`crate::seq`]).
    pub fn seq(&self) -> u16 {
        self.seq
    }

    /// The 32-bit media timestamp in the payload format's clock rate (§5.1;
    /// the rate comes from the profile/SDP, not the packet).
    pub fn timestamp(&self) -> u32 {
        self.timestamp
    }

    /// The synchronization source (§5.1 / §8).
    pub fn ssrc(&self) -> u32 {
        self.ssrc
    }

    /// CSRC identifiers (mixer contributors, §5.1) — usually empty.
    pub fn csrc(&self) -> impl Iterator<Item = u32> + 'a {
        self.csrc.chunks_exact(4).map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
    }

    /// The header extension `(defined-by-profile, data)` when present (§5.3.1).
    pub fn extension(&self) -> Option<(u16, &'a [u8])> {
        self.extension
    }

    /// The payload, padding stripped.
    pub fn payload(&self) -> &'a [u8] {
        self.payload
    }
}

/// Build an RTP packet into bytes — the send path's serializer, and the test
/// suites' fixture builder (round-trip pinning for [`RtpPacket::parse`]).
#[derive(Clone, Debug)]
pub struct RtpPacketBuilder {
    pub marker: bool,
    pub payload_type: u8,
    pub seq: u16,
    pub timestamp: u32,
    pub ssrc: u32,
}

impl RtpPacketBuilder {
    /// Serialize the fixed header + `payload` (§5.1; V=2, no padding, no
    /// extension, no CSRCs — the shapes a sender emits).
    pub fn build(&self, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + payload.len());
        out.push(0x80); // V=2, P=0, X=0, CC=0
        out.push((self.payload_type & 0x7F) | if self.marker { 0x80 } else { 0 });
        out.extend_from_slice(&self.seq.to_be_bytes());
        out.extend_from_slice(&self.timestamp.to_be_bytes());
        out.extend_from_slice(&self.ssrc.to_be_bytes());
        out.extend_from_slice(payload);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_parse_round_trip() {
        let b = RtpPacketBuilder {
            marker: true,
            payload_type: 96,
            seq: 0xFFFE,
            timestamp: 0xDEAD_BEEF,
            ssrc: 0x1234_5678,
        };
        let bytes = b.build(&[1, 2, 3, 4, 5]);
        let p = RtpPacket::parse(&bytes).unwrap();
        assert!(p.marker());
        assert_eq!(p.payload_type(), 96);
        assert_eq!(p.seq(), 0xFFFE);
        assert_eq!(p.timestamp(), 0xDEAD_BEEF);
        assert_eq!(p.ssrc(), 0x1234_5678);
        assert_eq!(p.payload(), &[1, 2, 3, 4, 5]);
        assert_eq!(p.csrc().count(), 0);
        assert!(p.extension().is_none());
    }

    #[test]
    fn csrc_extension_and_padding_parse() {
        // Hand-built: V=2, P=1, X=1, CC=1, M=0, PT=8, seq 7, ts 9, ssrc 3,
        // one CSRC (0x0A0B0C0D), ext id 0xBEDE len 1 word, payload [0xAA],
        // 3 pad bytes (last = 3).
        let mut d = vec![
            0xB1, 0x08, 0x00, 0x07, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x03,
        ];
        d.extend_from_slice(&[0x0A, 0x0B, 0x0C, 0x0D]); // CSRC
        d.extend_from_slice(&[0xBE, 0xDE, 0x00, 0x01, 1, 2, 3, 4]); // ext
        d.extend_from_slice(&[0xAA]); // payload
        d.extend_from_slice(&[0, 0, 3]); // padding
        let p = RtpPacket::parse(&d).unwrap();
        assert_eq!(p.payload_type(), 8);
        assert_eq!(p.csrc().collect::<Vec<_>>(), vec![0x0A0B_0C0D]);
        assert_eq!(p.extension(), Some((0xBEDE, &[1u8, 2, 3, 4][..])));
        assert_eq!(p.payload(), &[0xAA]);
    }

    #[test]
    fn malformed_datagrams_are_rejected() {
        assert_eq!(RtpPacket::parse(&[0x80; 11]).unwrap_err(), ParseError::Truncated);
        let mut wrong_ver = [0u8; 12];
        wrong_ver[0] = 0x40; // V=1
        assert_eq!(RtpPacket::parse(&wrong_ver).unwrap_err(), ParseError::Version);
        // CC=2 but no room for CSRCs.
        let mut short_csrc = [0u8; 12];
        short_csrc[0] = 0x82;
        assert_eq!(RtpPacket::parse(&short_csrc).unwrap_err(), ParseError::Truncated);
        // Padding count larger than the payload region.
        let mut bad_pad = [0u8; 13];
        bad_pad[0] = 0xA0; // V=2, P=1
        bad_pad[12] = 9;
        assert_eq!(RtpPacket::parse(&bad_pad).unwrap_err(), ParseError::Padding);
        // Extension length running past the datagram.
        let mut bad_ext = [0u8; 16];
        bad_ext[0] = 0x90; // V=2, X=1
        bad_ext[15] = 4; // 4 words claimed, none present
        assert_eq!(RtpPacket::parse(&bad_ext).unwrap_err(), ParseError::Truncated);
    }
}
