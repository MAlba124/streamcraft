//! Opus depayloading (RFC 7587 §4.2): one RTP payload IS exactly one Opus
//! packet as framed by RFC 6716 §3 — no fragmentation, no aggregation, no
//! payload header. Multi-frame bundling (up to the 120 ms packet-duration
//! cap) happens *inside* the Opus packet via its TOC byte, never at the RTP
//! layer (§4.2: "An RTP payload MUST contain exactly one Opus packet").
//!
//! Timing (§4.1): the RTP timestamp always runs at 48 kHz regardless of the
//! coded audio bandwidth, counting samples per mono channel; it corresponds
//! to the sample time of the packet's first encoded sample.
//!
//! DTX vs. loss (§3.1.3, RFC 3551 §4.1): under discontinuous transmission
//! the sender simply stops sending during silence, dropping whole frames so
//! successive timestamps still differ by a multiple of 120 (§3.1.3). A gap
//! in the *sequence numbers* means loss; contiguous sequence numbers with a
//! timestamp jump mean intentional DTX silence (§3.1.3: "A receiver can
//! distinguish between DTX and packet loss by looking for gaps in the
//! sequence number"). The marker bit carries no framing information — it is
//! set on the first packet of a talkspurt after such a silent period (RFC
//! 3551 §4.1) — so depacketizing ignores it; the element layer may use it to
//! adjust playout delay.
//!
//! Duplicate packets are the sequence layer's problem, not this function's:
//! §4.1 — the receiver "MUST provide at most one of those payloads to the
//! Opus decoder for decoding, and it MUST discard the others".

/// Depacketize one payload: the Opus packet, verbatim (§4.2). Kept as a
/// stateless function — the element wrapper derives pts from the 48 kHz RTP
/// timestamp (§4.1) and handles DTX/loss via the sequence layer. A malformed
/// (e.g. empty) payload is passed through untouched: TOC validation belongs
/// to the Opus decoder, and the RTP layer has no opinion on the bytes.
pub fn depay(payload: &[u8]) -> Vec<u8> {
    payload.to_vec()
}

#[cfg(test)]
mod tests {
    use super::depay;

    #[test]
    fn payload_is_the_packet_verbatim() {
        // §4.2: exactly one Opus packet per payload, byte-for-byte.
        let toc_and_data = [0x78, 0x01, 0x02, 0x03];
        assert_eq!(depay(&toc_and_data), toc_and_data);
        assert_eq!(depay(&[]), Vec::<u8>::new());
    }
}
