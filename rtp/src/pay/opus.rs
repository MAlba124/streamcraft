//! Opus payloading (RFC 7587 §4.2): one Opus packet IS one RTP payload,
//! verbatim — the trivial inverse of [`crate::depay::opus`].
//!
//! The surrounding header fields are the element's job: the timestamp
//! advances at 48 kHz by the packet's total frame duration (§4.2 Table 2 —
//! e.g. 960 per 20 ms frame; multi-frame packets add their increments), one
//! packet may cover at most 120 ms (§4.2), and the marker bit is set only on
//! the first packet after a DTX silence — the start of a talkspurt (RFC 3551
//! §4.1: "Applications without silence suppression MUST set the marker bit
//! to zero") — which requires cross-packet state this pure function does not
//! keep.

/// Payload one Opus packet: the bytes, verbatim (§4.2: "An RTP payload MUST
/// contain exactly one Opus packet").
pub fn pay(opus_packet: &[u8]) -> Vec<u8> {
    opus_packet.to_vec()
}

#[cfg(test)]
mod tests {
    use super::pay;
    use crate::depay::opus::depay;

    #[test]
    fn round_trip_is_identity() {
        let packet = [0xFC, 0xFF, 0xFE, 0x00, 0x42];
        assert_eq!(depay(&pay(&packet)), packet);
    }
}
