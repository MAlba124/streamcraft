//! Opus packet frame-packing parser (RFC 6716 §3.2).
//!
//! The TOC byte (§3.1, decoded by [`crate::toc::OpusTocByte`]) determines
//! how the rest of the packet is sliced into compressed Opus frames:
//!
//! * **Code 0** (§3.2.2) — one frame, the remaining `N - 1` bytes.
//! * **Code 1** (§3.2.3) — two equal-size frames; `(N - 1)` MUST be even
//!   (requirement R3), each frame is `(N - 1) / 2` bytes.
//! * **Code 2** (§3.2.4) — two frames; a one- or two-byte §3.2.1
//!   length sequence gives the size of the first frame, the rest is the
//!   second frame (requirement R4).
//! * **Code 3** (§3.2.5) — a signalled frame count `M` plus optional
//!   Opus padding and, for VBR, `M - 1` per-frame length sequences. The
//!   final frame consumes whatever remains before the trailing padding.
//!
//! This module performs the §3.2 layer only. It returns the
//! compressed-frame byte slices (borrowed from the input buffer) so the
//! SILK / CELT decoders can be wired up against them in a subsequent
//! round. Length zero is a legal §3.2.1 result (DTX / lost frame); such
//! frames appear as empty slices in the returned list.

use crate::toc::{FrameCountCode, OpusTocByte};
use crate::Error;

/// Maximum compressed frame length permitted by the §3.2.1 two-byte
/// length encoding. RFC 6716 §3.2.1: "The maximum representable length
/// is 255 \* 4 + 255 = 1275 bytes." Requirement R2 forbids any
/// individual frame from exceeding this.
pub const MAX_FRAME_BYTES: usize = 1275;

/// Maximum number of frames a code-3 packet may signal. RFC 6716
/// §3.2.5: "M MUST NOT be zero, and the audio duration contained
/// within a packet MUST NOT exceed 120 ms \[R5\]. This limits the
/// maximum frame count for any frame size to 48 (for 2.5 ms frames)".
pub const MAX_FRAMES_PER_PACKET: u8 = 48;

/// A fully-parsed Opus packet: TOC byte plus the per-frame slices
/// recovered by walking the §3.2 frame-packing layer.
///
/// Frame slices borrow from the original packet buffer; the parsed
/// packet therefore carries the same lifetime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusPacket<'a> {
    /// Decoded TOC byte (RFC 6716 §3.1).
    pub toc: OpusTocByte,
    frames: Vec<&'a [u8], crate::bump::DecodeBump>,
    /// Number of trailing Opus-padding bytes (§3.2.5). Always zero for
    /// codes 0/1/2; non-zero only for a code-3 packet whose `p` bit is
    /// set.
    pub padding: usize,
}

impl<'a> OpusPacket<'a> {
    /// Parse one complete Opus packet (TOC byte plus §3.2 frame packing).
    ///
    /// Returns [`Error::EmptyPacket`] if the buffer is empty (R1), or
    /// [`Error::MalformedPacket`] if any other §3.2 requirement is
    /// violated.
    pub fn parse(packet: &'a [u8]) -> Result<Self, Error> {
        if packet.is_empty() {
            return Err(Error::EmptyPacket);
        }
        let toc = OpusTocByte::from_byte(packet[0]);
        let body = &packet[1..];

        let (frames, padding) = match toc.frame_count_code {
            FrameCountCode::One => parse_code0(body)?,
            FrameCountCode::TwoEqual => parse_code1(body)?,
            FrameCountCode::TwoUnequal => parse_code2(body)?,
            FrameCountCode::Arbitrary => parse_code3(body)?,
        };

        Ok(Self {
            toc,
            frames,
            padding,
        })
    }

    /// Compressed frame payloads in packet order.
    ///
    /// Each slice borrows from the original packet buffer. A zero-length
    /// slice represents a §3.2.1 DTX / lost-frame marker.
    pub fn frames(&self) -> &[&'a [u8]] {
        &self.frames
    }

    /// Number of compressed frames in the packet (equals
    /// `self.frames().len()`).
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Build an [`OpusPacket`] from already-parsed components. This is
    /// the construction path used by the Appendix-B self-delimited
    /// parser (`framing_self_delim::parse_self_delimited`), which
    /// derives the frame slices using the Appendix-B extra length
    /// field rather than the §3.2 implicit-length rules. Crate-private
    /// to keep external callers on [`OpusPacket::parse`] /
    /// [`crate::framing_self_delim::parse_self_delimited`].
    pub(crate) fn new_self_delim(
        toc: OpusTocByte,
        frames: Vec<&'a [u8], crate::bump::DecodeBump>,
        padding: usize,
    ) -> Self {
        Self {
            toc,
            frames,
            padding,
        }
    }
}

/// Decode a §3.2.1 length sequence at the start of `bytes`.
///
/// Returns `(length, consumed_byte_count)` on success. The first byte
/// is interpreted per RFC 6716 §3.2.1:
///
/// * `0` — no frame (DTX / lost). Length 0, one byte consumed.
/// * `1..=251` — that value in bytes. One byte consumed.
/// * `252..=255` — a second byte is required; the total length is
///   `(second * 4) + first`. Two bytes consumed.
pub(crate) fn decode_length(bytes: &[u8]) -> Result<(usize, usize), Error> {
    let first = *bytes.first().ok_or(Error::MalformedPacket)? as usize;
    if first < 252 {
        Ok((first, 1))
    } else {
        let second = *bytes.get(1).ok_or(Error::MalformedPacket)? as usize;
        Ok((second * 4 + first, 2))
    }
}

/// §3.2.2 Code 0: the entire body is one frame.
///
/// R2: the implicit length MUST NOT exceed 1275 bytes.
fn parse_code0(body: &[u8]) -> Result<(Vec<&[u8], crate::bump::DecodeBump>, usize), Error> {
    if body.len() > MAX_FRAME_BYTES {
        return Err(Error::MalformedPacket);
    }
    // Profluens patch: the frame list lives in the per-packet bump arena (reset before this parse
    // at the decoder entry point). It borrows the packet bytes and is dropped at the next packet.
    let mut frames = Vec::with_capacity_in(1, crate::bump::DecodeBump);
    frames.push(body);
    Ok((frames, 0))
}

/// §3.2.3 Code 1: two equal-size frames.
///
/// R3: `(N - 1)` (= `body.len()`) MUST be even. Each frame is
/// `body.len() / 2` bytes. R2 bounds each frame to ≤ 1275 bytes.
fn parse_code1(body: &[u8]) -> Result<(Vec<&[u8], crate::bump::DecodeBump>, usize), Error> {
    if body.len() % 2 != 0 {
        return Err(Error::MalformedPacket);
    }
    let half = body.len() / 2;
    if half > MAX_FRAME_BYTES {
        return Err(Error::MalformedPacket);
    }
    let mut frames = Vec::with_capacity_in(2, crate::bump::DecodeBump);
    frames.push(&body[..half]);
    frames.push(&body[half..]);
    Ok((frames, 0))
}

/// §3.2.4 Code 2: two frames with an explicit length for the first.
///
/// R4: the packet must contain enough bytes after the TOC for a valid
/// length, and the decoded `N1` must not exceed the bytes remaining
/// after the length sequence.
fn parse_code2(body: &[u8]) -> Result<(Vec<&[u8], crate::bump::DecodeBump>, usize), Error> {
    let (n1, consumed) = decode_length(body)?;
    if n1 > MAX_FRAME_BYTES {
        return Err(Error::MalformedPacket);
    }
    let after_len = &body[consumed..];
    if n1 > after_len.len() {
        return Err(Error::MalformedPacket);
    }
    let frame1 = &after_len[..n1];
    let frame2 = &after_len[n1..];
    if frame2.len() > MAX_FRAME_BYTES {
        return Err(Error::MalformedPacket);
    }
    let mut frames = Vec::with_capacity_in(2, crate::bump::DecodeBump);
    frames.push(frame1);
    frames.push(frame2);
    Ok((frames, 0))
}

/// §3.2.5 Code 3: signalled frame count, optional padding, optional
/// per-frame VBR lengths.
///
/// `body` starts at the frame-count byte (§3.1 placed the TOC byte
/// already). Layout:
///
/// 1. Frame-count byte: bit 0 = `v` (VBR), bit 1 = `p` (padding),
///    bits 2..=7 = `M` (frame count, 1..=48).
/// 2. If `p` set: one or more padding-length bytes per §3.2.5; the
///    sum of those bytes (with the §3.2.5 "255 means 254 + read next"
///    chain) is the number of trailing padding bytes appended after
///    the last frame. The header bytes that *encode* the padding
///    length count toward the P budget per the RFC's definition of P
///    (header bytes + padding bytes), so we deduct both halves from
///    what remains before the frame data.
/// 3. If `v` set: `M - 1` §3.2.1 length sequences; the final frame
///    consumes whatever remains in the body after subtracting the
///    padding.
/// 4. Otherwise (CBR): every frame is `R / M` bytes where
///    `R = N - 2 - P` (R6: R must be a non-negative multiple of M).
fn parse_code3(body: &[u8]) -> Result<(Vec<&[u8], crate::bump::DecodeBump>, usize), Error> {
    // R6/R7: a code-3 packet has at least the frame-count byte.
    // §3.2.5 Figure 5 (RFC bit diagrams are MSB-first): bit 0 = `v`
    // is the MSB (0x80), bit 1 = `p` is 0x40, and `M` is the low six
    // bits.
    let fc = *body.first().ok_or(Error::MalformedPacket)?;
    let v_bit = fc & 0x80 != 0;
    let p_bit = fc & 0x40 != 0;
    let m = fc & 0x3F;
    if m == 0 || m > MAX_FRAMES_PER_PACKET {
        return Err(Error::MalformedPacket);
    }
    let mut cursor = 1usize;

    // §3.2.5 padding-length chain. Each byte 0..=254 contributes its
    // value as padding bytes; 255 contributes 254 and demands another
    // length byte. The header bytes used to encode the chain are part
    // of P (the total bytes "added to the packet" budget), per the
    // RFC's definition.
    let mut padding_bytes: usize = 0;
    let mut padding_header_bytes: usize = 0;
    if p_bit {
        loop {
            let byte = *body.get(cursor).ok_or(Error::MalformedPacket)? as usize;
            cursor += 1;
            padding_header_bytes += 1;
            if byte == 255 {
                padding_bytes += 254;
            } else {
                padding_bytes += byte;
                break;
            }
        }
    }
    let total_padding = padding_bytes + padding_header_bytes;
    // R6/R7: P (header bytes + padding bytes themselves) MUST be no
    // more than N - 2. Here body.len() = N - 1, so P MUST be no more
    // than body.len() - 1.
    if total_padding + 1 > body.len() {
        return Err(Error::MalformedPacket);
    }

    let m = m as usize;
    // Profluens patch: per-packet bump arena (see parse_code0).
    let mut frames: Vec<&[u8], crate::bump::DecodeBump> = Vec::with_capacity_in(m, crate::bump::DecodeBump);

    if v_bit {
        // VBR: read M-1 lengths, then derive the last frame size from
        // the remaining bytes after the padding tail.
        let mut declared_lengths: Vec<usize, crate::bump::DecodeBump> =
        Vec::with_capacity_in(m.saturating_sub(1), crate::bump::DecodeBump);
        for _ in 0..m.saturating_sub(1) {
            let (n, consumed) = decode_length(&body[cursor..])?;
            if n > MAX_FRAME_BYTES {
                return Err(Error::MalformedPacket);
            }
            declared_lengths.push(n);
            cursor += consumed;
        }
        // Bytes still owed: padding + sum(declared) + final-frame.
        let remaining = body.len() - cursor;
        let declared_sum: usize = declared_lengths.iter().copied().sum();
        if declared_sum + padding_bytes > remaining {
            return Err(Error::MalformedPacket);
        }
        let last_len = remaining - declared_sum - padding_bytes;
        if last_len > MAX_FRAME_BYTES {
            return Err(Error::MalformedPacket);
        }
        for n in declared_lengths {
            frames.push(&body[cursor..cursor + n]);
            cursor += n;
        }
        frames.push(&body[cursor..cursor + last_len]);
    } else {
        // CBR: R = N - 2 - P, frames are R/M bytes each. body.len()
        // = N - 1, so R = body.len() - 1 - P. After accounting for
        // the frame-count byte (already consumed at cursor==1) and
        // any padding-header bytes (consumed during the chain) and
        // padding_bytes (trailing), the remaining body slice is R.
        let r = body.len() - cursor - padding_bytes;
        if r % m != 0 {
            return Err(Error::MalformedPacket);
        }
        let per = r / m;
        if per > MAX_FRAME_BYTES {
            return Err(Error::MalformedPacket);
        }
        for _ in 0..m {
            frames.push(&body[cursor..cursor + per]);
            cursor += per;
        }
    }

    Ok((frames, padding_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toc::{ChannelMapping, FrameCountCode};

    fn toc(config: u8, stereo: bool, code: u8) -> u8 {
        (config << 3) | ((stereo as u8) << 2) | (code & 0x03)
    }

    // ----- §3.2.1 length decoding -----

    #[test]
    fn length_single_byte_values_0_through_251() {
        for value in [0u8, 1, 100, 200, 251] {
            let (len, n) = decode_length(&[value]).unwrap();
            assert_eq!(len, value as usize);
            assert_eq!(n, 1);
        }
    }

    #[test]
    fn length_two_byte_boundary() {
        // first=252 second=0 -> 0*4 + 252 = 252
        let (len, n) = decode_length(&[252, 0]).unwrap();
        assert_eq!(len, 252);
        assert_eq!(n, 2);
        // first=255 second=255 -> 255*4 + 255 = 1275 (= MAX_FRAME_BYTES)
        let (len, n) = decode_length(&[255, 255]).unwrap();
        assert_eq!(len, MAX_FRAME_BYTES);
        assert_eq!(n, 2);
        // first=253 second=10 -> 10*4 + 253 = 293
        let (len, n) = decode_length(&[253, 10]).unwrap();
        assert_eq!(len, 293);
        assert_eq!(n, 2);
    }

    #[test]
    fn length_two_byte_missing_second_rejects() {
        assert_eq!(decode_length(&[252]), Err(Error::MalformedPacket));
        assert_eq!(decode_length(&[255]), Err(Error::MalformedPacket));
    }

    #[test]
    fn length_empty_rejects() {
        assert_eq!(decode_length(&[]), Err(Error::MalformedPacket));
    }

    // ----- §3.2.2 Code 0 -----

    #[test]
    fn code0_single_frame_round_trip() {
        // config=1 (SILK NB 20 ms), mono, c=0.
        let mut packet = vec![toc(1, false, 0)];
        packet.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 1);
        assert_eq!(parsed.frames()[0], &[0xAA, 0xBB, 0xCC, 0xDD]);
        assert_eq!(parsed.padding, 0);
        assert_eq!(parsed.toc.frame_count_code, FrameCountCode::One);
    }

    #[test]
    fn code0_toc_only_yields_empty_frame() {
        // A code-0 packet with no payload is legal: §3.2.1 allows a
        // zero-length frame (DTX / lost). Frames() therefore returns
        // one empty slice.
        let packet = [toc(1, false, 0)];
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 1);
        assert!(parsed.frames()[0].is_empty());
    }

    #[test]
    fn code0_rejects_oversize_frame() {
        let mut packet = vec![toc(1, false, 0)];
        packet.resize(1 + MAX_FRAME_BYTES + 1, 0);
        assert_eq!(OpusPacket::parse(&packet), Err(Error::MalformedPacket));
    }

    // ----- §3.2.3 Code 1 -----

    #[test]
    fn code1_two_equal_split_at_midpoint() {
        let mut packet = vec![toc(1, false, 1)];
        packet.extend_from_slice(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66]);
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 2);
        assert_eq!(parsed.frames()[0], &[0x11, 0x22, 0x33]);
        assert_eq!(parsed.frames()[1], &[0x44, 0x55, 0x66]);
        assert_eq!(parsed.toc.frame_count_code, FrameCountCode::TwoEqual);
    }

    #[test]
    fn code1_rejects_odd_body_per_r3() {
        let mut packet = vec![toc(1, false, 1)];
        packet.extend_from_slice(&[0x11, 0x22, 0x33]);
        assert_eq!(OpusPacket::parse(&packet), Err(Error::MalformedPacket));
    }

    #[test]
    fn code1_toc_only_two_empty_frames() {
        let packet = [toc(1, false, 1)];
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 2);
        assert!(parsed.frames()[0].is_empty());
        assert!(parsed.frames()[1].is_empty());
    }

    // ----- §3.2.4 Code 2 -----

    #[test]
    fn code2_single_byte_length() {
        // N1 = 3, second frame = 4 bytes.
        let mut packet = vec![toc(1, false, 2), 3];
        packet.extend_from_slice(&[0xA1, 0xA2, 0xA3]); // frame 1 (3 bytes)
        packet.extend_from_slice(&[0xB1, 0xB2, 0xB3, 0xB4]); // frame 2
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 2);
        assert_eq!(parsed.frames()[0], &[0xA1, 0xA2, 0xA3]);
        assert_eq!(parsed.frames()[1], &[0xB1, 0xB2, 0xB3, 0xB4]);
    }

    #[test]
    fn code2_two_byte_length() {
        // N1 = 4*4 + 252 = 268.
        let n1 = 268usize;
        let mut packet = vec![toc(1, false, 2), 252, 4];
        packet.extend(std::iter::repeat(0xAA).take(n1));
        packet.extend_from_slice(&[0xBB; 5]); // frame 2 = 5 bytes
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 2);
        assert_eq!(parsed.frames()[0].len(), n1);
        assert!(parsed.frames()[0].iter().all(|&b| b == 0xAA));
        assert_eq!(parsed.frames()[1], &[0xBB; 5]);
    }

    #[test]
    fn code2_one_byte_packet_invalid() {
        // RFC §3.2.4: "a 1-byte code 2 packet is always invalid".
        let packet = [toc(1, false, 2)];
        assert_eq!(OpusPacket::parse(&packet), Err(Error::MalformedPacket));
    }

    #[test]
    fn code2_two_byte_packet_only_legal_when_lengths_are_zero() {
        // body = [0] -> N1 = 0, second frame = 0. Legal.
        let packet = [toc(1, false, 2), 0];
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 2);
        assert!(parsed.frames()[0].is_empty());
        assert!(parsed.frames()[1].is_empty());

        // body = [5] -> N1 = 5 but no bytes remain. Reject per R4.
        let packet = [toc(1, false, 2), 5];
        assert_eq!(OpusPacket::parse(&packet), Err(Error::MalformedPacket));

        // body = [253] -> demands a second length byte that isn't there.
        let packet = [toc(1, false, 2), 253];
        assert_eq!(OpusPacket::parse(&packet), Err(Error::MalformedPacket));
    }

    #[test]
    fn code2_n1_exceeds_remaining_rejected_per_r4() {
        let mut packet = vec![toc(1, false, 2), 10];
        // Only 5 bytes follow but N1 = 10.
        packet.extend_from_slice(&[0; 5]);
        assert_eq!(OpusPacket::parse(&packet), Err(Error::MalformedPacket));
    }

    // ----- §3.2.5 Code 3 -----

    /// §3.2.5 Figure 5 wire layout: RFC bit diagrams are MSB-first,
    /// so `v` = 0x80, `p` = 0x40, and `M` occupies the LOW six bits.
    /// This pins the layout against a reference-encoder-shaped CBR
    /// packet (frame-count byte 0x41 = v:0 p:1 M:1, one padding-length
    /// byte, 4 padding bytes) — a roundtrip test alone cannot catch a
    /// writer and parser that agree on a *reversed* layout, which is
    /// exactly the bug this test guards against (every code-3 CBR
    /// packet from a real encoder was rejected as malformed).
    #[test]
    fn code3_frame_count_byte_wire_layout_matches_rfc_figure5() {
        // TOC 0x4B: config 9 (SILK WB 20 ms), mono, code 3.
        let mut pkt = vec![0x4Bu8, 0x41, 0x04];
        pkt.extend_from_slice(&[0xAA; 53]);
        pkt.extend_from_slice(&[0x00; 4]);
        assert_eq!(pkt.len(), 60);
        let parsed = OpusPacket::parse(&pkt).expect("real-encoder CBR padded packet");
        assert_eq!(parsed.frame_count(), 1);
        assert_eq!(parsed.frames()[0].len(), 53);
        assert_eq!(parsed.padding, 4);

        // The VBR flag is the MSB: 0x80 | M, with M-1 length bytes.
        let vbr_pkt = [0x4Bu8, 0x80 | 2, 3, 0x11, 0x22, 0x33, 0x44, 0x55];
        let parsed = OpusPacket::parse(&vbr_pkt).expect("VBR M=2");
        assert_eq!(parsed.frame_count(), 2);
        assert_eq!(parsed.frames()[0], &[0x11, 0x22, 0x33]);
        assert_eq!(parsed.frames()[1], &[0x44, 0x55]);

        // A byte with the OLD (reversed) reading of "v=0 p=1 M=1"
        // (0x06 = M:6 in the RFC layout) parses as M=6 CBR — not as a
        // padded single-frame packet.
        let old_layout = [0x4Bu8, 0x06, 1, 2, 3, 4, 5, 6];
        let parsed = OpusPacket::parse(&old_layout).expect("M=6 CBR, R=6");
        assert_eq!(parsed.frame_count(), 6);
        assert_eq!(parsed.padding, 0);
    }

    #[test]
    fn code3_cbr_no_padding() {
        // M = 3, vbr=0, p=0; three equal-size frames of 4 bytes each.
        let m: u8 = 3;
        let fc = m; // v=0 p=0
        let mut packet = vec![toc(15, false, 3), fc];
        packet.extend_from_slice(&[0xA1, 0xA2, 0xA3, 0xA4]);
        packet.extend_from_slice(&[0xB1, 0xB2, 0xB3, 0xB4]);
        packet.extend_from_slice(&[0xC1, 0xC2, 0xC3, 0xC4]);
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 3);
        assert_eq!(parsed.frames()[0], &[0xA1, 0xA2, 0xA3, 0xA4]);
        assert_eq!(parsed.frames()[1], &[0xB1, 0xB2, 0xB3, 0xB4]);
        assert_eq!(parsed.frames()[2], &[0xC1, 0xC2, 0xC3, 0xC4]);
        assert_eq!(parsed.padding, 0);
    }

    #[test]
    fn code3_cbr_rejects_non_multiple_of_m_per_r6() {
        // M = 3 but only 5 payload bytes -> not a multiple of 3.
        let fc = 3 << 2;
        let mut packet = vec![toc(15, false, 3), fc];
        packet.extend_from_slice(&[0; 5]);
        assert_eq!(OpusPacket::parse(&packet), Err(Error::MalformedPacket));
    }

    #[test]
    fn code3_cbr_with_padding() {
        // M = 2, p=1, v=0, padding chain = single byte "5" -> 5 trailing
        // zero bytes. Then 2 frames of 3 bytes each.
        let m: u8 = 2;
        let fc = 0x40 | m; // p=1, v=0
        let mut packet = vec![toc(15, false, 3), fc, 5];
        packet.extend_from_slice(&[0xD1, 0xD2, 0xD3]);
        packet.extend_from_slice(&[0xE1, 0xE2, 0xE3]);
        packet.extend_from_slice(&[0x00; 5]);
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 2);
        assert_eq!(parsed.frames()[0], &[0xD1, 0xD2, 0xD3]);
        assert_eq!(parsed.frames()[1], &[0xE1, 0xE2, 0xE3]);
        assert_eq!(parsed.padding, 5);
    }

    #[test]
    fn code3_vbr_no_padding() {
        // M = 3, v=1, p=0. M-1 = 2 length sequences then last-frame
        // is implicit. Frame sizes: 2, 4, 1.
        let m: u8 = 3;
        let fc = 0x80 | m;
        let mut packet = vec![toc(15, false, 3), fc, 2, 4];
        packet.extend_from_slice(&[0xA1, 0xA2]);
        packet.extend_from_slice(&[0xB1, 0xB2, 0xB3, 0xB4]);
        packet.push(0xC1);
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 3);
        assert_eq!(parsed.frames()[0], &[0xA1, 0xA2]);
        assert_eq!(parsed.frames()[1], &[0xB1, 0xB2, 0xB3, 0xB4]);
        assert_eq!(parsed.frames()[2], &[0xC1]);
        assert_eq!(parsed.padding, 0);
    }

    #[test]
    fn code3_vbr_with_padding() {
        // Mirrors fixture: 4 VBR frames + 7 padding bytes.
        // Sizes used in trace: 99, 77, 66, 68 -> total 310; +3
        // length bytes (one each for the 3 declared) + frame-count
        // byte + padding-length byte + 7 padding bytes + TOC.
        // body (without TOC) = fc + pad_len + 3 lengths + 99 + 77 + 66
        //                      + 68 + 7 pad = 1 + 1 + 3 + 310 + 7 = 322.
        // Total packet = 323 bytes.
        let m: u8 = 4;
        let fc = 0xC0 | m; // p=1, v=1
        let pad_len = 7u8;
        let sizes = [99u8, 77, 66, 68];
        let mut packet = vec![toc(15, false, 3), fc, pad_len];
        // M-1 = 3 declared lengths.
        packet.extend_from_slice(&sizes[..3]);
        for &n in sizes.iter() {
            packet.extend(std::iter::repeat(0xAA).take(n as usize));
        }
        packet.extend_from_slice(&[0u8; 7]);
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 4);
        assert_eq!(parsed.frames()[0].len(), 99);
        assert_eq!(parsed.frames()[1].len(), 77);
        assert_eq!(parsed.frames()[2].len(), 66);
        assert_eq!(parsed.frames()[3].len(), 68);
        assert_eq!(parsed.padding, 7);
        // Confirm TOC fields survive composition.
        assert_eq!(parsed.toc.config, 15);
        assert_eq!(parsed.toc.channels, ChannelMapping::Mono);
        assert_eq!(parsed.toc.frame_count_code, FrameCountCode::Arbitrary);
    }

    #[test]
    fn code3_padding_chain_255_extension() {
        // Padding-chain spec: 255 contributes 254 padding bytes and
        // demands another length byte. Verify that "255, 3" => 257
        // padding bytes (254 + 3).
        let m: u8 = 1;
        let fc = 0x40 | m; // p=1, v=0
        let mut packet = vec![toc(15, false, 3), fc, 255, 3];
        // Frame body: any non-padding payload. 4 bytes is fine.
        packet.extend_from_slice(&[0xF1, 0xF2, 0xF3, 0xF4]);
        packet.extend(std::iter::repeat(0u8).take(257));
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 1);
        assert_eq!(parsed.frames()[0], &[0xF1, 0xF2, 0xF3, 0xF4]);
        assert_eq!(parsed.padding, 257);
    }

    #[test]
    fn code3_rejects_m_zero_per_r5() {
        let fc = 0; // M = 0
        let packet = [toc(15, false, 3), fc];
        assert_eq!(OpusPacket::parse(&packet), Err(Error::MalformedPacket));
    }

    #[test]
    fn code3_rejects_missing_frame_count_byte() {
        // Code-3 packet with only the TOC byte is invalid (R6/R7
        // require N >= 2).
        let packet = [toc(15, false, 3)];
        assert_eq!(OpusPacket::parse(&packet), Err(Error::MalformedPacket));
    }

    #[test]
    fn code3_rejects_padding_overrunning_body() {
        // M = 1, p=1, padding declared as 200 but only 5 bytes of
        // payload follow.
        let fc = 0x40 | 1;
        let mut packet = vec![toc(15, false, 3), fc, 200];
        packet.extend_from_slice(&[0; 5]);
        assert_eq!(OpusPacket::parse(&packet), Err(Error::MalformedPacket));
    }

    #[test]
    fn code3_vbr_rejects_lengths_exceeding_remaining() {
        // M = 3, v=1, lengths 100, 100 but only 50 bytes remain.
        let fc = 0x80 | 3;
        let mut packet = vec![toc(15, false, 3), fc, 100, 100];
        packet.extend_from_slice(&[0; 50]);
        assert_eq!(OpusPacket::parse(&packet), Err(Error::MalformedPacket));
    }

    #[test]
    fn empty_packet_rejected() {
        assert_eq!(OpusPacket::parse(&[]), Err(Error::EmptyPacket));
    }

    // ----- Composed cases: TOC + frame packing -----

    #[test]
    fn code3_vbr_max_frame_count_48() {
        // M = 48 (cap per R5 at 2.5 ms frames), v=1, p=0.
        // Each frame size = 1 byte; 47 declared + 1 implicit.
        let m: u8 = MAX_FRAMES_PER_PACKET;
        let fc = 0x80 | m;
        let mut packet = vec![toc(16, false, 3), fc];
        packet.extend(std::iter::repeat(1u8).take(47)); // 47 declared lengths of 1
        packet.extend(std::iter::repeat(0xCC).take(48));
        let parsed = OpusPacket::parse(&packet).unwrap();
        assert_eq!(parsed.frame_count(), 48);
        for f in parsed.frames() {
            assert_eq!(f, &[0xCC]);
        }
    }
}
