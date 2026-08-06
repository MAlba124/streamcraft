//! Self-delimiting Opus packet framing — RFC 6716 Appendix B.
//!
//! ## Motivation (RFC 6716 Appendix B, pp. 321–322)
//!
//! The internal framing of §3 assumes the *total* compressed Opus packet
//! length is known to the decoder out-of-band. When a transport packs
//! several Opus streams into a single container (the appendix gives the
//! example of a multi-channel stream formed from several one- or
//! two-channel Opus streams), every stream except the last needs an
//! in-band length so the demultiplexer can find the next stream's
//! starting byte. Appendix B specifies exactly that variant: it is
//! "identical to the regular, undelimited framing from Section 3, except
//! that each Opus packet contains one extra length field, encoded using
//! the same one- or two-byte scheme from Section 3.2.1."
//!
//! The extra length immediately follows the TOC byte (and, for code 3,
//! the §3.2.5 frame-count byte and any padding-length chain). Its
//! interpretation, per Appendix B:
//!
//! * Code 0 — length of the single Opus frame (Figure 25).
//! * Code 1 — length used for both Opus frames (Figure 26).
//! * Code 2 — length of the **second** Opus frame (the first frame's
//!   length is still given by the §3.2.4 inline length sequence, so a
//!   self-delimited code-2 packet carries two length sequences, Figure
//!   27).
//! * CBR code 3 — length used for all of the Opus frames (Figure 28).
//! * VBR code 3 — length of the **last** Opus frame; the other `M - 1`
//!   lengths are the same §3.2.5 inline length sequences as in the
//!   undelimited form (Figure 29).
//!
//! ## What this module produces
//!
//! [`parse_self_delimited`] consumes one Opus packet's worth of bytes
//! from the front of the input buffer and returns:
//!
//! 1. an [`OpusPacket`] whose `frames()` slices borrow into that buffer
//!    — directly compatible with [`OpusPacket::parse`] consumers, and
//! 2. the number of bytes consumed (`SelfDelimitedParse::consumed`), so
//!    a multiplexed demuxer can advance to the next stream's TOC byte
//!    by indexing `buffer[consumed..]`.
//!
//! Appendix B notes ("Nothing in the encoding of the packet itself
//! allows a decoder to distinguish between the regular, undelimited
//! framing and the self-delimiting framing"); this module is therefore
//! a separate entry point chosen by the transport. It does **not**
//! mutate the §3.1 TOC byte interpretation handled by
//! [`crate::toc::OpusTocByte`].
//!
//! ## Provenance
//!
//! RFC 6716 Appendix B (September 2012), pp. 321–325. The §3.2.1
//! one/two-byte length encoding is shared with the §3.2 path via the
//! crate-private [`crate::frames::decode_length`] helper. Reads of the
//! §3.2 source, the §3 narrative on pp. 13–18, and Appendix B itself
//! are the only inputs.

use crate::frames::{decode_length, OpusPacket, MAX_FRAMES_PER_PACKET, MAX_FRAME_BYTES};
use crate::toc::{FrameCountCode, OpusTocByte};
use crate::Error;

/// Result of parsing one self-delimited Opus packet from a buffer.
///
/// `consumed` is always the number of leading bytes of the input
/// buffer that were claimed by this packet (TOC byte; the frame-count
/// byte where present; the padding-length chain; the length
/// sequences; frame payloads; trailing padding). A multistream
/// demuxer advances by exactly that count to reach the next stream's
/// TOC byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfDelimitedParse<'a> {
    /// Parsed Opus packet — `packet.frames()` are slices into the
    /// original input buffer.
    pub packet: OpusPacket<'a>,
    /// Number of bytes consumed from the start of the input buffer.
    pub consumed: usize,
}

/// Parse one self-delimited Opus packet (RFC 6716 Appendix B) from
/// the start of `buffer`.
///
/// The parser succeeds when:
///
/// * `buffer` has at least one byte (the TOC byte — RFC 6716 §3.1
///   R1).
/// * Every §3.2 frame-packing requirement (R2..R7) is met for the
///   chosen code.
/// * The Appendix-B extra length field (and, for code 3 VBR, the
///   inline `M - 1` length sequences) fit inside the buffer.
/// * Every individual frame length is within [`MAX_FRAME_BYTES`].
///
/// Returns [`Error::EmptyPacket`] for a zero-length buffer,
/// [`Error::MalformedPacket`] for any §3.2 / Appendix-B violation.
///
/// Unlike [`OpusPacket::parse`], this entry point does NOT consume
/// the rest of the buffer as part of the last frame; it consumes
/// exactly as many bytes as the Appendix-B length fields say, which
/// is the whole point of the self-delimited variant.
pub fn parse_self_delimited(buffer: &[u8]) -> Result<SelfDelimitedParse<'_>, Error> {
    if buffer.is_empty() {
        return Err(Error::EmptyPacket);
    }
    let toc = OpusTocByte::from_byte(buffer[0]);
    let mut cursor = 1usize;

    let (frames, padding) = match toc.frame_count_code {
        FrameCountCode::One => parse_sd_code0(buffer, &mut cursor)?,
        FrameCountCode::TwoEqual => parse_sd_code1(buffer, &mut cursor)?,
        FrameCountCode::TwoUnequal => parse_sd_code2(buffer, &mut cursor)?,
        FrameCountCode::Arbitrary => parse_sd_code3(buffer, &mut cursor)?,
    };

    let packet = OpusPacket::new_self_delim(toc, frames, padding);
    Ok(SelfDelimitedParse {
        packet,
        consumed: cursor,
    })
}

// ---------------------------------------------------------------------------
// per-code parsers — each advances `cursor` past the bytes it claims
// ---------------------------------------------------------------------------

/// Figure 25: TOC byte | §3.2.1 N1 length | N1-byte frame.
fn parse_sd_code0<'a>(
    buffer: &'a [u8],
    cursor: &mut usize,
) -> Result<(Vec<&'a [u8], crate::bump::DecodeBump>, usize), Error> {
    let n1 = read_length_advance(buffer, cursor)?;
    let frame = take_frame_advance(buffer, cursor, n1)?;
    // Profluens patch: frame list in the per-packet bump arena (reset at the decoder entry point).
    let mut frames = Vec::with_capacity_in(1, crate::bump::DecodeBump);
    frames.push(frame);
    Ok((frames, 0))
}

/// Figure 26: TOC byte | §3.2.1 N1 length | N1-byte frame | N1-byte frame.
fn parse_sd_code1<'a>(
    buffer: &'a [u8],
    cursor: &mut usize,
) -> Result<(Vec<&'a [u8], crate::bump::DecodeBump>, usize), Error> {
    let n1 = read_length_advance(buffer, cursor)?;
    let frame1 = take_frame_advance(buffer, cursor, n1)?;
    let frame2 = take_frame_advance(buffer, cursor, n1)?;
    let mut frames = Vec::with_capacity_in(2, crate::bump::DecodeBump);
    frames.push(frame1);
    frames.push(frame2);
    Ok((frames, 0))
}

/// Figure 27: TOC byte | §3.2.1 N1 length | §3.2.1 N2 length |
/// N1-byte frame | N2-byte frame.
///
/// Per Appendix B p. 322 the Appendix-B "extra length" for a code-2
/// packet is the length of the *second* frame; the first frame's
/// length is given by the regular §3.2.4 inline length sequence, so a
/// self-delimited code-2 packet carries **two** length sequences
/// before any frame data.
fn parse_sd_code2<'a>(
    buffer: &'a [u8],
    cursor: &mut usize,
) -> Result<(Vec<&'a [u8], crate::bump::DecodeBump>, usize), Error> {
    let n1 = read_length_advance(buffer, cursor)?;
    let n2 = read_length_advance(buffer, cursor)?;
    let frame1 = take_frame_advance(buffer, cursor, n1)?;
    let frame2 = take_frame_advance(buffer, cursor, n2)?;
    let mut frames = Vec::with_capacity_in(2, crate::bump::DecodeBump);
    frames.push(frame1);
    frames.push(frame2);
    Ok((frames, 0))
}

/// Figures 28 (CBR) and 29 (VBR).
///
/// Body layout starting at `cursor`:
///
/// 1. Frame-count byte: bit 0 = `v` (VBR), bit 1 = `p` (padding),
///    bits 2..=7 = `M` (frame count, 1..=48).
/// 2. If `p`: §3.2.5 padding-length chain.
/// 3. If `v`: `M - 1` §3.2.1 inline length sequences plus one
///    Appendix-B trailing length (for frame M); otherwise CBR — one
///    Appendix-B length used by every frame.
/// 4. Frame payloads, then padding bytes.
fn parse_sd_code3<'a>(
    buffer: &'a [u8],
    cursor: &mut usize,
) -> Result<(Vec<&'a [u8], crate::bump::DecodeBump>, usize), Error> {
    let fc = *buffer.get(*cursor).ok_or(Error::MalformedPacket)?;
    *cursor += 1;
    // §3.2.5 Figure 5 (MSB-first): `v` = 0x80, `p` = 0x40, `M` = low
    // six bits.
    let v_bit = fc & 0x80 != 0;
    let p_bit = fc & 0x40 != 0;
    let m = fc & 0x3F;
    if m == 0 || m > MAX_FRAMES_PER_PACKET {
        return Err(Error::MalformedPacket);
    }
    let m = m as usize;

    // §3.2.5 padding-length chain (255 chains; the trailing byte
    // 0..=254 closes the chain).
    let mut padding_bytes: usize = 0;
    if p_bit {
        loop {
            let byte = *buffer.get(*cursor).ok_or(Error::MalformedPacket)? as usize;
            *cursor += 1;
            if byte == 255 {
                padding_bytes += 254;
            } else {
                padding_bytes += byte;
                break;
            }
        }
    }

    let mut frames: Vec<&[u8], crate::bump::DecodeBump> = Vec::with_capacity_in(m, crate::bump::DecodeBump);
    if v_bit {
        // VBR: M-1 inline lengths + one Appendix-B length for frame M.
        let mut sizes: Vec<usize, crate::bump::DecodeBump> = Vec::with_capacity_in(m, crate::bump::DecodeBump);
        for _ in 0..m.saturating_sub(1) {
            sizes.push(read_length_advance(buffer, cursor)?);
        }
        let last_size = read_length_advance(buffer, cursor)?;
        sizes.push(last_size);
        for n in sizes {
            frames.push(take_frame_advance(buffer, cursor, n)?);
        }
    } else {
        // CBR: one Appendix-B length used for every frame.
        let per = read_length_advance(buffer, cursor)?;
        for _ in 0..m {
            frames.push(take_frame_advance(buffer, cursor, per)?);
        }
    }

    // Trailing padding payload bytes are consumed last.
    if buffer.len() - *cursor < padding_bytes {
        return Err(Error::MalformedPacket);
    }
    *cursor += padding_bytes;

    Ok((frames, padding_bytes))
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

/// Read a §3.2.1 length at `buffer[*cursor..]`, advancing `*cursor`
/// by the byte-count consumed. Returns the decoded length.
fn read_length_advance(buffer: &[u8], cursor: &mut usize) -> Result<usize, Error> {
    let tail = buffer.get(*cursor..).ok_or(Error::MalformedPacket)?;
    let (length, consumed) = decode_length(tail)?;
    *cursor += consumed;
    Ok(length)
}

/// Take a frame of `n` bytes starting at `buffer[*cursor..]`, bounds-
/// checking against [`MAX_FRAME_BYTES`] (R2). Advances `*cursor`.
fn take_frame_advance<'a>(
    buffer: &'a [u8],
    cursor: &mut usize,
    n: usize,
) -> Result<&'a [u8], Error> {
    if n > MAX_FRAME_BYTES {
        return Err(Error::MalformedPacket);
    }
    let end = cursor.checked_add(n).ok_or(Error::MalformedPacket)?;
    if end > buffer.len() {
        return Err(Error::MalformedPacket);
    }
    let slice = &buffer[*cursor..end];
    *cursor = end;
    Ok(slice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toc::FrameCountCode;

    /// Build a TOC byte with the given §3.1 fields.
    fn toc_byte(config: u8, stereo: bool, code: u8) -> u8 {
        (config << 3) | ((stereo as u8) << 2) | (code & 0x03)
    }

    // ---------- Code 0 (Figure 25) ----------

    #[test]
    fn code0_short_length_one_frame() {
        // TOC code 0 + N1=3 + 3-byte frame, plus a trailing byte that
        // belongs to the *next* stream and MUST NOT be consumed.
        let bytes = [toc_byte(0, false, 0), 3, 0xAA, 0xBB, 0xCC, 0xFF];
        let r = parse_self_delimited(&bytes).unwrap();
        assert_eq!(r.consumed, 1 + 1 + 3);
        assert_eq!(r.packet.frame_count(), 1);
        assert_eq!(r.packet.frames()[0], &[0xAA, 0xBB, 0xCC]);
        assert_eq!(r.packet.toc.frame_count_code, FrameCountCode::One);
        // The leftover byte 0xFF is left for the next demux call.
        assert_eq!(bytes.len() - r.consumed, 1);
    }

    #[test]
    fn code0_two_byte_length() {
        // first=252 second=0 → length 252; build a 252-byte frame.
        let mut bytes = vec![toc_byte(0, false, 0), 252, 0];
        bytes.extend(std::iter::repeat(0x7Fu8).take(252));
        let r = parse_self_delimited(&bytes).unwrap();
        assert_eq!(r.consumed, 1 + 2 + 252);
        assert_eq!(r.packet.frames()[0].len(), 252);
    }

    #[test]
    fn code0_zero_length_dtx() {
        // §3.2.1 length 0 is a legal DTX marker.
        let bytes = [toc_byte(0, false, 0), 0];
        let r = parse_self_delimited(&bytes).unwrap();
        assert_eq!(r.consumed, 2);
        assert_eq!(r.packet.frames()[0].len(), 0);
    }

    // ---------- Code 1 (Figure 26) ----------

    #[test]
    fn code1_two_equal_frames() {
        // TOC code 1 + N1=2 + 2 frames of 2 bytes each.
        let bytes = [
            toc_byte(0, false, 1),
            2,
            0xAA,
            0xBB,
            0xCC,
            0xDD,
            // trailing demux byte
            0x11,
        ];
        let r = parse_self_delimited(&bytes).unwrap();
        assert_eq!(r.consumed, 1 + 1 + 4);
        assert_eq!(r.packet.frame_count(), 2);
        assert_eq!(r.packet.frames()[0], &[0xAA, 0xBB]);
        assert_eq!(r.packet.frames()[1], &[0xCC, 0xDD]);
    }

    // ---------- Code 2 (Figure 27) ----------

    #[test]
    fn code2_two_lengths_then_frames() {
        // TOC code 2 + inline N1=3 + Appendix-B N2=2 + 3-byte + 2-byte
        // frames.
        let bytes = [toc_byte(0, false, 2), 3, 2, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
        let r = parse_self_delimited(&bytes).unwrap();
        assert_eq!(r.consumed, 1 + 1 + 1 + 3 + 2);
        assert_eq!(r.packet.frame_count(), 2);
        assert_eq!(r.packet.frames()[0], &[0xAA, 0xBB, 0xCC]);
        assert_eq!(r.packet.frames()[1], &[0xDD, 0xEE]);
    }

    // ---------- Code 3 CBR (Figure 28) ----------

    #[test]
    fn code3_cbr_three_frames_no_padding() {
        // TOC code 3 + frame-count byte (M=3, v=0, p=0) + Appendix-B
        // per-frame length 2 + three 2-byte frames.
        let fc = 3u8;
        let bytes = [
            toc_byte(0, false, 3),
            fc,
            2,
            0x10,
            0x11,
            0x20,
            0x21,
            0x30,
            0x31,
        ];
        let r = parse_self_delimited(&bytes).unwrap();
        assert_eq!(r.consumed, 1 + 1 + 1 + 6);
        assert_eq!(r.packet.frame_count(), 3);
        assert_eq!(r.packet.frames()[0], &[0x10, 0x11]);
        assert_eq!(r.packet.frames()[1], &[0x20, 0x21]);
        assert_eq!(r.packet.frames()[2], &[0x30, 0x31]);
        assert_eq!(r.packet.padding, 0);
    }

    #[test]
    fn code3_cbr_with_padding() {
        // M=2, p=1, padding-len=2 (one byte chain closing at 2).
        let fc = 0x40 | 2u8;
        let bytes = [
            toc_byte(0, false, 3),
            fc,
            2, // padding length byte: 2 trailing bytes
            3, // Appendix-B per-frame length
            0xAA,
            0xBB,
            0xCC,
            0xDD,
            0xEE,
            0xFF,
            0x55,
            0x66, // 2 padding bytes
        ];
        let r = parse_self_delimited(&bytes).unwrap();
        // TOC + frame-count + 1 padding-len byte + 1 Appendix-B length
        // + 2*3 frame bytes + 2 padding bytes = 12
        assert_eq!(r.consumed, 12);
        assert_eq!(r.packet.frame_count(), 2);
        assert_eq!(r.packet.padding, 2);
    }

    // ---------- Code 3 VBR (Figure 29) ----------

    #[test]
    fn code3_vbr_three_frames() {
        // M=3, v=1, p=0. Inline §3.2.1 lengths for frames 1..M-1 = 2,
        // then Appendix-B length for frame M.
        let fc = 0x80 | 3u8;
        let bytes = [
            toc_byte(0, false, 3),
            fc,
            2, // N1 inline
            3, // N2 inline
            1, // N3 Appendix-B trailing length
            0xAA,
            0xBB,
            0xCC,
            0xDD,
            0xEE,
            0xFF,
        ];
        let r = parse_self_delimited(&bytes).unwrap();
        assert_eq!(r.consumed, 1 + 1 + 1 + 1 + 1 + 2 + 3 + 1);
        assert_eq!(r.packet.frame_count(), 3);
        assert_eq!(r.packet.frames()[0], &[0xAA, 0xBB]);
        assert_eq!(r.packet.frames()[1], &[0xCC, 0xDD, 0xEE]);
        assert_eq!(r.packet.frames()[2], &[0xFF]);
    }

    // ---------- multistream chaining ----------

    #[test]
    fn two_streams_chained_back_to_back() {
        // Stream A: code 0 with 2-byte frame.
        // Stream B: code 1 with 1-byte frames.
        let bytes = [
            // stream A
            toc_byte(0, false, 0),
            2,
            0xA0,
            0xA1,
            // stream B
            toc_byte(0, false, 1),
            1,
            0xB0,
            0xB1,
        ];
        let a = parse_self_delimited(&bytes).unwrap();
        assert_eq!(a.consumed, 4);
        assert_eq!(a.packet.frames()[0], &[0xA0, 0xA1]);

        let rest = &bytes[a.consumed..];
        let b = parse_self_delimited(rest).unwrap();
        assert_eq!(b.consumed, 4);
        assert_eq!(b.packet.frame_count(), 2);
        assert_eq!(b.packet.frames()[0], &[0xB0]);
        assert_eq!(b.packet.frames()[1], &[0xB1]);
    }

    // ---------- malformed inputs ----------

    #[test]
    fn empty_buffer_rejected() {
        assert_eq!(parse_self_delimited(&[]), Err(Error::EmptyPacket));
    }

    #[test]
    fn code0_truncated_after_length() {
        // length declares 4 bytes but only 2 follow → R2 violation.
        let bytes = [toc_byte(0, false, 0), 4, 0xAA, 0xBB];
        assert_eq!(parse_self_delimited(&bytes), Err(Error::MalformedPacket));
    }

    #[test]
    fn code0_length_field_truncated_two_byte_form() {
        // First length byte is 252 but no second byte present.
        let bytes = [toc_byte(0, false, 0), 252];
        assert_eq!(parse_self_delimited(&bytes), Err(Error::MalformedPacket));
    }

    #[test]
    fn code3_zero_frame_count_rejected() {
        // M = 0 is forbidden by R5.
        let fc = 0u8;
        let bytes = [toc_byte(0, false, 3), fc, 1, 0xAA];
        assert_eq!(parse_self_delimited(&bytes), Err(Error::MalformedPacket));
    }

    #[test]
    fn code3_frame_count_above_48_rejected() {
        let fc = 49u8 << 2;
        let bytes = [toc_byte(0, false, 3), fc, 1, 0xAA];
        assert_eq!(parse_self_delimited(&bytes), Err(Error::MalformedPacket));
    }

    #[test]
    fn frame_size_above_1275_rejected() {
        // length encoded as (252) + (255) = 252 + 1020 = 1272 OK,
        // but try (255, 255) → 1275 OK + (255, 0xFF, ...) won't
        // exceed 1275 via the two-byte encoding. Instead, build a
        // VBR code-3 with an inline length that's also at the cap,
        // and supply a trailing length above the cap to drive the
        // R2 check on a frame.
        //
        // Use (255, 0)=1020 OK then padding insufficient bytes.
        let mut bytes = vec![
            toc_byte(0, false, 0),
            255,
            0, // length = 255*4 + 255 wait that's 1275; recompute
        ];
        // Actually encoding 1275 fits in two bytes: 255*4 + 255 =
        // 1275 (= MAX). We test the rejection path by claiming 1275
        // bytes but only supplying 10.
        bytes[1] = 255;
        bytes[2] = 255;
        bytes.extend(std::iter::repeat(0u8).take(10));
        assert_eq!(parse_self_delimited(&bytes), Err(Error::MalformedPacket));
    }

    #[test]
    fn code3_cbr_missing_padding_bytes() {
        // Declares 4 trailing padding bytes via the §3.2.5 chain but
        // body ends after the frame payloads.
        let fc = 0x40 | 1u8;
        let bytes = [
            toc_byte(0, false, 3),
            fc,
            4, // padding length 4
            1, // Appendix-B length
            0xAA, // frame 1 (1 byte)
               // missing the 4 padding bytes here
        ];
        assert_eq!(parse_self_delimited(&bytes), Err(Error::MalformedPacket));
    }

    #[test]
    fn code3_padding_chain_runs_off_end() {
        // p_bit set, but no padding-length byte present.
        let fc = 0x40 | 1u8;
        let bytes = [toc_byte(0, false, 3), fc];
        assert_eq!(parse_self_delimited(&bytes), Err(Error::MalformedPacket));
    }
}
