//! Opus packet frame-packing **writer** (RFC 6716 §3.2 + Appendix B)
//! — the write-side mirror of [`crate::frames::OpusPacket::parse`] and
//! [`crate::framing_self_delim::parse_self_delimited`].
//!
//! Given a §3.1 TOC byte and the compressed frame payloads, these
//! entry points emit the §3.2 framing layer around them:
//!
//! * **Code 0** (§3.2.2) — `TOC | frame`.
//! * **Code 1** (§3.2.3) — `TOC | frame | frame` (equal sizes, R3).
//! * **Code 2** (§3.2.4) — `TOC | len(frame1) | frame1 | frame2`.
//! * **Code 3** (§3.2.5) — `TOC | count byte | [padding chain] |
//!   [VBR lengths] | frames | [padding]`, with the `v` / `p` bits,
//!   the §3.2.5 255-chained padding length, and — for VBR — the
//!   `M - 1` §3.2.1 length sequences (the last frame's length stays
//!   implicit).
//!
//! [`compose_self_delimited`] emits the Appendix-B variant instead:
//! one extra §3.2.1 length field making the packet self-terminating
//! (code 0: the frame length; code 1: the shared length; code 2: the
//! *second* frame's length; code 3 CBR: the shared length; code 3
//! VBR: the *last* frame's length — Figures 25-29), so packets can be
//! chained back-to-back inside a multistream payload and re-split by
//! [`crate::framing_self_delim::parse_self_delimited`].
//!
//! Every §3.2 requirement the parsers enforce is validated here
//! before writing (R1-R7 as applicable): per-frame lengths within
//! [`MAX_FRAME_BYTES`], code-1 length equality, code-3 frame counts
//! in `1..=48` with the R5 120 ms packet-duration bound, and CBR
//! length uniformity. Composed packets therefore always reparse to
//! the same frames, padding, and TOC byte.

use crate::frames::{MAX_FRAMES_PER_PACKET, MAX_FRAME_BYTES};
use crate::toc::{FrameCountCode, OpusTocByte};
use crate::Error;

/// Encode one §3.2.1 length sequence (0..=1275) into `out`.
///
/// * `0..=251` — one byte, the value itself.
/// * `252..=1275` — two bytes: `first ∈ 252..=255` with
///   `first ≡ length (mod 4)` offset from 252, then
///   `second = (length - first) / 4`, so that the §3.2.1 decode
///   `second * 4 + first` reproduces `length`.
///
/// Errors when `length > 1275` (R2: the two-byte form tops out at
/// `255 * 4 + 255`).
pub fn encode_length(length: usize, out: &mut Vec<u8>) -> Result<(), Error> {
    if length > MAX_FRAME_BYTES {
        return Err(Error::MalformedPacket);
    }
    if length < 252 {
        out.push(length as u8);
    } else {
        let first = 252 + ((length - 252) % 4);
        let second = (length - first) / 4;
        out.push(first as u8);
        out.push(second as u8);
    }
    Ok(())
}

/// Write the §3.2.5 padding-length chain for `padding` trailing bytes
/// into `out`: each `255` byte contributes 254 bytes and continues the
/// chain; the closing byte (`0..=254`) contributes its value.
fn write_padding_chain(mut padding: usize, out: &mut Vec<u8>) {
    while padding >= 255 {
        out.push(255);
        padding -= 254;
    }
    out.push(padding as u8);
}

/// Validate the shared per-code frame-shape rules and return the
/// parsed TOC (R2 per-frame bound; counts per code; code-1 equality).
fn validate_shape(toc_byte: u8, frames: &[&[u8]]) -> Result<OpusTocByte, Error> {
    let toc = OpusTocByte::from_byte(toc_byte);
    for f in frames {
        if f.len() > MAX_FRAME_BYTES {
            return Err(Error::MalformedPacket);
        }
    }
    match toc.frame_count_code {
        FrameCountCode::One => {
            if frames.len() != 1 {
                return Err(Error::MalformedPacket);
            }
        }
        FrameCountCode::TwoEqual => {
            if frames.len() != 2 || frames[0].len() != frames[1].len() {
                return Err(Error::MalformedPacket);
            }
        }
        FrameCountCode::TwoUnequal => {
            if frames.len() != 2 {
                return Err(Error::MalformedPacket);
            }
        }
        FrameCountCode::Arbitrary => {
            let m = frames.len();
            if m == 0 || m > MAX_FRAMES_PER_PACKET as usize {
                return Err(Error::MalformedPacket);
            }
            // R5: the packet's audio duration MUST NOT exceed 120 ms.
            if m as u32 * toc.frame_size_tenths_ms as u32 > 1200 {
                return Err(Error::MalformedPacket);
            }
        }
    }
    Ok(toc)
}

/// Compose one regular (undelimited) Opus packet: the §3.2 framing
/// layer around `frames`, selected by the TOC byte's frame-count code.
///
/// For a code-3 packet the CBR/VBR bit is chosen automatically: CBR
/// when every frame has the same length, VBR otherwise. Use
/// [`compose_packet_code3`] to force VBR framing or to append §3.2.5
/// padding.
///
/// Errors on any §3.2 shape violation (see the module docs).
pub fn compose_packet(toc_byte: u8, frames: &[&[u8]]) -> Result<Vec<u8>, Error> {
    let toc = validate_shape(toc_byte, frames)?;
    match toc.frame_count_code {
        FrameCountCode::One => Ok([&[toc_byte], frames[0]].concat()),
        FrameCountCode::TwoEqual => Ok([&[toc_byte], frames[0], frames[1]].concat()),
        FrameCountCode::TwoUnequal => {
            let mut out = Vec::with_capacity(3 + frames[0].len() + frames[1].len());
            out.push(toc_byte);
            encode_length(frames[0].len(), &mut out)?;
            out.extend_from_slice(frames[0]);
            out.extend_from_slice(frames[1]);
            Ok(out)
        }
        FrameCountCode::Arbitrary => {
            let cbr = frames.iter().all(|f| f.len() == frames[0].len());
            compose_packet_code3(toc_byte, frames, !cbr, 0)
        }
    }
}

/// Compose one **code-3** Opus packet with explicit VBR / padding
/// control (§3.2.5).
///
/// * `vbr` — write the `v` bit and the `M - 1` per-frame length
///   sequences. With `vbr == false` (CBR) every frame must have the
///   same length (R6).
/// * `padding` — number of trailing padding bytes (zeros); the
///   §3.2.5 255-chained padding-length header is derived from it and
///   the `p` bit set when non-zero.
///
/// The TOC byte must carry frame-count code 3.
pub fn compose_packet_code3(
    toc_byte: u8,
    frames: &[&[u8]],
    vbr: bool,
    padding: usize,
) -> Result<Vec<u8>, Error> {
    let toc = validate_shape(toc_byte, frames)?;
    if toc.frame_count_code != FrameCountCode::Arbitrary {
        return Err(Error::MalformedPacket);
    }
    if !vbr && frames.iter().any(|f| f.len() != frames[0].len()) {
        return Err(Error::MalformedPacket);
    }
    let m = frames.len();
    let mut out = Vec::new();
    out.push(toc_byte);
    out.push((u8::from(vbr) << 7) | (u8::from(padding > 0) << 6) | (m as u8));
    if padding > 0 {
        write_padding_chain(padding, &mut out);
    }
    if vbr {
        // The last frame's length stays implicit (§3.2.5).
        for f in &frames[..m - 1] {
            encode_length(f.len(), &mut out)?;
        }
    }
    for f in frames {
        out.extend_from_slice(f);
    }
    out.resize(out.len() + padding, 0);
    Ok(out)
}

/// Re-frame a **code-0** Opus packet (`TOC | frame`) as a §3.2.5
/// code-3 packet padded to **exactly** `target_len` bytes — the CBR
/// transport shaping §3.2.5 padding exists for. The compressed frame
/// bytes are unchanged, so the padded packet decodes identically to
/// the original; only the framing layer differs (frame-count code 3,
/// `M = 1`, CBR, the `p` bit + 255-chained padding length when padding
/// is present).
///
/// Any `target_len >= frame + 2` bytes is reachable exactly: sizes at
/// the 254-byte chain boundaries use a non-minimal padding chain (a
/// `255` continuation byte followed by a small terminal byte), which
/// the §3.2.5 parse folds to the same padding count.
pub fn pad_packet_to(packet: &[u8], target_len: usize) -> Result<Vec<u8>, Error> {
    if packet.len() < 2 {
        return Err(Error::MalformedPacket);
    }
    let toc = OpusTocByte::from_byte(packet[0]);
    if toc.frame_count_code != FrameCountCode::One {
        return Err(Error::MalformedPacket);
    }
    let frame = &packet[1..];
    if frame.len() > MAX_FRAME_BYTES {
        return Err(Error::MalformedPacket);
    }
    // Same config + stereo bit, frame-count code rewritten to 3.
    let toc3 = (packet[0] & !0b11) | 0b11;
    // TOC + count byte + frame = the minimum code-3 shape (no padding).
    let base = 2 + frame.len();
    if target_len < base {
        return Err(Error::MalformedPacket);
    }
    if target_len == base {
        return compose_packet_code3(toc3, &[frame], false, 0);
    }
    // `excess` bytes must be exactly consumed by the padding chain plus
    // the padding zeros: a chain of `c` bytes (c-1 continuation `255`s
    // + one terminal byte t in 0..=254) encodes (c-1)*254 + t padding
    // bytes, so `c` chain bytes reach any padding in
    // [(c-1)*254, (c-1)*254 + 254] — pick the c whose window contains
    // `excess - c`.
    let excess = target_len - base;
    let mut chain_bytes = None;
    for c in 1..=excess {
        let Some(padding) = excess.checked_sub(c) else {
            break;
        };
        let floor = (c - 1) * 254;
        if padding >= floor && padding <= floor + 254 {
            chain_bytes = Some((c, padding));
            break;
        }
        if floor > padding {
            break;
        }
    }
    let Some((c, padding)) = chain_bytes else {
        return Err(Error::MalformedPacket);
    };
    let mut out = Vec::with_capacity(target_len);
    out.push(toc3);
    // Count byte: M = 1, p = 1 (padding present), v = 0 (CBR).
    out.push(0x40 | 1u8);
    // c-1 continuation bytes (255 = 254 padding bytes + chain
    // continues), then the terminal byte.
    out.resize(out.len() + (c - 1), 255);
    out.push((padding - (c - 1) * 254) as u8);
    out.extend_from_slice(frame);
    out.resize(out.len() + padding, 0);
    debug_assert_eq!(out.len(), target_len);
    Ok(out)
}

/// Compose one **self-delimited** Opus packet (RFC 6716 Appendix B,
/// Figures 25-29) — the write-side mirror of
/// [`crate::framing_self_delim::parse_self_delimited`], for chaining
/// the first `N - 1` streams of a multistream payload.
///
/// The extra Appendix-B length field is placed exactly where the
/// parser expects it: after the TOC byte (codes 0/1), after the
/// §3.2.4 first-frame length (code 2), or after the frame-count byte
/// / padding chain / VBR inline lengths (code 3). `vbr` / `padding`
/// apply to code-3 packets only (pass `false` / `0` otherwise; a
/// non-code-3 TOC with padding or forced VBR errors). As in
/// [`compose_packet`], a code-3 packet chooses CBR automatically when
/// `vbr` is `false`, which then requires uniform frame lengths.
pub fn compose_self_delimited(
    toc_byte: u8,
    frames: &[&[u8]],
    vbr: bool,
    padding: usize,
) -> Result<Vec<u8>, Error> {
    let toc = validate_shape(toc_byte, frames)?;
    if toc.frame_count_code != FrameCountCode::Arbitrary && (vbr || padding > 0) {
        return Err(Error::MalformedPacket);
    }
    let mut out = Vec::new();
    out.push(toc_byte);
    match toc.frame_count_code {
        FrameCountCode::One => {
            // Figure 25: TOC | N1 | frame.
            encode_length(frames[0].len(), &mut out)?;
            out.extend_from_slice(frames[0]);
        }
        FrameCountCode::TwoEqual => {
            // Figure 26: TOC | N1 | frame | frame.
            encode_length(frames[0].len(), &mut out)?;
            out.extend_from_slice(frames[0]);
            out.extend_from_slice(frames[1]);
        }
        FrameCountCode::TwoUnequal => {
            // Figure 27: TOC | N1 | N2 | frame1 | frame2.
            encode_length(frames[0].len(), &mut out)?;
            encode_length(frames[1].len(), &mut out)?;
            out.extend_from_slice(frames[0]);
            out.extend_from_slice(frames[1]);
        }
        FrameCountCode::Arbitrary => {
            if !vbr && frames.iter().any(|f| f.len() != frames[0].len()) {
                return Err(Error::MalformedPacket);
            }
            let m = frames.len();
            out.push((u8::from(vbr) << 7) | (u8::from(padding > 0) << 6) | (m as u8));
            if padding > 0 {
                write_padding_chain(padding, &mut out);
            }
            if vbr {
                // Figure 29: M-1 inline lengths, then the Appendix-B
                // length of the LAST frame.
                for f in &frames[..m - 1] {
                    encode_length(f.len(), &mut out)?;
                }
                encode_length(frames[m - 1].len(), &mut out)?;
            } else {
                // Figure 28: one Appendix-B length shared by all frames.
                encode_length(frames[0].len(), &mut out)?;
            }
            for f in frames {
                out.extend_from_slice(f);
            }
            out.resize(out.len() + padding, 0);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{decode_length, OpusPacket};
    use crate::framing_self_delim::parse_self_delimited;
    use crate::toc::{Bandwidth, Mode};

    /// §3.2.5 Figure 5 wire layout on the WRITE side: the frame-count
    /// byte carries `v` in the MSB (0x80), `p` at 0x40, and `M` in the
    /// low six bits. Roundtrips through our own parser cannot catch a
    /// consistently reversed layout, so the emitted byte is pinned
    /// literally here.
    #[test]
    fn code3_frame_count_byte_emitted_msb_first() {
        // TOC 0x4B = config 9 (SILK WB 20 ms), mono, code 3.
        let f1: &[u8] = &[0xAA; 4];
        let f2: &[u8] = &[0xBB; 4];

        // CBR, no padding: fc = M.
        let pkt = compose_packet_code3(0x4B, &[f1, f2], false, 0).unwrap();
        assert_eq!(pkt[1], 2, "CBR no-padding fc byte");

        // CBR with padding: fc = 0x40 | M.
        let pkt = compose_packet_code3(0x4B, &[f1, f2], false, 3).unwrap();
        assert_eq!(pkt[1], 0x40 | 2, "CBR padded fc byte");
        assert_eq!(pkt[2], 3, "padding-length byte");

        // VBR: fc = 0x80 | M.
        let pkt = compose_packet_code3(0x4B, &[f1, f2], true, 0).unwrap();
        assert_eq!(pkt[1], 0x80 | 2, "VBR fc byte");
    }

    /// A tiny deterministic LCG.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n
        }
    }

    fn random_frame(rng: &mut Lcg, max_len: u32) -> Vec<u8> {
        let len = rng.below(max_len + 1) as usize;
        (0..len).map(|_| rng.next_u32() as u8).collect()
    }

    fn toc(code: FrameCountCode) -> u8 {
        // CELT-only FB 10 ms accommodates up to 12 frames under R5.
        OpusTocByte::compose_byte(Mode::CeltOnly, Bandwidth::Fb, 100, false, code).unwrap()
    }

    /// §3.2.1 write/read: every legal length 0..=1275 roundtrips
    /// through the shared decoder, and 1276 is rejected.
    #[test]
    fn encode_length_roundtrips_all_values() {
        for len in 0..=MAX_FRAME_BYTES {
            let mut buf = Vec::new();
            encode_length(len, &mut buf).unwrap();
            let (decoded, consumed) = decode_length(&buf).unwrap();
            assert_eq!((decoded, consumed), (len, buf.len()), "length {len}");
        }
        let mut buf = Vec::new();
        assert!(encode_length(MAX_FRAME_BYTES + 1, &mut buf).is_err());
    }

    /// Regular-framing roundtrip across all four codes: composed
    /// packets reparse to the same TOC byte, frames, and padding.
    #[test]
    fn compose_parse_roundtrip_all_codes() {
        let mut rng = Lcg(0x0385_C0DE);
        for round in 0..200 {
            let (toc_byte, frames): (u8, Vec<Vec<u8>>) = match rng.below(4) {
                0 => (toc(FrameCountCode::One), vec![random_frame(&mut rng, 1275)]),
                1 => {
                    let f = random_frame(&mut rng, 1275);
                    (toc(FrameCountCode::TwoEqual), vec![f.clone(), f])
                }
                2 => (
                    toc(FrameCountCode::TwoUnequal),
                    vec![random_frame(&mut rng, 1275), random_frame(&mut rng, 1275)],
                ),
                _ => {
                    let m = 1 + rng.below(12) as usize;
                    (
                        toc(FrameCountCode::Arbitrary),
                        (0..m).map(|_| random_frame(&mut rng, 300)).collect(),
                    )
                }
            };
            let slices: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
            let packet = compose_packet(toc_byte, &slices).expect("compose");
            let parsed = OpusPacket::parse(&packet).expect("parse");
            assert_eq!(
                parsed.toc,
                OpusTocByte::from_byte(toc_byte),
                "round {round}"
            );
            assert_eq!(parsed.frames(), &slices[..], "round {round}");
            assert_eq!(parsed.padding, 0, "round {round}");
        }
    }

    /// Code-3 with forced VBR (even for equal lengths) and padding
    /// (including the 255-chained lengths) roundtrips, and the parser
    /// reports the same trailing-padding byte count.
    #[test]
    fn compose_code3_vbr_and_padding_roundtrip() {
        let mut rng = Lcg(0x0AD5_0385);
        let toc_byte = toc(FrameCountCode::Arbitrary);
        for &padding in &[0usize, 1, 42, 253, 254, 255, 300, 600] {
            let m = 1 + rng.below(6) as usize;
            let frames: Vec<Vec<u8>> = (0..m).map(|_| random_frame(&mut rng, 200)).collect();
            let slices: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
            for vbr in [false, true] {
                if !vbr && !slices.iter().all(|f| f.len() == slices[0].len()) {
                    continue;
                }
                let packet =
                    compose_packet_code3(toc_byte, &slices, vbr, padding).expect("compose");
                let parsed = OpusPacket::parse(&packet).expect("parse");
                assert_eq!(parsed.frames(), &slices[..], "vbr={vbr} padding={padding}");
                assert_eq!(parsed.padding, padding, "vbr={vbr} padding={padding}");
            }
        }
    }

    /// Appendix-B roundtrip: self-delimited packets of every code
    /// reparse identically AND report `consumed == len`, and three
    /// packets chained back-to-back split correctly.
    #[test]
    fn compose_self_delimited_roundtrip_and_chain() {
        let mut rng = Lcg(0x5E1F_DE11);
        let mut chained = Vec::new();
        let mut expected: Vec<(u8, Vec<Vec<u8>>)> = Vec::new();
        for code_pick in 0..4u32 {
            let (toc_byte, frames, vbr, padding): (u8, Vec<Vec<u8>>, bool, usize) = match code_pick
            {
                0 => (
                    toc(FrameCountCode::One),
                    vec![random_frame(&mut rng, 400)],
                    false,
                    0,
                ),
                1 => {
                    let f = random_frame(&mut rng, 400);
                    (toc(FrameCountCode::TwoEqual), vec![f.clone(), f], false, 0)
                }
                2 => (
                    toc(FrameCountCode::TwoUnequal),
                    vec![random_frame(&mut rng, 400), random_frame(&mut rng, 400)],
                    false,
                    0,
                ),
                _ => (
                    toc(FrameCountCode::Arbitrary),
                    (0..5).map(|_| random_frame(&mut rng, 300)).collect(),
                    true,
                    77,
                ),
            };
            let slices: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
            let packet =
                compose_self_delimited(toc_byte, &slices, vbr, padding).expect("compose sd");
            // Individual roundtrip: exact consumption.
            let parsed = parse_self_delimited(&packet).expect("parse sd");
            assert_eq!(parsed.consumed, packet.len(), "code {code_pick}");
            assert_eq!(parsed.packet.frames(), &slices[..], "code {code_pick}");
            assert_eq!(parsed.packet.padding, padding, "code {code_pick}");
            chained.extend_from_slice(&packet);
            expected.push((toc_byte, frames));
        }
        // Chained: parse packets back-to-back from one buffer.
        let mut cursor = 0usize;
        for (toc_byte, frames) in &expected {
            let parsed = parse_self_delimited(&chained[cursor..]).expect("chained parse");
            assert_eq!(parsed.packet.toc, OpusTocByte::from_byte(*toc_byte));
            let slices: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
            assert_eq!(parsed.packet.frames(), &slices[..]);
            cursor += parsed.consumed;
        }
        assert_eq!(cursor, chained.len());
    }

    /// Shape violations are rejected: frame-count mismatches, unequal
    /// code-1 frames, oversize frames, empty / oversized code-3 frame
    /// lists, the R5 duration bound, CBR with non-uniform lengths, and
    /// padding / VBR options on non-code-3 packets.
    #[test]
    fn compose_rejects_shape_violations() {
        let f10 = vec![0u8; 10];
        let f11 = vec![0u8; 11];
        let big = vec![0u8; MAX_FRAME_BYTES + 1];
        // Frame-count mismatches per code.
        assert!(compose_packet(toc(FrameCountCode::One), &[&f10, &f10]).is_err());
        assert!(compose_packet(toc(FrameCountCode::TwoEqual), &[&f10]).is_err());
        assert!(compose_packet(toc(FrameCountCode::TwoUnequal), &[&f10]).is_err());
        assert!(compose_packet(toc(FrameCountCode::Arbitrary), &[]).is_err());
        // R3: code-1 frames must be equal-size.
        assert!(compose_packet(toc(FrameCountCode::TwoEqual), &[&f10, &f11]).is_err());
        // R2: per-frame bound.
        assert!(compose_packet(toc(FrameCountCode::One), &[&big]).is_err());
        // Code-3 frame-count / duration caps: 49 frames trip both the
        // M <= 48 bound and (at 2.5 ms) the R5 duration bound, while
        // 48 x 2.5 ms = 120 ms is exactly legal.
        let toc_25 = OpusTocByte::compose_byte(
            Mode::CeltOnly,
            Bandwidth::Fb,
            25,
            false,
            FrameCountCode::Arbitrary,
        )
        .unwrap();
        let many: Vec<&[u8]> = (0..49).map(|_| f10.as_slice()).collect();
        assert!(compose_packet(toc_25, &many).is_err());
        assert!(compose_packet(toc_25, &many[..48]).is_ok());
        // R5: 3 × 60 ms = 180 ms > 120 ms.
        let toc_60 = OpusTocByte::compose_byte(
            Mode::SilkOnly,
            Bandwidth::Nb,
            600,
            false,
            FrameCountCode::Arbitrary,
        )
        .unwrap();
        assert!(compose_packet(toc_60, &[&f10, &f10, &f10]).is_err());
        assert!(compose_packet(toc_60, &[&f10, &f10]).is_ok());
        // CBR with non-uniform lengths.
        assert!(
            compose_packet_code3(toc(FrameCountCode::Arbitrary), &[&f10, &f11], false, 0).is_err()
        );
        // compose_packet_code3 demands a code-3 TOC.
        assert!(compose_packet_code3(toc(FrameCountCode::One), &[&f10], false, 0).is_err());
        // Self-delimited: padding / VBR only for code 3.
        assert!(compose_self_delimited(toc(FrameCountCode::One), &[&f10], false, 4).is_err());
        assert!(compose_self_delimited(toc(FrameCountCode::One), &[&f10], true, 0).is_err());
        assert!(
            compose_self_delimited(toc(FrameCountCode::Arbitrary), &[&f10, &f11], false, 0)
                .is_err()
        );
    }

    /// Owns one deterministic 20 ms NB SILK frame script's buffers.
    struct SilkScript {
        frame_type: u8,
        gains: Vec<crate::silk_gains::GainSymbol>,
        i2: Vec<i8>,
        lsb: Vec<u8>,
        e_raw: Vec<i32>,
    }

    impl SilkScript {
        /// A simple valid NB 20 ms frame: 4 gain symbols, zero LSF
        /// stage-2 residual, no LTP (unvoiced/inactive types only),
        /// `pulses` unit pulses at each shell block's first sample.
        fn new(frame_type: u8, pulses: i32) -> Self {
            use crate::silk_excitation::{shell_block_count, SilkFrameSize, SHELL_BLOCK_SAMPLES};
            use crate::silk_gains::GainSymbol;
            assert!(frame_type < 4, "voiced scripts would need LTP symbols");
            let blocks = shell_block_count(Bandwidth::Nb, SilkFrameSize::TwentyMs).unwrap();
            let mut e_raw = vec![0i32; blocks * SHELL_BLOCK_SAMPLES];
            for b in 0..blocks {
                e_raw[b * SHELL_BLOCK_SAMPLES] = pulses;
            }
            SilkScript {
                frame_type,
                gains: vec![
                    GainSymbol::Independent(40),
                    GainSymbol::Delta(10),
                    GainSymbol::Delta(15),
                    GainSymbol::Delta(20),
                ],
                i2: vec![0i8; 10],
                lsb: vec![0u8; blocks],
                e_raw,
            }
        }

        fn symbols(&self) -> crate::silk_decode::SilkFrameSymbols<'_> {
            crate::silk_decode::SilkFrameSymbols {
                header: crate::silk_frame::SilkHeaderSymbols {
                    stereo: None,
                    mid_only_flag: None,
                    frame_type: self.frame_type,
                },
                gains: &self.gains,
                lsf_stage1: 5,
                lsf_stage2_i2: &self.i2,
                lsf_interp_w_q2: Some(4),
                ltp: None,
                lcg_seed: 1,
                excitation: crate::silk_excitation::ExcitationSymbols {
                    rate_level: 3,
                    lsb_counts: &self.lsb,
                    e_raw: &self.e_raw,
                },
            }
        }
    }

    /// End-to-end §3.2 + §4.5: two independently encoded 20 ms mono
    /// SILK frame bodies packed as code-1 / code-2 / code-3-VBR
    /// packets decode through `OpusDecoder::decode_packet` as two real
    /// SILK frames with the exact combined sample count.
    #[test]
    fn composed_multiframe_silk_packets_decode_end_to_end() {
        use crate::decoder::{FrameDecodeStatus, OpusDecoder};
        use crate::silk_packet_encode::encode_silk_only_packet_mono;

        // Reuse the packet encoder for two frame bodies: encode two
        // single-frame packets and strip their TOC bytes. The bodies
        // are self-contained §4.2 frames (each starts its own range
        // coder), which is exactly what a multi-frame packet carries.
        let script1 = SilkScript::new(0, 1);
        let script2 = SilkScript::new(2, 7);
        let (p1, _) = encode_silk_only_packet_mono(Bandwidth::Nb, 200, &[script1.symbols()])
            .expect("encode 1");
        let (p2, _) = encode_silk_only_packet_mono(Bandwidth::Nb, 200, &[script2.symbols()])
            .expect("encode 2");
        let body1 = &p1[1..];
        let body2 = &p2[1..];

        let toc_code2 = OpusTocByte::compose_byte(
            Mode::SilkOnly,
            Bandwidth::Nb,
            200,
            false,
            FrameCountCode::TwoUnequal,
        )
        .unwrap();
        let toc_code3 = OpusTocByte::compose_byte(
            Mode::SilkOnly,
            Bandwidth::Nb,
            200,
            false,
            FrameCountCode::Arbitrary,
        )
        .unwrap();
        let toc_code1 = OpusTocByte::compose_byte(
            Mode::SilkOnly,
            Bandwidth::Nb,
            200,
            false,
            FrameCountCode::TwoEqual,
        )
        .unwrap();

        let mut candidates: Vec<Vec<u8>> = vec![
            compose_packet(toc_code2, &[body1, body2]).unwrap(),
            compose_packet_code3(toc_code3, &[body1, body2], true, 9).unwrap(),
        ];
        // Code 1 needs equal sizes: pack the same body twice.
        candidates.push(compose_packet(toc_code1, &[body1, body1]).unwrap());

        for (idx, packet) in candidates.iter().enumerate() {
            let mut dec = OpusDecoder::new();
            let out = dec.decode_packet(packet).expect("decode");
            assert_eq!(out.frame_outcomes.len(), 2, "candidate {idx}");
            for fo in &out.frame_outcomes {
                assert_eq!(
                    fo.status,
                    FrameDecodeStatus::SilkParamsDecoded,
                    "candidate {idx}"
                );
            }
            assert_eq!(out.samples_per_channel(), 2 * 960, "candidate {idx}");
        }
    }

    /// `pad_packet_to` reaches EVERY target size from the code-3
    /// minimum up through several §3.2.5 chain boundaries (including
    /// the non-minimal-chain sizes), and each padded packet reparses to
    /// the identical single frame with the exact requested length.
    #[test]
    fn pad_packet_to_hits_every_target_exactly() {
        let mut rng = Lcg(0x0AD5_1391);
        let body = random_frame(&mut rng, 40);
        let mut packet = vec![toc(FrameCountCode::One)];
        packet.extend_from_slice(&body);

        let base = 2 + body.len();
        // Below the minimum: rejected.
        assert!(pad_packet_to(&packet, base - 1).is_err());
        for target in base..base + 900 {
            let padded = pad_packet_to(&packet, target).expect("pad");
            assert_eq!(padded.len(), target, "target {target}");
            let parsed = OpusPacket::parse(&padded).expect("parse");
            assert_eq!(parsed.frames().len(), 1, "target {target}");
            assert_eq!(parsed.frames()[0], &body[..], "target {target}");
            // The framing keeps the §3.1 config + channel bits.
            assert_eq!(
                parsed.toc.config,
                OpusTocByte::from_byte(packet[0]).config,
                "target {target}"
            );
        }
        // Only code-0 packets are re-framed.
        let code3 = compose_packet_code3(toc(FrameCountCode::Arbitrary), &[&body], false, 0)
            .expect("compose");
        assert!(pad_packet_to(&code3, code3.len() + 10).is_err());
    }
}
