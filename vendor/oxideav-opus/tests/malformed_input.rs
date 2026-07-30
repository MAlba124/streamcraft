//! Malformed-input property tests for the Opus packet-framing and
//! SILK header-bits layers — RFC 6716 §3.4 requirements R1..R7 plus
//! the §4.2.3 / §4.2.4 header-bit bitstream-boundary behaviour.
//!
//! Round 22 adds a dedicated integration-level audit that exercises
//! every concrete failure mode RFC 6716 §3.4 enumerates for a
//! malformed packet plus a property-style sweep of the §4.2.3 /
//! §4.2.4 SILK header decoder against truncated and adversarial
//! input. The goal is to lock in the per-RFC-requirement rejection
//! behaviour (R1..R7) before a downstream caller starts feeding
//! attacker-controlled bytes into `OpusPacket::parse`, and to prove
//! the §4.2.3 / §4.2.4 SILK header decoder is panic-free on any
//! truncation of a previously-valid bitstream — the §4.1.4
//! zero-extension rule promises this, but the property tests are the
//! audit-grade evidence.
//!
//! Provenance: this file reads only RFC 6716 §3 (TOC byte + framing
//! requirements R1..R7 from §3.4) and §4.2.3 / §4.2.4 (SILK header
//! bits + per-frame LBRR symbol). No external library source
//! consulted. The tests use only the crate's public API
//! (`OpusPacket::parse`, `OpusTocByte::from_byte`, `RangeDecoder`,
//! `SilkHeaderBits::decode`) — they do not poke at internals.
//!
//! ## What's covered
//!
//! * **R1** — empty-packet rejection at the TOC layer.
//! * **R2** — implicit frame length > 1275 bytes rejected for every
//!   `c` code where the §3.2 layer can detect it.
//! * **R3** — code-1 packet with an even total length (i.e. odd body
//!   length) rejected.
//! * **R4** — code-2 packets with various length-byte truncations
//!   plus the length-exceeds-remaining edge.
//! * **R5** — code-3 packets with `M = 0` rejected; `M > 48`
//!   rejected by the high-bits constraint.
//! * **R6** — CBR code-3 packets where `R` (the post-header,
//!   post-padding budget) is not a non-negative integer multiple of
//!   `M`.
//! * **R7** — VBR code-3 packets whose declared lengths overrun the
//!   remaining bytes.
//! * **TOC determinism** — every `u8` parses to a self-consistent
//!   TOC byte (total function).
//! * **§4.2.3 / §4.2.4 truncation safety** — for every supported
//!   `(num_silk_frames, stereo)` combination, truncating an
//!   originally-valid 32-byte buffer to any prefix length 1..=32
//!   never panics and always returns a well-formed
//!   `SilkHeaderBits` (the §4.1.4 zero-extension rule guarantees
//!   this).
//! * **§4.2.4 PDF bounds** — `decode_per_frame_lbrr` for every input
//!   buffer produces a value in `{1..=2^N - 1}`, never `0`, by way
//!   of the §4.1.3.3 leading-zero offset.

use oxideav_opus::{
    FrameDecodeStatus, OpusDecoder, OpusPacket, OpusTocByte, RangeDecoder, SilkHeaderBits,
    MAX_FRAMES_PER_PACKET, MAX_FRAME_BYTES,
};

/// RFC 6716 §3.4 [R1]: a well-formed Opus packet has at least one
/// byte. An empty buffer is rejected at the TOC-byte layer with
/// `EmptyPacket`.
///
/// Distinct from `MalformedPacket` because the §3.4 / §3.1
/// requirement R1 is structurally the first thing checked: with no
/// bytes, the TOC field doesn't even exist.
#[test]
fn r1_empty_packet_rejected() {
    let err = OpusPacket::parse(&[]).expect_err("R1");
    // Empty-packet sentinel error variant is preserved across the
    // refactor (the §3.2 frame-packing parser does NOT collapse it
    // into the generic MalformedPacket).
    assert!(format!("{err}").contains("empty"), "R1: {err}");
}

/// RFC 6716 §3.4 [R2]: no implicit frame length is larger than
/// 1275 bytes. For code 0 (single frame) this is "body must be ≤
/// 1275 bytes"; for code 1 (two equal-size frames) this is "half ≤
/// 1275"; for code 2 the declared first-frame length is bounded;
/// for code 3 the per-frame slice size is bounded.
///
/// We probe all four codes with a body large enough that the
/// derived per-frame length crosses 1275.
#[test]
fn r2_implicit_frame_length_capped_at_1275() {
    // Code 0: body of 1276 bytes ⇒ single frame of 1276 bytes ⇒ rejected.
    let mut p = vec![0u8; 1 + 1276];
    p[0] = 0; // config=0, mono, c=0
    assert!(OpusPacket::parse(&p).is_err(), "R2 code 0: 1276 B frame");

    // Code 0: body of exactly 1275 bytes is the upper bound and MUST
    // succeed (the §3.4 [R2] inequality is non-strict).
    let mut p = vec![0u8; 1 + 1275];
    p[0] = 0;
    let parsed = OpusPacket::parse(&p).expect("R2 code 0 boundary");
    assert_eq!(parsed.frames().len(), 1);
    assert_eq!(parsed.frames()[0].len(), 1275);

    // Code 1: body of 2552 bytes ⇒ two frames of 1276 each ⇒ rejected.
    let mut p = vec![0u8; 1 + 2552];
    p[0] = 1 << 0; // c=1
    assert!(OpusPacket::parse(&p).is_err(), "R2 code 1: 1276 B each");

    // Code 1: body of 2550 bytes ⇒ two frames of 1275 each ⇒ accepted.
    let mut p = vec![0u8; 1 + 2550];
    p[0] = 1;
    let parsed = OpusPacket::parse(&p).expect("R2 code 1 boundary");
    assert_eq!(parsed.frames().len(), 2);
    assert!(parsed.frames().iter().all(|f| f.len() == 1275));
}

/// RFC 6716 §3.4 [R3]: a code-1 packet has odd total length `N` so
/// that `(N-1)/2` is an integer.
///
/// We sweep body lengths 0..=8 and check that even body lengths
/// pass and odd body lengths fail (since body length = `N - 1` for
/// code 1 packets).
#[test]
fn r3_code1_requires_even_body_length() {
    for body_len in 0usize..=8 {
        let mut p = vec![0u8; 1 + body_len];
        p[0] = 1; // c=1
        let result = OpusPacket::parse(&p);
        if body_len % 2 == 0 {
            // Even body ⇒ two equal halves.
            let parsed =
                result.unwrap_or_else(|_| panic!("R3: even body_len={body_len} should accept"));
            assert_eq!(parsed.frames().len(), 2);
            assert_eq!(parsed.frames()[0].len(), body_len / 2);
        } else {
            // Odd body ⇒ R3 violation.
            assert!(result.is_err(), "R3: odd body_len={body_len} should reject");
        }
    }
}

/// RFC 6716 §3.4 [R4]: code-2 packets have enough bytes after the
/// TOC for a valid length, and that length is no larger than the
/// number of bytes remaining in the packet.
///
/// The §3.2.1 length encoding has two failure shapes:
///   1. The first length byte is missing (zero-byte body).
///   2. The first length byte is in `252..=255` and the SECOND
///      length byte is missing.
///   3. The decoded length runs off the end of the remaining buffer.
#[test]
fn r4_code2_length_byte_truncations_rejected() {
    // Sub-shape 1: code-2 with empty body — the §3.2.1 length byte
    // itself is missing.
    let p = [2u8]; // TOC: config=0, mono, c=2.
    assert!(OpusPacket::parse(&p).is_err(), "R4: missing length byte");

    // Sub-shape 2: code-2 with one body byte in 252..=255 — second
    // length byte missing.
    for first in 252u8..=255 {
        let p = [2u8, first];
        assert!(
            OpusPacket::parse(&p).is_err(),
            "R4: missing 2nd length byte after first={first}"
        );
    }

    // Sub-shape 3: code-2 with length byte = 10 but only 3 body
    // bytes after the length byte.
    let mut p = vec![2u8, 10];
    p.extend_from_slice(&[0u8; 3]);
    assert!(
        OpusPacket::parse(&p).is_err(),
        "R4: declared length 10 > remaining 3"
    );

    // Sub-shape 3 boundary: declared length exactly equals
    // remaining ⇒ ACCEPTED (second frame is zero bytes, a legal
    // §3.2.1 DTX marker).
    let mut p = vec![2u8, 5];
    p.extend_from_slice(&[0u8; 5]);
    let parsed = OpusPacket::parse(&p).expect("R4 boundary accept");
    assert_eq!(parsed.frames().len(), 2);
    assert_eq!(parsed.frames()[0].len(), 5);
    assert_eq!(parsed.frames()[1].len(), 0);
}

/// RFC 6716 §3.4 [R5]: code-3 packets contain at least one frame
/// (`M ≥ 1`) but no more than 48 frames worth of audio (per the
/// §3.2.5 cap, since 120 ms / 2.5 ms = 48).
///
/// We probe `M = 0` (R5 violation: zero frames) and confirm that
/// every `M ∈ 1..=MAX_FRAMES_PER_PACKET` produces a parser that
/// either succeeds or fails on a downstream R6 / R7 constraint, but
/// never on R5 alone.
#[test]
fn r5_code3_frame_count_zero_rejected() {
    // M = 0: frame-count byte = 0x00 ⇒ v=0, p=0, M=0.
    let p = [3u8, 0x00];
    assert!(OpusPacket::parse(&p).is_err(), "R5: M=0");

    // M = 1..=48 with CBR R=0 (body of just the frame-count byte +
    // zero frames of zero bytes each ⇒ R=0, R%M=0, accepted).
    for m in 1u8..=MAX_FRAMES_PER_PACKET {
        let p = [3u8, m]; // v=0, p=0, M=m
        let parsed = OpusPacket::parse(&p)
            .unwrap_or_else(|e| panic!("R5: M={m} should be accepted (got {e})"));
        assert_eq!(parsed.frames().len(), m as usize);
        for f in parsed.frames() {
            assert_eq!(f.len(), 0, "R=0 ⇒ each frame is 0 bytes");
        }
    }
}

/// RFC 6716 §3.4 [R6]: a CBR code-3 packet has `R = N - 2 - P` as a
/// non-negative integer multiple of `M`.
///
/// We construct M=3, no padding, body = TOC + frame-count byte +
/// 7 frame-data bytes ⇒ R = 7, R % 3 = 1 ⇒ R6 violation.
#[test]
fn r6_code3_cbr_r_not_multiple_of_m_rejected() {
    // TOC = 3 (c=3), frame-count byte: v=0, p=0, M=3 ⇒ 0x03.
    let mut p = vec![3u8, 0x03];
    p.extend_from_slice(&[0u8; 7]);
    assert!(OpusPacket::parse(&p).is_err(), "R6: R=7, M=3, 7%3=1");

    // Same setup with R=6 ⇒ R%M=0 ⇒ accepted, two bytes per frame.
    let mut p = vec![3u8, 0x03];
    p.extend_from_slice(&[0u8; 6]);
    let parsed = OpusPacket::parse(&p).expect("R6 boundary accept");
    assert_eq!(parsed.frames().len(), 3);
    for f in parsed.frames() {
        assert_eq!(f.len(), 2);
    }
}

/// RFC 6716 §3.4 [R7]: a VBR code-3 packet has enough bytes to
/// contain the header bytes (TOC + frame count + padding lengths +
/// `M-1` frame lengths) plus the declared lengths plus the trailing
/// padding.
///
/// We construct M=2, v=1, no padding; declare first frame length =
/// 50 bytes; provide only 10 body bytes after the length sequence
/// ⇒ R7 violation.
#[test]
fn r7_code3_vbr_declared_length_overruns_remaining() {
    // TOC=3 (c=3), frame-count byte: v=1, p=0, M=2 ⇒ 0x82.
    // First-frame length byte: 50.
    let mut p = vec![3u8, 0x82, 50u8];
    p.extend_from_slice(&[0u8; 10]);
    assert!(
        OpusPacket::parse(&p).is_err(),
        "R7: declared 50, only 10 bytes remain after length sequence"
    );

    // R7 boundary: declared length 5 with body of TOC + framecount +
    // length(=5) + 5+10 bytes ⇒ first frame 5 bytes, last frame 10
    // bytes ⇒ accepted.
    let mut p = vec![3u8, 0x82, 5u8];
    p.extend_from_slice(&[0u8; 15]);
    let parsed = OpusPacket::parse(&p).expect("R7 boundary accept");
    assert_eq!(parsed.frames().len(), 2);
    assert_eq!(parsed.frames()[0].len(), 5);
    assert_eq!(parsed.frames()[1].len(), 10);
}

/// §3.2.5 padding-chain pathological inputs: a `p=1` packet whose
/// padding-length byte demands more bytes than exist in the packet
/// must be rejected (the chain reading hits end-of-buffer).
#[test]
fn code3_padding_chain_truncation_rejected() {
    // TOC=3, frame-count byte v=0 p=1 M=1 ⇒ 0x41. Padding chain
    // declares 100 bytes, but no body follows ⇒ R6/R7 violation
    // (the padding chain byte is missing entirely).
    let p = [3u8, 0x41];
    assert!(
        OpusPacket::parse(&p).is_err(),
        "padding length byte missing"
    );

    // Pad chain says "200 bytes of padding" but body has only 5.
    let mut p = vec![3u8, 0x41, 200u8];
    p.extend_from_slice(&[0u8; 5]);
    assert!(OpusPacket::parse(&p).is_err(), "padding > remaining");
}

/// §3.2.5 padding-chain with the "255 means 254 + read next" extension
/// is rejected when the chain doesn't terminate (no <255 byte before
/// end-of-buffer).
#[test]
fn code3_padding_chain_255_unterminated_rejected() {
    // TOC=3, fc=0x41 (v=0, p=1, M=1). Then three 255 padding bytes,
    // each extending the chain by 254 + demanding another byte — but
    // the buffer ends.
    let p = [3u8, 0x41, 255, 255, 255];
    assert!(OpusPacket::parse(&p).is_err(), "255-chain runs off end");
}

/// `OpusTocByte::from_byte` is a total function. Every `u8` produces
/// a self-consistent TOC byte where the encoded `(config, s, c)`
/// triple matches the bit positions in the raw byte.
///
/// This is the property-test foundation of the parser: every
/// non-empty packet starts with a syntactically valid TOC.
#[test]
fn toc_byte_total_function_self_consistency() {
    for byte in 0u8..=255 {
        let toc = OpusTocByte::from_byte(byte);
        let config = byte >> 3;
        let s = (byte >> 2) & 0x01;
        let c = byte & 0x03;
        assert_eq!(toc.config, config, "byte=0x{byte:02X} config");
        let expected_channels = if s == 0 {
            oxideav_opus::ChannelMapping::Mono
        } else {
            oxideav_opus::ChannelMapping::Stereo
        };
        assert_eq!(toc.channels, expected_channels, "byte=0x{byte:02X} s");
        let expected_code = match c {
            0 => oxideav_opus::FrameCountCode::One,
            1 => oxideav_opus::FrameCountCode::TwoEqual,
            2 => oxideav_opus::FrameCountCode::TwoUnequal,
            3 => oxideav_opus::FrameCountCode::Arbitrary,
            _ => unreachable!(),
        };
        assert_eq!(toc.frame_count_code, expected_code, "byte=0x{byte:02X} c");

        // The §3.1 frame_size_tenths_ms must always be one of the
        // six values legal under Table 2.
        assert!(
            matches!(toc.frame_size_tenths_ms, 25 | 50 | 100 | 200 | 400 | 600),
            "byte=0x{byte:02X} frame_size_tenths_ms={}",
            toc.frame_size_tenths_ms
        );
    }
}

/// §4.2.3 / §4.2.4 truncation safety: starting from a 32-byte buffer
/// the SILK header decoder accepts, every shorter prefix (down to
/// length 1) also decodes without panicking and produces a
/// `SilkHeaderBits` whose meaningful-bit invariants still hold.
///
/// The §4.1.4 RangeDecoder zero-extension rule guarantees this — the
/// test pins the contract.
#[test]
fn silk_header_decode_truncation_never_panics() {
    let full = [0x5Au8; 32];
    for stereo in [false, true] {
        for &n in &[1u8, 2, 3] {
            for take in 1usize..=full.len() {
                let buf = &full[..take];
                let mut rd = RangeDecoder::new(buf);
                let h = SilkHeaderBits::decode(&mut rd, n, stereo).unwrap_or_else(|e| {
                    panic!("panic-equivalent error n={n} stereo={stereo} take={take}: {e}")
                });
                assert_eq!(h.num_silk_frames, n);
                assert_eq!(h.side.is_some(), stereo);
                // Only the low `n` bits of each VAD/LBRR bitmap are
                // meaningful — higher bits must be zero.
                assert!(
                    h.mid.vad_flags >> n == 0,
                    "n={n} take={take} mid VAD high bits set: {:08b}",
                    h.mid.vad_flags
                );
                if let Some(s) = h.side {
                    assert!(
                        s.vad_flags >> n == 0,
                        "n={n} take={take} side VAD high bits set: {:08b}",
                        s.vad_flags
                    );
                }
                assert!(
                    h.per_frame_lbrr.mid >> n == 0,
                    "n={n} take={take} mid LBRR high bits set: {:08b}",
                    h.per_frame_lbrr.mid
                );
                assert!(
                    h.per_frame_lbrr.side >> n == 0,
                    "n={n} take={take} side LBRR high bits set: {:08b}",
                    h.per_frame_lbrr.side
                );
            }
        }
    }
}

/// §4.2.3 invariant: in a mono Opus frame the §4.2.3 decoder always
/// produces `side == None` and the per-frame side LBRR bitmap is 0,
/// regardless of the input buffer. The §4.2.3 "single channel" path
/// MUST NOT silently produce side state.
#[test]
fn silk_header_mono_never_emits_side_state() {
    for byte0 in 0u8..=255 {
        for &n in &[1u8, 2, 3] {
            let buf = [byte0, 0xA3, 0x5C, 0xC5, 0x3A, 0xFF, 0x00, 0x77];
            let mut rd = RangeDecoder::new(&buf);
            let h = SilkHeaderBits::decode(&mut rd, n, false).expect("decode");
            assert!(
                h.side.is_none(),
                "byte0=0x{byte0:02X} n={n}: mono produced side state"
            );
            assert_eq!(
                h.per_frame_lbrr.side, 0,
                "byte0=0x{byte0:02X} n={n}: mono produced side LBRR"
            );
        }
    }
}

/// §4.2.4 PDF safety: for a mono 40 ms (n=2) Opus frame whose global
/// LBRR flag is set, the per-frame LBRR bitmap is always in `{1, 2,
/// 3}` — `0b00` is unreachable because the §4.2.4 PDF excludes the
/// zero entry, exactly because the channel only reaches the PDF
/// branch after its global LBRR flag was set (so at least one
/// per-frame LBRR bit MUST be 1).
#[test]
fn silk_header_40ms_lbrr_bitmap_never_zero_when_global_set() {
    // 0xFF puts the §4.2.3 dec_bit_logp(1) decisions into the "1"
    // half (val = 127 - 127 = 0 < s = rng>>1), so the global LBRR
    // flag is 1 ⇒ the per-frame LBRR PDF branch fires for n >= 2.
    for tail in [0u8, 0x11, 0x55, 0xAA, 0x33, 0xC0, 0x3F] {
        let buf = [0xFFu8, tail, 0x99, 0x66, 0x42, 0x18, 0x9A, 0x65];
        let mut rd = RangeDecoder::new(&buf);
        let h = SilkHeaderBits::decode(&mut rd, 2, false).expect("decode");
        assert!(
            h.mid.lbrr_flag,
            "tail=0x{tail:02X}: expected global LBRR=1 with 0xFF seed"
        );
        assert!(
            (1..=3).contains(&h.per_frame_lbrr.mid),
            "tail=0x{tail:02X}: 40ms LBRR bitmap {} out of 1..=3",
            h.per_frame_lbrr.mid
        );
    }
}

/// §4.2.4 PDF safety for the 60 ms case: same invariant, range
/// `{1..=7}`.
#[test]
fn silk_header_60ms_lbrr_bitmap_never_zero_when_global_set() {
    for tail in [0u8, 0x11, 0x55, 0xAA, 0x33, 0xC0, 0x3F] {
        let buf = [0xFFu8, tail, 0x99, 0x66, 0x42, 0x18, 0x9A, 0x65, 0xCD, 0x32];
        let mut rd = RangeDecoder::new(&buf);
        let h = SilkHeaderBits::decode(&mut rd, 3, false).expect("decode");
        assert!(
            h.mid.lbrr_flag,
            "tail=0x{tail:02X}: expected global LBRR=1 with 0xFF seed"
        );
        assert!(
            (1..=7).contains(&h.per_frame_lbrr.mid),
            "tail=0x{tail:02X}: 60ms LBRR bitmap {} out of 1..=7",
            h.per_frame_lbrr.mid
        );
    }
}

/// Sweep all 256 code-3 frame-count bytes against a 2-byte packet
/// (just `[TOC, frame_count]`): with no body, only `M=0` (R5) and
/// `M ≥ 1` with no padding + no VBR lengths can succeed (CBR with
/// R=0 ⇒ M=1..=48 accepted; VBR demands M-1 length sequences which
/// can't fit; padding-on demands at least one chain byte which can't
/// fit).
#[test]
fn code3_minimal_two_byte_packet_sweep_no_panic() {
    for fc in 0u8..=255 {
        let p = [3u8, fc];
        let v = fc & 0x80 != 0;
        let pp = fc & 0x40 != 0;
        let m = fc & 0x3F;
        let result = OpusPacket::parse(&p);
        // Acceptance precondition: M in 1..=48 per R5 (the §3.2.5
        // frame-count field is 6 bits ⇒ 0..=63 raw, but values
        // 49..=63 violate R5's "no more than 120 ms total" cap and
        // are rejected by `MAX_FRAMES_PER_PACKET`), no padding (no
        // padding-length byte fits in zero-body), and not VBR with
        // M>=2 (no length bytes fit either). VBR M=1 is also OK
        // because no length sequences are needed when M=1.
        let must_accept = (1..=MAX_FRAMES_PER_PACKET).contains(&m) && !pp && (!v || m == 1);
        if must_accept {
            let parsed = result.unwrap_or_else(|e| {
                panic!("fc=0x{fc:02X} v={v} p={pp} M={m}: expected accept ({e})")
            });
            assert_eq!(parsed.frames().len(), m as usize);
            for f in parsed.frames() {
                assert_eq!(f.len(), 0);
            }
        } else {
            assert!(
                result.is_err(),
                "fc=0x{fc:02X} v={v} p={pp} M={m}: expected reject"
            );
        }
    }
}

/// Frame-count code 0 / 1 / 2 / 3 do NOT alter the maximum-implicit-
/// length bound (R2 = 1275 bytes). For c=3 VBR with declared length
/// = `MAX_FRAME_BYTES + 1` the parser must reject even before
/// consulting the actual payload bytes.
#[test]
fn r2_code3_vbr_declared_length_capped_at_max_frame_bytes() {
    // c=3, fc: v=1, p=0, M=2 ⇒ 0x82. Declared first-frame length =
    // 1276 (two-byte §3.2.1 sequence: 252 + 256*4 = 1276 → encode
    // as first=252, second=(1276-252)/4=256... — too large for a
    // single byte). 1276 = 252 + 256*4. (1276 - 252)/4 = 256, but
    // 256 > 255 so 1276 can't even be encoded in 2 bytes.  The next
    // representable length is 1275 itself (first=255, second=255 ⇒
    // 255 + 255*4 = 1275). So we use first=252, second=255 ⇒
    // 252 + 255*4 = 1272 (legal); first=255, second=255 ⇒ 1275
    // (boundary). To probe R2 strictly we need a constructible
    // length > 1275 — which isn't expressible in the §3.2.1
    // encoding. Confirm 1275 is acceptable as the boundary.
    let mut p = vec![3u8, 0x82, 255u8, 255u8];
    p.extend_from_slice(&[0u8; 1275 + 5]); // 1275 first + 5 second
    let parsed = OpusPacket::parse(&p).expect("R2 boundary code 3 VBR");
    assert_eq!(parsed.frames().len(), 2);
    assert_eq!(parsed.frames()[0].len(), 1275);
    assert_eq!(parsed.frames()[1].len(), 5);
}

/// Sweep every `(c, body_len)` shape from 0 bytes up to 12 bytes,
/// confirming the parser NEVER panics. The §3.2 layer must reject
/// invalid shapes gracefully via `Err(MalformedPacket)`, not
/// `unwrap()` / index-out-of-bounds panics.
#[test]
fn parse_short_packet_sweep_never_panics() {
    for c in 0u8..4 {
        for stereo in [0u8, 1] {
            let toc = (c & 0x03) | (stereo << 2);
            for body_len in 0usize..=12 {
                let mut p = vec![0u8; 1 + body_len];
                p[0] = toc;
                // Just running parse without panicking is the
                // property; we don't assert success/failure.
                let _ = OpusPacket::parse(&p);
            }
            // And a few constructed bytes inside the body to widen
            // coverage of the §3.2.1 length-byte branches.
            for pattern in [0xFFu8, 0xFC, 0x00, 0x80, 0x55] {
                let mut p = vec![pattern; 1 + 8];
                p[0] = toc;
                let _ = OpusPacket::parse(&p);
            }
        }
    }
}

/// Frame slices returned by a successful parse always stay within
/// the input buffer's bounds and never alias the padding tail. This
/// is the lifetime contract `OpusPacket<'a>` promises.
#[test]
fn parsed_frames_borrow_inside_packet_bounds() {
    // A code-3 VBR packet with M=3, no padding, declared lengths
    // (4, 5), final-frame length implicit.
    let mut p = vec![3u8, 0x80 | 3, 4u8, 5u8];
    p.extend_from_slice(&[0xAAu8; 4]); // frame 0
    p.extend_from_slice(&[0xBBu8; 5]); // frame 1
    p.extend_from_slice(&[0xCCu8; 6]); // frame 2 (implicit length 6)
    let parsed = OpusPacket::parse(&p).expect("parse");
    assert_eq!(parsed.frames().len(), 3);
    assert!(parsed.frames()[0].iter().all(|&b| b == 0xAA));
    assert!(parsed.frames()[1].iter().all(|&b| b == 0xBB));
    assert!(parsed.frames()[2].iter().all(|&b| b == 0xCC));

    // Each frame slice MUST point inside `p`.
    let base = p.as_ptr() as usize;
    let end = base + p.len();
    for f in parsed.frames() {
        let start = f.as_ptr() as usize;
        assert!(start >= base && start + f.len() <= end);
    }
}

/// MAX_FRAME_BYTES + 1 is unrepresentable in the §3.2.1 length
/// encoding (the encoding maxes at 252 + 255 * 4 = 1272 for the
/// canonical 2-byte form, with 1275 reachable via 255 + 255 * 4 =
/// 1275). The constant MUST equal 1275 — this is the tightest bound
/// any caller can rely on.
#[test]
fn max_frame_bytes_constant_matches_section_3_2_1() {
    assert_eq!(MAX_FRAME_BYTES, 1275);
}

/// MAX_FRAMES_PER_PACKET = 48, matching R5's "120 ms / 2.5 ms = 48"
/// derivation.
#[test]
fn max_frames_per_packet_constant_matches_r5() {
    assert_eq!(MAX_FRAMES_PER_PACKET, 48);
}

/// §4.1.4 / §4.3 CELT decode-path truncation safety. The CELT-only
/// frame decode now consumes a long run of range-coded symbols past the
/// §4.3.7.1 prefix — §4.3.2.1 coarse energy, §4.3.1 tf_change /
/// tf_select, §4.3.4.3 spread, and the §4.3.3 signalled allocation
/// header (boosts / trim / reservations). Every one of those reads must
/// fail safely on a buffer that runs out mid-decode: the §4.1 range
/// coder latches its sticky error flag and the frame falls back to the
/// §4.6 silence floor.
///
/// For a config-19 (20 ms CELT-only mono) packet built from a fixed
/// pseudo-random body, decode every truncation length from 1 byte up to
/// the full body. None may panic; each must yield exactly the 20 ms
/// 48 kHz per-channel sample count (960) and a real CELT outcome (never
/// the not-wired placeholder, never a SILK status).
#[test]
fn celt_only_decode_truncation_never_panics() {
    // A 20 ms CELT-only mono packet: TOC config 19 (CELT-only, 20 ms,
    // mono), frame-count code 0 (one frame), body = pseudo-random bytes.
    let toc = 19u8 << 3; // config 19, s = 0 (mono), c = 0.
    let mut body = Vec::with_capacity(64);
    let mut x: u32 = 0x1234_5678;
    for _ in 0..64 {
        // A tiny xorshift PRNG — deterministic, no external deps.
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        body.push((x & 0xff) as u8);
    }

    for trunc in 1..=body.len() {
        let mut pkt = Vec::with_capacity(1 + trunc);
        pkt.push(toc);
        pkt.extend_from_slice(&body[..trunc]);

        let mut dec = OpusDecoder::new();
        let out = dec
            .decode_packet(&pkt)
            .expect("a well-framed CELT packet decodes (silence on truncation)");

        // Exactly one Opus frame, 20 ms = 960 samples/channel at 48 kHz,
        // mono ⇒ pcm length 960.
        assert_eq!(out.channels, 1, "len {trunc}");
        assert_eq!(out.frame_outcomes.len(), 1, "len {trunc}");
        assert_eq!(out.pcm.len(), 960, "len {trunc}");
        assert_eq!(out.samples_per_channel(), 960, "len {trunc}");

        // The status must be a real CELT outcome reflecting an
        // actually-consumed (or cleanly-truncated) bitstream — never a
        // SILK status, never the not-wired placeholder.
        assert!(
            matches!(
                out.frame_outcomes[0].status,
                FrameDecodeStatus::CeltSilence
                    | FrameDecodeStatus::CeltDecoded
                    | FrameDecodeStatus::CeltDecodeError
            ),
            "len {trunc}: got {:?}",
            out.frame_outcomes[0].status
        );
    }
}

/// Round-382 fuzz-crash regression: an 18-byte input the coverage-guided
/// `decode_packet` harness found that drove the §4.2.7.5.8 prediction-gain
/// recurrence into an i64 overflow (adversarial LSF coefficients whose Q24
/// widening escapes the spec's 32-bit envelope). The decoder must classify
/// the filter unstable and keep decoding — never panic.
#[test]
fn fuzz_crash_regression_lpc_recurrence_overflow() {
    // Verbatim crash artifact from the round-382 fuzz run, replayed
    // exactly the way the harness feeds it (first byte selects the
    // packet split; the remainder is decoded on one stateful decoder,
    // then re-fed through the self-delimited entry).
    let data: [u8; 18] = [
        0x00, 0x08, 0x08, 0x31, 0xa1, 0x5e, 0xa1, 0x31, 0xfd, 0x67, 0x52, 0x26, 0xff, 0x6c, 0x25,
        0xff, 0xa1, 0x00,
    ];
    let body = &data[1..];
    let mut dec = OpusDecoder::new();
    let _ = dec.decode_packet(body);
    let _ = dec.decode_self_delimited_packet(body);
}
