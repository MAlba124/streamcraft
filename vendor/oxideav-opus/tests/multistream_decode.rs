//! End-to-end multistream (multichannel) Opus decode validation — RFC
//! 7845 §3 + §5.1.1.
//!
//! These tests exercise the [`oxideav_opus::MultistreamDecoder`] path on
//! real, reference-encoder-produced SILK packets. A multistream Ogg
//! packet glues N independent Opus packets together (the first N−1 with
//! RFC 6716 Appendix-B self-delimited framing, the last with regular
//! framing); the decoder splits them, decodes each through its own
//! stateful sub-decoder, and assembles the C output channels per the
//! §5.1.1 channel-mapping rule.
//!
//! There is no committed multichannel fixture, so the multistream
//! packets here are *constructed* from the single-stream SILK fixtures by
//! wrapping their packets in the self-delimited framing — the same
//! framing a real multichannel encoder would use. The point is to
//! validate the split + per-stream decode + channel-map assembly, all of
//! which are codec-level operations independent of any container.
//!
//! No external library source is consulted; the only inputs are RFC 7845
//! §3 / §5.1.1, RFC 6716 §3.2 / Appendix B, and the committed fixtures.

use oxideav_opus::{ChannelMappingTable, MultistreamDecoder, OpusHead, OpusTocByte};

const FIXTURE_NB_MONO: &[u8] = include_bytes!("fixtures/silk-nb-mono-16kbps.opus");
const FIXTURE_MB_60MS: &[u8] = include_bytes!("fixtures/silk-mb-60ms-mono-20kbps.opus");
const FIXTURE_WB_STEREO: &[u8] = include_bytes!("fixtures/silk-wb-stereo-20kbps.opus");

/// Minimal test-only Ogg page de-laker (see `silk_fixture_decode.rs` for
/// the rationale): recovers the logical packets so we can pull real Opus
/// audio packets out of the fixture.
fn ogg_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut off = 0usize;
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    while off + 27 <= data.len() {
        assert_eq!(&data[off..off + 4], b"OggS");
        let nseg = data[off + 26] as usize;
        let seg_table_end = off + 27 + nseg;
        assert!(seg_table_end <= data.len());
        let segtab = &data[off + 27..seg_table_end];
        let mut p = seg_table_end;
        for &s in segtab {
            let seg_end = p + s as usize;
            assert!(seg_end <= data.len());
            cur.extend_from_slice(&data[p..seg_end]);
            p = seg_end;
            if s < 255 {
                packets.push(std::mem::take(&mut cur));
            }
        }
        off = p;
    }
    packets
}

/// The raw OpusHead bytes of the fixture (Ogg packet 0).
fn fixture_head_bytes() -> Vec<u8> {
    ogg_packets(FIXTURE_NB_MONO).remove(0)
}

/// The fixture's audio packets (after OpusHead + OpusTags).
fn fixture_audio_packets() -> Vec<Vec<u8>> {
    let mut p = ogg_packets(FIXTURE_NB_MONO);
    p.drain(..2);
    p
}

/// The 60 ms MB fixture's audio packets.
fn mb_audio_packets() -> Vec<Vec<u8>> {
    let mut p = ogg_packets(FIXTURE_MB_60MS);
    p.drain(..2);
    p
}

/// The WB stereo fixture's audio packets.
fn wb_stereo_audio_packets() -> Vec<Vec<u8>> {
    let mut p = ogg_packets(FIXTURE_WB_STEREO);
    p.drain(..2);
    p
}

/// Wrap a regular code-0 mono Opus packet (TOC byte + frame body) in the
/// RFC 6716 Appendix-B self-delimited framing: TOC | §3.2.1 length |
/// frame body. The fixture is config 1 (SILK/NB/20 ms), code 0, so each
/// audio packet is exactly TOC + one frame.
fn self_delimit_code0(packet: &[u8]) -> Vec<u8> {
    let toc = OpusTocByte::from_byte(packet[0]);
    // Confirm the assumption (single-frame, code 0): frame body is the
    // tail after the TOC byte.
    assert_eq!(toc.frame_count_range(), (1, 1), "fixture is code 0");
    let body = &packet[1..];
    let mut out = vec![packet[0]];
    // §3.2.1 one/two-byte length encoding of the body length.
    let len = body.len();
    if len < 252 {
        out.push(len as u8);
    } else {
        out.push((252 + (len - 252) % 4) as u8);
        out.push(((len - 252) / 4) as u8);
    }
    out.extend_from_slice(body);
    out
}

#[test]
fn fixture_opus_head_parses() {
    let head = OpusHead::parse(&fixture_head_bytes()).unwrap();
    // config 1 ⇒ SILK/NB; the fixture header is mono, family 0.
    assert_eq!(head.version, 1);
    assert_eq!(head.channel_count, 1);
    assert_eq!(head.mapping_family, 0);
    assert_eq!(head.mapping.stream_count, 1);
    assert_eq!(head.mapping.coupled_count, 0);
    assert_eq!(head.mapping.mapping, vec![0]);
    // The fixture's SILK NB input rate is 8 kHz; pre-skip is 312 samples.
    assert_eq!(head.input_sample_rate, 8000);
    assert_eq!(head.pre_skip, 312);
}

#[test]
fn single_stream_family0_matches_plain_decode() {
    // A family-0 mono OpusHead ⇒ N = 1: the multistream decoder must
    // produce exactly what a plain decoder does.
    let head = OpusHead::parse(&fixture_head_bytes()).unwrap();
    let mut ms = MultistreamDecoder::from_head(&head);
    assert_eq!(ms.output_channels(), 1);

    let mut plain = oxideav_opus::OpusDecoder::new();
    for pk in fixture_audio_packets() {
        let ms_out = ms.decode_packet(&pk).unwrap();
        let plain_out = plain.decode_packet(&pk).unwrap();
        assert_eq!(ms_out.channels, 1);
        assert_eq!(
            ms_out.pcm, plain_out.pcm,
            "N=1 multistream must equal plain decode"
        );
    }
}

#[test]
fn two_mono_streams_map_to_distinct_output_channels() {
    // Build a synthetic 2-stream packet: stream 0 = self-delimited mono
    // packet A, stream 1 = regular mono packet B (a *different* audio
    // packet so the two channels carry different signal). Map them to a
    // 2-channel output: out[0] ← stream 0, out[1] ← stream 1.
    //
    // Channel-mapping table: N = 2, M = 0 (both mono). Per §5.1.1, mono
    // stream s is index `s` (2*M = 0). So mapping [0, 1] routes stream 0
    // to the left channel and stream 1 to the right.
    let audio = fixture_audio_packets();
    let pkt_a = &audio[5];
    let pkt_b = &audio[40];

    let mut multistream_packet = self_delimit_code0(pkt_a);
    multistream_packet.extend_from_slice(pkt_b);

    let mapping = ChannelMappingTable {
        stream_count: 2,
        coupled_count: 0,
        mapping: vec![0, 1],
    };
    let mut ms = MultistreamDecoder::new(mapping);
    assert_eq!(ms.output_channels(), 2);

    let out = ms.decode_packet(&multistream_packet).unwrap();
    assert_eq!(out.channels, 2);
    assert_eq!(out.pcm.len(), out.samples_per_channel * 2);
    assert_eq!(out.samples_per_channel, 960); // 20 ms @ 48 kHz

    // Cross-check the two output channels against decoding each stream on
    // its own. The left output channel must equal a plain decode of
    // packet A; the right must equal a plain decode of packet B.
    let mut da = oxideav_opus::OpusDecoder::new();
    let mut db = oxideav_opus::OpusDecoder::new();
    let a = da.decode_packet(pkt_a).unwrap();
    let b = db.decode_packet(pkt_b).unwrap();
    for s in 0..out.samples_per_channel {
        assert_eq!(out.pcm[s * 2], a.pcm[s], "left channel sample {s}");
        assert_eq!(out.pcm[s * 2 + 1], b.pcm[s], "right channel sample {s}");
    }
}

#[test]
fn mismatched_stream_durations_rejected() {
    // RFC 7845 §3: every stream in an Ogg packet MUST have the same
    // duration. Pair a 20 ms NB stream with a 60 ms MB stream → reject.
    let nb = fixture_audio_packets();
    let mb = mb_audio_packets();
    let mut multistream_packet = self_delimit_code0(&nb[5]); // 20 ms
    multistream_packet.extend_from_slice(&mb[3]); // 60 ms

    let mapping = ChannelMappingTable {
        stream_count: 2,
        coupled_count: 0,
        mapping: vec![0, 1],
    };
    let mut ms = MultistreamDecoder::new(mapping);
    assert!(
        ms.decode_packet(&multistream_packet).is_err(),
        "mismatched stream durations must be rejected"
    );
}

#[test]
fn silence_index_255_yields_zero_channel() {
    // A 3-channel output where channel 2 is the reserved silence index
    // 255; channels 0 and 1 come from two mono streams.
    let audio = fixture_audio_packets();
    let mut multistream_packet = self_delimit_code0(&audio[5]);
    multistream_packet.extend_from_slice(&audio[40]);

    let mapping = ChannelMappingTable {
        stream_count: 2,
        coupled_count: 0,
        mapping: vec![0, 1, 255],
    };
    let mut ms = MultistreamDecoder::new(mapping);
    let out = ms.decode_packet(&multistream_packet).unwrap();
    assert_eq!(out.channels, 3);
    // The 255-mapped channel (output channel 2) is pure silence.
    for s in 0..out.samples_per_channel {
        assert_eq!(out.pcm[s * 3 + 2], 0, "silence channel sample {s}");
    }
    // ...while the other two carry the (non-trivially-routed) signal.
    let any_nonzero_left = (0..out.samples_per_channel).any(|s| out.pcm[s * 3] != 0);
    assert!(any_nonzero_left, "left channel should carry signal");
}

#[test]
fn coupled_stereo_stream_splits_to_left_right() {
    // A single coupled (stereo) stream: N = 1, M = 1. Per §5.1.1 the
    // two output channels are indices 0 (left) and 1 (right) of stream 0
    // (2*M = 2). The multistream output must be byte-identical to a plain
    // stereo decode of the same packet.
    let stereo = wb_stereo_audio_packets();
    let pkt = &stereo[5];

    let mapping = ChannelMappingTable {
        stream_count: 1,
        coupled_count: 1,
        mapping: vec![0, 1],
    };
    let mut ms = MultistreamDecoder::new(mapping);
    assert_eq!(ms.output_channels(), 2);
    let out = ms.decode_packet(pkt).unwrap();
    assert_eq!(out.channels, 2);

    let mut plain = oxideav_opus::OpusDecoder::new();
    let plain_out = plain.decode_packet(pkt).unwrap();
    assert_eq!(plain_out.channels, 2);
    // The coupled-stream L/R extraction must reproduce the plain stereo
    // interleave exactly.
    assert_eq!(out.pcm, plain_out.pcm);
}

#[test]
fn coupled_stream_swapped_channel_map() {
    // §5.1.1 mappings are arbitrary: map the coupled stream's right
    // channel to output 0 and its left to output 1 (a channel swap).
    let stereo = wb_stereo_audio_packets();
    let pkt = &stereo[7];

    let mapping = ChannelMappingTable {
        stream_count: 1,
        coupled_count: 1,
        mapping: vec![1, 0], // swapped
    };
    let mut ms = MultistreamDecoder::new(mapping);
    let out = ms.decode_packet(pkt).unwrap();

    let mut plain = oxideav_opus::OpusDecoder::new();
    let plain_out = plain.decode_packet(pkt).unwrap();
    for s in 0..out.samples_per_channel {
        // out[0] is plain right, out[1] is plain left.
        assert_eq!(
            out.pcm[s * 2],
            plain_out.pcm[s * 2 + 1],
            "swapped left sample {s}"
        );
        assert_eq!(
            out.pcm[s * 2 + 1],
            plain_out.pcm[s * 2],
            "swapped right sample {s}"
        );
    }
}

#[test]
fn duplicate_index_routes_same_stream_to_two_outputs() {
    // §5.1.1 permits a decoded channel to feed several output channels.
    // Map a single mono stream to BOTH output channels of a 2-channel
    // output (N = 1, mapping [0, 0]).
    let audio = fixture_audio_packets();
    let mapping = ChannelMappingTable {
        stream_count: 1,
        coupled_count: 0,
        mapping: vec![0, 0],
    };
    let mut ms = MultistreamDecoder::new(mapping);
    let out = ms.decode_packet(&audio[10]).unwrap();
    assert_eq!(out.channels, 2);
    // Both channels identical (same decoded channel duplicated).
    for s in 0..out.samples_per_channel {
        assert_eq!(
            out.pcm[s * 2],
            out.pcm[s * 2 + 1],
            "duplicated channel sample {s}"
        );
    }
}

#[test]
fn assemble_multistream_packet_matches_hand_construction() {
    // The write-side assembler must produce, for a pair of code-0
    // packets, exactly the bytes the hand construction above builds:
    // self-delimited stream 0 followed by the verbatim final stream.
    let audio = fixture_audio_packets();
    let pkt_a = &audio[5];
    let pkt_b = &audio[40];

    let assembled = oxideav_opus::assemble_multistream_packet(&[pkt_a, pkt_b]).expect("assemble");
    let mut hand = self_delimit_code0(pkt_a);
    hand.extend_from_slice(pkt_b);
    assert_eq!(assembled, hand);
}

#[test]
fn assembled_stream_pair_decodes_like_plain_decoders() {
    // Whole-stream write→read: assemble every consecutive fixture
    // packet with itself as a 2-mono-stream packet, decode with a
    // MultistreamDecoder, and require both output channels to equal a
    // plain stateful decode of the fixture stream.
    let audio = fixture_audio_packets();
    let mapping = ChannelMappingTable {
        stream_count: 2,
        coupled_count: 0,
        mapping: vec![0, 1],
    };
    let mut ms = MultistreamDecoder::new(mapping);
    let mut plain = oxideav_opus::OpusDecoder::new();
    for pk in &audio {
        let assembled = oxideav_opus::assemble_multistream_packet(&[pk, pk]).expect("assemble");
        let out = ms.decode_packet(&assembled).expect("multistream decode");
        let reference = plain.decode_packet(pk).expect("plain decode");
        assert_eq!(out.channels, 2);
        assert_eq!(out.samples_per_channel, reference.pcm.len());
        for s in 0..out.samples_per_channel {
            assert_eq!(out.pcm[s * 2], reference.pcm[s], "left sample {s}");
            assert_eq!(out.pcm[s * 2 + 1], reference.pcm[s], "right sample {s}");
        }
    }
}

#[test]
fn fixture_opus_head_recomposes_byte_identically() {
    // OpusHead::compose is the exact write-side mirror of parse: the
    // fixture's real family-0 identification header re-emits
    // byte-for-byte.
    let bytes = fixture_head_bytes();
    let head = OpusHead::parse(&bytes).unwrap();
    let composed = head.compose().expect("compose");
    assert_eq!(composed, bytes);
}

/// The 5.1 multistream fixture (family 1, 4 streams: 2 coupled +
/// 2 mono) and its reference decode — produced by the RFC 6716 §A
/// reference listing's multistream decoder (RFC 8251 patches applied),
/// pre-skip trimmed, end-trimmed to the granule length, in the
/// RFC 7845 §5.1.1.2 Vorbis channel order.
const FIXTURE_MS51: &[u8] = include_bytes!("fixtures/multistream-5.1.opus");
const FIXTURE_MS51_EXPECTED: &[u8] = include_bytes!("fixtures/multistream-5.1.expected.wav");

#[test]
fn multistream_51_fixture_decodes_at_the_float_noise_floor() {
    let packets = ogg_packets(FIXTURE_MS51);
    let head = OpusHead::parse(&packets[0]).expect("OpusHead");
    assert_eq!(head.channel_count, 6);
    assert_eq!(head.mapping_family, 1);
    assert_eq!(head.mapping.stream_count, 4);
    assert_eq!(head.mapping.coupled_count, 2);
    assert_eq!(head.pre_skip, 312);

    let mut dec = MultistreamDecoder::new(head.mapping.clone());
    let mut pcm: Vec<i16> = Vec::new();
    for p in &packets[2..] {
        let out = dec.decode_packet(p).expect("multistream decode");
        assert_eq!(out.channels, 6);
        pcm.extend_from_slice(&out.pcm);
    }

    // Pre-skip + end-trim to the granule length (48 000 samples).
    let ch = 6usize;
    let skip = head.pre_skip as usize;
    assert!(pcm.len() >= (skip + 48_000) * ch, "decode too short");
    let trimmed = &pcm[skip * ch..(skip + 48_000) * ch];

    // 44-byte canonical WAV header; s16le payload in Vorbis order.
    let payload = &FIXTURE_MS51_EXPECTED[44..];
    let want: Vec<i16> = payload
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(trimmed.len(), want.len());

    // Whole-stream SNR against the reference-listing decode: the
    // fixture is CELT-only, so our decode sits at the float-noise
    // floor (~100 dB measured); gate with margin.
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in trimmed.iter().zip(want.iter()) {
        let d = f64::from(*a) - f64::from(*b);
        num += d * d;
        den += f64::from(*b) * f64::from(*b);
    }
    let snr = 10.0 * (den / num.max(1e-30)).log10();
    assert!(snr > 90.0, "multistream-5.1 SNR {snr:.2} dB < 90");

    // And per channel: every one of the six Vorbis-order channels
    // reconstructs (no channel-order or routing slip).
    for c in 0..ch {
        let mut num_c = 0.0f64;
        let mut den_c = 0.0f64;
        for s in 0..48_000 {
            let d = f64::from(trimmed[s * ch + c]) - f64::from(want[s * ch + c]);
            num_c += d * d;
            den_c += f64::from(want[s * ch + c]) * f64::from(want[s * ch + c]);
        }
        let snr_c = 10.0 * (den_c / num_c.max(1e-30)).log10();
        assert!(snr_c > 80.0, "channel {c} SNR {snr_c:.2} dB < 80");
    }
}
