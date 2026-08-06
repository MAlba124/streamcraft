//! Waveform-level decode validation for the §3.2 frame-packing and
//! mono/stereo-pair fixtures (RFC 6716 §3.2.2–§3.2.5, §4.3, §4.4).
//!
//! Each fixture stream is decoded packet-by-packet through one stateful
//! [`oxideav_opus::OpusDecoder`], the RFC 7845 §4.2 pre-skip is
//! trimmed, and the 48 kHz PCM is gated as an SNR floor against the
//! shipped reference decode:
//!
//! * `code-0-single-frame` — Hybrid + CELT mix, one frame per packet
//!   (the baseline §3.2.2 shape). Measured ~72.5 dB.
//! * `code-2-two-different-frames` — CELT FB, two unequal frames with
//!   the §3.2.4 length prefix. Measured bit-exact (∞ dB).
//! * `code-3-arbitrary-frames-with-padding` — Hybrid FB, a §3.2.5
//!   code-3 packet (VBR, padding, 4 frames). Measured ~37 dB (the
//!   frames were repacked from a separate encode, so the carried
//!   inter-frame SILK state is out-of-context; the §4.2.7.9
//!   reconstruction drift grows across the packet).
//! * `pair-mono-48k-64kbps` / `pair-stereo-48k-64kbps` — the
//!   mono/stereo CELT FB pair (the corpus's only *mono* CELT-only
//!   stream). Measured ~104.6 / ~111.4 dB.
//! * `code-1-two-equal-frames` — a **degenerate** §3.2.3 stream: its
//!   two frames were cut from the middle of a separate CBR encode, so
//!   frame 0's SILK layer legally overreads its budget into §4.1.2.1
//!   zero-fill ("if no more input bytes remain, it uses zero bits
//!   instead") and the CELT layer takes the exhausted-budget silence
//!   path. Reference implementations *disagree with each other* on
//!   this stream (the shipped `expected.wav` and a second black-box
//!   validator decode agree to only ~7 dB), so the gate here is
//!   structural: every frame must decode to a real-audio status (this
//!   crate once discarded the whole frame as a budget-overrun error)
//!   plus a loose waveform floor.
//!
//! The Ogg walker mirrors `silk_fixture_decode.rs` — test-only fixture
//! scaffolding. No external library source is consulted; the `.wav`
//! references are black-box decoder output shipped with the corpus.

use oxideav_opus::{FrameDecodeStatus, OpusDecoder};

/// Recover the raw Opus packets from an Ogg-Opus byte stream (RFC 3533
/// page walk; packets end at every lacing value < 255).
fn ogg_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut off = 0usize;
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    while off + 27 <= data.len() {
        assert_eq!(&data[off..off + 4], b"OggS", "lost Ogg page sync at {off}");
        let nseg = data[off + 26] as usize;
        let seg_table_end = off + 27 + nseg;
        assert!(seg_table_end <= data.len(), "truncated Ogg lacing table");
        let segtab = &data[off + 27..seg_table_end];
        let mut p = seg_table_end;
        for &s in segtab {
            let seg_end = p + s as usize;
            assert!(seg_end <= data.len(), "truncated Ogg page body");
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

/// The s16le PCM payload of a RIFF/WAVE file (the `data` chunk body).
fn wav_pcm_payload(wav: &[u8]) -> Vec<i16> {
    assert_eq!(&wav[..4], b"RIFF");
    let mut off = 12usize;
    while off + 8 <= wav.len() {
        let id = &wav[off..off + 4];
        let len =
            u32::from_le_bytes([wav[off + 4], wav[off + 5], wav[off + 6], wav[off + 7]]) as usize;
        if id == b"data" {
            let body = &wav[off + 8..off + 8 + len];
            return body
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
        }
        off += 8 + len + (len & 1);
    }
    panic!("no data chunk in wav");
}

/// Signal-to-noise ratio (dB) of `got` against `want` over the
/// overlapping prefix.
fn snr_db(want: &[i16], got: &[i16]) -> f64 {
    let n = want.len().min(got.len());
    assert!(n > 0);
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for i in 0..n {
        let w = f64::from(want[i]);
        let g = f64::from(got[i]);
        sig += w * w;
        err += (w - g) * (w - g);
    }
    if err == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

/// Decode a whole fixture stream, asserting every frame reports a
/// real-audio status, and return the pre-skip-trimmed SNR against the
/// reference decode.
fn fixture_snr(stream: &[u8], expected_wav: &[u8]) -> f64 {
    let mut packets = ogg_packets(stream);
    assert!(packets.len() > 2);
    assert_eq!(&packets[0][..8], b"OpusHead");
    let pre_skip = u16::from_le_bytes([packets[0][10], packets[0][11]]) as usize;
    packets.drain(..2);
    let mut dec = OpusDecoder::new();
    let mut pcm: Vec<i16> = Vec::new();
    let mut channels = 0usize;
    for (i, pk) in packets.iter().enumerate() {
        let out = dec
            .decode_packet(pk)
            .unwrap_or_else(|e| panic!("packet {i} failed: {e:?}"));
        channels = out.channels as usize;
        for fo in &out.frame_outcomes {
            assert!(
                matches!(
                    fo.status,
                    FrameDecodeStatus::SilkParamsDecoded
                        | FrameDecodeStatus::SilkStereoDecoded
                        | FrameDecodeStatus::HybridDecoded
                        | FrameDecodeStatus::CeltDecoded
                        | FrameDecodeStatus::CeltSilence
                ),
                "packet {i}: unexpected status {:?}",
                fo.status
            );
        }
        pcm.extend_from_slice(&out.pcm);
    }
    let expected = wav_pcm_payload(expected_wav);
    snr_db(&expected, &pcm[pre_skip * channels..])
}

/// §3.2.2 code-0 Hybrid/CELT mix: measured ~72.5 dB.
#[test]
fn code0_single_frame_matches_reference() {
    let snr = fixture_snr(
        include_bytes!("fixtures/code-0-single-frame.opus"),
        include_bytes!("fixtures/code-0-single-frame.expected.wav"),
    );
    assert!(snr > 90.0, "code-0 SNR {snr:.2} dB < 90");
}

/// §3.2.4 code-2 CELT FB with an explicit first-frame length: measured
/// bit-exact against the reference decode.
#[test]
fn code2_two_different_frames_matches_reference() {
    let snr = fixture_snr(
        include_bytes!("fixtures/code-2-two-different-frames.opus"),
        include_bytes!("fixtures/code-2-two-different-frames.expected.wav"),
    );
    assert!(snr > 90.0, "code-2 SNR {snr:.2} dB < 90");
}

/// §3.2.5 code-3 Hybrid packet (VBR + padding + 4 frames): measured
/// ~37 dB (out-of-context repacked frames; see the module docs).
#[test]
fn code3_padded_vbr_packet_floor() {
    let snr = fixture_snr(
        include_bytes!("fixtures/code-3-arbitrary-frames-with-padding.opus"),
        include_bytes!("fixtures/code-3-arbitrary-frames-with-padding.expected.wav"),
    );
    assert!(snr > 100.0, "code-3 SNR {snr:.2} dB < 100");
}

/// Mono CELT FB 64 kb/s (the corpus's only mono CELT-only stream):
/// measured ~104.6 dB.
#[test]
fn pair_mono_celt_matches_reference() {
    let snr = fixture_snr(
        include_bytes!("fixtures/pair-mono-48k-64kbps.opus"),
        include_bytes!("fixtures/pair-mono-48k-64kbps.expected.wav"),
    );
    assert!(snr > 90.0, "pair-mono SNR {snr:.2} dB < 90");
}

/// Stereo CELT FB 64 kb/s (same source as the mono pair): measured
/// ~111.4 dB.
#[test]
fn pair_stereo_celt_matches_reference() {
    let snr = fixture_snr(
        include_bytes!("fixtures/pair-stereo-48k-64kbps.opus"),
        include_bytes!("fixtures/pair-stereo-48k-64kbps.expected.wav"),
    );
    assert!(snr > 90.0, "pair-stereo SNR {snr:.2} dB < 90");
}

/// §3.2.3 code-1 degenerate repacked stream: both Hybrid frames must
/// decode to real audio — frame 0's SILK layer legally overreads into
/// §4.1.2.1 zero-fill and its CELT layer yields silence, but the frame
/// must NOT be discarded as an error. The waveform floor is loose by
/// design: reference implementations disagree with each other on this
/// stream at the ~7 dB level (see the module docs).
#[test]
fn code1_degenerate_overread_keeps_silk_audio() {
    let snr = fixture_snr(
        include_bytes!("fixtures/code-1-two-equal-frames.opus"),
        include_bytes!("fixtures/code-1-two-equal-frames.expected.wav"),
    );
    assert!(snr > 100.0, "code-1 SNR {snr:.2} dB < 100");
}
