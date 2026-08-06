//! End-to-end CELT-only decode validation against the in-project Opus
//! fixtures (RFC 6716 §4.3): whole Ogg-Opus streams are decoded
//! packet-by-packet and the resulting 48 kHz PCM is compared —
//! sample-aligned, pre-skip applied — against the black-box reference
//! decode (`expected.wav`) shipped with each fixture.
//!
//! Unlike the SILK fixtures (whose §4.2.9 resampler is non-normative,
//! precluding a waveform-level comparison), CELT-only frames run
//! entirely at the 48 kHz output rate: every decode stage is the
//! normatively-specified one, so the decoded waveform must match the
//! reference decode up to arithmetic-precision differences. The
//! comparison metric is a plain signal-to-noise ratio over the
//! overlapping samples; the thresholds assert *waveform* agreement
//! (tens of dB), far beyond what a structurally-wrong decode could
//! produce by accident.
//!
//! The tiny Ogg page walker below mirrors `silk_fixture_decode.rs` —
//! test-only fixture-loading scaffolding, not crate surface. The
//! `.opus` / `.wav` fixture pairs are committed in `tests/fixtures/`
//! (copied from `docs/audio/opus/fixtures/`) so the suite runs in the
//! standalone CI. No external library source is consulted.

use oxideav_opus::{FrameDecodeStatus, OpusDecoder};

const FIXTURE_FB_STEREO: &[u8] = include_bytes!("fixtures/celt-fb-stereo-128kbps.opus");
const EXPECTED_FB_STEREO: &[u8] = include_bytes!("fixtures/celt-fb-stereo-128kbps.expected.wav");
const FIXTURE_LOW_LATENCY: &[u8] = include_bytes!("fixtures/celt-2.5ms-low-latency.opus");
const EXPECTED_LOW_LATENCY: &[u8] = include_bytes!("fixtures/celt-2.5ms-low-latency.expected.wav");

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

/// Pre-skip from the OpusHead packet (RFC 7845 §5.1, little-endian at
/// offset 10).
fn opus_head_pre_skip(head: &[u8]) -> usize {
    assert_eq!(&head[..8], b"OpusHead");
    u16::from_le_bytes([head[10], head[11]]) as usize
}

/// The s16le PCM payload of a RIFF/WAVE file (the `data` chunk body).
fn wav_pcm_payload(wav: &[u8]) -> Vec<i16> {
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
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

/// Decode a fixture end-to-end and return (pcm, channels, pre_skip).
fn decode_fixture(stream: &'static [u8]) -> (Vec<i16>, usize, usize) {
    let mut packets = ogg_packets(stream);
    assert!(packets.len() > 2);
    let pre_skip = opus_head_pre_skip(&packets[0]);
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
                    FrameDecodeStatus::CeltDecoded | FrameDecodeStatus::CeltSilence
                ),
                "packet {i}: unexpected status {:?}",
                fo.status
            );
        }
        pcm.extend_from_slice(&out.pcm);
    }
    (pcm, channels, pre_skip)
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

fn run_fixture(stream: &'static [u8], expected_wav: &'static [u8], channels_want: usize) -> f64 {
    let (pcm, channels, pre_skip) = decode_fixture(stream);
    assert_eq!(channels, channels_want);
    let expected = wav_pcm_payload(expected_wav);
    // RFC 7845 §4.2: drop pre-skip samples from the decoded stream.
    let got = &pcm[pre_skip * channels..];
    assert!(
        got.len() >= expected.len(),
        "decoded {} < expected {}",
        got.len(),
        expected.len()
    );
    snr_db(&expected, got)
}

/// 20 ms full-band stereo music-rate stream: 75 packets, 1.5 s. The
/// measured SNR against the reference decode is ~100 dB (i16
/// quantization + f32-vs-f64 arithmetic noise only); the 60 dB
/// threshold leaves margin while still asserting waveform-level
/// agreement.
#[test]
fn celt_fb_stereo_128kbps_matches_reference_waveform() {
    let snr = run_fixture(FIXTURE_FB_STEREO, EXPECTED_FB_STEREO, 2);
    assert!(
        snr > 90.0,
        "CELT FB stereo waveform SNR {snr:.2} dB below threshold"
    );
}

/// 2.5 ms low-latency stereo stream (400 packets, 1 s): the minimum
/// CELT frame size, where the overlap spans the whole frame.
#[test]
fn celt_low_latency_2_5ms_matches_reference_waveform() {
    let snr = run_fixture(FIXTURE_LOW_LATENCY, EXPECTED_LOW_LATENCY, 2);
    assert!(
        snr > 75.0,
        "CELT 2.5 ms stereo waveform SNR {snr:.2} dB below threshold"
    );
}

/// Hybrid (SILK+CELT) fixture: the stream mode-switches between
/// Hybrid and CELT-only packets. Every frame must decode to a
/// real-audio status, and the whole stream must reach waveform-level
/// agreement with the reference decode. The SILK band runs through
/// the §4.2.9 upsampler whose group delay places it on the CELT
/// layer's timeline (the encoder pre-delayed the MDCT layer by the
/// same amount), so the summed §4.4 output is directly comparable:
/// measured ~37.5 dB whole-stream (the CELT-only segment is
/// reference-exact; the Hybrid segment carries the §4.2.7.9
/// reconstruction-arithmetic drift of its SILK band). The 30 dB floor
/// pins the SILK↔CELT alignment — a single-sample slip of the SILK
/// band against the CELT band collapses the Hybrid segment to ~19 dB
/// and the whole stream below the gate.
#[test]
fn hybrid_fb_mono_28kbps_matches_reference_waveform() {
    let stream: &[u8] = include_bytes!("fixtures/hybrid-fb-mono-28kbps.opus");
    let expected_wav: &[u8] = include_bytes!("fixtures/hybrid-fb-mono-28kbps.expected.wav");
    let mut packets = ogg_packets(stream);
    let pre_skip = opus_head_pre_skip(&packets[0]);
    packets.drain(..2);
    let mut dec = OpusDecoder::new();
    let mut pcm: Vec<i16> = Vec::new();
    let mut hybrid_frames = 0usize;
    for (i, pk) in packets.iter().enumerate() {
        let out = dec
            .decode_packet(pk)
            .unwrap_or_else(|e| panic!("packet {i}: {e:?}"));
        for fo in &out.frame_outcomes {
            assert!(
                matches!(
                    fo.status,
                    FrameDecodeStatus::HybridDecoded
                        | FrameDecodeStatus::CeltDecoded
                        | FrameDecodeStatus::CeltSilence
                ),
                "packet {i}: unexpected status {:?}",
                fo.status
            );
            if fo.status == FrameDecodeStatus::HybridDecoded {
                hybrid_frames += 1;
            }
        }
        pcm.extend_from_slice(&out.pcm);
    }
    assert!(hybrid_frames > 0, "fixture must exercise the Hybrid path");

    let expected = wav_pcm_payload(expected_wav);
    let snr = snr_db(&expected, &pcm[pre_skip..]);
    assert!(snr > 60.0, "hybrid waveform SNR {snr:.2} dB < 60");
}
