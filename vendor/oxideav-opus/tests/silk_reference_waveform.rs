//! Waveform-level SILK decode validation against the reference decodes
//! of the fixture corpus (RFC 6716 §4.2).
//!
//! Where `tests/silk_fixture_decode.rs` validates routing / structure /
//! signal content, this suite pins the decoded **waveform**: each
//! SILK-bearing fixture stream is decoded packet-by-packet through one
//! stateful [`oxideav_opus::OpusDecoder`], the RFC 7845 §4.2 pre-skip
//! is trimmed, and the 48 kHz PCM is compared sample-aligned against
//! the shipped reference decode (`<name>.expected.wav`) as a plain SNR.
//!
//! What the gates pin:
//!
//! * The **§4.2.9 upsampler delay calibration** — the WB fixtures
//!   measure ~69–72 dB; a single sample of misalignment collapses them
//!   below ~45 dB, so their 55 dB floors are a sharp regression gate
//!   on the group delay (and on the §4.2.8 mono one-sample delay,
//!   whose omission costs one *input* sample = 3–6 output samples).
//! * The **Hybrid SILK↔CELT alignment** — the hybrid fixture's SILK
//!   band must land on the (bit-aligned) CELT layer's timeline; see
//!   `tests/celt_fixture_decode.rs` for its gate.
//! * The **reconstruction accuracy floor** for NB/MB. These ceilings
//!   (~19 / ~28 dB measured) are NOT alignment: §4.2.7.9 explicitly
//!   frees the reconstruction from bit-exactness ("small errors should
//!   only introduce proportionally small distortions"), and the
//!   fixture references come from a fixed-point reconstruction whose
//!   rounding recirculates through the LTP feedback on strongly
//!   periodic content. The floors assert we never regress below the
//!   established accuracy.
//!
//! The Ogg page walker mirrors `silk_fixture_decode.rs` — test-only
//! fixture-loading scaffolding, not crate surface. No external library
//! source is consulted; the `.wav` references are black-box decoder
//! output shipped with the staged corpus.

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

/// Decode a SILK-only fixture stream end-to-end (asserting every frame
/// takes a SILK decode path) and return the pre-skip-trimmed SNR
/// against its reference decode.
fn silk_fixture_snr(stream: &[u8], expected_wav: &[u8]) -> f64 {
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
                    FrameDecodeStatus::SilkParamsDecoded | FrameDecodeStatus::SilkStereoDecoded
                ),
                "packet {i}: non-SILK status {:?}",
                fo.status
            );
        }
        pcm.extend_from_slice(&out.pcm);
    }
    let expected = wav_pcm_payload(expected_wav);
    let got = &pcm[pre_skip * channels..];
    assert!(
        got.len() >= expected.len().saturating_sub(960 * channels),
        "decoded stream too short: {} < {}",
        got.len(),
        expected.len()
    );
    snr_db(&expected, got)
}

/// WB stereo (config 9, mid/side): decodes **bit-exactly** against the
/// reference decode (the §4.2.7.9 fixed-point reconstruction, the
/// integer §4.2.8 unmix and the reference §4.2.9 resampler leave no
/// arithmetic slack). The 100 dB floor pins bit-exactness with a
/// margin for any future last-LSB drift.
#[test]
fn silk_wb_stereo_waveform_matches_reference() {
    let snr = silk_fixture_snr(
        include_bytes!("fixtures/silk-wb-stereo-20kbps.opus"),
        include_bytes!("fixtures/silk-wb-stereo-20kbps.expected.wav"),
    );
    assert!(snr > 100.0, "WB stereo waveform SNR {snr:.2} dB < 100");
}

/// WB mono main path of the FEC-enabled stream (the §4.2.5 LBRR bits
/// are consumed but the regular frames decode normally). Decodes
/// bit-exactly against the reference decode.
#[test]
fn silk_wb_mono_fec_stream_waveform_matches_reference() {
    let snr = silk_fixture_snr(
        include_bytes!("fixtures/fec-on.opus"),
        include_bytes!("fixtures/fec-on.expected.wav"),
    );
    assert!(
        snr > 100.0,
        "WB mono (fec-on) waveform SNR {snr:.2} dB < 100"
    );
}

/// MB 60 ms mono (config 7, three SILK frames per packet). Decodes
/// bit-exactly against the reference decode — the former ~28 dB
/// "reconstruction drift" ceiling was the float realization of
/// §4.2.7.9; the fixed-point core removed it.
#[test]
fn silk_mb_60ms_waveform_floor() {
    let snr = silk_fixture_snr(
        include_bytes!("fixtures/silk-mb-60ms-mono-20kbps.opus"),
        include_bytes!("fixtures/silk-mb-60ms-mono-20kbps.expected.wav"),
    );
    assert!(snr > 100.0, "MB 60 ms waveform SNR {snr:.2} dB < 100");
}

/// NB mono 440 Hz sine (config 1) — the strongest LTP recirculation
/// in the corpus (perfectly periodic content). Decodes bit-exactly
/// against the reference decode; the LTP feedback recirculates the
/// *same* rounding as the reference, so nothing drifts.
#[test]
fn silk_nb_mono_waveform_floor() {
    let snr = silk_fixture_snr(
        include_bytes!("fixtures/silk-nb-mono-16kbps.opus"),
        include_bytes!("fixtures/silk-nb-mono-16kbps.expected.wav"),
    );
    assert!(snr > 100.0, "NB mono waveform SNR {snr:.2} dB < 100");
}

/// NB voice-silence-voice at 6 kb/s: near-DTX 6-byte packets whose
/// excitation is pure LCG comfort noise. Decodes bit-exactly against
/// the reference decode (the LCG sequence and its ±few-LSB noise floor
/// reproduce sample-for-sample).
#[test]
fn silk_silence_low_bitrate_waveform_floor() {
    let snr = silk_fixture_snr(
        include_bytes!("fixtures/silence-low-bitrate.opus"),
        include_bytes!("fixtures/silence-low-bitrate.expected.wav"),
    );
    assert!(snr > 100.0, "silence-low-bitrate SNR {snr:.2} dB < 100");
}
