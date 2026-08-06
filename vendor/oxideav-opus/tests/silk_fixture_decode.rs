//! End-to-end SILK-only decode validation against the in-project Opus
//! fixtures (RFC 6716 §4.2).
//!
//! These tests are the first whole-stream exercise of the
//! [`oxideav_opus::OpusDecoder::decode_packet`] path on real,
//! reference-encoder-produced data. Until now the crate's tests covered the decode
//! building blocks individually and the CELT synthesis backend in
//! isolation; here a complete Ogg-Opus stream is decoded packet-by-packet
//! and the resulting 48 kHz PCM is checked for structural correctness and
//! signal content.
//!
//! ## Why a test-only Ogg reader
//!
//! `oxideav-opus` is a **codec** crate: it consumes raw Opus packets and
//! produces PCM. Ogg page walking is **container** work that belongs in
//! `oxideav-ogg`, and the workspace's no-cross-crate-dev-dependency rule
//! keeps that crate out of this one's test graph. The fixtures, however,
//! are distributed as `.opus` (Ogg-encapsulated) files. The
//! `ogg_packets` helper below is therefore a *minimal, test-only*
//! Ogg page de-laker whose only job is to recover the raw Opus packets
//! that feed the codec under test. It is not part of the crate's public
//! surface and lives nowhere in `src/`; it is fixture-loading scaffolding,
//! analogous to taking a `[dev-dependencies]` on a container crate.
//!
//! ## Validation strategy
//!
//! This suite validates routing / structure / signal content; the
//! waveform-level comparison against each fixture's reference decode
//! (`expected.wav`) lives in `tests/silk_reference_waveform.rs`, whose
//! SNR gates pin the §4.2.9 upsampler delay calibration and the
//! §4.2.8 mono one-sample delay. These tests validate:
//!
//! 1. **Packet routing** — every packet's §3.1 TOC parses to the
//!    fixture's documented mode / bandwidth / channel count.
//! 2. **Whole-stream decode** — every audio packet decodes without an
//!    error status, across mono / stereo and NB / MB / WB / 20 ms / 60 ms.
//! 3. **Sample-count accounting** — the produced 48 kHz sample count
//!    matches the §3 frame-duration arithmetic.
//! 4. **Signal content** — for the 440 Hz NB sine fixture, a Goertzel
//!    probe confirms the decoded output is dominated by the 440 Hz tone
//!    (independent of the waveform-level gates, so a failure here
//!    points at gross routing/synthesis breakage rather than drift).
//!
//! The `.opus` fixture streams are committed in `tests/fixtures/` (copied
//! from the project's `docs/audio/opus/fixtures/` corpus) so the suite is
//! self-contained and runs in the crate's standalone CI without the
//! umbrella `docs/` submodule. No external library source is consulted.

use oxideav_opus::{Bandwidth, ChannelMapping, FrameDecodeStatus, Mode, OpusDecoder, OpusTocByte};

/// The three SILK fixture streams, embedded at compile time. Each is an
/// Ogg-Opus file produced by a reference encoder (a black-box validator) from a known
/// synthetic source; see `docs/audio/opus/fixtures/<name>/notes.md`.
const FIXTURE_NB_MONO: &[u8] = include_bytes!("fixtures/silk-nb-mono-16kbps.opus");
const FIXTURE_WB_STEREO: &[u8] = include_bytes!("fixtures/silk-wb-stereo-20kbps.opus");
const FIXTURE_MB_60MS_MONO: &[u8] = include_bytes!("fixtures/silk-mb-60ms-mono-20kbps.opus");

/// Map a fixture name to its embedded bytes.
fn fixture_bytes(name: &str) -> &'static [u8] {
    match name {
        "silk-nb-mono-16kbps" => FIXTURE_NB_MONO,
        "silk-wb-stereo-20kbps" => FIXTURE_WB_STEREO,
        "silk-mb-60ms-mono-20kbps" => FIXTURE_MB_60MS_MONO,
        other => panic!("unknown fixture {other}"),
    }
}

/// Recover the raw Opus packets from an Ogg-Opus byte stream.
///
/// Walks the Ogg pages (RFC 3533 page structure: a 27-byte fixed header
/// followed by an `nseg`-entry lacing table and the page body), gluing
/// segments into packets at every lacing value `< 255` (the standard Ogg
/// packet-termination convention). Returns every logical packet in stream
/// order; the caller drops the first two (the `OpusHead` and `OpusTags`
/// header packets per RFC 7845) to obtain the audio packets.
///
/// This is deliberately the smallest walker that recovers packets for the
/// well-formed single-stream fixtures; it is not a general Ogg demuxer.
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

/// The audio packets of a fixture (every logical packet after the two
/// RFC 7845 header packets).
fn fixture_audio_packets(name: &str) -> Vec<Vec<u8>> {
    let mut packets = ogg_packets(fixture_bytes(name));
    assert!(
        packets.len() >= 2,
        "{name}: expected at least OpusHead + OpusTags header packets"
    );
    // Drop OpusHead (packet 0) and OpusTags (packet 1).
    packets.drain(..2);
    packets
}

/// Decode every audio packet of `name` through one stateful
/// [`OpusDecoder`], concatenating the interleaved PCM. Panics on the first
/// hard decode error; returns the PCM, the channel count, and the total
/// per-channel 48 kHz sample count.
fn decode_fixture(name: &str) -> (Vec<i16>, u8, usize) {
    let packets = fixture_audio_packets(name);
    let mut dec = OpusDecoder::new();
    let mut pcm: Vec<i16> = Vec::new();
    let mut channels = 0u8;
    let mut per_channel_total = 0usize;
    for (i, pk) in packets.iter().enumerate() {
        let out = dec
            .decode_packet(pk)
            .unwrap_or_else(|e| panic!("{name}: packet {i} decode failed: {e:?}"));
        channels = out.channels;
        for fo in &out.frame_outcomes {
            assert!(
                !matches!(
                    fo.status,
                    FrameDecodeStatus::SilkDecodeError | FrameDecodeStatus::CeltDecodeError
                ),
                "{name}: packet {i} returned a decode-error status {:?}",
                fo.status
            );
            // SILK-only fixtures must take a SILK decode path, not the
            // CELT / unwired floor.
            assert!(
                matches!(
                    fo.status,
                    FrameDecodeStatus::SilkParamsDecoded | FrameDecodeStatus::SilkStereoDecoded
                ),
                "{name}: packet {i} took a non-SILK path: {:?}",
                fo.status
            );
            per_channel_total += fo.samples_per_channel;
        }
        pcm.extend_from_slice(&out.pcm);
    }
    assert_eq!(
        pcm.len(),
        per_channel_total * channels as usize,
        "{name}: interleaved PCM length disagrees with per-channel accounting"
    );
    (pcm, channels, per_channel_total)
}

/// Goertzel single-bin magnitude estimator: the per-sample magnitude of
/// frequency `f` (Hz) over `x` sampled at `sr` Hz.
fn goertzel_mag(x: &[i16], f: f64, sr: f64) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let w = 2.0 * std::f64::consts::PI * f / sr;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &v in x {
        let s0 = v as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).sqrt() / x.len() as f64
}

#[test]
fn silk_nb_mono_routes_and_decodes() {
    let name = "silk-nb-mono-16kbps";
    let packets = fixture_audio_packets(name);
    assert_eq!(packets.len(), 76, "{name}: packet count");

    // §3.1 TOC: config 1 ⇒ SILK / NB / 20 ms, mono (code 0 single frame).
    for pk in &packets {
        let toc = OpusTocByte::from_byte(pk[0]);
        assert_eq!(toc.mode, Mode::SilkOnly, "{name}: mode");
        assert_eq!(toc.bandwidth, Bandwidth::Nb, "{name}: bandwidth");
        assert_eq!(toc.channels, ChannelMapping::Mono, "{name}: mono");
    }

    let (pcm, channels, per_channel) = decode_fixture(name);
    assert_eq!(channels, 1);
    // 76 packets × one 20 ms frame × 960 samples/frame @ 48 kHz.
    assert_eq!(per_channel, 76 * 960);
    assert_eq!(pcm.len(), per_channel);

    // The fixture is a 440 Hz sine. Confirm the decoded output is
    // dominated by the 440 Hz bin relative to nearby probe frequencies,
    // over a steady mid-stream window (skip the leading transient /
    // pre-skip region). This is independent of the waveform gates in
    // `silk_reference_waveform.rs`: it checks the tone is present and
    // dominant, not exact
    // amplitude.
    let window = &pcm[10_000..40_000];
    let m440 = goertzel_mag(window, 440.0, 48_000.0);
    for off in [110.0, 220.0, 1320.0, 2000.0] {
        let other = goertzel_mag(window, off, 48_000.0);
        assert!(
            m440 > other * 3.0,
            "{name}: 440 Hz ({m440:.2}) not dominant over {off} Hz ({other:.2})"
        );
    }
    assert!(
        m440 > 1.0,
        "{name}: decoded 440 Hz energy too low ({m440:.2})"
    );
}

#[test]
fn silk_wb_stereo_routes_and_decodes() {
    let name = "silk-wb-stereo-20kbps";
    let packets = fixture_audio_packets(name);

    // §3.1 TOC: config 9 ⇒ SILK / WB / 20 ms, stereo.
    for pk in &packets {
        let toc = OpusTocByte::from_byte(pk[0]);
        assert_eq!(toc.mode, Mode::SilkOnly, "{name}: mode");
        assert_eq!(toc.bandwidth, Bandwidth::Wb, "{name}: bandwidth");
        assert_eq!(toc.channels, ChannelMapping::Stereo, "{name}: stereo");
    }

    let (pcm, channels, per_channel) = decode_fixture(name);
    assert_eq!(channels, 2);
    assert_eq!(per_channel, packets.len() * 960);
    assert_eq!(pcm.len(), per_channel * 2);
    // Stereo output must carry energy on both channels (the §4.2.8
    // mid/side unmix produced two distinct channels, not a mono copy in
    // one and silence in the other).
    let left_energy: u64 = pcm
        .iter()
        .step_by(2)
        .map(|&s| (s as i64 * s as i64) as u64)
        .sum();
    let right_energy: u64 = pcm
        .iter()
        .skip(1)
        .step_by(2)
        .map(|&s| (s as i64 * s as i64) as u64)
        .sum();
    assert!(left_energy > 0, "{name}: left channel is silent");
    assert!(right_energy > 0, "{name}: right channel is silent");
}

#[test]
fn silk_mb_60ms_mono_routes_and_decodes() {
    let name = "silk-mb-60ms-mono-20kbps";
    let packets = fixture_audio_packets(name);

    // §3.1 TOC: config 7 ⇒ SILK / MB / 60 ms (three 20 ms SILK frames),
    // mono. The 60 ms Opus frame is the largest SILK configuration and
    // exercises the multi-SILK-frame inter-frame state threading.
    for pk in &packets {
        let toc = OpusTocByte::from_byte(pk[0]);
        assert_eq!(toc.mode, Mode::SilkOnly, "{name}: mode");
        assert_eq!(toc.bandwidth, Bandwidth::Mb, "{name}: bandwidth");
        assert_eq!(toc.channels, ChannelMapping::Mono, "{name}: mono");
    }

    let (pcm, channels, per_channel) = decode_fixture(name);
    assert_eq!(channels, 1);
    // Each packet is a 60 ms Opus frame = 2880 samples/channel @ 48 kHz.
    assert_eq!(per_channel, packets.len() * 2880);
    assert_eq!(pcm.len(), per_channel);
    // The stream carries audio (non-silent output).
    let energy: u64 = pcm.iter().map(|&s| (s as i64 * s as i64) as u64).sum();
    assert!(energy > 0, "{name}: decoded output is entirely silent");
}
