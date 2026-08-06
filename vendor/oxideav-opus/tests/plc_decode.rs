//! Packet-loss concealment validation (RFC 6716 §4.4) on real Opus
//! streams: packets are dropped from the in-project fixtures and the
//! decoder's `conceal_loss` fills the gaps. The §4.4 acceptance
//! criteria are black-box signal properties — the output stays
//! continuous (no discontinuity artifacts at either join), the
//! concealment is real extrapolated audio (not silence) for a single
//! loss, and a burst of consecutive losses decays in energy to the
//! silence floor.
//!
//! The tiny Ogg page walker mirrors `silk_fixture_decode.rs` —
//! test-only fixture-loading scaffolding, not crate surface.

use oxideav_opus::{FrameDecodeStatus, OpusDecoder};

const FIXTURE_SILK_NB: &[u8] = include_bytes!("fixtures/silk-nb-mono-16kbps.opus");
const FIXTURE_CELT_FB: &[u8] = include_bytes!("fixtures/celt-fb-stereo-128kbps.opus");

/// Recover the raw Opus packets from an Ogg-Opus byte stream (RFC 3533
/// page walk; packets end at every lacing value < 255). Skips the two
/// header packets.
fn ogg_audio_packets(data: &[u8]) -> Vec<Vec<u8>> {
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
    packets.drain(..2);
    packets
}

/// Largest absolute sample-to-sample step per channel inside
/// `pcm[range]` of an interleaved buffer.
fn max_step(pcm: &[i16], channels: usize, from: usize, to: usize) -> i32 {
    let mut m = 0i32;
    for c in 0..channels {
        for i in (from.max(1))..to {
            let a = i32::from(pcm[i * channels + c]);
            let b = i32::from(pcm[(i - 1) * channels + c]);
            m = m.max((a - b).abs());
        }
    }
    m
}

fn energy(pcm: &[i16]) -> f64 {
    pcm.iter().map(|&s| f64::from(s) * f64::from(s)).sum()
}

/// Decode `packets`, dropping the packet at `lost_idx` and concealing
/// it. Returns (pcm, channels, per-packet sample counts, join sample
/// index of the concealed frame).
fn decode_with_loss(packets: &[Vec<u8>], lost_idx: usize) -> (Vec<i16>, usize, usize, usize) {
    let mut dec = OpusDecoder::new();
    let mut pcm: Vec<i16> = Vec::new();
    let mut channels = 1usize;
    let mut join_start = 0usize;
    let mut join_end = 0usize;
    for (i, pk) in packets.iter().enumerate() {
        if i == lost_idx {
            join_start = pcm.len();
            let out = dec.conceal_loss();
            assert_eq!(out.frame_outcomes.len(), 1);
            assert_eq!(out.frame_outcomes[0].status, FrameDecodeStatus::Concealed);
            channels = out.channels as usize;
            pcm.extend_from_slice(&out.pcm);
            join_end = pcm.len();
            continue;
        }
        let out = dec.decode_packet(pk).expect("decode");
        channels = out.channels as usize;
        pcm.extend_from_slice(&out.pcm);
    }
    (
        pcm,
        channels,
        join_start / channels.max(1),
        join_end / channels.max(1),
    )
}

/// A single mid-stream loss in the 440 Hz NB SILK stream: the
/// concealed frame is real (non-silent) extrapolated audio, both joins
/// are step-continuous, and the stream keeps decoding cleanly after.
#[test]
fn silk_single_loss_conceals_continuously() {
    let packets = ogg_audio_packets(FIXTURE_SILK_NB);
    assert!(packets.len() > 20);
    let lost = packets.len() / 2;
    let (pcm, channels, j0, j1) = decode_with_loss(&packets, lost);
    assert_eq!(channels, 1);

    // Real audio: the concealed frame carries energy comparable to its
    // neighbourhood (not silence, not an explosion).
    let frame = j1 - j0;
    let concealed = energy(&pcm[j0 * channels..j1 * channels]);
    let before = energy(&pcm[(j0 - frame) * channels..j0 * channels]);
    assert!(
        concealed > before * 0.05,
        "concealment near-silent: {concealed} vs {before}"
    );
    assert!(
        concealed < before * 4.0,
        "concealment overshoots: {concealed} vs {before}"
    );

    // Continuity: the steps across both joins are bounded by a small
    // multiple of the natural intra-signal step around them.
    let natural = max_step(&pcm, channels, j0 - 480, j0 - 1).max(1);
    let at_loss_join = max_step(&pcm, channels, j0 - 2, j0 + 2);
    let at_resume_join = max_step(&pcm, channels, j1 - 2, j1 + 2);
    assert!(
        at_loss_join <= natural * 4 + 64,
        "loss join step {at_loss_join} vs natural {natural}"
    );
    assert!(
        at_resume_join <= natural * 4 + 64,
        "resume join step {at_resume_join} vs natural {natural}"
    );
}

/// A single mid-stream loss in the CELT FB stereo music stream: same
/// §4.4 criteria on the pitch-repetition flavor.
#[test]
fn celt_single_loss_conceals_continuously() {
    let packets = ogg_audio_packets(FIXTURE_CELT_FB);
    assert!(packets.len() > 20);
    let lost = packets.len() / 2;
    let (pcm, channels, j0, j1) = decode_with_loss(&packets, lost);
    assert_eq!(channels, 2);

    let frame = j1 - j0;
    let concealed = energy(&pcm[j0 * channels..j1 * channels]);
    let before = energy(&pcm[(j0 - frame) * channels..j0 * channels]);
    assert!(
        concealed > before * 0.02,
        "concealment near-silent: {concealed} vs {before}"
    );
    assert!(
        concealed < before * 6.0,
        "concealment overshoots: {concealed} vs {before}"
    );

    // Music content has larger natural steps; the joins must stay
    // within the neighbourhood's own dynamics.
    let natural = max_step(&pcm, channels, j0 - 960, j0 - 1).max(1);
    let at_loss_join = max_step(&pcm, channels, j0 - 2, j0 + 2);
    let at_resume_join = max_step(&pcm, channels, j1 - 2, j1 + 2);
    assert!(
        at_loss_join <= natural * 4 + 64,
        "loss join step {at_loss_join} vs natural {natural}"
    );
    assert!(
        at_resume_join <= natural * 4 + 64,
        "resume join step {at_resume_join} vs natural {natural}"
    );
}

/// A burst of consecutive losses: the §4.4 energy decay drives the
/// concealment monotonically (allowing small ripple) to the silence
/// floor, and the first real packet after the burst decodes cleanly
/// with a smooth cross-lapped join.
#[test]
fn burst_loss_decays_to_silence_and_recovers() {
    let packets = ogg_audio_packets(FIXTURE_SILK_NB);
    let mut dec = OpusDecoder::new();
    let warmup = 10.min(packets.len() - 2);
    for pk in &packets[..warmup] {
        let _ = dec.decode_packet(pk).expect("decode");
    }

    let mut energies = Vec::new();
    let mut frame_len = 0usize;
    for _ in 0..30 {
        let out = dec.conceal_loss();
        frame_len = out.samples_per_channel();
        energies.push(energy(&out.pcm));
    }
    assert!(frame_len > 0);
    assert!(energies[0] > 0.0, "first concealed frame silent");
    for w in energies.windows(2).skip(1) {
        assert!(
            w[1] <= w[0] * 1.05,
            "energy grew across the loss burst: {energies:?}"
        );
    }
    assert!(
        energies[energies.len() - 1] < energies[0] * 1e-4,
        "burst did not decay toward silence: first {} last {}",
        energies[0],
        energies[energies.len() - 1]
    );

    // Recovery: the next real packet decodes as usual.
    let out = dec.decode_packet(&packets[warmup]).expect("decode");
    assert!(matches!(
        out.frame_outcomes[0].status,
        FrameDecodeStatus::SilkParamsDecoded | FrameDecodeStatus::SilkDecodeError
    ));
    assert_eq!(
        out.frame_outcomes[0].status,
        FrameDecodeStatus::SilkParamsDecoded
    );
}

/// The concealed duration mirrors the last decoded packet's frame
/// duration, and concealment with no history at all is 20 ms of
/// silence.
#[test]
fn concealment_duration_follows_the_stream() {
    // No history: 20 ms mono silence.
    let mut fresh = OpusDecoder::new();
    let out = fresh.conceal_loss();
    assert_eq!(out.channels, 1);
    assert_eq!(out.samples_per_channel(), 960);
    assert!(out.pcm.iter().all(|&s| s == 0));

    // After a 20 ms stereo CELT packet: 20 ms stereo concealment.
    let packets = ogg_audio_packets(FIXTURE_CELT_FB);
    let mut dec = OpusDecoder::new();
    let out = dec.decode_packet(&packets[0]).expect("decode");
    let per = out.samples_per_channel();
    let concealed = dec.conceal_loss();
    assert_eq!(concealed.channels, 2);
    assert_eq!(concealed.samples_per_channel(), per);
}

/// The 440 Hz NB sine fixture concealment stays 440 Hz-dominant — the
/// §4.4 extrapolation continues the signal's harmonic structure rather
/// than injecting unrelated content.
#[test]
fn silk_concealment_preserves_the_tone() {
    let packets = ogg_audio_packets(FIXTURE_SILK_NB);
    let mut dec = OpusDecoder::new();
    let warmup = 12.min(packets.len() - 1);
    for pk in &packets[..warmup] {
        let _ = dec.decode_packet(pk).expect("decode");
    }
    let out = dec.conceal_loss();
    let x: Vec<f64> = out.pcm.iter().map(|&s| f64::from(s)).collect();

    // Goertzel probe at 440 Hz vs a probe away from the tone.
    let goertzel = |f: f64| {
        let w = 2.0 * std::f64::consts::PI * f / 48_000.0;
        let c = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &v in &x {
            let s0 = v + c * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        s1 * s1 + s2 * s2 - c * s1 * s2
    };
    let tone = goertzel(440.0);
    let off = goertzel(1300.0).max(goertzel(2200.0));
    assert!(
        tone > off * 10.0,
        "concealment lost the 440 Hz dominance: tone {tone:.1} off {off:.1}"
    );
}
