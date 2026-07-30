//! Hybrid (SILK low band + CELT high band) encode → decode roundtrips
//! through the crate's own streaming decoder.
//!
//! The two layers are aligned on one timeline (see
//! `hybrid_packet_encode`): the whole encode→decode chain delays by
//! exactly 120 samples at 48 kHz, verified here by an SNR gate at that
//! lag against the input.

use oxideav_opus::decoder::{FrameDecodeStatus, OpusDecoder};
use oxideav_opus::hybrid_packet_encode::HybridEncoderMono;
use oxideav_opus::toc::Bandwidth;

const DELAY: usize = 120;

/// Broadband deterministic signal: voice-band tones + high-band tones
/// crossing the 8 kHz layer split.
fn sig(i: usize) -> f64 {
    let t = i as f64 / 48000.0;
    6000.0 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
        + 3000.0 * (2.0 * std::f64::consts::PI * 880.0 * t + 0.4).sin()
        + 2000.0 * (2.0 * std::f64::consts::PI * 3300.0 * t + 1.0).sin()
        + 1500.0 * (2.0 * std::f64::consts::PI * 10500.0 * t + 0.2).sin()
        + 1000.0 * (2.0 * std::f64::consts::PI * 14700.0 * t + 2.0).sin()
}

fn roundtrip(bw: Bandwidth, tenths: u16, payload: usize, frames: usize) -> f64 {
    let mut enc = HybridEncoderMono::new(bw, tenths).unwrap();
    let spf = enc.frame_samples();
    let mut dec = OpusDecoder::new();
    let mut input: Vec<i16> = Vec::new();
    let mut decoded: Vec<i16> = Vec::new();
    for f in 0..frames {
        let pcm: Vec<i16> = (0..spf)
            .map(|j| sig(f * spf + j).round().clamp(-32768.0, 32767.0) as i16)
            .collect();
        let packet = enc.encode_packet(&pcm, payload).expect("encode");
        assert_eq!(packet.len(), 1 + payload);
        let out = dec.decode_packet(&packet).expect("decode");
        assert_eq!(out.samples_per_channel(), spf);
        assert_eq!(
            out.frame_outcomes[0].status,
            FrameDecodeStatus::HybridDecoded,
            "frame {f}"
        );
        input.extend_from_slice(&pcm);
        decoded.extend_from_slice(&out.pcm);
    }
    // Settled-region SNR at the fixed 120-sample delay.
    let start = decoded.len() / 4;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for p in (start + DELAY)..decoded.len() {
        let x = f64::from(input[p - DELAY]);
        let d = f64::from(decoded[p]) - x;
        num += d * d;
        den += x * x;
    }
    10.0 * (den / num.max(1e-30)).log10()
}

#[test]
fn hybrid_fb_20ms_roundtrips() {
    let snr = roundtrip(Bandwidth::Fb, 200, 240, 25);
    println!("hybrid fb 20ms 96kbps: {snr:.2} dB");
    assert!(snr > 9.0, "snr {snr}");
}

#[test]
fn hybrid_fb_20ms_high_rate_roundtrips() {
    let snr = roundtrip(Bandwidth::Fb, 200, 280, 25);
    println!("hybrid fb 20ms 112kbps: {snr:.2} dB");
    assert!(snr > 14.0, "snr {snr}");
}

#[test]
fn hybrid_swb_20ms_roundtrips() {
    let snr = roundtrip(Bandwidth::Swb, 200, 240, 25);
    println!("hybrid swb 20ms: {snr:.2} dB");
    assert!(snr > 10.0, "snr {snr}");
}

#[test]
fn hybrid_fb_10ms_roundtrips() {
    let snr = roundtrip(Bandwidth::Fb, 100, 150, 50);
    println!("hybrid fb 10ms: {snr:.2} dB");
    assert!(snr > 9.0, "snr {snr}");
}

#[test]
fn hybrid_low_band_only_content_leans_on_silk() {
    // Content entirely under 8 kHz: the SILK layer carries it and the
    // CELT bands stay near-silent; the frame must still decode as
    // Hybrid with reasonable fidelity.
    let mut enc = HybridEncoderMono::new(Bandwidth::Fb, 200).unwrap();
    let mut dec = OpusDecoder::new();
    let mut input: Vec<i16> = Vec::new();
    let mut decoded: Vec<i16> = Vec::new();
    for f in 0..20 {
        let pcm: Vec<i16> = (0..960)
            .map(|j| {
                let t = (f * 960 + j) as f64 / 48000.0;
                ((2.0 * std::f64::consts::PI * 330.0 * t).sin() * 9000.0) as i16
            })
            .collect();
        let packet = enc.encode_packet(&pcm, 220).unwrap();
        let out = dec.decode_packet(&packet).unwrap();
        assert_eq!(
            out.frame_outcomes[0].status,
            FrameDecodeStatus::HybridDecoded
        );
        input.extend_from_slice(&pcm);
        decoded.extend_from_slice(&out.pcm);
    }
    let start = decoded.len() / 4;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for p in (start + DELAY)..decoded.len() {
        let x = f64::from(input[p - DELAY]);
        let d = f64::from(decoded[p]) - x;
        num += d * d;
        den += x * x;
    }
    let snr = 10.0 * (den / num.max(1e-30)).log10();
    println!("hybrid low-band tone: {snr:.2} dB");
    assert!(snr > 10.0, "snr {snr}");
}

#[test]
fn hybrid_rejects_busted_budget() {
    // A payload too small for the (rate-uncontrolled) SILK layer must
    // error rather than emit a corrupt packet.
    let mut enc = HybridEncoderMono::new(Bandwidth::Fb, 200).unwrap();
    let pcm: Vec<i16> = (0..960)
        .map(|j| ((j as f64 * 0.05).sin() * 8000.0) as i16)
        .collect();
    assert!(enc.encode_packet(&pcm, 20).is_err());
}

#[test]
fn hybrid_elected_payload_honours_roomy_elections_exactly() {
    // When the election leaves room past the SILK floor, the packet
    // is exactly the elected size and decodes as Hybrid.
    let mut enc = HybridEncoderMono::new(Bandwidth::Fb, 200).unwrap();
    let spf = enc.frame_samples();
    let mut dec = OpusDecoder::new();
    for f in 0..10 {
        let pcm: Vec<i16> = (0..spf)
            .map(|j| sig(f * spf + j).round().clamp(-32768.0, 32767.0) as i16)
            .collect();
        let packet = enc.encode_packet_elected(&pcm, 400).expect("encode");
        assert_eq!(packet.len(), 401, "roomy election must be exact");
        let out = dec.decode_packet(&packet).expect("decode");
        assert_eq!(out.samples_per_channel(), spf);
        assert_eq!(
            out.frame_outcomes[0].status,
            FrameDecodeStatus::HybridDecoded
        );
    }
}

#[test]
fn hybrid_elected_payload_raises_starving_elections_to_the_floor() {
    // A 2-byte election cannot carry the SILK layer: the size is
    // raised to the floor instead of erroring, and every raised
    // packet still decodes as Hybrid with the exact sample count —
    // across all four Hybrid configs (SWB/FB × 10/20 ms).
    for bw in [Bandwidth::Swb, Bandwidth::Fb] {
        for tenths in [100u16, 200] {
            let mut enc = HybridEncoderMono::new(bw, tenths).unwrap();
            let spf = enc.frame_samples();
            let mut dec = OpusDecoder::new();
            for f in 0..10 {
                let pcm: Vec<i16> = (0..spf)
                    .map(|j| sig(f * spf + j).round().clamp(-32768.0, 32767.0) as i16)
                    .collect();
                let packet = enc.encode_packet_elected(&pcm, 2).expect("floor raise");
                assert!(packet.len() > 3, "floor packet is SILK-sized");
                assert!(packet.len() <= 1276);
                let out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(out.samples_per_channel(), spf);
                assert_eq!(
                    out.frame_outcomes[0].status,
                    FrameDecodeStatus::HybridDecoded,
                    "bw {bw:?} tenths {tenths} frame {f}"
                );
            }
        }
    }
}
