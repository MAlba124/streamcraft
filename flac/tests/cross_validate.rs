//! Cross-validation against the reference decoder (spec: First-party codecs —
//! "cross-checked against a reference decode", a dev-only out-of-process oracle,
//! never linked in). If the `flac` CLI (libFLAC) is on `PATH`, encode a signal with
//! our encoder, run `flac -t` (verify) and `flac -d` (decode to WAV) on it, and
//! confirm libFLAC both accepts the stream and recovers the exact samples. When `flac`
//! is absent the test no-ops with a printed note — the in-crate round-trip
//! (`roundtrip.rs`) is the real gate; this is the extra net.

use std::io::Write;
use std::process::Command;

use pf_flac::{FlacEncoder, SampleFormat};

fn have_flac() -> bool {
    Command::new("flac")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf_flac_xv_{}_{}.{}", tag, std::process::id(), ext));
    p
}

/// Encode `n` interchannel S16 samples of a couple of sines into a complete FLAC file
/// (with finalised STREAMINFO) and return the path plus the interleaved i64 samples.
fn write_flac(tag: &str, n: usize, channels: u32, rate: u32) -> (std::path::PathBuf, Vec<i64>) {
    let mut interleaved = Vec::new();
    let mut expected = Vec::new();
    for i in 0..n {
        for c in 0..channels {
            let ph = 2.0 * std::f64::consts::PI * (300.0 + 55.0 * c as f64) * (i as f64) / 512.0;
            let s = (15000.0 * ph.sin()).round() as i64;
            let s = s.clamp(-32768, 32767);
            expected.push(s);
            interleaved.extend_from_slice(&(s as i16).to_le_bytes());
        }
    }
    let (mut enc, mut header) = FlacEncoder::new(rate, channels, SampleFormat::S16).unwrap();
    let mut frames = Vec::new();
    enc.encode_interleaved(&interleaved, &mut frames).unwrap();
    let body = enc.finish();
    header[pf_flac::streaminfo_offset()..pf_flac::streaminfo_offset() + body.len()]
        .copy_from_slice(&body);

    let path = temp_path(tag, "flac");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&header).unwrap();
    f.write_all(&frames).unwrap();
    (path, expected)
}

/// Parse a canonical PCM WAV (the shape `flac -d` writes) into interleaved i64 S16.
fn read_wav_s16(bytes: &[u8]) -> Vec<i64> {
    // Walk RIFF chunks to find `data`; assume 16-bit PCM as we asked flac to decode.
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut pos = 12;
    let mut data = &bytes[0..0];
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]])
            as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        if id == b"data" {
            data = body;
            break;
        }
        pos += 8 + size + (size & 1); // chunks are word-aligned
    }
    data.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as i64)
        .collect()
}

#[test]
fn libflac_accepts_and_losslessly_decodes_our_output() {
    if !have_flac() {
        eprintln!("cross_validate: `flac` CLI not on PATH — skipping (in-crate round-trip is the gate)");
        return;
    }

    for &(n, ch, rate) in &[(4096usize, 1u32, 44100u32), (5000, 2, 48000), (100, 2, 44100)] {
        let (flac_path, expected) = write_flac("xv", n, ch, rate);

        // 1. `flac -t`: libFLAC verifies the stream (sync, CRCs, MD5-if-present).
        let test = Command::new("flac")
            .arg("-t")
            .arg("-s")
            .arg(&flac_path)
            .output()
            .expect("run flac -t");
        assert!(
            test.status.success(),
            "flac -t rejected our output (n={n} ch={ch}): {}",
            String::from_utf8_lossy(&test.stderr)
        );

        // 2. `flac -d`: decode to WAV and compare samples exactly against our input.
        let wav_path = temp_path("xv", "wav");
        let dec = Command::new("flac")
            .arg("-d")
            .arg("-s")
            .arg("-f")
            .arg("-o")
            .arg(&wav_path)
            .arg(&flac_path)
            .output()
            .expect("run flac -d");
        assert!(
            dec.status.success(),
            "flac -d failed (n={n} ch={ch}): {}",
            String::from_utf8_lossy(&dec.stderr)
        );
        let wav = std::fs::read(&wav_path).expect("read decoded wav");
        let got = read_wav_s16(&wav);
        assert_eq!(
            got, expected,
            "libFLAC decode differs from our input (n={n} ch={ch})"
        );

        let _ = std::fs::remove_file(&flac_path);
        let _ = std::fs::remove_file(&wav_path);
    }
}
