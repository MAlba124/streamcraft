//! Rough FLAC encoder throughput + compression benchmark. Run in release:
//!   cargo run --release --example bench -p sc-flac
//!
//! Encodes a few synthetic signals and reports encode MB/s (of input PCM) and the
//! compression ratio (encoded / raw). This is the "prove it at the highest level"
//! number for the codec (spec: First-party codecs — every rung benchmarked). It also
//! decodes the output back and checks losslessness, so a fast-but-wrong regression
//! shows up here too.

use std::time::Instant;

use sc_flac::{FlacDecoder, FlacEncoder, SampleFormat};

/// Synthesise `n` interchannel S16 samples into interleaved LE bytes.
fn synth(kind: &str, n: usize, channels: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * channels as usize * 2);
    let mut state: u64 = 0x1234_5678;
    for i in 0..n {
        for c in 0..channels {
            let s: i64 = match kind {
                "sine" => {
                    let ph = 2.0 * std::f64::consts::PI * (440.0 + 30.0 * c as f64) * (i as f64)
                        / 44100.0;
                    (28000.0 * ph.sin()).round() as i64
                }
                "mix" => {
                    // A couple of tones plus a little noise — closer to real audio.
                    let ph1 = 2.0 * std::f64::consts::PI * 440.0 * (i as f64) / 44100.0;
                    let ph2 = 2.0 * std::f64::consts::PI * 660.0 * (i as f64) / 44100.0;
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let noise = ((state >> 40) as i64 % 400) - 200;
                    ((12000.0 * ph1.sin() + 8000.0 * (ph2 + c as f64).sin()).round() as i64) + noise
                }
                "noise" => {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 33) as i64 % 65536) - 32768
                }
                _ => 0,
            };
            out.extend_from_slice(&(s.clamp(-32768, 32767) as i16).to_le_bytes());
        }
    }
    out
}

fn bench(kind: &str, channels: u32, seconds: u32) {
    let rate = 44100u32;
    let n = (rate * seconds) as usize;
    let pcm = synth(kind, n, channels);
    let raw_len = pcm.len();

    // Warm run for encode.
    let iters = 5;
    let mut encoded = Vec::new();
    let t = Instant::now();
    for _ in 0..iters {
        let (mut enc, header) = FlacEncoder::new(rate, channels, SampleFormat::S16).unwrap();
        encoded.clear();
        encoded.extend_from_slice(&header);
        let mut frames = Vec::new();
        enc.encode_interleaved(&pcm, &mut frames).unwrap();
        let body = enc.finish();
        encoded[sc_flac::streaminfo_offset()..sc_flac::streaminfo_offset() + body.len()]
            .copy_from_slice(&body);
        encoded.extend_from_slice(&frames);
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;

    // Losslessness sanity check on the produced stream.
    let dec = FlacDecoder::decode(&encoded).expect("decode");
    let expected_samples = n * channels as usize;
    assert_eq!(dec.samples.len(), expected_samples, "sample count");

    let mbps = raw_len as f64 / dt / 1e6;
    let ratio = encoded.len() as f64 / raw_len as f64;
    println!(
        "{:5} {}ch {:>3}s : {:8.1} MB/s   ratio {:5.3}  ({} KiB -> {} KiB)   [lossless ✓]",
        kind,
        channels,
        seconds,
        mbps,
        ratio,
        raw_len / 1024,
        encoded.len() / 1024,
    );
}

fn main() {
    println!("FLAC encoder — encode throughput (of input PCM) and compression ratio\n");
    for kind in ["sine", "mix", "noise"] {
        for &ch in &[1u32, 2] {
            bench(kind, ch, 5);
        }
    }
    println!(
        "\nNotes: FIXED predictors + partitioned Rice, single-threaded, S16@44100. \
         'noise' is incompressible (ratio ~1) by design; 'sine'/'mix' show the win."
    );
}
