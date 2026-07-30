//! `quality_search` — a first-cut quality-targeting Opus transcoder: decode the source **once**,
//! then re-encode it at a sweep of bitrates (libopus VBR), decode each back, and score the
//! reconstruction with the perceptual NMR metric — mapping the **rate-distortion curve** and
//! binary-searching the **lowest bitrate that hits a target quality**.
//!
//!   cargo run -p pf-opus --example quality_search -- [input] [target_nmr_db]
//!
//! `input` defaults to the FLAC fixture; `target_nmr_db` (default 0.0) is the "good enough" NMR
//! (lower = stricter; NMR < 0 ≈ transparent). The source is decoded to 48 kHz s16 via ffmpeg so any
//! input works. The metric is a first cut — NMR is *relative*; swap in a MOS-calibrated metric
//! (ViSQOL) later without changing the search. See `audio/PERCEPTUAL_QUALITY.md`.

#![allow(clippy::disallowed_methods, clippy::chunks_exact_to_as_chunks)] // one-shot tool

use std::process::Command;

use pf_opus::libopus::{LibopusDecoder, LibopusEncoder};
use pf_opus::EncoderConfig;
use profluens_audio::quality::{best_offset, rms_normalize, PerceptualAnalyzer};

const RATE: usize = 48_000;

fn main() {
    let input = std::env::args().nth(1).unwrap_or_else(|| "fixtures/out/audio.flac".into());
    let target_nmr: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let (bytes, channels) = decode_source(&input).unwrap_or_else(|| {
        eprintln!("could not decode {input} with ffmpeg (is it installed?)");
        std::process::exit(2);
    });
    let orig: Vec<i16> = to_i16(&bytes);
    let samples_per_ch = orig.len() / channels;
    let dur_s = samples_per_ch as f64 / RATE as f64;
    println!("source: {input}  {channels}ch  {dur_s:.1}s  (target NMR ≤ {target_nmr:.1} dB)\n");

    // --- 1. Rate-distortion sweep (the curve) ---
    let sweep = [24u32, 32, 48, 64, 96, 128, 160, 192, 256];
    println!("  target   actual     NMR dB   verdict");
    println!("  ------   ------   --------   -------");
    let mut curve: Vec<(u32, f64, f64)> = Vec::new(); // (target kbps, actual kbps, nmr)
    for &kbps in &sweep {
        let (nmr, actual) = eval(&bytes, &orig, channels, kbps * 1000);
        curve.push((kbps, actual, nmr));
        println!(
            "  {kbps:>4}k   {actual:>5.1}k   {nmr:>8.2}   {}",
            if nmr <= target_nmr { "✓ transparent" } else { "  audible" },
        );
    }

    // --- 2. Binary-search the lowest bitrate meeting the target, between the sweep brackets ---
    // NMR decreases monotonically with bitrate, so find the first sweep point that passes and
    // refine the interval below it.
    let passing = curve.iter().position(|&(_, _, nmr)| nmr <= target_nmr);
    println!();
    match passing {
        None => {
            let (bk, ak, nmr) = *curve.last().unwrap();
            println!(
                "target NMR ≤ {target_nmr:.1} dB not reached in range; best is {bk}k \
                 (actual {ak:.1}k) at NMR {nmr:.2} dB — raise the ceiling or relax the target."
            );
        }
        Some(0) => {
            let (bk, ak, nmr) = curve[0];
            println!("target met at the sweep floor: {bk}k (actual {ak:.1}k), NMR {nmr:.2} dB.");
        }
        Some(i) => {
            let mut lo = sweep[i - 1] * 1000; // fails
            let mut hi = sweep[i] * 1000; // passes
            let (mut best_kbps, mut best_actual, mut best_nmr) = (hi, curve[i].1, curve[i].2);
            while hi - lo > 4_000 {
                let mid = ((lo + hi) / 2 / 1000) * 1000;
                let (nmr, actual) = eval(&bytes, &orig, channels, mid);
                println!("  probe {:>4}k → actual {actual:>5.1}k  NMR {nmr:>7.2}", mid / 1000);
                if nmr <= target_nmr {
                    best_kbps = mid;
                    best_actual = actual;
                    best_nmr = nmr;
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            println!(
                "\n→ lowest bitrate for NMR ≤ {target_nmr:.1} dB: **{}k target / {:.1}k actual** \
                 (NMR {:.2} dB)",
                best_kbps / 1000,
                best_actual,
                best_nmr,
            );
        }
    }
}

/// Encode `orig_bytes` at `bitrate` (VBR, complexity 10), decode it back, and return
/// `(nmr_db, actual_kbps)`.
fn eval(orig_bytes: &[u8], orig: &[i16], channels: usize, bitrate: u32) -> (f64, f64) {
    let cfg = EncoderConfig { bitrate_bps: bitrate, ..EncoderConfig::new(channels as u8) };
    let mut enc = LibopusEncoder::new(cfg).expect("libopus encoder");
    let fb = enc.frame_bytes();

    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut total_bytes = 0usize;
    let mut pkt = Vec::new();
    let mut frame = vec![0u8; fb];
    for chunk in orig_bytes.chunks(fb) {
        frame[..chunk.len()].copy_from_slice(chunk);
        frame[chunk.len()..].fill(0); // zero-pad a trailing partial frame
        enc.encode(&frame, &mut pkt).expect("encode");
        total_bytes += pkt.len();
        packets.push(pkt.clone());
    }

    let mut dec = LibopusDecoder::new();
    dec.set_channels(channels);
    let mut recon: Vec<i16> = Vec::new();
    let mut dpcm = Vec::new();
    for p in &packets {
        if dec.decode_packet_into(p, &mut dpcm).is_ok() {
            recon.extend_from_slice(&dpcm);
        }
    }

    let samples_per_ch = orig.len() / channels;
    let dur_s = samples_per_ch as f64 / RATE as f64;
    let kbps = (total_bytes as f64 * 8.0) / dur_s / 1000.0;
    (nmr(orig, &recon, channels), kbps)
}

/// Perceptual NMR (dB) of `test` vs `reference` on channel 0, aligned + level-matched.
fn nmr(reference: &[i16], test: &[i16], channels: usize) -> f64 {
    let r = to_f32(&deinterleave(reference, channels));
    let t = to_f32(&deinterleave(test, channels));
    let off = best_offset(&r, &t, 4000).max(0) as usize;
    let t = &t[off.min(t.len())..];
    let m = r.len().min(t.len());
    let r = r[..m].to_vec();
    let mut t = t[..m].to_vec();
    rms_normalize(&r, &mut t);
    PerceptualAnalyzer::new(RATE as u32).analyze(&r, &t).nmr_db
}

fn deinterleave(interleaved: &[i16], channels: usize) -> Vec<i16> {
    interleaved.iter().step_by(channels).copied().collect() // channel 0
}
fn to_f32(v: &[i16]) -> Vec<f32> {
    v.iter().map(|&s| s as f32 / 32768.0).collect()
}
fn to_i16(bytes: &[u8]) -> Vec<i16> {
    bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

/// Decode `input` to interleaved 48 kHz s16 via ffmpeg; returns `(bytes, channels)`.
fn decode_source(input: &str) -> Option<(Vec<u8>, usize)> {
    // Channel count (clamped to stereo — the encoder handles 1–2).
    let ch = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
               "-of", "csv=p=0", input])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|c| c.clamp(1, 2))
        .unwrap_or(2);
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i", input, "-f", "s16le", "-ar", "48000", "-ac", &ch.to_string(), "-"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some((out.stdout, ch))
}
