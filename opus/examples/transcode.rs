//! `transcode` — a FLAC→Opus demo CLI: pure-Rust FLAC decode (`pf-flac`) → reference **libopus**
//! encode → **Ogg-Opus** mux (`pf-ogg`), with optional **quality-targeted bitrate search**.
//!
//! ```text
//! cargo run -p pf-opus --example transcode -- in.flac out.opus [opts]
//! cargo run -p pf-opus --features qa --example transcode -- in.flac out.opus --target 4.0 --metric visqol
//! ```
//! Options:
//!   --bitrate <kbps>       fixed VBR bitrate (default 96)
//!   --target <q>           search the lowest bitrate meeting a quality target instead
//!   --metric <nmr|visqol>  target metric (default nmr; `visqol` needs `--features qa`)
//!   --complexity <0-10>    encoder complexity (default 10)
//!
//! With `--target`, a fast NMR sweep prints the rate-distortion curve, then a binary search homes
//! in on the target using the chosen metric — NMR (`≤` target, fast) or ViSQOL MOS-LQO (`≥` target,
//! calibrated but slow, so it scores only the first ~30 s). This is the hybrid: NMR maps the curve,
//! ViSQOL nails the MOS target.

#![allow(clippy::disallowed_methods, clippy::chunks_exact_to_as_chunks)] // one-shot CLI

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use pf_flac::FlacDecoder;
use pf_ogg::OggWriter;
use pf_opus::libopus::{LibopusDecoder, LibopusEncoder};
use pf_opus::EncoderConfig;
use profluens_audio::quality::PerceptualAnalyzer;
use profluens_audio::resample::{ChannelResampler, PolyphaseFilter};

const RATE: u32 = 48_000;
const FRAME_SAMPLES: usize = 960; // 20 ms @ 48 kHz

#[derive(Clone, Copy, PartialEq)]
enum Metric {
    Nmr,
    Visqol,
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!(
            "usage: transcode <in.flac> <out.opus> [--bitrate K | --target Q --metric nmr|visqol] \
             [--complexity C]"
        );
        std::process::exit(2);
    }
    let (input, output) = (&a[1], &a[2]);
    let mut bitrate = 96_000u32;
    let mut target: Option<f64> = None;
    let mut metric = Metric::Nmr;
    let mut complexity = 10u8;
    let mut i = 3;
    while i + 1 < a.len() {
        match a[i].as_str() {
            "--bitrate" => bitrate = a[i + 1].parse::<u32>().unwrap_or(96) * 1000,
            "--target" => target = a[i + 1].parse().ok(),
            "--metric" => metric = if a[i + 1] == "visqol" { Metric::Visqol } else { Metric::Nmr },
            "--complexity" => complexity = a[i + 1].parse().unwrap_or(10).min(10),
            x => {
                eprintln!("unknown option {x}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    // ViSQOL needs the `qa` feature; downgrade cleanly without it.
    #[cfg(not(feature = "qa"))]
    let metric = if matches!(metric, Metric::Visqol) {
        eprintln!("note: --metric visqol needs `--features qa`; using NMR");
        Metric::Nmr
    } else {
        metric
    };

    // 1. Decode the FLAC to 48 kHz s16 (pure-Rust decode + resample if needed).
    let (pcm, channels) = decode_flac_48k(input);
    let orig = to_i16(&pcm);
    let secs = orig.len() as f64 / channels as f64 / RATE as f64;
    let in_size = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    println!("in:  {input}  {channels}ch {secs:.1}s  {} KiB FLAC", in_size / 1024);

    // 2. Pick the bitrate: quality search or fixed.
    if let Some(t) = target {
        bitrate = search(&pcm, &orig, channels, metric, t, complexity);
    }

    // 3. Encode + mux to Ogg-Opus.
    let t0 = Instant::now();
    let packets = encode_all(&pcm, channels, bitrate, complexity);
    write_ogg_opus(output, channels, &packets);
    let out_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    println!(
        "out: {output}  {:.1} kbps  {} KiB Opus  ({:.0}% of source)  encoded {:.2?}",
        out_size as f64 * 8.0 / secs / 1000.0,
        out_size / 1024,
        out_size as f64 * 100.0 / in_size.max(1) as f64,
        t0.elapsed(),
    );
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode a FLAC file to interleaved 48 kHz `s16` bytes (first 1–2 channels), resampling if needed.
fn decode_flac_48k(path: &str) -> (Vec<u8>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| die(&format!("read {path}: {e}")));
    let dec = FlacDecoder::decode(&bytes).unwrap_or_else(|e| die(&format!("decode FLAC: {e:?}")));
    let total_ch = dec.info.channels as usize;
    let out_ch = total_ch.min(2);
    let bits = dec.info.bits_per_sample;
    let in_rate = dec.info.sample_rate;
    let n = dec.samples.len() / total_ch.max(1);
    let denom = (1i64 << (bits - 1)) as f32;

    // Deinterleave the first `out_ch` channels to f32 in [-1, 1].
    let mut planar: Vec<Vec<f32>> = (0..out_ch)
        .map(|c| (0..n).map(|i| dec.samples[i * total_ch + c] as f32 / denom).collect())
        .collect();

    if in_rate != RATE {
        let filter = PolyphaseFilter::design(in_rate, RATE, 32, 9.0);
        planar = planar
            .iter()
            .map(|lane| {
                let mut r = ChannelResampler::new(filter.clone());
                let mut out = Vec::new();
                r.process(lane, &mut out);
                out
            })
            .collect();
        println!("     resampled {in_rate} → {RATE} Hz");
    }

    let m = planar.iter().map(|l| l.len()).min().unwrap_or(0);
    let mut pcm = Vec::with_capacity(m * out_ch * 2);
    for i in 0..m {
        for lane in &planar {
            let s = (lane[i].clamp(-1.0, 1.0) * 32767.0) as i16;
            pcm.extend_from_slice(&s.to_le_bytes());
        }
    }
    (pcm, out_ch)
}

// ---------------------------------------------------------------------------
// Encode / decode
// ---------------------------------------------------------------------------

fn encode_all(pcm: &[u8], channels: usize, bitrate: u32, complexity: u8) -> Vec<Vec<u8>> {
    let cfg = EncoderConfig { bitrate_bps: bitrate, complexity, ..EncoderConfig::new(channels as u8) };
    let mut enc = LibopusEncoder::new(cfg).expect("libopus encoder");
    let fb = enc.frame_bytes();
    let mut packets = Vec::new();
    let mut pkt = Vec::new();
    let mut frame = vec![0u8; fb];
    for chunk in pcm.chunks(fb) {
        frame[..chunk.len()].copy_from_slice(chunk);
        frame[chunk.len()..].fill(0); // zero-pad a trailing partial frame
        enc.encode(&frame, &mut pkt).expect("encode");
        packets.push(pkt.clone());
    }
    packets
}

/// Encode at `bitrate`, decode back, return `(reconstruction_i16, actual_kbps)`.
fn enc_dec(pcm: &[u8], orig: &[i16], channels: usize, bitrate: u32, complexity: u8) -> (Vec<i16>, f64) {
    let packets = encode_all(pcm, channels, bitrate, complexity);
    let total: usize = packets.iter().map(|p| p.len()).sum();
    let mut dec = LibopusDecoder::new();
    dec.set_channels(channels);
    let mut recon = Vec::new();
    let mut p = Vec::new();
    for pk in &packets {
        if dec.decode_packet_into(pk, &mut p).is_ok() {
            recon.extend_from_slice(&p);
        }
    }
    let secs = orig.len() as f64 / channels as f64 / RATE as f64;
    (recon, total as f64 * 8.0 / secs / 1000.0)
}

// ---------------------------------------------------------------------------
// Quality search (hybrid: NMR sweep for the curve, chosen metric for the target)
// ---------------------------------------------------------------------------

fn search(pcm: &[u8], orig: &[i16], channels: usize, metric: Metric, target: f64, cx: u8) -> u32 {
    #[cfg(feature = "qa")]
    if metric == Metric::Visqol {
        ensure_model(); // write the model once, before the parallel section
    }

    // Each bitrate probe (encode → decode → score) is independent, so evaluate a whole grid in
    // parallel across cores — replacing a sequential binary search. Denser for cheap NMR, coarser
    // for the ~100× slower ViSQOL. A `tag` per ViSQOL call keeps its temp WAV paths thread-unique.
    let grid: Vec<u32> = match metric {
        Metric::Nmr => (24..=256).step_by(8).map(|k| k * 1000).collect(),
        Metric::Visqol => (32..=256).step_by(16).map(|k| k * 1000).collect(),
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(grid.len().max(1));
    // Work-queue rather than static `chunks(per)`: `chunks` splits N points into ceil(N/per)
    // pieces, spawning fewer threads than cores when N < 2·cores (15 points ÷ per=2 = 8 threads on
    // 12 cores) and load-imbalancing the tail. Instead spawn `threads` workers that each pull the
    // next grid index from a shared atomic until the grid is drained — every core stays busy and
    // the last, uneven items are picked up by whichever thread frees first.
    let grid_ref = &grid;
    let next = AtomicUsize::new(0);
    let counter = AtomicUsize::new(0);

    let t0 = Instant::now();
    let mut rows: Vec<(u32, f64, f64)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (next, counter) = (&next, &counter);
                s.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= grid_ref.len() {
                            break;
                        }
                        let k = grid_ref[i];
                        let (recon, actual) = enc_dec(pcm, orig, channels, k, cx);
                        let q = match metric {
                            Metric::Nmr => nmr(orig, &recon, channels),
                            Metric::Visqol => {
                                let tag = counter.fetch_add(1, Ordering::Relaxed);
                                visqol_mos(orig, &recon, channels, tag)
                            }
                        };
                        out.push((k, actual, q));
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    rows.sort_by_key(|r| r.0);

    let label = if metric == Metric::Visqol { "MOS" } else { "NMR" };
    let meets = |q: f64| match metric {
        Metric::Nmr => q <= target,
        Metric::Visqol => q >= target,
    };
    println!(
        "\n  rate-distortion sweep — {} points over {threads} threads in {:.1?}:",
        rows.len(),
        t0.elapsed(),
    );
    for &(k, actual, q) in &rows {
        println!(
            "   {:>4}k → {actual:>5.1}k   {label} {q:>7.2}  {}",
            k / 1000,
            if meets(q) { "✓" } else { "" },
        );
    }
    // Lowest grid bitrate meeting the target (rows are ascending).
    match rows.iter().find(|&&(_, _, q)| meets(q)) {
        Some(&(k, a, q)) => {
            println!("→ chosen: {}k target / {a:.1}k actual ({label} {q:.2})\n", k / 1000);
            k
        }
        None => {
            let last = *rows.last().expect("non-empty grid");
            println!("→ target not reached in range; using {}k\n", last.0 / 1000);
            last.0
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Perceptual NMR (dB) on channel 0, aligned + level-matched. Lower = better.
fn nmr(reference: &[i16], test: &[i16], channels: usize) -> f64 {
    let r = to_f32(&ch0(reference, channels, usize::MAX));
    let t = to_f32(&ch0(test, channels, usize::MAX));
    let off = profluens_audio::quality::best_offset(&r, &t, 4000).max(0) as usize;
    let t = &t[off.min(t.len())..];
    let m = r.len().min(t.len());
    let r = r[..m].to_vec();
    let mut t = t[..m].to_vec();
    profluens_audio::quality::rms_normalize(&r, &mut t);
    PerceptualAnalyzer::new(RATE).analyze(&r, &t).nmr_db
}

/// ViSQOL MOS-LQO (1–5), audio mode. Mono, first ~30 s (ViSQOL is slow + aligns internally, so no
/// pre-alignment needed). `tag` makes the temp WAV paths thread-unique for the parallel sweep.
/// Higher = better.
#[cfg(feature = "qa")]
fn visqol_mos(reference: &[i16], test: &[i16], channels: usize, tag: usize) -> f64 {
    use profluens_audio::format::{AudioFormat, SampleFormat};
    use visqol_rs::constants::{DEFAULT_WINDOW_SIZE, NUM_BANDS_AUDIO};
    use visqol_rs::variant::Variant;
    use visqol_rs::visqol_manager::VisqolManager;

    let cap = RATE as usize * 30;
    let fmt = AudioFormat::new(RATE, 1, SampleFormat::S16);
    let (rp, dp) = (format!("/tmp/sc_vq_ref_{tag}.wav"), format!("/tmp/sc_vq_deg_{tag}.wav"));
    std::fs::write(&rp, profluens_audio::wav::write_pcm_wav(&fmt, &i16_bytes(&ch0(reference, channels, cap))))
        .expect("write ref wav");
    std::fs::write(&dp, profluens_audio::wav::write_pcm_wav(&fmt, &i16_bytes(&ch0(test, channels, cap))))
        .expect("write deg wav");

    let mut v = VisqolManager::<NUM_BANDS_AUDIO>::new(
        Variant::Fullband { model_path: MODEL_PATH.to_string() },
        DEFAULT_WINDOW_SIZE,
    );
    // Profluens owns the bump arena; ViSQOL borrows it for the per-patch NSIM/convolution scratch
    // (reset between patches inside `run`). One arena per call — each parallel-search thread has
    // its own (`Arena` is `!Sync`); its chunks are reused across every patch of this comparison.
    let mut arena = profluens_core::memory::Arena::default();
    v.run(&rp, &dp, &mut arena).map(|r| r.moslqo).unwrap_or(1.0)
}

#[cfg(not(feature = "qa"))]
fn visqol_mos(_: &[i16], _: &[i16], _: usize, _tag: usize) -> f64 {
    unreachable!("ViSQOL requires --features qa")
}

/// Bundled ViSQOL audio SVR model, spilled to this path once (visqol-rs takes a file path).
#[cfg(feature = "qa")]
const MODEL_PATH: &str = "/tmp/sc_visqol_audio_model.txt";

#[cfg(feature = "qa")]
fn ensure_model() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        std::fs::write(MODEL_PATH, include_str!("../models/visqol_audio_model.txt")).expect("write model");
    }
}

// ---------------------------------------------------------------------------
// Ogg-Opus mux (RFC 7845) + small helpers
// ---------------------------------------------------------------------------

fn write_ogg_opus(path: &str, channels: usize, packets: &[Vec<u8>]) {
    const PRE_SKIP: u64 = 120;
    let mut w = OggWriter::new(0x5C0F_0007);
    let mut out = Vec::new();
    w.write_packet(&mut out, &opus_head(channels), 0).unwrap();
    w.flush(&mut out);
    w.write_packet(&mut out, &opus_tags(), 0).unwrap();
    w.flush(&mut out);
    for (k, p) in packets.iter().enumerate() {
        let granule = (k as u64 + 1) * FRAME_SAMPLES as u64 + PRE_SKIP;
        w.write_packet(&mut out, p, granule).unwrap();
    }
    w.finish(&mut out);
    std::fs::write(path, &out).unwrap_or_else(|e| die(&format!("write {path}: {e}")));
}

fn opus_head(channels: usize) -> Vec<u8> {
    let mut h = Vec::with_capacity(19);
    h.extend_from_slice(b"OpusHead");
    h.push(1);
    h.push(channels as u8);
    h.extend_from_slice(&120u16.to_le_bytes()); // pre-skip
    h.extend_from_slice(&RATE.to_le_bytes());
    h.extend_from_slice(&0u16.to_le_bytes()); // output gain
    h.push(0); // channel mapping family 0
    h
}

fn opus_tags() -> Vec<u8> {
    let vendor = b"streamcraft-transcode";
    let mut t = Vec::new();
    t.extend_from_slice(b"OpusTags");
    t.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    t.extend_from_slice(vendor);
    t.extend_from_slice(&0u32.to_le_bytes());
    t
}

fn ch0(interleaved: &[i16], channels: usize, cap: usize) -> Vec<i16> {
    interleaved.iter().step_by(channels).take(cap).copied().collect()
}
#[cfg(feature = "qa")]
fn i16_bytes(v: &[i16]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 2);
    for &s in v {
        b.extend_from_slice(&s.to_le_bytes());
    }
    b
}
fn to_i16(bytes: &[u8]) -> Vec<i16> {
    bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}
fn to_f32(v: &[i16]) -> Vec<f32> {
    v.iter().map(|&s| s as f32 / 32768.0).collect()
}
fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2)
}
