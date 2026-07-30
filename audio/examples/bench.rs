//! Rough polyphase-resampler throughput benchmark. Run in release:
//!   cargo run --release --example bench -p profluens-audio
//!
//! Resamples a few seconds of synthetic mono/stereo audio at the common rate conversions and
//! reports MB/s of *input* PCM (f32-domain, the resampler's working type) for each. This is the
//! "prove it at the highest level" number for the DSP (spec: First-party codecs — every rung
//! benchmarked). It also checks that a pure sine survives (dominant frequency preserved), so a
//! fast-but-wrong regression shows up here too.

use std::time::Instant;

use profluens_audio::{output_len, ChannelResampler, PolyphaseFilter};

const PI: f64 = std::f64::consts::PI;
const HALF_TAPS: usize = 32;
const BETA: f64 = 9.0;

/// Synthesise `n` mono f32 samples of a `freq`-Hz sine at `rate`.
fn synth_sine(freq: f64, rate: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.7 * (2.0 * PI * freq * i as f64 / rate as f64).sin()) as f32)
        .collect()
}

/// Single-bin DFT magnitude at `f` Hz (correctness probe).
fn mag_at(sig: &[f32], rate: u32, f: f64) -> f64 {
    let (mut re, mut im) = (0.0f64, 0.0f64);
    let w = 2.0 * PI * f / rate as f64;
    for (i, &s) in sig.iter().enumerate() {
        let ph = w * i as f64;
        re += s as f64 * ph.cos();
        im -= s as f64 * ph.sin();
    }
    (re * re + im * im).sqrt() / sig.len() as f64
}

fn bench(label: &str, in_rate: u32, out_rate: u32, channels: usize, seconds: u32) {
    let n = (in_rate * seconds) as usize;
    let freq = 1_000.0;
    // One shared filter design; one streaming resampler per channel.
    let filter = PolyphaseFilter::design(in_rate, out_rate, HALF_TAPS, BETA);
    let tpp = filter.taps_per_phase;

    // Per-channel input signals (all the same sine; enough to measure).
    let inputs: Vec<Vec<f32>> = (0..channels).map(|_| synth_sine(freq, in_rate, n)).collect();
    // Bytes of input PCM (f32 working domain) actually pushed through the filter.
    let in_bytes = (n * channels * std::mem::size_of::<f32>()) as f64;

    let iters = 5;
    let mut last_out: Vec<Vec<f32>> = Vec::new();
    let t = Instant::now();
    for _ in 0..iters {
        let mut resamplers: Vec<ChannelResampler> =
            (0..channels).map(|_| ChannelResampler::new(filter.clone())).collect();
        let mut outs: Vec<Vec<f32>> = (0..channels).map(|_| Vec::new()).collect();
        for (ch, r) in resamplers.iter_mut().enumerate() {
            r.process(&inputs[ch], &mut outs[ch]);
        }
        last_out = outs;
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;

    // Correctness: length ≈ ratio, and the 1 kHz tone survived on channel 0.
    let got = last_out[0].len() as i64;
    let want = output_len(n as u64, in_rate, out_rate) as i64;
    assert!((got - want).abs() <= 2, "{label}: len {got} vs ~{want}");
    let core = &last_out[0][tpp * 4..last_out[0].len() - tpp * 4];
    let peak = mag_at(core, out_rate, freq);
    let alias = mag_at(core, out_rate, 5_000.0);
    let clean = peak > 0.25 && alias < peak * 0.05;

    let mbps = in_bytes / dt / 1e6;
    let msamps = (n * channels) as f64 / dt / 1e6;
    println!(
        "{label:<20} {channels}ch {seconds}s : {mbps:8.1} MB/s  {msamps:7.1} Msamp/s   \
         L/M {}/{}  taps/phase {tpp}   [sine {}]",
        filter.l,
        filter.m,
        if clean { "ok" } else { "FAIL" },
    );
    assert!(clean, "{label}: sine not preserved (peak {peak}, alias {alias})");
}

fn main() {
    println!("audioresample — polyphase Kaiser-windowed-sinc throughput (input f32 PCM)\n");
    for &ch in &[1usize, 2] {
        bench("44100->48000 (up)", 44_100, 48_000, ch, 5);
        bench("48000->44100 (down)", 48_000, 44_100, ch, 5);
        bench("48000->24000 (÷2)", 48_000, 24_000, ch, 5);
        bench("8000->16000 (×2)", 8_000, 16_000, ch, 5);
    }
    println!(
        "\nNotes: single-threaded, f32 working type, {HALF_TAPS} taps/side Kaiser β={BETA}. \
         'up'/'down' are the awkward 147/160 ratios; the ÷2/×2 integer ratios are the cheap \
         cases (one polyphase branch on decimation)."
    );
}
