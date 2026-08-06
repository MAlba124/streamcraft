//! Rough audio-DSP throughput benchmark. Run in release:
//!   cargo run --release --example bench -p profluens-audio [resample|convert|downmix|all]
//!
//! Three rungs, all reporting MB/s of *input* PCM (spec: First-party codecs — every rung
//! benchmarked):
//!
//! * **resample** — a few seconds of synthetic mono/stereo audio at the common rate conversions,
//!   in the f32 working domain. Also checks that a pure sine survives (dominant frequency
//!   preserved, no alias image), so a fast-but-wrong regression shows up here too.
//! * **convert** — [`convert_interleaved`] over the sample-format pairs the decoders and sinks
//!   actually ask for, plus the stereo→mono remap.
//! * **downmix** — the ITU-R BS.775 5.1→stereo fold the player uses for AC-3/E-AC-3.

use std::time::Instant;

use profluens_audio::{
    convert_interleaved, downmix_to_stereo, output_len, remap_channels, ChannelResampler,
    PolyphaseFilter, SampleFormat,
};

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

/// Pseudo-random interleaved PCM, `bytes` long — a deterministic xorshift so runs are
/// comparable, and non-degenerate so no branch predicts perfectly.
fn synth_pcm(bytes: usize) -> Vec<u8> {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    (0..bytes)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 33) as u8
        })
        .collect()
}

/// Time `op` over `input` and print its input-MB/s. `iters` passes; the fastest wins (the loop is
/// deterministic, so the spread is scheduler and frequency noise).
fn bench_bytes(label: &str, input: &[u8], out_len: usize, mut op: impl FnMut(&[u8], &mut [u8])) {
    let mut out = vec![0u8; out_len];
    let iters = 20;
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        op(input, &mut out);
        best = best.min(t.elapsed().as_secs_f64());
    }
    // Keep the result alive so nothing is optimised away.
    std::hint::black_box(&out);
    println!("{label:<28} {:8.1} MB/s", input.len() as f64 / best / 1e6);
}

/// ~8 MiB of source PCM per case: past L2, small enough to stay quick.
const BYTES: usize = 8 << 20;

fn convert_benches() {
    println!("\naudioconvert — interleaved sample-format conversion (input PCM)\n");
    let src = synth_pcm(BYTES);
    for (from, to) in [
        (SampleFormat::S16, SampleFormat::F32),
        (SampleFormat::F32, SampleFormat::S16),
        (SampleFormat::S24, SampleFormat::F32),
        (SampleFormat::S32, SampleFormat::S16),
        (SampleFormat::S16, SampleFormat::S32),
        (SampleFormat::U8, SampleFormat::S16),
    ] {
        let n = (BYTES / (from.bytes() * 2)) * from.bytes() * 2; // whole stereo frames
        let out_len = (n / from.bytes()) * to.bytes();
        let label = format!("{from:?}->{to:?} 2ch");
        bench_bytes(&label, &src[..n], out_len, |i, o| {
            convert_interleaved(from, to, 2, i, o).expect("convert");
        });
    }
    let n = (BYTES / 4) * 4;
    bench_bytes("S16 stereo->mono", &src[..n], n / 2, |i, o| {
        remap_channels(SampleFormat::S16, 2, 1, i, o).expect("remap");
    });
}

fn downmix_benches() {
    println!("\naudiodownmix — ITU-R BS.775 fold to stereo (input PCM)\n");
    let src = synth_pcm(BYTES);
    for (fmt, ch) in [(SampleFormat::S16, 6usize), (SampleFormat::F32, 6), (SampleFormat::S16, 8)] {
        let stride = fmt.bytes() * ch;
        let n = (BYTES / stride) * stride;
        let out_len = (n / stride) * fmt.bytes() * 2;
        let label = format!("{fmt:?} {ch}ch -> 2ch");
        bench_bytes(&label, &src[..n], out_len, |i, o| {
            downmix_to_stereo(fmt, ch, i, o).expect("downmix");
        });
    }
}

fn resample_benches() {
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

/// `bench [resample|convert|downmix|all]` — one phase at a time so an A/B run can wrap
/// `perf stat`'s deterministic `instructions:u` around exactly the code under test.
fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if matches!(which.as_str(), "all" | "resample") {
        resample_benches();
    }
    if matches!(which.as_str(), "all" | "convert") {
        convert_benches();
    }
    if matches!(which.as_str(), "all" | "downmix") {
        downmix_benches();
    }
}
