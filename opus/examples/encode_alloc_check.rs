//! Hot-path allocation audit for the Opus **encode** path — the encoder mirror of
//! `examples/alloc_check.rs`. A counting global allocator measures how many heap allocations
//! `OpusEncoder::encode_frame` performs per packet in steady state (after warmup). The goal, as on
//! the decode side, is 0 in steady state; this harness tracks progress toward it.
//!
//!   cargo run -p pf-opus --example encode_alloc_check

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use pf_opus::{EncoderConfig, OpusEncoder, OPUS_RATE};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;
// SAFETY: delegates to the System allocator; only bumps a relaxed counter.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.realloc(p, l, new)
    }
}

#[global_allocator]
static A: Counting = Counting;

#[allow(clippy::disallowed_methods)] // one-shot measurement harness
fn main() {
    for &channels in &[1u8, 2] {
        let mut enc = OpusEncoder::new(EncoderConfig::new(channels)).unwrap();
        let flen = enc.frame_len();
        let n = enc.frame_samples();

        // A steady tone so every frame does the same work (transients/silence take different paths).
        let mut frame = vec![0i16; flen];
        let ch = channels as usize;

        let mut out = Vec::new();
        // Warm up: grow the output buffer + every reused internal scratch before measuring.
        for k in 0..200 {
            fill(&mut frame, k * n, n, ch);
            enc.encode_frame(&frame, &mut out).unwrap();
        }

        let measure = 500;
        let before = ALLOCS.load(Ordering::Relaxed);
        for k in 0..measure {
            fill(&mut frame, (200 + k) * n, n, ch);
            enc.encode_frame(&frame, &mut out).unwrap();
        }
        let steady = ALLOCS.load(Ordering::Relaxed) - before;
        println!(
            "{}ch  {:>5} packets measured  {:>7} allocs  {:.3}/pkt",
            channels,
            measure,
            steady,
            steady as f64 / measure as f64,
        );
    }
    println!("\n(goal: 0/pkt in steady state — the decode path reached it; see alloc_check.rs)");
}

/// Fill `frame` with a 440 Hz tone starting at sample `base`, replicated across `ch` channels.
fn fill(frame: &mut [i16], base: usize, n: usize, ch: usize) {
    for i in 0..n {
        let t = (base + i) as f64 / OPUS_RATE as f64;
        let s = (0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 16384.0) as i16;
        for c in 0..ch {
            frame[i * ch + c] = s;
        }
    }
}
