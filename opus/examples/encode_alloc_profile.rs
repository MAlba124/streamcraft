//! Backtrace-capturing allocation profiler for the encode hot path: over a small measured window
//! it captures a backtrace per allocation and aggregates by the innermost `oxideav_opus::` frame,
//! printing the hottest allocation call sites so the zero-alloc conversion can target them
//! biggest-first (the method the decoder's alloc work used).
//!
//!   cargo run -p pf-opus --example encode_alloc_profile

#![allow(clippy::disallowed_methods, clippy::manual_div_ceil)] // one-shot measurement harness

use std::alloc::{GlobalAlloc, Layout, System};
use std::backtrace::Backtrace;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use pf_opus::{EncoderConfig, OpusEncoder, OPUS_RATE};

static CAPTURE: AtomicBool = AtomicBool::new(false);
static SITES: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

thread_local! {
    static IN_HOOK: Cell<bool> = const { Cell::new(false) };
}

struct Profiler;
// SAFETY: delegates to System; the capture path is guarded against reentrancy (backtrace capture
// itself allocates) by the thread-local `IN_HOOK` flag.
unsafe impl GlobalAlloc for Profiler {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if CAPTURE.load(Ordering::Relaxed) {
            IN_HOOK.with(|f| {
                if !f.get() {
                    f.set(true);
                    record();
                    f.set(false);
                }
            });
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let np = System.realloc(p, l, new);
        if CAPTURE.load(Ordering::Relaxed) {
            IN_HOOK.with(|f| {
                if !f.get() {
                    f.set(true);
                    record();
                    f.set(false);
                }
            });
        }
        np
    }
}

/// Capture a backtrace and bump the count for its innermost `oxideav_opus::` frame.
fn record() {
    let bt = Backtrace::force_capture().to_string();
    let key = bt
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            // Lines look like "12: oxideav_opus::celt_pvq_encode::encode_pvq".
            let name = l.split_once(": ").map_or(l, |(_, r)| r);
            if name.starts_with("oxideav_opus::") && !name.contains("::bump::") {
                Some(name.to_string())
            } else {
                None
            }
        })
        .next()
        .unwrap_or_else(|| "<non-oxideav>".to_string());
    if let Ok(mut g) = SITES.lock() {
        *g.get_or_insert_with(HashMap::new).entry(key).or_insert(0) += 1;
    }
}

#[global_allocator]
static A: Profiler = Profiler;

#[allow(clippy::disallowed_methods)] // one-shot measurement harness
fn main() {
    let mut enc = OpusEncoder::new(EncoderConfig::new(2)).unwrap();
    let flen = enc.frame_len();
    let n = enc.frame_samples();
    let mut frame = vec![0i16; flen];
    let mut out = Vec::new();

    for k in 0..200 {
        fill(&mut frame, k * n, n);
        enc.encode_frame(&frame, &mut out).unwrap();
    }

    // Capture a handful of steady packets.
    const PKTS: usize = 5;
    CAPTURE.store(true, Ordering::Relaxed);
    for k in 0..PKTS {
        fill(&mut frame, (200 + k) * n, n);
        enc.encode_frame(&frame, &mut out).unwrap();
    }
    CAPTURE.store(false, Ordering::Relaxed);

    let g = SITES.lock().unwrap().take().unwrap_or_default();
    let mut sites: Vec<_> = g.into_iter().collect();
    sites.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    let total: usize = sites.iter().map(|(_, c)| *c).sum();
    println!("stereo encode — {} allocs over {PKTS} packets ({:.1}/pkt). Hottest sites:\n", total, total as f64 / PKTS as f64);
    for (site, count) in sites.iter().take(25) {
        println!("  {:>5}  ({:>5.1}/pkt)  {}", count, *count as f64 / PKTS as f64, site);
    }
}

fn fill(frame: &mut [i16], base: usize, n: usize) {
    for i in 0..n {
        let t = (base + i) as f64 / OPUS_RATE as f64;
        let s = (0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 16384.0) as i16;
        frame[i * 2] = s;
        frame[i * 2 + 1] = s;
    }
}
