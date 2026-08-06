//! Hot-path allocation audit for the Opus decode path: a counting global allocator measures how
//! many heap allocations `decode_packet_into` performs per packet in steady state (after warmup).
//! oxideav-opus allocates ~3270 transient `Vec`s per packet internally; a scoped bump arena
//! cannot safely eliminate them (the decoder keeps inter-packet state a per-packet reset frees —
//! verified: SEGV). Reaching 0 needs a crate-internals scratch-reuse refactor.
//!
//!   cargo run -p pf-opus --example alloc_check -- /path/to/opus_testvectors

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

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
    let dir = std::env::args().nth(1).unwrap_or_else(|| "/tmp/opus_testvectors".into());
    // Measure the whole RFC 6716 vector set (SILK / CELT / hybrid, mono + stereo) so the count
    // reflects every decode path, not just one vector's.
    let mut total_steady = 0usize;
    let mut total_measured = 0usize;
    let mut worst_per_pkt = 0.0f64;
    let mut worst_vec = 0;
    for n in 1..=12 {
        let Ok(bytes) = std::fs::read(format!("{dir}/testvector{n:02}.bit")) else { continue };
        let packets = split_bit(&bytes);
        if packets.len() < 20 {
            continue;
        }
        let mut dec = oxideav_opus::OpusDecoder::new();
        let mut pcm: Vec<i16> = Vec::new();
        // Warm up (grow every reused scratch/pending buffer + map the bump region) before measuring.
        let warmup = (packets.len() / 4).min(400);
        for p in &packets[..warmup] {
            let _ = dec.decode_packet_into(p, &mut pcm);
        }
        let measure = &packets[warmup..];
        let before = ALLOCS.load(Ordering::Relaxed);
        for p in measure {
            let _ = dec.decode_packet_into(p, &mut pcm);
        }
        let steady = ALLOCS.load(Ordering::Relaxed) - before;
        let per = steady as f64 / measure.len() as f64;
        println!("vec{n:02}  {:>5} pkts measured  {:>7} allocs  {:.4}/pkt", measure.len(), steady, per);
        total_steady += steady;
        total_measured += measure.len();
        if per > worst_per_pkt {
            worst_per_pkt = per;
            worst_vec = n;
        }
    }

    println!("\ntotal: {total_steady} allocs over {total_measured} packets");
    println!("per packet (mean)         : {:.4}", total_steady as f64 / total_measured as f64);
    println!("per packet (worst vector) : {worst_per_pkt:.4} (vec{worst_vec:02})");
    if total_steady == 0 {
        println!("RESULT: entirely allocation-free on the decode hot path ✓");
    } else {
        println!("RESULT: NOT allocation-free — {total_steady} allocs over {total_measured} packets");
    }
}

fn split_bit(data: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut i = 0;
    while i + 8 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 8;
        if i + len > data.len() {
            break;
        }
        packets.push(data[i..i + len].to_vec());
        i += len;
    }
    packets
}
