//! Rough Ogg mux/demux throughput benchmark. Run in release:
//!   cargo run --release --example bench -p sc-ogg
//!
//! Muxes and demuxes a large synthetic packet stream and reports MB/s of container
//! payload for each direction, plus the framing overhead (page headers + lacing as a
//! fraction of the muxed size). This is the "prove it at the highest level" number for
//! the container (spec: First-party codecs — every rung benchmarked). It also checks
//! that demux recovers the exact packets, so a fast-but-wrong regression shows here.

use std::time::Instant;

use sc_ogg::{demux_all, mux_packets, OggReader};

/// SplitMix64 for reproducible packet contents.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Build a packet set whose sizes cluster around `avg` bytes (the §5 "expected case" is
/// 50–200 byte packets, but codecs like FLAC/Opus vary), totalling ~`total_bytes`.
fn make_packets(avg: usize, total_bytes: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut r = Rng(seed);
    let mut pkts = Vec::new();
    let mut acc = 0usize;
    while acc < total_bytes {
        // Size in [avg/2, avg*3/2], never zero.
        let jitter = (r.next_u64() as usize) % avg.max(1);
        let size = (avg / 2 + jitter).max(1);
        let data: Vec<u8> = (0..size).map(|_| r.next_u64() as u8).collect();
        acc += size;
        pkts.push(data);
    }
    pkts
}

fn bench_case(label: &str, avg: usize, total_mb: usize) {
    let total_bytes = total_mb * 1024 * 1024;
    let packets = make_packets(avg, total_bytes, 0xC0FFEE ^ avg as u64);
    let payload_bytes: usize = packets.iter().map(|p| p.len()).sum();
    let refs: Vec<(&[u8], u64)> = packets
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_slice(), i as u64))
        .collect();

    // --- Mux ---
    let iters = 5;
    let mut muxed = Vec::new();
    let t = Instant::now();
    for _ in 0..iters {
        muxed = mux_packets(1, &refs);
    }
    let mux_dt = t.elapsed().as_secs_f64() / iters as f64;

    // --- Demux (one-shot) ---
    let t = Instant::now();
    let mut recovered = Vec::new();
    for _ in 0..iters {
        recovered = demux_all(&muxed);
    }
    let demux_dt = t.elapsed().as_secs_f64() / iters as f64;

    // --- Demux (streaming, 8 KiB chunks — the realistic transport case) ---
    let t = Instant::now();
    for _ in 0..iters {
        let mut reader = OggReader::new();
        let mut n = 0usize;
        for chunk in muxed.chunks(8192) {
            reader.push(chunk);
            while reader.next_packet().is_some() {
                n += 1;
            }
        }
        reader.finish();
        while reader.next_packet().is_some() {
            n += 1;
        }
        std::hint::black_box(n);
    }
    let stream_dt = t.elapsed().as_secs_f64() / iters as f64;

    // Correctness: recovered packets equal the originals.
    assert_eq!(recovered.len(), packets.len(), "packet count");
    for (r, p) in recovered.iter().zip(&packets) {
        assert_eq!(&r.data, p, "payload mismatch");
    }

    let mux_mbps = payload_bytes as f64 / mux_dt / 1e6;
    let demux_mbps = payload_bytes as f64 / demux_dt / 1e6;
    let stream_mbps = payload_bytes as f64 / stream_dt / 1e6;
    let overhead = (muxed.len() as f64 - payload_bytes as f64) / muxed.len() as f64 * 100.0;

    println!(
        "{label:<18} avg {avg:>5}B  {n:>7} pkts  mux {mux_mbps:>8.0} MB/s  \
         demux {demux_mbps:>8.0} MB/s  stream8k {stream_mbps:>8.0} MB/s  \
         overhead {overhead:>5.2}%",
        n = packets.len(),
    );
}

fn main() {
    println!("Ogg (RFC 3533) mux/demux throughput — MB/s of container payload\n");
    // Small packets (§5's expected 50–200B case), medium, and large multi-page packets.
    bench_case("tiny (~64B)", 64, 64);
    bench_case("small (~150B)", 150, 128);
    bench_case("medium (~1KiB)", 1024, 256);
    bench_case("large (~16KiB)", 16 * 1024, 256);
    bench_case("huge (~256KiB)", 256 * 1024, 256);
    println!(
        "\nNotes: single logical stream, single-threaded. 'overhead' is page headers + \
         lacing as a fraction of the muxed stream; it falls as packets grow (§5: ~0.5% \
         for large packets, more for tiny ones)."
    );
}
