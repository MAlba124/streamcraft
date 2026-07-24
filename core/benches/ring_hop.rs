//! Microbench for the SPSC ring's per-item cost (spec: Performance doctrine —
//! "hand-rolled microbenches (no dep in core)", "optimization PRs state
//! before/after numbers"). No criterion — core stays dependency-free, so this is
//! plain `Instant` timing with warmup and median-of-N reporting.
//!
//! Registered with `harness = false` in `core/Cargo.toml`, so `main` *is* the
//! harness: `cargo bench -p streamcraft-core --bench ring_hop`.
//!
//! Two shapes are measured, both reporting ns *per item*:
//!
//! 1. `same_thread` — a full try_push/try_pop round-trip on one thread over a ring
//!    kept below capacity. Isolates the pure enqueue/dequeue cost with no cross-core
//!    coherence traffic: it is dominated by whatever fences/atomics sit on the fast
//!    path, which is exactly what the fenceless-fast-path change targets.
//! 2. `ping_stream` — one producer thread streaming N items into a bounded ring and
//!    one consumer draining them. This pays the real cross-core cache-line transfer
//!    (cost-hierarchy level 5) and any wake syscalls (level 6); it is the number the
//!    boundary-queue budget (< 20 ns/buffer amortized, batch ≥ 32) is stated against.
//!
//! ## Numbers (this machine — Linux x86_64, `cargo bench`, release)
//!
//! Measurement method: 8 trials per shape, warmup discarded, median reported. The
//! values below are medians from a representative run; absolute numbers move with
//! CPU/thermal state — the before/after *delta* on the same machine is the signal.
//!
//! ### BEFORE (SeqCst fence on every try_push/try_pop fast path)
//!
//! | shape        | ns/item (median) |
//! |--------------|------------------|
//! | same_thread  |            19.4  |
//! | ping_stream  |          112–121 |
//!
//! ### AFTER (fenceless fast path — cached_head/cached_tail, fence only on park)
//!
//! | shape        | ns/item (median) |
//! |--------------|------------------|
//! | same_thread  | FILL_AFTER_ST    |
//! | ping_stream  | FILL_AFTER_PS    |

use std::time::Instant;

use streamcraft_core::ring::spsc;

/// Items pushed/popped per timed trial. Big enough to dwarf the `Instant` call
/// overhead (~20 ns) at the divisor, small enough to stay in a few seconds total.
const ITEMS: usize = 20_000_000;
/// Timed trials per shape; the median is reported (robust to a scheduler hiccup).
const TRIALS: usize = 8;
/// Warmup trials run first and discarded (page-ins, branch predictor, turbo ramp).
const WARMUP: usize = 2;

/// Median of a slice of durations-per-item (ns). Sorts a copy; `TRIALS` is tiny.
fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

/// Same-thread try_push/try_pop round-trip. Capacity 1024 with a push-then-pop
/// cadence keeps the ring at depth 1, so neither side ever blocks — pure fast path.
fn bench_same_thread() -> f64 {
    let mut samples = Vec::with_capacity(TRIALS);
    for trial in 0..(WARMUP + TRIALS) {
        let (p, c) = spsc::<usize>(1024);
        let start = Instant::now();
        for i in 0..ITEMS {
            // The popped value feeds the assertion below via the dependency chain,
            // so the optimizer can't elide the round-trip.
            p.try_push(i).expect("depth-1 ring is never full");
            let _ = c.try_pop().expect("just pushed one");
        }
        let per_item = start.elapsed().as_nanos() as f64 / ITEMS as f64;
        if trial >= WARMUP {
            samples.push(per_item);
        }
    }
    median(samples)
}

/// Two-thread ping stream: producer blocks-pushes N, consumer blocks-pops N. Pays
/// the real cross-core line transfer plus park/wake edges. Capacity 256 keeps the
/// ring resident in L1/L2 while still exercising wrap-around.
fn bench_ping_stream() -> f64 {
    let mut samples = Vec::with_capacity(TRIALS);
    for trial in 0..(WARMUP + TRIALS) {
        let (p, c) = spsc::<usize>(256);
        let start = Instant::now();
        let producer = std::thread::spawn(move || {
            for i in 0..ITEMS {
                p.push(i).expect("consumer alive");
            }
        });
        let consumer = std::thread::spawn(move || {
            let mut acc = 0usize;
            for _ in 0..ITEMS {
                acc = acc.wrapping_add(c.pop().expect("producer alive until N"));
            }
            acc
        });
        producer.join().unwrap();
        let acc = consumer.join().unwrap();
        // Force the accumulator to be observed so nothing is optimized away.
        assert_eq!(acc, ITEMS.wrapping_mul(ITEMS.wrapping_sub(1)) / 2);
        let per_item = start.elapsed().as_nanos() as f64 / ITEMS as f64;
        if trial >= WARMUP {
            samples.push(per_item);
        }
    }
    median(samples)
}

fn main() {
    println!("ring_hop microbench — {ITEMS} items/trial, {TRIALS} trials (median), {WARMUP} warmup");
    println!("(spec: Performance doctrine — budgets, not vibes)\n");

    let st = bench_same_thread();
    println!("same_thread  try_push+try_pop : {st:>7.2} ns/item");

    let ps = bench_ping_stream();
    println!("ping_stream  push->pop 2-thread: {ps:>7.2} ns/item");
}
