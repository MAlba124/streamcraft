//! Steady-state allocation audit for the **built-in elements**, one pipeline shape at a time,
//! through the real scheduler with a counting global allocator.
//!
//!   cargo run -p profluens-elements --release --example alloc_check
//!   PF_ALLOC_PROFILE=1 PF_ALLOC_SAMPLE=1 cargo run -p profluens-elements --release \
//!       --example alloc_check -- tee
//!
//! Each shape reports allocations per buffer *crossing the element under test* over the whole run
//! (startup amortized by running a long enough stream). Shapes are additive: subtracting the
//! `testsrc ! testsink` baseline from `testsrc ! X ! testsink` isolates X's own cost.

#![allow(clippy::disallowed_methods)] // one-shot measurement harness

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;
use profluens_elements::flow::{PassThrough, Queue, Tee};
use profluens_elements::io::{FileSink, FileSrc};
use profluens_elements::testing::{TestSink, TestSrc};

// A sink that *mutates* what it receives — the stand-in for any in-place transform. Behind a
// `tee` the payload is shared, so `as_mut_full()` takes `Memory`'s copy-on-write branch.
static MUT_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static MUT_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &MUT_OFFERS,
    dynamic: false,
    validate: None,
}];
static MUT_DESC: ElementDesc = ElementDesc {
    name: "mutsink",
    pads: &MUT_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};
struct MutSink;
impl Element for MutSink {
    fn desc(&self) -> &'static ElementDesc {
        &MUT_DESC
    }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _c: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(mut buf) = inputs.pop() {
            buf.memory.as_mut_full()[0] ^= 1;
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _c: &mut Ctx) {}
}

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static SAMPLING: AtomicBool = AtomicBool::new(false);
static SAMPLE_EVERY: AtomicUsize = AtomicUsize::new(1);
static STACKS: Mutex<Option<HashMap<String, (usize, usize)>>> = Mutex::new(None);

thread_local! {
    static IN_SAMPLER: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

impl Counting {
    fn note(size: usize) {
        let _ = IN_SAMPLER.try_with(|g| {
            if g.get() {
                return;
            }
            g.set(true);
            let n = ALLOCS.fetch_add(1, Ordering::Relaxed) + 1;
            BYTES.fetch_add(size, Ordering::Relaxed);
            PEAK.fetch_max(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
            if SAMPLING.load(Ordering::Relaxed)
                && n.is_multiple_of(SAMPLE_EVERY.load(Ordering::Relaxed))
            {
                let key = capture_site();
                if let Ok(mut guard) = STACKS.lock() {
                    let map = guard.get_or_insert_with(HashMap::new);
                    let e = map.entry(key).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += size;
                }
            }
            g.set(false);
        });
    }
}

const PLUMBING: &[&str] = &[
    "try_allocate_in", "allocate_in", "RawVec", "raw_vec", "exchange_malloc", "finish_grow",
    "grow_amortized", "grow_one", "do_reserve_and_handle", "reserve", "with_capacity",
    "from_iter", "to_vec", "into_vec", "extend_desugared", "SpecFrom", "SpecExtend",
    "spec_extend", "into_boxed_slice", "alloc::alloc", "alloc_impl", "grow_impl", "shrink_impl",
    "realloc_nonnull", "__rust",
];
const PLUMBING_EXACT: &[&str] = &["alloc", "realloc", "alloc_zeroed", "allocate", "grow", "shrink"];

fn capture_site() -> String {
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    let mut frames: Vec<&str> = Vec::new();
    for line in bt.lines() {
        let t = line.trim_start();
        let Some((num, rest)) = t.split_once(": ") else { continue };
        if !num.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        frames.push(rest.trim());
    }
    let start = frames.iter().rposition(|f| f.ends_with("note") || f.contains("capture_site"));
    let tail = start.map_or(&frames[..], |i| &frames[i + 1..]);
    let mut out: Vec<&str> = Vec::new();
    for f in tail {
        if PLUMBING.iter().any(|p| f.contains(p)) || PLUMBING_EXACT.contains(f) {
            continue;
        }
        out.push(f);
        if out.len() == 4 {
            break;
        }
    }
    if out.is_empty() { "<unknown>".to_string() } else { out.join(" \u{2190} ") }
}

// SAFETY: every request is delegated to `System`; the extra work is relaxed counters and (in
// profile mode) a re-entrancy-guarded backtrace capture.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size(), Ordering::Relaxed);
        Self::note(l.size());
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        LIVE.fetch_add(new.wrapping_sub(l.size()), Ordering::Relaxed);
        Self::note(new);
        System.realloc(p, l, new)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size(), Ordering::Relaxed);
        Self::note(l.size());
        System.alloc_zeroed(l)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn dump_sites() {
    if let Some(map) = STACKS.lock().unwrap().take() {
        let every = SAMPLE_EVERY.load(Ordering::Relaxed);
        let mut rows: Vec<(String, (usize, usize))> = map.into_iter().collect();
        rows.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        println!("  allocation sites (sampled 1/{every}):");
        for (site, (n, sz)) in rows.iter().take(18) {
            let site: String = site
                .split(" \u{2190} ")
                .map(|f| f.split_once('<').map_or(f, |(head, _)| head))
                .collect::<Vec<_>>()
                .join(" \u{2190} ");
            println!("    {:>8}  {:>8} B avg  {site}", n * every, sz / n.max(&1));
        }
    }
}

/// Run `build`, report allocations against the buffer count it returns.
fn measure(name: &str, want: Option<&str>, build: impl FnOnce() -> (Pipeline, Box<dyn Fn(&Pipeline) -> u64>)) {
    if let Some(w) = want {
        if !name.contains(w) {
            return;
        }
    }
    let (mut p, count) = build();
    let profile = std::env::var_os("PF_ALLOC_PROFILE").is_some();
    if profile {
        SAMPLE_EVERY.store(
            std::env::var("PF_ALLOC_SAMPLE").ok().and_then(|s| s.parse().ok()).unwrap_or(1),
            Ordering::Relaxed,
        );
        SAMPLING.store(true, Ordering::Relaxed);
    }
    let base = ALLOCS.load(Ordering::Relaxed);
    let base_bytes = BYTES.load(Ordering::Relaxed);
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
    p.run().expect("run");
    SAMPLING.store(false, Ordering::Relaxed);
    let allocs = ALLOCS.load(Ordering::Relaxed) - base;
    let bytes = BYTES.load(Ordering::Relaxed) - base_bytes;
    let bufs = count(&p);
    println!(
        "{name:<38} {bufs:>8} buffers  {allocs:>8} allocs  {:>7.4} allocs/buf  {:>6} KiB peak-live  {:>8} KiB total",
        allocs as f64 / bufs.max(1) as f64,
        PEAK.load(Ordering::Relaxed) / 1024,
        bytes / 1024,
    );
    if profile {
        dump_sites();
    }
}

fn main() {
    let want = std::env::args().nth(1);
    let want = want.as_deref();
    // A stream long enough that one-time startup amortizes: 64 MiB in 64 KiB slots = 1024 buffers.
    // `PF_ALLOC_MIB` scales it — a flat alloc *count* across two sizes proves the cost is one-time.
    let total: u64 = std::env::var("PF_ALLOC_MIB").ok().and_then(|s| s.parse().ok()).unwrap_or(64)
        << 20;
    let total = total;
    #[allow(non_snake_case)]
    let TOTAL = total;
    const SLOT: usize = 64 << 10;
    const SLOTS: u32 = 16;

    measure("testsrc ! testsink (baseline)", want, || {
        let (sink, _stats) = TestSink::new();
        let mut p = Pipeline::new();
        p.set_pool(SLOT, SLOTS);
        let src = p.add(TestSrc::new(TOTAL));
        let snk = p.add(sink);
        p.link((src, "src"), (snk, "sink")).unwrap();
        (p, Box::new(move |p: &Pipeline| p.counters(src).buffers_out))
    });

    measure("testsrc ! passthrough ! testsink", want, || {
        let (sink, _stats) = TestSink::new();
        let mut p = Pipeline::new();
        p.set_pool(SLOT, SLOTS);
        let src = p.add(TestSrc::new(TOTAL));
        let pt = p.add(PassThrough::new());
        let snk = p.add(sink);
        p.link((src, "src"), (pt, "sink")).unwrap();
        p.link((pt, "src"), (snk, "sink")).unwrap();
        (p, Box::new(move |p: &Pipeline| p.counters(src).buffers_out))
    });

    measure("testsrc ! queue ! testsink", want, || {
        let (sink, _stats) = TestSink::new();
        let mut p = Pipeline::new();
        p.set_pool(SLOT, SLOTS);
        let src = p.add(TestSrc::new(TOTAL));
        let q = p.add(Queue::new());
        let snk = p.add(sink);
        p.link((src, "src"), (q, "sink")).unwrap();
        p.link((q, "src"), (snk, "sink")).unwrap();
        (p, Box::new(move |p: &Pipeline| p.counters(src).buffers_out))
    });

    measure("testsrc ! tee(4) ! 4x testsink", want, || {
        let mut p = Pipeline::new();
        p.set_pool(SLOT, SLOTS);
        let src = p.add(TestSrc::new(TOTAL));
        let t = p.add(Tee::new(4));
        p.link((src, "src"), (t, "sink")).unwrap();
        for i in 0..4 {
            let (sink, _stats) = TestSink::new();
            let snk = p.add(sink);
            let name: &'static str = ["src_0", "src_1", "src_2", "src_3"][i];
            p.link((t, name), (snk, "sink")).unwrap();
        }
        (p, Box::new(move |p: &Pipeline| p.counters(src).buffers_out))
    });

    // Control: an in-place mutator with a *private* buffer takes the no-copy branch.
    measure("testsrc ! mutsink (unshared)", want, || {
        let mut p = Pipeline::new();
        p.set_pool(SLOT, SLOTS);
        let src = p.add(TestSrc::new(TOTAL));
        let m = p.add(MutSink);
        p.link((src, "src"), (m, "sink")).unwrap();
        (p, Box::new(move |p: &Pipeline| p.counters(src).buffers_out))
    });

    // The hazard: behind a tee the payload is shared, so the same mutation copy-on-writes a
    // whole slot per buffer — through `acquire_exact`, which falls back to an *unbounded* heap
    // box when the pool is dry.
    measure("testsrc ! tee(2) ! mutsink + testsink", want, || {
        let (sink, _stats) = TestSink::new();
        let mut p = Pipeline::new();
        p.set_pool(SLOT, SLOTS);
        let src = p.add(TestSrc::new(TOTAL));
        let t = p.add(Tee::new(2));
        let m = p.add(MutSink);
        let snk = p.add(sink);
        p.link((src, "src"), (t, "sink")).unwrap();
        p.link((t, "src_0"), (m, "sink")).unwrap();
        p.link((t, "src_1"), (snk, "sink")).unwrap();
        (p, Box::new(move |p: &Pipeline| p.counters(src).buffers_out))
    });

    // Same graph under pool pressure. Now `acquire_exact`'s `try_acquire` misses, `small_free`
    // refuses a slot-sized request, and every copy-on-write becomes a fresh *unpooled* heap box
    // — one malloc + free of a full slot per buffer, with no backpressure anywhere.
    measure("testsrc ! tee(2) ! mutsink + testsink [tiny pool]", want, || {
        let (sink, _stats) = TestSink::new();
        let mut p = Pipeline::new();
        p.set_pool(SLOT, 2);
        let src = p.add(TestSrc::new(TOTAL));
        let t = p.add(Tee::new(2));
        let m = p.add(MutSink);
        let snk = p.add(sink);
        p.link((src, "src"), (t, "sink")).unwrap();
        p.link((t, "src_0"), (m, "sink")).unwrap();
        p.link((t, "src_1"), (snk, "sink")).unwrap();
        (p, Box::new(move |p: &Pipeline| p.counters(src).buffers_out))
    });

    // Real IO. Write a fixture once, page-cache warm.
    let inp = std::env::temp_dir().join("pf_alloc_check_in.bin");
    let outp = std::env::temp_dir().join("pf_alloc_check_out.bin");
    if std::fs::metadata(&inp).map(|m| m.len()).unwrap_or(0) != TOTAL {
        std::fs::write(&inp, vec![0x5Au8; TOTAL as usize]).unwrap();
    }

    measure("filesrc ! testsink", want, || {
        let (sink, _stats) = TestSink::new();
        let mut p = Pipeline::new();
        p.set_pool(SLOT, SLOTS);
        let src = p.add(FileSrc::new(&inp));
        let snk = p.add(sink);
        p.link((src, "src"), (snk, "sink")).unwrap();
        (p, Box::new(move |p: &Pipeline| p.counters(src).buffers_out))
    });

    measure("filesrc ! filesink", want, || {
        let mut p = Pipeline::new();
        p.set_pool(SLOT, SLOTS);
        let src = p.add(FileSrc::new(&inp));
        let snk = p.add(FileSink::new(&outp));
        p.link((src, "src"), (snk, "sink")).unwrap();
        (p, Box::new(move |p: &Pipeline| p.counters(src).buffers_out))
    });

    measure("filesrc ! queue ! filesink", want, || {
        let mut p = Pipeline::new();
        p.set_pool(SLOT, SLOTS);
        let src = p.add(FileSrc::new(&inp));
        let q = p.add(Queue::new());
        let snk = p.add(FileSink::new(&outp));
        p.link((src, "src"), (q, "sink")).unwrap();
        p.link((q, "src"), (snk, "sink")).unwrap();
        (p, Box::new(move |p: &Pipeline| p.counters(src).buffers_out))
    });

    let _ = std::fs::remove_file(&outp);
}
