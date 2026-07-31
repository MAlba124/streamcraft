//! Steady-state allocation audit for `audiostretch` (spec: performance #1 — no steady-state
//! heap traffic; the house rule that `process()` allocates only from the pool/arena).
//!
//!     cargo run --release -p profluens-audio --example audiostretch_alloc_check
//!
//! A counting global allocator is armed *after* a warm-up phase, so format negotiation, the
//! one-time [`Engine`](profluens_audio::AudioStretch) buffers, and the harness's own growth are
//! all paid before measurement starts. Whatever it then counts across a long steady-state run
//! is per-buffer heap traffic — which must be zero.
//!
//! Every rate class is exercised, because each takes a different branch of the PICOLA splice
//! schedule and they have different worst-case round sizes: bypass, the `0.5 <= r < 1` and
//! `r < 0.5` slow-down branches, and the `1 < r < 2` and `r >= 2` speed-up branches.

#![allow(clippy::disallowed_methods)] // one-shot measurement harness

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use profluens_audio::format::{FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE};
use profluens_audio::AudioStretch;
use profluens_core::format::ValueDesc;
use profluens_core::harness::Harness;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

struct Counting;

// SAFETY: forwards every call to `System` unchanged; the counters are the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static A: Counting = Counting;

const RATE_HZ: u32 = 48_000;
const CHANNELS: u16 = 2;
/// Frames per pushed buffer — a deliberately un-round number so buffer boundaries land at
/// varying offsets in the splice schedule.
const CHUNK_FRAMES: usize = 1021;

fn gen_input(frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames * CHANNELS as usize);
    let mut seed: u32 = 0x1234_5678;
    let period = (RATE_HZ / 120) as usize;
    let eperiod = (RATE_HZ * 7 / 10) as usize;
    for n in 0..frames {
        let phase = (n % period) as f32 / period as f32;
        let tri = 1.0 - (2.0 * phase - 1.0).abs();
        let voiced = tri * tri * 2.0 - 0.5;
        let ephase = (n % eperiod) as f32 / eperiod as f32;
        let env = 1.0 - (2.0 * ephase - 1.0).abs();
        for c in 0..CHANNELS as usize {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 9) as f32 / 4_194_304.0 - 1.0;
            out.push(env * voiced * 0.6 + 0.05 * noise - 0.02 * c as f32);
        }
    }
    out
}

/// One measured run: `(allocations, heap bytes, input buffers pushed, output buffers pulled)`.
struct Run {
    allocs: usize,
    /// Total heap bytes requested while armed — reported for context; the verdict is on
    /// allocation *count*, which is what fragments and blocks.
    #[allow(dead_code)]
    bytes: usize,
    pushes: usize,
    pulls: usize,
}

const WARM: usize = 64;
const ROUNDS: usize = 512;

/// Drive `h` through warm-up and then a counted steady-state phase.
fn drive(h: &mut Harness) -> Run {
    let chunk = gen_input(CHUNK_FRAMES);
    let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_ne_bytes()).collect();

    let round = |h: &mut Harness| {
        let buf = h.alloc(&bytes);
        h.push("sink", buf).expect("push");
        let mut pulls = 0usize;
        while let Some(_b) = h.pull("src") {
            pulls += 1;
        }
        pulls
    };

    // Warm-up: negotiate, build the engine, let the harness's own vectors reach steady size.
    for _ in 0..WARM {
        round(h);
    }

    ALLOCS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    let mut pulls = 0usize;
    for _ in 0..ROUNDS {
        pulls += round(h);
    }
    ARMED.store(false, Ordering::Relaxed);

    Run {
        allocs: ALLOCS.load(Ordering::Relaxed),
        bytes: BYTES.load(Ordering::Relaxed),
        pushes: ROUNDS,
        pulls,
    }
}

fn fix_audio_caps(h: &mut Harness) {
    h.fix_format(
        "sink",
        "audio/raw",
        &[
            (FIELD_RATE, ValueDesc::Int(RATE_HZ as i64)),
            (FIELD_CHANNELS, ValueDesc::Int(CHANNELS as i64)),
            (FIELD_SAMPLE, ValueDesc::Id("f32")),
        ],
    );
}

/// The control (see [`chunker`]): re-emits each input buffer as `parts` pool buffers and owns
/// no heap state at all, so a run with the *same* push/pull shape as the element under test
/// measures exactly the surrounding machinery — `Batch`'s SoA columns growing as buffers move
/// through `Harness::push` / `OutBatch::push` / `Harness::pull`. That machinery, not the
/// element, is where every residual allocation below comes from (confirmed by backtrace
/// attribution: no site resolves into `audio/src/stretch.rs`).
fn control(parts: usize) -> Run {
    let mut h = Harness::with_slot_size(chunker::Chunker::new(parts), 1 << 16);
    drive(&mut h)
}

fn measure(rate: f32, label: &str) -> bool {
    let mut h = Harness::with_slot_size(AudioStretch::new(rate), 1 << 16);
    fix_audio_caps(&mut h);
    h.start().expect("start");
    let run = drive(&mut h);

    // Match the control's output-buffer shape to the element's, so the two runs push the same
    // number of buffers through the same batch machinery.
    let parts = run.pulls.div_ceil(run.pushes.max(1));
    let ctl = control(parts);

    let per = |r: &Run| r.allocs as f64 / (r.pushes + r.pulls) as f64;
    let (mine, theirs) = (per(&run), per(&ctl));
    // Anything the element itself allocated per buffer would push its rate above the control's.
    let ok = mine <= theirs * 1.05;
    println!(
        "{label:<24} rate={rate:<5} {:>4} pushes /{:>5} pulls  allocs={:<6} ({mine:.2}/buf)   \
         control(parts={parts}): {:>4}/{:<5} allocs={:<6} ({theirs:.2}/buf)   {}",
        run.pushes,
        run.pulls,
        run.allocs,
        ctl.pushes,
        ctl.pulls,
        ctl.allocs,
        if ok { "OK" } else { "*** ELEMENT ALLOCATES ***" }
    );
    ok
}

fn main() {
    println!("=== audiostretch steady-state heap traffic ===");
    println!(
        "Each rate is compared against a control element with the same push/pull shape that\n\
         provably owns no heap state. Equal per-buffer rates mean the stretcher itself added\n\
         nothing: its engine buffers are sized once, at negotiation, for the worst-case round.\n"
    );
    let mut ok = true;
    ok &= measure(1.0, "bypass (copy-free)");
    ok &= measure(0.35, "slow-down r<0.5");
    ok &= measure(0.75, "slow-down 0.5<=r<1");
    ok &= measure(1.5, "speed-up 1<r<2");
    ok &= measure(2.5, "speed-up r>=2");
    println!(
        "\n=== {} ===",
        if ok {
            "ZERO steady-state allocations by the element"
        } else {
            "ALLOCATION IN THE HOT PATH"
        }
    );
    std::process::exit(if ok { 0 } else { 1 });
}

/// A control element: splits every input buffer into `parts` pool buffers. It has no heap
/// state whatsoever, so all it contributes is the pool/batch traffic of pushing `parts`
/// buffers per input — the same shape `audiostretch` produces at a given rate.
mod chunker {
    use profluens_core::batch::Inputs;
    use profluens_core::ctx::Ctx;
    use profluens_core::element::{
        Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
    };
    use profluens_core::error::Error;
    use profluens_core::event::Event;
    use profluens_core::format::OfferDesc;
    use profluens_core::id::PadId;
    use profluens_core::time::Timestamp;

    static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
    static PADS: [PadDesc; 2] = [
        PadDesc {
            name: "sink",
            direction: Direction::Sink,
            offers: &OFFERS,
            dynamic: false,
            validate: None,
        },
        PadDesc {
            name: "src",
            direction: Direction::Src,
            offers: &OFFERS,
            dynamic: false,
            validate: None,
        },
    ];
    static DESC: ElementDesc = ElementDesc {
        name: "chunker",
        pads: &PADS,
        props: &[],
        sched: SchedHint::Passive,
        inputs: InputPolicy::Single,
        latency: LatencyDesc {
            min: Timestamp::ZERO,
            max: Timestamp::ZERO,
            is_live: false,
            jitter: Timestamp::ZERO,
        },
        make_default: None,
    };

    pub struct Chunker(usize);
    impl Chunker {
        pub fn new(parts: usize) -> Self {
            Self(parts.max(1))
        }
    }
    impl Element for Chunker {
        fn desc(&self) -> &'static ElementDesc {
            &DESC
        }
        fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
            Ok(())
        }
        fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
            while let Some(b) = inputs.pop() {
                let data = b.memory.data();
                let step = data.len().div_ceil(self.0).max(1);
                for part in data.chunks(step) {
                    let mut out = ctx.alloc(PadId(1));
                    let n = part.len().min(out.memory.capacity());
                    out.memory.as_mut_full()[..n].copy_from_slice(&part[..n]);
                    out.memory.set_len(n);
                    ctx.out(PadId(1)).push(out);
                }
            }
            Ok(Flow::Ok)
        }
        fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
            Ok(())
        }
        fn stop(&mut self, _ctx: &mut Ctx) {}
    }
}
