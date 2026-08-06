//! Regression: the **inline-group livelock**, and the head backlog cap that bounds it
//! (spec: Scheduling — "real queues exist only at group boundaries; *inside* a group the
//! scheduler paces producers"; `INLINE_INPUT_CAP` in `core/src/pipeline.rs`).
//!
//! Both of these are documented in prose at the two gates and had no test.
//!
//! The livelock: a fast source inlined with a deliberately latency-bounded consumer
//! (`filesrc ! flacdec …`) free-runs until the *shared* pool is exhausted by its own queued
//! output — at which point the consumer can no longer allocate **its** output, so by design it
//! stops consuming input, so it never frees a slot, and the group wedges forever. Nothing
//! upstream is blocked by a ring, because inside a group there is no ring; the only thing that
//! can pace the producer is the scheduler, and that is the gate under test here.
//!
//! The backlog cap: the same rule applied to a group *head* fed by a boundary ring. Every
//! buffered buffer pins a pool slot, so an uncapped head backlog is how a movie ate gigabytes.
//!
//! `ThrottledXform` below is the exact shape that triggers it: it consumes one buffer per
//! `process()` and only after it has secured its own output slot ("the decoder's carried
//! picture"). Without the gate this test hangs; the watchdog turns that into a loud failure
//! instead of a hung suite.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::PadId;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;

/// Mirrors `pipeline::INLINE_INPUT_CAP` (private). If that constant moves, the bound below
/// should move with it — the assertion is "bounded and small", not this exact number.
const INLINE_INPUT_CAP: usize = 8;
/// Buffers the source emits per `process()` call. The gate is checked *before* the producer
/// runs, so one whole burst can land on top of a nearly-full backlog.
const BURST: usize = 4;
/// Total buffers to push through. Big enough that an unbounded backlog would exhaust the pool
/// many times over.
const TOTAL: usize = 400;
/// Pool slots, deliberately **larger** than `INLINE_INPUT_CAP` so the scheduler gate — not the
/// pool cap — is what bounds the backlog. Without the gate the source fills all of these into
/// the transform's input queue and the group livelocks.
const SLOTS: u32 = 24;

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static XFORM_PADS: [PadDesc; 2] = [
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

const LAT: LatencyDesc = LatencyDesc {
    min: Timestamp::ZERO,
    max: Timestamp::ZERO,
    is_live: false,
    jitter: Timestamp::ZERO,
};

// ---------------------------------------------------------------------------
// Source: emits BURST buffers per pass, as fast as the scheduler will let it.
// ---------------------------------------------------------------------------
static SPEW_DESC: ElementDesc = ElementDesc {
    name: "spewsrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LAT,
    make_default: None,
};

struct SpewSrc {
    sent: usize,
}

impl Element for SpewSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SPEW_DESC
    }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> {
        if self.sent >= TOTAL {
            return Ok(Flow::Eos);
        }
        for _ in 0..BURST {
            if self.sent >= TOTAL {
                break;
            }
            // Pool backpressure: `None` means "no slot", which is exactly the state the
            // livelock made permanent.
            let Some(mut b) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
            b.memory.set_len(1);
            ctx.out(PadId(0)).push(b);
            self.sent += 1;
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _c: &mut Ctx) {}
}

// ---------------------------------------------------------------------------
// Latency-bounded transform: one buffer in, one buffer out, per pass — and it
// refuses to consume until it holds its own output slot. That "carried picture"
// discipline is what turns pool exhaustion into a livelock rather than a stall.
// ---------------------------------------------------------------------------
static XFORM_DESC: ElementDesc = ElementDesc {
    name: "throttledxform",
    pads: &XFORM_PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LAT,
    make_default: None,
};

struct ThrottledXform {
    /// Highest `inputs.len()` ever observed on entry to `process` — the backlog the gate is
    /// supposed to bound.
    peak: Arc<AtomicUsize>,
}

impl Element for ThrottledXform {
    fn desc(&self) -> &'static ElementDesc {
        &XFORM_DESC
    }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, mut i: Inputs<'_>) -> Result<Flow, Error> {
        self.peak.fetch_max(i.len(), Ordering::AcqRel);
        if i.is_empty() {
            return Ok(Flow::Ok);
        }
        // Secure the output slot FIRST; without one, do not consume input.
        let Some(mut out) = ctx.try_alloc(PadId(1)) else { return Ok(Flow::Ok) };
        let Some(inb) = i.pop() else { return Ok(Flow::Ok) };
        out.memory.set_len(inb.memory.len());
        drop(inb);
        ctx.out(PadId(1)).push(out);
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _c: &mut Ctx) {}
}

// ---------------------------------------------------------------------------
// Passive tail: drains everything it is handed, so slots keep coming back.
// ---------------------------------------------------------------------------
static SINK_DESC: ElementDesc = ElementDesc {
    name: "drainsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LAT,
    make_default: None,
};

struct DrainSink {
    n: Arc<AtomicU64>,
}

impl Element for DrainSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _c: &mut Ctx, mut i: Inputs<'_>) -> Result<Flow, Error> {
        while i.pop().is_some() {
            self.n.fetch_add(1, Ordering::Release);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _c: &mut Ctx) {}
}

/// Run `p` with a watchdog that stops it after `budget`, so a returned livelock fails the test
/// loudly instead of hanging the suite. Returns whether the watchdog had to fire.
fn run_with_watchdog(p: &mut Pipeline, budget: Duration) -> bool {
    let fired = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let handle = p.stop_handle();
    let watchdog = {
        let (fired, done) = (Arc::clone(&fired), Arc::clone(&done));
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + budget;
            while std::time::Instant::now() < deadline {
                if done.load(Ordering::Acquire) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            fired.store(true, Ordering::Release);
            handle.stop();
        })
    };
    p.run().expect("run");
    done.store(true, Ordering::Release);
    watchdog.join().expect("watchdog");
    fired.load(Ordering::Acquire)
}

/// The livelock regression. A fast source inlined with a latency-bounded consumer, sharing one
/// bounded pool, must run to completion — and the consumer's backlog must stay bounded by the
/// inline gate rather than by the pool running dry.
#[test]
fn inline_gate_paces_a_fast_source_and_the_group_never_livelocks() {
    let peak = Arc::new(AtomicUsize::new(0));
    let got = Arc::new(AtomicU64::new(0));

    let mut p = Pipeline::new();
    // One shared pool for the whole group: elements without an override share it, which is what
    // makes "the producer eats every slot" possible in the first place.
    p.set_pool(64, SLOTS);
    let src = p.add(SpewSrc { sent: 0 });
    let xform = p.add(ThrottledXform { peak: Arc::clone(&peak) });
    let snk = p.add(DrainSink { n: Arc::clone(&got) });
    p.link((src, "src"), (xform, "sink")).expect("src!xform");
    p.link((xform, "src"), (snk, "sink")).expect("xform!sink");

    // Generous: the work itself is microseconds. A livelock never finishes at all, so the
    // budget only has to be long enough that a loaded machine is not mistaken for one.
    let timed_out = run_with_watchdog(&mut p, Duration::from_secs(10));

    assert!(!timed_out, "pipeline livelocked — the inline backpressure gate is not pacing the source");
    assert_eq!(got.load(Ordering::Acquire), TOTAL as u64, "every buffer reached the tail");

    // The gate lets the producer run while the successor holds < INLINE_INPUT_CAP, and the
    // producer then emits a whole burst, so the true ceiling is CAP - 1 + BURST.
    let observed = peak.load(Ordering::Acquire);
    assert!(
        observed <= INLINE_INPUT_CAP + BURST,
        "inline backlog reached {observed} buffers (cap {INLINE_INPUT_CAP} + one {BURST}-buffer \
         burst) — the producer is not being paced by its successor's backlog"
    );
    assert!(observed > 1, "the transform never saw a backlog — the test is not exercising the gate");
}

/// The memory half of the same rule: because the backlog is bounded, so is pool occupancy. A
/// run that only ever allocates as many slot-sized boxes as the pool has slots is the
/// "steady-state allocation is flat" property that the movie OOM violated.
#[test]
fn a_bounded_backlog_keeps_pool_allocation_flat() {
    let peak = Arc::new(AtomicUsize::new(0));
    let got = Arc::new(AtomicU64::new(0));

    let mut p = Pipeline::new();
    p.set_pool(64, SLOTS);
    let src = p.add(SpewSrc { sent: 0 });
    let xform = p.add(ThrottledXform { peak: Arc::clone(&peak) });
    let snk = p.add(DrainSink { n: Arc::clone(&got) });
    p.link((src, "src"), (xform, "sink")).expect("src!xform");
    p.link((xform, "src"), (snk, "sink")).expect("xform!sink");

    let tap = p.tap_handle();
    let timed_out = run_with_watchdog(&mut p, Duration::from_secs(10));
    assert!(!timed_out, "pipeline livelocked");
    assert_eq!(got.load(Ordering::Acquire), TOTAL as u64);

    let pool = tap.pool();
    assert_eq!(
        pool.outstanding, 0,
        "every slot came back at end of run (outstanding leak = the freeze's failure mode)"
    );
    assert!(
        pool.slot_allocations <= SLOTS as u64,
        "{TOTAL} buffers cost {} slot allocations for a {SLOTS}-slot pool — recycling is not \
         flat, i.e. the pool is being missed and the run heap-allocates per buffer",
        pool.slot_allocations
    );
    assert!(
        pool.high_water <= SLOTS as u64,
        "occupancy peaked at {} of {SLOTS} slots",
        pool.high_water
    );
}
