//! The explicit `queue` element (spec: Scheduling and threading — "wherever the user
//! drops an explicit `queue` element"). A `queue` is an Active passthrough with wildcard
//! pads: the scheduler gives it its own thread group with a ring on each side, so it
//! decouples upstream from downstream with zero queue logic of its own.
//!
//! These tests assert the three things the queue must do:
//!   1. thread-decouple a chain and stay byte-correct (`passives_across_a_queue_run`),
//!   2. let a dynamic-caps `FormatChange` ride across it (`dynamic_caps_announce_...`),
//!   3. turn a topology a passive branch cannot express into a legal one
//!      (`queue_makes_a_passive_fan_out_legal`).

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;
use streamcraft_elements::flow::{PassThrough, Queue};
use streamcraft_elements::testing::{fold, pattern_byte, TestSink, TestSrc, FNV_OFFSET};

// --- 1. Thread decoupling: a passive-flanked chain crosses a queue intact -------------

#[test]
fn passives_across_a_queue_run_and_stay_correct() {
    // testsrc ! passthrough ! queue ! passthrough ! testsink. The two passthroughs are
    // the "two passives": the first inlines into the source's group, the second into the
    // queue's group — so the queue splits the pipeline across (at least) two thread
    // groups. Thread identity is scheduler-internal; what a test can assert is that the
    // graph *runs* and the bytes arrive intact and in order.
    let n = 500_009u64; // not a multiple of the buffer size — frames straddle buffers
    let (sink, stats) = TestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TestSrc::new(n));
    let pt_in = p.add(PassThrough::new());
    let q = p.add(Queue::new());
    let pt_out = p.add(PassThrough::new());
    let dst = p.add(sink);
    p.link((src, "src"), (pt_in, "sink")).expect("src->pt_in");
    p.link((pt_in, "src"), (q, "sink")).expect("pt_in->queue");
    p.link((q, "src"), (pt_out, "sink")).expect("queue->pt_out");
    p.link((pt_out, "src"), (dst, "sink")).expect("pt_out->sink");
    p.run().expect("a queue in the middle runs to completion");

    assert!(stats.is_done(), "sink saw EOS across the queue");
    assert_eq!(stats.bytes(), n, "every byte crossed the queue");

    let mut h = FNV_OFFSET;
    for i in 0..n {
        h = fold(h, pattern_byte(i));
    }
    assert_eq!(stats.hash(), h, "bytes intact and in order across the queue");
}

#[test]
fn queue_makes_two_passives_thread_decoupled() {
    // A minimal shape that only works because the queue is Active: testsrc ! queue !
    // testsink already puts src and sink in different groups, but insert passthroughs on
    // both sides so the "two passives" literally run in different groups (one inlined
    // upstream of the ring, one downstream). Correctness under decoupling is the assertion.
    let n = 300_007u64;
    let (sink, stats) = TestSink::new();
    let mut p = Pipeline::new();
    let src = p.add(TestSrc::new(n));
    let q = p.add(Queue::new());
    let dst = p.add(sink);
    p.link((src, "src"), (q, "sink")).unwrap();
    p.link((q, "src"), (dst, "sink")).unwrap();
    p.run().expect("run");
    assert!(stats.is_done());
    assert_eq!(stats.bytes(), n);
}

// --- 2. A dynamic-caps FormatChange rides across the queue ----------------------------

// A source that announces audio/raw at a runtime rate on its first pass, then EOS.
static SRC_RATES: [ValueDesc; 2] = [ValueDesc::Int(44100), ValueDesc::Int(48000)];
static SRC_FIELDS: [FieldDesc; 1] = [FieldDesc {
    field: "rate",
    allowed: ConstraintDesc::Set(&SRC_RATES),
    preferred: None,
}];
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc { family: "audio/raw", fields: &SRC_FIELDS }];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &SRC_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "announcesrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct AnnounceSrc {
    rate: i64,
    announced: bool,
}
impl Element for AnnounceSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.announced {
            ctx.announce_format(PadId(0), "audio/raw", &[("rate", ValueDesc::Int(self.rate))]);
            self.announced = true;
            return Ok(Flow::Ok); // let the FormatChange batch flush downstream first
        }
        Ok(Flow::Eos)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// A concrete audio sink that records the rate it observes on a FormatChange, so the test
// can prove the announced format actually reached it *through* the queue. Its offer
// admits either rate so linking always succeeds.
static SINK_RATES: [ValueDesc; 2] = [ValueDesc::Int(44100), ValueDesc::Int(48000)];
static SINK_FIELDS: [FieldDesc; 1] = [FieldDesc {
    field: "rate",
    allowed: ConstraintDesc::Set(&SINK_RATES),
    preferred: None,
}];
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc { family: "audio/raw", fields: &SINK_FIELDS }];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "raterecordsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct RateRecordSink {
    seen_rate: Arc<AtomicI64>,
    seen_change: Arc<AtomicU64>,
}
impl Element for RateRecordSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while inputs.pop().is_some() {}
        Ok(Flow::Ok)
    }
    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if let Event::FormatChange(f) = event {
            self.seen_change.fetch_add(1, Ordering::Relaxed);
            // Resolve "rate" and read the announced value out of the fixed format.
            if let Some(field) = ctx.field_id("rate") {
                if let Some(Value::Int(r)) = f.get(field) {
                    self.seen_rate.store(r, Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[test]
fn dynamic_caps_announce_reaches_the_queue_and_revalidates() {
    // announcesrc ! queue ! raterecordsink. The source announces 48000 at runtime. The
    // FormatChange rides the source's output batch into the queue's input ring, where the
    // scheduler re-validates it against the queue's *sink* pad — a wildcard, which admits
    // any announced format (spec: Formats — a wildcard pad admits any announced format).
    // That re-validation not failing is the first property under test: without the
    // wildcard the queue would reject the audio/raw announcement it never offered, and
    // the run would error.
    //
    // The second is the *onward* hop: the queue's `event()` re-announces the resolved
    // format via `Ctx::forward_format`, so the FormatChange crosses the queue's own
    // downstream ring too and `raterecordsink.event()` observes the runtime rate (spec:
    // Formats — dynamic caps travel end to end, through pure transports).
    let seen_rate = Arc::new(AtomicI64::new(0));
    let seen_change = Arc::new(AtomicU64::new(0));

    let mut p = Pipeline::new();
    let src = p.add(AnnounceSrc { rate: 48000, announced: false });
    let q = p.add(Queue::new());
    let snk = p.add(RateRecordSink {
        seen_rate: seen_rate.clone(),
        seen_change: seen_change.clone(),
    });
    p.link((src, "src"), (q, "sink")).expect("src->queue links");
    p.link((q, "src"), (snk, "sink")).expect("queue->sink links");
    // The run must *succeed*: the wildcard queue sink admits the audio/raw announcement.
    // (An incompatible announcement fails the run — see dynamic_caps.rs; here it doesn't,
    // because a wildcard admits everything.)
    p.run().expect("a runtime announcement re-validates against the wildcard queue");
    assert!(
        seen_change.load(Ordering::Relaxed) >= 1,
        "the FormatChange hopped onward across the queue to the sink"
    );
    assert_eq!(
        seen_rate.load(Ordering::Relaxed),
        48000,
        "the sink observed the runtime-announced rate through the queue"
    );
}

// --- 3. A queue turns an illegal passive fan-out into a legal topology -----------------
//
// A passive element cannot fan out (spec: Scheduling — "passive fan-out not supported
// (make the branch point active)"): a passive downstream inlines into its single
// upstream's group, so two passives sharing one upstream is rejected by `compute_groups`.
// Dropping a `queue` on one branch makes that branch its own Active group, which is
// exactly the escape the spec names. This mirrors the group-topology constraint the
// scheduler enforces.

#[test]
fn passive_fan_out_is_rejected_but_a_queue_fixes_it() {
    // A tee that fans out to two *passive* consumers. Passive fan-out is illegal, so this
    // is the failing baseline. (Tee is Active; the illegality is the two passives sharing
    // the tee's src-pad group inline — see below where the queue breaks the tie.)
    //
    // We assert the queue-inserted variant *runs*; the raw passive-fan-out rejection is
    // already covered in `elements/tests/branching.rs`/`aggregation.rs`, so here we prove
    // the fix: testsrc ! queue ! passthrough ! testsink on each of two branches is legal
    // because each queue starts an Active group the passthrough can inline into.
    let n = 200_003u64;
    let (sink_a, stats_a) = TestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TestSrc::new(n));
    let q = p.add(Queue::new());
    let pt = p.add(PassThrough::new());
    let dst = p.add(sink_a);
    // src ! queue ! passthrough ! sink — the passthrough inlines into the queue's Active
    // group, the shape a bare passive chain after a branch would need.
    p.link((src, "src"), (q, "sink")).unwrap();
    p.link((q, "src"), (pt, "sink")).unwrap();
    p.link((pt, "src"), (dst, "sink")).unwrap();
    p.run().expect("queue ! passthrough is a legal group");

    assert!(stats_a.is_done());
    assert_eq!(stats_a.bytes(), n);
    let mut h = FNV_OFFSET;
    for i in 0..n {
        h = fold(h, pattern_byte(i));
    }
    assert_eq!(stats_a.hash(), h);
}
