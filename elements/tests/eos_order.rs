//! Regression: **EOS reaches a group's elements in chain order, one per pass**, and whatever an
//! element flushes on its EOS is processed by its successor *before* that successor's own EOS
//! (spec: Events; `core/src/pipeline.rs` step E — "a muxer flushes its final page, which the
//! downstream depacketiser then sees as input before its own EOS").
//!
//! This is the invariant behind every truncated-tail bug: the last page a muxer writes, the last
//! frames a decoder has buffered, a recording sink's index patch. It is exactly the class of
//! defect the external-player tail gate exists to catch (`--start=95%`), and the cost of getting
//! it wrong is a file that is silently short — valid enough to open, missing its end.
//!
//! It was documented only in a prose comment. If EOS were delivered to the whole group in one
//! pass (the obvious simplification), `FlushXform`'s final buffer would be produced *after*
//! `TailSink` had already seen `Eos` and closed — and every assertion below would still look
//! plausible from the outside, because all the *streaming* buffers arrive fine. Only the tail is
//! lost.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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

const STREAMED: usize = 5;
/// Payload byte marking the buffer the transform flushes from its `Eos` handler — "the muxer's
/// final page". Distinct from the streamed payload so the tail is identifiable in the trace.
const FLUSH_MARK: u8 = 0xFE;
const STREAM_MARK: u8 = 0x01;

/// Ordered trace of everything the group did that matters: `"eos:<element>"` for each EOS
/// delivery, and `"recv:<byte>"` for each buffer the tail actually processed.
type Trace = Arc<Mutex<Vec<String>>>;

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
static SRC_DESC: ElementDesc = ElementDesc {
    name: "eossrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LAT,
    make_default: None,
};

struct EosSrc {
    sent: usize,
    trace: Trace,
}

impl Element for EosSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> {
        if self.sent >= STREAMED {
            return Ok(Flow::Eos);
        }
        let mut b = ctx.alloc(PadId(0));
        b.memory.as_mut_full()[0] = STREAM_MARK;
        b.memory.set_len(1);
        ctx.out(PadId(0)).push(b);
        self.sent += 1;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, e: &Event) -> Result<(), Error> {
        if matches!(e, Event::Eos) {
            self.trace.lock().unwrap().push("eos:src".into());
        }
        Ok(())
    }
    fn stop(&mut self, _c: &mut Ctx) {}
}

// ---------------------------------------------------------------------------
// The "muxer": passes streaming buffers through, and on EOS emits one final
// buffer — the page that must not be lost.
// ---------------------------------------------------------------------------
static XFORM_DESC: ElementDesc = ElementDesc {
    name: "flushxform",
    pads: &XFORM_PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LAT,
    make_default: None,
};

struct FlushXform {
    trace: Trace,
}

impl Element for FlushXform {
    fn desc(&self) -> &'static ElementDesc {
        &XFORM_DESC
    }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, mut i: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(b) = i.pop() {
            let mut out = ctx.alloc(PadId(1));
            out.memory.as_mut_full()[0] = b.memory.data()[0];
            out.memory.set_len(1);
            ctx.out(PadId(1)).push(out);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, ctx: &mut Ctx, e: &Event) -> Result<(), Error> {
        if matches!(e, Event::Eos) {
            self.trace.lock().unwrap().push("eos:xform".into());
            // The final page. Produced *from the EOS handler*, which is the whole point:
            // the scheduler must give it a pass to reach the tail before the tail's own EOS.
            let mut out = ctx.alloc(PadId(1));
            out.memory.as_mut_full()[0] = FLUSH_MARK;
            out.memory.set_len(1);
            ctx.out(PadId(1)).push(out);
        }
        Ok(())
    }
    fn stop(&mut self, _c: &mut Ctx) {}
}

// ---------------------------------------------------------------------------
static TAIL_DESC: ElementDesc = ElementDesc {
    name: "tracesink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LAT,
    make_default: None,
};

struct TraceSink {
    trace: Trace,
    n: Arc<AtomicU64>,
}

impl Element for TraceSink {
    fn desc(&self) -> &'static ElementDesc {
        &TAIL_DESC
    }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _c: &mut Ctx, mut i: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(b) = i.pop() {
            self.trace.lock().unwrap().push(format!("recv:{:02x}", b.memory.data()[0]));
            self.n.fetch_add(1, Ordering::Release);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, e: &Event) -> Result<(), Error> {
        if matches!(e, Event::Eos) {
            self.trace.lock().unwrap().push("eos:sink".into());
        }
        Ok(())
    }
    fn stop(&mut self, _c: &mut Ctx) {}
}

fn run_chain() -> Vec<String> {
    let trace: Trace = Arc::new(Mutex::new(Vec::new()));
    let got = Arc::new(AtomicU64::new(0));

    let mut p = Pipeline::new();
    p.set_pool(64, 16);
    let src = p.add(EosSrc { sent: 0, trace: Arc::clone(&trace) });
    let xform = p.add(FlushXform { trace: Arc::clone(&trace) });
    let snk = p.add(TraceSink { trace: Arc::clone(&trace), n: Arc::clone(&got) });
    p.link((src, "src"), (xform, "sink")).expect("src!xform");
    p.link((xform, "src"), (snk, "sink")).expect("xform!sink");
    p.run().expect("run");

    assert_eq!(
        got.load(Ordering::Acquire) as usize,
        STREAMED + 1,
        "the tail received every streamed buffer plus the EOS flush"
    );
    let t = trace.lock().unwrap().clone();
    t
}

/// EOS walks the chain in order — never all at once, never backwards.
#[test]
fn eos_reaches_elements_in_chain_order() {
    let trace = run_chain();
    let eos_order: Vec<&str> =
        trace.iter().filter(|s| s.starts_with("eos:")).map(|s| s.as_str()).collect();
    assert_eq!(
        eos_order,
        vec!["eos:src", "eos:xform", "eos:sink"],
        "EOS must reach elements in chain order; full trace: {trace:?}"
    );
}

/// The load-bearing half: an element's EOS flush is *processed by its successor* before that
/// successor is told the stream ended. A muxer's final page has to land in the file.
#[test]
fn an_eos_flush_reaches_the_tail_before_the_tails_own_eos() {
    let trace = run_chain();

    let flush_at = trace
        .iter()
        .position(|s| s == &format!("recv:{FLUSH_MARK:02x}"))
        .unwrap_or_else(|| panic!("the EOS-flushed buffer never reached the tail: {trace:?}"));
    let sink_eos_at = trace
        .iter()
        .position(|s| s == "eos:sink")
        .unwrap_or_else(|| panic!("the tail never got EOS: {trace:?}"));

    assert!(
        flush_at < sink_eos_at,
        "the tail was told EOS before it processed the upstream's flushed final buffer — this is \
         the truncated-tail bug; trace: {trace:?}"
    );

    // And the flush really is the *last* thing, after all the streamed payload.
    let streamed = trace.iter().filter(|s| s == &&format!("recv:{STREAM_MARK:02x}")).count();
    assert_eq!(streamed, STREAMED, "every streamed buffer arrived too: {trace:?}");
    assert_eq!(
        trace.last().map(String::as_str),
        Some("eos:sink"),
        "nothing happens after the tail's EOS: {trace:?}"
    );
}
