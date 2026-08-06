//! A failed group stops the pipeline (spec: robustness — fail loudly). The
//! regression shape is the NVR's: two chains with NO data path between them
//! (independent cameras). One chain's sink errors; the other would happily run
//! forever. `run()` joins group threads in order, so before the error-cascades-
//! stop fix the healthy chain's thread was joined first and never exited — the
//! error sat invisible in a later `JoinHandle` while the pipeline hung
//! (measured in the wild with the failed group's closed ring pinning pool
//! slots, wedging its upstream too). The contract under test: an element error
//! anywhere stops every group, and `run()` returns that error.

use std::sync::mpsc;

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
use profluens_elements::testing::TestSink;

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

static SRC_DESC: ElementDesc = ElementDesc {
    name: "endless_src",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

static FAIL_DESC: ElementDesc = ElementDesc {
    name: "failing_sink",
    pads: &SINK_PADS,
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

/// Emits small buffers forever — a live camera stand-in. Only the pipeline
/// stop ends it.
struct EndlessSrc;

impl Element for EndlessSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let n = buf.memory.capacity().min(64);
        buf.memory.as_mut_full()[..n].fill(0xAB);
        buf.memory.set_len(n);
        ctx.out(PadId(0)).push(buf);
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// Consumes a few buffers, then fails like a sink whose device/file vanished.
struct FailingSink {
    seen: u32,
    fail_at: u32,
}

impl Element for FailingSink {
    fn desc(&self) -> &'static ElementDesc {
        &FAIL_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(_buf) = inputs.pop() {
            self.seen += 1;
            if self.seen >= self.fail_at {
                return Err(Error::Resource("failing_sink: injected failure".into()));
            }
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// The unlucky-join-order wedge: chain A is healthy and endless, chain B fails
/// after a few buffers. `run()` must return chain B's error promptly instead
/// of joining chain A forever. Run under a watchdog so a regression fails the
/// suite instead of hanging it.
#[test]
fn element_error_stops_disjoint_siblings() {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut p = Pipeline::new();
        // Chain A: endless, healthy, no data path to chain B.
        let a_src = p.add(EndlessSrc);
        let (a_sink, _stats) = TestSink::new();
        let a_sink = p.add(a_sink);
        p.link((a_src, "src"), (a_sink, "sink")).expect("a_src ! a_sink");
        // Chain B: fails on the 3rd buffer.
        let b_src = p.add(EndlessSrc);
        let b_sink = p.add(FailingSink { seen: 0, fail_at: 3 });
        p.link((b_src, "src"), (b_sink, "sink")).expect("b_src ! b_sink");
        let _ = tx.send(p.run());
    });
    let result = rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("run() must return once a group fails — the error-cascades-stop contract");
    let err = result.expect_err("the injected sink failure must surface from run()");
    assert!(
        format!("{err:?}").contains("injected failure"),
        "run() returns the failing element's error, got: {err:?}"
    );
}
