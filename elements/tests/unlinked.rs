//! Unlinked-pad policy (spec: robustness): output pushed to a src pad nobody
//! linked is discarded by the scheduler — every pass, counted as drops — instead
//! of accumulating in the output table (the old behavior, which forced dummy
//! "drop-sinks" onto every unwatched demuxer track and cost real memory). The
//! linked branch must be entirely unaffected.

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

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 2] = [
    PadDesc {
        name: "a",
        direction: Direction::Src,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "b",
        direction: Direction::Src,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    },
];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "twopadsrc",
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

/// A two-track "demuxer": each round emits one buffer on pad `a` and one on pad
/// `b`. The test links only `a` — every `b` buffer must be dropped by policy.
struct TwoPadSrc {
    n: usize,
    sent: usize,
}

impl Element for TwoPadSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.sent >= self.n {
            return Ok(Flow::Eos);
        }
        for pad in [PadId(0), PadId(1)] {
            let Some(mut buf) = ctx.try_alloc(pad) else { return Ok(Flow::Ok) };
            buf.memory.set_len(1);
            buf.memory.as_mut_full()[0] = pad.0 as u8;
            ctx.out(pad).push(buf);
        }
        self.sent += 1;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "bytesink",
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

struct ByteSink {
    seen: Arc<Mutex<Vec<u8>>>,
}

impl Element for ByteSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut seen = self.seen.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            seen.push(buf.memory.data()[0]);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[test]
fn unlinked_pad_output_is_dropped_and_counted() {
    const ROUNDS: usize = 50;
    let seen = Arc::new(Mutex::new(Vec::new()));

    let mut p = Pipeline::new();
    let src = p.add(TwoPadSrc { n: ROUNDS, sent: 0 });
    let snk = p.add(ByteSink { seen: Arc::clone(&seen) });
    p.link((src, "a"), (snk, "sink")).expect("link pad a only");

    let tap = p.tap_handle();
    p.run().expect("run to EOS despite the unlinked pad");

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), ROUNDS, "the linked track arrived in full");
    assert!(seen.iter().all(|&b| b == 0), "only pad-a bytes reached the sink");

    let s = tap.snapshot(src).expect("source counters");
    assert_eq!(
        s.drops, ROUNDS as u64,
        "every pad-b buffer was dropped by policy and counted"
    );
    assert_eq!(s.buffers_out as usize, 2 * ROUNDS, "produced on both pads");
}
