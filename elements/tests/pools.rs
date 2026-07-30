//! Per-element output pools (spec: Formats — "pool negotiation is decoupled";
//! `Pipeline::set_element_pool` is its explicit v1). One pipeline-wide slot size
//! cannot serve a demuxer's small samples and a decoder's multi-MB frames — the
//! mismatch has cost real gigabytes. Here two producers in one pipeline get
//! different pools, observable as the capacity of the buffers their consumers
//! receive; an element without an override stays on the default pool.

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
    name: "capsrc",
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

/// Emits `n` buffers from its pool, then EOS — each buffer's capacity reveals
/// which pool served it.
struct CapSrc {
    n: usize,
    sent: usize,
}

impl Element for CapSrc {
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
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        buf.memory.set_len(1);
        ctx.out(PadId(0)).push(buf);
        self.sent += 1;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static SINK_DESC: ElementDesc = ElementDesc {
    name: "capsink",
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

/// Records the capacity of every buffer it receives.
struct CapSink {
    caps: Arc<Mutex<Vec<usize>>>,
}

impl Element for CapSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut caps = self.caps.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            caps.push(buf.memory.capacity());
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[test]
fn per_element_pools_serve_their_own_slot_sizes() {
    // Two independent chains in one pipeline: source A overridden to 4 KiB slots,
    // source B left on the (1 KiB) default.
    let mut p = Pipeline::new();
    p.set_pool(1024, 16);

    let a = p.add(CapSrc { n: 5, sent: 0 });
    let caps_a = Arc::new(Mutex::new(Vec::new()));
    let sink_a = p.add(CapSink { caps: Arc::clone(&caps_a) });
    p.link((a, "src"), (sink_a, "sink")).expect("a!sink_a");

    let b = p.add(CapSrc { n: 5, sent: 0 });
    let caps_b = Arc::new(Mutex::new(Vec::new()));
    let sink_b = p.add(CapSink { caps: Arc::clone(&caps_b) });
    p.link((b, "src"), (sink_b, "sink")).expect("b!sink_b");

    p.set_element_pool(a, 4096, 8);
    p.run().expect("run");

    let ca = caps_a.lock().unwrap();
    let cb = caps_b.lock().unwrap();
    assert_eq!(ca.len(), 5);
    assert_eq!(cb.len(), 5);
    assert!(ca.iter().all(|&c| c == 4096), "override pool served A: {ca:?}");
    assert!(cb.iter().all(|&c| c == 1024), "default pool served B: {cb:?}");
}
