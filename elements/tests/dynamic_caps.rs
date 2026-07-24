//! Dynamic-caps runtime re-validation (spec: Formats — dynamic caps). A source whose
//! static offer is broad enough to link, but which then announces a runtime format the
//! downstream pad forbids, must fail loudly (a NotNegotiated-style bus error) rather
//! than silently install the bad format. A compatible announcement passes through.

use streamcraft_core::batch::Inputs;
use streamcraft_core::bus::BusMessage;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

// A source offering audio/raw at rate ∈ {44100, 48000} (so it links against a strict
// sink), which then announces a caller-chosen rate at runtime.
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
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

struct AnnounceSrc {
    rate: i64,
    announced: bool,
}

impl AnnounceSrc {
    fn new(rate: i64) -> Self {
        Self { rate, announced: false }
    }
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

// A sink that only accepts audio/raw at rate == 44100.
static SINK_FIELDS: [FieldDesc; 1] = [FieldDesc {
    field: "rate",
    allowed: ConstraintDesc::Eq(ValueDesc::Int(44100)),
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
    name: "strictsink",
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

struct StrictSink;

impl Element for StrictSink {
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
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[test]
fn incompatible_runtime_announcement_fails_loudly() {
    // Links fine (the sink pins 44100 out of the source's {44100, 48000}); at runtime
    // the source announces 48000, which the sink's rate=Eq(44100) offer forbids.
    let mut p = Pipeline::new();
    let src = p.add(AnnounceSrc::new(48000));
    let snk = p.add(StrictSink);
    p.link((src, "src"), (snk, "sink")).expect("links at 44100");

    let res = p.run();
    assert!(res.is_err(), "an incompatible announcement must fail the run");

    // A NotNegotiated-style error, attributed to the sink, reached the bus.
    let mut saw_sink_error = false;
    while let Some(msg) = p.bus().try_recv() {
        if let BusMessage::Error { element, .. } = msg {
            if element == snk {
                saw_sink_error = true;
            }
        }
    }
    assert!(
        saw_sink_error,
        "the sink posted a re-validation error to the bus"
    );
}

#[test]
fn compatible_runtime_announcement_is_accepted() {
    // The same graph, but the source announces the rate the sink accepts (44100).
    let mut p = Pipeline::new();
    let src = p.add(AnnounceSrc::new(44100));
    let snk = p.add(StrictSink);
    p.link((src, "src"), (snk, "sink")).expect("links at 44100");
    p.run().expect("a compatible announcement runs cleanly");
}
