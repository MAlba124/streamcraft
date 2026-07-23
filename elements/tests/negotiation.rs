//! End-to-end format negotiation (spec: Formats — the caps-without-caps system).
//!
//! Two tiny typed elements declare concrete `audio/raw` offers as *static, string-
//! keyed* descriptors; `Pipeline::link` interns and intersects them, storing a
//! [`FixedFormat`] on the edge. The sink reads its negotiated format from `Ctx` in
//! `start()` and records it, so the test can assert the whole path — descriptor →
//! link-time intern → solve → `ctx.negotiated()` — landed on the expected values.
//!
//! Also covers the negative case: incompatible offers make `link()` fail loudly, and
//! a mismatched family fails too.

use std::sync::{Arc, Mutex};

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{
    ConstraintDesc, FieldDesc, FixedFormat, OfferDesc, Value, ValueDesc,
};
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

const ZERO_LAT: LatencyDesc = LatencyDesc {
    min: Timestamp::ZERO,
    max: Timestamp::ZERO,
    is_live: false,
    jitter: Timestamp::ZERO,
};

// --- A typed audio source: 44.1k or 48k, prefers 48k; s16 or f32 samples. ---------

static SRC_RATES: [ValueDesc; 2] = [ValueDesc::Int(44100), ValueDesc::Int(48000)];
static SRC_SAMPLES: [ValueDesc; 2] = [ValueDesc::Id("s16"), ValueDesc::Id("f32")];
static SRC_FIELDS: [FieldDesc; 2] = [
    FieldDesc {
        field: "rate",
        allowed: ConstraintDesc::Set(&SRC_RATES),
        preferred: Some(ValueDesc::Int(48000)),
    },
    FieldDesc {
        field: "sample",
        allowed: ConstraintDesc::Set(&SRC_SAMPLES),
        preferred: None,
    },
];
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc {
    family: "audio/raw",
    fields: &SRC_FIELDS,
}];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &SRC_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "audiotestsrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: ZERO_LAT,
    make_default: None,
};

#[derive(Default)]
struct AudioSrc;

impl Element for AudioSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        Ok(Flow::Eos) // negotiation is the point here, not data
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// --- A typed audio sink: fixed 48k, s16 — records what it negotiated. --------------

static SINK_FIELDS: [FieldDesc; 2] = [
    FieldDesc {
        field: "rate",
        allowed: ConstraintDesc::Eq(ValueDesc::Int(48000)),
        preferred: None,
    },
    FieldDesc {
        field: "sample",
        allowed: ConstraintDesc::Eq(ValueDesc::Id("s16")),
        preferred: None,
    },
];
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc {
    family: "audio/raw",
    fields: &SINK_FIELDS,
}];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "audiotestsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: ZERO_LAT,
    make_default: None,
};

/// The sink copies the format it saw in `start()` here, so the test can inspect what
/// `ctx.negotiated()` returned on the running element (not just the pipeline's edge).
struct AudioSink {
    seen: Arc<Mutex<Option<FixedFormat>>>,
}

impl Element for AudioSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // The negotiated format is available to the element the moment it starts.
        *self.seen.lock().unwrap() = ctx.negotiated(PadId(0)).cloned();
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// --- A deliberately incompatible sink: same family, but a rate nobody offers. ------

static BAD_FIELDS: [FieldDesc; 1] = [FieldDesc {
    field: "rate",
    allowed: ConstraintDesc::Eq(ValueDesc::Int(96000)), // src offers only 44100/48000
    preferred: None,
}];
static BAD_OFFERS: [OfferDesc; 1] = [OfferDesc {
    family: "audio/raw",
    fields: &BAD_FIELDS,
}];
static BAD_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &BAD_OFFERS,
    dynamic: false,
    validate: None,
}];
static BAD_DESC: ElementDesc = ElementDesc {
    name: "badsink",
    pads: &BAD_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: ZERO_LAT,
    make_default: None,
};

#[derive(Default)]
struct BadSink;

impl Element for BadSink {
    fn desc(&self) -> &'static ElementDesc {
        &BAD_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// --- A sink in a *different family* — negotiation must also reject this. ------------

static TEXT_OFFERS: [OfferDesc; 1] = [OfferDesc::any("text/raw")];
static TEXT_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &TEXT_OFFERS,
    dynamic: false,
    validate: None,
}];
static TEXT_DESC: ElementDesc = ElementDesc {
    name: "textsink",
    pads: &TEXT_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: ZERO_LAT,
    make_default: None,
};

#[derive(Default)]
struct TextSink;

impl Element for TextSink {
    fn desc(&self) -> &'static ElementDesc {
        &TEXT_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[test]
fn typed_audio_negotiates_to_expected_fixed_format() {
    let seen = Arc::new(Mutex::new(None));
    let mut p = Pipeline::new();
    let src = p.add(AudioSrc);
    let snk = p.add(AudioSink { seen: Arc::clone(&seen) });
    let link = p.link((src, "src"), (snk, "sink")).expect("compatible offers negotiate");

    // The pipeline stored the fixed format on the edge: 48000 Hz (src's preference,
    // which the sink also mandates) and s16 samples (the only sample both accept).
    let rate = p.field_id("rate").expect("rate field seen");
    let sample = p.field_id("sample").expect("sample field seen");
    let s16 = p.value_id("s16").expect("s16 value seen");

    let fixed = p.negotiated(link).expect("edge has a fixed format");
    assert_eq!(
        p.family_name(fixed.family),
        Some("audio/raw"),
        "family is the one both pads share"
    );
    assert_eq!(fixed.get(rate), Some(Value::Int(48000)), "rate fixated to 48k");
    assert_eq!(fixed.get(sample), Some(Value::Id(s16)), "sample fixated to s16");
    assert_eq!(fixed.fields().len(), 2, "exactly the two shared fields are fixed");

    // And the *element* saw the same format via ctx.negotiated() when it started.
    p.run().expect("run drains immediately (src goes straight to EOS)");
    let element_saw = seen.lock().unwrap().clone().expect("sink recorded its format");
    assert_eq!(element_saw.get(rate), Some(Value::Int(48000)));
    assert_eq!(element_saw.get(sample), Some(Value::Id(s16)));
}

#[test]
fn incompatible_rate_fails_negotiation_at_link() {
    let mut p = Pipeline::new();
    let src = p.add(AudioSrc);
    let bad = p.add(BadSink);
    // Same family, but the sink demands 96000 and the src offers only 44100/48000:
    // the field intersection is empty, so link() must fail loudly (never at run).
    let err = p.link((src, "src"), (bad, "sink")).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("no common format") || msg.contains("negotiation"),
        "error should explain the empty intersection, got: {msg}"
    );
}

#[test]
fn family_mismatch_fails_negotiation_at_link() {
    let mut p = Pipeline::new();
    let src = p.add(AudioSrc);
    let txt = p.add(TextSink);
    // audio/raw vs text/raw share no family — negotiation fails.
    assert!(
        p.link((src, "src"), (txt, "sink")).is_err(),
        "different families must not link"
    );
}
