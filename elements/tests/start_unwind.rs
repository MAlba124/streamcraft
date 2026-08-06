//! A group whose `start()` fails part-way must still stop the elements that already started.
//!
//! `element.rs` promises authors that start/stop run "exactly once, in order" and that they
//! "can't get it wrong", so nobody writes a `Drop` guard — an fd or device opened in `start()`
//! simply leaked when a later element's `start()` returned `Err`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] =
    [PadDesc { name: "src", direction: Direction::Src, offers: &SRC_OFFERS, dynamic: false, validate: None }];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "startsrc", pads: &SRC_PADS, props: &[], sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};
static SNK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SNK_PADS: [PadDesc; 1] =
    [PadDesc { name: "sink", direction: Direction::Sink, offers: &SNK_OFFERS, dynamic: false, validate: None }];
static SNK_DESC: ElementDesc = ElementDesc {
    name: "failsink", pads: &SNK_PADS, props: &[], sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

/// Starts successfully and records whether it was stopped.
struct Opener(Arc<AtomicUsize>);
impl Element for Opener {
    fn desc(&self) -> &'static ElementDesc { &SRC_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, _c: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> { Ok(Flow::Eos) }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> { Ok(()) }
    fn stop(&mut self, _c: &mut Ctx) { self.0.fetch_add(1, Ordering::SeqCst); }
}

/// Fails in `start()`, after the element above has already acquired its resource.
struct Failer;
impl Element for Failer {
    fn desc(&self) -> &'static ElementDesc { &SNK_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Err(Error::Todo("start refused")) }
    fn process(&mut self, _c: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> { Ok(Flow::Ok) }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> { Ok(()) }
    fn stop(&mut self, _c: &mut Ctx) {}
}

#[test]
fn a_failed_start_still_stops_the_elements_that_started() {
    let stopped = Arc::new(AtomicUsize::new(0));
    let mut p = Pipeline::new();
    let src = p.add(Opener(Arc::clone(&stopped)));
    let snk = p.add(Failer);
    p.link((src, "src"), (snk, "sink")).expect("link");
    assert!(p.run().is_err(), "the run must surface the start failure");
    assert_eq!(
        stopped.load(Ordering::SeqCst),
        1,
        "the element that started must be stopped when a later start() fails"
    );
}
