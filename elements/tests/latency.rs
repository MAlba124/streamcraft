//! The latency system (spec: Latency — declared by elements, computed per path by
//! the pipeline, enforced at sinks). A `Delay` transform *declares* 20 ms of
//! minimum latency without ever sleeping: the proof that latency is a graph
//! property, not a runtime measurement. `latency_report()` must show the declared
//! sum on the sink's path, and a `TimedTestSink` downstream of the declaration
//! must render at `pts + path_latency` — the scheduler shifts its deadline; the
//! sink code never mentions latency.

use std::sync::Arc;
use std::time::Duration;

use profluens_core::batch::Inputs;
use profluens_core::clock::MockClock;
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
use profluens_elements::testing::{TimedTestSink, TimedTestSrc};

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

const DELAY: Timestamp = Timestamp::from_millis(20);

static DELAY_DESC: ElementDesc = ElementDesc {
    name: "delay20",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: DELAY,
        max: DELAY,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Declares 20 ms of latency but forwards instantly — latency is a *declaration*
/// the pipeline compensates for, not something an element performs.
struct Delay;

impl Element for Delay {
    fn desc(&self) -> &'static ElementDesc {
        &DELAY_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            ctx.out(PadId(0)).push(buf);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[test]
fn latency_report_sums_declared_latencies_per_path() {
    // Two sinks: one behind a 20 ms declaration, one not. The report carries one
    // entry per sink, worst path first, with the per-element breakdown in path
    // order — a pure graph computation, callable before any run (spec: Latency).
    let mut p = Pipeline::new();
    let src_a = p.add(TimedTestSrc::new(1, Timestamp::from_millis(10)));
    let delay = p.add(Delay);
    let (sink, _stats) = TimedTestSink::new();
    let snk_a = p.add(sink);
    p.link((src_a, "src"), (delay, "sink")).expect("src!delay");
    p.link((delay, "src"), (snk_a, "sink")).expect("delay!sink");

    let src_b = p.add(TimedTestSrc::new(1, Timestamp::from_millis(10)));
    let (sink_b, _stats_b) = TimedTestSink::new();
    let snk_b = p.add(sink_b);
    p.link((src_b, "src"), (snk_b, "sink")).expect("src_b!sink_b");

    let report = p.latency_report();
    assert_eq!(report.paths.len(), 2, "one path per sink");
    assert_eq!(report.total(), DELAY, "pipeline latency = worst sink path");

    let worst = &report.paths[0];
    assert_eq!(worst.sink, snk_a);
    assert_eq!(worst.total, DELAY);
    assert!(!worst.is_live);
    let names: Vec<u32> = worst.per_element.iter().map(|(el, _)| el.0).collect();
    assert_eq!(names, vec![src_a.0, delay.0, snk_a.0], "path in source→sink order");
    assert_eq!(worst.per_element[1], (delay, DELAY), "the delay's own declaration");

    let quiet = &report.paths[1];
    assert_eq!(quiet.sink, snk_b);
    assert_eq!(quiet.total, Timestamp::ZERO, "undeclared path adds nothing");
}

#[test]
fn sink_render_deadline_shifts_by_path_latency() {
    // One buffer at PTS 0. Without the latency declaration the sink would render
    // it the instant the pipeline starts (clock at 0 ≥ deadline 0). With 20 ms
    // declared upstream, the sink's wait lands on `pts + 20ms`, so the run must
    // stay parked until the mock clock actually reaches 20 ms.
    let clock = MockClock::new();
    let (sink, stats) = TimedTestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TimedTestSrc::new(1, Timestamp::from_millis(10)));
    let delay = p.add(Delay);
    let snk = p.add(sink);
    p.link((src, "src"), (delay, "sink")).expect("src!delay");
    p.link((delay, "src"), (snk, "sink")).expect("delay!sink");
    p.set_clock(Arc::new(clock.clone()));

    let run = std::thread::spawn(move || p.run());

    // The negative assertion: with the clock parked at 0 the PTS-0 buffer must NOT
    // render — its compensated deadline is 20 ms away. (Same short-poll pattern as
    // the clock tests.)
    for _ in 0..60 {
        if run.is_finished() {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(!run.is_finished(), "sink rendered before its latency-shifted deadline");
    assert_eq!(stats.count(), 0, "PTS-0 buffer held back by 20 ms of path latency");

    // Cross the compensated deadline: everything renders and the run completes.
    clock.advance(DELAY);
    run.join().expect("run joined").expect("run ok");

    let renders = stats.renders();
    assert_eq!(renders.len(), 1);
    assert_eq!(renders[0].pts, Timestamp::ZERO);
    assert!(
        renders[0].rendered_at >= DELAY,
        "rendered at running time {:?}, before pts + path latency",
        renders[0].rendered_at
    );
}
