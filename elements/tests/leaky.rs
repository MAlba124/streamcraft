//! Leaky inbound rings (spec: Queues — leaky modes for live sources):
//! `Pipeline::set_queue_leaky(el, DropNewest)` makes the ring feeding `el` drop
//! incoming batches when full instead of blocking the producer — a live source
//! must never stall on a slow consumer. Drops surface in the *producer's* drops
//! counter; the default everywhere else stays lossless blocking backpressure.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use streamcraft_core::batch::Inputs;
use streamcraft_core::clock::MockClock;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::ring::Leaky;
use streamcraft_core::time::Timestamp;
use streamcraft_elements::testing::TimedTestSink;

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "livesrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: true, jitter: Timestamp::ZERO },
    make_default: None,
};

/// A "live" source: must emit all `n` buffers without ever blocking on downstream.
struct LiveSrc {
    n: usize,
    sent: Arc<AtomicUsize>,
}

impl Element for LiveSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        let sent = self.sent.load(Ordering::Relaxed);
        if sent >= self.n {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        buf.memory.set_len(1);
        buf.pts = Timestamp::from_millis(sent as u64);
        ctx.out(PadId(0)).push(buf);
        self.sent.store(sent + 1, Ordering::Relaxed);
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// A clock-parked sink behind a leaky ring: the live source finishes its whole
/// schedule without ever blocking, batches drop (counted on the source), and the
/// sink renders only what survived.
#[test]
fn leaky_ring_never_blocks_the_live_source() {
    const COUNT: usize = 400;
    let clock = MockClock::new();
    let (sink, stats) = TimedTestSink::new();
    let sent = Arc::new(AtomicUsize::new(0));

    let mut p = Pipeline::new();
    let src = p.add(LiveSrc { n: COUNT, sent: Arc::clone(&sent) });
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    // Ample pool so the ring is the binding constraint being tested.
    p.set_element_pool(src, 64, 1024);
    p.set_queue_leaky(snk, Leaky::DropNewest);
    p.set_clock(Arc::new(clock.clone()));
    let tap = p.tap_handle();

    let run = std::thread::spawn(move || p.run());

    // The sink parks on buffer 1's deadline (pts 1 ms, clock at 0) — with a
    // *blocking* ring the source would wedge within a few batches. Leaky: the
    // source must finish all 400 sends while the clock never moves.
    let mut settled = false;
    for _ in 0..500 {
        if sent.load(Ordering::Relaxed) >= COUNT {
            settled = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(settled, "live source finished without blocking (sent {})", sent.load(Ordering::Relaxed));

    // Release the sink; the run completes with only the surviving buffers rendered.
    while !run.is_finished() {
        clock.advance(Timestamp::from_millis(50));
        std::thread::yield_now();
    }
    run.join().expect("joined").expect("run ok");

    let rendered = stats.renders().len();
    let drops = tap.snapshot(src).expect("src counters").drops;
    assert!(rendered < COUNT, "a parked sink cannot have kept up ({rendered} of {COUNT})");
    assert!(drops > 0, "dropped batches surfaced in the producer's drops counter");
    assert!(rendered > 0, "the ring still delivered what fit");
}
