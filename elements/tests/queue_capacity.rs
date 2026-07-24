//! Per-element inbound ring capacity (`Pipeline::set_queue_capacity`, spec: Queues
//! — capacity belongs to the scheduler's rings, not to element code). Two
//! identical chains: a free-running source into a sink parked on the clock. The
//! only difference is the sink's inbound ring depth — the deeper ring must let its
//! source run measurably further ahead before backpressure parks it.

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
    name: "countingsrc",
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

/// Free-runs `n` tiny buffers with a 1 ms PTS grid, exposing how far it got — the
/// probe for where backpressure parked it.
struct CountingSrc {
    n: usize,
    sent: Arc<AtomicUsize>,
}

impl Element for CountingSrc {
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

const COUNT: usize = 400;

/// Build one chain; the sink parks on buffer 1 (PTS 1 ms) with the clock at 0, the
/// source free-runs until its downstream ring is full. Returns how many buffers the
/// source managed to emit before settling, then releases the clock and joins.
fn settled_progress(queue_cap: Option<usize>) -> usize {
    let clock = MockClock::new();
    let (sink, _stats) = TimedTestSink::new();
    let sent = Arc::new(AtomicUsize::new(0));

    let mut p = Pipeline::new();
    let src = p.add(CountingSrc { n: COUNT, sent: Arc::clone(&sent) });
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    // Ample identical pools for both variants, so the ring — not the pool — is the
    // binding backpressure being measured.
    p.set_element_pool(src, 64, 1024);
    if let Some(cap) = queue_cap {
        p.set_queue_capacity(snk, cap);
    }
    p.set_clock(Arc::new(clock.clone()));

    let run = std::thread::spawn(move || p.run());

    // Settle: the count is stable once the ring is full and the source is parked.
    let mut last = usize::MAX;
    loop {
        let now = sent.load(Ordering::Relaxed);
        if now == last {
            break;
        }
        last = now;
        std::thread::sleep(Duration::from_millis(20));
    }

    clock.advance(Timestamp::from_millis(COUNT as u64));
    run.join().expect("run joined").expect("run ok");
    last
}

#[test]
fn deeper_inbound_ring_lets_the_source_run_further_ahead() {
    let shallow = settled_progress(None); // default queue_cap = 4 batches
    let deep = settled_progress(Some(64));
    assert!(
        deep >= shallow + 16,
        "capacity override had no effect: default ring parked the source at \
         {shallow} buffers, 64-batch ring at {deep}"
    );
    assert!(
        deep < COUNT,
        "source was never backpressured at all ({deep} of {COUNT}) — the ring \
         cap is not binding and this test proves nothing"
    );
}
