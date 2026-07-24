//! Timed test elements (spec: Clocking and synchronization — the timed sink).
//!
//! [`TimedTestSrc`] emits a fixed number of tiny buffers stamped with increasing
//! running-time PTS (`i * period`), then EOS — a deterministic, IO-free producer.
//! [`TimedTestSink`] is an *active* sink that renders each buffer **on the pipeline
//! clock**: it `ctx.wait_until(pts)`s before recording it, so against a `MockClock` a
//! test drives hours of virtual time in milliseconds of wall time and asserts nothing
//! was ever rendered ahead of its deadline (spec: Testing — nothing ever sleeps).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use streamcraft_core::batch::Inputs;
use streamcraft_core::clock::WaitOutcome;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

/// A raw-byte stream: no fields, matches any peer that also speaks `bytes`.
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

// ---------------------------------------------------------------------------
// source
// ---------------------------------------------------------------------------

static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

static SRC_DESC: ElementDesc = ElementDesc {
    name: "timedtestsrc",
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

/// Emits `count` tiny buffers with `pts = i * period` (and matching `duration`), then
/// EOS. The payload is the little-endian buffer index, so a sink can double-check
/// order independently of the PTS.
pub struct TimedTestSrc {
    count: u64,
    period: Timestamp,
    produced: u64,
}

impl TimedTestSrc {
    /// Produce exactly `count` buffers spaced `period` apart in running time.
    pub fn new(count: u64, period: Timestamp) -> Self {
        Self {
            count,
            period,
            produced: 0,
        }
    }
}

impl Element for TimedTestSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.produced >= self.count {
            return Ok(Flow::Eos);
        }
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return Ok(Flow::Ok), // pool full → backpressure
        };
        let idx = self.produced;
        let bytes = idx.to_le_bytes();
        let dst = buf.memory.as_mut_full();
        let n = bytes.len().min(dst.len());
        dst[..n].copy_from_slice(&bytes[..n]);
        buf.memory.set_len(n);
        buf.pts = Timestamp::from_nanos(self.period.0.saturating_mul(idx));
        buf.duration = self.period;
        self.produced += 1;
        ctx.out(PadId(0)).push(buf);
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// ---------------------------------------------------------------------------
// sink
// ---------------------------------------------------------------------------

static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

static SINK_DESC: ElementDesc = ElementDesc {
    name: "timedtestsink",
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

/// One rendered buffer: the PTS it carried, and the running time the clock read at the
/// moment it was released for rendering. `rendered_at >= pts` always holds — the sink
/// never renders ahead of the clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Render {
    pub pts: Timestamp,
    pub rendered_at: Timestamp,
}

struct SinkShared {
    renders: Mutex<Vec<Render>>,
    count: AtomicU64,
    done: AtomicBool,
    interrupted: AtomicBool,
}

/// A handle to a [`TimedTestSink`]'s results, readable after the sink thread joins
/// (i.e. after `pipeline.run()` returns).
pub struct TimedSinkStats(Arc<SinkShared>);

impl TimedSinkStats {
    /// Every rendered buffer, in receive order.
    pub fn renders(&self) -> Vec<Render> {
        self.0.renders.lock().unwrap().clone()
    }

    /// Number of buffers rendered so far.
    pub fn count(&self) -> u64 {
        self.0.count.load(Ordering::Acquire)
    }

    /// Whether the sink saw EOS (the stream ended cleanly).
    pub fn is_done(&self) -> bool {
        self.0.done.load(Ordering::Acquire)
    }

    /// Whether a clock wait was cut short by an interrupt (flush / shutdown).
    pub fn was_interrupted(&self) -> bool {
        self.0.interrupted.load(Ordering::Acquire)
    }
}

/// An active sink that renders each buffer **on the pipeline clock**: it waits until
/// running time reaches the buffer's PTS (`ctx.wait_until(pts)`) before recording it,
/// so it paces the whole chain via backpressure exactly as a real device sink does.
pub struct TimedTestSink {
    shared: Arc<SinkShared>,
}

impl TimedTestSink {
    /// Returns the sink (to `add` to a pipeline) and a stats handle to read after
    /// `run()`.
    pub fn new() -> (Self, TimedSinkStats) {
        let shared = Arc::new(SinkShared {
            renders: Mutex::new(Vec::new()),
            count: AtomicU64::new(0),
            done: AtomicBool::new(false),
            interrupted: AtomicBool::new(false),
        });
        let stats = TimedSinkStats(Arc::clone(&shared));
        (Self { shared }, stats)
    }
}

impl Element for TimedTestSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            // Render on the clock: block until running time reaches this buffer's PTS.
            match ctx.wait_until(buf.pts) {
                WaitOutcome::Reached => {}
                WaitOutcome::Interrupted => {
                    self.shared.interrupted.store(true, Ordering::Release);
                    return Ok(Flow::Ok);
                }
            }
            let render = Render {
                pts: buf.pts,
                rendered_at: ctx.now(),
            };
            self.shared.renders.lock().unwrap().push(render);
            self.shared.count.fetch_add(1, Ordering::AcqRel);
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.shared.done.store(true, Ordering::Release);
    }
}
