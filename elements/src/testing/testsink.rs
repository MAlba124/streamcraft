//! `testsink` — folds received bytes into an FNV-1a hash and counts them, exposing
//! the result to the test via a shared [`TestSinkStats`] handle.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::time::Timestamp;

use super::{fold, FNV_OFFSET};

static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &[],
    dynamic: false,
    validate: None,
}];

static DESC: ElementDesc = ElementDesc {
    name: "testsink",
    pads: &PADS,
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

struct Shared {
    hash: AtomicU64,
    bytes: AtomicU64,
    done: AtomicBool,
}

/// A handle to a [`TestSink`]'s results, readable after `pipeline.run()` returns
/// (the sink thread has joined by then, so the values are final).
pub struct TestSinkStats(Arc<Shared>);

impl TestSinkStats {
    /// FNV-1a digest of every byte received, in order.
    pub fn hash(&self) -> u64 {
        self.0.hash.load(Ordering::Relaxed)
    }

    /// Total bytes received.
    pub fn bytes(&self) -> u64 {
        self.0.bytes.load(Ordering::Relaxed)
    }

    /// Whether the sink saw EOS (was stopped) — i.e. the stream ended cleanly.
    pub fn is_done(&self) -> bool {
        self.0.done.load(Ordering::Relaxed)
    }
}

pub struct TestSink {
    shared: Arc<Shared>,
    hash: u64,
    bytes: u64,
}

impl TestSink {
    /// Returns the sink (to `add` to a pipeline) and a stats handle to read after
    /// `run()`.
    pub fn new() -> (Self, TestSinkStats) {
        let shared = Arc::new(Shared {
            hash: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            done: AtomicBool::new(false),
        });
        let stats = TestSinkStats(Arc::clone(&shared));
        (
            Self {
                shared,
                hash: FNV_OFFSET,
                bytes: 0,
            },
            stats,
        )
    }
}

impl Element for TestSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            let data = buf.memory.data();
            for &b in data {
                self.hash = fold(self.hash, b);
            }
            self.bytes += data.len() as u64;
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.shared.hash.store(self.hash, Ordering::Relaxed);
        self.shared.bytes.store(self.bytes, Ordering::Relaxed);
        self.shared.done.store(true, Ordering::Relaxed);
    }
}
