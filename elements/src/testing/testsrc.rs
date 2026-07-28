//! `testsrc` — generates a reproducible byte pattern (no IO) then EOS.

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, PropDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{Constraint, OfferDesc, Value};
use streamcraft_core::id::PadId;
use streamcraft_core::log;
use streamcraft_core::log::Level;
use streamcraft_core::time::Timestamp;

use super::pattern_byte;

/// A raw-byte stream: no fields, matches any peer that also speaks `bytes`.
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

/// Total bytes to produce (spec: Plugins — settable via `parse("testsrc total=…")`).
/// Structural (`live: false`): read once in `start()`, not a mid-stream knob.
static PROPS: [PropDesc; 1] = [PropDesc {
    name: "total",
    allowed: Constraint::Any,
    live: false,
}];

// COLD: make_default boxes one instance per registry-created element, never per buffer.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "testsrc",
    pads: &PADS,
    props: &PROPS,
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(TestSrc::new(0))),
};

pub struct TestSrc {
    total: u64,
    produced: u64,
    /// The scheduler may call `process()` again after `Flow::Eos` (it drives to
    /// quiescence); log the completion transition once, not per call.
    eos_logged: bool,
}

impl TestSrc {
    /// Produce exactly `total_bytes` of the pattern, then EOS.
    pub fn new(total_bytes: u64) -> Self {
        Self {
            total: total_bytes,
            produced: 0,
            eos_logged: false,
        }
    }
}

impl Element for TestSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // A parsed `total=` property overrides the constructor value (spec: Plugins —
        // elements read their props in `start()`, falling back to constructor config).
        if let Some(Value::Int(n)) = ctx.prop("total") {
            if n >= 0 {
                self.total = n as u64;
            }
        }
        log!(&*ctx, Level::Debug, "start", total = self.total);
        self.eos_logged = false;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.produced >= self.total {
            if !self.eos_logged {
                self.eos_logged = true;
                log!(&*ctx, Level::Info, "eos", produced = self.produced);
            }
            return Ok(Flow::Eos);
        }
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return Ok(Flow::Ok), // pool full → backpressure
        };
        let cap = buf.memory.capacity() as u64;
        let n = cap.min(self.total - self.produced);
        let dst = buf.memory.as_mut_full();
        for j in 0..n {
            dst[j as usize] = pattern_byte(self.produced + j);
        }
        buf.memory.set_len(n as usize);
        self.produced += n;
        ctx.out(PadId(0)).push(buf);
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}
