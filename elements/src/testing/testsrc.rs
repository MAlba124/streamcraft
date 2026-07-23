//! `testsrc` — generates a reproducible byte pattern (no IO) then EOS.

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
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

static DESC: ElementDesc = ElementDesc {
    name: "testsrc",
    pads: &PADS,
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

pub struct TestSrc {
    total: u64,
    produced: u64,
}

impl TestSrc {
    /// Produce exactly `total_bytes` of the pattern, then EOS.
    pub fn new(total_bytes: u64) -> Self {
        Self {
            total: total_bytes,
            produced: 0,
        }
    }
}

impl Element for TestSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.produced >= self.total {
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
