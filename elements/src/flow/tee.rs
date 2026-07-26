//! `tee` — one input stream fanned out to N output branches by refcount
//! (spec: Scheduling — branching; the branching test's throwaway tee, promoted).
//! Zero-copy: each branch receives the same [`Memory`] via a refcount bump, so
//! a tee never allocates from any pool and never copies payload bytes. Events
//! and buffer metadata (pts/flags) forward to every branch verbatim.
//!
//! Construction picks the branch count ([`Tee::new`], up to [`MAX_BRANCHES`]);
//! only `src_0..src_{n-1}` are pushed to — the static pad table's spares stay
//! silent (the `MkvMuxN` static-pad convention while dynamic sink/src growth
//! is core-side pending work).
//!
//! [`Memory`]: streamcraft_core::memory::Memory

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

/// The static pad table's src pad budget.
pub const MAX_BRANCHES: usize = 8;

/// Format-agnostic: a tee forwards anything — the wildcard offer intersects
/// any family (spec: Formats), and the branches' peers negotiate with whatever
/// the upstream announces (`forward_format` rides the events).
static OFFERS: [OfferDesc; 1] = [OfferDesc::wildcard()];

const fn src_pad(name: &'static str) -> PadDesc {
    PadDesc { name, direction: Direction::Src, offers: &OFFERS, dynamic: false, validate: None }
}

static PADS: [PadDesc; MAX_BRANCHES + 1] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &OFFERS, dynamic: false, validate: None },
    src_pad("src_0"),
    src_pad("src_1"),
    src_pad("src_2"),
    src_pad("src_3"),
    src_pad("src_4"),
    src_pad("src_5"),
    src_pad("src_6"),
    src_pad("src_7"),
];

static DESC: ElementDesc = ElementDesc {
    name: "tee",
    pads: &PADS,
    props: &[],
    // Active: a tee is a branch point — each src pad feeds its own downstream
    // group (the branching-test lesson: inter-group branches come off a group
    // tail, and an Active head is its own tail).
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

/// Fan one stream out to `n` branches, refcount-only.
pub struct Tee {
    n: usize,
}

impl Tee {
    /// A tee feeding `src_0..src_{n-1}` (1 ≤ n ≤ [`MAX_BRANCHES`]).
    pub fn new(n: usize) -> Tee {
        assert!(
            (1..=MAX_BRANCHES).contains(&n),
            "tee supports 1..={MAX_BRANCHES} branches, got {n}"
        );
        Tee { n }
    }
}

impl Element for Tee {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            // Branches 1..n get refcount clones of the Memory; the original
            // moves to branch 0 — n pushes, zero copies, zero allocs. `Buffer`
            // itself is not `Clone` (device `sync` fences are single-owner), so
            // the shell is rebuilt per branch; a device-synced buffer through a
            // tee keeps its fence on branch 0 only (spec: Device memory — CPU
            // buffers, the tee's use today, carry `sync: None` anyway).
            for i in 1..self.n {
                let copy = streamcraft_core::buffer::Buffer {
                    memory: buf.memory.clone(),
                    pts: buf.pts,
                    dts: buf.dts,
                    duration: buf.duration,
                    flags: buf.flags,
                    format: buf.format,
                    sync: None,
                };
                ctx.out(PadId(i as u32 + 1)).push(copy);
            }
            ctx.out(PadId(1)).push(buf);
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}
