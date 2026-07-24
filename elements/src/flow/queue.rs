//! `queue` — the explicit thread-decoupling boundary (spec: Scheduling and threading —
//! "real queues exist only at group boundaries: around active elements, at branches, and
//! wherever the user drops an explicit `queue` element").
//!
//! It is deliberately the emptiest element in the tree: an **[`Active`] passthrough with
//! [`wildcard`] pads and zero queue logic of its own**. All of its value comes from the
//! scheduler, not from this file: because it is `Active`, the scheduler gives it its own
//! thread group with an SPSC ring on each side (spec: Queue internals). Those two rings
//! *are* the queue — thread decoupling (the upstream and downstream run on different
//! threads, paced only by the rings' bounded backpressure) plus bounded buffering. This
//! element only shovels buffers from the input ring to the output ring; the ring, its
//! capacity, its leaky policy, and its wakeups all live in core.
//!
//! [`Active`]: SchedHint::Active
//! [`wildcard`]: OfferDesc::wildcard
//!
//! **What crosses it, verbatim.** Buffers pass through untouched. What the milestone-1
//! batch transport preserves per buffer — `pts`, `duration`, `flags` — is carried
//! through; what that transport already drops on *every* hop (`dts`, the per-buffer
//! `sync` handle) is dropped here too, not by the queue but by [`Batch::push`]/`pop_front`
//! (spec: Batching — the SoA transport carries these columns; `dts`/`sync` are a
//! milestone-1 omission, not a queue behaviour). Nothing is added on EOS: the queue emits
//! exactly what arrived and no more.
//!
//! **Events ride the batches** (spec: Events travel with buffers through the same
//! queues). A `FormatChange` announced upstream rides the source's output batch into the
//! queue's input ring, where the scheduler re-validates it against the queue's *sink*
//! pad — a wildcard, so it is admitted whatever the family (spec: Formats — a wildcard
//! pad admits any announced format). `elements/tests/queue.rs` proves that crossing and
//! re-validation with this element's `event()` a pure no-op.
//!
//! FOLLOW-UP (main session): the *onward* hop — the queue re-emitting that FormatChange
//! to its own downstream — is not yet expressible from an element. The scheduler consumes
//! an element's inbound events rather than auto-riding them to that element's output, and
//! the only element→downstream format primitive, [`Ctx::announce_format`], needs
//! `&'static str` field names a generic forwarder cannot recover from a resolved
//! `FixedFormat` (`Ctx` exposes no `field_name`, no `FixedFormat`-based re-announce, and
//! no event-forward). Forwarding a FormatChange verbatim across an Active element needs a
//! core primitive (scheduler tail-event forwarding or a `Ctx` event-forward), owned by
//! the same session that owns the per-queue-capacity plumbing below.
//!
//! **Props: none in v1.** A per-queue capacity / leaky-policy knob needs scheduler
//! plumbing (ring sizing is chosen when the group's rings are built) owned by the main
//! session; that is a follow-up, noted here so an element author does not reach for a
//! prop that the transport cannot yet honour.

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

/// Both pads are wildcards — the queue is family-agnostic, adopting whatever its
/// neighbours negotiate (spec: Formats — a wildcard pad adopts the peer's family). This
/// is what lets one `queue` sit in an audio, video, or byte path unchanged.
static OFFERS: [OfferDesc; 1] = [OfferDesc::wildcard()];

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

static DESC: ElementDesc = ElementDesc {
    name: "queue",
    pads: &PADS,
    props: &[],
    // Active: the whole point. The scheduler puts a ring on each side and runs the queue
    // on its own thread group, decoupling upstream from downstream (spec: Scheduling).
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    // A pure transport adds no processing latency of its own; the ring residency it
    // *does* add is computed by the pipeline from the ring's capacity (spec: Latency —
    // worst-case queue residency is known because queues are bounded), not declared here.
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(Queue::new())),
};

/// The explicit `queue` element: an [`Active`](SchedHint::Active) passthrough. Holds no
/// state — all buffering is in the scheduler's rings.
#[derive(Default)]
pub struct Queue;

impl Queue {
    pub fn new() -> Self {
        Self
    }
}

impl Element for Queue {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Forward every input buffer to the single src pad, verbatim — `pop`/`push`
        // carry pts/duration/flags through the SoA columns; no copy, no allocation.
        while let Some(buf) = inputs.pop() {
            ctx.out(PadId(0)).push(buf);
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        // Events ride the batches; the scheduler delivers them here and rides them to the
        // downstream ring itself (spec: Events travel with buffers through the same
        // queues). A pure transport neither consumes nor rewrites them — see the module
        // docs and `elements/tests/queue.rs::dynamic_caps_announce_crosses_the_queue`.
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}
