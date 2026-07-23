//! The element contract (spec: Elements and pads; Aggregation). One small
//! object-safe trait; everything *static* lives in a descriptor, not on the trait.

use crate::batch::Inputs;
use crate::ctx::Ctx;
use crate::error::Error;
use crate::event::Event;
use crate::format::{Constraint, FixedFormat, OfferDesc};
use crate::time::Timestamp;

pub trait Element: Send {
    /// Points at a shared `static` descriptor — the element's identity and shape.
    fn desc(&self) -> &'static ElementDesc;
    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error>;
    fn process(&mut self, ctx: &mut Ctx, inputs: Inputs<'_>) -> Result<Flow, Error>;
    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error>;
    fn stop(&mut self, ctx: &mut Ctx);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Flow {
    Ok,
    NeedMore,
    Eos,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Sink,
    Src,
}

/// Passive = pure transform, callable inline; Active = blocks/waits, needs its own
/// thread (spec: Scheduling and threading).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SchedHint {
    Passive,
    Active,
}

/// Who waits, and how (spec: Aggregation). The scheduler implements the waiting;
/// the element only declares the policy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InputPolicy {
    None,
    Single,
    Any,
    All { by: AlignBy },
    AllDeadline { by: AlignBy, timeout: Timestamp },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AlignBy {
    Dts,
    RunningTime,
}

pub struct LatencyDesc {
    pub min: Timestamp,
    pub max: Timestamp,
    pub is_live: bool,
    pub jitter: Timestamp,
}

pub struct PadDesc {
    pub name: &'static str,
    pub direction: Direction,
    /// The formats this pad supports, as a static, string-keyed list of alternatives
    /// (spec: Formats — open vocabulary). Interned to id-based
    /// [`FormatOffer`](crate::format::FormatOffer)s and solved at *link time*, never
    /// per-buffer. Declaration order is preference order. Empty = matches nothing;
    /// byte/passthrough pads should offer [`OfferDesc::any`].
    pub offers: &'static [OfferDesc],
    pub dynamic: bool,
    /// Fixation escape hatch for coupled constraints (spec: Formats).
    pub validate: Option<fn(&FixedFormat) -> bool>,
}

pub struct PropDesc {
    pub name: &'static str,
    /// Reuses the format algebra for validation.
    pub allowed: Constraint,
    /// Settable at batch boundaries while `Playing`.
    pub live: bool,
}

pub struct ElementDesc {
    pub name: &'static str,
    pub pads: &'static [PadDesc],
    pub props: &'static [PropDesc],
    pub sched: SchedHint,
    pub inputs: InputPolicy,
    pub latency: LatencyDesc,
    /// Only for the string/parse path (spec: Plugins): default-construct then apply
    /// parsed props. Typed `T::new(..)` is how elements are actually made.
    pub make_default: Option<fn() -> Box<dyn Element>>,
}

/// A construction-time subgraph expander (spec: No bins — templates). Expands into
/// plain elements + links, exporting named boundary pads. TODO(step 8).
pub struct Template {
    _priv: (),
}
