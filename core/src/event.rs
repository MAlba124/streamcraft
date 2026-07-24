//! In-band events: travel *with* buffers through the same queues, ordered relative
//! to them (spec: Events, queries, and the bus). Delivered to `Element::event()`
//! at batch boundaries. There are no "non-serialized" events — the out-of-band
//! cases (flush/seek, a live property set) are pipeline operations the *scheduler*
//! delivers through the same `event()` door at the same batch-boundary safe point.

use crate::format::{FixedFormat, Value};
use crate::time::Timestamp;

pub enum Event {
    Segment { base: Timestamp, rate: f64 },
    FormatChange(FixedFormat),
    /// "Nothing through running time T" — the sparse-stream heartbeat, so a silent
    /// subtitle/audio track never stalls preroll or aggregation (spec: GAP protocol).
    Gap { until: Timestamp },
    Tags(TagList),
    /// A live property changed (spec: Dynamic element properties). Scheduler-delivered
    /// at the batch boundary — never mid-buffer; a set issued while batch N is in
    /// flight takes effect no earlier than N+1. `name` is the entry from this
    /// element's own `desc().props`; the value has already been validated against
    /// that entry's constraint. Elements that re-read via
    /// [`Ctx::prop`](crate::ctx::Ctx::prop) each batch may ignore this event.
    PropChanged { name: &'static str, value: Value },
    Eos,
    FlushStart,
    FlushStop,
    /// The transport paused (spec: Clocking — "pause is a clock op, not a state"):
    /// running time freezes, the scheduler parks the group after delivering this.
    /// A device sink reacts by holding its hardware (render silence, keep the ring);
    /// most elements ignore it.
    Paused,
    /// The transport resumed: running time continues (the pause interval is excised
    /// by re-basing), delivered just before the group runs again.
    Resumed,
}

/// Interned-key → value pairs, plus a blob reference for cover art. TODO(step 5).
pub struct TagList {
    _priv: (),
}
