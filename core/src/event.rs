//! In-band events: travel *with* buffers through the same queues, ordered relative
//! to them (spec: Events, queries, and the bus). Delivered to `Element::event()`
//! at batch boundaries. There are no "non-serialized" events.

use crate::format::FixedFormat;
use crate::time::Timestamp;

pub enum Event {
    Segment { base: Timestamp, rate: f64 },
    FormatChange(FixedFormat),
    /// "Nothing through running time T" — the sparse-stream heartbeat, so a silent
    /// subtitle/audio track never stalls preroll or aggregation (spec: GAP protocol).
    Gap { until: Timestamp },
    Tags(TagList),
    Eos,
    FlushStart,
    FlushStop,
}

/// Interned-key → value pairs, plus a blob reference for cover art. TODO(step 5).
pub struct TagList {
    _priv: (),
}
