//! Batches: the unit of work is a span, not a buffer (spec: Batching). SoA layout —
//! parallel arrays, so timestamp scans (QoS, latency, seek) touch one dense column.
//!
//! Milestone-1 transport is an owned [`Batch`] with `Vec` columns, reused across
//! iterations. Inputs are drained by *moving* owned buffers out ([`Inputs::pop`]),
//! because an active sink hands buffers to the reactor and must own them until the
//! write completes. The ring-buffer-slots-are-the-columns version (spec: Queue
//! internals) replaces the transport later behind [`BatchRef`] / [`OutBatch`].

use crate::buffer::{Buffer, BufferFlags};
use crate::event::Event;
use crate::id::{FormatId, MetaId};
use crate::memory::Memory;
use crate::time::Timestamp;

/// Owned SoA storage for a span of buffers. Format-homogeneous.
pub struct Batch {
    pub format: FormatId,
    pub memories: Vec<Memory>,
    pub pts: Vec<Timestamp>,
    pub durations: Vec<Timestamp>,
    pub flags: Vec<BufferFlags>,
    pub metas: Vec<Option<MetaRef>>,
    /// In-band events riding with this span, delivered to the downstream element's
    /// `event()` at the batch boundary *before* its buffers (spec: Events travel with
    /// buffers through the same queues). A `FormatChange` here is how a decoder announces
    /// a runtime format to its peer (spec: Formats — dynamic caps). Usually empty.
    pub events: Vec<Event>,
    /// The seek generation this span was produced under (spec: flush/seek). The scheduler
    /// stamps it on each batch crossing a ring and drops any batch whose generation is
    /// older than the current one on the consuming side — so a seek can discard the data
    /// already queued past the point of no return without racing the producer that is
    /// concurrently pushing fresh, post-seek data. `0` until the first seek.
    pub seek_gen: u64,
}

impl Batch {
    pub fn new(format: FormatId) -> Self {
        Self {
            format,
            memories: Vec::new(),
            pts: Vec::new(),
            durations: Vec::new(),
            flags: Vec::new(),
            metas: Vec::new(),
            events: Vec::new(),
            seek_gen: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Total used bytes across all buffers in the batch (for counters).
    pub fn total_bytes(&self) -> u64 {
        self.memories.iter().map(|m| m.len() as u64).sum()
    }

    /// Clear all columns, retaining capacity (drops the `Memory`s → recycled).
    pub fn clear(&mut self) {
        self.memories.clear();
        self.pts.clear();
        self.durations.clear();
        self.flags.clear();
        self.metas.clear();
        self.events.clear();
    }

    /// Append one buffer, decomposing it into the SoA columns.
    pub fn push(&mut self, buf: Buffer) {
        self.memories.push(buf.memory);
        self.pts.push(buf.pts);
        self.durations.push(buf.duration);
        self.flags.push(buf.flags);
        self.metas.push(None);
        // buf.dts / buf.format / buf.sync are not carried per-row in milestone 1.
    }

    /// Take the oldest buffer (FIFO), reconstructing it from the columns.
    pub fn pop_front(&mut self) -> Option<Buffer> {
        if self.memories.is_empty() {
            return None;
        }
        let _ = self.metas.remove(0);
        Some(Buffer {
            memory: self.memories.remove(0),
            pts: self.pts.remove(0),
            dts: Timestamp::NONE,
            duration: self.durations.remove(0),
            flags: self.flags.remove(0),
            format: self.format,
            sync: None,
        })
    }

    /// Move all buffers out of `other` (leaving it empty) onto the end of `self`,
    /// carrying its in-band events too so they stay ordered with the data.
    pub fn append(&mut self, other: &mut Batch) {
        self.memories.append(&mut other.memories);
        self.pts.append(&mut other.pts);
        self.durations.append(&mut other.durations);
        self.flags.append(&mut other.flags);
        self.metas.append(&mut other.metas);
        self.events.append(&mut other.events);
    }

    /// Attach an in-band event to this span (spec: Events travel with buffers).
    pub fn push_event(&mut self, event: Event) {
        self.events.push(event);
    }

    /// Take the in-band events off this batch, leaving it data-only.
    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    /// Whether this batch carries no buffers *and* no events (so pushing it downstream
    /// would be a no-op).
    pub fn is_inert(&self) -> bool {
        self.memories.is_empty() && self.events.is_empty()
    }

    pub fn view(&self) -> BatchRef<'_> {
        BatchRef {
            format: self.format,
            memories: &self.memories,
            pts: &self.pts,
            durations: &self.durations,
            flags: &self.flags,
            metas: &self.metas,
        }
    }
}

/// Borrowed SoA view over a batch. Cheap to copy (all fields are shared refs). Used
/// by passive elements that read input without taking ownership.
#[derive(Clone, Copy)]
pub struct BatchRef<'a> {
    pub format: FormatId,
    pub memories: &'a [Memory],
    pub pts: &'a [Timestamp],
    pub durations: &'a [Timestamp],
    pub flags: &'a [BufferFlags],
    pub metas: &'a [Option<MetaRef>],
}

impl<'a> BatchRef<'a> {
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    pub fn memory(&self, i: usize) -> &'a Memory {
        &self.memories[i]
    }
}

/// Points into a per-pool side arena of `(MetaId, POD payload)` entries
/// (spec: Metadata escape hatches).
#[derive(Clone, Copy)]
pub struct MetaRef {
    pub id: MetaId,
}

/// Writer into an element's output batch — building the batch *is* enqueueing it.
pub struct OutBatch<'a> {
    batch: &'a mut Batch,
}

impl<'a> OutBatch<'a> {
    pub(crate) fn new(batch: &'a mut Batch) -> Self {
        Self { batch }
    }

    pub fn push(&mut self, buf: Buffer) {
        self.batch.push(buf);
    }

    pub fn len(&self) -> usize {
        self.batch.len()
    }

    pub fn is_empty(&self) -> bool {
        self.batch.is_empty()
    }
}

/// The inputs delivered to `Element::process` (spec: Aggregation). Milestone 1 is a
/// single input batch, drained by *moving* owned buffers out via [`pop`](Self::pop).
pub struct Inputs<'a> {
    batch: Option<&'a mut Batch>,
}

impl<'a> Inputs<'a> {
    pub fn empty() -> Inputs<'static> {
        Inputs { batch: None }
    }

    pub fn owned(batch: &'a mut Batch) -> Self {
        Inputs { batch: Some(batch) }
    }

    pub fn is_empty(&self) -> bool {
        self.batch.as_ref().map_or(true, |b| b.is_empty())
    }

    pub fn len(&self) -> usize {
        self.batch.as_ref().map_or(0, |b| b.len())
    }

    /// Take the oldest input buffer (owned).
    pub fn pop(&mut self) -> Option<Buffer> {
        self.batch.as_mut().and_then(|b| b.pop_front())
    }

    /// Borrow the inputs without taking ownership (for passive read-only elements).
    pub fn view(&self) -> Option<BatchRef<'_>> {
        self.batch.as_ref().map(|b| b.view())
    }
}
