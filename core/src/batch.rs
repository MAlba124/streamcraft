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
///
/// Columns are private: consumed rows sit *before* the drain cursor (`head`), so
/// raw column access would see them. Read through [`view`](Self::view), drain
/// through [`pop_front`](Self::pop_front).
pub struct Batch {
    pub format: FormatId,
    memories: Vec<Memory>,
    pts: Vec<Timestamp>,
    durations: Vec<Timestamp>,
    flags: Vec<BufferFlags>,
    metas: Vec<Option<MetaRef>>,
    /// Drain cursor: rows before it have been handed out by [`pop_front`](Self::pop_front)
    /// (their `Memory` moved out, a dead husk left in the column). Popping is O(1) —
    /// no per-buffer column shifting (spec: Batching — queued hop budget). The columns
    /// compact (reset) when the last row is popped.
    head: usize,
    /// In-band events riding with this span, delivered to the downstream element's
    /// `event()` at the batch boundary *before* its buffers (spec: Events travel with
    /// buffers through the same queues). A `FormatChange` here is how a decoder announces
    /// a runtime format to its peer (spec: Formats — dynamic caps). Usually empty.
    events: Vec<Event>,
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
            head: 0,
            events: Vec::new(),
            seek_gen: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.memories.len() - self.head
    }

    pub fn is_empty(&self) -> bool {
        self.head == self.memories.len()
    }

    /// Total used bytes across all (undrained) buffers in the batch (for counters).
    pub fn total_bytes(&self) -> u64 {
        self.memories[self.head..].iter().map(|m| m.len() as u64).sum()
    }

    /// Clear all columns, retaining capacity (drops the `Memory`s → recycled).
    pub fn clear(&mut self) {
        self.reset_columns();
        self.events.clear();
    }

    /// Clear the data columns (not the events), resetting the drain cursor.
    fn reset_columns(&mut self) {
        self.memories.clear();
        self.pts.clear();
        self.durations.clear();
        self.flags.clear();
        self.metas.clear();
        self.head = 0;
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

    /// Take the oldest buffer (FIFO), reconstructing it from the columns. O(1): the
    /// drain cursor advances and the row's `Memory` is moved out; nothing shifts.
    pub fn pop_front(&mut self) -> Option<Buffer> {
        if self.head == self.memories.len() {
            return None;
        }
        let i = self.head;
        self.head += 1;
        let buf = Buffer {
            memory: self.memories[i].take(),
            pts: self.pts[i],
            dts: Timestamp::NONE,
            duration: self.durations[i],
            flags: self.flags[i],
            format: self.format,
            sync: None,
        };
        if self.head == self.memories.len() {
            self.reset_columns(); // fully drained — drop the husks, reuse capacity
        }
        Some(buf)
    }

    /// Move all (undrained) buffers out of `other` (leaving it empty) onto the end of
    /// `self`, carrying its in-band events too so they stay ordered with the data.
    pub fn append(&mut self, other: &mut Batch) {
        let h = other.head;
        self.memories.extend(other.memories.drain(h..));
        self.pts.extend_from_slice(&other.pts[h..]);
        self.durations.extend_from_slice(&other.durations[h..]);
        self.flags.extend_from_slice(&other.flags[h..]);
        self.metas.extend_from_slice(&other.metas[h..]);
        self.events.append(&mut other.events);
        other.reset_columns();
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
        self.is_empty() && self.events.is_empty()
    }

    /// The undrained rows, as parallel SoA slices.
    pub fn view(&self) -> BatchRef<'_> {
        BatchRef {
            format: self.format,
            memories: &self.memories[self.head..],
            pts: &self.pts[self.head..],
            durations: &self.durations[self.head..],
            flags: &self.flags[self.head..],
            metas: &self.metas[self.head..],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Pool;

    fn buf(pool: &Pool, fill: u8, len: usize) -> Buffer {
        let mut memory = pool.acquire();
        memory.as_mut_full()[..len].fill(fill);
        memory.set_len(len);
        Buffer {
            memory,
            pts: Timestamp::from_nanos(fill as u64),
            dts: Timestamp::NONE,
            duration: Timestamp::NONE,
            flags: BufferFlags::empty(),
            format: FormatId(0),
            sync: None,
        }
    }

    #[test]
    fn pop_front_is_fifo_and_view_tracks_the_cursor() {
        let pool = Pool::bounded(64, 8);
        let mut b = Batch::new(FormatId(0));
        for (fill, len) in [(1u8, 10usize), (2, 20), (3, 30)] {
            b.push(buf(&pool, fill, len));
        }
        assert_eq!(b.len(), 3);
        assert_eq!(b.total_bytes(), 60);

        let first = b.pop_front().unwrap();
        assert_eq!(first.memory.data()[0], 1, "FIFO: oldest first");
        assert_eq!(b.len(), 2);
        assert_eq!(b.total_bytes(), 50, "drained rows don't count");
        let v = b.view();
        assert_eq!(v.len(), 2, "view starts at the cursor");
        assert_eq!(v.memories[0].data()[0], 2);
        assert_eq!(v.pts[0], Timestamp::from_nanos(2));

        assert_eq!(b.pop_front().unwrap().memory.data()[0], 2);
        assert_eq!(b.pop_front().unwrap().memory.data()[0], 3);
        assert!(b.pop_front().is_none());
        assert!(b.is_empty());
    }

    #[test]
    fn drained_memory_recycles_immediately() {
        // The popped buffer owns the payload; the husk left in the column must not
        // pin the pool slot (a bounded pool would otherwise starve mid-drain).
        let pool = Pool::bounded(64, 2);
        let mut b = Batch::new(FormatId(0));
        b.push(buf(&pool, 1, 1));
        b.push(buf(&pool, 2, 1));
        assert!(pool.try_acquire().is_none(), "pool exhausted");
        let first = b.pop_front().unwrap();
        drop(first); // recycles even though the batch still holds the husk row
        assert!(pool.try_acquire().is_some(), "slot came back while batch part-drained");
    }

    #[test]
    fn push_after_partial_drain_appends_undrained() {
        let pool = Pool::bounded(64, 8);
        let mut b = Batch::new(FormatId(0));
        b.push(buf(&pool, 1, 1));
        b.push(buf(&pool, 2, 1));
        let _ = b.pop_front();
        b.push(buf(&pool, 3, 1));
        assert_eq!(b.len(), 2);
        assert_eq!(b.pop_front().unwrap().memory.data()[0], 2);
        assert_eq!(b.pop_front().unwrap().memory.data()[0], 3);
        assert!(b.pop_front().is_none());
    }

    #[test]
    fn append_moves_only_undrained_rows_and_events() {
        let pool = Pool::bounded(64, 8);
        let mut a = Batch::new(FormatId(0));
        let mut b = Batch::new(FormatId(0));
        b.push(buf(&pool, 1, 1));
        b.push(buf(&pool, 2, 1));
        b.push(buf(&pool, 3, 1));
        let _ = b.pop_front(); // 1 is gone
        b.push_event(Event::Eos);
        a.append(&mut b);
        assert!(b.is_empty() && b.take_events().is_empty(), "source emptied");
        assert_eq!(a.len(), 2, "only undrained rows moved");
        assert_eq!(a.take_events().len(), 1, "events ride along");
        assert_eq!(a.pop_front().unwrap().memory.data()[0], 2);
        assert_eq!(a.pop_front().unwrap().memory.data()[0], 3);
    }

    #[test]
    fn clear_resets_cursor_and_recycles() {
        let pool = Pool::bounded(64, 2);
        let mut b = Batch::new(FormatId(0));
        b.push(buf(&pool, 1, 1));
        b.push(buf(&pool, 2, 1));
        let _ = b.pop_front();
        b.clear();
        assert!(b.is_empty());
        assert!(pool.try_acquire().is_some(), "clear recycled the undrained row");
        b.push(buf(&pool, 4, 1));
        assert_eq!(b.pop_front().unwrap().memory.data()[0], 4);
    }
}
