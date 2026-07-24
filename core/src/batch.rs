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
    /// In-band events riding with this span, **positioned**: `(row, event)` means the
    /// event precedes row `row` (spec: Events travel with buffers, *ordered relative to
    /// them* — a mid-batch `FormatChange` applies exactly from its position, not to the
    /// whole batch). The drain cursor treats an undelivered event as a barrier:
    /// [`pop_front`](Self::pop_front) refuses to hand out row `row` until the scheduler
    /// has taken the event via [`due_event`](Self::due_event). Sorted by position
    /// (pushes stamp the current end, which is monotone). Usually empty.
    events: Vec<(u32, Event)>,
    /// The seek generation this span was produced under (spec: flush/seek). The scheduler
    /// stamps it on each batch crossing a ring and drops any batch whose generation is
    /// older than the current one on the consuming side — so a seek can discard the data
    /// already queued past the point of no return without racing the producer that is
    /// concurrently pushing fresh, post-seek data. `0` until the first seek.
    pub seek_gen: u64,
    /// When latency tracing is on (spec: Debuggability), the wall-clock instant the
    /// scheduler pushed this batch onto an inter-group ring — measured against the pop
    /// on the consuming side for ring residency. `None` when tracing is off (the
    /// untraced hot path never reads a clock for it).
    pub(crate) pushed_at: Option<std::time::Instant>,
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
            pushed_at: None,
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
        self.pushed_at = None;
    }

    /// Clear the data columns (not the events), resetting the drain cursor. Remaining
    /// event positions rebase by the drained rows (an end-of-batch event becomes due
    /// at 0 once every row is gone).
    fn reset_columns(&mut self) {
        let drained = self.head as u32;
        self.memories.clear();
        self.pts.clear();
        self.durations.clear();
        self.flags.clear();
        self.metas.clear();
        self.head = 0;
        for (pos, _) in &mut self.events {
            *pos = pos.saturating_sub(drained);
        }
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
    /// An undelivered in-band event at the cursor is a **barrier**: `None` until the
    /// scheduler delivers it ([`due_event`](Self::due_event)) — this is what makes a
    /// mid-batch `FormatChange` land between exactly the right two buffers.
    pub fn pop_front(&mut self) -> Option<Buffer> {
        if self.head == self.memories.len() {
            return None;
        }
        if let Some(&(pos, _)) = self.events.first() {
            if pos as usize <= self.head {
                return None; // event due before this row — barrier until delivered
            }
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
    /// `self`, carrying its in-band events too — re-positioned so they stay between
    /// exactly the same buffers they arrived between.
    pub fn append(&mut self, other: &mut Batch) {
        let h = other.head;
        let base = self.memories.len() as u32;
        self.memories.extend(other.memories.drain(h..));
        self.pts.extend_from_slice(&other.pts[h..]);
        self.durations.extend_from_slice(&other.durations[h..]);
        self.flags.extend_from_slice(&other.flags[h..]);
        self.metas.extend_from_slice(&other.metas[h..]);
        for (pos, ev) in other.events.drain(..) {
            // An event whose position was already at/behind other's cursor is due at
            // the seam; later ones keep their offset into the appended span.
            self.events.push((base + (pos.saturating_sub(h as u32)), ev));
        }
        other.reset_columns();
    }

    /// Attach an in-band event to this span at the **current end** (spec: Events
    /// travel with buffers): it precedes whatever is pushed next — so an element that
    /// announces mid-`process()` splits the batch exactly there.
    pub fn push_event(&mut self, event: Event) {
        self.events.push((self.memories.len() as u32, event));
    }

    /// Attach an in-band event at an explicit row position (scheduler use — e.g. an
    /// announcement whose position was captured when the element made it).
    pub(crate) fn push_event_at(&mut self, pos: u32, event: Event) {
        // Keep the position order invariant (positions are monotone by construction
        // on every normal path; a stray out-of-order insert would break the barrier).
        let at = self.events.partition_point(|&(p, _)| p <= pos);
        self.events.insert(at, (pos, event));
    }

    /// The next in-band event **due at the drain cursor** (its position has been
    /// reached), removed. The scheduler calls this before running the element, until
    /// it returns `None` — delivering each event after the element consumed exactly
    /// the buffers that preceded it.
    pub fn due_event(&mut self) -> Option<Event> {
        match self.events.first() {
            Some(&(pos, _)) if pos as usize <= self.head => Some(self.events.remove(0).1),
            _ => None,
        }
    }

    /// Whether any in-band events (delivered or not-yet-due) remain on this span.
    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    /// Pop the first event **regardless of position** — the fan-in path: an
    /// aggregator takes whole per-pad batches out of the scheduler's reach
    /// (`take_input_on`), so its events are delivered before the batch (the
    /// pre-positioning semantics), not at exact rows. Single-input elements get the
    /// positioned [`due_event`](Self::due_event) instead.
    pub(crate) fn next_event_any(&mut self) -> Option<Event> {
        if self.events.is_empty() {
            None
        } else {
            Some(self.events.remove(0).1)
        }
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
        b.push_event(Event::Eos); // positioned at b's end (after row 3)
        a.append(&mut b);
        assert!(b.is_empty() && !b.has_events(), "source emptied");
        assert_eq!(a.len(), 2, "only undrained rows moved");
        assert!(a.has_events(), "events ride along");
        // The event was positioned after b's last row: both rows pop before it is due.
        assert_eq!(a.pop_front().unwrap().memory.data()[0], 2);
        assert!(a.due_event().is_none(), "not due until its position is reached");
        assert_eq!(a.pop_front().unwrap().memory.data()[0], 3);
        assert!(matches!(a.due_event(), Some(Event::Eos)), "due after its preceding rows");
    }

    #[test]
    fn mid_batch_event_is_a_barrier_until_delivered() {
        let pool = Pool::bounded(64, 8);
        let mut b = Batch::new(FormatId(0));
        b.push(buf(&pool, 1, 1));
        b.push_event(Event::Eos); // precedes row 1 (the next push)
        b.push(buf(&pool, 2, 1));
        assert_eq!(b.pop_front().unwrap().memory.data()[0], 1, "row before the event flows");
        assert!(b.pop_front().is_none(), "barrier: row 2 held until the event is delivered");
        assert!(matches!(b.due_event(), Some(Event::Eos)), "event due exactly here");
        assert_eq!(b.pop_front().unwrap().memory.data()[0], 2, "row after the event flows");
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
