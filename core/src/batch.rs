//! Batches: the unit of work is a span, not a buffer (spec: Batching). SoA layout —
//! parallel arrays, so timestamp scans (QoS, latency, seek) touch one dense column.
//!
//! Milestone 1 uses an owned [`Batch`] with `Vec` columns, reused across iterations
//! (cleared, not freed) so the metadata storage doesn't reallocate. The
//! ring-buffer-slots-are-the-columns version (spec: Queue internals) replaces the
//! transport later behind [`BatchRef`] / [`OutBatch`].

use crate::buffer::{Buffer, BufferFlags};
use crate::id::{FormatId, MetaId, PadId};
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
        }
    }

    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Clear all columns, retaining capacity (drops the `Memory`s → recycled).
    pub fn clear(&mut self) {
        self.memories.clear();
        self.pts.clear();
        self.durations.clear();
        self.flags.clear();
        self.metas.clear();
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

/// Borrowed SoA view over a batch. Cheap to copy (all fields are shared refs).
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

    /// The `i`th buffer's memory (for reading its bytes).
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

/// The (possibly multiple) inputs delivered to `Element::process` (spec: Aggregation).
#[derive(Clone, Copy)]
pub struct Inputs<'a> {
    entries: &'a [(PadId, BatchRef<'a>)],
}

impl<'a> Inputs<'a> {
    pub fn empty() -> Inputs<'static> {
        Inputs { entries: &[] }
    }

    pub fn new(entries: &'a [(PadId, BatchRef<'a>)]) -> Self {
        Self { entries }
    }

    /// `Single`-policy sugar: the one input, if present.
    pub fn single(&self) -> Option<BatchRef<'a>> {
        self.entries.first().map(|(_, b)| *b)
    }

    pub fn get(&self, pad: PadId) -> Option<BatchRef<'a>> {
        self.entries.iter().find(|(p, _)| *p == pad).map(|(_, b)| *b)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
