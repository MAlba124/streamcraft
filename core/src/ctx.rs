//! `Ctx` — the element's world: output, memory, time, bus. Elements hold no
//! channels, no threads, no peers (spec: Elements and pads).
//!
//! Milestone 1 carries the essentials: a pool to allocate from, an output batch the
//! scheduler drains, and the bus. Clock, reactor IO, scratch arenas, and per-pad
//! pools come as their subsystems land.

use crate::batch::{Batch, OutBatch};
use crate::buffer::{Buffer, BufferFlags};
use crate::bus::{BusMessage, BusSender};
use crate::id::{ElementId, FormatId, PadId};
use crate::memory::{Arena, Pool};
use crate::time::Timestamp;

pub struct Ctx {
    pool: Pool,
    out: Batch,
    bus: BusSender,
    element: ElementId,
    out_format: FormatId,
}

impl Ctx {
    pub fn new(pool: Pool, bus: BusSender, element: ElementId, out_format: FormatId) -> Self {
        Self {
            out: Batch::new(out_format),
            pool,
            bus,
            element,
            out_format,
        }
    }

    /// Which element this context belongs to (used for bus messages).
    pub fn element(&self) -> ElementId {
        self.element
    }

    /// Allocate a buffer from this element's pool — never `malloc`, never per-buffer
    /// heap in steady state.
    pub fn alloc(&mut self, _pad: PadId) -> Buffer {
        Buffer {
            memory: self.pool.acquire(),
            pts: Timestamp::NONE,
            dts: Timestamp::NONE,
            duration: Timestamp::NONE,
            flags: BufferFlags::empty(),
            format: self.out_format,
            sync: None,
        }
    }

    /// SoA writer into this element's output batch.
    pub fn out(&mut self, _pad: PadId) -> OutBatch<'_> {
        OutBatch::new(&mut self.out)
    }

    /// Post a message to the bus.
    pub fn post(&mut self, msg: BusMessage) {
        self.bus.send(msg);
    }

    /// The pipeline clock. Milestone 1 has no clock yet (spec: Clocking).
    pub fn now(&self) -> Timestamp {
        Timestamp::NONE
    }

    /// Bump allocator, reset after each `process()` call (spec: Memory).
    pub fn scratch(&mut self) -> &mut Arena {
        todo!("spec: Memory — scratch arenas")
    }

    /// Scheduler hook: take the produced output batch to hand downstream.
    pub(crate) fn output_mut(&mut self) -> &mut Batch {
        &mut self.out
    }
}
