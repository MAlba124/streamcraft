//! `Ctx` — the element's world: input, output, memory, IO, bus (spec: Elements and
//! pads). Elements hold no channels, no threads, no peers.
//!
//! The scheduler owns each `Ctx` and moves data through it: it fills `input` with
//! upstream buffers and the IO `inbox` with completions before calling `process`,
//! then drains the `out` batch and the IO `outbox` (submissions) afterwards.

use std::fs::File;

use crate::batch::{Batch, OutBatch};
use crate::buffer::{Buffer, BufferFlags};
use crate::bus::{BusMessage, BusSender};
use crate::format::FixedFormat;
use crate::id::{ElementId, FormatId, PadId};
use crate::io::{Completion, Io, Submission};
use crate::memory::{Arena, Pool};
use crate::time::Timestamp;

pub struct Ctx {
    pool: Pool,
    input: Batch,
    out: Batch,
    bus: BusSender,
    element: ElementId,
    out_format: FormatId,
    /// The format negotiated on each of this element's pads, indexed by local pad
    /// index (`PadId.0` == index into `desc().pads`). `None` for an unlinked pad.
    /// Filled once, at link time, by the pipeline — read cheaply in `start()` /
    /// `process()` via [`negotiated`](Self::negotiated) (spec: Formats — elements read
    /// their fixed format from `Ctx`; never negotiated per-buffer).
    negotiated: Vec<Option<FixedFormat>>,
    credits: u32,
    // IO mailbox (spec: IO — submit/complete via ctx.io()).
    registration: Option<File>,
    io_in: Vec<Completion>,
    io_out: Vec<Submission>,
    next_op: u64,
}

impl Ctx {
    pub fn new(
        pool: Pool,
        bus: BusSender,
        element: ElementId,
        out_format: FormatId,
        credits: u32,
    ) -> Self {
        Self {
            input: Batch::new(out_format),
            out: Batch::new(out_format),
            pool,
            bus,
            element,
            out_format,
            negotiated: Vec::new(),
            credits,
            registration: None,
            io_in: Vec::new(),
            io_out: Vec::new(),
            next_op: 0,
        }
    }

    pub fn element(&self) -> ElementId {
        self.element
    }

    /// The format the pipeline fixed on `pad` at link time (spec: Formats — fixed
    /// formats live on edges; elements read theirs here). `None` if the pad is not
    /// linked (or the solve set no fixed fields on a bare `ANY` edge — the family is
    /// still available via the returned format). A cheap borrow: the solve ran once at
    /// link time, so this is never a per-buffer cost.
    pub fn negotiated(&self, pad: PadId) -> Option<&FixedFormat> {
        self.negotiated.get(pad.0 as usize).and_then(|f| f.as_ref())
    }

    /// Install the per-pad negotiated formats (pipeline → element, at link/`run`
    /// setup). Indexed by local pad index.
    pub(crate) fn set_negotiated(&mut self, formats: Vec<Option<FixedFormat>>) {
        self.negotiated = formats;
    }

    /// Allocate from this element's pool (unbounded).
    pub fn alloc(&mut self, _pad: PadId) -> Buffer {
        self.buffer(self.pool.acquire())
    }

    /// Allocate from the pool respecting its cap — `None` is the backpressure signal
    /// (spec: submission credits). IO sources gate reads on this.
    pub fn try_alloc(&mut self, _pad: PadId) -> Option<Buffer> {
        let memory = self.pool.try_acquire()?;
        Some(self.buffer(memory))
    }

    fn buffer(&self, memory: crate::memory::Memory) -> Buffer {
        Buffer {
            memory,
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

    /// The reactor submit/complete handle (spec: IO).
    pub fn io(&mut self) -> Io<'_> {
        Io::new(
            self.element,
            &mut self.registration,
            &mut self.io_in,
            &mut self.io_out,
            &mut self.next_op,
            self.credits,
        )
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

    // --- Scheduler hooks (crate-internal) ---

    pub(crate) fn output_mut(&mut self) -> &mut Batch {
        &mut self.out
    }

    pub(crate) fn take_input(&mut self) -> Batch {
        std::mem::replace(&mut self.input, Batch::new(self.out_format))
    }

    pub(crate) fn set_input(&mut self, input: Batch) {
        self.input = input;
    }

    /// Move buffers from another element's output batch into this element's input
    /// (leaves `other` empty, retaining both batches' capacity).
    pub(crate) fn input_append(&mut self, other: &mut Batch) {
        self.input.append(other);
    }

    pub(crate) fn input_is_empty(&self) -> bool {
        self.input.is_empty()
    }

    pub(crate) fn take_submissions(&mut self) -> Vec<Submission> {
        std::mem::take(&mut self.io_out)
    }

    pub(crate) fn deliver_completion(&mut self, c: Completion) {
        self.io_in.push(c);
    }

    pub(crate) fn inbox_empty(&self) -> bool {
        self.io_in.is_empty()
    }

    pub(crate) fn take_registration(&mut self) -> Option<File> {
        self.registration.take()
    }
}
