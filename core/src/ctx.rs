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
use crate::format::{FixedFormat, ValueDesc};
use crate::id::{ElementId, FormatId, PadId};
use crate::io::{Completion, Io, Submission};
use crate::log::{Field, Level, Log, Loggable};
use crate::memory::{Arena, Pool};
use crate::time::Timestamp;

/// A runtime format announcement queued by an element via [`Ctx::announce_format`]. The
/// scheduler drains it after `process()`, interns it against the (frozen) pipeline tables
/// into a [`FixedFormat`], and rides it downstream as an
/// [`Event::FormatChange`](crate::event::Event::FormatChange) so the peer re-fixates its
/// edge (spec: Formats — dynamic caps).
pub(crate) struct Announcement {
    #[allow(dead_code)] // which src pad announced — used once multi-src pads land
    pub(crate) pad: PadId,
    pub(crate) family: &'static str,
    pub(crate) fields: Vec<(&'static str, ValueDesc)>,
}

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
    /// This element's log emitter (spec: Debuggability — Logging). `None` when logging
    /// is disabled, which is the zero-overhead default: nothing is wired, so `log!`
    /// gates out on a plain `false` with no channel, thread, or atomic behind it. When
    /// enabled the pipeline installs a `Log` here whose owned `LogSink` is written only
    /// by this `Ctx`'s single group thread. (Spec says "per-group ring"; a per-element
    /// channel is a `Send`-clean refinement — the SPSC `Producer` is `Send` but not
    /// `Sync`, so one sink per `Ctx` keeps `Ctx: Send` without sharing a producer.)
    log: Option<Log>,
    /// A pending runtime format announcement (spec: Formats — dynamic caps). Set by
    /// [`announce_format`](Self::announce_format), drained by the scheduler after
    /// `process()`. `None` on the hot path for the overwhelming majority of elements.
    announced: Option<Announcement>,
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
            log: None,
            announced: None,
        }
    }

    pub fn element(&self) -> ElementId {
        self.element
    }

    /// Install this element's log emitter (pipeline → element, at `run()` setup, only
    /// when logging is enabled). The `Log` is pre-stamped with this element's id, so
    /// records it emits are already attributed. Absent this call the `Ctx` logs
    /// nothing at zero cost.
    pub(crate) fn set_log(&mut self, log: Log) {
        self.log = Some(log);
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

    /// Announce a runtime output format on `pad` (spec: Formats — dynamic caps). For an
    /// element whose output format is data-dependent — a decoder learning rate/channels
    /// from a header, a demuxer discovering a stream. `family` and the field names must be
    /// ones this pad already offered (they resolve against the link-time interned tables);
    /// the values are runtime. The scheduler turns this into a `FormatChange` that rides
    /// downstream, and the peer reads the concrete format via
    /// [`negotiated`](Self::negotiated). Cheap and rare — announce on change, not per
    /// buffer.
    pub fn announce_format(
        &mut self,
        pad: PadId,
        family: &'static str,
        fields: &[(&'static str, ValueDesc)],
    ) {
        self.announced = Some(Announcement {
            pad,
            family,
            fields: fields.to_vec(),
        });
    }

    /// Drain a pending announcement (scheduler hook, after `process()`).
    pub(crate) fn take_announcement(&mut self) -> Option<Announcement> {
        self.announced.take()
    }

    /// Update the negotiated format on a single pad at runtime, when a `FormatChange`
    /// crosses the edge (pipeline → element). Grows the table if needed (spec: Formats —
    /// dynamic caps).
    pub(crate) fn set_negotiated_one(&mut self, pad: PadId, format: FixedFormat) {
        let i = pad.0 as usize;
        if self.negotiated.len() <= i {
            self.negotiated.resize(i + 1, None);
        }
        self.negotiated[i] = Some(format);
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

/// Elements log through their `Ctx` — `log!(ctx, Level::Info, "event", k = v)` — which
/// forwards to the installed [`Log`] (spec: Debuggability). With no `Log` (logging
/// disabled), [`enabled`](Loggable::enabled) is a plain `false`, so the [`log!`] macro
/// short-circuits before touching the field expressions: the disabled path costs one
/// branch on an `Option` and nothing else.
impl Loggable for Ctx {
    #[inline]
    fn enabled(&self, level: Level) -> bool {
        match &self.log {
            Some(log) => log.enabled(level),
            None => false,
        }
    }

    fn emit(&self, level: Level, event: &'static str, fields: &[Field]) {
        if let Some(log) = &self.log {
            log.emit(level, event, fields);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::Bus;
    use crate::log::{log_channel, FieldValue, LevelFilter, Log};
    use std::sync::Arc;

    /// A minimal `Ctx` for the logging-path tests. The pool/bus are unused here; we
    /// only exercise `set_log` + `Loggable`.
    fn test_ctx(element: ElementId) -> Ctx {
        let pool = Pool::bounded(1024, 4);
        let (bus, _rx) = Bus::channel();
        Ctx::new(pool, bus, element, FormatId(0), 4)
    }

    #[test]
    fn ctx_with_no_log_is_disabled_and_silent() {
        // The zero-overhead default: no `Log` installed ⇒ every level gates out and the
        // macro never evaluates its arguments.
        let ctx = test_ctx(ElementId(1));
        assert!(!ctx.enabled(Level::Error));
        assert!(!ctx.enabled(Level::Trace));

        let mut calls = 0u32;
        let mut arg = || {
            calls += 1;
            7u64
        };
        crate::log!(&ctx, Level::Error, "suppressed", v = arg());
        assert_eq!(calls, 0, "no Log ⇒ arguments must not be evaluated");
    }

    #[test]
    fn log_through_ctx_reaches_the_drain_stamped_with_element() {
        // Prove an element-style emit (`log!(ctx, ...)`) crosses the channel and is
        // attributed to the element the pipeline stamped on the `Log`.
        let filter = Arc::new(LevelFilter::with_level(Level::Info));
        let (sink, drain) = log_channel(64);
        let mut log = Log::new(sink, filter);
        log.set_element(ElementId(5));

        let mut ctx = test_ctx(ElementId(5));
        ctx.set_log(log);

        assert!(ctx.enabled(Level::Info));
        crate::log!(&ctx, Level::Info, "hello", n = 42u64, ok = true);

        let rec = drain.try_next().expect("record reached the drain");
        assert_eq!(rec.event, "hello");
        assert_eq!(rec.element, ElementId(5), "stamped with the element id");
        assert_eq!(rec.level, Level::Info);
        assert_eq!(rec.fields().len(), 2);
        assert_eq!(rec.fields()[0].key, "n");
        assert!(matches!(rec.fields()[0].val, FieldValue::Uint(42)));
        assert!(matches!(rec.fields()[1].val, FieldValue::Bool(true)));
        assert!(drain.try_next().is_none(), "exactly one record");
    }

    #[test]
    fn ctx_gate_suppresses_levels_below_threshold() {
        // Hermetic: the `LevelFilter` is set directly (no `STREAMCRAFT_DEBUG`), so the
        // gate — not the environment — decides what passes.
        let filter = Arc::new(LevelFilter::with_level(Level::Warn));
        let (sink, drain) = log_channel(64);
        let mut log = Log::new(sink, Arc::clone(&filter));
        log.set_element(ElementId(3));

        let mut ctx = test_ctx(ElementId(3));
        ctx.set_log(log);

        crate::log!(&ctx, Level::Debug, "below", x = 1); // below Warn ⇒ suppressed
        crate::log!(&ctx, Level::Error, "boom", code = 500u32); // at/above ⇒ passes
        let rec = drain.try_next().expect("error passed the gate");
        assert_eq!(rec.event, "boom");
        assert!(drain.try_next().is_none(), "debug was gated out");

        // Raising the gate at runtime lets the finer level through immediately.
        filter.set(Some(Level::Trace));
        crate::log!(&ctx, Level::Debug, "now_visible", x = 2);
        let rec = drain.try_next().expect("debug passes once the gate is raised");
        assert_eq!(rec.event, "now_visible");
    }
}
