//! `Ctx` — the element's world: input, output, memory, IO, bus (spec: Elements and
//! pads). Elements hold no channels, no threads, no peers.
//!
//! The scheduler owns each `Ctx` and moves data through it: it fills `input` with
//! upstream buffers and the IO `inbox` with completions before calling `process`,
//! then drains the `out` batch and the IO `outbox` (submissions) afterwards.

use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::batch::{Batch, OutBatch};
use crate::buffer::{Buffer, BufferFlags};
use crate::bus::{BusMessage, BusSender};
use crate::clock::{Clock, WaitOutcome};
use crate::element::Direction;
use crate::event::Event;
use crate::format::{FixedFormat, OfferDesc, Value, ValueDesc, Vocabulary};
use crate::id::{ElementId, FieldId, FormatId, PadId, ValueId};
use crate::io::{Completion, Io, Submission};
use crate::log::{Field, Level, Log, Loggable};
use crate::memory::{Arena, Pool};
use crate::props::PropTable;
use crate::time::Timestamp;

/// A runtime format announcement queued by an element via [`Ctx::announce_format`]. The
/// scheduler drains it after `process()`, interns it against the (frozen) pipeline tables
/// into a [`FixedFormat`], and rides it downstream as an
/// [`Event::FormatChange`](crate::event::Event::FormatChange) so the peer re-fixates its
/// edge (spec: Formats — dynamic caps).
pub(crate) struct Announcement {
    /// Which src pad announced — the scheduler rides the `FormatChange` on that pad's
    /// output batch, so it travels to that pad's downstream (branching-correct).
    pub(crate) pad: PadId,
    /// The output-batch row the announcement precedes — captured at the moment of the
    /// announce call, so a mid-`process()` announcement splits the batch exactly
    /// there (spec: Events ordered relative to buffers).
    pub(crate) at: u32,
    pub(crate) payload: AnnouncePayload,
}

/// What an announcement carries: names to resolve through the vocabulary (the
/// decoder/demuxer path, [`Ctx::announce_format`]), or a format that is already
/// resolved (the forwarding path, [`Ctx::forward_format`] — a queue re-emitting a
/// `FormatChange` it received has no static names to offer).
pub(crate) enum AnnouncePayload {
    Named {
        family: &'static str,
        fields: Vec<(&'static str, ValueDesc)>,
    },
    Fixed(FixedFormat),
}

/// A pad an element instantiated at runtime via [`Ctx::add_pad`] during preroll (spec:
/// dynamic pads). The pipeline drains these: it registers each so [`link`] can find it
/// by name and posts a `PadAdded` bus message.
///
/// [`link`]: crate::pipeline::Pipeline::link
pub(crate) struct AddedPad {
    pub(crate) direction: Direction,
    pub(crate) name: String,
    pub(crate) offers: &'static [OfferDesc],
    pub(crate) pad: PadId,
}

/// Shared app→pipeline seek request (spec: Events, queries, and the bus — flush/seek). The
/// app publishes a target through a `SeekHandle`; the scheduler and the source/sink elements
/// each observe `gen` advancing and perform a coordinated flush. `to_byte` drives the source
/// (where to resume reading), `to_frame` resets the sink's play position. All fields are
/// relaxed/acquire atomics — reads sit on the flush path, never the per-buffer hot path.
#[derive(Default)]
pub struct SeekState {
    /// Bumped once per seek; every observer compares it to its own last-seen value.
    pub(crate) gen: AtomicU64,
    pub(crate) to_byte: AtomicU64,
    pub(crate) to_frame: AtomicU64,
}

/// A resolved seek target, read by an element from [`Ctx::seek_target`] when it handles
/// [`Event::FlushStart`](crate::event::Event::FlushStart).
#[derive(Clone, Copy, Debug)]
pub struct SeekTarget {
    /// Byte offset in the source byte stream to resume reading from.
    pub to_byte: u64,
    /// PCM frame (interchannel sample) index the target corresponds to — the play position
    /// a sink resets its counter to.
    pub to_frame: u64,
}

pub struct Ctx {
    pool: Pool,
    input: Batch,
    /// Per-src-pad output batches, indexed by local pad index (`PadId.0`). An element
    /// with one src pad uses a single slot; a branching element (a demuxer) writes each
    /// src pad's buffers to its own slot and the scheduler routes each to that pad's
    /// downstream ring (spec: dynamic pads / branching). Grown on demand, so a
    /// runtime-added pad is writable immediately.
    outs: Vec<Batch>,
    /// The src pad whose batch [`output_mut`](Self::output_mut) returns — the element's
    /// first src pad (a sink has none, so the slot is a harmless always-empty scratch).
    /// Set by [`configure_pads`](Self::configure_pads) at run setup.
    primary_out: usize,
    /// `Some(idx)` when the element has exactly one src pad: then `ctx.out(pad)` routes
    /// there regardless of the `pad` argument, preserving the pre-branching contract (one
    /// output, pad ignored). `None` for a branching element (≥2 src pads), where `out`
    /// routes strictly by pad, and for a sink (no src pad).
    single_src: Option<usize>,
    /// Pads instantiated at runtime via [`add_pad`](Self::add_pad), drained by the
    /// pipeline during preroll (spec: dynamic pads). Empty for the overwhelming majority.
    added_pads: Vec<AddedPad>,
    /// Next local pad index [`add_pad`](Self::add_pad) hands out — past the static pads
    /// (set by [`configure_pads`](Self::configure_pads)).
    next_pad: u32,
    /// Per-sink-pad input batches for a fan-in (aggregating) element — a muxer reads each
    /// upstream via [`take_input_on`](Self::take_input_on) (spec: Aggregation). Empty and
    /// unused for the single-input majority, which use the primary `input`.
    ins: Vec<Batch>,
    /// Per-sink-pad end-of-stream flags: the scheduler sets a pad closed when its upstream
    /// ring closes, so an aggregator knows that pad will produce no more and can stop
    /// waiting for it (spec: Aggregation). Grown on demand.
    pad_eos: Vec<bool>,
    bus: BusSender,
    element: ElementId,
    out_format: FormatId,
    /// Recycled output-batch shells (spec: Batching — zero-alloc transport): empty
    /// `Batch`es whose column capacity survived a trip downstream, returned by the
    /// consumer over the link's shell ring. [`take_output`](Self::take_output) reuses
    /// one instead of allocating fresh columns. Bounded (see `recycle_shell`).
    ///
    /// **Out-duty only.** Spent *input* shells go to `spare_inputs` instead: the two
    /// populations have different column-capacity histories (an out shell is sized by
    /// this element's emissions, an input shell by upstream's batches), and a shared
    /// pool paired them pessimally — `take_output` kept drawing a small input shell
    /// (or a cold one minted when the input side transiently drained the pool) and
    /// re-growing its columns from zero, ~75% of a remux's allocations.
    spare_shells: Vec<Batch>,
    /// Retention cap for `spare_shells`. The scheduler bumps it to the element's real
    /// shell circulation (downstream data+shell ring slots) — a cap below circulation
    /// makes `recycle_shell` drop warm shells during bursts while `take_output` mints
    /// cold ones, and every cold shell re-grows its columns from zero on first use.
    spare_cap: usize,
    /// Recycled *input* shells for [`take_input_on`](Self::take_input_on) replacements
    /// — a closed loop with [`recycle_input`](Self::recycle_input) inside an
    /// aggregator's pass (one take + one recycle per pad), so it is exactly balanced
    /// and never competes with the out-duty pool above.
    spare_inputs: Vec<Batch>,
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
    /// Pending runtime format announcements (spec: Formats — dynamic caps). Pushed by
    /// [`announce_format`](Self::announce_format), drained by the scheduler after
    /// `process()`. A queue, not a slot: a multi-track demuxer whose first samples for
    /// *several* pads land in one `process()` pass announces once per pad, and none may
    /// overwrite another (a single slot silently lost all but the last — found by a
    /// two-track remux). Empty on the hot path for the overwhelming majority of elements.
    announced: Vec<Announcement>,
    /// The pipeline's frozen interning tables (installed at run setup), so this element can
    /// resolve the names in its negotiated format — `field_id("rate")`, `value_name(id)`.
    /// `None` outside a run (spec: Formats — elements read their caps by name).
    vocabulary: Option<Arc<Vocabulary>>,
    /// The pipeline clock and the running-time base, installed at run setup (spec:
    /// Clocking and synchronization). [`now`](Self::now) returns `clock.now() -
    /// base_time`; [`wait_until`](Self::wait_until) parks on a `ClockWait` against this
    /// clock. `None` outside a run and for an unclocked byte pipeline, where `now()`
    /// reads `NONE` and `wait_until` returns at once (no pacing).
    clock: Option<Arc<dyn Clock>>,
    /// Raw ns of the running-time base — the pipeline's shared cell, read per call
    /// so a pause/resume re-base (spec: Clocking — pause is a clock op) is observed
    /// by the very next deadline computation. `u64::MAX` until a run samples it.
    base: Arc<std::sync::atomic::AtomicU64>,
    /// This element's computed upstream path latency (spec: Latency — enforced at
    /// sinks): the pipeline installs the worst source→here sum of declared minimum
    /// latencies, and [`wait_until`](Self::wait_until) folds it into the deadline —
    /// `base_time + running + path_latency` — so a sink automatically compensates
    /// what its upstream needs. Zero unless the graph declares latencies.
    path_latency: Timestamp,
    /// Latency tracing (spec: Debuggability): this element's histograms + the live
    /// gate, installed at run setup so [`wait_until`](Self::wait_until) can record how
    /// far past its deadline a sink wait actually returned. `None` outside a run.
    trace: Option<(Arc<crate::counters::ElementCounters>, Arc<std::sync::atomic::AtomicBool>)>,
    /// The pause transport (spec: Clocking — pause is a clock op), installed at run
    /// setup; `None` outside a run.
    pause: Option<Arc<crate::pipeline::PauseShared>>,
    /// Which of this element's src pads have a linked edge (indexed by local pad),
    /// installed at run setup. Lets a demuxer skip resolving samples for tracks
    /// nobody consumes (ZERO-COPY.md stage 5-lite). Empty outside a run — then
    /// [`pad_linked`](Self::pad_linked) reports `true` (fail open: emit; the
    /// scheduler's unlinked-drop policy is the backstop).
    linked_src_pads: Vec<bool>,
    /// Shared seek request, installed at run setup (spec: flush/seek). A source reads the
    /// byte target and a sink the frame target from [`seek_target`](Self::seek_target) when
    /// the scheduler delivers `FlushStart`. `None` outside a run / for a pipeline that is
    /// never seeked — no cost on the hot path.
    seek: Option<Arc<SeekState>>,
    /// Per-`process()` scratch bump allocator (spec: Memory). Reset by the scheduler after
    /// each `process()`, so temporaries carved here impose no steady-state heap traffic.
    scratch: Arena,
    /// This element's property mailbox + its `desc().props` table (spec: Dynamic element
    /// properties), installed at run setup. The app validates and parks sets in the
    /// table; the scheduler polls the dirty mask at batch boundaries (one relaxed load
    /// when idle) and the element pull-reads via [`prop`](Self::prop). `None` outside a
    /// run / for elements declaring no props — zero cost.
    props: Option<(&'static [crate::element::PropDesc], Arc<PropTable>)>,
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
            spare_shells: Vec::new(),
            spare_cap: 8,
            spare_inputs: Vec::new(),
            outs: Vec::new(),
            primary_out: 0,
            single_src: None,
            added_pads: Vec::new(),
            next_pad: 0,
            ins: Vec::new(),
            pad_eos: Vec::new(),
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
            announced: Vec::new(),
            vocabulary: None,
            clock: None,
            base: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            seek: None,
            scratch: Arena::default(),
            props: None,
            path_latency: Timestamp::ZERO,
            trace: None,
            pause: None,
            linked_src_pads: Vec::new(),
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
        let pad = self.resolve_src_pad(pad);
        let at = self.out_slot(pad).len() as u32;
        self.announced.push(Announcement {
            pad,
            at,
            payload: AnnouncePayload::Named { family, fields: fields.to_vec() },
        });
    }

    /// Re-announce an **already-resolved** format downstream on `pad` (spec: Formats
    /// — dynamic caps). The forwarding primitive for pure transports (a queue, a
    /// tee): on `Event::FormatChange(f)`, call `forward_format(src, f.clone())` so
    /// the change hops onward across this element — [`announce_format`]
    /// (Self::announce_format) cannot express this, since a generic forwarder has no
    /// static names for fields it never knew. Same queued, batch-boundary semantics as
    /// an announcement.
    pub fn forward_format(&mut self, pad: PadId, format: FixedFormat) {
        let pad = self.resolve_src_pad(pad);
        let at = self.out_slot(pad).len() as u32;
        self.announced.push(Announcement { pad, at, payload: AnnouncePayload::Fixed(format) });
    }

    /// Single-src leniency, as in `out(pad)`: the scheduler's drain routes strictly
    /// by pad index, so the announcement records the resolved pad.
    fn resolve_src_pad(&self, pad: PadId) -> PadId {
        match self.single_src {
            Some(idx) => PadId(idx as u32),
            None => pad,
        }
    }

    /// Drain the pending announcements, in announce order (scheduler hook, after
    /// `process()`).
    pub(crate) fn take_announcements(&mut self) -> Vec<Announcement> {
        std::mem::take(&mut self.announced)
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

    /// Install the frozen vocabulary (pipeline → element at run setup).
    pub(crate) fn set_vocabulary(&mut self, vocabulary: Arc<Vocabulary>) {
        self.vocabulary = Some(vocabulary);
    }

    /// Install the pipeline clock and running-time base (pipeline → element at run
    /// setup). See [`now`](Self::now) and [`wait_until`](Self::wait_until).
    pub(crate) fn set_clock(&mut self, clock: Arc<dyn Clock>, base: Arc<std::sync::atomic::AtomicU64>) {
        self.clock = Some(clock);
        self.base = base;
    }

    /// Install the linked-src-pad map (pipeline -> element at run setup).
    pub(crate) fn set_linked_pads(&mut self, linked: Vec<bool>) {
        self.linked_src_pads = linked;
    }

    /// Whether `pad` (a src pad) has a linked downstream edge. A demuxer may skip
    /// producing for unlinked pads entirely — cheaper than the scheduler's
    /// drop-and-count backstop, which still applies to whatever is emitted anyway.
    /// Outside a run (empty map) this reports `true` — fail open.
    pub fn pad_linked(&self, pad: PadId) -> bool {
        self.linked_src_pads.get(pad.0 as usize).copied().unwrap_or(true)
    }

    /// This element's negotiated output `FormatId` (what pool-allocated buffers are
    /// stamped with) — for elements constructing `Buffer`s around sliced `Memory`.
    pub fn out_format(&self) -> FormatId {
        self.out_format
    }

    /// Install the pause transport (pipeline -> element at run setup; spec: Clocking
    /// — pause is a clock op): a clock wait that expires while paused blocks here
    /// until resume, then re-derives its deadline from the re-based shared base.
    pub(crate) fn set_pause(&mut self, pause: Arc<crate::pipeline::PauseShared>) {
        self.pause = Some(pause);
    }

    /// The current running-time base (shared, pause-re-based).
    fn base_now(&self) -> Timestamp {
        let raw = self.base.load(std::sync::atomic::Ordering::Acquire);
        if raw == u64::MAX { Timestamp::ZERO } else { Timestamp(raw) }
    }

    /// Install this element's computed upstream path latency (pipeline → element at
    /// run setup; spec: Latency — computed per path from declared latencies).
    pub(crate) fn set_path_latency(&mut self, latency: Timestamp) {
        self.path_latency = latency;
    }

    /// Install the latency-tracing gate + histograms (pipeline → element at run setup;
    /// spec: Debuggability).
    pub(crate) fn set_trace(
        &mut self,
        counters: Arc<crate::counters::ElementCounters>,
        gate: Arc<std::sync::atomic::AtomicBool>,
    ) {
        self.trace = Some((counters, gate));
    }

    /// This element's upstream path latency — what [`wait_until`](Self::wait_until)
    /// already folds into its deadline. Exposed so QoS lateness checks can use the
    /// same reference point a render wait does.
    pub fn path_latency(&self) -> Timestamp {
        self.path_latency
    }

    /// Install the shared seek request (pipeline → element at run setup). See
    /// [`seek_target`](Self::seek_target).
    pub(crate) fn set_seek(&mut self, seek: Arc<SeekState>) {
        self.seek = Some(seek);
    }

    /// The current seek generation (spec: flush/seek), bumped once per seek. An element that
    /// blocks for a long time inside `process()` (e.g. a sink pushing a big batch to a device
    /// that drains in real time) samples this and bails out when it changes, so the scheduler
    /// can run the flush promptly instead of after the whole batch has played. `0` until the
    /// first seek / when no seek state is installed.
    pub fn seek_gen(&self) -> u64 {
        self.seek.as_ref().map_or(0, |s| s.gen.load(Ordering::Acquire))
    }

    /// The current seek target, if any seek has been requested (spec: flush/seek). A source
    /// element reads `to_byte` (where to resume reading) and a sink reads `to_frame` (the
    /// play position to reset to) when the scheduler delivers
    /// [`Event::FlushStart`](crate::event::Event::FlushStart). `None` until the first seek.
    pub fn seek_target(&self) -> Option<SeekTarget> {
        let s = self.seek.as_ref()?;
        if s.gen.load(Ordering::Acquire) == 0 {
            return None; // no seek has ever been requested
        }
        Some(SeekTarget {
            to_byte: s.to_byte.load(Ordering::Acquire),
            to_frame: s.to_frame.load(Ordering::Acquire),
        })
    }

    /// Resolve a format-field name to its interned id, for reading a negotiated
    /// [`FixedFormat`] by name (spec: Formats — an element acts on its caps). `None` before
    /// `run()` or for a name the graph never interned.
    pub fn field_id(&self, name: &str) -> Option<FieldId> {
        self.vocabulary.as_ref()?.field_id(name)
    }

    /// Resolve a categorical value name (e.g. a `sample` format) to its interned id.
    pub fn value_id(&self, name: &str) -> Option<ValueId> {
        self.vocabulary.as_ref()?.value_id(name)
    }

    /// Resolve an interned family id back to its name.
    pub fn family_name(&self, id: FormatId) -> Option<&str> {
        self.vocabulary.as_ref()?.family_name(id)
    }

    /// Resolve an interned categorical value id back to its name.
    pub fn value_name(&self, id: ValueId) -> Option<&str> {
        self.vocabulary.as_ref()?.value_name(id)
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

    /// Allocate a buffer of at least `n` usable bytes: a pooled slot when one is free
    /// and large enough, else an exactly-`n` heap allocation (never a slot-sized
    /// over-allocation — see [`Pool::acquire_exact`]). For bounded cold paths (an EOS
    /// flush emitting the tail of a stream); steady-state producers use
    /// [`try_alloc`](Self::try_alloc) + backpressure instead.
    pub fn alloc_exact(&mut self, _pad: PadId, n: usize) -> Buffer {
        self.buffer(self.pool.acquire_exact(n))
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

    /// SoA writer into the output batch for `pad` (spec: Elements and pads — an element
    /// writes each src pad independently). Grows the per-pad table on demand, so a
    /// runtime-added pad is writable immediately.
    pub fn out(&mut self, pad: PadId) -> OutBatch<'_> {
        OutBatch::new(self.out_batch(pad))
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

    /// Running time on the pipeline clock: `clock.now() - base_time`, so it starts near
    /// zero when the pipeline begins and only moves forward (spec: Clocking — elements
    /// read running time here, never a wall clock). Returns [`Timestamp::NONE`] when no
    /// clock is installed (an unclocked byte pipeline — the milestone-1 default).
    pub fn now(&self) -> Timestamp {
        match &self.clock {
            Some(c) => c.now().saturating_sub(self.base_now()),
            None => Timestamp::NONE,
        }
    }

    /// Block until the pipeline's running time reaches `running` **plus this
    /// element's path latency** — the deadline is `base_time + running +
    /// path_latency` (spec: Latency — "sinks render at `base_time + pts +
    /// path_latency`"; the compensation is automatic, not sink code). Returns
    /// [`WaitOutcome::Reached`], or [`WaitOutcome::Interrupted`] if the wait is cut
    /// short (flush / shutdown). A timed sink calls this with a buffer's PTS to
    /// render on the clock; upstream never waits (spec: Clocking — only sinks wait).
    /// With no clock installed this returns [`WaitOutcome::Reached`] at once — an
    /// unclocked pipeline imposes no pacing. A `running` of [`Timestamp::NONE`] is
    /// "infinitely late": only an interrupt ends the wait.
    pub fn wait_until(&self, running: Timestamp) -> WaitOutcome {
        let Some(c) = &self.clock else { return WaitOutcome::Reached };
        loop {
            // Deadline derived per iteration from the *shared* base: a pause/resume
            // re-bases it, and the loop below re-derives after every resume.
            let deadline = self
                .base_now()
                .saturating_add(running)
                .saturating_add(self.path_latency);
            match c.new_wait().wait_until(deadline) {
                WaitOutcome::Interrupted => return WaitOutcome::Interrupted,
                WaitOutcome::Reached => {
                    // Pause gate (spec: Clocking — pause is a clock op): a wait that
                    // expires while paused must not render — block until resume, then
                    // re-derive the deadline from the shifted base and wait again.
                    // One relaxed load on the playing path.
                    if let Some(p) = &self.pause {
                        if p.maybe_paused() && p.is_paused() {
                            if !p.block_while_paused() {
                                return WaitOutcome::Interrupted; // stop won
                            }
                            continue;
                        }
                    }
                    // Latency tracing (spec: Debuggability): how far past the deadline
                    // the wait actually returned — scheduling jitter as the sink
                    // experiences it (deterministic under MockClock).
                    if running.is_some() {
                        if let Some((counters, gate)) = &self.trace {
                            if gate.load(std::sync::atomic::Ordering::Relaxed) {
                                let late = c.now().saturating_sub(deadline);
                                counters.wait_lateness_ns.record(late.nanos().unwrap_or(0));
                            }
                        }
                    }
                    return WaitOutcome::Reached;
                }
            }
        }
    }

    /// The current value of one of this element's declared properties, by name (spec:
    /// Dynamic element properties). `None` while never set — fall back to the
    /// constructor value — and for a name not in `desc().props`. A lock-free seqlock
    /// read plus a short name scan: cheap enough to call once per batch for a
    /// continuous knob (volume, threshold); elements that prefer notification handle
    /// [`Event::PropChanged`](crate::event::Event::PropChanged) instead.
    pub fn prop(&self, name: &str) -> Option<Value> {
        let (descs, table) = self.props.as_ref()?;
        let idx = descs.iter().position(|p| p.name == name)?;
        table.get(idx)
    }

    /// Install the property mailbox (pipeline → element at run setup).
    pub(crate) fn set_props(
        &mut self,
        descs: &'static [crate::element::PropDesc],
        table: Arc<PropTable>,
    ) {
        self.props = Some((descs, table));
    }

    /// Poll the property dirty mask (scheduler hook, once per pass at the batch
    /// boundary). One relaxed load when nothing changed.
    pub(crate) fn poll_prop_changes(&self) -> u64 {
        match &self.props {
            Some((_, table)) => table.take_dirty(),
            None => 0,
        }
    }

    /// Resolve a dirty-bit index to `(name, current value)` for delivering
    /// [`Event::PropChanged`](crate::event::Event::PropChanged) (scheduler hook).
    pub(crate) fn prop_entry(&self, idx: usize) -> Option<(&'static str, Value)> {
        let (descs, table) = self.props.as_ref()?;
        let name = descs.get(idx)?.name;
        Some((name, table.get(idx)?))
    }

    /// A per-`process()` bump allocator for temporary scratch space (spec: Memory). Carve
    /// regions with `ctx.scratch().alloc_bytes(n)`; they are reused (the scheduler resets
    /// the arena after each `process()`), so scratch imposes no steady-state heap traffic.
    /// The regions borrow the `Ctx`, so finish with them before other `&mut ctx` calls
    /// (e.g. `ctx.out(pad)`) — typically build a temporary here, then copy the result into
    /// an output buffer.
    pub fn scratch(&self) -> &Arena {
        &self.scratch
    }

    /// Reset the scratch arena (scheduler hook, run after each `process()`).
    pub(crate) fn reset_scratch(&mut self) {
        self.scratch.reset();
    }

    // --- Scheduler hooks (crate-internal) ---

    /// Raw per-pad output slot, indexed strictly by `pad` and grown on demand. The
    /// scheduler's explicit per-pad routing uses this; the public [`out`](Self::out) goes
    /// through [`out_batch`](Self::out_batch), which adds the single-src-pad leniency.
    fn out_slot(&mut self, pad: PadId) -> &mut Batch {
        let i = pad.0 as usize;
        if self.outs.len() <= i {
            self.outs.resize_with(i + 1, || Batch::new(self.out_format));
        }
        &mut self.outs[i]
    }

    /// The batch a public `out(pad)` writes to. For a single-src-pad element every write
    /// lands on that one pad regardless of the `pad` argument (the pre-branching contract
    /// — pad ignored); a branching element routes strictly by `pad`.
    fn out_batch(&mut self, pad: PadId) -> &mut Batch {
        let pad = match self.single_src {
            Some(idx) => PadId(idx as u32),
            None => pad,
        };
        self.out_slot(pad)
    }

    /// Size the per-pad output table to this element's pad count and record its src pads
    /// (spec: dynamic pads / branching). `primary_out` is the first src pad (used by
    /// [`output_mut`](Self::output_mut)); `single_src` is set when there is exactly one,
    /// so `out()` keeps the pre-branching "pad ignored" contract. Called once at run
    /// setup, before `start()`. A sink (no src pad) gets `primary_out = 0`, a scratch.
    pub(crate) fn configure_pads(&mut self, npads: usize, src_pads: &[usize]) {
        self.outs = (0..npads).map(|_| Batch::new(self.out_format)).collect();
        self.primary_out = src_pads.first().copied().unwrap_or(0);
        self.single_src = if src_pads.len() == 1 { Some(src_pads[0]) } else { None };
        self.next_pad = npads as u32;
    }

    /// Instantiate a new pad at runtime and return its [`PadId`] (spec: dynamic pads).
    /// Called from an element's [`preroll`](crate::element::Element::preroll) after it
    /// discovers a stream: the id (past the static pads) is what the element writes to via
    /// [`out`](Self::out), and the pad is recorded so the pipeline can link it and post a
    /// `PadAdded`. `offers` is the pad's format menu, exactly like a static pad's.
    pub fn add_pad(
        &mut self,
        direction: Direction,
        name: &str,
        offers: &'static [OfferDesc],
    ) -> PadId {
        let pad = PadId(self.next_pad);
        self.next_pad += 1;
        self.added_pads.push(AddedPad {
            direction,
            name: name.to_string(),
            offers,
            pad,
        });
        pad
    }

    /// Drain the runtime-added pads (pipeline hook, during preroll).
    pub(crate) fn take_added_pads(&mut self) -> Vec<AddedPad> {
        std::mem::take(&mut self.added_pads)
    }

    /// The element's primary output batch (its first src pad). Used by the scheduler for
    /// the single-downstream push and intra-group hand-off.
    pub(crate) fn output_mut(&mut self) -> &mut Batch {
        self.out_slot(PadId(self.primary_out as u32))
    }

    /// The output batch for a specific src pad (the scheduler's per-pad routing).
    pub(crate) fn output_on(&mut self, pad: PadId) -> &mut Batch {
        self.out_slot(pad)
    }

    /// Take (replace with empty) the output batch for a src pad, to route it
    /// downstream. The replacement comes from the spare-shell stack when the consumer
    /// has returned one (column capacity intact — the steady-state zero-alloc path);
    /// only a link whose shells haven't circulated yet allocates.
    pub(crate) fn take_output(&mut self, pad: PadId) -> Batch {
        let fmt = self.out_format;
        let replacement = match self.spare_shells.pop() {
            Some(mut shell) => {
                shell.format = fmt;
                shell
            }
            None => Batch::new(fmt),
        };
        std::mem::replace(self.out_slot(pad), replacement)
    }

    /// Return an empty shell for [`take_output`](Self::take_output) to reuse
    /// (scheduler hook — shells arrive over the link's shell ring). Cleared here;
    /// capped so a burst can't hoard column memory.
    pub(crate) fn recycle_shell(&mut self, mut shell: Batch) {
        if self.spare_shells.len() < self.spare_cap {
            shell.clear();
            self.spare_shells.push(shell);
        }
    }

    /// Raise the spare-shell retention cap by `n` (scheduler hook, at group setup):
    /// called once per downstream link (data + shell ring slots) and per fan-in pad,
    /// so the cap tracks how many shells can legitimately be in flight at once.
    pub(crate) fn bump_spare_cap(&mut self, n: usize) {
        self.spare_cap += n;
    }

    /// Total buffers and used bytes across all src pads (for counters).
    pub(crate) fn total_output(&self) -> (u64, u64) {
        self.outs
            .iter()
            .fold((0u64, 0u64), |(n, b), o| (n + o.len() as u64, b + o.total_bytes()))
    }

    /// Clear every src pad's output batch (retaining capacity). Used when a group has no
    /// downstream (a sink) so any stray output is dropped rather than left to grow.
    pub(crate) fn clear_outputs(&mut self) {
        for out in &mut self.outs {
            out.clear();
        }
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

    /// The next in-band event due at the primary input's drain cursor (scheduler
    /// hook — delivered before running the element; spec: Events ordered relative
    /// to buffers, exact mid-batch positions).
    pub(crate) fn due_input_event(&mut self) -> Option<Event> {
        self.input.due_event()
    }

    /// Any event pending on a fan-in pad's input (delivered before the batch — see
    /// `Batch::next_event_any` for why fan-in is pre-positioned).
    pub(crate) fn any_input_event_on(&mut self, pad: PadId) -> Option<Event> {
        self.ins.get_mut(pad.0 as usize)?.next_event_any()
    }

    /// Whether undelivered in-band events remain on the primary input.
    pub(crate) fn input_has_events(&self) -> bool {
        self.input.has_events()
    }

    /// Buffers waiting on the primary sink pad — the scheduler's inline-backpressure
    /// gauge (see `run_group`: an element is not run while its co-grouped successor
    /// still holds a backlog).
    pub(crate) fn input_len(&self) -> usize {
        self.input.len()
    }

    /// True if no pad has pending input — the primary sink pad *and* every fan-in pad.
    /// The scheduler's quiescence check (works for single- and multi-input elements).
    pub(crate) fn inputs_empty(&self) -> bool {
        self.input.is_empty()
            && !self.input.has_events()
            && self.ins.iter().all(|b| b.is_empty() && !b.has_events())
    }

    /// Append an upstream batch to a specific sink pad's input (scheduler hook, fan-in).
    pub(crate) fn append_input_on(&mut self, pad: PadId, other: &mut Batch) {
        let i = pad.0 as usize;
        if self.ins.len() <= i {
            self.ins.resize_with(i + 1, || Batch::new(self.out_format));
        }
        self.ins[i].append(other);
    }

    /// Take (replace with empty) a sink pad's accumulated input — an aggregating element's
    /// per-pad read (spec: Aggregation). Pairs with [`is_pad_closed`](Self::is_pad_closed)
    /// so a muxer knows when a pad is done.
    pub fn take_input_on(&mut self, pad: PadId) -> Batch {
        let i = pad.0 as usize;
        if self.ins.len() <= i {
            self.ins.resize_with(i + 1, || Batch::new(self.out_format));
        }
        let fmt = self.out_format;
        let replacement = match self.spare_inputs.pop() {
            Some(mut shell) => {
                shell.format = fmt;
                shell
            }
            None => Batch::new(fmt),
        };
        std::mem::replace(&mut self.ins[i], replacement)
    }

    /// Attach an in-band event to a src pad's output at the **current end** of what
    /// this element has pushed (spec: Events — travel with buffers, ordered): it
    /// reaches the downstream element's `event()` after every buffer pushed so far is
    /// consumed. What a muxer uses to send [`Event::Patch`] after its final bytes.
    /// (Format announcements go through [`announce_format`](Self::announce_format) /
    /// [`forward_format`](Self::forward_format), which also re-fixate.)
    pub fn push_event(&mut self, pad: PadId, event: Event) {
        self.out_slot(pad).push_event(event);
    }

    /// Hand a spent batch from [`take_input_on`](Self::take_input_on) back for reuse.
    /// An aggregator owns whole input batches; returning the drained shell keeps its
    /// column capacity in circulation for the next `take_input_on` replacement instead
    /// of re-growing a fresh batch's columns from zero on every pass. Anything left in
    /// the batch is cleared (buffers recycle to their pool).
    pub fn recycle_input(&mut self, mut batch: Batch) {
        // One shell per fan-in pad plus slack fully covers the take/recycle loop; the
        // cap only guards against an element hoarding shells it never takes back.
        if self.spare_inputs.len() < 2 * self.ins.len().max(1) {
            batch.clear();
            self.spare_inputs.push(batch);
        }
    }

    /// Whether a sink pad's upstream has closed (no more input will arrive on it), so an
    /// aggregator can stop waiting for that pad and drain the rest (spec: Aggregation).
    pub fn is_pad_closed(&self, pad: PadId) -> bool {
        self.pad_eos.get(pad.0 as usize).copied().unwrap_or(false)
    }

    /// Mark a sink pad's upstream closed (scheduler hook).
    pub(crate) fn set_pad_closed(&mut self, pad: PadId) {
        let i = pad.0 as usize;
        if self.pad_eos.len() <= i {
            self.pad_eos.resize(i + 1, false);
        }
        self.pad_eos[i] = true;
    }

    /// The element's IO outbox, handed to `Reactor::submit` which *drains* it — the
    /// vec (and its capacity) stays here, so per-pass submission allocates nothing
    /// (ZERO-COPY.md stage 4.2).
    pub(crate) fn submissions_mut(&mut self) -> &mut Vec<Submission> {
        &mut self.io_out
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

    /// Drop the pipeline data staged in this `Ctx` — pending input, each src pad's output,
    /// and any fan-in per-pad inputs — recycling their buffers to the pool (spec: flush/seek).
    /// The scheduler calls this on each element of a group when a flush begins, so no pre-seek
    /// data lingers inside an element between passes.
    ///
    /// The IO mailbox (`io_in` completions / `io_out` submissions) is deliberately **left
    /// intact**: it is the element's own in-flight accounting (a source counts every submitted
    /// read and decrements only when it drains the completion), so clearing it here would
    /// strand completions the element never counted as done and desync that count — throttling
    /// or stalling the source across a seek. The source drops the *stale data* itself when it
    /// drains those completions (filesrc's `valid_from` seq floor), which keeps the count exact.
    pub(crate) fn discard_buffers(&mut self) {
        self.input.clear();
        for o in &mut self.outs {
            o.clear();
        }
        for i in &mut self.ins {
            i.clear();
        }
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
        log.set_element(ElementId(5), "testsrc");

        let mut ctx = test_ctx(ElementId(5));
        ctx.set_log(log);

        assert!(ctx.enabled(Level::Info));
        crate::log!(&ctx, Level::Info, "hello", n = 42u64, ok = true);

        let rec = drain.try_next().expect("record reached the drain");
        assert_eq!(rec.event, "hello");
        assert_eq!(rec.element, ElementId(5), "stamped with the element id");
        assert_eq!(rec.name, "testsrc", "stamped with the element name");
        assert_eq!(rec.level, Level::Info);
        assert_eq!(rec.fields().len(), 2);
        assert_eq!(rec.fields()[0].key, "n");
        assert!(matches!(rec.fields()[0].val, FieldValue::Uint(42)));
        assert!(matches!(rec.fields()[1].val, FieldValue::Bool(true)));
        assert!(drain.try_next().is_none(), "exactly one record");
    }

    #[test]
    fn now_and_wait_until_track_the_installed_clock() {
        use crate::clock::MockClock;
        use crate::time::Timestamp;

        let mut ctx = test_ctx(ElementId(1));
        // No clock installed → running time is NONE and a wait never blocks (no pacing).
        assert!(ctx.now().is_none());
        assert_eq!(ctx.wait_until(Timestamp::from_secs(1)), WaitOutcome::Reached);

        // Install a mock clock with the running-time base at 100ms (the shared-cell
        // form the pause transport re-bases in a real run).
        let clock = MockClock::new();
        ctx.set_clock(
            Arc::new(clock.clone()),
            Arc::new(std::sync::atomic::AtomicU64::new(
                Timestamp::from_millis(100).0,
            )),
        );
        assert_eq!(ctx.now(), Timestamp::ZERO, "at the base instant, running time is 0");
        clock.advance(Timestamp::from_millis(150));
        assert_eq!(ctx.now(), Timestamp::from_millis(50), "running = clock(150) - base(100)");

        // A running-time deadline already behind us returns at once...
        assert_eq!(
            ctx.wait_until(Timestamp::from_millis(50)),
            WaitOutcome::Reached
        );
        // ...and one exactly at the (advanced) clock, too — deadline = base + running.
        clock.advance(Timestamp::from_millis(150)); // clock 300ms → running 200ms
        assert_eq!(
            ctx.wait_until(Timestamp::from_millis(200)),
            WaitOutcome::Reached
        );
    }

    #[test]
    fn ctx_gate_suppresses_levels_below_threshold() {
        // Hermetic: the `LevelFilter` is set directly (no `STREAMCRAFT_DEBUG`), so the
        // gate — not the environment — decides what passes.
        let filter = Arc::new(LevelFilter::with_level(Level::Warn));
        let (sink, drain) = log_channel(64);
        let mut log = Log::new(sink, Arc::clone(&filter));
        log.set_element(ElementId(3), "testsrc");

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
