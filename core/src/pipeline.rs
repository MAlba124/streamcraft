//! The pipeline: owns topology, lifecycle, clock, latency (spec: The pipeline API).
//! Elements arrive already constructed — `pipeline.add(FileSrc::new(path))`.
//!
//! Milestone 1 → thread-group scheduler (spec: Scheduling and threading): the linear
//! chain is split into **groups** at active-element boundaries; each group runs on
//! its own thread (an active head element followed by any inline passive elements),
//! with a lock-free SPSC ring queue at each boundary and its own reactor. Passive
//! chains run inline as function calls; real queues exist only between groups. For an
//! all-active chain (e.g. `filesrc ! filesink`) this yields one thread per element
//! with a ring between them.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::batch::{Batch, Inputs};
use crate::bus::{Bus, BusMessage, BusSender, State};
use crate::clock::{Clock, InstantClock};
use crate::counters::{CounterSnapshot, ElementCounters, LatencyReport, TapHandle};
use crate::ctx::{Ctx, SeekState};
use crate::element::{Direction, Element, ElementDesc, Flow, SchedHint, Template};
use crate::error::Error;
use crate::event::Event;
use crate::format::{negotiate, FieldConstraint, FixedFormat, OfferDesc, Value, Vocabulary};
use crate::id::{ElementId, FieldId, FormatId, GroupId, Interner, LinkId, PadId, ValueId};
use crate::io::{Reactor, ReactorFactory, SyncReactor};
use crate::log::{log_channel, Level, LevelFilter, Log, LogDrain};
use crate::memory::Pool;
use crate::props::{validate_and_set, PropHandle, PropTable};
use crate::ring::{spsc, Consumer, Producer};
use crate::time::Timestamp;

/// Milestone raw-bytes format id. Real negotiation (spec: Formats) assigns these.
const BYTES: FormatId = FormatId(0);

/// How many unconsumed buffers a co-grouped element's input may hold before the
/// scheduler stops running its upstream (inline backpressure — see `run_group`).
/// A few batches, mirroring the boundary-ring capacity philosophy: enough to
/// amortize, small enough that no single chain position can starve the shared pool.
const INLINE_INPUT_CAP: usize = 8;

/// A negotiated edge: the two pads it joins (by element + local pad index) and the
/// [`FixedFormat`] the solver fixed on it at [`link`](Pipeline::link) time (spec:
/// Formats — fixed formats live on edges).
struct Edge {
    src: ElementId,
    src_pad: usize,
    sink: ElementId,
    sink_pad: usize,
    format: FixedFormat,
}

/// A pad instantiated at runtime (spec: dynamic pads), registered by
/// [`Pipeline::preroll`] so [`Pipeline::link`] can resolve it by name and the scheduler
/// can size the element's per-pad tables. Its `offers` are `&'static` (the element hands
/// `add_pad` a static offer menu), so negotiating a dynamic edge is identical to a static
/// one.
struct DynPad {
    name: String,
    direction: Direction,
    offers: &'static [OfferDesc],
    index: usize,
}

/// A pad that appeared during [`Pipeline::preroll`] (spec: dynamic pads). Link it to a
/// downstream with [`Pipeline::link`] using `name`; `pad` is its local id on `element`.
#[derive(Clone, Debug)]
pub struct AddedPadInfo {
    pub element: ElementId,
    pub pad: PadId,
    pub name: String,
}

/// A summary of a finished [`Pipeline::run`], for tests and profiling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunReport {
    /// Payload buffers ever heap-allocated. Flat == the zero-alloc criterion.
    pub pool_slot_allocations: u64,
    pub pool_high_water: u64,
    pub buffers: u64,
}

/// A cloneable handle that stops a running pipeline from another thread (e.g. a
/// Ctrl-C handler). Cooperative: group threads check it between operations and then
/// drain and exit, which closes the ring queues and cascades the stop downstream
/// (spec: milestone 2 — cancellation; hard-interrupting a blocked reactor/clock wait
/// is a follow-up).
#[derive(Clone)]
pub struct StopHandle(Arc<AtomicBool>);

/// The shared pause transport (spec: Clocking — "pause is a clock op, not a
/// state"). Pausing freezes **running time**: the resume re-bases `base_time`
/// forward by the paused interval (measured on the pipeline clock, so a device
/// clock that itself freezes while paused contributes zero shift), group threads
/// park at their pass gate after `Event::Paused` is delivered, and a sink whose
/// clock wait expires mid-pause blocks in `ctx.wait_until` until resume — then
/// re-derives its deadline from the shifted base.
pub(crate) struct PauseShared {
    /// Lock-free mirror of `inner.paused` for the per-pass / per-render fast checks
    /// (one relaxed load); the mutex is only taken when it reads `true` or on
    /// transitions.
    hint: AtomicBool,
    inner: Mutex<PauseInner>,
    cond: Condvar,
    /// Raw ns of the running-time base — the same cell `TapHandle` reads; resume
    /// shifts it forward so paused time never counts as running time.
    base: Arc<AtomicU64>,
    /// The pipeline stop flag, so a pause-parked wait still honours shutdown.
    stop: Arc<AtomicBool>,
}

struct PauseInner {
    paused: bool,
    /// Clock reading at the moment of pause — the resume shift is `now - pause_at`.
    pause_at: Timestamp,
    /// The selected pipeline clock, installed by `run()` (clock selection happens
    /// there); `None` before the first run — pause/resume still latch, without a
    /// base shift (there is no running time to preserve yet).
    clock: Option<Arc<dyn Clock>>,
}

impl PauseShared {
    fn new(base: Arc<AtomicU64>, stop: Arc<AtomicBool>) -> Arc<Self> {
        Arc::new(Self {
            hint: AtomicBool::new(false),
            inner: Mutex::new(PauseInner {
                paused: false,
                pause_at: Timestamp::ZERO,
                clock: None,
            }),
            cond: Condvar::new(),
            base,
            stop,
        })
    }

    /// The hot-path check: one relaxed load while playing.
    pub(crate) fn maybe_paused(&self) -> bool {
        self.hint.load(Ordering::Relaxed)
    }

    /// Install the selected clock (run() → here, after clock selection).
    fn install_clock(&self, clock: Arc<dyn Clock>) {
        self.inner.lock().unwrap().clock = Some(clock);
    }

    pub(crate) fn is_paused(&self) -> bool {
        self.inner.lock().unwrap().paused
    }

    fn pause(&self) {
        let mut i = self.inner.lock().unwrap();
        if !i.paused {
            i.pause_at = i.clock.as_ref().map_or(Timestamp::ZERO, |c| c.now());
            i.paused = true;
            self.hint.store(true, Ordering::Release);
        }
    }

    fn resume(&self) {
        let mut i = self.inner.lock().unwrap();
        if i.paused {
            // Excise the paused interval from running time: shift the base forward by
            // however far the clock moved while paused (a device clock frozen during
            // pause moves zero — both cases come out right). Shift and state flip are
            // under one lock, so a waiter re-deriving its deadline after the wake
            // sees the shifted base.
            if let Some(c) = &i.clock {
                let delta = c.now().saturating_sub(i.pause_at);
                let base = self.base.load(Ordering::Acquire);
                if base != u64::MAX {
                    self.base
                        .store(base.saturating_add(delta.nanos().unwrap_or(0)), Ordering::Release);
                }
            }
            i.paused = false;
            self.hint.store(false, Ordering::Release);
        }
        drop(i);
        self.cond.notify_all();
    }

    /// Block while paused (a sink's expired clock wait lands here; also the group
    /// gate). Polls the stop flag so shutdown still wins over a paused pipeline.
    /// Returns `false` if woken by stop.
    pub(crate) fn block_while_paused(&self) -> bool {
        let mut i = self.inner.lock().unwrap();
        while i.paused {
            if self.stop.load(Ordering::Acquire) {
                return false;
            }
            let (guard, _) = self
                .cond
                .wait_timeout(i, std::time::Duration::from_millis(100))
                .unwrap();
            i = guard;
        }
        true
    }
}

/// Cloneable pause/resume control (spec: Clocking — "pause is a clock op, not a
/// state"), the same shape as `StopHandle`/`SeekHandle`/`PropHandle`. Obtain from
/// [`Pipeline::pause_handle`]; usable from any thread while `run()` blocks.
#[derive(Clone)]
pub struct PauseHandle(pub(crate) Arc<PauseShared>);

impl PauseHandle {
    /// Freeze running time and park the streaming threads at their next safe point
    /// (each group delivers `Event::Paused` to its elements first, so a device sink
    /// holds its hardware). A sink already inside a clock wait finishes that wait
    /// but blocks before rendering — nothing renders while paused.
    pub fn pause(&self) {
        self.0.pause();
    }

    /// Continue: running time resumes exactly where it stopped (the paused interval
    /// is excised by re-basing), groups deliver `Event::Resumed` and run on.
    pub fn resume(&self) {
        self.0.resume();
    }

    /// Flip paused↔playing; returns `true` if now paused.
    pub fn toggle(&self) -> bool {
        if self.0.is_paused() {
            self.0.resume();
            false
        } else {
            self.0.pause();
            true
        }
    }

    pub fn is_paused(&self) -> bool {
        self.0.is_paused()
    }
}

impl StopHandle {
    pub fn stop(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_stopped(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// A handle to seek this pipeline from another thread while `run()` is blocking (spec:
/// flush/seek). Publishing a target makes the source resume reading at `to_byte`, discards
/// the data already in flight (out-of-band, so it is responsive rather than playing out the
/// queue first), and resets the reported play position to `to_frame`. Cloneable and cheap.
#[derive(Clone)]
pub struct SeekHandle(Arc<SeekState>);

impl SeekHandle {
    /// Request a seek: resume the source at byte offset `to_byte`, and reset the reported
    /// play position to PCM frame `to_frame`. The targets are published before the
    /// generation bump, so any observer that sees the new generation also sees the targets.
    /// Mapping a wall-clock time to a byte offset is the caller's job (a byte source has no
    /// notion of time); for FLAC without a seektable that is a proportional estimate the
    /// decoder then re-syncs from.
    pub fn seek(&self, to_byte: u64, to_frame: u64) {
        self.0.to_byte.store(to_byte, Ordering::Release);
        self.0.to_frame.store(to_frame, Ordering::Release);
        self.0.gen.fetch_add(1, Ordering::Release);
    }
}

pub struct Pipeline {
    elements: Vec<Option<Box<dyn Element>>>,
    /// Each element's `&'static` descriptor, captured at [`add`](Self::add). Outlives
    /// the element being moved into its group thread at `run()`, so name lookups,
    /// property validation, and handles keep working while streaming.
    descs: Vec<&'static ElementDesc>,
    /// Per-element property mailboxes (spec: Dynamic element properties), created at
    /// [`add`](Self::add) and shared with the element's `Ctx` (reader) and any
    /// [`PropHandle`] (writers).
    props: Vec<Arc<PropTable>>,
    edges: Vec<Edge>,
    // The three interning domains (spec: Formats — one interner per domain). Offers
    // declared as `&'static str` on pads are lowered through these at link time, so
    // ids are consistent across the whole graph and the solver only ever sees `u32`s.
    formats: Interner,
    fields: Interner,
    values: Interner,
    stop: Arc<AtomicBool>,
    /// Shared seek request (spec: flush/seek). Bumped by a [`SeekHandle`]; observed by every
    /// group thread (to flush) and by the source/sink elements (to re-seek / reset position).
    seek: Arc<SeekState>,
    bus_sender: BusSender,
    bus: Bus,
    slot_size: usize,
    pool_slots: usize,
    queue_cap: usize,
    credits: u32,
    reactor_factory: Option<ReactorFactory>,
    counters: Vec<Arc<ElementCounters>>,
    last_report: Option<RunReport>,
    /// Programmatic log level (spec: Debuggability). `None` leaves the gate to
    /// `STREAMCRAFT_DEBUG` alone; when neither is set, `run()` wires no logging at all
    /// (no channels, no drain thread) — the zero-overhead default that keeps the hot
    /// path clean when nobody is looking.
    log_level: Option<Level>,
    /// Capacity (in records) of each element's log ring. A full ring drops and counts
    /// (spec: never block a streaming thread), so this trades memory for tolerance of
    /// bursts before the low-priority drain catches up.
    log_queue_cap: usize,
    /// The pipeline clock (spec: Clocking and synchronization). Defaults to the real
    /// monotonic [`InstantClock`]; a test installs a `MockClock` for deterministic,
    /// time-compressed runs, and later a sink can provide a device clock. [`run`](Self::run)
    /// samples `base_time = clock.now()` at start, then every element shares this one
    /// timeline via `ctx.now()` / `ctx.wait_until()`.
    clock: Arc<dyn Clock>,
    /// Whether the application forced the clock via [`set_clock`](Self::set_clock).
    /// When false, [`run`](Self::run) lets the most downstream element that
    /// [`provide_clock`](crate::element::Element::provide_clock)s master the pipeline
    /// (spec: Clocking — an audio sink's device clock paces everyone).
    clock_explicit: bool,
    /// Raw ns of the running-time base sampled by the current/last [`run`](Self::run)
    /// (`u64::MAX` before the first run), shared into [`TapHandle`]s so observers can
    /// window counter deltas over running time (spec: Taps — bitrate is Δbytes/Δt).
    base_shared: Arc<AtomicU64>,
    /// The pause transport (spec: Clocking — pause is a clock op). See [`PauseHandle`].
    pause: Arc<PauseShared>,
    /// Start the next [`run`](Self::run) paused (preroll-and-hold: everything spins
    /// up, the first buffers queue to the sinks, nothing renders until
    /// [`PauseHandle::resume`]). See [`start_paused`](Self::start_paused).
    initial_paused: bool,
    /// Latency tracing gate (spec: Debuggability): while set, the scheduler times
    /// `process()` calls, stamps ring pushes for residency measurement, and sinks
    /// record wait overshoot — into the per-element histograms a [`TapHandle`]
    /// reads. Off (the default, unless `STREAMCRAFT_TRACE=1`) the streaming path
    /// pays one relaxed load per call. Toggle via [`set_tracing`](Self::set_tracing),
    /// live — it is read per pass.
    tracing: Arc<AtomicBool>,
    /// Per-element output-pool overrides (`(slot_size, slots)`), indexed by
    /// `ElementId`; `None` shares the pipeline default pool. See
    /// [`set_element_pool`](Self::set_element_pool).
    pool_overrides: Vec<Option<(usize, u32)>>,
    /// Per-element inbound-ring capacity overrides (in batches), indexed by
    /// `ElementId`; `None` uses `queue_cap`. See
    /// [`set_queue_capacity`](Self::set_queue_capacity).
    queue_overrides: Vec<Option<usize>>,
    /// Pads instantiated at runtime, per element (spec: dynamic pads). Indexed by
    /// `ElementId`; grown to match `elements` in [`add`](Self::add) and populated by
    /// [`preroll`](Self::preroll). Empty for a static pipeline.
    dyn_pads: Vec<Vec<DynPad>>,
}

impl Pipeline {
    pub fn new() -> Self {
        let (tx, rx) = Bus::channel();
        // Bound first: the pause transport shares the running-time base cell and the
        // stop flag (a paused wait must still honour shutdown).
        let stop = Arc::new(AtomicBool::new(false));
        let base_shared = Arc::new(AtomicU64::new(u64::MAX));
        Self {
            elements: Vec::new(),
            descs: Vec::new(),
            props: Vec::new(),
            edges: Vec::new(),
            formats: Interner::new(),
            fields: Interner::new(),
            values: Interner::new(),
            stop: Arc::clone(&stop),
            seek: Arc::new(SeekState::default()),
            bus_sender: tx,
            bus: rx,
            slot_size: 128 * 1024,
            // Sized so the ring (queue_cap batches) is the binding backpressure, not
            // the pool — the source blocks on a full ring, never spins on the pool.
            pool_slots: 64,
            queue_cap: 4,
            credits: 4,
            reactor_factory: None,
            counters: Vec::new(),
            last_report: None,
            log_level: None,
            log_queue_cap: 1024,
            clock: Arc::new(InstantClock::new()),
            clock_explicit: false,
            base_shared: Arc::clone(&base_shared),
            pause: PauseShared::new(base_shared, stop),
            initial_paused: false,
            tracing: Arc::new(AtomicBool::new(
                std::env::var_os("STREAMCRAFT_TRACE").is_some_and(|v| v != "0"),
            )),
            dyn_pads: Vec::new(),
            pool_overrides: Vec::new(),
            queue_overrides: Vec::new(),
        }
    }

    /// Enable/disable latency tracing (spec: Debuggability): per-element histograms of
    /// `process()` time, ring residency, and sink wait overshoot, read via
    /// [`TapHandle::latency`](crate::counters::TapHandle::latency). Also enabled by
    /// `STREAMCRAFT_TRACE=1`. Live — takes effect on the next scheduler pass; the cost
    /// while on is two monotonic clock reads per batch.
    pub fn set_tracing(&mut self, on: bool) {
        self.tracing.store(on, Ordering::Release);
    }

    /// Install a per-thread-group reactor factory (e.g. io_uring). Each group thread
    /// calls it to build its own reactor. Defaults to the dependency-free
    /// [`SyncReactor`] (spec: IO — the reactor is the portability boundary).
    pub fn set_reactor_factory(&mut self, factory: ReactorFactory) {
        self.reactor_factory = Some(factory);
    }

    /// Force the pipeline clock (spec: Clocking and synchronization), replacing the
    /// default real [`InstantClock`] — e.g. a `MockClock` for deterministic,
    /// time-compressed tests. Takes effect at the next [`run`](Self::run), which
    /// samples the running-time base from it; call before `run()`. A forced clock
    /// also wins over any element-provided one (see
    /// [`Element::provide_clock`](crate::element::Element::provide_clock)); without
    /// it, `run()` prefers the most downstream provider — the audio-master case.
    pub fn set_clock(&mut self, clock: Arc<dyn Clock>) {
        self.clock = clock;
        self.clock_explicit = true;
    }

    /// A handle to stop this pipeline from another thread (or a signal handler)
    /// while `run()` is blocking.
    pub fn stop_handle(&self) -> StopHandle {
        StopHandle(Arc::clone(&self.stop))
    }

    /// Pause/resume control usable from any thread while `run()` blocks (spec:
    /// Clocking — pause is a clock op, not a state). See [`PauseHandle`].
    pub fn pause_handle(&self) -> PauseHandle {
        PauseHandle(Arc::clone(&self.pause))
    }

    /// Start the next [`run`](Self::run) paused — preroll-and-hold: the pipeline
    /// spins up, decoders fill the queues to the sinks, and nothing renders until
    /// [`PauseHandle::resume`]. Running time starts at the resume (the pre-resume
    /// interval is excised like any other pause).
    pub fn start_paused(&mut self, on: bool) {
        self.initial_paused = on;
    }

    /// A handle to seek this pipeline from another thread while `run()` is blocking
    /// (spec: flush/seek). See [`SeekHandle`].
    pub fn seek_handle(&self) -> SeekHandle {
        SeekHandle(Arc::clone(&self.seek))
    }

    /// Configure the shared buffer pool for the next [`run`](Self::run) (spec: Memory —
    /// pools are sized at state changes, never per buffer). `slot_size` is each pooled
    /// buffer's capacity in bytes (video frames need `w*h*3/2` for I420); `slots` caps
    /// how many live at once — the allocation backstop behind the ring backpressure.
    /// Until per-link pool negotiation lands (spec: Formats — pool negotiation is
    /// decoupled), one pool serves the whole pipeline.
    pub fn set_pool(&mut self, slot_size: usize, slots: u32) {
        self.slot_size = slot_size.max(1);
        self.pool_slots = slots.max(1) as usize;
    }

    /// Programmatically enable logging up to `level`, formatting records to stderr on a
    /// dedicated drain thread during [`run`](Self::run) (spec: Debuggability — the "one
    /// line in `main`" dev path, without needing `STREAMCRAFT_DEBUG` in the
    /// environment). `STREAMCRAFT_DEBUG`, if set, still applies on top at `run()` time
    /// and takes the more verbose of the two globals.
    pub fn log_to_stderr(&mut self, level: Level) {
        self.log_level = Some(level);
    }

    /// Set (or clear, with `None`) the programmatic log level. See
    /// [`log_to_stderr`](Self::log_to_stderr).
    pub fn set_log_level(&mut self, level: Option<Level>) {
        self.log_level = level;
    }

    // --- Topology: legal in every state (spec: Runtime configuration) ---

    /// Elements arrive already constructed: `pipeline.add(FileSrc::new(path))`.
    pub fn add(&mut self, element: impl Element + 'static) -> ElementId {
        self.add_boxed(Box::new(element))
    }

    /// Add an already-boxed element (spec: Plugins — the parse layer builds elements by
    /// name via `ElementDesc::make_default`, which yields a `Box<dyn Element>`). The
    /// typed [`add`](Self::add) is the primary path; this is its type-erased twin.
    pub fn add_boxed(&mut self, element: Box<dyn Element>) -> ElementId {
        let id = ElementId(self.elements.len() as u32);
        let desc = element.desc();
        self.descs.push(desc);
        // The property mailbox and counters are created here — not per run — so
        // handles taken before `run()` observe the streaming run (spec: Dynamic
        // element properties; Taps). Counters are cumulative across runs.
        self.props.push(Arc::new(PropTable::new(desc.props.len())));
        self.counters.push(Arc::new(ElementCounters::default()));
        self.elements.push(Some(element));
        self.dyn_pads.push(Vec::new());
        self.pool_overrides.push(None);
        self.queue_overrides.push(None);
        id
    }

    /// Give one element its **own** output pool, overriding the pipeline default
    /// (spec: Formats — "pool negotiation is decoupled"; this is its explicit v1).
    /// One pipeline-wide slot size cannot serve a demuxer's ~16 KB samples and a
    /// decoder's multi-MB frames at once — that mismatch has cost real gigabytes.
    /// Elements without an override share the default pool ([`set_pool`](Self::set_pool)).
    /// Automatic sizing from the negotiated format is the follow-up; core stays
    /// format-agnostic, so it will arrive as element-declared requirements, not core
    /// interpreting families.
    pub fn set_element_pool(&mut self, el: ElementId, slot_size: usize, slots: u32) {
        if let Some(o) = self.pool_overrides.get_mut(el.0 as usize) {
            *o = Some((slot_size.max(1), slots.max(1)));
        }
    }

    /// Deepen (or shrink) the ring feeding **into** `el`, overriding the pipeline
    /// default capacity in batches (spec: Queues — capacity is the scheduler's, not
    /// the element's). Applies to every inter-group edge whose consumer is `el`;
    /// an inlined (intra-group) hop has no ring, so a buffering point must be an
    /// Active element — this is exactly how the explicit `queue` element gets its
    /// depth: add it, then size its inbound ring here.
    pub fn set_queue_capacity(&mut self, el: ElementId, batches: usize) {
        if let Some(o) = self.queue_overrides.get_mut(el.0 as usize) {
            *o = Some(batches.max(1));
        }
    }

    /// Link a src pad to a sink pad, negotiating their formats (spec: Formats — the
    /// pipeline solves the link as a constraint pass at link time). Both pads' static
    /// offers are lowered (interned) through this pipeline's interners and intersected;
    /// the resulting [`FixedFormat`] is stored on the edge and later handed to both
    /// elements via [`Ctx::negotiated`](crate::ctx::Ctx::negotiated). A pad name/
    /// direction mismatch, or an empty format intersection, is a loud error here —
    /// never a mid-stream failure.
    pub fn link(&mut self, src: (ElementId, &str), sink: (ElementId, &str)) -> Result<LinkId, Error> {
        let src_pad = self.check_pad(src.0, src.1, Direction::Src)?;
        let sink_pad = self.check_pad(sink.0, sink.1, Direction::Sink)?;

        // Lower both pads' string-keyed offers to id-based offers (interning family/
        // field/value names once, here), then intersect. Offer menus are `'static`
        // (static pads *and* dynamic ones, whose `add_pad` menu is `&'static`), so the
        // borrows survive the two `&mut self` lowering passes.
        let src_offers_desc = self
            .pad_offers(src.0, src_pad)
            .ok_or(Error::Todo("link: no offers for src pad"))?;
        let src_offers: Vec<_> = src_offers_desc
            .iter()
            .map(|o| o.lower(&mut self.formats, &mut self.fields, &mut self.values))
            .collect();
        let sink_offers_desc = self
            .pad_offers(sink.0, sink_pad)
            .ok_or(Error::Todo("link: no offers for sink pad"))?;
        let sink_offers: Vec<_> = sink_offers_desc
            .iter()
            .map(|o| o.lower(&mut self.formats, &mut self.fields, &mut self.values))
            .collect();

        let format = negotiate(&src_offers, &sink_offers).ok_or_else(|| {
            let (sn, sp) = (self.name_of(src.0), src.1);
            let (dn, dp) = (self.name_of(sink.0), sink.1);
            Error::Resource(format!(
                "link: no common format between {sn}.{sp} and {dn}.{dp} \
                 (offers {:?} vs {:?}) — negotiation failed",
                self.families(&src_offers),
                self.families(&sink_offers),
            ))
        })?;

        let id = LinkId(self.edges.len() as u32);
        self.edges.push(Edge {
            src: src.0,
            src_pad,
            sink: sink.0,
            sink_pad,
            format,
        });
        Ok(id)
    }

    /// Verify a named pad exists on an element with the expected direction, returning its
    /// local pad index (`PadId`). Searches the static `desc().pads` first, then the
    /// runtime-added (dynamic) pads registered by [`preroll`](Self::preroll).
    fn check_pad(&self, el: ElementId, pad: &str, dir: Direction) -> Result<usize, Error> {
        let e = self
            .elements
            .get(el.0 as usize)
            .and_then(|o| o.as_ref())
            .ok_or(Error::Todo("link: unknown element"))?;
        if let Some(i) = e.desc().pads.iter().position(|p| p.name == pad) {
            return if e.desc().pads[i].direction == dir {
                Ok(i)
            } else {
                Err(Error::Resource(format!(
                    "link: pad '{pad}' on '{}' has the wrong direction",
                    e.desc().name
                )))
            };
        }
        if let Some(dp) = self
            .dyn_pads
            .get(el.0 as usize)
            .and_then(|dps| dps.iter().find(|d| d.name == pad))
        {
            return if dp.direction == dir {
                Ok(dp.index)
            } else {
                Err(Error::Resource(format!(
                    "link: dynamic pad '{pad}' on '{}' has the wrong direction",
                    e.desc().name
                )))
            };
        }
        Err(Error::Resource(format!(
            "link: element '{}' has no pad '{pad}'",
            e.desc().name
        )))
    }

    /// The `&'static` offer menu for a pad by index — static (`desc().pads[i]`) or
    /// dynamic ([`preroll`](Self::preroll)-added). `&'static`, so a caller can hold it
    /// across the `&mut self` interning passes in [`link`](Self::link).
    fn pad_offers(&self, el: ElementId, index: usize) -> Option<&'static [OfferDesc]> {
        let e = self.elements.get(el.0 as usize)?.as_ref()?;
        let static_pads = e.desc().pads;
        if index < static_pads.len() {
            Some(static_pads[index].offers)
        } else {
            self.dyn_pads
                .get(el.0 as usize)?
                .iter()
                .find(|d| d.index == index)
                .map(|d| d.offers)
        }
    }

    /// An element's descriptor name, for link diagnostics. Reads the captured desc,
    /// so it works even while the element itself is off in its group thread.
    fn name_of(&self, el: ElementId) -> &'static str {
        self.descs.get(el.0 as usize).map_or("?", |d| d.name)
    }

    /// The family names of a set of lowered offers, resolved back to strings for a
    /// negotiation-failure message.
    fn families(&self, offers: &[crate::format::FormatOffer]) -> Vec<&str> {
        offers
            .iter()
            .map(|o| self.formats.resolve(o.family.0).unwrap_or("?"))
            .collect()
    }

    /// The format negotiated on this pipeline's edges, resolved into
    /// per-element/per-pad tables (indexed by `ElementId`, then local pad index) to
    /// hand each element's `Ctx`. An element sees the fixed format on every one of its
    /// linked pads. Built once at `run()` from the already-solved edges — no solving
    /// happens here.
    fn negotiated_by_element(&self) -> Vec<Vec<Option<FixedFormat>>> {
        let mut per: Vec<Vec<Option<FixedFormat>>> = self
            .elements
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let static_n = e.as_ref().map_or(0, |el| el.desc().pads.len());
                // Cover runtime-added pad indices too, so a dynamic edge's format lands.
                let dyn_n = self
                    .dyn_pads
                    .get(i)
                    .and_then(|d| d.iter().map(|p| p.index + 1).max())
                    .unwrap_or(0);
                vec![None; static_n.max(dyn_n)]
            })
            .collect();
        for edge in &self.edges {
            if let Some(slot) = per
                .get_mut(edge.src.0 as usize)
                .and_then(|v| v.get_mut(edge.src_pad))
            {
                *slot = Some(edge.format.clone());
            }
            if let Some(slot) = per
                .get_mut(edge.sink.0 as usize)
                .and_then(|v| v.get_mut(edge.sink_pad))
            {
                *slot = Some(edge.format.clone());
            }
        }
        per
    }

    /// The [`FixedFormat`] fixed on a link (spec: Formats — fixed formats live on
    /// edges). Available after a successful [`link`](Self::link), for tests and dumps.
    pub fn negotiated(&self, link: LinkId) -> Option<&FixedFormat> {
        self.edges.get(link.0 as usize).map(|e| &e.format)
    }

    /// Resolve an interned format-family id back to its name (for dumps / tests).
    pub fn family_name(&self, id: FormatId) -> Option<&str> {
        self.formats.resolve(id.0)
    }

    /// Resolve an interned field id back to its name (for dumps / tests).
    pub fn field_name(&self, id: FieldId) -> Option<&str> {
        self.fields.resolve(id.0)
    }

    /// The id a field name interned to, if this pipeline has seen it (i.e. some linked
    /// pad declared it). Lets a test look a negotiated field up by name.
    pub fn field_id(&self, name: &str) -> Option<FieldId> {
        self.fields.get(name).map(FieldId)
    }

    /// The id a categorical value name interned to, if seen. For tests asserting on a
    /// negotiated categorical value (pixel format, sample format).
    pub fn value_id(&self, name: &str) -> Option<ValueId> {
        self.values.get(name).map(ValueId)
    }

    // --- Preroll (spec: dynamic pads — topology settles, then freezes) ---

    /// Run each element's discovery phase so a demuxer can instantiate its dynamic src
    /// pads (spec: dynamic pads). Returns the pads that appeared — link each to a
    /// downstream with [`link`](Self::link) (by `name`) before calling [`run`](Self::run);
    /// each is also posted as a [`BusMessage::PadAdded`](crate::bus::BusMessage). Call
    /// after the static edges are linked. Topology is expected to settle here and then be
    /// frozen for the streaming phase — mid-stream topology change is out of scope. A
    /// static pipeline need not call this (every `preroll` defaults to a no-op).
    pub fn preroll(&mut self) -> Result<Vec<AddedPadInfo>, Error> {
        // A throwaway pool: preroll only instantiates pads, it does not stream buffers.
        let pool = Pool::bounded(self.slot_size, 1);
        let mut appeared = Vec::new();
        for i in 0..self.elements.len() {
            let static_len = match &self.elements[i] {
                Some(e) => e.desc().pads.len(),
                None => continue,
            };
            let id = ElementId(i as u32);
            let mut ctx = Ctx::new(pool.clone(), self.bus_sender.clone(), id, BYTES, self.credits);
            ctx.configure_pads(static_len, &[]); // hand out dynamic ids past the static pads
            if let Some(el) = self.elements[i].as_mut() {
                el.preroll(&mut ctx)?;
            }
            for ap in ctx.take_added_pads() {
                // A best-effort family for the notification; the concrete format is fixed
                // when the pad is linked (interning happens there too).
                let family = ap
                    .offers
                    .first()
                    .map(|o| FormatId(self.formats.intern(o.family)))
                    .unwrap_or(FormatId(0));
                self.bus_sender.send(BusMessage::PadAdded {
                    element: id,
                    pad: ap.pad,
                    format: FixedFormat::new(family),
                });
                appeared.push(AddedPadInfo { element: id, pad: ap.pad, name: ap.name.clone() });
                self.dyn_pads[i].push(DynPad {
                    name: ap.name,
                    direction: ap.direction,
                    offers: ap.offers,
                    index: ap.pad.0 as usize,
                });
            }
        }
        Ok(appeared)
    }

    // --- Running (finite: drives to EOS across all group threads) ---

    /// Start the chain, drive it to EOS, and stop it. Spawns one thread per group,
    /// joins them all, and returns the first error (if any).
    pub fn run(&mut self) -> Result<(), Error> {
        self.stop.store(false, Ordering::Release);
        let order = self.topo_order()?;
        // Clock selection (spec: Clocking): a clock forced by `set_clock` wins;
        // otherwise the most downstream element offering one masters the pipeline —
        // walked sinks-first so an audio sink's device clock beats anything upstream.
        if !self.clock_explicit {
            for id in order.iter().rev() {
                if let Some(el) = self.elements[id.0 as usize].as_mut() {
                    if let Some(c) = el.provide_clock() {
                        self.clock = c;
                        break;
                    }
                }
            }
        }
        // Sample the running-time base once, at the instant the pipeline starts (spec:
        // Clocking — running time = clock.now() - base_time). The clock and base are
        // shared into every group so all elements share one timeline.
        let clock = Arc::clone(&self.clock);
        let base_time = clock.now();
        // Publish the base so TapHandles can window counter deltas over running time.
        self.base_shared.store(base_time.0, Ordering::Release);
        // Arm the pause transport with the selected clock (its resume re-bases the
        // shared base); a start-paused run holds at the first gate/render.
        self.pause.install_clock(Arc::clone(&clock));
        if self.initial_paused {
            self.pause.pause();
        }
        let groups = self.compute_groups(&order)?;
        let ng = groups.len();
        let total = self.elements.len();
        // The default pool plus per-element overrides (spec: pool negotiation,
        // explicit v1 — see set_element_pool). Elements without an override share
        // the default; each override is its own recycling domain, so a demuxer's
        // small samples and a decoder's frames stop competing for one slot size.
        let pool = Pool::bounded(self.slot_size, self.pool_slots as u32);
        let mut distinct_pools: Vec<Pool> = vec![pool.clone()]; // for the run report
        let pools: Vec<Pool> = (0..total)
            .map(|i| match self.pool_overrides.get(i).copied().flatten() {
                Some((slot, slots)) => {
                    let p = Pool::bounded(slot, slots);
                    distinct_pools.push(p.clone());
                    p
                }
                None => pool.clone(),
            })
            .collect();
        let credits = self.credits;
        let factory: ReactorFactory = self
            .reactor_factory
            .clone()
            .unwrap_or_else(|| Arc::new(|| Ok(Box::new(SyncReactor::new()) as Box<dyn Reactor>)));

        // Map each element to its group, so an edge is intra-group (inlined — a function
        // call) or inter-group (a ring).
        let mut group_of = vec![0usize; total];
        for (gi, ids) in groups.iter().enumerate() {
            for id in ids {
                group_of[id.0 as usize] = gi;
            }
        }

        // One SPSC ring per inter-group edge. A group may have several downstream rings
        // (fan-out — a demuxer's src pads), each tagged with the tail's src pad; and
        // several upstream rings (fan-in — a muxer's sink pads), each tagged with the
        // head's sink pad so the aggregator reads each pad independently.
        let mut group_downstream: Vec<Vec<(PadId, Producer<Batch>, Consumer<Batch>)>> =
            (0..ng).map(|_| Vec::new()).collect();
        let mut group_upstream: Vec<Vec<(PadId, Consumer<Batch>, Producer<Batch>)>> =
            (0..ng).map(|_| Vec::new()).collect();
        for edge in &self.edges {
            let gs = group_of[edge.src.0 as usize];
            let gd = group_of[edge.sink.0 as usize];
            if gs == gd {
                continue; // inlined into one thread
            }
            if *groups[gs].last().unwrap() != edge.src {
                return Err(Error::Todo(
                    "inter-group branch from a non-tail element is unsupported \
                     (make the branch point active so it forms its own group)",
                ));
            }
            let cap = self
                .queue_overrides
                .get(edge.sink.0 as usize)
                .copied()
                .flatten()
                .unwrap_or(self.queue_cap);
            let (p, c) = spsc::<Batch>(cap);
            // The shell return ring (spec: Batching — zero-alloc transport): the
            // consumer sends drained batch shells back so the producer's
            // `take_output` reuses their column capacity instead of allocating.
            // Non-blocking on both ends; an empty/full shell ring just means an
            // alloc/drop — correctness never depends on it.
            let (shell_tx, shell_rx) = spsc::<Batch>(cap);
            group_downstream[gs].push((PadId(edge.src_pad as u32), p, shell_rx));
            group_upstream[gd].push((PadId(edge.sink_pad as u32), c, shell_tx));
        }

        // Per-element counters — created at `add()` and stable across runs, so a
        // TapHandle taken before this run reads them live (spec: Taps).
        let counters = self.counters.clone();
        let stop = Arc::clone(&self.stop);
        let seek = Arc::clone(&self.seek);

        // Formats fixed at link time, resolved per element so each group can install
        // them on its elements' `Ctx`s (spec: elements read their fixed format from
        // `Ctx`). `.take()` moves each element's table into its group thread.
        let mut negotiated = self.negotiated_by_element();

        // A read-only snapshot of the (now frozen) interners, shared into every group and
        // installed on each `Ctx`, so elements can resolve their negotiated format by name
        // and the scheduler can build announced formats (spec: Formats — dynamic caps).
        // Cloning is a one-time, off-hot-path cost.
        let vocabulary = Arc::new(Vocabulary {
            formats: self.formats.clone(),
            fields: self.fields.clone(),
            values: self.values.clone(),
        });

        // Logging (spec: Debuggability — Logging cont'd). Build the level filter once
        // from the programmatic level and `STREAMCRAFT_DEBUG`, then, *only if some level
        // is actually enabled*, wire one log channel per element and one low-priority
        // drain thread. When logging is off nothing here allocates or spawns — the
        // disabled path stays at zero cost, which is the point (performance is #1).
        let (mut per_elem_logs, log_drain) = self.build_logging(total);

        // Per-element pad shape (total pad count, src pad indices) — static pads plus any
        // runtime-added dynamic pads — so each Ctx sizes its per-pad output table and the
        // single-src leniency right (a demuxer with dynamic src pads is not single-src).
        let pad_infos: Vec<(usize, Vec<usize>)> = (0..total)
            .map(|i| {
                let static_pads = self.elements[i].as_ref().map(|e| e.desc().pads).unwrap_or(&[]);
                let mut n = static_pads.len();
                let mut srcs: Vec<usize> = static_pads
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.direction == Direction::Src)
                    .map(|(j, _)| j)
                    .collect();
                for dp in &self.dyn_pads[i] {
                    n = n.max(dp.index + 1);
                    if dp.direction == Direction::Src {
                        srcs.push(dp.index);
                    }
                }
                (n, srcs)
            })
            .collect();

        // Per-element upstream path latency (spec: Latency): computed once here from
        // declared latencies, installed into each Ctx so sink waits compensate it.
        let (path_latency, _) = self.compute_in_latency();

        // Spawn a thread per group.
        let mut handles: Vec<JoinHandle<Result<(), Error>>> = Vec::with_capacity(ng);
        for (gi, ids) in groups.into_iter().enumerate() {
            let mut elems: Vec<Box<dyn Element>> = Vec::with_capacity(ids.len());
            for id in &ids {
                let el = self.elements[id.0 as usize]
                    .take()
                    .ok_or(Error::Todo("element already consumed by a previous run"))?;
                elems.push(el);
            }
            let group_counters: Vec<Arc<ElementCounters>> =
                ids.iter().map(|id| Arc::clone(&counters[id.0 as usize])).collect();
            let group_props: Vec<Arc<PropTable>> =
                ids.iter().map(|id| Arc::clone(&self.props[id.0 as usize])).collect();
            let group_formats: Vec<Vec<Option<FixedFormat>>> = ids
                .iter()
                .map(|id| std::mem::take(&mut negotiated[id.0 as usize]))
                .collect();
            // Move each element's `Log` (if any) into its group thread, keyed by id.
            let group_logs: Vec<Option<Log>> = ids
                .iter()
                .map(|id| per_elem_logs[id.0 as usize].take())
                .collect();
            let upstream = std::mem::take(&mut group_upstream[gi]);
            let downstream = std::mem::take(&mut group_downstream[gi]);
            let group_pad_infos: Vec<(usize, Vec<usize>)> =
                ids.iter().map(|id| pad_infos[id.0 as usize].clone()).collect();
            let group_pools: Vec<Pool> =
                ids.iter().map(|id| pools[id.0 as usize].clone()).collect();
            let group_latencies: Vec<Timestamp> =
                ids.iter().map(|id| path_latency[id.0 as usize]).collect();
            let bus = self.bus_sender.clone();
            let factory = Arc::clone(&factory);
            let stop = Arc::clone(&stop);
            let tracing = Arc::clone(&self.tracing);
            let pause = Arc::clone(&self.pause);
            let base = Arc::clone(&self.base_shared);
            let seek = Arc::clone(&seek);
            let vocabulary = Arc::clone(&vocabulary);
            let clock = Arc::clone(&clock);
            handles.push(std::thread::spawn(move || {
                run_group(
                    elems, ids, group_formats, group_logs, upstream, downstream, factory,
                    group_pools, bus, credits, group_counters, group_props, stop, seek,
                    vocabulary, clock,
                    base, group_pad_infos, group_latencies, tracing, pause,
                )
            }));
        }

        // Join all groups; keep the first error.
        let mut first_err = None;
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                Err(_) => {
                    if first_err.is_none() {
                        first_err = Some(Error::Todo("group thread panicked"));
                    }
                }
            }
        }

        // Groups have joined, so every `Ctx` — and thus every `Log`/`LogSink` — is
        // dropped, closing the log rings. Join the drain thread, which then sees all
        // drains closed, flushes the tail, and exits. No-op when logging was disabled.
        if let Some(drain) = log_drain {
            let _ = drain.join();
        }

        // Aggregate over the default pool and every per-element override (each is
        // its own recycling domain; high-water sums as a worst-case footprint).
        let mut report = RunReport::default();
        for p in &distinct_pools {
            let s = p.stats();
            report.pool_slot_allocations += s.slot_allocations;
            report.pool_high_water += s.high_water;
            report.buffers += s.acquires;
        }
        self.last_report = Some(report);

        match first_err {
            None => {
                self.bus_sender.send(BusMessage::Eos);
                Ok(())
            }
            Some(e) => {
                self.bus_sender.send(BusMessage::Error {
                    element: ElementId(u32::MAX),
                    error: e.clone(),
                });
                Err(e)
            }
        }
    }

    /// Build the per-element log emitters and the drain thread for this run (spec:
    /// Debuggability — Logging). Returns `(logs, drain)` where `logs[i]` is the `Log`
    /// for element `i` (if logging is enabled for it) and `drain` is the join handle of
    /// the single low-priority thread that formats all records to stderr.
    ///
    /// The gate is decided once, here: the global level is `max(programmatic, env)` and
    /// per-target `name:level` rules from `STREAMCRAFT_DEBUG` raise matching elements
    /// above it. If nothing is enabled, this wires *nothing* — no channels, no thread —
    /// so a run with logging off pays only this one comparison.
    fn build_logging(&self, total: usize) -> (Vec<Option<Log>>, Option<LogDrainThread>) {
        // Fold the programmatic level and the env global into one shared gate; the more
        // verbose (higher rank) of the two wins.
        let filter = LevelFilter::new();
        if let Some(l) = self.log_level {
            filter.set(Some(l));
        }
        let spec = filter.apply_env(); // moves the global gate up if env set one
        if let Some(l) = self.log_level {
            // `apply_env` overwrites, so re-assert the programmatic floor afterwards.
            let env_rank = filter.level().map_or(0, Level::rank);
            if l.rank() > env_rank {
                filter.set(Some(l));
            }
        }
        let global_rank = filter.level().map_or(0, Level::rank);

        // Per-target overrides (`filesrc:debug`) matched against element names. A
        // nice-to-have on top of the global gate; ignored targets that raise nothing.
        let target_rank = |name: &str| -> u8 {
            spec.targets
                .iter()
                .filter(|(n, _)| n == name)
                .map(|(_, r)| *r)
                .max()
                .unwrap_or(0)
        };
        let any_target = spec.targets.iter().any(|(_, r)| *r > 0);

        // Nothing enabled anywhere → wire nothing (the zero-overhead disabled path).
        if global_rank == 0 && !any_target {
            return ((0..total).map(|_| None).collect(), None);
        }

        let shared = Arc::new(filter);
        let mut logs: Vec<Option<Log>> = Vec::with_capacity(total);
        let mut drains: Vec<LogDrain> = Vec::new();
        for i in 0..total {
            let name = self.name_of(ElementId(i as u32));
            let tr = target_rank(name);
            // Elements not matched by a target log at the global level; matched ones get
            // their own filter raised to `max(global, target)`. Both `>0` ⇒ a channel.
            let elem_rank = global_rank.max(tr);
            if elem_rank == 0 {
                logs.push(None);
                continue;
            }
            let elem_filter = if tr > global_rank {
                Arc::new(LevelFilter::with_level(
                    Level::from_rank(elem_rank).unwrap_or(Level::Error),
                ))
            } else {
                Arc::clone(&shared)
            };
            let (sink, drain) = log_channel(self.log_queue_cap);
            let mut log = Log::new(sink, elem_filter);
            log.set_element(ElementId(i as u32), name);
            logs.push(Some(log));
            drains.push(drain);
        }

        let drain = LogDrainThread::spawn(drains);
        (logs, Some(drain))
    }

    /// The report from the most recent [`run`](Self::run).
    pub fn last_report(&self) -> Option<RunReport> {
        self.last_report
    }

    /// Order the elements into a single source→sink chain (milestone topology).
    fn topo_order(&self) -> Result<Vec<ElementId>, Error> {
        let n = self.elements.len();
        if n == 0 {
            return Err(Error::Todo("empty pipeline"));
        }
        // Kahn's over the DAG. Both fan-out (one element → several downstreams via
        // distinct src pads) and fan-in (an active aggregator with several sink pads) are
        // fine; a passive element with two inputs is caught in compute_groups.
        let mut indeg = vec![0usize; n];
        let mut adj: Vec<Vec<ElementId>> = vec![Vec::new(); n];
        for e in &self.edges {
            indeg[e.sink.0 as usize] += 1;
            adj[e.src.0 as usize].push(e.sink);
        }
        let mut ready: Vec<ElementId> = indeg
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| ElementId(i as u32))
            .collect();
        // One or more sources are fine (a muxer graph fans several streams into one
        // aggregator); zero means a cycle with no entry point.
        if ready.is_empty() {
            return Err(Error::Todo("no source (cycle?)"));
        }
        let mut order = Vec::with_capacity(n);
        while let Some(u) = ready.pop() {
            order.push(u);
            for &v in &adj[u.0 as usize] {
                let d = &mut indeg[v.0 as usize];
                *d -= 1;
                if *d == 0 {
                    ready.push(v);
                }
            }
        }
        if order.len() != n {
            return Err(Error::Todo("disconnected graph or cycle"));
        }
        Ok(order)
    }

    /// Assign elements to thread groups over the (possibly branching) DAG: the source and
    /// every active element start a group; a passive element inlines into its single
    /// upstream's group, extending that group's chain (spec: Scheduling — passive chains
    /// run inline). Branching (fan-out) therefore lives *between* groups, wired as rings.
    /// Passive fan-out (a passive with two downstreams, or two passives sharing one
    /// upstream) is rejected — make the branch point active.
    fn compute_groups(&self, order: &[ElementId]) -> Result<Vec<Vec<ElementId>>, Error> {
        let n = self.elements.len();
        // Incoming edges per element. A passive element must have exactly one (it inlines
        // into that upstream's group); an active element may have several — a fan-in
        // aggregator (a muxer) is its own group head reading N upstream rings.
        let mut incoming: Vec<Vec<ElementId>> = vec![Vec::new(); n];
        for e in &self.edges {
            incoming[e.sink.0 as usize].push(e.src);
        }
        let mut group_of: Vec<Option<usize>> = vec![None; n];
        let mut groups: Vec<Vec<ElementId>> = Vec::new();
        for &id in order {
            let sched = self.elements[id.0 as usize]
                .as_ref()
                .ok_or(Error::Todo("element already consumed"))?
                .desc()
                .sched;
            let is_source = incoming[id.0 as usize].is_empty();
            if is_source || matches!(sched, SchedHint::Active) {
                group_of[id.0 as usize] = Some(groups.len());
                groups.push(vec![id]);
            } else {
                // Passive: inline into its single upstream's group, but only if that
                // upstream is the group's current tail — otherwise this is a passive
                // fan-out (unsupported), and >1 input is a passive aggregator (unsupported).
                if incoming[id.0 as usize].len() != 1 {
                    return Err(Error::Todo(
                        "passive element with multiple inputs (an aggregator must be active)",
                    ));
                }
                let up = incoming[id.0 as usize][0];
                let g = group_of[up.0 as usize]
                    .ok_or(Error::Todo("passive scheduled before its upstream (internal)"))?;
                if *groups[g].last().unwrap() != up {
                    return Err(Error::Todo(
                        "passive fan-out not supported (make the branch point active)",
                    ));
                }
                groups[g].push(id);
                group_of[id.0 as usize] = Some(g);
            }
        }
        Ok(groups)
    }

    // --- State and clock: pause is a clock op, not a state (TODO: later steps) ---

    pub fn add_subgraph(&mut self, _tpl: &Template) -> Result<GroupId, Error> {
        todo!("spec: No bins — templates")
    }

    pub fn link_filtered(
        &mut self,
        _src: (ElementId, &str),
        _sink: (ElementId, &str),
        _filter: &[FieldConstraint],
    ) -> Result<LinkId, Error> {
        todo!("spec: Formats — link constraints replace capsfilter")
    }

    pub fn relink(&mut self, _link: LinkId, _new_sink: (ElementId, &str)) -> Result<(), Error> {
        todo!("spec: Dynamic pipelines")
    }

    pub fn remove(&mut self, _el: ElementId) -> Result<(), Error> {
        todo!("spec: Runtime configuration")
    }

    pub fn set_state(&mut self, _s: State) -> Result<(), Error> {
        todo!("spec: Events, queries, and the bus")
    }

    pub fn pause(&mut self) {
        todo!("spec: Clocking — pause is a clock freeze")
    }

    pub fn seek(&mut self, _to: Timestamp) -> Result<(), Error> {
        todo!("spec: Events — seeking")
    }

    pub fn step(&mut self) -> Result<(), Error> {
        todo!("spec: Scheduling — step() mode")
    }

    /// Set an element property (spec: Dynamic element properties). The value is
    /// validated *here*, against the constraint the element declared in
    /// `desc().props` — an unknown name or out-of-range value fails loudly at the
    /// call site, reusing the format algebra's closed check. The element observes
    /// the new value in `start()` (next run) or, while streaming, at its next batch
    /// boundary via [`Event::PropChanged`] / [`Ctx::prop`](crate::ctx::Ctx::prop).
    /// Since `run()` blocks this thread, mid-run sets go through a
    /// [`prop_handle`](Self::prop_handle) instead (live properties only).
    pub fn set(&mut self, el: ElementId, prop: &str, v: Value) -> Result<(), Error> {
        let desc = self.descs.get(el.0 as usize).ok_or(Error::Todo("set: unknown element"))?;
        validate_and_set(el, desc, &self.props[el.0 as usize], prop, v, false)
    }

    /// Set a **string-valued** property (spec: Plugins — file paths ride `Value::Id`).
    /// `Value` has no string variant by design (it is POD, memcmp-comparable), so a
    /// string property is interned through this pipeline's value interner *here*, at set
    /// time, and stored as the resulting [`Value::Id`]. Because interning happens before
    /// `run()`, the string is in the value vocabulary snapshot the run reads, so the
    /// element resolves it in `start()` via
    /// [`ctx.value_name(id)`](crate::ctx::Ctx::value_name). Delegates to the validated
    /// [`set`](Self::set), so the property's declared constraint still gates it (use
    /// `Constraint::Any` for free-form paths).
    pub fn set_str(&mut self, el: ElementId, prop: &str, s: &str) -> Result<(), Error> {
        let id = ValueId(self.values.intern(s));
        self.set(el, prop, Value::Id(id))
    }

    /// The current value of an element's property, if set (spec: Dynamic element
    /// properties). Reads the same mailbox the element reads; for tests and tools
    /// inspecting a parsed pipeline before `run()`. A string set via
    /// [`set_str`](Self::set_str) reads back as a [`Value::Id`] — resolve it with
    /// [`value_name`](Self::value_name).
    pub fn prop_value(&self, el: ElementId, prop: &str) -> Option<Value> {
        let desc = self.descs.get(el.0 as usize)?;
        let idx = desc.props.iter().position(|p| p.name == prop)?;
        self.props.get(el.0 as usize)?.get(idx)
    }

    /// Resolve an interned categorical/string value id back to its name (spec: Formats —
    /// the value interner). Reverses a [`set_str`](Self::set_str) for tools and tests.
    pub fn value_name(&self, id: ValueId) -> Option<&str> {
        self.values.resolve(id.0)
    }

    /// A cloneable handle to set **live** properties from another thread while
    /// `run()` is blocking (spec: Dynamic element properties). Snapshots the
    /// elements present now — take it after the topology is built.
    pub fn prop_handle(&self) -> PropHandle {
        PropHandle::new(
            self.descs
                .iter()
                .zip(&self.props)
                .map(|(d, t)| (*d, Arc::clone(t)))
                .collect(),
        )
    }

    /// A cloneable stats tap: read any element's counters and the pipeline's
    /// running time from any thread, at zero streaming-path cost (spec: Taps —
    /// a tap is a pull of data the pipeline already keeps). Take it after the
    /// topology is built and the clock (if custom) is installed.
    pub fn tap_handle(&self) -> TapHandle {
        TapHandle::new(
            self.descs
                .iter()
                .zip(&self.counters)
                .map(|(d, c)| (d.name, Arc::clone(c)))
                .collect(),
            Arc::clone(&self.clock),
            Arc::clone(&self.base_shared),
        )
    }

    pub fn set_latency_budget(&mut self, _budget: Timestamp) {
        todo!("spec: Latency")
    }

    // --- Observability ---

    pub fn bus(&self) -> &Bus {
        &self.bus
    }

    /// Render the topology as Graphviz `dot`, clustering elements by thread group
    /// (spec: Debuggability — graph dump). Call before `run()` (elements are moved
    /// into their threads during a run).
    pub fn dump_dot(&self) -> String {
        let mut s = String::from("digraph streamcraft {\n  rankdir=LR;\n  node [shape=box];\n");
        for (i, e) in self.elements.iter().enumerate() {
            let name = e.as_ref().map_or("?", |el| el.desc().name);
            s.push_str(&format!("  e{i} [label=\"{name}\"];\n"));
        }
        // Group clusters — each group is one thread (spec: Scheduling).
        if let Ok(order) = self.topo_order() {
            if let Ok(groups) = self.compute_groups(&order) {
                for (gi, ids) in groups.iter().enumerate() {
                    s.push_str(&format!(
                        "  subgraph cluster_{gi} {{\n    label=\"group {gi} (thread)\";\n    style=dashed;\n"
                    ));
                    for id in ids {
                        s.push_str(&format!("    e{};\n", id.0));
                    }
                    s.push_str("  }\n");
                }
            }
        }
        for e in &self.edges {
            // Label the edge with the negotiated family (spec: Debuggability — the
            // dump shows, per edge, what was chosen).
            let fam = self.formats.resolve(e.format.family.0).unwrap_or("?");
            s.push_str(&format!(
                "  e{} -> e{} [label=\"{fam}\"];\n",
                e.src.0, e.sink.0
            ));
        }
        s.push_str("}\n");
        s
    }

    /// The computed per-path latency breakdown (spec: Latency — a graph traversal
    /// over data the pipeline already holds; no query protocol). Callable in any
    /// state; the same numbers a sink's `wait_until` compensates while playing.
    pub fn latency_report(&self) -> LatencyReport {
        let (in_lat, pred) = self.compute_in_latency();
        let n = self.descs.len();
        let mut has_out = vec![false; n];
        for e in &self.edges {
            has_out[e.src.0 as usize] = true;
        }
        let mut paths = Vec::new();
        for i in 0..n {
            if has_out[i] {
                continue; // not a sink
            }
            // Reconstruct the worst path, sink back to source.
            let mut chain = Vec::new();
            let mut cur = Some(i);
            let mut is_live = false;
            while let Some(c) = cur {
                let lat = &self.descs[c].latency;
                is_live |= lat.is_live;
                chain.push((ElementId(c as u32), lat.min));
                cur = pred[c];
            }
            chain.reverse();
            paths.push(crate::counters::PathLatency {
                sink: ElementId(i as u32),
                total: in_lat[i],
                per_element: chain,
                is_live,
            });
        }
        paths.sort_by(|a, b| b.total.cmp(&a.total));
        LatencyReport { paths }
    }

    /// The worst upstream declared-latency sum reaching each element (`in_lat`) and
    /// the predecessor on that worst path (`pred`), by DP over the topological order
    /// — the element's own latency is *not* included in its `in_lat` (a sink waits
    /// out its upstream, not itself; spec: Latency). Queue-residency terms join once
    /// formats carry rates.
    fn compute_in_latency(&self) -> (Vec<Timestamp>, Vec<Option<usize>>) {
        let n = self.descs.len();
        let mut in_lat = vec![Timestamp::ZERO; n];
        let mut pred: Vec<Option<usize>> = vec![None; n];
        if let Ok(order) = self.topo_order() {
            for id in order {
                let u = id.0 as usize;
                let via = in_lat[u].saturating_add(self.descs[u].latency.min);
                for e in self.edges.iter().filter(|e| e.src.0 as usize == u) {
                    let v = e.sink.0 as usize;
                    // First predecessor always wins the slot (in_lat[v] is still its
                    // ZERO init, and via >= ZERO); later ones only on a longer path.
                    if pred[v].is_none() || via > in_lat[v] {
                        in_lat[v] = via;
                        pred[v] = Some(u);
                    }
                }
            }
        }
        (in_lat, pred)
    }

    pub fn counters(&self, el: ElementId) -> CounterSnapshot {
        self.counters
            .get(el.0 as usize)
            .map(|c| c.snapshot())
            .unwrap_or_default()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// The single low-priority drain thread for a run (spec: Debuggability — Logging: a
/// low-priority [`LogDrain`] consumes; observation never perturbs what it observes).
/// It owns *all* elements' drains, polls them round-robin off the hot path, formats
/// each record to stderr, and parks briefly when every ring is momentarily empty. It
/// exits once every drain is closed (its `LogSink` dropped with the owning `Ctx`) and
/// emptied, so joining it after the group threads is a clean, bounded shutdown.
struct LogDrainThread {
    handle: JoinHandle<()>,
}

impl LogDrainThread {
    fn spawn(drains: Vec<LogDrain>) -> Self {
        let handle = std::thread::Builder::new()
            .name("sc-log-drain".into())
            .spawn(move || Self::run(drains))
            .expect("spawn log drain");
        Self { handle }
    }

    fn run(drains: Vec<LogDrain>) {
        use std::io::Write;
        let styled = crate::log::stderr_colors_enabled();
        let stderr = std::io::stderr();
        // Idle backoff: park a beat when a whole sweep found nothing, so the thread
        // costs no CPU while streaming is quiet, yet stays responsive under load.
        let idle_park = std::time::Duration::from_millis(2);
        loop {
            let mut progressed = false;
            let mut all_done = true;
            {
                let mut lock = stderr.lock();
                for d in &drains {
                    let mut drained_this = false;
                    while let Some(rec) = d.try_next() {
                        let _ = crate::log::format_record_styled(&mut lock, &rec, styled);
                        progressed = true;
                        drained_this = true;
                    }
                    // A channel is finished only when closed *and* fully drained; the
                    // producer may have pushed a last record just before dropping.
                    if !d.is_closed() || drained_this {
                        all_done = false;
                    }
                }
                let _ = lock.flush();
            }
            if all_done {
                break;
            }
            if !progressed {
                std::thread::sleep(idle_park);
            }
        }
    }

    fn join(self) -> std::thread::Result<()> {
        self.handle.join()
    }
}

/// Deliver a batch's in-band events to the group's head element (spec: Events travel with
/// buffers). A `FormatChange` first updates the head's negotiated format on its sink pad —
/// so the element sees the new format via `ctx.negotiated()` inside `event()` and its next
/// `process()` — then the event is handed to `event()` (spec: Formats — dynamic caps).
/// Runs once per received batch, off the per-buffer path.
fn deliver_events(
    elem: &mut dyn Element,
    ctx: &mut Ctx,
    sink_pad: Option<PadId>,
    events: Vec<Event>,
    vocabulary: &Vocabulary,
) -> Result<(), Error> {
    for ev in events {
        if let Event::FormatChange(f) = &ev {
            if let Some(pad) = sink_pad {
                // Re-validate the announced format against this pad's declared offers
                // before installing it (spec: Formats — dynamic caps: an incompatible
                // runtime format is a loud negotiation failure, never a silent install).
                // Only offers in the *announced family* gate this: if the peer speaks
                // that family but none of its offers admits the concrete values, that is
                // fatal — the target case, a decoder announcing (say) a rate the sink
                // cannot take. If the peer offers no such family at all (e.g. an
                // `audio/raw` refinement riding a `bytes` bridge to a byte sink), there
                // is nothing for this peer to re-fixate: install it and let a caps-reading
                // peer act while a caps-ignoring one ignores it. Underspecified fields are
                // never a conflict — the peer would fixate them, as at link time.
                let offers = elem.desc().pads[pad.0 as usize].offers;
                let family_offered = offers
                    .iter()
                    .any(|o| vocabulary.family_id(o.family) == Some(f.family));
                if family_offered && !vocabulary.offers_admit(offers, f) {
                    let element = ctx.element();
                    let err = Error::Element {
                        element,
                        message: format!(
                            "runtime format announcement not accepted by '{}' pad '{}' \
                             — dynamic-caps re-validation failed",
                            elem.desc().name,
                            elem.desc().pads[pad.0 as usize].name,
                        ),
                    };
                    ctx.post(BusMessage::Error { element, error: err.clone() });
                    return Err(err);
                }
                ctx.set_negotiated_one(pad, f.clone());
            }
        }
        elem.event(ctx, &ev)?;
    }
    Ok(())
}

/// One thread group's run loop (spec: Scheduling — a group runs on one thread). The
/// group is `[active_head, passive_tail...]`; the head pulls from the upstream ring
/// (unless it is the source) and the tail runs inline; the last element's output goes
/// to the downstream ring. Backpressure is the blocking ring push; the group parks on
/// the upstream ring when idle. Its reactor serves the group's IO.
#[allow(clippy::too_many_arguments)]
fn run_group(
    mut elements: Vec<Box<dyn Element>>,
    ids: Vec<ElementId>,
    formats: Vec<Vec<Option<FixedFormat>>>,
    logs: Vec<Option<Log>>,
    upstream: Vec<(PadId, Consumer<Batch>, Producer<Batch>)>,
    downstream: Vec<(PadId, Producer<Batch>, Consumer<Batch>)>,
    factory: ReactorFactory,
    pools: Vec<Pool>,
    bus: BusSender,
    credits: u32,
    counters: Vec<Arc<ElementCounters>>,
    props: Vec<Arc<PropTable>>,
    stop: Arc<AtomicBool>,
    seek: Arc<SeekState>,
    vocabulary: Arc<Vocabulary>,
    clock: Arc<dyn Clock>,
    base: Arc<AtomicU64>,
    pad_infos: Vec<(usize, Vec<usize>)>,
    path_latencies: Vec<Timestamp>,
    tracing: Arc<AtomicBool>,
    pause: Arc<PauseShared>,
) -> Result<(), Error> {
    let m = elements.len();
    let is_source = upstream.is_empty();
    // A fan-in (aggregator) head reads several upstream rings; a single-input head reads
    // one. The single-input head's sink pad (the one the ring feeds) is where a
    // FormatChange re-fixates; a fan-in head re-fixates per pad in the feed loop instead.
    let fan_in = upstream.len() > 1;
    let head_sink_pad: Option<PadId> = if is_source || fan_in {
        None
    } else {
        Some(upstream[0].0)
    };
    let mut reactor: Box<dyn Reactor> =
        factory().map_err(|e| Error::Resource(format!("reactor init: {e}")))?;
    let mut ctxs: Vec<Ctx> = ids
        .iter()
        .zip(&pools)
        .map(|(id, pool)| Ctx::new(pool.clone(), bus.clone(), *id, BYTES, credits))
        .collect();
    // Size each Ctx's per-pad output table and record its primary (first) src pad, so
    // `ctx.out(pad)` routes per pad and the scheduler can pull each src pad's output for
    // its downstream ring (spec: dynamic pads / branching).
    for (ctx, (npads, src_pads)) in ctxs.iter_mut().zip(pad_infos.iter()) {
        ctx.configure_pads(*npads, src_pads);
    }
    // Install each element's link-time-negotiated formats before `start()`, so an
    // element can read `ctx.negotiated(pad)` in `start()` as well as `process()`.
    for (ctx, per_pad) in ctxs.iter_mut().zip(formats) {
        ctx.set_negotiated(per_pad);
    }
    // Install each element's log emitter (present only when logging is enabled), so
    // `log!(ctx, ...)` works from `start()` onward. Each `Log` owns its own `LogSink`,
    // written solely by this group's thread — no producer is shared, so `Ctx` stays
    // `Send`.
    for (ctx, log) in ctxs.iter_mut().zip(logs) {
        if let Some(log) = log {
            ctx.set_log(log);
        }
    }
    // Install the frozen vocabulary so elements can resolve their negotiated format by name
    // and the produce hook can build announced formats (spec: Formats — dynamic caps).
    for ctx in ctxs.iter_mut() {
        ctx.set_vocabulary(Arc::clone(&vocabulary));
    }
    // Install the pipeline clock + the *shared* running-time base (re-based by
    // pause/resume) so elements read running time via `ctx.now()` and pace on it via
    // `ctx.wait_until()` (spec: Clocking), each element's computed upstream path
    // latency (spec: Latency — enforced at sinks), and the pause transport so an
    // expired wait blocks instead of rendering while paused.
    for (ctx, lat) in ctxs.iter_mut().zip(path_latencies) {
        ctx.set_clock(Arc::clone(&clock), Arc::clone(&base));
        ctx.set_path_latency(lat);
        ctx.set_pause(Arc::clone(&pause));
    }
    // Install the shared seek request so a source resolves its byte target and a sink its
    // frame target when the scheduler delivers `FlushStart` (spec: flush/seek).
    for ctx in ctxs.iter_mut() {
        ctx.set_seek(Arc::clone(&seek));
    }
    // Install the tracing gate + this element's histograms so `ctx.wait_until` can record
    // sink wait overshoot while tracing is on (spec: Debuggability).
    for (ctx, c) in ctxs.iter_mut().zip(&counters) {
        ctx.set_trace(Arc::clone(c), Arc::clone(&tracing));
    }
    // Install each element's property mailbox — only where props are declared, so the
    // no-props majority keeps the zero-cost `None` path (spec: Dynamic element properties).
    for i in 0..m {
        let descs = elements[i].desc().props;
        if !descs.is_empty() {
            ctxs[i].set_props(descs, Arc::clone(&props[i]));
        }
    }

    // Each member's (single) sink pad, for intra-group in-band event delivery — a
    // FormatChange from a co-grouped upstream installs the announced format there
    // (spec: Formats — dynamic caps; events are delivered before the buffers they
    // precede even when the hop is an inline function call, not a ring).
    let member_sink_pads: Vec<Option<PadId>> = elements
        .iter()
        .map(|e| {
            e.desc()
                .pads
                .iter()
                .position(|p| p.direction == Direction::Sink)
                .map(|i| PadId(i as u32))
        })
        .collect();

    // Start each element; hand any file it registered to this group's reactor.
    for i in 0..m {
        elements[i].start(&mut ctxs[i])?;
        if let Some(f) = ctxs[i].take_registration() {
            reactor.set_file(ids[i], f);
        }
    }

    let mut source_eos = false;
    let mut upstream_closed = false;
    let mut eos_next = 0usize;
    // The seek generation this group has flushed up to. A `SeekHandle::seek` bumps the
    // shared generation; observing the change here drives the out-of-band flush.
    let mut seek_gen = seek.gen.load(Ordering::Acquire);

    let result: Result<(), Error> = 'group: loop {
        // Cooperative cancellation: break, then stop + drop downstream, which closes
        // the ring and cascades the stop to the next group.
        if stop.load(Ordering::Acquire) {
            break Ok(());
        }

        // Flush/seek (spec: flush/seek). A bumped generation, observed out-of-band here
        // rather than carried in-band behind the queued data, means: reset each element's
        // stream state (a source re-seeks, a decoder re-syncs, a sink drops its device
        // buffer and re-bases its position — all via `FlushStart`) and discard everything
        // staged inside this group. Data already handed to the downstream ring is dropped
        // on the consuming side by the generation stamp below, so this never races the
        // producer that is concurrently pushing fresh post-seek data.
        let cur_gen = seek.gen.load(Ordering::Acquire);
        if cur_gen != seek_gen {
            for i in 0..m {
                if let Err(e) = elements[i].event(&mut ctxs[i], &Event::FlushStart) {
                    break 'group Err(e);
                }
            }
            for c in ctxs.iter_mut() {
                c.discard_buffers();
            }
            // A seek revives a stream that may already have ended: forget prior EOS.
            source_eos = false;
            upstream_closed = false;
            eos_next = 0;
            seek_gen = cur_gen;
        }

        let mut progressed = false;

        // Pause gate (spec: Clocking — pause is a clock op): notify every element
        // (a device sink holds its hardware on `Paused`), park until resume or stop,
        // notify again. One relaxed load while playing.
        if pause.maybe_paused() && pause.is_paused() {
            for i in 0..m {
                if let Err(e) = elements[i].event(&mut ctxs[i], &Event::Paused) {
                    break 'group Err(e);
                }
            }
            let _ = pause.block_while_paused(); // false = stop; the stop check below exits
            for i in 0..m {
                if let Err(e) = elements[i].event(&mut ctxs[i], &Event::Resumed) {
                    break 'group Err(e);
                }
            }
        }

        // A. Feed the head from upstream.
        if fan_in {
            // Aggregator: poll every upstream ring non-blockingly, feeding each into its
            // own sink pad and marking a pad closed when its ring closes (so a muxer can
            // stop waiting for it). An efficient multi-ring park is a follow-up; the
            // yield-on-no-progress net at the loop's end covers the idle case for now.
            let mut all_closed = true;
            for (pad, cons, shell_tx) in &upstream {
                while let Some(mut batch) = cons.try_pop() {
                    if batch.seek_gen < seek_gen {
                        continue; // stale pre-seek data (buffers recycle on drop)
                    }
                    counters[0].record_in(batch.len() as u64, batch.total_bytes());
                    if let Some(t) = batch.pushed_at.take() {
                        counters[0].queue_ns.record(t.elapsed().as_nanos() as u64);
                    }
                    let events = batch.take_events();
                    ctxs[0].append_input_on(*pad, &mut batch);
                    // Return the drained shell for the producer's take_output to reuse
                    // (columns keep their capacity; drop is the harmless fallback).
                    batch.clear();
                    let _ = shell_tx.try_push(batch);
                    if let Err(e) =
                        deliver_events(&mut *elements[0], &mut ctxs[0], Some(*pad), events, &vocabulary)
                    {
                        break 'group Err(e);
                    }
                    progressed = true;
                }
                if cons.is_closed() {
                    ctxs[0].set_pad_closed(*pad);
                } else {
                    all_closed = false;
                }
            }
            if all_closed {
                upstream_closed = true;
            }
        } else if let Some((_pad, up, shell_tx)) = upstream.first() {
            let closed = up.is_closed();
            let head_idle =
                ctxs[0].input_is_empty() && ctxs[0].inbox_empty() && reactor.is_idle();
            if !closed && head_idle {
                // Nothing to do but wait for input — block (no busy spin).
                match up.pop() {
                    Some(mut batch) if batch.seek_gen >= seek_gen => {
                        counters[0].record_in(batch.len() as u64, batch.total_bytes());
                        if let Some(t) = batch.pushed_at.take() {
                            counters[0].queue_ns.record(t.elapsed().as_nanos() as u64);
                        }
                        let events = batch.take_events();
                        ctxs[0].input_append(&mut batch);
                        batch.clear();
                        let _ = shell_tx.try_push(batch);
                        if let Err(e) = deliver_events(
                            &mut *elements[0],
                            &mut ctxs[0],
                            head_sink_pad,
                            events,
                            &vocabulary,
                        ) {
                            break 'group Err(e);
                        }
                        progressed = true;
                    }
                    Some(_) => {} // stale pre-seek data: drop (buffers recycle)
                    None => upstream_closed = true,
                }
            } else {
                // Backlog cap (the ring-fed twin of the inline gate): stop popping while
                // the head still holds INLINE_INPUT_CAP unconsumed buffers, so a slow
                // head (a video decoder) is bounded by its ring + this cap instead of
                // accumulating its whole upstream's output — every buffered buffer pins
                // a pool slot, and unbounded backlogs are how a movie eats gigabytes.
                while ctxs[0].input_len() < INLINE_INPUT_CAP {
                    let Some(mut batch) = up.try_pop() else { break };
                    if batch.seek_gen < seek_gen {
                        continue; // stale pre-seek data (buffers recycle on drop)
                    }
                    counters[0].record_in(batch.len() as u64, batch.total_bytes());
                    if let Some(t) = batch.pushed_at.take() {
                        counters[0].queue_ns.record(t.elapsed().as_nanos() as u64);
                    }
                    let events = batch.take_events();
                    ctxs[0].input_append(&mut batch);
                    batch.clear();
                    let _ = shell_tx.try_push(batch);
                    if let Err(e) = deliver_events(
                        &mut *elements[0],
                        &mut ctxs[0],
                        head_sink_pad,
                        events,
                        &vocabulary,
                    ) {
                        break 'group Err(e);
                    }
                    progressed = true;
                }
                if closed && up.is_empty() {
                    upstream_closed = true;
                }
            }
        }

        // B. Run the group's elements inline (head active, passive tail).
        let mut fatal = None;
        for i in 0..m {
            // Inline backpressure (spec: Scheduling — real queues exist only at group
            // boundaries; *inside* a group the scheduler paces producers): don't run
            // an element while its co-grouped successor still holds a backlog of
            // unconsumed input. Without this, a fast IO source inlined with a
            // deliberately-latency-bounded consumer (filesrc ! flacdec …) free-runs
            // until the shared pool is exhausted by its own queued output — at which
            // point the consumer can no longer allocate *its* output, never consumes,
            // never frees a slot, and the group livelocks. The tail ring stays the
            // inter-group bound; this is its inline analogue. Consumers below the
            // gated element still run, so backlogs always drain.
            if i + 1 < m && ctxs[i + 1].input_len() >= INLINE_INPUT_CAP {
                continue;
            }
            // Live property sets land here, at the batch boundary — a set issued
            // while batch N was in flight takes effect no earlier than N+1 (spec:
            // Dynamic element properties). Idle cost: one relaxed load.
            let mut dirty = ctxs[i].poll_prop_changes();
            while dirty != 0 {
                let bit = dirty.trailing_zeros() as usize;
                dirty &= dirty - 1;
                if let Some((name, value)) = ctxs[i].prop_entry(bit) {
                    if let Err(e) =
                        elements[i].event(&mut ctxs[i], &Event::PropChanged { name, value })
                    {
                        fatal = Some(e);
                        break;
                    }
                }
            }
            if fatal.is_some() {
                break;
            }
            // Latency tracing (spec: Debuggability): time `process()` only while the
            // flag is on — the untraced hot path pays one relaxed load per call.
            let trace_t0 =
                if tracing.load(Ordering::Relaxed) { Some(std::time::Instant::now()) } else { None };
            let flow = if i == 0 && (is_source || fan_in) {
                // A source has no input; a fan-in head reads its pads via
                // `ctx.take_input_on(pad)`, so both get an empty `Inputs` here.
                elements[0].process(&mut ctxs[0], Inputs::empty())
            } else {
                let mut input = ctxs[i].take_input();
                // In-band events from a co-grouped upstream ride the inline-appended
                // batch (a ring-fed head's events were already delivered at feed
                // time and never reach here). Deliver them now, before the buffers
                // they precede — this is what lets a passive consumer see a passive
                // producer's FormatChange (spec: Formats — dynamic caps).
                let events = input.take_events();
                if !events.is_empty() {
                    if let Err(e) = deliver_events(
                        &mut *elements[i],
                        &mut ctxs[i],
                        member_sink_pads[i],
                        events,
                        &vocabulary,
                    ) {
                        ctxs[i].set_input(input);
                        fatal = Some(e);
                        break;
                    }
                }
                let f = elements[i].process(&mut ctxs[i], Inputs::owned(&mut input));
                ctxs[i].set_input(input);
                f
            };
            if let Some(t0) = trace_t0 {
                counters[i].process_ns.record(t0.elapsed().as_nanos() as u64);
            }
            match flow {
                Ok(fl) => {
                    if i == 0 && is_source && matches!(fl, Flow::Eos) {
                        source_eos = true;
                    }
                }
                Err(e) => {
                    fatal = Some(e);
                    break;
                }
            }
            // Scratch lives only for the duration of `process()`; reclaim it now the call
            // has returned (its regions are dropped) so the next call reuses the memory.
            ctxs[i].reset_scratch();
            // Dynamic caps: if the element announced a runtime output format, attach a
            // FormatChange to its output batch so the peer re-fixates (spec: Formats).
            for ann in ctxs[i].take_announcements() {
                let fixed = match ann.payload {
                    crate::ctx::AnnouncePayload::Named { family, fields } => {
                        vocabulary.build_fixed(family, &fields)
                    }
                    // Already resolved — the forwarding path (a queue re-emitting a
                    // FormatChange it received; spec: Formats — dynamic caps).
                    crate::ctx::AnnouncePayload::Fixed(f) => Some(f),
                };
                if let Some(f) = fixed {
                    // Ride the FormatChange on the announced src pad's batch so it travels
                    // to that pad's downstream (correct when the element branches — a
                    // multi-track demuxer announces several pads in one pass).
                    ctxs[i].output_on(ann.pad).push_event(Event::FormatChange(f));
                }
            }
            let (nbuf, nbytes) = ctxs[i].total_output();
            // Only touch the counters when something was produced: an idle pass pays
            // no atomics, and `batches_*` counts real batches, not scheduler passes.
            if nbuf > 0 {
                counters[i].record_out(nbuf, nbytes);
            }
            reactor.submit(ctxs[i].take_submissions());
            if i + 1 < m {
                if nbuf > 0 {
                    counters[i + 1].record_in(nbuf, nbytes);
                }
                let (left, right) = ctxs.split_at_mut(i + 1);
                right[0].input_append(left[i].output_mut());
                // The inline hand-off moves only the primary src pad. Anything left on
                // another pad has nowhere to go — an *unlinked* pad on a non-tail member
                // (a demuxer's unwatched track, inlined mid-group). Same policy as the
                // tail's unrouted pads below: drop it, count it, recycle its pool slots
                // — accumulating here exhausts the element's pool and stalls the group
                // (spec: robustness).
                let (unrouted, _) = left[i].total_output();
                if unrouted > 0 {
                    counters[i].record_drops(unrouted);
                    left[i].clear_outputs();
                }
            }
        }
        if let Some(e) = fatal {
            break Err(e);
        }

        // C. Route the tail element's per-pad output to each downstream ring (blocking
        // backpressure). A branching tail (a demuxer) has one ring per src pad; a linear
        // tail has one; a sink has none. First reclaim any shells the consumer returned,
        // so `take_output` below reuses their column capacity (zero-alloc steady state).
        for (_, _, shell_rx) in &downstream {
            while let Some(shell) = shell_rx.try_pop() {
                ctxs[m - 1].recycle_shell(shell);
            }
        }
        for (src_pad, down, _) in &downstream {
            let mut out = ctxs[m - 1].take_output(*src_pad);
            if !out.is_inert() {
                // Stamp the generation so the consuming group can drop it if a seek
                // supersedes it before it is processed (spec: flush/seek).
                out.seek_gen = seek_gen;
                // Latency tracing: stamp the push instant so the consumer's pop measures
                // ring residency (spec: Debuggability).
                if tracing.load(Ordering::Relaxed) {
                    out.pushed_at = Some(std::time::Instant::now());
                }
                progressed = true;
                if down.push(out).is_err() {
                    break 'group Ok(()); // this downstream is gone
                }
                // Queue-fill high-water on the producing element (spec: Taps —
                // backpressure shows as the ring sitting at capacity).
                counters[m - 1].record_queue_fill(down.len() as u32);
            }
        }
        // Whatever is still in the output table has no ring: an *unlinked* src pad
        // (a demuxer track nobody connected). Policy: discard it here, every pass,
        // and count it as drops (spec: robustness) — an unlinked track must never
        // accumulate memory, stall the graph, or require a dummy sink.
        let (unrouted, _) = ctxs[m - 1].total_output();
        if unrouted > 0 {
            counters[m - 1].record_drops(unrouted);
        }
        ctxs[m - 1].clear_outputs();

        // D. Drive this group's reactor and route completions back to their elements.
        let completions = reactor.run_once();
        if !completions.is_empty() {
            progressed = true;
        }
        for (elem, c) in completions {
            if let Some(idx) = ids.iter().position(|e| *e == elem) {
                ctxs[idx].deliver_completion(c);
            }
        }

        // E. Termination: the head is finished and nothing is buffered or in flight.
        let head_done = if is_source { source_eos } else { upstream_closed };
        let quiescent = reactor.is_idle()
            && ctxs.iter().all(|c| c.inbox_empty() && c.inputs_empty());
        if head_done && quiescent {
            if eos_next < m {
                // Deliver EOS to one element per pass, in chain order, then loop so its
                // flushed output propagates to and is processed by the next element before
                // that element gets EOS (spec: Events — EOS reaches elements in order). A
                // sink drains its device buffer; a muxer flushes its final page, which the
                // downstream depacketiser then sees as input before its own EOS.
                if let Err(e) = elements[eos_next].event(&mut ctxs[eos_next], &Event::Eos) {
                    break 'group Err(e);
                }
                eos_next += 1;
                continue;
            }
            // Stamped on the group's head element (spec: Debuggability). Gated out at
            // zero cost unless Trace is enabled, so it never touches the hot path.
            crate::log!(&ctxs[0], Level::Trace, "group_done", is_source = is_source);
            break Ok(());
        }

        // Safety net against a pathological busy-spin (the pool is sized so the ring
        // is the real backpressure, so this should be rare): yield if a whole pass
        // made no progress.
        if !progressed {
            std::thread::yield_now();
        }
    };

    // Stop elements (reverse). Dropping `downstream` here closes the ring, signalling
    // EOS to the next group.
    for i in (0..m).rev() {
        elements[i].stop(&mut ctxs[i]);
    }
    drop(downstream);
    result
}
