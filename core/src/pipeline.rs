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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::batch::{Batch, Inputs};
use crate::bus::{Bus, BusMessage, BusSender, State};
use crate::counters::{CounterSnapshot, ElementCounters, LatencyReport};
use crate::ctx::Ctx;
use crate::element::{Direction, Element, Flow, SchedHint, Template};
use crate::error::Error;
use crate::format::{negotiate, FieldConstraint, FixedFormat, Value};
use crate::id::{ElementId, FieldId, FormatId, GroupId, Interner, LinkId, ValueId};
use crate::io::{Reactor, ReactorFactory, SyncReactor};
use crate::log::{log_channel, Level, LevelFilter, Log, LogDrain};
use crate::memory::Pool;
use crate::ring::{spsc, Consumer, Producer};
use crate::time::Timestamp;

/// Milestone raw-bytes format id. Real negotiation (spec: Formats) assigns these.
const BYTES: FormatId = FormatId(0);

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

impl StopHandle {
    pub fn stop(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_stopped(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

pub struct Pipeline {
    elements: Vec<Option<Box<dyn Element>>>,
    edges: Vec<Edge>,
    // The three interning domains (spec: Formats — one interner per domain). Offers
    // declared as `&'static str` on pads are lowered through these at link time, so
    // ids are consistent across the whole graph and the solver only ever sees `u32`s.
    formats: Interner,
    fields: Interner,
    values: Interner,
    stop: Arc<AtomicBool>,
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
}

impl Pipeline {
    pub fn new() -> Self {
        let (tx, rx) = Bus::channel();
        Self {
            elements: Vec::new(),
            edges: Vec::new(),
            formats: Interner::new(),
            fields: Interner::new(),
            values: Interner::new(),
            stop: Arc::new(AtomicBool::new(false)),
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
        }
    }

    /// Install a per-thread-group reactor factory (e.g. io_uring). Each group thread
    /// calls it to build its own reactor. Defaults to the dependency-free
    /// [`SyncReactor`] (spec: IO — the reactor is the portability boundary).
    pub fn set_reactor_factory(&mut self, factory: ReactorFactory) {
        self.reactor_factory = Some(factory);
    }

    /// A handle to stop this pipeline from another thread (or a signal handler)
    /// while `run()` is blocking.
    pub fn stop_handle(&self) -> StopHandle {
        StopHandle(Arc::clone(&self.stop))
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
        let id = ElementId(self.elements.len() as u32);
        self.elements.push(Some(Box::new(element)));
        id
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
        // field/value names once, here), then intersect. Descriptors are `'static`, so
        // these borrows are fine to hold across the two lowering passes.
        let src_desc = &self.elements[src.0 .0 as usize].as_ref().unwrap().desc().pads[src_pad];
        let src_offers: Vec<_> = src_desc
            .offers
            .iter()
            .map(|o| o.lower(&mut self.formats, &mut self.fields, &mut self.values))
            .collect();
        let sink_desc = &self.elements[sink.0 .0 as usize].as_ref().unwrap().desc().pads[sink_pad];
        let sink_offers: Vec<_> = sink_desc
            .offers
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

    /// Verify a named pad exists on an element with the expected direction, returning
    /// its index in the element's `desc().pads` (which is the pad's local `PadId`).
    fn check_pad(&self, el: ElementId, pad: &str, dir: Direction) -> Result<usize, Error> {
        let e = self
            .elements
            .get(el.0 as usize)
            .and_then(|o| o.as_ref())
            .ok_or(Error::Todo("link: unknown element"))?;
        match e.desc().pads.iter().position(|p| p.name == pad) {
            Some(i) if e.desc().pads[i].direction == dir => Ok(i),
            Some(_) => Err(Error::Resource(format!(
                "link: pad '{pad}' on '{}' has the wrong direction",
                e.desc().name
            ))),
            None => Err(Error::Resource(format!(
                "link: element '{}' has no pad '{pad}'",
                e.desc().name
            ))),
        }
    }

    /// An element's descriptor name, for link diagnostics.
    fn name_of(&self, el: ElementId) -> &'static str {
        self.elements
            .get(el.0 as usize)
            .and_then(|o| o.as_ref())
            .map_or("?", |e| e.desc().name)
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
            .map(|e| {
                let n = e.as_ref().map_or(0, |el| el.desc().pads.len());
                vec![None; n]
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

    // --- Running (finite: drives to EOS across all group threads) ---

    /// Start the chain, drive it to EOS, and stop it. Spawns one thread per group,
    /// joins them all, and returns the first error (if any).
    pub fn run(&mut self) -> Result<(), Error> {
        self.stop.store(false, Ordering::Release);
        let order = self.linear_order()?;
        let groups = self.compute_groups(&order)?;
        let ng = groups.len();
        let total = self.elements.len();
        let pool = Pool::bounded(self.slot_size, self.pool_slots as u32);
        let credits = self.credits;
        let factory: ReactorFactory = self
            .reactor_factory
            .clone()
            .unwrap_or_else(|| Arc::new(|| Ok(Box::new(SyncReactor::new()) as Box<dyn Reactor>)));

        // Rings between consecutive groups: ring i connects group i → group i+1.
        let mut producers: Vec<Option<Producer<Batch>>> = (0..ng).map(|_| None).collect();
        let mut consumers: Vec<Option<Consumer<Batch>>> = (0..ng).map(|_| None).collect();
        for i in 0..ng.saturating_sub(1) {
            let (p, c) = spsc::<Batch>(self.queue_cap);
            producers[i] = Some(p);
            consumers[i + 1] = Some(c);
        }

        // Per-element counters, shared with the app via `counters()` after run.
        let counters: Vec<Arc<ElementCounters>> =
            (0..total).map(|_| Arc::new(ElementCounters::default())).collect();
        self.counters = counters.clone();
        let stop = Arc::clone(&self.stop);

        // Formats fixed at link time, resolved per element so each group can install
        // them on its elements' `Ctx`s (spec: elements read their fixed format from
        // `Ctx`). `.take()` moves each element's table into its group thread.
        let mut negotiated = self.negotiated_by_element();

        // Logging (spec: Debuggability — Logging cont'd). Build the level filter once
        // from the programmatic level and `STREAMCRAFT_DEBUG`, then, *only if some level
        // is actually enabled*, wire one log channel per element and one low-priority
        // drain thread. When logging is off nothing here allocates or spawns — the
        // disabled path stays at zero cost, which is the point (performance is #1).
        let (mut per_elem_logs, log_drain) = self.build_logging(total);

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
            let group_formats: Vec<Vec<Option<FixedFormat>>> = ids
                .iter()
                .map(|id| std::mem::take(&mut negotiated[id.0 as usize]))
                .collect();
            // Move each element's `Log` (if any) into its group thread, keyed by id.
            let group_logs: Vec<Option<Log>> = ids
                .iter()
                .map(|id| per_elem_logs[id.0 as usize].take())
                .collect();
            let upstream = consumers[gi].take();
            let downstream = producers[gi].take();
            let pool = pool.clone();
            let bus = self.bus_sender.clone();
            let factory = Arc::clone(&factory);
            let stop = Arc::clone(&stop);
            handles.push(std::thread::spawn(move || {
                run_group(
                    elems, ids, group_formats, group_logs, upstream, downstream, factory, pool,
                    bus, credits, group_counters, stop,
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

        let s = pool.stats();
        self.last_report = Some(RunReport {
            pool_slot_allocations: s.slot_allocations,
            pool_high_water: s.high_water,
            buffers: s.acquires,
        });

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
            log.set_element(ElementId(i as u32));
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
    fn linear_order(&self) -> Result<Vec<ElementId>, Error> {
        let n = self.elements.len();
        if n == 0 {
            return Err(Error::Todo("empty pipeline"));
        }
        let mut indeg = vec![0usize; n];
        let mut next: Vec<Option<ElementId>> = vec![None; n];
        for e in &self.edges {
            indeg[e.sink.0 as usize] += 1;
            next[e.src.0 as usize] = Some(e.sink);
        }
        let mut start = None;
        for (i, &deg) in indeg.iter().enumerate() {
            if deg == 0 {
                if start.is_some() {
                    return Err(Error::Todo("not a linear chain (multiple sources)"));
                }
                start = Some(ElementId(i as u32));
            }
        }
        let mut cur = start.ok_or(Error::Todo("no source (cycle?)"))?;
        let mut order = Vec::with_capacity(n);
        loop {
            order.push(cur);
            if order.len() > n {
                return Err(Error::Todo("cycle detected"));
            }
            match next[cur.0 as usize] {
                Some(d) => cur = d,
                None => break,
            }
        }
        if order.len() != n {
            return Err(Error::Todo("disconnected graph (not a single chain)"));
        }
        Ok(order)
    }

    /// Split the ordered chain into thread groups: a new group begins at each active
    /// element; passive elements inline into the current group (spec: Scheduling).
    fn compute_groups(&self, order: &[ElementId]) -> Result<Vec<Vec<ElementId>>, Error> {
        let mut groups: Vec<Vec<ElementId>> = Vec::new();
        for &id in order {
            let sched = self.elements[id.0 as usize]
                .as_ref()
                .ok_or(Error::Todo("element already consumed"))?
                .desc()
                .sched;
            if groups.is_empty() || matches!(sched, SchedHint::Active) {
                groups.push(vec![id]);
            } else {
                groups.last_mut().unwrap().push(id);
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

    pub fn set(&mut self, _el: ElementId, _prop: &str, _v: Value) -> Result<(), Error> {
        todo!("spec: Runtime configuration — live vs. structural props")
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
        if let Ok(order) = self.linear_order() {
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

    pub fn latency_report(&self) -> LatencyReport {
        todo!("spec: Latency")
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
                        let _ = crate::log::format_record(&mut lock, &rec);
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
    upstream: Option<Consumer<Batch>>,
    downstream: Option<Producer<Batch>>,
    factory: ReactorFactory,
    pool: Pool,
    bus: BusSender,
    credits: u32,
    counters: Vec<Arc<ElementCounters>>,
    stop: Arc<AtomicBool>,
) -> Result<(), Error> {
    let m = elements.len();
    let is_source = upstream.is_none();
    let mut reactor: Box<dyn Reactor> =
        factory().map_err(|e| Error::Resource(format!("reactor init: {e}")))?;
    let mut ctxs: Vec<Ctx> = ids
        .iter()
        .map(|id| Ctx::new(pool.clone(), bus.clone(), *id, BYTES, credits))
        .collect();
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

    // Start each element; hand any file it registered to this group's reactor.
    for i in 0..m {
        elements[i].start(&mut ctxs[i])?;
        if let Some(f) = ctxs[i].take_registration() {
            reactor.set_file(ids[i], f);
        }
    }

    let mut source_eos = false;
    let mut upstream_closed = false;

    let result: Result<(), Error> = loop {
        // Cooperative cancellation: break, then stop + drop downstream, which closes
        // the ring and cascades the stop to the next group.
        if stop.load(Ordering::Acquire) {
            break Ok(());
        }
        let mut progressed = false;

        // A. Feed the head from upstream (non-source groups only).
        if let Some(up) = &upstream {
            let closed = up.is_closed();
            let head_idle =
                ctxs[0].input_is_empty() && ctxs[0].inbox_empty() && reactor.is_idle();
            if !closed && head_idle {
                // Nothing to do but wait for input — block (no busy spin).
                match up.pop() {
                    Some(mut batch) => {
                        counters[0].record_in(batch.len() as u64, batch.total_bytes());
                        ctxs[0].input_append(&mut batch);
                        progressed = true;
                    }
                    None => upstream_closed = true,
                }
            } else {
                while let Some(mut batch) = up.try_pop() {
                    counters[0].record_in(batch.len() as u64, batch.total_bytes());
                    ctxs[0].input_append(&mut batch);
                    progressed = true;
                }
                if closed {
                    upstream_closed = true;
                }
            }
        }

        // B. Run the group's elements inline (head active, passive tail).
        let mut fatal = None;
        for i in 0..m {
            let flow = if i == 0 && is_source {
                elements[0].process(&mut ctxs[0], Inputs::empty())
            } else {
                let mut input = ctxs[i].take_input();
                let f = elements[i].process(&mut ctxs[i], Inputs::owned(&mut input));
                ctxs[i].set_input(input);
                f
            };
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
            let (nbuf, nbytes) = {
                let out = ctxs[i].output_mut();
                (out.len() as u64, out.total_bytes())
            };
            counters[i].record_out(nbuf, nbytes);
            reactor.submit(ctxs[i].take_submissions());
            if i + 1 < m {
                counters[i + 1].record_in(nbuf, nbytes);
                let (left, right) = ctxs.split_at_mut(i + 1);
                right[0].input_append(left[i].output_mut());
            }
        }
        if let Some(e) = fatal {
            break Err(e);
        }

        // C. Push the last element's output downstream (blocking backpressure).
        if let Some(down) = &downstream {
            let out = std::mem::replace(ctxs[m - 1].output_mut(), Batch::new(BYTES));
            if !out.is_empty() {
                progressed = true;
                if down.push(out).is_err() {
                    break Ok(()); // downstream gone
                }
            }
        } else {
            ctxs[m - 1].output_mut().clear();
        }

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
            && ctxs.iter().all(|c| c.inbox_empty() && c.input_is_empty());
        if head_done && quiescent {
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
