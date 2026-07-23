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
use crate::format::{FieldConstraint, Value};
use crate::id::{ElementId, FormatId, GroupId, LinkId};
use crate::io::{Reactor, ReactorFactory, SyncReactor};
use crate::memory::Pool;
use crate::ring::{spsc, Consumer, Producer};
use crate::time::Timestamp;

/// Milestone raw-bytes format id. Real negotiation (spec: Formats) assigns these.
const BYTES: FormatId = FormatId(0);

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
    links: Vec<(ElementId, ElementId)>,
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
}

impl Pipeline {
    pub fn new() -> Self {
        let (tx, rx) = Bus::channel();
        Self {
            elements: Vec::new(),
            links: Vec::new(),
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

    // --- Topology: legal in every state (spec: Runtime configuration) ---

    /// Elements arrive already constructed: `pipeline.add(FileSrc::new(path))`.
    pub fn add(&mut self, element: impl Element + 'static) -> ElementId {
        let id = ElementId(self.elements.len() as u32);
        self.elements.push(Some(Box::new(element)));
        id
    }

    /// Link a src pad to a sink pad. Milestone: linear chains, single pads; pad names
    /// are recorded but not yet validated or negotiated (spec: Formats).
    pub fn link(&mut self, src: (ElementId, &str), sink: (ElementId, &str)) -> Result<LinkId, Error> {
        self.check_pad(src.0, src.1, Direction::Src)?;
        self.check_pad(sink.0, sink.1, Direction::Sink)?;
        let id = LinkId(self.links.len() as u32);
        self.links.push((src.0, sink.0));
        Ok(id)
    }

    /// Verify a named pad exists on an element with the expected direction.
    /// (Format negotiation over the pads' offers is wired here once elements declare
    /// them — for now offers are empty, so this checks pad name + direction.)
    fn check_pad(&self, el: ElementId, pad: &str, dir: Direction) -> Result<(), Error> {
        let e = self
            .elements
            .get(el.0 as usize)
            .and_then(|o| o.as_ref())
            .ok_or(Error::Todo("link: unknown element"))?;
        match e.desc().pads.iter().find(|p| p.name == pad) {
            Some(p) if p.direction == dir => Ok(()),
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
            let upstream = consumers[gi].take();
            let downstream = producers[gi].take();
            let pool = pool.clone();
            let bus = self.bus_sender.clone();
            let factory = Arc::clone(&factory);
            let stop = Arc::clone(&stop);
            handles.push(std::thread::spawn(move || {
                run_group(
                    elems, ids, upstream, downstream, factory, pool, bus, credits, group_counters,
                    stop,
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
        for (s, d) in &self.links {
            indeg[d.0 as usize] += 1;
            next[s.0 as usize] = Some(*d);
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
        for (src, dst) in &self.links {
            s.push_str(&format!("  e{} -> e{};\n", src.0, dst.0));
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

/// One thread group's run loop (spec: Scheduling — a group runs on one thread). The
/// group is `[active_head, passive_tail...]`; the head pulls from the upstream ring
/// (unless it is the source) and the tail runs inline; the last element's output goes
/// to the downstream ring. Backpressure is the blocking ring push; the group parks on
/// the upstream ring when idle. Its reactor serves the group's IO.
#[allow(clippy::too_many_arguments)]
fn run_group(
    mut elements: Vec<Box<dyn Element>>,
    ids: Vec<ElementId>,
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
