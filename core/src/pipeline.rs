//! The pipeline: owns topology, lifecycle, clock, latency (spec: The pipeline API).
//! Elements arrive already constructed — `pipeline.add(FileSrc::new(path))`.
//!
//! Milestone 1 implements a **single-threaded, reactor-driven linear scheduler**: a
//! chain of elements (source → … → sink) driven inline, with IO going through the
//! reactor (submit → execute → complete). Each round: every element processes
//! (draining its IO completions, consuming its input, emitting output and/or new IO
//! submissions), the reactor executes queued ops, and completions are routed back.
//! Thread groups, real queues, and an async/io_uring reactor (spec: Scheduling, IO)
//! replace the internals later; `play()` / `step()` grow from the same loop.

use crate::batch::Inputs;
use crate::bus::{Bus, BusMessage, BusSender, State};
use crate::counters::{CounterSnapshot, LatencyReport};
use crate::ctx::Ctx;
use crate::element::{Element, Flow, Template};
use crate::error::Error;
use crate::format::{FieldConstraint, Value};
use crate::id::{ElementId, FormatId, GroupId, LinkId};
use crate::io::{Reactor, SyncReactor};
use crate::memory::Pool;
use crate::time::Timestamp;

/// Milestone-1 raw-bytes format id. Real negotiation (spec: Formats) assigns these.
const BYTES: FormatId = FormatId(0);

/// A summary of a finished [`Pipeline::run`], for tests and profiling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunReport {
    /// Payload buffers ever heap-allocated. Flat == the zero-alloc criterion.
    pub pool_slot_allocations: u64,
    pub pool_high_water: u64,
    pub buffers: u64,
}

pub struct Pipeline {
    elements: Vec<Option<Box<dyn Element>>>,
    links: Vec<(ElementId, ElementId)>,
    bus_sender: BusSender,
    bus: Bus,
    slot_size: usize,
    pool_slots: usize,
    reactor: Option<Box<dyn Reactor>>,
    last_report: Option<RunReport>,
}

impl Pipeline {
    pub fn new() -> Self {
        let (tx, rx) = Bus::channel();
        Self {
            elements: Vec::new(),
            links: Vec::new(),
            bus_sender: tx,
            bus: rx,
            slot_size: 128 * 1024,
            pool_slots: 4,
            reactor: None,
            last_report: None,
        }
    }

    /// Install a custom IO reactor backend (e.g. io_uring). Defaults to the
    /// dependency-free [`SyncReactor`] when unset (spec: IO — the reactor is the
    /// portability boundary).
    pub fn set_reactor(&mut self, reactor: Box<dyn Reactor>) {
        self.reactor = Some(reactor);
    }

    // --- Topology: legal in every state (spec: Runtime configuration) ---

    /// Elements arrive already constructed: `pipeline.add(FileSrc::new(path))`.
    pub fn add(&mut self, element: impl Element + 'static) -> ElementId {
        let id = ElementId(self.elements.len() as u32);
        self.elements.push(Some(Box::new(element)));
        id
    }

    /// Link a src pad to a sink pad. Milestone 1: linear chains, single pads; pad
    /// names are recorded but not yet validated or negotiated (spec: Formats).
    pub fn link(&mut self, src: (ElementId, &str), sink: (ElementId, &str)) -> Result<LinkId, Error> {
        let _ = (src.1, sink.1);
        let id = LinkId(self.links.len() as u32);
        self.links.push((src.0, sink.0));
        Ok(id)
    }

    // --- Running (milestone 1: finite, drives to EOS) ---

    /// Start the chain, drive it to EOS, and stop it. Returns after the source is
    /// exhausted and all IO has drained through the sink.
    pub fn run(&mut self) -> Result<(), Error> {
        let order = self.linear_order()?;
        let n = order.len();
        let total = self.elements.len();
        let pool = Pool::bounded(self.slot_size, self.pool_slots as u32);
        let credits = self.pool_slots as u32;

        let mut chain: Vec<Box<dyn Element>> = Vec::with_capacity(n);
        for id in &order {
            let el = self.elements[id.0 as usize]
                .take()
                .ok_or(Error::Todo("element already consumed by a previous run"))?;
            chain.push(el);
        }
        let mut ctxs: Vec<Ctx> = order
            .iter()
            .map(|id| Ctx::new(pool.clone(), self.bus_sender.clone(), *id, BYTES, credits))
            .collect();

        // element id -> index in the ordered chain (for routing completions).
        let mut elem_to_idx = vec![0usize; total];
        for (i, id) in order.iter().enumerate() {
            elem_to_idx[id.0 as usize] = i;
        }

        let mut reactor: Box<dyn Reactor> = self
            .reactor
            .take()
            .unwrap_or_else(|| Box::new(SyncReactor::new()));
        let result = drive(&mut chain, &mut ctxs, &order, &elem_to_idx, reactor.as_mut(), n);

        // Stop in reverse order regardless of how the drive ended.
        for i in (0..n).rev() {
            chain[i].stop(&mut ctxs[i]);
        }

        let s = pool.stats();
        self.last_report = Some(RunReport {
            pool_slot_allocations: s.slot_allocations,
            pool_high_water: s.high_water,
            buffers: s.acquires,
        });

        match result {
            Ok(()) => {
                self.bus_sender.send(BusMessage::Eos);
                Ok(())
            }
            Err(e) => {
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

    /// Order the elements into a single source→sink chain (milestone 1 topology).
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

    pub fn dump_dot(&self) -> String {
        todo!("spec: Debuggability — graph dump")
    }

    pub fn latency_report(&self) -> LatencyReport {
        todo!("spec: Latency")
    }

    pub fn counters(&self, _el: ElementId) -> CounterSnapshot {
        todo!("spec: Debuggability — per-element counters")
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// The reactor-driven linear drive loop (spec: Scheduling — minimal scheduler).
fn drive(
    chain: &mut [Box<dyn Element>],
    ctxs: &mut [Ctx],
    order: &[ElementId],
    elem_to_idx: &[usize],
    reactor: &mut dyn Reactor,
    n: usize,
) -> Result<(), Error> {
    // Start each element; hand any file it registered to the reactor.
    for i in 0..n {
        chain[i].start(&mut ctxs[i])?;
        if let Some(f) = ctxs[i].take_registration() {
            reactor.set_file(order[i], f);
        }
    }

    let mut source_eos = false;
    let mut guard: u64 = 0;
    const GUARD_MAX: u64 = 1 << 40;

    loop {
        guard += 1;
        if guard > GUARD_MAX {
            return Err(Error::Todo("scheduler failed to make progress"));
        }

        // One pass: each element processes, then its output moves to the next input.
        for i in 0..n {
            let flow = if i == 0 {
                chain[0].process(&mut ctxs[0], Inputs::empty())?
            } else {
                let mut input = ctxs[i].take_input();
                let f = chain[i].process(&mut ctxs[i], Inputs::owned(&mut input))?;
                ctxs[i].set_input(input); // any unconsumed input persists
                f
            };
            if i == 0 && matches!(flow, Flow::Eos) {
                source_eos = true;
            }

            let subs = ctxs[i].take_submissions();
            reactor.submit(subs);

            if i + 1 < n {
                let (left, right) = ctxs.split_at_mut(i + 1);
                right[0].input_append(left[i].output_mut());
            } else {
                ctxs[i].output_mut().clear(); // sink produces nothing
            }
        }

        // Execute queued IO and route completions back to their elements.
        let completions = reactor.run_once();
        let did_io = !completions.is_empty();
        for (elem, c) in completions {
            ctxs[elem_to_idx[elem.0 as usize]].deliver_completion(c);
        }

        // Quiescent (nothing executed, no queued/in-flight work) and source done.
        if source_eos
            && !did_io
            && reactor.is_idle()
            && ctxs.iter().all(|c| c.inbox_empty() && c.input_is_empty())
        {
            break;
        }
    }

    Ok(())
}
