//! The pipeline: owns topology, lifecycle, clock, latency (spec: The pipeline API).
//! Elements arrive already constructed — `pipeline.add(FileSrc::new(path))`.
//!
//! Milestone 1 implements a **single-threaded linear scheduler**: a chain of
//! elements driven inline (source → … → sink), one batch per step, until the source
//! reports EOS. Thread groups, queues, and the reactor (spec: Scheduling) replace
//! `run()`'s internals later; `play()` / `step()` grow from the same code.

use crate::batch::{Batch, Inputs};
use crate::bus::{Bus, BusMessage, BusSender, State};
use crate::counters::{CounterSnapshot, LatencyReport};
use crate::ctx::Ctx;
use crate::element::{Element, Flow, Template};
use crate::error::Error;
use crate::format::{FieldConstraint, Value};
use crate::id::{ElementId, FormatId, GroupId, LinkId, PadId};
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
            last_report: None,
        }
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
    /// exhausted and all data has drained to the sink.
    pub fn run(&mut self) -> Result<(), Error> {
        let order = self.linear_order()?;
        let n = order.len();
        let pool = Pool::new(self.slot_size);

        let mut chain: Vec<Box<dyn Element>> = Vec::with_capacity(n);
        for id in &order {
            let el = self.elements[id.0 as usize]
                .take()
                .ok_or(Error::Todo("element already consumed by a previous run"))?;
            chain.push(el);
        }
        let mut ctxs: Vec<Ctx> = order
            .iter()
            .map(|id| Ctx::new(pool.clone(), self.bus_sender.clone(), *id, BYTES))
            .collect();

        let result = drive(&mut chain, &mut ctxs, n);

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

/// The single-threaded linear drive loop (spec: Scheduling — minimal scheduler).
fn drive(chain: &mut [Box<dyn Element>], ctxs: &mut [Ctx], n: usize) -> Result<(), Error> {
    for i in 0..n {
        chain[i].start(&mut ctxs[i])?;
    }

    // Buffers moving downstream between elements; reused (cleared) each step.
    let mut carry = Batch::new(BYTES);

    loop {
        // Source: pull one batch.
        let flow0 = chain[0].process(&mut ctxs[0], Inputs::empty())?;
        carry.clear();
        std::mem::swap(&mut carry, ctxs[0].output_mut());
        let had_output = !carry.is_empty();

        // Transforms and sink: feed each the previous element's output.
        for i in 1..n {
            {
                let entries = [(PadId(0), carry.view())];
                chain[i].process(&mut ctxs[i], Inputs::new(&entries))?;
            }
            carry.clear(); // recycle element i-1's buffers
            std::mem::swap(&mut carry, ctxs[i].output_mut());
        }
        carry.clear();

        if matches!(flow0, Flow::Eos) && !had_output {
            break;
        }
    }

    Ok(())
}
