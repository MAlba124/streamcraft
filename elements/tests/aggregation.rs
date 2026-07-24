//! Aggregation / fan-in in the scheduler (spec: Aggregation). Two sources with
//! interleaved PTS feed a muxer with two sink pads; the muxer merges them into one
//! PTS-ordered stream. This is the symmetric counterpart to branching: a group with
//! several *upstream* rings. The muxer only emits once every still-open pad has a
//! buffered head, so the merge order is deterministic regardless of thread timing.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use streamcraft_core::batch::Inputs;
use streamcraft_core::buffer::Buffer;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

// --- a source that stamps each buffer with a PTS and a marker byte ------------------
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "ptssrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct PtsSrc {
    marker: u8,
    start_ms: u64,
    step_ms: u64,
    count: usize,
    produced: usize,
}
impl PtsSrc {
    fn new(marker: u8, start_ms: u64, step_ms: u64, count: usize) -> Self {
        Self { marker, start_ms, step_ms, count, produced: 0 }
    }
}
impl Element for PtsSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.produced >= self.count {
            return Ok(Flow::Eos);
        }
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return Ok(Flow::Ok),
        };
        buf.memory.as_mut_full()[0] = self.marker;
        buf.memory.set_len(1);
        buf.pts = Timestamp::from_millis(self.start_ms + self.produced as u64 * self.step_ms);
        self.produced += 1;
        ctx.out(PadId(0)).push(buf);
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// --- a two-input muxer that merges by PTS -------------------------------------------
static MUX_PADS: [PadDesc; 3] = [
    PadDesc { name: "sink0", direction: Direction::Sink, offers: &OFFERS, dynamic: false, validate: None },
    PadDesc { name: "sink1", direction: Direction::Sink, offers: &OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &OFFERS, dynamic: false, validate: None },
];
static MUX_DESC: ElementDesc = ElementDesc {
    name: "interleavemux",
    pads: &MUX_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::All { by: streamcraft_core::element::AlignBy::RunningTime },
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};
const SINK0: PadId = PadId(0);
const SINK1: PadId = PadId(1);
const MSRC: PadId = PadId(2);

struct InterleaveMux {
    queues: [VecDeque<Buffer>; 2],
}
impl InterleaveMux {
    fn new() -> Self {
        Self { queues: [VecDeque::new(), VecDeque::new()] }
    }
    /// Emit while every still-open pad has a buffered head — then the lowest-PTS head is
    /// safe to release (no open pad can still produce something earlier). A closed pad is
    /// ignored once drained; a live but empty pad blocks emission until it delivers.
    fn emit_ready(&mut self, ctx: &mut Ctx) {
        loop {
            let mut min: Option<(usize, Timestamp)> = None;
            let mut blocked = false;
            for (i, q) in self.queues.iter().enumerate() {
                let pad = PadId(i as u32);
                match q.front() {
                    Some(buf) => {
                        if min.is_none_or(|(_, p)| buf.pts < p) {
                            min = Some((i, buf.pts));
                        }
                    }
                    None => {
                        if !ctx.is_pad_closed(pad) {
                            blocked = true; // a live pad might still deliver an earlier PTS
                        }
                    }
                }
            }
            if blocked {
                break;
            }
            match min {
                Some((i, _)) => {
                    let buf = self.queues[i].pop_front().unwrap();
                    ctx.out(MSRC).push(buf);
                }
                None => break,
            }
        }
    }
}
impl Element for InterleaveMux {
    fn desc(&self) -> &'static ElementDesc {
        &MUX_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        for (i, pad) in [SINK0, SINK1].into_iter().enumerate() {
            let mut batch = ctx.take_input_on(pad);
            while let Some(buf) = batch.pop_front() {
                self.queues[i].push_back(buf);
            }
        }
        self.emit_ready(ctx);
        Ok(Flow::Ok)
    }
    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::Eos) {
            self.emit_ready(ctx); // both pads closed → drain the remainder in PTS order
        }
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// --- a sink that records (pts_ms, marker) in receive order --------------------------
static REC_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static REC_DESC: ElementDesc = ElementDesc {
    name: "recordingsink",
    pads: &REC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

#[derive(Clone)]
struct Recorder(Arc<Mutex<Vec<(u64, u8)>>>);
struct RecordingSink {
    rec: Recorder,
}
impl Element for RecordingSink {
    fn desc(&self) -> &'static ElementDesc {
        &REC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            let pts_ms = buf.pts.nanos().unwrap_or(0) / 1_000_000;
            let marker = buf.memory.data().first().copied().unwrap_or(0xFF);
            self.rec.0.lock().unwrap().push((pts_ms, marker));
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[test]
fn muxer_merges_two_streams_in_pts_order() {
    use streamcraft_core::pipeline::Pipeline;

    // Stream A: PTS 0,20,40,60,80 (marker 0). Stream B: 10,30,50,70,90 (marker 1).
    let n = 5;
    let rec = Recorder(Arc::new(Mutex::new(Vec::new())));

    let mut p = Pipeline::new();
    let a = p.add(PtsSrc::new(0, 0, 20, n));
    let b = p.add(PtsSrc::new(1, 10, 20, n));
    let mux = p.add(InterleaveMux::new());
    let sink = p.add(RecordingSink { rec: rec.clone() });
    p.link((a, "src"), (mux, "sink0")).expect("a -> mux.sink0");
    p.link((b, "src"), (mux, "sink1")).expect("b -> mux.sink1");
    p.link((mux, "src"), (sink, "sink")).expect("mux -> sink");
    p.run().expect("run");

    let got = rec.0.lock().unwrap().clone();
    // Every buffer arrived, merged strictly by PTS (0,10,20,...,90), markers alternating.
    let want: Vec<(u64, u8)> = (0..2 * n as u64).map(|k| (k * 10, (k % 2) as u8)).collect();
    assert_eq!(got, want, "muxer emitted a single PTS-ordered stream from two inputs");
}
