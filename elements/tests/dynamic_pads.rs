//! Dynamic pads + preroll (spec: dynamic pads — the "god bin" keystone). A demuxer
//! instantiates one src pad per discovered stream during `preroll`; the app links each
//! (returned by `Pipeline::preroll`) to its own sink; the schedule then freezes and
//! streams. This is the multi-stream-demux → two-sinks milestone: each logical stream
//! reaches its own downstream chain, and `dump_dot` shows the settled topology.

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::PadId;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;
use profluens_elements::testing::{fold, TestSink, FNV_OFFSET};

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

// --- a source that tags each buffer with a stream id -------------------------------
//
// Produces `streams * per_stream` buffers round-robin across streams; buffer j is filled
// with the byte `(j % streams)`, so stream s carries `per_stream` buffers of all-`s`.
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "taggedsrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct TaggedSrc {
    streams: usize,
    per_stream: usize,
    buf_size: usize,
    produced: usize,
}
impl TaggedSrc {
    fn new(streams: usize, per_stream: usize, buf_size: usize) -> Self {
        Self { streams, per_stream, buf_size, produced: 0 }
    }
    fn total(&self) -> usize {
        self.streams * self.per_stream
    }
}
impl Element for TaggedSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.produced >= self.total() {
            return Ok(Flow::Eos);
        }
        let stream = (self.produced % self.streams) as u8;
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return Ok(Flow::Ok),
        };
        let dst = buf.memory.as_mut_full();
        for b in dst.iter_mut().take(self.buf_size) {
            *b = stream;
        }
        buf.memory.set_len(self.buf_size);
        self.produced += 1;
        ctx.out(PadId(0)).push(buf);
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// --- a demuxer with dynamic src pads -----------------------------------------------
//
// Only a static sink pad; it discovers `streams` streams at preroll and adds one src pad
// each, then routes every buffer to `pad[first_byte]` (its stream tag).
static DEMUX_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static DEMUX_DESC: ElementDesc = ElementDesc {
    name: "demux",
    pads: &DEMUX_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct Demux {
    streams: usize,
    pads: Vec<PadId>,
}
impl Demux {
    fn new(streams: usize) -> Self {
        Self { streams, pads: Vec::new() }
    }
}
impl Element for Demux {
    fn desc(&self) -> &'static ElementDesc {
        &DEMUX_DESC
    }
    fn preroll(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Discover streams (here: a known count) and expose one src pad per stream.
        for i in 0..self.streams {
            let pad = ctx.add_pad(Direction::Src, &format!("src{i}"), &OFFERS);
            self.pads.push(pad);
        }
        Ok(())
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            // Route by stream tag (the first byte). Forward the buffer unchanged — no copy.
            let Some(tag) = buf.memory.data().first().copied() else { continue };
            let pad = self.pads[(tag as usize) % self.pads.len()];
            ctx.out(pad).push(buf);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

fn repeated_hash(byte: u8, n: usize) -> u64 {
    let mut h = FNV_OFFSET;
    for _ in 0..n {
        h = fold(h, byte);
    }
    h
}

#[test]
fn demux_routes_each_discovered_stream_to_its_own_sink() {
    let (streams, per_stream, buf_size) = (2usize, 40usize, 500usize);
    let (sink0, st0) = TestSink::new();
    let (sink1, st1) = TestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TaggedSrc::new(streams, per_stream, buf_size));
    let demux = p.add(Demux::new(streams));
    let k0 = p.add(sink0);
    let k1 = p.add(sink1);
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");

    // Preroll: the demuxer instantiates its two src pads; link each to its sink.
    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 2, "demux exposed two runtime src pads");
    assert!(added.iter().all(|a| a.element == demux));
    p.link((added[0].element, &added[0].name), (k0, "sink")).expect("pad0 -> k0");
    p.link((added[1].element, &added[1].name), (k1, "sink")).expect("pad1 -> k1");

    p.run().expect("run");

    // src0 carries stream 0 (all 0x00) → sink 0; src1 carries stream 1 (all 0x01) → sink 1.
    let per_stream_bytes = (per_stream * buf_size) as u64;
    assert!(st0.is_done() && st1.is_done(), "both sinks saw EOS");
    assert_eq!(st0.bytes(), per_stream_bytes, "sink 0 got its whole stream");
    assert_eq!(st1.bytes(), per_stream_bytes, "sink 1 got its whole stream");
    assert_eq!(
        st0.hash(),
        repeated_hash(0, per_stream * buf_size),
        "sink 0 received only stream-0 bytes, in order"
    );
    assert_eq!(
        st1.hash(),
        repeated_hash(1, per_stream * buf_size),
        "sink 1 received only stream-1 bytes, in order"
    );
    assert_ne!(st0.hash(), st1.hash(), "the two streams were demultiplexed distinctly");
}

#[test]
fn dump_dot_shows_the_settled_topology() {
    let (sink0, _s0) = TestSink::new();
    let (sink1, _s1) = TestSink::new();
    let mut p = Pipeline::new();
    let src = p.add(TaggedSrc::new(2, 1, 8));
    let demux = p.add(Demux::new(2));
    let k0 = p.add(sink0);
    let k1 = p.add(sink1);
    p.link((src, "src"), (demux, "sink")).unwrap();
    let added = p.preroll().unwrap();
    p.link((added[0].element, &added[0].name), (k0, "sink")).unwrap();
    p.link((added[1].element, &added[1].name), (k1, "sink")).unwrap();

    let dot = p.dump_dot();
    assert!(dot.contains("demux") && dot.contains("taggedsrc"));
    // src → demux, then the demux fans out to both sinks (the settled dynamic topology).
    assert!(dot.contains("e0 -> e1"), "src → demux:\n{dot}");
    assert!(dot.contains("e1 -> e2"), "demux → sink 0:\n{dot}");
    assert!(dot.contains("e1 -> e3"), "demux → sink 1:\n{dot}");
}
