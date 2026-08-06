//! Branching / fan-out in the scheduler (spec: Scheduling — toward dynamic pads). A
//! `Tee` with two src pads routes each to its own downstream chain; the scheduler
//! builds a DAG of thread groups with one ring per inter-group edge. Distinct data per
//! pad (A = pattern, B = pattern+1) proves each pad reaches its *own* sink, so a
//! mis-routing (both pads to one ring) would fail loudly.

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
use profluens_elements::testing::{fold, pattern_byte, TestSink, TestSrc, FNV_OFFSET};

// A two-output tee: forwards each input buffer to `src0` unchanged and to `src1` with
// every byte incremented, so the two branches carry distinct streams. Active, so it is
// its own thread group and its two src pads each feed a separate downstream group.
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static PADS: [PadDesc; 3] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src0", direction: Direction::Src, offers: &OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src1", direction: Direction::Src, offers: &OFFERS, dynamic: false, validate: None },
];
static DESC: ElementDesc = ElementDesc {
    name: "tee",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

const SRC0: PadId = PadId(1);
const SRC1: PadId = PadId(2);

struct Tee;

impl Element for Tee {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            let bytes = buf.memory.data().to_vec();
            let n = bytes.len();

            // src0: verbatim.
            let mut b0 = ctx.alloc(SRC0);
            b0.memory.as_mut_full()[..n].copy_from_slice(&bytes);
            b0.memory.set_len(n);
            ctx.out(SRC0).push(b0);

            // src1: each byte + 1 — a distinct stream, so a mis-route is visible.
            let mut b1 = ctx.alloc(SRC1);
            let dst = b1.memory.as_mut_full();
            for (i, &x) in bytes.iter().enumerate() {
                dst[i] = x.wrapping_add(1);
            }
            b1.memory.set_len(n);
            ctx.out(SRC1).push(b1);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[test]
fn tee_routes_each_src_pad_to_its_own_sink() {
    let n = 300_007u64; // not a multiple of the buffer size — frames straddle buffers
    let (sink_a, stats_a) = TestSink::new();
    let (sink_b, stats_b) = TestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TestSrc::new(n));
    let tee = p.add(Tee);
    let a = p.add(sink_a);
    let b = p.add(sink_b);
    p.link((src, "src"), (tee, "sink")).expect("src->tee");
    p.link((tee, "src0"), (a, "sink")).expect("tee.src0->a");
    p.link((tee, "src1"), (b, "sink")).expect("tee.src1->b");
    p.run().expect("run");

    assert!(stats_a.is_done() && stats_b.is_done(), "both sinks saw EOS");
    assert_eq!(stats_a.bytes(), n, "sink A received the whole stream");
    assert_eq!(stats_b.bytes(), n, "sink B received the whole stream");

    // Each pad carried its own distinct data to its own sink.
    let (mut ha, mut hb) = (FNV_OFFSET, FNV_OFFSET);
    for i in 0..n {
        ha = fold(ha, pattern_byte(i));
        hb = fold(hb, pattern_byte(i).wrapping_add(1));
    }
    assert_eq!(stats_a.hash(), ha, "src0 → sink A: pattern, intact and in order");
    assert_eq!(stats_b.hash(), hb, "src1 → sink B: pattern+1, intact and in order");
    assert_ne!(stats_a.hash(), stats_b.hash(), "the two pads route distinctly");
}

#[test]
fn dump_dot_shows_the_fan_out() {
    let (sink_a, _a) = TestSink::new();
    let (sink_b, _b) = TestSink::new();
    let mut p = Pipeline::new();
    let src = p.add(TestSrc::new(0));
    let tee = p.add(Tee);
    let a = p.add(sink_a);
    let b = p.add(sink_b);
    p.link((src, "src"), (tee, "sink")).unwrap();
    p.link((tee, "src0"), (a, "sink")).unwrap();
    p.link((tee, "src1"), (b, "sink")).unwrap();

    let dot = p.dump_dot();
    assert!(dot.contains("tee"));
    // Two edges leave the tee (e1): the settled fan-out topology.
    assert!(dot.contains("e1 -> e2"), "tee → sink A edge:\n{dot}");
    assert!(dot.contains("e1 -> e3"), "tee → sink B edge:\n{dot}");
}
