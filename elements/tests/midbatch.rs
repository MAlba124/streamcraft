//! Mid-batch format-change boundaries (spec: Events ordered relative to buffers —
//! the last dynamic-caps gap, Stage 4.2's first half): an element that announces a
//! new format *between two pushes in one `process()` call* must have the
//! `FormatChange` delivered downstream **between exactly those two buffers**, not
//! before or after the whole batch. The machinery under test: the announcement
//! captures its output-row position; the event rides the batch positioned; the
//! consumer's drain cursor treats it as a barrier and the scheduler delivers it
//! when consumption reaches the row.

use std::sync::{Arc, Mutex};

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;

static SRC_FIELDS: [FieldDesc; 1] =
    [FieldDesc { field: "rate", allowed: ConstraintDesc::Any, preferred: None }];
static OFFERS: [OfferDesc; 1] = [OfferDesc { family: "midtest", fields: &SRC_FIELDS }];

static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "annsrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

/// One `process()` call emits buffers 0..3, announces `rate=999` **mid-call**, then
/// emits buffers 3..5 — all in the same output batch. Next call: EOS.
struct AnnSrc {
    emitted: bool,
}

impl Element for AnnSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.emitted {
            return Ok(Flow::Eos);
        }
        self.emitted = true;
        for i in 0..5u8 {
            if i == 3 {
                // The change applies from buffer 3 on — announced between pushes.
                ctx.announce_format(PadId(0), "midtest", &[("rate", ValueDesc::Int(999))]);
            }
            let mut buf = ctx.alloc(PadId(0));
            buf.memory.as_mut_full()[0] = i;
            buf.memory.set_len(1);
            ctx.out(PadId(0)).push(buf);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "ordersink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Item {
    Buf(u8),
    Rate(i64),
}

/// Records the exact interleaving of buffers and format changes as it experiences
/// them — the order is the assertion.
struct OrderSink {
    seen: Arc<Mutex<Vec<Item>>>,
}

impl Element for OrderSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut seen = self.seen.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            seen.push(Item::Buf(buf.memory.data()[0]));
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if let Event::FormatChange(f) = event {
            let rate = ctx
                .field_id("rate")
                .and_then(|id| f.get(id))
                .and_then(|v| match v {
                    Value::Int(n) => Some(n),
                    _ => None,
                })
                .unwrap_or(-1);
            self.seen.lock().unwrap().push(Item::Rate(rate));
        }
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// Across an inter-group ring (Active → Active): the change lands between buffer 2
/// and buffer 3 — exactly where it was announced — even though all five buffers
/// crossed in one batch.
#[test]
fn mid_batch_announce_lands_between_the_right_buffers() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(AnnSrc { emitted: false });
    let snk = p.add(OrderSink { seen: Arc::clone(&seen) });
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.run().expect("run");

    let got = seen.lock().unwrap().clone();
    assert_eq!(
        got,
        vec![
            Item::Buf(0),
            Item::Buf(1),
            Item::Buf(2),
            Item::Rate(999),
            Item::Buf(3),
            Item::Buf(4),
        ],
        "the FormatChange splits the batch exactly at its announcement position"
    );
}

/// The same property through an inline (passive) member: `annsrc ! queue !
/// ordersink` — the queue forwards the change (`Ctx::forward_format`), and the
/// position must survive both the inline hand-off and the forwarding hop.
#[test]
fn mid_batch_position_survives_a_forwarding_transport() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(AnnSrc { emitted: false });
    let q = p.add(profluens_elements::flow::Queue::new());
    let snk = p.add(OrderSink { seen: Arc::clone(&seen) });
    p.link((src, "src"), (q, "sink")).expect("src!q");
    p.link((q, "src"), (snk, "sink")).expect("q!snk");
    p.run().expect("run");

    let got = seen.lock().unwrap().clone();
    let rate_pos = got.iter().position(|i| matches!(i, Item::Rate(999))).expect("change arrived");
    let bufs_before: Vec<u8> = got[..rate_pos]
        .iter()
        .filter_map(|i| match i {
            Item::Buf(b) => Some(*b),
            _ => None,
        })
        .collect();
    assert_eq!(bufs_before, vec![0, 1, 2], "exactly the pre-announce buffers precede the change");
    let bufs_after: Vec<u8> = got[rate_pos..]
        .iter()
        .filter_map(|i| match i {
            Item::Buf(b) => Some(*b),
            _ => None,
        })
        .collect();
    assert_eq!(bufs_after, vec![3, 4], "exactly the post-announce buffers follow it");
}
