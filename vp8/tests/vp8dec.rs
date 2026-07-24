//! `Vp8Dec` element integration (spec: Milestone applications §5 — the decoder half).
//! Frames are encoded in-test with the same crate's encoder, decoded once directly as
//! the reference, then pushed through a pipeline `FrameSrc ! Vp8Dec ! PlaneRecordSink`
//! — output planes must be byte-identical to the reference decode, pts must pass
//! through, and the announced `video/raw` format must carry the real dimensions.
//! A corrupt mid-stream frame must degrade per-buffer (bus warning, next keyframe
//! recovers), never panic or kill the pipeline.

use std::sync::{Arc, Mutex};

use oxideav_vp8::encoder::{encode_keyframe, I420Frame, KeyframeParams};
use oxideav_vp8::Vp8DecoderState;

use streamcraft_core::batch::Inputs;
use streamcraft_core::bus::BusMessage;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

use sc_vp8::Vp8Dec;

const W: u32 = 128;
const H: u32 = 96;
/// One frame per 33 ms — a stand-in container timestamp grid.
const FRAME_NS: u64 = 33_000_000;

// ---- test elements -----------------------------------------------------------

static VP8_OFFERS: [OfferDesc; 1] = [OfferDesc::any("vp8")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &VP8_OFFERS,
    dynamic: false,
    validate: None,
}];

static FRAMESRC_DESC: ElementDesc = ElementDesc {
    name: "framesrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Emits pre-encoded VP8 frames, one per buffer (the demuxer contract), pts on the
/// 33 ms grid, then EOS.
struct FrameSrc {
    frames: Vec<Vec<u8>>,
    next: usize,
}

impl FrameSrc {
    fn new(frames: Vec<Vec<u8>>) -> Self {
        Self { frames, next: 0 }
    }
}

impl Element for FrameSrc {
    fn desc(&self) -> &'static ElementDesc {
        &FRAMESRC_DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.next >= self.frames.len() {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let bytes = &self.frames[self.next];
        assert!(buf.memory.capacity() >= bytes.len(), "encoded frame fits a pool slot");
        buf.memory.as_mut_full()[..bytes.len()].copy_from_slice(bytes);
        buf.memory.set_len(bytes.len());
        buf.pts = Timestamp::from_nanos(self.next as u64 * FRAME_NS);
        ctx.out(PadId(0)).push(buf);
        self.next += 1;
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static PIXFMTS: [ValueDesc; 1] = [ValueDesc::Id("i420")];
static RAW_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: "width", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "height", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "pixfmt", allowed: ConstraintDesc::Set(&PIXFMTS), preferred: None },
];
static RAW_OFFERS: [OfferDesc; 1] = [OfferDesc { family: "video/raw", fields: &RAW_FIELDS }];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &RAW_OFFERS,
    dynamic: false,
    validate: None,
}];

static RECORD_DESC: ElementDesc = ElementDesc {
    name: "planerecordsink",
    pads: &SINK_PADS,
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

#[derive(Default)]
struct Recorded {
    frames: Vec<(Timestamp, Vec<u8>)>,
    /// (width, height, pixfmt name) read from the announced format at FormatChange.
    format: Option<(i64, i64, String)>,
}

/// Records every received frame's pts + packed planes, and the announced format.
struct PlaneRecordSink {
    shared: Arc<Mutex<Recorded>>,
}

impl PlaneRecordSink {
    fn new() -> (Self, Arc<Mutex<Recorded>>) {
        let shared = Arc::new(Mutex::new(Recorded::default()));
        (Self { shared: Arc::clone(&shared) }, shared)
    }
}

impl Element for PlaneRecordSink {
    fn desc(&self) -> &'static ElementDesc {
        &RECORD_DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut rec = self.shared.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            rec.frames.push((buf.pts, buf.memory.data().to_vec()));
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if let Event::FormatChange(_) = event {
            // Read the installed format by name — the dynamic-caps consumer path.
            let f = ctx.negotiated(PadId(0)).expect("format installed").clone();
            let get = |name: &str| ctx.field_id(name).and_then(|id| f.get(id));
            let (Some(Value::Int(w)), Some(Value::Int(h)), Some(Value::Id(pf))) =
                (get("width"), get("height"), get("pixfmt"))
            else {
                panic!("announced format missing width/height/pixfmt");
            };
            let name = ctx.value_name(pf).expect("pixfmt name interned").to_string();
            self.shared.lock().unwrap().format = Some((w, h, name));
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// ---- fixtures ----------------------------------------------------------------

/// A deterministic I420 source picture; each frame index gets distinct content.
fn make_planes(idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let y = (0..w * h)
        .map(|i| ((i as u32).wrapping_mul(31).wrapping_add(idx * 97) >> 3) as u8)
        .collect();
    let u = (0..cw * ch).map(|i| (i as u32 + idx * 13) as u8).collect();
    let v = (0..cw * ch).map(|i| (i as u32 * 7 + idx * 29) as u8).collect();
    (y, u, v)
}

fn encode_frames(n: u32) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            let (y, u, v) = make_planes(i);
            let frame = I420Frame::packed(W, H, &y, &u, &v);
            encode_keyframe(&frame, &KeyframeParams::default()).expect("encode")
        })
        .collect()
}

/// Reference decode: the same frames through the library directly, packed Y|U|V.
fn reference_decode(frames: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut state = Vp8DecoderState::new();
    frames
        .iter()
        .map(|f| {
            let d = state.decode_frame(f).expect("reference decode");
            let mut packed = Vec::with_capacity(d.y.len() + d.u.len() + d.v.len());
            packed.extend_from_slice(&d.y);
            packed.extend_from_slice(&d.u);
            packed.extend_from_slice(&d.v);
            packed
        })
        .collect()
}

fn run_pipeline(frames: Vec<Vec<u8>>) -> (Pipeline, Arc<Mutex<Recorded>>) {
    let mut p = Pipeline::new();
    let (sink, recorded) = PlaneRecordSink::new();
    let src = p.add(FrameSrc::new(frames));
    let dec = p.add(Vp8Dec::new());
    let snk = p.add(sink);
    p.link((src, "src"), (dec, "sink")).expect("link src!dec");
    p.link((dec, "src"), (snk, "sink")).expect("link dec!sink");
    p.run().expect("run");
    (p, recorded)
}

// ---- tests -------------------------------------------------------------------

#[test]
fn decodes_bit_exactly_with_pts_passthrough_and_announced_format() {
    let frames = encode_frames(3);
    let reference = reference_decode(&frames);
    let (_p, recorded) = run_pipeline(frames);

    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 3, "every visible frame emitted");
    for (i, ((pts, planes), want)) in rec.frames.iter().zip(&reference).enumerate() {
        assert_eq!(*pts, Timestamp::from_nanos(i as u64 * FRAME_NS), "frame {i} pts");
        assert_eq!(planes, want, "frame {i} planes byte-identical to reference decode");
    }
    let (w, h, pf) = rec.format.clone().expect("format announced");
    assert_eq!((w, h), (W as i64, H as i64));
    assert_eq!(pf, "i420");
}

#[test]
fn corrupt_frame_warns_drops_and_recovers_at_next_keyframe() {
    let mut frames = encode_frames(2);
    // Splice garbage between the two keyframes.
    frames.insert(1, vec![0xDE; 512]);
    let reference = reference_decode(&[frames[0].clone(), frames[2].clone()]);
    let (p, recorded) = run_pipeline(frames);

    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 2, "good frames decoded, corrupt one dropped");
    assert_eq!(rec.frames[0].1, reference[0]);
    assert_eq!(rec.frames[1].1, reference[1]);

    let mut warnings = 0;
    while let Some(msg) = p.bus().try_recv() {
        if matches!(msg, BusMessage::Warning { .. }) {
            warnings += 1;
        }
    }
    assert!(warnings >= 1, "the dropped frame surfaced as a bus warning");
}

#[test]
fn garbage_only_stream_emits_nothing_and_never_panics() {
    let garbage: Vec<Vec<u8>> = (0..4).map(|i| vec![i as u8; 64 + i * 33]).collect();
    let (_p, recorded) = run_pipeline(garbage);
    assert!(recorded.lock().unwrap().frames.is_empty());
}
