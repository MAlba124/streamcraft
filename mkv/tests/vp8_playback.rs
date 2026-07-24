//! End-to-end **VP8-in-MKV playback path** (spec: `mkv/spec/MATROSKA.md`; RFC 9559 §12 codec
//! mappings; Milestone applications §5).
//!
//! The load-bearing property: a `V_VP8` track muxed by [`MatroskaWriter`] demuxes and decodes
//! back **bit-exact**. A few small frames are encoded with `oxideav-vp8`, decoded once directly
//! as the reference, then muxed into an in-memory MKV and pushed through a real pipeline
//! `ByteSrc ! MkvDemux ! Vp8Dec ! PlaneRecordSink` — the recorded planes must equal the
//! reference decode and the presentation timestamps must survive the container round-trip.
//!
//! WebM video stores frames raw (one Block = one VP8 frame, pts already computed by the
//! muxer), so the demuxer is naming-only here: it announces family `vp8` on the dynamic src
//! pad so linking `mkvdemux.src_track1 ! vp8dec.sink` negotiates, and forwards each frame
//! verbatim. The test element shapes mirror `vp8/tests/vp8dec.rs`; the demux-preroll wiring
//! mirrors `mkv/tests/demux_roundtrip.rs`.

use std::sync::{Arc, Mutex};

use oxideav_vp8::encoder::{encode_keyframe, I420Frame, KeyframeParams};
use oxideav_vp8::Vp8DecoderState;

use sc_mkv::ebml::id;
use sc_mkv::{MatroskaWriter, MkvDemux, TrackConfig};
use sc_vp8::Vp8Dec;

use streamcraft_core::batch::Inputs;
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

const W: u32 = 128;
const H: u32 = 96;
/// One frame per 33 ms — the container timestamp grid the muxer stamps and the demuxer
/// reproduces (scaled through the default 1 ms TimestampScale, which 33 ms is a whole multiple
/// of, so the round-trip is exact).
const FRAME_NS: u64 = 33_000_000;

// ---- byte source: streams the muxed MKV in chunks, then EOS -----------------------------

static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &SRC_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "bytesrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct ByteSrc {
    data: Vec<u8>,
    chunk: usize,
    pos: usize,
}

impl Element for ByteSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.pos = 0;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.pos >= self.data.len() {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let cap = buf.memory.capacity();
        let n = self.chunk.min(cap).min(self.data.len() - self.pos);
        buf.memory.as_mut_full()[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        buf.memory.set_len(n);
        ctx.out(PadId(0)).push(buf);
        self.pos += n;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// ---- plane-recording sink (copied from vp8/tests/vp8dec.rs) ------------------------------

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
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

#[derive(Default)]
struct Recorded {
    frames: Vec<(Timestamp, Vec<u8>)>,
    /// (width, height, pixfmt name) from the announced format at FormatChange.
    format: Option<(i64, i64, String)>,
}

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

// ---- fixtures ----------------------------------------------------------------------------

/// A deterministic I420 source picture; each frame index gets distinct content (identical to
/// the fixture in `vp8/tests/vp8dec.rs` so the two crates encode the same pictures).
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

/// Mux `frames` into an in-memory MKV as a single `V_VP8` track, one Block per frame stamped on
/// the 33 ms grid. Uses the N-track `MatroskaWriter` directly (the writer is the mux library;
/// `MkvMux` wraps it for FLAC — a V_VP8 mux element is naming-only over the same writer).
fn mux_vp8(frames: &[Vec<u8>]) -> Vec<u8> {
    let track = TrackConfig::vp8(1, W, H);
    let mut w = MatroskaWriter::new(vec![track]);
    let mut out = Vec::new();
    w.write_header(&mut out).unwrap();
    for (i, f) in frames.iter().enumerate() {
        // Every VP8 keyframe is independently decodable → keyframe = true.
        w.write_frame(&mut out, 1, i as u64 * FRAME_NS, f, true).unwrap();
    }
    w.finalize(&mut out);
    out
}

/// The header prefix `MkvDemux::new` needs: everything up to (not including) the first Cluster
/// — the EBML Header + Segment Info + Tracks (mirrors `demux_roundtrip.rs::header_prefix`).
fn header_prefix(stream: &[u8]) -> Vec<u8> {
    let cluster = stream
        .windows(4)
        .position(|w| w == id::CLUSTER)
        .expect("stream has at least one Cluster");
    stream[..cluster].to_vec()
}

/// Run `bytesrc(mkv) ! mkvdemux(header) ! vp8dec ! planerecordsink` and return the recorded
/// planes/format. The demuxer adds its src pad during preroll; the decoder + sink are added and
/// linked after (the demux-preroll wiring from `demux_roundtrip.rs`).
fn run_playback(stream: Vec<u8>, chunk: usize) -> Arc<Mutex<Recorded>> {
    let header = header_prefix(&stream);
    let mut p = Pipeline::new();
    let src = p.add(ByteSrc { data: stream, chunk, pos: 0 });
    let demux = p.add(MkvDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");

    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 1, "one V_VP8 track → one src pad");

    let (sink, recorded) = PlaneRecordSink::new();
    let dec = p.add(Vp8Dec::new());
    let snk = p.add(sink);
    let ap = &added[0];
    p.link((ap.element, &ap.name), (dec, "sink")).expect("demux pad -> vp8dec");
    p.link((dec, "src"), (snk, "sink")).expect("vp8dec -> sink");

    p.run().expect("run");
    recorded
}

// ---- tests -------------------------------------------------------------------------------

#[test]
fn vp8_in_mkv_decodes_bit_exact_with_pts() {
    let frames = encode_frames(3);
    let reference = reference_decode(&frames);
    let stream = mux_vp8(&frames);

    // Chunk small so MKV elements straddle input boundaries (exercises the demuxer's
    // cross-boundary buffering on the way to the decoder).
    let recorded = run_playback(stream, 500);
    let rec = recorded.lock().unwrap();

    assert_eq!(rec.frames.len(), 3, "every muxed VP8 frame decoded back");
    for (i, ((pts, planes), want)) in rec.frames.iter().zip(&reference).enumerate() {
        assert_eq!(
            *pts,
            Timestamp::from_nanos(i as u64 * FRAME_NS),
            "frame {i} pts survives the MKV round-trip"
        );
        assert_eq!(planes, want, "frame {i} planes bit-exact vs direct oxideav decode");
    }

    // The decoder announces authoritative dims from the VP8 bitstream (128x96, I420).
    let (w, h, pf) = rec.format.clone().expect("format announced downstream of the decoder");
    assert_eq!((w, h), (W as i64, H as i64));
    assert_eq!(pf, "i420");
}

#[test]
fn vp8_in_mkv_whole_stream_in_one_chunk() {
    // Same content, fed as one big chunk — the plane-exact property must not depend on chunking.
    let frames = encode_frames(2);
    let reference = reference_decode(&frames);
    let stream = mux_vp8(&frames);

    let recorded = run_playback(stream.clone(), stream.len().max(1));
    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 2);
    assert_eq!(rec.frames[0].1, reference[0]);
    assert_eq!(rec.frames[1].1, reference[1]);
}
