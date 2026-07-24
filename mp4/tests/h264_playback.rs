//! End-to-end **H.264-in-MP4 playback path** (spec: `spec/NOTES.md`; Milestone applications
//! §5). Mirrors `mkv/tests/vp8_playback.rs`, but the source is a real committed MP4 fixture
//! demuxed through [`Mp4Demux`] and decoded through `sc-h264`'s [`H264Dec`].
//!
//! The load-bearing property: the committed baseline H.264 MP4 (`tiny_h264.mp4`, 15 frames,
//! 128x96, no B-frames) demuxes — the demuxer reframes the length-prefixed NALs to Annex B
//! and prefixes the `avcC` parameter sets — and **decodes** all the way through
//! `Mp4Demux ! H264Dec ! PlaneRecordSink` to real I420 pictures: the right frame count, the
//! announced dimensions from the bitstream, non-empty packed planes, and monotone pts.
//!
//! The fixture is Baseline (no reorder), so decode order == display order and the decoder's
//! decode-order pts equals the container's display pts — no reorder caveat here. Ragged
//! chunking is swept to exercise the demuxer's cross-chunk buffering under a real decoder.

use std::sync::{Arc, Mutex};

use sc_h264::H264Dec;
use sc_mp4::Mp4Demux;

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

const W: i64 = 128;
const H: i64 = 96;

// ---- byte source: streams the MP4 file in chunks, then EOS -------------------------------

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
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return Ok(Flow::Ok),
        };
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

// ---- plane-recording sink (mirrors h264/tests/h264dec.rs) --------------------------------

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
    /// (pts_ns, plane_len) per decoded frame, in output order.
    frames: Vec<(Option<u64>, usize)>,
    /// (width, height, pixfmt) from the announced format.
    format: Option<(i64, i64, String)>,
}

struct PlaneRecordSink {
    shared: Arc<Mutex<Recorded>>,
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
            rec.frames.push((buf.pts.nanos(), buf.memory.data().len()));
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if let Event::FormatChange(_) = event {
            let f = ctx.negotiated(PadId(0)).expect("format installed").clone();
            let get = |name: &str| ctx.field_id(name).and_then(|id| f.get(id));
            if let (Some(Value::Int(w)), Some(Value::Int(h)), Some(Value::Id(pf))) =
                (get("width"), get("height"), get("pixfmt"))
            {
                let name = ctx.value_name(pf).map(|s| s.to_string()).unwrap_or_default();
                self.shared.lock().unwrap().format = Some((w, h, name));
            }
        }
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// ---- helpers -----------------------------------------------------------------------------

fn fixture_bytes(name: &str) -> Vec<u8> {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read fixture {name}: {e}"))
}

fn head_through_moov(file: &[u8]) -> Vec<u8> {
    let mut at = 0usize;
    while at + 8 <= file.len() {
        let size = u32::from_be_bytes([file[at], file[at + 1], file[at + 2], file[at + 3]]) as usize;
        if &file[at + 4..at + 8] == b"mdat" {
            return file[..at].to_vec();
        }
        let advance = if size == 0 { file.len() - at } else if size == 1 {
            u64::from_be_bytes(file[at + 8..at + 16].try_into().unwrap()) as usize
        } else {
            size
        };
        if advance == 0 {
            break;
        }
        at += advance;
    }
    file.to_vec()
}

/// Run `bytesrc(file, chunk) ! mp4demux ! h264dec ! planerecordsink` and return the recording.
fn run_playback(chunk: usize) -> Recorded {
    let file = fixture_bytes("tiny_h264.mp4");
    let head = head_through_moov(&file);

    let mut p = Pipeline::new();
    let src = p.add(ByteSrc { data: file, chunk, pos: 0 });
    let demux = p.add(Mp4Demux::new(head));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");

    // Preroll adds the demuxer's src pad(s). The fixture has a single video track.
    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 1, "one video track");

    let dec = p.add(H264Dec::new());
    let shared = Arc::new(Mutex::new(Recorded::default()));
    let sink = p.add(PlaneRecordSink { shared: Arc::clone(&shared) });
    p.link((added[0].element, &added[0].name), (dec, "sink")).expect("demux pad -> h264dec");
    p.link((dec, "src"), (sink, "sink")).expect("h264dec -> sink");

    p.run().expect("run playback");
    match Arc::try_unwrap(shared) {
        Ok(m) => m.into_inner().unwrap(),
        Err(_) => panic!("sink Arc still shared after run"),
    }
}

// ---- tests -------------------------------------------------------------------------------

/// The baseline H.264 MP4 decodes end-to-end: 15 frames out, each a full 128x96 packed I420
/// picture, the announced format carries the coded dimensions, and pts are monotone.
#[test]
fn h264_in_mp4_decodes_end_to_end() {
    let rec = run_playback(4096);

    // The decoder announces the bitstream dimensions (from the SPS the demuxer forwarded).
    let (w, h, pixfmt) = rec.format.expect("decoder announced a video/raw format");
    assert_eq!((w, h), (W, H), "announced dims == the coded 128x96");
    assert_eq!(pixfmt, "i420", "packed I420 output");

    // 15 coded frames → 15 decoded pictures.
    assert_eq!(rec.frames.len(), 15, "all 15 frames decode");

    // Each picture is a full packed I420 buffer: 128*96 + 2*(64*48) = 12288 + 6144 = 18432.
    let expect_bytes = (W * H + 2 * (W / 2) * (H / 2)) as usize;
    for (i, (_pts, len)) in rec.frames.iter().enumerate() {
        assert_eq!(*len, expect_bytes, "frame {i} is a full packed I420 picture");
    }

    // Every decoded frame carries a pts, monotone non-decreasing (baseline: no reorder).
    let ptss: Vec<u64> = rec.frames.iter().map(|(p, _)| p.expect("frame carries a pts")).collect();
    for w in ptss.windows(2) {
        assert!(w[1] >= w[0], "decoded pts monotone: {} !>= {}", w[1], w[0]);
    }
    assert_eq!(ptss[0], 0, "first frame at pts 0");
}

/// Ragged chunking under a real decoder: the demuxer's cross-chunk buffering feeds the
/// decoder correctly for every chunk size — the decoded frame count is invariant.
#[test]
fn playback_survives_ragged_chunking() {
    let reference = run_playback(1 << 20);
    let ref_ptss: Vec<Option<u64>> = reference.frames.iter().map(|(p, _)| *p).collect();
    for &chunk in &[1usize, 3, 17, 251, 1024, 4096] {
        let rec = run_playback(chunk);
        assert_eq!(rec.frames.len(), reference.frames.len(), "chunk {chunk}: same frame count");
        let ptss: Vec<Option<u64>> = rec.frames.iter().map(|(p, _)| *p).collect();
        assert_eq!(ptss, ref_ptss, "chunk {chunk}: identical pts sequence");
    }
}
