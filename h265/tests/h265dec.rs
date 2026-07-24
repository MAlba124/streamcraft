//! `H265Dec` element integration (spec: Milestone applications §5 — the decoder half).
//!
//! The committed golden fixtures are TINY real HEVC Annex B streams produced once with
//! system **x265** (via ffmpeg) during development and stripped to VPS/SPS/PPS + VCL
//! NALs (a few hundred bytes each — see the generation recipe in the fixture doc). Each
//! is split into **one access unit per buffer** (the demuxer contract) and pushed
//! through a pipeline `FrameSrc ! H265Dec ! PlaneRecordSink`. The output planes must be
//! byte-identical to a direct reference decode of the same stream through the library
//! ([`decode_annexb_sequence`]), and the announced `video/raw` format must carry the
//! real dimensions. A corrupt access unit spliced mid-stream must degrade per-buffer
//! (bus warning, next keyframe recovers), never panic. Garbage-only input emits nothing
//! and never panics. Finally an ffmpeg-oracle test cross-checks the decode against
//! ffmpeg itself, skipping cleanly when ffmpeg is not installed.

use std::sync::{Arc, Mutex};

use oxideav_h265::decode_annexb_sequence;

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

use sc_h265::H265Dec;

/// One frame per 33 ms — a stand-in container timestamp grid.
const FRAME_NS: u64 = 33_000_000;

// The committed golden fixtures (real x265 Annex B, SEI-stripped).
const TINY_I: &[u8] = include_bytes!("fixtures/tiny_i.hevc"); // 16x16, one IDR
const IP: &[u8] = include_bytes!("fixtures/ip.hevc"); // 16x16, IDR + P

// ---- Annex B access-unit splitter (the demuxer's job, done in-test) ----------

/// Split an Annex B byte stream into access units: a run of NAL units, where a new
/// AU begins at each VCL NAL whose `first_slice_segment_in_pic_flag` (top bit of the
/// first RBSP byte, after the 2-byte NAL header) is set. Non-VCL NALs (VPS/SPS/PPS/
/// SEI) attach to the AU that follows them. This mirrors the rule the library's
/// `SequenceDecoder` applies internally, so per-AU pushes and a whole-stream push
/// decode identically.
fn split_access_units(data: &[u8]) -> Vec<Vec<u8>> {
    // Collect (offset, start_code_len) of every NAL start code.
    let mut starts: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                starts.push((i, 3));
                i += 3;
                continue;
            }
            if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                starts.push((i, 4));
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    let mut aus: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    // Leading non-VCL NALs (VPS/SPS/PPS) belong to the AU of the VCL that follows, so
    // only start a *new* AU at a first-slice VCL when the current AU already holds one.
    let mut cur_has_vcl = false;
    for (n, &(off, sc)) in starts.iter().enumerate() {
        let end = starts.get(n + 1).map(|s| s.0).unwrap_or(data.len());
        let nal = &data[off..end];
        let nal_type = (nal[sc] >> 1) & 0x3f;
        let is_vcl = nal_type <= 31;
        let first_in_pic = is_vcl && (nal[sc + 2] & 0x80) != 0;
        if first_in_pic && cur_has_vcl {
            aus.push(std::mem::take(&mut cur));
            cur_has_vcl = false;
        }
        cur.extend_from_slice(nal);
        cur_has_vcl |= is_vcl;
    }
    if !cur.is_empty() {
        aus.push(cur);
    }
    aus
}

// ---- test elements -----------------------------------------------------------

static H265_OFFERS: [OfferDesc; 1] = [OfferDesc::any("h265/annexb")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &H265_OFFERS,
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

/// Emits pre-split HEVC access units, one per buffer (the demuxer contract), pts on
/// the 33 ms grid, then EOS.
struct FrameSrc {
    aus: Vec<Vec<u8>>,
    next: usize,
}

impl FrameSrc {
    fn new(aus: Vec<Vec<u8>>) -> Self {
        Self { aus, next: 0 }
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
        if self.next >= self.aus.len() {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let bytes = &self.aus[self.next];
        assert!(buf.memory.capacity() >= bytes.len(), "access unit fits a pool slot");
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

// ---- reference decode --------------------------------------------------------

/// Reference decode: the whole stream through the library directly, packed Y|Cb|Cr,
/// output-order, output frames only.
fn reference_decode(stream: &[u8]) -> Vec<Vec<u8>> {
    decode_annexb_sequence(stream)
        .expect("reference decode")
        .into_iter()
        .filter(|f| f.output)
        .map(|f| f.picture.to_planar_u8().expect("8-bit i420 reference"))
        .collect()
}

fn run_pipeline(aus: Vec<Vec<u8>>) -> (Pipeline, Arc<Mutex<Recorded>>) {
    let mut p = Pipeline::new();
    let (sink, recorded) = PlaneRecordSink::new();
    let src = p.add(FrameSrc::new(aus));
    let dec = p.add(H265Dec::new());
    let snk = p.add(sink);
    p.link((src, "src"), (dec, "sink")).expect("link src!dec");
    p.link((dec, "src"), (snk, "sink")).expect("link dec!sink");
    p.run().expect("run");
    (p, recorded)
}

// ---- tests -------------------------------------------------------------------

#[test]
fn intra_decodes_bit_exactly_with_announced_format() {
    let aus = split_access_units(TINY_I);
    assert_eq!(aus.len(), 1, "the tiny fixture is one access unit");
    let reference = reference_decode(TINY_I);
    assert_eq!(reference.len(), 1, "one output frame");

    let (_p, recorded) = run_pipeline(aus);
    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 1, "the frame was emitted");
    assert_eq!(
        rec.frames[0].1, reference[0],
        "planes byte-identical to the library reference decode"
    );

    let (w, h, pf) = rec.format.clone().expect("format announced");
    assert_eq!((w, h), (16, 16));
    assert_eq!(pf, "i420");
}

#[test]
fn inter_i_then_p_decodes_bit_exactly_with_pts() {
    let aus = split_access_units(IP);
    assert_eq!(aus.len(), 2, "IDR + P are two access units");
    let reference = reference_decode(IP);
    assert_eq!(reference.len(), 2, "two output frames");

    let (_p, recorded) = run_pipeline(aus);
    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 2, "both frames emitted");
    for (i, ((pts, planes), want)) in rec.frames.iter().zip(&reference).enumerate() {
        // No B-frames here, so decode order == output order and the ascending pts
        // re-attach is an identity: frame i keeps pts = i * FRAME_NS.
        assert_eq!(*pts, Timestamp::from_nanos(i as u64 * FRAME_NS), "frame {i} pts");
        assert_eq!(planes, want, "frame {i} planes byte-identical to reference");
    }
}

#[test]
fn corrupt_access_unit_warns_drops_and_recovers() {
    // Two independent IDR fixtures back to back, with garbage spliced between them:
    // the garbage AU must be dropped (bus warning), and the decoder must resync at
    // the second stream's parameter sets + IDR.
    let mut aus = split_access_units(TINY_I);
    aus.push(vec![0, 0, 1, 0x26, 0x01, 0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11]); // bogus IDR-ish
    aus.extend(split_access_units(TINY_I));

    let (p, recorded) = run_pipeline(aus);
    let rec = recorded.lock().unwrap();
    // The two good IDRs decode; the garbage one is dropped.
    assert_eq!(rec.frames.len(), 2, "both good frames decoded, corrupt one dropped");
    let reference = reference_decode(TINY_I);
    assert_eq!(rec.frames[0].1, reference[0]);
    assert_eq!(rec.frames[1].1, reference[0]);

    let mut warnings = 0;
    while let Some(msg) = p.bus().try_recv() {
        if matches!(msg, BusMessage::Warning { .. }) {
            warnings += 1;
        }
    }
    assert!(warnings >= 1, "the dropped access unit surfaced as a bus warning");
}

#[test]
fn garbage_only_stream_emits_nothing_and_never_panics() {
    let garbage: Vec<Vec<u8>> = (0..4u8)
        .map(|i| {
            let mut v = vec![0, 0, 1];
            v.extend((0..(48 + i * 21)).map(|j| j.wrapping_mul(i).wrapping_add(i)));
            v
        })
        .collect();
    let (_p, recorded) = run_pipeline(garbage);
    assert!(recorded.lock().unwrap().frames.is_empty());
}

// ---- ffmpeg oracle -----------------------------------------------------------

/// Decode a fixture with system ffmpeg to raw i420 (returns None if ffmpeg is absent
/// or the decode fails — the test then skips).
fn ffmpeg_decode_i420(hevc: &[u8]) -> Option<Vec<u8>> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    // ffmpeg reads the Annex B stream from stdin, writes raw yuv420p to stdout.
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error",
            "-f", "hevc", "-i", "pipe:0",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?; // ffmpeg not installed -> skip
    child.stdin.take()?.write_all(hevc).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() || out.stdout.is_empty() {
        return None;
    }
    Some(out.stdout)
}

#[test]
fn matches_ffmpeg_oracle() {
    let Some(ff) = ffmpeg_decode_i420(TINY_I) else {
        eprintln!("skipping ffmpeg oracle: ffmpeg unavailable or decode failed");
        return;
    };
    let aus = split_access_units(TINY_I);
    let (_p, recorded) = run_pipeline(aus);
    let rec = recorded.lock().unwrap();
    let ours: Vec<u8> = rec.frames.iter().flat_map(|(_, p)| p.clone()).collect();
    assert_eq!(ours.len(), ff.len(), "decoded size matches ffmpeg");
    assert_eq!(ours, ff, "H265Dec output byte-exact against the ffmpeg oracle");
}
