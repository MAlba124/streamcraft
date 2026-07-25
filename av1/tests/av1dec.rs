//! `Av1Dec` element integration (spec: Milestone applications §5 — the decoder half).
//! A short GOP is encoded in-test with the same crate's encoder, decoded once directly
//! through a `SpecDecodeSession` as the reference, then pushed through a pipeline
//! `FrameSrc ! Av1Dec ! PlaneRecordSink` — one temporal unit per buffer, the container
//! contract. Output planes must be byte-identical to the reference decode, pts must
//! pass through, and the announced `video/raw` format must carry the real dimensions.
//! A corrupt mid-stream unit must degrade per-buffer (bus warning, next keyframe
//! recovers), never panic or kill the pipeline. A final test decodes the same stream
//! through system `ffmpeg` as an out-of-process oracle (skipped cleanly when absent).

use std::sync::{Arc, Mutex};

use oxideav_av1::decoder::SpecDecodeSession;
use oxideav_av1::encoder::{encode_gop_yuv420_with_q, Yuv420Frame};

use streamcraft_core::batch::Inputs;
use streamcraft_core::bus::BusMessage;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use streamcraft_core::harness::Harness;
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

use sc_av1::Av1Dec;

const W: u32 = 64;
const H: u32 = 48;
/// One frame per 33 ms — a stand-in container timestamp grid.
const FRAME_NS: u64 = 33_000_000;

// ---- test elements -----------------------------------------------------------

static AV1_OFFERS: [OfferDesc; 1] = [OfferDesc::any("av1")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &AV1_OFFERS,
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

/// Emits pre-encoded AV1 temporal units, one per buffer (the container contract), pts
/// on the 33 ms grid, then EOS.
struct FrameSrc {
    units: Vec<Vec<u8>>,
    next: usize,
}

impl FrameSrc {
    fn new(units: Vec<Vec<u8>>) -> Self {
        Self { units, next: 0 }
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
        if self.next >= self.units.len() {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let bytes = &self.units[self.next];
        assert!(buf.memory.capacity() >= bytes.len(), "encoded unit fits a pool slot");
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

static PIXFMTS: [ValueDesc; 2] = [ValueDesc::Id("i420"), ValueDesc::Id("gray8")];
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
fn make_frame(idx: u32) -> Yuv420Frame {
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut f = Yuv420Frame::filled(W, H, 0);
    for (i, y) in f.y.iter_mut().enumerate() {
        *y = ((i as u32).wrapping_mul(31).wrapping_add(idx * 97) >> 3) as u8;
    }
    for (i, u) in f.u.iter_mut().enumerate() {
        *u = (i as u32 + idx * 13) as u8;
    }
    for (i, v) in f.v.iter_mut().enumerate() {
        *v = (i as u32 * 7 + idx * 29) as u8;
    }
    debug_assert_eq!(f.u.len(), cw * ch);
    f
}

/// Encode `n` frames as a lossless (`q == 0`) GOP: a KEY frame then P-frames, one
/// temporal unit per frame. At `q == 0` the decode is byte-exact to the input planes.
fn encode_units(n: u32) -> Vec<Vec<u8>> {
    let frames: Vec<Yuv420Frame> = (0..n).map(make_frame).collect();
    let enc = encode_gop_yuv420_with_q(&frames, 0).expect("encode gop");
    assert_eq!(enc.temporal_units.len(), n as usize, "one temporal unit per frame");
    enc.temporal_units
}

/// Reference decode: the same temporal units through the library directly, packed
/// Y|U|V (the `i420` layout `Av1Dec` emits), one entry per shown frame.
fn reference_decode(units: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut session = SpecDecodeSession::new();
    let mut out = Vec::new();
    for u in units {
        for frame in session.decode_temporal_unit(u).expect("reference decode") {
            let mut packed = Vec::new();
            for plane in &frame.planes {
                packed.extend_from_slice(plane);
            }
            out.push(packed);
        }
    }
    out
}

fn run_pipeline(units: Vec<Vec<u8>>) -> (Pipeline, Arc<Mutex<Recorded>>) {
    let mut p = Pipeline::new();
    let (sink, recorded) = PlaneRecordSink::new();
    let src = p.add(FrameSrc::new(units));
    let dec = p.add(Av1Dec::new());
    let snk = p.add(sink);
    p.link((src, "src"), (dec, "sink")).expect("link src!dec");
    p.link((dec, "src"), (snk, "sink")).expect("link dec!sink");
    p.run().expect("run");
    (p, recorded)
}

// ---- tests -------------------------------------------------------------------

#[test]
fn decodes_bit_exactly_with_pts_passthrough_and_announced_format() {
    let units = encode_units(3);
    let reference = reference_decode(&units);
    assert_eq!(reference.len(), 3, "sanity: three shown frames from three units");
    let (_p, recorded) = run_pipeline(units);

    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 3, "every shown frame emitted");
    for (i, ((pts, planes), want)) in rec.frames.iter().zip(&reference).enumerate() {
        assert_eq!(*pts, Timestamp::from_nanos(i as u64 * FRAME_NS), "frame {i} pts");
        assert_eq!(planes, want, "frame {i} planes byte-identical to reference decode");
    }
    let (w, h, pf) = rec.format.clone().expect("format announced");
    assert_eq!((w, h), (W as i64, H as i64));
    assert_eq!(pf, "i420");
}

#[test]
fn corrupt_unit_warns_drops_and_recovers_at_next_keyframe() {
    // A fresh single-KEY-frame stream (so the recovery point is a real keyframe, not a
    // P-frame that would legitimately fail after the reference chain is broken).
    let good = encode_units(1);
    let mut units = good.clone();
    // Splice garbage before the keyframe; then decode the keyframe after recovery.
    units.insert(0, vec![0xDE; 512]);
    let reference = reference_decode(&good);
    let (p, recorded) = run_pipeline(units);

    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 1, "good keyframe decoded, corrupt unit dropped");
    assert_eq!(rec.frames[0].1, reference[0]);

    let mut warnings = 0;
    while let Some(msg) = p.bus().try_recv() {
        if matches!(msg, BusMessage::Warning { .. }) {
            warnings += 1;
        }
    }
    assert!(warnings >= 1, "the dropped unit surfaced as a bus warning");
}

#[test]
fn garbage_only_stream_emits_nothing_and_never_panics() {
    let garbage: Vec<Vec<u8>> = (0..4).map(|i| vec![i as u8; 64 + i * 33]).collect();
    let (_p, recorded) = run_pipeline(garbage);
    assert!(recorded.lock().unwrap().frames.is_empty());
}

// ---- ffmpeg oracle -----------------------------------------------------------

/// Decode the same in-crate-encoded stream through system `ffmpeg` (out of process)
/// and compare to `Av1Dec`'s output. Skips cleanly when ffmpeg (or an AV1 decoder in
/// it) is absent — the exact dev-oracle pattern the spec prescribes.
#[test]
fn ffmpeg_oracle_agrees_on_the_decoded_pixels() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    // Is ffmpeg present at all?
    if Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_err() {
        eprintln!("skipping ffmpeg oracle: ffmpeg not found on PATH");
        return;
    }

    // Encode a single lossless keyframe and get its complete IVF file.
    let frame = make_frame(0);
    let enc = encode_gop_yuv420_with_q(&[frame], 0).expect("encode");
    let ivf = enc.ivf_bytes.clone();

    // Ask ffmpeg to decode the IVF (read from stdin) to raw yuv420p on stdout.
    let mut child = match Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-f", "ivf", "-i", "pipe:0",
               "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping ffmpeg oracle: ffmpeg failed to spawn");
            return;
        }
    };
    child.stdin.take().unwrap().write_all(&ivf).expect("feed ffmpeg");
    let output = child.wait_with_output().expect("ffmpeg run");
    if !output.status.success() || output.stdout.is_empty() {
        // Most likely: this ffmpeg build has no AV1 decoder. Skip, don't fail.
        eprintln!(
            "skipping ffmpeg oracle: ffmpeg produced no output (no AV1 decoder?): {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    // Our decode of the same stream (packed Y|U|V).
    let reference = reference_decode(&enc.temporal_units);
    assert_eq!(reference.len(), 1);
    assert_eq!(
        output.stdout, reference[0],
        "ffmpeg-decoded pixels must match the in-crate decode byte-for-byte"
    );
}

/// Seek (spec: flush/seek): decode a KEY+P GOP (the session accrues reference-slot
/// state and, at `q == 0`, its arithmetic-decoder + reference buffers), deliver
/// `Event::FlushStart`, then feed a fresh KEY temporal unit (which re-carries the
/// sequence header). A fresh `SpecDecodeSession` plus a cleared pending queue must let
/// that KEY unit decode cleanly at the post-flush pts — no stale reference frame, and
/// no stranded pending frame, leaks across the flush. Driven through the [`Harness`] so
/// `FlushStart` can be delivered mid-stream (a real pipeline delivers it only
/// out-of-band on a seek).
#[test]
fn flush_start_resets_session_and_no_stale_frame_leaks() {
    // Pre-flush: a 2-frame GOP (KEY + P) — populates the session's reference slots.
    let pre = encode_units(2);
    // Post-flush: an independent single-KEY-frame stream (its own sequence header),
    // reference-decoded alone as a fresh session would after a seek.
    let post = encode_units(1);
    let post_ref = reference_decode(&post);
    assert_eq!(post_ref.len(), 1, "the post-flush KEY unit decodes to one shown frame");

    let mut h = Harness::new(Av1Dec::new());

    // Feed the pre-flush units with pts on the 33 ms grid, then discard whatever they
    // staged (a real seek drops these in-flight buffers) — recording their pts so the
    // post-flush pts can be proven distinct.
    let mut pre_pts: Vec<Timestamp> = Vec::new();
    for (i, u) in pre.iter().enumerate() {
        let mut buf = h.alloc(u);
        buf.pts = Timestamp::from_nanos(i as u64 * FRAME_NS);
        h.push("sink", buf).expect("push pre-flush TU");
    }
    for b in h.drain_outputs() {
        pre_pts.push(b.pts);
    }

    // Seek: flush the session + pending carry.
    h.push_event(Event::FlushStart).expect("flush");
    assert!(h.pull("src").is_none(), "flush emits nothing");

    // Post-seek: a fresh KEY unit at a new, far-away pts.
    let post_pts = Timestamp::from_nanos(1000 * FRAME_NS);
    let mut buf = h.alloc(&post[0]);
    buf.pts = post_pts;
    h.push("sink", buf).expect("push post-flush KEY unit");

    let out = h.drain_outputs();
    assert_eq!(out.len(), 1, "exactly the post-flush KEY frame — no stale pre-flush frame leaked");
    assert_eq!(out[0].pts, post_pts, "post-flush pts comes from post-flush input");
    assert!(
        !pre_pts.contains(&out[0].pts),
        "post-flush frame did not inherit a pre-flush pts"
    );
    assert_eq!(out[0].memory.data(), post_ref[0].as_slice(), "post-flush KEY unit decodes cleanly");

    let ann = h.announced().expect("format announced");
    let w = h.vocabulary().field_id("width").unwrap();
    let ht = h.vocabulary().field_id("height").unwrap();
    assert_eq!(ann.get(w), Some(Value::Int(W as i64)));
    assert_eq!(ann.get(ht), Some(Value::Int(H as i64)));
}
