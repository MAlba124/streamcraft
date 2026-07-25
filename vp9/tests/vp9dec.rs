//! `Vp9Dec` element integration (spec: Milestone applications §5 — the decoder half).
//!
//! The backing `oxideav-vp9` 0.0.12 has no encoder, so the reference frames are a tiny
//! **committed golden** — a real 64x48 VP9 keyframe generated once with system ffmpeg
//! (`libvpx-vp9`, one key frame) and checked in under `tests/fixtures/`. The golden is
//! decoded once directly through the library as the reference, then pushed through a
//! pipeline `FrameSrc ! Vp9Dec ! PlaneRecordSink` — output planes must be
//! byte-identical to the reference decode, pts must pass through, and the announced
//! `video/raw` format must carry the real dimensions + pixel format.
//!
//! A corrupt frame must degrade per-buffer (bus warning, next keyframe recovers), a
//! garbage-only stream must emit nothing and never panic, an Annex B **superframe**
//! packet must split and decode its enclosed frame, and an ffmpeg-oracle test
//! cross-checks a runtime-generated stream (skipping cleanly when ffmpeg is absent).

use std::process::Command;
use std::sync::{Arc, Mutex};

use oxideav_vp9::decode_intra_frame;

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

use sc_vp9::Vp9Dec;

/// A real 64x48 VP9 keyframe (8-bit 4:2:0), one frame, generated with
/// `ffmpeg -f lavfi -i testsrc2=64x48 -frames:v 1 -c:v libvpx-vp9 -g 1 …` and stored
/// as its raw coded payload (IVF container stripped). ~1 KB.
const KEYFRAME: &[u8] = include_bytes!("fixtures/keyframe_64x48.vp9");

const W: i64 = 64;
const H: i64 = 48;
/// One frame per 33 ms — a stand-in container timestamp grid.
const FRAME_NS: u64 = 33_000_000;

// ---- test elements -----------------------------------------------------------

static VP9_OFFERS: [OfferDesc; 1] = [OfferDesc::any("vp9")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &VP9_OFFERS,
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

/// Emits VP9 coded chunks, one per buffer (the demuxer contract), pts on the 33 ms
/// grid, then EOS.
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
        assert!(buf.memory.capacity() >= bytes.len(), "coded chunk fits a pool slot");
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

// The record sink accepts any `video/raw` — the pixfmt set matches sc-vp9's src offer.
static PIXFMTS: [ValueDesc; 12] = [
    ValueDesc::Id("i420"),
    ValueDesc::Id("i422"),
    ValueDesc::Id("i440"),
    ValueDesc::Id("i444"),
    ValueDesc::Id("i420_10"),
    ValueDesc::Id("i422_10"),
    ValueDesc::Id("i440_10"),
    ValueDesc::Id("i444_10"),
    ValueDesc::Id("i420_12"),
    ValueDesc::Id("i422_12"),
    ValueDesc::Id("i440_12"),
    ValueDesc::Id("i444_12"),
];
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

// ---- helpers -----------------------------------------------------------------

/// Reference decode: the same frames through the library directly, packed Y|U|V. A
/// frame the library cannot decode (inter / garbage) contributes nothing — matching
/// the element's drop behaviour.
fn reference_decode(frames: &[Vec<u8>]) -> Vec<Vec<u8>> {
    frames
        .iter()
        .filter_map(|f| decode_intra_frame(f).ok().map(|d| d.to_planar_bytes()))
        .collect()
}

fn run_pipeline(frames: Vec<Vec<u8>>) -> (Pipeline, Arc<Mutex<Recorded>>) {
    let mut p = Pipeline::new();
    let (sink, recorded) = PlaneRecordSink::new();
    let src = p.add(FrameSrc::new(frames));
    let dec = p.add(Vp9Dec::new());
    let snk = p.add(sink);
    p.link((src, "src"), (dec, "sink")).expect("link src!dec");
    p.link((dec, "src"), (snk, "sink")).expect("link dec!sink");
    p.run().expect("run");
    (p, recorded)
}

/// Wrap coded frames into a single Annex B superframe chunk (1-byte sizes).
fn build_superframe(frames: &[&[u8]]) -> Vec<u8> {
    const MARKER: u8 = 0b110;
    // Pick the smallest byte width that holds every frame size (the keyframe is ~952 B,
    // so a single-byte size field would not fit — SzBytes must be >= 2 here).
    let max = frames.iter().map(|f| f.len()).max().unwrap_or(0);
    let szb: usize = if max > 0xff_ffff {
        4
    } else if max > 0xffff {
        3
    } else if max > 0xff {
        2
    } else {
        1
    };
    let m = MARKER << 5 | ((szb as u8 - 1) << 3) | (frames.len() as u8 - 1);
    let mut out = Vec::new();
    for f in frames {
        out.extend_from_slice(f);
    }
    out.push(m);
    for f in frames {
        let sz = f.len();
        for b in 0..szb {
            out.push(((sz >> (8 * b)) & 0xff) as u8);
        }
    }
    out.push(m);
    out
}

// ---- tests -------------------------------------------------------------------

#[test]
fn decodes_bit_exactly_with_pts_passthrough_and_announced_format() {
    let frames = vec![KEYFRAME.to_vec(), KEYFRAME.to_vec(), KEYFRAME.to_vec()];
    let reference = reference_decode(&frames);
    assert_eq!(reference.len(), 3, "the golden keyframe decodes via the library");
    let (_p, recorded) = run_pipeline(frames);

    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 3, "every frame emitted");
    for (i, ((pts, planes), want)) in rec.frames.iter().zip(&reference).enumerate() {
        assert_eq!(*pts, Timestamp::from_nanos(i as u64 * FRAME_NS), "frame {i} pts");
        assert_eq!(planes, want, "frame {i} planes byte-identical to reference decode");
    }
    let (w, h, pf) = rec.format.clone().expect("format announced");
    assert_eq!((w, h), (W, H));
    assert_eq!(pf, "i420", "8-bit 4:2:0 keyframe announces i420");
}

#[test]
fn corrupt_frame_warns_drops_and_recovers_at_next_keyframe() {
    // Good keyframe, garbage, good keyframe: the garbage is dropped, both keys decode.
    let frames = vec![KEYFRAME.to_vec(), vec![0xDE; 512], KEYFRAME.to_vec()];
    let reference = reference_decode(&[KEYFRAME.to_vec(), KEYFRAME.to_vec()]);
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

#[test]
fn superframe_packet_splits_and_decodes_the_enclosed_keyframe() {
    // One container packet carrying a single keyframe inside a valid superframe index
    // (the NumFrames == 1 legal case). The element must split it and decode the frame.
    let packet = build_superframe(&[KEYFRAME]);
    let reference = reference_decode(&[KEYFRAME.to_vec()]);
    let (_p, recorded) = run_pipeline(vec![packet]);

    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 1, "the superframe's enclosed keyframe decoded");
    assert_eq!(rec.frames[0].1, reference[0], "planes match the bare-keyframe decode");
}

#[test]
fn superframe_hidden_then_visible_emits_the_decodable_frames() {
    // A two-frame superframe: the golden keyframe twice (stand-ins for a hidden ARF +
    // its visible companion). Both are intra here, so both decode; the split + the
    // per-enclosed-frame drain are exercised. The last enclosed frame inherits the
    // packet pts.
    let packet = build_superframe(&[KEYFRAME, KEYFRAME]);
    let (_p, recorded) = run_pipeline(vec![packet]);

    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 2, "both enclosed frames decoded");
    // The last enclosed frame carries the packet pts (0 here — first packet).
    assert_eq!(rec.frames[1].0, Timestamp::from_nanos(0));
}

/// ffmpeg oracle: generate a fresh VP9 keyframe stream at test time, feed its coded
/// frames through the element, and require byte-exact agreement with the library's own
/// decode. Skips cleanly when ffmpeg is unavailable or lacks a VP9 encoder.
#[test]
fn ffmpeg_oracle_keyframe_stream_decodes_byte_exact() {
    let Some(frames) = ffmpeg_vp9_keyframes(80, 64) else {
        eprintln!("ffmpeg (with libvpx-vp9) not available; skipping oracle test");
        return;
    };
    assert!(!frames.is_empty(), "ffmpeg produced at least one coded frame");
    let reference = reference_decode(&frames);
    assert!(
        !reference.is_empty(),
        "at least one ffmpeg keyframe decodes via the library"
    );
    let (_p, recorded) = run_pipeline(frames);

    let rec = recorded.lock().unwrap();
    assert_eq!(
        rec.frames.len(),
        reference.len(),
        "the element emits exactly the library-decodable frames"
    );
    for (i, ((_pts, planes), want)) in rec.frames.iter().zip(&reference).enumerate() {
        assert_eq!(planes, want, "oracle frame {i} planes byte-identical");
    }
    let (w, h, _pf) = rec.format.clone().expect("format announced");
    assert_eq!((w, h), (80, 64), "announced dimensions match the ffmpeg input");
}

/// Seek (spec: flush/seek): the intra-only decoder holds no cross-frame reference
/// state, so its only flushable state is the backpressure carry (a decoded frame
/// awaiting a pool slot). With a one-slot pool the first keyframe decodes but its copy
/// stays *pending* (the slot is held by the previous, un-recycled buffer); `FlushStart`
/// must drop that carry so no pre-flush picture — with its pre-flush pts — leaks after
/// the seek. A fresh keyframe then decodes cleanly at the post-flush pts. Driven through
/// the [`Harness`] so `FlushStart` can be delivered mid-stream.
#[test]
fn flush_start_drops_the_pending_carry_and_no_stale_frame_leaks() {
    let reference = reference_decode(&[KEYFRAME.to_vec()]);
    assert_eq!(reference.len(), 1, "the golden keyframe decodes via the library");

    // A one-slot pool sized for the decoded frame: the first decode succeeds and emits,
    // but until that buffer is pulled (recycled) the pool is dry, so the *next* decode's
    // copy is left pending — exactly the carry the flush must drop.
    let need = reference[0].len();
    let mut h = Harness::with_pool(Vp9Dec::new(), need, 1);

    // Frame 1: decodes and emits into the single slot at pts 0.
    let mut b0 = h.alloc(KEYFRAME);
    b0.pts = Timestamp::from_nanos(0);
    h.push("sink", b0).expect("push frame 0");
    // Frame 2 at pts FRAME_NS: decodes but the pool is dry (slot 0 not yet pulled), so
    // the packed copy is stranded in the pending carry.
    let mut b1 = h.alloc(KEYFRAME);
    b1.pts = Timestamp::from_nanos(FRAME_NS);
    h.push("sink", b1).expect("push frame 1");

    // Simulate the scheduler discarding the in-flight (already-emitted) buffer on seek.
    let staged = h.drain_outputs();
    assert_eq!(staged.len(), 1, "one frame was staged before the pool went dry");
    assert_eq!(staged[0].pts, Timestamp::from_nanos(0), "staged frame is pre-flush frame 0");
    drop(staged); // recycles the slot, but the carry still holds the pre-flush frame 1

    // Seek: flush the pending carry.
    h.push_event(Event::FlushStart).expect("flush");

    // Post-seek: a fresh keyframe at a new pts. It must decode cleanly and be the *only*
    // frame that comes out — the pre-flush carried frame (pts FRAME_NS) must not appear.
    let post_pts = Timestamp::from_nanos(1000 * FRAME_NS);
    let mut b2 = h.alloc(KEYFRAME);
    b2.pts = post_pts;
    h.push("sink", b2).expect("push post-flush keyframe");

    let out = h.drain_outputs();
    assert_eq!(out.len(), 1, "exactly the post-flush keyframe — no stale carried frame leaked");
    assert_eq!(out[0].pts, post_pts, "post-flush pts comes from post-flush input");
    assert_eq!(out[0].memory.data(), reference[0].as_slice(), "post-flush keyframe decodes cleanly");

    let ann = h.announced().expect("format announced");
    let w = h.vocabulary().field_id("width").unwrap();
    let ht = h.vocabulary().field_id("height").unwrap();
    assert_eq!(ann.get(w), Some(Value::Int(W)));
    assert_eq!(ann.get(ht), Some(Value::Int(H)));
}

/// Encode `width x height` `testsrc2` as keyframe-only VP9 (`-g 1`) into IVF via
/// ffmpeg, then demux the IVF into raw coded frame payloads. Returns `None` when
/// ffmpeg is missing or the encode fails (no libvpx-vp9).
fn ffmpeg_vp9_keyframes(width: u32, height: u32) -> Option<Vec<Vec<u8>>> {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("sc_vp9_oracle_{width}x{height}_{}.ivf", std::process::id()));
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={width}x{height}:rate=1:duration=2"))
        .args([
            "-frames:v", "2", "-c:v", "libvpx-vp9", "-crf", "40", "-b:v", "0",
            "-g", "1", "-pix_fmt", "yuv420p", "-f", "ivf",
        ])
        .arg(&path)
        .status()
        .ok()?;
    if !status.success() {
        let _ = std::fs::remove_file(&path);
        return None;
    }
    let data = std::fs::read(&path).ok()?;
    let _ = std::fs::remove_file(&path);
    Some(demux_ivf(&data))
}

/// Minimal IVF demux: 32-byte file header, then per-frame a 4-byte LE size + 8-byte LE
/// timestamp + payload. Returns the raw coded frame payloads.
fn demux_ivf(data: &[u8]) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    if data.len() < 32 || &data[..4] != b"DKIF" {
        return frames;
    }
    let hdr_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut off = hdr_len;
    while off + 12 <= data.len() {
        let sz = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            as usize;
        let start = off + 12;
        let end = start + sz;
        if end > data.len() {
            break;
        }
        frames.push(data[start..end].to_vec());
        off = end;
    }
    frames
}
