//! End-to-end proof of the **bitmap (PGS) subtitle** path (spec: subtitle support; RFC 9559
//! §12.7 `S_HDMV/PGS`).
//!
//! Builds a synthetic PGS Display Set (a solid opaque-white rectangle at a known position),
//! feeds it through `pgsdec` to the `subtitle/bitmap` family, and composites it over a solid
//! video frame in `suboverlay.image`:
//!
//! ```text
//!   pgssrc  ─(subtitle/pgs)→ pgsdec ─(subtitle/bitmap)→ suboverlay.image
//!   videosrc ─(video/raw i420)──────────────────────────▶ suboverlay.video
//!   suboverlay.src ─(video/raw)→ planerecordsink
//! ```
//!
//! No external tools: the PGS bytes and the video frames are hand-built. The proof: the
//! composited frame's luma **changes inside the caption rectangle** (the white bitmap burned
//! in) and is **byte-for-byte unchanged everywhere else**. Headless — no display.
//!
//! Run: `nix develop --command cargo run --release -p sc-text --example pgs_overlay`

use std::sync::{Arc, Mutex};

use sc_text::pgs;
use sc_text::{PgsDec, SubtitleOverlay};

use streamcraft_elements::flow::Queue;

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{OfferDesc, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

use streamcraft_video::format::PixelFormat;
use streamcraft_video::geometry::frame_size;

/// The frame and caption geometry the proof asserts against.
const FRAME_W: u32 = 128;
const FRAME_H: u32 = 96;
const CAP_X: u32 = 30;
const CAP_Y: u32 = 60;
const CAP_W: u32 = 40;
const CAP_H: u32 = 16;

fn main() {
    let recorded = run();
    let rec = recorded.lock().unwrap();
    println!("pgs_overlay: recorded {} composited frames", rec.frames.len());
    assert!(!rec.frames.is_empty(), "the overlay produced frames end-to-end");

    // Take a frame from within the caption's on-screen span (pts 1.5 s, span [1, 3) s).
    let f = rec
        .frames
        .iter()
        .find(|f| {
            let t = f.pts.nanos().unwrap_or(0);
            (1_000_000_000..3_000_000_000).contains(&t)
        })
        .expect("a frame inside the caption span");

    // Compare against a pristine solid frame (what videosrc emitted before compositing).
    let baseline = vec![SOLID_LUMA; frame_size(PixelFormat::I420, FRAME_W, FRAME_H)];
    let ys = FRAME_W as usize;
    let mut inside_changed = 0usize;
    let mut outside_changed = 0usize;
    for y in 0..FRAME_H as usize {
        for x in 0..FRAME_W as usize {
            let inside = (CAP_X as usize..(CAP_X + CAP_W) as usize).contains(&x)
                && (CAP_Y as usize..(CAP_Y + CAP_H) as usize).contains(&y);
            let idx = y * ys + x;
            if f.planes[idx] != baseline[idx] {
                if inside {
                    inside_changed += 1;
                } else {
                    outside_changed += 1;
                }
            }
        }
    }
    println!(
        "pgs_overlay: caption rect {CAP_W}x{CAP_H}@({CAP_X},{CAP_Y}) — luma pixels changed \
         inside={inside_changed} outside={outside_changed}"
    );
    assert!(inside_changed > 0, "the caption rectangle's luma was painted by the bitmap");
    assert_eq!(outside_changed, 0, "nothing outside the caption rectangle changed");
    // The ink is near-white (the caption is opaque white over mid-grey).
    let some_bright = (0..FRAME_H as usize).any(|y| {
        (CAP_X as usize..(CAP_X + CAP_W) as usize).any(|x| f.planes[y * ys + x] > 200)
    });
    assert!(some_bright, "the white caption raised luma toward white");
    println!("pgs_overlay: PASS — pgsdec ! suboverlay.image burned the PGS bitmap into the frame");
}

/// Build one PGS Display Set: a solid opaque-white `CAP_W×CAP_H` object at `(CAP_X, CAP_Y)`,
/// against a `FRAME_W×FRAME_H` reference (1:1 with the frame, so no scaling). Bare segments
/// (no `.sup` `PG` header) — exactly what an mkv Block carries.
fn build_display_set() -> Vec<u8> {
    // PDS: palette id 0, index 1 = opaque white (Y=235, neutral chroma, A=255).
    let mut pds_body = vec![0x00u8, 0x00]; // palette_id, version
    pds_body.extend_from_slice(&[0x01, 235, 128, 128, 255]);
    let pds = segment(0x14, &pds_body);

    // ODS: object 0, first+last (0xC0), CAP_W×CAP_H, RLE = each row is a long colour-1 run.
    let mut rle = Vec::new();
    for _ in 0..CAP_H {
        // form 11: 64..16383 px of a colour — length CAP_W, colour 1, then EOL.
        rle.push(0x00);
        rle.push(0xC0 | ((CAP_W >> 8) as u8 & 0x3F));
        rle.push((CAP_W & 0xFF) as u8);
        rle.push(0x01); // colour index 1 (white)
        rle.push(0x00); // EOL
        rle.push(0x00);
    }
    let mut ods_body = vec![0x00u8, 0x00, 0x00, 0xC0]; // object_id, version, seq(first+last)
    let data_len = (4 + rle.len()) as u32; // width+height (4) + RLE
    ods_body.extend_from_slice(&data_len.to_be_bytes()[1..]); // u24
    ods_body.extend_from_slice(&(CAP_W as u16).to_be_bytes());
    ods_body.extend_from_slice(&(CAP_H as u16).to_be_bytes());
    ods_body.extend_from_slice(&rle);
    let ods = segment(0x15, &ods_body);

    // PCS: FRAME_W×FRAME_H reference, one object (id 0) at (CAP_X, CAP_Y).
    let mut pcs_body = Vec::new();
    pcs_body.extend_from_slice(&(FRAME_W as u16).to_be_bytes());
    pcs_body.extend_from_slice(&(FRAME_H as u16).to_be_bytes());
    pcs_body.push(0x10); // frame_rate
    pcs_body.extend_from_slice(&0u16.to_be_bytes()); // composition_number
    pcs_body.push(0x80); // composition_state = epoch start
    pcs_body.push(0x00); // palette_update_flag
    pcs_body.push(0x00); // palette_id
    pcs_body.push(0x01); // number_of_composition_objects
    pcs_body.extend_from_slice(&0u16.to_be_bytes()); // object_id
    pcs_body.push(0x00); // window_id
    pcs_body.push(0x00); // object_cropped_flag
    pcs_body.extend_from_slice(&(CAP_X as u16).to_be_bytes());
    pcs_body.extend_from_slice(&(CAP_Y as u16).to_be_bytes());
    let pcs = segment(0x16, &pcs_body);

    // WDS: one window matching the caption rectangle.
    let mut wds_body = vec![0x01u8]; // number_of_windows
    wds_body.push(0x00); // window_id
    wds_body.extend_from_slice(&(CAP_X as u16).to_be_bytes());
    wds_body.extend_from_slice(&(CAP_Y as u16).to_be_bytes());
    wds_body.extend_from_slice(&(CAP_W as u16).to_be_bytes());
    wds_body.extend_from_slice(&(CAP_H as u16).to_be_bytes());
    let wds = segment(0x17, &wds_body);

    let end = segment(0x80, &[]);

    let mut ds = Vec::new();
    ds.extend_from_slice(&pcs);
    ds.extend_from_slice(&wds);
    ds.extend_from_slice(&pds);
    ds.extend_from_slice(&ods);
    ds.extend_from_slice(&end);
    ds
}

/// Frame a PGS segment: `type u8, size u16 BE, payload`.
fn segment(seg_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut s = vec![seg_type];
    s.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    s.extend_from_slice(payload);
    s
}

fn run() -> Arc<Mutex<Recorded>> {
    let mut p = Pipeline::new();

    // Sanity: the synthetic Display Set decodes to the expected geometry before we wire it.
    let ds_bytes = build_display_set();
    let decoded = pgs::parse_display_set(&ds_bytes).expect("synthetic display set decodes");
    assert_eq!((decoded.width, decoded.height), (CAP_W, CAP_H));
    assert_eq!((decoded.x, decoded.y), (CAP_X, CAP_Y));
    println!(
        "pgs_overlay: synthetic display set: {}x{} @ ({},{}), ref {}x{}, rgba {} bytes",
        decoded.width, decoded.height, decoded.x, decoded.y, decoded.video_width,
        decoded.video_height, decoded.rgba.len()
    );

    let pgssrc = p.add(PgsSrc::new(ds_bytes));
    let videosrc = p.add(VideoSrc::new());
    let pgsdec = p.add(PgsDec::new());
    // A queue heads the subtitle branch (an active group head — the same reasoning as
    // overlay_mkv: the video branch and the subtitle branch must each be their own group head).
    let subq = p.add(Queue::new());
    let overlay = p.add(SubtitleOverlay::new());
    let (sink, recorded) = PlaneRecordSink::new();
    let snk = p.add(sink);

    p.link((videosrc, "src"), (overlay, "video")).expect("video -> overlay.video");
    p.link((pgssrc, "src"), (subq, "sink")).expect("pgs -> queue");
    p.link((subq, "src"), (pgsdec, "sink")).expect("queue -> pgsdec");
    p.link((pgsdec, "src"), (overlay, "image")).expect("pgsdec -> overlay.image");
    p.link((overlay, "src"), (snk, "sink")).expect("overlay -> sink");

    p.run().expect("run");
    recorded
}

// -------------------------------------------------------------------------------------------
// test elements: a one-shot PGS source, a solid-frame video source, a plane-recording sink
// -------------------------------------------------------------------------------------------

/// Mid-grey luma the solid video frames are filled with (so the white caption stands out).
const SOLID_LUMA: u8 = 128;
/// How many video frames the source emits (25 fps → the caption span [1, 3) s is covered).
const N_FRAMES: u64 = 100;

static PGS_SRC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("subtitle/pgs")];
static PGS_SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &PGS_SRC_OFFERS,
    dynamic: true,
    validate: None,
}];
static PGS_SRC_DESC: ElementDesc = ElementDesc {
    name: "pgssrc",
    pads: &PGS_SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

/// Emits one PGS Display Set (bare segments) as a single Block with PTS 1 s, duration 2 s
/// (the on-screen span [1, 3) s), then EOS. `announce`s the `subtitle/pgs` family first.
struct PgsSrc {
    ds: Vec<u8>,
    sent: bool,
    announced: bool,
}

impl PgsSrc {
    fn new(ds: Vec<u8>) -> Self {
        Self { ds, sent: false, announced: false }
    }
}

impl Element for PgsSrc {
    fn desc(&self) -> &'static ElementDesc {
        &PGS_SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.sent = false;
        self.announced = false;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.announced {
            ctx.announce_format(PadId(0), "subtitle/pgs", &[]);
            self.announced = true;
        }
        if self.sent {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let n = self.ds.len().min(buf.memory.capacity());
        buf.memory.as_mut_full()[..n].copy_from_slice(&self.ds[..n]);
        buf.memory.set_len(n);
        buf.pts = Timestamp::from_nanos(1_000_000_000); // 1 s
        buf.duration = Timestamp::from_nanos(2_000_000_000); // shown [1, 3) s
        ctx.out(PadId(0)).push(buf);
        self.sent = true;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static PIXFMTS: [ValueDesc; 1] = [ValueDesc::Id("i420")];
static VIDEO_SRC_FIELDS: [streamcraft_core::format::FieldDesc; 3] = [
    streamcraft_core::format::FieldDesc { field: "width", allowed: streamcraft_core::format::ConstraintDesc::Any, preferred: None },
    streamcraft_core::format::FieldDesc { field: "height", allowed: streamcraft_core::format::ConstraintDesc::Any, preferred: None },
    streamcraft_core::format::FieldDesc { field: "pixfmt", allowed: streamcraft_core::format::ConstraintDesc::Set(&PIXFMTS), preferred: None },
];
static VIDEO_SRC_OFFERS: [OfferDesc; 1] =
    [OfferDesc { family: "video/raw", fields: &VIDEO_SRC_FIELDS }];
static VIDEO_SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &VIDEO_SRC_OFFERS,
    dynamic: true,
    validate: None,
}];
static VIDEO_SRC_DESC: ElementDesc = ElementDesc {
    name: "videosrc",
    pads: &VIDEO_SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

/// Emits `N_FRAMES` solid mid-grey I420 frames at 25 fps, announcing `video/raw` first.
struct VideoSrc {
    n: u64,
    announced: bool,
}

impl VideoSrc {
    fn new() -> Self {
        Self { n: 0, announced: false }
    }
}

impl Element for VideoSrc {
    fn desc(&self) -> &'static ElementDesc {
        &VIDEO_SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.n = 0;
        self.announced = false;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.announced {
            ctx.announce_format(
                PadId(0),
                "video/raw",
                &[
                    ("width", ValueDesc::Int(FRAME_W as i64)),
                    ("height", ValueDesc::Int(FRAME_H as i64)),
                    ("pixfmt", ValueDesc::Id("i420")),
                ],
            );
            self.announced = true;
        }
        if self.n >= N_FRAMES {
            return Ok(Flow::Eos);
        }
        let need = frame_size(PixelFormat::I420, FRAME_W, FRAME_H);
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        if buf.memory.capacity() < need {
            // Shouldn't happen with the default slot size, but never write past the slot.
            buf.memory.set_len(0);
            ctx.out(PadId(0)).push(buf);
            return Ok(Flow::Ok);
        }
        let full = buf.memory.as_mut_full();
        let y_len = (FRAME_W * FRAME_H) as usize;
        full[..y_len].fill(SOLID_LUMA); // luma
        full[y_len..need].fill(128); // neutral chroma
        buf.memory.set_len(need);
        buf.pts = Timestamp::from_nanos(self.n * 40_000_000); // 25 fps
        ctx.out(PadId(0)).push(buf);
        self.n += 1;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static REC_PIXFMTS: [ValueDesc; 2] = [ValueDesc::Id("i420"), ValueDesc::Id("nv12")];
static REC_FIELDS: [streamcraft_core::format::FieldDesc; 3] = [
    streamcraft_core::format::FieldDesc { field: "width", allowed: streamcraft_core::format::ConstraintDesc::Any, preferred: None },
    streamcraft_core::format::FieldDesc { field: "height", allowed: streamcraft_core::format::ConstraintDesc::Any, preferred: None },
    streamcraft_core::format::FieldDesc { field: "pixfmt", allowed: streamcraft_core::format::ConstraintDesc::Set(&REC_PIXFMTS), preferred: None },
];
static REC_OFFERS: [OfferDesc; 1] = [OfferDesc { family: "video/raw", fields: &REC_FIELDS }];
static REC_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &REC_OFFERS,
    dynamic: false,
    validate: None,
}];
static REC_DESC: ElementDesc = ElementDesc {
    name: "planerecordsink",
    pads: &REC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct Frame {
    pts: Timestamp,
    planes: Vec<u8>,
}

#[derive(Default)]
struct Recorded {
    frames: Vec<Frame>,
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
        &REC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut rec = self.shared.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            rec.frames.push(Frame { pts: buf.pts, planes: buf.memory.data().to_vec() });
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}
