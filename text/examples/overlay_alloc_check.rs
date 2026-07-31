//! Steady-state allocation + copy-on-write audit for the **subtitle overlay** path.
//!
//! Runs `videosrc → suboverlay → sink` through the real scheduler with a counting global
//! allocator and reports allocations per composited frame, plus how often the overlay's
//! `Memory::as_mut_full()` copy-on-wrote the whole frame. Four graph shapes: plain, behind a
//! `tee` whose second branch drains, behind a `tee` whose second branch *holds* its clones, and
//! plain but with the source keeping a reference to each frame (a decoder's DPB).
//!
//! **How the CoW is detected, exactly.** The source records `buf.memory.data().as_ptr()` for
//! each frame just before pushing it; the sink records the pointer it receives. That slot is
//! continuously alive in between, so a CoW — which acquires a *fresh* slot while the old one is
//! still referenced — can never land on the same address. A mismatch is therefore a proven copy,
//! and a match is a proven in-place mutation.
//!
//! Each configuration runs in its own process: pipeline construction interns into process-global
//! state, so consecutive runs in one process are not independent.
//!
//!   cargo run --release -p pf-text --example overlay_alloc_check

#![allow(clippy::disallowed_methods)] // one-shot measurement harness

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;
use profluens_elements::flow::{Queue, Tee};
use profluens_video::format::PixelFormat;
use profluens_video::geometry::frame_size;

use pf_text::SubtitleOverlay;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new, Ordering::Relaxed);
        unsafe { System.realloc(p, l, new) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

const FRAME_W: u32 = 1920;
const FRAME_H: u32 = 1080;
const N_FRAMES: u64 = 100;

fn main() {
    // One configuration per process: pipeline construction interns into process-global state,
    // so consecutive runs in one process are not independent.
    let cfgs: [(&str, Fan); 4] = [
        ("video ! suboverlay ! sink", Fan::None),
        ("video ! tee ! (overlay | drain)", Fan::TeeDrain),
        ("video ! tee ! (overlay | HOLD)", Fan::TeeRetain),
        ("video ! suboverlay ! sink  [src holds a DPB ref]", Fan::SrcHold),
    ];
    let arg: Vec<String> = std::env::args().collect();
    let Some(sel) = arg.get(1).and_then(|a| a.parse::<usize>().ok()) else {
        // Driver: re-exec ourselves once per configuration.
        for i in 0..cfgs.len() * 2 {
            let out = std::process::Command::new(&arg[0]).arg(i.to_string()).output().expect("re-exec");
            print!("{}", String::from_utf8_lossy(&out.stdout));
        }
        println!("\nframe = {} bytes I420 {}x{}", frame_size(PixelFormat::I420, FRAME_W, FRAME_H), FRAME_W, FRAME_H);
        return;
    };
    let (label, tee) = cfgs[sel / 2];
    let caption = sel % 2 == 1;
    let r = run(tee, caption);
    let frames = r.sink_ptrs.len().max(1);
    println!(
        "{label:50}  caption={caption:5}  frames={:4}  allocs={:5} ({:5.1}/frame)  \
         heap B/frame={:8.0}  painted={:4}  CoW={:4} ({:5.1}%)  {:7.2} ms/frame",
        r.sink_ptrs.len(),
        r.allocs,
        r.allocs as f64 / frames as f64,
        r.bytes as f64 / frames as f64,
        r.painted,
        r.cow,
        100.0 * r.cow as f64 / frames as f64,
        r.ms / frames as f64,
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fan {
    None,
    TeeDrain,
    TeeRetain,
    /// The source keeps a refcount clone of each frame for a few frames — exactly what a video
    /// decoder does with a reference frame in its DPB. The overlay then receives a *shared*
    /// backing, so `Memory::as_mut_full()` must copy-on-write.
    SrcHold,
}

#[derive(Default)]
struct Report {
    allocs: usize,
    bytes: usize,
    cow: usize,
    ms: f64,
    painted: usize,
    sink_ptrs: Vec<usize>,
}

fn run(with_tee: Fan, caption: bool) -> Report {
    let probe: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let sink_ptrs: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));

    let mut p = Pipeline::new();
    // A 640x360 I420 frame is 345 600 B — well past the default slot; size the pool for it.
    p.set_pool(frame_size(PixelFormat::I420, FRAME_W, FRAME_H), 16);
    let videosrc = p.add(VideoSrc::new(Arc::clone(&probe), with_tee == Fan::SrcHold));
    let overlay = p.add(SubtitleOverlay::new());
    let painted = Arc::new(AtomicUsize::new(0));
    let snk = p.add(PtrSink { seen: Arc::clone(&sink_ptrs), painted: Arc::clone(&painted) });

    if matches!(with_tee, Fan::TeeDrain | Fan::TeeRetain) {
        let tee = p.add(Tee::new(2));
        let q = p.add(Queue::new());
        let drain = p.add(NullSink { retain: if with_tee == Fan::TeeRetain { Some(Vec::new()) } else { None } });
        p.link((videosrc, "src"), (tee, "sink")).expect("src->tee");
        p.link((tee, "src_0"), (overlay, "video")).expect("tee->overlay.video");
        p.link((tee, "src_1"), (q, "sink")).expect("tee->queue");
        p.link((q, "src"), (drain, "sink")).expect("queue->null");
    } else {
        p.link((videosrc, "src"), (overlay, "video")).expect("src->overlay.video");
    }
    // The text branch is always wired (an `InputPolicy::Any` fan-in with several unlinked sink
    // pads does not run); `caption` only decides whether a cue is actually emitted.
    let textsrc = p.add(TextSrc::new(caption));
    let tq = p.add(Queue::new());
    p.link((textsrc, "src"), (tq, "sink")).expect("text->queue");
    p.link((tq, "src"), (overlay, "text")).expect("queue->overlay.text");
    p.link((overlay, "src"), (snk, "sink")).expect("overlay->sink");

    let base_a = ALLOCS.load(Ordering::Relaxed);
    let base_b = BYTES.load(Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    p.run().expect("run");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let allocs = ALLOCS.load(Ordering::Relaxed) - base_a;
    let bytes = BYTES.load(Ordering::Relaxed) - base_b;

    let before = probe.lock().unwrap().clone();
    let after = sink_ptrs.lock().unwrap().clone();
    // Pairwise: a frame whose pointer changed between the probe and the sink was copied.
    let cow = before.iter().zip(after.iter()).filter(|(a, b)| a != b).count();
    Report { allocs, bytes, cow, ms, painted: painted.load(Ordering::Relaxed), sink_ptrs: after }
}

// ------------------------------------------------------------------ test elements

static PIXFMTS: [ValueDesc; 1] = [ValueDesc::Id("i420")];
static VIDEO_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: "width", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "height", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "pixfmt", allowed: ConstraintDesc::Set(&PIXFMTS), preferred: None },
];
static VIDEO_OFFERS: [OfferDesc; 1] = [OfferDesc { family: "video/raw", fields: &VIDEO_FIELDS }];

static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &VIDEO_OFFERS,
    dynamic: true,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "videosrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct VideoSrc {
    n: u64,
    announced: bool,
    seen: Arc<Mutex<Vec<usize>>>,
    /// A bounded ring of refcount clones — the DPB model (see `Fan::SrcHold`).
    dpb: Option<Vec<profluens_core::memory::Memory>>,
}
impl VideoSrc {
    fn new(seen: Arc<Mutex<Vec<usize>>>, hold: bool) -> Self {
        Self { n: 0, announced: false, seen, dpb: if hold { Some(Vec::new()) } else { None } }
    }
}
impl Element for VideoSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
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
            buf.memory.set_len(0);
            ctx.out(PadId(0)).push(buf);
            return Ok(Flow::Ok);
        }
        buf.memory.as_mut_full()[..need].fill(128);
        buf.memory.set_len(need);
        buf.pts = Timestamp::from_nanos(self.n * 40_000_000);
        // Record the backing pointer of the frame we are about to hand downstream. The slot
        // stays alive continuously from here to the sink, so a CoW inside the overlay must
        // land on a *different* slot — a pointer mismatch at the sink is a proven copy.
        self.seen.lock().unwrap().push(buf.memory.data().as_ptr() as usize);
        if let Some(dpb) = self.dpb.as_mut() {
            dpb.push(buf.memory.clone());
            // A realistic reference window: an H.264/HEVC DPB holds a handful of frames.
            if dpb.len() > 8 {
                dpb.remove(0);
            }
        }
        ctx.out(PadId(0)).push(buf);
        self.n += 1;
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
    offers: &VIDEO_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "ptrsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct PtrSink {
    seen: Arc<Mutex<Vec<usize>>>,
    painted: Arc<AtomicUsize>,
}
impl Element for PtrSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            self.seen.lock().unwrap().push(buf.memory.data().as_ptr() as usize);
            // Any luma away from the flat 128 the source wrote = something was composited.
            let y = &buf.memory.data()[..(FRAME_W * FRAME_H) as usize];
            if y[y.len() * 3 / 4..].iter().any(|&p| p != 128) {
                self.painted.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static NULL_DESC: ElementDesc = ElementDesc {
    name: "nullsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};
struct NullSink {
    /// `Some` = hold every buffer for the whole run, so the tee's refcount clone stays alive
    /// while the overlay mutates the original — the positive control for the CoW detector.
    retain: Option<Vec<profluens_core::buffer::Buffer>>,
}
impl Element for NullSink {
    fn desc(&self) -> &'static ElementDesc {
        &NULL_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(b) = inputs.pop() {
            if let Some(held) = self.retain.as_mut() {
                // A bounded hold (a decoder DPB / a slower peer branch): keep the refcount
                // clone alive across a few frames, then let it go so the pool never starves.
                held.push(b);
                if held.len() > 3 {
                    held.remove(0);
                }
            }
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// --- one long-lived text cue covering the whole run ---

static TEXT_OFFERS: [OfferDesc; 1] = [OfferDesc::any(pf_text::EVENTS_FAMILY)];
static TEXT_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &TEXT_OFFERS,
    dynamic: true,
    validate: None,
}];
static TEXT_DESC: ElementDesc = ElementDesc {
    name: "textsrc",
    pads: &TEXT_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};
struct TextSrc {
    sent: bool,
    announced: bool,
    emit: bool,
}
impl TextSrc {
    fn new(emit: bool) -> Self {
        Self { sent: false, announced: false, emit }
    }
}
impl Element for TextSrc {
    fn desc(&self) -> &'static ElementDesc {
        &TEXT_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.sent = false;
        self.announced = false;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.announced {
            ctx.announce_format(PadId(0), pf_text::EVENTS_FAMILY, &[]);
            self.announced = true;
        }
        if self.sent || !self.emit {
            return Ok(Flow::Eos);
        }
        let text = b"A caption that stays on screen";
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        buf.memory.as_mut_full()[..text.len()].copy_from_slice(text);
        buf.memory.set_len(text.len());
        buf.pts = Timestamp::from_nanos(0);
        buf.duration = Timestamp::from_nanos(N_FRAMES * 40_000_000 + 1_000_000_000);
        ctx.out(PadId(0)).push(buf);
        self.sent = true;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}
