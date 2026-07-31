//! Steady-state allocation + retention audit for the NVR **record** path:
//! `AU source → mkvsegmentsink`, run through the real scheduler with a counting
//! global allocator (the `transcode_alloc_check` pattern).
//!
//!   cargo run --release -p profluens-nvr --example record_alloc_check [-- SECONDS]
//!
//! Two numbers matter for a recorder that runs for weeks:
//!
//! * **allocations per access unit** in steady state — the per-frame cost;
//! * **live heap drift per segment rotation** — whether anything accumulates
//!   per closed segment (cue vectors, per-segment metadata, arena high-water).
//!   The counting allocator's `LIVE` is immune to allocator-arena noise, which
//!   is what makes RSS sampling inconclusive over a 15-minute window.

#![allow(clippy::disallowed_methods)] // one-shot measurement harness

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

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
use profluens_nvr::MkvSegmentSink;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static SAMPLING: AtomicBool = AtomicBool::new(false);
static STACKS: Mutex<Option<HashMap<String, (usize, usize)>>> = Mutex::new(None);
/// Access units the source has emitted (the per-frame denominator).
static AUS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static IN_SAMPLER: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

impl Counting {
    fn note(size: usize) {
        let _ = IN_SAMPLER.try_with(|g| {
            if g.get() {
                return;
            }
            g.set(true);
            let n = ALLOCS.fetch_add(1, Ordering::Relaxed) + 1;
            if SAMPLING.load(Ordering::Relaxed) && n.is_multiple_of(8) {
                let key = capture_site();
                if let Ok(mut guard) = STACKS.lock() {
                    let map = guard.get_or_insert_with(HashMap::new);
                    let e = map.entry(key).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += size;
                }
            }
            g.set(false);
        });
    }
}

const PLUMBING: &[&str] = &[
    "try_allocate_in", "allocate_in", "RawVec", "raw_vec", "exchange_malloc", "finish_grow",
    "grow_amortized", "grow_one", "do_reserve_and_handle", "reserve", "with_capacity",
    "from_iter", "to_vec", "into_vec", "extend_desugared", "SpecFrom", "SpecExtend",
    "spec_extend", "into_boxed_slice", "alloc::alloc", "alloc_impl", "grow_impl", "shrink_impl",
    "realloc_nonnull", "__rust",
];
const PLUMBING_EXACT: &[&str] = &["alloc", "realloc", "alloc_zeroed", "allocate", "grow", "shrink"];

fn capture_site() -> String {
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    let mut frames: Vec<&str> = Vec::new();
    for line in bt.lines() {
        let t = line.trim_start();
        let Some((num, rest)) = t.split_once(": ") else { continue };
        if !num.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        frames.push(rest.trim());
    }
    let start = frames.iter().rposition(|f| f.ends_with("note") || f.contains("capture_site"));
    let tail = start.map_or(&frames[..], |i| &frames[i + 1..]);
    let mut out: Vec<&str> = Vec::new();
    for f in tail {
        if PLUMBING.iter().any(|p| f.contains(p)) || PLUMBING_EXACT.contains(f) {
            continue;
        }
        out.push(f);
        if out.len() == 4 {
            break;
        }
    }
    if out.is_empty() { "<unknown>".to_string() } else { out.join(" ← ") }
}

// SAFETY: delegates every request to `System`; the extra work is a relaxed counter
// and (in profile mode) a re-entrancy-guarded backtrace capture.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size(), Ordering::Relaxed);
        Self::note(l.size());
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        LIVE.fetch_add(new.wrapping_sub(l.size()), Ordering::Relaxed);
        Self::note(new);
        System.realloc(p, l, new)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size(), Ordering::Relaxed);
        Self::note(l.size());
        System.alloc_zeroed(l)
    }
}

#[global_allocator]
static A: Counting = Counting;

// --- the AU source ------------------------------------------------------------

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("h264/annexb")];
static PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static DESC: ElementDesc = ElementDesc {
    name: "aureplay",
    pads: &PADS,
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

/// Replays a fixed AU set forever with advancing timestamps — a camera that
/// never stops, at whatever rate the pipeline can absorb.
struct AuReplay {
    aus: Vec<(Vec<u8>, bool)>,
    at: usize,
    frame: u64,
    total: u64,
    frame_ns: u64,
}

impl Element for AuReplay {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        loop {
            if self.frame >= self.total {
                return Ok(Flow::Eos);
            }
            let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
            let (au, _kf) = &self.aus[self.at];
            buf.memory.as_mut_full()[..au.len()].copy_from_slice(au);
            buf.memory.set_len(au.len());
            buf.pts = Timestamp::from_nanos(self.frame * self.frame_ns);
            ctx.out(PadId(0)).push(buf);
            self.at = (self.at + 1) % self.aus.len();
            self.frame += 1;
            AUS.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

fn fixture_aus(path: &Path) -> Vec<(Vec<u8>, bool)> {
    use pf_mkv::codec::{nal_head_from_config, Reframer};
    use pf_mkv::MatroskaReader;

    let bytes = std::fs::read(path).expect("read fixture");
    let mut r = MatroskaReader::new();
    r.push(&bytes).expect("parse fixture");
    let track = r
        .tracks()
        .iter()
        .find(|t| t.codec_id == "V_MPEG4/ISO/AVC")
        .cloned()
        .expect("h264 track");
    let (head, length_size) = nal_head_from_config(&track.codec_private, false).expect("avcC");
    let reframer = Reframer::Nal { length_size };
    let mut scratch = Vec::new();
    let mut out = Vec::new();
    while let Some(f) = r.next_frame() {
        if f.track_number != track.track_number {
            continue;
        }
        let body = reframer.reframe_into(&f.data, &mut scratch).expect("reframe");
        let mut au = Vec::with_capacity(head.len() + body.len());
        if f.keyframe {
            au.extend_from_slice(&head);
        }
        au.extend_from_slice(body);
        out.push((au, f.keyframe));
    }
    // The loop must restart on a keyframe so every wrap is a clean entry point.
    while !out.first().map(|(_, k)| *k).unwrap_or(true) {
        out.remove(0);
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fixture = PathBuf::from(
        args.first().cloned().unwrap_or_else(|| "/tmp/nvraudit/cam.mkv".to_string()),
    );
    // Frames to push; 25 fps nominal, 2 s segments → one rotation per 50 frames.
    let frames: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let profile = std::env::var_os("PF_ALLOC_PROFILE").is_some();

    let aus = fixture_aus(&fixture);
    println!("fixture: {} AUs, {} frames to push", aus.len(), frames);

    let rec = PathBuf::from("/tmp/nvraudit/allocrec");
    let _ = std::fs::remove_dir_all(&rec);

    let mut p = Pipeline::new();
    p.set_pool(1024 * 1024, 16);
    let src = p.add(AuReplay { aus, at: 0, frame: 0, total: frames, frame_ns: 40_000_000 });
    let seg = MkvSegmentSink::new(&rec, "cam0", 2.0, &[]);
    let stats = seg.stats_handle();
    let sink = p.add(seg);
    p.link((src, "src"), (sink, "sink")).expect("link");

    // Sample live heap + allocation count against segment closes, on a helper
    // thread — the drift per rotation is the number this harness exists for.
    let probe = stats.clone();
    std::thread::spawn(move || {
        let mut marks: Vec<(u64, u64, usize, usize)> = Vec::new();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            let s = *probe.lock().unwrap();
            marks.push((
                s.segments_closed,
                AUS.load(Ordering::Relaxed),
                ALLOCS.load(Ordering::Relaxed),
                LIVE.load(Ordering::Relaxed),
            ));
            if marks.len() % 8 == 0 {
                let (seg0, au0, al0, li0) = marks[marks.len() / 2];
                let (seg1, au1, al1, li1) = *marks.last().unwrap();
                let dseg = seg1.saturating_sub(seg0).max(1);
                let dau = au1.saturating_sub(au0).max(1);
                println!(
                    "segments={seg1} aus={au1} live={:.2} MB | window: {:.2} allocs/AU, \
                     live drift {:+} B/segment ({:+} B/AU)",
                    li1 as f64 / 1e6,
                    (al1 - al0) as f64 / dau as f64,
                    (li1 as i64 - li0 as i64) / dseg as i64,
                    (li1 as i64 - li0 as i64) / dau as i64,
                );
            }
        }
    });

    // Warm up, then take the steady-state window.
    let t0 = std::time::Instant::now();
    let run = std::thread::spawn(move || p.run());
    std::thread::sleep(std::time::Duration::from_secs(3));
    let (au0, al0, li0) = (
        AUS.load(Ordering::Relaxed),
        ALLOCS.load(Ordering::Relaxed),
        LIVE.load(Ordering::Relaxed),
    );
    let seg0 = stats.lock().unwrap().segments_closed;
    if profile {
        SAMPLING.store(true, Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_secs(5));
        SAMPLING.store(false, Ordering::Relaxed);
    }
    let r = run.join().expect("run thread");
    let (au1, al1, li1) = (
        AUS.load(Ordering::Relaxed),
        ALLOCS.load(Ordering::Relaxed),
        LIVE.load(Ordering::Relaxed),
    );
    let s = *stats.lock().unwrap();
    let dseg = s.segments_closed.saturating_sub(seg0).max(1);
    let dau = au1.saturating_sub(au0).max(1);
    println!("\n--- steady-state window ({:?}) ---", t0.elapsed());
    println!("result: {r:?}");
    println!("segments closed: {} (window {dseg})", s.segments_closed);
    println!("AUs: {au1} (window {dau}), frames written {}", s.frames_written);
    println!("allocations in window: {} → {:.2} per AU", al1 - al0, (al1 - al0) as f64 / dau as f64);
    println!(
        "live heap: {:.3} MB → {:.3} MB   drift {:+} B/segment, {:+} B/AU",
        li0 as f64 / 1e6,
        li1 as f64 / 1e6,
        (li1 as i64 - li0 as i64) / dseg as i64,
        (li1 as i64 - li0 as i64) / dau as i64,
    );
    if profile {
        if let Some(map) = STACKS.lock().unwrap().take() {
            let mut v: Vec<_> = map.into_iter().collect();
            v.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
            println!("\ntop allocation sites (sampled 1:8):");
            for (site, (n, bytes)) in v.into_iter().take(15) {
                println!("  {:7} × ~{:>9} B   {site}", n * 8, bytes * 8);
            }
        }
    }
}
