//! Steady-state allocation audit for the **WAV -> FLAC encode pipeline**:
//! `filesrc -> wavparse -> flacenc -> countsink`, through the real scheduler.
//!
//!     cargo run --release -p profluens-audio --example wav_to_flac_alloc_check -- in.wav
//!     PF_ALLOC_PROFILE=1 cargo run --release -p profluens-audio --example wav_to_flac_alloc_check -- in.wav
//!
//! Normalised **per emitted FLAC buffer**, with construct/preroll counted apart from
//! streaming, so two input lengths separate a real per-buffer cost from one-time startup.

#![allow(clippy::disallowed_methods)] // one-shot measurement harness

use std::cell::Cell;
use std::collections::HashMap;
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use pf_flac::{FlacEnc, SampleFormat as FlacFmt};
use profluens_audio::{parse_wav_header, SampleFormat, WavParse};
use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;
use profluens_elements::io::FileSrc;

use std::alloc::{GlobalAlloc, Layout, System};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static SAMPLING: AtomicBool = AtomicBool::new(false);
static SAMPLE_EVERY: AtomicUsize = AtomicUsize::new(1);
static STACKS: Mutex<Option<HashMap<String, (usize, usize)>>> = Mutex::new(None);

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
            BYTES.fetch_add(size, Ordering::Relaxed);
            PEAK.fetch_max(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
            if SAMPLING.load(Ordering::Relaxed)
                && n.is_multiple_of(SAMPLE_EVERY.load(Ordering::Relaxed))
            {
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
    "try_allocate_in",
    "allocate_in",
    "RawVec",
    "raw_vec",
    "exchange_malloc",
    "finish_grow",
    "grow_amortized",
    "grow_one",
    "do_reserve_and_handle",
    "reserve",
    "with_capacity",
    "from_iter",
    "to_vec",
    "into_vec",
    "extend_desugared",
    "SpecFrom",
    "SpecExtend",
    "spec_extend",
    "into_boxed_slice",
    "alloc::alloc",
    "alloc_impl",
    "grow_impl",
    "shrink_impl",
    "realloc_nonnull",
    "__rust",
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
        if out.len() == 6 {
            break;
        }
    }
    if out.is_empty() {
        "<unknown>".to_string()
    } else {
        out.join(" \u{2190} ")
    }
}

// SAFETY: delegates every request to `System`; the extra work is a relaxed counter and (in
// profile mode) a re-entrancy-guarded backtrace capture.
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

static PKT: AtomicUsize = AtomicUsize::new(0);

struct CountSink;

static SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("bytes"), OfferDesc::any("flac")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "buffercountsink",
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

impl Element for CountSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while inputs.pop().is_some() {
            PKT.fetch_add(1, Ordering::Relaxed);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

fn map_format(f: SampleFormat) -> Option<FlacFmt> {
    Some(match f {
        SampleFormat::S16 => FlacFmt::S16,
        SampleFormat::S24 => FlacFmt::S24,
        SampleFormat::S32 => FlacFmt::S32,
        _ => return None,
    })
}

fn main() {
    let inp = std::env::args().nth(1).expect("usage: wav_to_flac_alloc_check <in.wav>");
    if std::env::var_os("PF_ALLOC_PROFILE").is_some() {
        SAMPLE_EVERY.store(
            std::env::var("PF_ALLOC_SAMPLE").ok().and_then(|s| s.parse().ok()).unwrap_or(1),
            Ordering::Relaxed,
        );
    }

    let t0 = ALLOCS.load(Ordering::Relaxed);
    let head = {
        let mut f = std::fs::File::open(&inp).expect("open input");
        let mut buf = vec![0u8; 8192];
        let n = f.read(&mut buf).expect("read head");
        buf.truncate(n);
        buf
    };
    let fmt = parse_wav_header(&head).expect("parse WAV header").format;
    let flac_fmt = map_format(fmt.format).expect("integer-PCM sample format");

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&inp));
    let wav = p.add(WavParse::new());
    let enc = p.add(FlacEnc::new(fmt.sample_rate, fmt.channels as u32, flac_fmt));
    let snk = p.add(CountSink);
    p.link((src, "src"), (wav, "sink")).unwrap();
    p.link((wav, "src"), (enc, "sink")).unwrap();
    p.link((enc, "src"), (snk, "sink")).unwrap();
    let tap = p.tap_handle();
    let t1 = ALLOCS.load(Ordering::Relaxed);

    if std::env::var_os("PF_ALLOC_PROFILE").is_some() {
        SAMPLING.store(true, Ordering::Relaxed);
    }
    p.run().expect("pipeline run");
    SAMPLING.store(false, Ordering::Relaxed);
    let t2 = ALLOCS.load(Ordering::Relaxed);

    let s_wav = tap.snapshot(wav).unwrap_or_default();
    let s_enc = tap.snapshot(enc).unwrap_or_default();
    let bufs = s_enc.buffers_out.max(1) as f64;

    println!("input: {inp}  ({} Hz, {} ch, {:?})", fmt.sample_rate, fmt.channels, fmt.format);
    println!("wavparse: {:>8} buffers out, {:>12} B out", s_wav.buffers_out, s_wav.bytes_out);
    println!("flacenc : {:>8} buffers out, {:>12} B out", s_enc.buffers_out, s_enc.bytes_out);
    println!("PHASE                allocs   per emitted FLAC buffer");
    println!("construct+link  {:>10}   (one-time)", t1 - t0);
    println!("stream          {:>10}   {:>8.3}", t2 - t1, (t2 - t1) as f64 / bufs);
    println!("TOTAL           {:>10}   {:>8.3}", t2 - t0, (t2 - t0) as f64 / bufs);
    println!(
        "peak live heap: {:.2} MiB   total bytes allocated: {:.2} MiB",
        PEAK.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
        BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
    );
    println!(
        "SUMMARY frames={} pcmbufs={} setup={} stream={}",
        s_enc.buffers_out,
        s_wav.buffers_out,
        t1 - t0,
        t2 - t1
    );

    if let Some(map) = STACKS.lock().unwrap().take() {
        let every = SAMPLE_EVERY.load(Ordering::Relaxed);
        let mut rows: Vec<(String, (usize, usize))> = map.into_iter().collect();
        rows.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        println!("\nstream-phase allocation sites (sampled 1/{every}):");
        for (i, (site, (c, sz))) in rows.iter().enumerate() {
            if i < 25 {
                let short: String = site
                    .split(" \u{2190} ")
                    .map(|f| f.split_once('<').map_or(f, |(h, _)| h))
                    .collect::<Vec<_>>()
                    .join(" \u{2190} ");
                println!(
                    "  {:>8}  {:>7.3}/buf  {:>8} B avg  {short}",
                    c * every,
                    (c * every) as f64 / bufs,
                    sz / c.max(&1)
                );
            }
            println!("SITE\t{}\t{}\t{}", c * every, sz / c.max(&1), site);
        }
    }
}
