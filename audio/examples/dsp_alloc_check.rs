//! Steady-state allocation audit for the audio DSP hot paths — the resampler's streaming
//! `process()`, `audioconvert`'s sample-format conversion, and the BS.775 downmix — with a
//! counting global allocator and sampled backtrace attribution.
//!
//!   cargo run --release -p profluens-audio --example dsp_alloc_check
//!   PF_ALLOC_PROFILE=1 cargo run --release -p profluens-audio --example dsp_alloc_check
//!
//! Each rung is driven the way an element drives it: a fixed output buffer reused across
//! batches (`convert`/`downmix` are in-place APIs and must be *zero*-alloc), and, for the
//! resampler, one `ChannelResampler` fed realistic 1024-frame batches with the output `Vec`
//! reused. The number that matters is **allocations per batch in the steady state** — the
//! first batches legitimately grow the delay line and the output buffer to their working size.

#![allow(clippy::disallowed_methods)] // one-shot measurement harness

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use profluens_audio::{
    convert_interleaved, downmix_to_stereo, ChannelResampler, PolyphaseFilter, SampleFormat,
};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
/// Live heap bytes and their high-water mark — trading allocation *count* for retention must
/// not quietly balloon the peak.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static SAMPLING: AtomicBool = AtomicBool::new(false);
static SAMPLE_EVERY: AtomicUsize = AtomicUsize::new(1);
static STACKS: Mutex<Option<HashMap<String, (usize, usize)>>> = Mutex::new(None);

thread_local! {
    /// Re-entrancy guard: capturing/formatting a backtrace allocates, and those allocations must
    /// neither be counted nor recursively sampled.
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

/// Pure allocation plumbing — frames that say *how* memory was obtained, never *who* wanted it.
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
    "extend_from_slice",
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

/// Frames whose *whole* name is allocator plumbing (too short to match by substring safely).
const PLUMBING_EXACT: &[&str] = &["alloc", "realloc", "alloc_zeroed", "allocate", "grow", "shrink"];

/// The interesting part of the current backtrace: the first few frames below the allocator shim,
/// with generic `Vec`/`RawVec` plumbing dropped so the key names the *caller*.
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
    if out.is_empty() {
        "<unknown>".to_string()
    } else {
        out.join(" \u{2190} ")
    }
}

// SAFETY: delegates every request to `System`; the extra work is a relaxed counter and (in profile
// mode) a re-entrancy-guarded backtrace capture.
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

/// Run `f` `batches` times and report allocations *after* a warm-up window, so one-time buffer
/// growth is excluded and what remains is the genuine steady-state rate.
fn audit(label: &str, warmup: usize, batches: usize, mut f: impl FnMut(usize)) {
    for i in 0..warmup {
        f(i);
    }
    let base = ALLOCS.load(Ordering::Relaxed);
    let bytes = BYTES.load(Ordering::Relaxed);
    for i in 0..batches {
        f(warmup + i);
    }
    let n = ALLOCS.load(Ordering::Relaxed) - base;
    let b = BYTES.load(Ordering::Relaxed) - bytes;
    println!(
        "{label:<34} {:>8.3} allocs/batch  {:>10.0} B/batch   ({n} over {batches} batches)",
        n as f64 / batches as f64,
        b as f64 / batches as f64,
    );
}

const FRAMES: usize = 1024; // a typical decoder output batch

fn main() {
    if std::env::var_os("PF_ALLOC_PROFILE").is_some() {
        SAMPLE_EVERY.store(
            std::env::var("PF_ALLOC_SAMPLE").ok().and_then(|s| s.parse().ok()).unwrap_or(1),
            Ordering::Relaxed,
        );
        SAMPLING.store(true, Ordering::Relaxed);
    }
    println!("profluens-audio DSP — steady-state allocations per {FRAMES}-frame batch\n");

    // --- resampler: one streaming ChannelResampler, output Vec reused across batches ---------
    for (inr, outr, label) in [
        (44_100u32, 48_000u32, "resample 44100->48000"),
        (48_000, 44_100, "resample 48000->44100"),
        (48_000, 24_000, "resample 48000->24000"),
    ] {
        let filter = PolyphaseFilter::design(inr, outr, 32, 9.0);
        let mut r = ChannelResampler::new(filter);
        let input: Vec<f32> =
            (0..FRAMES).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let mut out: Vec<f32> = Vec::new();
        audit(label, 64, 512, |_| {
            out.clear();
            r.process(&input, &mut out);
            std::hint::black_box(&out);
        });
    }

    // --- audioconvert / downmix: in-place APIs into a reused buffer, must be zero-alloc -------
    let pcm: Vec<u8> = (0..FRAMES * 2 * 4).map(|i| (i * 37 % 251) as u8).collect();
    let mut obuf = vec![0u8; FRAMES * 8 * 4];
    for (from, to) in [
        (SampleFormat::S16, SampleFormat::F32),
        (SampleFormat::F32, SampleFormat::S16),
        (SampleFormat::S24, SampleFormat::S16),
    ] {
        let n = (pcm.len() / (from.bytes() * 2)) * from.bytes() * 2;
        let label = format!("convert {from:?}->{to:?} 2ch");
        audit(&label, 8, 512, |_| {
            convert_interleaved(from, to, 2, &pcm[..n], &mut obuf).expect("convert");
            std::hint::black_box(&obuf);
        });
    }
    for (fmt, ch) in [(SampleFormat::S16, 6usize), (SampleFormat::F32, 6)] {
        let stride = fmt.bytes() * ch;
        let n = (pcm.len() / stride) * stride;
        let label = format!("downmix {fmt:?} {ch}ch->2ch");
        audit(&label, 8, 512, |_| {
            downmix_to_stereo(fmt, ch, &pcm[..n], &mut obuf).expect("downmix");
            std::hint::black_box(&obuf);
        });
    }

    println!(
        "\ntotal allocations: {}   bytes: {}   peak live: {} B",
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
        PEAK.load(Ordering::Relaxed),
    );
    SAMPLING.store(false, Ordering::Relaxed);
    if let Some(map) = STACKS.lock().unwrap().take() {
        let every = SAMPLE_EVERY.load(Ordering::Relaxed);
        let mut rows: Vec<(String, (usize, usize))> = map.into_iter().collect();
        rows.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        println!("\nallocation sites (sampled 1/{every}):");
        for (site, (n, sz)) in rows.iter().take(20) {
            let site: String = site
                .split(" \u{2190} ")
                .map(|f| f.split_once('<').map_or(f, |(head, _)| head))
                .collect::<Vec<_>>()
                .join(" \u{2190} ");
            println!("  {:>7}  {:>8} B avg  {site}", n * every, sz / n.max(&1));
        }
    }
}
