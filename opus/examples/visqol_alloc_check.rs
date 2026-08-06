//! Steady-state allocation audit for **one ViSQOL (MOS-LQO) comparison** — the perceptual metric
//! behind `transcode --target` / `--recommend`.
//!
//! ```text
//! cargo run -p pf-opus --features qa --release --example visqol_alloc_check
//! VISQOL_ALLOC_PROFILE=1 cargo run -p pf-opus --features qa --release --example visqol_alloc_check
//! ```
//!
//! Encodes the WAV fixture through libopus, decodes it back, then scores the (reference, degraded)
//! pair with ViSQOL **twice** against a counting global allocator: the first run warms the one-time
//! caches (SVR model, per-thread FFT planner, gammatone coefficients, thread-local scratch), the
//! second is the reported steady-state number — what a `--target` search pays per grid point.
//!
//! `VISQOL_ALLOC_PROFILE=1` samples backtraces (1 in `VISQOL_ALLOC_SAMPLE`, default 8) during the
//! measured run and prints the top call sites, so a regression is attributable without heaptrack.

#![allow(clippy::disallowed_methods, clippy::chunks_exact_to_as_chunks)] // one-shot harness

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use pf_opus::libopus::{LibopusDecoder, LibopusEncoder};
use pf_opus::EncoderConfig;

const RATE: u32 = 48_000;

// ---------------------------------------------------------------------------
// Counting (optionally backtrace-sampling) global allocator
// ---------------------------------------------------------------------------

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
/// Live heap bytes and their high-water mark — a bump arena trades allocation *count* for
/// retention, so a patch that flattens the count must not quietly double the peak.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static SAMPLING: AtomicBool = AtomicBool::new(false);
static SAMPLE_EVERY: AtomicUsize = AtomicUsize::new(8);
static STACKS: Mutex<Option<HashMap<String, (usize, usize)>>> = Mutex::new(None);

thread_local! {
    /// Re-entrancy guard: capturing/formatting a backtrace allocates, and those allocations must
    /// neither be counted nor recursively sampled.
    static IN_SAMPLER: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

impl Counting {
    fn note(size: usize) {
        // The guard wraps the *whole* body: formatting a backtrace allocates, and those
        // allocations must be neither counted nor recursively sampled.
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
        // Frame lines look like `12: some::function::name`; the `at path:line` lines that follow
        // are indented differently and carry no symbol.
        let Some((num, rest)) = t.split_once(": ") else { continue };
        if !num.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        frames.push(rest.trim());
    }
    // Drop everything up to and including this harness's allocator shim.
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
        out.join(" ← ")
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

// ---------------------------------------------------------------------------

fn main() {
    let wav = std::env::args().nth(1).unwrap_or_else(|| "fixtures/out/audio.wav".into());
    let (mut pcm_i16, channels) = read_wav_s16(&wav);
    // ViSQOL scores the first 30 s per point in the real tool, and the per-patch loops scale with
    // duration — so repeat the (short) fixture up to `VISQOL_ALLOC_SECS` to measure that shape.
    let want_secs: f64 =
        std::env::var("VISQOL_ALLOC_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let want = (want_secs * RATE as f64) as usize * channels;
    if pcm_i16.len() < want {
        let base = pcm_i16.clone();
        while pcm_i16.len() < want {
            pcm_i16.extend_from_slice(&base);
        }
        pcm_i16.truncate(want);
    }
    let secs = pcm_i16.len() as f64 / channels as f64 / RATE as f64;
    println!("fixture: {wav}  {channels}ch  {secs:.1}s");

    // A realistic degraded signal: the same libopus encode→decode the quality search scores.
    let bitrate: u32 =
        std::env::var("VISQOL_ALLOC_BITRATE").ok().and_then(|s| s.parse().ok()).unwrap_or(64_000);
    let recon = enc_dec(&pcm_i16, channels, bitrate);

    let mut arena = profluens_core::memory::Arena::default();
    let mut v = new_manager();

    let before = ALLOCS.load(Ordering::Relaxed);
    let cold = score(&mut v, &pcm_i16, &recon, channels, &mut arena);
    let cold_allocs = ALLOCS.load(Ordering::Relaxed) - before;
    println!("cold run: MOS-LQO {cold:.17} [{:016x}]   allocations {cold_allocs}", cold.to_bits());

    if std::env::var_os("VISQOL_ALLOC_PROFILE").is_some() {
        SAMPLE_EVERY.store(
            std::env::var("VISQOL_ALLOC_SAMPLE").ok().and_then(|s| s.parse().ok()).unwrap_or(8),
            Ordering::Relaxed,
        );
        SAMPLING.store(true, Ordering::Relaxed);
    }
    let (b_allocs, b_bytes) = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
    let base_live = LIVE.load(Ordering::Relaxed);
    let warm = score(&mut v, &pcm_i16, &recon, channels, &mut arena);
    let allocs = ALLOCS.load(Ordering::Relaxed) - b_allocs;
    let bytes = BYTES.load(Ordering::Relaxed) - b_bytes;
    SAMPLING.store(false, Ordering::Relaxed);

    println!(
        "warm run: MOS-LQO {warm:.17} [{:016x}]   allocations {allocs}   bytes {:.1} MiB   \
         peak live +{:.1} MiB",
        warm.to_bits(),
        bytes as f64 / (1024.0 * 1024.0),
        PEAK.load(Ordering::Relaxed).saturating_sub(base_live) as f64 / (1024.0 * 1024.0)
    );
    assert_eq!(cold.to_bits(), warm.to_bits(), "cold and warm scores must be bit-identical");

    if let Some(map) = STACKS.lock().unwrap().take() {
        let every = SAMPLE_EVERY.load(Ordering::Relaxed);
        let mut rows: Vec<(String, (usize, usize))> = map.into_iter().collect();
        rows.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        let shown: usize = rows.iter().map(|(_, (n, _))| n * every).sum();
        println!("\nallocation sites (sampled 1/{every}, {shown} of {allocs} attributed):");
        for (site, (n, sz)) in rows.iter().take(60) {
            // Keep the key readable: ndarray's monomorphised names are book-length.
            let site: String = site
                .split(" ← ")
                .map(|f| f.split_once('<').map_or(f, |(head, _)| head))
                .collect::<Vec<_>>()
                .join(" ← ");
            println!("  {:>8}  {:>9} B avg  {site}", n * every, sz / n.max(&1));
        }
    }
}

fn new_manager() -> visqol_rs::visqol_manager::VisqolManager<{ visqol_rs::constants::NUM_BANDS_AUDIO }>
{
    use visqol_rs::constants::DEFAULT_WINDOW_SIZE;
    use visqol_rs::variant::Variant;
    const MODEL_PATH: &str = "/tmp/sc_visqol_audio_model.txt";
    if !std::path::Path::new(MODEL_PATH).exists() {
        std::fs::write(MODEL_PATH, include_str!("../models/visqol_audio_model.txt")).expect("model");
    }
    visqol_rs::visqol_manager::VisqolManager::new(
        Variant::Fullband { model_path: MODEL_PATH.to_string() },
        DEFAULT_WINDOW_SIZE,
    )
}

/// One ViSQOL comparison, built exactly as `transcode::visqol_mos` builds it (channel 0, first 30 s,
/// `x / 32767`).
fn score<const N: usize>(
    v: &mut visqol_rs::visqol_manager::VisqolManager<N>,
    reference: &[i16],
    test: &[i16],
    channels: usize,
    arena: &mut profluens_core::memory::Arena,
) -> f64 {
    use visqol_rs::audio_signal::AudioSignal;
    let cap = RATE as usize * 30;
    let to_signal = |pcm: &[i16]| {
        let samples: Vec<f64> =
            pcm.iter().step_by(channels).take(cap).map(|&x| x as f64 / 32767.0).collect();
        AudioSignal::new(&samples, RATE)
    };
    let mut ref_sig = to_signal(reference);
    let mut deg_sig = to_signal(test);
    v.compute_results(&mut ref_sig, &mut deg_sig, arena).map(|r| r.moslqo).unwrap_or(1.0)
}

/// libopus encode → decode round trip at `bitrate`, yielding the degraded signal.
fn enc_dec(orig: &[i16], channels: usize, bitrate: u32) -> Vec<i16> {
    let cfg = EncoderConfig { bitrate_bps: bitrate, complexity: 10, ..EncoderConfig::new(channels as u8) };
    let mut enc = LibopusEncoder::new(cfg).expect("libopus encoder");
    let fb = enc.frame_bytes();
    let mut dec = LibopusDecoder::new();
    dec.set_channels(channels);
    let mut recon = Vec::<i16>::with_capacity(orig.len());
    let (mut pkt, mut p) = (Vec::new(), Vec::new());
    let mut frame = vec![0u8; fb];
    let pcm: Vec<u8> = orig.iter().flat_map(|s| s.to_le_bytes()).collect();
    for chunk in pcm.chunks(fb) {
        frame[..chunk.len()].copy_from_slice(chunk);
        frame[chunk.len()..].fill(0);
        enc.encode(&frame, &mut pkt).expect("encode");
        if dec.decode_packet_into(&pkt, &mut p).is_ok() {
            recon.extend_from_slice(&p);
        }
    }
    recon
}

/// Minimal 16-bit PCM WAV reader (the fixtures are plain RIFF/PCM) — avoids pulling `hound` in.
fn read_wav_s16(path: &str) -> (Vec<i16>, usize) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(&b[0..4] == b"RIFF" && &b[8..12] == b"WAVE", "{path} is not a RIFF/WAVE file");
    let (mut i, mut channels, mut rate) = (12usize, 0usize, 0u32);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &b[i + 8..(i + 8 + sz).min(b.len())];
        match id {
            b"fmt " => {
                channels = u16::from_le_bytes(body[2..4].try_into().unwrap()) as usize;
                rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
                assert_eq!(bits, 16, "expected 16-bit PCM");
            }
            b"data" => {
                assert_eq!(rate, RATE, "expected a {RATE} Hz fixture");
                let s = body.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
                return (s, channels);
            }
            _ => {}
        }
        i += 8 + sz + (sz & 1);
    }
    panic!("{path}: no data chunk");
}
