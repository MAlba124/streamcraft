//! Steady-state allocation audit for the scan path (mirrors
//! `ogg/examples/ogg_read_alloc_check.rs`).
//!
//!     cargo run --release -p pf-tags --example scan_alloc_check
//!     cargo run --release -p pf-tags --example scan_alloc_check -- /path/to/music
//!     PF_ALLOC_PROFILE=1 cargo run --release -p pf-tags --example scan_alloc_check
//!
//! With no argument it generates a directory of fixtures (WAV with inline tags, WAV with
//! tags after `data`, FLAC with comments, FLAC with cover art) and scans it **twice**: the
//! first pass warms the pool free-list, the arena chunks and the slots' path buffers, the
//! second is the measurement. The target is **0 heap allocations per file** in that second
//! pass: everything the steady state needs is either a recycled pool buffer, a bump
//! allocation in the arena, or a reused slot field.
//!
//! One documented site remains, and only for outsized files: a metadata extent larger than
//! the pool's slot size (multi-MiB cover art), which `Pool::acquire_exact` serves from a
//! right-sized heap box **by design** — outside the slot budget, so it cannot starve the
//! scan. The `art_*.flac` fixtures below sit under that threshold precisely so the number
//! this prints is the true steady state; raise `PF_ART` past the slot size to watch the one
//! allocation appear.
//!
//! `PF_ALLOC_PROFILE=1` attributes the remaining allocations to their call sites.

#![allow(clippy::disallowed_methods)] // one-shot measurement harness

use std::cell::Cell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use pf_tags::{fixture, ScanConfig, Scanner};

use std::alloc::{GlobalAlloc, Layout, System};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static SAMPLING: AtomicBool = AtomicBool::new(false);
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
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size, Ordering::Relaxed);
            PEAK.fetch_max(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
            if SAMPLING.load(Ordering::Relaxed) {
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

// SAFETY: every request is delegated to `System`; the extra work is a relaxed counter and,
// in profile mode, a re-entrancy-guarded backtrace capture.
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

/// Write the fixture set: `n` of each shape, so slot recycling is exercised. Every format
/// the engine parses appears, and each in the shape that costs it the most reads — a WAV with
/// its `INFO` after `data`, a FLAC and an MP3 with cover art, an MP4 whose `moov` trails its
/// `mdat`, an Ogg whose comment packet carries a picture, an AIFF whose `ID3 ` chunk sits
/// behind the audio, and an APEv2-family file whose trailing tag is past the prefix.
fn generate(dir: &std::path::Path, n: usize) -> Vec<PathBuf> {
    let art_len: usize =
        std::env::var("PF_ART").ok().and_then(|s| s.parse().ok()).unwrap_or(24 * 1024);
    let art = vec![0xABu8; art_len];
    let mut paths = Vec::new();
    for i in 0..n {
        let tags = [("INAM", "Inline Title"), ("IART", "An Artist"), ("IPRD", "An Album")];
        let cases: [(String, Vec<u8>); 16] = [
            (format!("inline_{i}.wav"), fixture::wav(44_100, 2, 16, 2_000, &tags, false)),
            (format!("tail_{i}.wav"), fixture::wav(44_100, 2, 16, 2_000, &tags, true)),
            (
                format!("plain_{i}.flac"),
                fixture::flac(44_100, 2, 16, 132_300, &["TITLE=T", "ARTIST=A", "ALBUM=B"], None),
            ),
            (
                format!("art_{i}.flac"),
                fixture::flac(44_100, 2, 16, 132_300, &["TITLE=T"], Some(("image/jpeg", &art))),
            ),
            (
                format!("art_{i}.mp3"),
                fixture::mp3(
                    44_100,
                    128,
                    24,
                    &[("TIT2", "T"), ("TPE1", "A"), ("TXXX", "replaygain_track_gain=-2.5 dB")],
                    Some(("image/jpeg", &art)),
                    true,
                    Some(("T", "A", "B", "2011", 3)),
                    &[("REPLAYGAIN_ALBUM_GAIN", "-2.0 dB")],
                ),
            ),
            (
                format!("fast_{i}.m4a"),
                fixture::m4a(
                    44_100,
                    2,
                    132_300,
                    &[("©nam", "T"), ("©ART", "A"), ("©alb", "B")],
                    Some(3),
                    Some(("image/jpeg", &art)),
                    true,
                ),
            ),
            (
                format!("tail_{i}.m4a"),
                fixture::m4a(48_000, 2, 144_000, &[("©nam", "T"), ("©ART", "A")], Some(1), None, false),
            ),
            (
                format!("art_{i}.opus"),
                fixture::ogg_opus(
                    2,
                    312,
                    48_000,
                    &["TITLE=T", "ARTIST=A", "ALBUM=B"],
                    Some(("image/jpeg", &art)),
                    48_000 * 3 + 312,
                ),
            ),
            (
                format!("plain_{i}.ogg"),
                fixture::ogg_vorbis(2, 44_100, &["TITLE=T", "ARTIST=A", "ALBUM=B"], 44_100 * 3),
            ),
            (
                format!("art_{i}.spx"),
                fixture::ogg_speex(
                    16_000,
                    1,
                    &["TITLE=T", "ARTIST=A", "ALBUM=B"],
                    Some(("image/jpeg", &art)),
                    16_000 * 3,
                ),
            ),
            // The AIFF that costs the most: an `ID3 ` chunk behind the audio, so the chunk
            // walk asks for a positioned extent.
            (
                format!("tail_{i}.aiff"),
                fixture::aiff_id3(
                    44_100,
                    2,
                    44_100,
                    &[("NAME", "T"), ("AUTH", "A")],
                    &[("TIT2", "T"), ("TPE1", "A"), ("TALB", "B")],
                    true,
                ),
            ),
            // The APEv2 family: prefix plus one tail read each, no extent. Padded past the
            // pool slot so the tail read really happens rather than being served by the prefix.
            (
                format!("tagged_{i}.ape"),
                [
                    fixture::ape_file(3990, 44_100, 2, 73_728, 3, 4_644),
                    vec![0x71u8; 8_000],
                    fixture::ape_tag(&[("Title", "T"), ("Artist", "A"), ("Album", "B")]),
                    fixture::id3v1("T", "A", "B", "2001", 4),
                ]
                .concat(),
            ),
            (
                format!("tagged_{i}.wv"),
                [
                    fixture::wavpack(
                        44_100,
                        2,
                        88_200,
                        &[fixture::wv_sub_block(0x0A, &vec![0x33u8; 8_000])],
                    ),
                    fixture::ape_tag(&[
                        ("Title", "T"),
                        ("Artist", "A"),
                        ("Replaygain_Track_Gain", "-2.5 dB"),
                    ]),
                    fixture::id3v1("T", "A", "B", "2004", 6),
                ]
                .concat(),
            ),
            (
                format!("tagged_{i}.mpc"),
                [
                    fixture::mpc_sv8(0, 2, 88_200, 0, &[]),
                    vec![0x2Eu8; 8_000],
                    fixture::ape_tag(&[("Title", "T"), ("Artist", "A"), ("Album", "B")]),
                ]
                .concat(),
            ),
            // Matroska twice, because its two layouts cost different numbers of reads: the
            // front-loaded one every WebM in the wild uses (one op), and the one whose Tags
            // and Attachments trail the frames (two, the second sized off the first's headers).
            // Both must still steady-state at zero allocations.
            (
                format!("front_{i}.webm"),
                fixture::mkv(48_000.0, 2, 3_000.0, &[("TITLE", "T"), ("ARTIST", "A")], None),
            ),
            (
                format!("trailing_{i}.mkv"),
                fixture::mkv_trailing(
                    44_100.0,
                    2,
                    2_500.0,
                    &[("TITLE", "T"), ("ARTIST", "A"), ("ALBUM", "B")],
                    Some(("image/jpeg", &art)),
                    64 * 1024,
                ),
            ),
        ];
        for (name, bytes) in cases {
            let p = dir.join(name);
            std::fs::write(&p, &bytes).expect("write fixture");
            paths.push(p);
        }
    }
    paths
}

fn walk(path: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(md) = std::fs::symlink_metadata(path) else { return };
    if md.is_dir() {
        let Ok(rd) = std::fs::read_dir(path) else { return };
        for e in rd.flatten() {
            walk(&e.path(), out);
        }
    } else if md.is_file() {
        out.push(path.to_path_buf());
    }
}

/// One pass over `paths`, returning `(allocations, files, tags seen, picture bytes)`.
fn pass(scanner: &mut Scanner, paths: &[PathBuf], profile: bool) -> (usize, usize, usize, usize) {
    let (mut files, mut tags, mut pic) = (0usize, 0usize, 0usize);
    let before = ALLOCS.load(Ordering::Relaxed);
    if profile {
        SAMPLING.store(true, Ordering::Relaxed);
    }
    scanner.scan(paths.iter().map(|p| p.as_path()), |out| {
        if let Ok(r) = out.result {
            files += 1;
            tags += r.tags.len();
            pic += r.tags.pictures().iter().map(|p| p.data.len()).sum::<usize>();
            std::hint::black_box(r.props.duration_ns);
        }
    });
    SAMPLING.store(false, Ordering::Relaxed);
    (ALLOCS.load(Ordering::Relaxed) - before, files, tags, pic)
}

fn main() {
    let arg = std::env::args().nth(1);
    let profile = std::env::var_os("PF_ALLOC_PROFILE").is_some();
    let n: usize = std::env::var("PF_FIXTURES").ok().and_then(|s| s.parse().ok()).unwrap_or(32);

    let (dir, paths) = match arg {
        Some(d) => {
            let root = PathBuf::from(d);
            let mut v = Vec::new();
            walk(&root, &mut v);
            (None, v)
        }
        None => {
            let dir = fixture::scratch_dir("alloccheck");
            let v = generate(&dir, n);
            (Some(dir), v)
        }
    };
    assert!(!paths.is_empty(), "no files to scan");

    let cfg = ScanConfig::default();
    let mut scanner = Scanner::new(cfg);
    println!("files: {}   reactor: {:?}   in_flight: {}", paths.len(), scanner.reactor_kind(), cfg.slots());

    // Pass 1 warms every reusable structure: the pool free-list, the arena chunks, the slot
    // path buffers, the submission/completion vectors.
    let (warm, _, _, _) = pass(&mut scanner, &paths, false);
    let t0 = Instant::now();
    let (steady, files, tags, pic) = pass(&mut scanner, &paths, profile);
    let dt = t0.elapsed();

    let f = files.max(1) as f64;
    println!("tags seen: {tags}   picture bytes: {pic}");
    println!("PHASE                allocs    per file");
    println!("warm (pass 1)   {warm:>10}   {:>9.3}", warm as f64 / f);
    println!("steady (pass 2) {steady:>10}   {:>9.3}", steady as f64 / f);
    println!(
        "peak live heap: {:.2} MiB   total allocated: {:.2} MiB",
        PEAK.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
        BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
    );
    println!(
        "elapsed: {:.3} s   {:.0} files/sec",
        dt.as_secs_f64(),
        f / dt.as_secs_f64().max(f64::MIN_POSITIVE)
    );
    println!("SUMMARY files={files} warm={warm} steady={steady}");
    report(f);

    if let Some(d) = dir {
        std::fs::remove_dir_all(d).ok();
    }
}

/// Ranked attribution table + machine-readable `SITE` lines.
fn report(n: f64) {
    if let Some(map) = STACKS.lock().unwrap().take() {
        let mut rows: Vec<(String, (usize, usize))> = map.into_iter().collect();
        rows.sort_by_key(|(_, (c, _))| std::cmp::Reverse(*c));
        println!("\nsteady-phase allocation sites:");
        for (i, (site, (c, sz))) in rows.iter().enumerate() {
            if i < 20 {
                let short: String = site
                    .split(" \u{2190} ")
                    .map(|f| f.split_once('<').map_or(f, |(h, _)| h))
                    .collect::<Vec<_>>()
                    .join(" \u{2190} ");
                println!("  {c:>8}  {:>7.3}/file  {:>8} B avg  {short}", *c as f64 / n, sz / c.max(&1));
            }
            println!("SITE\t{c}\t{}\t{site}", sz / c.max(&1));
        }
    }
}
