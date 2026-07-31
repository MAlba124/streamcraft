//! `bench_scan` — the pf-tags scanner's standing performance regression tool.
//!
//!     cargo run --release -p pf-tags --example bench_scan -- ~/Music
//!     cargo run --release -p pf-tags --example bench_scan --features io-uring -- ~/Music
//!     cargo run --release -p pf-tags --example bench_scan -- --cold ~/Music
//!     cargo run --release -p pf-tags --example bench_scan -- --once ~/Music   # for `perf record`
//!
//! Walks a root, keeps the audio extensions, and times the scan the two ways that actually
//! separate designs:
//!
//! * **warm** (default) — every byte already in the page cache, so the wall clock is *parsing
//!   plus syscall overhead* and nothing else. This is the number that moves when the sniffer,
//!   the Ogg tail walk or an ID3 transcode gets cheaper.
//! * **`--cold`** — every file evicted from the page cache before each timed pass, so the wall
//!   clock is *storage latency at whatever queue depth the reactor manages to keep*. This is
//!   the number the whole reactor design argument rests on: `SyncReactor` issues one
//!   `pread` at a time per thread, `IoUringReactor` keeps `in_flight` of them outstanding, and
//!   only a cold cache can tell them apart.
//!
//! Both modes prime with one untimed pass (warm: to fill the page cache and the allocator's
//! free lists; cold: to fault the binary in and settle the CPU) and then run [`PASSES`] timed
//! passes, reporting best and median. Best is the honest floor — it is the pass least
//! disturbed by whatever else the machine was doing — and the median says whether that floor
//! is repeatable.
//!
//! Nothing is printed inside a timed pass. Results are consumed into relaxed atomic counters
//! and the total is `black_box`ed at the end, so the optimiser cannot notice that the scan's
//! output is unused and delete the parse.
//!
//! ## Cache eviction
//!
//! `--cold` calls [`profluens_core::io::fadvise`] — core's raw `fadvise64(2)` shim
//! (`core/src/io.rs:273`, the `StreamHygiene` machinery; `pub`, and its signature
//! `(fd, offset, len, advice)` with `len == 0` meaning "to end of file" is exactly what is
//! wanted here) — with [`POSIX_FADV_DONTNEED`] on every file. `DONTNEED` drops **clean** cached
//! pages only (`core/src/io.rs:230-236`), which is why this is safe on a read-only music
//! library: there is nothing dirty to lose, and a file that somebody else holds mapped simply
//! keeps its pages. Eviction happens outside the timed region.

// App-side harness: the directory walk, the eviction `File::open` and the argument parsing are
// one-time setup outside any pipeline — the exception `clippy.toml`'s header names, the same
// one `examples/pftags.rs` and `play/src/head.rs` take.
#![allow(clippy::disallowed_methods)]

use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use pf_tags::{scan_parallel, Format, ScanConfig, Scanner};

/// Extensions worth handing to the scanner. Deliberately the *file name* filter a library
/// indexer would use, not a content sniff: the point is to measure the scan, not a pre-pass.
const AUDIO_EXT: &[&str] = &[
    "mp3", "flac", "ogg", "oga", "opus", "spx", "m4a", "m4b", "mp4", "wav", "aif", "aiff",
    "aifc", "ape", "wv", "mpc", "webm",
];

/// Timed passes per configuration. Five is enough for a stable median without the run
/// drifting into a different thermal regime part-way through.
const PASSES: usize = 5;

/// Warm thread counts. 1 exercises [`Scanner`] directly (no threading at all), the rest go
/// through [`scan_parallel`].
const WARM_THREADS: &[usize] = &[1, 2, 4, 8];

/// Cold thread counts: the depth-1 floor and the depth-8 ceiling. The middle points say
/// nothing a cold run does not already say at these two.
const COLD_THREADS: &[usize] = &[1, 8];

fn main() {
    let mut root: Option<PathBuf> = None;
    let mut cold = false;
    let mut once = false;
    let mut cfg = ScanConfig::default();
    // `None` until `--passes` says otherwise: the sweep wants [`PASSES`] repeats for a
    // median, `--once` wants exactly one unless a profiler asked for more.
    let mut passes: Option<usize> = None;
    let mut thread_list: Option<Vec<usize>> = None;
    let mut evict_only = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--cold" => cold = true,
            "--once" => once = true,
            "--evict" => evict_only = true,
            "--in-flight" => {
                cfg.in_flight = args.next().and_then(|v| v.parse().ok()).unwrap_or(cfg.in_flight);
            }
            "--tail" => {
                cfg.tail_len = args.next().and_then(|v| v.parse().ok()).unwrap_or(cfg.tail_len);
            }
            "--passes" => passes = args.next().and_then(|v| v.parse().ok()),
            // `--threads 1,8` overrides the mode's default sweep. Interleaving two builds
            // A/B/A/B at one thread count is the only way to compare them on a box that
            // drifts with temperature, so a single-configuration run has to be expressible.
            "--threads" => {
                if let Some(v) = args.next() {
                    let list: Vec<usize> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                    if !list.is_empty() {
                        thread_list = Some(list);
                    }
                }
            }
            "-h" | "--help" => return usage(),
            other => root = Some(PathBuf::from(other)),
        }
    }
    let Some(root) = root else { return usage() };

    let mut paths = Vec::new();
    collect(&root, &mut paths);
    // Readdir order is not stable across runs; sorting makes every pass and every build see
    // the same sequence, which matters when comparing two binaries file-for-file.
    paths.sort();
    if paths.is_empty() {
        println!("no audio files under {}", root.display());
        return;
    }

    // `--evict`: drop the library and report what the kernel gave back, so "the cold mode is
    // really cold" is a measured claim and not an inference from the wall clock.
    if evict_only {
        let before = cached_kib();
        let t0 = Instant::now();
        evict(&paths);
        println!(
            "evicted {} file(s) in {:.3} s   Cached: {} -> {} KiB  (freed {} KiB)",
            paths.len(),
            t0.elapsed().as_secs_f64(),
            before,
            cached_kib(),
            before.saturating_sub(cached_kib()),
        );
        return;
    }

    let acc = Acc::default();

    // `--once`: single-threaded passes and nothing else — no priming pass, no sweep, no
    // table. This is the profiler's and the allocation tracer's entry point, where every
    // sample and every `malloc` in the process should belong to the scan itself.
    //
    // `--passes N` repeats the pass, because a sampling profiler needs a workload longer than
    // a scheduling quantum: one warm pass over this library is ~80 ms, which at 499 Hz is
    // ~40 samples — noise. `--once --passes 200` is ~16 s and ~8000 samples of the same
    // steady state. A single `--once` is still the right shape for a heap capture, where one
    // pass is exactly the "startup, then flat" story worth reading.
    if once {
        let n = passes.unwrap_or(1).max(1);
        let mut total = Duration::ZERO;
        let rchar0 = rchar();
        for _ in 0..n {
            total += pass_single(&paths, cfg, &acc);
        }
        let read = rchar().saturating_sub(rchar0);
        let files = acc.files.load(Ordering::Relaxed);
        println!(
            "{} files x {n} pass(es)  {:.3} s  {:.0} files/sec  {:.1} KiB read/file  checksum {}",
            paths.len(),
            total.as_secs_f64(),
            files as f64 / total.as_secs_f64().max(f64::MIN_POSITIVE),
            read as f64 / (n as f64 * paths.len() as f64 * 1024.0),
            std::hint::black_box(acc.checksum()),
        );
        return;
    }

    let passes = passes.unwrap_or(PASSES).max(1);
    let total_bytes: u64 = paths.iter().filter_map(|p| p.metadata().ok()).map(|m| m.len()).sum();
    println!("pf-tags bench_scan");
    println!("root      : {}", root.display());
    println!("reactor   : {:?}", Scanner::new(cfg).reactor_kind());
    println!("mode      : {}   passes {passes}", if cold { "cold" } else { "warm" });
    println!("in_flight : {}   tail {} B", cfg.slots(), cfg.tail_len);
    println!(
        "files     : {}   {:.2} GiB",
        paths.len(),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("by ext    : {}", ext_census(&paths));

    // The priming pass. Untimed, and the only pass whose sniffed-format census is printed:
    // it is a property of the library, not of the run.
    let prime = Acc::default();
    pass_single(&paths, cfg, &prime);
    println!("by format : {}", prime.format_census());
    println!(
        "art       : {} picture(s), {:.2} MiB, {} file(s) over the {} KiB slot",
        prime.pics.load(Ordering::Relaxed),
        prime.pic_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
        prime.big_art.load(Ordering::Relaxed),
        pf_tags::DEFAULT_PREFIX / 1024,
    );

    let threads: &[usize] =
        thread_list.as_deref().unwrap_or(if cold { COLD_THREADS } else { WARM_THREADS });
    println!("\n{:>8}{:>11}{:>11}{:>11}{:>11}   passes (ms)", "threads", "best ms", "med ms", "best f/s", "med f/s");
    for &t in threads {
        let mut times: Vec<Duration> = Vec::with_capacity(passes);
        for _ in 0..passes {
            if cold {
                evict(&paths);
            }
            let dt = if t == 1 {
                pass_single(&paths, cfg, &acc)
            } else {
                pass_parallel(&paths, t, cfg, &acc)
            };
            times.push(dt);
        }
        report(t, paths.len(), &times);
    }

    println!("\nchecksum  : {}", std::hint::black_box(acc.checksum()));
}

fn usage() {
    println!(
        "usage: bench_scan [--cold] [--once] [--passes N] [--threads 1,2,4,8] [--in-flight N] [--tail B] <dir>"
    );
}

/// One pass on this thread, through [`Scanner`] directly. The scanner (and therefore its
/// pool) is built inside the timed region on purpose: [`scan_parallel`] builds its pool and
/// its per-thread scanners inside its own call too, so timing them the same way is the only
/// way the 1-thread row is comparable with the others.
fn pass_single(paths: &[PathBuf], cfg: ScanConfig, acc: &Acc) -> Duration {
    let t0 = Instant::now();
    let mut sc = Scanner::new(cfg);
    sc.scan(paths.iter(), |out| acc.take(out));
    t0.elapsed()
}

/// One pass across `threads` workers. [`scan_parallel`] consumes its path list, so the clone
/// is unavoidable — it happens before the clock starts.
fn pass_parallel(paths: &[PathBuf], threads: usize, cfg: ScanConfig, acc: &Acc) -> Duration {
    let owned = paths.to_vec();
    let t0 = Instant::now();
    scan_parallel(owned, threads, cfg, |out| acc.take(out));
    t0.elapsed()
}

fn report(threads: usize, files: usize, times: &[Duration]) {
    let mut ms: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1e3).collect();
    ms.sort_by(f64::total_cmp);
    let best = ms[0];
    let med = ms[ms.len() / 2];
    let fps = |m: f64| files as f64 / (m / 1e3).max(f64::MIN_POSITIVE);
    // Passes in the order they ran, not sorted: a monotonically rising row is a thermal
    // story, a falling one is a cache story, and the summary statistics hide both.
    let raw: Vec<String> =
        times.iter().map(|d| format!("{:.1}", d.as_secs_f64() * 1e3)).collect();
    println!(
        "{threads:>8}{best:>11.2}{med:>11.2}{:>11.0}{:>11.0}   {}",
        fps(best),
        fps(med),
        raw.join(" ")
    );
}

/// Drop every one of these files from the page cache.
///
/// `POSIX_FADV_DONTNEED` on `[0, EOF)` per file, via core's raw `fadvise64(2)` shim. Failures
/// are ignored: eviction is advice, and a file that will not drop shows up as a fast pass, not
/// as a wrong one.
fn evict(paths: &[PathBuf]) {
    for p in paths {
        let Ok(f) = std::fs::File::open(p) else { continue };
        let _ = profluens_core::io::fadvise(
            f.as_raw_fd(),
            0,
            0,
            profluens_core::io::POSIX_FADV_DONTNEED,
        );
    }
}

/// `rchar` from `/proc/self/io`: bytes this process has obtained from read syscalls, whether
/// they came from the disk or the page cache.
///
/// The deterministic counterpart to the wall clock. The reactor reads
/// [`Memory::capacity`](profluens_core::memory::Memory::capacity) bytes per op — not the
/// number the planner asked for — so any change to how buffers are sized or recycled can
/// silently move the amount of IO the scan does. This number catches that where a warm
/// benchmark never would: it is identical run to run, and it is the quantity a *cold* scan
/// pays for.
///
/// **Reads `0` under the `io-uring` feature**, and that is not a bug: `rchar` counts bytes
/// returned by the `read`/`pread` syscalls, and an `IORING_OP_READ` completion never goes
/// through them. Compare buffer-sizing changes on the default (`SyncReactor`) build, where the
/// two reactors issue byte-for-byte the same reads.
fn rchar() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/io") else { return 0 };
    s.lines()
        .find_map(|l| l.strip_prefix("rchar:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

/// `Cached:` from `/proc/meminfo`, in KiB — the page-cache total the eviction is supposed to
/// shrink. Zero if the file cannot be read (a kernel without procfs); the caller only ever
/// prints it.
fn cached_kib() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/meminfo") else { return 0 };
    s.lines()
        .find_map(|l| l.strip_prefix("Cached:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// What a pass accumulates. Relaxed atomics because the workers share one `emit`, and because
/// the only thing that must be true of these numbers is that the compiler cannot prove them
/// dead.
#[derive(Default)]
struct Acc {
    files: AtomicU64,
    failed: AtomicU64,
    tags: AtomicU64,
    pics: AtomicU64,
    pic_bytes: AtomicU64,
    /// Pictures past the pool's slot size — the population that legitimately reaches
    /// `Pool::acquire_exact`'s oversized fallback, so an allocation audit has something to
    /// reconcile its one sanctioned site against.
    big_art: AtomicU64,
    ns: AtomicU64,
    fmt: [AtomicU64; FORMATS.len()],
}

impl Acc {
    fn take(&self, out: pf_tags::ScanOutcome<'_>) {
        match out.result {
            Err(_) => {
                self.failed.fetch_add(1, Ordering::Relaxed);
            }
            Ok(r) => {
                self.files.fetch_add(1, Ordering::Relaxed);
                self.fmt[fmt_index(r.format)].fetch_add(1, Ordering::Relaxed);
                self.tags.fetch_add(r.tags.len() as u64, Ordering::Relaxed);
                let pics = r.tags.pictures();
                self.pics.fetch_add(pics.len() as u64, Ordering::Relaxed);
                for p in pics {
                    self.pic_bytes.fetch_add(p.data.len() as u64, Ordering::Relaxed);
                    if p.data.len() > pf_tags::DEFAULT_PREFIX {
                        self.big_art.fetch_add(1, Ordering::Relaxed);
                    }
                }
                self.ns.fetch_add(r.props.duration_ns.unwrap_or(0), Ordering::Relaxed);
            }
        }
    }

    /// One number that depends on every field the scan produced, so no part of the parse can
    /// be proved dead.
    fn checksum(&self) -> u64 {
        self.files.load(Ordering::Relaxed)
            ^ self.failed.load(Ordering::Relaxed).rotate_left(8)
            ^ self.tags.load(Ordering::Relaxed).rotate_left(16)
            ^ self.pics.load(Ordering::Relaxed).rotate_left(24)
            ^ self.pic_bytes.load(Ordering::Relaxed).rotate_left(32)
            ^ self.ns.load(Ordering::Relaxed).rotate_left(40)
    }

    fn format_census(&self) -> String {
        let mut s = String::new();
        for (i, name) in FORMATS.iter().enumerate() {
            let n = self.fmt[i].load(Ordering::Relaxed);
            if n > 0 {
                s.push_str(&format!("{name} {n}  "));
            }
        }
        s.trim_end().to_string()
    }
}

const FORMATS: [&str; 14] = [
    "flac", "mp3", "mp4", "ogg/opus", "ogg/vorbis", "ogg/flac", "wav", "mkv", "unknown",
    "ogg/speex", "aiff", "ape", "wavpack", "musepack",
];

// Indices 0..=8 are frozen: `format_census` prints in index order and skips empty slots, so
// appending keeps a run over an existing library byte-identical.
fn fmt_index(f: Format) -> usize {
    match f {
        Format::Flac => 0,
        Format::Mp3 => 1,
        Format::Mp4 => 2,
        Format::OggOpus => 3,
        Format::OggVorbis => 4,
        Format::OggFlac => 5,
        Format::Wav => 6,
        Format::Mkv => 7,
        Format::Unknown => 8,
        Format::OggSpeex => 9,
        Format::Aiff => 10,
        Format::Ape => 11,
        Format::WavPack => 12,
        Format::Musepack => 13,
    }
}

fn ext_census(paths: &[PathBuf]) -> String {
    let mut counts: Vec<(usize, usize)> = AUDIO_EXT.iter().map(|_| (0, 0)).collect();
    for (i, e) in AUDIO_EXT.iter().enumerate() {
        counts[i] = (i, paths.iter().filter(|p| has_ext(p, e)).count());
    }
    counts.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    let mut s = String::new();
    for (i, n) in counts {
        if n > 0 {
            s.push_str(&format!("{} {n}  ", AUDIO_EXT[i]));
        }
    }
    s.trim_end().to_string()
}

fn has_ext(p: &Path, ext: &str) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// Recursive walk, std only, audio extensions only. Symlinked directories are not followed —
/// a library with a self-referential link should not hang the benchmark.
fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    let Ok(md) = std::fs::symlink_metadata(path) else { return };
    if md.is_dir() {
        let Ok(rd) = std::fs::read_dir(path) else { return };
        for e in rd.flatten() {
            collect(&e.path(), out);
        }
    } else if md.is_file() && AUDIO_EXT.iter().any(|e| has_ext(path, e)) {
        out.push(path.to_path_buf());
    }
}
