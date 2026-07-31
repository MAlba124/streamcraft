//! `pftags` — scan files and directories and print what the tag scanner found.
//!
//!     cargo run --release -p pf-tags --example pftags -- ~/music
//!     cargo run --release -p pf-tags --example pftags -- -j 8 --in-flight 24 a.flac dir/
//!
//! The human-validation tool for the engine: point it at a real library and check that the
//! formats, durations and tags match what the files actually contain — and watch the
//! files/sec, which is what the whole reactor design is for.
//!
//! Directories are walked recursively. Arguments are paths; `-j N` sets worker threads and
//! `--in-flight N` the per-thread reader depth.

// App-side CLI: argument collection and the directory walk are one-time setup outside any
// pipeline, the exception `clippy.toml`'s header names (cf. `play/src/head.rs`).
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use pf_tags::{scan_parallel, ScanConfig};

fn main() {
    let mut cfg = ScanConfig::default();
    let mut threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut all = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-a" | "--all" => all = true,
            "-j" | "--threads" => {
                threads = args.next().and_then(|v| v.parse().ok()).unwrap_or(threads)
            }
            "--in-flight" => {
                cfg.in_flight = args.next().and_then(|v| v.parse().ok()).unwrap_or(cfg.in_flight)
            }
            "--tail" => {
                cfg.tail_len = args.next().and_then(|v| v.parse().ok()).unwrap_or(cfg.tail_len)
            }
            "-h" | "--help" => return usage(),
            other => roots.push(PathBuf::from(other)),
        }
    }
    if roots.is_empty() {
        return usage();
    }

    let mut paths = Vec::new();
    for r in &roots {
        collect(r, &mut paths);
    }
    let total = paths.len();

    let files = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let art = AtomicU64::new(0);
    let art_bytes = AtomicU64::new(0);
    let audio_ns = AtomicU64::new(0);

    let t0 = Instant::now();
    scan_parallel(paths, threads, cfg, |out| {
        let name = out.path.display();
        match out.result {
            Err(e) => {
                failed.fetch_add(1, Ordering::Relaxed);
                // One `println!` per file: the macro locks stdout for the whole call, so a
                // multi-line record never interleaves with another worker's.
                println!("{name}\n  error: {e:?}");
            }
            Ok(r) => {
                files.fetch_add(1, Ordering::Relaxed);
                let dur = match r.props.duration_ns {
                    Some(ns) => {
                        audio_ns.fetch_add(ns, Ordering::Relaxed);
                        format!(
                            "{:.2}s {}",
                            ns as f64 / 1e9,
                            if r.props.duration_exact { "exact" } else { "estimated" }
                        )
                    }
                    None => "-".to_string(),
                };
                let rate = r.props.sample_rate.map_or_else(|| "-".into(), |v| format!("{v} Hz"));
                let ch = r.props.channels.map_or_else(|| "-".into(), |v| format!("{v}ch"));
                let pics = r.tags.pictures().len() as u64;
                let pbytes: u64 = r.tags.pictures().iter().map(|p| p.data.len() as u64).sum();
                art.fetch_add(pics, Ordering::Relaxed);
                art_bytes.fetch_add(pbytes, Ordering::Relaxed);
                // `--all` dumps every tag in file order (a key may repeat); the default is the
                // three fields a library view shows.
                let body = if all {
                    let mut s = String::new();
                    for (k, v) in r.tags.iter() {
                        s.push_str(&format!("\n  {k:<22} {v}"));
                    }
                    for p in r.tags.pictures() {
                        s.push_str(&format!("\n  {:<22} {} B", p.mime, p.data.len()));
                    }
                    s
                } else {
                    format!(
                        "\n  TITLE  {}\n  ARTIST {}\n  ALBUM  {}",
                        r.tags.get("TITLE").unwrap_or("-"),
                        r.tags.get("ARTIST").unwrap_or("-"),
                        r.tags.get("ALBUM").unwrap_or("-"),
                    )
                };
                println!(
                    "{name}\n  {:<10} {:>12}  {rate} {ch}  {} B{body}\n  art    {pics} picture(s), {pbytes} B",
                    r.format.name(),
                    dur,
                    r.file.len,
                );
            }
        }
    });
    let dt = t0.elapsed();

    let ok = files.load(Ordering::Relaxed);
    let bad = failed.load(Ordering::Relaxed);
    println!("\n--- {total} path(s), {threads} thread(s), in_flight {} ---", cfg.slots());
    println!("scanned : {ok}   failed: {bad}");
    println!("pictures: {} ({:.2} MiB)", art.load(Ordering::Relaxed), art_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0));
    println!("audio   : {:.1} s of playing time", audio_ns.load(Ordering::Relaxed) as f64 / 1e9);
    println!(
        "elapsed : {:.3} s   {:.0} files/sec",
        dt.as_secs_f64(),
        (ok + bad) as f64 / dt.as_secs_f64().max(f64::MIN_POSITIVE),
    );
}

fn usage() {
    println!("usage: pftags [-j THREADS] [--in-flight N] [--tail BYTES] [-a|--all] <file|dir>...");
}

/// Recursive directory walk, std only. Symlinked directories are not followed — a music
/// library with a self-referential link should not hang the scanner.
fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    let Ok(md) = std::fs::symlink_metadata(path) else {
        out.push(path.to_path_buf()); // let the scanner report why it cannot be opened
        return;
    };
    if md.is_dir() {
        let Ok(rd) = std::fs::read_dir(path) else { return };
        for e in rd.flatten() {
            collect(&e.path(), out);
        }
    } else if md.is_file() {
        out.push(path.to_path_buf());
    }
}
