//! `scraft-launch` — a gst-launch-style runner for streamcraft (spec: Plugins —
//! "a gst-launch equivalent is cheap once the registry exists and invaluable for
//! debugging and bug reports").
//!
//! It aggregates every in-tree plugin crate's `register(&mut Registry)` into one
//! registry, parses a launch string into a [`Pipeline`], and drives it to EOS.
//!
//! ```text
//! scraft-launch "filesrc path=in.wav ! wavparse ! flacenc ! filesink path=out.flac"
//! scraft-launch --dump-dot "testsrc total=65536 ! testsink"
//! scraft-launch --counters   "testsrc total=65536 ! testsink"
//! scraft-launch --log info    "filesrc path=x ! filesink path=y"
//! ```
//!
//! Flags (before the launch string):
//! * `--dump-dot`    — print the pipeline as Graphviz `dot` and exit (no run).
//! * `--counters`    — after the run, print each element's `CounterSnapshot`.
//! * `--log LEVEL`   — enable stderr logging up to LEVEL (error/warn/info/debug/trace).
//! * `--no-progress` — suppress the live progress line (auto-off when stderr isn't a tty).
//! * `--list`        — list every registered element name and exit.
//! * `-h`/`--help`   — usage.
//!
//! ## Progress
//! While running, a single in-place stderr line shows running time, bytes produced
//! by the chain's first element, bytes consumed by its last, and the sink-side
//! throughput over the last poll window. It is a pure *tap* (spec: Taps — a pull of
//! counters the pipeline already keeps, polled from the app's thread): the streaming
//! path pays nothing. Shown only when stderr is a terminal, so piped/CI output stays
//! clean; `--no-progress` forces it off.
//!
//! ## Ctrl-C
//! Not handled in v1: the standard library exposes no signal API and this tool takes
//! no new dependencies (the `ctrlc` crate is the usual answer). `run()` drives the
//! finite pipeline to EOS on its own, so a launch string that terminates needs no
//! interrupt; a `StopHandle`-based handler behind a signal dependency is a follow-up
//! (see `Pipeline::stop_handle`).

use std::io::{IsTerminal, Write};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use streamcraft_core::counters::TapHandle;
use streamcraft_core::id::ElementId;
use streamcraft_core::log::Level;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::registry::Registry;

/// Build the registry with every in-tree plugin crate's elements (spec: Plugins —
/// each crate exposes `register`, the app calls them all). pipewire is omitted (heavy
/// native dep, and a launch tool rarely wants a device sink).
fn build_registry() -> Registry {
    let mut r = Registry::new();
    streamcraft_elements::register(&mut r);
    streamcraft_audio::register(&mut r);
    sc_flac::register(&mut r);
    sc_ogg::register(&mut r);
    r
}

struct Options {
    launch: String,
    dump_dot: bool,
    counters: bool,
    log: Option<Level>,
    no_progress: bool,
}

/// Parse argv (already skipping the program name) into [`Options`], or a message to
/// print (help/list/usage error). Total: never panics on bad flags.
fn parse_args(args: Vec<String>) -> Result<Options, String> {
    let mut launch: Option<String> = None;
    let mut dump_dot = false;
    let mut counters = false;
    let mut log = None;
    let mut no_progress = false;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--dump-dot" => dump_dot = true,
            "--counters" => counters = true,
            "--no-progress" => no_progress = true,
            "--log" => {
                let lvl = it.next().ok_or_else(|| "--log needs a LEVEL argument".to_string())?;
                log = Some(
                    Level::from_name(&lvl)
                        .ok_or_else(|| format!("unknown log level '{lvl}' (error/warn/info/debug/trace)"))?,
                );
            }
            "-h" | "--help" => return Err(usage()),
            "--list" => return Err(list_elements()),
            other if other.starts_with('-') && other != "-" => {
                return Err(format!("unknown flag '{other}'\n\n{}", usage()));
            }
            // The first non-flag positional is the launch string.
            _ => {
                if launch.is_some() {
                    return Err(format!(
                        "unexpected extra argument '{arg}' — quote the whole launch string\n\n{}",
                        usage()
                    ));
                }
                launch = Some(arg);
            }
        }
    }
    let launch = launch.ok_or_else(|| format!("missing launch string\n\n{}", usage()))?;
    Ok(Options { launch, dump_dot, counters, log, no_progress })
}

fn usage() -> String {
    "usage: scraft-launch [--dump-dot] [--counters] [--log LEVEL] [--no-progress] \"elem prop=val ! elem ! …\"\n\
     \n\
     flags:\n\
     \x20 --dump-dot     print the pipeline as Graphviz dot and exit\n\
     \x20 --counters     print each element's counters after the run\n\
     \x20 --log LEVEL    log to stderr up to LEVEL (error/warn/info/debug/trace)\n\
     \x20 --no-progress  suppress the live progress line (auto-off when not a tty)\n\
     \x20 --list         list registered element names and exit\n\
     \x20 -h, --help     this message"
        .to_string()
}

fn list_elements() -> String {
    let r = build_registry();
    let mut s = String::from("registered elements:\n");
    for name in r.names() {
        s.push_str("  ");
        s.push_str(name);
        s.push('\n');
    }
    s
}

fn main() -> ExitCode {
    let opts = match parse_args(std::env::args().skip(1).collect()) {
        Ok(o) => o,
        Err(msg) => {
            // Help/list is not an error; a usage problem is. Both print to stderr; the
            // exit code distinguishes them.
            eprintln!("{msg}");
            return if msg.starts_with("usage:") || msg.starts_with("registered elements:") {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            };
        }
    };

    let registry = build_registry();
    let mut pipeline = Pipeline::new();
    if let Some(level) = opts.log {
        pipeline.log_to_stderr(level);
    }

    let ids = match registry.parse(&mut pipeline, &opts.launch) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("parse error: {}", e.message);
            return ExitCode::FAILURE;
        }
    };

    if opts.dump_dot {
        print!("{}", pipeline.dump_dot());
        return ExitCode::SUCCESS;
    }

    // Take the stats tap before the run so it reads the counters live/after (spec: Taps).
    let tap = pipeline.tap_handle();

    // Live progress: an observer thread polling the tap while `run()` blocks this
    // thread. Zero cost on the streaming path; tty-gated so piped output stays clean.
    let progress = (!opts.no_progress && std::io::stderr().is_terminal())
        .then(|| Progress::spawn(tap.clone(), ids.first().copied(), ids.last().copied()));

    let result = pipeline.run();

    if let Some(p) = progress {
        p.finish(&tap, ids.last().copied());
    }

    if opts.counters {
        print_counters(&tap, &ids);
    }

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("run error: {e:?}");
            ExitCode::FAILURE
        }
    }
}

/// The live progress line: an observer thread refreshing one in-place stderr line
/// from the tap a few times a second (spec: Taps tier 1 — cumulative counters plus
/// running time; the window and the arithmetic are the observer's).
struct Progress {
    stop: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<()>,
}

impl Progress {
    const POLL: Duration = Duration::from_millis(100);

    fn spawn(tap: TapHandle, first: Option<ElementId>, last: Option<ElementId>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("scl-progress".into())
            .spawn(move || {
                // Windowed sink rate: bytes consumed by the last element since the
                // previous poll, over wall time (the pipeline clock is the elapsed
                // display; the window uses wall time so a paused clock reads 0 B/s).
                let mut prev_bytes = 0u64;
                let mut prev_at = Instant::now();
                while !stop2.load(Ordering::Acquire) {
                    std::thread::sleep(Self::POLL);
                    let produced = first.and_then(|id| tap.snapshot(id)).map_or(0, |s| s.bytes_out);
                    let consumed = last.and_then(|id| tap.snapshot(id)).map_or(0, |s| s.bytes_in);
                    let now = Instant::now();
                    let dt = now.duration_since(prev_at).as_secs_f64();
                    let rate = if dt > 0.0 {
                        (consumed.saturating_sub(prev_bytes)) as f64 / dt
                    } else {
                        0.0
                    };
                    prev_bytes = consumed;
                    prev_at = now;
                    let elapsed = tap.now().nanos().map_or(0.0, |n| n as f64 / 1e9);
                    let mut err = std::io::stderr().lock();
                    let _ = write!(
                        err,
                        "\r  {elapsed:7.1}s   in {:>10}   out {:>10}   {:>10}/s   ",
                        fmt_size(produced),
                        fmt_size(consumed),
                        fmt_size(rate as u64),
                    );
                    let _ = err.flush();
                }
            })
            .expect("spawn progress thread");
        Self { stop, handle }
    }

    /// Stop the observer and replace the live line with a final summary.
    fn finish(self, tap: &TapHandle, last: Option<ElementId>) {
        self.stop.store(true, Ordering::Release);
        let _ = self.handle.join();
        let consumed = last.and_then(|id| tap.snapshot(id)).map_or(0, |s| s.bytes_in);
        let elapsed = tap.now().nanos().map_or(0.0, |n| n as f64 / 1e9);
        let avg = if elapsed > 0.0 { consumed as f64 / elapsed } else { 0.0 };
        eprintln!(
            "\r  done: {} in {elapsed:.1}s ({}/s avg){:12}",
            fmt_size(consumed),
            fmt_size(avg as u64),
            ""
        );
    }
}

/// `1234` → `1.2 KiB` — human sizes for the progress line, 1 decimal, binary units.
fn fmt_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// Print each element's counters after the run (spec: Debuggability — per-element
/// counters). Reads through the [`TapHandle`](streamcraft_core::counters::TapHandle),
/// the zero-streaming-cost stats pull.
fn print_counters(tap: &streamcraft_core::counters::TapHandle, ids: &[streamcraft_core::id::ElementId]) {
    println!("counters:");
    for &id in ids {
        match tap.snapshot(id) {
            Some(c) => println!(
                "  e{}: buffers {}/{} bytes {}/{} batches {}/{} queue_hw {}",
                id.0,
                c.buffers_in,
                c.buffers_out,
                c.bytes_in,
                c.bytes_out,
                c.batches_in,
                c.batches_out,
                c.queue_high_water,
            ),
            None => println!("  e{}: (no counters)", id.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fmt_size;

    #[test]
    fn fmt_size_picks_binary_units() {
        assert_eq!(fmt_size(0), "0 B");
        assert_eq!(fmt_size(999), "999 B");
        assert_eq!(fmt_size(1024), "1.0 KiB");
        assert_eq!(fmt_size(1536), "1.5 KiB");
        assert_eq!(fmt_size(12 * 1024 * 1024), "12.0 MiB");
        assert_eq!(fmt_size(5 * 1024 * 1024 * 1024), "5.0 GiB");
    }
}
