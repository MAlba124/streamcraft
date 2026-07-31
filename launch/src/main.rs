//! `profluens` — the CLI: `launch` (gst-launch), `inspect` (gst-inspect), `dot`
//! (spec: Plugins — "a gst-launch equivalent is cheap once the registry exists and
//! invaluable for debugging and bug reports").
//!
//! It aggregates every in-tree plugin crate's `register(&mut Registry)` into one
//! registry; `launch` parses a pipeline into a [`Pipeline`] and drives it to EOS.
//!
//! ```text
//! profluens launch filesrc path=in.wav ! wavparse ! flacenc ! filesink path=out.flac
//! profluens launch --counters testsrc total=65536 ! testsink
//! profluens launch --pool 4194304x16 videotestsrc width=1280 height=720 frames=30 ! videocksink
//! profluens dot testsrc total=65536 ! testsink
//! profluens inspect flacenc
//! profluens inspect            # list every registered element
//! ```
//!
//! Switches come **first**; everything after the first non-switch argument is the
//! pipeline, taken verbatim and joined with spaces — no quoting needed (the
//! gst-launch contract; interactive shells with `!` history expansion may still
//! prefer quotes).
//!
//! `launch` switches:
//! * `--counters`      — after the run, print each element's `CounterSnapshot`.
//! * `--health`        — after the run, print buffer-pool and scheduler-idle health: whether the
//!                       pool reached a steady state, how much slot headroom was left, and whether
//!                       the scheduler was woken by published work or fell back on its backstop
//!                       tick. Costs nothing to collect, so it works on any pipeline unprompted.
//! * `--log LEVEL`     — enable stderr logging up to LEVEL (error/warn/info/debug/trace).
//! * `--no-progress`   — suppress the live progress line (auto-off when stderr isn't a tty).
//! * `--pool SIZExN`   — size the buffer pool: `SIZE` bytes per slot, `N` slots (e.g.
//!   `4194304x16`). A raw-video frame must fit one slot, so big frames need this raised
//!   from the default. Parsed leniently (`x` or `*`, whitespace-tolerant).
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

use profluens_core::counters::TapHandle;
use profluens_core::id::ElementId;
use profluens_core::log::Level;
use profluens_core::pipeline::Pipeline;
use profluens_core::registry::Registry;

/// Build the registry with every in-tree plugin crate's elements (spec: Plugins —
/// each crate exposes `register`, the app calls them all). pipewire is omitted (heavy
/// native dep, and a launch tool rarely wants a device sink).
fn build_registry() -> Registry {
    let mut r = Registry::new();
    profluens_elements::register(&mut r);
    profluens_audio::register(&mut r);
    pf_aac::register(&mut r);
    pf_flac::register(&mut r);
    pf_opus::register(&mut r);
    pf_ogg::register(&mut r);
    profluens_video::register(&mut r);
    pf_vp8::register(&mut r);
    pf_vp9::register(&mut r);
    pf_av1::register(&mut r);
    pf_h264::register(&mut r);
    pf_h265::register(&mut r);
    pf_mkv::register(&mut r);
    pf_mp4::register(&mut r);
    pf_mp3::register(&mut r);
    pf_sdl3::register(&mut r);
    // Capability-gated: registers `vaapih264dec` only when the VA-API driver
    // advertises the decode profile (and `PF_NO_VAAPI` is unset) — a machine
    // without the hardware never sees the element.
    pf_vaapi::register(&mut r);
    r
}

#[cfg_attr(test, derive(Debug))]
struct Options {
    launch: String,
    dump_dot: bool,
    counters: bool,
    /// `--health`: print buffer-pool and scheduler-idle health after the run. Always cheap — the
    /// counters behind it are relaxed adds on the pool and on the scheduler's *parking* path, never
    /// on the per-buffer path — so unlike `--trace` it needs nothing enabled up front.
    health: bool,
    /// `--trace`: enable latency tracing and print per-element histograms after the
    /// run (p50/p90/p99/max of process time, ring residency, sink wait overshoot).
    trace: bool,
    log: Option<Level>,
    no_progress: bool,
    /// `(slot_size_bytes, slots)` from `--pool`, or `None` to keep the pipeline default.
    pool: Option<(usize, u32)>,
}

/// A parsed invocation: run/dot a pipeline, or inspect (one element, or list all).
#[cfg_attr(test, derive(Debug))]
enum Cmd {
    Pipeline(Options),
    Inspect(Option<String>),
}

/// Parse argv (already skipping the program name) into a [`Cmd`], or a message to
/// print (help/usage error). Total: never panics on bad input.
///
/// Contract (the gst-launch shape): a subcommand first, then that subcommand's
/// switches, then — for `launch`/`dot` — *everything else verbatim* as the pipeline,
/// joined with single spaces. Nothing after the first non-switch word is interpreted
/// as a switch, so `volume=-3`-style words can never collide with flags.
fn parse_args(args: Vec<String>) -> Result<Cmd, String> {
    let mut it = args.into_iter();
    let sub = it.next().ok_or_else(usage)?;
    match sub.as_str() {
        "launch" | "dot" => {
            let dump_dot = sub == "dot";
            let mut counters = false;
    let mut health = false;
            let mut trace = false;
            let mut log = None;
            let mut no_progress = false;
            let mut pool = None;
            let mut words: Vec<String> = Vec::new();
            while let Some(arg) = it.next() {
                if !words.is_empty() {
                    words.push(arg);
                    continue;
                }
                match arg.as_str() {
                    "--counters" => counters = true,
            "--health" => health = true,
                    "--trace" => trace = true,
                    "--no-progress" => no_progress = true,
                    "--log" => {
                        // Switch values still come from the iterator; we are before
                        // the pipeline, so this cannot swallow a pipeline word.
                        let lvl = take_value(&mut it, "--log LEVEL")?;
                        log = Some(Level::from_name(&lvl).ok_or_else(|| {
                            format!("unknown log level '{lvl}' (error/warn/info/debug/trace)")
                        })?);
                    }
                    "--pool" => {
                        let spec = take_value(&mut it, "--pool SIZExSLOTS")?;
                        pool = Some(parse_pool(&spec)?);
                    }
                    "-h" | "--help" => return Err(usage()),
                    other if other.starts_with('-') && other != "-" => {
                        return Err(format!("unknown switch '{other}'\n\n{}", usage()));
                    }
                    _ => words.push(arg), // first pipeline word — verbatim from here on
                }
            }
            if words.is_empty() {
                return Err(format!("missing pipeline\n\n{}", usage()));
            }
            Ok(Cmd::Pipeline(Options {
                launch: words.join(" "),
                dump_dot,
                counters,
                health,
                trace,
                log,
                no_progress,
                pool,
            }))
        }
        "inspect" => {
            let name = it.next();
            if let Some(extra) = it.next() {
                return Err(format!("inspect takes at most one element name (got '{extra}')\n\n{}", usage()));
            }
            Ok(Cmd::Inspect(name))
        }
        "list" => Ok(Cmd::Inspect(None)),
        "help" | "-h" | "--help" => Err(usage()),
        other => Err(format!("unknown command '{other}'\n\n{}", usage())),
    }
}

fn take_value(it: &mut impl Iterator<Item = String>, what: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{what}: missing value"))
}

/// Parse a `--pool` spec — `SLOT_SIZExSLOTS` (`4194304x16`) — into `(slot_size, slots)`.
/// Lenient: the separator is `x`, `X`, or `*`, surrounding whitespace is trimmed, and
/// both sides must be positive integers. Total — a malformed spec is a named `Err`,
/// never a panic (the parse layer's fuzz-friendly contract). `SLOTS` is clamped to
/// `u32`; an out-of-range slot count is an error rather than a silent wrap.
fn parse_pool(spec: &str) -> Result<(usize, u32), String> {
    let sep = spec
        .find(['x', 'X', '*'])
        .ok_or_else(|| format!("--pool '{spec}' must be SIZExSLOTS (e.g. 4194304x16)"))?;
    let size_s = spec[..sep].trim();
    let slots_s = spec[sep + 1..].trim();
    let slot_size: usize = size_s
        .parse()
        .map_err(|_| format!("--pool slot size '{size_s}' is not a non-negative integer"))?;
    let slots: u32 = slots_s
        .parse()
        .map_err(|_| format!("--pool slot count '{slots_s}' is not a u32"))?;
    if slot_size == 0 || slots == 0 {
        return Err(format!("--pool '{spec}' — both slot size and slot count must be > 0"));
    }
    Ok((slot_size, slots))
}

fn usage() -> String {
    "usage: profluens <command> …\n\
     \n\
     commands:\n\
     \x20 launch [switches] ELEM prop=val ! ELEM ! …   run a pipeline to EOS\n\
     \x20 dot    [switches] ELEM ! …                   print the pipeline as Graphviz dot\n\
     \x20 inspect [NAME]                               describe an element / list all\n\
     \x20 list                                         list registered element names\n\
     \x20 help                                         this message\n\
     \n\
     switches come first; everything after the first non-switch word is the pipeline\n\
     (no quoting needed).\n\
     \n\
     launch switches:\n\
     \x20 --counters        print each element's counters after the run\n\
\x20 --health          pool + scheduler idle health after the run (always cheap)\n\
\x20 --trace           latency tracing: per-element p50/p99/max after the run\n\
\x20 (interactive)     while running on a tty: 'p'⏎ pause/resume, 'q'⏎ stop\n\
     \x20 --log LEVEL       log to stderr up to LEVEL (error/warn/info/debug/trace)\n\
     \x20 --no-progress     suppress the live progress line (auto-off when not a tty)\n\
     \x20 --pool SIZExSLOTS size the buffer pool: SIZE bytes/slot, SLOTS slots (e.g. 4194304x16);\n\
     \x20                   a raw-video frame must fit one slot, so big frames need this raised"
        .to_string()
}

/// `inspect` with no name / `list`: every registered element, one per line.
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
        Ok(Cmd::Pipeline(o)) => o,
        Ok(Cmd::Inspect(Some(name))) => {
            return match build_registry().describe(&name) {
                Some(card) => {
                    print!("{card}");
                    ExitCode::SUCCESS
                }
                None => {
                    eprintln!("no element '{name}' — 'profluens inspect' lists them all");
                    ExitCode::FAILURE
                }
            };
        }
        Ok(Cmd::Inspect(None)) => {
            print!("{}", list_elements());
            return ExitCode::SUCCESS;
        }
        Err(msg) => {
            // Help is not an error; a usage problem is. Both print to stderr; the
            // exit code distinguishes them.
            eprintln!("{msg}");
            return if msg.starts_with("usage:") {
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
    if let Some((slot_size, slots)) = opts.pool {
        // A raw-video frame must fit one pool slot; large frames need this raised from
        // the default (spec: Formats — pool sizing).
        pipeline.set_pool(slot_size, slots);
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

    if opts.trace {
        pipeline.set_tracing(true);
    }

    // Take the stats tap before the run so it reads the counters live/after (spec: Taps).
    let tap = pipeline.tap_handle();

    // Live progress: an observer thread polling the tap while `run()` blocks this
    // thread. Zero cost on the streaming path; tty-gated so piped output stays clean.
    let progress = (!opts.no_progress && std::io::stderr().is_terminal())
        .then(|| Progress::spawn(tap.clone(), ids.first().copied(), ids.last().copied()));

    // Transport control from stdin while running (tty only — a piped stdin must not
    // be consumed): `p⏎` pause/resume (spec: Clocking — pause is a clock op), `q⏎`
    // stop. Line-based on purpose: no raw-mode termios, no dependencies. The reader
    // thread parks on stdin and is not joined — it dies with the process.
    if std::io::stdin().is_terminal() && std::io::stderr().is_terminal() {
        let pause = pipeline.pause_handle();
        let stop = pipeline.stop_handle();
        std::thread::Builder::new()
            .name("scl-stdin".into())
            .spawn(move || {
                let stdin = std::io::stdin();
                let mut line = String::new();
                loop {
                    line.clear();
                    if stdin.read_line(&mut line).unwrap_or(0) == 0 {
                        return; // stdin closed
                    }
                    match line.trim() {
                        "p" => {
                            let paused = pause.toggle();
                            eprintln!("\r{}", if paused { "⏸ paused ('p'⏎ resumes)" } else { "▶ playing" });
                        }
                        "q" => {
                            eprintln!("\rstopping…");
                            stop.stop();
                            return;
                        }
                        "" => {}
                        other => eprintln!("\r'{other}'? — 'p'⏎ pause/resume, 'q'⏎ stop"),
                    }
                }
            })
            .expect("spawn stdin control thread");
    }

    let result = pipeline.run();

    if let Some(p) = progress {
        p.finish(&tap, ids.last().copied());
    }

    if opts.counters {
        print_counters(&tap, &ids);
    }
    if opts.health {
        print_health(&tap);
    }
    if opts.trace {
        print_latency(&tap, &ids);
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
                    // `park_timeout`, not `sleep`: `finish` unparks after setting `stop`, so the
                    // CLI exits at once instead of waiting out however much of the poll interval
                    // happened to be left — up to `POLL` on every single invocation.
                    std::thread::park_timeout(Self::POLL);
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
        self.handle.thread().unpark(); // `stop` is published before this, so the wake cannot be lost
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
/// counters). Reads through the [`TapHandle`](profluens_core::counters::TapHandle),
/// the zero-streaming-cost stats pull.
/// Buffer-pool and scheduler-idle health — the two questions a profiler would otherwise be needed
/// for: "did this pipeline settle into recycling memory, and was it ever asleep when it should have
/// been working?"
///
/// `slot_allocations` flat against `acquires` is the zero-steady-state-allocation goal met; a large
/// ratio means the pool is missing and falling back to the heap. `outstanding` against `max_slots`
/// is the backpressure headroom, and `high_water` is what the run actually needed — i.e. how to size
/// `--pool`. On the scheduler side `tick expiries` is the number to read: a park normally ends
/// because work was published, so expiries near zero is healthy, while a large fraction means some
/// path made work available without announcing it and the pipeline stepped at the 10 ms backstop
/// rather than at the data rate.
fn print_health(tap: &profluens_core::counters::TapHandle) {
    let pool = tap.pool();
    let sched = tap.scheduler();
    println!("pool:");
    println!(
        "  slots {:>10} outstanding  {:>10} high water  {:>10} capacity",
        pool.outstanding, pool.high_water, pool.max_slots
    );
    let reuse = if pool.acquires > 0 {
        100.0 * (1.0 - pool.slot_allocations as f64 / pool.acquires as f64)
    } else {
        0.0
    };
    println!(
        "  {:>10} acquires  {:>10} recycles  {:>10} heap allocations ({reuse:.1}% reused)",
        pool.acquires, pool.recycles, pool.slot_allocations
    );
    println!("scheduler:");
    let expiry_pct =
        if sched.parks > 0 { 100.0 * sched.tick_expiries as f64 / sched.parks as f64 } else { 0.0 };
    println!(
        "  {:>10} parks  {:>10} tick expiries ({expiry_pct:.1}%)  {:>10.3}s parked",
        sched.parks,
        sched.tick_expiries,
        sched.parked_ns as f64 / 1e9
    );
    if expiry_pct > 25.0 && sched.parks > 20 {
        println!(
            "  note: most parks ended on the backstop tick, not on published work — the pipeline \
             was stepping at the tick rate rather than the data rate."
        );
    }
}

fn print_counters(tap: &profluens_core::counters::TapHandle, ids: &[profluens_core::id::ElementId]) {
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

/// `--trace`: per-element latency table (spec: Debuggability). Quantiles are computed
/// here, on the observer's side, from the histograms the scheduler filled — the tap
/// model: the pipeline keeps counts, the observer does arithmetic.
fn print_latency(tap: &profluens_core::counters::TapHandle, ids: &[profluens_core::id::ElementId]) {
    let fmt = |ns: u64| -> String {
        if ns >= 1_000_000_000 {
            format!("{:.2}s", ns as f64 / 1e9)
        } else if ns >= 1_000_000 {
            format!("{:.1}ms", ns as f64 / 1e6)
        } else if ns >= 1_000 {
            format!("{:.1}µs", ns as f64 / 1e3)
        } else {
            format!("{ns}ns")
        }
    };
    println!("latency (p50/p99/max over the whole run):");
    for &id in ids {
        let Some((process, queue, wait)) = tap.latency(id) else { continue };
        let name = tap.element_name(id).unwrap_or("?");
        let mut line = format!("  {name}#{}:", id.0);
        for (label, h) in [("process", process), ("queue", queue), ("wait_late", wait)] {
            if h.count == 0 {
                continue;
            }
            line.push_str(&format!(
                " {label} {}/{}/{} (n={})",
                fmt(h.quantile_ns(0.5)),
                fmt(h.quantile_ns(0.99)),
                fmt(h.max_ns),
                h.count
            ));
        }
        println!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::{fmt_size, parse_args, parse_pool, Cmd};

    fn argv(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn launch_joins_unquoted_pipeline_words_after_switches() {
        let Ok(Cmd::Pipeline(o)) = parse_args(argv(&[
            "launch", "--counters", "--pool", "1024x4", "testsrc", "total=65536", "!", "testsink",
        ])) else {
            panic!("expected a pipeline command");
        };
        assert_eq!(o.launch, "testsrc total=65536 ! testsink");
        assert!(o.counters && !o.dump_dot);
        assert_eq!(o.pool, Some((1024, 4)));
    }

    #[test]
    fn switch_looking_words_after_the_pipeline_starts_stay_pipeline() {
        // Nothing after the first non-switch word is interpreted — the gst contract.
        let Ok(Cmd::Pipeline(o)) =
            parse_args(argv(&["launch", "testsrc", "--counters", "!", "testsink"]))
        else {
            panic!("expected a pipeline command");
        };
        assert_eq!(o.launch, "testsrc --counters ! testsink");
        assert!(!o.counters, "--counters inside the pipeline is data, not a switch");
    }

    #[test]
    fn dot_is_launch_without_running() {
        let Ok(Cmd::Pipeline(o)) = parse_args(argv(&["dot", "testsrc", "!", "testsink"])) else {
            panic!("expected a pipeline command");
        };
        assert!(o.dump_dot);
        assert_eq!(o.launch, "testsrc ! testsink");
    }

    #[test]
    fn inspect_routes_name_and_bare_list() {
        assert!(matches!(
            parse_args(argv(&["inspect", "flacenc"])),
            Ok(Cmd::Inspect(Some(n))) if n == "flacenc"
        ));
        assert!(matches!(parse_args(argv(&["inspect"])), Ok(Cmd::Inspect(None))));
        assert!(matches!(parse_args(argv(&["list"])), Ok(Cmd::Inspect(None))));
        assert!(parse_args(argv(&["inspect", "a", "b"])).is_err());
    }

    #[test]
    fn bad_invocations_error_without_panicking() {
        assert!(parse_args(argv(&[])).is_err());
        assert!(parse_args(argv(&["frobnicate"])).is_err());
        assert!(parse_args(argv(&["launch"])).is_err(), "missing pipeline");
        assert!(parse_args(argv(&["launch", "--pool"])).is_err(), "missing switch value");
        assert!(parse_args(argv(&["launch", "--nope", "a", "!", "b"])).is_err());
        // help is a usage message, flagged via the Err channel but starts with usage:
        let msg = parse_args(argv(&["help"])).unwrap_err();
        assert!(msg.starts_with("usage:"));
    }

    #[test]
    fn fmt_size_picks_binary_units() {
        assert_eq!(fmt_size(0), "0 B");
        assert_eq!(fmt_size(999), "999 B");
        assert_eq!(fmt_size(1024), "1.0 KiB");
        assert_eq!(fmt_size(1536), "1.5 KiB");
        assert_eq!(fmt_size(12 * 1024 * 1024), "12.0 MiB");
        assert_eq!(fmt_size(5 * 1024 * 1024 * 1024), "5.0 GiB");
    }

    #[test]
    fn parse_pool_accepts_lenient_specs() {
        // The documented form, plus the tolerated separators and surrounding whitespace.
        assert_eq!(parse_pool("4194304x16"), Ok((4_194_304, 16)));
        assert_eq!(parse_pool("1024X4"), Ok((1024, 4)));
        assert_eq!(parse_pool("2048*8"), Ok((2048, 8)));
        assert_eq!(parse_pool("  65536 x 32 "), Ok((65_536, 32)));
    }

    #[test]
    fn parse_pool_rejects_bad_specs_without_panicking() {
        // The contract is "Err, never panic" (the parse layer's fuzz-friendly rule).
        let bad = [
            "", "x", "16", "4194304", "x16", "16x", "axb", "-1x4", "4x-1", "0x16",
            "16x0", "16x16x16", "16.5x4", "999999999999999999999999x4",
            "16x999999999999999999999999", "  ", "*", "16 16",
        ];
        for s in bad {
            assert!(parse_pool(s).is_err(), "{s:?} should be a clean Err");
        }
    }
}
