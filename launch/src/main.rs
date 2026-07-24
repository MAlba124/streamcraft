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
//! * `--dump-dot`   — print the pipeline as Graphviz `dot` and exit (no run).
//! * `--counters`   — after the run, print each element's `CounterSnapshot`.
//! * `--log LEVEL`  — enable stderr logging up to LEVEL (error/warn/info/debug/trace).
//! * `--list`       — list every registered element name and exit.
//! * `-h`/`--help`  — usage.
//!
//! ## Ctrl-C
//! Not handled in v1: the standard library exposes no signal API and this tool takes
//! no new dependencies (the `ctrlc` crate is the usual answer). `run()` drives the
//! finite pipeline to EOS on its own, so a launch string that terminates needs no
//! interrupt; a `StopHandle`-based handler behind a signal dependency is a follow-up
//! (see `Pipeline::stop_handle`).

use std::process::ExitCode;

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
}

/// Parse argv (already skipping the program name) into [`Options`], or a message to
/// print (help/list/usage error). Total: never panics on bad flags.
fn parse_args(args: Vec<String>) -> Result<Options, String> {
    let mut launch: Option<String> = None;
    let mut dump_dot = false;
    let mut counters = false;
    let mut log = None;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--dump-dot" => dump_dot = true,
            "--counters" => counters = true,
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
    Ok(Options { launch, dump_dot, counters, log })
}

fn usage() -> String {
    "usage: scraft-launch [--dump-dot] [--counters] [--log LEVEL] \"elem prop=val ! elem ! …\"\n\
     \n\
     flags:\n\
     \x20 --dump-dot   print the pipeline as Graphviz dot and exit\n\
     \x20 --counters   print each element's counters after the run\n\
     \x20 --log LEVEL  log to stderr up to LEVEL (error/warn/info/debug/trace)\n\
     \x20 --list       list registered element names and exit\n\
     \x20 -h, --help   this message"
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

    let result = pipeline.run();

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
