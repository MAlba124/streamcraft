//! `scplay` — the general-purpose player CLI (spec: Milestone applications §5). Probe a
//! file, autoplug the pipeline, play it:
//!
//! ```text
//! scplay [--no-window] [--no-audio] [--stats] [--max-secs N] FILE
//! ```
//!
//! Contract (a separate validation harness runs against this):
//! - one `track <pad>: <what happened>` line per discovered track;
//! - `--no-window` → decoded video to a drop sink (headless); `--no-audio` → same for audio
//!   (no PipeWire device grabbed);
//! - `--stats` → the play_file-style 3 s per-element counter lines;
//! - `--max-secs N` → cooperative stop after N seconds, exit 0;
//! - stdin: `p`⏎ pause toggle, `q`⏎ quit, a single digit ⏎ seeks to n×10 % of duration;
//! - exit 0 on clean EOS or max-secs stop; nonzero with ONE stderr line when the container is
//!   unknown or NO track could be linked (audio-only / video-only files are success);
//! - the bus is drained at exit (Warnings, Qos), like play_file.

// The adopted h264 decoder allocates ~1.5M times/s internally (see play_file); mimalloc buys
// the headroom until the churn is fixed upstream. Binary-only — the library stays
// allocator-agnostic. Gated behind the default `mimalloc` feature: build
// `--no-default-features` to fall back to the system allocator so heaptrack/valgrind can see
// the decode-path allocations (mimalloc bypasses the libc malloc they interpose).
#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::process::ExitCode;
use std::time::Duration;

use sc_play::{Player, SinkChoice, SinkPolicy};
use streamcraft_core::bus::BusMessage;
use streamcraft_core::time::Timestamp;

/// Parsed CLI options. Switches come first; the single positional is the file.
struct Options {
    file: String,
    no_window: bool,
    no_audio: bool,
    stats: bool,
    max_secs: Option<u64>,
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut file: Option<String> = None;
    let (mut no_window, mut no_audio, mut stats) = (false, false, false);
    let mut max_secs = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--no-window" => no_window = true,
            "--no-audio" => no_audio = true,
            "--stats" => stats = true,
            "--max-secs" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--max-secs needs a value (seconds)".to_string())?
                    .parse::<u64>()
                    .map_err(|_| "--max-secs value must be a non-negative integer".to_string())?;
                max_secs = Some(v);
            }
            "-h" | "--help" => return Err(usage()),
            other if other.starts_with("--") => return Err(format!("unknown switch '{other}'\n{}", usage())),
            _ => {
                if file.replace(arg.clone()).is_some() {
                    return Err(format!("expected exactly one FILE argument\n{}", usage()));
                }
            }
        }
    }
    let file = file.ok_or_else(usage)?;
    Ok(Options { file, no_window, no_audio, stats, max_secs })
}

fn usage() -> String {
    "usage: scplay [--no-window] [--no-audio] [--stats] [--max-secs N] FILE".to_string()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("{msg}");
            // A bare help request is success; a malformed invocation is not.
            return if msg.starts_with("usage:") { ExitCode::SUCCESS } else { ExitCode::from(2) };
        }
    };

    let policy = SinkPolicy {
        video: if opts.no_window { SinkChoice::Drop } else { SinkChoice::Device },
        audio: if opts.no_audio { SinkChoice::Drop } else { SinkChoice::Device },
    };

    // Build: probe → head → seek index → preroll → autoplug. A failure here (unknown
    // container, unreadable head) is the one-line error + nonzero exit.
    let mut player = match Player::open(&opts.file, policy) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };

    // The per-track report (the harness parses these).
    println!(
        "{}: {} track(s)",
        player.kind().label(),
        player.tracks().len(),
    );
    for t in player.tracks() {
        println!("track {}: {}", t.pad, t.summary);
    }
    if let Some(d) = player.duration() {
        println!("duration: {:.1}s", d.0 as f64 / 1e9);
    }

    // A recognized container with zero linkable tracks is a failure (nonzero, one line).
    // Audio-only / video-only are fine — `any_track_linked` is true for both.
    if !player.any_track_linked() {
        eprintln!("no track could be linked to any decoder");
        return ExitCode::from(1);
    }

    // Transport controls off the pipeline handles (spec: Clocking — pause is a clock op;
    // flush/seek — the digit-seek). All spawned threads die with the process; none is joined.
    install_stats(&mut player, opts.stats);
    install_max_secs(&mut player, opts.max_secs);
    install_stdin_controls(&mut player);
    install_window_controls(&mut player);

    println!(
        "playing {} — 'p'⏎ pause/resume, 'q'⏎ quit, 0-9⏎ seek (or Ctrl-C)…",
        opts.file
    );
    let result = player.run();

    // Drain the bus like play_file: surface sink warnings (a display/GPU fallback reports
    // here) and Qos frame drops, which would otherwise be invisible.
    while let Some(msg) = player.pipeline.bus().try_recv() {
        match msg {
            BusMessage::Warning { error, .. } => eprintln!("warning: {error:?}"),
            BusMessage::Qos { lateness_ns, .. } => {
                eprintln!("qos: frame dropped {:.1} ms late", lateness_ns as f64 / 1e6)
            }
            _ => {}
        }
    }

    match result {
        Ok(()) => {
            println!("done.");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("run error: {e:?}");
            ExitCode::from(1)
        }
    }
}

/// The `--stats` observer: named per-element `buffers in→out (+delta)` lines every 3 s, from
/// the live tap (zero cost to the streaming threads — spec: Taps).
fn install_stats(player: &mut Player, on: bool) {
    if !on {
        return;
    }
    let tap = player.pipeline.tap_handle();
    let watched: Vec<(String, streamcraft_core::id::ElementId)> =
        player.watched().into_iter().map(|(n, id)| (n.to_string(), id)).collect();
    std::thread::spawn(move || {
        let mut prev: Vec<(u64, u64)> = vec![(0, 0); watched.len()];
        loop {
            std::thread::sleep(Duration::from_secs(3));
            let mut line = String::from("stats:");
            for (i, (name, id)) in watched.iter().enumerate() {
                if let Some(c) = tap.snapshot(*id) {
                    let (pi, po) = prev[i];
                    line.push_str(&format!(
                        " {name} {}→{} (+{}/+{}) qhw={} drops={} |",
                        c.buffers_in,
                        c.buffers_out,
                        c.buffers_in - pi,
                        c.buffers_out - po,
                        c.queue_high_water,
                        c.drops,
                    ));
                    prev[i] = (c.buffers_in, c.buffers_out);
                }
            }
            eprintln!("{line}");
        }
    });
}

/// `--max-secs N`: a cooperative stop after N wall seconds (exits 0). A watcher thread trips
/// the [`StopHandle`](streamcraft_core::pipeline::Pipeline::stop_handle); `run()` then returns
/// `Ok`.
fn install_max_secs(player: &mut Player, max_secs: Option<u64>) {
    let Some(secs) = max_secs else { return };
    let stop = player.pipeline.stop_handle();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(secs));
        stop.stop();
    });
}

/// Line-based stdin transport: `p`⏎ pause toggle, `q`⏎ quit, a single digit ⏎ seeks to
/// n×10 % of duration (spec: Clocking — pause is a clock op; flush/seek — the scope-free time
/// seek). No raw-mode termios, no deps; the reader parks on stdin and dies with the process.
fn install_stdin_controls(player: &mut Player) {
    let pause = player.pipeline.pause_handle();
    let stop = player.pipeline.stop_handle();
    let seek = player.pipeline.seek_handle();
    let tap = player.pipeline.tap_handle();
    let index = player.seek_index().clone();
    let duration = player.duration().unwrap_or(Timestamp::NONE);
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut line = String::new();
        loop {
            line.clear();
            if std::io::BufRead::read_line(&mut stdin.lock(), &mut line).unwrap_or(0) == 0 {
                return; // stdin closed
            }
            match line.trim() {
                "p" => {
                    let paused = pause.toggle();
                    println!("{}", if paused { "⏸ paused" } else { "▶ playing" });
                }
                "q" => {
                    stop.stop();
                    return;
                }
                d if d.len() == 1 && d.as_bytes()[0].is_ascii_digit() => {
                    let Some(dur) = duration.nanos() else {
                        println!("seek: unknown duration");
                        continue;
                    };
                    let frac = (d.as_bytes()[0] - b'0') as u64;
                    let target = Timestamp(dur / 10 * frac);
                    match index.resolve(target, duration) {
                        // Seek to the RESOLVED cue time, not the request — rebasing to the
                        // request while content resumes at the earlier cue leaves video
                        // permanently late (frozen picture over playing audio). See
                        // SeekIndex::resolve.
                        Some((byte, landed)) => {
                            seek.seek(byte, landed);
                            println!(
                                "⇥ seek to {:.1}s → landed {:.1}s (byte {byte}); position now {:.1}s",
                                target.0 as f64 / 1e9,
                                landed.0 as f64 / 1e9,
                                tap.now().nanos().map(|n| n as f64 / 1e9).unwrap_or(-1.0),
                            );
                        }
                        None => println!("seek: no mapping available"),
                    }
                }
                _ => {}
            }
        }
    });
}

/// Window transport: drain the raw-wire presenter's [`PlayerControl`] (clicks from the video
/// window) into the same pause/seek/stop handles, and publish duration + pause state back for
/// the HUD. No-op unless `SC_PRESENT`'s `waylandvideosink` is in use. Polls at ~33 Hz; the
/// thread dies with the process (none is joined).
fn install_window_controls(player: &mut Player) {
    let Some(control) = player.player_control() else {
        return;
    };
    let pause = player.pipeline.pause_handle();
    let stop = player.pipeline.stop_handle();
    let seek = player.pipeline.seek_handle();
    let index = player.seek_index().clone();
    let duration = player.duration().unwrap_or(Timestamp::NONE);
    control.set_duration_ns(duration.nanos().unwrap_or(0));
    std::thread::spawn(move || loop {
        for cmd in control.drain() {
            match cmd {
                sc_present::UiCommand::TogglePause => {
                    let paused = pause.toggle();
                    control.set_paused(paused);
                }
                sc_present::UiCommand::Quit => {
                    stop.stop();
                    return;
                }
                sc_present::UiCommand::SeekFraction(f) => {
                    let Some(dur) = duration.nanos() else { continue };
                    let target = Timestamp((dur as f64 * f as f64) as u64);
                    if let Some((byte, landed)) = index.resolve(target, duration) {
                        seek.seek(byte, landed);
                    }
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(30));
    });
}
