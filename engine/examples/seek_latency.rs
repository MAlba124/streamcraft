//! **Where does a seek's latency go?** — measured live, against a real device, in release.
//!
//! ```text
//! cargo run --release -p pf-player-engine --example seek_latency -- track.flac [--queued next.flac]
//! ```
//!
//! Reported symptom: "large delay when seeking until the audio continues." That is a claim about
//! four layers at once, and the only useful answer separates them:
//!
//! | stage | what has happened | who is responsible |
//! |---|---|---|
//! | dispatch | `Engine::seek` resolved the target and bumped the seek generation | the engine |
//! | flush published | the sink handled `FlushStart` and published the ring's flush point | the scheduler + the sink |
//! | flush applied | the device callback jumped the ring's read head; the old audio is gone | the device |
//! | first write | the re-primed pipeline pushed its first post-seek byte | source + decoder + chain |
//! | **first pull** | the device callback pulled that byte — **the audible instant** | everything above |
//!
//! Run it in **release**. A debug build measures FLAC decoding (historically ~10× slower) rather
//! than the seek, which is exactly how this path has been misdiagnosed before.
//!
//! It seeks repeatedly across the file, prints the per-layer breakdown for each, and finishes
//! with the distribution of the number that matters. Pass `--queued` to run the seeks with a
//! second track pre-rolled behind the first — the shared-output case, which is what the
//! application is actually doing when the complaint arrives, and the one where a queued track's
//! gated pipeline could in principle delay the live one's flush.

// Example binary: a scripted measurement that prints and formats freely. Nothing here is on a
// media path or inside `process()` — the documented `clippy.toml` exception.
#![allow(clippy::disallowed_methods)]

use std::time::{Duration, Instant};

use pf_pipewire::probe::{self, Stage};
use pf_player_engine::{Engine, EngineEvent, Track};

/// How long to let playback settle before the first seek, and between seeks. Long enough that
/// the ring is full again and the measurement is of a steady state rather than of a start-up.
const SETTLE: Duration = Duration::from_millis(1_200);

/// How long a single seek may take before we give up waiting for it to become audible.
const SEEK_PATIENCE: Duration = Duration::from_secs(3);

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path = None;
    let mut queued = None;
    let mut gain_db = None;
    let mut drag = 0usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--queued" => queued = it.next().cloned(),
            // A ReplayGain stage, which the application always builds its music tracks with —
            // so the measured chain is the one the user actually hears.
            "--gain-db" => gain_db = it.next().and_then(|v| v.parse::<f32>().ok()),
            // Simulate a dragged scrubber: this many seeks in quick succession, then measure
            // how long after the *last* one the audio comes back.
            "--drag" => drag = it.next().and_then(|v| v.parse().ok()).unwrap_or(25),
            _ => path = Some(a.clone()),
        }
    }
    let Some(path) = path else {
        return Err(
            "usage: seek_latency <file> [--queued <file>] [--gain-db <db>] [--drag <n>]".into()
        );
    };

    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: this is a debug build. FLAC decoding is roughly ten times slower here and \
             the post-seek re-prime is exactly where that shows. Re-run with --release."
        );
    }

    let engine = Engine::new().map_err(|e| format!("cannot open the audio device: {e:?}"))?;
    let mut track = Track::file(&path);
    if let Some(db) = gain_db {
        track = track.with_gain_db(db);
    }
    engine.play_now(track);

    // Wait for it to actually be producing sound, and for a duration to arrive.
    let deadline = Instant::now() + Duration::from_secs(10);
    while !engine.is_playing() {
        report_errors(&engine);
        if Instant::now() > deadline {
            return Err("the track never started".into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let duration = engine.duration().ok_or("the file declares no duration to seek within")?;
    println!("playing {path} ({duration:.1?})");

    if let Some(q) = &queued {
        engine.enqueue(Track::file(q)).map_err(|e| format!("enqueue: {e:?}"))?;
        println!("queued {q} behind it — measuring with a second pipeline pre-rolled");
    }

    // Seek around the file, avoiding both ends so nothing is measuring an EOS.
    let targets: Vec<Duration> = [0.15, 0.55, 0.30, 0.70, 0.20, 0.60, 0.35, 0.80]
        .iter()
        .map(|f| duration.mul_f64(*f))
        .collect();

    let mut audible = Vec::new();
    println!();
    for (i, target) in targets.iter().enumerate() {
        std::thread::sleep(SETTLE);
        report_errors(&engine);

        // A dragged scrubber emits a burst of seeks, not one. Issue the burst first; the
        // measurement is armed for the *last* of them, because that is the one the user is
        // waiting on — everything before it they were still dragging through.
        if drag > 0 {
            let span = duration.mul_f64(0.25);
            for k in 0..drag {
                let t = target.saturating_sub(span) + span.mul_f64(k as f64 / drag as f64);
                engine.seek(t);
                std::thread::sleep(Duration::from_millis(20)); // ~50 Hz, a real drag
            }
        }

        probe::arm();
        let called = Instant::now();
        engine.seek(*target);
        let returned = called.elapsed();

        // Wait for the audible stage, or give up.
        let until = Instant::now() + SEEK_PATIENCE;
        while probe::at(Stage::FirstPull).is_none() && Instant::now() < until {
            std::thread::sleep(Duration::from_micros(200));
        }
        probe::disarm();

        println!("seek {} -> {:>6.2?}", i + 1, target);
        println!("    Engine::seek() returned in {:.2} ms", returned.as_secs_f64() * 1000.0);
        println!("    {}", probe::report());
        if let Some(d) = probe::at(Stage::FirstPull) {
            audible.push(d);
        } else {
            println!("    !! never became audible within {SEEK_PATIENCE:?}");
        }
    }

    engine.stop();
    summarise(&audible);
    Ok(())
}

fn report_errors(engine: &Engine) {
    for ev in engine.poll_events() {
        if let EngineEvent::Error { message } = ev {
            eprintln!("engine error: {message}");
        }
    }
}

fn summarise(audible: &[Duration]) {
    if audible.is_empty() {
        println!("\nno seek ever became audible");
        return;
    }
    let mut v: Vec<f64> = audible.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    v.sort_by(f64::total_cmp);
    let sum: f64 = v.iter().sum();
    println!(
        "\nAUDIBLE seek latency over {} seeks: min {:.1} ms, median {:.1} ms, max {:.1} ms, mean {:.1} ms",
        v.len(),
        v[0],
        v[v.len() / 2],
        v[v.len() - 1],
        sum / v.len() as f64
    );
}
