//! Live smoke test: the engine against a **real** PipeWire device, audibly.
//!
//! ```text
//! cargo run -p pf-player-engine --example live_smoke -- a.flac b.flac
//! ```
//!
//! The deterministic proof lives in `tests/gapless.rs`; this is the other half — the same code
//! path against a real graph, a real clock and a real DAC, on a scripted timeline that exercises
//! every control the API exposes. Give it two files whose audio is *continuous* across the seam
//! (the recipe in the transcript generates two halves of one unbroken tone) and the boundary is
//! either inaudible or obvious.
//!
//! Alongside the ear, it measures the thing the ear cannot: the playback **runway**
//! (`Engine::buffered`) sampled every few milliseconds through the boundary. A gap is by
//! definition the runway reaching zero, so "the minimum runway across the handoff" is the
//! number that says gapless, and it is printed at the end.

// Example binary: a scripted transcript, printing and formatting freely. Nothing here is on a
// media path or inside `process()` — the documented `clippy.toml` exception.
#![allow(clippy::disallowed_methods)]

use std::time::{Duration, Instant};

use pf_player_engine::{Engine, EngineEvent, Track};

/// One step of the script: when to do it (seconds from start) and what it is.
struct Step {
    at: f64,
    what: &'static str,
    run: fn(&Engine, &str),
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [a, b] = &args[..] else {
        return Err("usage: live_smoke <a.flac> <b.flac>".into());
    };

    let engine = Engine::new().map_err(|e| format!("cannot open the audio device: {e:?}"))?;
    println!("engine open — output renders {:?}", engine.output().format());

    let script: &[Step] = &[
        Step { at: 1.0, what: "enqueue B (the gapless append)", run: |_, _| {} },
        Step { at: 4.5, what: "pause", run: |e, _| e.pause() },
        Step { at: 5.5, what: "resume", run: |e, _| e.resume() },
        Step { at: 6.5, what: "volume 0.30 (ramped)", run: |e, _| e.set_volume(0.30) },
        Step { at: 7.5, what: "volume 1.00 (ramped)", run: |e, _| e.set_volume(1.00) },
        Step { at: 8.5, what: "set_idle(true) — park the device", run: |e, _| e.set_idle(true) },
        Step { at: 9.5, what: "set_idle(false) — unpark", run: |e, _| e.set_idle(false) },
        // Last on purpose: a mid-file seek currently ends the track (a known upstream defect —
        // see `Engine::seek`), so anything after it would not get to run.
        Step { at: 10.5, what: "seek to 2.0 s", run: |e, _| e.seek(Duration::from_secs(2)) },
    ];

    let start = Instant::now();
    engine.play_now(Track::file(a));
    println!("[ 0.00] play_now({a})");

    let mut next_step = 0usize;
    let mut next_tick = Duration::from_millis(250);
    let mut changed_at: Option<f64> = None;
    // The runway, sampled continuously from the moment the next track is queued until well past
    // the boundary. Zero at any point in that window is a gap.
    let mut min_runway = Duration::MAX;
    let mut watching = false;
    let mut ended = false;

    loop {
        let t = start.elapsed().as_secs_f64();

        for ev in engine.poll_events() {
            match ev {
                EngineEvent::TrackStarted => println!("[{t:5.2}] event: TrackStarted"),
                EngineEvent::TrackChanged => {
                    println!("[{t:5.2}] event: TrackChanged  <-- the gapless boundary");
                    changed_at = Some(t);
                }
                EngineEvent::TrackEnded => {
                    println!("[{t:5.2}] event: TrackEnded (queue empty; tail still draining)");
                    ended = true;
                }
                EngineEvent::DurationKnown(d) => {
                    println!("[{t:5.2}] event: DurationKnown({:.2} s)", d.as_secs_f64())
                }
                EngineEvent::Error { message } => println!("[{t:5.2}] event: ERROR {message}"),
            }
        }

        if watching && !ended {
            min_runway = min_runway.min(engine.buffered());
        }

        if next_step < script.len() && t >= script[next_step].at {
            let step = &script[next_step];
            if next_step == 0 {
                match engine.enqueue(Track::file(b)) {
                    Ok(()) => println!("[{t:5.2}] {} -> queue_len {}", step.what, engine.queue_len()),
                    Err(e) => println!("[{t:5.2}] enqueue refused: {e}"),
                }
                watching = true;
            } else {
                (step.run)(&engine, b);
                println!(
                    "[{t:5.2}] {} (paused={}, idle={}, volume={:.2})",
                    step.what,
                    engine.is_paused(),
                    engine.is_idle(),
                    engine.volume()
                );
            }
            next_step += 1;
        }

        if start.elapsed() >= next_tick {
            next_tick += Duration::from_millis(250);
            println!(
                "[{t:5.2}] pos {:6.2} / {:>6} s | queue {} | runway {:5.0} ms",
                engine.position().as_secs_f64(),
                engine
                    .duration()
                    .map(|d| format!("{:.2}", d.as_secs_f64()))
                    .unwrap_or_else(|| "?".into()),
                engine.queue_len(),
                engine.buffered().as_secs_f64() * 1000.0,
            );
        }

        if ended && engine.buffered() < Duration::from_millis(5) {
            println!("[{t:5.2}] playout complete");
            break;
        }
        if t > 40.0 {
            println!("[{t:5.2}] giving up — the script did not complete");
            break;
        }
        std::thread::sleep(Duration::from_millis(4));
    }

    println!("---");
    match changed_at {
        Some(at) => println!("boundary  : TrackChanged at {at:.2} s"),
        None => println!("boundary  : NEVER FIRED"),
    }
    if min_runway == Duration::MAX {
        println!("min runway: not sampled");
    } else {
        println!(
            "min runway: {:.0} ms across the handoff  ({})",
            min_runway.as_secs_f64() * 1000.0,
            if min_runway.is_zero() { "UNDERRUN — that is a gap" } else { "never reached zero" }
        );
    }

    let at = Instant::now();
    drop(engine);
    println!("teardown  : {:?}", at.elapsed());
    Ok(())
}
