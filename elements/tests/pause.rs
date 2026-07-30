//! The pause transport (spec: Clocking — "pause is a clock op, not a state"):
//! pausing freezes running time; nothing renders while paused — even a sink whose
//! clock wait expires mid-pause blocks before rendering; resume excises the paused
//! interval by re-basing, so the remaining schedule plays exactly as if the pause
//! never happened. All driven by a MockClock — deterministic, no real sleeps.

use std::sync::Arc;
use std::time::Duration;

use profluens_core::clock::MockClock;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;
use profluens_elements::testing::{TimedTestSink, TimedTestSrc};

/// Poll until `cond` or ~200 ms of real time (the negative-assertion pattern from
/// the clock tests).
fn settle(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..100 {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

/// Pause holds rendering even when the clock crosses every deadline, and resume
/// completes the schedule with the paused interval excised.
#[test]
fn pause_holds_rendering_resume_excises_the_interval() {
    let clock = MockClock::new();
    let (sink, stats) = TimedTestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TimedTestSrc::new(5, Timestamp::from_millis(10)));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.set_clock(Arc::new(clock.clone()));
    let pause = p.pause_handle();

    let run = std::thread::spawn(move || p.run());

    // Buffer 0 (pts 0) renders as soon as the pipeline spins up.
    assert!(settle(|| stats.count() >= 1), "first buffer rendered");
    let rendered_before_pause = stats.count();

    // Pause, then blow the clock past EVERY remaining deadline. The sinks' waits
    // all expire — and must block in the pause gate instead of rendering.
    pause.pause();
    clock.advance(Timestamp::from_secs(10));
    assert!(
        !settle(|| stats.count() > rendered_before_pause),
        "nothing rendered while paused (got {} > {})",
        stats.count(),
        rendered_before_pause
    );

    // Resume: the 10 s advance happened while paused, so it is excised — the
    // remaining buffers' deadlines sit just past the resume point; a small further
    // advance releases them all.
    pause.resume();
    while !run.is_finished() {
        clock.advance(Timestamp::from_millis(20));
        std::thread::yield_now();
    }
    run.join().expect("joined").expect("run ok");
    assert_eq!(stats.renders().len(), 5, "full schedule after resume");
}

/// `start_paused`: the pipeline spins up and holds — preroll-and-hold — until the
/// first resume.
#[test]
fn start_paused_holds_until_resume() {
    let clock = MockClock::new();
    let (sink, stats) = TimedTestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TimedTestSrc::new(3, Timestamp::from_millis(10)));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.set_clock(Arc::new(clock.clone()));
    p.start_paused(true);
    let pause = p.pause_handle();

    let run = std::thread::spawn(move || p.run());

    // Even buffer 0 (pts 0, deadline already reached) must not render.
    assert!(
        !settle(|| stats.count() > 0),
        "held everything while start-paused (rendered {})",
        stats.count()
    );
    assert!(pause.is_paused());

    pause.resume();
    while !run.is_finished() {
        clock.advance(Timestamp::from_millis(20));
        std::thread::yield_now();
    }
    run.join().expect("joined").expect("run ok");
    assert_eq!(stats.renders().len(), 3, "full schedule after the first resume");
}

/// Toggle flips the transport; stop still wins over a paused pipeline (no hang).
#[test]
fn toggle_and_stop_while_paused() {
    let clock = MockClock::new();
    let (sink, _stats) = TimedTestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TimedTestSrc::new(100, Timestamp::from_millis(10)));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.set_clock(Arc::new(clock.clone()));
    let pause = p.pause_handle();
    let stop = p.stop_handle();

    assert!(pause.toggle(), "now paused");
    assert!(pause.is_paused());

    let run = std::thread::spawn(move || p.run());
    std::thread::sleep(Duration::from_millis(20)); // let it park in the gate

    // Shutdown must unstick a paused pipeline (the pause block polls the stop flag).
    stop.stop();
    clock.advance(Timestamp::from_secs(100));
    let joined = std::thread::spawn(move || run.join());
    for _ in 0..300 {
        if joined.is_finished() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(joined.is_finished(), "stop unstuck the paused pipeline");
    let _ = joined.join();
}
