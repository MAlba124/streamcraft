//! Time-based seeking (spec: flush/seek + Clocking): a seek rebases running time
//! to the target, so post-seek deadlines (`base + pts + latency`) are immediately
//! schedulable — a forward seek does not stall the sinks, a backward seek does not
//! release a late burst, and pause composes with seek. All driven by a MockClock —
//! deterministic, no real sleeps (spec: Testing — nothing ever sleeps).

use std::sync::Arc;
use std::time::Duration;

use streamcraft_core::clock::MockClock;
use streamcraft_core::id::ElementId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;
use streamcraft_elements::testing::{TimedTestSink, TimedTestSrc};

/// Poll until `cond` or ~200 ms of real time (the negative-assertion pattern from
/// the pause tests).
fn settle(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..100 {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

const PERIOD: Timestamp = Timestamp::from_millis(10);

/// A forward seek beyond elapsed clock time (the negative-base case): the stream
/// resumes at the target after advancing only ~the remaining schedule — NOT the
/// target's absolute distance. Without the rebase, deadlines would sit at
/// `base + target…end` and the run below would need ~300 ms of advances; with it,
/// ~110 ms suffice.
#[test]
fn forward_seek_schedules_immediately() {
    let clock = MockClock::new();
    let (sink, stats) = TimedTestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TimedTestSrc::new(30, PERIOD)); // 300 ms stream
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.set_clock(Arc::new(clock.clone()));
    let seek = p.seek_handle();

    let run = std::thread::spawn(move || p.run());
    assert!(settle(|| stats.count() >= 1), "first buffer rendered");

    // Seek to 200 ms while the clock is still near zero: base goes negative.
    let target = Timestamp::from_millis(200);
    seek.seek(0, target);

    // Drain with bounded advances; count the virtual time we had to add.
    let mut advanced = Timestamp::ZERO;
    let mut guard = 0;
    while !run.is_finished() {
        clock.advance(PERIOD);
        advanced = advanced.saturating_add(PERIOD);
        std::thread::sleep(Duration::from_millis(1));
        guard += 1;
        assert!(guard < 5_000, "run did not finish (stalled seek?)");
    }
    run.join().expect("joined").expect("run ok");

    // The post-seek tail rendered: pts from ~200 ms to the end, paced on the clock.
    let renders = stats.renders();
    let post: Vec<_> = renders.iter().filter(|r| r.pts >= target).collect();
    assert!(!post.is_empty(), "post-seek buffers rendered: {renders:?}");
    assert_eq!(post.last().unwrap().pts, Timestamp::from_millis(290), "tail completed");
    for r in &post {
        assert!(r.rendered_at >= r.pts, "never renders ahead of the clock: {r:?}");
    }
    // The no-stall property: finishing needed ~the remaining schedule (~100 ms),
    // nowhere near the target's absolute distance (200 ms+). Generous margin.
    assert!(
        advanced < Timestamp::from_millis(160),
        "forward seek must not stall: advanced {advanced:?}"
    );
}

/// A backward seek replays from the target, paced — deadlines rebased so earlier
/// pts are *future* again, not a late burst.
#[test]
fn backward_seek_replays_paced() {
    let clock = MockClock::new();
    let (sink, stats) = TimedTestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TimedTestSrc::new(30, PERIOD));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.set_clock(Arc::new(clock.clone()));
    let seek = p.seek_handle();

    let run = std::thread::spawn(move || p.run());

    // Play forward to ~100 ms of running time.
    let mut guard = 0;
    while stats.count() < 10 {
        clock.advance(PERIOD);
        std::thread::sleep(Duration::from_millis(1));
        guard += 1;
        assert!(guard < 5_000, "warm-up did not reach 10 renders");
    }

    // Seek back to 20 ms.
    let target = Timestamp::from_millis(20);
    seek.seek(0, target);

    // Lock-step drain: advance one period, then let the sink catch up before the
    // next tick — so virtual time never races ahead of the render thread.
    let mut guard = 0;
    while !run.is_finished() {
        let c = stats.count();
        clock.advance(PERIOD);
        let _ = settle(|| stats.count() > c || run.is_finished());
        guard += 1;
        assert!(guard < 1_000, "run did not finish after backward seek");
    }
    run.join().expect("joined").expect("run ok");

    // The replay boundary is where pts jumps backward (in-flight pre-seek
    // renders may still land after the seek call returns — the flush is
    // asynchronous — so slicing by count would race).
    let renders = stats.renders();
    let boundary = renders
        .windows(2)
        .position(|w| w[1].pts < w[0].pts)
        .map(|i| i + 1)
        .expect("replay boundary (a backward pts jump) exists");
    let post = &renders[boundary..];
    assert!(!post.is_empty(), "replayed tail rendered");
    assert!(
        post[0].pts <= Timestamp::from_millis(40),
        "replay starts near the 20 ms target, got {:?}",
        post[0].pts
    );
    assert_eq!(post.last().unwrap().pts, Timestamp::from_millis(290));
    for w in post.windows(2) {
        assert!(w[1].pts > w[0].pts, "monotonic replay: {w:?}");
    }
    for r in post {
        assert!(r.rendered_at >= r.pts, "never ahead of the clock: {r:?}");
    }
    // The discriminator: the first replayed buffer is re-scheduled on the
    // *rebased* clock — rendered on time against its own pts, not ~80 ms late
    // (which is what the un-rebased running time of ~100 ms would produce).
    assert!(
        post[0].rendered_at.saturating_sub(post[0].pts) <= Timestamp::from_millis(60),
        "replay re-scheduled on the rebased clock: {:?}",
        post[0]
    );
}

/// Seek while paused: nothing renders until resume; after resume the stream runs
/// from the target with the paused interval excised (the rebase and the resume
/// shift compose: running = target + post-resume elapsed).
#[test]
fn seek_while_paused_resumes_from_target() {
    let clock = MockClock::new();
    let (sink, stats) = TimedTestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TimedTestSrc::new(30, PERIOD));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.set_clock(Arc::new(clock.clone()));
    let seek = p.seek_handle();
    let pause = p.pause_handle();
    let tap = p.tap_handle();

    let run = std::thread::spawn(move || p.run());
    assert!(settle(|| stats.count() >= 1), "first buffer rendered");

    pause.pause();
    // Let the source drain its backpressure headroom so the re-prime is visible.
    std::thread::sleep(Duration::from_millis(10));
    let produced_before = tap.snapshot(ElementId(0)).unwrap().buffers_out;

    let target = Timestamp::from_millis(150);
    seek.seek(0, target);

    // The flush must run NOW, while paused (spec: flush/seek): the seek wake
    // flushes at the gate and one re-prime pass produces post-seek buffers —
    // that is what lets a display sink preroll the seeked frame.
    assert!(
        settle(|| tap.snapshot(ElementId(0)).unwrap().buffers_out > produced_before),
        "source re-primes while paused (flush not deferred to resume)"
    );
    // And the position reads the target while paused (frozen at the rebase).
    assert_eq!(tap.now(), target, "position shows the seek target while paused");
    let held = stats.count();

    // Blow the clock past every deadline while paused: the gate must hold.
    clock.advance(Timestamp::from_secs(10));
    assert!(
        !settle(|| stats.count() > held),
        "nothing renders while paused (got {} > {held})",
        stats.count()
    );

    pause.resume();
    let mut guard = 0;
    while !run.is_finished() {
        clock.advance(PERIOD);
        std::thread::sleep(Duration::from_millis(1));
        guard += 1;
        assert!(guard < 5_000, "run did not finish after resume");
    }
    run.join().expect("joined").expect("run ok");

    // First post-resume render is the seek target, rendered close to its own pts
    // (running restarted at the target — the 10 s paused advance was excised).
    let renders = stats.renders();
    let post = &renders[held as usize..];
    assert!(!post.is_empty(), "post-resume renders exist");
    assert_eq!(post[0].pts, target, "resumes at the seek target");
    assert!(
        post[0].rendered_at.saturating_sub(post[0].pts) <= Timestamp::from_millis(50),
        "paused interval excised: rendered_at {:?} vs pts {:?}",
        post[0].rendered_at,
        post[0].pts
    );
}

/// Observers (TapHandle → the scope's position display) see the position jump to
/// the target at the instant of the seek.
#[test]
fn tap_position_jumps_to_seek_target() {
    let clock = MockClock::new();
    let (sink, stats) = TimedTestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TimedTestSrc::new(30, PERIOD));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.set_clock(Arc::new(clock.clone()));
    let seek = p.seek_handle();
    let tap = p.tap_handle();

    let run = std::thread::spawn(move || p.run());
    assert!(settle(|| stats.count() >= 1), "first buffer rendered");

    let target = Timestamp::from_millis(120);
    seek.seek(0, target);
    // No clock advance between the seek and this read: exact under MockClock.
    assert_eq!(tap.now(), target, "position reads the seek target");

    let mut guard = 0;
    while !run.is_finished() {
        clock.advance(PERIOD);
        std::thread::sleep(Duration::from_millis(1));
        guard += 1;
        assert!(guard < 5_000, "run did not finish");
    }
    run.join().expect("joined").expect("run ok");
}

/// Rapid back-to-back seeks: the last target wins; the stream tail comes from it.
#[test]
fn rapid_seeks_last_target_wins() {
    let clock = MockClock::new();
    let (sink, stats) = TimedTestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TimedTestSrc::new(30, PERIOD));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.set_clock(Arc::new(clock.clone()));
    let seek = p.seek_handle();

    let run = std::thread::spawn(move || p.run());
    assert!(settle(|| stats.count() >= 1), "first buffer rendered");

    seek.seek(0, Timestamp::from_millis(100));
    seek.seek(0, Timestamp::from_millis(240));

    let mut guard = 0;
    while !run.is_finished() {
        clock.advance(PERIOD);
        std::thread::sleep(Duration::from_millis(1));
        guard += 1;
        assert!(guard < 5_000, "run did not finish");
    }
    run.join().expect("joined").expect("run ok");

    // Once the second target's range starts rendering, nothing earlier appears
    // again — the tail is target-2's schedule through the end of the stream.
    let renders = stats.renders();
    let t2 = Timestamp::from_millis(240);
    let first_t2 = renders.iter().position(|r| r.pts >= t2).expect("t2 range rendered");
    for r in &renders[first_t2..] {
        assert!(r.pts >= t2, "no pre-target-2 pts after the last flush: {r:?}");
    }
    assert_eq!(renders.last().unwrap().pts, Timestamp::from_millis(290));
}
