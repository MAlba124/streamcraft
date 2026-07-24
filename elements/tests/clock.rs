//! Clocking and synchronization (spec: Clocking — the flagship). A `TimedTestSink`
//! renders each buffer on the pipeline clock; driven by a `MockClock`, these tests
//! prove the sink genuinely blocks on the clock, that hours of virtual time run in
//! milliseconds of wall time (deterministic, no real sleeps on the streaming side),
//! and that two sinks slaved to one clock share a schedule.

use std::sync::Arc;
use std::time::{Duration, Instant as StdInstant};

use streamcraft_core::clock::MockClock;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;
use streamcraft_elements::testing::{TimedTestSink, TimedTestSrc};

#[test]
fn timed_sink_waits_for_the_clock_then_releases() {
    // Two buffers one hour apart: buffer 0 is due at running-time 0 (renders at once),
    // buffer 1 is due one hour in. With the mock clock sitting at 0 the sink must park
    // in `wait_until(1h)`, so `run()` cannot finish. Advancing the clock past the
    // deadline releases it — the only real time spent is the short poll below.
    let clock = MockClock::new();
    let (sink, stats) = TimedTestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TimedTestSrc::new(2, Timestamp::from_secs(3600)));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.set_clock(Arc::new(clock.clone()));

    let run = std::thread::spawn(move || p.run());

    // Let the pipeline spin up: buffer 0 renders immediately, then the sink parks on
    // `wait_until(1h)` for buffer 1. Confirm it stays parked (only buffer 0 rendered,
    // run not finished). A small real poll — the negative assertion — mirrors the clock
    // unit tests.
    for _ in 0..60 {
        if run.is_finished() {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        !run.is_finished(),
        "run finished before the clock reached the second buffer's deadline"
    );
    assert_eq!(stats.count(), 1, "only the running-time-0 buffer rendered while parked");

    // Cross the deadline: the wait returns Reached, buffer 1 renders, run() ends.
    clock.advance(Timestamp::from_secs(3600));
    run.join().expect("run thread joined").expect("run ok");

    assert!(stats.is_done(), "sink saw EOS");
    let renders = stats.renders();
    assert_eq!(renders.len(), 2);
    assert_eq!(renders[0].pts, Timestamp::ZERO);
    assert_eq!(renders[1].pts, Timestamp::from_secs(3600));
    assert!(
        renders[1].rendered_at >= renders[1].pts,
        "rendered no earlier than its PTS"
    );
}

#[test]
fn timed_sink_compresses_hours_into_milliseconds() {
    // Ten hours of buffers spaced one second apart, rendered on a mock clock the test
    // thread advances as fast as it can: virtual hours, wall milliseconds, deterministic.
    let count = 36_000u64; // 36000 * 1s = 10 hours of running time
    let period = Timestamp::from_secs(1);
    let clock = MockClock::new();
    let (sink, stats) = TimedTestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TimedTestSrc::new(count, period));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.set_clock(Arc::new(clock.clone()));

    let start = StdInstant::now();
    let run = std::thread::spawn(move || p.run());

    // Drive virtual time forward in coarse jumps until the run completes. Advancing past
    // a deadline is harmless — the sink renders every buffer whose PTS the clock reached.
    let step = Timestamp::from_secs(300); // 5 virtual minutes per jump
    while !run.is_finished() {
        clock.advance(step);
        std::thread::yield_now();
    }
    run.join().expect("run joined").expect("run ok");
    let wall = start.elapsed();

    assert!(stats.is_done(), "sink saw EOS");
    let renders = stats.renders();
    assert_eq!(renders.len() as u64, count, "every buffer rendered exactly once");

    // The PTS schedule is exact and monotonic, and nothing rendered ahead of the clock.
    for (i, r) in renders.iter().enumerate() {
        assert_eq!(r.pts, Timestamp::from_secs(i as u64), "buffer {i} PTS");
        assert!(
            r.rendered_at >= r.pts,
            "buffer {i} rendered before its deadline"
        );
    }
    assert_eq!(
        renders.last().unwrap().pts,
        Timestamp::from_secs(count - 1),
        "the span really was ~10 virtual hours"
    );
    assert!(
        wall < Duration::from_secs(20),
        "10 virtual hours took {wall:?} of wall time — should be well under a second"
    );
}

#[test]
fn two_sinks_share_one_clock_and_stay_in_sync() {
    // Branching isn't available yet (linear pipelines only), so "a two-sink graph on a
    // shared clock" is shown here as two independent pipelines slaved to the SAME
    // MockClock: advancing the one clock gates both sinks, each renders the identical
    // schedule, and neither ever renders ahead of its deadline. (A single branched graph
    // feeding two sinks awaits dynamic pads.)
    let count = 200u64;
    let period = Timestamp::from_millis(10);
    let master = MockClock::new();

    let (sink_a, stats_a) = TimedTestSink::new();
    let mut pa = Pipeline::new();
    let a_src = pa.add(TimedTestSrc::new(count, period));
    let a_snk = pa.add(sink_a);
    pa.link((a_src, "src"), (a_snk, "sink")).expect("link a");
    pa.set_clock(Arc::new(master.clone()));

    let (sink_b, stats_b) = TimedTestSink::new();
    let mut pb = Pipeline::new();
    let b_src = pb.add(TimedTestSrc::new(count, period));
    let b_snk = pb.add(sink_b);
    pb.link((b_src, "src"), (b_snk, "sink")).expect("link b");
    pb.set_clock(Arc::new(master.clone()));

    let run_a = std::thread::spawn(move || pa.run());
    let run_b = std::thread::spawn(move || pb.run());

    let step = Timestamp::from_millis(5);
    while !run_a.is_finished() || !run_b.is_finished() {
        master.advance(step);
        std::thread::yield_now();
    }
    run_a.join().expect("a joined").expect("a ok");
    run_b.join().expect("b joined").expect("b ok");

    let ra = stats_a.renders();
    let rb = stats_b.renders();
    assert_eq!(ra.len() as u64, count, "sink A rendered the whole schedule");
    assert_eq!(rb.len() as u64, count, "sink B rendered the whole schedule");
    for i in 0..count as usize {
        let expected = Timestamp::from_millis(10 * i as u64);
        assert_eq!(ra[i].pts, expected, "A buffer {i} PTS");
        assert_eq!(rb[i].pts, expected, "B buffer {i} PTS");
        assert!(ra[i].rendered_at >= ra[i].pts, "A buffer {i} not early");
        assert!(rb[i].rendered_at >= rb[i].pts, "B buffer {i} not early");
    }
}
