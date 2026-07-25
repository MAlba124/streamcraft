//! Regression: idle group threads must **park**, not spin (spec: Scheduling —
//! see `PauseShared::park_idle`). A pool-starved producer paced by a real-time
//! sink used to spin-yield through empty passes at full tilt — ~1 core-second of
//! CPU over a ~1 s paced run, measured as `run_group` self time + `sched_yield`
//! + idle batch churn on movie playback. With the idle-park eventcount the same
//! run costs only the real work (tens of ms).
//!
//! This file is its own test binary (cargo autodiscovery), so the process-wide
//! CPU counter below sees only this pipeline — no cross-test interference.

use std::time::Duration;

use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;
use streamcraft_elements::testing::{TimedTestSink, TimedTestSrc};

/// Whole-process CPU time (utime + stime) from `/proc/self/stat`, in ms. CPU
/// time — unlike wall time — is not inflated by scheduler delays on a loaded
/// machine, so the assertion below is load-independent.
fn process_cpu_ms() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("read /proc/self/stat");
    // `pid (comm) state ppid ... utime stime ...` — comm may contain spaces but
    // is parenthesized, so index from after the last ')': state is field 3, and
    // utime/stime are fields 14/15, i.e. 11/12 past the paren.
    let after = stat.rsplit(')').next().expect("stat has a comm field");
    let fields: Vec<&str> = after.split_whitespace().collect();
    let utime: u64 = fields[11].parse().expect("utime");
    let stime: u64 = fields[12].parse().expect("stime");
    // USER_HZ is 100 on every Linux ABI this crate targets (x86_64/aarch64).
    (utime + stime) * 1000 / 100
}

/// A paced, pool-starved pipeline burns (almost) no CPU: the source group parks
/// while the sink plays out on the real clock, and is woken event-driven when a
/// slot frees (the consumer's progressing pass bumps the idle eventcount).
#[test]
fn idle_groups_park_instead_of_spinning() {
    // 20 buffers, 50 ms apart on the (default) real clock → ~1 s of paced
    // playback. The pool holds only 2 slots, so the source is pool-starved for
    // most of every 50 ms period — exactly the "no consumable input, no IO, an
    // element that can't progress" state that used to spin.
    let (sink, stats) = TimedTestSink::new();
    let mut p = Pipeline::new();
    p.set_pool(1024, 2);
    let src = p.add(TimedTestSrc::new(20, Timestamp::from_millis(50)));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");

    let cpu_before = process_cpu_ms();
    let t0 = std::time::Instant::now();
    p.run().expect("run");
    let wall = t0.elapsed();
    let cpu = process_cpu_ms().saturating_sub(cpu_before);

    assert_eq!(stats.count(), 20, "every buffer rendered");
    // Throughput sanity: parking must not break pacing — the sink really played
    // the schedule out (last pts is 950 ms).
    assert!(wall >= Duration::from_millis(900), "sink did not pace: {wall:?}");
    // The old spin burned ~a full core for the whole run (> 900 ms CPU). Parked
    // groups leave only real work — measured ~10-40 ms; 500 ms keeps a wide
    // non-flaky margin while still failing loudly on any full-tilt spin.
    assert!(
        cpu < 500,
        "group threads burned {cpu} ms CPU over a ~{wall:?} idle-paced run — spinning?"
    );
}
