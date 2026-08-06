//! **Seek-latency probe** — where the time between "the user seeked" and "the user hears it"
//! actually goes.
//!
//! A seek crosses four layers before a sample changes, and each one can hide a stall from the
//! others: the app's call, the pipeline's flush, the sink's re-primed write into the output ring,
//! and the real-time callback pulling that byte out. "Seeks feel slow" is not actionable until
//! those four are separated, and none of them can be timed from outside — the last two happen on
//! the device thread.
//!
//! So this records one timestamp at each, into process-global atomics.
//!
//! ```ignore
//! probe::arm();                       // t0: about to seek
//! engine.seek(target);
//! probe::mark(Stage::Dispatched);     // t1: the generation bump has been published
//! // …the sink and the RT callback fill in the rest by themselves…
//! eprintln!("{}", probe::report().unwrap());
//! ```
//!
//! # Cost when it is off
//!
//! One relaxed `bool` load per marked site, and every marked site is per-*block* or per-*batch* —
//! never per sample. [`arm`] is the only thing that turns it on, so a process that never calls it
//! pays two predictable-branch loads in the RT callback and nothing else. Nothing here allocates,
//! locks, or syscalls: `Instant` is a vDSO read, and the stages are plain `AtomicU64` nanosecond
//! offsets. It is therefore safe to leave compiled into release builds, which is the point — the
//! numbers that matter are release numbers, and a debug build measures the decoder rather than
//! the seek (debug FLAC decode is roughly ten times slower, historically the thing that made
//! seek latency look catastrophic when it was not).

// Diagnostics module: the report formats a String. It is built once, by hand, off any media
// path — the documented `clippy.toml` exception.
#![allow(clippy::disallowed_methods)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The stages of one seek, in the order they must happen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    /// The seek generation has been bumped and published — the app's call has returned.
    Dispatched = 0,
    /// The sink handled `FlushStart` and published the ring's flush point. The pipeline now
    /// knows about the seek; nothing has been dropped or re-primed yet.
    FlushPublished = 1,
    /// The real-time callback observed that flush point and jumped the ring's read head. The
    /// stale audio is now unreachable — this is the instant the *old* audio stops.
    FlushApplied = 2,
    /// The sink wrote the first post-seek byte into the ring. The pipeline has re-primed:
    /// source re-seek, decoder resync, decode, convert, resample.
    FirstWrite = 3,
    /// The real-time callback pulled a post-seek byte. **This is the audible instant** — the
    /// number a user would recognise as "the seek took this long".
    FirstPull = 4,
}

const STAGES: usize = 5;

static ENABLED: AtomicBool = AtomicBool::new(false);
/// Nanoseconds since an arbitrary process-wide origin, captured by [`arm`].
static ARMED_AT: AtomicU64 = AtomicU64::new(0);
/// Per-stage offset from `ARMED_AT`, in nanoseconds. `0` means "not reached".
static STAGE_NS: [AtomicU64; STAGES] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// The process-wide origin all timestamps are measured from. `OnceLock` rather than a lazy
/// `Instant` so the RT callback never initialises it.
fn origin() -> Instant {
    use std::sync::OnceLock;
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

fn now_ns() -> u64 {
    origin().elapsed().as_nanos() as u64
}

/// Start timing a seek: clear the stages and take t0. Call **immediately before** the seek.
///
/// Arming from a second thread while a seek is in flight simply restarts the measurement; this
/// is a diagnostic for one seek at a time, not a concurrent profiler.
pub fn arm() {
    // Establish the origin off the RT thread, while we are allowed to.
    let t0 = now_ns();
    for s in &STAGE_NS {
        s.store(0, Ordering::Relaxed);
    }
    ARMED_AT.store(t0, Ordering::Release);
    ENABLED.store(true, Ordering::Release);
}

/// Stop recording. [`report`] still reads whatever was captured.
pub fn disarm() {
    ENABLED.store(false, Ordering::Release);
}

/// Record `stage`, if this is the first time it has been reached since [`arm`].
///
/// The whole call is behind one relaxed load when the probe is off. First-write-wins, so a
/// stage that happens repeatedly (every block, every batch) keeps the instant that matters.
#[inline]
pub fn mark(stage: Stage) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    mark_cold(stage);
}

#[inline(never)]
fn mark_cold(stage: Stage) {
    let armed = ARMED_AT.load(Ordering::Acquire);
    let t = now_ns().saturating_sub(armed).max(1); // 0 is the "not reached" sentinel
    let _ = STAGE_NS[stage as usize].compare_exchange(
        0,
        t,
        Ordering::Release,
        Ordering::Relaxed,
    );
}

/// Record `stage`, but only once `after` has already happened.
///
/// The sites for [`Stage::FirstWrite`] and [`Stage::FirstPull`] are the ordinary per-batch and
/// per-block paths — they run constantly, including in the window between [`arm`] and the seek
/// actually reaching the pipeline. Plain [`mark`] there would timestamp the last *pre*-seek
/// write, which is the opposite of the question. Gating on the preceding stage is what makes
/// "first" mean "first post-seek".
#[inline]
pub fn mark_after(stage: Stage, after: Stage) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    if STAGE_NS[after as usize].load(Ordering::Acquire) != 0 {
        mark_cold(stage);
    }
}

/// Whether the probe is currently recording — for a caller that wants to skip building
/// diagnostic values of its own.
#[inline]
pub fn is_armed() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// How long a stage took from the arming instant, or `None` if it was never reached.
pub fn at(stage: Stage) -> Option<Duration> {
    match STAGE_NS[stage as usize].load(Ordering::Acquire) {
        0 => None,
        ns => Some(Duration::from_nanos(ns)),
    }
}

/// A one-line summary of the last armed seek: cumulative time to each stage, and in brackets the
/// increment that stage added. `-` is a stage that never happened.
pub fn report() -> String {
    let names = [
        ("dispatch", Stage::Dispatched),
        ("flush published", Stage::FlushPublished),
        ("flush applied (RT)", Stage::FlushApplied),
        ("first write", Stage::FirstWrite),
        ("first pull (AUDIBLE)", Stage::FirstPull),
    ];
    let mut out = String::new();
    let mut prev = Duration::ZERO;
    for (name, stage) in names {
        match at(stage) {
            None => out.push_str(&format!("{name}: -  |  ")),
            Some(d) => {
                out.push_str(&format!(
                    "{name}: {:.2} ms (+{:.2})  |  ",
                    d.as_secs_f64() * 1000.0,
                    d.saturating_sub(prev).as_secs_f64() * 1000.0
                ));
                prev = d;
            }
        }
    }
    out
}
