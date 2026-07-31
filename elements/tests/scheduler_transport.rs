//! Scheduler transport regression tests: pause/resume delivery, and stop latency.
//!
//! Every case here reproduced a real defect. `Event::Resumed` was delivered only from the pause
//! gate's own park, so a resume landing while a group was anywhere else — blocked in `up.pop()`,
//! inside a long `process()`, or falling through the gate on a tick — was dropped, and left the
//! group's `paused_announced` latch set so the *next* `Paused` was suppressed too. A device sink
//! latches its hardware-paused flag off that pair, so a dropped `Resumed` left the audio device
//! paused for good (30 cycles delivered 1 `Paused` and 0 `Resumed` before the fix). `stop()`
//! likewise never woke parked groups, leaving shutdown quantised to the 10 ms idle backstop, or
//! 100 ms when paused.
//!
//! The timing bounds are deliberately loose — milliseconds against microsecond-scale fixes — so
//! they fail on a return of tick quantisation rather than on ordinary scheduler jitter.
//!
//! Not covered: a sink starved across an *entire* pause window. It stays blocked in `up.pop()`
//! the whole time and so never reaches the gate, delivering neither `Paused` nor `Resumed` —
//! balanced, but a device sink is never told the pipeline paused. Whether that matters is a
//! design question (a starved sink has nothing to play), and a test for it would be asserting
//! a precondition the scheduler does not establish, so there is deliberately no test here
//! rather than one that passes vacuously at 0/0.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;
use profluens_core::id::PadId;

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
const LAT: LatencyDesc =
    LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO };

// ---------------------------------------------------------------------------
// A sink that consumes exactly ONE buffer per process() call and produces
// nothing. Passive variant (inlines into its upstream's group) and active
// variant (its own group head).
// ---------------------------------------------------------------------------
static SLOW_PASSIVE_DESC: ElementDesc = ElementDesc {
    name: "slowpassive",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LAT,
    make_default: None,
};
static SLOW_ACTIVE_DESC: ElementDesc = ElementDesc {
    name: "slowactive",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LAT,
    make_default: None,
};

struct SlowSink { passive: bool, n: Arc<AtomicU64> }
impl Element for SlowSink {
    fn desc(&self) -> &'static ElementDesc {
        if self.passive { &SLOW_PASSIVE_DESC } else { &SLOW_ACTIVE_DESC }
    }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, _ctx: &mut Ctx, mut i: Inputs<'_>) -> Result<Flow, Error> {
        if i.pop().is_some() {
            self.n.fetch_add(1, Ordering::Release);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> { Ok(()) }
    fn stop(&mut self, _c: &mut Ctx) {}
}

// ---------------------------------------------------------------------------
// scheduler 3: stop latency while every group is idle-parked.
// ---------------------------------------------------------------------------
static IDLE_SRC_DESC: ElementDesc = ElementDesc {
    name: "idlesrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LAT,
    make_default: None,
};
struct IdleSrc(Arc<AtomicBool>);
impl Element for IdleSrc {
    fn desc(&self) -> &'static ElementDesc { &IDLE_SRC_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, _c: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> {
        self.0.store(true, Ordering::Release);
        Ok(Flow::Ok) // never produces: the group idles forever
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> { Ok(()) }
    fn stop(&mut self, _c: &mut Ctx) {}
}

#[test]
fn stop_wakes_idle_parked_groups_at_once() {
    let ran = Arc::new(AtomicBool::new(false));
    let n = Arc::new(AtomicU64::new(0));
    let mut p = Pipeline::new();
    let src = p.add(IdleSrc(Arc::clone(&ran)));
    let snk = p.add(SlowSink { passive: false, n });
    p.link((src, "src"), (snk, "sink")).expect("link");
    let stop = p.stop_handle();
    let h = std::thread::spawn(move || p.run());
    while !ran.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::sleep(Duration::from_millis(30)); // let everyone settle into park_idle
    let t0 = Instant::now();
    stop.stop();
    let _ = h.join().expect("join");
    let dt = t0.elapsed();
    println!("scheduler: stop -> run() returned in {dt:?}");
    // Loose on purpose: the fix took this from 9.2 ms to ~0.1 ms, so anything under the 10 ms
    // backstop tick proves the wake is event-driven. A tight bound only measures machine load.
    assert!(dt < Duration::from_millis(8), "stop took {dt:?} (park-tick quantised?)");
}

// ---------------------------------------------------------------------------
// scheduler 4: stop latency while paused (the 100 ms pause-gate tick).
// ---------------------------------------------------------------------------
#[test]
fn stop_wakes_paused_groups_at_once() {
    let ran = Arc::new(AtomicBool::new(false));
    let n = Arc::new(AtomicU64::new(0));
    let mut p = Pipeline::new();
    let src = p.add(IdleSrc(Arc::clone(&ran)));
    let snk = p.add(SlowSink { passive: false, n });
    p.link((src, "src"), (snk, "sink")).expect("link");
    let stop = p.stop_handle();
    let pause = p.pause_handle();
    let h = std::thread::spawn(move || p.run());
    while !ran.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(1));
    }
    pause.pause();
    std::thread::sleep(Duration::from_millis(150));
    let t0 = Instant::now();
    stop.stop();
    let _ = h.join().expect("join");
    let dt = t0.elapsed();
    println!("scheduler: stop-while-paused -> run() returned in {dt:?}");
    // 60 ms -> ~0.1 ms; the pause gate's own backstop is 100 ms, so well under it is the signal.
    assert!(dt < Duration::from_millis(50), "stop-while-paused took {dt:?}");
}

// ---------------------------------------------------------------------------
// scheduler 5: resume latency for an idle-parked group.
// ---------------------------------------------------------------------------
static RESUME_SRC_DESC: ElementDesc = ElementDesc {
    name: "resumesrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LAT,
    make_default: None,
};
struct ResumeSrc { seen_resume: Arc<AtomicU64>, seen_paused: Arc<AtomicU64>, ran: Arc<AtomicBool> }
impl Element for ResumeSrc {
    fn desc(&self) -> &'static ElementDesc { &RESUME_SRC_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, _c: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> {
        self.ran.store(true, Ordering::Release);
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, e: &Event) -> Result<(), Error> {
        match e {
            Event::Resumed => { self.seen_resume.fetch_add(1, Ordering::Release); }
            Event::Paused => { self.seen_paused.fetch_add(1, Ordering::Release); }
            _ => {}
        }
        Ok(())
    }
    fn stop(&mut self, _c: &mut Ctx) {}
}

// ---------------------------------------------------------------------------
// scheduler 7: every pause must deliver exactly one Event::Paused and every resume
// exactly one Event::Resumed. A resume that lands while the group is NOT inside
// `park_while_paused` skips the gate entirely.
// ---------------------------------------------------------------------------
#[test]
fn paused_and_resumed_deliveries_balance() {
    const CYCLES: u64 = 60;
    let seen_r = Arc::new(AtomicU64::new(0));
    let seen_p = Arc::new(AtomicU64::new(0));
    let ran = Arc::new(AtomicBool::new(false));
    let n = Arc::new(AtomicU64::new(0));
    let mut p = Pipeline::new();
    let src = p.add(ResumeSrc {
        seen_resume: Arc::clone(&seen_r),
        seen_paused: Arc::clone(&seen_p),
        ran: Arc::clone(&ran),
    });
    let snk = p.add(SlowSink { passive: false, n });
    p.link((src, "src"), (snk, "sink")).expect("link");
    let stop = p.stop_handle();
    let pause = p.pause_handle();
    let h = std::thread::spawn(move || p.run());
    while !ran.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(1));
    }
    for i in 0..CYCLES {
        pause.pause();
        // Sweep the delay across the gate's fall-through pass + park_idle window.
        std::thread::sleep(Duration::from_micros(200 + (i % 30) * 500));
        pause.resume();
        std::thread::sleep(Duration::from_millis(2));
    }
    std::thread::sleep(Duration::from_millis(50));
    let (np, nr) = (seen_p.load(Ordering::Acquire), seen_r.load(Ordering::Acquire));
    stop.stop();
    let _ = h.join().expect("join");
    println!("scheduler: {CYCLES} pause/resume cycles -> {np} Paused, {nr} Resumed delivered");
    // Balance is the invariant, not one delivery per `pause()` call: a pause/resume pair that
    // lands entirely inside one `process()` is legitimately coalesced and delivers neither. The
    // bug was a delivered `Paused` with no matching `Resumed`, which leaves a device latched off
    // for good — so assert `np == nr`, and that the run actually exercised the path. Asserting
    // `(CYCLES, CYCLES)` is strictly stronger than the invariant and fails under load.
    assert_eq!(np, nr, "every delivered Paused needs its Resumed (got {np} / {nr})");
    assert!(np > 0, "no pause was observed at all — the test exercised nothing");
}

#[test]
fn resume_is_delivered_promptly() {
    let seen = Arc::new(AtomicU64::new(0));
    let ran = Arc::new(AtomicBool::new(false));
    let n = Arc::new(AtomicU64::new(0));
    let mut p = Pipeline::new();
    let src = p.add(ResumeSrc { seen_resume: Arc::clone(&seen), seen_paused: Arc::new(AtomicU64::new(0)), ran: Arc::clone(&ran) });
    let snk = p.add(SlowSink { passive: false, n });
    p.link((src, "src"), (snk, "sink")).expect("link");
    let stop = p.stop_handle();
    let pause = p.pause_handle();
    let h = std::thread::spawn(move || p.run());
    while !ran.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(1));
    }
    let mut worst = Duration::ZERO;
    let mut over_1ms = 0;
    for i in 0..200u64 {
        pause.pause();
        std::thread::sleep(Duration::from_millis(3));
        let t0 = Instant::now();
        pause.resume();
        while seen.load(Ordering::Acquire) <= i {
            std::hint::spin_loop();
        }
        let dt = t0.elapsed();
        if dt > Duration::from_millis(1) {
            over_1ms += 1;
        }
        worst = worst.max(dt);
    }
    println!("scheduler: {over_1ms}/200 resumes took > 1 ms");
    stop.stop();
    let _ = h.join().expect("join");
    println!("scheduler: worst resume -> Event::Resumed latency {worst:?}");
    // Worst of 200 was 535 us once the wake is published; the 10 ms tick is the thing to catch.
    assert!(worst < Duration::from_millis(8), "resume latency {worst:?}");
}

// ---------------------------------------------------------------------------
// scheduler 8: same balance check, but on a *busy* pipeline. A pass that made
// progress makes the pause gate return `Tick` and fall through to another pass;
// that pass ends in `park_idle`, OUTSIDE the gate. A resume landing in that
// window skips the gate entirely on the next loop top (`maybe_paused()` is
// already false) — `Event::Resumed` is never delivered and `paused_announced`
// stays true, so the NEXT pause delivers no `Event::Paused` either.
// ---------------------------------------------------------------------------
static BUSY_SRC_DESC: ElementDesc = ElementDesc {
    name: "busysrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LAT,
    make_default: None,
};
struct BusySrc;
impl Element for BusySrc {
    fn desc(&self) -> &'static ElementDesc { &BUSY_SRC_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, ctx: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> {
        let Some(mut b) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        b.memory.set_len(0);
        ctx.out(PadId(0)).push(b);
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> { Ok(()) }
    fn stop(&mut self, _c: &mut Ctx) {}
}

static BUSY_SINK_DESC: ElementDesc = ElementDesc {
    name: "busysink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LAT,
    make_default: None,
};
struct BusySink {
    paused: bool,
    seen_p: Arc<AtomicU64>,
    seen_r: Arc<AtomicU64>,
    ran: Arc<AtomicBool>,
}
impl Element for BusySink {
    fn desc(&self) -> &'static ElementDesc { &BUSY_SINK_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, _c: &mut Ctx, mut i: Inputs<'_>) -> Result<Flow, Error> {
        self.ran.store(true, Ordering::Release);
        if self.paused {
            return Ok(Flow::Ok); // the pipewire-sink pattern: stage, don't render
        }
        while i.pop().is_some() {}
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, e: &Event) -> Result<(), Error> {
        match e {
            Event::Paused => { self.paused = true; self.seen_p.fetch_add(1, Ordering::Release); }
            Event::Resumed => { self.paused = false; self.seen_r.fetch_add(1, Ordering::Release); }
            _ => {}
        }
        Ok(())
    }
    fn stop(&mut self, _c: &mut Ctx) {}
}

#[test]
fn paused_and_resumed_balance_on_a_busy_pipeline() {
    const CYCLES: u64 = 40;
    let seen_p = Arc::new(AtomicU64::new(0));
    let seen_r = Arc::new(AtomicU64::new(0));
    let ran = Arc::new(AtomicBool::new(false));
    let mut p = Pipeline::new();
    p.set_pool(64, 32);
    let src = p.add(BusySrc);
    let snk = p.add(BusySink {
        paused: false,
        seen_p: Arc::clone(&seen_p),
        seen_r: Arc::clone(&seen_r),
        ran: Arc::clone(&ran),
    });
    p.link((src, "src"), (snk, "sink")).expect("link");
    let stop = p.stop_handle();
    let pause = p.pause_handle();
    let h = std::thread::spawn(move || p.run());
    while !ran.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(1));
    }
    for i in 0..CYCLES {
        pause.pause();
        // Sweep the resume delay across the fall-through pass + 10 ms park_idle.
        std::thread::sleep(Duration::from_micros(200 + (i % 40) * 400));
        pause.resume();
        std::thread::sleep(Duration::from_millis(3));
    }
    std::thread::sleep(Duration::from_millis(50));
    let (np, nr) = (seen_p.load(Ordering::Acquire), seen_r.load(Ordering::Acquire));
    stop.stop();
    let _ = h.join().expect("join");
    println!("scheduler: {CYCLES} pause/resume cycles on a busy pipeline -> {np} Paused, {nr} Resumed");
    // Balance is the invariant, not one delivery per `pause()` call: a pause/resume pair that
    // lands entirely inside one `process()` is legitimately coalesced and delivers neither. The
    // bug was a delivered `Paused` with no matching `Resumed`, which leaves a device latched off
    // for good — so assert `np == nr`, and that the run actually exercised the path. Asserting
    // `(CYCLES, CYCLES)` is strictly stronger than the invariant and fails under load.
    assert_eq!(np, nr, "every delivered Paused needs its Resumed (got {np} / {nr})");
    assert!(np > 0, "no pause was observed at all — the test exercised nothing");
}

// ---------------------------------------------------------------------------
// scheduler 10: a sink that is BUSY (long `process()`) when the resume lands.
// It announced `Paused` at the gate, then `last_progressed == true` made the
// gate return `Tick` and fall through. Every subsequent pass keeps progressing
// (the ring has data), so it never re-enters `park_while_paused` — and the
// resume is therefore never observed by the gate.
// ---------------------------------------------------------------------------
static SLOWPROC_SINK_DESC: ElementDesc = ElementDesc {
    name: "slowprocsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LAT,
    make_default: None,
};
struct SlowProcSink { seen_p: Arc<AtomicU64>, seen_r: Arc<AtomicU64>, ran: Arc<AtomicBool> }
impl Element for SlowProcSink {
    fn desc(&self) -> &'static ElementDesc { &SLOWPROC_SINK_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, _c: &mut Ctx, mut i: Inputs<'_>) -> Result<Flow, Error> {
        self.ran.store(true, Ordering::Release);
        if i.pop().is_some() {
            // A real device sink spends most of a buffer's playback inside push().
            let t = Instant::now();
            while t.elapsed() < Duration::from_millis(8) {
                std::hint::spin_loop();
            }
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, e: &Event) -> Result<(), Error> {
        match e {
            Event::Paused => { self.seen_p.fetch_add(1, Ordering::Release); }
            Event::Resumed => { self.seen_r.fetch_add(1, Ordering::Release); }
            _ => {}
        }
        Ok(())
    }
    fn stop(&mut self, _c: &mut Ctx) {}
}

#[test]
fn resume_reaches_a_sink_busy_in_process() {
    const CYCLES: u64 = 30;
    let seen_p = Arc::new(AtomicU64::new(0));
    let seen_r = Arc::new(AtomicU64::new(0));
    let ran = Arc::new(AtomicBool::new(false));
    let mut p = Pipeline::new();
    p.set_pool(64, 32);
    let src = p.add(BusySrc);
    let snk = p.add(SlowProcSink {
        seen_p: Arc::clone(&seen_p),
        seen_r: Arc::clone(&seen_r),
        ran: Arc::clone(&ran),
    });
    p.link((src, "src"), (snk, "sink")).expect("link");
    let stop = p.stop_handle();
    let pause = p.pause_handle();
    let h = std::thread::spawn(move || p.run());
    while !ran.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(1));
    }
    for _ in 0..CYCLES {
        pause.pause();
        std::thread::sleep(Duration::from_millis(25));
        pause.resume();
        std::thread::sleep(Duration::from_millis(25));
    }
    std::thread::sleep(Duration::from_millis(50));
    let (np, nr) = (seen_p.load(Ordering::Acquire), seen_r.load(Ordering::Acquire));
    stop.stop();
    let _ = h.join().expect("join");
    println!("scheduler: {CYCLES} pause/resume cycles, busy sink -> {np} Paused, {nr} Resumed");
    // Balance is the invariant, not one delivery per `pause()` call: a pause/resume pair that
    // lands entirely inside one `process()` is legitimately coalesced and delivers neither. The
    // bug was a delivered `Paused` with no matching `Resumed`, which leaves a device latched off
    // for good — so assert `np == nr`, and that the run actually exercised the path. Asserting
    // `(CYCLES, CYCLES)` is strictly stronger than the invariant and fails under load.
    assert_eq!(np, nr, "every delivered Paused needs its Resumed (got {np} / {nr})");
    assert!(np > 0, "no pause was observed at all — the test exercised nothing");
}
