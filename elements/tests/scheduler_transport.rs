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

//! TEMPORARY scheduler audit probes. Delete after the audit.

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
static XFORM_PADS: [PadDesc; 2] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &OFFERS, dynamic: false, validate: None },
];

const LAT: LatencyDesc =
    LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO };

// ---------------------------------------------------------------------------
// Burst source: emit `burst` buffers as fast as it can, then stall (produce
// nothing) for `stall`, then emit `burst` more, then EOS.
// ---------------------------------------------------------------------------
static BURST_DESC: ElementDesc = ElementDesc {
    name: "burstsrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LAT,
    make_default: None,
};

struct BurstSrc {
    burst: u64,
    produced: u64,
    stall: Duration,
    stall_started: Option<Instant>,
    stalled_done: bool,
}

impl BurstSrc {
    fn new(burst: u64, stall: Duration) -> Self {
        Self { burst, produced: 0, stall, stall_started: None, stalled_done: false }
    }
}

impl Element for BurstSrc {
    fn desc(&self) -> &'static ElementDesc { &BURST_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, ctx: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> {
        if self.produced >= self.burst && !self.stalled_done {
            let t0 = *self.stall_started.get_or_insert_with(Instant::now);
            if t0.elapsed() < self.stall {
                return Ok(Flow::Ok); // stall: no output
            }
            self.stalled_done = true;
        }
        if self.produced >= self.burst * 2 {
            return Ok(Flow::Eos);
        }
        // Emit the whole burst as ONE batch so the consumer really receives a
        // multi-buffer backlog in a single pop.
        for _ in 0..self.burst {
            let Some(mut b) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
            b.memory.as_mut_full()[..1].copy_from_slice(&[1u8]);
            b.memory.set_len(1);
            self.produced += 1;
            ctx.out(PadId(0)).push(b);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> { Ok(()) }
    fn stop(&mut self, _c: &mut Ctx) {}
}

// ---------------------------------------------------------------------------
// Active passthrough head: moves every input buffer to its output.
// ---------------------------------------------------------------------------
static PASS_DESC: ElementDesc = ElementDesc {
    name: "passhead",
    pads: &XFORM_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LAT,
    make_default: None,
};

struct PassHead;
impl Element for PassHead {
    fn desc(&self) -> &'static ElementDesc { &PASS_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, ctx: &mut Ctx, mut i: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(b) = i.pop() {
            ctx.out(PadId(1)).push(b);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> { Ok(()) }
    fn stop(&mut self, _c: &mut Ctx) {}
}

// ---------------------------------------------------------------------------
// A sink that consumes exactly ONE buffer per process() call and produces
// nothing. Passive variant (inlines into its upstream's group) and active
// variant (its own group head).
// ---------------------------------------------------------------------------
struct Counter(Arc<AtomicU64>);

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
// scheduler 6: live property-set latency on an idle-parked group.
// ---------------------------------------------------------------------------
static PROP_SRC_DESC: ElementDesc = ElementDesc {
    name: "propsrc",
    pads: &SRC_PADS,
    props: &[profluens_core::element::PropDesc {
        name: "knob",
        allowed: profluens_core::format::Constraint::Any,
        live: true,
    }],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LAT,
    make_default: None,
};
struct PropSrc { seen: Arc<AtomicU64>, ran: Arc<AtomicBool> }
impl Element for PropSrc {
    fn desc(&self) -> &'static ElementDesc { &PROP_SRC_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, _c: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> {
        self.ran.store(true, Ordering::Release);
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, e: &Event) -> Result<(), Error> {
        if matches!(e, Event::PropChanged { .. }) {
            self.seen.fetch_add(1, Ordering::Release);
        }
        Ok(())
    }
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
    assert!(dt < Duration::from_millis(3), "stop took {dt:?} (park-tick quantised?)");
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
    assert!(dt < Duration::from_millis(3), "stop-while-paused took {dt:?}");
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
    assert_eq!((np, nr), (CYCLES, CYCLES), "Paused/Resumed deliveries must balance");
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
    assert!(worst < Duration::from_millis(3), "resume latency {worst:?}");
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
    assert_eq!((np, nr), (CYCLES, CYCLES), "Paused/Resumed deliveries must balance");
}

// ---------------------------------------------------------------------------
// scheduler 9: an upstream-starved sink announces `Paused` at the gate, then falls
// through (last_progressed was true) into a pass that blocks in `up.pop()`.
// A resume landing while it is blocked there is never seen by the gate: on the
// next loop top `maybe_paused()` is already false, so `Event::Resumed` is never
// delivered and `paused_announced` stays true — the following pause then
// delivers no `Event::Paused` either.
// ---------------------------------------------------------------------------
static SLOW_SRC_DESC: ElementDesc = ElementDesc {
    name: "slowdripsrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LAT,
    make_default: None,
};
struct SlowDripSrc { last: Option<Instant> }
impl Element for SlowDripSrc {
    fn desc(&self) -> &'static ElementDesc { &SLOW_SRC_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, ctx: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> {
        let now = Instant::now();
        if let Some(t) = self.last {
            if now.duration_since(t) < Duration::from_millis(4) {
                return Ok(Flow::Ok);
            }
        }
        self.last = Some(now);
        let Some(mut b) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        b.memory.set_len(0);
        ctx.out(PadId(0)).push(b);
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> { Ok(()) }
    fn stop(&mut self, _c: &mut Ctx) {}
}

static DRAIN_SINK_DESC: ElementDesc = ElementDesc {
    name: "drainsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LAT,
    make_default: None,
};
struct DrainSink { seen_p: Arc<AtomicU64>, seen_r: Arc<AtomicU64>, ran: Arc<AtomicBool> }
impl Element for DrainSink {
    fn desc(&self) -> &'static ElementDesc { &DRAIN_SINK_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, _c: &mut Ctx, mut i: Inputs<'_>) -> Result<Flow, Error> {
        self.ran.store(true, Ordering::Release);
        while i.pop().is_some() {}
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
    assert_eq!((np, nr), (CYCLES, CYCLES), "Paused/Resumed deliveries must balance");
}
