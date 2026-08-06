//! Regression: waiting at a growing file's frontier must cost **no CPU**
//! (spec: Scheduling — a group with nothing to do parks, it never spins).
//!
//! `growing_src.rs` proves the same property exactly, from `SchedulerStats`. This
//! file is the independent cross-check in the currency that actually matters — CPU
//! time — and it lives on its own because the process-wide counter it reads only
//! means anything when this pipeline is the only thing running in the binary
//! (cargo autodiscovery gives each test file its own process; see `park_cpu.rs`,
//! which carries the same constraint for the same reason).
//!
//! Test setup does ordinary blocking IO; the reactor rule the `disallowed_methods`
//! lint enforces is about `Element` code, hence the file-wide `#[allow]`.
#![allow(clippy::disallowed_methods)]

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
use profluens_elements::io::GrowingFileSrc;

/// Whole-process CPU time (utime + stime) from `/proc/self/stat`, in ms. CPU time —
/// unlike wall time — is not inflated by scheduler delays on a loaded machine, so the
/// assertion below is load-independent. (Verbatim from `park_cpu.rs`.)
fn process_cpu_ms() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("read /proc/self/stat");
    let after = stat.rsplit(')').next().expect("stat has a comm field");
    let fields: Vec<&str> = after.split_whitespace().collect();
    let utime: u64 = fields[11].parse().expect("utime");
    let stime: u64 = fields[12].parse().expect("stime");
    // USER_HZ is 100 on every Linux ABI this crate targets (x86_64/aarch64).
    (utime + stime) * 1000 / 100
}

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

static COUNT_DESC: ElementDesc = ElementDesc {
    name: "growingcountsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

struct CountSink(Arc<Mutex<u64>>);

impl Element for CountSink {
    fn desc(&self) -> &'static ElementDesc {
        &COUNT_DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut n = self.0.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            *n += buf.memory.len() as u64;
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// A pipeline that spends a whole second caught up with a stalled writer burns
/// (almost) no CPU: it has no read to submit, so every pass is idle and the group
/// parks on the idle eventcount, re-checking the watermark once per 10 ms tick.
#[test]
fn waiting_at_the_frontier_costs_no_cpu() {
    const HAVE: usize = 64_000;
    let path = std::env::temp_dir().join(format!("pf_growing_cpu_{}.bin", std::process::id()));
    File::create(&path).expect("create");
    let data: Vec<u8> = (0..HAVE).map(|i| (i as u64).wrapping_mul(0x9E37_79B9) as u8).collect();
    {
        let mut f = OpenOptions::new().append(true).open(&path).expect("append open");
        f.write_all(&data).expect("write");
        f.sync_data().expect("sync");
    }

    let (src, frontier) = GrowingFileSrc::new(&path);
    let received = Arc::new(Mutex::new(0u64));
    let mut p = Pipeline::new();
    let s = p.add(src);
    let k = p.add(CountSink(Arc::clone(&received)));
    p.link((s, "src"), (k, "sink")).expect("link");
    frontier.advance(HAVE as u64);

    let cpu_before = process_cpu_ms();
    let t0 = std::time::Instant::now();
    let watcher = Arc::clone(&received);
    let run = std::thread::spawn(move || p.run());

    // Let it drain the published bytes, then sit at the frontier for a full second.
    for _ in 0..500 {
        if *watcher.lock().unwrap() == HAVE as u64 {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    std::thread::sleep(Duration::from_millis(1000));
    frontier.finish(HAVE as u64);
    run.join().expect("joined").expect("run");

    let wall = t0.elapsed();
    let cpu = process_cpu_ms().saturating_sub(cpu_before);
    assert_eq!(*received.lock().unwrap(), HAVE as u64, "every published byte was delivered");
    assert!(wall >= Duration::from_millis(900), "the frontier wait really happened: {wall:?}");
    // A spin burns ~a core per group for the whole second (> 1000 ms CPU); parked
    // groups leave only the ~64 KB of real work plus ~100 park wakeups per group.
    // 400 ms keeps a wide non-flaky margin while failing loudly on any spin.
    assert!(
        cpu < 400,
        "the frontier wait burned {cpu} ms CPU over a ~{wall:?} run — spinning?"
    );
    let _ = std::fs::remove_file(&path);
}
