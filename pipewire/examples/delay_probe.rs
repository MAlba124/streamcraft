//! `delay_probe` — live validation of the sink's output-latency compensation against a real
//! PipeWire daemon: `tonesrc ! pipewireaudiosink`, printing the write-side position, the
//! measured output delay (`pw_time`) and the compensated (audible) position side by side.
//!
//! Usage: `cargo run --release -p pf-pipewire --example delay_probe [seconds]`
//!
//! It plays an audible 440 Hz tone (quiet), so a listener can confirm the sink still works,
//! and asserts the invariants the compensation is supposed to hold:
//!
//! * the reported delay is plausible for a playback device (`0 < delay < 1 s`),
//! * the compensated position never exceeds the write-side position,
//! * the gap between them tracks the reported delay,
//! * both positions advance,
//! * after a seek the position never reads before the target.
//!
//! Point it at a specific node — a high-latency `pw-loopback` target, say — with
//! `PIPEWIRE_NODE=<id>` or by making that node the default sink.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use pf_pipewire::PipeWireAudioSink;
use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;

const RATE: u32 = 48_000;
const CHANNELS: usize = 2;
const TONE_HZ: f64 = 440.0;

static FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: "rate", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "channels", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "sample", allowed: ConstraintDesc::Any, preferred: None },
];
static OFFERS: [OfferDesc; 1] = [OfferDesc { family: "audio/raw", fields: &FIELDS }];
static PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: true,
    validate: None,
}];
static DESC: ElementDesc = ElementDesc {
    name: "tonesrc",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// A 440 Hz sine in interleaved stereo s16, announced as `audio/raw` so the sink configures
/// its device from it exactly as a decoder would.
struct ToneSrc {
    phase: f64,
    frames_left: u64,
    total_frames: u64,
    announced: bool,
}

impl ToneSrc {
    fn new(secs: f64) -> Self {
        let total_frames = (secs * RATE as f64) as u64;
        Self { phase: 0.0, frames_left: total_frames, total_frames, announced: false }
    }
}

impl Element for ToneSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.announced {
            ctx.announce_format(
                PadId(0),
                "audio/raw",
                &[
                    ("rate", ValueDesc::Int(RATE as i64)),
                    ("channels", ValueDesc::Int(CHANNELS as i64)),
                    ("sample", ValueDesc::Id("s16")),
                ],
            );
            self.announced = true;
        }
        if self.frames_left == 0 {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else {
            return Ok(Flow::Ok); // pool full → backpressure
        };
        let stride = 2 * CHANNELS; // s16
        let frames = (buf.memory.capacity() / stride).min(self.frames_left as usize);
        let step = std::f64::consts::TAU * TONE_HZ / RATE as f64;
        let dst = buf.memory.as_mut_full();
        for f in 0..frames {
            let s = (self.phase.sin() * 6000.0) as i16; // quiet, well below full scale
            self.phase += step;
            for c in 0..CHANNELS {
                let at = (f * CHANNELS + c) * 2;
                dst[at..at + 2].copy_from_slice(&s.to_le_bytes());
            }
        }
        buf.memory.set_len(frames * stride);
        self.frames_left -= frames as u64;
        ctx.out(PadId(0)).push(buf);
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // Be seek-aware, or the seek below ends the run: the scheduler discards the buffers
        // this source had already generated ahead into the graph, and with `frames_left`
        // already spent on them it would EOS immediately. Re-basing to the target keeps the
        // tone playing so the post-seek position can actually be watched.
        if matches!(event, Event::FlushStart) {
            if let Some(ns) = ctx.seek_target().and_then(|t| t.to_time.nanos()) {
                let at = (ns as u128 * RATE as u128 / 1_000_000_000) as u64;
                self.frames_left = self.total_frames.saturating_sub(at);
            }
        }
        Ok(())
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// One sampled observation of the three quantities under test.
#[derive(Clone, Copy)]
struct Sample {
    write: f64,
    pos: f64,
    delay: f64,
}

fn main() {
    let secs: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(6.0);

    let mut p = Pipeline::new();
    let src = p.add(ToneSrc::new(secs));
    let sink = PipeWireAudioSink::new();
    let ctrl = sink.control();
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("tonesrc -> pipewireaudiosink");
    let seek = p.seek_handle();

    let done = Arc::new(AtomicBool::new(false));
    let run = {
        let done = Arc::clone(&done);
        std::thread::spawn(move || {
            let r = p.run();
            done.store(true, Ordering::Release);
            // Drain `LatencyChanged` here rather than concurrently: `Bus::try_recv` borrows
            // the pipeline, which this thread owns. Nothing is lost by waiting — the message
            // is classified `Critical`, so the bus never drops it.
            // COLD: one drain after the pipeline has stopped, not a per-buffer allocation.
            #[allow(clippy::disallowed_methods)]
            let mut latency: Vec<(u64, u64)> = Vec::new();
            while let Some(msg) = p.bus().try_recv() {
                if let BusMessage::LatencyChanged { old, new } = msg {
                    latency.push((old.0, new.0));
                }
            }
            (r, latency)
        })
    };

    println!("  t     write-side   position    delay     gap");
    let started = Instant::now();
    // COLD: the probe harness's own observation log (~10/s), off the media path entirely.
    #[allow(clippy::disallowed_methods)]
    let mut samples: Vec<Sample> = Vec::new();
    let mut seeked = false;
    let mut post_seek_min = f64::INFINITY;
    let mut seek_landed = false;
    // Seek *backward*, to a target far enough below the current position that "the seek has
    // landed" is unambiguous — and, because the ring is then empty and the write-side counter
    // sits exactly on the target, this is precisely the case where an uncompensated
    // subtraction would report a position before the target the user asked for.
    let seek_to = (secs / 4.0).max(0.5);

    while !done.load(Ordering::Acquire) {
        let s = Sample {
            write: ctrl.write_position_secs(),
            pos: ctrl.position_secs(),
            delay: ctrl.output_delay().as_secs_f64(),
        };
        println!(
            "{:6.2}  {:9.3} s {:9.3} s {:7.1} ms {:7.1} ms",
            started.elapsed().as_secs_f64(),
            s.write,
            s.pos,
            s.delay * 1e3,
            (s.write - s.pos) * 1e3,
        );
        if s.delay > 0.0 {
            samples.push(s);
        }
        if seek_landed {
            post_seek_min = post_seek_min.min(s.pos);
        }
        // Halfway through, seek back and watch the position never dip below the target.
        if !seeked && started.elapsed() > Duration::from_secs_f64(secs / 2.0) {
            let before = ctrl.position_secs();
            println!("  -- seek {before:.3} s -> {seek_to:.3} s --");
            seek.seek(0, Timestamp::from_nanos((seek_to * 1e9) as u64));
            seeked = true;
            // The seek is asynchronous: the sink re-bases its counters when the scheduler
            // delivers `FlushStart`, not when `seek()` returns. Sampling from here would just
            // record *pre*-seek positions, so first wait (briefly) for it to land, then watch
            // the window where the ring is empty and the floor is the only thing holding the
            // position at the target.
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline && !done.load(Ordering::Acquire) {
                if ctrl.position_secs() < before - 0.2 {
                    seek_landed = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            // Sample tightly across the landing: that is where a naive subtraction dips.
            for _ in 0..400 {
                if seek_landed {
                    post_seek_min = post_seek_min.min(ctrl.position_secs());
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let (result, latency_msgs) = run.join().expect("run thread joined");
    match result {
        Ok(()) => println!("pipeline finished cleanly"),
        Err(e) => println!("pipeline error: {e:?}"),
    }
    for (old, new) in &latency_msgs {
        println!(
            "bus: LatencyChanged {:.1} ms -> {:.1} ms",
            *old as f64 / 1e6,
            *new as f64 / 1e6
        );
    }

    // --- assertions -------------------------------------------------------------------
    let mut failures = 0;
    let mut check = |ok: bool, what: &str| {
        println!("{} {what}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failures += 1;
        }
    };

    check(!samples.is_empty(), "the device reported a delay at all");
    if samples.is_empty() {
        std::process::exit(1);
    }

    let min_d = samples.iter().map(|s| s.delay).fold(f64::INFINITY, f64::min);
    let max_d = samples.iter().map(|s| s.delay).fold(0.0, f64::max);
    let mean_d = samples.iter().map(|s| s.delay).sum::<f64>() / samples.len() as f64;
    println!(
        "delay: min {:.1} ms  mean {:.1} ms  max {:.1} ms  (n={})",
        min_d * 1e3,
        mean_d * 1e3,
        max_d * 1e3,
        samples.len()
    );
    check(min_d > 0.0, "delay > 0");
    check(max_d < 1.0, "delay < 1 s");
    check(
        samples.iter().all(|s| s.pos <= s.write + 1e-9),
        "compensated position never exceeds write-side",
    );
    // The gap is the delay, except where the post-seek floor is deliberately holding the
    // position up (which makes the gap *smaller* than the delay, never larger).
    check(
        samples.iter().all(|s| s.write - s.pos <= s.delay + 1e-3),
        "gap never exceeds the reported delay",
    );
    let settled: Vec<&Sample> = samples.iter().filter(|s| s.pos > s.delay * 2.0).collect();
    check(
        !settled.is_empty()
            && settled.iter().all(|s| ((s.write - s.pos) - s.delay).abs() < 5e-3),
        "away from the seek floor, gap == delay (within 5 ms)",
    );
    check(
        samples.last().map(|s| s.pos).unwrap_or(0.0) > samples[0].pos,
        "position advances",
    );
    check(
        samples.windows(2).all(|w| w[1].pos >= w[0].pos - 1e-9 || w[1].pos < w[0].pos - 0.5),
        "position is monotonic except across the seek",
    );
    check(seeked && seek_landed, "the seek landed (so the post-seek policy was exercised)");
    // Informational on the bus (core has no dynamic-latency consumer — see the sink's
    // `post_latency_change`), but it must actually be posted, once, with the measured value.
    check(
        latency_msgs.iter().any(|(_, new)| {
            (*new as f64 / 1e9 - mean_d).abs() < 0.02 // within 20 ms of what we measured
        }),
        "the measured delay was announced as BusMessage::LatencyChanged",
    );
    check(latency_msgs.len() <= 4, "LatencyChanged is rate-limited, not per graph cycle");
    if seek_landed {
        println!(
            "seek target {seek_to:.3} s; lowest position observed after it landed: \
             {post_seek_min:.3} s"
        );
        check(post_seek_min >= seek_to - 1e-3, "position never reads before the seek target");
    }

    if failures > 0 {
        eprintln!("{failures} check(s) failed");
        std::process::exit(1);
    }
    println!("all checks passed");
}
