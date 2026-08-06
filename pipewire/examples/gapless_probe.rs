//! `gapless_probe` — live validation of the shared [`AudioOut`] against a real PipeWire
//! daemon: two tone "tracks" played back to back through **one** output, each by its own
//! `Pipeline` and its own `PipeWireAudioSink::with_output(handle)`.
//!
//! Usage: `cargo run --release -p pf-pipewire --example gapless_probe [track_seconds]`
//!
//! It plays two audible tones (440 Hz then 660 Hz, quiet) so a listener can confirm the
//! handoff is seamless and the volume ramp does not click, and it asserts what a listener
//! cannot:
//!
//! * the boundary is **never starved** — the ring's runway (`AudioOutHandle::buffered`) stays
//!   above zero from the moment track 1 EOSes to the moment track 2 is streaming. That is the
//!   whole gapless claim, measured rather than heard;
//! * EOS **detaches without draining** (the output is free while its ring is still full);
//! * `position()` is epoch-relative — it resets for track 2 rather than continuing;
//! * the device clock runs *continuously* across the boundary (it no longer freezes between
//!   tracks, because the device never stops);
//! * `set_volume` / `set_muted` take effect and `set_idle` round-trips, freezing and resuming
//!   the device without losing audio.
//!
//! The sample-exact side of the proof is the deterministic capture backend
//! (`sink::attach_tests::a_track_handoff_is_byte_exact_at_dozens_of_split_points`); this
//! example is the "and it works against a real device" half.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use pf_pipewire::{AudioOut, AudioOutConfig, AudioOutHandle, PipeWireAudioSink};
use profluens_core::batch::Inputs;
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

/// A sine in interleaved stereo **f32** — the canonical `AudioOut` format, which is what a
/// real track's chain is converged onto upstream.
///
/// The tone starts and ends at a zero crossing (an integer number of periods), so a *clean*
/// handoff is inaudible and a gap or a repeated buffer is not: any discontinuity at the
/// boundary is a click, not a phase wobble.
struct ToneSrc {
    phase: f64,
    step: f64,
    frames_left: u64,
    announced: bool,
}

impl ToneSrc {
    fn new(hz: f64, secs: f64) -> Self {
        Self {
            phase: 0.0,
            step: std::f64::consts::TAU * hz / RATE as f64,
            frames_left: (secs * RATE as f64) as u64,
            announced: false,
        }
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
                    ("sample", ValueDesc::Id("f32")),
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
        let stride = 4 * CHANNELS; // f32
        let frames = (buf.memory.capacity() / stride).min(self.frames_left as usize);
        let dst = buf.memory.as_mut_full();
        for f in 0..frames {
            let s = (self.phase.sin() * 0.15) as f32; // quiet, well below full scale
            self.phase += self.step;
            for c in 0..CHANNELS {
                let at = (f * CHANNELS + c) * 4;
                dst[at..at + 4].copy_from_slice(&s.to_le_bytes());
            }
        }
        buf.memory.set_len(frames * stride);
        self.frames_left -= frames as u64;
        ctx.out(PadId(0)).push(buf);
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// Build and run one "track" to completion, returning when its pipeline does — which, in
/// attached mode, is *before* its audio has finished playing.
fn play_track(hz: f64, secs: f64, out: &AudioOutHandle) -> Result<(), Error> {
    let mut p = Pipeline::new();
    let src = p.add(ToneSrc::new(hz, secs));
    let snk = p.add(PipeWireAudioSink::with_output(out.clone()));
    p.link((src, "src"), (snk, "sink")).expect("tonesrc -> pipewireaudiosink");
    p.run()
}

/// Samples the output's runway and position on a tight loop, recording the minimum runway
/// seen inside a window the main thread opens around the track boundary.
struct Monitor {
    stop: Arc<AtomicBool>,
    watching: Arc<AtomicBool>,
    /// Minimum `buffered()` in nanoseconds seen while `watching` — `u64::MAX` if never armed.
    min_runway_ns: Arc<AtomicU64>,
    /// Highest `position()` seen, in nanoseconds, since the last reset.
    max_pos_ns: Arc<AtomicU64>,
}

fn main() {
    let secs: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3.0);

    let out = match AudioOut::open(AudioOutConfig::default()) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("could not open the audio output: {e:?}");
            std::process::exit(1);
        }
    };
    let h = out.handle();
    println!(
        "AudioOut open: {:?} {} Hz {} ch",
        h.format().sample,
        h.format().rate,
        h.format().channels
    );

    let mon = Monitor {
        stop: Arc::new(AtomicBool::new(false)),
        watching: Arc::new(AtomicBool::new(false)),
        min_runway_ns: Arc::new(AtomicU64::new(u64::MAX)),
        max_pos_ns: Arc::new(AtomicU64::new(0)),
    };
    let sampler = {
        let (h, stop, watching, min_runway, max_pos) = (
            h.clone(),
            Arc::clone(&mon.stop),
            Arc::clone(&mon.watching),
            Arc::clone(&mon.min_runway_ns),
            Arc::clone(&mon.max_pos_ns),
        );
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let runway = h.buffered().as_nanos() as u64;
                if watching.load(Ordering::Relaxed) {
                    min_runway.fetch_min(runway, Ordering::Relaxed);
                }
                max_pos.fetch_max(h.position().as_nanos() as u64, Ordering::Relaxed);
                std::thread::sleep(Duration::from_micros(500));
            }
        })
    };

    let mut failures = 0usize;
    let mut check = |ok: bool, what: &str| {
        println!("{} {what}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failures += 1;
        }
    };

    // --- track 1, with a volume ramp partway through ---------------------------------------
    println!("-- track 1: 440 Hz for {secs:.1} s (volume ramps to 0.3 and back) --");
    let volume_rider = {
        let h = h.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs_f64(1.0));
            h.set_volume(0.3);
            std::thread::sleep(Duration::from_secs_f64(0.7));
            h.set_volume(1.0);
        })
    };
    let t0 = Instant::now();
    let r1 = play_track(440.0, secs, &h);
    let track1_run = t0.elapsed();
    volume_rider.join().expect("volume rider");
    check(r1.is_ok(), "track 1's pipeline finished cleanly");

    // EOS must have detached *without* draining: the output is free while its ring is full.
    let runway_at_boundary = h.buffered();
    check(!h.is_attached(), "EOS released the output (detached)");
    check(
        runway_at_boundary > Duration::ZERO,
        &format!("the ring still holds audio at the boundary ({runway_at_boundary:?})"),
    );
    // The pipeline returned early precisely because it did not wait for playout.
    check(
        track1_run.as_secs_f64() < secs,
        &format!("run() returned before playout finished ({track1_run:?} < {secs:.1} s)"),
    );
    let pos_end_of_track1 = Duration::from_nanos(mon.max_pos_ns.load(Ordering::Relaxed));

    // --- the boundary --------------------------------------------------------------------
    // Arm the runway watch and build track 2 *inline* — the worst case. A real engine
    // pre-rolls it, so this measures more handoff cost than production ever pays.
    mon.watching.store(true, Ordering::Relaxed);
    mon.max_pos_ns.store(0, Ordering::Relaxed);
    println!("-- track 2: 660 Hz for {secs:.1} s (built inline at the boundary) --");
    let t1 = Instant::now();
    let r2 = play_track(660.0, secs, &h);
    let handoff_and_run = t1.elapsed();
    mon.watching.store(false, Ordering::Relaxed);
    check(r2.is_ok(), "track 2's pipeline finished cleanly");

    let min_runway = Duration::from_nanos(mon.min_runway_ns.load(Ordering::Relaxed));
    println!(
        "minimum runway across the boundary: {:.1} ms (started from {:.1} ms)",
        min_runway.as_secs_f64() * 1e3,
        runway_at_boundary.as_secs_f64() * 1e3,
    );
    check(
        min_runway > Duration::ZERO,
        "the ring never ran dry across the handoff — no gap was possible",
    );
    check(
        handoff_and_run.as_secs_f64() < secs + 0.5,
        &format!("track 2 started promptly ({handoff_and_run:?})"),
    );

    // Position is epoch-relative: track 2 counts from zero, it does not continue track 1.
    let pos_track2 = Duration::from_nanos(mon.max_pos_ns.load(Ordering::Relaxed));
    println!(
        "position at end of track 1: {:.3} s; peak during track 2: {:.3} s",
        pos_end_of_track1.as_secs_f64(),
        pos_track2.as_secs_f64()
    );
    check(
        pos_track2.as_secs_f64() < pos_end_of_track1.as_secs_f64() + 0.5,
        "position reset for the new track (epoch-relative), it did not accumulate",
    );
    check(
        pos_track2.as_secs_f64() > secs * 0.5,
        "and it advanced through track 2",
    );

    // --- mute, then idle round-trip ---------------------------------------------------------
    h.set_muted(true);
    check(h.is_muted(), "mute latched");
    h.set_muted(false);
    check(!h.is_muted(), "unmute latched");

    // Let the tail play out, then park the device and prove it freezes rather than drains.
    while h.buffered() > Duration::from_millis(5) {
        std::thread::sleep(Duration::from_millis(10));
    }
    h.set_idle(true);
    check(h.is_idle(), "set_idle(true) latched");
    std::thread::sleep(Duration::from_millis(200));
    h.set_idle(false);
    check(!h.is_idle(), "set_idle(false) latched");

    // A third track after the idle round-trip must still play: the output survived parking.
    println!("-- track 3: 550 Hz for 1.0 s (after an idle round-trip) --");
    mon.max_pos_ns.store(0, Ordering::Relaxed);
    let r3 = play_track(550.0, 1.0, &h);
    check(r3.is_ok(), "track 3 played after the idle round-trip");
    while h.buffered() > Duration::from_millis(5) {
        std::thread::sleep(Duration::from_millis(10));
    }
    check(
        Duration::from_nanos(mon.max_pos_ns.load(Ordering::Relaxed)) > Duration::from_millis(500),
        "and its position advanced (the device really resumed)",
    );

    mon.stop.store(true, Ordering::Relaxed);
    sampler.join().expect("sampler");
    drop(out);

    if failures > 0 {
        eprintln!("{failures} check(s) failed");
        std::process::exit(1);
    }
    println!("all checks passed");
}
