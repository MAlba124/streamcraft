//! `pipewireaudiosink` — plays interleaved PCM through PipeWire (spec: Milestone
//! applications — play an audio file).
//!
//! It is an **active** element on its own thread. Its `process()` pushes decoded PCM into a
//! bounded lock-free SPSC byte ring ([`crate::ring`]); a device backend's **real-time**
//! callback pulls from that ring to fill device buffers. The RT callback must never lock or
//! allocate, so the ring's consumer side is wait-free — the only blocking (a full-ring producer
//! parking; the EOS drain) happens on the profluens render thread. When the ring is full,
//! `process()` blocks, so the whole pipeline is paced by the audio device (backpressure is the
//! clock; full clock-slaving through PipeWire for A/V sync is a follow-up). The PCM format is
//! data-dependent, so the sink advertises a broad `dynamic` `audio/raw` sink and configures
//! itself from the runtime `FormatChange` a decoder announces (spec: dynamic caps).
//!
//! ## Two modes
//!
//! * **Single owner** ([`PipeWireAudioSink::new`]) — the historical arrangement, unchanged.
//!   The element opens a private [`AudioOut`] sized from the negotiated format, drains the
//!   ring on `Event::Eos`, and tears the device down on `stop()`. Every existing pipeline
//!   (pfplay, NVR, the sdl3 player) uses this.
//! * **Attached** ([`PipeWireAudioSink::with_output`]) — the gapless arrangement (spec:
//!   gapless.md, Phase 2). The device belongs to the application, not the pipeline; the
//!   element is a thin producer that attaches on its first byte and, at EOS, **detaches
//!   without draining** so the next track's sink can queue audio behind the tail still in
//!   flight. `stop()` never touches the shared output.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::clock::{Clock, ClockWait};
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, FixedFormat, OfferDesc, Value, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

use crate::out::{
    frames_to_ns, open_epoch, position_ns, AudioOut, AudioOutConfig, AudioOutHandle,
    CanonicalFormat, Playback, SampleFormat,
};
use crate::ring::Producer;

// The unit tests below build `Playback` states by hand and check the `pw_time` conversion and
// the position packing; `use super::*` in the test module picks these up.
#[cfg(test)]
use crate::out::{pack_mono, MONO_EPOCH_MASK, MONO_POS_BITS, MONO_POS_MASK};
#[cfg(test)]
use crate::pw_backend::{out_delay_ns, PwTime};
#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicU32};

// --- audio/raw sink offer (broad; the concrete format arrives via dynamic caps) --------

const FAMILY: &str = "audio/raw";
/// Seconds of PCM the single-owner mode's private ring holds — the historical ~0.5 s jitter
/// buffer.
const OWNED_RING_SECS: f32 = 0.5;
static SAMPLE_VALUES: [ValueDesc; 5] = [
    ValueDesc::Id("u8"),
    ValueDesc::Id("s16"),
    ValueDesc::Id("s24"),
    ValueDesc::Id("s32"),
    ValueDesc::Id("f32"),
];
static SINK_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: "rate", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "channels", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "sample", allowed: ConstraintDesc::Set(&SAMPLE_VALUES), preferred: None },
];
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &SINK_FIELDS }];
static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: true, // format is announced at runtime
    validate: None,
}];
static DESC: ElementDesc = ElementDesc {
    name: "pipewireaudiosink",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active, // its own thread; paces the graph by blocking on the device
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: true,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

// --- the element -----------------------------------------------------------------------

/// A cloneable handle to control and observe playback (spec: Clocking — pause). Obtain it
/// from [`PipeWireAudioSink::control`] before adding the sink to the pipeline. Pausing makes
/// the device render silence and hold its buffer — no audio is lost and the whole pipeline
/// backpressures to a stop; resuming continues from exactly where it left off.
///
/// In attached mode this observes the *shared* [`AudioOut`]'s counters, so `pause` pauses the
/// whole output. The per-track, app-facing view is [`AudioOutHandle`].
#[derive(Clone)]
pub struct AudioControl {
    pb: Arc<Playback>,
}

impl AudioControl {
    pub fn pause(&self) {
        self.pb.paused.store(true, Ordering::Relaxed);
    }
    pub fn resume(&self) {
        self.pb.paused.store(false, Ordering::Relaxed);
    }
    /// Flip paused↔playing; returns the new state (`true` = now paused).
    pub fn toggle(&self) -> bool {
        !self.pb.paused.fetch_xor(true, Ordering::Relaxed)
    }
    pub fn is_paused(&self) -> bool {
        self.pb.paused.load(Ordering::Relaxed)
    }
    /// Seconds of audio the listener has actually **heard** — the audible play position,
    /// within the current attach epoch. See [`position_ns`] for the delay compensation, the
    /// post-seek floor and the monotonic clamp it applies.
    pub fn position_secs(&self) -> f64 {
        self.position_ns() as f64 / 1_000_000_000.0
    }

    /// [`Self::position_secs`] in nanoseconds — the shared implementation.
    fn position_ns(&self) -> u64 {
        position_ns(&self.pb)
    }

    /// Seconds of audio handed to the device — the **write-side** position, which leads what
    /// is audible by [`Self::output_delay`]. This is what this method used to return under the
    /// name `position_secs`; it is kept for callers that genuinely mean "how much have I
    /// pushed" (buffer accounting) rather than "where is playback".
    ///
    /// Unlike [`Self::position_secs`] this is *not* epoch-relative: under a shared
    /// [`AudioOut`] it counts every track the output has rendered.
    pub fn write_position_secs(&self) -> f64 {
        let rate = self.pb.rate.load(Ordering::Relaxed);
        if rate == 0 {
            0.0
        } else {
            self.pb.frames.load(Ordering::Relaxed) as f64 / rate as f64
        }
    }

    /// The measured delay between handing a frame to PipeWire and hearing it: the stream's
    /// converter plus the graph plus the device, as PipeWire reports it in `pw_time`. Zero
    /// until the stream reaches STREAMING and the first snapshot lands.
    ///
    /// Exposed raw so A/V logic can compensate the (deliberately un-compensated) device
    /// clock itself — see [`AudioDeviceClock`].
    pub fn output_delay(&self) -> Duration {
        Duration::from_nanos(self.pb.out_delay_ns.load(Ordering::Relaxed))
    }
}

/// The audio device as a [`Clock`] (spec: Clocking — a sink provides a device
/// clock): `now()` is nanoseconds of audio the hardware has *actually rendered*
/// (`clock_frames / rate`), so a pipeline mastered on it paces to the DAC's true
/// rate — the audio-master A/V-sync arrangement. It reads zero until the device is
/// configured and running, stops while paused (video freezes with the audio, as it
/// should), and is never re-based by seeks. Waits against it poll (~1 ms slices):
/// the RT callback only bumps atomics and can never notify a waiter.
///
/// This clock is **write-side and stays that way**: it counts frames handed to PipeWire, not
/// frames a speaker has produced, and it is deliberately *not* reduced by the measured output
/// delay. Two reasons. It is the pipeline's timebase, so shifting it would move every
/// `wait_until` deadline in the graph at once and re-open the latency model that
/// `path_latency` computes statically at `run()`; and its "monotonic, never adjusted"
/// contract is what video pacing is built on, whereas the measured delay moves when the
/// graph re-routes. A consumer that wants presentation-accurate alignment subtracts
/// [`AudioControl::output_delay`] from this clock itself — output-latency compensation is a
/// consumer-side concern, and [`AudioControl::position_secs`] is the ready-made answer for
/// the common case of "where is playback".
pub struct AudioDeviceClock {
    pb: Arc<Playback>,
}

impl Clock for AudioDeviceClock {
    fn now(&self) -> Timestamp {
        let rate = self.pb.rate.load(Ordering::Relaxed);
        if rate == 0 {
            return Timestamp::ZERO;
        }
        Timestamp::from_nanos(frames_to_ns(self.pb.clock_frames.load(Ordering::Relaxed), rate))
    }

    fn new_wait(&self) -> ClockWait {
        ClockWait::polling(
            Arc::new(AudioDeviceClock { pb: Arc::clone(&self.pb) }),
            std::time::Duration::from_millis(1),
        )
    }
}

/// Which [`AudioOut`] this sink renders through, and who owns it.
enum Mode {
    /// Single owner: the element opens a private output at configure time and drops it on
    /// `stop()`. `None` until the format is known (and again after `stop()`).
    Owned(Option<AudioOut>),
    /// Attached: the application owns the output; the element only borrows its producer end.
    Shared(AudioOutHandle),
}

/// Plays interleaved PCM through PipeWire. Construct with [`PipeWireAudioSink::new`] (single
/// owner) or [`PipeWireAudioSink::with_output`] (attached to an app-owned [`AudioOut`]).
pub struct PipeWireAudioSink {
    mode: Mode,
    /// Producer end of the PCM ring while attached. In single-owner mode it is taken at
    /// configure; in attached mode **lazily, on the first byte actually written**.
    producer: Option<Producer>,
    /// The negotiated format has been resolved — and, attached, checked against the output's
    /// canonical format.
    started: bool,
    /// Pause latch + play-position counters, shared with the render callback and the app's
    /// [`AudioControl`] handle. Attached, these are the shared output's counters.
    playback: Arc<Playback>,
    /// Change detector for [`Self::post_latency_change`]: the delay last announced on the
    /// bus, and when. Element-thread only — never shared with the RT callback.
    posted_delay_ns: u64,
    last_latency_post: Option<Instant>,
}

impl Default for PipeWireAudioSink {
    fn default() -> Self {
        Self {
            mode: Mode::Owned(None),
            producer: None,
            started: false,
            playback: Arc::new(Playback::default()),
            posted_delay_ns: 0,
            last_latency_post: None,
        }
    }
}

impl PipeWireAudioSink {
    /// A single-owner sink: it opens its own PipeWire device from whatever format the stream
    /// negotiates, and closes it on `stop()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// A sink that renders through an application-owned [`AudioOut`] — the gapless mode.
    ///
    /// The element becomes a pure producer:
    ///
    /// * It attaches **on its first actual byte write**, not in `start()`. A pre-rolled
    ///   next-track pipeline must be able to sit paused, fully negotiated, without touching
    ///   the ring the current track is still playing out of.
    /// * Attach is **exclusive**. The engine sequences handoffs; two sinks attached at once
    ///   would interleave PCM into one ring, so a second attach is a loud error rather than
    ///   something absorbed.
    /// * On `Event::Eos` it **detaches without draining** — the ring keeps whatever it holds
    ///   and the device plays it out while the next sink queues behind it. That tail is the
    ///   gapless seam.
    /// * `stop()` detaches (if attached) and never tears the output down.
    ///
    /// The negotiated input format must equal [`AudioOutHandle::format`] exactly; anything
    /// else is a wiring bug in the chain that was supposed to converge on it, and is reported
    /// as an error rather than played as garbage.
    pub fn with_output(out: AudioOutHandle) -> Self {
        let playback = Arc::clone(out.playback());
        Self {
            mode: Mode::Shared(out),
            producer: None,
            started: false,
            playback,
            posted_delay_ns: 0,
            last_latency_post: None,
        }
    }

    /// A handle to pause/resume and read the play position. Call before
    /// `pipeline.add(sink)` (the pipeline takes ownership of the sink); the handle keeps
    /// working for the pipeline's lifetime.
    pub fn control(&self) -> AudioControl {
        AudioControl { pb: Arc::clone(&self.playback) }
    }

    /// Configure from the negotiated `audio/raw` format. Idempotent — the first format wins
    /// (a mid-stream format change would need a device reconfigure, a follow-up).
    ///
    /// Single owner: opens the device. Attached: verifies the format matches the output's.
    fn configure(&mut self, ctx: &Ctx, f: &FixedFormat) -> Result<(), Error> {
        if self.started {
            return Ok(());
        }
        let fmt = read_format(ctx, f)?;

        if let Mode::Shared(handle) = &self.mode {
            let want = handle.format();
            if fmt != want {
                return Err(Error::Resource(format!(
                    "pipewireaudiosink: negotiated {:?} {} Hz {} ch but the shared AudioOut \
                     renders {:?} {} Hz {} ch — the chain must converge on the canonical format",
                    fmt.sample, fmt.rate, fmt.channels, want.sample, want.rate, want.channels
                )));
            }
            self.started = true;
            return Ok(());
        }

        let out = AudioOut::open_pw(
            AudioOutConfig { format: fmt, ring_secs: OWNED_RING_SECS },
            Arc::clone(&self.playback),
        )?;
        self.producer = Some(out.handle().attach()?);
        self.mode = Mode::Owned(Some(out));
        self.started = true;
        Ok(())
    }

    /// Hand the producer end back to a shared output (a no-op when not attached, and in
    /// single-owner mode where the ring dies with the element).
    fn detach(&mut self) {
        if let Mode::Shared(handle) = &self.mode {
            if let Some(producer) = self.producer.take() {
                handle.detach(producer);
            }
        }
    }

    /// Seek (spec: flush/seek): drop the PCM buffered in the ring so the new position's audio
    /// plays immediately instead of after the stale tail, and re-base the play-position
    /// counter to the seek target's frame — relative to the current attach epoch.
    ///
    /// **Honest trade-off** (attached mode): the ring is a single byte stream shared with
    /// whatever is still in flight. A seek issued in the instant *after* the previous track
    /// detached but before its tail has played out therefore drops that tail too. This is
    /// accepted rather than engineered around: the user is jumping, so the previous track's
    /// last few hundred milliseconds are not what they asked to hear, and the alternative (a
    /// per-producer partition of the ring) would put a branch in the RT pull path for a case
    /// that is audible only as "the seek was clean".
    fn flush_start(&mut self, ctx: &Ctx) {
        // Not attached in shared mode: nothing of ours is in the ring, and flushing would
        // destroy the *previous* track's in-flight tail. A pre-rolled next-track pipeline
        // flushes on the way to its start position, and that must be inert here.
        if matches!(self.mode, Mode::Shared(_)) && self.producer.is_none() {
            return;
        }
        if let Some(producer) = &self.producer {
            producer.flush();
        }
        if let Some(t) = ctx.seek_target() {
            // The seek target is stream time; this sink's play-position unit is PCM frames —
            // derive it from the negotiated rate (published at configure; zero before then,
            // when there is no position to reset anyway).
            let rate = self.playback.rate.load(Ordering::Relaxed) as u128;
            if let Some(ns) = t.to_time.nanos() {
                let frames = (ns as u128 * rate / 1_000_000_000) as u64;
                // Order matters for `position_ns`: re-base the write-side counter and the
                // post-seek floor first, publish the new epoch last. A reader that sampled
                // pre-seek state then fails its compare-exchange on the epoch word and retries
                // against this one, so a backward seek can never leave the monotonic clamp
                // pinned at the old position. The re-base is *epoch-relative*: the counter
                // includes everything the shared output has rendered, so the target frame is
                // measured from where this attach epoch began (zero in single-owner mode,
                // where the arithmetic is unchanged).
                let epoch_start = self.playback.epoch_start_frames.load(Ordering::Relaxed);
                self.playback.frames.store(epoch_start.saturating_add(frames), Ordering::Relaxed);
                open_epoch(&self.playback, ns);
            }
        }
        self.playback.flush_stream.store(true, Ordering::Relaxed);
    }

    /// Announce a materially changed output delay on the bus as `BusMessage::LatencyChanged`.
    ///
    /// **Informational only, by design.** Core sizes its latency model once: `run()` calls
    /// `Pipeline::compute_in_latency`, which sums the *static* `LatencyDesc` of each element
    /// along the longest path, and installs the result into every group's `Ctx` via
    /// `set_path_latency` before `start()`. There is no re-computation hook and no
    /// bus-to-pipeline feedback, so posting this changes no scheduling — it is for the
    /// application (and the introspection tap, the only in-tree consumer today). Rebuilding
    /// core's latency machinery to consume it is out of scope here. It is also why this
    /// element's `LatencyDesc` above stays zero: it is a `&'static` property read before any
    /// device exists, so there is nowhere to put a measurement that only the running stream
    /// can produce.
    ///
    /// Called from the element thread (the only place with a `&mut Ctx`), which is why the
    /// RT callback merely publishes an atomic and never touches the bus itself.
    fn post_latency_change(&mut self, ctx: &mut Ctx) {
        /// Below this, a change is quantum jitter rather than a route change.
        const MATERIAL_NS: u64 = 10_000_000; // 10 ms
        /// Rate limit: the delay is re-measured every graph cycle (~23 ms here).
        const MIN_INTERVAL: Duration = Duration::from_millis(500);

        let now_ns = self.playback.out_delay_ns.load(Ordering::Relaxed);
        if now_ns.abs_diff(self.posted_delay_ns) < MATERIAL_NS {
            return;
        }
        if self.last_latency_post.is_some_and(|t| t.elapsed() < MIN_INTERVAL) {
            return;
        }
        let old = Timestamp::from_nanos(self.posted_delay_ns);
        ctx.post(BusMessage::LatencyChanged { old, new: Timestamp::from_nanos(now_ns) });
        self.posted_delay_ns = now_ns;
        self.last_latency_post = Some(Instant::now());
    }
}

/// Read the `audio/raw` triple out of a negotiated format.
fn read_format(ctx: &Ctx, f: &FixedFormat) -> Result<CanonicalFormat, Error> {
    let rate = ctx
        .field_id("rate")
        .and_then(|id| f.get(id))
        .and_then(int_value)
        .ok_or(Error::Todo("pipewireaudiosink: negotiated format has no rate"))? as u32;
    let channels = ctx
        .field_id("channels")
        .and_then(|id| f.get(id))
        .and_then(int_value)
        .ok_or(Error::Todo("pipewireaudiosink: negotiated format has no channels"))?
        as u32;
    let sample = ctx
        .field_id("sample")
        .and_then(|id| f.get(id))
        .and_then(|v| match v {
            Value::Id(vid) => ctx.value_name(vid),
            _ => None,
        })
        .ok_or(Error::Todo("pipewireaudiosink: negotiated format has no sample format"))?;
    let sample = SampleFormat::from_name(sample).ok_or_else(|| {
        Error::Resource(format!("pipewireaudiosink: unsupported sample '{sample}'"))
    })?;
    Ok(CanonicalFormat { sample, rate, channels })
}

fn int_value(v: Value) -> Option<i64> {
    match v {
        Value::Int(n) => Some(n),
        _ => None,
    }
}

impl Element for PipeWireAudioSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn provide_clock(&mut self) -> Option<Arc<dyn Clock>> {
        // Master the pipeline on the DAC's rendered-frame count (unless the app
        // forced a clock). Offered before the device is configured; the clock just
        // reads zero until audio actually starts flowing. Attached to a shared output it is
        // the *output's* clock, which keeps running across a track boundary.
        Some(Arc::new(AudioDeviceClock { pb: Arc::clone(&self.playback) }))
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        // The device is configured lazily, once the format is known (FormatChange or a
        // fully-fixed link-time format read from `ctx.negotiated`). Attaching is lazier
        // still — see `with_output`.
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.started {
            // No FormatChange yet? Try a fully-fixed link-time format.
            match ctx.negotiated(PadId(0)).cloned() {
                Some(f) => self.configure(ctx, &f)?,
                None => return Ok(Flow::Ok), // wait for the format before touching the device
            }
        }
        // Before the paused bail below: a route change while paused is still worth announcing.
        self.post_latency_change(ctx);
        // A single pooled buffer can be ~0.7 s of audio while the device ring holds ~0.5 s, so
        // `push` blocks for most of a buffer's playback. Push in small slices and bail the
        // moment a seek is requested (the generation changes), so the scheduler can run the
        // flush promptly rather than after the whole batch has drained (spec: flush/seek).
        // The partial data already pushed is dropped by the sink's `FlushStart` handler.
        const SLICE: usize = 8 * 1024; // ~46 ms at 44.1 kHz stereo s16 — well under the ring
        // While paused, consume nothing (spec: Clocking + flush/seek): the RT
        // callback holds the ring, so pushing would park this group *inside*
        // process where a resume's `Event::Resumed` can never reach it. Input
        // stays staged; the re-prime happens on the first pass after resume.
        if self.playback.paused.load(Ordering::Relaxed) {
            return Ok(Flow::Ok);
        }
        let start_gen = ctx.seek_gen();
        while let Some(buf) = inputs.pop() {
            let data = buf.memory.data();
            if data.is_empty() {
                continue; // an empty buffer is not a write, and must not trigger an attach
            }
            // Lazy attach: the *first actual byte* is what claims the shared output, so a
            // pre-rolled, paused next-track pipeline never touches the ring the current track
            // is playing out of.
            if self.producer.is_none() {
                if let Mode::Shared(handle) = &self.mode {
                    let producer = handle.attach()?;
                    self.producer = Some(producer);
                }
            }
            let Some(producer) = &self.producer else { break };
            let mut off = 0;
            while off < data.len() {
                let end = (off + SLICE).min(data.len());
                // Interruptible push: bail the instant a seek is requested — even mid-slice on
                // a full ring, which is the paused case (the RT callback holds the ring, so a
                // plain blocking push would never return and the flush could not propagate).
                if !producer.push_interruptible(&data[off..end], || ctx.seek_gen() != start_gen) {
                    if producer.is_closed() {
                        // The output went away underneath us (the app dropped its `AudioOut`,
                        // or the device thread died). Report it — the alternative is a
                        // pipeline that silently plays nothing.
                        return Err(Error::Resource(
                            "pipewireaudiosink: the audio output closed while attached".into(),
                        ));
                    }
                    return Ok(Flow::Ok); // seek: stop; the loop top flushes
                }
                off = end;
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FormatChange(f) => self.configure(ctx, f)?,
            Event::FlushStart => self.flush_start(ctx),
            // The pipeline delivers EOS at end-of-stream.
            Event::Eos => match &self.mode {
                // Single owner: play out the buffered audio before `stop()` closes the device.
                Mode::Owned(_) => {
                    if let Some(producer) = &self.producer {
                        producer.drain();
                    }
                }
                // Shared: detach *without* draining. The bytes already in the ring keep
                // playing while the next track's sink attaches behind them — that overlap is
                // exactly what makes the boundary sample-continuous.
                Mode::Shared(_) => self.detach(),
            },
            // Transport pause (spec: Clocking — pause is a clock op): hold the
            // hardware — the RT callback renders silence and keeps the ring, so no
            // audio is lost and the device clock (and with it the whole pipeline's
            // running time) freezes. Resume continues exactly where it left off.
            Event::Paused => self.playback.paused.store(true, Ordering::Relaxed),
            Event::Resumed => self.playback.paused.store(false, Ordering::Relaxed),
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        match &mut self.mode {
            Mode::Owned(out) => {
                if let Some(producer) = self.producer.take() {
                    producer.close(); // wake anything parked on the ring
                }
                // Dropping the output asks the PipeWire loop to quit and joins its thread.
                drop(out.take());
            }
            // Shared: give the producer back and leave the output completely alone. Idempotent
            // — a second `stop()`, or a stop that never attached, takes `None`.
            Mode::Shared(handle) => {
                if let Some(producer) = self.producer.take() {
                    handle.detach(producer);
                }
            }
        }
        self.started = false;
    }
}

impl Drop for PipeWireAudioSink {
    /// A sink dropped without `stop()` — a pipeline unwinding on an error — must still hand
    /// the producer back in shared mode: dropping it closes the ring, which would kill the
    /// application's output for every future track. Single-owner mode wants precisely the
    /// opposite (the ring dies with the element), so it is left alone.
    fn drop(&mut self) {
        self.detach();
    }
}

// The PipeWire integration itself needs a running server (verified by the `play` example on a
// real desktop). The device-independent part with the concurrency subtlety — the lock-free
// PCM hand-off ring the RT callback pulls from — is unit-tested in [`crate::ring`]. The
// `pw_time` unit conversion and the two position policies are device-independent too, and
// are unit-tested below against hand-built snapshots; the live end is `examples/delay_probe`.
// The attach/detach seam is proven sample-exactly against the deterministic capture backend
// in `mod attach_tests` at the end of this file.

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Playback` in the state the RT callback would leave it: configured rate, some
    /// write-side frames rendered, a measured delay.
    fn pb_at(rate: u32, frames: u64, delay_ns: u64) -> AudioControl {
        let pb = Playback { rate: AtomicU32::new(rate), ..Playback::default() };
        pb.frames.store(frames, Ordering::Relaxed);
        pb.out_delay_ns.store(delay_ns, Ordering::Relaxed);
        AudioControl { pb: Arc::new(pb) }
    }

    // --- tick -> ns conversion ---------------------------------------------------------

    #[test]
    fn delay_ticks_convert_with_the_pw_time_rate_fraction() {
        // The usual case: rate is 1/48000, so a tick is one frame at 48 kHz.
        // 4800 ticks = 100 ms.
        let t = PwTime { rate_num: 1, rate_denom: 48_000, delay: 4_800, ..PwTime::default() };
        assert_eq!(out_delay_ns(t, 48_000), 100_000_000);

        // 44.1 kHz graph: 441 ticks = 10 ms exactly.
        let t = PwTime { rate_num: 1, rate_denom: 44_100, delay: 441, ..PwTime::default() };
        assert_eq!(out_delay_ns(t, 44_100), 10_000_000);
    }

    #[test]
    fn delay_ticks_convert_with_an_odd_rate_fraction() {
        // A non-unit numerator: ticks of 5/1000 s = 5 ms each, so 7 ticks = 35 ms. This is
        // the case that a naive `delay / denom` would get wrong by the numerator.
        let t = PwTime { rate_num: 5, rate_denom: 1_000, delay: 7, ..PwTime::default() };
        assert_eq!(out_delay_ns(t, 48_000), 35_000_000);

        // A rate that does not divide a second evenly: 1/44100 ticks, 44_100 ticks = 1 s.
        let t = PwTime { rate_num: 1, rate_denom: 44_100, delay: 44_100, ..PwTime::default() };
        assert_eq!(out_delay_ns(t, 44_100), 1_000_000_000);

        // Rounding is truncating, not wrong-by-a-factor: 1 tick at 1/44100 is 22675.7 ns.
        let t = PwTime { rate_num: 1, rate_denom: 44_100, delay: 1, ..PwTime::default() };
        assert_eq!(out_delay_ns(t, 44_100), 22_675);
    }

    #[test]
    fn queued_and_buffered_convert_in_the_stream_rate_not_the_graph_rate() {
        // The graph runs at 48 kHz (so does `rate`), the stream at 44.1 kHz. The delay term
        // uses the fraction; the frame counts use the stream rate.
        let t = PwTime {
            rate_num: 1,
            rate_denom: 48_000,
            delay: 4_800,      // 100 ms
            queued: 4_410,     // 100 ms at 44.1 kHz
            buffered: 441,     // 10 ms at 44.1 kHz
        };
        assert_eq!(out_delay_ns(t, 44_100), 210_000_000);
    }

    #[test]
    fn negative_and_degenerate_snapshots_never_panic() {
        // "it is acceptable to clamp negative delays to 0" — pipewire/stream.h.
        let t = PwTime { rate_num: 1, rate_denom: 48_000, delay: -9_600, ..PwTime::default() };
        assert_eq!(out_delay_ns(t, 48_000), 0);

        // A zero denominator (an unconfigured snapshot) must not divide by zero.
        let t = PwTime { rate_num: 1, rate_denom: 0, delay: 4_800, buffered: 480, ..PwTime::default() };
        assert_eq!(out_delay_ns(t, 48_000), 10_000_000); // delay dropped, buffered still counts

        // A zero stream rate must not divide by zero either.
        let t = PwTime { rate_num: 1, rate_denom: 48_000, delay: 4_800, queued: 99, buffered: 99 };
        assert_eq!(out_delay_ns(t, 0), 100_000_000);

        // Absurd values saturate instead of wrapping or panicking.
        let t = PwTime { rate_num: u32::MAX, rate_denom: 1, delay: i64::MAX, ..PwTime::default() };
        assert_eq!(out_delay_ns(t, 48_000), u64::MAX);
    }

    // --- the compensated position ------------------------------------------------------

    #[test]
    fn position_is_write_side_minus_the_delay() {
        // 48000 frames written = 1 s write-side, 100 ms of it still in flight.
        let c = pb_at(48_000, 48_000, 100_000_000);
        assert!((c.write_position_secs() - 1.0).abs() < 1e-9);
        assert!((c.position_secs() - 0.9).abs() < 1e-6);
        assert_eq!(c.output_delay(), Duration::from_millis(100));
        assert!(c.position_secs() <= c.write_position_secs());
    }

    #[test]
    fn position_is_zero_before_the_device_is_configured() {
        let c = pb_at(0, 0, 0);
        assert_eq!(c.position_secs(), 0.0);
        assert_eq!(c.write_position_secs(), 0.0);
        assert_eq!(c.output_delay(), Duration::ZERO);
    }

    #[test]
    fn position_never_reads_before_zero_at_the_start_of_a_stream() {
        // Less audio written than the delay: the naive subtraction is negative.
        let c = pb_at(48_000, 480, 100_000_000); // 10 ms written, 100 ms delay
        assert_eq!(c.position_secs(), 0.0);
    }

    #[test]
    fn position_never_reads_before_the_seek_target() {
        // Post-seek state: `frames` re-based to 30 s, floor at 30 s, ring empty.
        let c = pb_at(48_000, 30 * 48_000, 100_000_000);
        c.pb.pos_floor_ns.store(30_000_000_000, Ordering::Relaxed);
        c.pb.pos_mono.store(pack_mono(1, 30_000_000), Ordering::Relaxed);
        assert!((c.position_secs() - 30.0).abs() < 1e-6, "held at the target, not 29.9");

        // Still held while the write side is inside the delay window.
        c.pb.frames.store(30 * 48_000 + 2_400, Ordering::Relaxed); // +50 ms, delay is 100 ms
        assert!((c.position_secs() - 30.0).abs() < 1e-6);

        // Once the write side passes target + delay, the subtraction takes over.
        c.pb.frames.store(30 * 48_000 + 9_600, Ordering::Relaxed); // +200 ms
        assert!((c.position_secs() - 30.1).abs() < 1e-6, "got {}", c.position_secs());
    }

    #[test]
    fn a_relative_seek_computed_from_the_position_does_not_drift() {
        // The example's arrow-key seek is `position_secs() + delta`. Without the floor each
        // hop would lose `delay`; with it, ten hops land exactly where they should.
        let c = pb_at(48_000, 0, 150_000_000);
        let mut target = 0.0f64;
        for _ in 0..10 {
            target = c.position_secs() + 10.0;
            // The sink's FlushStart handler, in its documented order.
            let ns = (target * 1e9) as u64;
            c.pb.frames.store(ns / 1_000_000_000 * 48_000, Ordering::Relaxed);
            c.pb.pos_floor_ns.store(ns, Ordering::Relaxed);
            let epoch = (c.pb.pos_mono.load(Ordering::Relaxed) >> MONO_POS_BITS) + 1;
            c.pb.pos_mono.store(pack_mono(epoch, ns / 1_000), Ordering::Release);
        }
        assert!((target - 100.0).abs() < 1e-6, "ten +10 s hops landed at {target}");
    }

    #[test]
    fn a_jittering_delay_never_walks_the_position_backwards() {
        let c = pb_at(48_000, 48_000, 100_000_000); // 1 s written, 100 ms delay -> 0.9 s
        let first = c.position_secs();
        assert!((first - 0.9).abs() < 1e-6);

        // The next graph cycle measures a quantum more delay while the write side has not
        // moved: naively 0.877 s, i.e. backwards.
        c.pb.out_delay_ns.store(123_000_000, Ordering::Relaxed);
        let second = c.position_secs();
        assert!(second >= first, "{second} < {first}");
        assert!((second - first).abs() < 1e-3, "clamped, not jumped: {second}");

        // A delay that shrinks again is free to move the position forward.
        c.pb.out_delay_ns.store(80_000_000, Ordering::Relaxed);
        assert!((c.position_secs() - 0.92).abs() < 1e-6);
    }

    #[test]
    fn a_backward_seek_resets_the_monotonic_clamp() {
        let c = pb_at(48_000, 60 * 48_000, 100_000_000);
        assert!((c.position_secs() - 59.9).abs() < 1e-6); // arms the clamp at 59.9 s

        // Seek back to 10 s, in the FlushStart order.
        c.pb.frames.store(10 * 48_000, Ordering::Relaxed);
        c.pb.pos_floor_ns.store(10_000_000_000, Ordering::Relaxed);
        let epoch = (c.pb.pos_mono.load(Ordering::Relaxed) >> MONO_POS_BITS).wrapping_add(1);
        c.pb.pos_mono.store(pack_mono(epoch, 10_000_000), Ordering::Release);

        assert!((c.position_secs() - 10.0).abs() < 1e-6, "got {}", c.position_secs());
    }

    #[test]
    fn the_clamp_word_packs_and_survives_epoch_wraparound() {
        assert_eq!(pack_mono(0, 0), 0);
        assert_eq!(pack_mono(1, 5) >> MONO_POS_BITS, 1);
        assert_eq!(pack_mono(7, 12_345) & MONO_POS_MASK, 12_345);
        // The epoch is masked into its 20 bits, so a wrapping bump can never bleed into the
        // position field and inflate it.
        let wrapped = pack_mono(MONO_EPOCH_MASK.wrapping_add(1), 42);
        assert_eq!(wrapped & MONO_POS_MASK, 42);
        assert_eq!(wrapped >> MONO_POS_BITS, 0);
        // A position beyond the field saturates rather than wrapping into the epoch.
        assert_eq!(pack_mono(3, u64::MAX) & MONO_POS_MASK, MONO_POS_MASK);
        assert_eq!(pack_mono(3, u64::MAX) >> MONO_POS_BITS, 3);
    }

    #[test]
    fn concurrent_readers_agree_and_never_observe_a_regression() {
        // The compare-exchange loop is the only shared mutable state on the read path;
        // hammer it from several threads and assert each thread sees a monotonic sequence.
        let c = pb_at(48_000, 0, 20_000_000);
        let stop = Arc::new(AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let c = c.clone();
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut last = 0.0f64;
                    while !stop.load(Ordering::Relaxed) {
                        let p = c.position_secs();
                        assert!(p >= last - 1e-9, "went backwards: {p} < {last}");
                        last = p;
                    }
                    last
                })
            })
            .collect();
        for i in 1..=2_000u64 {
            c.pb.frames.store(i * 48, Ordering::Relaxed); // +1 ms per step
            // Jitter the delay by a quantum either way, as the RT callback would.
            c.pb.out_delay_ns.store(20_000_000 + (i % 3) * 1_000_000, Ordering::Relaxed);
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            let last = r.join().expect("reader thread panicked");
            assert!(last <= c.write_position_secs(), "compensated exceeded write-side");
        }
    }
}

/// The attached (gapless) mode, driven end-to-end against the deterministic capture backend:
/// a real `PipeWireAudioSink` in a real element harness, writing into a real `AudioOut` whose
/// only difference from the shipping one is that a test steps its clock instead of a DAC.
///
/// The centrepiece is [`handoff_at`]: a known byte sequence is split at an arbitrary point,
/// pushed through two separate sinks either side of an EOS-detach/attach handoff, and the
/// device's recording is compared with the unbroken sequence byte for byte.
// Test module: harness rigs, byte fixtures and assertions allocate freely — none of this is on
// a media path or inside `process()`.
#[allow(clippy::disallowed_methods)]
#[cfg(test)]
mod attach_tests {
    use super::*;
    use crate::out::testing::{open_capture, CaptureHandle};
    use crate::out::{AudioOutConfig, CanonicalFormat, SampleFormat};
    use profluens_core::bus::Bus;
    use profluens_core::harness::Harness;
    use profluens_core::id::{ElementId, FormatId};
    use profluens_core::memory::Pool;

    const RATE: u32 = 48_000;
    const CANON: CanonicalFormat =
        CanonicalFormat { sample: SampleFormat::F32, rate: RATE, channels: 2 };

    fn cfg() -> AudioOutConfig {
        AudioOutConfig { format: CANON, ring_secs: 0.5 }
    }

    /// Install the canonical format on the sink's pad, as link-time negotiation would.
    fn fix(h: &mut Harness, rate: i64) {
        h.fix_format(
            "sink",
            "audio/raw",
            &[
                ("rate", ValueDesc::Int(rate)),
                ("channels", ValueDesc::Int(2)),
                ("sample", ValueDesc::Id("f32")),
            ],
        );
    }

    fn sink_harness(out: &AudioOutHandle) -> Harness {
        let mut h = Harness::new(PipeWireAudioSink::with_output(out.clone()));
        fix(&mut h, RATE as i64);
        h
    }

    /// A deterministic, non-repeating byte sequence (a splitmix-style hash of the index). Any
    /// byte lost, duplicated or reordered at the seam shows up as a mismatch at a *specific
    /// index*, which is what makes a failure diagnosable rather than "it sounds wrong".
    fn seq_bytes(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| {
                let x = (i as u64)
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (x >> 33) as u8
            })
            .collect()
    }

    /// Push `bytes` through the harness the way a pipeline does: in buffer-sized chunks, with
    /// the device consuming in between. (A bulk push larger than the ring would simply block —
    /// that is the backpressure doing its job.)
    fn feed(h: &mut Harness, cap: &CaptureHandle, bytes: &[u8]) {
        const CHUNK: usize = 4096;
        let mut off = 0;
        while off < bytes.len() {
            let end = (off + CHUNK).min(bytes.len());
            let buf = h.alloc(&bytes[off..end]);
            h.push("sink", buf).expect("push");
            off = end;
            cap.drain(512);
        }
    }

    /// A `Ctx` with nothing installed — enough for `stop()` and for events that read no
    /// negotiated state. The harness owns its own `Ctx` and cannot lend one out, and `stop()`
    /// is the one lifecycle call it does not expose.
    fn bare_ctx() -> (Ctx, Bus) {
        let (tx, bus) = Bus::channel();
        (Ctx::new(Pool::bounded(1024, 4), tx, ElementId(0), FormatId(0), 1), bus)
    }

    // --- the continuity proof --------------------------------------------------------------

    /// One handoff: sink A writes `seq[..split]`, EOS-detaches (leaving its tail in flight),
    /// sink B attaches and writes `seq[split..]`. The device's recording must be `seq`.
    fn handoff_at(split: usize, seq: &[u8]) {
        let (out, cap) = open_capture(cfg());
        let h = out.handle();

        let mut a = sink_harness(&h);
        feed(&mut a, &cap, &seq[..split]);
        // EOS detaches *without* draining: whatever is still in the ring keeps playing.
        a.push_event(Event::Eos).expect("eos");
        assert!(!h.is_attached(), "split {split}: EOS must release the output");
        drop(a);

        let mut b = sink_harness(&h);
        feed(&mut b, &cap, &seq[split..]);
        b.push_event(Event::Eos).expect("eos");
        drop(b);
        cap.drain(512);

        let got = cap.captured();
        assert_eq!(got.len(), seq.len(), "split {split}: byte count");
        if got != seq {
            let at = got.iter().zip(seq).position(|(a, b)| a != b).unwrap_or(0);
            panic!("split {split}: first divergence at byte {at}");
        }
    }

    #[test]
    fn a_track_handoff_is_byte_exact_at_dozens_of_split_points() {
        // 64 KiB of audio, split at 48 offsets that are deliberately *not* aligned to the
        // quantum, the buffer size, or even the 8-byte frame — the ring is a byte stream and
        // the seam must be exact at byte granularity.
        const TOTAL: usize = 64 * 1024;
        let seq = seq_bytes(TOTAL);
        for k in 0..48usize {
            handoff_at((k * 1367 + 3) % TOTAL, &seq);
        }
        // And the degenerate ends: a track that wrote nothing, and one that wrote everything.
        handoff_at(0, &seq);
        handoff_at(1, &seq);
        handoff_at(TOTAL - 1, &seq);
        handoff_at(TOTAL, &seq);
    }

    #[test]
    fn eos_detaches_without_draining_so_the_tail_keeps_playing() {
        // The property the whole design rests on: at EOS the ring still holds audio.
        let (out, cap) = open_capture(cfg());
        let h = out.handle();
        let mut a = sink_harness(&h);
        let buf = a.alloc(&seq_bytes(4096));
        a.push("sink", buf).expect("push");
        assert!(h.is_attached());
        assert_eq!(cap.available(), 4096, "nothing rendered yet");

        a.push_event(Event::Eos).expect("eos");
        assert!(!h.is_attached(), "detached");
        assert_eq!(cap.available(), 4096, "the tail is still in flight, not drained or dropped");
    }

    // --- the attach protocol ----------------------------------------------------------------

    #[test]
    fn a_negotiated_but_silent_sink_never_attaches() {
        // The pre-rolled next track: started, format-checked, paused — and it must not have
        // claimed the output, because the current track still owns it.
        let (out, _cap) = open_capture(cfg());
        let h = out.handle();
        let mut b = sink_harness(&h);
        b.start().expect("start");
        b.crank().expect("crank"); // runs process() with no input: configures, writes nothing
        assert!(!h.is_attached(), "start()/preroll must not attach");
        // An empty buffer is not a write either.
        let empty = b.alloc(&[]);
        b.push("sink", empty).expect("push");
        assert!(!h.is_attached(), "an empty buffer is not a byte written");
    }

    #[test]
    fn a_prerolled_sinks_flush_cannot_damage_the_previous_tracks_tail() {
        // A pre-resume seek/flush on the *next* track's pipeline must be completely inert:
        // the ring belongs to the track still playing.
        let (out, cap) = open_capture(cfg());
        let h = out.handle();
        let mut a = sink_harness(&h);
        let buf = a.alloc(&seq_bytes(4096));
        a.push("sink", buf).expect("push");
        assert_eq!(cap.available(), 4096);

        let mut b = sink_harness(&h);
        b.crank().expect("crank"); // configured, not attached
        b.push_event(Event::FlushStart).expect("flush");
        b.push_event(Event::FlushStop).expect("flush stop");
        assert_eq!(cap.available(), 4096, "the previous track's audio is untouched");
        assert!(h.is_attached(), "and A still owns the output");
    }

    #[test]
    fn a_second_attach_while_one_is_live_is_a_loud_error() {
        let (out, cap) = open_capture(cfg());
        let h = out.handle();
        let mut a = sink_harness(&h);
        let buf = a.alloc(&seq_bytes(512));
        a.push("sink", buf).expect("push");

        let mut b = sink_harness(&h);
        let buf = b.alloc(&seq_bytes(512));
        let r = b.push("sink", buf);
        assert!(r.is_err(), "two sinks on one output must not be silently interleaved");
        // A's stream is unharmed.
        cap.drain(512);
        assert_eq!(cap.captured(), seq_bytes(512));
    }

    #[test]
    fn a_format_mismatch_is_reported_not_played() {
        let (out, _cap) = open_capture(cfg());
        let h = out.handle();
        let mut s = Harness::new(PipeWireAudioSink::with_output(h.clone()));
        fix(&mut s, 44_100); // the chain failed to converge on the canonical rate
        let buf = s.alloc(&seq_bytes(64));
        let r = s.push("sink", buf);
        assert!(r.is_err(), "a mismatched format must not reach the device as garbage");
        assert!(!h.is_attached());
    }

    // --- teardown and hostile sequences --------------------------------------------------------

    #[test]
    fn stop_detaches_and_is_idempotent() {
        let (out, _cap) = open_capture(cfg());
        let h = out.handle();
        let (mut ctx, _bus) = bare_ctx();

        let mut sink = PipeWireAudioSink::with_output(h.clone());
        // Reach in and attach the way `process()` would on its first byte.
        sink.producer = Some(h.attach().expect("attach"));
        sink.started = true;
        assert!(h.is_attached());

        sink.stop(&mut ctx);
        assert!(!h.is_attached(), "stop detaches");
        sink.stop(&mut ctx);
        assert!(!h.is_attached(), "a second stop is a no-op, not a double detach");
        // The output is intact and re-attachable.
        let _p = h.attach().expect("the output outlived its sink");
    }

    #[test]
    fn stop_and_flush_on_a_sink_that_never_attached_are_no_ops() {
        let (out, _cap) = open_capture(cfg());
        let h = out.handle();
        let (mut ctx, _bus) = bare_ctx();
        let mut sink = PipeWireAudioSink::with_output(h.clone());
        sink.start(&mut ctx).expect("start");
        sink.event(&mut ctx, &Event::FlushStart).expect("flush");
        sink.event(&mut ctx, &Event::Eos).expect("eos");
        sink.stop(&mut ctx);
        sink.stop(&mut ctx);
        drop(sink);
        assert!(!h.is_attached());
        let _p = h.attach().expect("nothing was damaged");
    }

    #[test]
    fn a_sink_dropped_without_stopping_still_returns_the_producer() {
        // The pipeline-unwinds-on-error path. Dropping the producer would close the shared
        // ring and kill the output for every future track, so `Drop` must detach.
        let (out, _cap) = open_capture(cfg());
        let h = out.handle();
        let mut a = sink_harness(&h);
        let buf = a.alloc(&seq_bytes(256));
        a.push("sink", buf).expect("push");
        assert!(h.is_attached());

        drop(a); // no stop(), no EOS
        assert!(!h.is_attached(), "Drop hands the producer back");
        let _p = h.attach().expect("the output survived an unclean teardown");
    }

    #[test]
    fn an_output_dropped_underneath_an_attached_sink_errors_on_the_next_write() {
        let (out, _cap) = open_capture(cfg());
        let h = out.handle();
        let mut a = sink_harness(&h);
        let buf = a.alloc(&seq_bytes(256));
        a.push("sink", buf).expect("push");

        drop(out); // the app tore its audio output down mid-track
        let buf = a.alloc(&seq_bytes(256));
        let r = a.push("sink", buf);
        assert!(r.is_err(), "a closed output is an error, not silence");
        drop(a); // and unwinding from here must not misbehave either
    }

    #[test]
    fn a_shared_output_survives_many_sequential_tracks() {
        // The soak: forty handoffs on one output. Track lengths are deliberately *not*
        // multiples of the 8-byte frame — real tracks do not end on a frame boundary, let
        // alone a device quantum — so each seam falls mid-frame and the next track's first
        // bytes complete the previous one's last frame. That is exactly the case a
        // per-producer buffer partition would get wrong.
        let stride = CANON.stride();
        let (out, cap) = open_capture(cfg());
        let h = out.handle();
        let mut expect = Vec::new();
        for track in 0..40usize {
            let seq = seq_bytes(2048 + track * 7);
            expect.extend_from_slice(&seq);
            let mut s = sink_harness(&h);
            feed(&mut s, &cap, &seq);
            s.push_event(Event::Eos).expect("eos");
            drop(s);
        }
        cap.drain(512);
        let got = cap.captured();
        // A device renders whole frames, so a trailing sub-frame remainder is still in the
        // ring — everything before it must be the unbroken concatenation.
        let whole = expect.len() / stride * stride;
        assert_eq!(got.len(), whole, "every whole frame reached the device");
        assert!(expect.len() - whole < stride, "only a sub-frame remainder is left");
        assert_eq!(got, expect[..whole], "forty tracks, one unbroken byte stream");
        assert!(!h.is_attached());
    }
}
