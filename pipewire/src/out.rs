//! `AudioOut` — the **application-owned** audio output: one device, one ring, one set of
//! playback counters, outliving any individual pipeline (spec: gapless.md, Phase 2).
//!
//! The pipewire sink used to own all of that privately, which made a track boundary a device
//! teardown: the stream closed, the ring died, the clock froze, and the next track paid a
//! fresh connect. Here the device is hoisted out of the element. A sink becomes a thin
//! *producer* that [attaches](AudioOutHandle::attach) to an `AudioOut`, pushes PCM into its
//! ring, and detaches at EOS **without draining** — the in-flight bytes keep playing while the
//! next track's sink attaches behind them. Handoff cost is a scheduler pass against a ~0.5 s
//! ring, so the seam is sample-continuous.
//!
//! Three pieces live here:
//!
//! * [`AudioOut`] / [`AudioOutHandle`] — the app-facing object and its cloneable control
//!   handle (volume, mute, idle parking, position, output delay).
//! * [`Renderer`] — the **device-agnostic** render step: pull from the ring, apply the volume
//!   ramp, advance the counters. Every backend calls exactly this from its callback, so the
//!   deterministic capture backend exercises the same code the real device does.
//! * [`DeviceBackend`] — the seam between that logic and an actual device. Two impls today
//!   ([`crate::pw_backend::PwBackend`] and [`testing::CaptureBackend`]); CoreAudio is the
//!   intended third.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use profluens_core::error::Error;

use crate::ring::{self, Consumer, Producer};

// --- format ----------------------------------------------------------------------------

/// An interleaved PCM sample encoding this crate can hand to a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    U8,
    S16,
    S24,
    S32,
    F32,
}

impl SampleFormat {
    /// Parse the `audio/raw` `sample` vocabulary name (`"s16"`, `"f32"`, …).
    pub fn from_name(name: &str) -> Option<SampleFormat> {
        Some(match name {
            "u8" => SampleFormat::U8,
            "s16" => SampleFormat::S16,
            "s24" => SampleFormat::S24,
            "s32" => SampleFormat::S32,
            "f32" => SampleFormat::F32,
            _ => return None,
        })
    }

    /// Bytes per single-channel sample.
    pub fn bytes(self) -> usize {
        match self {
            SampleFormat::U8 => 1,
            SampleFormat::S16 => 2,
            SampleFormat::S24 => 3,
            SampleFormat::S32 | SampleFormat::F32 => 4,
        }
    }

    /// The byte value that renders as silence: zero for signed/float PCM, mid-scale for
    /// unsigned U8.
    pub(crate) fn silence(self) -> u8 {
        if matches!(self, SampleFormat::U8) {
            0x80
        } else {
            0
        }
    }
}

/// The fixed format an [`AudioOut`] renders. It is chosen once, at
/// [`open`](AudioOut::open), and never changes — that is the whole point: every track's chain
/// is converged onto it upstream, so a track boundary never re-latches the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalFormat {
    pub sample: SampleFormat,
    pub rate: u32,
    pub channels: u32,
}

impl Default for CanonicalFormat {
    /// f32, 48 kHz, stereo — PipeWire's native float format and the canonical chain target.
    fn default() -> Self {
        Self { sample: SampleFormat::F32, rate: 48_000, channels: 2 }
    }
}

impl CanonicalFormat {
    /// Bytes per interchannel frame.
    pub fn stride(&self) -> usize {
        self.sample.bytes() * self.channels as usize
    }
}

/// How to open an [`AudioOut`].
#[derive(Clone, Copy, Debug)]
pub struct AudioOutConfig {
    pub format: CanonicalFormat,
    /// Seconds of PCM the hand-off ring holds. This is the jitter buffer *and* the budget a
    /// track handoff has to complete in — 0.5 s against a millisecond-scale handoff.
    pub ring_secs: f32,
}

impl Default for AudioOutConfig {
    fn default() -> Self {
        Self { format: CanonicalFormat::default(), ring_secs: 0.5 }
    }
}

// --- shared playback state -------------------------------------------------------------

/// Shared playback state between the element, the device's real-time callback and the app's
/// handles: the pause latch, the frame counters, the volume target and the position
/// machinery. Relaxed atomics — the RT callback only ever loads and adds, no lock, no
/// allocation.
pub(crate) struct Playback {
    pub(crate) paused: AtomicBool,
    /// Whole interchannel frames handed to the device — the **write-side** position. Audio is
    /// still in flight past this point; [`Playback::out_delay_ns`] is how far it leads what
    /// the speaker is actually producing.
    pub(crate) frames: AtomicU64,
    /// Frames rendered since the device started, *never* re-based by seeks or attaches
    /// (unlike `frames`, which tracks stream position): the monotonic timebase behind
    /// `AudioDeviceClock`. Pausing stops it — and with it the pipeline clock. Under a shared
    /// [`AudioOut`] it keeps running across a track boundary, which is exactly what a
    /// pipeline mastered on it needs.
    ///
    /// This counter stays **write-side**: it is deliberately *not* compensated for the
    /// output delay, because the pipeline's whole latency model is built on it and the
    /// contract "never re-based, never adjusted" is what video pacing relies on.
    pub(crate) clock_frames: AtomicU64,
    /// Sample rate, published once at open.
    pub(crate) rate: AtomicU32,
    /// Set on seek; the device backend flushes whatever it has already committed downstream
    /// (notably a rate converter's buffer) so the new position is audible promptly.
    pub(crate) flush_stream: AtomicBool,
    /// Measured output delay in nanoseconds: how far `frames` leads audible playback. Written
    /// only by the backend, read by anyone.
    pub(crate) out_delay_ns: AtomicU64,
    /// Post-seek floor for the compensated position, in nanoseconds: the stream time the last
    /// `FlushStart` re-based to. See [`position_ns`].
    pub(crate) pos_floor_ns: AtomicU64,
    /// Monotonic clamp for the compensated position, packed as `epoch << 44 | micros` (see
    /// [`pack_mono`]).
    pub(crate) pos_mono: AtomicU64,
    /// The value of `frames` at which the **current attach epoch**'s audio begins — the
    /// device-side frame index where the attached producer's first byte lands (see
    /// [`epoch_frames`]). Zero for a single-owner sink, whose device starts when its one and
    /// only track does, which is why that mode's position arithmetic is bit-for-bit what it
    /// always was; [`NO_EPOCH`] until a shared output's first attach.
    pub(crate) epoch_start_frames: AtomicU64,
    /// Master volume target, linear, as `f32::to_bits`. `1.0` is unity.
    pub(crate) volume_bits: AtomicU32,
    /// Master mute. Ramped like volume — the effective target is `0.0` while set.
    pub(crate) muted: AtomicBool,
    /// Idle parking: the device is asked to stop being scheduled entirely. Distinct from
    /// `paused` (which keeps the callback running and rendering silence).
    pub(crate) idle: AtomicBool,
    /// Bytes still in the ring as of the last rendered block — the playback *runway*.
    /// Published by [`Renderer::render`] (one relaxed store per block) because the ring's
    /// depth is otherwise reachable only from whichever side holds an end of it, and during a
    /// track handoff that is nobody. See [`AudioOutHandle::buffered`].
    pub(crate) ring_bytes: AtomicUsize,
}

impl Default for Playback {
    fn default() -> Self {
        Self {
            paused: AtomicBool::new(false),
            frames: AtomicU64::new(0),
            clock_frames: AtomicU64::new(0),
            rate: AtomicU32::new(0),
            flush_stream: AtomicBool::new(false),
            out_delay_ns: AtomicU64::new(0),
            pos_floor_ns: AtomicU64::new(0),
            pos_mono: AtomicU64::new(0),
            epoch_start_frames: AtomicU64::new(0),
            // Unity, not `0` — a default-constructed `Playback` must not be silent.
            volume_bits: AtomicU32::new(1.0f32.to_bits()),
            muted: AtomicBool::new(false),
            idle: AtomicBool::new(false),
            ring_bytes: AtomicUsize::new(0),
        }
    }
}

impl Playback {
    /// The gain the ramp is currently walking toward: the master volume, or zero while muted.
    fn target_gain(&self) -> f32 {
        if self.muted.load(Ordering::Relaxed) {
            0.0
        } else {
            f32::from_bits(self.volume_bits.load(Ordering::Relaxed))
        }
    }
}

/// [`Playback::epoch_start_frames`] before any producer has ever attached.
///
/// A shared [`AudioOut`] starts its device the moment the app opens it, long before a track
/// exists, and it renders silence until one does — so `frames` is already advancing with
/// nothing to attribute it to. The sentinel makes the epoch-relative subtraction saturate to
/// zero, so `position()` reports "nothing has played" instead of the output's idle uptime. A
/// single-owner sink's device starts *with* its track, so that mode leaves the epoch at 0 and
/// its arithmetic is unchanged.
pub(crate) const NO_EPOCH: u64 = u64::MAX;

/// Whole frames at `rate` → nanoseconds. Saturating; `rate == 0` (device not configured yet)
/// reads zero rather than dividing by zero.
pub(crate) fn frames_to_ns(frames: u64, rate: u32) -> u64 {
    if rate == 0 {
        return 0;
    }
    u64::try_from(frames as u128 * 1_000_000_000 / rate as u128).unwrap_or(u64::MAX)
}

/// Packing for [`Playback::pos_mono`]: the low 44 bits hold the clamped position in
/// **microseconds** (≈203 days of range), the high 20 an epoch counter.
///
/// One word, so a reader's compare-exchange observes an epoch change (a seek, or an attach
/// starting a new track) atomically and discards a candidate it computed from stale state —
/// without that, a backward seek could leave the clamp pinned at the old (higher) position
/// forever. The clamp is µs-granular, so the only imprecision it can introduce is under 1 µs,
/// and only when it actually bites.
pub(crate) const MONO_POS_BITS: u32 = 44;
pub(crate) const MONO_POS_MASK: u64 = (1 << MONO_POS_BITS) - 1;
pub(crate) const MONO_EPOCH_MASK: u64 = (1 << (64 - MONO_POS_BITS)) - 1;

pub(crate) fn pack_mono(epoch: u64, micros: u64) -> u64 {
    ((epoch & MONO_EPOCH_MASK) << MONO_POS_BITS) | micros.min(MONO_POS_MASK)
}

/// Open a new position epoch at `floor_ns`: re-base the floor and bump the epoch counter so
/// every in-flight reader of [`position_ns`] discards its candidate and retries.
///
/// Order matters and is the caller's contract: re-base `frames` (and, for an attach,
/// `epoch_start_frames`) **first**, publish the epoch **last**.
pub(crate) fn open_epoch(pb: &Playback, floor_ns: u64) {
    pb.pos_floor_ns.store(floor_ns, Ordering::Relaxed);
    let epoch = (pb.pos_mono.load(Ordering::Relaxed) >> MONO_POS_BITS).wrapping_add(1);
    pb.pos_mono.store(pack_mono(epoch, floor_ns / 1_000), Ordering::Release);
}

/// Nanoseconds of audio the listener has actually **heard**, within the current epoch.
///
/// Handing PCM to a device is not the same as playing it: the frames travel through the
/// stream's converter, the graph and the device before they reach a speaker, and on a
/// Bluetooth A2DP sink that is 150–350 ms. This is the write-side position minus the measured
/// output delay, under three policies:
///
/// * **Epoch-relative.** `frames` counts everything the device has taken since the output
///   opened, including previous tracks; `epoch_start_frames` is where the current attach's
///   audio begins, so the difference is *this* track's position. A single-owner sink leaves
///   the epoch start at zero and the arithmetic reduces to what it always was.
/// * **Post-seek floor.** Right after a flush the ring is empty and the write-side counter
///   has been re-based straight to the seek target, so a naive subtraction would report a
///   position *before* the target the user just asked for — and, because a relative seek is
///   computed from this value, each one would drift backwards by the delay. The position is
///   therefore floored at the last re-base watermark: it holds at the target until the write
///   side has advanced past `target + delay`, from which point the subtraction takes over on
///   its own. (Playback from the start, or a fresh attach, is the same rule with a watermark
///   of zero — which is also the "never negative" clamp.)
/// * **Monotonic within an epoch.** The delay estimate is re-measured every graph cycle and
///   can wobble by a quantum, which would show up as the position ticking backwards between
///   two reads. The value is clamped to the highest already reported since the last epoch
///   change. A seek *or an attach* resets the clamp, so those still move the position — only
///   jitter is suppressed. A real latency increase (a route change to Bluetooth) is therefore
///   absorbed as a pause in the position rather than a jump backwards, and the raw output
///   delay still reports it truthfully.
pub(crate) fn position_ns(pb: &Playback) -> u64 {
    let rate = pb.rate.load(Ordering::Relaxed);
    if rate == 0 {
        return 0;
    }
    loop {
        // Sample the clamp word first. The compare-exchange below re-checks it, so an epoch
        // change landing while we compute is caught and we simply retry against its state.
        let before = pb.pos_mono.load(Ordering::Acquire);
        let epoch_start = pb.epoch_start_frames.load(Ordering::Relaxed);
        let written = pb.frames.load(Ordering::Relaxed).saturating_sub(epoch_start);
        let write_ns = frames_to_ns(written, rate);
        let delay_ns = pb.out_delay_ns.load(Ordering::Relaxed);
        let floor_ns = pb.pos_floor_ns.load(Ordering::Relaxed);
        let cand_ns = write_ns.saturating_sub(delay_ns).max(floor_ns);
        let clamp_us = before & MONO_POS_MASK;
        let cand_us = cand_ns / 1_000;
        let want = pack_mono(before >> MONO_POS_BITS, cand_us.max(clamp_us));
        if pb
            .pos_mono
            .compare_exchange(before, want, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            return if cand_us >= clamp_us { cand_ns } else { clamp_us * 1_000 };
        }
    }
}

// --- the volume ramp ---------------------------------------------------------------------

/// Ramp length for a volume/mute change. A step change in gain is a click; ~5 ms of linear
/// interpolation is the standard fix (short enough to feel instant, long enough that the
/// resulting spectral splatter is inaudible).
const RAMP_SECS: f32 = 0.005;

/// Walk `buf`'s samples from gain `g0` to `g1`, linearly, one step per interchannel frame.
///
/// The gain is held constant across a frame's channels (a per-channel ramp would smear the
/// stereo image) and advances by `dg = (g1 - g0) / frames` per frame, so the block *ends*
/// approaching `g1` and the next block starts exactly at `g1` — continuous across blocks,
/// with a bounded per-frame step and no accumulated error.
fn ramp(buf: &mut [u8], bps: usize, channels: usize, g0: f32, dg: f32, mut scale: impl FnMut(&mut [u8], f64)) {
    let mut g = g0;
    let mut c = 0usize;
    for s in buf.chunks_exact_mut(bps) {
        scale(s, g as f64);
        c += 1;
        if c == channels {
            c = 0;
            g += dg;
        }
    }
}

/// Apply a linear gain ramp from `g0` to `g1` across `buf`, in place, for `fmt`.
///
/// Integer formats round-to-nearest and saturate at their full-scale bounds, so a gain above
/// unity clips rather than wrapping (a wrap is a hard, extremely audible fault). The
/// arithmetic goes through `f64` for the integer paths: `f32`'s 24-bit mantissa cannot hold an
/// `s32` sample, and rounding a 24-bit sample through it would lose the bottom bit.
///
/// Allocation-free and branch-light; called from the RT callback only when the gain is not a
/// no-op (see [`Renderer::apply_ramp`]).
fn apply_gain(fmt: SampleFormat, buf: &mut [u8], channels: usize, g0: f32, g1: f32) {
    let frames = buf.len() / (fmt.bytes() * channels.max(1));
    if frames == 0 || channels == 0 {
        return;
    }
    let dg = (g1 - g0) / frames as f32;
    match fmt {
        SampleFormat::F32 => ramp(buf, 4, channels, g0, dg, |s, g| {
            let v = f32::from_le_bytes([s[0], s[1], s[2], s[3]]) * g as f32;
            s.copy_from_slice(&v.to_le_bytes());
        }),
        SampleFormat::S16 => ramp(buf, 2, channels, g0, dg, |s, g| {
            let v = i16::from_le_bytes([s[0], s[1]]) as f64 * g;
            let v = v.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
            s.copy_from_slice(&v.to_le_bytes());
        }),
        SampleFormat::S32 => ramp(buf, 4, channels, g0, dg, |s, g| {
            let v = i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64 * g;
            let v = v.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            s.copy_from_slice(&v.to_le_bytes());
        }),
        SampleFormat::S24 => ramp(buf, 3, channels, g0, dg, |s, g| {
            // 24-bit little-endian, sign-extended into an i32 by shifting the top byte up
            // and arithmetic-shifting back down.
            let raw = s[0] as i32 | (s[1] as i32) << 8 | (s[2] as i32) << 16;
            let v = ((raw << 8) >> 8) as f64 * g;
            let v = v.round().clamp(-8_388_608.0, 8_388_607.0) as i32;
            s[0] = v as u8;
            s[1] = (v >> 8) as u8;
            s[2] = (v >> 16) as u8;
        }),
        SampleFormat::U8 => ramp(buf, 1, channels, g0, dg, |s, g| {
            // Unsigned 8-bit PCM is offset binary: 0x80 is silence, so scale about it.
            let v = ((s[0] as f64 - 128.0) * g + 128.0).round().clamp(0.0, 255.0);
            s[0] = v as u8;
        }),
    }
}

// --- the device-agnostic render step ------------------------------------------------------

/// The render step every [`DeviceBackend`] runs: pull PCM from the ring into the device's
/// buffer, apply the volume ramp, advance the counters.
///
/// This is deliberately *all* of the logic — a backend contributes buffers and a clock, and
/// nothing else. That is what makes [`testing::CaptureBackend`] a real proof rather than a
/// parallel implementation: the bytes a capture test compares came through exactly the code
/// the PipeWire RT callback runs.
///
/// **RT-safe**: no locks, no allocation, no syscalls. The ring's consumer side is wait-free
/// and the ramp is pure arithmetic.
pub(crate) struct Renderer {
    consumer: Consumer,
    pb: Arc<Playback>,
    fmt: CanonicalFormat,
    stride: usize,
    silence: u8,
    /// The gain actually applied to the last sample rendered — the ramp's state. Owned by
    /// whichever thread runs [`render`](Self::render), never shared.
    gain: f32,
    /// Maximum gain change per interchannel frame: `1 / (RAMP_SECS * rate)`.
    ramp_step: f32,
}

impl Renderer {
    fn new(consumer: Consumer, pb: Arc<Playback>, fmt: CanonicalFormat) -> Self {
        let rate = fmt.rate.max(1) as f32;
        // Start at the current target rather than unity, so an output opened muted (or at a
        // stored volume) does not ramp up from nothing on its very first block.
        let gain = pb.target_gain();
        Self {
            consumer,
            pb,
            fmt,
            stride: fmt.stride(),
            silence: fmt.sample.silence(),
            gain,
            ramp_step: 1.0 / (RAMP_SECS * rate),
        }
    }

    /// The format this renderer was built for.
    pub(crate) fn format(&self) -> CanonicalFormat {
        self.fmt
    }

    /// The shared counters, for a backend that publishes into them (a measured output delay).
    pub(crate) fn playback(&self) -> &Arc<Playback> {
        &self.pb
    }

    /// Bytes currently readable from the ring — how much real audio a render can still pull
    /// before it starts padding silence.
    pub(crate) fn available(&self) -> usize {
        self.consumer.available()
    }

    /// Fill `out` with the next device block and return how many bytes were written (always a
    /// whole number of frames; a trailing partial frame is left untouched).
    ///
    /// * **Paused**: renders silence and holds the ring — no audio is lost, no counter moves,
    ///   and the upstream producer backpressures to a stop.
    /// * **Underrun**: the ring pads with silence rather than replaying stale bytes, and the
    ///   counters still advance, so the clock keeps running when nothing is attached.
    pub(crate) fn render(&mut self, out: &mut [u8]) -> usize {
        let n = (out.len() / self.stride) * self.stride; // whole frames only
        let dst = &mut out[..n];
        if self.pb.paused.load(Ordering::Relaxed) {
            dst.fill(self.silence);
            self.pb.ring_bytes.store(self.consumer.available(), Ordering::Relaxed);
            return n;
        }
        self.consumer.pull(dst);
        self.pb.ring_bytes.store(self.consumer.available(), Ordering::Relaxed);
        let frames = n / self.stride;
        self.pb.frames.fetch_add(frames as u64, Ordering::Relaxed);
        self.pb.clock_frames.fetch_add(frames as u64, Ordering::Relaxed);
        self.apply_ramp(dst, frames);
        n
    }

    /// Advance the volume ramp across this block and apply it.
    ///
    /// The gain moves toward the target by at most `ramp_step` per frame, so a block of
    /// `frames` frames can close at most `ramp_step * frames` of the gap; when the gap is
    /// smaller than that the clamp saturates and the block ends *exactly* on the target.
    ///
    /// Unity-to-unity is the overwhelmingly common case (nobody has touched the volume) and
    /// costs one float compare — the sample loop is never entered.
    fn apply_ramp(&mut self, dst: &mut [u8], frames: usize) {
        let target = self.pb.target_gain();
        if frames == 0 || (self.gain == 1.0 && target == 1.0) {
            return;
        }
        let max = self.ramp_step * frames as f32;
        let end = self.gain + (target - self.gain).clamp(-max, max);
        apply_gain(self.fmt.sample, dst, self.fmt.channels as usize, self.gain, end);
        self.gain = end;
    }
}

// --- the device seam ----------------------------------------------------------------------

/// The seam between [`AudioOut`]'s ring/counter/gain logic and an actual device.
///
/// A backend owns *only* the device: it obtains buffers, calls [`Renderer::render`] to fill
/// them, and hands them on. It must not reimplement pulling, gain, or counting — those live in
/// the renderer precisely so every backend behaves identically and the deterministic capture
/// backend can stand in for the real one in tests.
///
/// Implementations today: [`crate::pw_backend::PwBackend`] (PipeWire, the production path) and
/// [`testing::CaptureBackend`] (no device, a virtual clock driven by the test). **CoreAudio is
/// the intended third impl** — an `AudioUnit` render callback maps onto
/// [`start`](Self::start)'s "keep the renderer, call it from the device callback" contract
/// one-to-one, `set_idle` maps onto `AudioOutputUnitStop`/`Start` (and, above it, the
/// `AVAudioSession` activation the platform requires), and `stop` onto disposing the unit.
///
/// Contract:
///
/// * [`start`](Self::start) is called exactly once, from [`AudioOut::open`], before any handle
///   is issued. It takes ownership of the [`Renderer`] (and with it the ring's consumer end).
/// * The backend should publish a measured output delay into
///   `renderer.playback().out_delay_ns` whenever it can; leaving it zero simply means the
///   reported position is the write-side one.
/// * [`set_idle`](Self::set_idle) parks/unparks the device. **Freeze semantics**: buffered
///   audio is preserved and playback resumes exactly where it stopped (see
///   [`AudioOutHandle::set_idle`]).
/// * [`stop`](Self::stop) is called exactly once, from `AudioOut`'s `Drop`, and must be
///   idempotent and must not deadlock if the device thread has already exited. Dropping the
///   renderer there closes the ring, which is how a still-attached producer learns its output
///   is gone.
pub(crate) trait DeviceBackend: Send + Sync {
    fn start(&self, renderer: Renderer) -> Result<(), Error>;
    fn set_idle(&self, idle: bool);
    fn stop(&self);
}

// --- AudioOut ------------------------------------------------------------------------------

/// The producer slot. Exactly one sink may hold the ring's producer end at a time; `attached`
/// distinguishes "checked out by a sink" from "the output is being torn down".
struct Slot {
    producer: Option<Producer>,
    attached: bool,
}

/// Everything an [`AudioOutHandle`] can reach — kept behind one `Arc` so a handle (and an
/// attached sink) stays sound after the [`AudioOut`] itself is dropped.
struct Shared {
    pb: Arc<Playback>,
    fmt: CanonicalFormat,
    stride: usize,
    slot: Mutex<Slot>,
    backend: Arc<dyn DeviceBackend>,
    /// The owning [`AudioOut`] has been dropped: no new attach can succeed, and an attached
    /// producer will see its ring closed on the next write.
    closed: AtomicBool,
}

/// The application-owned audio output: one device, one ring, one clock, for the life of the
/// app. Create it once, hand [`handle`](Self::handle)s to whatever needs them, and let sink
/// elements come and go against it.
///
/// Dropping it stops the device and closes the ring; a sink still attached at that moment
/// fails its next write with a clean error rather than misbehaving.
pub struct AudioOut {
    shared: Arc<Shared>,
}

impl AudioOut {
    /// Open the PipeWire output for `cfg`.
    pub fn open(cfg: AudioOutConfig) -> Result<Self, Error> {
        Self::open_pw(cfg, Arc::new(Playback::default()))
    }

    /// The PipeWire output over caller-supplied counters — how the single-owner sink opens a
    /// private device against the `Arc<Playback>` it handed out an `AudioControl` for long
    /// before any device existed.
    pub(crate) fn open_pw(cfg: AudioOutConfig, pb: Arc<Playback>) -> Result<Self, Error> {
        Self::open_with(cfg, pb, Arc::new(crate::pw_backend::PwBackend::new()))
    }

    /// The full constructor: caller-supplied counters and backend.
    ///
    /// The counters are a parameter because the single-owner sink hands out an `AudioControl`
    /// (and a device clock) *before* the format is known and the device exists — it therefore
    /// owns the `Arc<Playback>` from construction and lends it to the output it later opens.
    pub(crate) fn open_with(
        cfg: AudioOutConfig,
        pb: Arc<Playback>,
        backend: Arc<dyn DeviceBackend>,
    ) -> Result<Self, Error> {
        let fmt = cfg.format;
        if fmt.rate == 0 || fmt.channels == 0 {
            return Err(Error::Resource(format!(
                "AudioOut: degenerate format ({} Hz, {} ch)",
                fmt.rate, fmt.channels
            )));
        }
        let stride = fmt.stride();
        // Ring capacity: `ring_secs` of audio, floored so a tiny rate still buffers sensibly.
        let secs = if cfg.ring_secs.is_finite() && cfg.ring_secs > 0.0 { cfg.ring_secs } else { 0.5 };
        let frames = ((fmt.rate as f64 * secs as f64) as usize).max(4096);
        let (producer, consumer) = ring::spsc(stride * frames);

        pb.rate.store(fmt.rate, Ordering::Relaxed);
        // No epoch until something attaches: the device is about to start rendering silence,
        // and none of it belongs to a track. (The single-owner sink attaches inside the same
        // `configure` call that opened this, so it never observes the sentinel.)
        pb.epoch_start_frames.store(NO_EPOCH, Ordering::Relaxed);
        backend.start(Renderer::new(consumer, Arc::clone(&pb), fmt))?;

        Ok(Self {
            shared: Arc::new(Shared {
                pb,
                fmt,
                stride,
                slot: Mutex::new(Slot { producer: Some(producer), attached: false }),
                backend,
                closed: AtomicBool::new(false),
            }),
        })
    }

    /// A cloneable, `Send + Sync` handle: volume, mute, idle, position, and the attach point
    /// a sink element needs.
    pub fn handle(&self) -> AudioOutHandle {
        AudioOutHandle { shared: Arc::clone(&self.shared) }
    }

    /// The fixed format this output renders.
    pub fn format(&self) -> CanonicalFormat {
        self.shared.fmt
    }
}

impl Drop for AudioOut {
    fn drop(&mut self) {
        // Order: mark closed so a racing `attach` fails cleanly, release the parked producer
        // (closing the ring even when no sink is attached), then stop the device — which drops
        // the renderer, and with it the consumer end, waking any producer parked on a full
        // ring so it can observe the closure instead of timing out forever.
        self.shared.closed.store(true, Ordering::Release);
        if let Ok(mut slot) = self.shared.slot.lock() {
            slot.producer = None;
        }
        self.shared.backend.stop();
    }
}

/// A cloneable control handle on an [`AudioOut`]. Everything here is safe to call from any
/// thread at any time, including after the `AudioOut` has been dropped (the calls become
/// inert; [`attach`](Self::attach) reports the closure).
#[derive(Clone)]
pub struct AudioOutHandle {
    shared: Arc<Shared>,
}

impl AudioOutHandle {
    /// The fixed format the output renders. A sink attaching to it must have negotiated
    /// exactly this.
    pub fn format(&self) -> CanonicalFormat {
        self.shared.fmt
    }

    /// Set the master volume: linear gain, clamped to `[0.0, 4.0]` (+12 dB of headroom for a
    /// quiet source; integer formats saturate rather than wrap, so a boost clips at worst).
    /// Applied in the render callback over a ~5 ms linear ramp, so a change never clicks.
    /// NaN is ignored.
    pub fn set_volume(&self, v: f32) {
        if v.is_nan() {
            return;
        }
        self.shared.pb.volume_bits.store(v.clamp(0.0, 4.0).to_bits(), Ordering::Relaxed);
    }

    /// The master volume target most recently set (not the instantaneous ramp value).
    pub fn volume(&self) -> f32 {
        f32::from_bits(self.shared.pb.volume_bits.load(Ordering::Relaxed))
    }

    /// Mute/unmute. Ramped exactly like [`set_volume`](Self::set_volume) — the effective
    /// target is zero while muted, and unmuting ramps back to the stored volume.
    pub fn set_muted(&self, m: bool) {
        self.shared.pb.muted.store(m, Ordering::Relaxed);
    }

    pub fn is_muted(&self) -> bool {
        self.shared.pb.muted.load(Ordering::Relaxed)
    }

    /// The measured delay between handing a frame to the device and hearing it: the stream's
    /// converter plus the graph plus the device. Zero until the backend publishes a first
    /// measurement.
    pub fn output_delay(&self) -> Duration {
        Duration::from_nanos(self.shared.pb.out_delay_ns.load(Ordering::Relaxed))
    }

    /// Audible playback position **within the current attach epoch** — i.e. into the track
    /// whose sink is attached now. Delay-compensated, floored at a seek target, and monotonic
    /// within the epoch; see [`position_ns`] for the exact policies.
    pub fn position(&self) -> Duration {
        Duration::from_nanos(position_ns(&self.shared.pb))
    }

    /// Park (or unpark) the device: the app is not playing anything and wants the output to
    /// stop being scheduled — a real power saving, and the hook an iOS `AVAudioSession`
    /// deactivation belongs on.
    ///
    /// **Policy: freeze, not drain.** Anything left in the ring stays there and plays when the
    /// output is unparked; the frame counters (and therefore the device clock and the
    /// position) stop advancing. This is deliberate: idle parking happens when playback is
    /// stopped, so there is nothing to drain, and freezing means an accidental park during
    /// playback loses no audio — it stutters, which is recoverable and obvious, rather than
    /// silently discarding a buffer.
    pub fn set_idle(&self, idle: bool) {
        self.shared.pb.idle.store(idle, Ordering::Relaxed);
        self.shared.backend.set_idle(idle);
    }

    pub fn is_idle(&self) -> bool {
        self.shared.pb.idle.load(Ordering::Relaxed)
    }

    /// How much audio is still queued ahead of the device — the playback **runway**, as of the
    /// last block it rendered.
    ///
    /// This is the number an engine schedules against: it is how long a track handoff (or a
    /// stalled decoder) may take before the listener hears a gap. It stays readable *between*
    /// tracks, when neither the outgoing nor the incoming sink holds an end of the ring, which
    /// is precisely the moment it matters — and it is why the live gapless probe can assert
    /// "the runway never reached zero across the boundary" instead of asking someone to listen.
    pub fn buffered(&self) -> Duration {
        let rate = self.shared.pb.rate.load(Ordering::Relaxed);
        let frames = self.shared.pb.ring_bytes.load(Ordering::Relaxed) / self.shared.stride.max(1);
        Duration::from_nanos(frames_to_ns(frames as u64, rate))
    }

    /// Whether a sink currently holds the producer end.
    pub fn is_attached(&self) -> bool {
        self.shared.slot.lock().map(|s| s.attached).unwrap_or(false)
    }

    /// The shared counters — how the single-owner sink keeps handing out its `AudioControl`
    /// and device clock.
    pub(crate) fn playback(&self) -> &Arc<Playback> {
        &self.shared.pb
    }

    /// Take the ring's producer end and open a new position epoch on it.
    ///
    /// **Exclusive**: a second attach while one is live is an error, not a queue. The engine
    /// sequences track handoffs (track N detaches at EOS, then N+1 attaches); a violation of
    /// that sequence is a wiring bug, and absorbing it silently would mean two sinks
    /// interleaving PCM into one ring — garbage audio that is very hard to trace back.
    ///
    /// The epoch starts at [`epoch_frames`]: the device-side frame index where the bytes this
    /// producer is about to push will actually be heard, i.e. past whatever the previous track
    /// still has in flight.
    pub(crate) fn attach(&self) -> Result<Producer, Error> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(Error::Resource("AudioOut: attach after the output was closed".into()));
        }
        let mut slot = self.shared.slot.lock().unwrap_or_else(|e| e.into_inner());
        if slot.attached {
            return Err(Error::Resource(
                "AudioOut: a producer is already attached (attach is exclusive)".into(),
            ));
        }
        let Some(producer) = slot.producer.take() else {
            return Err(Error::Resource("AudioOut: attach after the output was closed".into()));
        };
        slot.attached = true;
        let pb = &self.shared.pb;
        pb.epoch_start_frames
            .store(epoch_frames(pb, &producer, self.shared.stride), Ordering::Relaxed);
        open_epoch(pb, 0);
        Ok(producer)
    }

    /// Hand the producer end back. Taking it **by value** is the design: a sink holds it in an
    /// `Option` and detaches with `take()`, so a double detach cannot be expressed, and a sink
    /// dropped without stopping still returns it (see the element's `Drop`).
    pub(crate) fn detach(&self, producer: Producer) {
        let mut slot = self.shared.slot.lock().unwrap_or_else(|e| e.into_inner());
        slot.producer = Some(producer);
        slot.attached = false;
    }
}

/// The value of `Playback::frames` at which audio pushed from now on becomes audible: what the
/// device has already taken, plus the ring backlog still queued ahead of it.
///
/// Read as an exact pair. `frames` is sampled either side of the backlog and the reading is
/// retried if the render callback moved it in between — otherwise the sum would double-count
/// (or lose) whatever the device consumed between the two loads, which at a 1024-frame quantum
/// is a 21 ms error in the reported track position. The retry is bounded because a callback
/// fires every quantum, not every nanosecond; the loop gives up after a few tries and accepts
/// a reading rather than spinning, which cannot happen in practice but must not be able to
/// hang if it did.
fn epoch_frames(pb: &Playback, producer: &Producer, stride: usize) -> u64 {
    let mut last = 0;
    for _ in 0..8 {
        let before = pb.frames.load(Ordering::Relaxed);
        let backlog = (producer.len() / stride.max(1)) as u64;
        let after = pb.frames.load(Ordering::Relaxed);
        last = before.saturating_add(backlog);
        if before == after {
            return last;
        }
    }
    last
}

// --- the deterministic capture backend ------------------------------------------------------

/// A [`DeviceBackend`] with no device: a virtual clock the test steps by hand, recording every
/// sample it consumes.
///
/// This is what makes gaplessness *provable*. A real device is wall-clock paced and its output
/// is unobservable; the capture backend renders exactly when a test says so, in exactly the
/// block sizes it says, and keeps the bytes — so "producer A detached mid-quantum and producer
/// B continued" becomes a bit-for-bit array comparison instead of a listening session.
///
/// It runs the *same* [`Renderer`] the PipeWire backend runs, so what it proves is a property
/// of the shipping code path, not of a test double.
// Test-support module: the recording buffer and its snapshots are ordinary heap `Vec`s, which
// is correct here — nothing in this module runs on a media path or inside `process()`.
#[allow(clippy::disallowed_methods)]
pub mod testing {
    use super::{AudioOut, AudioOutConfig, DeviceBackend, Playback, Renderer};
    use profluens_core::error::Error;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Shared between the backend (which is handed the renderer) and the handle (which drives
    /// it). A plain `Mutex` — nothing here is real-time.
    struct CaptureInner {
        renderer: Mutex<Option<Renderer>>,
        captured: Mutex<Vec<u8>>,
        scratch: Mutex<Vec<u8>>,
    }

    /// The [`DeviceBackend`] half.
    pub(crate) struct CaptureBackend {
        inner: Arc<CaptureInner>,
    }

    impl DeviceBackend for CaptureBackend {
        fn start(&self, renderer: Renderer) -> Result<(), Error> {
            *self.inner.renderer.lock().unwrap_or_else(|e| e.into_inner()) = Some(renderer);
            Ok(())
        }
        fn set_idle(&self, _idle: bool) {
            // Nothing to park: `step` reads the shared `idle` latch and renders nothing while
            // it is set, which is the freeze the real backend achieves by deactivating.
        }
        fn stop(&self) {
            // Dropping the renderer drops the ring's consumer, closing the ring — exactly what
            // a real device thread exiting does.
            *self.inner.renderer.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }

    /// The test-facing half: step the virtual device, inspect what it consumed.
    #[derive(Clone)]
    pub struct CaptureHandle {
        inner: Arc<CaptureInner>,
    }

    /// Open an [`AudioOut`] backed by the capture device. Cannot fail — there is no device.
    pub fn open_capture(cfg: AudioOutConfig) -> (AudioOut, CaptureHandle) {
        let inner = Arc::new(CaptureInner {
            renderer: Mutex::new(None),
            captured: Mutex::new(Vec::new()),
            scratch: Mutex::new(Vec::new()),
        });
        let backend = Arc::new(CaptureBackend { inner: Arc::clone(&inner) });
        let out = AudioOut::open_with(cfg, Arc::new(Playback::default()), backend)
            .expect("the capture backend never fails to start");
        (out, CaptureHandle { inner })
    }

    impl CaptureHandle {
        /// Render exactly `frames` frames, as a device with that quantum would: an underrun
        /// becomes silence and still advances the counters. Returns the bytes appended to the
        /// recording (zero while parked idle, or after the output was dropped).
        pub fn step(&self, frames: usize) -> usize {
            let mut guard = self.inner.renderer.lock().unwrap_or_else(|e| e.into_inner());
            let Some(r) = guard.as_mut() else { return 0 };
            if r.playback().idle.load(Ordering::Relaxed) {
                return 0; // parked: the device is not being scheduled at all
            }
            let want = frames * r.format().stride();
            let mut scratch = self.inner.scratch.lock().unwrap_or_else(|e| e.into_inner());
            scratch.clear();
            scratch.resize(want, 0);
            let n = r.render(&mut scratch);
            self.inner
                .captured
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend_from_slice(&scratch[..n]);
            n
        }

        /// Render at most `max_frames`, but only as many whole frames as the ring actually
        /// holds — never padding. The tool for a continuity test, where recorded silence would
        /// be an artefact of the test's pacing rather than of the code under test.
        /// Returns the bytes appended.
        pub fn step_available(&self, max_frames: usize) -> usize {
            let frames = {
                let guard = self.inner.renderer.lock().unwrap_or_else(|e| e.into_inner());
                let Some(r) = guard.as_ref() else { return 0 };
                (r.available() / r.format().stride()).min(max_frames)
            };
            if frames == 0 {
                return 0;
            }
            self.step(frames)
        }

        /// Drain the ring in `quantum`-frame steps until it holds less than one frame.
        /// Returns the bytes appended.
        pub fn drain(&self, quantum: usize) -> usize {
            let mut total = 0;
            loop {
                let n = self.step_available(quantum.max(1));
                if n == 0 {
                    return total;
                }
                total += n;
            }
        }

        /// Bytes the ring currently holds.
        pub fn available(&self) -> usize {
            let guard = self.inner.renderer.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().map(|r| r.available()).unwrap_or(0)
        }

        /// A copy of everything the virtual device has rendered so far.
        pub fn captured(&self) -> Vec<u8> {
            self.inner.captured.lock().unwrap_or_else(|e| e.into_inner()).clone()
        }

        /// Bytes recorded so far, without copying them.
        pub fn captured_len(&self) -> usize {
            self.inner.captured.lock().unwrap_or_else(|e| e.into_inner()).len()
        }

        /// Forget the recording (keeps the device and its counters running).
        pub fn clear(&self) {
            self.inner.captured.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }

        /// Publish an output delay, as a real backend measures one — so the position policies
        /// (delay compensation, the post-seek floor, the monotonic clamp) can be exercised
        /// deterministically.
        pub fn set_delay(&self, d: Duration) {
            let guard = self.inner.renderer.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(r) = guard.as_ref() {
                r.playback().out_delay_ns.store(d.as_nanos() as u64, Ordering::Relaxed);
            }
        }
    }
}

// Test module: fixtures and assertions allocate freely — nothing here is on a media path.
#[allow(clippy::disallowed_methods)]
#[cfg(test)]
mod tests {
    use super::testing::open_capture;
    use super::*;

    fn f32_bytes(samples: &[f32]) -> Vec<u8> {
        #[allow(clippy::disallowed_methods)] // test fixture
        let mut v = Vec::with_capacity(samples.len() * 4);
        for s in samples {
            v.extend_from_slice(&s.to_le_bytes());
        }
        v
    }

    fn f32_samples(bytes: &[u8]) -> Vec<f32> {
        bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    }

    /// Push more audio than the ring holds, the way a running pipeline does: interleave the
    /// writes with the device draining. A bulk push of a second of audio into a half-second
    /// ring simply blocks forever with nothing consuming — which is the backpressure working,
    /// not a bug.
    fn feed(p: &Producer, cap: &super::testing::CaptureHandle, bytes: &[u8]) {
        const CHUNK: usize = 8 * 1024;
        let mut off = 0;
        while off < bytes.len() {
            let end = (off + CHUNK).min(bytes.len());
            assert!(p.push_interruptible(&bytes[off..end], || false), "ring closed mid-feed");
            off = end;
            cap.drain(1024);
        }
        cap.drain(1024);
    }

    // --- the gain ramp -----------------------------------------------------------------

    #[test]
    fn unity_gain_leaves_every_byte_untouched() {
        // The default path: nobody touched the volume, so the sample loop must never run and
        // the bytes must come out exactly as pushed (bit-for-bit, including denormals).
        let src = f32_bytes(&[0.5, -0.25, 1.0, -1.0, 1e-30, f32::MAX]);
        let (out, cap) = open_capture(AudioOutConfig {
            format: CanonicalFormat { sample: SampleFormat::F32, rate: 48_000, channels: 2 },
            ring_secs: 0.5,
        });
        let h = out.handle();
        let p = h.attach().unwrap();
        p.push_interruptible(&src, || false);
        cap.drain(16);
        assert_eq!(cap.captured(), src, "unity gain is a pure passthrough");
    }

    #[test]
    fn a_volume_change_ramps_and_never_steps_more_than_the_bound() {
        // Constant full-scale DC in, so the captured samples *are* the gain envelope.
        const RATE: u32 = 48_000;
        const FRAMES: usize = 4_096;
        let src = f32_bytes(&vec![1.0f32; FRAMES * 2]);
        let (out, cap) = open_capture(AudioOutConfig {
            format: CanonicalFormat { sample: SampleFormat::F32, rate: RATE, channels: 2 },
            ring_secs: 0.5,
        });
        let h = out.handle();
        h.set_volume(0.25);
        let p = h.attach().unwrap();
        p.push_interruptible(&src, || false);
        // Render in small blocks, as a device would.
        for _ in 0..64 {
            cap.step_available(64);
        }
        let got = f32_samples(&cap.captured());
        assert_eq!(got.len(), FRAMES * 2);

        // The gain starts at unity and walks down to the target.
        let bound = 1.0 / (RAMP_SECS * RATE as f32) + 1e-6;
        let mut prev = 1.0f32;
        for (i, &s) in got.iter().enumerate() {
            assert!(s.is_finite() && (0.0..=1.0).contains(&s), "sample {i} out of range: {s}");
            assert!((s - prev).abs() <= bound, "step {i}: {prev} -> {s} exceeds {bound}");
            // Channels of one frame share a gain exactly.
            if i % 2 == 1 {
                assert_eq!(s, got[i - 1], "frame {} channels differ", i / 2);
            }
            prev = s;
        }
        assert!((got[0] - 1.0).abs() < 1e-6, "starts at unity");
        // 0.75 of gap at 1/240 per frame needs 180 frames; long before 4096 it must be exact.
        assert_eq!(*got.last().unwrap(), 0.25, "target reached exactly, and held");
    }

    #[test]
    fn mute_ramps_to_silence_and_unmute_ramps_back() {
        const RATE: u32 = 48_000;
        let src = f32_bytes(&vec![1.0f32; 2_048 * 2]);
        let (out, cap) = open_capture(AudioOutConfig {
            format: CanonicalFormat { sample: SampleFormat::F32, rate: RATE, channels: 2 },
            ring_secs: 0.5,
        });
        let h = out.handle();
        let p = h.attach().unwrap();

        h.set_muted(true);
        p.push_interruptible(&src, || false);
        cap.drain(64);
        let muted = f32_samples(&cap.captured());
        let bound = 1.0 / (RAMP_SECS * RATE as f32) + 1e-6;
        assert!(muted.windows(2).all(|w| (w[1] - w[0]).abs() <= bound), "mute is ramped");
        assert_eq!(*muted.last().unwrap(), 0.0, "reaches exact silence");

        cap.clear();
        h.set_muted(false);
        p.push_interruptible(&src, || false);
        cap.drain(64);
        let back = f32_samples(&cap.captured());
        assert!(back.windows(2).all(|w| (w[1] - w[0]).abs() <= bound), "unmute is ramped");
        assert_eq!(*back.last().unwrap(), 1.0, "returns to the stored volume exactly");
    }

    #[test]
    fn volume_is_clamped_and_nan_is_ignored() {
        let (out, _cap) = open_capture(AudioOutConfig::default());
        let h = out.handle();
        h.set_volume(-3.0);
        assert_eq!(h.volume(), 0.0);
        h.set_volume(99.0);
        assert_eq!(h.volume(), 4.0);
        h.set_volume(0.7);
        assert_eq!(h.volume(), 0.7);
        h.set_volume(f32::NAN);
        assert_eq!(h.volume(), 0.7, "NaN cannot poison the gain");
    }

    #[test]
    fn integer_gain_saturates_instead_of_wrapping() {
        // A boost above unity on full-scale s16 must clip, not wrap to the opposite rail.
        let mut buf = Vec::new();
        for v in [i16::MAX, i16::MIN, 1000i16, -1000] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        apply_gain(SampleFormat::S16, &mut buf, 1, 4.0, 4.0);
        let got: Vec<i16> =
            buf.as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c)).collect();
        assert_eq!(got, vec![i16::MAX, i16::MIN, 4000, -4000]);
    }

    #[test]
    fn every_sample_format_scales_about_its_own_zero() {
        // Half gain on a mid-scale value, for each encoding: the U8 path must scale about
        // 0x80 (offset binary), the signed ones about 0.
        let mut u8b = vec![128u8, 228, 28];
        apply_gain(SampleFormat::U8, &mut u8b, 1, 0.5, 0.5);
        assert_eq!(u8b, vec![128, 178, 78]);

        let mut s24 = vec![0x00, 0x00, 0x40, 0x00, 0x00, 0xC0]; // +2^22, -2^22
        apply_gain(SampleFormat::S24, &mut s24, 1, 0.5, 0.5);
        let dec = |s: &[u8]| ((s[0] as i32 | (s[1] as i32) << 8 | (s[2] as i32) << 16) << 8) >> 8;
        assert_eq!(dec(&s24[0..3]), 1 << 21);
        assert_eq!(dec(&s24[3..6]), -(1 << 21));

        let mut s32 = Vec::new();
        s32.extend_from_slice(&1_000_000i32.to_le_bytes());
        apply_gain(SampleFormat::S32, &mut s32, 1, 0.5, 0.5);
        assert_eq!(i32::from_le_bytes([s32[0], s32[1], s32[2], s32[3]]), 500_000);
    }

    // --- the ring / counter contract -----------------------------------------------------

    #[test]
    fn an_empty_ring_renders_silence_and_still_advances_the_clock() {
        // Nothing attached, nothing pushed: the device must keep running (so a pipeline
        // mastered on the device clock does not stall) and must emit silence, not garbage.
        let (out, cap) = open_capture(AudioOutConfig::default());
        let h = out.handle();
        for _ in 0..10 {
            assert_eq!(cap.step(256), 256 * 8, "a full block every time");
        }
        assert!(cap.captured().iter().all(|&b| b == 0), "silence, not stale bytes");
        assert!(!h.is_attached());
        assert_eq!(h.position(), Duration::ZERO, "no epoch has played");
    }

    #[test]
    fn attach_is_exclusive_and_detach_returns_the_slot() {
        let (out, _cap) = open_capture(AudioOutConfig::default());
        let h = out.handle();
        let p = h.attach().expect("first attach");
        assert!(h.is_attached());
        let second = h.attach();
        assert!(second.is_err(), "a second attach must be loud, not queued");
        h.detach(p);
        assert!(!h.is_attached());
        let _p2 = h.attach().expect("attach again after detach");
    }

    #[test]
    fn attach_after_the_output_is_dropped_is_an_error_not_a_crash() {
        let (out, _cap) = open_capture(AudioOutConfig::default());
        let h = out.handle();
        drop(out);
        assert!(h.attach().is_err(), "attach on a dead output");
        // The rest of the handle stays callable and inert.
        h.set_volume(0.5);
        h.set_muted(true);
        h.set_idle(true);
        assert_eq!(h.volume(), 0.5);
        assert_eq!(h.position(), Duration::ZERO);
    }

    #[test]
    fn a_producer_outliving_the_output_sees_a_closed_ring() {
        // The Arc topology under test: the sink holds the producer, the app drops the
        // AudioOut. Nothing may dangle; the write must simply fail.
        let (out, _cap) = open_capture(AudioOutConfig::default());
        let h = out.handle();
        let p = h.attach().unwrap();
        assert!(!p.is_closed());
        drop(out);
        assert!(p.is_closed(), "the device going away closes the ring");
        assert!(!p.push_interruptible(&[0u8; 64], || false), "and the write fails");
        h.detach(p); // returning it to a dead output is harmless
    }

    #[test]
    fn idle_freezes_the_device_and_preserves_the_ring() {
        // Documented policy: park == freeze. Buffered audio is neither drained nor dropped,
        // and the counters stop.
        let src = f32_bytes(&[0.5; 512]);
        let (out, cap) = open_capture(AudioOutConfig::default());
        let h = out.handle();
        let p = h.attach().unwrap();
        p.push_interruptible(&src, || false);

        h.set_idle(true);
        assert!(h.is_idle());
        let before = cap.available();
        assert_eq!(cap.step(64), 0, "a parked device renders nothing");
        assert_eq!(cap.available(), before, "and consumes nothing");
        assert_eq!(cap.captured_len(), 0);

        h.set_idle(false);
        cap.drain(64);
        assert_eq!(cap.captured(), src, "every byte survived the park, in order");
    }

    // --- the position epoch ----------------------------------------------------------------

    #[test]
    fn position_is_epoch_relative_across_a_handoff() {
        // Two epochs on one output: each one's position must start at zero, not carry the
        // previous track's elapsed time.
        const RATE: u32 = 48_000;
        let cfg = AudioOutConfig {
            format: CanonicalFormat { sample: SampleFormat::F32, rate: RATE, channels: 2 },
            ring_secs: 0.5,
        };
        let (out, cap) = open_capture(cfg);
        let h = out.handle();
        let stride = cfg.format.stride();

        let a = h.attach().unwrap();
        feed(&a, &cap, &vec![0u8; RATE as usize * stride]); // 1 s
        let pos_a = h.position();
        assert!(
            (pos_a.as_secs_f64() - 1.0).abs() < 1e-3,
            "track A played a second, got {pos_a:?}"
        );
        h.detach(a);

        let b = h.attach().unwrap();
        assert_eq!(h.position(), Duration::ZERO, "a fresh epoch starts at zero");
        feed(&b, &cap, &vec![0u8; RATE as usize * stride / 2]); // 0.5 s
        let pos_b = h.position();
        assert!((pos_b.as_secs_f64() - 0.5).abs() < 1e-3, "track B is at 0.5 s, got {pos_b:?}");
    }

    #[test]
    fn an_epoch_opened_behind_a_tail_does_not_count_the_tail() {
        // The gapless case: track A still has audio in the ring when track B attaches. B's
        // position must not start ticking until A's tail has actually played out.
        const RATE: u32 = 48_000;
        let cfg = AudioOutConfig {
            format: CanonicalFormat { sample: SampleFormat::F32, rate: RATE, channels: 2 },
            ring_secs: 0.5,
        };
        let (out, cap) = open_capture(cfg);
        let h = out.handle();
        let stride = cfg.format.stride();

        let a = h.attach().unwrap();
        // 0.2 s of A, none of it rendered yet.
        a.push_interruptible(&vec![0u8; RATE as usize / 5 * stride], || false);
        h.detach(a);
        let b = h.attach().unwrap();
        assert_eq!(h.position(), Duration::ZERO);

        // Render exactly A's tail: B has not been heard at all.
        cap.step(RATE as usize / 5);
        assert_eq!(h.position(), Duration::ZERO, "the previous track's tail is not ours");

        // Now B's own audio plays.
        b.push_interruptible(&vec![0u8; RATE as usize / 10 * stride], || false); // 0.1 s
        cap.drain(1024);
        assert!((h.position().as_secs_f64() - 0.1).abs() < 1e-3, "{:?}", h.position());
    }

    #[test]
    fn a_seek_inside_a_later_epoch_still_holds_at_its_target() {
        // The sink's `FlushStart` rebase, in its documented order, against an epoch that does
        // *not* start at frame zero: `frames = epoch_start + target`, floor at the target,
        // epoch published last. (Driving the real seek needs a pipeline's `SeekState`, which
        // core keeps `pub(crate)`; this pins the arithmetic `flush_start` performs, the way
        // the sink's own seek tests do.)
        const RATE: u32 = 48_000;
        let cfg = AudioOutConfig {
            format: CanonicalFormat { sample: SampleFormat::F32, rate: RATE, channels: 2 },
            ring_secs: 0.5,
        };
        let (out, cap) = open_capture(cfg);
        let h = out.handle();
        let stride = cfg.format.stride();

        // Burn a first epoch so the second starts at a non-zero frame count.
        let a = h.attach().unwrap();
        feed(&a, &cap, &vec![0u8; RATE as usize * stride]); // 1 s of track A
        h.detach(a);
        let _b = h.attach().unwrap();
        let pb = &h.shared.pb;
        let epoch_start = pb.epoch_start_frames.load(Ordering::Relaxed);
        assert_eq!(epoch_start, RATE as u64, "epoch B starts one second in");

        cap.set_delay(Duration::from_millis(100));
        let ns = 30_000_000_000u64;
        let frames = ns / 1_000_000_000 * RATE as u64;
        pb.frames.store(epoch_start + frames, Ordering::Relaxed);
        open_epoch(pb, ns);
        assert!(
            (h.position().as_secs_f64() - 30.0).abs() < 1e-6,
            "held at the target, not 29.9: {:?}",
            h.position()
        );

        // Once the write side passes target + delay, the subtraction takes over.
        pb.frames.store(epoch_start + frames + RATE as u64 / 5, Ordering::Relaxed); // +200 ms
        assert!((h.position().as_secs_f64() - 30.1).abs() < 1e-3, "{:?}", h.position());
    }

    #[test]
    fn position_is_compensated_by_the_measured_delay_and_never_goes_backwards() {
        const RATE: u32 = 48_000;
        let cfg = AudioOutConfig {
            format: CanonicalFormat { sample: SampleFormat::F32, rate: RATE, channels: 2 },
            ring_secs: 0.5,
        };
        let (out, cap) = open_capture(cfg);
        let h = out.handle();
        let a = h.attach().unwrap();
        feed(&a, &cap, &vec![0u8; RATE as usize * cfg.format.stride()]);

        cap.set_delay(Duration::from_millis(100));
        let p1 = h.position();
        assert!((p1.as_secs_f64() - 0.9).abs() < 1e-3, "1 s written, 100 ms in flight: {p1:?}");
        // A jittering measurement must not walk the reported position backwards.
        cap.set_delay(Duration::from_millis(140));
        assert!(h.position() >= p1);
        cap.set_delay(Duration::from_millis(60));
        assert!((h.position().as_secs_f64() - 0.94).abs() < 1e-3);
    }
}
