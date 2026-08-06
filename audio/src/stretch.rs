//! `audiostretch` — pitch-preserving playback-rate change (spec: Formats — dynamic caps;
//! Writing elements; Dynamic element properties). Interleaved `audio/raw` PCM arrives on the
//! sink pad; the *same* audio, played faster or slower **at the original pitch**, leaves on the
//! src pad in the same rate / channels / sample format. Only the number of frames changes.
//!
//! This is the podcast speed control: a plain resample plays 1.5× speech a fifth higher
//! (chipmunk), so instead the stream is time-scale-modified with **PICOLA** — Pointer Interval
//! Control OverLap and Add (N. Morita and F. Itakura, *"Time-scale modification algorithm for
//! speech by use of Pointer Interval Control OverLap and Add (PICOLA) and its evaluation"*,
//! Proc. ICASSP 1986). The DSP core is ported verbatim from the author's music player
//! (`mskp-stretch`), which follows the PICOLA lineage popularised by Bill Cox's *Sonic* (the
//! time stretcher behind Android TTS and most audiobook apps): the speaker's pitch period is
//! found with an AMDF search, and *whole periods* are dropped (speed-up) or repeated
//! (slow-down), joined by a period-long linear crossfade. Because every splice is exactly one
//! pitch period, joins are phase-aligned by construction and there is no fixed splice cadence
//! to buzz at — the two artifacts that make block-based WSOLA sound robotic on speech.
//!
//! # Timestamp policy (read this before wiring it up)
//!
//! A time stretcher breaks the usual identity between *playback time* and *source time*: one
//! wall-clock second of output consumes `rate` seconds of input. The element therefore keeps
//! **two** clocks, and it is deliberate which one lands on the buffers:
//!
//! * **Outgoing `pts` is playback time** — a continuous, gap-free grid over the frames this
//!   element has *emitted*: `pts = base + emitted_frames / sample_rate`. Since the stretcher
//!   emits `1/rate` output frames per input frame by construction, this is exactly "input pts
//!   scaled by `1/rate`", accumulated instead of recomputed. Accumulating is what makes a
//!   mid-stream rate change free of discontinuities: the grid never jumps, it only starts
//!   advancing through the source faster or slower. A clock-driven sink can pace on it
//!   directly, exactly as it would on an unstretched stream, and `pts` stays monotonic across
//!   rate changes, buffer boundaries, and the EOS tail.
//! * **Source time is published out-of-band**, through [`StretchPosition`] — a cheap
//!   `Arc`-of-atomics handle taken from the element before it is added to the pipeline (the
//!   `TestSinkStats` pattern). `position()` returns `base + consumed_frames / sample_rate`,
//!   i.e. how far into the *episode* the audio just emitted came from. This is the direct port
//!   of `mskp-stretch`'s `Control::position()`, which exists for the same reason: the music
//!   player's `playback_pos()` cannot use the audio backend's wall-clock `get_pos` at ≠1×, so
//!   it reads the stretcher's own source-time counter instead.
//!
//! The alternative — stamping source time on the buffers — was rejected: it makes `pts`
//! non-monotonic in rate (a 2× stream would advance its pts at twice the clock), so every
//! downstream sink would have to know the rate to schedule anything. Source position is
//! app-facing state, not pipeline state, so it travels on an app-facing channel.
//!
//! An app that cannot hold the handle can still recover source time, but only with the full
//! rate history: `source(t) = base + ∫ rate(τ) dτ` over playback time `τ ∈ [0, t]`. With a
//! single constant rate that collapses to `source = base + (pts − base) * rate`. The handle
//! exists precisely so that live rate changes do not force the app to integrate.
//!
//! Both timelines share one `base`, rebased on a seek to the seek target — so at the instant
//! of a seek, playback time and source time coincide, and diverge again afterwards.
//!
//! # The `rate` property (live)
//!
//! `rate` is settable while `Playing` (`live: true`), applied at the batch boundary the
//! scheduler delivers [`Event::PropChanged`] on. The value universe has no float, so three
//! spellings are accepted and all mean the same thing:
//!
//! | spelling | example | meaning |
//! |---|---|---|
//! | [`Value::Rat`] | `rate=3/2` | exact fraction — the precise form |
//! | [`Value::Int`] | `rate=2` | whole multiple |
//! | [`Value::Id`] | `rate=1.5` | decimal text, parsed as `f32` (what `parse()` yields) |
//!
//! Clamped to `[0.25, 4.0]`, the hard range of the source `Control::set_rate` (the music
//! player's UI exposes the narrower `0.5 ..= 3.0`). Rates within `0.02` of 1.0 **bypass** the
//! stretcher: input buffers are then forwarded by move, with no copy and no DSP at all.
//!
//! # Adaptation notes (pull `Source` → push element)
//!
//! The DSP — period search, splice schedule, crossfade, fractional-frame carry — is
//! byte-for-byte the original. Only the I/O shell changed:
//!
//! * The original *pulls* (`fill_input` calls `inner.next()` until it has what a round needs).
//!   A push element cannot pull, so each round is **gated** on the same threshold instead: a
//!   splice runs only once `3 * max_period` frames are buffered, a copy run once
//!   `min(copy_remaining, 1024)` are. The original's short-input branches (the final
//!   pass-through tail) are reached only at EOS, exactly as `inner_done` reaches them there.
//!   The sample values are therefore identical, not merely close.
//! * The original's `out` `VecDeque` is replaced by a fixed-capacity round buffer flushed to
//!   pool slots after every round. Chunk boundaries differ from rodio's one-sample-at-a-time
//!   pull; sample values do not.
//! * In bypass the original still waits for a full 1024-frame chunk; here a short buffered
//!   remainder is flushed immediately so the copy-free forward path can resume. Bypass output
//!   is a verbatim copy either way, so this changes chunking only.
//! * `frac` and `prev_min_diff` are scrubbed if a non-finite input sample poisons them. For
//!   finite input this is unreachable and cannot change the output; it stops a NaN payload
//!   from permanently wedging the splice schedule.
//! * The element works in `f32` internally (as the original does) and converts at the edges
//!   with [`crate::convert::convert_interleaved`], so any `audio/raw` sample format is
//!   accepted. An `f32` stream round-trips through a `memcpy`, bit-exactly.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, PropDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{Constraint, ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

use crate::convert::{convert_interleaved, converted_len};
use crate::format::{AudioFormat, SampleFormat, FAMILY, FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE};

const SINK: PadId = PadId(0);
const SRC: PadId = PadId(1);

// --- PICOLA tuning, ported verbatim from `mskp-stretch` -------------------------------

/// Pitch range the period search covers. 65 Hz reaches a deep male voice; 400 Hz a high one.
/// Unvoiced/silent stretches have no true period — the search still returns *some* minimum,
/// which splices fine (noise has no phase to misalign).
const MIN_PITCH_HZ: usize = 65;
const MAX_PITCH_HZ: usize = 400;
/// The AMDF coarse pass runs on a mixdown decimated to roughly this rate; the winner is then
/// refined at full resolution.
const AMDF_FREQ: usize = 4000;
/// Rates this close to 1.0 bypass the stretcher entirely.
const BYPASS_EPSILON: f32 = 0.02;
/// Frames per refill in bypass mode, and cap per verbatim-copy round.
const BYPASS_CHUNK: usize = 1024;

/// Rate clamp — the hard range of the source `Control::set_rate`. (The music player's UI
/// offers `0.5 ..= 3.0` within it; the wider range is what the algorithm stays sane over.)
pub const RATE_MIN: f32 = 0.25;
pub const RATE_MAX: f32 = 4.0;

/// Worst-case added latency: a splice round holds `3 * max_period` frames, and
/// `max_period = sample_rate / 65`, so the window is `3/65 s` at *any* sample rate.
const WINDOW_LATENCY_NS: u64 = 3 * 1_000_000_000 / MIN_PITCH_HZ as u64;

// --- Pads / props / descriptor --------------------------------------------------------

// Both pads speak `audio/raw` (any rate/channels/sample) plus the `bytes` escape, mirroring
// `audioconvert`/`audioresample`. The stretcher never changes the format, so the src pad
// announces exactly what the sink negotiated; it is `dynamic` only because that announcement
// happens at runtime. Every sample-format name is offered so the `sample` value is interned
// for the announcement to resolve against.
static SAMPLE_VALUES: [ValueDesc; 5] = [
    ValueDesc::Id("u8"),
    ValueDesc::Id("s16"),
    ValueDesc::Id("s24"),
    ValueDesc::Id("s32"),
    ValueDesc::Id("f32"),
];
static FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: FIELD_RATE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: FIELD_CHANNELS, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: FIELD_SAMPLE, allowed: ConstraintDesc::Set(&SAMPLE_VALUES), preferred: None },
];
static OFFERS: [OfferDesc; 2] =
    [OfferDesc { family: FAMILY, fields: &FIELDS }, OfferDesc::any("bytes")];

static PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &OFFERS,
        dynamic: true, // the (unchanged) format is announced at runtime
        validate: None,
    },
];

/// Playback rate, pitch preserved. **Live** (spec: Dynamic element properties): a set lands at
/// the next batch boundary and the splice schedule picks it up on the following round — see the
/// module docs for the three accepted spellings.
static PROPS: [PropDesc; 1] = [PropDesc { name: "rate", allowed: Constraint::Any, live: true }];

// COLD: the `make_default` factory boxes one element at construction, not per buffer.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "audiostretch",
    pads: &PADS,
    props: &PROPS,
    // Passive: a pure transform that inlines into the upstream group (spec: Scheduling).
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        // Bypass adds nothing; a stretching round holds up to one search window (3/65 s).
        min: Timestamp::ZERO,
        max: Timestamp::from_nanos(WINDOW_LATENCY_NS),
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Default 1.0 (bypass); override via the `rate` prop at parse time.
    make_default: Some(|| Box::new(AudioStretch::new(1.0))),
};

// --- Source-time position handle ------------------------------------------------------

/// The counters [`StretchPosition`] reads. Relaxed throughout: these are monotone progress
/// counters for a UI, not a synchronisation channel.
struct PositionShared {
    /// Source-time position in nanoseconds of the audio last emitted.
    pos_nanos: AtomicU64,
    /// Input frames consumed since the last seek.
    in_frames: AtomicU64,
    /// Output frames emitted since the last seek.
    out_frames: AtomicU64,
    /// The rate currently applied, as `f32` bits — atomics have no native `f32`
    /// (the source `Control` does the same).
    rate_bits: AtomicU32,
}

/// App-side view of an [`AudioStretch`]'s **source-time** position — the direct port of
/// `mskp-stretch`'s `Control`, minus the rate setter (that is the live `rate` prop now).
///
/// Take it with [`AudioStretch::position`] *before* handing the element to the pipeline; it is
/// `Arc`-backed, so the clone is cheap and stays valid for the element's whole life. Every
/// read is a relaxed atomic load — safe to poll from a UI frame callback.
///
/// ```ignore
/// let el = AudioStretch::new(1.5);
/// let pos = el.position();          // keep this
/// let id = pipeline.add(el);
/// // … later, from the UI thread:
/// let source_time = pos.position(); // where in the episode we are
/// props.set(id, "rate", Value::Rat(2, 1)).unwrap();
/// ```
#[derive(Clone)]
pub struct StretchPosition(Arc<PositionShared>);

impl StretchPosition {
    /// Source time (episode time) of the audio most recently emitted — what a progress bar
    /// should show. Advances at `rate` × wall clock.
    pub fn position(&self) -> Timestamp {
        Timestamp::from_nanos(self.0.pos_nanos.load(Ordering::Relaxed))
    }

    /// Snap the reported position — for a decode-ahead consumer that has seeked before the
    /// stretcher itself has processed the flush. The next round of real progress overwrites
    /// it (same contract as the source `Control::set_position`).
    pub fn set_position(&self, pos: Timestamp) {
        self.0.pos_nanos.store(pos.nanos().unwrap_or(0), Ordering::Relaxed);
    }

    /// Input frames consumed since the last seek.
    pub fn source_frames(&self) -> u64 {
        self.0.in_frames.load(Ordering::Relaxed)
    }

    /// Output frames emitted since the last seek. `output_frames / sample_rate` is the
    /// playback-time offset carried in the outgoing `pts`.
    pub fn output_frames(&self) -> u64 {
        self.0.out_frames.load(Ordering::Relaxed)
    }

    /// The rate the element is currently applying (post-clamp).
    pub fn rate(&self) -> f32 {
        f32::from_bits(self.0.rate_bits.load(Ordering::Relaxed))
    }
}

// --- The PICOLA engine ----------------------------------------------------------------

/// The ported stretcher: fixed-capacity input buffer, period search, splice schedule. Built
/// once the input format is known, torn down and rebuilt if it changes. Everything it touches
/// per round lives in these buffers — `generate` never allocates.
struct Engine {
    fmt: AudioFormat,
    channels: usize,
    /// Period search bounds in frames (`sample_rate / pitch`).
    min_period: usize,
    max_period: usize,
    /// Decimation stride for the coarse AMDF pass.
    amdf_skip: usize,
    /// Frames a splice round must have buffered (`3 * max_period`).
    window: usize,
    /// Buffered input, interleaved f32; `start` is the read head (in *samples*).
    /// Compacted in place instead of drained per round; never reallocated.
    buf: Vec<f32>,
    start: usize,
    /// Frames still to copy verbatim before the next splice (the run between splices at rates
    /// below 2× / above 0.5×).
    copy_remaining: usize,
    /// Last measured period and its AMDF score. Reused through silence, and preferred over a
    /// new measurement that matches *worse* than the last one did (Sonic's hysteresis) —
    /// unvoiced frames have no true period, and letting their random minima through jitters
    /// the splice cadence.
    prev_period: usize,
    prev_min_diff: f32,
    /// Scratch: mono mixdown of the search window (full-res), and its decimated copy.
    mono: Vec<f32>,
    mono_coarse: Vec<f32>,
    /// Fractional frames of skip/insert owed, carried between splices so the long-run tempo
    /// is exact.
    frac: f32,
    /// Input frames consumed since the last seek.
    consumed: u64,
    /// One round's output, interleaved f32.
    out: Vec<f32>,
    /// Byte staging: decoded input, `out` as native-endian f32 bytes, and the re-encoded
    /// output. All fixed-capacity.
    in_stage: Vec<u8>,
    out_f32: Vec<u8>,
    out_pcm: Vec<u8>,
    /// The capacities every buffer above was built with. The house rule (spec: performance #1
    /// — no steady-state heap traffic) is that none of them ever grows again, so
    /// [`Engine::assert_no_growth`] can check that invariant directly instead of trusting a
    /// reviewer to re-derive the worst-case round size. See
    /// `examples/audiostretch_alloc_check.rs` for the matching runtime measurement.
    caps: [usize; 7],
}

impl Engine {
    /// Size every buffer for the worst-case round and allocate them **once**.
    // COLD: called on format negotiation (once per stream), not per buffer. Every `Vec` here is
    // sized for the worst-case round and never grows again — `generate()` is allocation-free.
    #[allow(clippy::disallowed_methods)]
    fn new(fmt: AudioFormat) -> Self {
        let sr = fmt.sample_rate as usize;
        let ch = (fmt.channels as usize).max(1);
        let max_period = (sr / MIN_PITCH_HZ).max(2);
        let amdf_skip = (sr / AMDF_FREQ).max(1);
        let window = 3 * max_period;
        // Room for a full search window plus a feed chunk, so `feed` can always make progress
        // after a drain without the buffer ever reallocating.
        let buf_frames = window + 2 * BYPASS_CHUNK;
        // Worst-case round output: the EOS tail passes through up to a whole window; a
        // slow-down round emits `2 * period`; a bypass/copy round emits `BYPASS_CHUNK`.
        let out_frames = window.max(2 * max_period).max(BYPASS_CHUNK);
        Self {
            fmt,
            channels: ch,
            min_period: (sr / MAX_PITCH_HZ).max(1),
            max_period,
            amdf_skip,
            window,
            buf: Vec::with_capacity(buf_frames * ch),
            start: 0,
            copy_remaining: 0,
            prev_period: 0,
            prev_min_diff: 0.0,
            mono: Vec::with_capacity(2 * max_period),
            mono_coarse: Vec::with_capacity(2 * max_period / amdf_skip + 2),
            frac: 0.0,
            consumed: 0,
            out: Vec::with_capacity(out_frames * ch),
            in_stage: Vec::with_capacity(BYPASS_CHUNK * ch * 4),
            out_f32: Vec::with_capacity(out_frames * ch * 4),
            out_pcm: Vec::with_capacity(out_frames * ch * 4),
            caps: [0; 7],
        }
        .with_recorded_caps()
    }

    fn with_recorded_caps(mut self) -> Self {
        self.caps = self.capacities();
        self
    }

    fn capacities(&self) -> [usize; 7] {
        [
            self.buf.capacity(),
            self.mono.capacity(),
            self.mono_coarse.capacity(),
            self.out.capacity(),
            self.in_stage.capacity(),
            self.out_f32.capacity(),
            self.out_pcm.capacity(),
        ]
    }

    /// Every buffer was sized for the worst-case round at [`Engine::new`]; a grown capacity
    /// means a round overran its bound and the element allocated on the hot path. Checked in
    /// debug builds after every round, so the crate's own test suite (which drives all five
    /// branches of the splice schedule) is the regression guard.
    fn assert_no_growth(&self) {
        debug_assert_eq!(
            self.capacities(),
            self.caps,
            "audiostretch: a round grew one of the engine's fixed buffers — \
             [buf, mono, mono_coarse, out, in_stage, out_f32, out_pcm]"
        );
    }

    /// Drop every trace of the audio seen so far (seek / format change). Capacities survive.
    fn reset(&mut self) {
        self.buf.clear();
        self.start = 0;
        self.copy_remaining = 0;
        self.prev_period = 0;
        self.prev_min_diff = 0.0;
        self.frac = 0.0;
        self.consumed = 0;
        self.out.clear();
    }

    fn buffered_frames(&self) -> usize {
        (self.buf.len() - self.start) / self.channels
    }

    /// Free frames at the tail, without reallocating.
    fn free_frames(&self) -> usize {
        (self.buf.capacity() - self.buf.len()) / self.channels
    }

    /// Slide the unread remainder to the front, reclaiming the consumed prefix.
    fn compact(&mut self) {
        if self.start == 0 {
            return;
        }
        let keep = self.buf.len() - self.start;
        self.buf.copy_within(self.start.., 0);
        self.buf.truncate(keep);
        self.start = 0;
    }

    /// Append `bytes` (whole interchannel frames of `self.fmt`) as interleaved f32. The caller
    /// guarantees `bytes` holds at most [`Engine::free_frames`] frames, so no reallocation
    /// happens. Returns the frames appended.
    fn append(&mut self, bytes: &[u8]) -> Option<usize> {
        let len = converted_len(self.fmt.format, SampleFormat::F32, self.channels, bytes.len())?;
        if len > self.in_stage.capacity() {
            return None;
        }
        self.in_stage.clear();
        self.in_stage.resize(len, 0);
        convert_interleaved(
            self.fmt.format,
            SampleFormat::F32,
            self.channels,
            bytes,
            &mut self.in_stage,
        )?;
        debug_assert!(len / 4 <= self.buf.capacity() - self.buf.len(), "engine buf would grow");
        for word in self.in_stage.as_chunks::<4>().0 {
            self.buf.push(f32::from_ne_bytes(*word));
        }
        Some(len / 4 / self.channels)
    }

    /// AMDF pitch period of the input at the read head: the lag in `[min_period, max_period]`
    /// minimizing the mean `|s[i] - s[i+lag]|`. Coarse pass on a decimated mixdown, refined at
    /// full resolution. Needs `2 * max_period` frames buffered.
    ///
    /// (Ported verbatim; AMDF period detection is Ross et al., *"Average magnitude difference
    /// function pitch extractor"*, IEEE Trans. ASSP 22(5), 1974.)
    fn find_pitch_period(&mut self) -> usize {
        let ch = self.channels;
        let input = &self.buf[self.start..];
        self.mono.clear();
        self.mono.extend(
            input
                .chunks_exact(ch)
                .take(2 * self.max_period)
                .map(|f| f.iter().sum::<f32>()),
        );

        // Silence carries no period; keep the speaker's last one so pauses splice at the same
        // cadence.
        let energy: f32 = self.mono.iter().map(|s| s * s).sum();
        if energy < 1e-6 && self.prev_period != 0 {
            return self.prev_period;
        }

        let skip = self.amdf_skip;
        self.mono_coarse.clear();
        self.mono_coarse.extend(self.mono.iter().step_by(skip));

        let amdf = |signal: &[f32], lag: usize| -> f32 {
            let n = lag.min(signal.len().saturating_sub(lag));
            if n == 0 {
                return f32::MAX;
            }
            let diff: f32 = signal[..n]
                .iter()
                .zip(&signal[lag..lag + n])
                .map(|(a, b)| (a - b).abs())
                .sum();
            diff / n as f32
        };

        let (lo, hi) = (self.min_period / skip, self.max_period / skip);
        let mut best = lo.max(1);
        let mut best_diff = f32::MAX;
        for lag in lo.max(1)..=hi {
            let d = amdf(&self.mono_coarse, lag);
            if d < best_diff {
                best_diff = d;
                best = lag;
            }
        }

        // Refine around the coarse winner at full resolution
        let center = best * skip;
        let lo = center.saturating_sub(2 * skip).max(self.min_period);
        let hi = (center + 2 * skip).min(self.max_period);
        let mut best = lo;
        let mut best_diff = f32::MAX;
        for lag in lo..=hi {
            let d = amdf(&self.mono, lag);
            if d < best_diff {
                best_diff = d;
                best = lag;
            }
        }

        // A match that fits worse than the previous frame's did is likely an unvoiced stretch;
        // keep the established period instead.
        let chosen = if self.prev_period != 0 && best_diff > self.prev_min_diff {
            self.prev_period
        } else {
            best
        };
        self.prev_period = best;
        // Adaptation: a non-finite score (NaN input) would make every later comparison false
        // and freeze the hysteresis. Unreachable for finite audio.
        self.prev_min_diff = if best_diff.is_finite() { best_diff } else { f32::MAX };
        chosen.clamp(1, self.max_period)
    }

    /// Emit `frames` frames crossfading `from` (fading out) into `to` (fading in), both frame
    /// offsets relative to the read head. The linear ramp spans the whole splice — one pitch
    /// period — so joins are gentle.
    fn overlap_add(&mut self, frames: usize, from: usize, to: usize) {
        let ch = self.channels;
        let base = self.start;
        // Defensive: the splice schedule never asks for more than `2 * max_period` frames and a
        // round only runs with `3 * max_period` buffered, so this clamp is unreachable. It is
        // here so a future schedule change degrades into a short crossfade, not a panic.
        let avail = self.buffered_frames().saturating_sub(from.max(to));
        let frames = frames.min(avail);
        for i in 0..frames {
            let t = i as f32 / frames as f32;
            for c in 0..ch {
                let a = self.buf[base + (from + i) * ch + c];
                let b = self.buf[base + (to + i) * ch + c];
                self.out.push(a * (1.0 - t) + b * t);
            }
        }
    }

    /// Copy frames verbatim from the read head and consume them.
    fn pass_through(&mut self, frames: usize) {
        let n = frames * self.channels;
        let s = self.start;
        self.out.extend_from_slice(&self.buf[s..s + n]);
        self.consume(frames);
    }

    fn consume(&mut self, frames: usize) {
        self.start += frames * self.channels;
        self.consumed += frames as u64;
    }

    /// One round: a bypass block, a verbatim run, or one pitch-synchronous splice, appended to
    /// `out`. Returns `false` when the round cannot run yet — the push-model replacement for
    /// the original's `fill_input` pull (see the module's adaptation notes). `eos` unlocks the
    /// original's short-input branches, which a pull source reaches via `inner_done`.
    fn generate(&mut self, rate: f32, eos: bool) -> bool {
        // Adaptation: scrub a NaN carried in from non-finite input, which would otherwise turn
        // every later `ideal`/`copy` into NaN and pin the splice length at its clamp forever.
        if !self.frac.is_finite() {
            self.frac = 0.0;
        }
        let buffered = self.buffered_frames();

        if (rate - 1.0).abs() < BYPASS_EPSILON {
            if buffered == 0 {
                return false;
            }
            let frames = buffered.min(BYPASS_CHUNK);
            self.pass_through(frames);
            // A later return to stretching restarts cleanly
            self.copy_remaining = 0;
            self.frac = 0.0;
            return true;
        }

        // The run between splices is plain copy
        if self.copy_remaining > 0 {
            let want = self.copy_remaining.min(BYPASS_CHUNK);
            if buffered < want && !eos {
                return false;
            }
            let frames = buffered.min(self.copy_remaining).min(BYPASS_CHUNK);
            if frames == 0 {
                self.copy_remaining = 0;
                // State advanced; the next round takes the splice branch and stops there.
                return true;
            }
            self.pass_through(frames);
            self.copy_remaining -= frames;
            return true;
        }

        // A splice needs a full search window plus the period being blended
        if buffered < self.window {
            if !eos || buffered == 0 {
                return false;
            }
            // Tail end of the episode: pass the remainder through. Tempo drifts for the final
            // few tens of ms; inaudible.
            self.pass_through(buffered);
            return true;
        }

        let period = self.find_pitch_period();

        if rate > 1.0 {
            // Speed up: blend one period into the next, dropping a period. The blend length
            // shrinks as the rate grows (Sonic's schedule): at exactly 2x every period is
            // spliced with its neighbor; above that the splice itself must span less than a
            // period.
            if rate >= 2.0 {
                let ideal = period as f32 / (rate - 1.0) + self.frac;
                let new_frames = (ideal as usize).clamp(1, period);
                self.frac = ideal - new_frames as f32;
                self.overlap_add(new_frames, 0, period);
                self.consume(period + new_frames);
            } else {
                let copy = period as f32 * (2.0 - rate) / (rate - 1.0) + self.frac;
                self.copy_remaining = copy as usize;
                self.frac = copy - self.copy_remaining as f32;
                self.overlap_add(period, 0, period);
                self.consume(2 * period);
            }
        } else {
            // Slow down: emit one period verbatim, then blend back to repeat it, inserting
            // `new_frames` of output the input doesn't advance through.
            let ch = self.channels;
            if rate < 0.5 {
                let ideal = period as f32 * rate / (1.0 - rate) + self.frac;
                let new_frames = (ideal as usize).clamp(1, period);
                self.frac = ideal - new_frames as f32;
                let s = self.start;
                self.out.extend_from_slice(&self.buf[s..s + period * ch]);
                self.overlap_add(new_frames, period, 0);
                self.consume(new_frames);
            } else {
                let copy = period as f32 * (2.0 * rate - 1.0) / (1.0 - rate) + self.frac;
                self.copy_remaining = copy as usize;
                self.frac = copy - self.copy_remaining as f32;
                let s = self.start;
                self.out.extend_from_slice(&self.buf[s..s + period * ch]);
                self.overlap_add(period, period, 0);
                self.consume(period);
            }
        }
        true
    }

    /// Re-encode `out` into the negotiated sample format, leaving the bytes in `out_pcm`.
    fn encode_round(&mut self) -> Result<(), Error> {
        self.out_f32.clear();
        for s in &self.out {
            self.out_f32.extend_from_slice(&s.to_ne_bytes());
        }
        let len =
            converted_len(SampleFormat::F32, self.fmt.format, self.channels, self.out_f32.len())
                .ok_or(Error::Todo("audiostretch: ragged round output"))?;
        self.out_pcm.clear();
        self.out_pcm.resize(len, 0);
        convert_interleaved(
            SampleFormat::F32,
            self.fmt.format,
            self.channels,
            &self.out_f32,
            &mut self.out_pcm,
        )
        .ok_or(Error::Todo("audiostretch: re-encode failed"))?;
        Ok(())
    }
}

// --- The element ----------------------------------------------------------------------

/// Pitch-preserving playback-rate change. Construct with [`AudioStretch::new`]; take
/// [`AudioStretch::position`] before adding it to a pipeline if the app needs source time.
pub struct AudioStretch {
    /// The rate currently applied, already clamped.
    rate: f32,
    /// Input format learned from negotiated caps. `None` until known.
    input: Option<AudioFormat>,
    /// The PICOLA engine, built once the input format is known.
    engine: Option<Engine>,
    /// Whether the (unchanged) output format has been announced downstream.
    announced: bool,
    /// Input interchannel-frame stride in bytes; `0` until known.
    in_stride: usize,
    /// Carry for a partial input frame straddling two buffers, so channel lanes never desync
    /// (mirrors `audioconvert`). Capacity is one frame — filling it never allocates.
    carry: Vec<u8>,
    /// Shared source-time counters (see [`StretchPosition`]).
    shared: Arc<PositionShared>,
    /// Origin of both timelines: the first input `pts` seen, or the seek target after a flush.
    base: Timestamp,
    base_locked: bool,
    /// Output frames emitted since the last seek — the playback-time `pts` grid.
    out_frames: u64,
}

impl AudioStretch {
    /// A stretcher running at `rate` (1.0 = bypass), discovering its format at runtime from the
    /// upstream's negotiated `audio/raw` caps (spec: dynamic caps — consumer side).
    // COLD: constructs the element once. `carry` is a one-frame cross-buffer accumulator, not
    // per-buffer scratch, and the shared counters are a single `Arc`.
    #[allow(clippy::disallowed_methods)]
    pub fn new(rate: f32) -> Self {
        let shared = Arc::new(PositionShared {
            pos_nanos: AtomicU64::new(0),
            in_frames: AtomicU64::new(0),
            out_frames: AtomicU64::new(0),
            rate_bits: AtomicU32::new(1.0f32.to_bits()),
        });
        let mut el = Self {
            rate: 1.0,
            input: None,
            engine: None,
            announced: false,
            in_stride: 0,
            carry: Vec::new(),
            shared,
            base: Timestamp::ZERO,
            base_locked: false,
            out_frames: 0,
        };
        el.set_rate(rate);
        el
    }

    /// A stretcher with the input format pinned at construction (statically-wired / testable
    /// form). Still announces its format downstream.
    pub fn with_input(input: AudioFormat, rate: f32) -> Self {
        let mut s = Self::new(rate);
        s.set_input(input);
        s
    }

    /// A handle on the **source-time** position — see the module's timestamp policy. Cheap to
    /// clone; take it before the element is moved into the pipeline.
    pub fn position(&self) -> StretchPosition {
        StretchPosition(Arc::clone(&self.shared))
    }

    /// The rate currently applied (post-clamp).
    pub fn rate(&self) -> f32 {
        self.rate
    }

    /// The input format, once known.
    pub fn input_format(&self) -> Option<AudioFormat> {
        self.input
    }

    /// Apply a new rate, clamped to `[RATE_MIN, RATE_MAX]`. A non-finite request is ignored
    /// (rather than poisoning the splice schedule).
    fn set_rate(&mut self, rate: f32) {
        if !rate.is_finite() {
            return;
        }
        self.rate = rate.clamp(RATE_MIN, RATE_MAX);
        self.shared.rate_bits.store(self.rate.to_bits(), Ordering::Relaxed);
    }

    /// Decode a `rate` property value. The value universe has no float, so an exact rational, a
    /// whole multiple, and decimal text (what `parse("audiostretch rate=1.5")` interns) are all
    /// accepted — see the module docs.
    fn rate_from_value(ctx: &Ctx, v: Value) -> Option<f32> {
        match v {
            Value::Int(n) => Some(n as f32),
            Value::Rat(n, d) if d != 0 => Some(n as f32 / d as f32),
            Value::Rat(_, _) => None,
            Value::Id(id) => ctx.value_name(id).and_then(|s| s.parse::<f32>().ok()),
        }
    }

    /// Record the input format and (re)build the engine. A genuinely different format re-arms
    /// the downstream announcement.
    fn set_input(&mut self, input: AudioFormat) {
        if self.input == Some(input) {
            return;
        }
        self.in_stride = input.frame_stride();
        self.input = Some(input);
        self.announced = false;
        self.carry.clear();
        self.carry.reserve(self.in_stride.max(1));
        self.engine = Some(Engine::new(input));
    }

    /// Infer the input [`AudioFormat`] from the sink's negotiated `audio/raw` caps, reading
    /// every field by name (the `audioconvert` / `audioresample` / `flacdec` pattern).
    fn learn_from_sink(&mut self, ctx: &Ctx) {
        if self.input.is_some() {
            return;
        }
        let Some(fixed) = ctx.negotiated(SINK) else { return };
        let int_field = |name: &str| -> Option<i64> {
            ctx.field_id(name).and_then(|id| fixed.get(id)).and_then(|v| match v {
                Value::Int(n) => Some(n),
                _ => None,
            })
        };
        let (Some(rate), Some(channels)) = (int_field(FIELD_RATE), int_field(FIELD_CHANNELS))
        else {
            return;
        };
        if rate <= 0 || channels <= 0 {
            return;
        }
        let sample = ctx
            .field_id(FIELD_SAMPLE)
            .and_then(|id| fixed.get(id))
            .and_then(|v| match v {
                Value::Id(vid) => ctx.value_name(vid),
                _ => None,
            })
            .and_then(SampleFormat::from_caps_name);
        // Need the sample format to interpret bytes; without it, wait for a fuller caps.
        if let Some(fmt) = sample {
            self.set_input(AudioFormat::new(rate as u32, channels as u16, fmt));
        }
    }

    /// Announce the output format on the src pad — identical to the input, since the stretcher
    /// changes only how many frames there are (spec: dynamic caps — producer side).
    fn announce_output(&mut self, ctx: &mut Ctx) {
        if self.announced {
            return;
        }
        let Some(inp) = self.input else { return };
        ctx.announce_format(
            SRC,
            FAMILY,
            &[
                (FIELD_RATE, ValueDesc::Int(inp.sample_rate as i64)),
                (FIELD_CHANNELS, ValueDesc::Int(inp.channels as i64)),
                (FIELD_SAMPLE, ValueDesc::Id(inp.format.caps_name())),
            ],
        );
        self.announced = true;
    }

    /// Nanoseconds for `frames` at the negotiated rate. `u128` intermediate so a multi-day
    /// stream cannot overflow the `frames * 1e9` product, then saturated back into the
    /// [`Timestamp`] range (`NONE` is one above `MAX`, so it can never be produced by accident).
    fn frames_to_nanos(&self, frames: u64) -> Option<Timestamp> {
        let inp = self.input?;
        if inp.sample_rate == 0 {
            return None;
        }
        let ns = (frames as u128 * 1_000_000_000) / inp.sample_rate as u128;
        Some(Timestamp::from_nanos(ns.min(Timestamp::MAX.0 as u128) as u64))
    }

    /// Playback-time `pts` for the current `out_frames` — the continuous grid documented at the
    /// top of this file.
    fn out_pts(&self) -> Timestamp {
        match self.frames_to_nanos(self.out_frames) {
            Some(off) => self.base.saturating_add(off),
            None => Timestamp::NONE,
        }
    }

    /// Adopt the first real input `pts` as the origin of both timelines.
    fn adopt_base(&mut self, pts: Timestamp) {
        if !self.base_locked && pts.is_some() {
            self.base = pts;
            self.base_locked = true;
        }
    }

    /// Publish source time = `base + consumed / sample_rate` (the port of the original's
    /// `consume()` → `Control::set_position`), plus the raw frame counters.
    fn publish_position(&self) {
        let Some(engine) = self.engine.as_ref() else { return };
        let Some(off) = self.frames_to_nanos(engine.consumed) else { return };
        let pos = self.base.saturating_add(off);
        self.shared.pos_nanos.store(pos.nanos().unwrap_or(0), Ordering::Relaxed);
        self.shared.in_frames.store(engine.consumed, Ordering::Relaxed);
        self.shared.out_frames.store(self.out_frames, Ordering::Relaxed);
    }

    /// True when the engine holds nothing — the precondition for the copy-free bypass forward.
    fn is_idle(&self) -> bool {
        self.carry.is_empty()
            && self.engine.as_ref().is_none_or(|e| e.buffered_frames() == 0)
    }

    fn in_bypass(&self) -> bool {
        (self.rate - 1.0).abs() < BYPASS_EPSILON
    }

    /// Feed whole input frames into the engine, draining rounds as room is needed. Chunked so
    /// the engine's fixed-capacity input buffer never has to grow.
    fn feed(&mut self, ctx: &mut Ctx, bytes: &[u8]) -> Result<(), Error> {
        let stride = self.in_stride;
        if stride == 0 {
            return Err(Error::Todo("audiostretch: input frame stride unknown"));
        }
        debug_assert_eq!(bytes.len() % stride, 0, "feed takes whole frames");
        let mut off = 0;
        while off < bytes.len() {
            // Drain first so the buffer has room, then top it up.
            self.drain(ctx, false)?;
            let taken = {
                let Some(engine) = self.engine.as_mut() else {
                    return Err(Error::Todo("audiostretch: no engine"));
                };
                if engine.free_frames() < BYPASS_CHUNK {
                    engine.compact();
                }
                let room = engine.free_frames().min(BYPASS_CHUNK);
                if room == 0 {
                    // Unreachable: a drained engine always leaves at least `2 * BYPASS_CHUNK`
                    // frames free. Reported rather than spun on.
                    return Err(Error::Todo("audiostretch: input buffer made no room"));
                }
                let take = room.min((bytes.len() - off) / stride);
                engine
                    .append(&bytes[off..off + take * stride])
                    .ok_or(Error::Todo("audiostretch: input decode failed"))?;
                take * stride
            };
            off += taken;
        }
        self.drain(ctx, false)
    }

    /// Run rounds until none can fire, flushing each one downstream.
    fn drain(&mut self, ctx: &mut Ctx, eos: bool) -> Result<(), Error> {
        let rate = self.rate;
        loop {
            let progressed = match self.engine.as_mut() {
                Some(engine) => {
                    engine.out.clear();
                    let p = engine.generate(rate, eos);
                    engine.assert_no_growth();
                    p
                }
                None => return Ok(()),
            };
            if !progressed {
                return Ok(());
            }
            self.flush_round(ctx)?;
            self.publish_position();
        }
    }

    /// Encode the round sitting in the engine's `out` and push it on the src pad.
    fn flush_round(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        let pcm = {
            let Some(engine) = self.engine.as_mut() else { return Ok(()) };
            if engine.out.is_empty() {
                return Ok(());
            }
            engine.encode_round()?;
            engine.assert_no_growth();
            // Move the staging buffer out so `self` is free for the push; it goes straight
            // back (the `audiodownmix` pattern). No allocation — `take` leaves an empty `Vec`.
            std::mem::take(&mut engine.out_pcm)
        };
        let r = self.push_pcm(ctx, &pcm);
        if let Some(engine) = self.engine.as_mut() {
            engine.out_pcm = pcm;
            engine.out_pcm.clear();
        }
        r
    }

    /// Push PCM on the src pad, chunked to the pool slot size but always on a whole-frame
    /// boundary, stamping the playback-time `pts` grid (the `mp3dec` sample-counter pattern:
    /// stamp, advance, then derive `duration` as the delta, so rounding never drifts).
    fn push_pcm(&mut self, ctx: &mut Ctx, bytes: &[u8]) -> Result<(), Error> {
        let stride = self.in_stride;
        if stride == 0 {
            return Err(Error::Todo("audiostretch: output frame stride unknown"));
        }
        let mut off = 0;
        while off < bytes.len() {
            let mut buf = ctx.alloc(SRC);
            let cap = buf.memory.capacity();
            let frames = cap / stride;
            if frames == 0 {
                return Err(Error::Todo(
                    "audiostretch: pool slot smaller than one interchannel frame",
                ));
            }
            let n = (frames * stride).min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            buf.pts = self.out_pts();
            self.out_frames += (n / stride) as u64;
            buf.duration = self.out_pts().saturating_sub(buf.pts);
            ctx.out(SRC).push(buf);
            off += n;
        }
        Ok(())
    }
}

impl Element for AudioStretch {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // A `rate=` parked before `run()` (parse string or `Pipeline::set`) is the initial
        // value; fall back to the constructor's otherwise.
        if let Some(v) = ctx.prop("rate") {
            if let Some(r) = Self::rate_from_value(ctx, v) {
                self.set_rate(r);
            }
        }
        // Link-time negotiation may already have fixed the sink format.
        self.learn_from_sink(ctx);
        self.carry.clear();
        self.publish_position();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.input.is_none() {
            self.learn_from_sink(ctx);
        }
        self.announce_output(ctx);

        while let Some(buf) = inputs.pop() {
            if self.input.is_none() {
                return Err(Error::Todo(
                    "audiostretch: PCM arrived before an audio/raw input format was known \
                     (construct with AudioStretch::with_input, or feed an announcing upstream)",
                ));
            }
            self.adopt_base(buf.pts);

            // Copy-free bypass: at 1.0 the buffer is forwarded by *move*, no DSP and no memcpy.
            // Only when nothing is held back, so ordering and frame alignment are preserved.
            if self.in_bypass() && self.is_idle() && buf.memory.data().len() % self.in_stride == 0
            {
                let frames = (buf.memory.data().len() / self.in_stride) as u64;
                if let Some(engine) = self.engine.as_mut() {
                    // Returning to stretching later must restart cleanly (as the original's
                    // bypass round does).
                    engine.copy_remaining = 0;
                    engine.frac = 0.0;
                    engine.consumed += frames;
                }
                let mut buf = buf;
                buf.pts = self.out_pts();
                self.out_frames += frames;
                buf.duration = self.out_pts().saturating_sub(buf.pts);
                ctx.out(SRC).push(buf);
                self.publish_position();
                continue;
            }

            // Whole interchannel frames only; a partial frame is completed from the head of the
            // next buffer so the channel lanes never desync. `carry` holds at most one frame and
            // was reserved at negotiation, so this never allocates.
            let mut data = buf.memory.data();
            if !self.carry.is_empty() {
                let need = self.in_stride - self.carry.len();
                let take = need.min(data.len());
                self.carry.extend_from_slice(&data[..take]);
                data = &data[take..];
                if self.carry.len() < self.in_stride {
                    continue; // still short of a frame
                }
                let frame = std::mem::take(&mut self.carry);
                let r = self.feed(ctx, &frame);
                self.carry = frame;
                self.carry.clear();
                r?;
            }
            let whole = data.len() - (data.len() % self.in_stride);
            self.feed(ctx, &data[..whole])?;
            self.carry.extend_from_slice(&data[whole..]);
            // `buf` recycles here on drop.
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // The live knob. Delivered at a batch boundary, before this pass's `process()`, so
            // the new rate governs the very next round (spec: Dynamic element properties).
            Event::PropChanged { name: "rate", value } => {
                if let Some(r) = Self::rate_from_value(ctx, *value) {
                    self.set_rate(r);
                }
            }
            // Seek (spec: flush/seek): every overlap/window/pointer is dropped, so post-seek
            // output can never blend pre-seek audio. Both timelines rebase on the seek target,
            // which is where playback time and source time coincide again.
            Event::FlushStart => {
                self.carry.clear();
                if let Some(engine) = self.engine.as_mut() {
                    engine.reset();
                }
                self.out_frames = 0;
                match ctx.seek_target() {
                    Some(t) if t.to_time.is_some() => {
                        self.base = t.to_time;
                        self.base_locked = true;
                    }
                    _ => {
                        self.base = Timestamp::ZERO;
                        self.base_locked = false;
                    }
                }
                self.publish_position();
            }
            // Drain the tail: the final overlap window is passed through rather than held back
            // (the original's short-input branch, reached there via `inner_done`). A leftover
            // `carry` is an incomplete final frame — dropped so output stays frame-aligned.
            Event::Eos => {
                self.announce_output(ctx);
                self.drain(ctx, true)?;
                self.carry.clear();
            }
            Event::FormatChange(_) => self.learn_from_sink(ctx),
            _ => {}
        }
        if self.input.is_none() {
            self.learn_from_sink(ctx);
        }
        Ok(())
    }

    // COLD: teardown; releases the engine's fixed buffers once, not per buffer.
    #[allow(clippy::disallowed_methods)]
    fn stop(&mut self, _ctx: &mut Ctx) {
        self.carry = Vec::new();
        self.engine = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_is_clamped_to_the_supported_range() {
        assert_eq!(AudioStretch::new(1.5).rate(), 1.5);
        assert_eq!(AudioStretch::new(0.01).rate(), RATE_MIN);
        assert_eq!(AudioStretch::new(99.0).rate(), RATE_MAX);
        // A non-finite request is ignored rather than poisoning the schedule.
        assert_eq!(AudioStretch::new(f32::NAN).rate(), 1.0);
    }

    #[test]
    fn position_handle_tracks_the_element() {
        let el = AudioStretch::new(2.0);
        let pos = el.position();
        assert_eq!(pos.rate(), 2.0);
        assert_eq!(pos.position(), Timestamp::ZERO);
        pos.set_position(Timestamp::from_secs(5));
        assert_eq!(pos.position(), Timestamp::from_secs(5));
    }

    #[test]
    fn engine_geometry_matches_the_original() {
        let e = Engine::new(AudioFormat::new(48_000, 2, SampleFormat::F32));
        assert_eq!(e.min_period, 48_000 / 400);
        assert_eq!(e.max_period, 48_000 / 65);
        assert_eq!(e.amdf_skip, 48_000 / 4_000);
        assert_eq!(e.window, 3 * e.max_period);
    }
}
