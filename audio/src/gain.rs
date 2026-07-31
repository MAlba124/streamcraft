//! `audiogain` — linear volume, with a declick ramp (spec: Formats — dynamic caps; Writing
//! elements; Dynamic element properties). Interleaved `audio/raw` PCM arrives on the sink pad
//! and leaves on the src pad scaled by a multiplier, in the *same* rate, channels and sample
//! format. Nothing but the amplitude changes: no resampling, no re-quantising, no channel
//! remap.
//!
//! # What it is for
//!
//! Two jobs, one element:
//!
//! * **Per-track ReplayGain.** A player that has ReplayGain metadata for a track (or an album)
//!   inserts one `audiogain` per audio branch and sets `gain` to the stored correction, so
//!   every track in a playlist plays at the same loudness instead of the next one blowing the
//!   listener's head off. The stored value is in **decibels**; this element's `gain` is a
//!   **linear multiplier**, and converting is deliberately the caller's job — `10f32.powf(db /
//!   20.0)` — because a dB knob would have to pick a floor for −∞ and this element should not
//!   own that policy. Set it once at build time, or live via the property when the user walks
//!   to the next track.
//! * **General volume.** The `gain` and `mute` properties are both live (spec: Dynamic element
//!   properties), so a volume slider and a mute button drive them directly at batch boundaries.
//!
//! # Properties
//!
//! | prop | spellings | meaning |
//! |---|---|---|
//! | `gain` | [`Value::Rat`] `gain=1/2`, [`Value::Int`] `gain=2`, [`Value::Id`] `gain=0.85` | linear multiplier, default `1.0`, clamped to `[0.0, 4.0]` |
//! | `mute` | [`Value::Int`] `mute=1` / `mute=0`, [`Value::Id`] `mute=true` / `mute=false` | silence without forgetting `gain` |
//!
//! The value universe has no float variant, so `gain` takes the three spellings
//! [`crate::stretch::AudioStretch`]'s `rate` established: an exact rational (the precise form),
//! a whole multiple, and decimal text — which is what the launch-string parser interns for
//! `audiogain gain=0.85`. `mute` takes `1`/`0`, the workspace spelling for a boolean property,
//! *and* the text forms, because `parse` turns a bare `mute=true` into an interned id and
//! silently ignoring it would be a trap. Muting does not clear `gain`: unmuting returns to the
//! value that was set.
//!
//! # The declick ramp
//!
//! A gain change applied between one buffer and the next is a **step** in the waveform, and a
//! step is broadband — that is the click you hear when a volume key or a mute button lands mid
//! note. So a change is interpolated linearly over [`RAMP_MS`] ms of *output* instead of being
//! applied at once, which bounds the slope and puts the artefact below audibility. This is a
//! deliberate divergence from rodio, whose `Sink::set_volume` steps instantly. The ramp is
//! per-frame, not per-sample: every channel of an interchannel frame is scaled by the same
//! value, or the ramp itself would become a moving inter-channel imbalance.
//!
//! At a **discontinuity** there is no previous sample to be continuous with, so the ramp is
//! *snapped* rather than run: on `start()` (a pipeline built at `gain=0.5` must not fade in
//! from unity) and on `Event::FlushStart` (post-seek audio starts at the gain in force, and a
//! half-finished ramp from before the seek is meaningless).
//!
//! # Arithmetic
//!
//! * **Integer formats saturate.** The product is formed in `f64` — wide enough to hold a
//!   31-bit `S32` sample times 4.0 exactly, which neither `i32` nor `f32` is — then rounded and
//!   clamped to the format's full scale. Nothing ever wraps, so a loud sample at `gain=4.0`
//!   comes out pinned at full scale, not inverted.
//! * **`F32` is not clamped.** Float PCM is the pipeline's headroom format: a downstream
//!   `audioconvert` clamps when it narrows to an integer format, and clamping here as well
//!   would throw away headroom an intervening stage might have brought back. `NaN`/`inf` in a
//!   float stream therefore pass through as `NaN`/`inf`; they cannot wedge the element, since
//!   the ramp is driven by frame counts and never by sample values.
//!
//! # Unity fast path
//!
//! At `gain == 1.0`, unmuted, with no ramp in flight, an incoming buffer is forwarded by
//! **move** — no pool slot, no memcpy, no arithmetic (the `audiostretch` bypass idiom). So
//! leaving an `audiogain` permanently in a playback graph costs a comparison per buffer.

use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, PropDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{Constraint, ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

use crate::format::{
    negotiated_audio_format, AudioFormat, SampleFormat, FAMILY, FIELD_CHANNELS, FIELD_RATE,
    FIELD_SAMPLE,
};

const SINK: PadId = PadId(0);
const SRC: PadId = PadId(1);

/// Gain clamp. Zero is silence; 4.0 (+12 dB) is as much boost as makes sense before the caller
/// should be reaching for a limiter instead.
pub const GAIN_MIN: f32 = 0.0;
pub const GAIN_MAX: f32 = 4.0;

/// Declick ramp length in milliseconds of output — long enough to keep the slope inaudible,
/// short enough that a volume key still feels instant.
pub const RAMP_MS: usize = 5;

// --- Pads / props / descriptor --------------------------------------------------------

// Both pads speak `audio/raw` (any rate/channels/sample) plus the `bytes` escape, mirroring
// `audioconvert`/`audiostretch`. Gain never changes the format, so the src pad announces
// exactly what the sink negotiated; it is `dynamic` only because that announcement happens at
// runtime. Every sample-format name is offered so the `sample` value is interned for the
// announcement to resolve against.
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

/// Both **live** (spec: Dynamic element properties): a set lands at the next batch boundary and
/// starts a declick ramp — see the module docs for the accepted spellings.
static PROPS: [PropDesc; 2] = [
    PropDesc { name: "gain", allowed: Constraint::Any, live: true },
    PropDesc { name: "mute", allowed: Constraint::Any, live: true },
];

// COLD: the `make_default` factory boxes one element at construction, not per buffer.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "audiogain",
    pads: &PADS,
    props: &PROPS,
    // Passive: a pure transform that inlines into the upstream group (spec: Scheduling).
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        // Nothing is held back: at most one partial interchannel frame, which is not a whole
        // frame of delay.
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Default unity (the fast path); override via the `gain` prop at parse time.
    make_default: Some(|| Box::new(AudioGain::new(1.0))),
};

// --- Sample arithmetic ----------------------------------------------------------------

/// `sample * gain`, rounded to nearest and clamped to `[lo, hi]` — **saturating, never
/// wrapping**.
///
/// The product is formed in `f64` on purpose. An `S32` sample carries 31 significant bits,
/// which `f32`'s 24-bit mantissa would quantise before the clamp ever ran, and full-scale
/// `i32` times 4.0 overflows `i32` outright; `f64` represents both exactly.
#[inline]
fn saturate(sample: i32, gain: f32, lo: i32, hi: i32) -> i32 {
    let v = (sample as f64 * gain as f64).round();
    // Integer PCM cannot carry a NaN and a non-finite gain is rejected at the property, so this
    // is belt and braces: `f64::clamp` propagates a NaN input and `NaN as i32` saturates to 0 —
    // silence, not a panic.
    v.clamp(lo as f64, hi as f64) as i32
}

/// Scale one sample, in place. `s` is exactly one sample's bytes in `fmt`'s wire layout —
/// little-endian throughout, matching [`crate::convert`] and [`crate::format::AudioFrameRef`].
#[inline]
fn scale_sample(fmt: SampleFormat, gain: f32, s: &mut [u8]) {
    match fmt {
        // Deliberately unclamped — see the module docs on headroom.
        SampleFormat::F32 => {
            let v = f32::from_le_bytes([s[0], s[1], s[2], s[3]]) * gain;
            s[..4].copy_from_slice(&v.to_le_bytes());
        }
        // U8 is unsigned-biased-by-128; unbias, scale, re-bias.
        SampleFormat::U8 => {
            s[0] = (saturate(s[0] as i32 - 128, gain, -128, 127) + 128) as u8;
        }
        SampleFormat::S16 => {
            let v = saturate(
                i16::from_le_bytes([s[0], s[1]]) as i32,
                gain,
                i16::MIN as i32,
                i16::MAX as i32,
            );
            s[..2].copy_from_slice(&(v as i16).to_le_bytes());
        }
        // 24-bit two's complement in 3 bytes: sign-extend by shifting up then arithmetic-down
        // (the `AudioFrameRef::sample_i32` idiom), then write back the low 3 bytes.
        SampleFormat::S24 => {
            let raw = s[0] as i32 | (s[1] as i32) << 8 | (s[2] as i32) << 16;
            let v = saturate((raw << 8) >> 8, gain, -(1 << 23), (1 << 23) - 1);
            s[..3].copy_from_slice(&v.to_le_bytes()[..3]);
        }
        SampleFormat::S32 => {
            let v = saturate(i32::from_le_bytes([s[0], s[1], s[2], s[3]]), gain, i32::MIN, i32::MAX);
            s[..4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

/// Copy `dst.len()` bytes from logical offset `off` of the concatenation `head ++ body`.
///
/// The two slices are the carried partial frame completed from this buffer's head, and the
/// whole frames behind it. Reading them as one logical stream is what lets the output be
/// written **straight into a pool slot** — no staging `Vec`, so `process()` allocates nothing.
#[inline]
fn copy_concat(head: &[u8], body: &[u8], off: usize, dst: &mut [u8]) {
    let mut written = 0;
    if off < head.len() {
        written = (head.len() - off).min(dst.len());
        dst[..written].copy_from_slice(&head[off..off + written]);
    }
    if written < dst.len() {
        let start = (off + written) - head.len();
        let n = dst.len() - written;
        dst[written..].copy_from_slice(&body[start..start + n]);
    }
}

// --- The element ----------------------------------------------------------------------

/// Linear volume with a declick ramp. Construct with [`AudioGain::new`].
pub struct AudioGain {
    /// Target linear gain, already clamped to `[GAIN_MIN, GAIN_MAX]`.
    gain: f32,
    /// Mute overrides the target with zero while set. `gain` is remembered, so unmuting ramps
    /// back to it.
    mute: bool,
    /// The multiplier applied to the *next* output frame. Walks toward
    /// [`effective_gain`](AudioGain::effective_gain) one ramp step per frame.
    current: f32,
    /// Output frames left in the declick ramp, and the per-frame increment.
    ramp_remaining: usize,
    ramp_step: f32,
    /// Input format learned from negotiated caps. `None` until known.
    input: Option<AudioFormat>,
    /// Whether the (unchanged) output format has been announced downstream.
    announced: bool,
    /// Input interchannel-frame stride in bytes; `0` until known.
    in_stride: usize,
    /// Carry for a partial input frame straddling two buffers, so channel lanes never desync
    /// and the ramp stays whole-frame (mirrors `audioconvert`). Capacity is one frame —
    /// filling it never allocates.
    carry: Vec<u8>,
    /// The mid-stream format this element has already complained about, so the warning is
    /// posted once per *distinct* change — see [`warn_if_relatched`](AudioGain::warn_if_relatched).
    warned_relatch: Option<AudioFormat>,
}

impl AudioGain {
    /// A gain stage at `gain` (1.0 = the copy-free unity path), discovering its format at
    /// runtime from the upstream's negotiated `audio/raw` caps (spec: dynamic caps — consumer
    /// side).
    ///
    /// `gain` is a **linear** multiplier, clamped to `[GAIN_MIN, GAIN_MAX]`. For ReplayGain,
    /// convert the stored decibels first: `AudioGain::new(10f32.powf(db / 20.0))`.
    // COLD: constructs the element once. `carry` is a one-frame cross-buffer accumulator, not
    // per-buffer scratch.
    #[allow(clippy::disallowed_methods)]
    pub fn new(gain: f32) -> Self {
        let mut el = Self {
            gain: 1.0,
            mute: false,
            current: 1.0,
            ramp_remaining: 0,
            ramp_step: 0.0,
            input: None,
            announced: false,
            in_stride: 0,
            carry: Vec::new(),
            warned_relatch: None,
        };
        el.set_gain(gain);
        // Nothing has been emitted, so there is no waveform to be continuous with.
        el.snap();
        el
    }

    /// A gain stage with the input format pinned at construction (statically-wired / testable
    /// form). Still announces its format downstream.
    pub fn with_input(input: AudioFormat, gain: f32) -> Self {
        let mut g = Self::new(gain);
        g.set_input(input);
        g.snap();
        g
    }

    /// The target gain currently configured (post-clamp). Not the value being applied while a
    /// ramp is in flight — see [`applied_gain`](Self::applied_gain).
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Whether the element is muted.
    pub fn muted(&self) -> bool {
        self.mute
    }

    /// The multiplier the next output frame will be scaled by — the ramp's current position.
    pub fn applied_gain(&self) -> f32 {
        self.current
    }

    /// The input format, once known.
    pub fn input_format(&self) -> Option<AudioFormat> {
        self.input
    }

    /// The gain the ramp is walking toward: the target, or zero while muted.
    fn effective_gain(&self) -> f32 {
        if self.mute {
            0.0
        } else {
            self.gain
        }
    }

    /// True when the element is a no-op *right now*: unity target, unmuted, no ramp in flight.
    /// The precondition for forwarding a buffer by move.
    fn is_unity(&self) -> bool {
        self.ramp_remaining == 0 && !self.mute && self.gain == 1.0 && self.current == 1.0
    }

    /// Apply a new target gain, clamped, and start the declick ramp toward it. A non-finite
    /// request is **ignored** rather than poisoning the multiplier — a NaN gain would turn the
    /// whole stream into NaN with no way back.
    fn set_gain(&mut self, gain: f32) {
        if !gain.is_finite() {
            return;
        }
        self.gain = gain.clamp(GAIN_MIN, GAIN_MAX);
        self.retarget();
    }

    /// Set the mute flag and ramp toward the new effective gain.
    fn set_mute(&mut self, mute: bool) {
        if mute == self.mute {
            return;
        }
        self.mute = mute;
        self.retarget();
    }

    /// Ramp length in output frames — [`RAMP_MS`] at the negotiated rate, at least one frame so
    /// a ramp always makes progress. An unknown rate degrades to an instant change (nothing has
    /// been emitted yet, so there is nothing to click).
    fn ramp_frames(&self) -> usize {
        match self.input {
            Some(inp) => ((inp.sample_rate as usize * RAMP_MS) / 1000).max(1),
            None => 1,
        }
    }

    /// Begin interpolating from the value in force toward the new effective gain (see the
    /// module docs on why a step is not acceptable).
    fn retarget(&mut self) {
        let target = self.effective_gain();
        if target == self.current {
            self.ramp_remaining = 0;
            self.ramp_step = 0.0;
            return;
        }
        let frames = self.ramp_frames();
        self.ramp_remaining = frames;
        self.ramp_step = (target - self.current) / frames as f32;
    }

    /// Jump straight to the effective gain with no ramp — the right thing at a discontinuity,
    /// where there is no previous sample to be continuous with.
    fn snap(&mut self) {
        self.current = self.effective_gain();
        self.ramp_remaining = 0;
        self.ramp_step = 0.0;
    }

    /// The gain for the frame about to be written, advancing the ramp by one frame.
    #[inline]
    fn advance(&mut self) -> f32 {
        let g = self.current;
        if self.ramp_remaining > 0 {
            self.ramp_remaining -= 1;
            // Snap exactly on the last step: accumulating `ramp_step` would leave the steady
            // state a few ULPs away from the requested gain forever.
            self.current = if self.ramp_remaining == 0 {
                self.effective_gain()
            } else {
                self.current + self.ramp_step
            };
        }
        g
    }

    /// Decode a `gain` value. The value universe has no float, so an exact rational, a whole
    /// multiple, and decimal text (what `parse("audiogain gain=0.85")` interns) are all
    /// accepted — the spellings `audiostretch`'s `rate` established.
    fn gain_from_value(ctx: &Ctx, v: Value) -> Option<f32> {
        match v {
            Value::Int(n) => Some(n as f32),
            Value::Rat(n, d) if d != 0 => Some(n as f32 / d as f32),
            Value::Rat(_, _) => None,
            Value::Id(id) => ctx.value_name(id).and_then(|s| s.parse::<f32>().ok()),
        }
    }

    /// Decode a `mute` value. `1`/`0` is the workspace spelling for a boolean property; the
    /// launch-string parser interns a bare `mute=true` as an id instead, so the text forms are
    /// accepted too rather than being silently dropped.
    fn mute_from_value(ctx: &Ctx, v: Value) -> Option<bool> {
        match v {
            Value::Int(n) => Some(n != 0),
            Value::Id(id) => match ctx.value_name(id)? {
                "1" | "true" | "on" | "yes" => Some(true),
                "0" | "false" | "off" | "no" => Some(false),
                _ => None,
            },
            Value::Rat(_, _) => None,
        }
    }

    /// Record the input format and derive the frame stride. A genuinely different format
    /// re-arms the downstream announcement.
    fn set_input(&mut self, input: AudioFormat) {
        if self.input == Some(input) {
            return;
        }
        self.in_stride = input.frame_stride();
        self.input = Some(input);
        self.announced = false;
        self.carry.clear();
        self.carry.reserve(self.in_stride.max(1));
    }

    /// Infer the input format from the sink's negotiated `audio/raw` caps (spec: dynamic caps —
    /// consumer side). A pinned or already-learned format wins and short-circuits.
    fn learn_from_sink(&mut self, ctx: &Ctx) {
        if self.input.is_some() {
            return;
        }
        if let Some(fmt) = negotiated_audio_format(ctx, SINK) {
            self.set_input(fmt);
        }
    }

    /// Announce the output format on the src pad — identical to the input, since gain changes
    /// only amplitudes (spec: dynamic caps — producer side).
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

    /// Complain on the bus when the sink's negotiated format has moved away from the one this
    /// element latched — the same contract the other `audio/raw` glue elements hold. The stride
    /// and sample format are pinned at the first negotiation, so a mid-stream change would be
    /// read as the old layout: whole-frame boundaries land in the wrong place and every sample
    /// is decoded at the wrong width. Following the change is gapless-playback work with its
    /// own design and is not attempted here.
    ///
    /// Posted once per *distinct* format, and only from the event path, so it can never fire
    /// per buffer.
    fn warn_if_relatched(&mut self, ctx: &mut Ctx) {
        let Some(latched) = self.input else { return };
        let Some(offered) = negotiated_audio_format(ctx, SINK) else { return };
        if offered == latched || self.warned_relatch == Some(offered) {
            return;
        }
        self.warned_relatch = Some(offered);
        let element = ctx.element();
        ctx.post(BusMessage::Warning {
            element,
            error: Error::Element {
                element,
                message: format!(
                    "audiogain: ignoring a mid-stream input format change ({latched} -> \
                     {offered}); this element latches its input format at the first negotiation \
                     and keeps reading every later buffer as the latched one. Rebuild the branch \
                     (or insert a fresh audiogain) to follow the change."
                ),
            },
        });
    }

    /// Nanoseconds spanned by `frames` at the negotiated rate. `u128` intermediate so a
    /// multi-day stream cannot overflow the `frames * 1e9` product, then saturated back into
    /// the [`Timestamp`] range.
    fn frames_to_ts(&self, frames: u64) -> Timestamp {
        let Some(inp) = self.input else { return Timestamp::ZERO };
        if inp.sample_rate == 0 {
            return Timestamp::ZERO;
        }
        let ns = (frames as u128 * 1_000_000_000) / inp.sample_rate as u128;
        Timestamp::from_nanos(ns.min(Timestamp::MAX.0 as u128) as u64)
    }

    /// Multiply `pcm` (whole interchannel frames, in place) by the ramping gain — one gain
    /// value per frame, so the channels of a frame never drift apart.
    fn apply(&mut self, pcm: &mut [u8]) {
        let Some(inp) = self.input else { return };
        let bps = inp.format.bytes();
        for frame in pcm.chunks_exact_mut(self.in_stride) {
            let g = self.advance();
            // Unity is bit-exact by skipping, not by multiplying: `NaN * 1.0` may re-quantise a
            // signalling payload, and the steady state of an `audiogain` left in a graph at
            // unity should touch nothing at all.
            if g == 1.0 {
                continue;
            }
            for s in frame.chunks_exact_mut(bps) {
                scale_sample(inp.format, g, s);
            }
        }
    }

    /// Scale `head ++ body` (whole interchannel frames) into pool slots and push them on the
    /// src pad, chunked to the slot size but always on a whole-frame boundary.
    ///
    /// The bytes are copied straight into the slot and scaled **in place**, so no staging
    /// buffer exists and `process()` performs no heap allocation (spec: performance — no
    /// steady-state heap traffic). `base` is the incoming buffer's `pts`; outputs carry it
    /// forward on the frame grid so a clock-driven sink downstream keeps its schedule.
    fn emit_gained(
        &mut self,
        ctx: &mut Ctx,
        head: &[u8],
        body: &[u8],
        base: Timestamp,
    ) -> Result<(), Error> {
        let stride = self.in_stride;
        if stride == 0 {
            return Err(Error::Todo("audiogain: input frame stride unknown"));
        }
        let total = head.len() + body.len();
        debug_assert_eq!(total % stride, 0, "emit_gained takes whole frames");
        let (mut off, mut done) = (0usize, 0u64);
        while off < total {
            let mut buf = ctx.alloc(SRC);
            let frames = buf.memory.capacity() / stride;
            if frames == 0 {
                return Err(Error::Todo(
                    "audiogain: pool slot smaller than one interchannel frame",
                ));
            }
            let n = (frames * stride).min(total - off);
            let dst = &mut buf.memory.as_mut_full()[..n];
            copy_concat(head, body, off, dst);
            self.apply(dst);
            buf.memory.set_len(n);
            // The carried frame belongs to the previous input buffer, so a ragged stream's pts
            // can be up to one frame early. That is below the resolution of any sink's
            // scheduling and keeps the grid gapless, which matters more.
            buf.pts = if base.is_some() {
                base.saturating_add(self.frames_to_ts(done))
            } else {
                Timestamp::NONE
            };
            buf.duration = self.frames_to_ts((n / stride) as u64);
            ctx.out(SRC).push(buf);
            off += n;
            done += (n / stride) as u64;
        }
        Ok(())
    }
}

impl Element for AudioGain {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // A `gain=` / `mute=` parked before `run()` (parse string or `Pipeline::set`) is the
        // initial value; fall back to the constructor's otherwise.
        if let Some(v) = ctx.prop("gain") {
            if let Some(g) = Self::gain_from_value(ctx, v) {
                self.set_gain(g);
            }
        }
        if let Some(v) = ctx.prop("mute") {
            if let Some(m) = Self::mute_from_value(ctx, v) {
                self.set_mute(m);
            }
        }
        // Link-time negotiation may already have fixed the sink format.
        self.learn_from_sink(ctx);
        self.carry.clear();
        // Start *at* the configured gain rather than fading up to it from the default.
        self.snap();
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
                    "audiogain: PCM arrived before an audio/raw input format was known \
                     (construct with AudioGain::with_input, or feed an announcing upstream)",
                ));
            }
            let stride = self.in_stride;

            // Unity fast path: the buffer is forwarded by *move*, with no pool slot and no
            // memcpy. Only when nothing is held back and the buffer is whole frames, so
            // ordering and frame alignment are preserved.
            if self.is_unity() && self.carry.is_empty() && buf.memory.data().len() % stride == 0 {
                ctx.out(SRC).push(buf);
                continue;
            }

            // Whole interchannel frames only: the ramp advances once per frame, so a frame
            // split across two buffers has to be reassembled before either half is scaled. The
            // carry holds at most one frame and was reserved at negotiation, so this never
            // allocates. `take` leaves an empty `Vec` behind, freeing `self` for `emit_gained`.
            let mut data = buf.memory.data();
            let mut frame = std::mem::take(&mut self.carry);
            let had_carry = !frame.is_empty();
            if had_carry {
                let take = (stride - frame.len()).min(data.len());
                frame.extend_from_slice(&data[..take]);
                data = &data[take..];
            }
            let complete = frame.len() == stride;
            let whole = data.len() - (data.len() % stride);
            let head: &[u8] = if complete { &frame } else { &[] };
            let r = self.emit_gained(ctx, head, &data[..whole], buf.pts);
            // Put the carry back. `frame` first, so its one-frame capacity is restored either
            // way; then, unless it is still an incomplete frame in its own right, overwrite it
            // with this buffer's ragged tail.
            self.carry = frame;
            if complete || !had_carry {
                self.carry.clear();
                self.carry.extend_from_slice(&data[whole..]);
            }
            r?;
            // `buf` recycles here on drop.
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // The live knobs. Delivered at a batch boundary, before this pass's `process()`, so
            // the ramp starts on the very next frame (spec: Dynamic element properties).
            Event::PropChanged { name: "gain", value } => {
                if let Some(g) = Self::gain_from_value(ctx, *value) {
                    self.set_gain(g);
                }
            }
            Event::PropChanged { name: "mute", value } => {
                if let Some(m) = Self::mute_from_value(ctx, *value) {
                    self.set_mute(m);
                }
            }
            // Seek (spec: flush/seek). The carried partial frame is *pre-seek* audio: keeping
            // it would prepend those bytes to the first post-seek buffer, shifting every later
            // sample by part of a frame and permanently rotating the channel lanes. The ramp
            // goes too — it was interpolating toward a target across audio that is no longer
            // adjacent to what comes next, so it is snapped instead (there is no waveform on
            // the far side of a seek to be continuous with). The configured gain, the mute
            // flag and the learned format all survive: a seek moves the read head, it does not
            // change the volume or what the upstream is sending.
            Event::FlushStart => {
                self.carry.clear();
                self.snap();
            }
            // A leftover carry is an incomplete final frame — dropped so output stays
            // frame-aligned.
            Event::Eos => {
                self.announce_output(ctx);
                self.carry.clear();
            }
            Event::FormatChange(_) => {
                self.learn_from_sink(ctx);
                self.warn_if_relatched(ctx);
            }
            _ => {}
        }
        if self.input.is_none() {
            self.learn_from_sink(ctx);
        }
        Ok(())
    }

    // COLD: teardown; releases the carry once, not per buffer.
    #[allow(clippy::disallowed_methods)]
    fn stop(&mut self, _ctx: &mut Ctx) {
        self.carry = Vec::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_is_clamped_and_non_finite_requests_are_ignored() {
        assert_eq!(AudioGain::new(0.5).gain(), 0.5);
        assert_eq!(AudioGain::new(-1.0).gain(), GAIN_MIN);
        assert_eq!(AudioGain::new(99.0).gain(), GAIN_MAX);
        // A NaN gain would turn the stream into NaN with no way back; it is dropped instead.
        assert_eq!(AudioGain::new(f32::NAN).gain(), 1.0);
        assert_eq!(AudioGain::new(f32::INFINITY).gain(), 1.0);
    }

    #[test]
    fn a_fresh_element_starts_at_its_configured_gain() {
        // No fade-in from unity: there is no previous audio to be continuous with.
        let g = AudioGain::new(0.25);
        assert_eq!(g.applied_gain(), 0.25);
        assert!(!g.is_unity());
        assert!(AudioGain::new(1.0).is_unity(), "unity is the copy-free fast path");
    }

    #[test]
    fn integer_scaling_saturates_instead_of_wrapping() {
        // Full scale times 4 must pin, not invert.
        assert_eq!(saturate(i16::MAX as i32, 4.0, i16::MIN as i32, i16::MAX as i32), 32767);
        assert_eq!(saturate(i16::MIN as i32, 4.0, i16::MIN as i32, i16::MAX as i32), -32768);
        assert_eq!(saturate(i32::MAX, 4.0, i32::MIN, i32::MAX), i32::MAX);
        assert_eq!(saturate(i32::MIN, 2.0, i32::MIN, i32::MAX), i32::MIN);
        // …and an ordinary value is just rounded.
        assert_eq!(saturate(1000, 0.5, i16::MIN as i32, i16::MAX as i32), 500);
        assert_eq!(saturate(1001, 0.5, i16::MIN as i32, i16::MAX as i32), 501); // round half away
    }

    #[test]
    fn s24_scaling_round_trips_through_the_three_byte_layout() {
        // -1 is 0xFFFFFF; halving it must stay negative and stay in range.
        let mut s = [0xFFu8, 0xFF, 0xFF];
        scale_sample(SampleFormat::S24, 0.5, &mut s);
        let raw = s[0] as i32 | (s[1] as i32) << 8 | (s[2] as i32) << 16;
        assert_eq!((raw << 8) >> 8, -1, "round-half-away keeps -0.5 at -1");
        // Full-scale positive at 4x pins at the 24-bit maximum.
        let mut s = [0xFFu8, 0xFF, 0x7F]; // 8388607
        scale_sample(SampleFormat::S24, 4.0, &mut s);
        assert_eq!(s, [0xFF, 0xFF, 0x7F]);
    }

    #[test]
    fn u8_scaling_respects_the_128_bias() {
        // 128 is silence and must stay silence at any gain.
        let mut s = [128u8];
        scale_sample(SampleFormat::U8, 0.5, &mut s);
        assert_eq!(s, [128]);
        // 255 (= +127) at 4x pins at +127.
        let mut s = [255u8];
        scale_sample(SampleFormat::U8, 4.0, &mut s);
        assert_eq!(s, [255]);
    }

    #[test]
    fn ramp_lands_exactly_on_the_target() {
        // Accumulating the step would leave the steady state a few ULPs off forever.
        let mut g = AudioGain::with_input(AudioFormat::new(48_000, 2, SampleFormat::F32), 1.0);
        g.set_gain(0.5);
        let frames = g.ramp_frames();
        assert_eq!(frames, 240, "5 ms at 48 kHz");
        for _ in 0..frames {
            g.advance();
        }
        assert_eq!(g.applied_gain(), 0.5, "the last step snaps exactly onto the target");
        assert_eq!(g.ramp_remaining, 0);
    }

    #[test]
    fn copy_concat_reads_the_two_slices_as_one_stream() {
        let (head, body) = ([1u8, 2, 3, 4], [5u8, 6, 7, 8, 9, 10, 11, 12]);
        let mut dst = [0u8; 6];
        copy_concat(&head, &body, 0, &mut dst);
        assert_eq!(dst, [1, 2, 3, 4, 5, 6]);
        copy_concat(&head, &body, 6, &mut dst);
        assert_eq!(dst, [7, 8, 9, 10, 11, 12]);
        // Wholly past the head.
        let mut dst = [0u8; 4];
        copy_concat(&head, &body, 4, &mut dst);
        assert_eq!(dst, [5, 6, 7, 8]);
    }
}
