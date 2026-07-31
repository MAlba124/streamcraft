//! `audioresample` — the profluens element wrapping the [`crate::resample`] polyphase
//! resampler (spec: Formats — sample-rate conversion, deliberately split out of `audioconvert`).
//! Interleaved `audio/raw` PCM arrives on the sink pad at the negotiated input rate; the same
//! audio, band-limited and re-sampled to a construction-time **target rate**, leaves on the src
//! pad — same channel count and sample format, only the rate changes.
//!
//! A passive transform (like [`crate::convert_element::AudioConvert`] and `wavparse`): it inlines
//! into the upstream group and never blocks. Like `audioconvert` it is **both a consumer and a
//! producer of dynamic caps**:
//!
//! * It learns its **input** format (rate, channels, sample) **by name** from the negotiated
//!   caps on the sink pad — `ctx.field_id("rate")` / `"channels"` / `"sample"`, reversing the
//!   `sample` id via `ctx.value_name` (the [`crate::convert_element::AudioConvert`] / `flacdec` /
//!   `pipewireaudiosink` pattern). No out-of-band hint: [`AudioResample::new`] infers everything
//!   from caps.
//! * It **announces** its output `audio/raw` on the src pad the moment the input is known — same
//!   channels and sample format, `rate` = the target — via [`Ctx::announce_format`], so its own
//!   downstream re-fixates (spec: dynamic caps — producer side).
//!
//! # How it resamples
//!
//! The DSP works in `f32` (see [`crate::resample`]). Each incoming buffer is: decoded to
//! interleaved `f32` (reusing the tested [`crate::convert`] library), de-interleaved into planar
//! per-channel spans, run through one streaming [`ChannelResampler`] per channel (so filter state
//! — the delay line and phase — carries across buffers), re-interleaved, then re-encoded to the
//! input sample format. Only whole interchannel frames are processed; a partial trailing frame is
//! carried to the next buffer (mirrors `audioconvert`/`flacenc`). If the input rate already equals
//! the target, PCM passes through byte-for-byte (the announcement still fires).

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

use crate::convert::convert_interleaved_vec;
use crate::format::{
    negotiated_audio_format, AudioFormat, SampleFormat, FAMILY, FIELD_CHANNELS, FIELD_RATE,
    FIELD_SAMPLE,
};
use crate::resample::{ChannelResampler, PolyphaseFilter};

const SINK: PadId = PadId(0);
const SRC: PadId = PadId(1);

/// Filter length knob: `half_taps` input samples reach each side of the interpolation point.
/// 32 gives a long, sharp Kaiser-windowed-sinc (good stop-band) while staying cheap; the element
/// is fixed-quality for now (a `quality` prop is a natural follow-up).
const HALF_TAPS: usize = 32;
/// Kaiser β ≈ 9 → roughly −85 dB side-lobes, a good general-purpose audio anti-alias.
const KAISER_BETA: f64 = 9.0;

// Both pads prefer a broad `audio/raw` and also accept raw `bytes` (to bridge `bytes`-typed
// transports exactly as `audioconvert`/`wavparse` do). `audio/raw` is listed first so it wins
// whenever the peer speaks it. Every known sample format is offered so the `sample` field name is
// interned and negotiation matches any raw-audio peer.
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
static OFFERS: [OfferDesc; 2] = [
    OfferDesc { family: FAMILY, fields: &FIELDS },
    OfferDesc::any("bytes"),
];

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
        dynamic: true, // output rate is announced at runtime
        validate: None,
    },
];

/// The target output rate in Hz (spec: Plugins — `parse("audioresample rate=48000")`).
/// Structural (`live: false`): the engine is built around it in `start()`. The input
/// rate is discovered from caps; this is only the *output* rate.
static PROPS: [PropDesc; 1] = [PropDesc {
    name: "rate",
    allowed: Constraint::Any,
    live: false,
}];

// COLD: the `make_default` factory boxes one element at construction, not per buffer.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "audioresample",
    pads: &PADS,
    props: &PROPS,
    // Passive: a pure transform that inlines into the upstream group (spec: Scheduling).
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        // The FIR imposes a group delay; a precise figure waits on the clock (Milestone 1 has
        // none). Left zero like the other passive audio transforms for now.
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Default target 48 kHz; override via the `rate` prop at parse time.
    make_default: Some(|| Box::new(AudioResample::new(48_000))),
};

/// Per-channel streaming resamplers plus the shared input geometry, built once the input format
/// is known.
struct Engine {
    /// One streaming resampler per channel (each carries its own delay line + phase).
    channels: Vec<ChannelResampler>,
    /// Input sample format (outputs are re-encoded to the same format).
    sample: SampleFormat,
    /// Channel count.
    nch: usize,
}

impl Engine {
    /// Drop the audio held in every channel's delay line (spec: flush/seek) — see
    /// [`ChannelResampler::reset`] for why a streaming FIR must forget across a seek. The
    /// filter design and all buffer capacities survive, so this allocates nothing.
    fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.reset();
        }
    }
}

/// Resamples interleaved PCM to a fixed target sample **rate**, preserving channels and sample
/// format. Construct with [`AudioResample::new`] (input format inferred from caps).
pub struct AudioResample {
    /// The output sample rate every produced buffer is written at — set at construction.
    target_rate: u32,
    /// Input format, learned at runtime from negotiated caps. `None` until known.
    input: Option<AudioFormat>,
    /// The resampling engine, built once the input format is known. `None` for a pass-through
    /// (input rate == target) or before the format is known.
    engine: Option<Engine>,
    /// Whether the output format has been announced downstream yet (announce once).
    announced: bool,
    /// Input frame stride in bytes (`channels * sample.bytes()`); `0` until known.
    in_stride: usize,
    /// Carry for a partial input frame straddling two buffers (only whole interchannel frames are
    /// resampled — channel lanes must never desync).
    carry: Vec<u8>,
    /// The `(rate, channels)` recovered from caps when the sample format is not yet resolvable —
    /// so [`output_format`](AudioResample::output_format) reports the negotiated stream early.
    partial_in: Option<(u32, u16)>,
    /// The mid-stream format this element has already complained about, so the warning is
    /// posted once per *distinct* change — see [`warn_if_relatched`](AudioResample::warn_if_relatched).
    warned_relatch: Option<AudioFormat>,
}

impl AudioResample {
    /// A resampler to `target_rate` Hz, discovering its input format at runtime from the
    /// upstream's negotiated `audio/raw` caps (spec: dynamic caps — consumer side). The whole
    /// input [`AudioFormat`] is read by name off the negotiated [`FixedFormat`] — same pattern as
    /// [`AudioConvert::new`](crate::convert_element::AudioConvert::new).
    // COLD: constructs the element once; `carry` is a reused cross-buffer accumulator, not per-buffer scratch.
    #[allow(clippy::disallowed_methods)]
    pub fn new(target_rate: u32) -> Self {
        assert!(target_rate > 0, "audioresample: target rate must be positive");
        Self {
            target_rate,
            input: None,
            engine: None,
            announced: false,
            in_stride: 0,
            carry: Vec::new(),
            partial_in: None,
            warned_relatch: None,
        }
    }

    /// A resampler with the input format pinned at construction (statically-wired / testable
    /// form). Still announces the output format downstream.
    pub fn with_input(input: AudioFormat, target_rate: u32) -> Self {
        let mut r = Self::new(target_rate);
        r.set_input(input);
        r
    }

    /// The target output sample rate.
    pub fn target_rate(&self) -> u32 {
        self.target_rate
    }

    /// The input format, once known (from negotiation or [`AudioResample::with_input`]).
    pub fn input_format(&self) -> Option<AudioFormat> {
        self.input
    }

    /// The output format this element produces once rate/channels are known: the input's channels
    /// and sample format, `sample_rate` = the target. Available from a pinned/inferred input or,
    /// early, from the rate/channels recovered from caps (sample format then defaults harmlessly
    /// to the eventual value — callers use this only for the announced rate/channels).
    pub fn output_format(&self) -> Option<AudioFormat> {
        if let Some(f) = self.input {
            return Some(AudioFormat::new(self.target_rate, f.channels, f.format));
        }
        self.partial_in
            .map(|(_, ch)| AudioFormat::new(self.target_rate, ch, SampleFormat::F32))
    }

    /// Record the input format, (re)build the resampling engine, and derive the stride. A
    /// genuinely new format re-arms the announcement and resets filter state.
    fn set_input(&mut self, input: AudioFormat) {
        if self.input == Some(input) {
            return;
        }
        self.in_stride = input.frame_stride();
        self.input = Some(input);
        self.announced = false;
        self.carry.clear();

        // Pass-through when the rate already matches: no engine, bytes forwarded verbatim.
        if input.sample_rate == self.target_rate {
            self.engine = None;
            return;
        }
        // One shared filter design; clone its (small) coefficient tables into one streaming
        // resampler per channel so each carries independent delay-line/phase state.
        let filter = PolyphaseFilter::design(input.sample_rate, self.target_rate, HALF_TAPS, KAISER_BETA);
        let nch = input.channels as usize;
        let channels = (0..nch).map(|_| ChannelResampler::new(filter.clone())).collect();
        self.engine = Some(Engine { channels, sample: input.format, nch });
    }

    /// Infer the input [`AudioFormat`] from the sink's negotiated `audio/raw` caps, reading every
    /// field by name (spec: Formats — an element acts on its caps). Mirrors
    /// [`AudioConvert`](crate::convert_element::AudioConvert)'s consumer path.
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

        match sample {
            Some(fmt) => self.set_input(AudioFormat::new(rate as u32, channels as u16, fmt)),
            None => self.partial_in = Some((rate as u32, channels as u16)),
        }
    }

    /// Complain on the bus when the sink's negotiated format has moved away from the one this
    /// element latched.
    ///
    /// [`learn_from_sink`](Self::learn_from_sink) pins the input format at the *first*
    /// negotiation and short-circuits ever after, so a mid-stream change is **ignored**: the
    /// filter was designed for the old input rate and the old channel count, and later buffers
    /// keep being decoded as the old sample format. Following the change means redesigning the
    /// filter, re-announcing downstream and re-fixating the tail of the graph — gapless-playback
    /// work with its own design, deliberately not attempted here. So the contract is: keep the
    /// old format, but say so, loudly.
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
                    "audioresample: ignoring a mid-stream input format change ({latched} -> \
                     {offered}); the polyphase filter is designed once for the latched input and \
                     every later buffer is still resampled as the latched format. Rebuild the \
                     branch (or insert a fresh audioresample) to follow the change."
                ),
            },
        });
    }

    /// Announce the output `audio/raw` (channels + sample from the input, `rate` = target) on the
    /// src pad, once, as soon as the full input format is known (spec: dynamic caps — producer).
    /// Deferred until the sample format resolves so the announced `sample` id is correct.
    fn announce_output(&mut self, ctx: &mut Ctx) {
        if self.announced {
            return;
        }
        let Some(inp) = self.input else { return };
        ctx.announce_format(
            SRC,
            FAMILY,
            &[
                (FIELD_RATE, ValueDesc::Int(self.target_rate as i64)),
                (FIELD_CHANNELS, ValueDesc::Int(inp.channels as i64)),
                (FIELD_SAMPLE, ValueDesc::Id(inp.format.caps_name())),
            ],
        );
        self.announced = true;
    }

    /// Resample `bytes` (whole interchannel frames of the input format) and push the result on
    /// the src pad. Pass-through (no engine) forwards the bytes verbatim.
    fn emit_resampled(&mut self, ctx: &mut Ctx, bytes: &[u8]) -> Result<(), Error> {
        if bytes.is_empty() {
            return Ok(());
        }
        let Some(engine) = self.engine.as_mut() else {
            // Rate unchanged: forward verbatim.
            return Self::emit_bytes(ctx, bytes);
        };
        let nch = engine.nch;

        // 1. Decode the input bytes to interleaved f32 via the tested conversion library.
        let inter = convert_interleaved_vec(engine.sample, SampleFormat::F32, nch, bytes)
            .ok_or(Error::Todo("audioresample: ragged frame in emit"))?;
        let frames = inter.len() / (4 * nch); // f32 == 4 bytes

        // All per-buffer de-interleave / resample scratch is carved from the per-`process()`
        // arena (`ctx.scratch()`), so steady-state resampling adds no heap traffic. Scoped so the
        // arena borrow of `ctx` ends before the `&mut ctx` `emit_bytes` below — only the re-encoded
        // `pcm` (an owned Vec independent of the arena) escapes.
        let pcm = {
            let arena = ctx.scratch();

            // 2. De-interleave into planar per-channel f32, resample each, collect planar outputs.
            let mut planar_out: Vec<Vec<f32, &_>, &_> = Vec::with_capacity_in(nch, arena);
            let mut max_out = 0usize;
            for ch in 0..nch {
                let mut lane: Vec<f32, &_> = Vec::with_capacity_in(frames, arena);
                for f in 0..frames {
                    lane.push(f32::from_le_bytes([
                        inter[(f * nch + ch) * 4],
                        inter[(f * nch + ch) * 4 + 1],
                        inter[(f * nch + ch) * 4 + 2],
                        inter[(f * nch + ch) * 4 + 3],
                    ]));
                }
                let mut out: Vec<f32, &_> = Vec::new_in(arena);
                engine.channels[ch].process(&lane, &mut out);
                max_out = max_out.max(out.len());
                planar_out.push(out);
            }
            // Every channel shares L/M and history sizing, so they emit the same count in lock-step.
            debug_assert!(
                planar_out.iter().all(|c| c.len() == max_out),
                "channel output counts diverged"
            );
            if max_out == 0 {
                return Ok(());
            }

            // 3. Re-interleave to f32 bytes (arena scratch), then re-encode to the input format.
            let mut inter_out: Vec<u8, &_> = Vec::with_capacity_in(max_out * nch * 4, arena);
            inter_out.resize(max_out * nch * 4, 0u8);
            for f in 0..max_out {
                for (ch, lane) in planar_out.iter().enumerate() {
                    let v = lane.get(f).copied().unwrap_or(0.0);
                    let off = (f * nch + ch) * 4;
                    inter_out[off..off + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            convert_interleaved_vec(SampleFormat::F32, engine.sample, nch, &inter_out)
                .ok_or(Error::Todo("audioresample: re-encode failed"))?
        };
        Self::emit_bytes(ctx, &pcm)
    }

    /// Push raw `bytes` on the src pad, chunked to the pool slot size (mirrors `audioconvert`).
    fn emit_bytes(ctx: &mut Ctx, bytes: &[u8]) -> Result<(), Error> {
        let mut off = 0;
        while off < bytes.len() {
            let mut buf = ctx.alloc(SRC);
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "audioresample: zero-capacity pool slot");
            let n = cap.min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            ctx.out(SRC).push(buf);
            off += n;
        }
        Ok(())
    }
}

impl Element for AudioResample {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // A parsed `rate=` overrides the constructor target rate (spec: Plugins).
        if let Some(Value::Int(r)) = ctx.prop("rate") {
            if r > 0 {
                self.target_rate = r as u32;
            }
        }
        // Link-time negotiation may already have fixed the sink format; infer the input now so a
        // statically-negotiated graph needs no runtime event.
        self.learn_from_sink(ctx);
        self.carry.clear();
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
                    "audioresample: PCM arrived before an audio/raw input format was known \
                     (construct with AudioResample::with_input, or feed an announcing upstream)",
                ));
            }
            let data = buf.memory.data();
            // Resample whole interchannel frames only; carry any partial frame to the next buffer.
            if self.carry.is_empty() {
                let whole = data.len() - (data.len() % self.in_stride);
                self.emit_resampled(ctx, &data[..whole])?;
                self.carry.extend_from_slice(&data[whole..]);
            } else {
                let mut joined = std::mem::take(&mut self.carry);
                joined.extend_from_slice(data);
                let whole = joined.len() - (joined.len() % self.in_stride);
                self.emit_resampled(ctx, &joined[..whole])?;
                self.carry.clear();
                self.carry.extend_from_slice(&joined[whole..]);
            }
            // `buf` recycles here on drop.
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Seek (spec: flush/seek). Two kinds of state are pre-seek audio and both go:
            //
            // * `carry`, the tail of a partial interchannel frame. Prepending it to the first
            //   post-seek buffer shifts every later sample by part of a frame, permanently
            //   rotating the channel lanes (a stereo stream comes out L/R swapped).
            // * the per-channel FIR delay lines. A streaming resampler convolves each output
            //   across the buffer boundary *by design*, so without this the first filter
            //   length of post-seek output is a blend of audio from both sides of the seek —
            //   an audible smear exactly at the point the listener asked for a clean cut.
            //
            // The filter design and the learned format both survive: a seek moves the read
            // head, it does not change the stream's rate, channels or sample format.
            Event::FlushStart => {
                self.carry.clear();
                if let Some(engine) = self.engine.as_mut() {
                    engine.reset();
                }
            }
            // A runtime format announcement arrives as a FormatChange on our sink; the pipeline
            // updates `ctx.negotiated(sink)` before calling us, so the concrete format is
            // already there. Re-arm inference (a no-op once learned), and warn when the new
            // format disagrees with what we latched.
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

    fn stop(&mut self, _ctx: &mut Ctx) {
        // A leftover carry is an incomplete final frame (ragged input) — drop it so output stays
        // frame-aligned. (Flushing the FIR tail with zeros to emit the last group-delay samples
        // is a natural follow-up; Milestone-1 tests assert on the steady-state body.)
        self.carry.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_tracks_target_rate_and_input_shape() {
        let r = AudioResample::with_input(AudioFormat::new(44_100, 2, SampleFormat::S16), 48_000);
        assert_eq!(r.target_rate(), 48_000);
        assert_eq!(r.input_format(), Some(AudioFormat::new(44_100, 2, SampleFormat::S16)));
        assert_eq!(
            r.output_format(),
            Some(AudioFormat::new(48_000, 2, SampleFormat::S16)),
            "output keeps channels + sample format, only the rate changes"
        );
    }

    #[test]
    fn equal_rate_is_pass_through_engine_none() {
        let r = AudioResample::with_input(AudioFormat::new(48_000, 1, SampleFormat::S16), 48_000);
        assert!(r.engine.is_none(), "matching rate needs no resampling engine");
    }
}
