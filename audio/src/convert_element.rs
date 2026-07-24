//! `audioconvert` — the streamcraft element wrapping the [`crate::convert`] library (spec:
//! Formats — dynamic caps; Writing elements). Interleaved `audio/raw` PCM arrives on the
//! sink pad; the same audio in a target [`SampleFormat`] leaves on the src pad. Same rate,
//! same channels — only the sample *representation* changes (rate conversion is a separate
//! element; see the crate follow-ups).
//!
//! A passive transform, like [`crate::wav::WavParse`] and `flacenc`: it inlines into the
//! upstream group and never blocks. It is **both a consumer and a producer of dynamic caps**:
//!
//! * It **announces** its output format on the src pad the moment the input is known — same
//!   rate and channels, `sample` = the construction-time target — via
//!   [`Ctx::announce_format`], so its own downstream re-fixates (spec: dynamic caps — the
//!   producer side, using the `&'static` field/value names the scheduler resolves).
//! * It learns its **input** format from the negotiated caps on the sink pad (spec: dynamic
//!   caps — the consumer side). Every field is read **by name** off the negotiated
//!   [`FixedFormat`] through the shared [`Vocabulary`](streamcraft_core::format::Vocabulary):
//!   `ctx.field_id("rate")` resolves the field id, `f.get(id)` the value, and
//!   `ctx.value_name(id)` reverses the interned `sample` id back to a caps name (the same
//!   pattern `flacdec` and `pipewireaudiosink` use). So [`AudioConvert::new`] **infers** the
//!   full input [`AudioFormat`] — rate, channels *and* sample format — with no out-of-band
//!   hint, whenever the sink carries a concrete `audio/raw` format (a link-time-fixed edge,
//!   or a runtime `FormatChange` delivered to this element). [`AudioConvert::with_input`]
//!   remains for callers that already know the input format and want it pinned up front.
//!
//! If the input sample format already equals the target, the payload passes through byte for
//! byte (the announcement still fires, so downstream sees the format).

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::convert::{convert_interleaved, converted_len};
use crate::format::{AudioFormat, SampleFormat, FAMILY, FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE};

const SINK: PadId = PadId(0);
const SRC: PadId = PadId(1);

// Both pads prefer a broad `audio/raw` but also accept raw `bytes`. The sink takes whatever
// the upstream fixates; the src is `dynamic` and announces its concrete (target-format)
// output at runtime. Every known sample format is offered so `audio/raw` negotiation matches
// any raw-audio peer and the `sample` field name is interned for the announcement to resolve
// against. The trailing `bytes` offer lets the element bridge the `bytes`-typed transports —
// `filesrc`/`filesink` and the `bytes`-pad codecs (`flacenc`) — exactly as `wavparse` does,
// while still being an `audio/raw`-first transform. `audio/raw` is listed first so it wins
// whenever the peer also speaks it (declaration order is preference order — see `negotiate`).
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
        dynamic: true, // output sample format is announced at runtime
        validate: None,
    },
];

static DESC: ElementDesc = ElementDesc {
    name: "audioconvert",
    pads: &PADS,
    props: &[],
    // Passive: a pure transform that inlines into the upstream group (spec: Scheduling).
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Converts interleaved PCM to a fixed target [`SampleFormat`], preserving rate and channels.
pub struct AudioConvert {
    /// The sample format every output buffer is written in — set at construction.
    target: SampleFormat,
    /// The input format, learned at runtime from negotiated caps (or pinned via
    /// [`AudioConvert::with_input`]). `None` until known.
    input: Option<AudioFormat>,
    /// Whether the output format has been announced downstream yet (announce once, after the
    /// input format is known — spec: announce on change, not per buffer).
    announced: bool,
    /// Input frame stride in bytes (`channels * input.bytes()`), so we only ever convert
    /// whole interchannel frames. `0` until `input` is known.
    in_stride: usize,
    /// Carry for a partial input frame straddling two buffers (mirrors `flacenc`): only
    /// whole interchannel frames are handed to the converter.
    carry: Vec<u8>,
    /// Reused conversion output buffer, so steady-state conversion does not allocate.
    scratch: Vec<u8>,
    /// The `(rate, channels)` recovered from negotiated caps when the sample format is not
    /// (yet) resolvable — e.g. the vocabulary is present but the negotiated `sample` id has
    /// no name, or the edge fixed rate/channels but left `sample` open. Used so
    /// [`output_format`](AudioConvert::output_format) reports the negotiated rate/channels
    /// even before a full input format is pinned. Once all three fields resolve, `input` is
    /// set and this is redundant.
    partial_in: Option<(u32, u16)>,
}

impl AudioConvert {
    /// A converter targeting `target`, discovering its input format at runtime from the
    /// upstream's negotiated `audio/raw` caps (spec: dynamic caps — consumer side).
    ///
    /// The whole input [`AudioFormat`] — **rate**, **channels** *and* **sample format** — is
    /// read by name off the negotiated [`FixedFormat`] via the shared
    /// [`Vocabulary`](streamcraft_core::format::Vocabulary): `ctx.field_id("rate")` /
    /// `"channels"` / `"sample"`, then `ctx.value_name(id)` reverses the interned `sample` id
    /// to its caps name (the [`flacdec`]/[`pipewireaudiosink`] pattern). No out-of-band hint
    /// is needed; it works the moment the sink carries a concrete `audio/raw` format (a
    /// link-time-fixed edge, read in [`start`], or a runtime `FormatChange`, read in
    /// [`event`]/[`process`]). Use [`with_input`] to pin a known input format up front
    /// instead.
    ///
    /// [`with_input`]: AudioConvert::with_input
    /// [`start`]: Element::start
    /// [`event`]: Element::event
    /// [`process`]: Element::process
    /// [`flacdec`]: https://docs.rs/sc-flac
    /// [`pipewireaudiosink`]: https://docs.rs/sc-pipewire
    pub fn new(target: SampleFormat) -> Self {
        Self {
            target,
            input: None,
            announced: false,
            in_stride: 0,
            carry: Vec::new(),
            scratch: Vec::new(),
            partial_in: None,
        }
    }

    /// A converter with the input format pinned at construction, converting `input` PCM to
    /// `target`. Needs no announcing upstream, so it is the statically-wired / testable form
    /// (`filesrc(raw PCM) ! AudioConvert::with_input(in, target) ! filesink`). It still
    /// announces the output format downstream.
    pub fn with_input(input: AudioFormat, target: SampleFormat) -> Self {
        let mut c = Self::new(target);
        c.set_input(input);
        c
    }

    /// The target output sample format.
    pub fn target(&self) -> SampleFormat {
        self.target
    }

    /// The input format, once known (from negotiation or [`AudioConvert::with_input`]).
    pub fn input_format(&self) -> Option<AudioFormat> {
        self.input
    }

    /// The output format this element produces, once the rate/channels are known: same rate
    /// and channels as the input, `format` = the target. Available from a pinned input or,
    /// on the runtime-caps path, from the rate/channels recovered from negotiated caps.
    pub fn output_format(&self) -> Option<AudioFormat> {
        if let Some(f) = self.input {
            return Some(AudioFormat::new(f.sample_rate, f.channels, self.target));
        }
        self.partial_in
            .map(|(rate, ch)| AudioFormat::new(rate, ch, self.target))
    }

    /// Record the input format and derive the frame stride. Idempotent for the same format;
    /// a genuinely different format re-arms the announcement so downstream is re-notified.
    fn set_input(&mut self, input: AudioFormat) {
        if self.input == Some(input) {
            return;
        }
        self.in_stride = input.frame_stride();
        self.input = Some(input);
        self.announced = false;
        self.carry.clear();
    }

    /// Announce the output `audio/raw` format (same rate/channels, `sample` = target) on the
    /// src pad, once, as soon as the rate/channels are known (spec: dynamic caps — producer
    /// side). The output rate/channels equal the input's, so this can fire from either a
    /// pinned input or the rate/channels recovered from negotiated caps — even before the
    /// input *sample* format is resolved.
    fn announce_output(&mut self, ctx: &mut Ctx) {
        if self.announced {
            return;
        }
        let Some(out) = self.output_format() else { return };
        ctx.announce_format(
            SRC,
            FAMILY,
            &[
                (FIELD_RATE, ValueDesc::Int(out.sample_rate as i64)),
                (FIELD_CHANNELS, ValueDesc::Int(out.channels as i64)),
                (FIELD_SAMPLE, ValueDesc::Id(self.target.caps_name())),
            ],
        );
        self.announced = true;
    }

    /// Convert `bytes` (whole interchannel frames of the input format) into the target and
    /// push the result on the src pad, chunked to the pool slot size (mirrors `flacenc`).
    /// Zero-length input pushes nothing.
    fn emit_converted(&mut self, ctx: &mut Ctx, bytes: &[u8]) -> Result<(), Error> {
        let Some(inp) = self.input else {
            return Err(Error::Todo("audioconvert: input format unknown"));
        };
        let channels = inp.channels as usize;

        // Pass-through: identical sample format needs no conversion, just forward the bytes.
        if inp.format == self.target {
            return Self::emit_bytes(ctx, bytes);
        }

        let out_len = converted_len(inp.format, self.target, channels, bytes.len())
            .ok_or(Error::Todo("audioconvert: ragged frame in emit"))?;
        self.scratch.clear();
        self.scratch.resize(out_len, 0);
        convert_interleaved(inp.format, self.target, channels, bytes, &mut self.scratch)
            .ok_or(Error::Todo("audioconvert: conversion failed"))?;

        // Move the scratch out for pushing, then reclaim its allocation.
        let out = std::mem::take(&mut self.scratch);
        let r = Self::emit_bytes(ctx, &out);
        self.scratch = out;
        self.scratch.clear();
        r
    }

    /// Push raw `bytes` on the src pad, chunked to the pool slot size.
    fn emit_bytes(ctx: &mut Ctx, bytes: &[u8]) -> Result<(), Error> {
        let mut off = 0;
        while off < bytes.len() {
            let mut buf = ctx.alloc(SRC);
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "audioconvert: zero-capacity pool slot");
            let n = cap.min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            ctx.out(SRC).push(buf);
            off += n;
        }
        Ok(())
    }
}

impl Element for AudioConvert {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Link-time negotiation may already have fixed the sink format; infer the input
        // format now so a statically-negotiated graph needs no runtime event.
        self.learn_from_sink(ctx);
        self.carry.clear();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Late-arriving caps: an upstream may have announced after `start()`.
        if self.input.is_none() {
            self.learn_from_sink(ctx);
        }
        // Announce our output format as soon as the input is known (once).
        self.announce_output(ctx);

        while let Some(buf) = inputs.pop() {
            if self.input.is_none() {
                // No format yet — cannot interpret the bytes. A wiring error: no `with_input`
                // and an upstream that never gave us a usable `audio/raw` format. Fail loudly
                // rather than emit garbage.
                return Err(Error::Todo(
                    "audioconvert: PCM arrived before an audio/raw input format was known \
                     (construct with AudioConvert::with_input, or feed an announcing upstream)",
                ));
            }
            let data = buf.memory.data();

            // Convert whole interchannel frames only; carry any partial frame to the next
            // buffer so channel lanes never desync (mirrors `flacenc`).
            if self.carry.is_empty() {
                let whole = data.len() - (data.len() % self.in_stride);
                self.emit_converted(ctx, &data[..whole])?;
                self.carry.extend_from_slice(&data[whole..]);
            } else {
                let mut joined = std::mem::take(&mut self.carry);
                joined.extend_from_slice(data);
                let whole = joined.len() - (joined.len() % self.in_stride);
                self.emit_converted(ctx, &joined[..whole])?;
                self.carry.clear();
                self.carry.extend_from_slice(&joined[whole..]);
            }
            // `buf` recycles here on drop.
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // An upstream's runtime format announcement arrives as a FormatChange on our sink
        // (spec: dynamic caps). The pipeline updates `ctx.negotiated(sink)` *before* calling
        // us, so the fresh concrete format is already on the sink pad either way — a
        // FormatChange re-arms inference, and any other event is a cheap no-op once learned.
        if matches!(event, Event::FormatChange(_)) || self.input.is_none() {
            self.learn_from_sink(ctx);
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // A leftover carry is an incomplete final interchannel frame (ragged input) — drop
        // it so the output stays frame-aligned.
        self.carry.clear();
        self.scratch = Vec::new();
    }
}

impl AudioConvert {
    /// Infer the input [`AudioFormat`] from the `audio/raw` [`FixedFormat`] currently
    /// negotiated on the sink pad, reading every field **by name** through the shared
    /// vocabulary (spec: Formats — an element acts on its caps). A pinned input (from
    /// [`with_input`](AudioConvert::with_input)) or an already-learned one always wins and
    /// short-circuits.
    ///
    /// `rate` and `channels` are plain [`Value::Int`]s; `sample` is an interned categorical
    /// id reversed to its caps name via [`Ctx::value_name`] and mapped with
    /// [`SampleFormat::from_caps_name`] — exactly how `flacdec` announces and
    /// `pipewireaudiosink` consumes. When all three resolve, [`set_input`](Self::set_input)
    /// pins the format and re-arms the downstream announcement. When only rate/channels are
    /// readable (no vocabulary yet, an unmapped `sample`, or an edge that left `sample`
    /// open), they are cached as a [`partial_in`](Self::partial_in) hint so
    /// [`output_format`](Self::output_format) still reflects the negotiated stream.
    fn learn_from_sink(&mut self, ctx: &Ctx) {
        if self.input.is_some() {
            return; // pinned or already learned
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

        // The categorical `sample` id, reversed to a caps name, then to a `SampleFormat`.
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
            // Rate/channels known but the sample format is not resolvable yet — record the
            // partial so output_format reports the negotiated stream; input stays unset.
            None => self.partial_in = Some((rate as u32, channels as u16)),
        }
    }
}
