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
//!   caps — the consumer side). The rate and channels ride a `FormatChange`/negotiated
//!   [`FixedFormat`] as plain [`Value::Int`]s and are read positionally (the sample id is
//!   interned and cannot be reversed from an element until [`Ctx`] exposes the interners —
//!   see the note on [`AudioConvert::new`]). For a fully-pinned, upstream-independent input —
//!   which is what tests and statically-wired graphs use — construct with
//!   [`AudioConvert::with_input`].
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
use streamcraft_core::format::{ConstraintDesc, FieldDesc, FixedFormat, OfferDesc, Value, ValueDesc};
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
    /// The `(rate, channels)` recovered from negotiated caps when only those are readable
    /// (the runtime-caps path, sample format still pending an id→name accessor on [`Ctx`]).
    /// Used so [`output_format`](AudioConvert::output_format) reports the negotiated
    /// rate/channels even before a full input format is pinned.
    partial_in: Option<(u32, u16)>,
}

impl AudioConvert {
    /// A converter targeting `target`, discovering its input format at runtime from the
    /// upstream's negotiated `audio/raw` caps (spec: dynamic caps — consumer side).
    ///
    /// The input **rate** and **channels** are recovered from the negotiated
    /// [`FixedFormat`] (they are plain [`Value::Int`]s). The input **sample format** is an
    /// interned categorical id; an element cannot reverse it to a name until [`Ctx`] exposes
    /// the interning tables (the `&'static`→id direction is already available for
    /// *announcing*, via [`Ctx::announce_format`]; the id→`&'static` direction that
    /// consumers need is the natural follow-up). Until then, prefer [`with_input`] whenever
    /// the input format is known to the caller, which is the case for tests and every
    /// statically-wired graph.
    ///
    /// [`with_input`]: AudioConvert::with_input
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
        // Link-time negotiation may already have fixed the sink format; pick up what we can
        // so a statically-negotiated graph learns rate/channels without a runtime event.
        if let Some(fixed) = ctx.negotiated(SINK) {
            self.merge_negotiated(fixed);
        }
        self.carry.clear();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Late-arriving caps: an upstream may have announced after `start()`.
        if self.input.is_none() {
            if let Some(fixed) = ctx.negotiated(SINK) {
                self.merge_negotiated(fixed);
            }
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
        // (spec: dynamic caps). The pipeline updates `ctx.negotiated(sink)` before calling
        // us; read the concrete format from either the event or the negotiated caps.
        if let Event::FormatChange(fixed) = event {
            self.merge_negotiated(fixed);
        } else if self.input.is_none() {
            if let Some(fixed) = ctx.negotiated(SINK) {
                self.merge_negotiated(fixed);
            }
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
    /// Recover the input rate and channels from a negotiated `audio/raw` [`FixedFormat`],
    /// caching them so [`output_format`](AudioConvert::output_format) and diagnostics reflect
    /// the negotiated stream even on the runtime-caps path. A pinned input (from
    /// [`with_input`](AudioConvert::with_input)) always wins and short-circuits.
    ///
    /// Rate and channels are plain [`Value::Int`]s and ride the format in the canonical
    /// announcement order `[rate, channels, sample]` (see the src offer and the way decoders
    /// like `flacdec` announce); [`FixedFormat::fields`] preserves insertion order, so they
    /// read positionally with no interner. The `sample` id is categorical and cannot be
    /// reversed to a [`SampleFormat`] from an element yet — that needs the id→name table the
    /// pipeline holds but does not expose on [`Ctx`] (the announce direction is already
    /// available; the consume direction is the natural follow-up). Until then this cannot
    /// finish inferring the input *sample format* from caps, so it records the rate/channels
    /// as a partial hint and leaves `input` unset; construct with
    /// [`with_input`](AudioConvert::with_input) to run without an id→name accessor.
    fn merge_negotiated(&mut self, fixed: &FixedFormat) {
        if self.input.is_some() {
            return; // pinned or already learned
        }
        // Positional read of the two integer fields (rate, channels) in announcement order.
        let ints: Vec<i64> = fixed
            .fields()
            .iter()
            .filter_map(|(_, v)| match v {
                Value::Int(n) => Some(*n),
                _ => None,
            })
            .collect();
        if let (Some(&rate), Some(&channels)) = (ints.first(), ints.get(1)) {
            if rate > 0 && channels > 0 {
                self.partial_in = Some((rate as u32, channels as u16));
            }
        }
    }
}
