//! `audiodownmix` — fold a multichannel `audio/raw` stream down to **stereo** (spec: Formats
//! — dynamic caps; Writing elements). Interleaved PCM of any channel count arrives on the
//! sink pad; interleaved *stereo* PCM in the same sample format and rate leaves on the src
//! pad. The mix is the ITU-R BS.775-3 "Lo/Ro" matrix — see [`crate::convert::downmix_to_stereo`]
//! for the coefficients, the assumed canonical channel order, and why LFE is dropped.
//!
//! Why this element exists: the 5.1 AC-3 / E-AC-3 movie tracks (`Nord`'s AC-3, `Avatar`'s
//! DDP5.1) have **no companion stereo track**, and the PipeWire device sink is driven at
//! stereo. Without a fold, a 6-channel stream either fails to open the device or plays only
//! the front pair. (The 5.1 *AAC* movie tracks ship an explicit 2-channel mix alongside, so
//! the autoplugger prefers that and this element is never inserted for them — higher quality
//! than any matrix fold.)
//!
//! A passive transform, like [`crate::convert_element::AudioConvert`]: it inlines into the
//! upstream group and never blocks. It is both a consumer and a producer of dynamic caps —
//! it learns rate/channels/sample from the negotiated sink caps (reading each field by name
//! through the shared vocabulary, the `audioconvert`/`flacdec` pattern) and announces its
//! stereo output on the src pad the moment the input is known. `channels <= 2` passes
//! through byte-for-byte (the announcement still fires so downstream sees the format).

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, PropDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::convert::downmix_to_stereo;
use crate::format::{SampleFormat, FAMILY, FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE};

const SINK: PadId = PadId(0);
const SRC: PadId = PadId(1);

// Both pads speak `audio/raw` (any rate/channels/sample) plus the `bytes` escape, mirroring
// `audioconvert`. The sink takes whatever the upstream fixates; the src is `dynamic` and
// announces its concrete stereo output at runtime. Every sample-format name is offered so
// the `sample` value is interned for the announcement to resolve against.
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
    PadDesc { name: "sink", direction: Direction::Sink, offers: &OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &OFFERS, dynamic: true, validate: None },
];

static PROPS: [PropDesc; 0] = [];

static DESC: ElementDesc = ElementDesc {
    name: "audiodownmix",
    pads: &PADS,
    props: &PROPS,
    // Passive: a pure transform that inlines into the upstream group (spec: Scheduling).
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Config-free (spec: Plugins — `parse("… ! audiodownmix ! …")`).
    make_default: Some(|| Box::new(AudioDownmix::new())),
};

/// Folds interleaved multichannel PCM to stereo, preserving rate and sample format.
#[derive(Default)]
pub struct AudioDownmix {
    /// Input `(rate, channels, sample)` learned from negotiated caps. `None` until known.
    input: Option<(u32, u16, SampleFormat)>,
    /// Whether the stereo output format has been announced downstream (announce once).
    announced: bool,
    /// Input interchannel-frame stride in bytes (`channels * sample.bytes()`); `0` until known.
    in_stride: usize,
    /// Carry for a partial input frame straddling two buffers, so channel lanes never
    /// desync (mirrors `audioconvert`).
    carry: Vec<u8>,
    /// Reused downmix output buffer, so steady state does not allocate.
    scratch: Vec<u8>,
}

impl AudioDownmix {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the input format and derive the frame stride. A genuinely different format
    /// re-arms the downstream announcement.
    fn set_input(&mut self, rate: u32, channels: u16, sample: SampleFormat) {
        if self.input == Some((rate, channels, sample)) {
            return;
        }
        self.in_stride = channels as usize * sample.bytes();
        self.input = Some((rate, channels, sample));
        self.announced = false;
        self.carry.clear();
    }

    /// Infer the input format from the `audio/raw` [`FixedFormat`] on the sink pad, reading
    /// each field by name through the shared vocabulary (the `audioconvert` pattern).
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
        let (Some(rate), Some(channels)) = (int_field(FIELD_RATE), int_field(FIELD_CHANNELS)) else {
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
            self.set_input(rate as u32, channels as u16, fmt);
        }
    }

    /// Announce the stereo output format (same rate, `channels = 2`, same sample) on the src
    /// pad, once the input is known (spec: dynamic caps — producer side).
    fn announce_output(&mut self, ctx: &mut Ctx) {
        if self.announced {
            return;
        }
        let Some((rate, _ch, fmt)) = self.input else { return };
        ctx.announce_format(
            SRC,
            FAMILY,
            &[
                (FIELD_RATE, ValueDesc::Int(rate as i64)),
                (FIELD_CHANNELS, ValueDesc::Int(2)),
                (FIELD_SAMPLE, ValueDesc::Id(fmt.caps_name())),
            ],
        );
        self.announced = true;
    }

    /// Downmix `bytes` (whole interchannel frames) to stereo and push on the src pad. When
    /// the input is already mono/stereo the bytes pass through unchanged (mono is left as-is
    /// — a stereo device plays a single-channel stream fine, and duplicating it is the
    /// converter's job, not the downmixer's).
    fn emit(&mut self, ctx: &mut Ctx, bytes: &[u8]) -> Result<(), Error> {
        let Some((_rate, channels, fmt)) = self.input else {
            return Err(Error::Todo("audiodownmix: input format unknown"));
        };
        if channels <= 2 {
            return Self::emit_bytes(ctx, bytes);
        }
        let frames = bytes.len() / self.in_stride;
        let out_len = frames * 2 * fmt.bytes();
        self.scratch.clear();
        self.scratch.resize(out_len, 0);
        downmix_to_stereo(fmt, channels as usize, bytes, &mut self.scratch)
            .ok_or(Error::Todo("audiodownmix: fold failed (ragged frame)"))?;
        let out = std::mem::take(&mut self.scratch);
        let r = Self::emit_bytes(ctx, &out);
        self.scratch = out;
        self.scratch.clear();
        r
    }

    /// Push raw `bytes` on the src pad, chunked to the pool slot size (mirrors `audioconvert`).
    fn emit_bytes(ctx: &mut Ctx, bytes: &[u8]) -> Result<(), Error> {
        let mut off = 0;
        while off < bytes.len() {
            let mut buf = ctx.alloc(SRC);
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "audiodownmix: zero-capacity pool slot");
            let n = cap.min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            ctx.out(SRC).push(buf);
            off += n;
        }
        Ok(())
    }
}

impl Element for AudioDownmix {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
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
                    "audiodownmix: PCM arrived before an audio/raw input format was known",
                ));
            }
            let data = buf.memory.data();
            // Fold whole interchannel frames only; carry any partial frame to the next buffer
            // so the channel lanes never desync (mirrors `audioconvert`).
            if self.carry.is_empty() {
                let whole = data.len() - (data.len() % self.in_stride);
                self.emit(ctx, &data[..whole])?;
                self.carry.extend_from_slice(&data[whole..]);
            } else {
                let mut joined = std::mem::take(&mut self.carry);
                joined.extend_from_slice(data);
                let whole = joined.len() - (joined.len() % self.in_stride);
                self.emit(ctx, &joined[..whole])?;
                self.carry.clear();
                self.carry.extend_from_slice(&joined[whole..]);
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::FormatChange(_)) || self.input.is_none() {
            self.learn_from_sink(ctx);
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.carry.clear();
        self.scratch = Vec::new();
    }
}
