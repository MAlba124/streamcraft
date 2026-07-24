//! `flacdec` — the incremental FLAC decoder element (spec: Milestone applications §3;
//! Formats — dynamic caps). FLAC bytes arrive on the sink pad; raw interleaved PCM leaves
//! on the src pad. It decodes frame-by-frame as bytes arrive (never buffering the whole
//! stream — latency #2), inlining into the upstream group like a transform.
//!
//! The PCM format (rate / channels / bit depth) lives in STREAMINFO, not in a static
//! descriptor, so `flacdec` cannot declare it up front. Its src pad advertises a broad,
//! `dynamic` `audio/raw` template and then **announces** the concrete format downstream at
//! runtime — via [`Ctx::announce_format`] — the moment the header is decoded. That
//! announcement rides downstream as a `FormatChange`, and the peer reads the fixed format
//! from `ctx.negotiated()` (spec: Formats — dynamic caps). This is the decoder proving the
//! runtime-caps path end to end.

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::decoder::StreamDecoder;

// `audio/raw` family/field/value names. Kept as literals (not a dep on streamcraft-audio)
// so sc-flac stays core-only; they match that crate's convention, and since the pipeline
// interns by string the ids line up with any audio peer that uses the same names.
const FAMILY: &str = "audio/raw";
const F_RATE: &str = "rate";
const F_CHANNELS: &str = "channels";
const F_SAMPLE: &str = "sample";

static SAMPLE_VALUES: [ValueDesc; 4] = [
    ValueDesc::Id("s8"),
    ValueDesc::Id("s16"),
    ValueDesc::Id("s24"),
    ValueDesc::Id("s32"),
];

// A broad `audio/raw` template: any rate/channels, any integer sample format. The concrete
// values are announced at runtime from STREAMINFO — the src pad is `dynamic` for exactly
// this reason (spec: Formats — dynamic caps).
static SRC_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_RATE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_CHANNELS, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_SAMPLE, allowed: ConstraintDesc::Set(&SAMPLE_VALUES), preferred: None },
];
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }];
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &SINK_OFFERS,
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &SRC_OFFERS,
        dynamic: true, // format is data-dependent — announced at runtime
        validate: None,
    },
];

static DESC: ElementDesc = ElementDesc {
    name: "flacdec",
    pads: &PADS,
    props: &[],
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

/// The `audio/raw` sample-format name for a FLAC bit depth.
fn sample_name(bits: u32) -> &'static str {
    match bits {
        8 => "s8",
        16 => "s16",
        24 => "s24",
        _ => "s32",
    }
}

/// Incrementally decodes a FLAC stream to interleaved PCM.
#[derive(Default)]
pub struct FlacDec {
    dec: StreamDecoder,
    announced: bool,
    /// Bytes per single-channel sample, known once the header is decoded.
    bytes_per_sample: usize,
}

impl FlacDec {
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialise a decoded frame's samples as interleaved little-endian PCM **directly
    /// into pool output buffers** — no per-frame intermediate allocation (spec: Memory —
    /// zero steady-state heap traffic). Each sample is `bytes` wide (the low bytes of the
    /// sign-extended two's-complement `i64` are the correct LE PCM for that width); a
    /// buffer is flushed and a fresh one taken whenever the next whole sample would not
    /// fit, so a sample never straddles a buffer boundary.
    fn emit_frame(ctx: &mut Ctx, samples: &[i64], bytes: usize) -> Result<(), Error> {
        if samples.is_empty() {
            return Ok(());
        }
        let mut buf = ctx.alloc(PadId(1));
        let cap = buf.memory.capacity();
        debug_assert!(bytes > 0 && cap >= bytes, "flacdec: pool slot too small for a sample");
        let mut n = 0; // bytes written into the current buffer
        for &s in samples {
            if n + bytes > cap {
                buf.memory.set_len(n);
                ctx.out(PadId(1)).push(buf);
                buf = ctx.alloc(PadId(1));
                n = 0;
            }
            buf.memory.as_mut_full()[n..n + bytes].copy_from_slice(&s.to_le_bytes()[..bytes]);
            n += bytes;
        }
        // The final buffer always holds at least one sample here (empty frames returned
        // early, and a flush is always followed by a write).
        buf.memory.set_len(n);
        ctx.out(PadId(1)).push(buf);
        Ok(())
    }
}

impl Element for FlacDec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.dec = StreamDecoder::new();
        self.announced = false;
        self.bytes_per_sample = 0;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            self.dec.push(buf.memory.data());
            // `buf` recycles on drop here; decode every whole frame now buffered.
            loop {
                let frame = match self.dec.pull() {
                    Ok(Some(f)) => f,
                    Ok(None) => break, // need more input
                    Err(e) => return Err(Error::Resource(format!("flacdec: {e:?}"))),
                };

                // The first decoded frame means the header is known: announce the runtime
                // audio format so the peer can configure itself (spec: dynamic caps).
                if !self.announced {
                    let info = self
                        .dec
                        .info()
                        .ok_or(Error::Todo("flacdec: frame decoded without streaminfo"))?;
                    self.bytes_per_sample = (info.bits_per_sample as usize).div_ceil(8);
                    ctx.announce_format(
                        PadId(1),
                        FAMILY,
                        &[
                            (F_RATE, ValueDesc::Int(info.sample_rate as i64)),
                            (F_CHANNELS, ValueDesc::Int(info.channels as i64)),
                            (F_SAMPLE, ValueDesc::Id(sample_name(info.bits_per_sample))),
                        ],
                    );
                    self.announced = true;
                }

                Self::emit_frame(ctx, &frame.samples, self.bytes_per_sample)?;
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.dec = StreamDecoder::new();
        self.announced = false;
        self.bytes_per_sample = 0;
    }
}
