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
    /// Decoded samples of the current frame not yet emitted — the backpressure carry.
    /// Bounded to one frame: the decode loop never pulls another frame until this drains,
    /// so a slow sink cannot make the decoder run ahead and balloon the pool.
    pending: Vec<i64>,
    /// How many of `pending` have already been emitted.
    pending_pos: usize,
}

impl FlacDec {
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit the pending frame's not-yet-written samples as interleaved little-endian PCM,
    /// **directly into pool buffers** (no per-frame intermediate). Each sample is
    /// `bytes_per_sample` wide (the low bytes of the sign-extended two's-complement `i64`
    /// are the correct LE PCM). With `bounded` it uses the capped pool (`try_alloc`) and
    /// returns `Ok(false)` when the pool is full — the caller then yields so the sink
    /// drains (backpressure). With `!bounded` it uses the unbounded pool (never fails), to
    /// guarantee the stream tail is flushed at EOS. Returns `Ok(true)` once fully drained.
    fn emit_pending(&mut self, ctx: &mut Ctx, bounded: bool) -> Result<bool, Error> {
        let bytes = self.bytes_per_sample;
        while self.pending_pos < self.pending.len() {
            debug_assert!(bytes > 0, "flacdec: emit before header decoded");
            let mut buf = if bounded {
                match ctx.try_alloc(PadId(1)) {
                    Some(b) => b,
                    None => return Ok(false), // pool full → backpressure
                }
            } else {
                ctx.alloc(PadId(1))
            };
            let per = (buf.memory.capacity() / bytes).max(1); // whole samples per buffer
            let end = (self.pending_pos + per).min(self.pending.len());
            let dst = buf.memory.as_mut_full();
            let mut n = 0;
            for &s in &self.pending[self.pending_pos..end] {
                dst[n..n + bytes].copy_from_slice(&s.to_le_bytes()[..bytes]);
                n += bytes;
            }
            buf.memory.set_len(n);
            ctx.out(PadId(1)).push(buf);
            self.pending_pos = end;
        }
        self.pending.clear();
        self.pending_pos = 0;
        Ok(true)
    }

    /// Announce the runtime `audio/raw` format once, from STREAMINFO (spec: dynamic caps).
    /// Called after the first frame decodes, so `info()` is populated.
    fn announce_if_needed(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        if self.announced {
            return Ok(());
        }
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
        self.pending.clear();
        self.pending_pos = 0;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        loop {
            // Flush any carry from a prior backpressured call before decoding more.
            if !self.emit_pending(ctx, true)? {
                // Pool full: yield without consuming more input, so the sink drains first.
                // Leaving input buffered keeps the group non-quiescent (backpressure).
                return Ok(Flow::Ok);
            }
            // Decode the next buffered frame; if the decoder is hungry, feed it one input
            // buffer; when there is neither a frame nor more input, we are done this call.
            match self.dec.pull() {
                Ok(Some(frame)) => {
                    self.announce_if_needed(ctx)?; // first frame → header known
                    self.pending = frame.samples;
                    self.pending_pos = 0;
                    if !self.emit_pending(ctx, true)? {
                        return Ok(Flow::Ok); // pool filled mid-frame — carry the rest, yield
                    }
                }
                Ok(None) => match inputs.pop() {
                    Some(buf) => self.dec.push(buf.memory.data()), // `buf` recycles on drop
                    None => return Ok(Flow::Ok),
                },
                Err(e) => return Err(Error::Resource(format!("flacdec: {e:?}"))),
            }
        }
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::Eos) {
            // End of stream: flush the carry and every remaining buffered frame through the
            // unbounded pool, so the tail is emitted even if we were backpressured here.
            self.emit_pending(ctx, false)?;
            loop {
                match self.dec.pull() {
                    Ok(Some(frame)) => {
                        self.announce_if_needed(ctx)?;
                        self.pending = frame.samples;
                        self.pending_pos = 0;
                        self.emit_pending(ctx, false)?;
                    }
                    Ok(None) => break, // no more whole frames (a truncated tail is dropped)
                    Err(e) => return Err(Error::Resource(format!("flacdec: {e:?}"))),
                }
            }
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.dec = StreamDecoder::new();
        self.announced = false;
        self.bytes_per_sample = 0;
        self.pending.clear();
        self.pending_pos = 0;
    }
}
