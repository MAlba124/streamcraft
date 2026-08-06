//! `flacenc` — the profluens element wrapping [`FlacEncoder`] (spec: Milestone
//! applications §3; Writing elements). A passive transform: raw **interleaved** PCM
//! buffers arrive on the sink pad, encoded FLAC bytes leave on the src pad. It inlines
//! into the upstream active element's group like [`passthrough`], so it never blocks.
//!
//! Audio parameters come from, in increasing precedence: the [`FlacEnc::new`]
//! constructor, parse-path props, and the **negotiated/announced input format**
//! (spec: Formats — dynamic caps, the consumer side). The sink pad offers
//! `audio/raw` (preferred — so a decoder like `flacdec` links directly and its
//! runtime `FormatChange` announcement sets rate/channels/sample) plus a legacy
//! `bytes` bridge (raw interleaved PCM from `filesrc`, or `wavparse`, whose
//! announcement rides the bridge and is learned the same way). A format change
//! after the STREAMINFO header has been emitted is a loud error — FLAC cannot
//! change stream parameters mid-file.
//!
//! It writes a **streaming**, forward-only FLAC stream (see
//! [`FlacEncoder::new_streaming`]): the STREAMINFO header is emitted once, up front,
//! with "unknown" (0) frame sizes and total-sample count — spec-legal (§8.2) and
//! required because a sequential sink (`filesink`) cannot be back-patched. The output
//! is a fully valid, streamable FLAC file.

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, PropDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{
    Constraint, ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc,
};
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

use crate::encoder::{FlacEncoder, SampleFormat};

/// Block size (interchannel samples) the encoder aims for per frame. 4096 is the
/// common libFLAC default and within the streamable subset for <=48 kHz (§7).
const BLOCK_SIZE: u32 = 4096;

// `audio/raw` names, matching `flacdec`'s announce vocabulary (interned by string,
// so the ids line up graph-wide).
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

static SINK_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_RATE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_CHANNELS, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_SAMPLE, allowed: ConstraintDesc::Set(&SAMPLE_VALUES), preferred: None },
];

/// Sink: typed `audio/raw` first (a decoder links directly and its announcement sets
/// the parameters), then the legacy `bytes` bridge (raw PCM from `filesrc`, or
/// `wavparse` announcing over the bridge). Declaration order is preference order.
static SINK_OFFERS: [OfferDesc; 2] = [
    OfferDesc { family: FAMILY, fields: &SINK_FIELDS },
    OfferDesc::any("bytes"),
];
/// Src: encoded FLAC bytes, matching `filesink`/`oggmux`/`flacdec` peers.
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

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
        dynamic: false,
        validate: None,
    },
];

/// The interleaved-PCM parameters, for the parse path (spec: Plugins —
/// `parse("… ! flacenc rate=44100 channels=2 format=s16 ! …")`). Until `wavparse`
/// propagates the format through caps, `flacenc` needs these out of band; the parse
/// layer supplies them as props (the typed [`FlacEnc::new`] is the other path). All
/// structural (`live: false`): the encoder is built in `start()`.
static PROPS: [PropDesc; 3] = [
    PropDesc { name: "rate", allowed: Constraint::Any, live: false },
    PropDesc { name: "channels", allowed: Constraint::Any, live: false },
    PropDesc { name: "format", allowed: Constraint::Any, live: false },
];

// `make_default` boxes one element instance at registry/parse time, never per frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "flacenc",
    pads: &PADS,
    props: &PROPS,
    // Passive: pure transform, inlines into the upstream group (spec: Scheduling).
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Default parameters (44.1 kHz stereo S16, the spec-target case); override via
    // the `rate`/`channels`/`format` props at parse time.
    make_default: Some(|| Box::new(FlacEnc::new(44_100, 2, SampleFormat::S16))),
};

/// Map an `audio/raw`-style sample-format name to the encoder's [`SampleFormat`], for
/// the parse path's `format=` property. Unknown names leave the constructor default.
fn sample_format_from_name(s: &str) -> Option<SampleFormat> {
    Some(match s {
        "s8" => SampleFormat::S8,
        "s16" => SampleFormat::S16,
        "s24" => SampleFormat::S24,
        "s32" => SampleFormat::S32,
        _ => return None,
    })
}

pub struct FlacEnc {
    sample_rate: u32,
    channels: u32,
    format: SampleFormat,
    encoder: Option<FlacEncoder>,
    header_sent: bool,
    /// Bytes per interchannel sample (channels * bytes-per-sample) — the frame stride.
    frame_stride: usize,
    /// Carry for a partial interchannel sample straddling two input buffers, so we
    /// only ever hand whole interchannel samples to the encoder.
    carry: Vec<u8>,
    /// Reused encode output buffer, so steady-state encoding does not allocate.
    scratch_out: Vec<u8>,
}

impl FlacEnc {
    /// Create a FLAC encoder element for interleaved `format` PCM at `sample_rate` Hz
    /// with `channels` channels. Parameters are validated lazily in `start()`.
    // One-time construction: `carry`/`scratch_out` are reused across frames, not re-allocated.
    #[allow(clippy::disallowed_methods)]
    pub fn new(sample_rate: u32, channels: u32, format: SampleFormat) -> Self {
        let frame_stride = channels as usize * format.bytes_per_sample();
        Self {
            sample_rate,
            channels,
            format,
            encoder: None,
            header_sent: false,
            frame_stride,
            carry: Vec::new(),
            scratch_out: Vec::new(),
        }
    }

    /// Adopt rate/channels/sample from the sink pad's installed `audio/raw` format —
    /// the link-time fixation in `start()`, or a runtime `FormatChange` announcement
    /// (spec: Formats — dynamic caps, the consumer side; the `AudioConvert` pattern).
    /// Fields absent from the format keep their current values, so a partially
    /// specified link-time fixation is refined by the upstream's announcement (which
    /// always arrives before its data). Returns whether anything changed.
    fn learn_from_caps(&mut self, ctx: &Ctx) -> bool {
        let Some(f) = ctx.negotiated(PadId(0)) else { return false };
        if ctx.family_name(f.family) != Some(FAMILY) {
            return false; // the bytes bridge fixes no audio parameters
        }
        let mut changed = false;
        if let Some(Value::Int(r)) = ctx.field_id(F_RATE).and_then(|id| f.get(id)) {
            if r > 0 && r as u32 != self.sample_rate {
                self.sample_rate = r as u32;
                changed = true;
            }
        }
        if let Some(Value::Int(c)) = ctx.field_id(F_CHANNELS).and_then(|id| f.get(id)) {
            if c > 0 && c as u32 != self.channels {
                self.channels = c as u32;
                changed = true;
            }
        }
        if let Some(Value::Id(id)) = ctx.field_id(F_SAMPLE).and_then(|id| f.get(id)) {
            if let Some(sf) = ctx.value_name(id).and_then(sample_format_from_name) {
                if sf != self.format {
                    self.format = sf;
                    changed = true;
                }
            }
        }
        if changed {
            self.frame_stride = self.channels as usize * self.format.bytes_per_sample();
        }
        changed
    }

    /// (Re)build the encoder from the current parameters. Any carried partial sample
    /// belongs to the old layout, so it is dropped with the old encoder.
    fn rebuild_encoder(&mut self) -> Result<(), Error> {
        let (encoder, _header) =
            FlacEncoder::new_streaming(self.sample_rate, self.channels, self.format, BLOCK_SIZE)
                .map_err(|e| Error::Resource(format!("flacenc: {e:?}")))?;
        self.encoder = Some(encoder);
        self.carry.clear();
        Ok(())
    }

    /// Push `bytes` as an output buffer on the src pad, chunked to the pool slot size
    /// so no single copy exceeds a buffer. Zero-length input pushes nothing.
    fn emit(ctx: &mut Ctx, bytes: &[u8]) -> Result<(), Error> {
        let mut off = 0;
        while off < bytes.len() {
            let mut buf = ctx.alloc(PadId(1));
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "flacenc: zero-capacity pool slot");
            let n = cap.min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            ctx.out(PadId(1)).push(buf);
            off += n;
        }
        Ok(())
    }
}

impl Element for FlacEnc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Parsed props override the constructor parameters (spec: Plugins). `rate` and
        // `channels` are ints; `format` is an interned sample-format name.
        if let Some(Value::Int(r)) = ctx.prop("rate") {
            if r > 0 {
                self.sample_rate = r as u32;
            }
        }
        if let Some(Value::Int(c)) = ctx.prop("channels") {
            if c > 0 {
                self.channels = c as u32;
            }
        }
        if let Some(Value::Id(id)) = ctx.prop("format") {
            if let Some(sf) = ctx.value_name(id).and_then(sample_format_from_name) {
                self.format = sf;
            }
        }
        self.frame_stride = self.channels as usize * self.format.bytes_per_sample();
        // The negotiated input format (if the edge is typed `audio/raw`) outranks
        // props and constructor; a runtime announcement refines it again before any
        // data flows (see `event`).
        self.learn_from_caps(ctx);

        // Build the encoder; the STREAMINFO header is emitted on the first `process`
        // (output can only be pushed while holding an `out` batch mid-`process`), so
        // a pre-data `FormatChange` can still rebuild with the right parameters.
        self.rebuild_encoder()?;
        self.header_sent = false;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            // Emit the STREAMINFO header exactly once, right before the first
            // encoded bytes — *lazily*, so an upstream's format announcement (which
            // always precedes its data) can still set the parameters. Emitting it
            // eagerly on the first `process` call would race a decoder upstream
            // that needs a pass of lookahead before it announces.
            if !self.header_sent {
                let (_enc, header) = FlacEncoder::new_streaming(
                    self.sample_rate,
                    self.channels,
                    self.format,
                    BLOCK_SIZE,
                )
                .map_err(|e| Error::Resource(format!("flacenc: {e:?}")))?;
                Self::emit(ctx, &header)?;
                self.header_sent = true;
            }

            // Prepend any carried partial sample, then encode whole interchannel
            // samples and carry the remainder to the next buffer.
            let data = buf.memory.data();
            let combined_len = self.carry.len() + data.len();
            let whole = combined_len - (combined_len % self.frame_stride);

            self.scratch_out.clear();
            let encoder = self
                .encoder
                .as_mut()
                .ok_or(Error::Todo("flacenc not started"))?;

            if self.carry.is_empty() {
                // Fast path: no carry, encode directly from the buffer.
                let split = data.len() - (data.len() % self.frame_stride);
                encoder
                    .encode_interleaved(&data[..split], &mut self.scratch_out)
                    .map_err(|e| Error::Resource(format!("flacenc: {e:?}")))?;
                self.carry.extend_from_slice(&data[split..]);
            } else {
                // Splice carry + new data once, encode the whole-sample prefix.
                let mut joined = std::mem::take(&mut self.carry);
                joined.extend_from_slice(data);
                encoder
                    .encode_interleaved(&joined[..whole], &mut self.scratch_out)
                    .map_err(|e| Error::Resource(format!("flacenc: {e:?}")))?;
                self.carry.clear();
                self.carry.extend_from_slice(&joined[whole..]);
            }

            // `buf` recycles here on drop; move the encoded bytes downstream.
            let out = std::mem::take(&mut self.scratch_out);
            Self::emit(ctx, &out)?;
            self.scratch_out = out; // reclaim the allocation
            self.scratch_out.clear();
        }

        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // The upstream's runtime format announcement (a decoder's header, wavparse's
        // fmt chunk) arrives here before its data (spec: dynamic caps). Before the
        // header went out, adopting it is free — rebuild the encoder. After, FLAC
        // cannot change stream parameters mid-file: fail loudly, never mislabel.
        if matches!(event, Event::FormatChange(_)) && self.learn_from_caps(ctx) {
            if self.header_sent {
                return Err(Error::Element {
                    element: ctx.element(),
                    message: format!(
                        "flacenc: input format changed mid-stream (now {} Hz, {} ch, {:?}) \
                         after STREAMINFO was written — unsupported",
                        self.sample_rate, self.channels, self.format
                    ),
                });
            }
            self.rebuild_encoder()?;
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // Any carried bytes are an incomplete final interchannel sample (ragged input)
        // — dropping them keeps the stream well-formed. FLAC needs no EOS marker (§9).
        self.encoder = None;
        self.carry.clear();
    }
}
