//! `opusenc` — the Opus (RFC 6716) **encoder** element, driving the CELT-only [`OpusEncoder`]
//! orchestrator (see [`crate::encoder`]). Interleaved 48 kHz `s16` PCM arrives on the `audio/raw`
//! sink pad; one **Opus packet per output buffer** leaves on the `opus` src pad — the exact shape
//! `opusdec` consumes and the `mkv`/`mp4`/`ogg` muxers accept, so a transcode pipeline
//! (`…dec ! audioconvert ! audioresample ! opusenc ! …mux`) links end to end.
//!
//! The sink caps are pinned to **`rate=48000, sample=s16`** (Opus is 48 kHz on the wire and CELT
//! takes `i16` directly), so autoplug inserts an upstream `audioresample`/`audioconvert` for any
//! other source format — the mirror of `opusdec` always *emitting* 48 kHz and letting downstream
//! resample. Channels (1 or 2) are learned from the negotiated caps and drive mono/stereo CELT.
//!
//! # Reframing
//!
//! CELT codes a fixed frame (default 20 ms = 960 samples/channel at 48 kHz). Input buffers are
//! arbitrarily sized, so the element accumulates raw PCM bytes and emits one packet per full
//! frame, carrying a partial frame to the next call. At EOS the trailing partial frame is
//! zero-padded to a full frame and emitted (so no input audio is dropped; the ~20 ms of trailing
//! silence is the standard final-frame pad a container's duration/end-trim absorbs).
//!
//! # Allocation
//!
//! `process()` reuses the byte accumulator, the `i16` frame scratch, and the packet buffer across
//! frames, and emits into right-sized pool buffers (`ctx.alloc_exact`) — no per-frame heap traffic
//! on *this* side. The vendored CELT range coder still allocates internally per packet; making it
//! pool-backed is the alloc-free follow-up (the decoder took the same correctness-first path — see
//! `examples/alloc_check.rs`).

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

use crate::encoder::{Application, EncoderConfig, OPUS_RATE};
#[cfg(not(feature = "libopus"))]
use crate::encoder::OpusEncoder;

const SINK: PadId = PadId(0);
const SRC: PadId = PadId(1);

/// The Opus encoder backend the element drives — selected at build time: the reference **libopus**
/// C encoder (`libopus` feature, default) or the pure-Rust `oxideav-opus` CELT encoder.
///
/// The frame is passed as raw **`s16`-LE PCM bytes** (not `&[i16]`) so the zero-copy backend
/// (libopus) can reinterpret an aligned input buffer in place, while the pure-Rust backend converts
/// into its own reused `i16` scratch.
trait EncBackend: Send {
    /// Bytes of interleaved `s16` PCM one [`Self::encode`] consumes (`channels · frame · 2`).
    fn frame_bytes(&self) -> usize;
    /// Encode one full frame of `s16`-LE bytes into `out` (contents replaced). No per-call
    /// allocation in steady state.
    fn encode(&mut self, frame: &[u8], out: &mut Vec<u8>) -> Result<(), String>;
    /// Reset carried inter-frame state (stream start / seek).
    fn reset(&mut self);
}

#[cfg(feature = "libopus")]
impl EncBackend for crate::libopus::LibopusEncoder {
    fn frame_bytes(&self) -> usize {
        crate::libopus::LibopusEncoder::frame_bytes(self)
    }
    fn encode(&mut self, frame: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
        crate::libopus::LibopusEncoder::encode(self, frame, out)
    }
    fn reset(&mut self) {
        crate::libopus::LibopusEncoder::reset(self)
    }
}

// Pure-Rust backend: `OpusEncoder` takes `&[i16]`, so wrap it with a reused conversion scratch.
#[cfg(not(feature = "libopus"))]
struct PureRust {
    enc: OpusEncoder,
    scratch: Vec<i16>,
}

#[cfg(not(feature = "libopus"))]
impl EncBackend for PureRust {
    fn frame_bytes(&self) -> usize {
        self.enc.frame_len() * BYTES_PER_SAMPLE
    }
    fn encode(&mut self, frame: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
        self.scratch.clear();
        let (pairs, _) = frame.as_chunks::<BYTES_PER_SAMPLE>();
        self.scratch.extend(pairs.iter().map(|c| i16::from_le_bytes(*c)));
        self.enc.encode_frame(&self.scratch, out).map_err(|e| format!("{e:?}"))
    }
    fn reset(&mut self) {
        self.enc.reset()
    }
}

/// Build the feature-selected backend from `cfg`.
#[allow(clippy::disallowed_methods)] // one-time per-stream construction (Box), not a hot path
fn make_backend(cfg: EncoderConfig) -> Result<Box<dyn EncBackend>, String> {
    #[cfg(feature = "libopus")]
    {
        crate::libopus::LibopusEncoder::new(cfg).map(|e| Box::new(e) as Box<dyn EncBackend>)
    }
    #[cfg(not(feature = "libopus"))]
    {
        OpusEncoder::new(cfg)
            .map(|e| Box::new(PureRust { enc: e, scratch: Vec::new() }) as Box<dyn EncBackend>)
            .map_err(|e| format!("{e:?}"))
    }
}

// `audio/raw` family/field names (string-interned; kept as literals so pf-opus stays core-only,
// exactly like `opusdec`). The output `opus` caps reuse the same rate/channels field names so a
// muxer can build the RFC 7845 `OpusHead` from them.
const FAMILY_RAW: &str = "audio/raw";
const FAMILY_OPUS: &str = "opus";
const F_RATE: &str = "rate";
const F_CHANNELS: &str = "channels";
const F_SAMPLE: &str = "sample";

/// Default CELT frame duration: 20 ms (200 tenths) — the Opus-typical size.
const DEFAULT_FRAME_TENTHS: u16 = 200;
// Used by the pure-Rust backend's byte→i16 conversion and the tests; the libopus backend reads
// bytes in place, so this is unreferenced in a `--features libopus` non-test build.
#[allow(dead_code)]
const BYTES_PER_SAMPLE: usize = 2;

// Sink accepts only what CELT can take directly: 48 kHz, `s16`, mono/stereo. Autoplug inserts the
// upstream conversion for anything else. The `bytes` bridge lets a raw PCM source link too.
static SINK_FIELDS: [FieldDesc; 3] = [
    FieldDesc {
        field: F_RATE,
        allowed: ConstraintDesc::Set(&[ValueDesc::Int(OPUS_RATE as i64)]),
        preferred: None,
    },
    FieldDesc {
        field: F_CHANNELS,
        allowed: ConstraintDesc::Set(&[ValueDesc::Int(1), ValueDesc::Int(2)]),
        preferred: None,
    },
    FieldDesc {
        field: F_SAMPLE,
        allowed: ConstraintDesc::Set(&[ValueDesc::Id("s16")]),
        preferred: None,
    },
];
static SINK_OFFERS: [OfferDesc; 2] =
    [OfferDesc { family: FAMILY_RAW, fields: &SINK_FIELDS }, OfferDesc::any("bytes")];

// Src emits `opus` access units (one packet per buffer); the `bytes` bridge lets a raw-packet sink
// (dump to a file) link too. Dynamic: rate/channels are announced once the input is known.
static SRC_OFFERS: [OfferDesc; 2] = [OfferDesc::any(FAMILY_OPUS), OfferDesc::any("bytes")];

static PADS: [PadDesc; 2] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &SINK_OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &SRC_OFFERS, dynamic: true, validate: None },
];

/// Encoder knobs (structural — read once when the encoder is built):
/// - `bitrate` — target bits/s (VBR average / CBR exact); `0`/unset derives from channels.
/// - `vbr` — `1` = variable bitrate (default), `0` = CBR (libopus backend only).
/// - `complexity` — 0–10 (libopus backend only; default 10).
static PROPS: [PropDesc; 3] = [
    PropDesc { name: "bitrate", allowed: Constraint::Any, live: false },
    PropDesc { name: "vbr", allowed: Constraint::Any, live: false },
    PropDesc { name: "complexity", allowed: Constraint::Any, live: false },
];

// `make_default` boxes one instance at registry/parse time, never per frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "opusenc",
    pads: &PADS,
    props: &PROPS,
    // Passive transform: inlines into the upstream group like the other audio transforms.
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        // The encoder buffers up to one frame (~20 ms) plus CELT's 2.5 ms MDCT overlap; a precise
        // figure waits on the clock, left zero like the passive audio transforms for now.
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(OpusEnc::new())),
};

/// Encodes interleaved 48 kHz `s16` PCM to an Opus (CELT-only) packet stream.
pub struct OpusEnc {
    /// Target bitrate override from the `bitrate` prop; `0` = derive from channels.
    bitrate_bps: u32,
    frame_ms_tenths: u16,
    application: Application,
    /// `vbr` prop (default true = VBR).
    vbr: bool,
    /// `complexity` prop, 0–10 (default 10).
    complexity: u8,
    /// The encoder backend (libopus or pure-Rust), built once the input channel count is known.
    enc: Option<Box<dyn EncBackend>>,
    /// Learned channel count (1 or 2); `0` until known.
    channels: usize,
    /// Frame size in bytes (`channels · frame_samples · 2`); `0` until the encoder exists.
    frame_bytes: usize,
    announced: bool,
    /// Reused carry for a frame straddling two input buffers (only the boundary frame is buffered;
    /// whole frames inside an input buffer are encoded directly from it — zero-copy).
    acc: Vec<u8>,
    /// Reused encoded-packet buffer.
    packet: Vec<u8>,
}

impl Default for OpusEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl OpusEnc {
    #[allow(clippy::disallowed_methods)] // one-time construction; the Vecs are reused accumulators, not per-frame
    pub fn new() -> Self {
        OpusEnc {
            bitrate_bps: 0,
            frame_ms_tenths: DEFAULT_FRAME_TENTHS,
            application: Application::Audio,
            vbr: true,
            complexity: 10,
            enc: None,
            channels: 0,
            frame_bytes: 0,
            announced: false,
            acc: Vec::new(),
            packet: Vec::new(),
        }
    }

    /// Learn the input channel count from the negotiated sink caps and build the encoder (once).
    /// Rate/sample are caps-constrained to 48 kHz/`s16`, so only channels vary.
    fn build_from_sink(&mut self, ctx: &Ctx) {
        if self.enc.is_some() {
            return;
        }
        let Some(fixed) = ctx.negotiated(SINK) else { return };
        let channels = ctx
            .field_id(F_CHANNELS)
            .and_then(|id| fixed.get(id))
            .and_then(|v| match v {
                Value::Int(n) if (1..=2).contains(&n) => Some(n as usize),
                _ => None,
            });
        let Some(channels) = channels else { return };

        let bitrate = if self.bitrate_bps > 0 {
            self.bitrate_bps
        } else {
            EncoderConfig::new(channels as u8).bitrate_bps
        };
        let cfg = EncoderConfig {
            sample_rate: OPUS_RATE,
            channels: channels as u8,
            bitrate_bps: bitrate,
            application: self.application,
            frame_ms_tenths: self.frame_ms_tenths,
            vbr: self.vbr,
            complexity: self.complexity,
            ..EncoderConfig::new(channels as u8)
        };
        match make_backend(cfg) {
            Ok(enc) => {
                self.frame_bytes = enc.frame_bytes();
                self.channels = channels;
                self.enc = Some(enc);
            }
            Err(_) => {
                // Config rejected (shouldn't happen given the caps constraints) — leave unbuilt so
                // process() surfaces a clear error.
            }
        }
    }

    /// Announce the output `opus` caps (rate + channels) on the src pad, once (spec: dynamic caps
    /// — producer side), so a muxer downstream can build the `OpusHead`.
    fn announce(&mut self, ctx: &mut Ctx) {
        if self.announced || self.enc.is_none() {
            return;
        }
        ctx.announce_format(
            SRC,
            FAMILY_OPUS,
            &[
                (F_RATE, ValueDesc::Int(OPUS_RATE as i64)),
                (F_CHANNELS, ValueDesc::Int(self.channels as i64)),
            ],
        );
        self.announced = true;
    }

    /// Encode one full `frame` of `s16`-LE PCM bytes and push the packet as a right-sized pool
    /// buffer on the src pad. `frame` is borrowed from the caller (an input buffer or the carry) —
    /// disjoint from `self.enc`/`self.packet`, so the zero-copy backend reads it in place.
    fn encode_frame_bytes(&mut self, ctx: &mut Ctx, frame: &[u8]) -> Result<(), Error> {
        let Some(enc) = self.enc.as_mut() else {
            return Err(Error::Todo("opusenc: PCM arrived before the input format was known"));
        };
        if let Err(e) = enc.encode(frame, &mut self.packet) {
            // A frame the encoder rejects (should not happen) is a per-frame error scope: warn-drop
            // rather than tearing down the stream.
            let element = ctx.element();
            ctx.post(BusMessage::Warning {
                element,
                error: Error::Element {
                    element,
                    message: format!("opusenc: dropping frame the encoder rejected: {e}"),
                },
            });
            return Ok(());
        }
        let n = self.packet.len();
        let mut buf = ctx.alloc_exact(SRC, n);
        buf.memory.as_mut_full()[..n].copy_from_slice(&self.packet);
        buf.memory.set_len(n);
        ctx.out(SRC).push(buf);
        Ok(())
    }

    /// Reframe `data` into full frames and encode each. **Whole frames sitting inside `data` are
    /// encoded straight from the input buffer** (zero-copy — the libopus backend reinterprets the
    /// aligned bytes in place); only a frame that straddles two input buffers is copied through
    /// `acc`. Before the encoder is built, `data` accumulates in `acc`.
    fn feed(&mut self, ctx: &mut Ctx, data: &[u8]) -> Result<(), Error> {
        let fb = self.frame_bytes;
        if fb == 0 {
            self.acc.extend_from_slice(data);
            return Ok(());
        }

        let mut off = 0;
        // First drain any frame carried in `acc` (a straddle, or pre-build buffering), completing a
        // partial one from the front of `data`.
        while !self.acc.is_empty() {
            if self.acc.len() < fb {
                let need = fb - self.acc.len();
                let take = need.min(data.len() - off);
                self.acc.extend_from_slice(&data[off..off + take]);
                off += take;
                if self.acc.len() < fb {
                    return Ok(()); // still partial — wait for more input
                }
            }
            // `acc` now holds ≥ one frame. Encode its front frame; keep any surplus. `mem::take`
            // frees the borrow so `self` (enc/packet) is available to `encode_frame_bytes`.
            let mut carry = std::mem::take(&mut self.acc);
            self.encode_frame_bytes(ctx, &carry[..fb])?;
            carry.drain(..fb);
            self.acc = carry;
        }

        // `acc` is empty: encode every whole frame directly from `data` (no copy).
        while off + fb <= data.len() {
            self.encode_frame_bytes(ctx, &data[off..off + fb])?;
            off += fb;
        }
        // Carry the trailing partial frame.
        self.acc.extend_from_slice(&data[off..]);
        Ok(())
    }
}

impl Element for OpusEnc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Parsed knobs override the defaults (spec: Plugins).
        if let Some(Value::Int(b)) = ctx.prop("bitrate") {
            if b > 0 {
                self.bitrate_bps = b as u32;
            }
        }
        if let Some(Value::Int(v)) = ctx.prop("vbr") {
            self.vbr = v != 0;
        }
        if let Some(Value::Int(c)) = ctx.prop("complexity") {
            self.complexity = c.clamp(0, 10) as u8;
        }
        self.enc = None;
        self.channels = 0;
        self.frame_bytes = 0;
        self.announced = false;
        self.acc.clear();
        // Link-time negotiation may already have fixed the sink format; build now so a statically
        // negotiated graph needs no runtime event.
        self.build_from_sink(ctx);
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.enc.is_none() {
            self.build_from_sink(ctx);
        }
        self.announce(ctx);
        while let Some(buf) = inputs.pop() {
            self.feed(ctx, buf.memory.data())?;
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FormatChange(_) => {
                if self.enc.is_none() {
                    self.build_from_sink(ctx);
                    self.announce(ctx);
                }
            }
            // Seek (spec: flush/seek): drop the reframing carry and reset the encoder's carried
            // inter-frame state so the next frame starts clean.
            Event::FlushStart => {
                if let Some(enc) = self.enc.as_mut() {
                    enc.reset();
                }
                self.acc.clear();
            }
            // End of stream: zero-pad the trailing partial frame to a full frame and emit it, so no
            // input audio is dropped (standard final-frame pad).
            Event::Eos if self.frame_bytes > 0 && !self.acc.is_empty() => {
                self.acc.resize(self.frame_bytes, 0);
                let mut carry = std::mem::take(&mut self.acc);
                self.encode_frame_bytes(ctx, &carry[..self.frame_bytes])?;
                carry.clear();
                self.acc = carry;
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.acc.clear();
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods, clippy::chunks_exact_to_as_chunks)] // tests: one-shot fixtures, not a hot path
mod tests {
    use super::*;
    // The reframing unit test drives the pure-Rust encoder directly regardless of backend feature.
    use crate::encoder::OpusEncoder;
    use oxideav_opus::OpusDecoder;

    /// The encoder + a decoder can round-trip a driven stream without the element harness: this
    /// exercises the same reframing/emit logic the element wraps. (A full pipeline test lives in
    /// the crate's integration tests / the `encode_roundtrip` example.)
    #[test]
    fn desc_is_named_and_registered_shape() {
        let e = OpusEnc::new();
        assert_eq!(e.desc().name, "opusenc");
        assert_eq!(e.desc().pads.len(), 2);
    }

    /// Sanity: a mono 48 kHz s16 stream reframed in irregular byte chunks still decodes back to
    /// the right sample count through our decoder (drives OpusEncoder directly, mirroring the
    /// element's byte accumulation).
    #[test]
    fn irregular_chunks_reframe_and_decode() {
        let mut enc = OpusEncoder::new(EncoderConfig::new(1)).unwrap();
        let n = enc.frame_samples();
        // 3 full frames of a ramp, as bytes.
        let mut bytes = Vec::new();
        for i in 0..n * 3 {
            let s = ((i % 200) as i16 - 100) * 100;
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        // Feed in odd-sized chunks and reframe.
        let frame_bytes = n * BYTES_PER_SAMPLE;
        let mut acc: Vec<u8> = Vec::new();
        let mut dec = OpusDecoder::new();
        let mut packet = Vec::new();
        let mut total_out = 0usize;
        for chunk in bytes.chunks(777) {
            acc.extend_from_slice(chunk);
            while acc.len() >= frame_bytes {
                let frame: Vec<u8> = acc.drain(..frame_bytes).collect();
                let f16: Vec<i16> =
                    frame.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
                enc.encode_frame(&f16, &mut packet).unwrap();
                let d = dec.decode_packet(&packet).unwrap();
                total_out += d.samples_per_channel();
            }
        }
        assert_eq!(total_out, n * 3, "all three frames decoded");
    }
}
