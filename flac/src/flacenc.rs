//! `flacenc` — the streamcraft element wrapping [`FlacEncoder`] (spec: Milestone
//! applications §3; Writing elements). A passive transform: raw **interleaved** PCM
//! buffers arrive on the sink pad, encoded FLAC bytes leave on the src pad. It inlines
//! into the upstream active element's group like [`passthrough`], so it never blocks.
//!
//! Audio parameters are constructor arguments ([`FlacEnc::new`]) because format
//! negotiation is being built in parallel and this element must not depend on it — the
//! `wavparse` upstream will eventually hand these over, but for now the caller wires
//! them (spec target: S16 mono/stereo at 44100/48000).
//!
//! It writes a **streaming**, forward-only FLAC stream (see
//! [`FlacEncoder::new_streaming`]): the STREAMINFO header is emitted once, up front,
//! with "unknown" (0) frame sizes and total-sample count — spec-legal (§8.2) and
//! required because a sequential sink (`filesink`) cannot be back-patched. The output
//! is a fully valid, streamable FLAC file.

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::encoder::{FlacEncoder, SampleFormat};

/// Block size (interchannel samples) the encoder aims for per frame. 4096 is the
/// common libFLAC default and within the streamable subset for <=48 kHz (§7).
const BLOCK_SIZE: u32 = 4096;

static PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &[],
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &[],
        dynamic: false,
        validate: None,
    },
];

static DESC: ElementDesc = ElementDesc {
    name: "flacenc",
    pads: &PADS,
    props: &[],
    // Passive: pure transform, inlines into the upstream group (spec: Scheduling).
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

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        // Build the encoder and stash the header to emit on the first `process` (we
        // can only push output when we hold an `out` batch mid-`process`).
        let (encoder, _header) =
            FlacEncoder::new_streaming(self.sample_rate, self.channels, self.format, BLOCK_SIZE)
                .map_err(|e| Error::Resource(format!("flacenc: {e:?}")))?;
        self.encoder = Some(encoder);
        self.header_sent = false;
        self.carry.clear();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Emit the STREAMINFO header exactly once, before any frames.
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

        while let Some(buf) = inputs.pop() {
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

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // Any carried bytes are an incomplete final interchannel sample (ragged input)
        // — dropping them keeps the stream well-formed. FLAC needs no EOS marker (§9).
        self.encoder = None;
        self.carry.clear();
    }
}
