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

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::bus::BusMessage;
use profluens_core::error::Error;
use profluens_core::event::{Event, TagList};
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

use crate::decoder::StreamDecoder;

// `audio/raw` family/field/value names. Kept as literals (not a dep on profluens-audio)
// so pf-flac stays core-only; they match that crate's convention, and since the pipeline
// interns by string the ids line up with any audio peer that uses the same names.
const FAMILY: &str = "audio/raw";
const F_RATE: &str = "rate";
const F_CHANNELS: &str = "channels";
const F_SAMPLE: &str = "sample";

/// How many FLAC frames `process()` decodes before yielding its output batch downstream.
/// Small on purpose: it bounds the batch so the first audio reaches the sink after ~one
/// frame's decode rather than after a whole read's worth (spec: latency #2). A FLAC frame is
/// ~0.1 s of audio, so a handful still leaves plenty of jitter buffer downstream.
const FRAMES_PER_BATCH: u32 = 2;

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
// Typed `audio/raw` first; then the `bytes` bridge, so a byte sink (`filesink` — dump
// decoded PCM to a file) still links and the announcement rides it tolerantly (spec:
// Formats — an `audio/raw` refinement riding a `bytes` bridge to a byte sink).
static SRC_OFFERS: [OfferDesc; 2] =
    [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }, OfferDesc::any("bytes")];
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

// `make_default` boxes one element instance at registry/parse time, never per frame.
#[allow(clippy::disallowed_methods)]
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
    // Name-constructible (spec: Plugins): the decoder is config-free — it learns its
    // format from the FLAC stream and announces it via dynamic caps.
    make_default: Some(|| Box::new(FlacDec::new())),
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
    /// Sample rate and channel count from STREAMINFO, known once the header is decoded —
    /// the two numbers the PTS is derived from. Zero until then.
    rate: u32,
    channels: usize,
    /// Interchannel samples emitted so far: the stream position, and hence the timestamp,
    /// of the next buffer. Re-based to the seek target on `FlushStart`.
    samples_emitted: u64,
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
            buf.pts = self.pts_now();
            self.samples_emitted += ((end - self.pending_pos) / self.channels.max(1)) as u64;
            buf.duration = self.pts_now().saturating_sub(buf.pts);
            ctx.out(PadId(1)).push(buf);
            self.pending_pos = end;
        }
        self.pending.clear();
        self.pending_pos = 0;
        Ok(true)
    }

    /// PTS for the current `samples_emitted`, on the sample grid from a zero base (the same
    /// convention `wavparse` and `mp3dec` use for an elementary audio stream: a stream with
    /// no container has no timestamps of its own, so its position *is* its time).
    /// `NONE` until STREAMINFO has been decoded and the rate is known.
    fn pts_now(&self) -> Timestamp {
        if self.rate == 0 {
            return Timestamp::NONE;
        }
        Timestamp::from_nanos(
            (u128::from(self.samples_emitted) * 1_000_000_000 / u128::from(self.rate)) as u64,
        )
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
        self.rate = info.sample_rate;
        self.channels = info.channels as usize;
        ctx.announce_format(
            PadId(1),
            FAMILY,
            &[
                (F_RATE, ValueDesc::Int(info.sample_rate as i64)),
                (F_CHANNELS, ValueDesc::Int(info.channels as i64)),
                (F_SAMPLE, ValueDesc::Id(sample_name(info.bits_per_sample))),
            ],
        );
        self.emit_tags(ctx);
        self.announced = true;
        Ok(())
    }

    /// Emit the stream's metadata tags once, GStreamer-style: an in-band [`Event::Tags`] travelling
    /// downstream (a muxer writes them into its container) **and** a [`BusMessage::Tags`] for the
    /// application (a player's "now playing" / cover art). No-op if the file carried no tags.
    fn emit_tags(&mut self, ctx: &mut Ctx) {
        // One-time clone of the parsed tags (header path) so the borrow of `self.dec` is released
        // before we touch `ctx` for the picture buffers below.
        let ft = self.dec.tags().clone();
        if ft.is_empty() {
            return;
        }
        let mut tags = TagList::new();
        for (key, value) in &ft.comments {
            tags.add(key, value);
        }
        // Cover art: copy each picture's bytes into a one-time buffer (a large image heap-allocates
        // via `alloc_exact`). The `TagList` clone below is a refcount bump, not a copy.
        for (mime, data) in &ft.pictures {
            let mut buf = ctx.alloc_exact(PadId(1), data.len());
            buf.memory.as_mut_full()[..data.len()].copy_from_slice(data);
            buf.memory.set_len(data.len());
            tags.add_picture(mime, buf.memory.clone());
        }
        let element = ctx.element();
        ctx.push_event(PadId(1), Event::Tags(tags.clone()));
        ctx.post(BusMessage::Tags { element, tags });
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
        self.rate = 0;
        self.channels = 0;
        self.samples_emitted = 0;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Flush any carry from a prior backpressured call before decoding more.
        if !self.emit_pending(ctx, true)? {
            // Pool full: yield without consuming more input, so the sink drains first.
            // Leaving input buffered keeps the group non-quiescent (backpressure).
            return Ok(Flow::Ok);
        }
        let mut emitted = 0u32;
        loop {
            // Decode the next buffered frame; if the decoder is hungry, feed it one input
            // buffer; when there is neither a frame nor more input, we are done this call.
            // Decode straight into the reused `pending` buffer — no per-frame allocation
            // (the decoder reuses its own scratch too).
            match self.dec.pull_into(&mut self.pending) {
                Ok(Some(())) => {
                    self.pending_pos = 0;
                    self.announce_if_needed(ctx)?; // first frame → header known
                    if !self.emit_pending(ctx, true)? {
                        return Ok(Flow::Ok); // pool filled mid-frame — carry the rest, yield
                    }
                    // Yield the batch after a few frames rather than draining all buffered
                    // input into one large batch: the first audio reaches the sink after ~one
                    // frame instead of after the whole batch is decoded — much lower latency,
                    // especially after a seek (spec: latency #2). The scheduler re-invokes us
                    // with the remaining input still buffered, so throughput is unaffected.
                    emitted += 1;
                    if emitted >= FRAMES_PER_BATCH {
                        return Ok(Flow::Ok);
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
        if matches!(event, Event::FlushStart) {
            // Seek (spec: flush/seek): drop the buffered bytes and the pending carry, and
            // arrange to re-sync to the next frame boundary on the bytes that arrive after
            // the source's byte-seek. STREAMINFO (and thus the announced format) is kept —
            // the header lives only at the file start, so a mid-stream seek carries none.
            self.dec.seek_reset();
            self.pending.clear();
            self.pending_pos = 0;
            // …and re-base the sample counter the PTS is derived from. A FLAC frame header
            // does carry a sample number, but the streaming decoder does not surface it,
            // and in any case the seek target is the position the *pipeline* rebased its
            // running time to — so taking it from there is what keeps buffer timestamps and
            // running time telling the same story. Without this the post-seek audio is
            // stamped with pre-seek times and appears to jump backwards. The source reads
            // `to_byte` from the same target; the decoder, which owns this stream's time
            // base, reads `to_time`.
            if let Some(t) = ctx.seek_target() {
                if let (Some(ns), true) = (t.to_time.nanos(), self.rate > 0) {
                    self.samples_emitted =
                        (u128::from(ns) * u128::from(self.rate) / 1_000_000_000) as u64;
                }
            }
            return Ok(());
        }
        if matches!(event, Event::Eos) {
            // End of stream: flush the carry and every remaining buffered frame through the
            // unbounded pool, so the tail is emitted even if we were backpressured here.
            self.emit_pending(ctx, false)?;
            loop {
                match self.dec.pull_into(&mut self.pending) {
                    Ok(Some(())) => {
                        self.pending_pos = 0;
                        self.announce_if_needed(ctx)?;
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
        self.rate = 0;
        self.channels = 0;
        self.samples_emitted = 0;
    }
}
