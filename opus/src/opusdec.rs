//! `opusdec` — the Opus (RFC 6716 / RFC 7845) decoder element, backed by the pure-Rust
//! `oxideav-opus` (git master; see the crate's lib.rs history for why the published 0.0.13 is a
//! non-decoding scaffold). One **Opus packet per input buffer** on the `opus` sink pad; decoded
//! interleaved `s16` PCM at 48 kHz leaves on the `audio/raw` src pad.
//!
//! The concrete `audio/raw` format (channels) is data-dependent, so the src pad is `dynamic` and
//! the format is announced at runtime via [`Ctx::announce_format`] — from the `OpusHead` (RFC
//! 7845 §5.1) identification packet when the container front-loads it (Ogg-Opus), else from the
//! first decoded packet. An `OpusTags` comment packet is consumed silently. The first `pre_skip`
//! samples (RFC 7845 §4.2) are trimmed and the header's `output_gain` applied.
//!
//! Backpressure mirrors `flacdec`: decoded PCM is emitted straight into pool buffers with a
//! bounded pending carry, so a slow sink cannot make the decoder run ahead and balloon the pool.
//! Opus is always 48 kHz on decode; sample-rate conversion is a downstream `audioresample`.

use oxideav_opus::{apply_output_gain, OpusDecoder, OpusHead, OPUS_HEAD_MAGIC};

use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

// `audio/raw` family/field/value names (string-interned, so they line up with any audio peer
// using the same convention — kept as literals so pf-opus stays core-only, like flacdec).
const FAMILY: &str = "audio/raw";
const F_RATE: &str = "rate";
const F_CHANNELS: &str = "channels";
const F_SAMPLE: &str = "sample";

/// Opus decode output is always 48 kHz (RFC 6716 §2) interleaved `s16`.
const OUTPUT_RATE: i64 = 48_000;
const BYTES_PER_SAMPLE: usize = 2;

static SRC_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_RATE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_CHANNELS, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc {
        field: F_SAMPLE,
        allowed: ConstraintDesc::Set(&[ValueDesc::Id("s16")]),
        preferred: None,
    },
];
// Typed `audio/raw` first, then the `bytes` bridge so a byte sink (dump PCM to a file) still
// links and the announcement rides it tolerantly (as flacdec does).
static SRC_OFFERS: [OfferDesc; 2] =
    [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }, OfferDesc::any("bytes")];
// Opus access units arrive one packet per buffer, tagged `opus` by the demuxer (a `bytes`
// bridge is also accepted for a raw-packet source).
static SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("opus"), OfferDesc::any("bytes")];

static PADS: [PadDesc; 2] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &SINK_OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &SRC_OFFERS, dynamic: true, validate: None },
];

// `make_default` boxes one instance at registry/parse time, never per frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "opusdec",
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
    // Config-free: it learns channels from OpusHead / the first packet and announces via
    // dynamic caps (spec: Plugins — name-constructible).
    make_default: Some(|| Box::new(OpusDec::new())),
};

/// Decodes an Opus packet stream to interleaved 48 kHz `s16` PCM.
pub struct OpusDec {
    dec: OpusDecoder,
    announced: bool,
    /// Channel count of the announced format (from `OpusHead` or the first decoded packet).
    channels: usize,
    /// Per-channel samples still to trim from the output front (RFC 7845 §4.2 pre-skip).
    preskip_remaining: u32,
    /// `OpusHead` output gain (Q7.8 dB); applied to every decoded sample, `0` = unity.
    output_gain_q7_8: i16,
    /// Decoded interleaved PCM not yet emitted — the bounded backpressure carry.
    pending: Vec<i16>,
    pending_pos: usize,
}

impl Default for OpusDec {
    fn default() -> Self {
        Self::new()
    }
}

impl OpusDec {
    #[allow(clippy::disallowed_methods)] // one-time construction at registry/parse time
    pub fn new() -> Self {
        OpusDec {
            dec: OpusDecoder::new(),
            announced: false,
            channels: 0,
            preskip_remaining: 0,
            output_gain_q7_8: 0,
            pending: Vec::new(),
            pending_pos: 0,
        }
    }

    /// Announce the runtime `audio/raw` format once (spec: dynamic caps): 48 kHz, `s16`, the
    /// packet's channel count.
    fn announce(&mut self, ctx: &mut Ctx, channels: usize) {
        if self.announced {
            return;
        }
        self.channels = channels.max(1);
        ctx.announce_format(
            PadId(1),
            FAMILY,
            &[
                (F_RATE, ValueDesc::Int(OUTPUT_RATE)),
                (F_CHANNELS, ValueDesc::Int(self.channels as i64)),
                (F_SAMPLE, ValueDesc::Id("s16")),
            ],
        );
        self.announced = true;
    }

    /// Emit the pending PCM as interleaved little-endian `s16` **straight into pool buffers**
    /// (no per-packet intermediate). `bounded` uses the capped pool and returns `Ok(false)` when
    /// it is full (backpressure); `!bounded` uses the unbounded pool to guarantee the EOS tail
    /// flushes. `Ok(true)` once fully drained.
    #[allow(clippy::disallowed_methods)] // pool-backed; the pending Vec is a reused carry, not per-frame
    fn emit_pending(&mut self, ctx: &mut Ctx, bounded: bool) -> Result<bool, Error> {
        while self.pending_pos < self.pending.len() {
            let mut buf = if bounded {
                match ctx.try_alloc(PadId(1)) {
                    Some(b) => b,
                    None => return Ok(false), // pool full → backpressure
                }
            } else {
                ctx.alloc(PadId(1))
            };
            let per = (buf.memory.capacity() / BYTES_PER_SAMPLE).max(1); // whole i16 samples per buffer
            let end = (self.pending_pos + per).min(self.pending.len());
            let dst = buf.memory.as_mut_full();
            let mut n = 0;
            for &s in &self.pending[self.pending_pos..end] {
                dst[n..n + BYTES_PER_SAMPLE].copy_from_slice(&s.to_le_bytes());
                n += BYTES_PER_SAMPLE;
            }
            buf.memory.set_len(n);
            ctx.out(PadId(1)).push(buf);
            self.pending_pos = end;
        }
        self.pending.clear();
        self.pending_pos = 0;
        Ok(true)
    }

    /// Decode one audio packet into `pending` (gain-applied, pre-skip-trimmed). The RFC 7845
    /// header/comment packets are handled by the caller.
    fn decode_audio(&mut self, ctx: &mut Ctx, packet: &[u8]) -> Result<(), Error> {
        let audio = match self.dec.decode_packet(packet) {
            Ok(a) => a,
            // Per-packet error scope (spec: error handling): warn-drop a corrupt packet rather
            // than tearing down the stream; a following clean packet resumes decode.
            Err(e) => {
                let element = ctx.element();
                ctx.post(BusMessage::Warning {
                    element,
                    error: Error::Element {
                        element,
                        message: format!("opusdec: dropping undecodable packet: {e:?}"),
                    },
                });
                return Ok(());
            }
        };
        let ch = (audio.channels as usize).max(1);
        self.announce(ctx, ch);
        let mut pcm = audio.pcm;
        if self.output_gain_q7_8 != 0 {
            apply_output_gain(&mut pcm, self.output_gain_q7_8);
        }
        // Trim the RFC 7845 §4.2 pre-skip (per-channel samples) off the output front.
        if self.preskip_remaining > 0 {
            let spc = pcm.len() / ch; // per-channel samples in this packet
            let drop_spc = self.preskip_remaining.min(spc as u32) as usize;
            pcm.drain(0..drop_spc * ch);
            self.preskip_remaining -= drop_spc as u32;
        }
        self.pending = pcm;
        self.pending_pos = 0;
        Ok(())
    }
}

impl Element for OpusDec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        *self = OpusDec::new();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Flush any carry from a prior backpressured call before decoding more.
        if !self.emit_pending(ctx, true)? {
            return Ok(Flow::Ok); // pool full: yield, leaving input buffered (backpressure)
        }
        while let Some(buf) = inputs.pop() {
            let data = buf.memory.data();
            if data.len() >= 8 && &data[..8] == OPUS_HEAD_MAGIC.as_slice() {
                // RFC 7845 §5.1 identification header: configure channels / pre-skip / gain.
                if let Ok(head) = OpusHead::parse(data) {
                    self.preskip_remaining = head.pre_skip as u32;
                    self.output_gain_q7_8 = head.output_gain_q7_8;
                    self.announce(ctx, head.channel_count as usize);
                }
                continue;
            }
            if data.len() >= 8 && &data[..8] == b"OpusTags" {
                continue; // RFC 7845 §5.2 comment header — consumed silently
            }
            self.decode_audio(ctx, data)?;
            if !self.emit_pending(ctx, true)? {
                return Ok(Flow::Ok); // pool filled mid-packet — carry the rest, yield
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Seek (spec: flush/seek): reset the decoder's inter-packet state and drop the
            // carry. The announced format is kept; pre-skip is NOT re-armed mid-stream (it is a
            // stream-start artifact only).
            Event::FlushStart => {
                self.dec.reset();
                self.pending.clear();
                self.pending_pos = 0;
            }
            // End of stream: flush the carry through the unbounded pool so the tail is emitted
            // even if this element was backpressured (Opus packets are self-contained — no
            // decoder-buffered frames to drain).
            Event::Eos => {
                self.emit_pending(ctx, false)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.pending.clear();
        self.pending_pos = 0;
    }
}
