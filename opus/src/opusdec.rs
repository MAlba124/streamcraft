//! `opusdec` — the Opus (RFC 6716 / RFC 7845) decoder element. The decode backend is selected by
//! the `libopus` feature (default): the reference **libopus** C decoder (`libopus-sys`), or the
//! pure-Rust `oxideav-opus` decoder with `--no-default-features` — see the [`DecBackend`] trait.
//! One **Opus packet per input buffer** on the `opus` sink pad; decoded interleaved `s16` PCM at
//! 48 kHz leaves on the `audio/raw` src pad.
//!
//! The concrete `audio/raw` format (channels) is data-dependent, so the src pad is `dynamic` and
//! the format is announced at runtime via [`Ctx::announce_format`] — from the `OpusHead` (RFC
//! 7845 §5.1) identification packet when the container front-loads it (Ogg-Opus), else from the
//! first decoded packet. An `OpusTags` comment packet is consumed silently. The first `pre_skip`
//! samples (RFC 7845 §4.2) are trimmed and the header's `output_gain` applied.
//!
//! Output buffers are stamped on the decoder's own 48 kHz sample grid — see [`OpusDec::pts_now`]
//! for why the decoder, not the container, owns this timeline — and that counter is re-based to
//! the seek target on `FlushStart`.
//!
//! Backpressure mirrors `flacdec`: decoded PCM is emitted straight into pool buffers with a
//! bounded pending carry, so a slow sink cannot make the decoder run ahead and balloon the pool.
//! Opus is always 48 kHz on decode; sample-rate conversion is a downstream `audioresample`.

// `OpusHead`/`OpusTags`/gain are RFC 7845 container metadata (not decode), so they stay pure-Rust
// regardless of the decode backend.
use oxideav_opus::{apply_output_gain, OpusHead, OPUS_HEAD_MAGIC};

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

/// The Opus decoder backend — selected at build time: the reference **libopus** C decoder
/// (`libopus` feature, default) or the pure-Rust `oxideav-opus` decoder (`--no-default-features`).
/// Both decode one packet into a reused `i16` buffer and report the output channel count.
trait DecBackend: Send {
    /// Fix the output channel count from an `OpusHead` hint (libopus creates a fixed-width decoder;
    /// the pure-Rust decoder learns per-packet and ignores this).
    fn set_channels(&mut self, channels: usize);
    /// Decode one packet into `pcm` (interleaved `s16`, contents replaced); returns channel count.
    fn decode_into(&mut self, packet: &[u8], pcm: &mut Vec<i16>) -> Result<u8, String>;
    /// Reset carried inter-packet state (seek).
    fn reset(&mut self);
}

#[cfg(feature = "libopus")]
impl DecBackend for crate::libopus::LibopusDecoder {
    fn set_channels(&mut self, channels: usize) {
        crate::libopus::LibopusDecoder::set_channels(self, channels)
    }
    fn decode_into(&mut self, packet: &[u8], pcm: &mut Vec<i16>) -> Result<u8, String> {
        crate::libopus::LibopusDecoder::decode_packet_into(self, packet, pcm)
    }
    fn reset(&mut self) {
        crate::libopus::LibopusDecoder::reset(self)
    }
}

#[cfg(not(feature = "libopus"))]
impl DecBackend for oxideav_opus::OpusDecoder {
    fn set_channels(&mut self, _channels: usize) {} // learns channels per packet
    fn decode_into(&mut self, packet: &[u8], pcm: &mut Vec<i16>) -> Result<u8, String> {
        oxideav_opus::OpusDecoder::decode_packet_into(self, packet, pcm).map_err(|e| format!("{e:?}"))
    }
    fn reset(&mut self) {
        oxideav_opus::OpusDecoder::reset(self)
    }
}

#[allow(clippy::disallowed_methods)] // one-time construction at registry/parse time
fn make_decoder() -> Box<dyn DecBackend> {
    #[cfg(feature = "libopus")]
    {
        Box::new(crate::libopus::LibopusDecoder::new())
    }
    #[cfg(not(feature = "libopus"))]
    {
        Box::new(oxideav_opus::OpusDecoder::new())
    }
}

/// Decodes an Opus packet stream to interleaved 48 kHz `s16` PCM.
pub struct OpusDec {
    dec: Box<dyn DecBackend>,
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
    /// Interchannel samples emitted so far: the stream position, and hence the timestamp, of
    /// the next buffer. Re-based to the seek target on `FlushStart`.
    samples_emitted: u64,
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
            dec: make_decoder(),
            announced: false,
            channels: 0,
            preskip_remaining: 0,
            output_gain_q7_8: 0,
            pending: Vec::new(),
            pending_pos: 0,
            samples_emitted: 0,
        }
    }

    /// PTS for the current `samples_emitted`, on the 48 kHz output grid from a zero base —
    /// the convention `flacdec`/`mp3dec` use for a stream whose container supplies no time.
    /// Opus needs it most in Ogg, where there is none to supply: RFC 3533 §4 is explicit that
    /// Ogg "has no concept of 'time'", and the page granule is an end-of-page position that
    /// says nothing about where a packet *starts*. The decoder is the first element that knows
    /// how many samples it has produced, so it owns this stream's time base and stamps it here.
    /// [`OUTPUT_RATE`] is a nonzero constant (RFC 6716 §2), so this needs no unknown-rate guard.
    fn pts_now(&self) -> Timestamp {
        Timestamp::from_nanos(
            (u128::from(self.samples_emitted) * 1_000_000_000 / OUTPUT_RATE as u128) as u64,
        )
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
            // Stamp the buffer on the decoder's own timeline, then advance it by the
            // interchannel samples just written (the same shape as `flacdec::emit_pending`).
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

    /// Decode one audio packet into the reused `pending` buffer (gain-applied, pre-skip-trimmed).
    /// Uses `decode_packet_into` so a steady decode does **no per-packet heap allocation** — the
    /// `pending` buffer's capacity is retained across packets (the profluens no-alloc patch, like
    /// `flacdec::pull_into` and `aacdec`'s reused `pending_pcm`). The RFC 7845 header/comment
    /// packets are handled by the caller.
    fn decode_audio(&mut self, ctx: &mut Ctx, packet: &[u8]) -> Result<(), Error> {
        // Decode straight into the reused `pending` buffer (its capacity is retained across
        // packets), so a steady decode drives no per-packet heap traffic on either backend: libopus
        // is allocation-free on the hot path (C99 VLA scratch); the pure-Rust decoder uses a
        // per-packet `DecodeBump` arena + a reused PVQ scratch (vendor/oxideav-opus/PROFLUENS-PATCHES.md).
        let channels = match self.dec.decode_into(packet, &mut self.pending) {
            Ok(ch) => (ch as usize).max(1),
            // Per-packet error scope (spec: error handling): warn-drop a corrupt packet rather
            // than tearing down the stream; a following clean packet resumes decode.
            Err(e) => {
                let element = ctx.element();
                ctx.post(BusMessage::Warning {
                    element,
                    error: Error::Element {
                        element,
                        message: format!("opusdec: dropping undecodable packet: {e}"),
                    },
                });
                self.pending.clear();
                self.pending_pos = 0;
                return Ok(());
            }
        };
        self.pending_pos = 0;
        self.announce(ctx, channels);
        if self.output_gain_q7_8 != 0 {
            apply_output_gain(&mut self.pending, self.output_gain_q7_8);
        }
        // Trim the RFC 7845 §4.2 pre-skip (per-channel samples) off the output front (only at
        // stream start, so the front drain is not a steady-state cost).
        if self.preskip_remaining > 0 {
            let spc = self.pending.len() / channels; // per-channel samples in this packet
            let drop_spc = self.preskip_remaining.min(spc as u32) as usize;
            self.pending.drain(0..drop_spc * channels);
            self.preskip_remaining -= drop_spc as u32;
        }
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
                    // Fix the decoder's output width from the header (so a mono-coded frame in a
                    // stereo stream still outputs stereo).
                    self.dec.set_channels(head.channel_count as usize);
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
                // …and re-base the sample counter the PTS is derived from. An Opus packet
                // carries no position of its own, and the Ogg page granule that would give one
                // is not visible this side of the demuxer — but the seek target *is* the
                // position the pipeline rebased its running time to, so taking it from there
                // is what keeps buffer timestamps and running time telling the same story.
                // Without it the post-seek audio is stamped with pre-seek times and appears to
                // jump backwards. The source reads `to_byte` from this same target; the
                // decoder, which owns this stream's time base, reads `to_time`.
                if let Some(t) = ctx.seek_target() {
                    if let Some(ns) = t.to_time.nanos() {
                        self.samples_emitted =
                            (u128::from(ns) * OUTPUT_RATE as u128 / 1_000_000_000) as u64;
                    }
                }
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
