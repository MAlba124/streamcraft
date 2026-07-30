//! `adtsparse` — de-frame a bare **ADTS** AAC byte stream (a raw `.aac`/`.adts` file) into the
//! `AudioSpecificConfig` + raw access units that [`crate::AacDec`] expects (spec: Formats —
//! a byte source has no framing; this is the elementary-stream counterpart of a demuxer's
//! reframing). ADTS carries the codec config **in every frame header** rather than out of
//! band, so a container's CodecPrivate/ASC head is absent; this element synthesizes it.
//!
//! Pipeline: `filesrc(.aac) ! adtsparse ! aacdec ! …`. On the sink pad a raw ADTS byte stream
//! arrives; on the src pad the element emits, in order:
//! 1. **one** `AudioSpecificConfig` buffer (ISO/IEC 14496-3 §1.6.2.1), synthesized from the
//!    first ADTS header — this is the "first buffer = ASC head" `aacdec` keys on;
//! 2. then **one raw AAC access unit per buffer** — each ADTS frame's payload with its 7- or
//!    9-byte header stripped.
//!
//! The framer is self-syncing (ISO/IEC 14496-3 §1.A.3.2.1): it scans for the 12-bit `0xFFF`
//! syncword with a layer field of `00` (the ADTS marker, distinguishing it from an MPEG-audio
//! frame — the same test [`crate` typefind] uses), validates the frame length against the
//! buffered bytes, and resyncs past a bad byte rather than trusting a stray sync. Untrusted
//! input is a P0: every field read is bounds-checked; a malformed header advances one byte and
//! rescans, never panics.

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

const SRC: PadId = PadId(1);

// Sink takes the raw ADTS byte stream (`filesrc` speaks `bytes`); src emits the AAC family
// `aacdec` consumes (its sink offers `aac`), plus the `bytes` escape.
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_OFFERS: [OfferDesc; 2] = [OfferDesc::any("aac"), OfferDesc::any("bytes")];

static PADS: [PadDesc; 2] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &SINK_OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &SRC_OFFERS, dynamic: false, validate: None },
];

// COLD: `make_default` boxes one element instance at pipeline construction, never per frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "adtsparse",
    pads: &PADS,
    props: &[],
    // Passive: a pure byte→packet transform, inlines into the upstream group.
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Config-free (spec: Plugins — `parse("filesrc ! adtsparse ! aacdec ! …")`).
    make_default: Some(|| Box::new(AdtsParse::new())),
};

/// De-frames an ADTS byte stream into an ASC head + raw AAC access units.
#[derive(Default)]
pub struct AdtsParse {
    /// Accumulated bytes not yet consumed as whole ADTS frames (a frame may straddle two
    /// input buffers; the tail carries to the next `process`).
    buf: Vec<u8>,
    /// Whether the synthesized `AudioSpecificConfig` head has been emitted (once, before any AU).
    configured: bool,
}

/// One parsed ADTS fixed header (ISO/IEC 14496-3 §1.A.2.2.1 `adts_fixed_header`).
struct AdtsHeader {
    /// MPEG-4 Audio Object Type = ADTS `profile` + 1 (§1.A.2.2.1: profile is `AOT − 1`).
    aot: u8,
    /// Sampling-frequency index (§1.A.2.2.1 `sampling_frequency_index`, 0..=12 valid).
    freq_index: u8,
    /// Channel configuration (§1.A.2.2.1 `channel_configuration`, 1..=7).
    channel_config: u8,
    /// Total frame length in bytes, header included (§1.A.2.2.2 `aac_frame_length`, 13 bits).
    frame_length: usize,
    /// Header size: 7 bytes, or 9 when a 2-byte CRC follows (`protection_absent == 0`).
    header_len: usize,
}

impl AdtsParse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse an ADTS frame header at `b[0..]` if `b` holds the 7 fixed-header bytes and the
    /// syncword + layer mark a genuine ADTS frame. Returns `None` on too-few bytes or a
    /// non-ADTS pattern (the caller then advances one byte and rescans).
    fn parse_header(b: &[u8]) -> Option<AdtsHeader> {
        if b.len() < 7 {
            return None;
        }
        // 12-bit syncword 0xFFF, then MPEG version (ignored) + 2-bit layer that MUST be `00`
        // for AAC (§1.A.2.2.1) — the same discriminator that separates ADTS from MP3.
        if b[0] != 0xFF || (b[1] & 0xF0) != 0xF0 || (b[1] >> 1) & 0x03 != 0 {
            return None;
        }
        let protection_absent = b[1] & 0x01;
        let profile = (b[2] >> 6) & 0x03;
        let freq_index = (b[2] >> 2) & 0x0F;
        let channel_config = ((b[2] & 0x01) << 2) | ((b[3] >> 6) & 0x03);
        // 13-bit aac_frame_length spanning bytes 3..6.
        let frame_length =
            (((b[3] & 0x03) as usize) << 11) | ((b[4] as usize) << 3) | ((b[5] as usize) >> 5);
        let header_len = if protection_absent == 1 { 7 } else { 9 };
        // A frame must at least contain its header, and only indices 0..=12 are real rates.
        if frame_length < header_len || freq_index > 12 {
            return None;
        }
        Some(AdtsHeader {
            aot: profile + 1,
            freq_index,
            channel_config,
            frame_length,
            header_len,
        })
    }

    /// Synthesize the 2-byte `AudioSpecificConfig` (ISO/IEC 14496-3 §1.6.2.1) from an ADTS
    /// header: `AOT`(5) | `samplingFrequencyIndex`(4) | `channelConfiguration`(4) | 000(3) —
    /// the GA-specific trailing flags (`frameLengthFlag`/`dependsOnCoreCoder`/`extensionFlag`)
    /// are zero for a plain ADTS stream. Valid for `aot < 31` and `freq_index < 15`, which ADTS
    /// guarantees (both are narrower fields there).
    fn synth_asc(h: &AdtsHeader) -> [u8; 2] {
        let b0 = (h.aot << 3) | (h.freq_index >> 1);
        let b1 = ((h.freq_index & 1) << 7) | (h.channel_config << 3);
        [b0, b1]
    }

    /// Push `payload` as one buffer on the src pad (chunked to the pool slot — an AU or the
    /// tiny ASC always fits, but the chunk loop matches the crate's other emitters).
    fn emit(ctx: &mut Ctx, payload: &[u8]) {
        let mut off = 0;
        while off < payload.len() {
            let mut b = ctx.alloc(SRC);
            let cap = b.memory.capacity();
            let n = cap.min(payload.len() - off);
            b.memory.as_mut_full()[..n].copy_from_slice(&payload[off..off + n]);
            b.memory.set_len(n);
            ctx.out(SRC).push(b);
            off += n;
        }
    }

    /// Consume as many whole ADTS frames as `self.buf` holds, emitting the ASC head (once) and
    /// each frame's raw AU. Leaves a partial trailing frame in `self.buf`.
    fn drain(&mut self, ctx: &mut Ctx) {
        let mut pos = 0;
        loop {
            // Find the next syncword from `pos`.
            let Some(sync) = self.buf[pos..].windows(2).position(|w| w[0] == 0xFF && (w[1] & 0xF6) == 0xF0)
            else {
                // No sync in the remainder — keep only a trailing byte that might begin one.
                pos = self.buf.len().saturating_sub(1);
                break;
            };
            let start = pos + sync;
            let Some(h) = Self::parse_header(&self.buf[start..]) else {
                // A false sync (or not enough bytes yet). If we have ≥7 bytes it was genuinely
                // bad — skip this byte and rescan; otherwise wait for more input.
                if self.buf.len() - start >= 7 {
                    pos = start + 1;
                    continue;
                }
                pos = start;
                break;
            };
            if self.buf.len() - start < h.frame_length {
                // The frame is not fully buffered yet — carry from its start.
                pos = start;
                break;
            }
            let frame = &self.buf[start..start + h.frame_length];
            if !self.configured {
                Self::emit(ctx, &Self::synth_asc(&h));
                self.configured = true;
            }
            // The raw access unit: the frame past its (7- or 9-byte) header.
            Self::emit(ctx, &frame[h.header_len..]);
            pos = start + h.frame_length;
        }
        // Drop everything consumed; keep the unparsed tail.
        if pos > 0 {
            self.buf.drain(..pos);
        }
    }
}

impl Element for AdtsParse {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.buf.clear();
        self.configured = false;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            self.buf.extend_from_slice(buf.memory.data());
            self.drain(ctx);
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    // COLD: teardown, once per stream — releases the carry buffer's backing storage.
    #[allow(clippy::disallowed_methods)]
    fn stop(&mut self, _ctx: &mut Ctx) {
        self.buf = Vec::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal 7-byte-header ADTS frame with `payload` and the given config, protection
    /// absent (no CRC). Used to drive header/ASC unit checks without a pipeline.
    fn adts_frame(profile: u8, freq_index: u8, chan: u8, payload: &[u8]) -> Vec<u8> {
        let frame_len = 7 + payload.len();
        let mut f = vec![
            0xFF,
            0xF1, // sync + MPEG-4 + layer 00 + protection_absent=1
            (profile << 6) | (freq_index << 2) | (chan >> 2),
            ((chan & 0x03) << 6) | ((frame_len >> 11) as u8 & 0x03),
            ((frame_len >> 3) & 0xFF) as u8,
            (((frame_len & 0x07) << 5) as u8) | 0x1F,
            0xFC,
        ];
        f.extend_from_slice(payload);
        f
    }

    #[test]
    fn parses_header_fields() {
        let f = adts_frame(1 /*LC*/, 4 /*44.1k*/, 2 /*stereo*/, &[0xDE, 0xAD]);
        let h = AdtsParse::parse_header(&f).expect("valid ADTS header");
        assert_eq!(h.aot, 2, "AOT = profile + 1 (LC = 2)");
        assert_eq!(h.freq_index, 4);
        assert_eq!(h.channel_config, 2);
        assert_eq!(h.frame_length, 9);
        assert_eq!(h.header_len, 7);
    }

    #[test]
    fn synthesizes_two_byte_asc() {
        // AOT=2, freq_index=4, chan=2 → 00010 0100 0010 000 = 0x12 0x10.
        let h = AdtsHeader { aot: 2, freq_index: 4, channel_config: 2, frame_length: 9, header_len: 7 };
        assert_eq!(AdtsParse::synth_asc(&h), [0x12, 0x10]);
    }

    #[test]
    fn rejects_non_adts_and_mp3_layer() {
        assert!(AdtsParse::parse_header(&[0x00; 7]).is_none(), "no syncword");
        // MP3 (layer III = 01) shares the 0xFFF sync but must be rejected here.
        let mut mp3 = vec![0xFF, 0xFB, 0x90, 0, 0, 0, 0];
        mp3.resize(7, 0);
        assert!(AdtsParse::parse_header(&mp3).is_none(), "layer != 00 is not ADTS");
    }
}
