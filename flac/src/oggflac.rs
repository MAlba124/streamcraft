//! `oggflacdeframe` — de-frames a FLAC-in-Ogg logical bitstream back into a **native FLAC
//! byte stream** (spec: `spec/ogg-flac-mapping.md`; xiph "Ogg Mapping for FLAC",
//! <https://xiph.org/flac/ogg_mapping.html>). Ogg packets arrive on the sink pad (one per
//! input buffer, as [`OggDemux`](../../ogg/src/element.rs) emits them); the bytes leaving
//! the src pad are exactly what a native `.flac` file contains — `"fLaC"` + STREAMINFO +
//! any further metadata blocks + all frames — which [`FlacDec`](crate::FlacDec)'s
//! `StreamDecoder` already decodes.
//!
//! It sits between `oggdemux` and `flacdec`:
//! `filesrc ! oggdemux ! oggflacdeframe ! flacdec ! pipewireaudiosink`.
//!
//! ## What the mapping does, and how de-framing undoes it (spec: the mapping doc)
//! The mapping wraps a native FLAC stream in Ogg packets, changing the bytes in exactly one
//! place: the **first** packet is prefixed with a 9-byte mapping header —
//! `0x7F "FLAC" <major> <minor> <count:be16>` — before the native `"fLaC"` signature and
//! STREAMINFO. Every later packet is *already* native FLAC on the wire (one metadata block,
//! then one frame per packet). So de-framing is: **strip the 9-byte prefix from the first
//! packet, forward every packet's bytes verbatim** — the concatenation is the native
//! stream. The header packet count is advisory: the native last-metadata-block flag marks
//! the metadata/audio boundary inside the reconstructed stream, so this element never needs
//! to count packets to stay in sync (spec: the mapping doc, "Header packet count is
//! advisory").
//!
//! ## Format
//! Both pads carry raw [`bytes`](profluens_core::format::OfferDesc::any) — an Ogg-FLAC
//! packet stream in, a native FLAC byte stream out — matching the `oggdemux` src pad and the
//! `flacdec` sink pad, so it links to both at link time with no typed vocabulary. This
//! element is a **passive byte→byte transform** (bytes in, bytes out, inlining into the
//! upstream group like `oggdemux`/`flacdec`; spec: Scheduling — passive elements run
//! inline). It does **not** decode audio or announce `audio/raw`: the downstream `flacdec`
//! announces the concrete format from the STREAMINFO now at the head of the native stream
//! (spec: Formats — dynamic caps).
//!
//! ## Robustness
//! A decoder parses untrusted input, so a first packet that is shorter than the 9-byte
//! prefix, or that lacks the `0x7F "FLAC"` signature, is reported as an [`Error`] rather
//! than mis-de-framed or panicking (spec: a crash on bad input is a P0).

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

// The src pad's local index (== position in the element's `pads` array). The sink pad
// (index 0) needs no id: input is drained via `Inputs::pop`, which is pad-agnostic — same
// convention as `oggdemux`/`flacdec`.
const SRC: PadId = PadId(1);

/// The FLAC-to-Ogg mapping-header packet type: the first byte of the first packet
/// (spec: the mapping doc §1). `0x7F` is outside the native FLAC frame sync (`0xFF`) and
/// the metadata-block-type range, so it unambiguously identifies the mapping packet.
const MAPPING_PACKET_TYPE: u8 = 0x7F;
/// The mapping signature following the type byte: ASCII `"FLAC"` (spec: the mapping doc §1).
const MAPPING_SIGNATURE: &[u8; 4] = b"FLAC";
/// Bytes of the mapping header to strip from the first packet before the native `"fLaC"`:
/// packet type (1) + `"FLAC"` (4) + major version (1) + minor version (1) + header packet
/// count big-endian (2) = 9 (spec: the mapping doc §1).
const MAPPING_PREFIX_LEN: usize = 9;

/// Raw `bytes` on both pads: an Ogg-FLAC packet stream on the sink, a native FLAC byte
/// stream on the src. Matches the `oggdemux` src / `flacdec` sink `bytes` pads.
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    },
];

// `make_default` boxes one element instance at registry/parse time, never per frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "oggflacdeframe",
    pads: &PADS,
    props: &[],
    // Passive: a pure byte→byte transform, inlines into the upstream group.
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Name-constructible (spec: Plugins): config-free byte→byte de-framer.
    make_default: Some(|| Box::new(OggFlacDeframe::new())),
};

/// De-frames a FLAC-in-Ogg logical bitstream into a native FLAC byte stream (spec:
/// `spec/ogg-flac-mapping.md`). One Ogg packet arrives per input buffer (from `oggdemux`);
/// the first packet's 9-byte mapping prefix is stripped and every packet's remaining bytes
/// are forwarded verbatim, reconstructing exactly what a native `.flac` file contains.
#[derive(Default)]
pub struct OggFlacDeframe {
    /// False until the mapping-header (first) packet has been seen and its prefix stripped.
    /// The strip only ever applies to the very first packet, so a later frame packet that
    /// `oggdemux` splits across several buffers is forwarded whole with no special-casing.
    seen_header: bool,
}

impl OggFlacDeframe {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the mapping header on the first packet and return the native-FLAC tail
    /// (everything from `"fLaC"` onward). Errors on a malformed / too-short header rather
    /// than panicking (spec: decoders parse untrusted input).
    // Runs once per stream (first packet only); the `to_string` is a cold error message.
    #[allow(clippy::disallowed_methods)]
    fn strip_mapping_header(data: &[u8]) -> Result<&[u8], Error> {
        if data.len() < MAPPING_PREFIX_LEN {
            return Err(Error::Resource(format!(
                "oggflacdeframe: first packet is {} bytes, shorter than the {MAPPING_PREFIX_LEN}-byte FLAC-to-Ogg mapping header",
                data.len()
            )));
        }
        if data[0] != MAPPING_PACKET_TYPE || &data[1..5] != MAPPING_SIGNATURE {
            return Err(Error::Resource(
                "oggflacdeframe: first packet is not a FLAC-to-Ogg mapping header (missing 0x7F \"FLAC\" signature)".to_string(),
            ));
        }
        // Bytes 5..7 are the mapping version, 7..9 the big-endian header packet count. Both
        // are read (bounds-checked by the length guard above) but not enforced: only mapping
        // v1 exists, and the count is advisory (spec: the mapping doc). What matters is the
        // native FLAC bytes from offset 9.
        Ok(&data[MAPPING_PREFIX_LEN..])
    }

    /// Push `bytes` of the reconstructed native FLAC stream on the src pad, chunked to the
    /// pool slot size so no single copy exceeds a buffer (same pattern as `oggdemux`/`oggmux`
    /// / `flacdec`). Empty `bytes` push nothing — a native byte stream has no packet
    /// boundary to preserve, so an empty forward is simply a no-op.
    fn emit(ctx: &mut Ctx, bytes: &[u8]) {
        let mut off = 0;
        while off < bytes.len() {
            let mut buf = ctx.alloc(SRC);
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "oggflacdeframe: zero-capacity pool slot");
            let n = cap.min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            ctx.out(SRC).push(buf);
            off += n;
        }
    }
}

impl Element for OggFlacDeframe {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.seen_header = false;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            // `buf` recycles on drop at the end of this iteration.
            if self.seen_header {
                // A metadata block or one native FLAC frame — forward verbatim.
                Self::emit(ctx, buf.memory.data());
            } else {
                // The mapping header (first packet): strip its 9-byte prefix, forward the
                // native `"fLaC"` + STREAMINFO tail. The whole mapping packet arrives in this
                // one buffer (51 bytes, well under a pool slot).
                let native = Self::strip_mapping_header(buf.memory.data())?;
                Self::emit(ctx, native);
                self.seen_header = true;
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        // Nothing to flush: every packet is forwarded within `process`, and the native FLAC
        // stream needs no end-of-stream marker (§9: frames just stop). EOS propagates to the
        // downstream decoder through the normal chain-ordered delivery.
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.seen_header = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_the_nine_byte_mapping_prefix() {
        // A minimal mapping packet: 0x7F "FLAC" v1.0, count 0, then a stand-in native tail.
        let mut pkt = vec![0x7F];
        pkt.extend_from_slice(b"FLAC");
        pkt.extend_from_slice(&[0x01, 0x00]); // major.minor
        pkt.extend_from_slice(&0u16.to_be_bytes()); // header packet count
        pkt.extend_from_slice(b"fLaCnative-streaminfo-bytes");
        let native = OggFlacDeframe::strip_mapping_header(&pkt).expect("valid header");
        assert_eq!(native, b"fLaCnative-streaminfo-bytes");
    }

    #[test]
    fn rejects_a_short_first_packet() {
        // Fewer than 9 bytes cannot hold the mapping prefix.
        let err = OggFlacDeframe::strip_mapping_header(&[0x7F, b'F', b'L']);
        assert!(err.is_err(), "short first packet must error, not panic");
    }

    #[test]
    fn rejects_a_bad_signature() {
        // Right length, wrong signature (not 0x7F "FLAC").
        let bogus = [0u8; 16];
        let err = OggFlacDeframe::strip_mapping_header(&bogus);
        assert!(err.is_err(), "missing 0x7F \"FLAC\" must error");
    }
}
