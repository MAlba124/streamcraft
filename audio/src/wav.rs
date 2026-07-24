//! WAV (RIFF/WAVE) parsing: a pure header parser plus the [`WavParse`] element
//! (spec: Milestone applications §3). WAV is trivial and self-describing, so this is a
//! hand-written reference container parser — no external crate.
//!
//! [`WavParse`] is a passive transform: the WAV bytes arrive on the sink pad, it locates
//! the `data` chunk, and streams the raw interleaved PCM out the src pad as `bytes`
//! (stripping the RIFF header and any trailing chunks). The parsed [`AudioFormat`] is
//! exposed via [`WavParse::format`] and — because format negotiation is still link-time
//! only — the downstream encoder is told the rate/channels/format out of band for now
//! (the example parses the header directly). Typed `audio/raw` output waits on runtime
//! caps propagation.

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::format::{AudioFormat, SampleFormat};

/// Cap on how many bytes we buffer looking for the `data` chunk before declaring the
/// input not-a-WAV. Real headers are tens of bytes; metadata (`LIST`/`INFO`) rarely more
/// than a few KiB.
const MAX_HEADER: usize = 1 << 20;

/// The result of parsing a WAV header: the format plus where the PCM payload begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WavHeader {
    pub format: AudioFormat,
    /// Byte offset of the first PCM sample (start of the `data` chunk body).
    pub data_offset: usize,
    /// Declared payload length, or `None` for a streaming/unknown size (`0` or `0xFFFFFFFF`).
    pub data_len: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WavError {
    /// Not enough bytes yet to finish parsing — accumulate more and retry.
    NeedMore,
    NotRiff,
    NotWave,
    /// A `fmt ` chunk that is too short or has zero channels.
    BadFmt,
    /// A `data` chunk appeared before any `fmt ` chunk.
    NoFmt,
    /// A PCM encoding this parser does not support (`tag`/`bits`).
    Unsupported { tag: u16, bits: u16 },
    /// Chunk sizes that run off the end of the address space.
    Malformed,
}

fn le_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn le_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Parse a WAV header from the front of `b`, scanning chunks until `data` is reached.
/// Returns [`WavError::NeedMore`] when `b` is a valid-so-far but incomplete prefix, so a
/// streaming caller can accumulate and retry.
pub fn parse_wav_header(b: &[u8]) -> Result<WavHeader, WavError> {
    if b.len() < 12 {
        return Err(WavError::NeedMore);
    }
    if &b[0..4] != b"RIFF" {
        return Err(WavError::NotRiff);
    }
    if &b[8..12] != b"WAVE" {
        return Err(WavError::NotWave);
    }

    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (tag, channels, rate, bits)

    loop {
        if pos + 8 > b.len() {
            return Err(WavError::NeedMore);
        }
        let id = &b[pos..pos + 4];
        let size32 = le_u32(b, pos + 4);
        let size = size32 as usize;
        let body = pos + 8;

        if id == b"fmt " {
            if size < 16 {
                return Err(WavError::BadFmt);
            }
            if body + 16 > b.len() {
                return Err(WavError::NeedMore);
            }
            let tag = le_u16(b, body);
            let channels = le_u16(b, body + 2);
            let rate = le_u32(b, body + 4);
            // body+8 byte-rate, body+12 block-align (derived, not trusted here)
            let bits = le_u16(b, body + 14);
            // WAVE_FORMAT_EXTENSIBLE: the real code is the first 2 bytes of the SubFormat
            // GUID at body+24 (after cbSize@16, validBitsPerSample@18, channelMask@20).
            let real_tag = if tag == 0xFFFE {
                if body + 26 > b.len() {
                    return Err(WavError::NeedMore);
                }
                le_u16(b, body + 24)
            } else {
                tag
            };
            fmt = Some((real_tag, channels, rate, bits));
        } else if id == b"data" {
            let (tag, channels, rate, bits) = fmt.ok_or(WavError::NoFmt)?;
            if channels == 0 {
                return Err(WavError::BadFmt);
            }
            let sf = SampleFormat::from_wav(tag, bits).ok_or(WavError::Unsupported { tag, bits })?;
            let format = AudioFormat::new(rate, channels, sf);
            let data_len = if size32 == 0 || size32 == u32::MAX {
                None
            } else {
                Some(size as u64)
            };
            return Ok(WavHeader {
                format,
                data_offset: body,
                data_len,
            });
        }

        // Advance past this chunk; RIFF pads odd-sized bodies to an even boundary.
        let advance = size + (size & 1);
        pos = body.checked_add(advance).ok_or(WavError::Malformed)?;
    }
}

/// Write a canonical 44-byte-header PCM WAV wrapping `pcm` — the muxer half, so
/// `wav → … → wav` round-trips in tests/examples and a future `wavenc` has a home.
pub fn write_pcm_wav(format: &AudioFormat, pcm: &[u8]) -> Vec<u8> {
    let block_align = format.frame_stride() as u16;
    let byte_rate = format.sample_rate * block_align as u32;
    let data_len = pcm.len() as u32;
    let mut w = Vec::with_capacity(44 + pcm.len());
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVE");
    w.extend_from_slice(b"fmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    let tag: u16 = if format.format.is_float() { 3 } else { 1 };
    w.extend_from_slice(&tag.to_le_bytes());
    w.extend_from_slice(&format.channels.to_le_bytes());
    w.extend_from_slice(&format.sample_rate.to_le_bytes());
    w.extend_from_slice(&byte_rate.to_le_bytes());
    w.extend_from_slice(&block_align.to_le_bytes());
    w.extend_from_slice(&(format.format.bits() as u16).to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    w.extend_from_slice(pcm);
    w
}

// --- The element ----------------------------------------------------------------------

/// Bytes in, bytes out: `wavparse` strips the container and streams PCM. Both pads speak
/// raw `bytes` so it links between `filesrc`/`filesink` and the (currently `bytes`-typed)
/// `flacenc` in the milestone-3 chain.
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

static DESC: ElementDesc = ElementDesc {
    name: "wavparse",
    pads: &PADS,
    props: &[],
    // Passive: a pure transform that inlines into the upstream group (spec: Scheduling).
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Name-constructible (spec: Plugins): config-free; it learns the format from the
    // WAV header.
    make_default: Some(|| Box::new(WavParse::new())),
};

/// Parses a WAV stream and emits its interleaved PCM payload.
#[derive(Default)]
pub struct WavParse {
    /// The header once located; `None` while still buffering the leading chunks.
    header: Option<WavHeader>,
    /// Bytes accumulated while the header is still incomplete (dropped once parsed).
    acc: Vec<u8>,
    /// PCM bytes emitted so far, to honour `data_len` and stop at the payload's end.
    data_emitted: u64,
}

impl WavParse {
    pub fn new() -> Self {
        Self::default()
    }

    /// The parsed audio format, available once the header has been seen.
    pub fn format(&self) -> Option<AudioFormat> {
        self.header.as_ref().map(|h| h.format)
    }

    /// Emit `bytes` as PCM on the src pad, clamped to the declared `data_len` (so trailing
    /// post-`data` chunks are dropped) and chunked to the pool slot size.
    fn emit_data(&mut self, ctx: &mut Ctx, bytes: &[u8]) -> Result<(), Error> {
        let bytes = match self.header.as_ref().and_then(|h| h.data_len) {
            Some(total) => {
                let remaining = total.saturating_sub(self.data_emitted) as usize;
                &bytes[..bytes.len().min(remaining)]
            }
            None => bytes,
        };
        let mut off = 0;
        while off < bytes.len() {
            let mut buf = ctx.alloc(PadId(1));
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "wavparse: zero-capacity pool slot");
            let n = cap.min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            ctx.out(PadId(1)).push(buf);
            off += n;
        }
        self.data_emitted += bytes.len() as u64;
        Ok(())
    }
}

impl Element for WavParse {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.header = None;
        self.acc.clear();
        self.data_emitted = 0;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            if self.header.is_some() {
                // Past the header: the whole buffer is PCM payload.
                self.emit_data(ctx, buf.memory.data())?;
                continue;
            }

            self.acc.extend_from_slice(buf.memory.data());
            match parse_wav_header(&self.acc) {
                Ok(h) => {
                    // The bytes past the header, already buffered, are the first PCM.
                    let tail = self.acc.split_off(h.data_offset);
                    self.acc = Vec::new(); // done accumulating; release the header bytes
                    self.header = Some(h);
                    self.emit_data(ctx, &tail)?;
                }
                Err(WavError::NeedMore) => {
                    if self.acc.len() > MAX_HEADER {
                        return Err(Error::Resource(
                            "wavparse: no WAV data chunk within header budget".into(),
                        ));
                    }
                    // Wait for more input.
                }
                Err(e) => return Err(Error::Resource(format!("wavparse: bad WAV: {e:?}"))),
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.header = None;
        self.acc = Vec::new();
        self.data_emitted = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a canonical 44-byte PCM WAV header for `data_len` payload bytes.
    fn wav_header(format: &AudioFormat, data_len: u32) -> Vec<u8> {
        let block_align = format.frame_stride() as u16;
        let byte_rate = format.sample_rate * block_align as u32;
        let mut h = Vec::new();
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&(36 + data_len).to_le_bytes());
        h.extend_from_slice(b"WAVE");
        h.extend_from_slice(b"fmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        let tag: u16 = if format.format.is_float() { 3 } else { 1 };
        h.extend_from_slice(&tag.to_le_bytes());
        h.extend_from_slice(&format.channels.to_le_bytes());
        h.extend_from_slice(&format.sample_rate.to_le_bytes());
        h.extend_from_slice(&byte_rate.to_le_bytes());
        h.extend_from_slice(&block_align.to_le_bytes());
        h.extend_from_slice(&(format.format.bits() as u16).to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&data_len.to_le_bytes());
        h
    }

    #[test]
    fn parses_canonical_pcm_header() {
        let fmt = AudioFormat::new(44_100, 2, SampleFormat::S16);
        let mut wav = wav_header(&fmt, 8);
        wav.extend_from_slice(&[0u8; 8]);
        let h = parse_wav_header(&wav).expect("valid header");
        assert_eq!(h.format, fmt);
        assert_eq!(h.data_offset, 44);
        assert_eq!(h.data_len, Some(8));
    }

    #[test]
    fn skips_unknown_chunks_before_data() {
        let fmt = AudioFormat::new(48_000, 1, SampleFormat::S24);
        // A LIST chunk (odd size 5 → padded to 6) sits between fmt and data.
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&0u32.to_le_bytes()); // riff size unused by the parser
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // 1ch
        wav.extend_from_slice(&48_000u32.to_le_bytes());
        wav.extend_from_slice(&(48_000u32 * 3).to_le_bytes());
        wav.extend_from_slice(&3u16.to_le_bytes()); // block align
        wav.extend_from_slice(&24u16.to_le_bytes()); // bits
        wav.extend_from_slice(b"LIST");
        wav.extend_from_slice(&5u32.to_le_bytes());
        wav.extend_from_slice(b"hello"); // 5 bytes
        wav.push(0); // pad to even
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&6u32.to_le_bytes());
        let data_at = wav.len();
        wav.extend_from_slice(&[0u8; 6]);

        let h = parse_wav_header(&wav).expect("valid with LIST chunk");
        assert_eq!(h.format, fmt);
        assert_eq!(h.data_offset, data_at);
        assert_eq!(h.data_len, Some(6));
    }

    #[test]
    fn incomplete_header_reports_needmore() {
        let fmt = AudioFormat::new(44_100, 2, SampleFormat::S16);
        let wav = wav_header(&fmt, 100);
        // Truncate mid-header at every length: all should ask for more, never panic/err.
        for n in 0..wav.len() {
            assert_eq!(parse_wav_header(&wav[..n]), Err(WavError::NeedMore), "len {n}");
        }
        assert!(parse_wav_header(&wav).is_ok());
    }

    #[test]
    fn rejects_non_riff_and_unsupported() {
        assert_eq!(parse_wav_header(b"NOPExxxxWAVE"), Err(WavError::NotRiff));
        let mut not_wave = b"RIFF\x00\x00\x00\x00AVI ".to_vec();
        not_wave.resize(12, 0);
        assert_eq!(parse_wav_header(&not_wave), Err(WavError::NotWave));
        // 12-bit PCM is unsupported.
        let fmt = AudioFormat::new(44_100, 1, SampleFormat::S16);
        let mut wav = wav_header(&fmt, 0);
        wav[34] = 12; // overwrite bits-per-sample field
        assert!(matches!(parse_wav_header(&wav), Err(WavError::Unsupported { bits: 12, .. })));
    }
}
