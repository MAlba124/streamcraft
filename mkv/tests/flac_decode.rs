//! Integration: a muxed A_FLAC track demuxes and **decodes bit-exact** through the existing
//! `FlacDec` (spec: `mkv/spec/MATROSKA.md`; milestones 5/6, the container half). This is the
//! `filesrc ! mkvdemux ! flacdec ! …` milestone on a byte stream: MKV bytes in, PCM out,
//! sample-for-sample equal to decoding the original native FLAC stream.
//!
//! Dev-dependency on `pf-flac` (no cycle — the container is codec-agnostic and `pf-flac` does
//! not depend on `pf-mkv`), used to make real A_FLAC frames + STREAMINFO and to decode.

use std::sync::{Arc, Mutex};

use pf_flac::{FlacDec, FlacDecoder, FlacEncoder, SampleFormat};
use pf_mkv::ebml::id;
use pf_mkv::{MatroskaWriter, MkvDemux, TrackConfig};
use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::PadId;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;

// --- a source that streams a byte blob in fixed chunks, then EOS ---------------------

static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &SRC_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "bytesrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct ByteSrc {
    data: Vec<u8>,
    chunk: usize,
    pos: usize,
}
impl Element for ByteSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.pos = 0;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.pos >= self.data.len() {
            return Ok(Flow::Eos);
        }
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return Ok(Flow::Ok),
        };
        let cap = buf.memory.capacity();
        let n = self.chunk.min(cap).min(self.data.len() - self.pos);
        buf.memory.as_mut_full()[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        buf.memory.set_len(n);
        ctx.out(PadId(0)).push(buf);
        self.pos += n;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// --- a sink concatenating decoded PCM bytes -----------------------------------------

static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("audio/raw")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "pcmsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct PcmSink {
    got: Arc<Mutex<Vec<u8>>>,
}
impl Element for PcmSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut got = self.got.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            got.extend_from_slice(buf.memory.data());
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// --- helpers ------------------------------------------------------------------------

/// A deterministic S16 signal of `frames` interchannel samples across `channels`.
fn signal(frames: usize, channels: u32) -> Vec<i16> {
    let mut v = Vec::with_capacity(frames * channels as usize);
    let mut phase = 0i32;
    for _ in 0..frames {
        for c in 0..channels {
            let s = ((phase.wrapping_mul(31) + c as i32 * 1234) & 0x3FFF) as i16 - 0x2000;
            v.push(s);
            phase = phase.wrapping_add(1);
        }
    }
    v
}

/// Encode `samples` (interleaved S16) into a native FLAC stream and the per-frame byte list.
/// Returns `(native_stream, codec_private, frames)` where `codec_private` is `fLaC` +
/// finalised STREAMINFO (what A_FLAC CodecPrivate holds) and `native_stream` is the full
/// native `.flac` bytes (head + frames) — the reference the demux reconstruction must match.
fn encode(samples: &[i16], channels: u32, rate: u32) -> (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) {
    let (mut enc, mut header) = FlacEncoder::new(rate, channels, SampleFormat::S16).unwrap();
    // Encode in blocks of 4096 interchannel samples (one FLAC frame each).
    let block = 4096usize;
    let per = channels as usize;
    let mut frames = Vec::new();
    let mut i = 0usize;
    while i < samples.len() {
        let end = (i + block * per).min(samples.len());
        let mut pcm = Vec::with_capacity((end - i) * 2);
        for &s in &samples[i..end] {
            pcm.extend_from_slice(&s.to_le_bytes());
        }
        let mut frame = Vec::new();
        enc.encode_interleaved(&pcm, &mut frame).unwrap();
        frames.push(frame);
        i = end;
    }
    let body = enc.finish();
    let off = pf_flac::streaminfo_offset();
    header[off..off + body.len()].copy_from_slice(&body);
    let codec_private = header.clone();
    // The native reference stream is head + every frame concatenated.
    let mut native = header;
    for f in &frames {
        native.extend_from_slice(f);
    }
    (native, codec_private, frames)
}

/// Mux one A_FLAC track (codec_private + frames) into an MKV byte stream via the N-track
/// writer, on a millisecond timeline.
fn mux(codec_private: &[u8], frames: &[Vec<u8>], rate: u32, channels: u32) -> Vec<u8> {
    let track = TrackConfig::flac(1, codec_private.to_vec(), rate as f64, channels, 16);
    let mut w = MatroskaWriter::new(vec![track]);
    let mut out = Vec::new();
    w.write_header(&mut out).unwrap();
    let block_ns = 4096u64 * 1_000_000_000 / rate as u64;
    for (i, f) in frames.iter().enumerate() {
        w.write_frame(&mut out, 1, i as u64 * block_ns, f, true).unwrap();
    }
    w.finalize(&mut out);
    out
}

/// The demuxer discovers tracks from its constructor header (through the first Cluster).
fn header_prefix(stream: &[u8]) -> Vec<u8> {
    let cluster = stream.windows(4).position(|w| w == id::CLUSTER).expect("a Cluster");
    stream[..cluster].to_vec()
}

/// Run `bytesrc(mkv) ! mkvdemux ! flacdec ! pcmsink` and return the decoded PCM bytes.
fn run(stream: Vec<u8>) -> Vec<u8> {
    let header = header_prefix(&stream);
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(ByteSrc { data: stream, chunk: 1024, pos: 0 });
    let demux = p.add(MkvDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");

    // Preroll: the demuxer exposes one src pad for the single track; link it into flacdec.
    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 1, "one A_FLAC track → one src pad");
    let dec = p.add(FlacDec::new());
    let sink = p.add(PcmSink { got: Arc::clone(&got) });
    p.link((added[0].element, &added[0].name), (dec, "sink")).expect("demux pad -> flacdec");
    p.link((dec, "src"), (sink, "sink")).expect("flacdec -> pcmsink");

    p.run().expect("run");
    let out = got.lock().unwrap().clone();
    out
}

fn pcm_to_i64(bytes: &[u8]) -> Vec<i64> {
    bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as i64).collect()
}

#[test]
fn muxed_a_flac_decodes_bit_exact_through_flacdec() {
    for &(frames, ch, rate) in &[
        (1usize, 2u32, 44_100u32),  // one interchannel sample, one short frame
        (4096, 1, 48_000),          // exactly one full block, mono
        (10_000, 2, 44_100),        // several frames (> block), stereo
        (13_337, 1, 44_100),        // several frames, mono, non-block-multiple
    ] {
        let samples = signal(frames, ch);
        let expected: Vec<i64> = samples.iter().map(|&s| s as i64).collect();

        let (native, codec_private, flac_frames) = encode(&samples, ch, rate);

        // Sanity: the native encode is itself lossless (isolates a codec bug from a mux bug).
        let ref_dec = FlacDecoder::decode(&native).expect("native decode");
        assert_eq!(ref_dec.samples, expected, "native FLAC lossless (frames={frames} ch={ch})");

        // The pipeline: mux → mkvdemux (reconstructs native FLAC) → flacdec must be bit-exact.
        let stream = mux(&codec_private, &flac_frames, rate, ch);
        let pcm = run(stream);
        let decoded = pcm_to_i64(&pcm);
        assert_eq!(
            decoded, expected,
            "A_FLAC mux → demux → flacdec bit-exact (frames={frames} ch={ch} rate={rate})"
        );
    }
}
