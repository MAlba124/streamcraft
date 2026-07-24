//! End-to-end element test: `MkvMux` in a real [`Pipeline`] (spec: `mkv/spec/MATROSKA.md`).
//!
//! A source emits real FLAC frames as buffers → `MkvMux` muxes them into a Matroska byte
//! stream → a sink concatenates the bytes. We then parse the captured stream with the
//! minimal in-crate EBML reader and assert it is a well-formed MKV: EBML Header at the
//! start, one A_FLAC TrackEntry with the CodecPrivate the element was constructed with, and
//! one SimpleBlock per input frame carrying the exact frame bytes on track 1. This mirrors
//! `sc-ogg`'s `element_roundtrip.rs` scaffold (a byte-collecting sink around the mux).

mod common;

use std::sync::{Arc, Mutex};

use common::{parse_simple_block, walk};
use sc_flac::{FlacEncoder, SampleFormat};
use sc_mkv::ebml::id;
use sc_mkv::MkvMux;
use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

const MASTERS: &[&[u8]] = &[
    id::EBML,
    id::SEGMENT,
    id::INFO,
    id::TRACKS,
    id::TRACK_ENTRY,
    id::AUDIO,
    id::CLUSTER,
];

// =====================================================================================
// A source that emits a fixed list of byte buffers (one per frame), then EOS. Copied in
// shape from sc-ogg's element_roundtrip PacketSrc.
// =====================================================================================

static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &SRC_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "framesrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

struct FrameSrc {
    frames: Vec<Vec<u8>>,
    next: usize,
}

impl Element for FrameSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.next = 0;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.next >= self.frames.len() {
            return Ok(Flow::Eos);
        }
        let f = &self.frames[self.next];
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return Ok(Flow::Ok),
        };
        assert!(f.len() <= buf.memory.capacity(), "frame exceeds pool slot");
        buf.memory.as_mut_full()[..f.len()].copy_from_slice(f);
        buf.memory.set_len(f.len());
        ctx.out(PadId(0)).push(buf);
        self.next += 1;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// =====================================================================================
// A sink concatenating all received bytes (the raw MKV stream). Same shape as sc-ogg's
// ByteSink.
// =====================================================================================

static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "bytesink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

struct ByteSink {
    got: Arc<Mutex<Vec<u8>>>,
}

impl Element for ByteSink {
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

/// Real FLAC head + frames, one frame per block (shared with `roundtrip.rs`'s generator
/// shape but local here so the test files stay independent test crates).
fn make_flac(sample_rate: u32, channels: u32, block: usize, n: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let (mut enc, mut header) = FlacEncoder::new(sample_rate, channels, SampleFormat::S16).unwrap();
    let mut frames = Vec::new();
    let mut phase = 0i32;
    for _ in 0..n {
        let mut pcm = Vec::new();
        for _ in 0..block {
            for c in 0..channels {
                let v = ((phase.wrapping_mul(53) + c as i32 * 700) & 0x3FFF) as i16 - 0x2000;
                pcm.extend_from_slice(&v.to_le_bytes());
                phase = phase.wrapping_add(1);
            }
        }
        let mut frame = Vec::new();
        enc.encode_interleaved(&pcm, &mut frame).unwrap();
        frames.push(frame);
    }
    let body = enc.finish();
    let off = sc_flac::streaminfo_offset();
    header[off..off + body.len()].copy_from_slice(&body);
    (header, frames)
}

/// Run `framesrc(frames) ! mkvmux(flac) ! bytesink` and return the muxed MKV bytes.
fn run_mux(codec_private: Vec<u8>, frames: Vec<Vec<u8>>) -> Vec<u8> {
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(FrameSrc { frames, next: 0 });
    let mux = p.add(MkvMux::flac(codec_private, 48_000.0, 2, 16));
    let sink = p.add(ByteSink { got: Arc::clone(&got) });
    p.link((src, "src"), (mux, "sink")).expect("src->mux");
    p.link((mux, "src"), (sink, "sink")).expect("mux->sink");
    p.run().expect("run");
    let out = got.lock().unwrap().clone();
    out
}

#[test]
fn mkvmux_pipeline_produces_valid_mkv() {
    let (codec_private, frames) = make_flac(48_000, 2, 4096, 5);
    let stream = run_mux(codec_private.clone(), frames.clone());

    // EBML Header at the very start.
    assert!(!stream.is_empty(), "mux produced bytes");
    assert_eq!(&stream[..4], id::EBML, "stream starts with the EBML Header ID");

    let els = walk(&stream, MASTERS);
    assert!(els.iter().any(|(_, e)| e.id == id::SEGMENT), "Segment present");

    // Exactly one A_FLAC TrackEntry with the constructed CodecPrivate.
    let entries = els.iter().filter(|(_, e)| e.id == id::TRACK_ENTRY).count();
    assert_eq!(entries, 1, "one TrackEntry (single-track element)");
    let mut cid = None;
    let mut cpriv = None;
    for (_d, el) in &els {
        if el.id == id::CODEC_ID {
            let (s, e) = el.data.unwrap();
            cid = Some(String::from_utf8(stream[s..e].to_vec()).unwrap());
        } else if el.id == id::CODEC_PRIVATE {
            let (s, e) = el.data.unwrap();
            cpriv = Some(stream[s..e].to_vec());
        }
    }
    assert_eq!(cid.as_deref(), Some("A_FLAC"), "CodecID A_FLAC");
    assert_eq!(cpriv.as_ref(), Some(&codec_private), "CodecPrivate as constructed");

    // One SimpleBlock per input frame, on track 1, exact bytes, timestamps non-decreasing.
    let mut base = 0i64;
    let mut blocks = Vec::new();
    for (_d, el) in &els {
        if el.id == id::TIMESTAMP {
            let (s, e) = el.data.unwrap();
            let mut v = 0i64;
            for &b in &stream[s..e] {
                v = (v << 8) | b as i64;
            }
            base = v;
        } else if el.id == id::SIMPLE_BLOCK {
            let (s, e) = el.data.unwrap();
            let blk = parse_simple_block(&stream[s..e]);
            blocks.push((base + blk.rel_ts as i64, blk));
        }
    }
    assert_eq!(blocks.len(), frames.len(), "one SimpleBlock per frame");
    let mut last = i64::MIN;
    for (i, ((ts, blk), frame)) in blocks.iter().zip(frames.iter()).enumerate() {
        assert_eq!(blk.track, 1, "block {i} on track 1");
        assert_eq!(&blk.frame, frame, "block {i} exact FLAC frame bytes");
        assert!(*ts >= last, "block {i} timestamp non-decreasing");
        last = *ts;
    }
}

#[test]
fn mkvmux_empty_stream_writes_nothing() {
    // No frames → the header is emitted lazily on the first frame, so an empty stream
    // produces no bytes (nothing to finalize either — Segment is streamed).
    let (codec_private, _f) = make_flac(48_000, 2, 4096, 0);
    let stream = run_mux(codec_private, Vec::new());
    assert!(stream.is_empty(), "an empty input produces no MKV bytes");
}
