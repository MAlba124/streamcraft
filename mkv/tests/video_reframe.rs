//! Demux-level codec-id mapping + avcC/hvcC → Annex B reframing (spec: `mkv/spec/MATROSKA.md`;
//! RFC 9559 §12 codec mappings; ISO/IEC 14496-15).
//!
//! `codec.rs` unit-tests the record parsing and NAL reframing against golden byte vectors;
//! these tests exercise the same logic **through the demuxer element**: a hand-built MKV with a
//! V_MPEG4/ISO/AVC (or V_MPEGH/ISO/HEVC) track, whose Blocks carry length-prefixed NALs, must
//! demux to the Annex B parameter-set head followed by each Block's NALs in start-code form,
//! on a pad announcing the `h264/annexb` (resp. `h265/annexb`) family. WebM V_VP8 must map to
//! `vp8` and forward frames verbatim. Malformed configuration records/blocks warn-and-drop,
//! never panic.

use std::sync::{Arc, Mutex};

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

const START_CODE: [u8; 4] = [0x00, 0x00, 0x00, 0x01];

// ---- byte source + recording sink (mirrors demux_roundtrip.rs) ---------------------------

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
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
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

// A sink offering every demux family, recording each received buffer as (bytes, pts).
static SINK_OFFERS: [OfferDesc; 7] = [
    OfferDesc::any("flac"),
    OfferDesc::any("bytes"),
    OfferDesc::any("vp8"),
    OfferDesc::any("vp9"),
    OfferDesc::any("av1"),
    OfferDesc::any("h264/annexb"),
    OfferDesc::any("h265/annexb"),
];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "recsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

#[derive(Default, Debug)]
struct Recorded {
    /// Each received buffer's bytes, kept per-buffer (one demux buffer == one access unit for a
    /// NAL track), plus the head buffer.
    buffers: Vec<Vec<u8>>,
}

struct RecSink {
    rec: Arc<Mutex<Recorded>>,
}

impl Element for RecSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut rec = self.rec.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            rec.buffers.push(buf.memory.data().to_vec());
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// ---- helpers -----------------------------------------------------------------------------

/// The header prefix `MkvDemux::new` needs (up to the first Cluster).
fn header_prefix(stream: &[u8]) -> Vec<u8> {
    let cluster = stream
        .windows(4)
        .position(|w| w == pf_mkv::ebml::id::CLUSTER)
        .expect("stream has a Cluster");
    stream[..cluster].to_vec()
}

/// Run `bytesrc(stream) ! mkvdemux(header) ! recsink` for a single-track stream. Returns the
/// per-buffer bytes the sink recorded, in order.
fn run(stream: Vec<u8>, chunk: usize) -> Vec<Vec<u8>> {
    let header = header_prefix(&stream);
    let mut p = Pipeline::new();
    let src = p.add(ByteSrc { data: stream, chunk, pos: 0 });
    let demux = p.add(MkvDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");

    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 1, "one track → one src pad");

    let rec = Arc::new(Mutex::new(Recorded::default()));
    let sink = p.add(RecSink { rec: Arc::clone(&rec) });
    let ap = &added[0];
    p.link((ap.element, &ap.name), (sink, "sink")).expect("pad -> sink");
    p.run().expect("run");

    Arc::try_unwrap(rec).unwrap().into_inner().unwrap().buffers
}

/// A minimal AVCDecoderConfigurationRecord with one SPS and one PPS, lengthSize 4.
fn avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut cfg = vec![1, 0x42, 0x00, 0x0A, 0xFF, 0xE1];
    cfg.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    cfg.extend_from_slice(sps);
    cfg.push(1);
    cfg.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    cfg.extend_from_slice(pps);
    cfg
}

/// A minimal HEVCDecoderConfigurationRecord: 22-octet fixed prefix (lengthSize 4 at octet 21),
/// then three arrays (VPS, SPS, PPS), one NAL each.
fn hvcc(vps: &[u8], sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut cfg = vec![1u8];
    cfg.extend_from_slice(&[0u8; 20]);
    cfg.push(0xFC | 0x03); // octet 21: lengthSizeMinusOne = 3
    cfg.push(3); // numOfArrays
    for (t, nal) in [(32u8, vps), (33, sps), (34, pps)] {
        cfg.push(t);
        cfg.extend_from_slice(&1u16.to_be_bytes());
        cfg.extend_from_slice(&(nal.len() as u16).to_be_bytes());
        cfg.extend_from_slice(nal);
    }
    cfg
}

/// A Block of length-prefixed NALs (4-byte big-endian lengths) from a list of NAL byte slices.
fn length_prefixed(nals: &[&[u8]]) -> Vec<u8> {
    let mut b = Vec::new();
    for nal in nals {
        b.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        b.extend_from_slice(nal);
    }
    b
}

/// One NAL in Annex B form: 4-byte start code + bytes.
fn annex_b(nals: &[&[u8]]) -> Vec<u8> {
    let mut b = Vec::new();
    for nal in nals {
        b.extend_from_slice(&START_CODE);
        b.extend_from_slice(nal);
    }
    b
}

// ---- tests -------------------------------------------------------------------------------

/// A V_MPEG4/ISO/AVC track: the demux emits the SPS+PPS Annex B head first, then each Block's
/// length-prefixed NALs reframed to start-code form — one buffer per access unit.
#[test]
fn avc_track_reframes_to_annex_b() {
    let sps = [0x67, 0x42, 0x00, 0x0A, 0x11, 0x22];
    let pps = [0x68, 0xCE, 0x3C, 0x80];
    let cfg = avcc(&sps, &pps);

    // Two access units, each a couple of length-prefixed NALs.
    let au0 = length_prefixed(&[&[0x65, 0xAA, 0xBB], &[0x41, 0xCC]]);
    let au1 = length_prefixed(&[&[0x41, 0xDD, 0xEE, 0xFF]]);

    let track = TrackConfig::video(1, "V_MPEG4/ISO/AVC", cfg, 320, 240);
    let mut w = MatroskaWriter::new(vec![track]);
    let mut stream = Vec::new();
    w.write_header(&mut stream).unwrap();
    w.write_frame(&mut stream, 1, 0, &au0, true).unwrap();
    w.write_frame(&mut stream, 1, 33_000_000, &au1, false).unwrap();
    w.finalize(&mut stream);

    let bufs = run(stream, 500);
    // Buffer 0: the parameter-set head (SPS then PPS as start-code NALs).
    // Buffer 1: au0 reframed; Buffer 2: au1 reframed.
    assert_eq!(bufs.len(), 3, "head + one buffer per access unit");
    assert_eq!(bufs[0], annex_b(&[&sps, &pps]), "SPS+PPS Annex B head");
    assert_eq!(bufs[1], annex_b(&[&[0x65, 0xAA, 0xBB], &[0x41, 0xCC]]), "AU0 reframed");
    assert_eq!(bufs[2], annex_b(&[&[0x41, 0xDD, 0xEE, 0xFF]]), "AU1 reframed");
}

/// A V_MPEGH/ISO/HEVC track: VPS+SPS+PPS Annex B head, then each Block reframed.
#[test]
fn hevc_track_reframes_to_annex_b() {
    let vps = [0x40, 0x01, 0x0C, 0x01];
    let sps = [0x42, 0x01, 0x01, 0x02];
    let pps = [0x44, 0x01, 0xC0];
    let cfg = hvcc(&vps, &sps, &pps);
    let au0 = length_prefixed(&[&[0x26, 0x01, 0xAF], &[0x02, 0x01, 0xD0]]);

    let track = TrackConfig::video(1, "V_MPEGH/ISO/HEVC", cfg, 640, 360);
    let mut w = MatroskaWriter::new(vec![track]);
    let mut stream = Vec::new();
    w.write_header(&mut stream).unwrap();
    w.write_frame(&mut stream, 1, 0, &au0, true).unwrap();
    w.finalize(&mut stream);

    let bufs = run(stream, 64);
    assert_eq!(bufs.len(), 2, "head + one access unit");
    assert_eq!(bufs[0], annex_b(&[&vps, &sps, &pps]), "VPS+SPS+PPS Annex B head");
    assert_eq!(bufs[1], annex_b(&[&[0x26, 0x01, 0xAF], &[0x02, 0x01, 0xD0]]), "AU0 reframed");
}

/// A WebM V_VP8 track carries frames raw: no head, frames forwarded verbatim.
#[test]
fn vp8_track_forwards_frames_verbatim() {
    let f0: &[u8] = &[0x10, 0x20, 0x30, 0x40];
    let f1: &[u8] = &[0x50, 0x60];
    let track = TrackConfig::vp8(1, 128, 96);
    let mut w = MatroskaWriter::new(vec![track]);
    let mut stream = Vec::new();
    w.write_header(&mut stream).unwrap();
    w.write_frame(&mut stream, 1, 0, f0, true).unwrap();
    w.write_frame(&mut stream, 1, 33_000_000, f1, true).unwrap();
    w.finalize(&mut stream);

    let bufs = run(stream, 500);
    assert_eq!(bufs, vec![f0.to_vec(), f1.to_vec()], "no head; VP8 frames verbatim");
}

/// A malformed avcC CodecPrivate must not kill the demuxer: it degrades to no head +
/// passthrough frames (the pad still exists, the stream still runs) — never a panic.
#[test]
fn malformed_avcc_degrades_without_panic() {
    // A truncated record (version byte only): head parse fails, falls back to passthrough.
    let cfg = vec![1u8];
    let f0: &[u8] = &[0x00, 0x00, 0x00, 0x02, 0x65, 0xAB]; // one length-prefixed NAL
    let track = TrackConfig::video(1, "V_MPEG4/ISO/AVC", cfg, 320, 240);
    let mut w = MatroskaWriter::new(vec![track]);
    let mut stream = Vec::new();
    w.write_header(&mut stream).unwrap();
    w.write_frame(&mut stream, 1, 0, f0, true).unwrap();
    w.finalize(&mut stream);

    let bufs = run(stream, 500);
    // No head (record unparseable) and the frame passes through verbatim (passthrough fallback).
    assert_eq!(bufs, vec![f0.to_vec()], "malformed record → passthrough, no panic");
}
