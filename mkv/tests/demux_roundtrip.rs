//! End-to-end demux round-trip: `MatroskaWriter`/`MkvMux` → `MkvDemux` in a real
//! [`Pipeline`] (spec: `mkv/spec/MATROSKA.md`; dynamic pads).
//!
//! The load-bearing property: what the mux side writes, the demux side reads back
//! **bit-exact per track** — every frame's bytes, track routing, and timestamps survive the
//! container round-trip. The demuxer exposes one runtime src pad per track (settled during
//! `preroll`, then frozen — spec: dynamic pads), each linked to its own byte-collecting sink.
//!
//! Covered here:
//! - a single-track A_FLAC file through the `MkvMux` element → `MkvDemux`;
//! - a **two-track** file (stereo 48k + mono 44.1k) through the N-track `MatroskaWriter` (the
//!   mux library `MkvMux` wraps) → `MkvDemux`, each track's frames routed to its own pad;
//! - timestamps: each frame's demuxed pts (ns) matches the muxed timeline.

use std::sync::{Arc, Mutex};

use sc_flac::{FlacEncoder, SampleFormat};
use sc_mkv::ebml::id;
use sc_mkv::{MatroskaWriter, MkvDemux, MkvMux, TrackConfig};
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

// =====================================================================================
// A source that streams a fixed byte buffer in fixed-size chunks, then EOS. Chunking
// exercises the demuxer's cross-boundary buffering (elements straddle chunk edges).
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

// =====================================================================================
// A sink that records each received buffer as one (bytes, pts_ns) entry, in order. One per
// demuxer src pad, so a track's reconstructed stream and its timestamps are captured.
// =====================================================================================

static SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("flac"), OfferDesc::any("bytes")];
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
    /// All bytes received on the pad, concatenated in order (the reconstructed track stream).
    bytes: Vec<u8>,
    /// The pts (ns) of each received buffer, in order (`None` for a buffer with no pts).
    ptss: Vec<Option<u64>>,
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
            rec.bytes.extend_from_slice(buf.memory.data());
            rec.ptss.push(buf.pts.nanos());
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// =====================================================================================
// FLAC + MKV builders (real frames from the hand-written encoder).
// =====================================================================================

/// Real native FLAC head (`fLaC` + finalised STREAMINFO) plus `n` frames of `block`
/// interchannel samples. Returns `(codec_private, frames)`.
fn make_flac(sample_rate: u32, channels: u32, block: usize, n: usize, seed: i32) -> (Vec<u8>, Vec<Vec<u8>>) {
    let (mut enc, mut header) = FlacEncoder::new(sample_rate, channels, SampleFormat::S16).unwrap();
    let mut frames = Vec::new();
    let mut phase = seed;
    for _ in 0..n {
        let mut pcm = Vec::new();
        for _ in 0..block {
            for c in 0..channels {
                let v = ((phase.wrapping_mul(41) + c as i32 * 900) & 0x3FFF) as i16 - 0x2000;
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

/// The demuxer needs its track set at construction (spec: constructor-supplied discovery).
/// Split off the header prefix of a muxed stream up to (but not including) the first Cluster —
/// this covers the EBML Header + Segment Info + Tracks, which is exactly what `MkvDemux::new`
/// needs to discover the tracks.
fn header_prefix(stream: &[u8]) -> Vec<u8> {
    let cluster = stream
        .windows(4)
        .position(|w| w == id::CLUSTER)
        .expect("stream has at least one Cluster");
    stream[..cluster].to_vec()
}

/// Run `bytesrc(stream) ! mkvdemux(header) ! recsink*` and return one [`Recorded`] per track,
/// keyed by track number in ascending order. Links each discovered src pad to its own sink.
fn run_demux(stream: Vec<u8>, chunk: usize, track_numbers: &[u64]) -> Vec<(u64, Recorded)> {
    let header = header_prefix(&stream);
    let mut p = Pipeline::new();
    let src = p.add(ByteSrc { data: stream, chunk, pos: 0 });
    let demux = p.add(MkvDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");

    // Preroll: the demuxer instantiates one src pad per track. Link each to a fresh sink.
    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), track_numbers.len(), "one src pad per track");

    let mut recs: Vec<(u64, Arc<Mutex<Recorded>>)> = Vec::new();
    for (i, ap) in added.iter().enumerate() {
        let rec = Arc::new(Mutex::new(Recorded::default()));
        let sink = p.add(RecSink { rec: Arc::clone(&rec) });
        p.link((ap.element, &ap.name), (sink, "sink")).expect("pad -> sink");
        // The demuxer names pads `src_track<N>` in track order (file order).
        recs.push((track_numbers[i], rec));
    }

    p.run().expect("run");

    recs.into_iter()
        .map(|(tn, rec)| (tn, Arc::try_unwrap(rec).unwrap().into_inner().unwrap()))
        .collect()
}

// =====================================================================================
// Tests
// =====================================================================================

/// A single A_FLAC track through the `MkvMux` **element** (`framesrc ! mkvmux ! bytesink`),
/// then demuxed: the reconstructed native FLAC stream is `CodecPrivate` (fLaC + STREAMINFO)
/// followed by every frame's bytes, in order, and the demuxed timestamps match the muxed
/// timeline.
#[test]
fn single_track_element_roundtrip() {
    let (codec_private, frames) = make_flac(48_000, 2, 4096, 6, 1);

    // Mux with the element, capturing its MKV output.
    let stream = mux_with_element(codec_private.clone(), &frames);

    // Demux and check the reconstructed native FLAC stream is head + frames concatenated.
    let recs = run_demux(stream, 4096, &[1]);
    assert_eq!(recs.len(), 1, "one track");
    let (tn, rec) = &recs[0];
    assert_eq!(*tn, 1);

    let mut want = codec_private.clone();
    for f in &frames {
        want.extend_from_slice(f);
    }
    assert_eq!(rec.bytes, want, "reconstructed native FLAC stream is head + frames, bit-exact");
    assert_eq!(&rec.bytes[..4], b"fLaC", "reconstruction begins with the native fLaC marker");
}

/// A two-track file (stereo 48k + mono 44.1k) through the **N-track** `MatroskaWriter` (the
/// mux library) → `MkvDemux`: each track's frames route to its own src pad, bit-exact, and the
/// two reconstructed streams are distinct (each carries its own STREAMINFO head).
#[test]
fn two_track_library_roundtrip() {
    let (cp1, frames1) = make_flac(48_000, 2, 4096, 5, 1);
    let (cp2, frames2) = make_flac(44_100, 1, 4096, 5, 777);

    let tracks = vec![
        TrackConfig::flac(1, cp1.clone(), 48_000.0, 2, 16),
        TrackConfig::flac(2, cp2.clone(), 44_100.0, 1, 16),
    ];
    let mut w = MatroskaWriter::new(tracks);
    let mut stream = Vec::new();
    w.write_header(&mut stream).unwrap();
    // Interleave the two tracks on a shared millisecond timeline.
    let dur = 85_000_000u64; // ~one 48k block, ns
    for i in 0..5 {
        w.write_frame(&mut stream, 1, i as u64 * dur, &frames1[i], true).unwrap();
        w.write_frame(&mut stream, 2, i as u64 * dur, &frames2[i], true).unwrap();
    }
    w.finalize(&mut stream);

    let recs = run_demux(stream, 500, &[1, 2]);
    assert_eq!(recs.len(), 2, "two tracks → two src pads");

    // Track 1: head cp1 + frames1.
    let (_, r1) = recs.iter().find(|(tn, _)| *tn == 1).expect("track 1");
    let mut want1 = cp1.clone();
    for f in &frames1 {
        want1.extend_from_slice(f);
    }
    assert_eq!(r1.bytes, want1, "track 1 reconstructed bit-exact");

    // Track 2: head cp2 + frames2.
    let (_, r2) = recs.iter().find(|(tn, _)| *tn == 2).expect("track 2");
    let mut want2 = cp2.clone();
    for f in &frames2 {
        want2.extend_from_slice(f);
    }
    assert_eq!(r2.bytes, want2, "track 2 reconstructed bit-exact");

    assert_ne!(r1.bytes, r2.bytes, "the two tracks demultiplexed distinctly");
}

/// Timestamps survive the round-trip: each demuxed buffer's pts equals the muxed frame pts,
/// scaled through TimestampScale. The codec head rides on the first frame's pts, so the pts
/// sequence a sink sees is `[t0, t0, t1, t2, …]` (head + first frame share `t0`).
#[test]
fn timestamps_survive_roundtrip() {
    let (codec_private, frames) = make_flac(48_000, 2, 4096, 4, 1);
    let stream = mux_with_element(codec_private, &frames);
    let recs = run_demux(stream, 4096, &[1]);
    let (_, rec) = &recs[0];

    // The muxer stamps PTS from the buffer PTS; the element source below carries none, so the
    // muxer synthesises a cadence of one 4096-sample block at 48 kHz per frame. Demuxed pts
    // are that timeline rounded to the millisecond TimestampScale.
    let block_ns = 4096u64 * 1_000_000_000 / 48_000; // ~85_333_333 ns
    let scale = 1_000_000u64; // default TimestampScale (1 ms/tick)
    let expected_frame_pts: Vec<u64> = (0..frames.len() as u64)
        .map(|i| {
            let ns = i * block_ns;
            ((ns + scale / 2) / scale) * scale // round to nearest tick, back to ns
        })
        .collect();

    // First buffer is the codec head at frame 0's pts; then one buffer per frame.
    let got: Vec<u64> = rec.ptss.iter().map(|p| p.expect("demuxed buffers carry a pts")).collect();
    assert_eq!(got.len(), frames.len() + 1, "codec head + one buffer per frame");
    assert_eq!(got[0], expected_frame_pts[0], "codec head rides the first frame's pts");
    assert_eq!(&got[1..], &expected_frame_pts[..], "each frame's pts matches the muxed timeline");
}

/// Mux `frames` for one A_FLAC track through the `MkvMux` **element** and return the MKV byte
/// stream (`framesrc ! mkvmux ! bytesink`).
fn mux_with_element(codec_private: Vec<u8>, frames: &[Vec<u8>]) -> Vec<u8> {
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(FrameSrc { frames: frames.to_vec(), next: 0 });
    let mux = p.add(MkvMux::flac(codec_private, 48_000.0, 2, 16));
    let sink = p.add(ByteCollect { got: Arc::clone(&got) });
    p.link((src, "src"), (mux, "sink")).expect("src -> mux");
    p.link((mux, "src"), (sink, "sink")).expect("mux -> sink");
    p.run().expect("run mux");
    let out = got.lock().unwrap().clone();
    out
}

// A minimal frame source + byte-collecting sink for the mux side (mirrors
// `element_roundtrip.rs`, local so the test crates stay independent).

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

struct ByteCollect {
    got: Arc<Mutex<Vec<u8>>>,
}
impl Element for ByteCollect {
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
