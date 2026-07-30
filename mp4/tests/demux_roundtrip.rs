//! End-to-end demux through the [`Mp4Demux`] element in a real [`Pipeline`] (spec:
//! `spec/NOTES.md`; dynamic pads). Mirrors `mkv/tests/demux_roundtrip.rs`.
//!
//! The load-bearing properties over the committed H.264-in-MP4 fixture:
//! - **one dynamic src pad per track**, settled during `preroll` then frozen;
//! - each sample leaves as **one output buffer**, Annex-B-reframed from the file's
//!   length-prefixed NALs, prefixed by the `avcC` parameter-set head on the first sample;
//! - **pts survives** (the container's composition timeline, ns);
//! - **ragged chunking** — the file is streamed in many small chunk sizes that straddle
//!   sample boundaries, exercising the cross-chunk byte buffering; the result is identical
//!   regardless of chunk size.

use std::sync::{Arc, Mutex};

use pf_mp4::Mp4Demux;
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

// ---- byte source: streams a fixed buffer in fixed-size chunks, then EOS ------------------

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

// ---- recording sink: one per demuxer src pad, captures (bytes, pts) per buffer -----------

static SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("h264/annexb"), OfferDesc::any("bytes")];
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
    /// Every buffer's bytes concatenated in order (the reconstructed elementary stream).
    bytes: Vec<u8>,
    /// The pts (ns) of each received buffer, in order.
    ptss: Vec<Option<u64>>,
    /// The byte length of each received buffer, in order (one per output buffer / sample +
    /// the codec head buffer).
    lens: Vec<usize>,
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
            rec.lens.push(buf.memory.data().len());
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// ---- helpers -----------------------------------------------------------------------------

fn fixture_bytes(name: &str) -> Vec<u8> {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read fixture {name}: {e}"))
}

/// The file head through `moov` (the prefix a `+faststart` file puts before `mdat`).
fn head_through_moov(file: &[u8]) -> Vec<u8> {
    let mut at = 0usize;
    while at + 8 <= file.len() {
        let size = u32::from_be_bytes([file[at], file[at + 1], file[at + 2], file[at + 3]]) as usize;
        if &file[at + 4..at + 8] == b"mdat" {
            return file[..at].to_vec();
        }
        let advance = if size == 0 { file.len() - at } else if size == 1 {
            u64::from_be_bytes(file[at + 8..at + 16].try_into().unwrap()) as usize
        } else {
            size
        };
        if advance == 0 {
            break;
        }
        at += advance;
    }
    file.to_vec()
}

/// Run `bytesrc(file, chunk) ! mp4demux(head) ! recsink*` and return one [`Recorded`] per
/// track, in the demuxer's pad order (file/track order).
fn run_demux(file: Vec<u8>, chunk: usize) -> Vec<Recorded> {
    let head = head_through_moov(&file);
    let mut p = Pipeline::new();
    let src = p.add(ByteSrc { data: file, chunk, pos: 0 });
    let demux = p.add(Mp4Demux::new(head));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");

    // Preroll: the demuxer instantiates one src pad per track. Link each to a fresh sink.
    let added = p.preroll().expect("preroll");
    let mut recs: Vec<Arc<Mutex<Recorded>>> = Vec::new();
    for ap in &added {
        let rec = Arc::new(Mutex::new(Recorded::default()));
        let sink = p.add(RecSink { rec: Arc::clone(&rec) });
        p.link((ap.element, &ap.name), (sink, "sink")).expect("pad -> sink");
        recs.push(rec);
    }

    p.run().expect("run");
    recs.into_iter()
        .map(|rec| Arc::try_unwrap(rec).unwrap().into_inner().unwrap())
        .collect()
}

// ---- tests -------------------------------------------------------------------------------

const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// The fixture's single H.264 track demuxes to one src pad; the reconstructed stream is the
/// Annex B parameter-set head (SPS+PPS) followed by one Annex B access unit per sample, each
/// carrying a pts. Every emitted buffer begins with a start code (a valid Annex B stream).
#[test]
fn single_h264_track_reconstructs_annex_b() {
    let file = fixture_bytes("tiny_h264.mp4");
    let recs = run_demux(file, 4096);
    assert_eq!(recs.len(), 1, "one track → one src pad");
    let rec = &recs[0];

    // The stream begins with the avcC parameter-set head (a start-code SPS, then PPS).
    assert_eq!(&rec.bytes[..4], &START_CODE, "reconstruction begins with an Annex B start code");
    // 15 samples + 1 codec-head buffer = 16 output buffers.
    assert_eq!(rec.lens.len(), 16, "codec head + one buffer per sample");
    // Every buffer is a start-code-framed Annex B chunk.
    let mut off = 0;
    for &len in &rec.lens {
        assert!(len >= 4, "each buffer is at least a start code + payload");
        assert_eq!(&rec.bytes[off..off + 4], &START_CODE, "each output buffer starts with a start code");
        off += len;
    }
    assert_eq!(off, rec.bytes.len(), "buffer lengths tile the reconstructed stream");
}

/// Timestamps survive the container round-trip: the demuxed pts sequence is monotone
/// non-decreasing and every buffer carries a pts. The codec head rides the first sample's pts.
#[test]
fn demuxed_timestamps_are_present_and_monotone() {
    let file = fixture_bytes("tiny_h264.mp4");
    let recs = run_demux(file, 4096);
    let rec = &recs[0];
    let ptss: Vec<u64> = rec.ptss.iter().map(|p| p.expect("every demuxed buffer carries a pts")).collect();
    // 15 fps → 15 samples; the head shares the first sample's pts, so 16 pts values.
    assert_eq!(ptss.len(), 16);
    assert_eq!(ptss[0], ptss[1], "codec head rides the first sample's pts");
    // The first presented sample is at 0 (baseline, no ctts, no edit trim needed).
    assert_eq!(ptss[0], 0, "first sample presents at 0");
    // Sample pts (skipping the head buffer) are strictly increasing at ~1/15 s spacing.
    for w in ptss[1..].windows(2) {
        assert!(w[1] > w[0], "sample pts strictly increase: {} !> {}", w[1], w[0]);
    }
    // Last sample ~ 14/15 s (rounded through the 15360 timescale).
    let last = *ptss.last().unwrap();
    let expected = 14u64 * 1_000_000_000 / 15; // ~933_333_333 ns
    assert!(
        last.abs_diff(expected) < 2_000_000,
        "last pts ~ 14/15 s (got {last}, want ~{expected})"
    );
}

/// Ragged chunking sweep: the reconstructed stream + pts are identical for every chunk size,
/// so the cross-chunk byte buffering handles sample boundaries at arbitrary offsets.
#[test]
fn ragged_chunking_is_chunk_size_invariant() {
    let file = fixture_bytes("tiny_h264.mp4");
    let reference = run_demux(file.clone(), 1 << 20); // whole-file chunk
    let ref_rec = &reference[0];

    for &chunk in &[1usize, 2, 3, 7, 13, 64, 137, 512, 1000, 4096] {
        let got = run_demux(file.clone(), chunk);
        assert_eq!(got.len(), 1, "chunk {chunk}: still one track");
        assert_eq!(got[0].bytes, ref_rec.bytes, "chunk {chunk}: reconstructed stream identical");
        assert_eq!(got[0].ptss, ref_rec.ptss, "chunk {chunk}: pts sequence identical");
        assert_eq!(got[0].lens, ref_rec.lens, "chunk {chunk}: per-buffer framing identical");
    }
}

/// The B-frame fixture demuxes too (ctts present): still one track, one buffer per sample +
/// head, and every buffer carries a pts (the reorder is handled by the container timeline).
#[test]
fn bframes_fixture_demuxes() {
    let file = fixture_bytes("bframes_h264.mp4");
    let recs = run_demux(file, 4096);
    assert_eq!(recs.len(), 1);
    let rec = &recs[0];
    assert_eq!(rec.lens.len(), 16, "codec head + 15 samples");
    assert!(rec.ptss.iter().all(|p| p.is_some()), "every buffer carries a pts");
    assert_eq!(&rec.bytes[..4], &START_CODE);
}
