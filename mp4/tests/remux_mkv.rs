//! **MP4 → MKV remux**: `mp4demux(passthrough) ! mkvmux(from_caps)` in a real pipeline
//! over the checked-in H.264 fixture. Passthrough is the load-bearing idea: MP4 stores
//! length-prefixed NALs plus the raw `avcC` record — exactly Matroska's
//! `V_MPEG4/ISO/AVC` shape (RFC 9559 §12) — so a remux must not touch a single sample
//! byte. Verified against `Mp4Reader` as the oracle:
//!
//! - the muxed track is `V_MPEG4/ISO/AVC` with `CodecPrivate` == the `avcC` record,
//!   verbatim, and the announced coded dimensions;
//! - every sample's bytes survive bit-exact (still length-prefixed — no Annex B);
//! - the sample table's **sync bits** survive as SimpleBlock keyframe flags;
//! - timestamps survive to the millisecond TimestampScale (round-to-nearest).

use std::sync::{Arc, Mutex};

use sc_mkv::MatroskaReader;
use sc_mp4::{Mp4Demux, Mp4Reader};
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

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
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
        let n = self.chunk.min(buf.memory.capacity()).min(self.data.len() - self.pos);
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

static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "bytecollect",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

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
        let advance = if size == 0 {
            file.len() - at
        } else if size == 1 {
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

/// Round `ns` to the muxer's millisecond TimestampScale (round-to-nearest, back to ns).
fn quantize_ms(ns: u64) -> u64 {
    let scale = sc_mkv::DEFAULT_TIMESTAMP_SCALE;
    ((ns + scale / 2) / scale) * scale
}

#[test]
fn h264_mp4_remuxes_to_conformant_mkv() {
    let file = fixture_bytes("tiny_h264.mp4");
    let head = head_through_moov(&file);

    // Oracle: every sample (bytes verbatim, pts, sync) + the raw avcC record.
    let mut oracle = Mp4Reader::new(&head).expect("oracle resolves");
    let record = oracle.tracks()[0].entry.config_record.clone();
    assert!(!record.is_empty(), "fixture has an avcC record");
    let (width, height) = (oracle.tracks()[0].width, oracle.tracks()[0].height);
    oracle.push(&file);
    let mut samples: Vec<(u64, bool, Vec<u8>)> = Vec::new();
    // Copy the sample out first: `ResolvedSample` borrows the reader, and `ticks_to_ns`
    // needs the reader again.
    while let Some((ti, pts, sync, bytes)) =
        oracle.next_sample().map(|s| (s.track_index, s.pts, s.sync, s.bytes.to_vec()))
    {
        samples.push((oracle.ticks_to_ns(ti, pts), sync, bytes));
    }
    assert!(!samples.is_empty(), "fixture yields samples");

    // The remux pipeline: bytesrc ! mp4demux(passthrough) ! mkvmux ! collect.
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(ByteSrc { data: file, chunk: 1000, pos: 0 });
    let demux = p.add(Mp4Demux::passthrough(head));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");
    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 1, "fixture is single-track");
    let mux = p.add(sc_mkv::MkvMux::from_caps());
    let sink = p.add(ByteCollect { got: Arc::clone(&got) });
    p.link((added[0].element, &added[0].name), (mux, "sink")).expect("demux -> mux");
    p.link((mux, "src"), (sink, "sink")).expect("mux -> sink");
    p.run().expect("remux runs");
    let out = got.lock().unwrap().clone();

    // Verify the produced Matroska against the oracle.
    let mut r = MatroskaReader::new();
    r.push(&out).expect("remuxed MKV parses");
    assert_eq!(r.tracks().len(), 1);
    let t = &r.tracks()[0];
    assert_eq!(t.codec_id, "V_MPEG4/ISO/AVC");
    assert_eq!(t.codec_private, record, "CodecPrivate is the avcC record, verbatim");
    assert_eq!((t.pixel_width, t.pixel_height), (width, height), "announced dims");

    let mut frames = Vec::new();
    while let Some(f) = r.next_frame() {
        frames.push(f);
    }
    assert_eq!(frames.len(), samples.len(), "one SimpleBlock per sample");
    for (i, (f, (pts_ns, sync, bytes))) in frames.iter().zip(&samples).enumerate() {
        assert_eq!(&f.data, bytes, "sample {i} bytes bit-exact (still length-prefixed)");
        assert_eq!(f.pts_ns, quantize_ms(*pts_ns), "sample {i} pts to the ms tick");
        assert_eq!(f.keyframe, *sync, "sample {i} sync bit survived as the keyframe flag");
    }
}
