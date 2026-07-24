//! **Remux**: `mkvdemux ! mkvmux` in a real pipeline (spec: dynamic caps — the
//! caps-driven muxer). The load-bearing properties:
//!
//! - [`MkvMux::from_caps`] builds its track entirely from the demuxer's runtime
//!   announcement (+ the in-band FLAC head for `A_FLAC`) — no constructor arguments;
//! - every frame's bytes and (millisecond-quantised) timestamps survive the
//!   demux→remux round-trip bit-exact;
//! - the container's **keyframe bits** survive: the demuxer tags `KEYFRAME`/`DELTA`
//!   per frame and the muxer writes them back (an untagged all-keyframe FLAC stream
//!   and an explicitly mixed VP8 stream both come out right);
//! - a family the muxer cannot write conformantly (`av1` today) fails the run
//!   loudly at the announcement, not silently as a corrupt file.

use std::sync::{Arc, Mutex};

use sc_flac::{FlacEncoder, SampleFormat};
use sc_mkv::ebml::id;
use sc_mkv::{MatroskaReader, MatroskaWriter, MkvDemux, MkvMux, TrackConfig};
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

// --- byte source / byte sink (mirrors demux_roundtrip.rs, local for independence) ------

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

// --- helpers ---------------------------------------------------------------------------

/// Real native FLAC head (`fLaC` + finalised STREAMINFO) plus `n` encoded frames.
fn make_flac(sample_rate: u32, channels: u32, n: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let (mut enc, mut header) = FlacEncoder::new(sample_rate, channels, SampleFormat::S16).unwrap();
    let mut frames = Vec::new();
    let mut phase = 7i32;
    for _ in 0..n {
        let mut pcm = Vec::new();
        for _ in 0..4096 {
            for c in 0..channels {
                let v = ((phase.wrapping_mul(37) + c as i32 * 501) & 0x3FFF) as i16 - 0x2000;
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

/// Header prefix (everything before the first Cluster) — what `MkvDemux::new` needs.
fn header_prefix(stream: &[u8]) -> Vec<u8> {
    let cluster = stream
        .windows(4)
        .position(|w| w == id::CLUSTER)
        .expect("stream has at least one Cluster");
    stream[..cluster].to_vec()
}

/// Run the remux pipeline `bytesrc(stream) ! mkvdemux ! mkvmux(from_caps) ! bytecollect`
/// and return the remuxed MKV bytes. `Err` if the run fails (the unsupported-family test).
fn remux(stream: Vec<u8>, chunk: usize) -> Result<Vec<u8>, Error> {
    let header = header_prefix(&stream);
    let got = Arc::new(Mutex::new(Vec::new()));

    let mut p = Pipeline::new();
    let src = p.add(ByteSrc { data: stream, chunk, pos: 0 });
    let demux = p.add(MkvDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");

    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 1, "single-track remux: one discovered src pad");

    let mux = p.add(MkvMux::from_caps());
    let sink = p.add(ByteCollect { got: Arc::clone(&got) });
    p.link((added[0].element, &added[0].name), (mux, "sink")).expect("demux -> mux");
    p.link((mux, "src"), (sink, "sink")).expect("mux -> sink");

    p.run()?;
    let out = got.lock().unwrap().clone();
    Ok(out)
}

/// Parse an MKV byte stream fully: `(tracks, frames)`.
fn read_all(stream: &[u8]) -> (Vec<sc_mkv::Track>, Vec<sc_mkv::Frame>) {
    let mut r = MatroskaReader::new();
    r.push(stream).expect("remuxed stream parses");
    let mut frames = Vec::new();
    while let Some(f) = r.next_frame() {
        frames.push(f);
    }
    (r.tracks().to_vec(), frames)
}

// --- tests -----------------------------------------------------------------------------

/// FLAC: the caps-driven muxer learns the whole track from the announcement + the
/// in-band native head — CodecPrivate, audio params (parsed from STREAMINFO), frames
/// and timestamps all bit-exact after demux → remux.
#[test]
fn flac_remux_is_bit_exact() {
    let (head, frames) = make_flac(48_000, 2, 6);

    // Source file: one A_FLAC track, frames on an exact-millisecond timeline (so the
    // ms-quantised container timestamps round-trip exactly).
    let mut w = MatroskaWriter::new(vec![TrackConfig::flac(1, head.clone(), 48_000.0, 2, 16)]);
    let mut src = Vec::new();
    w.write_header(&mut src).unwrap();
    for (i, f) in frames.iter().enumerate() {
        w.write_frame(&mut src, 1, i as u64 * 85_000_000, f, true).unwrap();
    }
    w.finalize(&mut src);

    let out = remux(src, 900).expect("remux runs");
    let (tracks, got) = read_all(&out);

    assert_eq!(tracks.len(), 1);
    let t = &tracks[0];
    assert_eq!(t.codec_id, "A_FLAC");
    assert_eq!(t.codec_private, head, "CodecPrivate absorbed from the in-band head, verbatim");
    assert_eq!(
        (t.sampling_frequency, t.channels, t.bit_depth),
        (48_000.0, 2, 16),
        "audio params parsed from STREAMINFO (RFC 9639 §8.2)"
    );

    assert_eq!(got.len(), frames.len(), "one block per source frame");
    for (i, (g, want)) in got.iter().zip(&frames).enumerate() {
        assert_eq!(&g.data, want, "frame {i} bytes bit-exact");
        assert_eq!(g.pts_ns, i as u64 * 85_000_000, "frame {i} timestamp preserved");
        assert!(g.keyframe, "frame {i}: FLAC frames are all keyframes");
    }
}

/// VP8: track dimensions come from the announcement, and the container's keyframe
/// bits survive the round-trip — the demuxer's explicit `KEYFRAME`/`DELTA` tagging is
/// what makes a remuxed video stream still seekable.
#[test]
fn vp8_remux_preserves_dims_and_keyframes() {
    // Synthetic VP8 "frames" (the container never inspects payloads) with a mixed
    // keyframe pattern on a 20 ms timeline.
    let frames: Vec<Vec<u8>> = (0..8u8).map(|i| vec![i; 40 + i as usize]).collect();
    let key = [true, false, false, true, false, true, false, false];

    let mut w = MatroskaWriter::new(vec![TrackConfig::vp8(1, 320, 240)]);
    let mut src = Vec::new();
    w.write_header(&mut src).unwrap();
    for (i, f) in frames.iter().enumerate() {
        w.write_frame(&mut src, 1, i as u64 * 20_000_000, f, key[i]).unwrap();
    }
    w.finalize(&mut src);

    let out = remux(src, 700).expect("remux runs");
    let (tracks, got) = read_all(&out);

    assert_eq!(tracks.len(), 1);
    let t = &tracks[0];
    assert_eq!(t.codec_id, "V_VP8");
    assert_eq!((t.pixel_width, t.pixel_height), (320, 240), "dims from the announcement");

    assert_eq!(got.len(), frames.len());
    for (i, (g, want)) in got.iter().zip(&frames).enumerate() {
        assert_eq!(&g.data, want, "frame {i} bytes bit-exact");
        assert_eq!(g.pts_ns, i as u64 * 20_000_000, "frame {i} timestamp preserved");
        assert_eq!(
            g.keyframe, key[i],
            "frame {i}: keyframe bit preserved through demux flags → mux"
        );
    }
}

/// A family the muxer cannot write conformantly yet (`av1` — needs an av1C
/// CodecPrivate) fails the run loudly at the announcement instead of producing a
/// silently non-conformant file.
#[test]
fn unsupported_family_fails_loudly() {
    let mut w = MatroskaWriter::new(vec![TrackConfig::video(1, "V_AV1", Vec::new(), 640, 360)]);
    let mut src = Vec::new();
    w.write_header(&mut src).unwrap();
    w.write_frame(&mut src, 1, 0, &[0u8; 64], true).unwrap();
    w.finalize(&mut src);

    assert!(remux(src, 4096).is_err(), "av1 remux must be a loud error, not a corrupt file");
}
