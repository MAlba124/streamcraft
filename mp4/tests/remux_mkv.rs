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
    let duration = oracle.tracks()[0].duration_ns;
    assert!(duration > 0, "fixture declares an mdhd duration");
    oracle.push_bytes(&file);
    let mut samples: Vec<(u64, bool, Vec<u8>)> = Vec::new();
    while let Some(s) = oracle.next_sample() {
        samples.push((
            oracle.ticks_to_ns(s.track_index, s.pts),
            s.sync,
            s.payload.data().to_vec(),
        ));
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
    let want_dur = duration;
    let got_dur = r.duration_ns().expect("remux declares Info\\Duration — not a live stream");
    assert!(
        got_dur.abs_diff(want_dur) <= sc_mkv::DEFAULT_TIMESTAMP_SCALE,
        "duration {got_dur} ns ≈ mdhd duration {want_dur} ns (within one ms tick)"
    );

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

/// **Multi-track remux**: `mp4demux(passthrough) ! mkvmuxn` over the two-track (avc1 +
/// mp4a) fixture — video AND audio in one output file. The load-bearing properties:
///
/// - two Matroska tracks: `V_MPEG4/ISO/AVC` with the `avcC` record as CodecPrivate and
///   `A_AAC` with the esds-carried **AudioSpecificConfig** as CodecPrivate (RFC 9559
///   §12) — both verbatim;
/// - every sample of *both* tracks bit-exact (video still length-prefixed, audio raw
///   AAC access units — no ADTS);
/// - per-track sync bits survive as SimpleBlock keyframe flags (every AAC AU is sync);
/// - blocks interleave in global pts order across the tracks;
/// - `Info\Duration` = the max of the tracks' mdhd durations.
#[test]
fn av_mp4_remuxes_to_two_track_mkv() {
    let file = fixture_bytes("tiny_av.mp4");
    let head = head_through_moov(&file);

    // Oracle: per-track config records + every sample, routed by track.
    let mut oracle = Mp4Reader::new(&head).expect("oracle resolves");
    assert_eq!(oracle.tracks().len(), 2, "fixture is two-track");
    let vidx = oracle.tracks().iter().position(|t| t.width != 0).expect("video track");
    let aidx = oracle.tracks().iter().position(|t| t.family() == "aac").expect("aac track");
    let record = oracle.tracks()[vidx].entry.config_record.clone();
    let asc = oracle.tracks()[aidx].entry.config_record.clone();
    assert!(!record.is_empty(), "fixture has an avcC record");
    assert!(!asc.is_empty(), "fixture's esds yields an AudioSpecificConfig");
    let (rate, channels) =
        (oracle.tracks()[aidx].entry.sample_rate, oracle.tracks()[aidx].entry.channels);
    let want_dur = oracle.tracks().iter().map(|t| t.duration_ns).max().unwrap();
    let pad_names: Vec<String> = oracle
        .tracks()
        .iter()
        .map(|t| format!("src_track{}", t.track_id))
        .collect();
    oracle.push_bytes(&file);
    // Per-track (pts_ns, sync, bytes) in file (== decode) order.
    let mut want: Vec<Vec<(u64, bool, Vec<u8>)>> = vec![Vec::new(), Vec::new()];
    while let Some(s) = oracle.next_sample() {
        want[s.track_index].push((
            oracle.ticks_to_ns(s.track_index, s.pts),
            s.sync,
            s.payload.data().to_vec(),
        ));
    }
    assert!(!want[vidx].is_empty() && !want[aidx].is_empty(), "both tracks yield samples");

    // The remux pipeline: bytesrc ! mp4demux(passthrough) ⇉ mkvmuxn ! collect.
    // Video → sink_0 (track 1), audio → sink_1 (track 2).
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(ByteSrc { data: file, chunk: 1000, pos: 0 });
    let demux = p.add(Mp4Demux::passthrough(head));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");
    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 2, "one dynamic pad per track");
    let mux = p.add(sc_mkv::MkvMux::multi(2));
    let sink = p.add(ByteCollect { got: Arc::clone(&got) });
    for (i, &tidx) in [vidx, aidx].iter().enumerate() {
        let ap = added
            .iter()
            .find(|ap| ap.name == pad_names[tidx])
            .unwrap_or_else(|| panic!("pad {} discovered", pad_names[tidx]));
        p.link((ap.element, &ap.name), (mux, &format!("sink_{i}")))
            .expect("demux -> mux");
    }
    p.link((mux, "src"), (sink, "sink")).expect("mux -> sink");
    p.run().expect("remux runs");
    let out = got.lock().unwrap().clone();

    // Verify the produced Matroska against the oracle.
    let mut r = MatroskaReader::new();
    r.push(&out).expect("remuxed MKV parses");
    let tracks = r.tracks().to_vec();
    assert_eq!(tracks.len(), 2, "two TrackEntries");
    // The video track is the Cluster anchor and leads the Tracks element.
    assert_eq!(tracks[0].codec_id, "V_MPEG4/ISO/AVC");
    assert_eq!(tracks[0].track_number, 1, "video pad (sink_0) → track 1");
    assert_eq!(tracks[0].codec_private, record, "video CodecPrivate = avcC verbatim");
    assert_eq!(tracks[1].codec_id, "A_AAC");
    assert_eq!(tracks[1].track_number, 2, "audio pad (sink_1) → track 2");
    assert_eq!(tracks[1].codec_private, asc, "audio CodecPrivate = the ASC verbatim");
    assert_eq!(tracks[1].sampling_frequency, rate as f64);
    assert_eq!(tracks[1].channels, channels);

    let got_dur = r.duration_ns().expect("Info\\Duration declared");
    assert!(
        got_dur.abs_diff(want_dur) <= sc_mkv::DEFAULT_TIMESTAMP_SCALE,
        "duration {got_dur} ns ≈ max mdhd duration {want_dur} ns (within one ms tick)"
    );

    // Frames: bit-exact per track in decode order, keyframe bits, pts to the ms tick,
    // and globally pts-ordered interleave (non-strict — both tracks start at pts 0).
    let mut seen = [0usize, 0usize]; // [track1(video), track2(audio)]
    let mut last_pts = 0u64;
    while let Some(f) = r.next_frame() {
        assert!(f.pts_ns >= last_pts, "blocks interleave in pts order");
        last_pts = f.pts_ns;
        let (tidx, si) = match f.track_number {
            1 => (vidx, 0),
            2 => (aidx, 1),
            n => panic!("unexpected track {n}"),
        };
        let (pts_ns, sync, ref bytes) = want[tidx][seen[si]];
        assert_eq!(&f.data, bytes, "track {} sample {} bit-exact", f.track_number, seen[si]);
        assert_eq!(f.pts_ns, quantize_ms(pts_ns), "track {} sample {} pts", f.track_number, seen[si]);
        assert_eq!(f.keyframe, sync, "track {} sample {} keyframe bit", f.track_number, seen[si]);
        seen[si] += 1;
    }
    assert_eq!(seen[0], want[vidx].len(), "every video sample arrived");
    assert_eq!(seen[1], want[aidx].len(), "every audio sample arrived");
}
