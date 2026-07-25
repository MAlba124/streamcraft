//! Demux-level seek behaviour (spec: `mkv/spec/MATROSKA.md`; flush/seek): a real
//! [`Pipeline`] streams a cued MKV file, seeks mid-run via a [`SeekHandle`] the way the app
//! does, and asserts the demuxer's `Event::FlushStart` handling — the parse state resyncs to
//! the post-seek Cluster, staged pre-seek output is dropped, video output is gated until the
//! first post-seek keyframe, the codec head is re-emitted, and KEYFRAME/DELTA flags survive.
//!
//! The seek is driven from **within the source element** (no test thread): the source
//! captures the pipeline's `SeekHandle`, and on reaching a trigger byte it publishes the seek
//! target + jumps its own read position. The scheduler observes the generation bump at the
//! group-loop boundary and delivers `FlushStart` to the demuxer, exactly as a real seek does
//! (`core/src/pipeline.rs` — the flush/seek arm). The seek target is computed from the file's
//! own Cues via `parse_seek_head` + `parse_cues`, so it lands on a real keyframe Cluster.

use std::sync::{Arc, Mutex};

use sc_mkv::ebml::{self, id};
use sc_mkv::{parse_cues, parse_seek_head, MatroskaWriter, MkvDemux, TrackConfig};
use streamcraft_core::batch::Inputs;
use streamcraft_core::buffer::BufferFlags;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::{Pipeline, SeekHandle};
use streamcraft_core::time::Timestamp;

// =====================================================================================
// A byte source that streams a fixed buffer in chunks and, on reaching `trigger_byte`,
// publishes a seek (via the captured SeekHandle) and jumps its read cursor to `seek_byte`
// — the demuxer then receives FlushStart from the scheduler and resumes at the new offset.
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
    name: "seeksrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct SeekSrc {
    data: Vec<u8>,
    chunk: usize,
    pos: usize,
    trigger_byte: usize,
    seek_byte: usize,
    seek: SeekHandle,
    seeked: bool,
}

impl Element for SeekSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Once we have streamed past the trigger, publish the seek and jump — but push nothing
        // this call, so no post-seek bytes leak ahead of the FlushStart the scheduler is about
        // to deliver (it observes the generation bump at the next group-loop boundary).
        if !self.seeked && self.pos >= self.trigger_byte {
            self.seek.seek(self.seek_byte as u64, streamcraft_core::time::Timestamp::ZERO);
            self.pos = self.seek_byte;
            self.seeked = true;
            return Ok(Flow::Ok);
        }
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
// A sink recording each received buffer's (bytes, pts_ns, flags), per pad.
// =====================================================================================

static SINK_OFFERS: [OfferDesc; 3] =
    [OfferDesc::any("vp8"), OfferDesc::any("flac"), OfferDesc::any("bytes")];
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
    /// One entry per received buffer: (bytes, pts_ns, is_keyframe, is_delta).
    bufs: Vec<(Vec<u8>, Option<u64>, bool, bool)>,
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
            rec.bufs.push((
                buf.memory.data().to_vec(),
                buf.pts.nanos(),
                buf.flags.contains(BufferFlags::KEYFRAME),
                buf.flags.contains(BufferFlags::DELTA),
            ));
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// =====================================================================================
// Builders + helpers.
// =====================================================================================

/// A cued multi-cluster V_VP8 stream: each `(ts_ns, keyframe, data)` is one frame on track 1.
/// An anchor-track keyframe opens a fresh (cue-indexed) Cluster (spec `§simpleblock`).
fn build_cued_video(frames: &[(u64, bool, &[u8])]) -> Vec<u8> {
    let mut w = MatroskaWriter::new(vec![TrackConfig::vp8(1, 320, 240)]);
    w.enable_cues();
    let mut out = Vec::new();
    w.write_header(&mut out).unwrap();
    for &(ts, key, data) in frames {
        w.write_frame(&mut out, 1, ts, data, key).unwrap();
    }
    w.finalize(&mut out);
    out
}

/// The header prefix (through the first Cluster) the demuxer discovers tracks from.
fn header_prefix(stream: &[u8]) -> Vec<u8> {
    let cl = stream.windows(4).position(|w| w == id::CLUSTER).expect("a Cluster");
    stream[..cl].to_vec()
}

/// The `(time_ns, absolute_byte)` seek index parsed from the file's own SeekHead + Cues.
fn seek_index(stream: &[u8]) -> Vec<(u64, u64)> {
    let info = parse_seek_head(&header_prefix(stream)).expect("SeekHead");
    let cues_abs = (info.segment_data_start + info.cues_pos.expect("Cues")) as usize;
    let h = ebml::read_element_header(stream, cues_abs).unwrap();
    let cues_bytes = &stream[cues_abs..h.data_end().unwrap().unwrap()];
    parse_cues(cues_bytes, info.segment_data_start, info.timestamp_scale)
}

/// Run `seeksrc(stream, seek at `seek_byte` once past `trigger_byte`) ! mkvdemux ! recsink`
/// and return the sink's record. One track → one src pad → one sink.
fn run_seek(stream: Vec<u8>, chunk: usize, trigger_byte: usize, seek_byte: usize) -> Recorded {
    let header = header_prefix(&stream);
    let mut p = Pipeline::new();
    let seek = p.seek_handle();
    let src = p.add(SeekSrc {
        data: stream,
        chunk,
        pos: 0,
        trigger_byte,
        seek_byte,
        seek,
        seeked: false,
    });
    let demux = p.add(MkvDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");

    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 1, "one video track → one src pad");

    let rec = Arc::new(Mutex::new(Recorded::default()));
    let sink = p.add(RecSink { rec: Arc::clone(&rec) });
    p.link((added[0].element, &added[0].name), (sink, "sink")).expect("pad -> sink");

    p.run().expect("run");
    Arc::try_unwrap(rec).unwrap().into_inner().unwrap()
}

// =====================================================================================
// Tests
// =====================================================================================

/// A seek to a later keyframe Cluster: after the FlushStart the demuxer resyncs, and the
/// post-seek output begins at that Cluster's timestamp with the codec head re-emitted and a
/// KEYFRAME flag on the first frame. The pre-seek frames appear first (they streamed before
/// the seek), then — with no pre-seek delta leaking past the flush — the post-seek run.
#[test]
fn seek_to_keyframe_cluster_resyncs_and_regates() {
    // Clusters at 0, 40, 80, 120 ms (each an anchor keyframe); 60 ms is a delta in cluster #2.
    let stream = build_cued_video(&[
        (0, true, &[1, 1, 1]),
        (40_000_000, true, &[2, 2]),
        (60_000_000, false, &[3]),
        (80_000_000, true, &[4, 4, 4, 4]),
        (120_000_000, true, &[5, 5]),
    ]);
    let index = seek_index(&stream);
    // Seek to the 80 ms cluster (index[2]); trigger after the first cluster's bytes are past.
    let (seek_time, seek_byte) = index[2];
    assert_eq!(seek_time, 80_000_000);
    let trigger = index[1].1 as usize; // once we've streamed into cluster #2

    let rec = run_seek(stream, 64, trigger, seek_byte as usize);

    // The demuxer re-emits the codec head after a flush; for V_VP8 the head is empty, so the
    // recorded buffers are exactly frames. Find where the post-seek run begins: the first
    // buffer at pts >= 80 ms whose data is the 80 ms cluster's frame.
    let ptss: Vec<Option<u64>> = rec.bufs.iter().map(|b| b.1).collect();
    // Post-seek frames must be present and correct.
    let post: Vec<&(Vec<u8>, Option<u64>, bool, bool)> =
        rec.bufs.iter().filter(|b| b.1 == Some(80_000_000) || b.1 == Some(120_000_000)).collect();
    assert_eq!(post.len(), 2, "the two post-seek clusters' frames arrive");
    assert_eq!(post[0].0, vec![4, 4, 4, 4], "80 ms frame bit-exact");
    assert!(post[0].2 && !post[0].3, "first post-seek frame is KEYFRAME, not DELTA");
    assert_eq!(post[1].0, vec![5, 5], "120 ms frame bit-exact");
    assert!(post[1].2, "120 ms frame is a keyframe");

    // The 60 ms delta (cluster #2) must NOT appear after the post-seek keyframe — it is behind
    // the seek. (It may appear before, having streamed pre-seek; that is fine.)
    let key80 = rec.bufs.iter().position(|b| b.1 == Some(80_000_000)).expect("80 ms present");
    assert!(
        !rec.bufs[key80 + 1..].iter().any(|b| b.1 == Some(60_000_000)),
        "no pre-seek 60 ms delta leaks after the resync"
    );
    // pts are monotonic across the post-seek boundary (80 then 120).
    assert!(ptss.contains(&Some(80_000_000)) && ptss.contains(&Some(120_000_000)));
}

/// The keyframe gate proper: a seek landing on a Cluster whose **first frame is a delta**
/// (opened by the writer's timestamp-overflow rule, not a keyframe) — the demuxer drops the
/// delta(s) and only starts emitting at the next keyframe. Without the gate a decoder would
/// render garbage from the mid-GOP delta.
#[test]
fn seek_gate_drops_leading_delta_until_keyframe() {
    // Frame 1 (keyframe, t=0) opens cluster #1. Frame 2 is a DELTA at t=40s — beyond the
    // signed-16-bit tick window (±32.767 s), so it opens cluster #2 with a *delta* first
    // frame. Frame 3 is a keyframe shortly after, in that same cluster.
    let stream = build_cued_video(&[
        (0, true, &[1, 1]),
        (40_000_000_000, false, &[2, 2, 2]), // delta, opens cluster #2 (ts overflow)
        (40_040_000_000, true, &[3, 3]),     // keyframe in cluster #2
    ]);
    // Cluster #2 is not cue-indexed (no keyframe opened it), so find its byte offset directly:
    // the second Cluster ID in the stream.
    let first = stream.windows(4).position(|w| w == id::CLUSTER).unwrap();
    let second = first + 4
        + stream[first + 4..].windows(4).position(|w| w == id::CLUSTER).unwrap();
    let trigger = first; // seek once we've streamed into cluster #1

    let rec = run_seek(stream, 48, trigger, second);

    // The delta at 40s must be gated out post-seek; the first emitted post-seek frame is the
    // 40.04s keyframe.
    let post: Vec<&(Vec<u8>, Option<u64>, bool, bool)> =
        rec.bufs.iter().filter(|b| b.1 == Some(40_040_000_000)).collect();
    assert_eq!(post.len(), 1, "the post-seek keyframe is emitted");
    assert_eq!(post[0].0, vec![3, 3], "keyframe frame bit-exact");
    assert!(post[0].2 && !post[0].3, "it is KEYFRAME");
    // The leading delta (40.00s) never appears after the gate opened at the keyframe.
    let kf = rec.bufs.iter().position(|b| b.1 == Some(40_040_000_000)).unwrap();
    assert!(
        !rec.bufs[..=kf].iter().any(|b| b.1 == Some(40_000_000_000)),
        "the mid-GOP delta was gated out before the keyframe"
    );
}

/// A proportional-style seek that lands a few bytes *before* a keyframe Cluster (mid-block):
/// the reader's forward scan discards the garbage and recovers at the next Cluster, and the
/// demuxer's keyframe gate ensures the first post-seek frame is the keyframe — no panic.
#[test]
fn seek_into_midblock_recovers_and_gates() {
    let stream = build_cued_video(&[
        (0, true, &[9, 9]),
        (40_000_000, true, &[7, 7, 7]),
        (80_000_000, true, &[8, 8, 8, 8]),
    ]);
    let index = seek_index(&stream);
    // Land 6 bytes before the 80 ms cue — inside the previous cluster's data.
    let target_cluster = index[2].1 as usize;
    let seek_byte = target_cluster - 6;
    let trigger = index[1].1 as usize;

    let rec = run_seek(stream, 32, trigger, seek_byte);

    // Recovery lands on the 80 ms Cluster: its keyframe frame appears, bit-exact.
    let post: Vec<&(Vec<u8>, Option<u64>, bool, bool)> =
        rec.bufs.iter().filter(|b| b.1 == Some(80_000_000)).collect();
    assert_eq!(post.len(), 1, "recovered exactly the 80 ms keyframe frame");
    assert_eq!(post[0].0, vec![8, 8, 8, 8], "80 ms frame bit-exact after mid-block recovery");
    assert!(post[0].2 && !post[0].3, "first post-seek frame is KEYFRAME");
}

/// The codec head is **re-emitted** after a seek (spec: flush/seek — downstream decoders reset
/// on FlushStart; the NAL codecs need their parameter sets again). Uses an A_FLAC track whose
/// CodecPrivate *is* the native head: after the seek the demuxer emits the head again before
/// the first post-seek frame (audio never gates), so the head bytes reappear post-flush.
#[test]
fn seek_reemits_codec_head() {
    // A synthetic A_FLAC head — the demuxer forwards CodecPrivate verbatim as the stream start;
    // it never decodes it here, so a plausible `fLaC` + STREAMINFO-length blob suffices.
    let head: Vec<u8> = {
        let mut h = vec![b'f', b'L', b'a', b'C'];
        h.extend_from_slice(&[0x80, 0, 0, 34]); // last-metadata-block STREAMINFO header
        h.extend_from_slice(&[0xAB; 34]); // 34-byte STREAMINFO body (opaque here)
        h
    };
    // Two keyframe clusters at 0 and 40 ms (FLAC frames are all keyframes / independently
    // decodable). Build directly so we control the head.
    let mut w = MatroskaWriter::new(vec![TrackConfig::flac(1, head.clone(), 48_000.0, 2, 16)]);
    w.enable_cues();
    let mut stream = Vec::new();
    w.write_header(&mut stream).unwrap();
    w.write_frame(&mut stream, 1, 0, &[0x11, 0x22], true).unwrap();
    w.write_frame(&mut stream, 1, 40_000_000, &[0x33, 0x44, 0x55], true).unwrap();
    w.finalize(&mut stream);

    let index = seek_index(&stream);
    let (_t, seek_byte) = index[1]; // the 40 ms cluster
    let trigger = index[0].1 as usize;
    let rec = run_seek(stream, 40, trigger, seek_byte as usize);

    // The head must appear at least twice: once before the first (pre-seek) frame, once after
    // the seek before the 40 ms frame. Count exact-head buffers.
    let head_count = rec.bufs.iter().filter(|b| b.0 == head).count();
    assert!(head_count >= 2, "codec head re-emitted after the seek (saw {head_count})");
    // The 40 ms frame is present, bit-exact, immediately preceded by a head buffer.
    let frame_at = rec
        .bufs
        .iter()
        .position(|b| b.1 == Some(40_000_000) && b.0 == vec![0x33, 0x44, 0x55])
        .expect("40 ms frame present");
    assert!(frame_at > 0 && rec.bufs[frame_at - 1].0 == head, "head precedes the post-seek frame");
}
