//! `MkvMuxN` — the multi-track muxer element — end to end in a real [`Pipeline`]
//! (spec: Aggregation; RFC 9559). Two announcing sources (an `h264/avcc`-shaped video
//! track and an `aac` track) fan into `mkvmuxn`; the output must be one well-formed
//! two-track Matroska stream:
//!
//! - per-track `CodecPrivate` from the **first buffer** (the raw record / the raw
//!   AudioSpecificConfig — RFC 9559 §12), verbatim;
//! - every frame's bytes bit-exact per track, in per-track order;
//! - **global pts-ordered interleave** across tracks (the merge pops the smallest
//!   head-of-queue timestamp), deterministic regardless of thread timing;
//! - SimpleBlock keyframe bits preserved (audio untagged-as-keyframe per the DELTA
//!   convention);
//! - `Info\Duration` = the max of the pads' announced durations.
//!
//! Plus the loud failure modes: a linked pad that never announces while others stream
//! (the bounded pre-header queue), a mislinked expected-count, and data before any
//! announcement.

use std::sync::{Arc, Mutex};

use pf_mkv::{mux_multi, MatroskaReader, MkvMux, MkvMuxN};
use profluens_core::batch::Inputs;
use profluens_core::buffer::BufferFlags;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;

// ---- an announcing frame source ----------------------------------------------------
//
// Announces a family + fields on its src pad, emits an optional codec head as the first
// buffer (the first-buffer-is-CodecPrivate contract), then one frame per process pass
// with pts + keyframe/delta tags. The offer menu declares both families *and their
// field names* so the announcement's names are interned at link time (spec: Formats —
// dynamic caps).

static ANN_FIELDS: [FieldDesc; 5] = [
    FieldDesc { field: "width", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "height", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "rate", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "channels", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "duration", allowed: ConstraintDesc::Any, preferred: None },
];
static SRC_OFFERS: [OfferDesc; 2] = [
    OfferDesc { family: "h264/avcc", fields: &ANN_FIELDS },
    OfferDesc { family: "aac", fields: &ANN_FIELDS },
];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &SRC_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "annsrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

/// One frame to emit: pts (ns), keyframe?, payload bytes.
type Frame = (u64, bool, Vec<u8>);

struct AnnSrc {
    family: &'static str,
    fields: Vec<(&'static str, ValueDesc)>,
    /// Emitted as the leading buffer (CodecPrivate) when non-empty.
    head: Vec<u8>,
    frames: Vec<Frame>,
    at: usize,
    announced: bool,
    head_sent: bool,
}

impl AnnSrc {
    fn new(family: &'static str, fields: Vec<(&'static str, ValueDesc)>, head: Vec<u8>, frames: Vec<Frame>) -> Self {
        Self { family, fields, head, frames, at: 0, announced: false, head_sent: false }
    }

    fn emit(ctx: &mut Ctx, bytes: &[u8], pts: Option<u64>, flags: BufferFlags) -> bool {
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return false };
        assert!(buf.memory.capacity() >= bytes.len(), "test frame exceeds pool slot");
        buf.memory.as_mut_full()[..bytes.len()].copy_from_slice(bytes);
        buf.memory.set_len(bytes.len());
        if let Some(t) = pts {
            buf.pts = Timestamp::from_nanos(t);
        }
        buf.flags = flags;
        ctx.out(PadId(0)).push(buf);
        true
    }
}

impl Element for AnnSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.at = 0;
        self.announced = false;
        self.head_sent = false;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.announced {
            ctx.announce_format(PadId(0), self.family, &self.fields);
            self.announced = true;
        }
        if !self.head_sent {
            if !self.head.is_empty() && !Self::emit(ctx, &self.head.clone(), None, BufferFlags::empty()) {
                return Ok(Flow::Ok); // pool dry — retry
            }
            self.head_sent = true;
            return Ok(Flow::Ok);
        }
        if self.at >= self.frames.len() {
            return Ok(Flow::Eos);
        }
        let (pts, key, bytes) = self.frames[self.at].clone();
        let flags = if key { BufferFlags::KEYFRAME } else { BufferFlags::DELTA };
        if Self::emit(ctx, &bytes, Some(pts), flags) {
            self.at += 1;
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// ---- a byte-collecting sink ---------------------------------------------------------

static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "collectsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct CollectSink {
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl Element for CollectSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut sink = self.bytes.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            sink.extend_from_slice(buf.memory.data());
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// ---- fixtures -----------------------------------------------------------------------

/// A fabricated (structurally plausible) avcC record: configurationVersion 1 (the guard
/// the muxer checks), profile/level, lengthSizeMinusOne 3, one SPS, one PPS (ISO/IEC
/// 14496-15 §5.3.3.1). The muxer takes it verbatim — content is never parsed.
fn avcc_record() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0xD9];
    let pps = [0x68u8, 0xCB, 0x83, 0xCB, 0x20];
    let mut rec = vec![1u8, 0x42, 0xC0, 0x1E, 0xFF, 0xE1];
    rec.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    rec.extend_from_slice(&sps);
    rec.push(1);
    rec.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    rec.extend_from_slice(&pps);
    rec
}

/// AudioSpecificConfig 0x11 0x90: AOT 2 (AAC-LC), samplingFrequencyIndex 3 (48 kHz),
/// channelConfiguration 2 (ISO/IEC 14496-3 §1.6.2.1).
fn asc() -> Vec<u8> {
    vec![0x11, 0x90]
}

/// Video frames: 10 length-prefixed-NAL-shaped payloads at 40 ms cadence, keyframes at
/// 0 and 5 (two Clusters via the anchor-keyframe cadence).
fn video_frames() -> Vec<Frame> {
    (0..10u64)
        .map(|i| {
            let body = vec![if i % 5 == 0 { 0x65u8 } else { 0x41 }, i as u8, 0xAA, 0xBB];
            let mut f = (body.len() as u32).to_be_bytes().to_vec();
            f.extend_from_slice(&body);
            (i * 40_000_000, i % 5 == 0, f)
        })
        .collect()
}

/// Audio frames: 15 raw-AU-shaped payloads at 20 ms cadence offset by 10 ms, so every
/// pts across both tracks is distinct → one deterministic global merge order. Untagged
/// keyframe (KEYFRAME flag), like a demuxed all-sync audio track.
fn audio_frames() -> Vec<Frame> {
    (0..15u64)
        .map(|i| (10_000_000 + i * 20_000_000, true, vec![0x21, i as u8, 0xCC]))
        .collect()
}

// ---- the tests ----------------------------------------------------------------------

/// The full two-track happy path (see the module docs for the properties).
#[test]
fn two_tracks_mux_pts_ordered_bit_exact() {
    let vframes = video_frames();
    let aframes = audio_frames();
    let (dur_v, dur_a) = (400_000_000u64, 300_000_000u64);

    let out = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let v = p.add(AnnSrc::new(
        "h264/avcc",
        vec![
            ("width", ValueDesc::Int(320)),
            ("height", ValueDesc::Int(240)),
            ("duration", ValueDesc::Int(dur_v as i64)),
        ],
        avcc_record(),
        vframes.clone(),
    ));
    let a = p.add(AnnSrc::new(
        "aac",
        vec![
            ("rate", ValueDesc::Int(48_000)),
            ("channels", ValueDesc::Int(2)),
            ("duration", ValueDesc::Int(dur_a as i64)),
        ],
        asc(),
        aframes.clone(),
    ));
    let mux = p.add(MkvMux::multi(2));
    let sink = p.add(CollectSink { bytes: Arc::clone(&out) });
    p.link((v, "src"), (mux, "sink_0")).expect("v -> mux.sink_0");
    p.link((a, "src"), (mux, "sink_1")).expect("a -> mux.sink_1");
    p.link((mux, "src"), (sink, "sink")).expect("mux -> sink");
    p.run().expect("run");

    let bytes = out.lock().unwrap().clone();
    let mut r = MatroskaReader::new();
    r.push(&bytes).expect("output parses as Matroska");

    // Tracks: pad 0 → track 1 (video), pad 1 → track 2 (audio); the video track leads
    // the Tracks element (it is the Cluster anchor).
    let tracks = r.tracks().to_vec();
    assert_eq!(tracks.len(), 2, "two TrackEntries");
    assert_eq!(tracks[0].codec_id, "V_MPEG4/ISO/AVC");
    assert_eq!(tracks[0].track_number, 1, "pad 0 → track 1");
    assert_eq!(tracks[0].codec_private, avcc_record(), "CodecPrivate = the record verbatim");
    assert_eq!((tracks[0].pixel_width, tracks[0].pixel_height), (320, 240));
    assert_eq!(tracks[1].codec_id, "A_AAC");
    assert_eq!(tracks[1].track_number, 2, "pad 1 → track 2");
    assert_eq!(tracks[1].codec_private, asc(), "CodecPrivate = the ASC verbatim");
    assert_eq!(tracks[1].sampling_frequency, 48_000.0);
    assert_eq!(tracks[1].channels, 2);

    // Info\Duration = max over the pads' announced durations.
    assert_eq!(r.duration_ns(), Some(dur_v.max(dur_a)), "duration = max across pads");

    // Frames: bit-exact per track in order, pts to the ms TimestampScale, keyframe
    // bits preserved, and the *global* block order strictly pts-ascending (all pts
    // distinct by construction — the merge order is deterministic).
    let (mut vi, mut ai) = (0usize, 0usize);
    let mut last_pts = 0u64;
    let mut blocks = 0usize;
    while let Some(f) = r.next_frame() {
        assert!(
            blocks == 0 || f.pts_ns > last_pts,
            "block {blocks}: pts {} not ascending after {last_pts} — interleave broken",
            f.pts_ns
        );
        last_pts = f.pts_ns;
        blocks += 1;
        match f.track_number {
            1 => {
                let (pts, key, ref bytes) = vframes[vi];
                assert_eq!(f.data, *bytes, "video frame {vi} bit-exact");
                assert_eq!(f.pts_ns, pts, "video frame {vi} pts");
                assert_eq!(f.keyframe, key, "video frame {vi} keyframe bit");
                vi += 1;
            }
            2 => {
                let (pts, _, ref bytes) = aframes[ai];
                assert_eq!(f.data, *bytes, "audio frame {ai} bit-exact");
                assert_eq!(f.pts_ns, pts, "audio frame {ai} pts");
                assert!(f.keyframe, "audio frame {ai} is a keyframe (untagged/KEYFRAME)");
                ai += 1;
            }
            n => panic!("unexpected track {n}"),
        }
    }
    assert_eq!((vi, ai), (vframes.len(), aframes.len()), "every frame arrived");
}

/// A single linked pad exercises the plain single-input head path (the scheduler feeds
/// `Inputs`, not the per-pad batches) — a 1-track `MkvMuxN` still muxes correctly.
#[test]
fn single_pad_aac_mux() {
    let aframes = audio_frames();
    let out = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let a = p.add(AnnSrc::new(
        "aac",
        vec![("rate", ValueDesc::Int(44_100)), ("channels", ValueDesc::Int(1))],
        asc(),
        aframes.clone(),
    ));
    let mux = p.add(MkvMuxN::new());
    let sink = p.add(CollectSink { bytes: Arc::clone(&out) });
    p.link((a, "src"), (mux, "sink_0")).expect("a -> mux.sink_0");
    p.link((mux, "src"), (sink, "sink")).expect("mux -> sink");
    p.run().expect("run");

    let bytes = out.lock().unwrap().clone();
    let mut r = MatroskaReader::new();
    r.push(&bytes).expect("output parses");
    assert_eq!(r.tracks().len(), 1);
    assert_eq!(r.tracks()[0].codec_id, "A_AAC");
    assert_eq!(r.tracks()[0].codec_private, asc());
    assert_eq!(r.tracks()[0].sampling_frequency, 44_100.0);
    let mut i = 0usize;
    while let Some(f) = r.next_frame() {
        assert_eq!(f.data, aframes[i].2, "frame {i} bit-exact");
        i += 1;
    }
    assert_eq!(i, aframes.len());
}

/// `MkvMux::multi(n)` validates the linked-pad count loudly at start.
#[test]
fn expected_count_mismatch_is_loud() {
    let out = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let a = p.add(AnnSrc::new(
        "aac",
        vec![("rate", ValueDesc::Int(48_000)), ("channels", ValueDesc::Int(2))],
        asc(),
        audio_frames(),
    ));
    let mux = p.add(MkvMux::multi(2)); // but only one pad gets linked
    let sink = p.add(CollectSink { bytes: Arc::clone(&out) });
    p.link((a, "src"), (mux, "sink_0")).expect("a -> mux.sink_0");
    p.link((mux, "src"), (sink, "sink")).expect("mux -> sink");
    let err = p.run().expect_err("mislinked count must fail");
    let msg = format!("{err:?}");
    assert!(msg.contains("2 track"), "names the expected count: {msg}");
}

// ---- the silent-pad failure mode ----------------------------------------------------
//
// A two-src-pad source (one element, so the whole feed lives in one scheduler group and
// terminates with the pipeline): pad `a` announces and floods tiny frames, pad `b` is
// linked but never announces and never sends. The muxer must not hang waiting for `b`'s
// config — the bounded pre-header queue turns into a loud error.

static FLOOD_PADS: [PadDesc; 2] = [
    PadDesc { name: "a", direction: Direction::Src, offers: &SRC_OFFERS, dynamic: false, validate: None },
    PadDesc { name: "b", direction: Direction::Src, offers: &SRC_OFFERS, dynamic: false, validate: None },
];
static FLOOD_DESC: ElementDesc = ElementDesc {
    name: "floodsrc",
    pads: &FLOOD_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct FloodSrc {
    sent: usize,
    total: usize,
    announced: bool,
    head_sent: bool,
}

impl Element for FloodSrc {
    fn desc(&self) -> &'static ElementDesc {
        &FLOOD_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.announced {
            ctx.announce_format(
                PadId(0),
                "h264/avcc",
                &[("width", ValueDesc::Int(64)), ("height", ValueDesc::Int(48))],
            );
            self.announced = true;
        }
        if !self.head_sent {
            if !AnnSrc::emit(ctx, &avcc_record(), None, BufferFlags::empty()) {
                return Ok(Flow::Ok);
            }
            self.head_sent = true;
            return Ok(Flow::Ok);
        }
        if self.sent >= self.total {
            return Ok(Flow::Eos);
        }
        let pts = self.sent as u64 * 1_000_000;
        if AnnSrc::emit(ctx, &[0x00], Some(pts), BufferFlags::DELTA) {
            self.sent += 1;
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// A linked pad that never announces while another streams: the pre-header queue bound
/// ([`mux_multi::PRECONFIG_MAX_BUFS`]) fails the pipeline loudly, naming the pad — a
/// diagnosable error instead of a silent stall.
#[test]
fn silent_linked_pad_is_a_loud_error() {
    let out = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(FloodSrc {
        sent: 0,
        total: mux_multi::PRECONFIG_MAX_BUFS + 64,
        announced: false,
        head_sent: false,
    });
    let mux = p.add(MkvMuxN::new());
    let sink = p.add(CollectSink { bytes: Arc::clone(&out) });
    p.link((src, "a"), (mux, "sink_0")).expect("a -> mux.sink_0");
    p.link((src, "b"), (mux, "sink_1")).expect("b -> mux.sink_1");
    p.link((mux, "src"), (sink, "sink")).expect("mux -> sink");
    // The muxer's queued frames pin the source's pool slots (the retention loop), so the
    // source pool must supply more buffers than the count bound for the *count* bound to
    // fire first — tiny slots, many of them. (In a real pipeline with MiB-sized slots
    // the byte bound fires first, below the pool's capacity: 64 MiB < e.g. 96 MiB.)
    p.set_element_pool(src, 256, (mux_multi::PRECONFIG_MAX_BUFS + 256) as u32);
    let err = p.run().expect_err("silent pad must fail loudly, not hang");
    let msg = format!("{err:?}");
    assert!(msg.contains("sink_1"), "the silent pad is named: {msg}");
    assert!(msg.contains("never announced"), "the cause is named: {msg}");
}

/// Data on a linked pad before any announcement is a loud error (a caps-driven muxer
/// cannot guess its codec) — via the single-input path.
#[test]
fn data_before_announcement_is_loud() {
    // An `AnnSrc` with an empty family announces nothing: emit head+frames raw.
    struct RawSrc {
        sent: bool,
    }
    impl Element for RawSrc {
        fn desc(&self) -> &'static ElementDesc {
            &SRC_DESC
        }
        fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
            Ok(())
        }
        fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
            if self.sent {
                return Ok(Flow::Eos);
            }
            if AnnSrc::emit(ctx, &[1, 2, 3], Some(0), BufferFlags::KEYFRAME) {
                self.sent = true;
            }
            Ok(Flow::Ok)
        }
        fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
            Ok(())
        }
        fn stop(&mut self, _ctx: &mut Ctx) {}
    }

    let out = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(RawSrc { sent: false });
    let mux = p.add(MkvMuxN::new());
    let sink = p.add(CollectSink { bytes: Arc::clone(&out) });
    p.link((src, "src"), (mux, "sink_0")).expect("src -> mux");
    p.link((mux, "src"), (sink, "sink")).expect("mux -> sink");
    let err = p.run().expect_err("unannounced data must fail");
    assert!(
        format!("{err:?}").contains("before any format announcement"),
        "the cause is named: {err:?}"
    );
}
