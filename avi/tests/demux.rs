//! End-to-end `AviDemux` tests in a real [`Pipeline`], plus a direct-parser cross-check.
//!
//! The load-bearing properties:
//! - **Discovery**: `FileSrc(tiny.avi) ! avidemux` prerolls exactly two src pads — one video
//!   (`mpeg4/asp`, 160×120) and one audio (`ac3`, 44100 Hz) — matching what `ffprobe` reports
//!   for the fixture.
//! - **Demux completeness**: each pad's capturing sink receives exactly the stream's chunk
//!   count and byte total the direct parser (and ffprobe's `nb_frames`) says — 75 video
//!   frames, 87 audio frames, no bytes lost.
//! - **Robustness**: a truncated / corrupt AVI does not panic the parser (this crate's P0).
//!
//! The fixture `tests/fixtures/tiny.avi` was produced once by
//! `ffmpeg -f lavfi -i testsrc2=s=160x120:d=3 -f lavfi -i sine -t 3 -c:v mpeg4 -vtag XVID
//!  -c:a ac3 tiny.avi` (185 KB), so these tests need no ffmpeg and no 2 GB file at test time.

use std::sync::{Arc, Mutex};

use sc_avi::riff::{self, MoviWalker, StreamKind};
use sc_avi::AviDemux;
use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;
use streamcraft_elements::io::FileSrc;

const TINY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny.avi");

/// Read the fixture's header prefix (app-side test setup — the sanctioned reactor exception).
#[allow(clippy::disallowed_methods)] // test setup reads the fixture directly
fn header_prefix(path: &str) -> Vec<u8> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).expect("open fixture");
    let mut buf = vec![0u8; 1024 * 1024];
    let n = f.read(&mut buf).expect("read header");
    buf.truncate(n);
    buf
}

// =====================================================================================
// A sink that records, per received buffer, its byte length + a running total + PTS list.
// =====================================================================================

static SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("bytes"), OfferDesc::any("mpeg4/asp")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "capturesink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active, // its own group, so buffers cross a real ring
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

#[derive(Default)]
struct Capture {
    /// One entry per received buffer: (byte length, pts_ns or -1).
    buffers: Vec<(usize, i64)>,
    total_bytes: u64,
}

struct CaptureSink {
    cap: Arc<Mutex<Capture>>,
}

impl Element for CaptureSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut cap = self.cap.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            let len = buf.memory.data().len();
            cap.total_bytes += len as u64;
            cap.buffers.push((len, buf.pts.nanos().map(|n| n as i64).unwrap_or(-1)));
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// Run `filesrc(tiny.avi) ! avidemux ! capturesink(per pad)` and return the per-pad captures
/// in discovery order.
fn run_demux() -> Vec<Arc<Mutex<Capture>>> {
    let mut p = Pipeline::new();
    // Video frames here are small (≤ a few KB), so the default slot easily holds each chunk
    // as one buffer — the per-chunk boundary the counts assert.
    p.set_pool(1024 * 1024, 32);
    let header = header_prefix(TINY);
    let src = p.add(FileSrc::new(TINY));
    let demux = p.add(AviDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src ! demux");

    let added = p.preroll().expect("preroll");
    let mut caps = Vec::new();
    for ap in &added {
        let cap = Arc::new(Mutex::new(Capture::default()));
        let sink = p.add(CaptureSink { cap: Arc::clone(&cap) });
        p.link((ap.element, &ap.name), (sink, "sink")).expect("demux ! sink");
        caps.push(cap);
    }
    p.run().expect("run");
    caps
}

#[test]
fn preroll_discovers_video_and_audio_pads() {
    let mut p = Pipeline::new();
    let header = header_prefix(TINY);
    let src = p.add(FileSrc::new(TINY));
    let demux = p.add(AviDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src ! demux");
    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 2, "two src pads (video + audio)");
    assert_eq!(added[0].name, "src_stream0");
    assert_eq!(added[1].name, "src_stream1");
}

#[test]
fn demux_emits_every_chunk_with_correct_counts() {
    // Ground truth from the direct parser (== ffprobe's nb_frames: 75 video, 87 audio).
    let (v_chunks, v_bytes, a_chunks, a_bytes) = parser_counts();
    assert_eq!(v_chunks, 75, "ffprobe reports 75 video frames");
    assert_eq!(a_chunks, 87, "ffprobe reports 87 audio frames");

    let caps = run_demux();
    assert_eq!(caps.len(), 2);
    let v = caps[0].lock().unwrap();
    let a = caps[1].lock().unwrap();
    assert_eq!(v.buffers.len(), v_chunks, "one output buffer per video chunk");
    assert_eq!(v.total_bytes, v_bytes, "no video bytes lost");
    assert_eq!(a.buffers.len(), a_chunks, "one output buffer per audio chunk");
    assert_eq!(a.total_bytes, a_bytes, "no audio bytes lost");
}

#[test]
fn video_pts_advances_monotonically_at_frame_rate() {
    let caps = run_demux();
    let v = caps[0].lock().unwrap();
    // 25 fps → 40 ms/frame. The first frame is PTS 0, and PTS is strictly increasing.
    assert_eq!(v.buffers[0].1, 0, "first video frame at PTS 0");
    let mut prev = -1i64;
    for (i, &(_, pts)) in v.buffers.iter().enumerate() {
        assert!(pts >= 0, "video frame {i} carries a PTS");
        assert!(pts > prev, "video PTS strictly increasing at frame {i}");
        prev = pts;
    }
    // Frame 1 is ~40 ms after frame 0 (25 fps). Allow the integer-division rounding.
    let dt = v.buffers[1].1 - v.buffers[0].1;
    assert!((39_000_000..=41_000_000).contains(&dt), "≈40 ms between video frames, got {dt} ns");
}

#[test]
fn audio_pts_is_monotonic() {
    let caps = run_demux();
    let a = caps[1].lock().unwrap();
    let mut prev = -1i64;
    for (i, &(_, pts)) in a.buffers.iter().enumerate() {
        assert!(pts >= 0, "audio frame {i} carries a PTS");
        assert!(pts > prev || i == 0, "audio PTS non-decreasing at frame {i}");
        prev = pts;
    }
}

#[test]
fn seek_index_from_idx1_has_video_keyframes() {
    // The idx1 → SeekIndex helper resolves the fixture's video keyframes to byte offsets.
    let header = header_prefix(TINY);
    let file_len = std::fs::metadata(TINY).unwrap().len();
    let (idx1_off, idx1_size) = find_idx1(TINY, file_len).expect("tiny.avi has an idx1");
    let payload = read_at(TINY, idx1_off + 8, idx1_size as usize);
    let si = sc_avi::build_seek_index(&header, &payload, file_len).expect("build seek index");
    assert!(!si.entries.is_empty(), "at least one video keyframe indexed");
    assert_eq!(si.entries[0].0, 0, "first keyframe at time 0");
    assert_eq!(si.file_len, Some(file_len));
    // Entries ascend in time and point inside the file.
    let mut prev_t = 0u64;
    for &(t, b) in &si.entries {
        assert!(t >= prev_t, "keyframe times ascend");
        assert!(b < file_len, "keyframe byte offset inside the file");
        prev_t = t;
    }
    // The resolved byte for the first keyframe lands at the movi data start (frame 0).
    let (_h, movi) = riff::probe_header(&header).unwrap();
    assert_eq!(si.entries[0].1, movi.unwrap().movi_data_start, "frame 0 → movi data start");
}

#[test]
fn truncated_avi_does_not_panic() {
    // Feed progressively longer prefixes of the fixture to the parser — each must either
    // parse cleanly or error, never panic (this crate's untrusted-input P0).
    #[allow(clippy::disallowed_methods)] // test setup reads the fixture directly
    let full = std::fs::read(TINY).unwrap();
    for cut in [0usize, 4, 12, 64, 256, 4096, full.len() / 2, full.len() - 1] {
        let _ = riff::probe_header(&full[..cut.min(full.len())]);
    }
    // And a corrupt movi walk from a mid-file offset must terminate without panicking.
    let mut w = MoviWalker::new();
    w.push(&full[full.len() / 3..]);
    let mut n = 0;
    while w.next_chunk().is_some() && n < 1_000_000 {
        n += 1;
    }
}

// ---- helpers: direct-parser ground truth + idx1 location (test setup, sanctioned IO) ----

/// Walk the fixture's movi with the same [`MoviWalker`] the element uses; return
/// (video_chunks, video_bytes, audio_chunks, audio_bytes).
fn parser_counts() -> (usize, u64, usize, u64) {
    #[allow(clippy::disallowed_methods)] // test setup reads the fixture directly
    let full = std::fs::read(TINY).unwrap();
    let (avi, movi) = riff::probe_header(&full).expect("probe");
    let movi = movi.unwrap();
    let video_idx = avi.streams.iter().find(|s| matches!(s.kind, StreamKind::Video)).unwrap().index;
    let audio_idx = avi.streams.iter().find(|s| matches!(s.kind, StreamKind::Audio)).unwrap().index;
    let mut w = MoviWalker::new();
    w.push(&full[movi.movi_data_start as usize..]);
    let (mut vc, mut vb, mut ac, mut ab) = (0usize, 0u64, 0usize, 0u64);
    while let Some(c) = w.next_chunk() {
        if c.stream_index == video_idx {
            vc += 1;
            vb += c.data.len() as u64;
        } else if c.stream_index == audio_idx {
            ac += 1;
            ab += c.data.len() as u64;
        }
    }
    (vc, vb, ac, ab)
}

#[allow(clippy::disallowed_methods)] // test setup reads the fixture directly
fn read_at(path: &str, off: u64, len: usize) -> Vec<u8> {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(path).unwrap();
    let mut buf = vec![0u8; len];
    let n = f.read_at(&mut buf, off).unwrap();
    buf.truncate(n);
    buf
}

#[allow(clippy::disallowed_methods)] // test setup reads the fixture directly
fn find_idx1(path: &str, file_len: u64) -> Option<(u64, u32)> {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(path).ok()?;
    let window = (4 * 1024 * 1024).min(file_len) as usize;
    let start = file_len - window as u64;
    let mut buf = vec![0u8; window];
    let n = f.read_at(&mut buf, start).ok()?;
    buf.truncate(n);
    let mut found = None;
    let mut i = 0;
    while i + 8 <= buf.len() {
        if &buf[i..i + 4] == b"idx1" {
            let size = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]);
            let abs = start + i as u64;
            if abs + 8 + size as u64 <= file_len {
                found = Some((abs, size));
            }
        }
        i += 1;
    }
    found
}
