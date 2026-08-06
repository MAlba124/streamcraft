//! profluens-video integration tests (spec: Milestone applications §5 — the
//! no-decoder video path; Testing — one integration binary per crate). One binary to
//! keep link time down:
//!
//! 1. **geometry tables** — the plane math end to end, including the odd-dimension
//!    4:2:0 convention;
//! 2. **`VideoTestSrc ! VideoCkSink` under a `MockClock`** — frames render on their fps
//!    deadlines, hours of virtual time in milliseconds of wall time, and the
//!    seed-deterministic pixels round-trip via the checksum;
//! 3. **`FileSrc ! RawVideoParse ! VideoCkSink`** over a temp raw file written with
//!    `std::fs` — the parser chunks the byte stream into frames across buffer
//!    boundaries, stamped on the fps grid;
//! 4. **a negotiation failure** — a sink constrained to a pixfmt the source does not
//!    offer fails loudly at `link()`.

use std::sync::Arc;
use std::time::{Duration, Instant as StdInstant};

use profluens_core::batch::Inputs;
use profluens_core::clock::MockClock;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::pipeline::Pipeline;
use profluens_core::time::{Rational, Timestamp};

use profluens_elements::io::FileSrc;
use profluens_video::format::PixelFormat;
use profluens_video::geometry::{frame_size, plane_count, plane_geometry, plane_offset};
use profluens_video::sink::checksum;
use profluens_video::testsrc::{frame_pattern_byte, frame_pts};
use profluens_video::{RawVideoParse, VideoCkSink, VideoFormat, VideoTestSrc};

// --- 1. Geometry tables -------------------------------------------------------------

#[test]
fn geometry_planes_pack_tightly_and_size_is_the_sum() {
    // (pixfmt, w, h, planes[(stride, height)]) — even and odd dims (odd exercises the
    // 4:2:0 chroma round-UP convention).
    let cases: &[(PixelFormat, u32, u32, &[(usize, usize)])] = &[
        (PixelFormat::I420, 4, 4, &[(4, 4), (2, 2), (2, 2)]),
        (PixelFormat::I420, 3, 3, &[(3, 3), (2, 2), (2, 2)]), // chroma ceil(3/2)=2
        (PixelFormat::Nv12, 4, 4, &[(4, 4), (4, 2)]),
        (PixelFormat::Nv12, 3, 5, &[(3, 5), (4, 3)]), // chroma stride 2*ceil(3/2)=4
        (PixelFormat::Rgb24, 5, 3, &[(15, 3)]),
        (PixelFormat::Gray8, 5, 3, &[(5, 3)]),
    ];
    for &(pixfmt, w, h, planes) in cases {
        assert_eq!(plane_count(pixfmt), planes.len(), "{pixfmt:?} {w}x{h}");
        let mut off = 0usize;
        let mut total = 0usize;
        for (i, &(stride, height)) in planes.iter().enumerate() {
            let g = plane_geometry(pixfmt, w, h, i).unwrap();
            assert_eq!((g.stride, g.height, g.size), (stride, height, stride * height));
            assert_eq!(plane_offset(pixfmt, w, h, i), Some(off));
            off += g.size;
            total += g.size;
        }
        assert_eq!(frame_size(pixfmt, w, h), total, "{pixfmt:?} {w}x{h} total");
    }
}

// --- 2. VideoTestSrc ! VideoCkSink under a MockClock --------------------------------

#[test]
fn videotestsrc_renders_on_fps_deadlines_and_pixels_round_trip() {
    // 16x16 I420 at 30 fps. Enough frames to be ~an hour of virtual time (30 fps × 3600
    // s = 108000), so the run compresses hours to milliseconds and never sleeps.
    let fps = Rational::new(30, 1);
    let format = VideoFormat::new(16, 16, PixelFormat::I420, fps);
    let seed = 0xABCD_1234u64;
    let count = 108_000u64; // one virtual hour at 30 fps

    let clock = MockClock::new();
    let (sink, stats) = VideoCkSink::new();
    let mut p = Pipeline::new();
    let src = p.add(VideoTestSrc::new(format, seed, count));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("video/raw negotiates");
    p.set_clock(Arc::new(clock.clone()));

    let start = StdInstant::now();
    let run = std::thread::spawn(move || p.run());

    // Drive virtual time forward in coarse jumps until the run completes.
    let step = Timestamp::from_secs(60); // one virtual minute per jump
    while !run.is_finished() {
        clock.advance(step);
        std::thread::yield_now();
    }
    run.join().expect("run joined").expect("run ok");
    let wall = start.elapsed();

    assert!(stats.is_done(), "sink saw EOS");
    let renders = stats.renders();
    assert_eq!(renders.len() as u64, count, "every frame rendered exactly once");

    // Spot-check the schedule + pixels rather than regenerate all 108k frames.
    let frame_bytes = frame_size(format.pixfmt, format.width, format.height);
    for &i in &[0u64, 1, 2, 100, 29, 30, count - 1] {
        let r = renders[i as usize];
        assert_eq!(r.pts, frame_pts(fps, i), "frame {i} PTS on the fps grid");
        assert!(r.rendered_at >= r.pts, "frame {i} not rendered ahead of its deadline");
        // Recompute the expected frame content and its checksum.
        let expect: Vec<u8> = (0..frame_bytes)
            .map(|pos| frame_pattern_byte(seed, i, pos))
            .collect();
        assert_eq!(r.checksum, checksum(&expect), "frame {i} pixels round-trip");
    }
    // The whole virtual hour really did run in a blink of wall time.
    assert!(
        wall < Duration::from_secs(20),
        "one virtual hour took {wall:?} of wall time — should be well under a second"
    );
}

#[test]
fn videocksink_parks_until_the_clock_reaches_a_deadline() {
    // Two frames, the second one virtual-hour away: the sink must park in wait_until
    // until the clock crosses it, so the run cannot finish early.
    let fps = Rational::new(1, 3600); // one frame per hour
    let format = VideoFormat::new(2, 2, PixelFormat::Gray8, fps);
    let clock = MockClock::new();
    let (sink, stats) = VideoCkSink::new();
    let mut p = Pipeline::new();
    let src = p.add(VideoTestSrc::new(format, 1, 2));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.set_clock(Arc::new(clock.clone()));

    let run = std::thread::spawn(move || p.run());
    for _ in 0..60 {
        if run.is_finished() {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(!run.is_finished(), "run finished before the 2nd frame's deadline");
    assert_eq!(stats.count(), 1, "only frame 0 (due at t=0) rendered while parked");

    clock.advance(Timestamp::from_secs(3600));
    run.join().expect("joined").expect("run ok");
    let renders = stats.renders();
    assert_eq!(renders.len(), 2);
    assert_eq!(renders[1].pts, Timestamp::from_secs(3600));
    assert!(renders[1].rendered_at >= renders[1].pts);
}

// --- 3. FileSrc ! RawVideoParse ! VideoCkSink over a generated temp raw file ---------

#[test]
fn filesrc_rawvideoparse_chunks_a_raw_file_into_frames() {
    let fps = Rational::new(25, 1);
    let format = VideoFormat::new(16, 16, PixelFormat::I420, fps);
    let frame_bytes = frame_size(format.pixfmt, format.width, format.height);
    let frames = 40u64;

    // Write a raw file: `frames` concatenated frames, each a distinct byte pattern so
    // the checksum catches any mis-chunking across buffer boundaries.
    let mut raw = Vec::with_capacity(frames as usize * frame_bytes);
    for f in 0..frames {
        for pos in 0..frame_bytes {
            raw.push(frame_pattern_byte(0xF00D, f, pos));
        }
    }
    let dir = std::env::temp_dir();
    let path = dir.join(format!("pf_video_test_{}.raw", std::process::id()));
    std::fs::write(&path, &raw).expect("write temp raw file");

    let clock = MockClock::new();
    let (sink, stats) = VideoCkSink::new();
    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&path));
    let parse = p.add(RawVideoParse::new(format));
    let snk = p.add(sink);
    p.link((src, "src"), (parse, "sink")).expect("filesrc -> rawvideoparse (bytes)");
    p.link((parse, "src"), (snk, "sink")).expect("rawvideoparse -> videocksink");
    p.set_clock(Arc::new(clock.clone()));

    let run = std::thread::spawn(move || p.run());
    let step = Timestamp::from_millis(200);
    while !run.is_finished() {
        clock.advance(step);
        std::thread::yield_now();
    }
    let run_result = run.join().expect("joined");
    std::fs::remove_file(&path).ok();
    run_result.expect("run ok");

    assert!(stats.is_done(), "sink saw EOS");
    let renders = stats.renders();
    assert_eq!(renders.len() as u64, frames, "one buffer per whole frame");
    for (f, r) in renders.iter().enumerate() {
        assert_eq!(r.pts, frame_pts(fps, f as u64), "frame {f} PTS on the fps grid");
        let expect: Vec<u8> = (0..frame_bytes)
            .map(|pos| frame_pattern_byte(0xF00D, f as u64, pos))
            .collect();
        assert_eq!(r.checksum, checksum(&expect), "frame {f} chunked intact");
    }
}

// --- 4. Negotiation failure: a sink pins a pixfmt the source does not offer ----------
//
// VideoTestSrc offers the broad video/raw menu (every pixfmt), so to force a *link-time*
// failure we use a narrow source that offers only i420 against a sink that demands gray8
// — an empty intersection the solver rejects at link(), exactly the negotiation.rs shape.

static NARROW_PIXFMT: [ValueDesc; 1] = [ValueDesc::Id("i420")];
static NARROW_FIELDS: [FieldDesc; 1] = [FieldDesc {
    field: "pixfmt",
    allowed: ConstraintDesc::Set(&NARROW_PIXFMT),
    preferred: None,
}];
static NARROW_OFFERS: [OfferDesc; 1] = [OfferDesc { family: "video/raw", fields: &NARROW_FIELDS }];
static NARROW_SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &NARROW_OFFERS,
    dynamic: false,
    validate: None,
}];
static NARROW_SRC_DESC: ElementDesc = ElementDesc {
    name: "i420only_src",
    pads: &NARROW_SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: ZERO_LAT,
    make_default: None,
};

const ZERO_LAT: LatencyDesc = LatencyDesc {
    min: Timestamp::ZERO,
    max: Timestamp::ZERO,
    is_live: false,
    jitter: Timestamp::ZERO,
};

#[derive(Default)]
struct I420OnlySrc;
impl Element for I420OnlySrc {
    fn desc(&self) -> &'static ElementDesc {
        &NARROW_SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        Ok(Flow::Eos)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static GRAY_PIXFMT: [ValueDesc; 1] = [ValueDesc::Id("gray8")];
static GRAY_FIELDS: [FieldDesc; 1] = [FieldDesc {
    field: "pixfmt",
    allowed: ConstraintDesc::Set(&GRAY_PIXFMT),
    preferred: None,
}];
static GRAY_OFFERS: [OfferDesc; 1] = [OfferDesc { family: "video/raw", fields: &GRAY_FIELDS }];
static GRAY_SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &GRAY_OFFERS,
    dynamic: false,
    validate: None,
}];
static GRAY_SINK_DESC: ElementDesc = ElementDesc {
    name: "gray8only_sink",
    pads: &GRAY_SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: ZERO_LAT,
    make_default: None,
};

#[derive(Default)]
struct Gray8OnlySink;
impl Element for Gray8OnlySink {
    fn desc(&self) -> &'static ElementDesc {
        &GRAY_SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[test]
fn incompatible_pixfmt_fails_negotiation_at_link() {
    let mut p = Pipeline::new();
    let src = p.add(I420OnlySrc);
    let snk = p.add(Gray8OnlySink);
    // Same family (video/raw), but pixfmt ∈ {i420} vs {gray8} — empty intersection.
    let err = p.link((src, "src"), (snk, "sink")).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("no common format") || msg.contains("negotiation"),
        "error should explain the empty pixfmt intersection, got: {msg}"
    );
}

// --- 5. Parse-path props override the constructor defaults (spec: Plugins) -----------
//
// The registry builds `VideoTestSrc` with default construction parameters and refines
// them from props in `start()`. This exercises the same refinement path through
// `Pipeline::set` / `set_str`: construct with placeholder dims/seed/frames, override every
// knob via props, then prove the sink rendered the *overridden* format — the checksums
// only match the overridden `(seed, dims)`, and the count only matches `frames`.

#[test]
fn videotestsrc_props_override_constructor_defaults() {
    use profluens_core::format::Value;

    // Placeholder construction values — every one is overridden below.
    let ctor = VideoFormat::new(320, 240, PixelFormat::Rgb24, Rational::new(25, 1));
    // The overrides the props install.
    let want = VideoFormat::new(8, 8, PixelFormat::Gray8, Rational::new(30, 1));
    let want_seed = 0xC0FF_EEu64;
    let want_frames = 7u64;

    let clock = MockClock::new();
    let (sink, stats) = VideoCkSink::new();
    let mut p = Pipeline::new();
    let src = p.add(VideoTestSrc::new(ctor, 1, 1)); // ctor seed=1, frames=1
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");

    // Override each construction parameter through the property mailbox.
    p.set(src, "width", Value::Int(want.width as i64)).expect("width");
    p.set(src, "height", Value::Int(want.height as i64)).expect("height");
    p.set_str(src, "pixfmt", want.pixfmt.caps_name()).expect("pixfmt");
    p.set(src, "fps", Value::Rat(want.fps.num, want.fps.den)).expect("fps");
    p.set(src, "frames", Value::Int(want_frames as i64)).expect("frames");
    p.set(src, "seed", Value::Int(want_seed as i64)).expect("seed");

    p.set_clock(Arc::new(clock.clone()));
    let run = std::thread::spawn(move || p.run());
    let step = Timestamp::from_millis(200);
    while !run.is_finished() {
        clock.advance(step);
        std::thread::yield_now();
    }
    run.join().expect("joined").expect("run ok");

    let renders = stats.renders();
    // `frames` override took effect: exactly `want_frames` frames, not the ctor's 1.
    assert_eq!(renders.len() as u64, want_frames, "frames prop overrode the constructor");

    // The overridden `(seed, dims, fps)` reproduce every rendered frame exactly.
    let frame_bytes = frame_size(want.pixfmt, want.width, want.height);
    for (i, r) in renders.iter().enumerate() {
        assert_eq!(r.pts, frame_pts(want.fps, i as u64), "frame {i} on the overridden fps grid");
        let expect: Vec<u8> = (0..frame_bytes)
            .map(|pos| frame_pattern_byte(want_seed, i as u64, pos))
            .collect();
        assert_eq!(r.checksum, checksum(&expect), "frame {i} matches the overridden format");
    }
}
