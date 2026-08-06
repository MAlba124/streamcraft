//! AUDIT reproducer: a backward media-timestamp step inside an open segment.
//!
//! A camera whose media clock is stepped backwards (an NTP correction on the
//! camera, an RTP timestamp discontinuity kept under the same SSRC, or an
//! `rtpsession` re-anchor after a restart) delivers an access unit whose pts is
//! *below* the open segment's `base_pts`. `MkvSegmentSink::handle_au` computes
//! the block timestamp as `pts - seg.base_pts` on `u64`, so the step wraps.

use std::path::{Path, PathBuf};
use std::process::Command;

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
use profluens_nvr::MkvSegmentSink;

fn tool(name: &str) -> Option<PathBuf> {
    let out = Command::new("which").arg(name).output().ok()?;
    out.status.success().then(|| {
        PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_string())
    })
}

fn generate_fixture(ffmpeg: &Path, path: &Path) {
    let st = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25"])
        .args(["-t", "12", "-c:v", "libx264", "-preset", "ultrafast"])
        .args(["-g", "25", "-bf", "0", "-pix_fmt", "yuv420p"])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(st.success());
}

#[allow(clippy::disallowed_methods)]
fn read_all(path: &Path) -> Vec<u8> {
    std::fs::read(path).expect("read fixture")
}

/// Annex-B AUs pulled out of the fixture, with the parameter sets prefixed to
/// every keyframe (the camera convention `camsrc` also follows).
fn fixture_aus(path: &Path) -> Vec<(Vec<u8>, bool)> {
    use pf_mkv::codec::{nal_head_from_config, Reframer};
    use pf_mkv::MatroskaReader;

    let bytes = read_all(path);
    let mut r = MatroskaReader::new();
    r.push(&bytes).expect("parse fixture");
    let track = r
        .tracks()
        .iter()
        .find(|t| t.codec_id == "V_MPEG4/ISO/AVC")
        .cloned()
        .expect("h264 track");
    let (head, length_size) =
        nal_head_from_config(&track.codec_private, false).expect("avcC parses");
    let reframer = Reframer::Nal { length_size };
    let mut scratch = Vec::new();
    let mut out = Vec::new();
    while let Some(f) = r.next_frame() {
        if f.track_number != track.track_number {
            continue;
        }
        let body = reframer.reframe_into(&f.data, &mut scratch).expect("reframe");
        let mut au = Vec::with_capacity(head.len() + body.len());
        if f.keyframe {
            au.extend_from_slice(&head);
        }
        au.extend_from_slice(body);
        out.push((au, f.keyframe));
    }
    out
}

// --- a source that replays AUs with app-chosen timestamps ---------------------

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("h264/annexb")];
static PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];
static DESC: ElementDesc = ElementDesc {
    name: "aureplay",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

struct AuReplay {
    aus: Vec<(Vec<u8>, u64)>,
    at: usize,
}

impl Element for AuReplay {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        while self.at < self.aus.len() {
            let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
            let (au, pts) = &self.aus[self.at];
            buf.memory.as_mut_full()[..au.len()].copy_from_slice(au);
            buf.memory.set_len(au.len());
            buf.pts = Timestamp::from_nanos(*pts);
            ctx.out(PadId(0)).push(buf);
            self.at += 1;
        }
        Ok(Flow::Eos)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// A camera stepping its media clock back 5 s, 3 s into a 30 s segment.
#[test]
fn backward_timestamp_step_inside_a_segment() {
    let (Some(ffmpeg), Some(ffprobe)) = (tool("ffmpeg"), tool("ffprobe")) else {
        eprintln!("ffmpeg/ffprobe absent — skipping");
        return;
    };
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("timestep");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let fixture = tmp.join("src.mkv");
    generate_fixture(&ffmpeg, &fixture);

    // 25 fps, running time already at 100 s (a recorder that has been up a
    // while); step the clock back 5 s at frame 80 — mid-GOP, so the recovery
    // path has to idle to the next IDR rather than reopening immediately.
    const T0: u64 = 100_000_000_000;
    const STEP_AT: usize = 80;
    const STEP_BACK_NS: u64 = 5_000_000_000;
    let aus: Vec<(Vec<u8>, u64)> = fixture_aus(&fixture)
        .into_iter()
        .enumerate()
        .map(|(i, (au, _kf))| {
            let t = T0 + i as u64 * 40_000_000;
            (au, if i >= STEP_AT { t - STEP_BACK_NS } else { t })
        })
        .collect();

    let rec = tmp.join("rec");
    let mut p = Pipeline::new();
    p.set_pool(1024 * 1024, 16);
    let src = p.add(AuReplay { aus, at: 0 });
    // 30 s segments: the whole 12 s clip is one segment, so the step lands
    // inside it rather than at a rotation.
    let sink = p.add(MkvSegmentSink::new(&rec, "cam0", 30.0, &[]));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("pipeline run");

    let mut files: Vec<PathBuf> =
        std::fs::read_dir(&rec).unwrap().map(|e| e.unwrap().path()).collect();
    files.sort();
    // The discontinuity cuts the segment: one file per clock epoch.
    assert_eq!(files.len(), 2, "the clock step should cut the segment, got {files:?}");

    let mut total_frames = 0usize;
    for f in &files {
        let dur = Command::new(&ffprobe)
            .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1"])
            .arg(f)
            .output()
            .expect("ffprobe");
        let pkts = Command::new(&ffprobe)
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time"])
            .args(["-of", "csv=p=0"])
            .arg(f)
            .output()
            .expect("ffprobe packets");
        let times: Vec<f64> = String::from_utf8_lossy(&pkts.stdout)
            .lines()
            .filter_map(|l| l.trim().parse::<f64>().ok())
            .collect();
        let max = times.iter().cloned().fold(f64::MIN, f64::max);
        eprintln!(
            "{}: {} packets, max pts {max:.3}s, {}",
            f.file_name().unwrap().to_string_lossy(),
            times.len(),
            String::from_utf8_lossy(&dur.stdout).trim(),
        );
        total_frames += times.len();
        assert!(
            max < 60.0,
            "{}: packet at {max:.3}s in a 12 s clip — the `pts - base_pts` \
             subtraction wrapped on the backward clock step",
            f.display()
        );
        // Monotone within the file: a backward step inside a segment breaks
        // every seek, whether or not it also wraps.
        assert!(
            times.windows(2).all(|w| w[1] >= w[0]),
            "{}: non-monotone packet timestamps",
            f.display()
        );
    }
    // At most one GOP (25 frames) is dropped re-acquiring an entry point.
    assert!(total_frames >= 275, "lost more than a GOP at the cut: {total_frames}/300");
}
