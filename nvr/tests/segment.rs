//! Offline gate for [`MkvSegmentSink`]: a real x264 MKV (ffmpeg-generated,
//! skipped gracefully without ffmpeg — the sanctioned oracle policy) plays
//! through `filesrc ! mkvdemux ! mkvsegmentsink` and must land as rotated,
//! self-contained segments that (a) our own reader parses with the patched
//! Duration, (b) `ffprobe` identifies with the right codec and length, and
//! (c) headless `mpv` decodes fully *and* from the 95% tail — ffprobe and mpv
//! disagree often enough that both are required (the unknown-size-Cluster
//! lesson).

use std::path::{Path, PathBuf};
use std::process::Command;

use profluens_core::pipeline::Pipeline;
use profluens_elements::io::FileSrc;
use profluens_elements::testing::TestSink;
use profluens_nvr::MkvSegmentSink;

fn tool(name: &str) -> Option<PathBuf> {
    let out = Command::new("which").arg(name).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_string()))
}

/// 25 s of x264 (1 s GOP, no B-frames — the IP-camera shape) via ffmpeg.
fn generate_fixture(ffmpeg: &Path, path: &Path) {
    let st = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25"])
        .args(["-t", "25", "-c:v", "libx264", "-preset", "ultrafast"])
        .args(["-g", "25", "-bf", "0", "-pix_fmt", "yuv420p"])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(st.success(), "ffmpeg fixture generation failed");
}

/// Everything before the first Cluster (the `play_file` probe).
// Test-side probe on the test thread, not element IO (clippy.toml).
#[allow(clippy::disallowed_methods)]
fn header_prefix(path: &Path) -> Vec<u8> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).expect("open fixture");
    let mut head = vec![0u8; 8 * 1024 * 1024];
    let n = f.read(&mut head).expect("read head");
    head.truncate(n);
    let cluster = head
        .windows(4)
        .position(|w| w == pf_mkv::ebml::id::CLUSTER)
        .expect("no Cluster in fixture head");
    head.truncate(cluster);
    head
}

fn ffprobe_duration_and_codec(ffprobe: &Path, file: &Path) -> (f64, String) {
    let out = Command::new(ffprobe)
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .arg("stream=codec_name:format=duration")
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(file)
        .output()
        .expect("run ffprobe");
    assert!(out.status.success(), "ffprobe failed on {}", file.display());
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let field = |k: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(k))
            .unwrap_or_else(|| panic!("ffprobe output missing {k} for {}:\n{text}", file.display()))
            .to_string()
    };
    (field("duration=").parse().expect("duration parses"), field("codec_name="))
}

/// Headless decode gate; `start` like `"95%"` exercises the file tail.
fn mpv_decodes(mpv: &Path, file: &Path, start: Option<&str>) {
    let mut cmd = Command::new(mpv);
    cmd.args(["--no-config", "--really-quiet", "--vo=null", "--ao=null", "--untimed"]);
    if let Some(s) = start {
        cmd.arg(format!("--start={s}"));
    }
    let st = cmd.arg(file).status().expect("run mpv");
    assert!(
        st.success(),
        "mpv failed on {} (start={start:?})",
        file.display()
    );
}

#[test]
fn segments_rotate_and_validate_externally() {
    let (Some(ffmpeg), Some(ffprobe), Some(mpv)) = (tool("ffmpeg"), tool("ffprobe"), tool("mpv"))
    else {
        eprintln!("ffmpeg/ffprobe/mpv not all present — skipping the oracle gate");
        return;
    };

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("segment_sink");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let fixture = tmp.join("source.mkv");
    generate_fixture(&ffmpeg, &fixture);
    let header = header_prefix(&fixture);

    let rec_dir = tmp.join("rec");
    let mut p = Pipeline::new();
    // One AU per buffer is the sink's contract; 4 MiB slots dwarf any 640x360 AU.
    p.set_pool(4 * 1024 * 1024, 24);
    let src = p.add(FileSrc::new(fixture.to_str().unwrap()));
    let demux = p.add(pf_mkv::MkvDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src ! demux");
    let added = p.preroll().expect("preroll");

    let sink = p.add(MkvSegmentSink::new(&rec_dir, "cam0", 5.0, &[]));
    let mut linked = false;
    for ap in &added {
        if !linked && p.link((ap.element, &ap.name), (sink, "sink")).is_ok() {
            linked = true;
            continue;
        }
        let (drop_sink, _stats) = TestSink::new();
        let d = p.add(drop_sink);
        p.link((ap.element, &ap.name), (d, "sink")).expect("drop link");
    }
    assert!(linked, "no demux pad linked to mkvsegmentsink");
    p.run().expect("pipeline run");

    let mut segments: Vec<PathBuf> = std::fs::read_dir(&rec_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    segments.sort();
    assert!(
        (4..=6).contains(&segments.len()),
        "25 s at 5 s/segment should land 5±1 files, got {}: {segments:?}",
        segments.len()
    );

    let mut total = 0.0f64;
    for (i, seg) in segments.iter().enumerate() {
        // (a) our reader: parses, and the patched Duration is real (not 0).
        let head = header_prefix(seg);
        let mut r = pf_mkv::MatroskaReader::new();
        r.push(&head).expect("segment header parses");
        let dur_ns = r.duration_ns().expect("segment has a Duration");
        assert!(dur_ns > 0, "Duration patch applied on {}", seg.display());

        // (b) ffprobe agrees on codec and rough length.
        let (dur_s, codec) = ffprobe_duration_and_codec(&ffprobe, seg);
        assert_eq!(codec, "h264", "codec on {}", seg.display());
        assert!(
            (dur_s - dur_ns as f64 / 1e9).abs() < 0.25,
            "ffprobe {dur_s:.2}s vs our Duration {:.2}s on {}",
            dur_ns as f64 / 1e9,
            seg.display()
        );
        let interior = i + 1 < segments.len();
        if interior {
            assert!(
                (dur_s - 5.0).abs() < 1.0,
                "interior segment ≈5 s, got {dur_s:.2}s on {}",
                seg.display()
            );
        }
        total += dur_s;

        // (c) mpv plays it — whole file and the tail.
        mpv_decodes(&mpv, seg, None);
        mpv_decodes(&mpv, seg, Some("95%"));
    }
    // No recording gaps: the segments together cover the source (one frame of
    // slack per cut for the duration estimate).
    assert!(
        (total - 25.0).abs() < 0.5,
        "segment durations sum to the source length, got {total:.2}s"
    );
}
