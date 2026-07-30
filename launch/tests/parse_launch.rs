//! End-to-end tests for the registry / parse-launch path (spec: Plugins). These build
//! real pipelines from launch strings — the same surface `pf-launch` drives — and
//! run them to EOS, so a regression in the grammar, the descriptors, or the string
//! props shows up here.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use pf_flac::FlacDecoder;
use profluens_audio::{write_pcm_wav, AudioFormat, SampleFormat};
use profluens_core::pipeline::Pipeline;
use profluens_core::registry::Registry;

/// The registry every test uses — the same aggregation `pf-launch` builds.
fn registry() -> Registry {
    let mut r = Registry::new();
    profluens_elements::register(&mut r);
    profluens_audio::register(&mut r);
    pf_flac::register(&mut r);
    pf_ogg::register(&mut r);
    profluens_video::register(&mut r);
    pf_vp8::register(&mut r);
    pf_vp9::register(&mut r);
    pf_av1::register(&mut r);
    pf_h264::register(&mut r);
    pf_h265::register(&mut r);
    r
}

/// A process-unique temp path, so parallel test threads never collide on a fixed name.
fn temp_path(stem: &str, ext: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("pf_launch_{stem}_{pid}_{n}.{ext}"))
}

/// Generate a small stereo S16 WAV (a 440 Hz sine, 0.1 s) in the temp dir.
fn synth_wav(path: &std::path::Path) -> AudioFormat {
    let fmt = AudioFormat::new(44_100, 2, SampleFormat::S16);
    let n = fmt.sample_rate as usize / 10; // 0.1 s
    let mut pcm = Vec::with_capacity(n * 4);
    for i in 0..n {
        let t = i as f64 / fmt.sample_rate as f64;
        let s = (8_000.0 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as i16;
        pcm.extend_from_slice(&s.to_le_bytes()); // L
        pcm.extend_from_slice(&s.to_le_bytes()); // R
    }
    std::fs::write(path, write_pcm_wav(&fmt, &pcm)).unwrap();
    fmt
}

#[test]
fn testsrc_to_testsink_runs_to_eos() {
    // The headline acceptance one-liner: a byte source into a sink, driven to EOS.
    let mut p = Pipeline::new();
    let ids = registry()
        .parse(&mut p, "testsrc total=65536 ! testsink")
        .expect("parses");
    assert_eq!(ids.len(), 2);
    p.run().expect("runs to EOS");
    // The sink consumed exactly what the source produced.
    let src = p.counters(ids[0]);
    let sink = p.counters(ids[1]);
    assert_eq!(src.bytes_out, 65_536);
    assert_eq!(sink.bytes_in, 65_536);
}

#[test]
fn wav_to_flac_end_to_end() {
    // `filesrc ! wavparse ! flacenc ! filesink` on a generated WAV, then decode the
    // FLAC to prove it is a well-formed stream carrying the right sample count.
    let wav = temp_path("in", "wav");
    let flac = temp_path("out", "flac");
    let fmt = synth_wav(&wav);
    let expected_frames = (fmt.sample_rate as usize / 10) as u64;

    let launch = format!(
        "filesrc path={} ! wavparse ! flacenc rate=44100 channels=2 format=s16 ! filesink path={}",
        wav.display(),
        flac.display(),
    );
    let mut p = Pipeline::new();
    registry().parse(&mut p, &launch).expect("parses");
    p.run().expect("wav→flac runs");

    let bytes = std::fs::read(&flac).expect("flac written");
    assert!(!bytes.is_empty(), "flac output is non-empty");
    let decoded = FlacDecoder::decode(&bytes).expect("output is decodable FLAC");
    assert_eq!(decoded.info.sample_rate, 44_100);
    assert_eq!(decoded.info.channels, 2);
    assert_eq!(decoded.info.bits_per_sample, 16);
    // `samples` is interleaved: channels * frames.
    assert_eq!(decoded.samples.len() as u64, expected_frames * 2);

    let _ = std::fs::remove_file(&wav);
    let _ = std::fs::remove_file(&flac);
}

#[test]
fn file_copy_via_filesrc_filesink() {
    // A pure byte copy through the parse layer: string path props on both ends.
    let src = temp_path("src", "bin");
    let dst = temp_path("dst", "bin");
    let data: Vec<u8> = (0..50_000u32).map(|i| (i * 7) as u8).collect();
    std::fs::write(&src, &data).unwrap();

    let launch = format!("filesrc path={} ! filesink path={}", src.display(), dst.display());
    let mut p = Pipeline::new();
    registry().parse(&mut p, &launch).expect("parses");
    p.run().expect("copy runs");

    let copied = std::fs::read(&dst).unwrap();
    assert_eq!(copied, data, "byte-exact copy");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn dump_dot_reflects_parsed_topology() {
    let mut p = Pipeline::new();
    registry()
        .parse(&mut p, "testsrc total=1024 ! testsink")
        .expect("parses");
    let dot = p.dump_dot();
    assert!(dot.contains("testsrc"), "dot names the source");
    assert!(dot.contains("testsink"), "dot names the sink");
    assert!(dot.contains("->"), "dot has the edge");
}

#[test]
fn parse_errors_do_not_run() {
    // A malformed launch string is a clean error, never a panic or a half-built run.
    let mut p = Pipeline::new();
    let err = registry()
        .parse(&mut p, "testsrc ! nosuchelement ! testsink")
        .expect_err("unknown element must fail");
    assert!(err.message.contains("nosuchelement"), "{}", err.message);
}

// --- The raw-video path through the parse layer (spec: Milestone applications §5) ----

#[test]
fn videotestsrc_to_videocksink_runs_to_eos() {
    // The headline video one-liner: a raw-video source into the clock sink, driven to
    // EOS. `videocksink` renders on the default real clock, so keep the frame count
    // small (30 frames at the default 30 fps ≈ 1 s of wall time).
    let mut p = Pipeline::new();
    let ids = registry()
        .parse(&mut p, "videotestsrc frames=30 ! videocksink")
        .expect("parses");
    assert_eq!(ids.len(), 2);
    p.run().expect("runs to EOS");
    // The sink saw one buffer per frame.
    let sink = p.counters(ids[1]);
    assert_eq!(sink.buffers_in, 30, "one buffer per frame reached the sink");
}

#[test]
fn videotestsrc_props_override_and_run_to_eos() {
    // Prop plumbing end to end: overriding width/height/frames through the parse layer
    // produces a valid, smaller pipeline that runs to EOS. (The stats handle is
    // unreachable through the parse path, so EOS + the sink's buffer count is the
    // assertion.)
    let mut p = Pipeline::new();
    let ids = registry()
        .parse(&mut p, "videotestsrc width=64 height=48 frames=5 ! videocksink")
        .expect("parses");
    p.run().expect("runs to EOS");
    let sink = p.counters(ids[1]);
    assert_eq!(sink.buffers_in, 5, "the frames prop took effect");
}

#[test]
fn video_decoders_are_registered_and_listed() {
    // Every video decoder registers and appears in the sorted name list `--list` prints.
    let names = registry().names();
    for dec in ["vp8dec", "vp9dec", "av1dec", "h264dec", "h265dec"] {
        assert!(names.contains(&dec), "{dec} should be registered");
        // A registered decoder is also constructible by name (it has a make_default).
        let mut p = Pipeline::new();
        // A decoder alone is a valid single-stage chain to construct; it just needs an
        // input to actually run, so we only assert it parses/builds here.
        let launch = format!("{dec}");
        assert!(registry().parse(&mut p, &launch).is_ok(), "{dec} builds by name");
    }
    // The raw-video elements are listed too.
    for e in ["videotestsrc", "rawvideoparse", "videocksink"] {
        assert!(names.contains(&e), "{e} should be registered");
    }
}
