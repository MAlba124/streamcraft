//! `WaylandVideoSink` element tests that run *everywhere* (no compositor needed). They
//! prove the pipeline wiring: the sink negotiates `video/raw`, learns its format from
//! `VideoTestSrc`'s announcement, paces on the clock, and — crucially — degrades to
//! dropping frames (never hangs, never errors the pipeline) when no display is available.
//!
//! To make the "no display" path deterministic regardless of the CI box, these tests clear
//! `WAYLAND_DISPLAY`/`XDG_RUNTIME_DIR` for their process so the sink's `connect()` fails and
//! it disables cleanly. They run serially (a shared static lock) because env vars are
//! process-global.

use std::sync::{Arc, Mutex};

use streamcraft_core::clock::MockClock;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::{Rational, Timestamp};
use streamcraft_video::format::PixelFormat;
use streamcraft_video::{VideoFormat, VideoTestSrc};

use sc_wayland::WaylandVideoSink;

/// Serialise env-var mutation across the tests in this binary.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Run `f` with the Wayland environment removed, so the sink cannot connect to a display
/// and takes its graceful "drop frames" path. Restores the vars afterwards.
fn without_display<R>(f: impl FnOnce() -> R) -> R {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let saved_display = std::env::var_os("WAYLAND_DISPLAY");
    let saved_runtime = std::env::var_os("XDG_RUNTIME_DIR");
    std::env::remove_var("WAYLAND_DISPLAY");
    std::env::remove_var("XDG_RUNTIME_DIR");
    let r = f();
    if let Some(v) = saved_display {
        std::env::set_var("WAYLAND_DISPLAY", v);
    }
    if let Some(v) = saved_runtime {
        std::env::set_var("XDG_RUNTIME_DIR", v);
    }
    r
}

#[test]
fn negotiates_and_runs_to_eos_without_a_display() {
    without_display(|| {
        // 64x48 I420 at 30 fps, 30 frames — a few seconds of virtual time under a MockClock.
        let fps = Rational::new(30, 1);
        let format = VideoFormat::new(64, 48, PixelFormat::I420, fps);
        let frame_bytes = (64 * 48 * 3) / 2;

        let clock = MockClock::new();
        let mut p = Pipeline::new();
        p.set_pool(frame_bytes, 8);
        p.set_clock(Arc::new(clock.clone()));
        let src = p.add(VideoTestSrc::new(format, 0xABCD, 30));
        let sink = p.add(WaylandVideoSink::new());
        p.link((src, "src"), (sink, "sink"))
            .expect("videotestsrc -> waylandvideosink negotiates video/raw");

        // Drive virtual time forward so clock-paced frames all come due, and the run ends.
        let run = std::thread::spawn(move || p.run());
        let step = Timestamp::from_millis(200);
        while !run.is_finished() {
            clock.advance(step);
            std::thread::yield_now();
        }
        run.join().expect("run joined").expect("pipeline completes cleanly with no display");
    });
}

#[test]
fn gray8_format_also_negotiates_without_a_display() {
    without_display(|| {
        let fps = Rational::new(25, 1);
        let format = VideoFormat::new(32, 32, PixelFormat::Gray8, fps);
        let frame_bytes = 32 * 32;

        let clock = MockClock::new();
        let mut p = Pipeline::new();
        p.set_pool(frame_bytes, 8);
        p.set_clock(Arc::new(clock.clone()));
        let src = p.add(VideoTestSrc::new(format, 7, 10));
        let sink = p.add(WaylandVideoSink::new());
        p.link((src, "src"), (sink, "sink"))
            .expect("gray8 video/raw negotiates against the sink offer");

        let run = std::thread::spawn(move || p.run());
        let step = Timestamp::from_millis(200);
        while !run.is_finished() {
            clock.advance(step);
            std::thread::yield_now();
        }
        run.join().expect("joined").expect("gray8 pipeline completes");
    });
}
