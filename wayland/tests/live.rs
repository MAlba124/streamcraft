//! Live Wayland integration test (spec: task deliverable 5 — "one live integration test
//! that skips cleanly when `WAYLAND_DISPLAY` is unset"). On a box with a real session it
//! opens a window and presents 30 frames of `VideoTestSrc` under the real clock, asserting
//! the pipeline completes with no error. With no compositor it returns immediately.
//!
//! The sink is designed to degrade (drop frames) when it cannot reach a compositor, so this
//! test never hangs or fails in CI — it just does nothing there.

use std::sync::Arc;

use streamcraft_core::bus::BusMessage;
use streamcraft_core::clock::InstantClock;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Rational;
use streamcraft_video::format::PixelFormat;
use streamcraft_video::{VideoFormat, VideoTestSrc};

use sc_wayland::{WaylandClient, WaylandVideoSink};

/// True when a Wayland session appears to be available, so the live path can run.
fn have_wayland() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some()
}

#[test]
fn presents_thirty_frames_on_a_real_compositor_or_skips() {
    if !have_wayland() {
        eprintln!("no WAYLAND_DISPLAY — skipping live Wayland test");
        return;
    }
    // If a display var is set but no compositor actually answers (stale env), skip too
    // rather than fail — the connect probe tells us.
    if WaylandClient::connect().is_err() {
        eprintln!("WAYLAND_DISPLAY set but no compositor answered — skipping");
        return;
    }

    let width = 320u32;
    let height = 240u32;
    let fps = Rational::new(30, 1);
    let format = VideoFormat::new(width, height, PixelFormat::I420, fps);
    let frame_bytes = (width as usize * height as usize * 3) / 2;

    let mut p = Pipeline::new();
    p.set_pool(frame_bytes, 8);
    p.set_clock(Arc::new(InstantClock::new()));
    let src = p.add(VideoTestSrc::new(format, 0x5EED, 30));
    let sink = p.add(WaylandVideoSink::new().with_title("streamcraft — live test"));
    p.link((src, "src"), (sink, "sink")).expect("link");

    let run = p.run();

    // Drain the bus: no Error message should have been posted. (A QoS drop or a benign
    // warning may appear on a slow compositor; only a hard Error is a failure.)
    let mut hard_error = None;
    while let Some(msg) = p.bus().try_recv() {
        if let BusMessage::Error { error, .. } = msg {
            hard_error = Some(error);
        }
    }

    assert!(run.is_ok(), "pipeline run returned an error: {run:?}");
    assert!(hard_error.is_none(), "a hard error was posted: {hard_error:?}");
}
