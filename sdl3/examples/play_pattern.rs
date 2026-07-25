//! `videotestsrc ! sdl3videosink` — the human acceptance test for the hand-written
//! Wayland shm sink (spec: Milestone applications §5). Opens a real window and plays a
//! deterministic test pattern at ~640x360@30 for a few seconds, paced on the pipeline clock.
//!
//! Usage: `cargo run -p sc-sdl3 --example play_pattern`
//!
//! Requires a running Wayland session (`$WAYLAND_DISPLAY` set). With no compositor the sink
//! degrades to dropping frames, so this still runs to completion headless — but you only see
//! a window on a real desktop.

use std::sync::Arc;

use streamcraft_core::clock::InstantClock;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Rational;
use streamcraft_video::format::PixelFormat;
use streamcraft_video::{VideoFormat, VideoTestSrc};

use sc_sdl3::Sdl3VideoSink;

fn main() {
    // 640x360 I420 at 30 fps, a few seconds' worth of frames.
    let width = 640u32;
    let height = 360u32;
    let fps = Rational::new(30, 1);
    let seconds = 4u64;
    let count = seconds * 30;

    let format = VideoFormat::new(width, height, PixelFormat::I420, fps);

    let mut p = Pipeline::new();
    // The pool must hold at least one full I420 frame (w*h*3/2) per slot.
    let frame_bytes = (width as usize * height as usize * 3) / 2;
    p.set_pool(frame_bytes, 8);
    // Real time so the window plays at wall-clock speed.
    p.set_clock(Arc::new(InstantClock::new()));

    let src = p.add(VideoTestSrc::new(format, 0xC0FFEE, count));
    let sink = p.add(Sdl3VideoSink::new().with_title("streamcraft — play_pattern"));
    p.link((src, "src"), (sink, "sink"))
        .expect("videotestsrc -> sdl3videosink");

    println!("playing {width}x{height}@30 test pattern for {seconds}s …");
    match p.run() {
        Ok(()) => println!("done"),
        Err(e) => {
            eprintln!("error: {e:?}");
            std::process::exit(1);
        }
    }
}
