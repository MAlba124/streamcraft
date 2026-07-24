//! `videotestsrc ! vkvideosink` — the human acceptance test for the GPU renderer
//! (spec: Milestone applications §5). Opens a real window and plays the deterministic
//! test pattern, converted on the GPU and presented as dma-bufs, paced on the clock.
//!
//! Usage: `cargo run --release -p sc-vk --example play_pattern_vk`
//!
//! Needs a Wayland session with linux-dmabuf and a Vulkan device that exports
//! dma-bufs; otherwise the sink posts one warning naming waylandvideosink and the
//! pipeline still completes (frames drop).

use std::sync::Arc;

use streamcraft_core::clock::InstantClock;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Rational;
use streamcraft_video::format::PixelFormat;
use streamcraft_video::{VideoFormat, VideoTestSrc};

use sc_vk::VkVideoSink;

fn main() {
    let width = 640u32;
    let height = 360u32;
    let fps = Rational::new(30, 1);
    let seconds = 4u64;
    let count = seconds * 30;

    let format = VideoFormat::new(width, height, PixelFormat::I420, fps);

    let mut p = Pipeline::new();
    let frame_bytes = (width as usize * height as usize * 3) / 2;
    p.set_pool(frame_bytes, 8);
    p.set_clock(Arc::new(InstantClock::new()));

    let src = p.add(VideoTestSrc::new(format, 0xC0FFEE, count));
    let sink = p.add(VkVideoSink::new().with_title("streamcraft — play_pattern (vk)"));
    p.link((src, "src"), (sink, "sink")).expect("videotestsrc -> vkvideosink");

    println!("playing {width}x{height}@30 test pattern on the GPU for {seconds}s …");
    let result = p.run();
    // Surface sink warnings (e.g. a fallback to the CPU path) — silence here is the
    // proof the dma-buf path actually presented.
    while let Some(msg) = p.bus().try_recv() {
        if let streamcraft_core::bus::BusMessage::Warning { error, .. } = msg {
            eprintln!("warning: {error:?}");
        }
    }
    match result {
        Ok(()) => println!("done"),
        Err(e) => {
            eprintln!("error: {e:?}");
            std::process::exit(1);
        }
    }
}
