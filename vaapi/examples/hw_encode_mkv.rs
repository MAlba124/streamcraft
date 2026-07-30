//! Hardware-encode a test pattern into an MKV file:
//! `videotestsrc → vaapi{h264,h265,vp8}enc → mkvmuxn → filesink`.
//!
//! Usage: `cargo run -p pf-vaapi --example hw_encode_mkv -- <h264|h265|vp8> [out.mkv] [frames]`
//!
//! This is the external-validation vehicle for the encoders: the produced file
//! must open in ffprobe *and* headless mpv (the workspace's muxer rule — the two
//! disagree often enough that both are required), proving the whole chain —
//! negotiation (`h264/avcc`/`h265/hvcc`/`vp8` lanes), the avcC/hvcC records this
//! crate authors, keyframe flags, and the hardware bitstreams themselves —
//! against decoders that share no code with this workspace.

use std::path::PathBuf;

use pf_mkv::MkvMuxN;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Rational;
use profluens_elements::io::FileSink;
use profluens_video::{PixelFormat, VideoFormat, VideoTestSrc};

fn main() {
    let mut args = std::env::args().skip(1);
    let codec = args.next().unwrap_or_else(|| "h264".to_string());
    let out = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/pf_hw_{codec}.mkv")));
    let frames: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(90);

    let Some(caps) = pf_vaapi::probe() else {
        eprintln!("no VA-API device available");
        std::process::exit(1);
    };
    let family = match codec.as_str() {
        "h264" => "h264/annexb",
        "h265" => "h265/annexb",
        "vp8" => "vp8",
        other => {
            eprintln!("unknown codec {other:?} (want h264|h265|vp8)");
            std::process::exit(2);
        }
    };
    if !caps.supports_encode(family) {
        eprintln!("device does not advertise {codec} encode");
        std::process::exit(1);
    }

    // 640×360: even dimensions, but 360 is not MB/CTB aligned — the coded picture
    // pads to 368 and the crop must round-trip through the container (ffprobe
    // must report 640x360).
    let fmt = VideoFormat::new(640, 360, PixelFormat::I420, Rational { num: 30, den: 1 });

    let mut p = Pipeline::new();
    // Raw 640×360 I420 frames are ~346 KiB — larger than default pool slots.
    p.set_pool(1024 * 1024, 16);
    let src = p.add(VideoTestSrc::new(fmt, 7, frames));
    // 30-frame GOP = 1-second clusters at 30 fps, the typical Matroska cadence.
    // The mkv writer emits **sized** Clusters (the ecosystem-safe form), so it
    // holds every frame of the open cluster — one full GOP — as refcounted pool
    // slots until the next keyframe closes it. The encoder's pool must therefore
    // be deeper than the GOP, or the pipeline deadlocks: writer waits for frame
    // N+1, encoder waits for a slot the writer holds.
    let enc = match codec.as_str() {
        "h264" => p.add(pf_vaapi::VaapiH264Enc::new().with_gop(30)),
        "h265" => p.add(pf_vaapi::VaapiH265Enc::new().with_gop(30)),
        _ => p.add(pf_vaapi::VaapiVp8Enc::new().with_gop(30)),
    };
    // Per-element pools also break shared-pool deadlock cycles (the class the
    // RTP payloader hit): compressed output must not compete with the source's
    // 1 MiB raw frames, nor the muxer's coalescing writer wait on a pool the
    // source has drained.
    p.set_element_pool(enc, 512 * 1024, 48);
    let mux = p.add(MkvMuxN::new());
    p.set_element_pool(mux, 512 * 1024, 8);
    let sink = p.add(FileSink::new(&out));
    p.link((src, "src"), (enc, "sink")).expect("link src→enc");
    p.link((enc, "src"), (mux, "sink_0")).expect("link enc→mux");
    p.link((mux, "src"), (sink, "sink")).expect("link mux→sink");

    match p.run() {
        Ok(()) => {
            let n = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
            println!("OK: {codec} → {} ({n} bytes, {frames} frames)", out.display());
        }
        Err(e) => {
            eprintln!("FAILED: {e:?}");
            std::process::exit(1);
        }
    }
}
