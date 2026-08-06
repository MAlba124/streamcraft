//! `decode_annexb` — diagnostic: feed a raw Annex B HEVC byte stream straight to the
//! upstream `SequenceDecoder` (no pipeline), reporting decode progress and timing —
//! isolates the library from the element/scheduler when a real-world stream
//! misbehaves.
//!
//! ```text
//! cargo run --release -p pf-h265 --example decode_annexb -- IN.h265 [chunk_bytes]
//! ```

use oxideav_h265::SequenceDecoder;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(input) = args.next() else {
        eprintln!("usage: decode_annexb IN.h265 [chunk_bytes]");
        std::process::exit(2);
    };
    let chunk: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64 * 1024);

    let data = std::fs::read(&input).expect("read input");
    eprintln!("{}: {} bytes, {}-byte pushes", input, data.len(), chunk);

    let mut dec = SequenceDecoder::new();
    let t0 = Instant::now();
    let mut frames = 0usize;
    let mut errors = 0usize;
    for (i, c) in data.chunks(chunk).enumerate() {
        let t = Instant::now();
        match dec.push_annexb(c) {
            Ok(()) => {}
            Err(e) => {
                errors += 1;
                if errors <= 5 {
                    eprintln!("push {i}: ERROR {e:?}");
                }
            }
        }
        let got = dec.take_decoded();
        if !got.is_empty() || t.elapsed().as_millis() > 200 {
            for f in &got {
                eprintln!(
                    "  frame poc={} {}x{} bd={} output={} (+{} ms)",
                    f.poc,
                    f.picture.width_luma(),
                    f.picture.height_luma(),
                    f.picture.bit_depth_luma(),
                    f.output,
                    t.elapsed().as_millis()
                );
            }
            frames += got.len();
        } else {
            frames += got.len();
        }
    }
    let t = Instant::now();
    let _ = dec.flush();
    let tail = dec.take_decoded();
    eprintln!(
        "flush: +{} frames in {} ms",
        tail.len(),
        t.elapsed().as_millis()
    );
    frames += tail.len();
    eprintln!(
        "total: {frames} frames, {errors} push errors, {:.2}s",
        t0.elapsed().as_secs_f64()
    );
}
