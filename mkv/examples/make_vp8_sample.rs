//! `make_vp8_sample` — author a small VP8-in-MKV file with our own writer + the
//! adopted encoder (spec: Milestone applications §5 — the test clip for
//! `play_mkv`). All-keyframe on purpose: every frame is independently decodable,
//! which keeps the generator trivial and seeking exact.
//!
//! ```text
//! cargo run --release -p sc-mkv --example make_vp8_sample -- OUT.mkv [SECONDS] [WIDTH HEIGHT]
//! ```

use oxideav_vp8::encoder::{encode_keyframe, I420Frame, KeyframeParams};
use sc_mkv::{MatroskaWriter, TrackConfig};

const FPS: u64 = 30;

/// A deterministic animated test picture: a diagonally scrolling luma gradient with
/// a chroma sweep, so playback visibly moves.
fn make_planes(idx: u32, w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let t = idx as usize * 4;
    let y = (0..h)
        .flat_map(|row| (0..w).map(move |col| (((row + col + t) * 255 / (w + h)) & 0xFF) as u8))
        .collect();
    let u = (0..ch)
        .flat_map(|row| (0..cw).map(move |col| ((col * 2 + t + row) & 0xFF) as u8))
        .collect();
    let v = (0..ch)
        .flat_map(|row| (0..cw).map(move |col| ((row * 2 + 255 - (t % 256) + col) & 0xFF) as u8))
        .collect();
    (y, u, v)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(out) = args.next() else {
        eprintln!("usage: make_vp8_sample OUT.mkv [SECONDS] [WIDTH HEIGHT]");
        std::process::exit(2);
    };
    let secs: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let w: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(320);
    let h: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(240);
    let frames = secs * FPS;
    let frame_ns = 1_000_000_000 / FPS;

    let mut writer = MatroskaWriter::new(vec![TrackConfig::vp8(1, w, h)]);
    let mut bytes = Vec::new();
    writer.write_header(&mut bytes).expect("header");
    for i in 0..frames {
        let (y, u, v) = make_planes(i as u32, w as usize, h as usize);
        let frame = I420Frame::packed(w, h, &y, &u, &v);
        let coded = encode_keyframe(&frame, &KeyframeParams::default()).expect("encode");
        writer
            .write_frame(&mut bytes, 1, i * frame_ns, &coded, true)
            .expect("write frame");
    }
    writer.finalize(&mut bytes);
    std::fs::write(&out, &bytes).expect("write file");
    println!(
        "wrote {out}: {}x{w}x{h} VP8 keyframes, {secs}s @ {FPS}fps, {} bytes",
        frames,
        bytes.len()
    );
}
