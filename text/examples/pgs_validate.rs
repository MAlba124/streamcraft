//! Offline validation: decode the real Coneheads PGS stream (`/tmp/con.sup`, extracted with
//! `ffmpeg -map 0:s:0 -c copy`) with `pf_text::pgs` and report the decode.
//!
//! Run: `nix develop --command cargo run --release -p pf-text --example pgs_validate`

use pf_text::pgs;

const SUP: &str = "/tmp/con.sup";

fn main() {
    #[allow(clippy::disallowed_methods)] // offline validation tool — not element code
    let Ok(data) = std::fs::read(SUP) else {
        eprintln!("pgs_validate: {SUP} not found — run the ffmpeg extract first, skipping");
        return;
    };
    println!("pgs_validate: {} bytes of raw PGS/SUP", data.len());

    // Peel the .sup PG/timestamp framing into per-Display-Set bare segment blobs.
    let sets = pgs::split_sup(&data);
    println!("pgs_validate: {} display sets", sets.len());

    let mut shown = 0usize;
    let mut cleared = 0usize;
    let mut dropped = 0usize;
    let mut samples = 0usize;
    let mut total_opaque = 0u64;

    for (i, (pts90k, _dts, ds_bytes)) in sets.iter().enumerate() {
        match pgs::parse_display_set(ds_bytes) {
            Ok(ds) => {
                if ds.is_clear() {
                    cleared += 1;
                } else {
                    shown += 1;
                    // Count non-transparent pixels — a sanity signal the RLE decoded real ink.
                    let opaque = ds.rgba.as_chunks::<4>().0.iter().filter(|p| p[3] > 0).count();
                    total_opaque += opaque as u64;
                    if samples < 6 {
                        let pts_s = *pts90k as f64 / 90_000.0;
                        println!(
                            "  DS#{i}: pts={pts_s:.3}s  bitmap {}x{} @ ({},{})  ref {}x{}  \
                             rgba={}B  opaque_px={opaque}",
                            ds.width, ds.height, ds.x, ds.y, ds.video_width, ds.video_height,
                            ds.rgba.len()
                        );
                        // A couple of concrete pixels for a "non-garbage" spot-check.
                        if opaque > 0 {
                            if let Some(px) = ds.rgba.as_chunks::<4>().0.iter().find(|p| p[3] > 0) {
                                println!(
                                    "        first opaque pixel RGBA = ({},{},{},{})",
                                    px[0], px[1], px[2], px[3]
                                );
                            }
                            // Count near-white opaque pixels — the glyph *fill* (white text),
                            // distinct from the dark low-alpha outline edge.
                            let white = ds
                                .rgba
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .filter(|p| p[3] > 200 && p[0] > 200 && p[1] > 200 && p[2] > 200)
                                .count();
                            println!("        near-white opaque (glyph fill) pixels = {white}");
                        }
                        samples += 1;
                    }
                }
            }
            Err(e) => {
                dropped += 1;
                if dropped <= 3 {
                    println!("  DS#{i}: DROPPED — {}", e.reason());
                }
            }
        }
    }

    println!(
        "pgs_validate: shown={shown} cleared={cleared} dropped={dropped}  \
         total_opaque_px={total_opaque}"
    );
    assert!(shown > 0, "at least one caption bitmap decoded");
    assert!(total_opaque > 0, "decoded bitmaps carry non-empty (opaque) ink");
    assert!(dropped * 20 < sets.len().max(1), "the vast majority of display sets decode cleanly");
    println!("pgs_validate: PASS — real Coneheads PGS decodes to non-empty RGBA captions");
}
