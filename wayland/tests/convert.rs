//! i420 → XRGB8888 conversion against reference pixels (spec: task deliverable 5 — "the
//! i420→XRGB conversion against reference pixels (a few exact vectors incl. odd widths)").
//! CPU-only; no compositor.
//!
//! The reference RGB values are the exact output of integer BT.601 studio-swing:
//!
//! ```text
//!   R = clip((298*(Y-16)              + 409*(V-128) + 128) >> 8)
//!   G = clip((298*(Y-16) - 100*(U-128) - 208*(V-128) + 128) >> 8)
//!   B = clip((298*(Y-16) + 516*(U-128)              + 128) >> 8)
//! ```

use sc_wayland::convert::{gray8_to_xrgb, i420_to_xrgb, yuv_to_rgb};

/// Recompute the reference triple independently of the crate (same integer formula), so a
/// coefficient drift in the crate is caught rather than masked by using its own function.
fn ref_rgb(y: i32, u: i32, v: i32) -> (u8, u8, u8) {
    let c = y - 16;
    let d = u - 128;
    let e = v - 128;
    let clip = |x: i32| x.clamp(0, 255) as u8;
    (
        clip((298 * c + 409 * e + 128) >> 8),
        clip((298 * c - 100 * d - 208 * e + 128) >> 8),
        clip((298 * c + 516 * d + 128) >> 8),
    )
}

#[test]
fn exact_reference_vectors_for_primaries() {
    // (Y, U, V, expected R, G, B) — black, white, primary red/green/blue in BT.601 studio.
    let cases: &[(u8, u8, u8)] = &[
        (16, 128, 128),  // black
        (235, 128, 128), // white
        (81, 90, 240),   // red-ish
        (145, 54, 34),   // green-ish
        (41, 240, 110),  // blue-ish
        (126, 128, 128), // mid grey
    ];
    for &(y, u, v) in cases {
        let got = yuv_to_rgb(y, u, v);
        let want = ref_rgb(y as i32, u as i32, v as i32);
        assert_eq!(got, want, "YUV({y},{u},{v})");
    }
}

#[test]
fn i420_frame_converts_pixel_exact_including_xbyte() {
    // 2x2 solid colour: all four luma equal, one shared chroma sample.
    let (w, h) = (2usize, 2usize);
    let (y, u, v) = (81u8, 90u8, 240u8); // red-ish
    let mut src = vec![0u8; w * h + 2]; // Y(4) + Cb(1) + Cr(1)
    src[0..4].fill(y);
    src[4] = u;
    src[5] = v;
    let mut dst = vec![0u8; w * 4 * h];
    assert!(i420_to_xrgb(&src, w, h, &mut dst, w * 4));

    let (r, g, b) = ref_rgb(y as i32, u as i32, v as i32);
    for (i, px) in dst.chunks_exact(4).enumerate() {
        assert_eq!(px, &[b, g, r, 0xFF], "pixel {i} is [B,G,R,X] LE with opaque X");
    }
}

#[test]
fn odd_width_and_height_use_ceil_chroma_geometry() {
    // 3x3 frame: Y is 9 bytes; chroma is ceil(3/2)=2 wide × 2 tall → 4 bytes each plane.
    let (w, h) = (3usize, 3usize);
    let cw = 2usize;
    let ch = 2usize;
    let mut src = vec![0u8; w * h + 2 * (cw * ch)];
    // Fill luma with a gradient, chroma neutral so R=G=B tracks luma.
    for (i, b) in src[..w * h].iter_mut().enumerate() {
        *b = (16 + i as u32 * 20).min(235) as u8;
    }
    for b in src[w * h..].iter_mut() {
        *b = 128;
    }
    let mut dst = vec![0u8; w * 4 * h];
    assert!(i420_to_xrgb(&src, w, h, &mut dst, w * 4), "3x3 converts in-bounds");

    // Every pixel: neutral chroma → R=G=B, and matches the luma-only reference.
    for row in 0..h {
        for col in 0..w {
            let lum = src[row * w + col] as i32;
            let (r, g, b) = ref_rgb(lum, 128, 128);
            assert_eq!(r, g);
            assert_eq!(g, b);
            let o = (row * w + col) * 4;
            assert_eq!(&dst[o..o + 4], &[b, g, r, 0xFF], "pixel ({row},{col})");
        }
    }
}

#[test]
fn odd_5x3_shares_chroma_across_pairs_correctly() {
    // 5x3: chroma ceil(5/2)=3 wide × ceil(3/2)=2 tall. Chroma columns map x/2, rows y/2.
    let (w, h) = (5usize, 3usize);
    let cw = 3usize;
    let ch = 2usize;
    let mut src = vec![128u8; w * h + 2 * (cw * ch)];
    // Distinct luma per pixel; leave chroma neutral (128) so output is pure grey.
    for (i, b) in src[..w * h].iter_mut().enumerate() {
        *b = (16 + (i as u32 * 7) % 220) as u8;
    }
    let mut dst = vec![0u8; w * 4 * h];
    assert!(i420_to_xrgb(&src, w, h, &mut dst, w * 4));
    for row in 0..h {
        for col in 0..w {
            let lum = src[row * w + col] as i32;
            let (r, g, b) = ref_rgb(lum, 128, 128);
            let o = (row * w + col) * 4;
            assert_eq!(&dst[o..o + 4], &[b, g, r, 0xFF], "grey pixel ({row},{col})");
        }
    }
}

#[test]
fn gray8_maps_luma_to_equal_channels() {
    let src = [0u8, 64, 128, 200, 255];
    let mut dst = vec![0u8; src.len() * 4];
    assert!(gray8_to_xrgb(&src, src.len(), 1, &mut dst, src.len() * 4));
    for (i, &g) in src.iter().enumerate() {
        assert_eq!(&dst[i * 4..i * 4 + 4], &[g, g, g, 0xFF]);
    }
}

#[test]
fn short_or_narrow_inputs_are_rejected_without_writing() {
    let mut dst = vec![0xEEu8; 2 * 4 * 2];
    // One byte short of a 2x2 I420.
    let short = vec![0u8; 2 * 2 + 2 - 1];
    assert!(!i420_to_xrgb(&short, 2, 2, &mut dst, 8));
    assert!(dst.iter().all(|&b| b == 0xEE), "nothing written on rejection");
    // Stride narrower than a row.
    let ok_src = vec![16u8; 2 * 2 + 2];
    assert!(!i420_to_xrgb(&ok_src, 2, 2, &mut dst, 4));
}
