//! I420 (and Gray8) → XRGB8888 colour conversion (spec: the task's "i420 → XRGB8888
//! (integer BT.601, scalar; convert directly into the shm buffer — one pass")).
//!
//! `wl_shm` `XRGB8888` is a little-endian 32-bit pixel: byte 0 = Blue, byte 1 = Green,
//! byte 2 = Red, byte 3 = unused (X). So each pixel word, read as a native `u32`, is
//! `0x00RRGGBB`; on the wire (LE) that lands as `[B, G, R, X]`, which is what the
//! compositor samples.
//!
//! The colour maths is **integer BT.601 studio-swing** (the ITU-R BT.601 limited-range
//! matrix, the ubiquitous SD/`videotestsrc` default). Derivation, per Rec. ITU-R
//! BT.601-7 (03/2011) — see `REFERENCES.md`:
//!
//! - §2.5.1 gives the luma weights Kr = 0.299, Kb = 0.114 (Kg = 1 − Kr − Kb), i.e.
//!   R = Y' + 2(1−Kr)·Cr, B = Y' + 2(1−Kb)·Cb, G from the weight identity.
//! - §3.5 quantizes studio swing: Y' spans 16..235 (219 steps), Cb/Cr span ±112
//!   around 128, so the full-range expansion scales by 255/219 (luma) and 255/112·(…)
//!   (chroma) before the matrix.
//!
//! Folding the range expansion into the matrix and scaling by 256 for 8-bit integer
//! arithmetic (rounding term +128, shift >> 8) yields:
//!
//! ```text
//!   C = Y - 16 ;  D = U - 128 ;  E = V - 128
//!   R = clip(( 298*C           + 409*E + 128) >> 8)
//!   G = clip(( 298*C -  100*D  - 208*E + 128) >> 8)
//!   B = clip(( 298*C +  516*D          + 128) >> 8)
//! ```
//!
//! where each coefficient is the analytic BT.601 value rounded to the nearest 1/256:
//! 298 = ⌈256·255/219⌋; 409 = ⌈256·(255/224)·2(1−Kr)⌋; 516 = ⌈256·(255/224)·2(1−Kb)⌋;
//! and the G-row terms 208/100 are those chroma gains weighted by Kr/Kg and Kb/Kg
//! (Kg = 1 − Kr − Kb). Rounding error is < 1/2 LSB per term — within the ±1 LSB
//! agreement the conversion tests assert.
//!
//! One pass, no intermediate buffer: for each output row we index the matching luma row
//! and the *subsampled* chroma row (`y/2`), and each chroma sample serves two horizontal
//! luma pixels (`x/2`). Odd widths/heights use the round-up chroma geometry from
//! `streamcraft-video` (`ceil(w/2)`), so the trailing column/row still has a chroma sample.

/// Clamp an `i32` to a `u8` (studio→full-range clip).
#[inline]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Write one XRGB8888 pixel (`[B, G, R, X]`, LE) at `dst[o..o+4]`.
#[inline]
fn put_xrgb(dst: &mut [u8], o: usize, r: u8, g: u8, b: u8) {
    dst[o] = b;
    dst[o + 1] = g;
    dst[o + 2] = r;
    dst[o + 3] = 0xFF; // X byte: ignored by XRGB, set opaque for ARGB-compatible reuse
}

/// Convert one BT.601 (Y, U, V) sample triple to (R, G, B).
#[inline]
pub fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let c = y as i32 - 16;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    let r = clip((298 * c + 409 * e + 128) >> 8);
    let g = clip((298 * c - 100 * d - 208 * e + 128) >> 8);
    let b = clip((298 * c + 516 * d + 128) >> 8);
    (r, g, b)
}

/// Convert a tightly-packed I420 frame to XRGB8888 directly into `dst`, one output row at a
/// time. `dst` must be at least `dst_stride * height` bytes and `dst_stride >= width * 4`;
/// `src` must be a tightly-packed I420 frame of `(width, height)` (Y plane `w*h`, then Cb
/// `ceil(w/2)*ceil(h/2)`, then Cr the same). Returns `false` (writing nothing) if any size
/// is inconsistent, so a caller never indexes out of bounds on a malformed frame.
///
/// `dst_stride` lets the caller target an shm buffer whose row stride is `width * 4`
/// (our pools use the tight stride) or a wider negotiated stride.
pub fn i420_to_xrgb(
    src: &[u8],
    width: usize,
    height: usize,
    dst: &mut [u8],
    dst_stride: usize,
) -> bool {
    if width == 0 || height == 0 {
        return true;
    }
    let cw = width.div_ceil(2); // ceil(w/2) — chroma width
    let ch = height.div_ceil(2); // ceil(h/2) — chroma height
    let y_size = width * height;
    let c_size = cw * ch;
    if src.len() < y_size + 2 * c_size {
        return false;
    }
    if dst_stride < width * 4 || dst.len() < dst_stride * height {
        return false;
    }
    let (y_plane, rest) = src.split_at(y_size);
    let (u_plane, v_rest) = rest.split_at(c_size);
    let v_plane = &v_rest[..c_size];

    for row in 0..height {
        let y_row = &y_plane[row * width..row * width + width];
        let crow = row / 2;
        let u_row = &u_plane[crow * cw..crow * cw + cw];
        let v_row = &v_plane[crow * cw..crow * cw + cw];
        let dst_row = &mut dst[row * dst_stride..row * dst_stride + width * 4];
        for col in 0..width {
            let ccol = col / 2;
            let (r, g, b) = yuv_to_rgb(y_row[col], u_row[ccol], v_row[ccol]);
            put_xrgb(dst_row, col * 4, r, g, b);
        }
    }
    true
}

/// Convert a tightly-packed Gray8 frame to XRGB8888 (luma replicated across R=G=B), one
/// row at a time, directly into `dst`. Same size contract as [`i420_to_xrgb`].
pub fn gray8_to_xrgb(
    src: &[u8],
    width: usize,
    height: usize,
    dst: &mut [u8],
    dst_stride: usize,
) -> bool {
    if width == 0 || height == 0 {
        return true;
    }
    if src.len() < width * height {
        return false;
    }
    if dst_stride < width * 4 || dst.len() < dst_stride * height {
        return false;
    }
    for row in 0..height {
        let s = &src[row * width..row * width + width];
        let dst_row = &mut dst[row * dst_stride..row * dst_stride + width * 4];
        for col in 0..width {
            let g = s[col];
            put_xrgb(dst_row, col * 4, g, g, g);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference BT.601 studio-swing conversions computed from the integer formulas above.
    // These are exact for the primaries/grey the maths defines, so a regression in the
    // coefficients or clipping is caught by an exact byte compare.
    #[test]
    fn primary_colours_convert_to_expected_rgb() {
        // Black: Y=16, U=V=128 → (0,0,0).
        assert_eq!(yuv_to_rgb(16, 128, 128), (0, 0, 0));
        // White: Y=235, U=V=128 → ~(255,255,255) (298*219>>8 = 255 after clip).
        assert_eq!(yuv_to_rgb(235, 128, 128), (255, 255, 255));
        // Mid grey: Y=126 (~0.5 studio), U=V=128 → equal channels, no colour cast.
        let (r, g, b) = yuv_to_rgb(126, 128, 128);
        assert_eq!(r, g);
        assert_eq!(g, b);
    }

    #[test]
    fn saturated_chroma_clips_into_range() {
        // Extreme red-ish chroma must clip, not wrap. V=255 with mid luma pushes R high.
        let (r, _g, _b) = yuv_to_rgb(128, 128, 255);
        assert_eq!(r, 255, "R saturates to 255, no overflow wrap");
        // Extreme opposite pushes a channel below 0 → clipped to 0.
        let (_r, _g, b) = yuv_to_rgb(16, 0, 128);
        // B = clip(298*0 + 516*(-128) + 128 >> 8) = clip(negative) = 0
        assert_eq!(b, 0);
    }

    #[test]
    fn i420_2x2_solid_red_gives_red_xrgb_pixels() {
        // 2x2 I420: one chroma sample covers all four luma pixels. Choose Y/U/V for red.
        // Red in BT.601 studio ≈ Y=81, U=90, V=240.
        let (w, h) = (2usize, 2usize);
        let mut src = vec![0u8; w * h + 2 * 1]; // Y(4) + Cb(1) + Cr(1)
        src[0..4].fill(81);
        src[4] = 90; // Cb
        src[5] = 240; // Cr
        let mut dst = vec![0u8; w * 4 * h];
        assert!(i420_to_xrgb(&src, w, h, &mut dst, w * 4));
        let (er, eg, eb) = yuv_to_rgb(81, 90, 240);
        for px in dst.chunks_exact(4) {
            assert_eq!(px[0], eb, "B");
            assert_eq!(px[1], eg, "G");
            assert_eq!(px[2], er, "R");
            assert_eq!(px[3], 0xFF, "X byte opaque");
        }
    }

    #[test]
    fn i420_odd_width_uses_ceil_chroma_and_stays_in_bounds() {
        // 3x1 frame: Y is 3 bytes, chroma ceil(3/2)=2 wide, ceil(1/2)=1 tall → 2 each.
        let (w, h) = (3usize, 1usize);
        let cw = 2usize;
        let mut src = vec![0u8; w * h + 2 * cw];
        // Distinct luma per column, distinct chroma per chroma-column.
        src[0] = 16; // col 0 luma (black-ish)
        src[1] = 235; // col 1 luma (white-ish)
        src[2] = 126; // col 2 luma (grey)
        // chroma: neutral so RGB tracks luma
        src[w] = 128; // Cb[0]
        src[w + 1] = 128; // Cb[1]
        src[w + cw] = 128; // Cr[0]
        src[w + cw + 1] = 128; // Cr[1]
        let mut dst = vec![0u8; w * 4 * h];
        assert!(i420_to_xrgb(&src, w, h, &mut dst, w * 4), "odd width converts");
        // Column 0 black, column 1 white, column 2 grey — R=G=B for neutral chroma.
        assert_eq!(&dst[0..4], &[0, 0, 0, 0xFF]);
        assert_eq!(&dst[4..8], &[255, 255, 255, 0xFF]);
        let (r, g, b) = yuv_to_rgb(126, 128, 128);
        assert_eq!(&dst[8..12], &[b, g, r, 0xFF]);
    }

    #[test]
    fn rejects_short_source_and_destination() {
        let mut dst = vec![0u8; 2 * 4 * 2];
        // Source one byte short of a 2x2 I420 frame.
        let short = vec![0u8; 2 * 2 + 2 - 1];
        assert!(!i420_to_xrgb(&short, 2, 2, &mut dst, 8));
        // Destination too small.
        let src = vec![0u8; 2 * 2 + 2];
        let mut tiny = vec![0u8; 4];
        assert!(!i420_to_xrgb(&src, 2, 2, &mut tiny, 8));
        // Stride narrower than a row.
        assert!(!i420_to_xrgb(&src, 2, 2, &mut dst, 4));
    }

    #[test]
    fn gray8_replicates_luma_across_channels() {
        let src = [0u8, 128, 255, 64];
        let mut dst = vec![0u8; 4 * 4];
        assert!(gray8_to_xrgb(&src, 4, 1, &mut dst, 16));
        for (i, &lum) in src.iter().enumerate() {
            let px = &dst[i * 4..i * 4 + 4];
            assert_eq!(px, &[lum, lum, lum, 0xFF]);
        }
    }

    #[test]
    fn wide_dst_stride_leaves_row_padding_untouched_between_rows() {
        // dst_stride wider than width*4: conversion writes only width*4 per row.
        let (w, h) = (2usize, 2usize);
        let src = vec![16u8; w * h + 2]; // black-ish frame
        let stride = w * 4 + 8; // 8 bytes of padding per row
        let mut dst = vec![0x77u8; stride * h];
        assert!(i420_to_xrgb(&src, w, h, &mut dst, stride));
        // The padding bytes at the end of each row are left as-is (0x77).
        for row in 0..h {
            let pad = &dst[row * stride + w * 4..row * stride + stride];
            assert!(pad.iter().all(|&b| b == 0x77), "row {row} padding untouched");
        }
    }
}
