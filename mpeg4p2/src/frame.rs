//! A planar 4:2:0 picture buffer and half-pel motion compensation (ISO/IEC
//! 14496-2 §7.5 "Motion compensation decoding" and §7.6 "Interpolation for
//! sub-pixel accuracy"). The picture is stored MB-aligned (width/height rounded up
//! to a multiple of 16) with unpadded planes; motion vectors that point outside
//! the picture are clamped by edge extension in the sample fetch, which is the
//! §7.6.2 "unrestricted motion vector" edge-emulation behaviour the target file
//! relies on.

/// A decoded 4:2:0 planar picture. Planes are MB-aligned: `stride`/`cheight`
/// carry the padded geometry so motion compensation from an aligned reference is
/// exact; the element crops to the display `width`/`height` on output.
#[derive(Clone)]
pub struct Picture {
    /// Display (cropped) luma dimensions.
    pub width: usize,
    pub height: usize,
    /// MB-aligned (padded) luma dimensions == plane strides.
    pub lstride: usize,
    pub lheight: usize,
    pub cstride: usize,
    pub cheight: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl Picture {
    /// Allocate an MB-aligned picture for `width`×`height` display dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        let lstride = width.div_ceil(16) * 16;
        let lheight = height.div_ceil(16) * 16;
        let cstride = lstride / 2;
        let cheight = lheight / 2;
        Picture {
            width,
            height,
            lstride,
            lheight,
            cstride,
            cheight,
            y: vec![0; lstride * lheight],
            u: vec![128; cstride * cheight],
            v: vec![128; cstride * cheight],
        }
    }

    /// Extend the picture's borders by replicating edge samples (§7.6.2
    /// unrestricted-MV edge emulation). A 16-pixel margin covers the maximum
    /// half-pel motion reach for the fcodes this decoder handles; but since our
    /// sample fetch clamps coordinates itself, this is a no-op safety hook kept
    /// for clarity. (Left intentionally empty — clamping in `fetch_*` handles it.)
    pub fn pad_edges(&mut self) {}
}

/// Clamp a coordinate to `[0, max-1]` for edge-emulated sample fetch.
#[inline]
fn clampc(v: isize, max: usize) -> usize {
    if v < 0 {
        0
    } else if v as usize >= max {
        max - 1
    } else {
        v as usize
    }
}

/// Fetch one luma sample with edge clamping.
#[inline]
fn luma_at(p: &Picture, x: isize, y: isize) -> i32 {
    let xc = clampc(x, p.lstride);
    let yc = clampc(y, p.lheight);
    p.y[yc * p.lstride + xc] as i32
}

#[inline]
fn chroma_at(plane: &[u8], stride: usize, cheight: usize, x: isize, y: isize) -> i32 {
    let xc = clampc(x, stride);
    let yc = clampc(y, cheight);
    plane[yc * stride + xc] as i32
}

/// Motion-compensate one 8×8 (or 16×16 tiled as four 8×8) luma block from `refp`
/// into `dst` (an 8×8 scratch, row-major), given the integer + half-pel motion
/// vector. `mvx`/`mvy` are in half-pel units. `rounding` is `vop_rounding_type`
/// (§7.6.2: 0 rounds up (+1), 1 rounds down). Bilinear half-pel interpolation
/// (§7.6.2).
#[allow(clippy::too_many_arguments)]
pub fn mc_luma_block(
    dst: &mut [i32; 64],
    refp: &Picture,
    bx: usize,
    by: usize,
    mvx: i32,
    mvy: i32,
    rounding: u32,
) {
    let ix = (mvx >> 1) as isize;
    let iy = (mvy >> 1) as isize;
    let hx = (mvx & 1) != 0;
    let hy = (mvy & 1) != 0;
    let round = 1 - rounding as i32; // +1 when rounding_type==0
    for r in 0..8 {
        for c in 0..8 {
            let x = bx as isize + c as isize + ix;
            let y = by as isize + r as isize + iy;
            let v = match (hx, hy) {
                (false, false) => luma_at(refp, x, y),
                (true, false) => (luma_at(refp, x, y) + luma_at(refp, x + 1, y) + round) >> 1,
                (false, true) => (luma_at(refp, x, y) + luma_at(refp, x, y + 1) + round) >> 1,
                (true, true) => {
                    (luma_at(refp, x, y)
                        + luma_at(refp, x + 1, y)
                        + luma_at(refp, x, y + 1)
                        + luma_at(refp, x + 1, y + 1)
                        + round
                        + 1)
                        >> 2
                }
            };
            dst[r * 8 + c] = v;
        }
    }
}

/// Motion-compensate an 8×8 chroma block from a chroma plane. `mvx`/`mvy` are the
/// *chroma* half-pel motion vector (already derived from the luma MV per §7.6.2:
/// chroma MV = round(luma MV / 2)).
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma_block(
    dst: &mut [i32; 64],
    plane: &[u8],
    stride: usize,
    cheight: usize,
    bx: usize,
    by: usize,
    mvx: i32,
    mvy: i32,
    rounding: u32,
) {
    let ix = (mvx >> 1) as isize;
    let iy = (mvy >> 1) as isize;
    let hx = (mvx & 1) != 0;
    let hy = (mvy & 1) != 0;
    let round = 1 - rounding as i32;
    for r in 0..8 {
        for c in 0..8 {
            let x = bx as isize + c as isize + ix;
            let y = by as isize + r as isize + iy;
            let at = |xx: isize, yy: isize| chroma_at(plane, stride, cheight, xx, yy);
            let v = match (hx, hy) {
                (false, false) => at(x, y),
                (true, false) => (at(x, y) + at(x + 1, y) + round) >> 1,
                (false, true) => (at(x, y) + at(x, y + 1) + round) >> 1,
                (true, true) => (at(x, y) + at(x + 1, y) + at(x, y + 1) + at(x + 1, y + 1) + round + 1) >> 2,
            };
            dst[r * 8 + c] = v;
        }
    }
}

/// Derive the chroma motion vector from a luma MV (§7.6.2). For 4:2:0 the chroma
/// MV is the luma MV summed component-wise then rounded toward the nearest even
/// half-pel with a lookup-based rounding. XviD/ISO uses the "round toward zero of
/// half" table: chroma = (lumasum) with a specific rounding. Here we implement the
/// standard 4-MV averaging path via `chroma_mv_from_sum`.
pub fn round_chroma_mv(sum: i32) -> i32 {
    // §7.6.2: cmv = sign(sum) * (roundtab[|sum| % 16] + (|sum|/16)*2 ... )
    // The reference rounding table maps the sum of the (up to four) luma MVs to a
    // chroma MV. For a single-MV MB this is just the luma MV halved with rounding
    // toward the nearest half-pel using the table below.
    // roundtab16[k] for k in 0..16 (ISO §7.6.2 Table): the chroma-MV rounding.
    const ROUNDTAB: [i32; 16] = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2];
    let s = sum;
    let a = s.unsigned_abs() as usize;
    let base = (a / 16) * 2;
    let frac = ROUNDTAB[a % 16];
    let mag = base as i32 + frac;
    if s < 0 {
        -mag
    } else {
        mag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_mv_copies_samples() {
        let mut p = Picture::new(16, 16);
        for (i, b) in p.y.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let mut dst = [0i32; 64];
        // MV (2,0) half-pel == integer +1 in x
        mc_luma_block(&mut dst, &p, 0, 0, 2, 0, 0);
        for r in 0..8 {
            for c in 0..8 {
                let expect = p.y[r * p.lstride + (c + 1)] as i32;
                assert_eq!(dst[r * 8 + c], expect);
            }
        }
    }

    #[test]
    fn halfpel_x_is_average() {
        let mut p = Picture::new(16, 16);
        p.y.fill(0);
        p.y[0] = 10;
        p.y[1] = 20;
        let mut dst = [0i32; 64];
        // MV (1,0): half-pel in x, rounding 0 → (10+20+1)>>1 = 15
        mc_luma_block(&mut dst, &p, 0, 0, 1, 0, 0);
        assert_eq!(dst[0], 15);
    }

    #[test]
    fn round_chroma_mv_symmetry() {
        assert_eq!(round_chroma_mv(0), 0);
        assert_eq!(round_chroma_mv(2), 0);
        assert_eq!(round_chroma_mv(-2), 0);
        assert_eq!(round_chroma_mv(16), 2);
        assert_eq!(round_chroma_mv(-16), -2);
    }
}
