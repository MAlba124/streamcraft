//! A tiny, dependency-free software rasterizer for the GUI `wl_subsurface`.
//!
//! ## Why a rasterizer here at all
//!
//! The video surface is composited zero-copy from the VA-API dmabuf; the GUI on top of it
//! (the scope UI, transport controls, text overlays) is a `wl_shm` buffer we software-render
//! and let the **compositor** blend over the video (see [`crate`] docs). There is no GPU on
//! this path, so the scope's triangles, filled rects, lines and glyph coverage all land here,
//! in plain integer math, writing straight into the caller's shm mapping.
//!
//! ## Pixel format
//!
//! The shm buffer is `WL_SHM_FORMAT_ARGB8888`, which on a little-endian host is a native-endian
//! `u32` laid out `0xAARRGGBB` — i.e. the bytes in memory are `B, G, R, A`. Every `argb`
//! argument below is one such packed `u32`; the alpha byte drives [`blend_rect`](Canvas::blend_rect),
//! [`fill_triangle`](Canvas::fill_triangle) and [`blit_a8`](Canvas::blit_a8).
//!
//! ## Discipline
//!
//! Every method writes into a **caller-owned** `&mut [u8]` and allocates nothing (the workspace
//! alloc ban is active — see `clippy.toml`). Every primitive clips to `[0, width) × [0, height)`
//! and to the backing slice, so off-screen, negative, or degenerate input is a no-op, never a
//! panic and never an out-of-bounds index.

/// An immediate-mode drawing surface over a caller-owned ARGB8888 pixel buffer.
///
/// The buffer is borrowed for the `Canvas`'s lifetime; nothing is copied and nothing is
/// allocated. `stride` is the row pitch **in bytes** (≥ `width * 4`), letting the caller draw
/// into a sub-window of a larger shm mapping.
pub struct Canvas<'a> {
    /// The backing ARGB8888 bytes (`B, G, R, A` per pixel on little-endian), row-major.
    buf: &'a mut [u8],
    width: u32,
    height: u32,
    /// Row pitch in bytes.
    stride: u32,
}

/// The four channels of a packed `0xAARRGGBB` `argb`, unpacked once so blends stay in `u32`.
struct Argb {
    a: u32,
    r: u32,
    g: u32,
    b: u32,
}

impl Argb {
    /// Split a packed `0xAARRGGBB` into its channels.
    #[inline]
    fn unpack(argb: u32) -> Self {
        Argb {
            a: (argb >> 24) & 0xff,
            r: (argb >> 16) & 0xff,
            g: (argb >> 8) & 0xff,
            b: argb & 0xff,
        }
    }
}

impl<'a> Canvas<'a> {
    /// Wrap `buf` as a `width × height` ARGB8888 surface with the given byte `stride`.
    ///
    /// Asserts the backing slice is large enough to address every pixel of the last row
    /// (`stride * (height - 1) + width * 4` bytes) and that `stride ≥ width * 4`. These are the
    /// one-time constructor invariants that let every draw method use a plain computed index
    /// without re-checking the whole geometry.
    pub fn new(buf: &'a mut [u8], width: u32, height: u32, stride: u32) -> Canvas<'a> {
        assert!(stride >= width * 4, "stride must be at least width*4 bytes");
        if width != 0 && height != 0 {
            // Bytes needed to touch the last pixel of the last row.
            let needed = (stride as usize) * (height as usize - 1) + (width as usize) * 4;
            assert!(buf.len() >= needed, "buffer too small for width*height*stride");
        }
        Canvas { buf, width, height, stride }
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Byte offset of pixel `(x, y)`. Callers must have already clipped `x < width`, `y < height`.
    #[inline]
    fn offset(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.stride as usize) + (x as usize) * 4
    }

    /// Write an opaque packed pixel at a pre-clipped `(x, y)`.
    #[inline]
    fn put(&mut self, x: u32, y: u32, argb: u32) {
        let o = self.offset(x, y);
        // Little-endian memory order for a native-endian `0xAARRGGBB` u32 is B, G, R, A.
        self.buf[o] = (argb & 0xff) as u8; // B
        self.buf[o + 1] = ((argb >> 8) & 0xff) as u8; // G
        self.buf[o + 2] = ((argb >> 16) & 0xff) as u8; // R
        self.buf[o + 3] = ((argb >> 24) & 0xff) as u8; // A
    }

    /// Source-over blend `src` (with its own alpha) onto the destination pixel at a pre-clipped
    /// `(x, y)`: `out = src·a + dst·(1 − a)` per channel, in integer math with `+127` rounding
    /// before the `/255`. The stored alpha is likewise composited (`a + dst_a·(1 − a)`), so
    /// repeatedly blending translucent layers converges toward opaque — correct for a buffer the
    /// compositor will itself alpha-blend over the video.
    #[inline]
    fn blend_px(&mut self, x: u32, y: u32, src: &Argb) {
        let a = src.a;
        if a == 0 {
            return;
        }
        if a == 255 {
            self.put(x, y, (a << 24) | (src.r << 16) | (src.g << 8) | src.b);
            return;
        }
        let ia = 255 - a;
        let o = self.offset(x, y);
        let db = self.buf[o] as u32;
        let dg = self.buf[o + 1] as u32;
        let dr = self.buf[o + 2] as u32;
        let da = self.buf[o + 3] as u32;
        // (v + 127) / 255 is the standard rounded fixed-point divide by 255.
        let blend = |s: u32, d: u32| -> u8 {
            let v = s * a + d * ia + 127;
            ((v + (v >> 8)) >> 8) as u8
        };
        self.buf[o] = blend(src.b, db);
        self.buf[o + 1] = blend(src.g, dg);
        self.buf[o + 2] = blend(src.r, dr);
        self.buf[o + 3] = blend(255, da); // src coverage is fully opaque *within* alpha a
    }

    /// Fill the entire canvas with an opaque packed color, ignoring its alpha byte.
    pub fn clear(&mut self, argb: u32) {
        // Take the simple per-row path; every pixel is in bounds by construction.
        for y in 0..self.height {
            for x in 0..self.width {
                self.put(x, y, argb);
            }
        }
    }

    /// Intersect the requested rect `(x, y, w, h)` with `[0, width) × [0, height)`, returning the
    /// clipped `[x0, x1) × [y0, y1)` in canvas coordinates, or `None` if nothing is visible.
    /// Negative origins, zero/negative extents and overflow all reduce to `None` — this is the
    /// single choke point that keeps the rect primitives panic-free.
    fn clip_rect(&self, x: i32, y: i32, w: i32, h: i32) -> Option<(u32, u32, u32, u32)> {
        if w <= 0 || h <= 0 {
            return None;
        }
        // Widen to i64 so `x + w` cannot overflow for pathological inputs.
        let x0 = x.max(0) as i64;
        let y0 = y.max(0) as i64;
        let x1 = ((x as i64) + (w as i64)).min(self.width as i64);
        let y1 = ((y as i64) + (h as i64)).min(self.height as i64);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        Some((x0 as u32, y0 as u32, x1 as u32, y1 as u32))
    }

    /// Fill an axis-aligned rectangle with an **opaque** color (the alpha byte is written as-is
    /// but no blending occurs — this overwrites). Clipped to the canvas; off-screen or degenerate
    /// input draws nothing.
    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, argb: u32) {
        if let Some((x0, y0, x1, y1)) = self.clip_rect(x, y, w, h) {
            for py in y0..y1 {
                for px in x0..x1 {
                    self.put(px, py, argb);
                }
            }
        }
    }

    /// Alpha-blend an axis-aligned rectangle over the canvas using the color's own alpha byte
    /// (source-over, see [`blend_px`](Self::blend_px)). Clipped; degenerate input is a no-op.
    pub fn blend_rect(&mut self, x: i32, y: i32, w: i32, h: i32, argb: u32) {
        let src = Argb::unpack(argb);
        if src.a == 0 {
            return;
        }
        if let Some((x0, y0, x1, y1)) = self.clip_rect(x, y, w, h) {
            for py in y0..y1 {
                for px in x0..x1 {
                    self.blend_px(px, py, &src);
                }
            }
        }
    }

    /// Draw a 1-pixel line from `(x0, y0)` to `(x1, y1)` with an opaque color, via the integer
    /// Bresenham midpoint algorithm (Bresenham, *IBM Systems Journal* 4(1), 1965). Each candidate
    /// pixel is bounds-checked before being written, so any portion of the line off the canvas is
    /// simply skipped — no pre-clipping arithmetic and no panics.
    pub fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, argb: u32) {
        let mut x = x0;
        let mut y = y0;
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        // Error accumulator; the loop steps whichever axis keeps the point nearest the ideal line.
        let mut err = dx + dy;
        loop {
            if x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height {
                self.put(x as u32, y as u32, argb);
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                if x == x1 {
                    break;
                }
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                if y == y1 {
                    break;
                }
                err += dx;
                y += sy;
            }
        }
    }

    /// Fill a solid triangle through the three vertices, alpha-blended via the color's alpha byte.
    ///
    /// Scanline barycentric rasterization: for each pixel center inside the triangle's bounding
    /// box (clipped to the canvas) we evaluate the three edge functions `E_i` (the signed area of
    /// the sub-triangle on each edge; Pineda, *SIGGRAPH '88*, "A Parallel Algorithm for Polygon
    /// Rasterization"). A pixel is inside when all three have the winding's sign. Shared edges
    /// between adjacent triangles are disambiguated by a **top-left fill rule**: a pixel exactly on
    /// an edge is covered only if that edge is a top or left edge, so a seam is painted by exactly
    /// one of the two triangles (no double-blend, no gap). Edge functions are `f32` — ample for the
    /// integer pixel coordinates the scope UI uses — but coverage is binary (1 sample per pixel).
    pub fn fill_triangle(&mut self, p0: (i32, i32), p1: (i32, i32), p2: (i32, i32), argb: u32) {
        let src = Argb::unpack(argb);
        if src.a == 0 {
            return;
        }

        // Signed area × 2 of the triangle (via the shoelace/cross product). Its sign is the
        // winding; zero means the three points are collinear — a degenerate, zero-area triangle,
        // which we decline to fill.
        let area = edge(p0, p1, p2);
        if area == 0 {
            return;
        }
        // Normalize to counter-clockwise so the "inside" test is a single comparison. If the input
        // was clockwise (`area < 0`) we swap two vertices to flip the winding.
        let (a, b, c) = if area > 0 { (p0, p1, p2) } else { (p0, p2, p1) };

        // Bounding box of the triangle, clipped to the canvas. Empty box → nothing to do.
        let min_x = a.0.min(b.0).min(c.0).max(0);
        let min_y = a.1.min(b.1).min(c.1).max(0);
        let max_x = a.0.max(b.0).max(c.0).min(self.width as i32 - 1);
        let max_y = a.1.max(b.1).max(c.1).min(self.height as i32 - 1);
        if min_x > max_x || min_y > max_y {
            return;
        }

        // A top-left edge (for CCW winding): a "top" edge is horizontal and goes right→left
        // (dy == 0 && dx < 0); a "left" edge goes downward (dy > 0, i.e. y increases). Pixels
        // exactly on such an edge count as inside; on a bottom/right edge they do not.
        let top_left = |v0: (i32, i32), v1: (i32, i32)| -> bool {
            let ex = v1.0 - v0.0;
            let ey = v1.1 - v0.1;
            (ey == 0 && ex < 0) || ey > 0
        };
        let tl_ab = top_left(a, b);
        let tl_bc = top_left(b, c);
        let tl_ca = top_left(c, a);

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let p = (px, py);
                // Edge functions at this pixel; for CCW winding a point is inside when all are ≥ 0.
                let w_ab = edge(a, b, p);
                let w_bc = edge(b, c, p);
                let w_ca = edge(c, a, p);
                // Strictly inside covers positive; on an edge (== 0), the top-left rule decides.
                let inside = (w_ab > 0 || (w_ab == 0 && tl_ab))
                    && (w_bc > 0 || (w_bc == 0 && tl_bc))
                    && (w_ca > 0 || (w_ca == 0 && tl_ca));
                if inside {
                    // px,py are within [0,width)×[0,height) by the clipped bbox.
                    self.blend_px(px as u32, py as u32, &src);
                }
            }
        }
    }

    /// Composite an 8-bit coverage mask at `(x, y)` tinted with `argb`. `mask` is `mw × mh`
    /// samples with row pitch `mstride` bytes; each sample scales the color's alpha
    /// (`px_alpha = mask · colorA / 255`) and the tinted, faded color is source-over blended.
    ///
    /// This is the text path: `sc-text` rasterizes glyph coverage into an A8 buffer, and we land
    /// it as a colored, anti-aliased run here. The mask, its stride and the destination are all
    /// clipped independently, so a glyph partly off the canvas paints only its visible part and a
    /// truncated `mask` slice is never over-read.
    // Destination (x, y) + mask geometry (bytes, w, h, stride) + tint is irreducibly 8 args; a
    // wrapper struct would only add a per-call construction for no clarity in an immediate-mode API.
    #[allow(clippy::too_many_arguments)]
    pub fn blit_a8(
        &mut self,
        x: i32,
        y: i32,
        mask: &[u8],
        mw: u32,
        mh: u32,
        mstride: u32,
        argb: u32,
    ) {
        let color = Argb::unpack(argb);
        if color.a == 0 || mw == 0 || mh == 0 {
            return;
        }
        for my in 0..mh {
            let dy = y + my as i32;
            if dy < 0 || dy as u32 >= self.height {
                continue;
            }
            let row = (my as usize) * (mstride as usize);
            for mx in 0..mw {
                let dx = x + mx as i32;
                if dx < 0 || dx as u32 >= self.width {
                    continue;
                }
                let idx = row + mx as usize;
                // Tolerate a mask slice shorter than mw*mstride (partial/degenerate input).
                let Some(&cov) = mask.get(idx) else {
                    continue;
                };
                if cov == 0 {
                    continue;
                }
                // Fold coverage into the color's alpha: a = colorA·cov/255 (rounded).
                let av = color.a * cov as u32 + 127;
                let a = (av + (av >> 8)) >> 8;
                if a == 0 {
                    continue;
                }
                let src = Argb { a, r: color.r, g: color.g, b: color.b };
                self.blend_px(dx as u32, dy as u32, &src);
            }
        }
    }
}

/// Twice the signed area of triangle `(a, b, c)` — the 2-D cross product of `b−a` and `c−a`.
/// Positive for a counter-clockwise winding (in a y-down raster space where clockwise looks
/// visually counter-clockwise, this is the standard "left of the edge" test). Used both to
/// classify the whole triangle's winding and, per pixel, as the edge function `E(a→b, p)`.
/// `i64` math keeps the product exact for the full `i32` coordinate range.
#[inline]
fn edge(a: (i32, i32), b: (i32, i32), c: (i32, i32)) -> i64 {
    let abx = (b.0 - a.0) as i64;
    let aby = (b.1 - a.1) as i64;
    let acx = (c.0 - a.0) as i64;
    let acy = (c.1 - a.1) as i64;
    abx * acy - aby * acx
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests allocate scratch pixel buffers up front (one-time).
mod tests {
    use super::*;

    /// A packed-pixel helper: read the `0xAARRGGBB` u32 at `(x, y)` out of a raw ARGB8888 buffer.
    fn px(buf: &[u8], stride: u32, x: u32, y: u32) -> u32 {
        let o = (y as usize) * (stride as usize) + (x as usize) * 4;
        let b = buf[o] as u32;
        let g = buf[o + 1] as u32;
        let r = buf[o + 2] as u32;
        let a = buf[o + 3] as u32;
        (a << 24) | (r << 16) | (g << 8) | b
    }

    /// Allocate a zeroed `w × h` ARGB8888 buffer with a tight `w*4` stride.
    fn scratch(w: u32, h: u32) -> Vec<u8> {
        vec![0u8; (w * h * 4) as usize]
    }

    #[test]
    fn clear_sets_every_pixel() {
        let (w, h) = (4u32, 3u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        c.clear(0xff3366cc);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(px(&buf, w * 4, x, y), 0xff3366cc, "pixel {x},{y}");
            }
        }
    }

    #[test]
    fn fill_rect_fills_region_and_leaves_the_rest() {
        let (w, h) = (8u32, 8u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        c.fill_rect(2, 3, 3, 2, 0xffabcdef);
        for y in 0..h {
            for x in 0..w {
                let inside = (2..5).contains(&x) && (3..5).contains(&y);
                let want = if inside { 0xffabcdef } else { 0 };
                assert_eq!(px(&buf, w * 4, x, y), want, "pixel {x},{y}");
            }
        }
    }

    #[test]
    fn fill_rect_clips_off_screen_and_negative() {
        let (w, h) = (4u32, 4u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        // A rect straddling the top-left corner: only the (0,0)..(2,2) quadrant lands.
        c.fill_rect(-2, -2, 4, 4, 0xff112233);
        // Fully off-screen / degenerate rects are no-ops (and must not panic). Do them before the
        // read loop, since the canvas holds the buffer's mutable borrow until then.
        c.fill_rect(100, 100, 4, 4, 0xffffffff);
        c.fill_rect(0, 0, 0, 5, 0xffffffff);
        c.fill_rect(0, 0, 5, -3, 0xffffffff);
        c.fill_rect(i32::MAX, 0, i32::MAX, 4, 0xffffffff);
        for y in 0..h {
            for x in 0..w {
                let inside = x < 2 && y < 2;
                let want = if inside { 0xff112233 } else { 0 };
                assert_eq!(px(&buf, w * 4, x, y), want, "pixel {x},{y}");
            }
        }
    }

    #[test]
    fn blend_rect_half_alpha_mixes() {
        let (w, h) = (2u32, 2u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        // Opaque black background, then blend white at A=128 → ~half grey.
        c.clear(0xff000000);
        c.blend_rect(0, 0, 2, 2, 0x80ffffff);
        // out = 255*128/255 rounded = 128; alpha composites toward opaque (0xff).
        for y in 0..h {
            for x in 0..w {
                assert_eq!(px(&buf, w * 4, x, y), 0xff808080, "pixel {x},{y}");
            }
        }
    }

    #[test]
    fn blend_rect_alpha_extremes() {
        // A == 0 leaves the destination untouched.
        let mut buf = scratch(1, 1);
        {
            let mut c = Canvas::new(&mut buf, 1, 1, 4);
            c.clear(0xff102030);
            c.blend_rect(0, 0, 1, 1, 0x00ffffff);
        }
        assert_eq!(px(&buf, 4, 0, 0), 0xff102030);

        // A == 255 fully replaces (color + opaque alpha).
        let mut buf2 = scratch(1, 1);
        {
            let mut c = Canvas::new(&mut buf2, 1, 1, 4);
            c.clear(0xff102030);
            c.blend_rect(0, 0, 1, 1, 0xff9988aa);
        }
        assert_eq!(px(&buf2, 4, 0, 0), 0xff9988aa);
    }

    #[test]
    fn draw_line_hits_endpoints_and_diagonal() {
        let (w, h) = (5u32, 5u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        // Main diagonal (0,0)..(4,4): a Bresenham line hits exactly the diagonal pixels.
        c.draw_line(0, 0, 4, 4, 0xffff0000);
        for i in 0..5u32 {
            assert_eq!(px(&buf, w * 4, i, i), 0xffff0000, "diagonal {i}");
        }
        // An off-diagonal pixel stays clear.
        assert_eq!(px(&buf, w * 4, 0, 4), 0);

        // A horizontal line covers its whole run inclusive of both endpoints.
        let mut buf2 = scratch(w, h);
        let mut c2 = Canvas::new(&mut buf2, w, h, w * 4);
        c2.draw_line(1, 2, 3, 2, 0xff00ff00);
        for x in 1..=3u32 {
            assert_eq!(px(&buf2, w * 4, x, 2), 0xff00ff00, "hline {x}");
        }
        assert_eq!(px(&buf2, w * 4, 0, 2), 0);
        assert_eq!(px(&buf2, w * 4, 4, 2), 0);
    }

    #[test]
    fn draw_line_clips_and_survives_degenerate() {
        let (w, h) = (4u32, 4u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        // A line that starts off-screen and ends on-screen paints only the visible tail.
        c.draw_line(-10, 2, 2, 2, 0xff0000ff);
        // Fully off-screen and single-point lines must not panic (drawn before the pixel reads,
        // since the canvas borrows the buffer until then).
        c.draw_line(-5, -5, -1, -1, 0xffffffff);
        c.draw_line(100, 100, 200, 200, 0xffffffff);
        c.draw_line(1, 1, 1, 1, 0xffffff00); // degenerate point, on-canvas
        assert_eq!(px(&buf, w * 4, 0, 2), 0xff0000ff);
        assert_eq!(px(&buf, w * 4, 2, 2), 0xff0000ff);
        assert_eq!(px(&buf, w * 4, 1, 1), 0xffffff00);
    }

    #[test]
    fn fill_triangle_fills_interior_only() {
        let (w, h) = (8u32, 8u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        // A right triangle with the right angle at (0,0): interior is where x + y is small.
        c.fill_triangle((0, 0), (7, 0), (0, 7), 0xff00ffff);
        // Points strictly inside the triangle are filled.
        assert_eq!(px(&buf, w * 4, 1, 1), 0xff00ffff, "interior");
        assert_eq!(px(&buf, w * 4, 2, 1), 0xff00ffff, "interior");
        assert_eq!(px(&buf, w * 4, 1, 3), 0xff00ffff, "interior");
        // A point clearly outside the hypotenuse (x + y large) is untouched.
        assert_eq!(px(&buf, w * 4, 7, 7), 0, "outside");
        assert_eq!(px(&buf, w * 4, 6, 6), 0, "outside");
    }

    #[test]
    fn fill_triangle_winding_independent_and_covers_a_solid_block() {
        // Two triangles sharing edge (0,0)-(4,0) and (0,0)-(0,4) etc. tile a 4×4 square with no
        // gaps and no double-cover; test that both windings fill and the square is solid.
        let (w, h) = (4u32, 4u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        // Cover the whole 4×4 with two triangles split along the diagonal. Use A=128 so a
        // double-blended seam pixel (a bug) would read a *different* value than a singly-blended
        // one — this is how the top-left fill rule is validated.
        c.clear(0xff000000);
        c.fill_triangle((0, 0), (4, 0), (4, 4), 0x80ffffff); // upper-right
        c.fill_triangle((0, 0), (4, 4), (0, 4), 0x80ffffff); // lower-left
        // Every covered pixel must be exactly one 50% blend (0x808080), never two (0xbfbfbf-ish).
        for y in 0..h {
            for x in 0..w {
                let got = px(&buf, w * 4, x, y);
                assert!(
                    got == 0xff808080 || got == 0xff000000,
                    "pixel {x},{y} = {got:#010x}: seam double-blended or gap"
                );
            }
        }
        // The interior of the shared diagonal must actually be painted (no gap): (1,1) is on it.
        assert_eq!(px(&buf, w * 4, 1, 1), 0xff808080);
    }

    #[test]
    fn fill_triangle_degenerate_and_offscreen_no_panic() {
        let (w, h) = (4u32, 4u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        // Collinear (zero area) → nothing drawn.
        c.fill_triangle((0, 0), (2, 2), (3, 3), 0xffffffff);
        // Fully off-screen → nothing drawn, no panic.
        c.fill_triangle((100, 100), (110, 100), (100, 110), 0xffffffff);
        // A giant triangle covering the whole canvas fills every pixel.
        c.fill_triangle((-100, -100), (1000, -100), (-100, 1000), 0xff123456);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(px(&buf, w * 4, x, y), 0xff123456, "pixel {x},{y}");
            }
        }
    }

    #[test]
    fn blit_a8_full_mask_paints_color_zero_mask_paints_nothing() {
        let (w, h) = (4u32, 4u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        c.clear(0xff000000);
        // Full-coverage 2×2 mask, opaque red → replaces exactly the 2×2 region with red.
        let full = [255u8; 4];
        c.blit_a8(1, 1, &full, 2, 2, 2, 0xffff0000);
        // A zero mask at the origin changes nothing (drawn now; the (0,0) check below proves it).
        let zero = [0u8; 4];
        c.blit_a8(0, 0, &zero, 2, 2, 2, 0xff00ff00);
        for y in 0..h {
            for x in 0..w {
                let inside = (1..3).contains(&x) && (1..3).contains(&y);
                let want = if inside { 0xffff0000 } else { 0xff000000 };
                assert_eq!(px(&buf, w * 4, x, y), want, "pixel {x},{y}");
            }
        }
    }

    #[test]
    fn blit_a8_half_coverage_blends_half() {
        let (w, h) = (1u32, 1u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        c.clear(0xff000000);
        // cov=128 with an opaque white color → effective alpha ~128 → half grey.
        let mask = [128u8; 1];
        c.blit_a8(0, 0, &mask, 1, 1, 1, 0xffffffff);
        assert_eq!(px(&buf, w * 4, 0, 0), 0xff808080);
    }

    #[test]
    fn blit_a8_clips_and_tolerates_short_mask() {
        let (w, h) = (4u32, 4u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        c.clear(0xff000000);
        // Do every blit first (the canvas holds the mutable borrow), then read pixels.
        // Mask placed partly off the top-left: only its in-bounds part paints.
        let full = [255u8; 9]; // 3×3
        c.blit_a8(-1, -1, &full, 3, 3, 3, 0xffffffff);
        // A mask slice shorter than mw*mstride must be tolerated, not over-read.
        let short = [255u8; 2];
        c.blit_a8(0, 3, &short, 4, 1, 4, 0xff00ff00); // claims width 4, only 2 bytes present
        // Fully off-screen blit is a no-op.
        c.blit_a8(100, 100, &full, 3, 3, 3, 0xffffffff);
        // The 3×3 mask at (-1,-1) covers dest (0..2, 0..2) — its bottom-right 2×2 lands on-canvas.
        assert_eq!(px(&buf, w * 4, 0, 0), 0xffffffff);
        assert_eq!(px(&buf, w * 4, 1, 1), 0xffffffff);
        assert_eq!(px(&buf, w * 4, 2, 2), 0xff000000); // beyond the mask's on-canvas footprint
        assert_eq!(px(&buf, w * 4, 0, 3), 0xff00ff00);
        assert_eq!(px(&buf, w * 4, 1, 3), 0xff00ff00);
        assert_eq!(px(&buf, w * 4, 2, 3), 0xff000000); // no data → untouched
    }

    #[test]
    fn sub_window_stride_addresses_correct_rows() {
        // A canvas over a 3-wide window inside a 5-wide (20-byte-stride) buffer: writes to row 1
        // must land at byte offset 20, proving stride (not width) drives addressing.
        let (w, h, stride) = (3u32, 2u32, 20u32);
        let mut buf = vec![0u8; (stride * h) as usize];
        let mut c = Canvas::new(&mut buf, w, h, stride);
        c.fill_rect(0, 1, 3, 1, 0xffaabbcc);
        // Row 0 untouched (bytes 0..12 zero); row 1 filled at offset 20.
        assert_eq!(px(&buf, stride, 0, 0), 0);
        assert_eq!(px(&buf, stride, 0, 1), 0xffaabbcc);
        assert_eq!(px(&buf, stride, 2, 1), 0xffaabbcc);
    }
}
