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

    /// Source-over blend `src` at a pre-clipped `(x, y)` scaled by a fractional `cov` in
    /// `[0.0, 1.0]` — the anti-aliasing primitive. Coverage folds into the source alpha
    /// (`a' = src.a · cov`, rounded), so an edge pixel that a shape covers 40% blends as if the
    /// color's alpha were 40% of nominal. `cov ≤ 0` is a no-op; `cov ≥ 1` is the plain full blend.
    /// This keeps every AA path (triangle edges, Wu lines, circle rims) on the same rounded
    /// integer source-over as the solid primitives — no separate premultiplied bookkeeping.
    #[inline]
    fn blend_px_cov(&mut self, x: u32, y: u32, color: &Argb, cov: f32) {
        if cov <= 0.0 {
            return;
        }
        if cov >= 1.0 {
            self.blend_px(x, y, color);
            return;
        }
        // a' = round(color.a * cov). color.a ≤ 255 and cov < 1, so this stays in [0, 254].
        let a = (color.a as f32 * cov + 0.5) as u32;
        if a == 0 {
            return;
        }
        let src = Argb { a, r: color.r, g: color.g, b: color.b };
        self.blend_px(x, y, &src);
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

    /// Draw an anti-aliased 1-pixel line from `(x0, y0)` to `(x1, y1)` with the color's alpha, via
    /// **Xiaolin Wu's algorithm** (Wu, *SIGGRAPH '91*, "An Efficient Antialiasing Technique"). The
    /// line is walked along its major axis one step at a time; at each step the two pixels
    /// straddling the ideal line get complementary coverage from the fractional part of the minor
    /// coordinate, so a shallow diagonal fades between rows instead of stair-stepping. Endpoints are
    /// plotted with the classic Wu endpoint weighting (coverage tapered by the fractional overhang).
    /// Every candidate pixel is bounds-checked, so any portion off the canvas is simply skipped —
    /// no panics. A single point (`x0,y0 == x1,y1`) plots one full-coverage pixel.
    pub fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, argb: u32) {
        let color = Argb::unpack(argb);
        if color.a == 0 {
            return;
        }
        // Degenerate zero-length line: a single pixel, no fractional weighting.
        if x0 == x1 && y0 == y1 {
            if x0 >= 0 && y0 >= 0 && (x0 as u32) < self.width && (y0 as u32) < self.height {
                self.blend_px(x0 as u32, y0 as u32, &color);
            }
            return;
        }

        let mut fx0 = x0 as f32;
        let mut fy0 = y0 as f32;
        let mut fx1 = x1 as f32;
        let mut fy1 = y1 as f32;

        // Wu iterates along the axis of greater extent ("major axis"); if the line is steeper than
        // 45° we transpose x↔y so the inner loop always steps the major axis by one. `steep` is
        // remembered so plotting un-transposes the coordinates.
        let steep = (fy1 - fy0).abs() > (fx1 - fx0).abs();
        if steep {
            core::mem::swap(&mut fx0, &mut fy0);
            core::mem::swap(&mut fx1, &mut fy1);
        }
        // Walk left→right along the major axis.
        if fx0 > fx1 {
            core::mem::swap(&mut fx0, &mut fx1);
            core::mem::swap(&mut fy0, &mut fy1);
        }

        let dx = fx1 - fx0;
        let dy = fy1 - fy0;
        // dx > 0 here (equal-endpoint handled above, and dx ≥ |dy| ≥ 0 after the transpose).
        let gradient = dy / dx;

        // Plot at major coordinate `mx`, minor coordinate `my`, with coverage `cov`, honoring the
        // transpose. `mx`/`my` are the along-axis / across-axis pixel indices respectively.
        let plot = |canvas: &mut Self, mx: i32, my: i32, cov: f32| {
            let (px, py) = if steep { (my, mx) } else { (mx, my) };
            if px >= 0 && py >= 0 && (px as u32) < canvas.width && (py as u32) < canvas.height {
                canvas.blend_px_cov(px as u32, py as u32, &color, cov);
            }
        };

        // Fractional part and its complement (`rfpart`) — Wu's coverage weights.
        let fpart = |v: f32| v - v.floor();
        let rfpart = |v: f32| 1.0 - fpart(v);

        // --- Start endpoint ---
        let xend0 = fx0.round();
        let yend0 = fy0 + gradient * (xend0 - fx0);
        let xgap0 = rfpart(fx0 + 0.5);
        let xpxl0 = xend0 as i32;
        let ypxl0 = yend0.floor() as i32;
        plot(self, xpxl0, ypxl0, rfpart(yend0) * xgap0);
        plot(self, xpxl0, ypxl0 + 1, fpart(yend0) * xgap0);

        // --- End endpoint ---
        let xend1 = fx1.round();
        let yend1 = fy1 + gradient * (xend1 - fx1);
        let xgap1 = fpart(fx1 + 0.5);
        let xpxl1 = xend1 as i32;
        let ypxl1 = yend1.floor() as i32;
        plot(self, xpxl1, ypxl1, rfpart(yend1) * xgap1);
        plot(self, xpxl1, ypxl1 + 1, fpart(yend1) * xgap1);

        // --- Interior span ---
        // Track the ideal minor coordinate; the two nearest pixels split coverage by its fraction.
        let mut inter = yend0 + gradient;
        for mx in (xpxl0 + 1)..xpxl1 {
            let base = inter.floor() as i32;
            let f = inter - inter.floor();
            plot(self, mx, base, 1.0 - f);
            plot(self, mx, base + 1, f);
            inter += gradient;
        }
    }

    /// Fill an **anti-aliased** solid triangle through the three vertices, source-over blended via
    /// the color's alpha byte.
    ///
    /// Rasterization is Pineda's edge-function scan (Pineda, *SIGGRAPH '88*, "A Parallel Algorithm
    /// for Polygon Rasterization"): for each pixel center in the triangle's clipped bounding box we
    /// evaluate the three signed edge functions `E_i` in exact `i64` (range-safe for the full
    /// `i32` coordinate space). Instead of a binary inside test, each edge contributes **analytic
    /// coverage**: the perpendicular signed distance of the pixel center to the edge is
    /// `E_i / |edge_i|`, and that edge's coverage is `saturate(dist + 0.5)` — 1 well inside, 0 well
    /// outside, a linear ramp across the ~1-pixel band centered on the true edge. The pixel's
    /// coverage is the **product** of the three edge coverages, so a corner (near two edges) tapers
    /// correctly and interior pixels stay at coverage 1. That coverage scales the source-over blend
    /// (see [`blend_px_cov`](Self::blend_px_cov)), giving smooth slopes and diagonals with no MSAA
    /// samples and no allocation.
    ///
    /// **Seam note:** two AA triangles sharing an edge each paint that edge at partial coverage, so
    /// the seam blends twice (each side ~50% at the exact diagonal). This is inherent to
    /// coverage-AA without a coverage-accumulation buffer; interior fill remains exact.
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
        // Normalize to counter-clockwise so every edge function is ≥ 0 inside. If the input was
        // clockwise (`area < 0`) we swap two vertices to flip the winding.
        let (a, b, c) = if area > 0 { (p0, p1, p2) } else { (p0, p2, p1) };

        // Bounding box, expanded one pixel on every side so the coverage ramp of an edge that lies
        // just outside the integer bbox still gets sampled, then clipped to the canvas.
        let min_x = (a.0.min(b.0).min(c.0) - 1).max(0);
        let min_y = (a.1.min(b.1).min(c.1) - 1).max(0);
        let max_x = (a.0.max(b.0).max(c.0) + 1).min(self.width as i32 - 1);
        let max_y = (a.1.max(b.1).max(c.1) + 1).min(self.height as i32 - 1);
        if min_x > max_x || min_y > max_y {
            return;
        }

        // Reciprocal edge lengths convert an edge function (twice-signed-area of pixel vs edge)
        // into a perpendicular pixel distance: |E_i| / length_i. A zero-length edge can't happen
        // here (area != 0 ⇒ no two vertices coincide), but guard the divide anyway.
        let inv_len = |v0: (i32, i32), v1: (i32, i32)| -> f32 {
            let ex = (v1.0 - v0.0) as f32;
            let ey = (v1.1 - v0.1) as f32;
            let len = (ex * ex + ey * ey).sqrt();
            if len > 0.0 { 1.0 / len } else { 0.0 }
        };
        let il_ab = inv_len(a, b);
        let il_bc = inv_len(b, c);
        let il_ca = inv_len(c, a);

        // saturate(dist + 0.5): coverage of one edge for a pixel whose signed distance (positive
        // inside) to that edge is `dist`. The 0.5 offset centers the 1-px ramp on the true edge.
        let edge_cov = |e: i64, il: f32| -> f32 {
            (e as f32 * il + 0.5).clamp(0.0, 1.0)
        };

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let p = (px, py);
                // Exact i64 edge functions (positive inside for the CCW-normalized winding).
                let w_ab = edge(a, b, p);
                let w_bc = edge(b, c, p);
                let w_ca = edge(c, a, p);
                // Per-edge analytic coverage, multiplied for the pixel's total coverage.
                let cov = edge_cov(w_ab, il_ab) * edge_cov(w_bc, il_bc) * edge_cov(w_ca, il_ca);
                if cov > 0.0 {
                    // px,py are within [0,width)×[0,height) by the clipped bbox.
                    self.blend_px_cov(px as u32, py as u32, &src, cov);
                }
            }
        }
    }

    /// Fill an **anti-aliased** disc centered at `(cx, cy)` with radius `r`, source-over blended via
    /// the color's alpha byte. Coverage comes from the radial signed distance `r − dist(center)`:
    /// `saturate(r − dist + 0.5)` gives a 1-pixel-wide smooth rim, so GUI knobs and handles get the
    /// same soft edge as SDL's. Non-positive `r` and a zero-alpha color are no-ops; the sampled box
    /// is clipped to the canvas so an off-screen or partly-clipped circle never panics.
    pub fn fill_circle(&mut self, cx: f32, cy: f32, r: f32, argb: u32) {
        let src = Argb::unpack(argb);
        if src.a == 0 || !(r > 0.0) {
            return;
        }
        // Bounding box of the disc, padded a pixel for the rim ramp, clipped to the canvas.
        let min_x = ((cx - r - 1.0).floor() as i32).max(0);
        let min_y = ((cy - r - 1.0).floor() as i32).max(0);
        let max_x = ((cx + r + 1.0).ceil() as i32).min(self.width as i32 - 1);
        let max_y = ((cy + r + 1.0).ceil() as i32).min(self.height as i32 - 1);
        if min_x > max_x || min_y > max_y {
            return;
        }
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let dx = px as f32 + 0.5 - cx;
                let dy = py as f32 + 0.5 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                // Positive inside; the 0.5 offset centers the antialiased rim on radius r.
                let cov = (r - dist + 0.5).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend_px_cov(px as u32, py as u32, &src, cov);
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
        // Main diagonal (0,0)..(4,4): gradient == 1, so the *interior* diagonal pixels land at
        // full coverage; only the two endpoints are Wu-tapered (see the endpoint test below).
        c.draw_line(0, 0, 4, 4, 0xffff0000);
        for i in 1..4u32 {
            assert_eq!(px(&buf, w * 4, i, i), 0xffff0000, "diagonal {i}");
        }
        // On a 45° line every step is exact, so no off-diagonal spill on the interior rows.
        assert_eq!(px(&buf, w * 4, 0, 4), 0);
        assert_eq!(px(&buf, w * 4, 1, 2), 0);

        // A horizontal line sits on one exact row (gradient 0 ⇒ fpart(yend) == 0), so it never
        // bleeds onto neighbouring rows. Its *interior* pixel (x=2) is full coverage; the two
        // integer endpoints (x=1, x=3) are Wu-tapered to ~50% (xgap weighting), which is expected.
        let mut buf2 = scratch(w, h);
        let mut c2 = Canvas::new(&mut buf2, w, h, w * 4);
        c2.draw_line(1, 2, 3, 2, 0xff00ff00);
        assert_eq!(px(&buf2, w * 4, 2, 2), 0xff00ff00, "hline interior full");
        let a_l = px(&buf2, w * 4, 1, 2) >> 24;
        let a_r = px(&buf2, w * 4, 3, 2) >> 24;
        assert!(a_l > 0 && a_l < 255, "hline start endpoint Wu-tapered: {a_l:#x}");
        assert!(a_r > 0 && a_r < 255, "hline end endpoint Wu-tapered: {a_r:#x}");
        assert_eq!(px(&buf2, w * 4, 0, 2), 0);
        assert_eq!(px(&buf2, w * 4, 4, 2), 0);
        // A horizontal line does not bleed onto the neighbouring rows.
        assert_eq!(px(&buf2, w * 4, 2, 1), 0);
        assert_eq!(px(&buf2, w * 4, 2, 3), 0);
    }

    #[test]
    fn draw_line_wu_shallow_diagonal_has_partial_coverage() {
        // A shallow line (gradient 0.5) must split coverage between the two straddling rows on the
        // off-integer steps — the defining property of Wu antialiasing vs. the old aliased
        // Bresenham (which put down solid pixels only). (0,0)..(4,2): at x=1 the ideal y is 0.5,
        // so rows 0 and 1 each get ~50% coverage.
        let (w, h) = (6u32, 4u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        c.draw_line(0, 0, 4, 2, 0xffffffff); // opaque white
        // Column x=1: both row 0 and row 1 carry intermediate (non-0, non-255) alpha.
        let a_top = px(&buf, w * 4, 1, 0) >> 24;
        let a_bot = px(&buf, w * 4, 1, 1) >> 24;
        assert!(a_top > 0 && a_top < 255, "wu top pixel not partial: {a_top:#x}");
        assert!(a_bot > 0 && a_bot < 255, "wu bottom pixel not partial: {a_bot:#x}");
        // Complementary coverage: the two straddling pixels' alphas sum to roughly full (a small
        // rounding overshoot past 255 is fine — each side rounds its own ~50% independently).
        let sum = a_top + a_bot;
        assert!((230..=258).contains(&sum), "wu coverage not complementary: {sum}");
    }

    #[test]
    fn draw_line_clips_and_survives_degenerate() {
        let (w, h) = (4u32, 4u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        // A line that starts off-screen and ends on-screen paints only the visible tail. Interior
        // span pixels of this horizontal line are full coverage; the on-canvas end *endpoint*
        // (x=2) is Wu-tapered to ~50% (xgap weighting), so we only assert full on an interior px.
        c.draw_line(-10, 2, 2, 2, 0xff0000ff);
        // Fully off-screen and single-point lines must not panic (drawn before the pixel reads,
        // since the canvas borrows the buffer until then).
        c.draw_line(-5, -5, -1, -1, 0xffffffff);
        c.draw_line(100, 100, 200, 200, 0xffffffff);
        c.draw_line(1, 1, 1, 1, 0xffffff00); // degenerate point, on-canvas
        assert_eq!(px(&buf, w * 4, 0, 2), 0xff0000ff, "interior span pixel");
        assert_eq!(px(&buf, w * 4, 1, 2), 0xff0000ff, "interior span pixel");
        // The degenerate single-point line lands one full-coverage pixel.
        assert_eq!(px(&buf, w * 4, 1, 1), 0xffffff00);
    }

    #[test]
    fn fill_triangle_fills_interior_only() {
        let (w, h) = (8u32, 8u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        // A right triangle with the right angle at (0,0): interior is where x + y is small.
        c.fill_triangle((0, 0), (7, 0), (0, 7), 0xff00ffff);
        // Points well inside the triangle (far from every edge) stay at full coverage under AA.
        assert_eq!(px(&buf, w * 4, 1, 1), 0xff00ffff, "interior");
        assert_eq!(px(&buf, w * 4, 2, 1), 0xff00ffff, "interior");
        assert_eq!(px(&buf, w * 4, 1, 3), 0xff00ffff, "interior");
        // A point clearly outside the hypotenuse (x + y large) is untouched.
        assert_eq!(px(&buf, w * 4, 7, 7), 0, "outside");
        assert_eq!(px(&buf, w * 4, 6, 6), 0, "outside");
    }

    #[test]
    fn fill_triangle_edge_is_antialiased() {
        // The defining AA property: pixels straddling a sloped edge take *intermediate* alpha, not
        // just 0 or 255. Triangle (0,0),(8,0),(0,8): the hypotenuse x+y=8 runs the diagonal, and a
        // pixel sitting on it (e.g. (4,4), (3,5)) is ~half-covered → alpha strictly between 0/255.
        let (w, h) = (10u32, 10u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        c.fill_triangle((0, 0), (8, 0), (0, 8), 0xff00ffff); // opaque cyan
        for &(x, y) in &[(4u32, 4u32), (5, 3), (3, 5)] {
            let a = px(&buf, w * 4, x, y) >> 24;
            assert!(a > 0 && a < 255, "edge pixel {x},{y} not antialiased: alpha {a:#x}");
        }
        // A deep-interior pixel is still fully opaque (coverage 1).
        assert_eq!(px(&buf, w * 4, 1, 1) >> 24, 255, "interior must stay solid");
        // A pixel well outside stays clear.
        assert_eq!(px(&buf, w * 4, 8, 8), 0, "outside stays clear");
    }

    #[test]
    fn fill_triangle_interior_fill_is_exact_seam_is_aa() {
        // Two triangles sharing the diagonal (0,0)-(4,4) tile a 4×4 square. Under coverage-based AA
        // the shared diagonal is *not* a clean single 50% blend anymore: each triangle paints the
        // seam pixels at ~50% coverage (their centers sit exactly on the shared edge), so the seam
        // is composited as two partial source-over blends. Two 0.5-coverage passes over black give
        // only ~0.75 combined coverage (0.5 + 0.5·0.5), i.e. the seam is *under-covered* — the
        // inherent AA seam artifact when there is no coverage-accumulation buffer. We assert:
        //   1. deep-interior pixels (coverage 1) are an exact single 50% blend (0x808080), and
        //   2. a seam pixel differs from that clean blend (it is the AA seam, not the top-left rule).
        let (w, h) = (4u32, 4u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        c.clear(0xff000000);
        c.fill_triangle((0, 0), (4, 0), (4, 4), 0x80ffffff); // upper-right
        c.fill_triangle((0, 0), (4, 4), (0, 4), 0x80ffffff); // lower-left

        // (1) Interior fill is exact: a pixel deep inside the upper-right triangle (well below the
        // top edge, well right of the diagonal) is one exact 50% blend — coverage 1 there.
        assert_eq!(px(&buf, w * 4, 3, 1), 0xff808080, "upper-right interior");
        // And one deep inside the lower-left triangle.
        assert_eq!(px(&buf, w * 4, 1, 3), 0xff808080, "lower-left interior");

        // (2) The shared diagonal is the AA seam: pixel (2,2) is covered ~0.5 by each triangle, so
        // its composited grey is *not* the clean single 50% blend (0x80) but the partial-coverage
        // seam value (~0x70) — documenting AA seam behavior instead of the old top-left rule.
        let seam = px(&buf, w * 4, 2, 2) & 0xff; // blue channel of the grey
        assert!(seam != 0x80 && seam > 0, "seam should be AA (partial), got {seam:#x}");
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
    fn fill_circle_center_solid_rim_antialiased_outside_clear() {
        let (w, h) = (16u32, 16u32);
        let mut buf = scratch(w, h);
        let mut c = Canvas::new(&mut buf, w, h, w * 4);
        // A disc centered at (8,8) radius 5, opaque white.
        c.fill_circle(8.0, 8.0, 5.0, 0xffffffff);
        // Degenerate / off-screen circles are no-ops and must not panic — run them now (the canvas
        // holds the mutable borrow) so the pixel reads below prove they changed nothing.
        c.fill_circle(8.0, 8.0, 0.0, 0xffff0000); // r == 0
        c.fill_circle(8.0, 8.0, -3.0, 0xffff0000); // r < 0
        c.fill_circle(100.0, 100.0, 5.0, 0xffff0000); // off-screen
        // The center is fully inside → full coverage (and untouched by the no-ops).
        assert_eq!(px(&buf, w * 4, 8, 8) >> 24, 255, "center solid");
        // A pixel far outside the disc is untouched.
        assert_eq!(px(&buf, w * 4, 0, 0), 0, "far outside clear");
        assert_eq!(px(&buf, w * 4, 15, 15), 0, "far corner clear");
        // A rim pixel (near dist == r) has intermediate coverage — the AA soft edge. At (13,8) the
        // center distance is ~5, so coverage tapers to a softened alpha.
        let rim = px(&buf, w * 4, 13, 8) >> 24;
        assert!(rim < 255, "rim pixel should be softened, got {rim:#x}");
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
