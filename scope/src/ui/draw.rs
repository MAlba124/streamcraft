//! The draw list: the UI's per-frame geometry, as plain data.
//!
//! Widgets never touch SDL. They append primitives — filled rects, outlines,
//! lines-as-quads, textured quads, text runs — to a [`DrawList`]. The list is a
//! flat, ordered vector of [`Prim`]s (z is implied by submission order) plus a
//! clip-rect stack. Because it is *only data*, the whole UI is testable without a
//! window: feed widget calls a mock frame and assert the emitted primitives.
//!
//! Two things consume the list:
//! - the SDL backend, which tessellates primitives into interleaved vertex/index
//!   arrays and issues as few `SDL_RenderGeometryRaw` calls as possible (batched
//!   by texture and clip rect — see [`DrawList::build`]);
//! - the tests, which read [`DrawList::prims`] directly.

use crate::ui::arena::Arena;

/// RGBA8 colour. Straight bytes so tests read like `Color::rgb(0x20, 0x24, 0x2c)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }
    pub const TRANSPARENT: Color = Color::rgba(0, 0, 0, 0);
    pub const WHITE: Color = Color::rgb(255, 255, 255);
    pub const BLACK: Color = Color::rgb(0, 0, 0);

    /// Linear-ish alpha fade (multiplies the alpha channel by `f` in 0..=1).
    pub fn with_alpha(self, a: u8) -> Self {
        Self { a, ..self }
    }

    /// The float RGBA SDL wants (`SDL_FColor`), straight 0..1 division.
    pub fn to_f32(self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            self.a as f32 / 255.0,
        ]
    }
}

/// An axis-aligned rectangle in logical pixels (origin top-left, +y down — SDL's
/// convention).
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }
    pub fn right(&self) -> f32 {
        self.x + self.w
    }
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }
    /// Is the point inside (half-open on the far edges)?
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }
    /// Shrink by `pad` on every side (never below zero size).
    pub fn inset(&self, pad: f32) -> Rect {
        Rect::new(self.x + pad, self.y + pad, (self.w - 2.0 * pad).max(0.0), (self.h - 2.0 * pad).max(0.0))
    }
    /// The intersection of two rects, or a zero-size rect if disjoint. Used for
    /// nested clip rects (a child clip is always clamped inside its parent).
    pub fn intersect(&self, other: &Rect) -> Rect {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = self.right().min(other.right());
        let y1 = self.bottom().min(other.bottom());
        Rect::new(x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0))
    }
}

/// Which texture a primitive samples. The backend keeps at most two live textures
/// so the whole frame flushes in ~two `SDL_RenderGeometryRaw` batches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TexId {
    /// Solid colour (a 1x1 white texel, or SDL's null texture).
    None,
    /// The bitmap-font glyph atlas.
    Font,
}

/// One drawable: an axis-aligned quad (two triangles). Outlines and lines are
/// emitted as several of these; text is emitted as one per glyph. UVs are in
/// normalised atlas space and ignored for [`TexId::None`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Prim {
    /// Axis-aligned bounds. The quad's corners when `corners` is `None`.
    pub rect: Rect,
    /// Explicit corners (TL, TR, BR, BL winding) for rotated quads — diagonal
    /// lines. `None` = the axis-aligned `rect` fast path.
    pub corners: Option<[(f32, f32); 4]>,
    /// Normalised source rect in the atlas (`TexId::Font` only).
    pub uv: Rect,
    pub color: Color,
    pub tex: TexId,
    /// Clip rect this primitive is scissored to. `None` = the whole framebuffer.
    pub clip: Option<Rect>,
}

/// A contiguous run of primitives sharing one texture and one clip rect — exactly
/// one `SDL_RenderGeometryRaw` call. The vertex/index data lives in a per-frame
/// arena; these slices point into it.
pub struct Batch<'a> {
    pub tex: TexId,
    pub clip: Option<Rect>,
    /// Interleaved `[x, y]` pairs, two floats per vertex.
    pub xy: &'a [f32],
    /// `SDL_FColor`-compatible `[r, g, b, a]`, four floats per vertex.
    pub color: &'a [f32],
    /// `[u, v]` pairs, two floats per vertex.
    pub uv: &'a [f32],
    pub indices: &'a [u16],
    pub num_vertices: usize,
}

/// The per-frame draw list. Cleared, not freed, each frame.
#[derive(Default)]
pub struct DrawList {
    prims: Vec<Prim>,
    /// Clip-rect stack; the effective clip is the top (already intersected with
    /// its parent when pushed).
    clip_stack: Vec<Rect>,
}

impl DrawList {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rewind for a new frame (keeps `Vec` capacity — no per-frame heap traffic).
    pub fn clear(&mut self) {
        self.prims.clear();
        self.clip_stack.clear();
    }

    /// The recorded primitives, in submission (z) order. Tests read this.
    pub fn prims(&self) -> &[Prim] {
        &self.prims
    }

    /// Current effective clip rect (`None` = unclipped).
    pub fn current_clip(&self) -> Option<Rect> {
        self.clip_stack.last().copied()
    }

    /// Push a clip rect, clamped inside the current one. Everything drawn until the
    /// matching [`pop_clip`](DrawList::pop_clip) is scissored to it.
    pub fn push_clip(&mut self, rect: Rect) {
        let clamped = match self.clip_stack.last() {
            Some(parent) => parent.intersect(&rect),
            None => rect,
        };
        self.clip_stack.push(clamped);
    }

    pub fn pop_clip(&mut self) {
        self.clip_stack.pop();
    }

    fn push_prim(&mut self, rect: Rect, uv: Rect, color: Color, tex: TexId) {
        if color.a == 0 && tex == TexId::None {
            return; // fully transparent solid: nothing to draw
        }
        self.prims.push(Prim { rect, corners: None, uv, color, tex, clip: self.current_clip() });
    }

    /// A solid quad with explicit corners (TL, TR, BR, BL winding) — rotated
    /// geometry (diagonal lines). `rect` is set to the bounding box.
    fn push_quad_corners(&mut self, corners: [(f32, f32); 4], color: Color) {
        if color.a == 0 {
            return;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, y) in &corners {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        self.prims.push(Prim {
            rect: Rect::new(x0, y0, x1 - x0, y1 - y0),
            corners: Some(corners),
            uv: Rect::default(),
            color,
            tex: TexId::None,
            clip: self.current_clip(),
        });
    }

    /// A filled solid rectangle.
    pub fn fill_rect(&mut self, rect: Rect, color: Color) {
        self.push_prim(rect, Rect::default(), color, TexId::None);
    }

    /// A rectangle outline `thickness` px wide, drawn as four inset bars.
    pub fn rect_outline(&mut self, rect: Rect, thickness: f32, color: Color) {
        let t = thickness.max(0.0);
        if t <= 0.0 || rect.w <= 0.0 || rect.h <= 0.0 {
            return;
        }
        // top, bottom, left, right (corners covered by top/bottom bars).
        self.fill_rect(Rect::new(rect.x, rect.y, rect.w, t), color);
        self.fill_rect(Rect::new(rect.x, rect.bottom() - t, rect.w, t), color);
        self.fill_rect(Rect::new(rect.x, rect.y + t, t, (rect.h - 2.0 * t).max(0.0)), color);
        self.fill_rect(Rect::new(rect.right() - t, rect.y + t, t, (rect.h - 2.0 * t).max(0.0)), color);
    }

    /// A line from `a` to `b`, drawn as a quad of the given thickness. Axis-aligned
    /// lines take the crisp `Rect` fast path; diagonals emit a properly rotated
    /// quad (the graph view's routed edges are mostly diagonal).
    pub fn line(&mut self, ax: f32, ay: f32, bx: f32, by: f32, thickness: f32, color: Color) {
        let t = thickness.max(1.0);
        if (ay - by).abs() < 0.5 {
            // horizontal
            let x0 = ax.min(bx);
            let w = (ax - bx).abs();
            self.fill_rect(Rect::new(x0, ay - t / 2.0, w, t), color);
        } else if (ax - bx).abs() < 0.5 {
            // vertical
            let y0 = ay.min(by);
            let h = (ay - by).abs();
            self.fill_rect(Rect::new(ax - t / 2.0, y0, t, h), color);
        } else {
            // Rotated quad: offset both endpoints by ±half-thickness along the
            // unit normal of the line direction.
            let (dx, dy) = (bx - ax, by - ay);
            let len = (dx * dx + dy * dy).sqrt();
            let (nx, ny) = (-dy / len * t * 0.5, dx / len * t * 0.5);
            self.push_quad_corners(
                [
                    (ax + nx, ay + ny),
                    (bx + nx, by + ny),
                    (bx - nx, by - ny),
                    (ax - nx, ay - ny),
                ],
                color,
            );
        }
    }

    /// A textured quad sampling the font atlas at `uv` (normalised), tinted `color`.
    /// Used by the text path; exposed for completeness.
    pub fn textured_quad(&mut self, rect: Rect, uv: Rect, color: Color) {
        self.push_prim(rect, uv, color, TexId::Font);
    }

    /// Number of primitives (tests / diagnostics).
    pub fn len(&self) -> usize {
        self.prims.len()
    }
    pub fn is_empty(&self) -> bool {
        self.prims.is_empty()
    }

    /// Tessellate the primitive list into batches ready for `SDL_RenderGeometryRaw`.
    ///
    /// Consecutive primitives sharing `(tex, clip)` coalesce into one batch, so a
    /// typical frame — solid panels, then one atlas run of text — flushes in a
    /// handful of draw calls (often two). All vertex/index storage comes from
    /// `arena`, so building the batches allocates nothing on the heap in steady
    /// state.
    ///
    /// Returns `(batches, total_indices)`; the caller iterates and issues one raw
    /// geometry call per [`Batch`].
    pub fn build<'a>(&self, arena: &'a mut Arena) -> Vec<Batch<'a>> {
        // First pass: how many verts/indices total, and the batch boundaries.
        // Each quad = 4 verts, 6 indices. Runs break when (tex, clip) changes.
        let n = self.prims.len();
        if n == 0 {
            return Vec::new();
        }
        let total_verts = n * 4;
        let total_indices = n * 6;

        // All four streams in one arena bump (avoids overlapping `&'a mut` borrows;
        // see Arena::alloc_geometry).
        let geo = arena.alloc_geometry(total_verts, total_indices);
        let xy = geo.xy;
        let col = geo.col;
        let uv = geo.uv;
        let idx = geo.idx;

        // Boundaries: (start_prim, end_prim, tex, clip).
        let mut bounds: Vec<(usize, usize, TexId, Option<Rect>)> = Vec::new();
        let mut run_start = 0usize;
        for i in 1..=n {
            let brk = i == n || {
                let a = &self.prims[i - 1];
                let b = &self.prims[i];
                a.tex != b.tex || !clip_eq(a.clip, b.clip)
            };
            if brk {
                let p = &self.prims[run_start];
                bounds.push((run_start, i, p.tex, p.clip));
                run_start = i;
            }
        }

        // Second pass: fill the arena streams, quad by quad.
        for (i, p) in self.prims.iter().enumerate() {
            let v0 = i * 4;
            let r = p.rect;
            let corners = p.corners.unwrap_or([
                (r.x, r.y),
                (r.right(), r.y),
                (r.right(), r.bottom()),
                (r.x, r.bottom()),
            ]);
            let uvc = [
                (p.uv.x, p.uv.y),
                (p.uv.right(), p.uv.y),
                (p.uv.right(), p.uv.bottom()),
                (p.uv.x, p.uv.bottom()),
            ];
            let c = p.color.to_f32();
            for k in 0..4 {
                let v = v0 + k;
                xy[v * 2] = corners[k].0;
                xy[v * 2 + 1] = corners[k].1;
                uv[v * 2] = uvc[k].0;
                uv[v * 2 + 1] = uvc[k].1;
                col[v * 4] = c[0];
                col[v * 4 + 1] = c[1];
                col[v * 4 + 2] = c[2];
                col[v * 4 + 3] = c[3];
            }
            // Two triangles: 0-1-2, 0-2-3. Indices are absolute vertex numbers;
            // each batch re-bases them below.
            let o = i * 6;
            let base = v0 as u16;
            idx[o] = base;
            idx[o + 1] = base + 1;
            idx[o + 2] = base + 2;
            idx[o + 3] = base;
            idx[o + 4] = base + 2;
            idx[o + 5] = base + 3;
        }

        // SAFETY-free slicing: we hand each batch the *whole* vertex arrays plus its
        // own index sub-slice (indices are absolute vertex numbers, so a batch can
        // reference the shared vertex arrays directly). This is the simplest correct
        // form; SDL walks only the indices we pass.
        //
        // We must re-borrow the arena slices immutably to hand out overlapping
        // shared references. Convert to raw and back within the arena's lifetime.
        let xy_ref: &'a [f32] = xy;
        let col_ref: &'a [f32] = col;
        let uv_ref: &'a [f32] = uv;
        let idx_ref: &'a [u16] = idx;

        bounds
            .into_iter()
            .map(|(s, e, tex, clip)| Batch {
                tex,
                clip,
                xy: xy_ref,
                color: col_ref,
                uv: uv_ref,
                indices: &idx_ref[s * 6..e * 6],
                num_vertices: total_verts,
            })
            .collect()
    }
}

/// Clip-rect equality with an epsilon (float rects that came from the same source
/// should coalesce even if recomputed).
fn clip_eq(a: Option<Rect>, b: Option<Rect>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            (x.x - y.x).abs() < 0.01
                && (x.y - y.y).abs() < 0.01
                && (x.w - y.w).abs() < 0.01
                && (x.h - y.h).abs() < 0.01
        }
        _ => false,
    }
}
