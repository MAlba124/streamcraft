//! Text rendering for the scope UI: a baked, anti-aliased glyph atlas.
//!
//! # Typeface, provenance & licence
//!
//! The embedded typeface is **JetBrains Mono Regular 2.304** — © 2020 The
//! JetBrains Mono Project Authors, **SIL Open Font License 1.1** (full text in
//! `scope/src/ui/FONT_LICENSE.txt`). Glyphs are pre-rasterised offline by
//! `scope/tools/bake_font.py` (run command in its header) into one A8 atlas
//! (`font_atlas.a8`) plus a generated metrics table (`font_data.rs`) at a fixed
//! set of pixel sizes — so the crate ships a real, hinted, anti-aliased typeface
//! with **zero runtime dependencies** and no build-time font pipeline.
//!
//! # Model
//!
//! Monospace: one `advance` per size; the pen is the **top-left of the line box**
//! and each glyph quad is offset by its bearings (`off_x`, `off_y`). A requested
//! pixel size snaps to the nearest baked size and the quads scale by the
//! remainder — with linear filtering this keeps continuous zoom smooth while
//! exact sizes stay pixel-crisp. Layout and measurement are pure (tested without
//! a window); only the atlas *upload* touches SDL.

use crate::ui::draw::{Color, DrawList, Rect};
use crate::ui::font_data::{self, SizeMetrics};

/// The UI's default text size in pixels (widgets multiply by `Theme.text_scale`).
pub const DEFAULT_PX: f32 = 14.0;

/// Atlas dimensions in pixels (from the bake).
pub const ATLAS_W: usize = font_data::ATLAS_W;
pub const ATLAS_H: usize = font_data::ATLAS_H;

/// Index 0 in each size's glyph table is the replacement box for anything unmapped.
const REPLACEMENT: usize = 0;

/// A measured/laid-out font. Cheap to clone; holds nothing per-frame. The atlas
/// *pixels* live here; the atlas *texture* is created by the backend from
/// [`Font::atlas_rgba`].
#[derive(Clone)]
pub struct Font {
    /// RGBA8 atlas (white + baked alpha), `ATLAS_W * ATLAS_H * 4` bytes, row-major.
    atlas: Vec<u8>,
}

impl Default for Font {
    fn default() -> Self {
        Self::new()
    }
}

impl Font {
    /// Expand the baked A8 coverage into the white-tint RGBA atlas once.
    pub fn new() -> Self {
        let a8 = font_data::ATLAS_A8;
        let mut atlas = vec![0u8; ATLAS_W * ATLAS_H * 4];
        for (i, &a) in a8.iter().enumerate() {
            let o = i * 4;
            atlas[o] = 0xFF;
            atlas[o + 1] = 0xFF;
            atlas[o + 2] = 0xFF;
            atlas[o + 3] = a;
        }
        Self { atlas }
    }

    /// The RGBA8 atlas pixels for the backend to upload (row-major, no padding).
    pub fn atlas_rgba(&self) -> &[u8] {
        &self.atlas
    }
    pub fn atlas_size(&self) -> (usize, usize) {
        (ATLAS_W, ATLAS_H)
    }

    /// The baked size nearest `px`, plus the residual quad scale to reach `px` exactly.
    fn size_for(px: f32) -> (&'static SizeMetrics, f32) {
        let mut best = &font_data::SIZES[0];
        for s in font_data::SIZES.iter() {
            if (s.px - px).abs() < (best.px - px).abs() {
                best = s;
            }
        }
        (best, px / best.px)
    }

    /// Advance width of one glyph at the default size (monospace: same for all).
    pub fn cell_w(&self) -> f32 {
        let (s, r) = Self::size_for(DEFAULT_PX);
        s.advance * r
    }
    /// Line height at the default size.
    pub fn line_h(&self) -> f32 {
        self.line_h_px(DEFAULT_PX)
    }
    /// Line height at an arbitrary pixel size.
    pub fn line_h_px(&self, px: f32) -> f32 {
        let (s, r) = Self::size_for(px);
        s.line_h * r
    }

    /// Pixel size of `text` at the default size (no wrapping; newlines start a
    /// new line). Width = longest line, height = line count × line height.
    pub fn measure(&self, text: &str) -> (f32, f32) {
        self.measure_px(text, DEFAULT_PX)
    }

    /// [`Font::measure`] at an arbitrary pixel size.
    pub fn measure_px(&self, text: &str, px: f32) -> (f32, f32) {
        let (s, r) = Self::size_for(px);
        let mut max_cols = 0usize;
        let mut cols = 0usize;
        let mut lines = 1usize;
        for ch in text.chars() {
            if ch == '\n' {
                max_cols = max_cols.max(cols);
                cols = 0;
                lines += 1;
            } else {
                cols += 1;
            }
        }
        max_cols = max_cols.max(cols);
        (max_cols as f32 * s.advance * r, lines as f32 * s.line_h * r)
    }

    /// Width of a single line at the default size — the common widget case.
    pub fn measure_line(&self, text: &str) -> f32 {
        let (s, r) = Self::size_for(DEFAULT_PX);
        text.chars().count() as f32 * s.advance * r
    }

    /// Append `text` as textured glyph quads, top-left of the line box at
    /// `(x, y)`, tinted `color`, at `scale` × the default size. Returns the pen
    /// after the last glyph. Newlines wrap to `x` on the next line. Glyphs with
    /// no ink (space) emit no quad.
    pub fn layout_into(
        &self,
        dl: &mut DrawList,
        text: &str,
        x: f32,
        y: f32,
        scale: f32,
        color: Color,
    ) -> (f32, f32) {
        self.layout_px(dl, text, x, y, DEFAULT_PX * scale, color)
    }

    /// [`Font::layout_into`] at an explicit pixel size (the graph view zooms).
    pub fn layout_px(
        &self,
        dl: &mut DrawList,
        text: &str,
        x: f32,
        y: f32,
        px: f32,
        color: Color,
    ) -> (f32, f32) {
        let (s, r) = Self::size_for(px);
        let mut pen_x = x;
        let mut pen_y = y;
        for ch in text.chars() {
            if ch == '\n' {
                pen_x = x;
                pen_y += s.line_h * r;
                continue;
            }
            let g = &s.glyphs[glyph_index(ch)];
            if g.w > 0 {
                let quad = Rect::new(
                    pen_x + g.off_x as f32 * r,
                    pen_y + g.off_y as f32 * r,
                    g.w as f32 * r,
                    g.h as f32 * r,
                );
                let uv = Rect::new(
                    g.x as f32 / ATLAS_W as f32,
                    g.y as f32 / ATLAS_H as f32,
                    g.w as f32 / ATLAS_W as f32,
                    g.h as f32 / ATLAS_H as f32,
                );
                dl.textured_quad(quad, uv, color);
            }
            pen_x += s.advance * r;
        }
        (pen_x, pen_y)
    }
}

/// Map a `char` to a glyph-table index: printable ASCII → `1 + (c - 0x20)`,
/// everything else → the replacement box at 0.
fn glyph_index(ch: char) -> usize {
    let c = ch as u32;
    if (0x20..=0x7E).contains(&c) {
        (c - 0x20 + 1) as usize
    } else {
        REPLACEMENT
    }
}
