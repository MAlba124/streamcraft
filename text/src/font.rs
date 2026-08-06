//! A self-contained A8 (8-bit coverage) bitmap font for CPU subtitle rasterisation.
//!
//! # Provenance
//!
//! The glyph atlas ([`font_data::ATLAS_A8`] / `font_atlas.a8`) and its metrics table
//! ([`font_data::SIZES`]) are **copied verbatim** from `scope/src/ui/font_data.rs` +
//! `scope/src/ui/font_atlas.a8` — the scope UI's baked JetBrains Mono atlas (SIL Open Font
//! License 1.1; the licence is copied to `text/src/FONT_LICENSE.txt`, and the origin +
//! `bake_font.py` tool live in the `scope` crate). We copy rather than depend on
//! `profluens-scope` because that crate pulls SDL3/GPU — far too heavy for a pure-CPU
//! overlay element. A single self-contained bitmap font in this crate is the right call.
//!
//! # Model — what this file adds over the copy
//!
//! The scope `font.rs` lays glyphs into a GPU [`DrawList`] of textured quads. An overlay
//! blitting onto CPU planes needs the opposite: a **coverage bitmap** it can alpha-over onto
//! luma. So this module reads the same atlas and metrics but *rasterises* a UTF-8 string to a
//! small owned A8 buffer ([`rasterize_line`] / [`Bitmap`]) by nearest-sampling the atlas per
//! destination pixel. Monospace: one advance per size; the pen is the top-left of the line box
//! and each glyph is offset by its baked bearings.
//!
//! v1 uses the atlas at its **native baked pixel size** (no scaling) — the overlay picks the
//! baked size nearest a target height, keeping glyphs pixel-crisp. Sub-pixel scaling (the
//! scope zoom path) is a future refinement.

use crate::font_data::{self, GlyphMetrics, SizeMetrics};

/// Index 0 in each size's glyph table is the replacement box for anything unmapped (matches
/// the scope bake: glyph 0 = box, 1..=95 = ASCII 0x20..=0x7E).
const REPLACEMENT: usize = 0;

/// An owned A8 coverage bitmap: `w * h` bytes, row-major, one coverage value per pixel
/// (0 = transparent, 255 = fully inked). The overlay alpha-overs this onto a video plane.
pub struct Bitmap {
    pub w: usize,
    pub h: usize,
    pub cov: Vec<u8>,
}

impl Bitmap {
    // Vec::new() allocates nothing (zero capacity) — the empty-coverage fallback for a zero-size
    // line. `Bitmap` owns a `Vec<u8>` by its public API, so the empty case is a null Vec, no heap.
    #[allow(clippy::disallowed_methods)]
    fn empty() -> Self {
        Bitmap { w: 0, h: 0, cov: Vec::new() }
    }

    /// Coverage at `(x, y)`, or 0 out of bounds.
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        if x < self.w && y < self.h {
            self.cov[y * self.w + x]
        } else {
            0
        }
    }
}

/// The baked [`SizeMetrics`] whose pixel size is nearest `px` (the atlas is baked at a fixed
/// set of sizes; v1 renders at the chosen one with no residual scaling). Never empty — the
/// bake ships at least one size.
pub fn size_nearest(px: f32) -> &'static SizeMetrics {
    let mut best = &font_data::SIZES[0];
    for s in font_data::SIZES.iter() {
        if (s.px - px).abs() < (best.px - px).abs() {
            best = s;
        }
    }
    best
}

/// The line-box height (px) at a baked size — the vertical advance between stacked lines.
pub fn line_height(s: &SizeMetrics) -> usize {
    s.line_h.round() as usize
}

/// The pixel width one line of `text` occupies at size `s` (monospace: `char_count ×
/// advance`). Newlines are the caller's concern (they split lines before measuring).
pub fn line_width(s: &SizeMetrics, text: &str) -> usize {
    (text.chars().count() as f32 * s.advance).round() as usize
}

/// Rasterise one line of UTF-8 `text` to an A8 coverage [`Bitmap`] at baked size `s`. The
/// bitmap is exactly `line_width × line_h`; each glyph's baked quad is placed at the pen
/// (top-left of the line box) plus its bearings, nearest-sampled from the atlas. Characters
/// outside printable ASCII render the replacement box; a space inks nothing.
pub fn rasterize_line(s: &SizeMetrics, text: &str) -> Bitmap {
    let w = line_width(s, text);
    let h = line_height(s);
    if w == 0 || h == 0 {
        return Bitmap::empty();
    }
    let mut cov = vec![0u8; w * h];
    let advance = s.advance;
    let mut pen_x = 0.0f32;
    let (aw, ah) = (font_data::ATLAS_W, font_data::ATLAS_H);
    let atlas = font_data::ATLAS_A8;
    for ch in text.chars() {
        let g = &s.glyphs[glyph_index(ch)];
        if g.w > 0 && g.h > 0 {
            blit_glyph(&mut cov, w, h, pen_x, g, atlas, aw, ah);
        }
        pen_x += advance;
    }
    Bitmap { w, h, cov }
}

/// Nearest-sample one atlas glyph into the destination coverage buffer at the pen. The glyph's
/// destination top-left is `(pen_x + off_x, off_y)`; its source is the atlas rect
/// `(g.x, g.y, g.w, g.h)`. Pixels landing outside the line box are clipped (an over-tall glyph
/// simply crops — the box is `line_h`).
#[allow(clippy::too_many_arguments)]
fn blit_glyph(
    dst: &mut [u8],
    dw: usize,
    dh: usize,
    pen_x: f32,
    g: &GlyphMetrics,
    atlas: &[u8],
    aw: usize,
    ah: usize,
) {
    let dst_x0 = (pen_x + g.off_x as f32).round() as i32;
    let dst_y0 = g.off_y as i32;
    for gy in 0..g.h as usize {
        let sy = g.y as usize + gy;
        if sy >= ah {
            break;
        }
        let dy = dst_y0 + gy as i32;
        if dy < 0 || dy as usize >= dh {
            continue;
        }
        for gx in 0..g.w as usize {
            let sx = g.x as usize + gx;
            if sx >= aw {
                break;
            }
            let dx = dst_x0 + gx as i32;
            if dx < 0 || dx as usize >= dw {
                continue;
            }
            let a = atlas[sy * aw + sx];
            if a == 0 {
                continue;
            }
            // Coverage max-composite (glyphs in one line never overlap horizontally in a
            // monospace layout; max is a safe idempotent merge at the seams).
            let idx = dy as usize * dw + dx as usize;
            if a > dst[idx] {
                dst[idx] = a;
            }
        }
    }
}

/// Map a `char` to a glyph-table index: printable ASCII → `1 + (c - 0x20)`, everything else →
/// the replacement box at 0 (identical to the scope bake's mapping).
fn glyph_index(ch: char) -> usize {
    let c = ch as u32;
    if (0x20..=0x7E).contains(&c) {
        (c - 0x20 + 1) as usize
    } else {
        REPLACEMENT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_size_snaps_to_a_baked_size() {
        // The atlas is baked at a known set (10, 14, …); a request lands on one of them.
        let s = size_nearest(14.0);
        assert!(font_data::SIZES.iter().any(|b| b.px == s.px));
        // Extreme requests clamp to the smallest/largest baked size.
        let small = size_nearest(1.0);
        let big = size_nearest(1000.0);
        assert!(small.px <= big.px);
    }

    #[test]
    fn a_space_inks_nothing_but_advances() {
        let s = size_nearest(14.0);
        let bm = rasterize_line(s, "   ");
        assert!(bm.w > 0, "three spaces still occupy width (advance)");
        assert!(bm.cov.iter().all(|&c| c == 0), "spaces leave zero coverage");
    }

    #[test]
    fn glyphs_produce_ink() {
        let s = size_nearest(14.0);
        let bm = rasterize_line(s, "Hi");
        let inked = bm.cov.iter().filter(|&&c| c > 0).count();
        assert!(inked > 0, "letters leave coverage");
        assert_eq!(bm.h, line_height(s));
        assert_eq!(bm.w, line_width(s, "Hi"));
    }

    #[test]
    fn empty_text_is_an_empty_bitmap() {
        let s = size_nearest(14.0);
        let bm = rasterize_line(s, "");
        assert_eq!((bm.w, bm.h), (0, 0));
        assert!(bm.cov.is_empty());
    }

    #[test]
    fn ink_stays_within_the_line_box() {
        // Every rasterised pixel is addressable — no out-of-bounds writes (the clip works).
        let s = size_nearest(14.0);
        let bm = rasterize_line(s, "Ag|_^");
        assert_eq!(bm.cov.len(), bm.w * bm.h);
    }
}
