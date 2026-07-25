//! Bitmap font and glyph atlas for the scope UI.
//!
//! # Font provenance & licence
//!
//! The embedded typeface is **"Scope 8x8"**, an original 8x8 monospace bitmap font
//! authored by hand for this crate (see [`GLYPHS`]). It is released into the
//! **public domain (CC0 1.0)** — see `scope/src/ui/FONT_LICENSE.txt`. No third-party
//! font data is vendored, so there is no attribution obligation and nothing to keep
//! in sync. (The task suggested Spleen 8x16 / BSD-1-Clause as an acceptable option;
//! an original public-domain glyph set is strictly more permissive and keeps the
//! whole font auditable in one array.)
//!
//! Each glyph is 8x8: one `u8` per row, MSB = leftmost pixel. ASCII 0x20..=0x7E is
//! covered; every other code point renders the replacement box at index 0.
//!
//! # Atlas
//!
//! At startup the glyphs are unpacked into a single RGBA8 atlas texture laid out as
//! a 16-wide grid of 8x8 cells (16x8 cells = 128x64 px, one cell per byte value
//! 0..=127). A glyph pixel becomes opaque white (`0xFFFFFFFF`); background is fully
//! transparent, so the draw list's per-vertex colour tints the text and blending
//! composites it. Text emission ([`Font::layout_into`]) walks a string and appends
//! one textured quad per glyph to a [`DrawList`], advancing the pen by the fixed
//! cell width. This is all pure — [`Font::measure`] and layout are tested without a
//! window; only the atlas *upload* touches SDL.

use crate::ui::draw::{Color, DrawList, Rect};

/// Glyph cell size in pixels (square, monospace).
pub const CELL: usize = 8;
/// Cells per atlas row (one row per 16 code points → 8 rows for 0..=127).
pub const ATLAS_COLS: usize = 16;
pub const ATLAS_ROWS: usize = 8;
/// Atlas dimensions in pixels.
pub const ATLAS_W: usize = ATLAS_COLS * CELL;
pub const ATLAS_H: usize = ATLAS_ROWS * CELL;

/// Index 0 in [`GLYPHS`] is the replacement box for anything unmapped.
const REPLACEMENT: usize = 0;

/// A measured/laid-out font. Cheap to clone (all sizing is constant); holds nothing
/// per-frame. The atlas *pixels* live here; the atlas *texture* is created by the
/// backend from [`Font::atlas_rgba`].
#[derive(Clone)]
pub struct Font {
    /// RGBA8 atlas, `ATLAS_W * ATLAS_H * 4` bytes, row-major.
    atlas: Vec<u8>,
    /// Logical advance/line height (== CELL for this monospace font).
    cell_w: f32,
    cell_h: f32,
}

impl Default for Font {
    fn default() -> Self {
        Self::new()
    }
}

impl Font {
    /// Build the font and rasterise its atlas once.
    pub fn new() -> Self {
        let mut atlas = vec![0u8; ATLAS_W * ATLAS_H * 4];
        // Byte value b in 0..=127 → cell (b % 16, b / 16). Printable ASCII maps to
        // its glyph; the rest to the replacement box.
        for b in 0u8..128 {
            let glyph = glyph_for(b);
            let cx = (b as usize % ATLAS_COLS) * CELL;
            let cy = (b as usize / ATLAS_COLS) * CELL;
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..CELL {
                    // MSB is the leftmost pixel.
                    if bits & (0x80 >> col) != 0 {
                        let px = cx + col;
                        let py = cy + row;
                        let o = (py * ATLAS_W + px) * 4;
                        atlas[o] = 0xFF;
                        atlas[o + 1] = 0xFF;
                        atlas[o + 2] = 0xFF;
                        atlas[o + 3] = 0xFF;
                    }
                }
            }
        }
        Self { atlas, cell_w: CELL as f32, cell_h: CELL as f32 }
    }

    /// The RGBA8 atlas pixels for the backend to upload (row-major, no padding).
    pub fn atlas_rgba(&self) -> &[u8] {
        &self.atlas
    }
    pub fn atlas_size(&self) -> (usize, usize) {
        (ATLAS_W, ATLAS_H)
    }

    /// Advance width of one glyph cell (monospace: same for every code point).
    pub fn cell_w(&self) -> f32 {
        self.cell_w
    }
    /// Line height.
    pub fn line_h(&self) -> f32 {
        self.cell_h
    }

    /// Pixel size of `text` at scale 1 (no wrapping; newlines start a new line).
    /// Width = longest line, height = line count × line height.
    pub fn measure(&self, text: &str) -> (f32, f32) {
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
        (max_cols as f32 * self.cell_w, lines as f32 * self.cell_h)
    }

    /// Width of a single line (no newline handling) — the common widget case.
    pub fn measure_line(&self, text: &str) -> f32 {
        text.chars().count() as f32 * self.cell_w
    }

    /// Append `text` to `dl` as textured glyph quads, top-left at `(x, y)`, tinted
    /// `color`, at integer `scale`. Returns the pen position after the last glyph.
    /// Newlines wrap to `x` on the next line. Spaces (and any glyph whose cell is
    /// empty) emit no quad — the atlas run stays tight.
    pub fn layout_into(&self, dl: &mut DrawList, text: &str, x: f32, y: f32, scale: f32, color: Color) -> (f32, f32) {
        let gw = self.cell_w * scale;
        let gh = self.cell_h * scale;
        let mut pen_x = x;
        let mut pen_y = y;
        for ch in text.chars() {
            if ch == '\n' {
                pen_x = x;
                pen_y += gh;
                continue;
            }
            let b = ascii_index(ch);
            // Skip space and other empty cells: no quad, just advance.
            if !glyph_is_blank(b) {
                let cx = (b as usize % ATLAS_COLS) as f32 * CELL as f32;
                let cy = (b as usize / ATLAS_COLS) as f32 * CELL as f32;
                let uv = Rect::new(
                    cx / ATLAS_W as f32,
                    cy / ATLAS_H as f32,
                    CELL as f32 / ATLAS_W as f32,
                    CELL as f32 / ATLAS_H as f32,
                );
                dl.textured_quad(Rect::new(pen_x, pen_y, gw, gh), uv, color);
            }
            pen_x += gw;
        }
        (pen_x, pen_y)
    }
}

/// Map a `char` to an atlas byte index: printable ASCII → itself, else replacement.
fn ascii_index(ch: char) -> u8 {
    let c = ch as u32;
    if (0x20..=0x7E).contains(&c) {
        c as u8
    } else {
        REPLACEMENT as u8
    }
}

/// The 8-row glyph for atlas byte `b` (0..=127).
fn glyph_for(b: u8) -> [u8; 8] {
    let c = b as u32;
    if (0x20..=0x7E).contains(&c) {
        GLYPHS[(c - 0x20 + 1) as usize]
    } else {
        GLYPHS[REPLACEMENT]
    }
}

/// True if the atlas byte's glyph has no set pixels (space, or unmapped-to-blank).
fn glyph_is_blank(b: u8) -> bool {
    glyph_for(b).iter().all(|&row| row == 0)
}

// Row bit patterns, MSB = leftmost of 8 columns. Index 0 is the replacement box;
// indices 1..=95 are ASCII 0x20 (space) .. 0x7E (~) in order.
//
// Authored by hand for this crate; public domain (CC0). Kept as literal binary so a
// glyph reads like a tiny picture in source — the "1" bits trace each character.
#[rustfmt::skip]
static GLYPHS: [[u8; 8]; 96] = [
    // 0: replacement box (filled border with hollow centre)
    [0b11111110, 0b10000010, 0b10000010, 0b10000010, 0b10000010, 0b10000010, 0b11111110, 0b00000000],
    // 0x20 ' '
    [0,0,0,0,0,0,0,0],
    // 0x21 '!'
    [0b00110000,0b00110000,0b00110000,0b00110000,0b00110000,0b00000000,0b00110000,0b00000000],
    // 0x22 '"'
    [0b01101100,0b01101100,0b01101100,0b00000000,0b00000000,0b00000000,0b00000000,0b00000000],
    // 0x23 '#'
    [0b01101100,0b01101100,0b11111110,0b01101100,0b11111110,0b01101100,0b01101100,0b00000000],
    // 0x24 '$'
    [0b00110000,0b01111100,0b11000000,0b01111000,0b00001100,0b11111000,0b00110000,0b00000000],
    // 0x25 '%'
    [0b11000110,0b11001100,0b00011000,0b00110000,0b01100000,0b11001100,0b10000110,0b00000000],
    // 0x26 '&'
    [0b00111000,0b01101100,0b00111000,0b01110110,0b11011100,0b11001100,0b01110110,0b00000000],
    // 0x27 '\''
    [0b00110000,0b00110000,0b01100000,0b00000000,0b00000000,0b00000000,0b00000000,0b00000000],
    // 0x28 '('
    [0b00011000,0b00110000,0b01100000,0b01100000,0b01100000,0b00110000,0b00011000,0b00000000],
    // 0x29 ')'
    [0b01100000,0b00110000,0b00011000,0b00011000,0b00011000,0b00110000,0b01100000,0b00000000],
    // 0x2A '*'
    [0b00000000,0b01101100,0b00111000,0b11111110,0b00111000,0b01101100,0b00000000,0b00000000],
    // 0x2B '+'
    [0b00000000,0b00110000,0b00110000,0b11111100,0b00110000,0b00110000,0b00000000,0b00000000],
    // 0x2C ','
    [0b00000000,0b00000000,0b00000000,0b00000000,0b00000000,0b00110000,0b00110000,0b01100000],
    // 0x2D '-'
    [0b00000000,0b00000000,0b00000000,0b11111110,0b00000000,0b00000000,0b00000000,0b00000000],
    // 0x2E '.'
    [0b00000000,0b00000000,0b00000000,0b00000000,0b00000000,0b00110000,0b00110000,0b00000000],
    // 0x2F '/'
    [0b00000110,0b00001100,0b00011000,0b00110000,0b01100000,0b11000000,0b10000000,0b00000000],
    // 0x30 '0'
    [0b01111100,0b11000110,0b11001110,0b11010110,0b11100110,0b11000110,0b01111100,0b00000000],
    // 0x31 '1'
    [0b00110000,0b01110000,0b00110000,0b00110000,0b00110000,0b00110000,0b11111100,0b00000000],
    // 0x32 '2'
    [0b01111100,0b11000110,0b00000110,0b00011100,0b01110000,0b11000000,0b11111110,0b00000000],
    // 0x33 '3'
    [0b01111100,0b11000110,0b00000110,0b00111100,0b00000110,0b11000110,0b01111100,0b00000000],
    // 0x34 '4'
    [0b00011100,0b00111100,0b01101100,0b11001100,0b11111110,0b00001100,0b00011110,0b00000000],
    // 0x35 '5'
    [0b11111110,0b11000000,0b11111100,0b00000110,0b00000110,0b11000110,0b01111100,0b00000000],
    // 0x36 '6'
    [0b00111100,0b01100000,0b11000000,0b11111100,0b11000110,0b11000110,0b01111100,0b00000000],
    // 0x37 '7'
    [0b11111110,0b11000110,0b00001100,0b00011000,0b00110000,0b00110000,0b00110000,0b00000000],
    // 0x38 '8'
    [0b01111100,0b11000110,0b11000110,0b01111100,0b11000110,0b11000110,0b01111100,0b00000000],
    // 0x39 '9'
    [0b01111100,0b11000110,0b11000110,0b01111110,0b00000110,0b00001100,0b01111000,0b00000000],
    // 0x3A ':'
    [0b00000000,0b00110000,0b00110000,0b00000000,0b00000000,0b00110000,0b00110000,0b00000000],
    // 0x3B ';'
    [0b00000000,0b00110000,0b00110000,0b00000000,0b00000000,0b00110000,0b00110000,0b01100000],
    // 0x3C '<'
    [0b00001100,0b00011000,0b00110000,0b01100000,0b00110000,0b00011000,0b00001100,0b00000000],
    // 0x3D '='
    [0b00000000,0b00000000,0b11111110,0b00000000,0b11111110,0b00000000,0b00000000,0b00000000],
    // 0x3E '>'
    [0b01100000,0b00110000,0b00011000,0b00001100,0b00011000,0b00110000,0b01100000,0b00000000],
    // 0x3F '?'
    [0b01111100,0b11000110,0b00001100,0b00011000,0b00110000,0b00000000,0b00110000,0b00000000],
    // 0x40 '@'
    [0b01111100,0b11000110,0b11011110,0b11011110,0b11011100,0b11000000,0b01111100,0b00000000],
    // 0x41 'A'
    [0b00111000,0b01101100,0b11000110,0b11000110,0b11111110,0b11000110,0b11000110,0b00000000],
    // 0x42 'B'
    [0b11111100,0b11000110,0b11000110,0b11111100,0b11000110,0b11000110,0b11111100,0b00000000],
    // 0x43 'C'
    [0b01111100,0b11000110,0b11000000,0b11000000,0b11000000,0b11000110,0b01111100,0b00000000],
    // 0x44 'D'
    [0b11111000,0b11001100,0b11000110,0b11000110,0b11000110,0b11001100,0b11111000,0b00000000],
    // 0x45 'E'
    [0b11111110,0b11000000,0b11000000,0b11111100,0b11000000,0b11000000,0b11111110,0b00000000],
    // 0x46 'F'
    [0b11111110,0b11000000,0b11000000,0b11111100,0b11000000,0b11000000,0b11000000,0b00000000],
    // 0x47 'G'
    [0b01111100,0b11000110,0b11000000,0b11001110,0b11000110,0b11000110,0b01111110,0b00000000],
    // 0x48 'H'
    [0b11000110,0b11000110,0b11000110,0b11111110,0b11000110,0b11000110,0b11000110,0b00000000],
    // 0x49 'I'
    [0b01111000,0b00110000,0b00110000,0b00110000,0b00110000,0b00110000,0b01111000,0b00000000],
    // 0x4A 'J'
    [0b00011110,0b00001100,0b00001100,0b00001100,0b11001100,0b11001100,0b01111000,0b00000000],
    // 0x4B 'K'
    [0b11000110,0b11001100,0b11011000,0b11110000,0b11011000,0b11001100,0b11000110,0b00000000],
    // 0x4C 'L'
    [0b11000000,0b11000000,0b11000000,0b11000000,0b11000000,0b11000000,0b11111110,0b00000000],
    // 0x4D 'M'
    [0b11000110,0b11101110,0b11111110,0b11010110,0b11000110,0b11000110,0b11000110,0b00000000],
    // 0x4E 'N'
    [0b11000110,0b11100110,0b11110110,0b11011110,0b11001110,0b11000110,0b11000110,0b00000000],
    // 0x4F 'O'
    [0b01111100,0b11000110,0b11000110,0b11000110,0b11000110,0b11000110,0b01111100,0b00000000],
    // 0x50 'P'
    [0b11111100,0b11000110,0b11000110,0b11111100,0b11000000,0b11000000,0b11000000,0b00000000],
    // 0x51 'Q'
    [0b01111100,0b11000110,0b11000110,0b11000110,0b11011110,0b11001100,0b01110110,0b00000000],
    // 0x52 'R'
    [0b11111100,0b11000110,0b11000110,0b11111100,0b11011000,0b11001100,0b11000110,0b00000000],
    // 0x53 'S'
    [0b01111100,0b11000110,0b11000000,0b01111100,0b00000110,0b11000110,0b01111100,0b00000000],
    // 0x54 'T'
    [0b11111100,0b00110000,0b00110000,0b00110000,0b00110000,0b00110000,0b00110000,0b00000000],
    // 0x55 'U'
    [0b11000110,0b11000110,0b11000110,0b11000110,0b11000110,0b11000110,0b01111100,0b00000000],
    // 0x56 'V'
    [0b11000110,0b11000110,0b11000110,0b11000110,0b11000110,0b01101100,0b00111000,0b00000000],
    // 0x57 'W'
    [0b11000110,0b11000110,0b11000110,0b11010110,0b11111110,0b11101110,0b11000110,0b00000000],
    // 0x58 'X'
    [0b11000110,0b11000110,0b01101100,0b00111000,0b01101100,0b11000110,0b11000110,0b00000000],
    // 0x59 'Y'
    [0b11000110,0b11000110,0b01101100,0b00111000,0b00110000,0b00110000,0b00110000,0b00000000],
    // 0x5A 'Z'
    [0b11111110,0b00000110,0b00001100,0b00011000,0b00110000,0b01100000,0b11111110,0b00000000],
    // 0x5B '['
    [0b01111000,0b01100000,0b01100000,0b01100000,0b01100000,0b01100000,0b01111000,0b00000000],
    // 0x5C '\\'
    [0b11000000,0b01100000,0b00110000,0b00011000,0b00001100,0b00000110,0b00000010,0b00000000],
    // 0x5D ']'
    [0b01111000,0b00011000,0b00011000,0b00011000,0b00011000,0b00011000,0b01111000,0b00000000],
    // 0x5E '^'
    [0b00010000,0b00111000,0b01101100,0b11000110,0b00000000,0b00000000,0b00000000,0b00000000],
    // 0x5F '_'
    [0b00000000,0b00000000,0b00000000,0b00000000,0b00000000,0b00000000,0b00000000,0b11111111],
    // 0x60 '`'
    [0b01100000,0b00110000,0b00011000,0b00000000,0b00000000,0b00000000,0b00000000,0b00000000],
    // 0x61 'a'
    [0b00000000,0b00000000,0b01111100,0b00000110,0b01111110,0b11000110,0b01111110,0b00000000],
    // 0x62 'b'
    [0b11000000,0b11000000,0b11111100,0b11000110,0b11000110,0b11000110,0b11111100,0b00000000],
    // 0x63 'c'
    [0b00000000,0b00000000,0b01111100,0b11000110,0b11000000,0b11000110,0b01111100,0b00000000],
    // 0x64 'd'
    [0b00000110,0b00000110,0b01111110,0b11000110,0b11000110,0b11000110,0b01111110,0b00000000],
    // 0x65 'e'
    [0b00000000,0b00000000,0b01111100,0b11000110,0b11111110,0b11000000,0b01111100,0b00000000],
    // 0x66 'f'
    [0b00011100,0b00110110,0b00110000,0b01111100,0b00110000,0b00110000,0b00110000,0b00000000],
    // 0x67 'g'
    [0b00000000,0b00000000,0b01111110,0b11000110,0b11000110,0b01111110,0b00000110,0b01111100],
    // 0x68 'h'
    [0b11000000,0b11000000,0b11111100,0b11000110,0b11000110,0b11000110,0b11000110,0b00000000],
    // 0x69 'i'
    [0b00110000,0b00000000,0b01110000,0b00110000,0b00110000,0b00110000,0b01111000,0b00000000],
    // 0x6A 'j'
    [0b00001100,0b00000000,0b00011100,0b00001100,0b00001100,0b11001100,0b11001100,0b01111000],
    // 0x6B 'k'
    [0b11000000,0b11000000,0b11001100,0b11011000,0b11110000,0b11011000,0b11001100,0b00000000],
    // 0x6C 'l'
    [0b01110000,0b00110000,0b00110000,0b00110000,0b00110000,0b00110000,0b01111000,0b00000000],
    // 0x6D 'm'
    [0b00000000,0b00000000,0b11101100,0b11111110,0b11010110,0b11000110,0b11000110,0b00000000],
    // 0x6E 'n'
    [0b00000000,0b00000000,0b11111100,0b11000110,0b11000110,0b11000110,0b11000110,0b00000000],
    // 0x6F 'o'
    [0b00000000,0b00000000,0b01111100,0b11000110,0b11000110,0b11000110,0b01111100,0b00000000],
    // 0x70 'p'
    [0b00000000,0b00000000,0b11111100,0b11000110,0b11000110,0b11111100,0b11000000,0b11000000],
    // 0x71 'q'
    [0b00000000,0b00000000,0b01111110,0b11000110,0b11000110,0b01111110,0b00000110,0b00000110],
    // 0x72 'r'
    [0b00000000,0b00000000,0b11011100,0b11100110,0b11000000,0b11000000,0b11000000,0b00000000],
    // 0x73 's'
    [0b00000000,0b00000000,0b01111110,0b11000000,0b01111100,0b00000110,0b11111100,0b00000000],
    // 0x74 't'
    [0b00110000,0b00110000,0b11111100,0b00110000,0b00110000,0b00110110,0b00011100,0b00000000],
    // 0x75 'u'
    [0b00000000,0b00000000,0b11000110,0b11000110,0b11000110,0b11000110,0b01111110,0b00000000],
    // 0x76 'v'
    [0b00000000,0b00000000,0b11000110,0b11000110,0b11000110,0b01101100,0b00111000,0b00000000],
    // 0x77 'w'
    [0b00000000,0b00000000,0b11000110,0b11000110,0b11010110,0b11111110,0b01101100,0b00000000],
    // 0x78 'x'
    [0b00000000,0b00000000,0b11000110,0b01101100,0b00111000,0b01101100,0b11000110,0b00000000],
    // 0x79 'y'
    [0b00000000,0b00000000,0b11000110,0b11000110,0b11000110,0b01111110,0b00000110,0b01111100],
    // 0x7A 'z'
    [0b00000000,0b00000000,0b11111110,0b00001100,0b00011000,0b01100000,0b11111110,0b00000000],
    // 0x7B '{'
    [0b00011100,0b00110000,0b00110000,0b01100000,0b00110000,0b00110000,0b00011100,0b00000000],
    // 0x7C '|'
    [0b00110000,0b00110000,0b00110000,0b00110000,0b00110000,0b00110000,0b00110000,0b00000000],
    // 0x7D '}'
    [0b01110000,0b00011000,0b00011000,0b00001100,0b00011000,0b00011000,0b01110000,0b00000000],
    // 0x7E '~'
    [0b01110110,0b11011100,0b00000000,0b00000000,0b00000000,0b00000000,0b00000000,0b00000000],
];
