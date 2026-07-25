//! One integration binary for the whole scope UI layer (table-driven where the
//! shape allows). Every test here is window-free: widgets emit a [`DrawList`] of
//! plain data against a mock [`Input`], so we assert on rects / glyph runs / clip
//! state / interaction outcomes without SDL. The backend's FFI is exercised only by
//! the headless `ui_demo --frames` run in CI, not here.

use streamcraft_scope::ui::arena::Arena;
use streamcraft_scope::ui::draw::{Color, DrawList, Rect, TexId};
use streamcraft_scope::ui::font::{Font, ATLAS_H, ATLAS_W};
use streamcraft_scope::ui::widgets::{LogView, UiState};
use streamcraft_scope::ui::{Id, Input, Key, Mods, MouseButton, Ui};

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

#[test]
fn arena_bump_and_reset() {
    let mut a = Arena::with_capacity(16);
    let s = a.alloc_slice::<u32>(4);
    assert_eq!(s.len(), 4);
    for (i, e) in s.iter_mut().enumerate() {
        *e = i as u32;
    }
    assert_eq!(a.used(), 16); // 4 * 4 bytes
    a.reset();
    assert_eq!(a.used(), 0);
    // Reuse after reset yields a fresh zeroed slice.
    let s2 = a.alloc_slice::<u32>(2);
    assert_eq!(s2, &[0, 0]);
}

#[test]
fn arena_grows_past_capacity() {
    let mut a = Arena::with_capacity(8);
    let s = a.alloc_slice::<u64>(10); // 80 bytes > 8
    assert_eq!(s.len(), 10);
    assert!(a.capacity() >= 80);
    assert_eq!(a.peak(), 80);
}

#[test]
fn arena_alignment_is_respected() {
    let mut a = Arena::with_capacity(64);
    // Force a misaligned head, then allocate a wider type.
    let _b = a.alloc_slice::<u8>(1);
    let w = a.alloc_slice::<u32>(1);
    let addr = w.as_ptr() as usize;
    assert_eq!(addr % std::mem::align_of::<u32>(), 0);
}

#[test]
fn arena_geometry_streams_are_disjoint_and_zeroed() {
    let mut a = Arena::with_capacity(4);
    let g = a.alloc_geometry(4, 6); // 4 verts, 6 indices
    assert_eq!(g.xy.len(), 8);
    assert_eq!(g.col.len(), 16);
    assert_eq!(g.uv.len(), 8);
    assert_eq!(g.idx.len(), 6);
    assert!(g.xy.iter().all(|&v| v == 0.0));
    assert!(g.idx.iter().all(|&v| v == 0));
    // Writing one stream does not corrupt the others.
    for e in g.xy.iter_mut() {
        *e = 1.0;
    }
    g.idx[0] = 42;
    assert!(g.col.iter().all(|&v| v == 0.0));
    assert!(g.uv.iter().all(|&v| v == 0.0));
    assert_eq!(g.idx[0], 42);
    assert_eq!(g.idx[1], 0);
}

// ---------------------------------------------------------------------------
// Draw list
// ---------------------------------------------------------------------------

#[test]
fn draw_fill_rect_records_one_prim() {
    let mut dl = DrawList::new();
    dl.fill_rect(Rect::new(1.0, 2.0, 3.0, 4.0), Color::rgb(10, 20, 30));
    assert_eq!(dl.len(), 1);
    let p = dl.prims()[0];
    assert_eq!(p.rect, Rect::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(p.color, Color::rgb(10, 20, 30));
    assert_eq!(p.tex, TexId::None);
    assert_eq!(p.clip, None);
}

#[test]
fn draw_transparent_solid_is_skipped() {
    let mut dl = DrawList::new();
    dl.fill_rect(Rect::new(0.0, 0.0, 5.0, 5.0), Color::TRANSPARENT);
    assert_eq!(dl.len(), 0);
}

#[test]
fn draw_rect_outline_is_four_bars() {
    let mut dl = DrawList::new();
    dl.rect_outline(Rect::new(0.0, 0.0, 20.0, 10.0), 2.0, Color::WHITE);
    // top, bottom, left, right
    assert_eq!(dl.len(), 4);
}

#[test]
fn draw_clip_stack_intersects_and_pops() {
    let mut dl = DrawList::new();
    dl.push_clip(Rect::new(0.0, 0.0, 100.0, 100.0));
    dl.push_clip(Rect::new(50.0, 50.0, 100.0, 100.0)); // clamped into parent
    let c = dl.current_clip().unwrap();
    assert_eq!(c, Rect::new(50.0, 50.0, 50.0, 50.0));
    dl.fill_rect(Rect::new(60.0, 60.0, 5.0, 5.0), Color::WHITE);
    assert_eq!(dl.prims()[0].clip, Some(Rect::new(50.0, 50.0, 50.0, 50.0)));
    dl.pop_clip();
    assert_eq!(dl.current_clip(), Some(Rect::new(0.0, 0.0, 100.0, 100.0)));
    dl.pop_clip();
    assert_eq!(dl.current_clip(), None);
}

#[test]
fn draw_line_axis_aligned_stays_thin() {
    let mut dl = DrawList::new();
    dl.line(0.0, 5.0, 30.0, 5.0, 2.0, Color::WHITE); // horizontal
    let p = dl.prims()[0];
    assert_eq!(p.rect.w, 30.0);
    assert_eq!(p.rect.h, 2.0);
    assert_eq!(p.rect.y, 4.0); // centred on y=5, half-thickness up
}

#[test]
fn draw_build_batches_by_texture_and_clip() {
    // Two untextured, then one textured quad, then untextured again => 3 batches
    // (texture changes break runs). All under no clip.
    let mut dl = DrawList::new();
    dl.fill_rect(Rect::new(0.0, 0.0, 1.0, 1.0), Color::WHITE);
    dl.fill_rect(Rect::new(1.0, 0.0, 1.0, 1.0), Color::WHITE);
    dl.textured_quad(Rect::new(2.0, 0.0, 1.0, 1.0), Rect::new(0.0, 0.0, 0.5, 0.5), Color::WHITE);
    dl.fill_rect(Rect::new(3.0, 0.0, 1.0, 1.0), Color::WHITE);

    let mut arena = Arena::with_capacity(1024);
    let batches = dl.build(&mut arena);
    assert_eq!(batches.len(), 3);
    assert_eq!(batches[0].tex, TexId::None);
    assert_eq!(batches[0].indices.len(), 12); // two quads
    assert_eq!(batches[1].tex, TexId::Font);
    assert_eq!(batches[1].indices.len(), 6);
    assert_eq!(batches[2].tex, TexId::None);
    // Vertex count is shared across batches (indices are absolute).
    assert_eq!(batches[0].num_vertices, 16); // 4 prims * 4 verts
}

#[test]
fn draw_build_empty_is_no_batches() {
    let dl = DrawList::new();
    let mut arena = Arena::with_capacity(16);
    assert!(dl.build(&mut arena).is_empty());
}

#[test]
fn draw_build_quad_winding_and_positions() {
    let mut dl = DrawList::new();
    dl.fill_rect(Rect::new(10.0, 20.0, 4.0, 6.0), Color::rgba(255, 0, 0, 255));
    let mut arena = Arena::with_capacity(256);
    let b = dl.build(&mut arena).pop().unwrap();
    // 4 verts: TL, TR, BR, BL.
    assert_eq!(&b.xy[0..2], &[10.0, 20.0]);
    assert_eq!(&b.xy[2..4], &[14.0, 20.0]);
    assert_eq!(&b.xy[4..6], &[14.0, 26.0]);
    assert_eq!(&b.xy[6..8], &[10.0, 26.0]);
    // Two triangles 0-1-2, 0-2-3.
    assert_eq!(b.indices, &[0, 1, 2, 0, 2, 3]);
    // Colour is red, full alpha.
    assert_eq!(&b.color[0..4], &[1.0, 0.0, 0.0, 1.0]);
}

// ---------------------------------------------------------------------------
// Color / Rect helpers
// ---------------------------------------------------------------------------

#[test]
fn color_to_f32_is_normalised() {
    assert_eq!(Color::rgb(255, 0, 0).to_f32(), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(Color::rgba(0, 0, 0, 128).to_f32()[3], 128.0 / 255.0);
}

#[test]
fn rect_contains_is_half_open() {
    let r = Rect::new(0.0, 0.0, 10.0, 10.0);
    assert!(r.contains(0.0, 0.0));
    assert!(r.contains(9.9, 9.9));
    assert!(!r.contains(10.0, 5.0)); // right edge excluded
    assert!(!r.contains(-0.1, 5.0));
}

#[test]
fn rect_intersect_disjoint_is_zero_size() {
    let a = Rect::new(0.0, 0.0, 5.0, 5.0);
    let b = Rect::new(10.0, 10.0, 5.0, 5.0);
    let i = a.intersect(&b);
    assert_eq!(i.w, 0.0);
    assert_eq!(i.h, 0.0);
}

// ---------------------------------------------------------------------------
// Id hashing
// ---------------------------------------------------------------------------

#[test]
fn id_is_stable_and_label_sensitive() {
    assert_eq!(Id::new(0, "play"), Id::new(0, "play"));
    assert_ne!(Id::new(0, "play"), Id::new(0, "pause"));
    // Different parent seed => different id for the same label (nesting).
    assert_ne!(Id::new(1, "row"), Id::new(2, "row"));
}

// ---------------------------------------------------------------------------
// Font
// ---------------------------------------------------------------------------

#[test]
fn font_measure_single_line() {
    // Monospace: width is exactly chars × advance, height one line, at any size.
    let f = Font::new();
    let (w, h) = f.measure("ABCD");
    assert_eq!(w, 4.0 * f.cell_w());
    assert_eq!(h, f.line_h());
    assert_eq!(f.measure_line("hi"), 2.0 * f.cell_w());
}

#[test]
fn font_measure_multiline() {
    let f = Font::new();
    let (w, h) = f.measure("ab\ncdef\ng");
    assert_eq!(w, 4.0 * f.cell_w()); // longest line "cdef"
    assert_eq!(h, 3.0 * f.line_h()); // 3 lines
}

#[test]
fn font_measure_scales_linearly_with_px() {
    // measure_px at 2× the size is exactly 2× as wide (monospace advance is
    // linear in px: snapped size × residual ratio).
    let f = Font::new();
    let (w14, _) = f.measure_px("stream", 14.0);
    let (w28, _) = f.measure_px("stream", 28.0);
    // Within 1% — different baked sizes rasterise to slightly different ratios.
    assert!((w28 - 2.0 * w14).abs() < 0.01 * w28, "w14={w14} w28={w28}");
    // Unbaked sizes interpolate: strictly between the neighbours' widths.
    let (w20, _) = f.measure_px("stream", 20.0);
    let (w18, _) = f.measure_px("stream", 18.0);
    let (w24, _) = f.measure_px("stream", 24.0);
    assert!(w18 < w20 && w20 < w24);
}

#[test]
fn font_atlas_dimensions() {
    let f = Font::new();
    assert_eq!(f.atlas_size(), (ATLAS_W, ATLAS_H));
    assert_eq!(f.atlas_rgba().len(), ATLAS_W * ATLAS_H * 4);
}

/// Fraction of texels with nonzero alpha inside a quad's uv rect.
fn uv_coverage(f: &Font, uv: Rect) -> f32 {
    let atlas = f.atlas_rgba();
    let (x0, y0) = ((uv.x * ATLAS_W as f32) as usize, (uv.y * ATLAS_H as f32) as usize);
    let (w, h) = ((uv.w * ATLAS_W as f32) as usize, (uv.h * ATLAS_H as f32) as usize);
    let mut lit = 0usize;
    for py in y0..(y0 + h).min(ATLAS_H) {
        for px in x0..(x0 + w).min(ATLAS_W) {
            if atlas[(py * ATLAS_W + px) * 4 + 3] != 0 {
                lit += 1;
            }
        }
    }
    lit as f32 / (w * h).max(1) as f32
}

#[test]
fn font_glyphs_have_ink_and_distinct_uvs() {
    // Each printable glyph's quad must point at atlas texels with real coverage,
    // and different glyphs at different atlas regions.
    let f = Font::new();
    let mut dl = DrawList::new();
    f.layout_into(&mut dl, "AW.", 0.0, 0.0, 1.0, Color::WHITE);
    assert_eq!(dl.len(), 3);
    let uvs: Vec<Rect> = dl.prims().iter().map(|p| p.uv).collect();
    for uv in &uvs {
        assert!(uv_coverage(&f, *uv) > 0.15, "glyph uv region looks blank");
    }
    assert!(uvs[0].x != uvs[1].x || uvs[0].y != uvs[1].y);
    assert!(uvs[1].x != uvs[2].x || uvs[1].y != uvs[2].y);
}

#[test]
fn font_space_emits_no_quad_but_advances() {
    let f = Font::new();
    let mut dl = DrawList::new();
    // "a b" -> two glyph quads (a, b), the space emits nothing.
    let (pen_x, _) = f.layout_into(&mut dl, "a b", 0.0, 0.0, 1.0, Color::WHITE);
    assert_eq!(dl.len(), 2);
    assert_eq!(pen_x, 3.0 * f.cell_w()); // pen advanced past all 3 cells
}

#[test]
fn font_non_ascii_uses_replacement_box() {
    let f = Font::new();
    let mut dl = DrawList::new();
    // A non-ASCII char maps to the replacement box glyph (index 0), which is not
    // blank, so it emits a quad.
    f.layout_into(&mut dl, "é", 0.0, 0.0, 1.0, Color::WHITE);
    assert_eq!(dl.len(), 1);
    assert_eq!(dl.prims()[0].tex, TexId::Font);
}

#[test]
fn font_glyph_quads_carry_bearings() {
    // An AA glyph quad sits at pen + bearing and is narrower than the advance
    // (JetBrains Mono side bearings) — i.e. we are not drawing full cells.
    let f = Font::new();
    let mut dl = DrawList::new();
    f.layout_into(&mut dl, "o", 10.0, 20.0, 1.0, Color::WHITE);
    let q = dl.prims()[0].rect;
    assert!(q.x >= 10.0, "bearing must not move ink left of the pen");
    assert!(q.y > 20.0, "lowercase 'o' starts below the line-box top");
    assert!(q.w < f.cell_w(), "ink narrower than the advance");
}

// ---------------------------------------------------------------------------
// Widgets — pure interaction against a mock Input
// ---------------------------------------------------------------------------

/// A mock input placing the mouse at `(x, y)` with the left button in the given
/// edge state.
fn mock_input(x: f32, y: f32) -> Input {
    Input {
        mouse_x: x,
        mouse_y: y,
        window_w: 400.0,
        window_h: 300.0,
        ..Input::default()
    }
}

fn press_left(i: &mut Input) {
    i.mouse_down[0] = true;
    i.mouse_pressed[0] = true;
}
fn release_left(i: &mut Input) {
    i.mouse_down[0] = false;
    i.mouse_released[0] = true;
}

#[test]
fn widget_label_emits_text_prims() {
    let font = Font::new();
    let input = mock_input(-1.0, -1.0);
    let mut state = UiState::default();
    let mut ui = Ui::new(&input, &font, &mut state, (400.0, 300.0));
    ui.begin_panel("Panel", Rect::new(0.0, 0.0, 200.0, 200.0));
    let before = ui.prim_count();
    ui.label("Hi");
    let after = ui.prim_count();
    // At least the two glyph quads for "Hi".
    assert!(after - before >= 2, "label should emit glyph quads");
    ui.end_panel();
    // The finished draw list has a font batch somewhere.
    let dl = ui.finish();
    assert!(dl.prims().iter().any(|p| p.tex == TexId::Font));
}

#[test]
fn widget_button_click_requires_press_and_release_inside() {
    let font = Font::new();
    let mut state = UiState::default();

    // Frame 1: press inside the button (button occupies a known rect in a panel).
    // The panel body starts below the title bar; place the mouse well inside.
    let panel = Rect::new(0.0, 0.0, 200.0, 200.0);

    // Press.
    let mut input = mock_input(30.0, 40.0);
    press_left(&mut input);
    let clicked_press = {
        let mut ui = Ui::new(&input, &font, &mut state, (400.0, 300.0));
        ui.begin_panel("P", panel);
        let c = ui.button("Go");
        ui.end_panel();
        c
    };
    assert!(!clicked_press, "press alone is not a click");

    // Release on the same spot => click fires.
    let mut input2 = mock_input(30.0, 40.0);
    release_left(&mut input2);
    let clicked_release = {
        let mut ui = Ui::new(&input2, &font, &mut state, (400.0, 300.0));
        ui.begin_panel("P", panel);
        let c = ui.button("Go");
        ui.end_panel();
        c
    };
    assert!(clicked_release, "press then release inside should click");
}

#[test]
fn widget_button_no_click_when_released_outside() {
    let font = Font::new();
    let mut state = UiState::default();
    let panel = Rect::new(0.0, 0.0, 200.0, 200.0);

    let mut input = mock_input(30.0, 40.0);
    press_left(&mut input);
    {
        let mut ui = Ui::new(&input, &font, &mut state, (400.0, 300.0));
        ui.begin_panel("P", panel);
        let _ = ui.button("Go");
        ui.end_panel();
    }
    // Release far away.
    let mut input2 = mock_input(999.0, 999.0);
    release_left(&mut input2);
    let clicked = {
        let mut ui = Ui::new(&input2, &font, &mut state, (400.0, 300.0));
        ui.begin_panel("P", panel);
        let c = ui.button("Go");
        ui.end_panel();
        c
    };
    assert!(!clicked, "release outside should not click");
}

#[test]
fn widget_toggle_flips_on_click() {
    let font = Font::new();
    let mut state = UiState::default();
    let panel = Rect::new(0.0, 0.0, 200.0, 200.0);
    let mut value = false;

    // Press then release over the toggle row.
    let mut down = mock_input(20.0, 40.0);
    press_left(&mut down);
    {
        let mut ui = Ui::new(&down, &font, &mut state, (400.0, 300.0));
        ui.begin_panel("P", panel);
        ui.toggle("flag", &mut value);
        ui.end_panel();
    }
    assert!(!value, "not toggled until release");

    let mut up = mock_input(20.0, 40.0);
    release_left(&mut up);
    let changed = {
        let mut ui = Ui::new(&up, &font, &mut state, (400.0, 300.0));
        ui.begin_panel("P", panel);
        let ch = ui.toggle("flag", &mut value);
        ui.end_panel();
        ch
    };
    assert!(changed);
    assert!(value, "toggle flips to true on click");
}

#[test]
fn widget_tab_strip_selects_on_click() {
    let font = Font::new();
    let mut state = UiState::default();
    let mut selected = 0usize;
    let tabs = ["A", "B", "C"];

    // Panel is 300 wide; body ~ 288 wide; three tabs ~ 96 px each. Click near the
    // middle of the second tab. Press then release.
    let panel = Rect::new(0.0, 0.0, 300.0, 200.0);
    let click_x = 6.0 + 96.0 + 40.0; // into tab index 1
    let click_y = 40.0;

    let mut down = mock_input(click_x, click_y);
    press_left(&mut down);
    {
        let mut ui = Ui::new(&down, &font, &mut state, (400.0, 300.0));
        ui.begin_panel("Tabs", panel);
        ui.tab_strip(&tabs, &mut selected);
        ui.end_panel();
    }
    let mut up = mock_input(click_x, click_y);
    release_left(&mut up);
    {
        let mut ui = Ui::new(&up, &font, &mut state, (400.0, 300.0));
        ui.begin_panel("Tabs", panel);
        ui.tab_strip(&tabs, &mut selected);
        ui.end_panel();
    }
    assert_eq!(selected, 1, "clicking the second tab selects index 1");
}

#[test]
fn widget_fill_bar_clamps_and_overlays() {
    let font = Font::new();
    let input = mock_input(-1.0, -1.0);
    let mut state = UiState::default();
    let mut ui = Ui::new(&input, &font, &mut state, (400.0, 300.0));
    // Over-1.0 fraction is clamped: the fill quad must not exceed the bar width.
    let bar = Rect::new(0.0, 0.0, 100.0, 16.0);
    ui.fill_bar_in(bar, 2.0, Color::rgb(0, 255, 0), "x");
    let dl = ui.finish();
    // Find the green fill quad (colour matches, tex None).
    let fill = dl
        .prims()
        .iter()
        .find(|p| p.color == Color::rgb(0, 255, 0) && p.tex == TexId::None)
        .expect("fill quad present");
    assert!(fill.rect.w <= bar.w + 0.01, "clamped fill width");
}

// ---------------------------------------------------------------------------
// LogView — scroll & stick-to-bottom
// ---------------------------------------------------------------------------

#[test]
fn log_view_bounded_ring_evicts_oldest() {
    let mut log = LogView::new(3);
    for i in 0..5 {
        log.push(format!("line {i}"), Color::WHITE);
    }
    assert_eq!(log.len(), 3);
}

#[test]
fn log_view_sticks_to_bottom_by_default() {
    let mut log = LogView::new(100);
    for i in 0..50 {
        log.push(format!("l{i}"), Color::WHITE);
    }
    assert!(log.is_stuck());
}

#[test]
fn log_view_scroll_up_unsticks_then_back_resticks() {
    let mut log = LogView::new(100);
    for i in 0..50 {
        log.push(format!("l{i}"), Color::WHITE);
    }
    let visible = 10;
    // Scroll up (positive wheel delta => older lines) unsticks.
    log.scroll_by(5, visible);
    assert!(!log.is_stuck(), "scrolling up unsticks");
    assert!(log.offset() < 40);
    // Scroll all the way back down re-sticks.
    log.scroll_by(-100, visible);
    assert!(log.is_stuck(), "scrolling to the tail re-sticks");
    assert_eq!(log.offset(), 40); // 50 - 10 visible
}

#[test]
fn log_view_renders_visible_tail() {
    let font = Font::new();
    let input = mock_input(-1.0, -1.0);
    let mut state = UiState::default();
    let mut log = LogView::new(100);
    for i in 0..30 {
        log.push(format!("line{i}"), Color::WHITE);
    }
    let mut ui = Ui::new(&input, &font, &mut state, (400.0, 300.0));
    // A short viewport: 3 lines tall (line height 8 => ~24px inner content).
    ui.log_view_in(Rect::new(0.0, 0.0, 200.0, 30.0), &mut log);
    let dl = ui.finish();
    // Some glyph quads emitted for the tail lines (clipped).
    assert!(dl.prims().iter().any(|p| p.tex == TexId::Font));
}

// ---------------------------------------------------------------------------
// Input snapshot semantics
// ---------------------------------------------------------------------------

#[test]
fn input_begin_frame_clears_edges_keeps_levels() {
    let mut i = Input {
        mouse_x: 5.0,
        mouse_down: [true, false, false],
        wheel: 3.0,
        window_w: 640.0,
        ..Input::default()
    };
    press_left(&mut i);
    i.text.push_str("abc");
    i.keys.push((Key::Enter, Mods::default()));

    i.begin_frame();
    // Edges cleared.
    assert!(!i.pressed(MouseButton::Left));
    assert_eq!(i.wheel, 0.0);
    assert!(i.text.is_empty());
    assert!(i.keys.is_empty());
    // Levels kept.
    assert_eq!(i.mouse_x, 5.0);
    assert!(i.down(MouseButton::Left));
    assert_eq!(i.window_w, 640.0);
}

#[test]
fn input_key_pressed_lookup() {
    let mut i = Input::default();
    i.keys.push((Key::Escape, Mods { ctrl: true, ..Default::default() }));
    assert!(i.key_pressed(Key::Escape));
    assert!(!i.key_pressed(Key::Enter));
}

// ---------------------------------------------------------------------------
// App: pure camera / hit-test / formatting helpers
// ---------------------------------------------------------------------------

use streamcraft_scope::app::{dist_point_segment, fit_camera, fmt_time, zoom_at};

#[test]
fn point_segment_distance() {
    // Perpendicular foot inside the segment.
    assert_eq!(dist_point_segment((5.0, 3.0), (0.0, 0.0), (10.0, 0.0)), 3.0);
    // Beyond an endpoint: distance to the endpoint.
    assert_eq!(dist_point_segment((13.0, 4.0), (0.0, 0.0), (10.0, 0.0)), 5.0);
    // Degenerate zero-length segment.
    assert_eq!(dist_point_segment((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)), 5.0);
}

#[test]
fn zoom_keeps_anchor_stationary() {
    // The content point under the anchor must stay under it across a zoom step:
    // screen = pan + p*zoom; solve p at the anchor before and after.
    let (pan, zoom) = ((30.0, -12.0), 0.8);
    let anchor = (100.0, 60.0);
    let p_before = ((anchor.0 - pan.0) / zoom, (anchor.1 - pan.1) / zoom);
    let ((px, py), z2) = zoom_at(pan, zoom, anchor, 1.25);
    let p_after = ((anchor.0 - px) / z2, (anchor.1 - py) / z2);
    assert!((p_before.0 - p_after.0).abs() < 1e-3);
    assert!((p_before.1 - p_after.1).abs() < 1e-3);
    assert!((z2 - 1.0).abs() < 1e-6);
}

#[test]
fn zoom_clamps_to_range() {
    let ((_, _), z) = zoom_at((0.0, 0.0), 2.9, (0.0, 0.0), 10.0);
    assert_eq!(z, 3.0);
    let ((_, _), z) = zoom_at((0.0, 0.0), 0.3, (0.0, 0.0), 0.01);
    assert_eq!(z, 0.2);
}

#[test]
fn fit_camera_centres_content() {
    // A 100x50 content box in a 400x300 body: fully visible, centred.
    let ((px, py), z) = fit_camera((10.0, 20.0, 110.0, 70.0), 400.0, 300.0);
    for (cx, cy) in [(10.0, 20.0), (110.0, 70.0), (60.0, 45.0)] {
        let sx = px + cx * z;
        let sy = py + cy * z;
        assert!((0.0..=400.0).contains(&sx), "x {sx} out of body");
        assert!((0.0..=300.0).contains(&sy), "y {sy} out of body");
    }
    // Centre of content lands at the centre of the body.
    let (sx, sy) = (px + 60.0 * z, py + 45.0 * z);
    assert!((sx - 200.0).abs() < 1e-3 && (sy - 150.0).abs() < 1e-3);
}

#[test]
fn time_formatting() {
    assert_eq!(fmt_time(0), "0:00");
    assert_eq!(fmt_time(59_500_000_000), "0:59");
    assert_eq!(fmt_time(83_000_000_000), "1:23");
    assert_eq!(fmt_time(3_723_000_000_000), "1:02:03");
}

#[test]
fn draw_diagonal_line_is_a_rotated_quad() {
    // A 45° line must emit an actual rotated quad, not its bounding rect (the
    // "gray rectangle" fan-out bug): corners sit ±t/2 along the line normal.
    let mut dl = DrawList::new();
    dl.line(0.0, 0.0, 10.0, 10.0, 2.0, Color::WHITE);
    let p = dl.prims()[0];
    let c = p.corners.expect("diagonal line must carry explicit corners");
    let inv = std::f32::consts::FRAC_1_SQRT_2; // unit normal component for 45°
    assert!((c[0].0 - -inv).abs() < 1e-4 && (c[0].1 - inv).abs() < 1e-4);
    assert!((c[2].0 - (10.0 + inv)).abs() < 1e-4 && (c[2].1 - (10.0 - inv)).abs() < 1e-4);
    // Opposite edges stay `thickness` apart: |c0 - c3| == 2.
    let d = ((c[0].0 - c[3].0).powi(2) + (c[0].1 - c[3].1).powi(2)).sqrt();
    assert!((d - 2.0).abs() < 1e-4);
    // Bbox covers the quad.
    assert!(p.rect.x <= -inv && p.rect.right() >= 10.0 + inv);
    // Axis-aligned lines keep the fast path (no corners).
    let mut dl2 = DrawList::new();
    dl2.line(0.0, 5.0, 30.0, 5.0, 2.0, Color::WHITE);
    assert!(dl2.prims()[0].corners.is_none());
}

#[test]
fn draw_build_rotated_quad_vertices() {
    let mut dl = DrawList::new();
    dl.line(0.0, 0.0, 8.0, 6.0, 2.0, Color::WHITE); // 3-4-5 diagonal
    let mut arena = Arena::with_capacity(256);
    let b = dl.build(&mut arena).pop().unwrap();
    // Vertices are the rotated corners (normal = (-0.6, 0.8)): v0 = a + n,
    // v3 = a - n.
    assert!((b.xy[0] - -0.6).abs() < 1e-4 && (b.xy[1] - 0.8).abs() < 1e-4);
    assert!((b.xy[6] - 0.6).abs() < 1e-4 && (b.xy[7] - -0.8).abs() < 1e-4);
    assert_eq!(b.indices, &[0, 1, 2, 0, 2, 3]);
}
