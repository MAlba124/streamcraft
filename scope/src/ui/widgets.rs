//! The immediate-mode context [`Ui`] and the scope's core widgets.
//!
//! Each frame the app makes a fresh [`Ui`] over an [`Input`] snapshot, calls
//! widgets (which append to the [`DrawList`] and read the input), then takes the
//! finished draw list back with [`Ui::finish`]. Widget *identity* comes from the
//! call-site label ([`Id`]); the context tracks a single **hot** id (hovered) and
//! **active** id (mouse-captured, e.g. a button being held) across the frame — the
//! standard immediate-mode interaction bookkeeping (Muratori 2005).
//!
//! Retained state that must outlive a frame (toggle values, selected tab, log
//! scroll position) lives in the *caller* and is passed by `&mut` — the
//! immediate-mode discipline. The `Ui` itself owns nothing across frames.

use crate::ui::draw::{Color, DrawList, Rect};
use crate::ui::font::Font;
use crate::ui::{Id, Input, MouseButton};

/// Colours and metrics for the inspector's dark theme. Plain data; a caller can
/// tweak a field before building the frame.
#[derive(Clone, Copy, Debug)]
pub struct Theme {
    pub bg: Color,
    pub panel_bg: Color,
    pub panel_border: Color,
    pub title_bg: Color,
    pub text: Color,
    pub text_dim: Color,
    pub accent: Color,
    pub button: Color,
    pub button_hot: Color,
    pub button_active: Color,
    pub bar_bg: Color,
    pub bar_fill: Color,
    /// Padding inside a panel body.
    pub pad: f32,
    /// Height of one row (label / kv / button).
    pub row_h: f32,
    /// Height of a panel title bar.
    pub title_h: f32,
    /// Text scale (integer multiple of the 8px cell).
    pub text_scale: f32,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            bg: Color::rgb(0x14, 0x16, 0x1a),
            panel_bg: Color::rgb(0x1e, 0x22, 0x28),
            panel_border: Color::rgb(0x33, 0x3a, 0x44),
            title_bg: Color::rgb(0x2a, 0x30, 0x3a),
            text: Color::rgb(0xd8, 0xdd, 0xe4),
            text_dim: Color::rgb(0x8a, 0x93, 0xa0),
            accent: Color::rgb(0x4a, 0xa8, 0xff),
            button: Color::rgb(0x33, 0x3a, 0x46),
            button_hot: Color::rgb(0x42, 0x4c, 0x5c),
            button_active: Color::rgb(0x27, 0x5a, 0x8c),
            bar_bg: Color::rgb(0x2a, 0x30, 0x3a),
            bar_fill: Color::rgb(0x4a, 0xa8, 0xff),
            pad: 6.0,
            row_h: 20.0,
            title_h: 22.0,
            text_scale: 1.0,
        }
    }
}

/// A vertical layout cursor inside a rectangle. Widgets advance it downward; a
/// simple row/column model, no constraint solver (per the spec).
#[derive(Clone, Copy, Debug)]
pub struct Cursor {
    /// The content rect the cursor lays out within.
    pub area: Rect,
    /// Current pen y (top of the next row).
    pub y: f32,
    /// Left edge / indent.
    pub x: f32,
    /// Gap inserted between rows.
    pub gap: f32,
}

impl Cursor {
    fn new(area: Rect, gap: f32) -> Self {
        Self { area, y: area.y, x: area.x, gap }
    }
    /// Reserve a full-width row `h` px tall and advance past it.
    fn row(&mut self, h: f32) -> Rect {
        let r = Rect::new(self.x, self.y, (self.area.right() - self.x).max(0.0), h);
        self.y += h + self.gap;
        r
    }
    /// Remaining vertical space below the cursor.
    fn remaining_h(&self) -> f32 {
        (self.area.bottom() - self.y).max(0.0)
    }
}

/// The result of an interactive widget: was it clicked / is it hovered / held.
#[derive(Clone, Copy, Debug, Default)]
pub struct Response {
    pub hovered: bool,
    pub held: bool,
    /// A full click: pressed and released over the widget this interaction.
    pub clicked: bool,
}

/// Persistent interaction state the caller carries between frames (there is no
/// retained widget tree, so hot/active must survive as plain fields). One per `Ui`
/// session; the demo keeps a single instance in its app state.
#[derive(Clone, Copy, Debug, Default)]
pub struct UiState {
    /// The id the mouse is currently captured by (e.g. button held down).
    active: Option<Id>,
}

/// Per-frame immediate-mode context.
pub struct Ui<'a> {
    input: &'a Input,
    font: &'a Font,
    theme: Theme,
    dl: DrawList,
    /// Screen rect (0,0,w,h).
    screen: Rect,
    /// Hovered id this frame (computed as widgets are visited, last-wins == topmost
    /// since z is submission order).
    hot: Option<Id>,
    state: &'a mut UiState,
    /// Layout-cursor stack: panels push a body cursor, widgets use the top.
    cursors: Vec<Cursor>,
    /// Seed for id hashing, mixed from the enclosing panel so labels nest.
    id_seed: u64,
}

impl<'a> Ui<'a> {
    /// Begin a frame. `size` is the window size in logical pixels.
    pub fn new(input: &'a Input, font: &'a Font, state: &'a mut UiState, size: (f32, f32)) -> Self {
        Ui {
            input,
            font,
            theme: Theme::default(),
            dl: DrawList::new(),
            screen: Rect::new(0.0, 0.0, size.0, size.1),
            hot: None,
            state,
            cursors: Vec::new(),
            id_seed: 0,
        }
    }

    /// Override the default theme.
    pub fn with_theme(mut self, theme: Theme) -> Self {
        self.theme = theme;
        self
    }

    pub fn theme(&self) -> &Theme {
        &self.theme
    }
    pub fn font(&self) -> &Font {
        self.font
    }
    pub fn input(&self) -> &Input {
        self.input
    }
    pub fn screen(&self) -> Rect {
        self.screen
    }
    /// Read-only access to the draw list (tests, diagnostics).
    pub fn draw_list(&self) -> &DrawList {
        &self.dl
    }
    /// Mutable draw-list access for bespoke drawing (graph panel later).
    pub fn draw_list_mut(&mut self) -> &mut DrawList {
        &mut self.dl
    }

    /// Fill the whole screen with the theme background — call once at frame start.
    pub fn clear_background(&mut self) {
        self.dl.fill_rect(self.screen, self.theme.bg);
    }

    /// Finish the frame and take the accumulated draw list.
    pub fn finish(self) -> DrawList {
        self.dl
    }

    fn id(&self, label: &str) -> Id {
        // The full salted label (incl. any `##salt`) feeds identity; only the
        // display part is drawn.
        Id::new(self.id_seed, strip_salt(label).1)
    }

    /// The layout cursor the current widget should use.
    fn cursor(&mut self) -> &mut Cursor {
        self.cursors.last_mut().expect("widget called outside a panel/layout scope")
    }

    /// Reserve the next full-width row `h` px tall from the current layout cursor.
    /// A thin wrapper so callers compute `h` (which usually reads `self.theme`)
    /// *before* borrowing the cursor — sidestepping a self-borrow conflict.
    fn next_row(&mut self, h: f32) -> Rect {
        self.cursor().row(h)
    }

    // ---- text helpers -------------------------------------------------------

    /// Draw a line of text top-left at `(x, y)`; returns its pixel width.
    pub fn text(&mut self, x: f32, y: f32, s: &str, color: Color) -> f32 {
        let scale = self.theme.text_scale;
        self.font.layout_into(&mut self.dl, s, x, y, scale, color);
        self.font.measure_line(s) * scale
    }

    /// Draw a line of text at an explicit pixel size (the graph view zooms text
    /// with its content, bypassing the theme scale); returns its pixel width.
    pub fn text_px(&mut self, x: f32, y: f32, s: &str, px: f32, color: Color) -> f32 {
        self.font.layout_px(&mut self.dl, s, x, y, px, color);
        self.font.measure_px(s, px).0
    }

    /// Shift the current panel body's layout cursor up by `off` px — the hook for
    /// caller-managed wheel scrolling of overflowing panel content (the panel clip
    /// pushed by [`Ui::begin_panel`] hides what moves out of the body).
    pub fn scroll_body(&mut self, off: f32) {
        if let Some(c) = self.cursors.last_mut() {
            c.y -= off;
        }
    }

    /// Vertically centre a line of text within `row` (left-aligned at `row.x + pad`).
    fn text_in_row(&mut self, row: Rect, s: &str, color: Color, pad: f32) {
        let scale = self.theme.text_scale;
        let ty = row.y + (row.h - self.font.line_h() * scale) * 0.5;
        self.font.layout_into(&mut self.dl, s, row.x + pad, ty.floor(), scale, color);
    }

    /// Right-align a line of text within `row`.
    fn text_in_row_right(&mut self, row: Rect, s: &str, color: Color, pad: f32) {
        let scale = self.theme.text_scale;
        let w = self.font.measure_line(s) * scale;
        let tx = row.right() - pad - w;
        let ty = row.y + (row.h - self.font.line_h() * scale) * 0.5;
        self.font.layout_into(&mut self.dl, s, tx.floor(), ty.floor(), scale, color);
    }

    // ---- interaction core ---------------------------------------------------

    /// Standard immediate-mode interaction over `rect` with identity `id`. Updates
    /// hot/active and returns hovered/held/clicked. `clip` limits hit-testing to a
    /// visible region (log view rows outside the viewport don't react).
    fn interact(&mut self, id: Id, rect: Rect, clip: Option<Rect>) -> Response {
        let (mx, my) = (self.input.mouse_x, self.input.mouse_y);
        let inside = rect.contains(mx, my) && clip.map(|c| c.contains(mx, my)).unwrap_or(true);
        let mut resp = Response::default();
        if inside {
            self.hot = Some(id);
            resp.hovered = true;
        }
        if self.state.active == Some(id) {
            resp.held = true;
            if self.input.released(MouseButton::Left) {
                if inside {
                    resp.clicked = true;
                }
                self.state.active = None;
            }
        } else if inside && self.input.pressed(MouseButton::Left) {
            self.state.active = Some(id);
            resp.held = true;
        }
        resp
    }

    // ---- panels & layout ----------------------------------------------------

    /// Begin a titled panel occupying `rect`. Draws the frame + title bar and pushes
    /// a body layout cursor; body widgets called until [`Ui::end_panel`] lay out
    /// vertically inside. Returns the body content rect.
    pub fn begin_panel(&mut self, title: &str, rect: Rect) -> Rect {
        let t = self.theme;
        self.dl.fill_rect(rect, t.panel_bg);
        self.dl.rect_outline(rect, 1.0, t.panel_border);
        // Title bar.
        let title_bar = Rect::new(rect.x, rect.y, rect.w, t.title_h);
        self.dl.fill_rect(title_bar, t.title_bg);
        self.text_in_row(title_bar, strip_salt(title).0, t.text, t.pad);
        // Body cursor.
        let body = Rect::new(
            rect.x + t.pad,
            rect.y + t.title_h + t.pad,
            (rect.w - 2.0 * t.pad).max(0.0),
            (rect.h - t.title_h - 2.0 * t.pad).max(0.0),
        );
        self.dl.push_clip(Rect::new(rect.x, rect.y + t.title_h, rect.w, (rect.h - t.title_h).max(0.0)));
        self.cursors.push(Cursor::new(body, 4.0));
        // Nest ids under this panel's title.
        self.id_seed ^= Id::new(0, strip_salt(title).0).0;
        body
    }

    /// End the current panel (pops its clip + layout cursor).
    pub fn end_panel(&mut self) {
        self.cursors.pop();
        self.dl.pop_clip();
        // Un-nest: xor is its own inverse, but we simply reset to 0 at frame top
        // between panels; here restore by re-xoring is unnecessary because panels
        // are not nested in this UI. Reset the seed to the frame root.
        self.id_seed = 0;
    }

    // ---- widgets ------------------------------------------------------------

    /// A plain left-aligned text label occupying one row.
    pub fn label(&mut self, s: &str) {
        let h = self.theme.row_h;
        let row = self.next_row(h);
        let (c, p) = (self.theme.text, self.theme.pad);
        self.text_in_row(row, s, c, p);
    }

    /// A dim sub-heading label.
    pub fn label_dim(&mut self, s: &str) {
        let h = self.theme.row_h;
        let row = self.next_row(h);
        let (c, p) = (self.theme.text_dim, self.theme.pad);
        self.text_in_row(row, s, c, p);
    }

    /// A key/value row: dim `key` on the left, bright `value` on the right.
    pub fn kv(&mut self, key: &str, value: &str) {
        let h = self.theme.row_h;
        let row = self.next_row(h);
        let t = self.theme;
        self.text_in_row(row, key, t.text_dim, t.pad);
        self.text_in_row_right(row, value, t.text, t.pad);
    }

    /// A fill bar showing `frac` (clamped 0..1) in `color`, with `overlay` text
    /// centred on top. One row tall.
    pub fn fill_bar(&mut self, frac: f32, color: Color, overlay: &str) {
        let h = self.theme.row_h;
        let row = self.next_row(h);
        self.fill_bar_in(row, frac, color, overlay);
    }

    /// Draw a fill bar into an explicit rect (used by the panel body and by the log
    /// scrollbar). Separated so it is directly unit-testable against a rect.
    pub fn fill_bar_in(&mut self, rect: Rect, frac: f32, color: Color, overlay: &str) {
        let t = self.theme;
        let f = frac.clamp(0.0, 1.0);
        self.dl.fill_rect(rect, t.bar_bg);
        if f > 0.0 {
            self.dl.fill_rect(Rect::new(rect.x, rect.y, rect.w * f, rect.h), color);
        }
        self.dl.rect_outline(rect, 1.0, t.panel_border);
        if !overlay.is_empty() {
            let scale = t.text_scale;
            let tw = self.font.measure_line(overlay) * scale;
            let tx = rect.x + (rect.w - tw) * 0.5;
            let ty = rect.y + (rect.h - self.font.line_h() * scale) * 0.5;
            self.font.layout_into(&mut self.dl, overlay, tx.floor(), ty.floor(), scale, t.text);
        }
    }

    /// A clickable button spanning one row. Returns true on click (press+release
    /// inside). Hot/active colouring follows the theme.
    pub fn button(&mut self, label: &str) -> bool {
        let h = self.theme.row_h + 4.0;
        let row = self.next_row(h);
        self.button_in(label, row)
    }

    /// Draw a button into an explicit rect (used by the tab strip). Returns clicked.
    pub fn button_in(&mut self, label: &str, rect: Rect) -> bool {
        let id = self.id(label);
        let resp = self.interact(id, rect, self.dl.current_clip());
        let t = self.theme;
        let bg = if resp.held {
            t.button_active
        } else if resp.hovered {
            t.button_hot
        } else {
            t.button
        };
        self.dl.fill_rect(rect, bg);
        self.dl.rect_outline(rect, 1.0, t.panel_border);
        // Centred label.
        let scale = t.text_scale;
        let disp = strip_salt(label).0;
        let tw = self.font.measure_line(disp) * scale;
        let tx = rect.x + (rect.w - tw) * 0.5;
        let ty = rect.y + (rect.h - self.font.line_h() * scale) * 0.5;
        self.font.layout_into(&mut self.dl, disp, tx.floor(), ty.floor(), scale, t.text);
        resp.clicked
    }

    /// A toggle (checkbox): draws a box + label, flips `*value` on click. Returns
    /// true if the value changed this frame.
    pub fn toggle(&mut self, label: &str, value: &mut bool) -> bool {
        let h = self.theme.row_h;
        let row = self.next_row(h);
        let id = self.id(label);
        let box_sz = (row.h - 4.0).max(8.0);
        let box_rect = Rect::new(row.x, row.y + (row.h - box_sz) * 0.5, box_sz, box_sz);
        let resp = self.interact(id, row, self.dl.current_clip());
        let t = self.theme;
        let changed = resp.clicked;
        if changed {
            *value = !*value;
        }
        // Box.
        self.dl.fill_rect(box_rect, if resp.hovered { t.button_hot } else { t.button });
        self.dl.rect_outline(box_rect, 1.0, t.panel_border);
        if *value {
            self.dl.fill_rect(box_rect.inset(3.0), t.accent);
        }
        // Label to the right of the box.
        let text_row = Rect::new(box_rect.right() + 6.0, row.y, (row.right() - box_rect.right() - 6.0).max(0.0), row.h);
        self.text_in_row(text_row, strip_salt(label).0, t.text, 0.0);
        changed
    }

    /// A horizontal tab strip. `tabs` are the labels; `selected` is the current
    /// index (mutated on click). Returns true if the selection changed. Lays out
    /// one row tall, tabs evenly divided.
    pub fn tab_strip(&mut self, tabs: &[&str], selected: &mut usize) -> bool {
        if tabs.is_empty() {
            return false;
        }
        let rh = self.theme.row_h + 4.0;
        let row = self.next_row(rh);
        let tw = row.w / tabs.len() as f32;
        let t = self.theme;
        let mut changed = false;
        for (i, name) in tabs.iter().enumerate() {
            let r = Rect::new(row.x + i as f32 * tw, row.y, tw, row.h);
            let id = Id::new(self.id_seed ^ 0x7ab5, name);
            let resp = self.interact(id, r, self.dl.current_clip());
            let is_sel = i == *selected;
            let bg = if is_sel {
                t.button_active
            } else if resp.hovered {
                t.button_hot
            } else {
                t.button
            };
            self.dl.fill_rect(r, bg);
            self.dl.rect_outline(r, 1.0, t.panel_border);
            let scale = t.text_scale;
            let disp = strip_salt(name).0;
            let tx = r.x + (r.w - self.font.measure_line(disp) * scale) * 0.5;
            let ty = r.y + (r.h - self.font.line_h() * scale) * 0.5;
            let col = if is_sel { t.text } else { t.text_dim };
            self.font.layout_into(&mut self.dl, disp, tx.floor(), ty.floor(), scale, col);
            if resp.clicked && !is_sel {
                *selected = i;
                changed = true;
            }
        }
        changed
    }

    /// Render a [`LogView`] filling the remaining panel body (or an explicit rect via
    /// [`Ui::log_view_in`]). Shows the tail of the ring, sticks to the bottom unless
    /// the user has scrolled up, and responds to the mouse wheel while hovered.
    pub fn log_view(&mut self, log: &mut LogView) {
        let h = self.cursor().remaining_h();
        let row = self.next_row(h);
        self.log_view_in(row, log);
    }

    /// Draw a log view into an explicit rect.
    pub fn log_view_in(&mut self, rect: Rect, log: &mut LogView) {
        let t = self.theme;
        self.dl.fill_rect(rect, Color::rgb(0x0f, 0x11, 0x15));
        self.dl.rect_outline(rect, 1.0, t.panel_border);

        let line_h = self.font.line_h() * t.text_scale;
        let inner = rect.inset(3.0);
        let visible_lines = (inner.h / line_h).floor().max(0.0) as usize;
        let total = log.lines.len();

        // Wheel scrolling while hovered: wheel > 0 scrolls up (toward older lines).
        let hovered = inner.contains(self.input.mouse_x, self.input.mouse_y);
        if hovered && self.input.wheel != 0.0 {
            // Each wheel notch moves 3 lines (typical terminal feel).
            let delta = (self.input.wheel * 3.0).round() as i64;
            log.scroll_by(delta, visible_lines);
        }

        // Compute the first line to show. When stuck, always show the tail.
        let max_off = total.saturating_sub(visible_lines);
        if log.stuck {
            log.offset = max_off;
        } else {
            log.offset = log.offset.min(max_off);
        }
        let start = log.offset;

        self.dl.push_clip(inner);
        let mut y = inner.y;
        for line in log.lines.iter().skip(start).take(visible_lines) {
            self.font.layout_into(&mut self.dl, &line.text, inner.x, y, t.text_scale, line.color);
            y += line_h;
        }
        self.dl.pop_clip();

        // A thin scrollbar on the right when content overflows.
        if total > visible_lines {
            let track = Rect::new(rect.right() - 4.0, inner.y, 3.0, inner.h);
            self.dl.fill_rect(track, t.bar_bg);
            let frac_shown = visible_lines as f32 / total as f32;
            let thumb_h = (inner.h * frac_shown).max(8.0);
            let frac_off = if max_off == 0 { 0.0 } else { start as f32 / max_off as f32 };
            let thumb_y = inner.y + (inner.h - thumb_h) * frac_off;
            self.dl.fill_rect(Rect::new(track.x, thumb_y, track.w, thumb_h), t.text_dim);
        }
    }

    /// Number of primitives emitted so far this frame (diagnostics / tests).
    pub fn prim_count(&self) -> usize {
        self.dl.len()
    }
}

/// One log line: text plus a tint (e.g. warnings amber, errors red).
#[derive(Clone, Debug)]
pub struct LogLine {
    pub text: String,
    pub color: Color,
}

/// A bounded, scrollable log buffer — app-owned retained state for [`Ui::log_view`].
///
/// New lines push onto the back; when full, the oldest is dropped (a ring by
/// capacity). "Stick to bottom" auto-follows new output unless the user has scrolled
/// up, at which point it holds position until they scroll back to the end.
#[derive(Clone, Debug)]
pub struct LogView {
    lines: std::collections::VecDeque<LogLine>,
    capacity: usize,
    /// Index of the first visible line (from the top of the buffer).
    offset: usize,
    /// True while auto-following the tail.
    stuck: bool,
}

impl LogView {
    /// A log holding at most `capacity` lines, initially stuck to the bottom.
    pub fn new(capacity: usize) -> Self {
        Self {
            lines: std::collections::VecDeque::with_capacity(capacity.min(1 << 16)),
            capacity: capacity.max(1),
            offset: 0,
            stuck: true,
        }
    }

    /// Append a line, evicting the oldest if at capacity. Preserves the user's
    /// scroll position when they've scrolled up (offset shifts with eviction).
    pub fn push(&mut self, text: impl Into<String>, color: Color) {
        if self.lines.len() == self.capacity {
            self.lines.pop_front();
            // Eviction shifts every index down by one; keep the view stable.
            self.offset = self.offset.saturating_sub(1);
        }
        self.lines.push_back(LogLine { text: text.into(), color });
    }

    /// Current number of buffered lines.
    pub fn len(&self) -> usize {
        self.lines.len()
    }
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
    /// Is the view currently following the tail?
    pub fn is_stuck(&self) -> bool {
        self.stuck
    }
    /// First visible line index.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Scroll by `delta` lines (negative = toward the tail / down, positive = up).
    /// Re-sticks to the bottom when scrolled to (or past) the last page.
    pub fn scroll_by(&mut self, delta: i64, visible_lines: usize) {
        let max_off = self.lines.len().saturating_sub(visible_lines) as i64;
        // wheel up (positive) shows older lines => decrease offset.
        let new = (self.offset as i64 - delta).clamp(0, max_off.max(0));
        self.offset = new as usize;
        self.stuck = self.offset as i64 >= max_off;
    }
}

/// Split an imgui-style `"label##salt"` into `(display, id_source)`. The salt keeps
/// two same-looking widgets distinct without cluttering the visible text; both parts
/// feed the id, only the display part is drawn.
fn strip_salt(label: &str) -> (&str, &str) {
    match label.find("##") {
        Some(i) => (&label[..i], label),
        None => (label, label),
    }
}
