//! The player's immediate-mode UI — transport bar, track menus, and stats overlay,
//! built each frame from live state onto scope's [`Ui`] toolkit (spec: profluens.md UI
//! `<update>` §1209 — "the same philosophy as the rest of SC (very performant, per-frame
//! arenas)"). Nothing here is retained across frames except the small [`PlayerUiState`]
//! the caller owns; the widgets rebuild the picture from the data every frame, the
//! Muratori (2005) immediate-mode discipline scope itself follows.
//!
//! The transport bar auto-hides after ~2.5 s of mouse idle and fades back in on motion,
//! so the video is unobstructed during playback. The stats overlay is a scope-style panel
//! reading the tap counters — zero cost when hidden (it draws nothing).

use profluens_core::pipeline::SeekIndex;
use profluens_core::time::Timestamp;
use profluens_scope::ui::draw::{Color, Rect};
use profluens_scope::ui::{MouseButton, Ui};

/// How long the transport bar lingers after the last mouse motion before fading out.
const IDLE_HIDE_SECS: f32 = 2.5;
/// The fade in/out duration once idle/motion flips.
const FADE_SECS: f32 = 0.25;
/// The transport bar's height and the timeline track's height.
const BAR_H: f32 = 64.0;

/// One track row for a menu (audio / subtitle), distilled from `player.tracks()`.
pub struct TrackRow {
    pub label: String,
    /// True for the track that is actually being played (v1 plays the first of each kind).
    pub active: bool,
    /// True for a subtitle track (routed to the subtitle menu instead of the audio menu).
    pub subtitle: bool,
}

/// Retained UI state the caller owns between frames (the immediate-mode discipline: the
/// `Ui` itself owns nothing). Seek-drag is tracked here so a click-and-drag on the timeline
/// scrubs continuously.
///
/// scope's `UiState` (hot/active bookkeeping) is deliberately NOT a field here: `Ui::new`
/// borrows it mutably for the frame's lifetime, which would then conflict with borrowing
/// `PlayerUiState` for [`build`]. The caller keeps a separate `UiState` local and passes
/// both disjointly.
pub struct PlayerUiState {
    /// Seconds since the mouse last moved — drives the transport-bar auto-hide.
    idle_secs: f32,
    /// Last mouse position, to detect motion (SDL gives absolute coords each frame).
    last_mouse: (f32, f32),
    /// Whether the stats overlay is shown (toggled by Tab / i).
    pub show_stats: bool,
    /// True while the user is dragging the timeline thumb (scrubbing).
    scrubbing: bool,
    /// The paused state as the UI last knew it (for the play/pause glyph).
    pub paused: bool,
    /// Muted state (best-effort; only meaningful when a volume prop exists).
    pub muted: bool,
}

impl Default for PlayerUiState {
    fn default() -> Self {
        Self {
            idle_secs: 0.0,
            last_mouse: (-1.0, -1.0),
            show_stats: false,
            scrubbing: false,
            paused: false,
            muted: false,
        }
    }
}

impl PlayerUiState {
    /// Advance the idle timer from this frame's `dt` and the current mouse position; the
    /// bar shows while the mouse is moving or recently moved, and while scrubbing.
    pub fn tick_idle(&mut self, dt: f32, mouse: (f32, f32)) {
        if (mouse.0 - self.last_mouse.0).abs() > 0.5 || (mouse.1 - self.last_mouse.1).abs() > 0.5 {
            self.idle_secs = 0.0;
        } else {
            self.idle_secs += dt;
        }
        self.last_mouse = mouse;
    }

    /// The transport bar's current opacity (0 fully hidden … 1 fully shown). Fades over
    /// [`FADE_SECS`] around the [`IDLE_HIDE_SECS`] threshold. Always fully shown while
    /// scrubbing or while the pointer is over the bar region.
    fn bar_alpha(&self, over_bar: bool) -> f32 {
        if self.scrubbing || over_bar {
            return 1.0;
        }
        let past = self.idle_secs - IDLE_HIDE_SECS;
        if past <= 0.0 {
            1.0
        } else {
            (1.0 - past / FADE_SECS).clamp(0.0, 1.0)
        }
    }

    /// True once the transport chrome has fully faded out (idle past the hide+fade window) and
    /// nothing is being scrubbed — the composited frame is then identical to the fully-hidden
    /// steady state and need not be redrawn until a new video frame arrives or the user acts.
    /// A stationary pointer parked over the bar keeps it lit via [`bar_alpha`]'s `over_bar`
    /// path; that frame is likewise static, and any real pointer motion / button / key is
    /// caught by the caller's input-activity gate. This is what lets a paused player idle at
    /// ~0% CPU instead of re-tessellating and re-presenting the same pixels every vsync.
    pub fn chrome_settled_hidden(&self) -> bool {
        !self.scrubbing && self.idle_secs >= IDLE_HIDE_SECS + FADE_SECS
    }
}

/// Everything the UI needs to draw one frame, gathered by the caller from the player.
pub struct UiFrame<'a> {
    /// Current playback position (running time), if known.
    pub position: Timestamp,
    /// Total duration, if known.
    pub duration: Timestamp,
    pub paused: bool,
    /// The audio+subtitle track rows for the menus.
    pub tracks: &'a [TrackRow],
    /// Whether a live `volume` prop exists on the audio sink (enables the mute control).
    pub has_volume: bool,
    /// The seek index, for turning a timeline click into a byte target.
    pub seek_index: &'a SeekIndex,
    /// The per-element stats rows for the overlay (label, in, out, Δin, Δout, qhw, drops).
    pub stats: &'a [StatRow],
    /// Computed video fps and cumulative paced drops for the overlay header.
    pub video_fps: f32,
    pub qos_drops: u64,
    /// Whether the stream has ended.
    pub eos: bool,
}

/// One element's counters for the stats overlay.
pub struct StatRow {
    pub label: &'static str,
    pub buffers_in: u64,
    pub buffers_out: u64,
    pub d_in: u64,
    pub d_out: u64,
    pub queue_high_water: u32,
    pub drops: u64,
}

/// A UI action the caller must carry out against the player (the UI itself holds no
/// handles — it returns intent, the binary drives `pf-play`). Immediate-mode all the way
/// down: one action per frame at most.
#[derive(Clone, Copy, Debug, Default)]
pub struct UiActions {
    /// Toggle pause/resume was requested (the play/pause button).
    pub toggle_pause: bool,
    /// Toggle mute was requested (the volume/mute button).
    pub toggle_mute: bool,
    /// A seek to this stream time was requested (timeline click/drag).
    pub seek_to: Option<Timestamp>,
}

/// Build the whole player UI for one frame onto `ui`, returning the actions the user
/// triggered. `dt` is the frame delta (seconds), for the auto-hide timer.
pub fn build(state: &mut PlayerUiState, ui: &mut Ui, frame: &UiFrame, dt: f32) -> UiActions {
    let mut actions = UiActions::default();
    let (ww, wh) = (ui.screen().w, ui.screen().h);
    let mouse = (ui.input().mouse_x, ui.input().mouse_y);
    state.tick_idle(dt, mouse);
    state.paused = frame.paused;

    // The stats overlay (top-left), when toggled. Drawn first so the transport bar sits
    // above it if they ever overlap.
    if state.show_stats {
        draw_stats(ui, frame, ww);
    }

    // The transport bar (bottom), auto-hiding. Compute opacity from idle + hover.
    let bar_rect = Rect::new(0.0, wh - BAR_H, ww, BAR_H);
    let over_bar = bar_rect.contains(mouse.0, mouse.1);
    let alpha = state.bar_alpha(over_bar);
    if alpha > 0.01 {
        draw_transport(state, ui, frame, bar_rect, alpha, &mut actions);
    }

    // The track menus (right side), shown together with the transport bar (same fade).
    if alpha > 0.01 && !frame.tracks.is_empty() {
        draw_track_menu(ui, frame, ww, alpha);
    }

    // A centred "ENDED" hint when the stream finished.
    if frame.eos {
        let s = "— ENDED —";
        let tw = ui.font().measure_line(s);
        ui.text((ww - tw) * 0.5, 24.0, s, Color::rgba(0xff, 0xff, 0xff, 0xcc));
    }

    actions
}

/// The bottom transport bar: play/pause, a seekable timeline, elapsed/total, and a
/// mute control when a live volume prop exists. Alpha-faded per the auto-hide timer.
fn draw_transport(
    state: &mut PlayerUiState,
    ui: &mut Ui,
    frame: &UiFrame,
    bar: Rect,
    alpha: f32,
    actions: &mut UiActions,
) {
    let a = |c: Color| c.with_alpha((alpha * c.a as f32) as u8);
    let dl = ui.draw_list_mut();
    // A translucent gradient-ish panel (single fill; the video shows faintly under it).
    dl.fill_rect(bar, a(Color::rgba(0x10, 0x12, 0x16, 0xd0)));
    dl.fill_rect(Rect::new(bar.x, bar.y, bar.w, 1.0), a(Color::rgba(0x33, 0x3a, 0x44, 0xff)));

    let pad = 14.0;
    let btn_sz = 28.0;
    // Vertically center everything (button, timeline, labels) on the bar's midline so they
    // align, rather than the button riding the top and the timeline the bottom.
    let cy = bar.y + (BAR_H - btn_sz) * 0.5;

    // Play/pause button (a simple glyph: "►" play / "II" pause via two bars).
    let play_rect = Rect::new(bar.x + pad, cy, btn_sz, btn_sz);
    if button_glyph(state, ui, "play##transport", play_rect, alpha) {
        actions.toggle_pause = true;
    }
    draw_playpause_glyph(ui, play_rect, state.paused, alpha);

    // Mute button (only when a volume prop exists).
    let mut right_edge = bar.right() - pad;
    if frame.has_volume {
        let mute_rect = Rect::new(right_edge - btn_sz, cy, btn_sz, btn_sz);
        if button_glyph(state, ui, "mute##transport", mute_rect, alpha) {
            actions.toggle_mute = true;
        }
        let g = if state.muted { "x" } else { "))" };
        let gw = ui.font().measure_line(g);
        let a2 = |c: Color| c.with_alpha((alpha * c.a as f32) as u8);
        let tx = mute_rect.x + (mute_rect.w - gw) * 0.5;
        let ty = mute_rect.y + (mute_rect.h - ui.font().line_h()) * 0.5;
        ui.text(tx.floor(), ty.floor(), g, a2(Color::rgb(0xd8, 0xdd, 0xe4)));
        right_edge -= btn_sz + 8.0;
    }

    // Time readouts on the right: "elapsed / total".
    let pos = frame.position.nanos().unwrap_or(0);
    let dur = frame.duration.nanos();
    let time_str = match dur {
        Some(d) => format!("{} / {}", fmt_time(pos), fmt_time(d)),
        None => fmt_time(pos),
    };
    let a3 = |c: Color| c.with_alpha((alpha * c.a as f32) as u8);
    let tw = ui.font().measure_line(&time_str);
    let time_x = right_edge - tw;
    let time_y = cy + (btn_sz - ui.font().line_h()) * 0.5;
    ui.text(time_x.floor(), time_y.floor(), &time_str, a3(Color::rgb(0xd8, 0xdd, 0xe4)));

    // The seekable timeline between the play button and the time readout — centered on the
    // same midline as the button and labels.
    let track_x = play_rect.right() + pad;
    let track_w = (time_x - 12.0 - track_x).max(10.0);
    let track_h = 8.0;
    let track = Rect::new(track_x, bar.y + (BAR_H - track_h) * 0.5, track_w, track_h);
    if let Some(t) = draw_timeline(state, ui, track, pos, dur, frame.seek_index, alpha) {
        actions.seek_to = Some(t);
    }
}

/// The timeline: a fill bar showing progress, clickable/draggable to seek. Returns the
/// requested seek time when the user clicks or drags. Resolving the target time→byte is
/// the caller's job (via `SeekIndex::resolve`, like pfplay's digit-seek) — here we just
/// produce the target *time* from the click x, and confirm the index CAN resolve it.
fn draw_timeline(
    state: &mut PlayerUiState,
    ui: &mut Ui,
    track: Rect,
    pos_ns: u64,
    dur_ns: Option<u64>,
    index: &SeekIndex,
    alpha: f32,
) -> Option<Timestamp> {
    // Snapshot all input reads up front, then release the `&Input` borrow before touching
    // `draw_list_mut` (they both borrow `ui`).
    let input = ui.input();
    let mouse = (input.mouse_x, input.mouse_y);
    let pressed = input.pressed(MouseButton::Left);
    let released = input.released(MouseButton::Left);
    let down = input.down(MouseButton::Left);
    // A generous hit region (taller than the visible track, so the thin bar is easy to grab).
    let hit = Rect::new(track.x, track.y - 8.0, track.w, track.h + 16.0);
    let over = hit.contains(mouse.0, mouse.1);

    // Begin/continue/end a scrub drag.
    if over && pressed {
        state.scrubbing = true;
    }
    if !down {
        state.scrubbing = false;
    }

    let frac = match dur_ns {
        Some(d) if d > 0 => (pos_ns as f32 / d as f32).clamp(0.0, 1.0),
        _ => 0.0,
    };
    // While scrubbing, the thumb tracks the mouse (visual feedback before the seek lands).
    let draw_frac = if state.scrubbing {
        ((mouse.0 - track.x) / track.w).clamp(0.0, 1.0)
    } else {
        frac
    };

    let a = |c: Color| c.with_alpha((alpha * c.a as f32) as u8);
    let dl = ui.draw_list_mut();
    dl.fill_rect(track, a(Color::rgba(0x2a, 0x30, 0x3a, 0xff)));
    dl.fill_rect(Rect::new(track.x, track.y, track.w * draw_frac, track.h), a(Color::rgb(0x4a, 0xa8, 0xff)));
    // The thumb.
    let thumb_x = track.x + track.w * draw_frac;
    dl.fill_rect(Rect::new(thumb_x - 3.0, track.y - 4.0, 6.0, track.h + 8.0), a(Color::rgb(0xd8, 0xdd, 0xe4)));

    // A seek fires on release-after-scrub, or a plain click in the track. We only produce
    // a target the index can actually resolve — else the click is ignored (informational).
    let want_seek = (state.scrubbing && released) || (over && pressed);
    if want_seek {
        if let Some(d) = dur_ns {
            let clicked_frac = ((mouse.0 - track.x) / track.w).clamp(0.0, 1.0);
            let target = Timestamp((clicked_frac * d as f32) as u64);
            // Confirm the index can map it (keyframe floor or proportional) before asking.
            if index.resolve(target, Timestamp(d)).is_some() {
                return Some(target);
            }
        }
    }
    None
}

/// A right-side menu listing audio + subtitle tracks and marking which plays. Switching
/// live is future work (needs a re-wire); v1 is informational, per the brief.
fn draw_track_menu(ui: &mut Ui, frame: &UiFrame, ww: f32, alpha: f32) {
    let w = 260.0;
    let x = ww - w - 14.0;
    let rows = frame.tracks.len();
    let h = 30.0 + rows as f32 * 20.0;
    let panel = Rect::new(x, 14.0, w, h);
    let a = |c: Color| c.with_alpha((alpha * c.a as f32) as u8);
    {
        let dl = ui.draw_list_mut();
        dl.fill_rect(panel, a(Color::rgba(0x1e, 0x22, 0x28, 0xdc)));
        dl.rect_outline(panel, 1.0, a(Color::rgb(0x33, 0x3a, 0x44)));
    }
    ui.text(panel.x + 10.0, panel.y + 6.0, "Tracks", a(Color::rgb(0x8a, 0x93, 0xa0)));
    let mut y = panel.y + 28.0;
    for t in frame.tracks {
        let kind = if t.subtitle { "sub" } else { "aud" };
        let marker = if t.active { "●" } else { "○" };
        let line = format!("{marker} [{kind}] {}", t.label);
        let col = if t.active {
            a(Color::rgb(0x4a, 0xa8, 0xff))
        } else {
            a(Color::rgb(0xd8, 0xdd, 0xe4))
        };
        ui.text(panel.x + 10.0, y, &line, col);
        y += 20.0;
    }
}

/// The scope-style stats panel: per-element buffers in→out with deltas, queue high-water,
/// drops, plus a computed video fps and the paced-drop count. Zero cost when hidden (the
/// caller only calls this when `show_stats`).
fn draw_stats(ui: &mut Ui, frame: &UiFrame, _ww: f32) {
    let panel = Rect::new(14.0, 14.0, 360.0, 30.0 + (frame.stats.len() as f32 + 2.0) * 18.0);
    let body = ui.begin_panel("stats", panel);
    let _ = body;
    ui.kv("video fps", &format!("{:.1}", frame.video_fps));
    ui.kv("paced drops", &frame.qos_drops.to_string());
    for s in frame.stats {
        ui.kv(
            s.label,
            &format!(
                "{}→{} (+{}/+{}) qhw={} drop={}",
                s.buffers_in, s.buffers_out, s.d_in, s.d_out, s.queue_high_water, s.drops
            ),
        );
    }
    ui.end_panel();
}

/// A borderless icon button that highlights on hover/press. Returns clicked. Uses raw
/// input+draw (scope's `button_in` draws a filled box, which we don't want over video).
fn button_glyph(state: &mut PlayerUiState, ui: &mut Ui, id: &str, rect: Rect, alpha: f32) -> bool {
    let input = ui.input();
    let over = rect.contains(input.mouse_x, input.mouse_y);
    let clicked = over && input.pressed(MouseButton::Left);
    let _ = (state, id); // hot/active tracking not needed for these momentary buttons
    let a = |c: Color| c.with_alpha((alpha * c.a as f32) as u8);
    let bg = if over {
        a(Color::rgba(0x42, 0x4c, 0x5c, 0xcc))
    } else {
        a(Color::rgba(0x2a, 0x30, 0x3a, 0x88))
    };
    let dl = ui.draw_list_mut();
    dl.fill_rect(rect, bg);
    clicked
}

/// Draw the play (►) / pause (II) glyph centred in `rect`, as solid primitives (the 8x8
/// bitmap font has no good triangle glyph).
fn draw_playpause_glyph(ui: &mut Ui, rect: Rect, paused: bool, alpha: f32) {
    let col = Color::rgb(0xd8, 0xdd, 0xe4).with_alpha((alpha * 255.0) as u8);
    let cx = rect.x + rect.w * 0.5;
    let cy = rect.y + rect.h * 0.5;
    let dl = ui.draw_list_mut();
    if paused {
        // Two vertical bars = a "paused" state shows the play triangle to resume.
        let s = 6.0;
        dl.line(cx - s, cy - s, cx + s, cy, 2.0, col);
        dl.line(cx + s, cy, cx - s, cy + s, 2.0, col);
        dl.line(cx - s, cy - s, cx - s, cy + s, 2.0, col);
    } else {
        // Playing → show the pause bars (click to pause).
        dl.fill_rect(Rect::new(cx - 5.0, cy - 6.0, 3.0, 12.0), col);
        dl.fill_rect(Rect::new(cx + 2.0, cy - 6.0, 3.0, 12.0), col);
    }
}

/// Format a nanosecond timestamp as `H:MM:SS` (or `M:SS` under an hour).
fn fmt_time(ns: u64) -> String {
    let total = ns / 1_000_000_000;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_formatting() {
        assert_eq!(fmt_time(0), "0:00");
        assert_eq!(fmt_time(65 * 1_000_000_000), "1:05");
        assert_eq!(fmt_time(3661 * 1_000_000_000), "1:01:01");
    }

    #[test]
    fn idle_timer_shows_bar_on_motion_and_hides_after_threshold() {
        let mut s = PlayerUiState::default();
        // Motion resets idle; the bar is fully shown.
        s.tick_idle(0.1, (10.0, 10.0));
        s.tick_idle(0.1, (20.0, 20.0));
        assert_eq!(s.bar_alpha(false), 1.0);
        // No motion for well past the hide threshold + fade → hidden.
        for _ in 0..40 {
            s.tick_idle(0.1, (20.0, 20.0));
        }
        assert_eq!(s.bar_alpha(false), 0.0);
        // Hovering the bar keeps it shown regardless.
        assert_eq!(s.bar_alpha(true), 1.0);
    }
}
