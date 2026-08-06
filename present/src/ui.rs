//! [`WindowUi`] — the interactive-overlay layer shared by the two windowed video sinks
//! ([`WaylandVideoSink`](crate::sink::WaylandVideoSink), zero-copy dmabuf; and
//! [`WaylandRawSink`](crate::rawsink::WaylandRawSink), software `video/raw`). It owns everything
//! that is independent of *how* the video pixels reach the screen: pumping the compositor
//! socket, routing pointer clicks to pause/seek, the auto-hiding HUD control bar, the subtitle
//! caption, and answering window close. Each sink drives it from its clock-wait pump so the
//! window stays live between frames and while paused.

use std::sync::Arc;

use profluens_core::time::Timestamp;

use pf_text::{font, pgs};

use crate::control::{PlayerControl, UiCommand};
use crate::raster::Canvas;
use crate::window::Window;

/// The control-bar overlay height in px.
pub(crate) const BAR_H: i32 = 40;
/// Where the timeline rail starts (after the play glyph + the `MM:SS` time) and its right pad.
const TRACK_X: i32 = 108;
const TRACK_PAD: i32 = 16;
/// How long the HUD lingers after the last pointer activity before auto-hiding (playing only —
/// a paused player keeps its controls up).
const HUD_LINGER: std::time::Duration = std::time::Duration::from_secs(3);

/// One decoded PGS caption + its on-screen span.
struct SubCue {
    start: Timestamp,
    end: Timestamp,
    ds: pgs::DisplaySet,
}

/// The interactive overlay state for one window. Embedded in each sink; driven every clock-wait
/// slice via [`pump`](Self::pump).
pub(crate) struct WindowUi {
    /// The window↔app control channel (clicks → pause/seek; duration/paused → HUD). `None` =
    /// display-only (no interactive controls).
    control: Option<Arc<PlayerControl>>,
    /// The current subtitle caption (latest-wins), shown for `[start, end)` on its own subsurface.
    subtitle: Option<SubCue>,
    /// HUD auto-hide: stay visible until this instant (bumped on pointer activity); hidden when
    /// elapsed *and* playing (a paused player always shows its controls). `None` = never shown.
    hud_deadline: Option<std::time::Instant>,
    /// Whether the HUD subsurface is currently mapped — the redraw/hide edge trigger.
    hud_mapped: bool,
    /// The wall-second and pause state the HUD was last drawn at, so it redraws only on a real
    /// content change (the clock ticking or a play/pause flip), not every pump.
    hud_sec: u64,
    hud_paused: bool,
    /// The `[start, end)` span of the caption currently committed to the subtitle subsurface, so
    /// it is re-rendered only when the visible cue changes — not every pump while one is up.
    sub_shown: Option<(Timestamp, Timestamp)>,
    /// `Quit` was pushed to the control channel on window close — once only.
    quit_sent: bool,
}

impl WindowUi {
    // COLD: constructor.
    #[allow(clippy::disallowed_methods)]
    pub(crate) fn new() -> Self {
        WindowUi {
            control: None,
            subtitle: None,
            hud_deadline: None,
            hud_mapped: false,
            hud_sec: u64::MAX,
            hud_paused: false,
            sub_shown: None,
            quit_sent: false,
        }
    }

    /// Attach the window↔app control channel (clicks → pause/seek; duration/pause → HUD).
    pub(crate) fn set_control(&mut self, control: Arc<PlayerControl>) {
        self.control = Some(control);
    }

    /// Reveal the controls for [`HUD_LINGER`] (e.g. on window open) — then they auto-hide.
    pub(crate) fn arm_hud(&mut self) {
        self.hud_deadline = Some(std::time::Instant::now() + HUD_LINGER);
    }

    /// Update the current caption from a `subtitle/bitmap` buffer (latest-wins; a clear erases
    /// it). Timed by the buffer's pts + duration.
    pub(crate) fn push_subtitle(&mut self, ds: pgs::DisplaySet, pts: Timestamp, duration: Timestamp) {
        if ds.is_clear() {
            self.subtitle = None;
            return;
        }
        let Some(start_ns) = pts.nanos() else { return };
        // A PGS composition stays until the next composition/clear; use its duration when set,
        // else a generous default so a lost clear doesn't strand a caption forever.
        let dur_ns = duration.nanos().filter(|&d| d > 0).unwrap_or(10_000_000_000);
        let start = Timestamp::from_nanos(start_ns);
        let end = Timestamp::from_nanos(start_ns.saturating_add(dur_ns));
        self.subtitle = Some(SubCue { start, end, ds });
    }

    /// Drop the current caption on a flush/seek — the subtitle track re-emits for the new
    /// position (the pump hides it next pass once `pts` leaves its span).
    pub(crate) fn on_flush(&mut self) {
        self.subtitle = None;
    }

    /// One unit of window upkeep, decoupled from the video rate: pump the socket, re-letterbox
    /// on resize, route clicks to pause/seek, signal `Quit` on close, and refresh the HUD +
    /// caption for running position `pos_ns`. Idempotent/throttled, so calling it at the clock
    /// wait's ~8 ms slice rate is cheap. Video presentation itself is the sink's job.
    pub(crate) fn pump(&mut self, w: &mut Window, pos_ns: u64) {
        let _ = w.dispatch(false);
        // A resize (esp. while paused, when no new frame arrives) re-fits the retained frame.
        let _ = w.reflow();

        // Route left-clicks: the timeline rail → seek, elsewhere → pause toggle.
        if let Some(ctrl) = self.control.as_ref() {
            let (ww, wh) = w.size();
            while let Some((cx, cy)) = w.take_click() {
                if cy >= wh - BAR_H && cx >= TRACK_X && cx <= ww - TRACK_PAD {
                    let tw = (ww - TRACK_X - TRACK_PAD).max(1);
                    let frac = ((cx - TRACK_X) as f32 / tw as f32).clamp(0.0, 1.0);
                    ctrl.push(UiCommand::SeekFraction(frac));
                } else {
                    ctrl.push(UiCommand::TogglePause);
                }
            }
            // Window close → tell the app to quit (once); the pipeline stop unwinds everything.
            if w.should_close() && !self.quit_sent {
                ctrl.push(UiCommand::Quit);
                self.quit_sent = true;
            }
        }

        // HUD: shown while paused or for a few seconds after pointer activity, then auto-hidden
        // so idle playback pays nothing for it. While visible, redraw only on a real content
        // change (the wall-second ticking or a play/pause flip) — not every pump.
        if w.has_gui() {
            if w.take_pointer_activity() {
                self.arm_hud();
            }
            let (dur_ns, paused) = self
                .control
                .as_ref()
                .map(|c| (c.duration_ns(), c.paused()))
                .unwrap_or((0, false));
            let show = paused || self.hud_deadline.is_some_and(|d| std::time::Instant::now() < d);
            let sec = pos_ns / 1_000_000_000;
            if show {
                if !self.hud_mapped || paused != self.hud_paused || sec != self.hud_sec {
                    self.hud_mapped = true;
                    self.hud_paused = paused;
                    self.hud_sec = sec;
                    let (ww, wh) = w.size();
                    let hud = Hud { width: ww, pos_ns, dur_ns, paused };
                    let _ = w.present_gui(0, wh - BAR_H, ww, BAR_H, |c| draw_controls(c, &hud));
                }
            } else if self.hud_mapped {
                self.hud_mapped = false;
                let _ = w.hide_gui();
            }
        }

        // Caption for the current position — (re)rendered only when the visible cue changes.
        let pts = Timestamp::from_nanos(pos_ns);
        let want = self
            .subtitle
            .as_ref()
            .filter(|c| pts >= c.start && pts < c.end)
            .map(|c| (c.start, c.end));
        if want != self.sub_shown {
            match self.subtitle.as_ref().filter(|_| want.is_some()) {
                Some(c) => {
                    let ds = &c.ds;
                    let _ = w.present_subtitle(
                        ds.video_width as i32,
                        ds.video_height as i32,
                        ds.x as i32,
                        ds.y as i32,
                        ds.width as i32,
                        ds.height as i32,
                        &ds.rgba,
                    );
                }
                None => {
                    let _ = w.hide_subtitle();
                }
            }
            self.sub_shown = want;
        }
    }
}

/// The HUD's per-frame state (position/duration/pause) for [`draw_controls`].
struct Hud {
    width: i32,
    pos_ns: u64,
    dur_ns: u64,
    paused: bool,
}

/// Draw the player HUD control bar into `c` — a `width×BAR_H` ARGB canvas, `0` alpha where the
/// video shows through. Anti-aliased shapes + text (the compositor blends this `wl_subsurface`
/// over the video, no GPU): a play/pause glyph, the elapsed time, and a live timeline with a
/// progress fill + seek knob.
fn draw_controls(c: &mut Canvas, h: &Hud) {
    const ACCENT: u32 = 0xff30_ffa0; // profluens green
    const INK: u32 = 0xffe6_edf5;
    let width = h.width;
    c.clear(0); // fully transparent — video shows through everywhere we don't draw
    c.blend_rect(0, 0, width, BAR_H, 0xd010_1822); // translucent dark bar

    // Transport glyph: show the action the click performs, not the current state — a play
    // triangle while paused (click resumes), pause bars while playing (click pauses).
    if h.paused {
        c.fill_triangle((16, 11), (16, 29), (34, 20), ACCENT);
    } else {
        c.fill_rect(16, 11, 5, 18, ACCENT);
        c.fill_rect(26, 11, 5, 18, ACCENT);
    }

    // Elapsed time "MM:SS", anti-aliased text — formatted into a stack buffer (no heap).
    let secs = h.pos_ns / 1_000_000_000;
    let (mm, ss) = (secs / 60, secs % 60);
    let buf = [
        b'0' + ((mm / 10) % 10) as u8,
        b'0' + (mm % 10) as u8,
        b':',
        b'0' + (ss / 10) as u8,
        b'0' + (ss % 10) as u8,
    ];
    let text = std::str::from_utf8(&buf).unwrap_or("00:00");
    let s = font::size_nearest(18.0);
    let bm = font::rasterize_line(s, text);
    let ty = ((BAR_H - bm.h as i32) / 2).max(0);
    c.blit_a8(46, ty, &bm.cov, bm.w as u32, bm.h as u32, bm.w as u32, INK);

    // Timeline: rail, progress fill (position/duration), and the seek knob.
    let (tx, cy) = (TRACK_X, BAR_H / 2);
    let tw = (width - tx - TRACK_PAD).max(1);
    c.blend_rect(tx, cy - 1, tw, 2, 0x6050_6274); // rail
    let frac = if h.dur_ns > 0 {
        (h.pos_ns as f64 / h.dur_ns as f64).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    let fill = (tw as f32 * frac) as i32;
    c.fill_rect(tx, cy - 1, fill, 2, ACCENT); // progress
    c.fill_circle((tx + fill) as f32, cy as f32, 5.0, ACCENT); // seek knob
}
