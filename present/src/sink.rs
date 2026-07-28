//! [`WaylandVideoSink`] (`waylandvideosink`) — a clock-paced, zero-copy video sink that
//! presents the VA-API decoder's `video/gpu` dmabuf frames through the raw-wire [`Window`],
//! no libwayland and no GPU renderer. The drop-in replacement for `sdl3videosink` on the
//! zero-copy path.
//!
//! Each input `video/gpu` buffer carries a [`GpuFrameHeader`] (token + display geometry); the
//! sink pairs it with the out-of-band [`GpuFrame`] (dmabuf fds + planes) from the shared
//! [`GpuFrameChannel`], paces it on `ctx.wait_until(pts)`, imports it as a `wl_buffer`
//! (`zwp_linux_dmabuf`), presents it, then **releases the token** so the decoder can reuse the
//! surface — the anti-tear contract. A missing compositor dmabuf global disables the sink
//! (drops frames) rather than failing the pipeline.

use std::sync::Arc;

use streamcraft_core::bus::BusMessage;
use streamcraft_core::clock::WaitOutcome;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::log;
use streamcraft_core::log::Level;
use streamcraft_core::time::Timestamp;

use sc_text::{font, pgs};
use sc_vaapi::gpuframe::{GpuFrame, GpuFrameChannel, GpuFrameHeader, GPU_OFFER};

use crate::raster::Canvas;
use crate::window::{DmabufVideo, Plane, Window};

/// The video (metronome) pad and the subtitle-bitmap side pad.
const VIDEO: PadId = PadId(0);
const SUBTITLE: PadId = PadId(1);

static VIDEO_OFFERS: [OfferDesc; 1] = [GPU_OFFER];
static SUB_OFFERS: [OfferDesc; 1] = [OfferDesc::any(sc_text::BITMAP_FAMILY)];
static PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &VIDEO_OFFERS,
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "subtitle",
        direction: Direction::Sink,
        offers: &SUB_OFFERS,
        dynamic: false,
        validate: None,
    },
];

static DESC: ElementDesc = ElementDesc {
    name: "waylandvideosink",
    pads: &PADS,
    props: &[],
    // Its own thread; paces the graph on the clock (spec: Scheduling — a windowed sink).
    sched: SchedHint::Active,
    // Any: the video pad is the metronome; the subtitle pad is a side input read per-pass.
    inputs: InputPolicy::Any,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(WaylandVideoSink::new())),
};

/// The raw-wire Wayland zero-copy video sink.
pub struct WaylandVideoSink {
    /// The decoder↔sink frame hand-off (set by the player at wiring time).
    channel: Option<Arc<GpuFrameChannel>>,
    /// The presentation window, opened lazily on the first frame (sized to the display rect).
    window: Option<Window>,
    /// Set once presentation is impossible (no compositor dmabuf / window failed) — from then
    /// on frames are accepted and dropped so the pipeline never hangs.
    disabled: bool,
    /// The next frame after a flush presents immediately (no clock wait) — preroll.
    preroll_next: bool,
    /// Frames presented — drives the GUI overlay's decoupled (lower-rate) update cadence.
    frames: u64,
    /// The current subtitle caption (latest-wins), shown for `[start, end)` (its own subsurface).
    subtitle: Option<SubCue>,
    /// The window↔app control channel (clicks → pause/seek; duration/paused → HUD). Set by the
    /// player; `None` = display-only (no interactive controls).
    control: Option<Arc<crate::control::PlayerControl>>,
    /// HUD auto-hide: stay visible until this instant (bumped on pointer activity); hidden when
    /// elapsed *and* playing (a paused player always shows its controls). `None` = never shown.
    hud_deadline: Option<std::time::Instant>,
    /// Whether the HUD subsurface is currently mapped — the redraw/hide edge trigger.
    hud_mapped: bool,
    /// The wall-second and pause state the HUD was last drawn at, so it redraws only on a real
    /// content change (the clock ticking or a play/pause flip), not every pump — the overlay
    /// was previously re-rendered ~15×/s for nothing (the "2% CPU while playing").
    hud_sec: u64,
    hud_paused: bool,
    /// The `[start, end)` span of the caption currently committed to the subtitle subsurface, so
    /// it is re-rendered only when the visible cue changes — not every pump while one is up.
    sub_shown: Option<(Timestamp, Timestamp)>,
    /// `Quit` was pushed to the control channel on window close — once only.
    quit_sent: bool,
}

/// One decoded PGS caption + its on-screen span.
struct SubCue {
    start: Timestamp,
    end: Timestamp,
    ds: pgs::DisplaySet,
}

impl Default for WaylandVideoSink {
    fn default() -> Self {
        Self::new()
    }
}

impl WaylandVideoSink {
    /// A sink with no channel yet — [`with_channel`](Self::with_channel) wires the hand-off.
    // COLD: constructor.
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        WaylandVideoSink {
            channel: None,
            window: None,
            disabled: false,
            preroll_next: false,
            frames: 0,
            subtitle: None,
            control: None,
            hud_deadline: None,
            hud_mapped: false,
            hud_sec: u64::MAX,
            hud_paused: false,
            sub_shown: None,
            quit_sent: false,
        }
    }

    /// Attach the shared [`GpuFrameChannel`] (the same `Arc` the zero-copy decoder holds).
    pub fn with_channel(mut self, channel: Arc<GpuFrameChannel>) -> Self {
        self.channel = Some(channel);
        self
    }

    /// Attach the window↔app control channel for interactive controls (clicks → pause/seek).
    pub fn with_control(mut self, control: Arc<crate::control::PlayerControl>) -> Self {
        self.control = Some(control);
        self
    }

    /// Present one `video/gpu` buffer: pair it to its [`GpuFrame`], pace, import + show, release.
    fn present(&mut self, ctx: &mut Ctx, bytes: &[u8], pts: Timestamp) -> Result<(), Error> {
        let Some(header) = GpuFrameHeader::from_bytes(bytes) else {
            // A malformed descriptor — drop it (per-buffer error scope), keep the pipeline alive.
            return Ok(());
        };
        let preroll = std::mem::take(&mut self.preroll_next);
        let pos_ns = pts.nanos().unwrap_or(0);
        if !preroll {
            // Pump the window *throughout* the clock wait (and while paused) — input, HUD,
            // close and resize stay live instead of freezing until the next frame. `self` and
            // `ctx` are disjoint borrows, so the pump closure can hold `&mut self`.
            match ctx.wait_until_pumping(pts, &mut || self.pump_window(pos_ns)) {
                WaitOutcome::Reached => {}
                WaitOutcome::Interrupted => return Ok(()),
            }
        }
        if self.disabled {
            // Still claim the frame so the decoder's surface is freed (no stall).
            self.claim_and_release(header.token);
            return Ok(());
        }

        // Pair with the out-of-band dmabuf frame; if it has not arrived (or no channel), drop.
        let Some(mut frame) = self.take(header.token) else { return Ok(()) };

        // Open the window on the first real frame, sized to the display geometry.
        if self.window.is_none() {
            match Window::open("streamcraft", header.disp_w.max(1) as i32, header.disp_h.max(1) as i32) {
                Ok(w) => {
                    if !w.has_dmabuf() {
                        log!(&*ctx, Level::Warn, "no_dmabuf", note = "compositor lacks zwp_linux_dmabuf_v1");
                        self.disabled = true;
                    }
                    self.window = Some(w);
                    // Reveal the controls briefly on open (then they auto-hide) so the user
                    // sees them without having to move the pointer first.
                    self.hud_deadline = Some(std::time::Instant::now() + HUD_LINGER);
                }
                Err(e) => {
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Resource(format!("waylandvideosink: window open failed: {e}")),
                    });
                    self.disabled = true;
                }
            }
        }
        if self.disabled {
            self.release(header.token);
            frame.close_fds();
            return Ok(());
        }
        let window = self.window.as_mut().expect("window present");

        // Map the VA-exported frame onto the presenter's dmabuf import params. Plane count is
        // bounded (NV12 = 2, any DRM layout ≤ 4), so this stays on the stack — no per-frame heap.
        let mut pbuf = [Plane { object_index: 0, offset: 0, pitch: 0 }; 4];
        let n = frame.planes.len().min(pbuf.len());
        for (dst, pl) in pbuf.iter_mut().zip(frame.planes.iter()) {
            *dst = Plane { object_index: pl.object_index, offset: pl.offset, pitch: pl.pitch };
        }
        let video = DmabufVideo {
            fds: &frame.fds,
            planes: &pbuf[..n],
            drm_format: frame.drm_format,
            drm_modifier: frame.drm_modifier,
            coded_w: frame.coded_w as i32,
            coded_h: frame.coded_h as i32,
            crop_x: frame.crop_x as i32,
            crop_y: frame.crop_y as i32,
            disp_w: frame.disp_w as i32,
            disp_h: frame.disp_h as i32,
            token: header.token,
        };
        let present = window.present_dmabuf(&video);
        // The compositor took its own dmabuf references at import, so our dup'd fds can go. The
        // frame's TOKEN is now held by the presenter (with the wl_buffer) and released back only
        // when the compositor is done sampling it — NOT here — so the decoder can't overwrite a
        // surface mid-scanout (the anti-tear fix for stale/torn frames).
        frame.close_fds();
        match present {
            Ok(()) => {
                self.frames = self.frames.wrapping_add(1);
                // Refresh the window once for this freshly-presented frame: release the surfaces
                // the compositor just finished with, redraw the HUD/caption, route any clicks.
                self.pump_window(pos_ns);
                // The user closed the window (button / Esc / q): stop presenting locally. The
                // pump has already pushed `Quit` to the app so the whole pipeline unwinds.
                if self.window.as_ref().is_some_and(|w| w.should_close()) {
                    self.on_close(ctx);
                }
            }
            Err(e) => {
                // Never shown → free the surface now so the decoder is not starved.
                self.release(header.token);
                let element = ctx.element();
                ctx.post(BusMessage::Warning {
                    element,
                    error: Error::Resource(format!("waylandvideosink: present failed: {e}")),
                });
                self.disabled = true;
            }
        }
        Ok(())
    }

    /// One unit of window upkeep, decoupled from the video rate — run every clock-wait slice
    /// (so the window stays live between frames and while paused) and once per presented frame:
    /// pump the socket, free compositor-released decoder surfaces, re-letterbox on resize, route
    /// clicks to pause/seek, signal `Quit` on close, and refresh the HUD + caption. `pos_ns` is
    /// the running position to show. All hits are throttled/idempotent, so calling this at the
    /// ~8 ms slice rate is cheap. No `ctx` — nothing here needs the scheduler.
    fn pump_window(&mut self, pos_ns: u64) {
        let Self {
            window,
            subtitle,
            channel,
            control,
            hud_deadline,
            hud_mapped,
            hud_sec,
            hud_paused,
            sub_shown,
            quit_sent,
            ..
        } = self;
        let Some(w) = window.as_mut() else { return };
        let _ = w.dispatch(false);
        // Deferred token release: hand back the surfaces the compositor is done sampling.
        if let Some(ch) = channel.as_ref() {
            while let Some(tok) = w.next_released_token() {
                ch.release(tok);
            }
        }
        // A resize while paused draws no new frame — re-fit the retained one here.
        let _ = w.reflow();
        // Route left-clicks: the timeline rail → seek, elsewhere → pause toggle.
        if let Some(ctrl) = control.as_ref() {
            let (ww, wh) = w.size();
            while let Some((cx, cy)) = w.take_click() {
                if cy >= wh - BAR_H && cx >= TRACK_X && cx <= ww - TRACK_PAD {
                    let tw = (ww - TRACK_X - TRACK_PAD).max(1);
                    let frac = ((cx - TRACK_X) as f32 / tw as f32).clamp(0.0, 1.0);
                    ctrl.push(crate::control::UiCommand::SeekFraction(frac));
                } else {
                    ctrl.push(crate::control::UiCommand::TogglePause);
                }
            }
            // Window close → tell the app to quit (once); the pipeline stop unwinds everything.
            if w.should_close() && !*quit_sent {
                ctrl.push(crate::control::UiCommand::Quit);
                *quit_sent = true;
            }
        }
        // HUD: shown while paused or for a few seconds after pointer activity, then auto-hidden
        // so idle playback pays nothing for it. While visible, redraw only when the content
        // actually changes (the wall-second ticks or play/pause flips) — not every pump.
        if w.has_gui() {
            if w.take_pointer_activity() {
                *hud_deadline = Some(std::time::Instant::now() + HUD_LINGER);
            }
            let (dur_ns, paused) = control
                .as_ref()
                .map(|c| (c.duration_ns(), c.paused()))
                .unwrap_or((0, false));
            let show = paused || hud_deadline.is_some_and(|d| std::time::Instant::now() < d);
            let sec = pos_ns / 1_000_000_000;
            if show {
                if !*hud_mapped || paused != *hud_paused || sec != *hud_sec {
                    *hud_mapped = true;
                    *hud_paused = paused;
                    *hud_sec = sec;
                    let (ww, wh) = w.size();
                    let hud = Hud { width: ww, pos_ns, dur_ns, paused };
                    let _ = w.present_gui(0, wh - BAR_H, ww, BAR_H, |c| draw_controls(c, &hud));
                }
            } else if *hud_mapped {
                *hud_mapped = false;
                let _ = w.hide_gui();
            }
        }
        // Caption for the current position — (re)rendered only when the visible cue changes, not
        // every pump (present_subtitle rasterizes + uploads the bitmap; doing it at the pump rate
        // was a second constant-redraw cost whenever a subtitle was on screen).
        let pts = Timestamp::from_nanos(pos_ns);
        let want = subtitle
            .as_ref()
            .filter(|c| pts >= c.start && pts < c.end)
            .map(|c| (c.start, c.end));
        if want != *sub_shown {
            match subtitle.as_ref().filter(|_| want.is_some()) {
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
            *sub_shown = want;
        }
    }

    /// Take the frame for `token` from the channel (if wired + arrived).
    fn take(&self, token: u64) -> Option<GpuFrame> {
        self.channel.as_ref()?.take(token)
    }

    /// Release a presented (or dropped) token so the decoder reclaims its surface.
    fn release(&self, token: u64) {
        if let Some(ch) = &self.channel {
            ch.release(token);
        }
    }

    /// On the disabled/drop path: claim the frame (to free its fds + surface) and release it.
    fn claim_and_release(&self, token: u64) {
        if let Some(mut f) = self.take(token) {
            f.close_fds();
        }
        self.release(token);
    }

    /// Update the current subtitle caption from a `subtitle/bitmap` buffer (latest-wins; a
    /// clear erases it). Timed by the buffer's pts + duration.
    fn push_subtitle(&mut self, ds: pgs::DisplaySet, pts: Timestamp, duration: Timestamp) {
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

    /// The window was closed by the user — warn on the bus (the app ends playback) and disable.
    fn on_close(&mut self, ctx: &mut Ctx) {
        log!(&*ctx, Level::Debug, "window_closed");
        let element = ctx.element();
        ctx.post(BusMessage::Warning {
            element,
            error: Error::Resource("waylandvideosink: window closed by user".into()),
        });
        self.window = None;
        self.disabled = true;
    }
}

/// The control-bar overlay height in px.
const BAR_H: i32 = 40;

/// How long the HUD lingers after the last pointer activity before auto-hiding (playing only —
/// a paused player keeps its controls up).
const HUD_LINGER: std::time::Duration = std::time::Duration::from_secs(3);

/// Where the timeline rail starts (after the play glyph + the `MM:SS` time) and its right pad.
const TRACK_X: i32 = 108;
const TRACK_PAD: i32 = 16;

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
    const ACCENT: u32 = 0xff30_ffa0; // streamcraft green
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

impl Element for WaylandVideoSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        // The window opens lazily on the first frame (sized to the display geometry).
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: streamcraft_core::batch::Inputs<'_>) -> Result<Flow, Error> {
        // Subtitle side-input first: update the current caption before this pass's frames (so a
        // caption landing in the same batch as its first video frame is already visible). A
        // malformed bitmap is dropped — untrusted peer data never panics.
        let mut sub_batch = ctx.take_input_on(SUBTITLE);
        while let Some(buf) = sub_batch.pop_front() {
            if let Some(ds) = pgs::decode_bitmap(buf.memory.data()) {
                self.push_subtitle(ds, buf.pts, buf.duration);
            }
        }
        ctx.recycle_input(sub_batch);

        // The video pad — the metronome.
        let entry_gen = ctx.seek_gen();
        let mut vbatch = ctx.take_input_on(VIDEO);
        while let Some(buf) = vbatch.pop_front() {
            if ctx.seek_gen() != entry_gen {
                ctx.recycle_input(vbatch);
                return Ok(Flow::Ok); // a mid-batch seek staled the rest (spec: flush/seek)
            }
            self.present(ctx, buf.memory.data(), buf.pts)?;
        }
        ctx.recycle_input(vbatch);
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FlushStart => {
                // Orphan any decoder run-ahead frames still queued for us: a seek discarded
                // their `video/gpu` buffers, so we will never pace/take them. Releasing them
                // frees the decoder's surfaces (else its pool starves → the seek stalls) and,
                // crucially, clears stale entries so a post-seek `take(token)` can't return a
                // pre-seek frame on a recycled token (the "seek shows the wrong/old frame"
                // unreliability). While playing the decoder runs ahead, so several frames sit
                // here at seek time — which is exactly why seeking while playing was slower and
                // flakier than while paused (paused → no run-ahead → nothing to orphan).
                if let Some(ch) = &self.channel {
                    ch.flush_pending();
                }
                // Drop the pre-seek caption; the subtitle track re-emits for the new position
                // (the pump hides it next pass when `pts` no longer falls in its span).
                self.subtitle = None;
                self.preroll_next = true;
            }
            Event::Eos => {
                // Keep the last frame on screen; drain window events so a close still answers.
                if let Some(w) = self.window.as_mut() {
                    let _ = w.dispatch(false);
                    if w.should_close() {
                        self.on_close(ctx);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.window = None;
    }
}
