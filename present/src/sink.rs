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

use sc_text::pgs;
use sc_vaapi::gpuframe::{GpuFrame, GpuFrameChannel, GpuFrameHeader, GPU_OFFER};

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
    /// Frames presented.
    frames: u64,
    /// The shared interactive overlay layer (controls, HUD, caption, close) — driven from the
    /// clock-wait pump.
    ui: crate::ui::WindowUi,
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
            ui: crate::ui::WindowUi::new(),
        }
    }

    /// Attach the shared [`GpuFrameChannel`] (the same `Arc` the zero-copy decoder holds).
    pub fn with_channel(mut self, channel: Arc<GpuFrameChannel>) -> Self {
        self.channel = Some(channel);
        self
    }

    /// Attach the window↔app control channel for interactive controls (clicks → pause/seek).
    pub fn with_control(mut self, control: Arc<crate::control::PlayerControl>) -> Self {
        self.ui.set_control(control);
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
                    self.ui.arm_hud();
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

    /// One unit of window upkeep, decoupled from the video rate — run every clock-wait slice (so
    /// the window stays live between frames and while paused) and once per presented frame. Frees
    /// the compositor-released decoder surfaces (deferred token release, the anti-tear contract),
    /// then hands off to the shared [`WindowUi`](crate::ui::WindowUi) for input/HUD/caption/close.
    fn pump_window(&mut self, pos_ns: u64) {
        let Self { window, channel, ui, .. } = self;
        let Some(w) = window.as_mut() else { return };
        // Deferred token release (before the UI pump's own dispatch also runs): hand back the
        // surfaces the compositor is done sampling so the decoder can reuse them.
        let _ = w.dispatch(false);
        if let Some(ch) = channel.as_ref() {
            while let Some(tok) = w.next_released_token() {
                ch.release(tok);
            }
        }
        ui.pump(w, pos_ns);
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
                self.ui.push_subtitle(ds, buf.pts, buf.duration);
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
                self.ui.on_flush();
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
