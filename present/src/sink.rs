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
use streamcraft_core::log;
use streamcraft_core::log::Level;
use streamcraft_core::time::Timestamp;

use sc_vaapi::gpuframe::{GpuFrame, GpuFrameChannel, GpuFrameHeader, GPU_OFFER};

use crate::window::{DmabufVideo, Plane, Window};

static SINK_OFFERS: [streamcraft_core::format::OfferDesc; 1] = [GPU_OFFER];
static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];

static DESC: ElementDesc = ElementDesc {
    name: "waylandvideosink",
    pads: &PADS,
    props: &[],
    // Its own thread; paces the graph on the clock (spec: Scheduling — a windowed sink).
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
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
        WaylandVideoSink { channel: None, window: None, disabled: false, preroll_next: false }
    }

    /// Attach the shared [`GpuFrameChannel`] (the same `Arc` the zero-copy decoder holds).
    pub fn with_channel(mut self, channel: Arc<GpuFrameChannel>) -> Self {
        self.channel = Some(channel);
        self
    }

    /// Present one `video/gpu` buffer: pair it to its [`GpuFrame`], pace, import + show, release.
    fn present(&mut self, ctx: &mut Ctx, bytes: &[u8], pts: Timestamp) -> Result<(), Error> {
        let Some(header) = GpuFrameHeader::from_bytes(bytes) else {
            // A malformed descriptor — drop it (per-buffer error scope), keep the pipeline alive.
            return Ok(());
        };
        let preroll = std::mem::take(&mut self.preroll_next);
        if !preroll {
            match ctx.wait_until(pts) {
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
        };
        let present = window.present_dmabuf(&video);
        // The compositor now holds its own dmabuf references (the buffer was created), so our
        // dup'd fds can go; then release the token — the decoder may reuse the surface.
        frame.close_fds();
        self.release(header.token);
        match present {
            Ok(()) => {
                // Pump window events (close/ping/release-recycle) without blocking the graph.
                if let Some(w) = self.window.as_mut() {
                    let _ = w.dispatch(false);
                    if w.should_close() {
                        self.on_close(ctx);
                    }
                }
            }
            Err(e) => {
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

    fn process(&mut self, ctx: &mut Ctx, mut inputs: streamcraft_core::batch::Inputs<'_>) -> Result<Flow, Error> {
        let entry_gen = ctx.seek_gen();
        while let Some(buf) = inputs.pop() {
            if ctx.seek_gen() != entry_gen {
                return Ok(Flow::Ok); // a mid-batch seek staled the rest (spec: flush/seek)
            }
            self.present(ctx, buf.memory.data(), buf.pts)?;
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FlushStart => self.preroll_next = true,
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
