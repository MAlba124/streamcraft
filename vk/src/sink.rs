//! [`VkVideoSink`] — the GPU-accelerated Wayland video sink (spec: Milestone
//! applications §5; the sc-vk crate docs for the architecture). It mirrors
//! `sc-wayland`'s [`WaylandVideoSink`] element shape exactly — active, clock-paced
//! (`ctx.wait_until(pts)`), QoS drop-late with [`BusMessage::Qos`], dynamic-caps
//! configured, seek-aware, degrade-to-drops on display loss — but the per-frame work
//! is one memcpy + one fence-synchronised compute dispatch on the GPU instead of a
//! CPU conversion loop: the shader renders straight into an exported DMA-BUF that the
//! hand-written Wayland client presents via `zwp_linux_dmabuf_v1`.
//!
//! Fallback policy: if Vulkan, dma-buf export, or the compositor's dmabuf support is
//! missing, the sink posts a single bus `Warning` naming `waylandvideosink` (the CPU
//! path) and drops frames — the pipeline always completes.
//!
//! [`WaylandVideoSink`]: sc_wayland::WaylandVideoSink

use streamcraft_core::batch::Inputs;
use streamcraft_core::bus::BusMessage;
use streamcraft_core::clock::WaitOutcome;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{
    ConstraintDesc, FieldDesc, FixedFormat, OfferDesc, Value, ValueDesc,
};
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use sc_wayland::protocol::{DRM_FORMAT_MOD_LINEAR, DRM_FORMAT_XRGB8888};
use sc_wayland::WaylandClient;

use crate::gpu::{Gpu, Renderer};

const SINK: PadId = PadId(0);

// `video/raw` vocabulary — literals, matching every other video peer (interned by string).
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
const F_FPS: &str = "fps";
const PIXFMT_I420: &str = "i420";

/// How many exported framebuffers cycle through the compositor (double buffering
/// plus one in flight — the same shape as the shm swapchain's triple buffer).
const SLOTS: usize = 3;

// The shader converts i420 only (v1); the offer is constrained accordingly.
static PIXFMT_VALUES: [ValueDesc; 1] = [ValueDesc::Id(PIXFMT_I420)];
static SINK_FIELDS: [FieldDesc; 4] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
    FieldDesc { field: F_FPS, allowed: ConstraintDesc::Any, preferred: None },
];
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &SINK_FIELDS }];
static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: true, // dimensions/fps announced at runtime
    validate: None,
}];
static DESC: ElementDesc = ElementDesc {
    name: "vkvideosink",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active, // its own thread; paces the graph on the clock
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: true,
        jitter: Timestamp::ZERO,
    },
    // Config-free (window + GPU come up lazily once the format is known), so
    // parse-launch can construct it.
    make_default: Some(|| Box::new(VkVideoSink::new())),
};

/// The negotiated frame geometry, configured from a `FormatChange`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FrameFormat {
    width: usize,
    height: usize,
    /// ns per frame for the QoS lateness test; `0` disables QoS (unknown fps).
    frame_dur_ns: u64,
}

/// The GPU-accelerated Wayland video sink. Construct with [`VkVideoSink::new`]; set a
/// window title with [`with_title`](Self::with_title).
pub struct VkVideoSink {
    title: String,
    client: Option<WaylandClient>,
    gpu: Option<Gpu>,
    renderer: Option<Renderer>,
    format: Option<FrameFormat>,
    disabled: bool,
    close_reported: bool,
}

impl VkVideoSink {
    pub fn new() -> Self {
        Self {
            title: "streamcraft (vk)".to_string(),
            client: None,
            gpu: None,
            renderer: None,
            format: None,
            disabled: false,
            close_reported: false,
        }
    }

    /// Set the window title. Chainable at construction.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Post one warning and degrade to dropping frames for the rest of the run.
    fn disable(&mut self, ctx: &mut Ctx, why: String) {
        let element = ctx.element();
        ctx.post(BusMessage::Warning {
            element,
            error: Error::Resource(format!("vkvideosink: {why} — use waylandvideosink for the CPU path")),
        });
        self.disabled = true;
    }

    fn read_format(ctx: &Ctx, f: &FixedFormat) -> Option<FrameFormat> {
        let int = |name: &str| {
            ctx.field_id(name).and_then(|id| f.get(id)).and_then(|v| match v {
                Value::Int(n) if n > 0 => Some(n as usize),
                _ => None,
            })
        };
        let width = int(F_WIDTH)?;
        let height = int(F_HEIGHT)?;
        let pix = ctx
            .field_id(F_PIXFMT)
            .and_then(|id| f.get(id))
            .and_then(|v| match v {
                Value::Id(vid) => ctx.value_name(vid),
                _ => None,
            })?;
        if pix != PIXFMT_I420 {
            return None;
        }
        let frame_dur_ns = ctx
            .field_id(F_FPS)
            .and_then(|id| f.get(id))
            .and_then(|v| match v {
                Value::Rat(num, den) if num > 0 && den != 0 => {
                    Some((den.unsigned_abs() as u64).saturating_mul(1_000_000_000) / num as u64)
                }
                _ => None,
            })
            .unwrap_or(0);
        Some(FrameFormat { width, height, frame_dur_ns })
    }

    /// Bring up (or re-size) the whole path: window, Vulkan, exported framebuffers,
    /// dmabuf imports. Any missing capability degrades with one warning.
    fn configure(&mut self, ctx: &mut Ctx, f: &FixedFormat) -> Result<(), Error> {
        let Some(fmt) = Self::read_format(ctx, f) else {
            return Ok(()); // not enough fields yet — wait for a fuller announcement
        };
        let resized = self.format.is_some_and(|old| old != fmt);
        self.format = Some(fmt);
        if self.disabled || (self.renderer.is_some() && !resized) {
            return Ok(());
        }

        // Window first (cheap, and tells us about dmabuf support).
        if self.client.is_none() {
            match WaylandClient::connect() {
                Ok(mut c) => {
                    if let Err(e) = c.create_window(&self.title) {
                        self.disable(ctx, format!("window creation failed: {e:?}"));
                        return Ok(());
                    }
                    self.client = Some(c);
                }
                Err(_) => {
                    // Headless (CI): drop silently, like the shm sink.
                    self.disabled = true;
                    return Ok(());
                }
            }
        }
        let client = self.client.as_mut().expect("connected above");
        if !client.has_dmabuf() {
            self.disable(ctx, "compositor has no linux-dmabuf support".into());
            return Ok(());
        }
        if !client.dmabuf_supports(DRM_FORMAT_XRGB8888, DRM_FORMAT_MOD_LINEAR) {
            self.disable(ctx, "compositor does not import XRGB8888+LINEAR dmabufs".into());
            return Ok(());
        }

        // Vulkan (once per sink).
        if self.gpu.is_none() {
            match Gpu::new(true) {
                Ok(g) => self.gpu = Some(g),
                Err(e) => {
                    self.disable(ctx, format!("vulkan init failed: {e}"));
                    return Ok(());
                }
            }
        }
        let gpu = self.gpu.as_ref().expect("initialised above");

        // (Re)build the exported framebuffers at the new geometry.
        if let Some(old) = self.renderer.take() {
            self.client.as_mut().expect("client").clear_dmabuf_buffers();
            old.destroy(gpu);
        }
        let renderer = match Renderer::new(gpu, fmt.width, fmt.height, SLOTS, true) {
            Ok(r) => r,
            Err(e) => {
                self.disable(ctx, format!("renderer init failed: {e}"));
                return Ok(());
            }
        };
        let client = self.client.as_mut().expect("client");
        for slot in 0..renderer.slot_count() {
            let fd = renderer.slot_fd(slot).expect("export mode has fds");
            if let Err(e) = client.import_dmabuf(
                slot as u32,
                fmt.width,
                fmt.height,
                DRM_FORMAT_XRGB8888,
                DRM_FORMAT_MOD_LINEAR,
                fd,
                0,
                renderer.stride(),
            ) {
                renderer.destroy(gpu);
                self.disable(ctx, format!("dmabuf import failed: {e:?}"));
                return Ok(());
            }
        }
        match client.take_dmabuf_import_failed() {
            Ok(false) => {}
            Ok(true) => {
                renderer.destroy(gpu);
                self.disable(ctx, "compositor rejected the exported dmabuf".into());
                return Ok(());
            }
            Err(e) => {
                renderer.destroy(gpu);
                self.disable(ctx, format!("dmabuf import roundtrip failed: {e:?}"));
                return Ok(());
            }
        }
        self.renderer = Some(renderer);
        Ok(())
    }

    /// Render one frame on the clock, with QoS — the GPU twin of the shm sink's path.
    fn render(&mut self, ctx: &mut Ctx, data: &[u8], pts: Timestamp) -> Result<(), Error> {
        let Some(fmt) = self.format else { return Ok(()) };

        // QoS before waiting: a frame already > 1 frame late is dropped unconverted.
        if fmt.frame_dur_ns > 0 && pts.is_some() {
            let now = ctx.now();
            if now.is_some() && now.0 > pts.0 {
                let lateness = now.0 - pts.0;
                if lateness > fmt.frame_dur_ns {
                    let sink = ctx.element();
                    ctx.post(BusMessage::Qos { sink, lateness_ns: lateness as i64 });
                    return Ok(());
                }
            }
        }
        match ctx.wait_until(pts) {
            WaitOutcome::Reached => {}
            WaitOutcome::Interrupted => return Ok(()),
        }
        if self.disabled {
            return Ok(());
        }
        let (Some(client), Some(gpu), Some(renderer)) =
            (self.client.as_mut(), self.gpu.as_ref(), self.renderer.as_mut())
        else {
            return Ok(());
        };

        // Acquire a compositor-released slot, convert into it on the GPU, present it.
        let slot = match client.acquire_dmabuf_slot() {
            Ok(Some(s)) => s,
            Ok(None) => {
                if !self.close_reported {
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Resource("vkvideosink: window closed by user".into()),
                    });
                    self.close_reported = true;
                }
                self.disabled = true;
                return Ok(());
            }
            Err(e) => {
                self.disable(ctx, format!("acquire failed: {e:?}"));
                return Ok(());
            }
        };
        if let Err(e) = renderer.render(gpu, data, slot as usize) {
            self.disable(ctx, format!("gpu render failed: {e}"));
            return Ok(());
        }
        match client.present_dmabuf(slot, fmt.width, fmt.height) {
            Ok(true) => {}
            Ok(false) => {
                if !self.close_reported {
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Resource("vkvideosink: window closed by user".into()),
                    });
                    self.close_reported = true;
                }
                self.disabled = true;
            }
            Err(e) => self.disable(ctx, format!("present failed: {e:?}")),
        }
        Ok(())
    }
}

impl Default for VkVideoSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for VkVideoSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(()) // window + GPU come up lazily once the format is known
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.format.is_none() {
            if let Some(f) = ctx.negotiated(SINK).cloned() {
                self.configure(ctx, &f)?;
            }
        }
        let entry_gen = ctx.seek_gen();
        while let Some(buf) = inputs.pop() {
            if ctx.seek_gen() != entry_gen {
                return Ok(Flow::Ok); // seek landed mid-batch: bail, let the flush run
            }
            self.render(ctx, buf.memory.data(), buf.pts)?;
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FormatChange(f) => self.configure(ctx, f)?,
            // Keep the display responsive (ping/pong, release, close) around flushes
            // and at EOS; the last frame stays on screen until stop().
            Event::FlushStart | Event::Eos => {
                if let Some(c) = self.client.as_mut() {
                    let _ = c.pump();
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        if let (Some(r), Some(g)) = (self.renderer.take(), self.gpu.as_ref()) {
            if let Some(c) = self.client.as_mut() {
                c.clear_dmabuf_buffers();
            }
            r.destroy(g);
        }
        if let Some(mut c) = self.client.take() {
            c.destroy();
        }
        self.gpu = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desc_is_an_active_dynamic_i420_sink() {
        let d = VkVideoSink::new();
        let pad = &d.desc().pads[0];
        assert_eq!(pad.name, "sink");
        assert_eq!(pad.direction, Direction::Sink);
        assert!(pad.dynamic);
        assert!(matches!(d.desc().sched, SchedHint::Active));
        assert_eq!(d.desc().name, "vkvideosink");
    }
}
