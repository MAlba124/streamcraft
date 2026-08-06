//! [`Sdl3VideoSink`] — the SDL3 video sink (spec: Milestone applications §5 — play a
//! video file, the display half; windowing/graphics ride SDL3). Semantics are the
//! `WaylandVideoSink` contract it replaces, unchanged: an **active** sink that renders
//! each frame **on the pipeline clock** (`ctx.wait_until(pts)`), reads its concrete
//! format from the upstream decoder's `FormatChange` announcement (dynamic caps), and
//! cooperates with flush/seek by bailing out of long waits when `ctx.seek_gen()`
//! changes.
//!
//! Per frame:
//! * **QoS** first: a frame already more than one frame-duration late is dropped with
//!   [`BusMessage::Qos`] before any upload cost;
//! * pace with `ctx.wait_until(pts)` (an `Interrupted` wait returns cleanly);
//! * upload the I420 planes to a streaming GPU texture and present (YUV→RGB is the
//!   renderer's job now — the CPU conversion pass the shm sink carried is gone);
//! * drain window events between frames. On close the sink posts a bus `Warning`
//!   once and thereafter accepts frames as silent drops — it never hangs the graph.
//!
//! On `Eos` the last presented frame stays on screen (the window lives until
//! `stop`). No compositor / no display degrades to dropping frames, not failure —
//! a headless CI box must still finish the pipeline.

use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::clock::WaitOutcome;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{
    ConstraintDesc, FieldDesc, FixedFormat, OfferDesc, Value, ValueDesc,
};
use profluens_core::id::PadId;
use profluens_core::log;
use profluens_core::log::Level;
use profluens_core::time::Timestamp;

use crate::gpu::renderer::PlaneFormat;
use crate::gpu::{Colorimetry, GpuRenderer};
use crate::video::VideoWindow;

const SINK: PadId = PadId(0);

// `video/raw` vocabulary — literals so the crate does not pin field-name identity to
// profluens-video (the pipeline interns by string, so ids line up with any video peer).
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
const F_FPS: &str = "fps";
// The pinned colorimetry contract (a parallel task adds the producer side — these
// names are the exact fields the GPU color pipeline reads). All optional; the
// defaults rule in `gpu::Colorimetry::resolve` applies when a field is absent.
const F_MATRIX: &str = "matrix";
const F_RANGE: &str = "range";
const F_TRANSFER: &str = "transfer";
const F_PRIMARIES: &str = "primaries";
const PIXFMT_I420: &str = "i420";
const PIXFMT_GRAY8: &str = "gray8";
const PIXFMT_NV12: &str = "nv12";

// The sink offers a broad `video/raw` template constrained to the pixel formats we can
// present (i420 natively; gray8 as flat-chroma IYUV). Dimensions/fps arrive at runtime
// via the decoder's FormatChange, so the pad is `dynamic`.
static PIXFMT_VALUES: [ValueDesc; 3] =
    [ValueDesc::Id(PIXFMT_I420), ValueDesc::Id(PIXFMT_GRAY8), ValueDesc::Id(PIXFMT_NV12)];
// The colorimetry fields are declared with `ConstraintDesc::Any` so the sink accepts
// whatever the producer announces (or none). Declaring them matters: an announced
// field only survives interning when the consumer also declares it (hard-won repo
// lesson), so absent these four the GPU pipeline could never read colorimetry.
static SINK_FIELDS: [FieldDesc; 8] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
    FieldDesc { field: F_FPS, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_MATRIX, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_RANGE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_TRANSFER, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PRIMARIES, allowed: ConstraintDesc::Any, preferred: None },
];
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &SINK_FIELDS }];
static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: true, // dimensions/fps announced at runtime
    validate: None,
}];
// COLD: `make_default` boxes one element instance at pipeline construction, never per frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "sdl3videosink",
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
    // Config-free: the window is created once the format is known, so parse-launch
    // can construct it.
    make_default: Some(|| Box::new(Sdl3VideoSink::new())),
};

/// The pixel format we accept and know how to present.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pix {
    I420,
    Gray8,
    /// Two-plane 4:2:0 (Y + interleaved CbCr) — what hardware decoders emit;
    /// SDL renders it natively, so this path never converts on the CPU.
    Nv12,
}

impl Pix {
    fn from_name(s: &str) -> Option<Pix> {
        match s {
            PIXFMT_I420 => Some(Pix::I420),
            PIXFMT_GRAY8 => Some(Pix::Gray8),
            PIXFMT_NV12 => Some(Pix::Nv12),
            _ => None,
        }
    }
}

/// The negotiated frame geometry the sink configured from a `FormatChange`.
#[derive(Clone, Copy)]
struct FrameFormat {
    width: usize,
    height: usize,
    pixfmt: Pix,
    /// Nanoseconds per frame (from fps), used for the QoS "more than one frame late"
    /// test. `0` when fps is unknown — then QoS is disabled (never drop for lateness).
    frame_dur_ns: u64,
    /// The resolved colorimetry (parsed from the negotiated matrix/range/transfer/
    /// primaries fields with the H.273 defaults rule) — drives the GPU color pipeline.
    colorimetry: Colorimetry,
}

impl Pix {
    /// The GPU plane arrangement for this pixel format.
    fn plane_format(self) -> PlaneFormat {
        match self {
            Pix::I420 => PlaneFormat::I420,
            Pix::Nv12 => PlaneFormat::Nv12,
            Pix::Gray8 => PlaneFormat::Gray8,
        }
    }
}

/// Which presentation backend the sink is using.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// The owned GPU render pipeline (custom SPIR-V color shaders).
    Gpu,
    /// The classic `SDL_Renderer` fixed-function YUV path.
    Classic,
}

/// The `PF_RENDER` env override: `gpu` forces the GPU path (no fallback attempt is
/// skipped, but classic is not tried first), `classic` forces the classic path, and
/// anything else (or unset) is `auto` — try GPU, fall back to classic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BackendPref {
    Auto,
    Gpu,
    Classic,
}

impl BackendPref {
    fn from_env() -> BackendPref {
        match std::env::var("PF_RENDER").ok().as_deref() {
            Some("gpu") => BackendPref::Gpu,
            Some("classic") => BackendPref::Classic,
            _ => BackendPref::Auto,
        }
    }
}

/// The SDL3 video sink. Construct with [`Sdl3VideoSink::new`] and set a window title
/// with [`with_title`](Self::with_title).
pub struct Sdl3VideoSink {
    title: String,
    window: Option<VideoWindow>,
    /// The owned GPU renderer, when the GPU backend is active (else the classic
    /// `SDL_Renderer` path in `window` presents).
    gpu: Option<GpuRenderer>,
    /// Which backend is presenting (once a window is open).
    backend: Backend,
    format: Option<FrameFormat>,
    /// Set once we could not open a window / hit a fatal render error, so we stop
    /// trying to present and accept frames as drops (never hang the pipeline).
    disabled: bool,
    /// Whether we already posted the "window closed" warning (post once).
    close_reported: bool,
    /// Render the next frame immediately — no QoS, no clock wait (spec: flush/seek).
    /// Set on `FlushStart` so the first post-seek frame *prerolls* to the screen:
    /// while paused the user sees the seeked frame; while playing the picture
    /// updates instantly instead of waiting out the first deadline.
    preroll_next: bool,
}

impl Sdl3VideoSink {
    // COLD: one-time element construction — sets the default window title string once.
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        Self {
            title: "profluens".to_string(),
            window: None,
            gpu: None,
            backend: Backend::Classic,
            format: None,
            disabled: false,
            close_reported: false,
            preroll_next: false,
        }
    }

    /// Set the window title. Chainable at construction.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Read a `video/raw` `FixedFormat` (link-time or FormatChange) into a
    /// [`FrameFormat`]. `None` if a required field is missing or the pixfmt is
    /// unsupported.
    fn read_format(ctx: &Ctx, f: &FixedFormat) -> Option<FrameFormat> {
        let width = ctx.field_id(F_WIDTH).and_then(|id| f.get(id)).and_then(int_value)? as usize;
        let height = ctx.field_id(F_HEIGHT).and_then(|id| f.get(id)).and_then(int_value)? as usize;
        let pix_name = ctx
            .field_id(F_PIXFMT)
            .and_then(|id| f.get(id))
            .and_then(|v| match v {
                Value::Id(vid) => ctx.value_name(vid),
                _ => None,
            })?;
        let pixfmt = Pix::from_name(pix_name)?;
        // fps is optional; when present compute the per-frame duration for QoS.
        let frame_dur_ns = ctx
            .field_id(F_FPS)
            .and_then(|id| f.get(id))
            .and_then(|v| match v {
                Value::Rat(num, den) if num > 0 && den != 0 => {
                    // ns/frame = den/num * 1e9
                    Some((den.unsigned_abs() as u64).saturating_mul(1_000_000_000) / num as u64)
                }
                _ => None,
            })
            .unwrap_or(0);
        // Colorimetry (all optional; the H.273 defaults rule fills the gaps). Each is
        // a categorical `Value::Id` resolved back to its name.
        let read_id = |field: &str| -> Option<&str> {
            ctx.field_id(field)
                .and_then(|id| f.get(id))
                .and_then(|v| match v {
                    Value::Id(vid) => ctx.value_name(vid),
                    _ => None,
                })
        };
        let colorimetry = Colorimetry::resolve(
            height,
            read_id(F_MATRIX),
            read_id(F_RANGE),
            read_id(F_TRANSFER),
            read_id(F_PRIMARIES),
        );
        Some(FrameFormat { width, height, pixfmt, frame_dur_ns, colorimetry })
    }

    /// Configure the sink from a negotiated format: learn the geometry and, on first
    /// use, open the window. Failure to open one disables the sink (frames become
    /// drops) rather than failing the pipeline — a headless CI box has no display,
    /// and the sink should degrade, not crash.
    fn configure(&mut self, ctx: &mut Ctx, f: &FixedFormat) -> Result<(), Error> {
        let Some(fmt) = Self::read_format(ctx, f) else {
            return Ok(()); // not enough fields yet; wait for a fuller announcement
        };
        log!(
            &*ctx,
            Level::Debug,
            "configured",
            width = fmt.width,
            height = fmt.height,
            frame_dur_ns = fmt.frame_dur_ns,
        );
        self.format = Some(fmt);

        // A live geometry / pixfmt change on an already-open window is fine: both
        // backends recreate their plane textures on the next present when the format
        // record differs. Only open a window / pick a backend on first use.
        if self.disabled || self.window.is_some() {
            return Ok(());
        }
        self.open_backend(ctx, fmt);
        Ok(())
    }

    /// Open a window and choose the presentation backend: `PF_RENDER=gpu|classic`
    /// forces one; otherwise (auto) try the owned GPU pipeline and fall back to the
    /// classic `SDL_Renderer` path on any GPU init failure. Failing to open any window
    /// (headless / CI) disables the sink (frames drop) — never fails the pipeline.
    fn open_backend(&mut self, ctx: &mut Ctx, fmt: FrameFormat) {
        let pref = BackendPref::from_env();
        let c = fmt.colorimetry;
        let (w, h) = (fmt.width as u32, fmt.height as u32);

        // Try GPU unless explicitly classic.
        if pref != BackendPref::Classic {
            match VideoWindow::open_windowed(&self.title, w, h) {
                // SAFETY: `win` is a live window we just created on this thread with
                // the video subsystem up, and it is stored in `self.window` (kept
                // alive) alongside the returned renderer; the renderer unclaims it
                // before the window is destroyed (see Drop ordering in on_close/stop).
                #[allow(unsafe_code)]
                Ok(win) => match unsafe { GpuRenderer::new(win.raw_window()) } {
                    Ok(gpu) => {
                        log!(
                            &*ctx, Level::Info, "backend",
                            render = "gpu", driver = gpu.driver(),
                            matrix = c.matrix.name(), range = c.range.name(),
                            transfer = c.transfer.name(), primaries = c.primaries.name(),
                        );
                        self.window = Some(win);
                        self.gpu = Some(gpu);
                        self.backend = Backend::Gpu;
                        return;
                    }
                    Err(_e) => {
                        // GPU init failed: drop this windowed window (no classic
                        // renderer attached) and fall through to the classic path,
                        // unless the user forced GPU.
                        drop(win);
                        if pref == BackendPref::Gpu {
                            log!(&*ctx, Level::Debug, "gpu_forced_failed");
                            self.disabled = true;
                            return;
                        }
                        log!(&*ctx, Level::Debug, "gpu_fallback_classic");
                    }
                },
                Err(_e) => {
                    // No display at all: disable (headless degrade).
                    log!(&*ctx, Level::Debug, "no_display_disabled");
                    self.disabled = true;
                    return;
                }
            }
        }

        // Classic path (forced, or GPU fell back).
        match VideoWindow::open(&self.title, w, h) {
            Ok(win) => {
                log!(
                    &*ctx, Level::Info, "backend",
                    render = "classic",
                    matrix = c.matrix.name(), range = c.range.name(),
                    transfer = c.transfer.name(), primaries = c.primaries.name(),
                );
                self.window = Some(win);
                self.backend = Backend::Classic;
            }
            Err(_e) => {
                log!(&*ctx, Level::Debug, "no_display_disabled");
                self.disabled = true;
            }
        }
    }

    /// Present one frame's bytes on the clock, with QoS. Returns `Ok(())` always — a
    /// presentation failure disables the sink rather than failing the pipeline.
    fn render(&mut self, ctx: &mut Ctx, data: &[u8], pts: Timestamp) -> Result<(), Error> {
        let Some(fmt) = self.format else { return Ok(()) };

        // Preroll (spec: flush/seek): the first frame after a flush presents
        // immediately — no QoS, no wait — so a paused seek shows its new frame.
        let preroll = std::mem::take(&mut self.preroll_next);

        // QoS: if the deadline is already behind us by more than one frame, drop it —
        // tested *before* the wait, so a late frame costs no upload (spec: QoS).
        if !preroll && fmt.frame_dur_ns > 0 && pts.is_some() {
            let now = ctx.now();
            if now.is_some() && now.0 > pts.0 {
                let lateness = now.0 - pts.0;
                if lateness > fmt.frame_dur_ns {
                    log!(&*ctx, Level::Debug, "qos_drop", lateness_ns = lateness);
                    let sink = ctx.element();
                    ctx.post(BusMessage::Qos { sink, lateness_ns: lateness as i64 });
                    return Ok(());
                }
            }
        }

        // Pace on the clock. An interrupt (flush/shutdown) returns cleanly.
        if !preroll {
            match ctx.wait_until(pts) {
                WaitOutcome::Reached => {}
                WaitOutcome::Interrupted => return Ok(()),
            }
        }

        if self.disabled {
            return Ok(());
        }
        let Some(window) = self.window.as_mut() else { return Ok(()) };

        // Answer close promptly even when frames are sparse.
        if window.pump().closed {
            self.on_close(ctx);
            return Ok(());
        }

        // Present via the active backend: the owned GPU pipeline (colorimetry-aware
        // custom shaders) or the classic fixed-function SDL_Renderer path.
        let shown = match (self.backend, self.gpu.as_mut()) {
            (Backend::Gpu, Some(gpu)) => gpu.present(
                data,
                fmt.width,
                fmt.height,
                fmt.pixfmt.plane_format(),
                fmt.colorimetry,
            ),
            _ => match fmt.pixfmt {
                Pix::I420 => window.present_i420(data, fmt.width, fmt.height),
                Pix::Gray8 => window.present_gray8(data, fmt.width, fmt.height),
                Pix::Nv12 => window.present_nv12(data, fmt.width, fmt.height),
            },
        };
        match shown {
            Ok(drawn) => {
                log!(&*ctx, Level::Trace, "present", pts = pts, drawn = drawn);
            }
            Err(e) => {
                // A render error after a successful open is real: report and degrade.
                let element = ctx.element();
                ctx.post(BusMessage::Warning { element, error: e });
                self.disabled = true;
            }
        }
        Ok(())
    }

    /// The user closed the window: report once, then drop frames silently.
    fn on_close(&mut self, ctx: &mut Ctx) {
        log!(&*ctx, Level::Debug, "window_closed");
        if !self.close_reported {
            let element = ctx.element();
            ctx.post(BusMessage::Warning {
                element,
                error: Error::Resource("sdl3videosink: window closed by user".into()),
            });
            self.close_reported = true;
        }
        self.disabled = true;
        // Drop the GPU renderer *before* the window: it unclaims the window from its
        // device, and the window is destroyed on its own drop.
        self.gpu = None;
        self.window = None;
    }
}

impl Default for Sdl3VideoSink {
    fn default() -> Self {
        Self::new()
    }
}

fn int_value(v: Value) -> Option<i64> {
    match v {
        Value::Int(n) => Some(n),
        _ => None,
    }
}

impl Element for Sdl3VideoSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        // The window is created lazily once the format is known (FormatChange or a
        // fully-fixed link-time format), exactly like the pipewire sink configures
        // its device on first use.
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // On first data, if no FormatChange arrived, try a fully-fixed link-time format.
        if self.format.is_none() {
            if let Some(f) = ctx.negotiated(SINK).cloned() {
                self.configure(ctx, &f)?;
            }
        }

        // A seek that lands mid-batch makes the remaining frames stale: bail and let
        // the scheduler run the flush (spec: flush/seek — blocking sinks bail on
        // ctx.seek_gen()).
        let entry_gen = ctx.seek_gen();
        while let Some(buf) = inputs.pop() {
            if ctx.seek_gen() != entry_gen {
                return Ok(Flow::Ok);
            }
            // `buf` is owned (moved out of the batch), so borrowing its bytes does
            // not clash with the `&mut ctx` the render path needs — the upload reads
            // straight from the decoder's buffer, no per-frame copy.
            self.render(ctx, buf.memory.data(), buf.pts)?;
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FormatChange(f) => self.configure(ctx, f)?,
            // Flush/seek and EOS: nothing is carried (one frame per buffer), but
            // drain window events so a close is answered promptly while no frames
            // flow; on EOS the last frame stays on screen until stop().
            Event::FlushStart | Event::Eos => {
                if matches!(event, Event::FlushStart) {
                    self.preroll_next = true;
                }
                if let Some(w) = self.window.as_mut() {
                    if w.pump().closed {
                        self.on_close(ctx);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // Tear down GPU renderer first (it unclaims the window), then the window
        // (Drop of VideoWindow closes SDL cleanly).
        self.gpu = None;
        self.window = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixfmt_names_round_trip() {
        assert!(matches!(Pix::from_name("i420"), Some(Pix::I420)));
        assert!(matches!(Pix::from_name("gray8"), Some(Pix::Gray8)));
        assert!(matches!(Pix::from_name("nv12"), Some(Pix::Nv12)));
    }
}
