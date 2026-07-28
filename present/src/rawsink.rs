//! [`WaylandRawSink`] (`waylandrawsink`) — the **software** (`video/raw`) counterpart of
//! [`WaylandVideoSink`](crate::sink::WaylandVideoSink). For codecs with no zero-copy VA-API path
//! (software VP8/VP9/etc.), the decoder emits CPU `video/raw` (i420 / nv12 / gray8); this sink
//! converts each frame to `wl_shm` ARGB on the CPU and presents it through the same raw-wire
//! [`Window`] — no libwayland, no GPU renderer, no SDL. It reuses the shared
//! [`WindowUi`](crate::ui::WindowUi) for the auto-hiding HUD, pointer controls, and close, and
//! the same clock pacing as the dmabuf sink. Subtitles arrive pre-composited (the CPU
//! [`SubtitleOverlay`] burns them into `video/raw` upstream), so there is no subtitle pad.
//!
//! A compositor without `wl_subcompositor`/`wp_viewporter`, or a failed window open, disables the
//! sink (frames drop) rather than failing the pipeline — a headless box should degrade.

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
use streamcraft_core::log;
use streamcraft_core::log::Level;
use streamcraft_core::time::Timestamp;

use crate::ui::WindowUi;
use crate::window::Window;

const SINK: PadId = PadId(0);

// `video/raw` vocabulary — string literals (the pipeline interns by name, so ids line up with
// any video peer regardless of which crate declared them). Mirrors `sdl3videosink`'s offer.
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
const F_FPS: &str = "fps";
const F_MATRIX: &str = "matrix";
const F_RANGE: &str = "range";
const F_TRANSFER: &str = "transfer";
const F_PRIMARIES: &str = "primaries";
const PIXFMT_I420: &str = "i420";
const PIXFMT_GRAY8: &str = "gray8";
const PIXFMT_NV12: &str = "nv12";

static PIXFMT_VALUES: [ValueDesc; 3] =
    [ValueDesc::Id(PIXFMT_I420), ValueDesc::Id(PIXFMT_GRAY8), ValueDesc::Id(PIXFMT_NV12)];
// Colorimetry fields are declared `Any` (accept whatever the producer announces, or none) — a
// field only survives interning when the consumer also declares it (repo lesson).
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
    dynamic: true, // dimensions/fps announced at runtime via FormatChange
    validate: None,
}];

// COLD: `make_default` boxes one instance at construction, never per frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "waylandrawsink",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active, // its own thread; paces the graph on the clock
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(WaylandRawSink::new())),
};

/// The pixel formats this sink converts on the CPU.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pix {
    I420,
    Nv12,
    Gray8,
}

impl Pix {
    fn from_name(s: &str) -> Option<Pix> {
        match s {
            PIXFMT_I420 => Some(Pix::I420),
            PIXFMT_NV12 => Some(Pix::Nv12),
            PIXFMT_GRAY8 => Some(Pix::Gray8),
            _ => None,
        }
    }
}

/// Fixed-point (.8) YCbCr→RGB coefficients: `R = (yc·(Y−yoff) + rv·(V−128))>>8`, etc.
#[derive(Clone, Copy)]
struct Coef {
    yc: i32,
    rv: i32,
    gv: i32,
    gu: i32,
    bu: i32,
    yoff: i32,
}

impl Coef {
    /// Resolve from the announced matrix + range (H.273-ish: unknown matrix defaults by height —
    /// ≥720 → BT.709 else BT.601; unknown range → limited). Covers the common SD/HD/UHD cases.
    fn resolve(matrix: Option<&str>, range: Option<&str>, height: usize) -> Coef {
        let full = matches!(range, Some(r) if r.contains("full") || r.contains("jpeg") || r.contains("pc"));
        let m = matrix.unwrap_or("");
        let kind = if m.contains("2020") {
            2020
        } else if m.contains("709") {
            709
        } else if m.contains("601") || m.contains("470") || m.contains("170") || m.contains("240") {
            601
        } else if height >= 720 {
            709
        } else {
            601
        };
        match (kind, full) {
            (709, false) => Coef { yc: 298, rv: 459, gv: -137, gu: -55, bu: 541, yoff: 16 },
            (709, true) => Coef { yc: 256, rv: 403, gv: -120, gu: -48, bu: 475, yoff: 0 },
            (2020, false) => Coef { yc: 298, rv: 459, gv: -137, gu: -55, bu: 541, yoff: 16 },
            (2020, true) => Coef { yc: 256, rv: 439, gv: -170, gu: -49, bu: 561, yoff: 0 },
            (_, false) => Coef { yc: 298, rv: 409, gv: -208, gu: -100, bu: 516, yoff: 16 },
            (_, true) => Coef { yc: 256, rv: 359, gv: -183, gu: -88, bu: 454, yoff: 0 },
        }
    }
}

/// The negotiated frame geometry + how to convert it.
#[derive(Clone, Copy)]
struct FrameFormat {
    width: usize,
    height: usize,
    pix: Pix,
    coef: Coef,
    /// ns/frame (from fps) for the QoS "more than one frame late → drop" test; `0` disables it.
    frame_dur_ns: u64,
}

/// The software `video/raw` → `wl_shm` ARGB Wayland sink.
pub struct WaylandRawSink {
    window: Option<Window>,
    disabled: bool,
    preroll_next: bool,
    /// The negotiated format, learned from a `FormatChange` (or a fully-fixed link-time format).
    format: Option<FrameFormat>,
    /// Reused ARGB8888 scratch (`wl_shm` byte order); (re)sized only on a format change (cold).
    argb: Vec<u8>,
    /// The shared interactive overlay layer (HUD, pointer controls, close).
    ui: WindowUi,
}

impl Default for WaylandRawSink {
    fn default() -> Self {
        Self::new()
    }
}

impl WaylandRawSink {
    // COLD: constructor.
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        WaylandRawSink {
            window: None,
            disabled: false,
            preroll_next: false,
            format: None,
            argb: Vec::new(),
            ui: WindowUi::new(),
        }
    }

    /// The default element handle (for the autoplug controller).
    pub fn new_element() -> Box<dyn Element> {
        Box::new(Self::new())
    }

    /// Attach the window↔app control channel (clicks → pause/seek; duration/pause → HUD).
    pub fn with_control(mut self, control: std::sync::Arc<crate::control::PlayerControl>) -> Self {
        self.ui.set_control(control);
        self
    }

    /// Learn geometry from a negotiated `video/raw` format and size the ARGB scratch. Cold: runs
    /// on a `FormatChange` (or first-frame link-time format), never per frame.
    #[allow(clippy::disallowed_methods)] // one-time scratch (re)size on format change, not the hot path
    fn configure(&mut self, ctx: &Ctx, f: &FixedFormat) {
        let read_int = |name: &str| -> Option<i64> {
            ctx.field_id(name).and_then(|id| f.get(id)).and_then(|v| match v {
                Value::Int(n) => Some(n),
                _ => None,
            })
        };
        let read_id = |name: &str| -> Option<&str> {
            ctx.field_id(name).and_then(|id| f.get(id)).and_then(|v| match v {
                Value::Id(vid) => ctx.value_name(vid),
                _ => None,
            })
        };
        let (Some(width), Some(height)) = (read_int(F_WIDTH), read_int(F_HEIGHT)) else {
            return; // not enough fields yet — wait for a fuller announcement
        };
        let (width, height) = (width.max(1) as usize, height.max(1) as usize);
        let Some(pix) = read_id(F_PIXFMT).and_then(Pix::from_name) else {
            return;
        };
        let coef = Coef::resolve(read_id(F_MATRIX), read_id(F_RANGE), height);
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
        self.argb.resize(width * height * 4, 0);
        self.format = Some(FrameFormat { width, height, pix, coef, frame_dur_ns });
        log!(&*ctx, Level::Info, "configure", width = width as u64, height = height as u64);
    }

    /// Convert + present one `video/raw` frame on the clock, pumping the window throughout.
    fn render(&mut self, ctx: &mut Ctx, data: &[u8], pts: Timestamp) -> Result<(), Error> {
        let Some(fmt) = self.format else { return Ok(()) };
        let preroll = std::mem::take(&mut self.preroll_next);
        let pos_ns = pts.nanos().unwrap_or(0);

        // QoS: drop a frame already more than one frame-time late — tested *before* the wait +
        // convert, so a late frame costs no work (spec: QoS).
        if !preroll && fmt.frame_dur_ns > 0 && pts.is_some() {
            let now = ctx.now();
            if now.is_some() && now.0 > pts.0 && now.0 - pts.0 > fmt.frame_dur_ns {
                let sink = ctx.element();
                ctx.post(BusMessage::Qos { sink, lateness_ns: (now.0 - pts.0) as i64 });
                return Ok(());
            }
        }

        // Pace on the clock, keeping the window live (input/HUD/close) throughout — and while
        // paused. `self` and `ctx` are disjoint borrows, so the pump can hold `&mut self`.
        if !preroll {
            match ctx.wait_until_pumping(pts, &mut || self.pump_window(pos_ns)) {
                WaitOutcome::Reached => {}
                WaitOutcome::Interrupted => return Ok(()),
            }
        }
        if self.disabled {
            return Ok(());
        }

        // Open the window on the first frame, sized to the video.
        if self.window.is_none() {
            match Window::open("streamcraft", fmt.width as i32, fmt.height as i32) {
                Ok(w) => {
                    if !w.has_gui() {
                        log!(&*ctx, Level::Warn, "no_subcompositor");
                        self.disabled = true;
                    }
                    self.window = Some(w);
                    self.ui.arm_hud();
                }
                Err(e) => {
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Resource(format!("waylandrawsink: window open failed: {e}")),
                    });
                    self.disabled = true;
                }
            }
        }
        if self.disabled {
            return Ok(());
        }

        // CPU colour-convert into the reused ARGB scratch, then present via shm.
        convert(fmt, data, &mut self.argb);
        let (w, h) = (fmt.width as i32, fmt.height as i32);
        let present = self
            .window
            .as_mut()
            .expect("window present")
            .present_raw(&self.argb, w, h);
        match present {
            Ok(()) => {
                self.pump_window(pos_ns);
                if self.window.as_ref().is_some_and(|w| w.should_close()) {
                    self.on_close(ctx);
                }
            }
            Err(e) => {
                let element = ctx.element();
                ctx.post(BusMessage::Warning {
                    element,
                    error: Error::Resource(format!("waylandrawsink: present failed: {e}")),
                });
                self.disabled = true;
            }
        }
        Ok(())
    }

    /// Window upkeep, decoupled from the video rate (see [`WindowUi::pump`]).
    fn pump_window(&mut self, pos_ns: u64) {
        if let Some(w) = self.window.as_mut() {
            self.ui.pump(w, pos_ns);
        }
    }

    /// The user closed the window — warn on the bus (the app ends playback) and disable.
    fn on_close(&mut self, ctx: &mut Ctx) {
        log!(&*ctx, Level::Debug, "window_closed");
        let element = ctx.element();
        ctx.post(BusMessage::Warning {
            element,
            error: Error::Resource("waylandrawsink: window closed by user".into()),
        });
        self.window = None;
        self.disabled = true;
    }
}

impl Element for WaylandRawSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(()) // the window opens lazily on the first frame
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // First data with no FormatChange yet: try a fully-fixed link-time format.
        if self.format.is_none() {
            if let Some(f) = ctx.negotiated(SINK).cloned() {
                self.configure(ctx, &f);
            }
        }
        let entry_gen = ctx.seek_gen();
        while let Some(buf) = inputs.pop() {
            if ctx.seek_gen() != entry_gen {
                return Ok(Flow::Ok); // a mid-batch seek staled the rest (spec: flush/seek)
            }
            self.render(ctx, buf.memory.data(), buf.pts)?;
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FormatChange(f) => self.configure(ctx, f),
            Event::FlushStart => {
                self.ui.on_flush();
                self.preroll_next = true;
            }
            Event::Eos => {
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

/// Convert one `video/raw` frame to `wl_shm` ARGB8888 (little-endian `[B,G,R,A]`, opaque) in
/// `out` (`width*height*4`). Assumes tightly packed planes (stride == width), as the decoders
/// emit. Chroma is 4:2:0 (i420 / nv12); gray8 is luma-only (neutral chroma).
fn convert(fmt: FrameFormat, data: &[u8], out: &mut [u8]) {
    let (w, h) = (fmt.width, fmt.height);
    let c = fmt.coef;
    let (cw, ch) = ((w + 1) / 2, (h + 1) / 2);
    let y_plane = &data[..(w * h).min(data.len())];
    // Locate the chroma source for this pixel format.
    let sample_uv = |x: usize, y: usize| -> (i32, i32) {
        match fmt.pix {
            Pix::Gray8 => (128, 128),
            Pix::I420 => {
                let u0 = w * h;
                let v0 = u0 + cw * ch;
                let idx = (y / 2) * cw + (x / 2);
                let u = data.get(u0 + idx).copied().unwrap_or(128) as i32;
                let v = data.get(v0 + idx).copied().unwrap_or(128) as i32;
                (u, v)
            }
            Pix::Nv12 => {
                let uv0 = w * h;
                let idx = uv0 + (y / 2) * (cw * 2) + (x / 2) * 2;
                let u = data.get(idx).copied().unwrap_or(128) as i32;
                let v = data.get(idx + 1).copied().unwrap_or(128) as i32;
                (u, v)
            }
        }
    };
    for y in 0..h {
        for x in 0..w {
            let yy = y_plane.get(y * w + x).copied().unwrap_or(0) as i32;
            let (u, v) = sample_uv(x, y);
            let yv = c.yc * (yy - c.yoff);
            let r = ((yv + c.rv * (v - 128)) >> 8).clamp(0, 255) as u8;
            let g = ((yv + c.gv * (v - 128) + c.gu * (u - 128)) >> 8).clamp(0, 255) as u8;
            let b = ((yv + c.bu * (u - 128)) >> 8).clamp(0, 255) as u8;
            let d = (y * w + x) * 4;
            out[d] = b;
            out[d + 1] = g;
            out[d + 2] = r;
            out[d + 3] = 255;
        }
    }
}
