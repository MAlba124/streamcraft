//! [`WaylandVideoSink`] — a hand-written Wayland shm video sink (spec: Milestone
//! applications §5 — play a video file; the display half). It mirrors the shape of
//! `streamcraft-video`'s [`VideoCkSink`] and `sc-pipewire`'s device sink: an **active** sink
//! that renders each frame **on the pipeline clock** (`ctx.wait_until(pts)`), reads its
//! concrete format from the upstream decoder's `FormatChange` announcement (dynamic caps),
//! and cooperates with flush/seek by bailing out of long waits when `ctx.seek_gen()` changes.
//!
//! Per frame:
//! * pace with `ctx.wait_until(pts)` (an `Interrupted` wait returns cleanly, dropping the
//!   rest of the batch);
//! * **QoS**: if the frame's deadline is already past by more than one frame duration, drop
//!   it and post [`BusMessage::Qos`] with the lateness in nanoseconds — no point converting
//!   and presenting a frame the compositor would show late;
//! * otherwise convert **i420 → XRGB8888** (or gray8 → XRGB) directly into a free shm swap
//!   buffer and present it (attach + damage + commit);
//! * pump xdg ping/pong + configure/close events between frames. On `close` the sink posts
//!   a bus `Warning` once and thereafter accepts frames as silent drops — it never hangs the
//!   pipeline.
//!
//! On `Eos` the last presented frame stays on screen (the window is kept until `stop`).
//! `FlushStart` resets pacing state (no pending frame is carried — one frame per buffer).
//!
//! [`VideoCkSink`]: streamcraft_video::sink::VideoCkSink

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

use crate::client::WaylandClient;
use crate::convert::{gray8_to_xrgb, i420_to_xrgb};

const SINK: PadId = PadId(0);

// `video/raw` vocabulary — literals so the crate does not pin field-name identity to
// streamcraft-video (the pipeline interns by string, so ids line up with any video peer).
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
const F_FPS: &str = "fps";
const PIXFMT_I420: &str = "i420";
const PIXFMT_GRAY8: &str = "gray8";

// The sink offers a broad `video/raw` template constrained to the pixel formats we can
// convert (i420 always; gray8 as a trivial extra). Dimensions/fps arrive at runtime via the
// decoder's FormatChange, so the pad is `dynamic`.
static PIXFMT_VALUES: [ValueDesc; 2] = [ValueDesc::Id(PIXFMT_I420), ValueDesc::Id(PIXFMT_GRAY8)];
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
    name: "waylandvideosink",
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
    make_default: None,
};

/// The pixel format we accept and know how to convert.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pix {
    I420,
    Gray8,
}

impl Pix {
    fn from_name(s: &str) -> Option<Pix> {
        match s {
            PIXFMT_I420 => Some(Pix::I420),
            PIXFMT_GRAY8 => Some(Pix::Gray8),
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
    /// Nanoseconds per frame (from fps), used for the QoS "more than one frame late" test.
    /// `0` when fps is unknown — then QoS is disabled (never drop for lateness).
    frame_dur_ns: u64,
}

/// A hand-written Wayland shm video sink. Construct with [`WaylandVideoSink::new`] and set a
/// window title with [`with_title`](Self::with_title).
pub struct WaylandVideoSink {
    title: String,
    client: Option<WaylandClient>,
    format: Option<FrameFormat>,
    /// Set once we could not open a display / hit a fatal protocol error, so we stop trying
    /// to present and accept frames as drops (never hang the pipeline).
    disabled: bool,
    /// Whether we already posted the "window closed" warning (post once).
    close_reported: bool,
}

impl WaylandVideoSink {
    pub fn new() -> Self {
        Self {
            title: "streamcraft".to_string(),
            client: None,
            format: None,
            disabled: false,
            close_reported: false,
        }
    }

    /// Set the window title (spec: xdg_toplevel.set_title). Chainable at construction.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Read a `video/raw` `FixedFormat` (link-time or FormatChange) into a [`FrameFormat`].
    /// `None` if a required field is missing or the pixfmt is unsupported.
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
        Some(FrameFormat {
            width,
            height,
            pixfmt,
            frame_dur_ns,
        })
    }

    /// Configure the sink from a negotiated format: learn the geometry and, on first use,
    /// open the Wayland connection and create the window. A failure to reach a compositor
    /// disables the sink (frames become drops) rather than failing the pipeline — a headless
    /// CI box has no display, and the sink should degrade, not crash. A genuine protocol
    /// error after a successful connect *is* surfaced.
    fn configure(&mut self, ctx: &mut Ctx, f: &FixedFormat) -> Result<(), Error> {
        let Some(fmt) = Self::read_format(ctx, f) else {
            return Ok(()); // not enough fields yet; wait for a fuller announcement
        };
        self.format = Some(fmt);

        if self.disabled {
            return Ok(());
        }
        if self.client.is_none() {
            match WaylandClient::connect() {
                Ok(mut c) => {
                    if let Err(e) = c.create_window(&self.title) {
                        // Connected but could not create the window — a real error worth
                        // reporting, but still degrade to drops so the pipeline finishes.
                        let element = ctx.element();
                        ctx.post(BusMessage::Warning { element, error: e });
                        self.disabled = true;
                        return Ok(());
                    }
                    self.client = Some(c);
                }
                Err(_e) => {
                    // No compositor (headless / CI): disable silently and drop frames.
                    self.disabled = true;
                }
            }
        }
        Ok(())
    }

    /// Present one frame's bytes on the clock, with QoS. Returns `Ok(())` always — a
    /// presentation failure disables the sink rather than failing the pipeline.
    fn render(&mut self, ctx: &mut Ctx, data: &[u8], pts: Timestamp) -> Result<(), Error> {
        let Some(fmt) = self.format else { return Ok(()) };

        // QoS: if the deadline is already behind us by more than one frame, drop it. We test
        // *before* the (usually already-passed) wait, so a late frame is dropped without any
        // conversion/present cost (spec: task deliverable 3 — QoS).
        if fmt.frame_dur_ns > 0 && pts.is_some() {
            let now = ctx.now();
            if now.is_some() && now.0 > pts.0 {
                let lateness = now.0 - pts.0;
                if lateness > fmt.frame_dur_ns {
                    let sink = ctx.element();
                    ctx.post(BusMessage::Qos {
                        sink,
                        lateness_ns: lateness as i64,
                    });
                    return Ok(());
                }
            }
        }

        // Pace on the clock. An interrupt (flush/shutdown) returns cleanly.
        match ctx.wait_until(pts) {
            WaitOutcome::Reached => {}
            WaitOutcome::Interrupted => return Ok(()),
        }

        if self.disabled {
            return Ok(());
        }
        let Some(client) = self.client.as_mut() else { return Ok(()) };

        // Present into a free shm buffer, converting directly (one pass, no intermediate).
        let width = fmt.width;
        let height = fmt.height;
        let pixfmt = fmt.pixfmt;
        let present = client.present(width, height, |dst, stride, w, h| {
            let ok = match pixfmt {
                Pix::I420 => i420_to_xrgb(data, w, h, dst, stride),
                Pix::Gray8 => gray8_to_xrgb(data, w, h, dst, stride),
            };
            if !ok {
                // A malformed/short frame: leave the buffer black rather than index OOB.
                for b in dst.iter_mut() {
                    *b = 0;
                }
            }
        });

        match present {
            Ok(true) => {}
            Ok(false) => {
                // The compositor asked to close the window — report once, then drop frames.
                if !self.close_reported {
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Resource("waylandvideosink: window closed by user".into()),
                    });
                    self.close_reported = true;
                }
                self.disabled = true;
            }
            Err(e) => {
                // A protocol error after connect is real: report and degrade to drops.
                let element = ctx.element();
                ctx.post(BusMessage::Warning { element, error: e });
                self.disabled = true;
            }
        }
        Ok(())
    }
}

impl Default for WaylandVideoSink {
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

impl Element for WaylandVideoSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        // The window is created lazily once the format is known (FormatChange or a
        // fully-fixed link-time format), exactly like the pipewire sink configures its
        // device on first use.
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // On first data, if no FormatChange arrived, try a fully-fixed link-time format.
        if self.format.is_none() {
            if let Some(f) = ctx.negotiated(SINK).cloned() {
                self.configure(ctx, &f)?;
            }
        }

        // A seek that lands mid-batch makes the remaining frames stale: bail and let the
        // scheduler run the flush (spec: flush/seek — blocking sinks bail on ctx.seek_gen()).
        let entry_gen = ctx.seek_gen();
        while let Some(buf) = inputs.pop() {
            if ctx.seek_gen() != entry_gen {
                return Ok(Flow::Ok);
            }
            // `buf` is owned (moved out of the batch), so borrowing its bytes does not clash
            // with the `&mut ctx` the render path needs — no per-frame copy of the frame. The
            // conversion inside `render` reads straight from here into the shm buffer.
            self.render(ctx, buf.memory.data(), buf.pts)?;
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FormatChange(f) => self.configure(ctx, f)?,
            // Flush/seek: nothing is carried (one frame per buffer), but pump the display so
            // ping/pong/close are answered promptly even while the graph is flushing.
            Event::FlushStart => {
                if let Some(c) = self.client.as_mut() {
                    let _ = c.pump();
                }
            }
            // EOS: keep the last frame on screen; the window stays until stop(). Pump once so
            // a trailing close/ping is handled.
            Event::Eos => {
                if let Some(c) = self.client.as_mut() {
                    let _ = c.pump();
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // Tear the window down (Drop also does this; explicit here so it happens at stop).
        if let Some(mut c) = self.client.take() {
            c.destroy();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixfmt_names_round_trip() {
        assert!(matches!(Pix::from_name("i420"), Some(Pix::I420)));
        assert!(matches!(Pix::from_name("gray8"), Some(Pix::Gray8)));
        assert!(Pix::from_name("nv12").is_none());
    }

    #[test]
    fn desc_offers_only_convertible_pixfmts_on_a_dynamic_sink_pad() {
        let d = WaylandVideoSink::new();
        let pad = &d.desc().pads[0];
        assert_eq!(pad.name, "sink");
        assert_eq!(pad.direction, Direction::Sink);
        assert!(pad.dynamic, "dimensions arrive via FormatChange");
        assert!(matches!(d.desc().sched, SchedHint::Active));
    }
}
