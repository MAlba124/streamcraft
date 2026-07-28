//! `vp8dec` — the VP8 decode element (spec: Milestone applications §5; Formats —
//! dynamic caps). VP8 elementary-stream frames arrive on the sink pad — **one frame
//! per buffer**, the demuxer contract — and raw I420 video leaves on the src pad,
//! one frame per buffer, planes packed Y then U then V (the `video/raw` layout the
//! streamcraft-video helpers describe).
//!
//! Frame dimensions live in the keyframe header, not in a static descriptor, so the
//! src pad advertises a broad, `dynamic` `video/raw` template and **announces** the
//! concrete format at runtime — via [`Ctx::announce_format`] — when the first frame
//! decodes (the flacdec pattern, spec: Formats — dynamic caps).
//!
//! Error scope is per-buffer (spec: Supervision — a bit-flipped frame is weather,
//! not a verdict): a frame that fails to decode posts a bus `Warning` and is
//! dropped; decoding resumes at the next keyframe. Invisible frames (`show_frame ==
//! 0`, altref updates) update the reference slots inside the decoder but emit
//! nothing downstream.

use oxideav_vp8::Vp8DecoderState;

use streamcraft_core::batch::Inputs;
use streamcraft_core::bus::BusMessage;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_video::color;
use streamcraft_core::time::Timestamp;

// `video/raw` family/field/value names. Kept as literals (not a dep on
// streamcraft-video) so sc-vp8 stays core-only, exactly as sc-flac does for
// `audio/raw`; the pipeline interns by string, so the ids line up with any video
// peer using the same names.
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
const PIXFMT_I420: &str = "i420";

const SRC_PAD: PadId = PadId(1);
/// The sink pad's local index (pads are `[sink, src]`).
const SINK_PAD: PadId = PadId(0);

static PIXFMT_VALUES: [ValueDesc; 1] = [ValueDesc::Id(PIXFMT_I420)];

// A broad `video/raw` template: any dimensions, I420 only (all VP8 output is
// 4:2:0). The concrete width/height are announced at runtime from the keyframe
// header — the src pad is `dynamic` for exactly this reason.
static SRC_FIELDS: [FieldDesc; 7] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
    // Colorimetry passthrough (spec: Formats; ITU-T H.273 via streamcraft-video's
    // color vocab): announced only when the demuxer declared it upstream.
    FieldDesc { field: color::FIELD_MATRIX, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_RANGE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_TRANSFER, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_PRIMARIES, allowed: ConstraintDesc::Any, preferred: None },
];
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }];
// One VP8 elementary frame per buffer, as a demuxer (mkv V_VP8, IVF) hands them out.
// The sink offer declares the colorimetry field names so the demuxer's Colour
// announcement survives interning (announced field names resolve only against
// declared offers — the duration-field lesson); `Any` admits every closed value.
static SINK_COLOR_FIELDS: [FieldDesc; 4] = [
    FieldDesc { field: color::FIELD_MATRIX, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_RANGE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_TRANSFER, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_PRIMARIES, allowed: ConstraintDesc::Any, preferred: None },
];
static SINK_OFFERS: [OfferDesc; 1] =
    [OfferDesc { family: "vp8", fields: &SINK_COLOR_FIELDS }];

static PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &SINK_OFFERS,
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &SRC_OFFERS,
        dynamic: true, // dimensions are data-dependent — announced at runtime
        validate: None,
    },
];

// COLD: `make_default` boxes one element instance at pipeline construction, never per frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "vp8dec",
    pads: &PADS,
    props: &[],
    // Active: a video decode is far beyond the inline passive budget (spec:
    // Scheduling — the Passive contract is enforced, and a frame decode is
    // milliseconds, not nanoseconds). Its own thread group pipelines decode
    // against demux and display, and a branching demuxer upstream stays a legal
    // group tail (its consumers are all group heads).
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Default-constructs trivially: dimensions come from the keyframe header, not props.
    make_default: Some(|| Box::new(Vp8Dec::new())),
};

/// A decoded frame waiting for a pool slot — the backpressure carry, bounded to
/// one frame: `process` never decodes another until this drains, so a slow sink
/// cannot make the decoder run ahead and balloon the heap.
struct PendingFrame {
    frame: oxideav_vp8::Vp8DecodedFrame,
    pts: Timestamp,
    duration: Timestamp,
}

/// Decodes VP8 elementary frames to packed I420.
pub struct Vp8Dec {
    state: Vp8DecoderState,
    announced: bool,
    pending: Option<PendingFrame>,
}

impl Vp8Dec {
    pub fn new() -> Self {
        Self {
            state: Vp8DecoderState::new(),
            announced: false,
            pending: None,
        }
    }

    /// Copy the pending decoded frame into a pool buffer and emit it. `false` means
    /// the pool is exhausted — leave it pending (backpressure).
    ///
    /// The plane copy is the adoption's known cost (see lib.rs): the upstream
    /// decoder owns its output `Vec`s, so until a `decode_frame_into` lands this
    /// element pays one packed-I420 copy per frame — bounded, and the only copy on
    /// the path.
    fn emit_pending(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        let Some(p) = self.pending.take() else { return Ok(true) };
        let need = p.frame.y.len() + p.frame.u.len() + p.frame.v.len();
        let Some(mut buf) = ctx.try_alloc(SRC_PAD) else {
            self.pending = Some(p);
            return Ok(false);
        };
        if buf.memory.capacity() < need {
            // Pool slots are sized by the pipeline; a frame that cannot fit any
            // slot would silently truncate — fail loudly instead (pool sizing /
            // pool negotiation is the pipeline's pending spec work).
            return Err(Error::Element {
                element: ctx.element(),
                message: format!(
                    "vp8dec: {}x{} I420 frame needs {need} bytes but pool slots hold {} — \
                     raise the pipeline pool slot size",
                    p.frame.width,
                    p.frame.height,
                    buf.memory.capacity()
                ),
            });
        }
        let dst = buf.memory.as_mut_full();
        let (ylen, ulen) = (p.frame.y.len(), p.frame.u.len());
        dst[..ylen].copy_from_slice(&p.frame.y);
        dst[ylen..ylen + ulen].copy_from_slice(&p.frame.u);
        dst[ylen + ulen..need].copy_from_slice(&p.frame.v);
        buf.memory.set_len(need);
        buf.pts = p.pts;
        buf.duration = p.duration;
        ctx.out(SRC_PAD).push(buf);
        Ok(true)
    }
}

impl Default for Vp8Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Vp8Dec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        loop {
            // Drain the carry first; if the pool is out of slots, stop pulling
            // input — un-popped buffers stay in the input batch for the next pass.
            if !self.emit_pending(ctx)? {
                return Ok(Flow::Ok);
            }
            let Some(inbuf) = inputs.pop() else { break };
            let decoded = match self.state.decode_frame(inbuf.memory.data()) {
                Ok(f) => f,
                Err(e) => {
                    // Per-buffer error scope: warn, drop, resync at the next
                    // keyframe (spec: Supervision).
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: format!("vp8dec: frame dropped: {e:?}"),
                        },
                    });
                    continue;
                }
            };
            if !self.announced {
                let mut fields = vec![
                        (F_WIDTH, ValueDesc::Int(decoded.width as i64)),
                        (F_HEIGHT, ValueDesc::Int(decoded.height as i64)),
                        (F_PIXFMT, ValueDesc::Id(PIXFMT_I420)),
                    ];
                    color_passthrough(ctx, SINK_PAD, &mut fields);
                    ctx.announce_format(SRC_PAD, FAMILY, &fields);
                self.announced = true;
            }
            // An invisible frame (§9.1 show_frame == 0) only updates the decoder's
            // reference slots — nothing is presented downstream.
            if self.state.last_frame_shown() == Some(false) {
                continue;
            }
            self.pending = Some(PendingFrame {
                frame: decoded,
                pts: inbuf.pts,
                duration: inbuf.duration,
            });
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Flush/seek: drop the carry and all reference state — the stream
            // resumes at a keyframe, which rebuilds every slot (spec: flush/seek).
            Event::FlushStart => {
                self.pending = None;
                self.state = Vp8DecoderState::new();
            }
            // Drain any carried frame at EOS so the last picture is not lost.
            Event::Eos => {
                let _ = self.emit_pending(ctx)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// Colorimetry passthrough (spec: Formats — dynamic caps): forward the container's
/// announced colour fields from the negotiated sink format onto `video/raw`, names
/// re-anchored to `'static` through the closed color vocabulary (ITU-T H.273 via
/// `streamcraft_video::color`). Absent/unknown fields stay unannounced — the
/// renderer defaults by resolution.
fn color_passthrough(ctx: &Ctx, pad: PadId, out: &mut Vec<(&'static str, ValueDesc)>) {
    let Some(fmt) = ctx.negotiated(pad) else { return };
    type Statify = fn(&str) -> Option<&'static str>;
    let table: [(&'static str, Statify); 4] = [
        (color::FIELD_MATRIX, |s| color::Matrix::from_caps_name(s).map(|v| v.caps_name())),
        (color::FIELD_RANGE, |s| color::Range::from_caps_name(s).map(|v| v.caps_name())),
        (color::FIELD_TRANSFER, |s| color::TransferFn::from_caps_name(s).map(|v| v.caps_name())),
        (color::FIELD_PRIMARIES, |s| color::Primaries::from_caps_name(s).map(|v| v.caps_name())),
    ];
    for (field, statify) in table {
        let Some(fid) = ctx.field_id(field) else { continue };
        let Some(streamcraft_core::format::Value::Id(vid)) = fmt.get(fid) else { continue };
        let Some(stat) = ctx.value_name(vid).and_then(statify) else { continue };
        out.push((field, ValueDesc::Id(stat)));
    }
}
