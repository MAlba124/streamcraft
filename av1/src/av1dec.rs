//! `av1dec` — the AV1 decode element (spec: Milestone applications §5; Formats —
//! dynamic caps). AV1 temporal units arrive on the sink pad — **one temporal unit
//! per buffer**, the container contract (Matroska `V_AV1`, an ISOBMFF `av01` sample,
//! one IVF record) — and raw video leaves on the src pad, one frame per buffer,
//! planes packed Y then U then V (the `video/raw` layout the streamcraft-video
//! helpers describe).
//!
//! Frame dimensions and the pixel format live in the sequence / frame headers, not in
//! a static descriptor, so the src pad advertises a broad, `dynamic` `video/raw`
//! template and **announces** the concrete format at runtime — via
//! [`Ctx::announce_format`] — when the first frame decodes (the flacdec / vp8dec
//! pattern, spec: Formats — dynamic caps).
//!
//! A temporal unit yields **zero, one, or several** shown frames: an invisible altref
//! updates the decoder's reference slots and emits nothing, and a later
//! `show_existing_frame` unit emits a stored frame. The upstream
//! [`SpecDecodeSession`] applies the §7.4 output discipline internally and returns
//! exactly the shown frames in output order — the AV1 analogue of how vp8dec handles
//! `show_frame == 0`.
//!
//! Error scope is per-buffer (spec: Supervision — a bit-flipped unit is weather, not a
//! verdict): a temporal unit that fails to decode, or that produces an output shape
//! our raw-video vocabulary cannot name (4:2:2 / 4:4:4 / 10-/12-bit — see the crate
//! docs), posts a bus `Warning` and is dropped; decoding resumes at the next keyframe.

use oxideav_av1::decoder::{SpecDecodeSession, SpecFrame};

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
// streamcraft-video) so sc-av1 stays core-only, exactly as sc-vp8 does for
// `video/raw`; the pipeline interns by string, so the ids line up with any video
// peer using the same names.
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
// The two AV1 8-bit output shapes streamcraft-video's vocabulary can name (see the
// crate docs' capability table): planar 4:2:0 (Y,U,V) and monochrome (Y only).
const PIXFMT_I420: &str = "i420";
const PIXFMT_GRAY8: &str = "gray8";

const SRC_PAD: PadId = PadId(1);
/// The sink pad's local index (pads are `[sink, src]`).
const SINK_PAD: PadId = PadId(0);

static PIXFMT_VALUES: [ValueDesc; 2] = [ValueDesc::Id(PIXFMT_I420), ValueDesc::Id(PIXFMT_GRAY8)];

// A broad `video/raw` template: any dimensions, i420 or gray8 (the 8-bit output
// shapes we admit). The concrete width/height/pixfmt are announced at runtime from
// the headers — the src pad is `dynamic` for exactly this reason.
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
// One AV1 temporal unit per buffer, as a demuxer (mkv V_AV1, IVF) hands them out.
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
    [OfferDesc { family: "av1", fields: &SINK_COLOR_FIELDS }];

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
        dynamic: true, // dimensions + pixfmt are data-dependent — announced at runtime
        validate: None,
    },
];

// COLD: `make_default` boxes one element instance at pipeline construction, never per frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "av1dec",
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
    // Default-constructs trivially: dimensions come from the sequence header, not props.
    make_default: Some(|| Box::new(Av1Dec::new())),
};

/// The `video/raw` pixel format a decoded [`SpecFrame`] maps onto, and the total
/// packed-plane byte count. `None` for a shape our vocabulary cannot name (4:2:2 /
/// 4:4:4 / 10-/12-bit) — the caller rejects it loudly per the per-buffer error scope.
fn pixfmt_for(frame: &SpecFrame) -> Option<(&'static str, usize)> {
    // The upstream decoder surfaces 8-bit output as one byte per sample; 10/12-bit as
    // two little-endian bytes. Our vocabulary names only 8-bit formats.
    if frame.bit_depth != 8 {
        return None;
    }
    let packed: usize = frame.planes.iter().map(Vec::len).sum();
    match frame.planes.len() {
        1 => Some((PIXFMT_GRAY8, packed)), // monochrome: Y only
        3 => Some((PIXFMT_I420, packed)),  // 4:2:0: Y, U, V — rejected below if not
        _ => None,
    }
}

/// A decoded frame waiting for a pool slot — the backpressure carry. AV1 differs from
/// VP8 in that one temporal unit can produce several shown frames, so the carry is a
/// bounded queue rather than a single slot: `process` stops pulling input while
/// anything is pending, so a slow sink cannot make the decoder run ahead and balloon
/// the heap (the queue never exceeds the shown-frame count of one temporal unit).
struct PendingFrame {
    frame: SpecFrame,
    pixfmt: &'static str,
    need: usize,
    pts: Timestamp,
    duration: Timestamp,
}

/// Decodes AV1 temporal units to packed 8-bit `i420` / `gray8` video.
pub struct Av1Dec {
    session: SpecDecodeSession,
    announced: bool,
    pending: std::collections::VecDeque<PendingFrame>,
}

impl Av1Dec {
    pub fn new() -> Self {
        Self {
            session: SpecDecodeSession::new(),
            announced: false,
            pending: std::collections::VecDeque::new(),
        }
    }

    /// Drain the carry into pool buffers. `false` means the pool is exhausted — leave
    /// the rest pending (backpressure).
    ///
    /// The plane copy is the adoption's known cost (see lib.rs): the upstream decoder
    /// owns its output `Vec`s, so until a `decode_into` lands this element pays one
    /// packed copy per frame — bounded, and the only copy on the path.
    fn emit_pending(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        while let Some(p) = self.pending.front() {
            let Some(mut buf) = ctx.try_alloc(SRC_PAD) else {
                return Ok(false);
            };
            if buf.memory.capacity() < p.need {
                // Pool slots are sized by the pipeline; a frame that cannot fit any
                // slot would silently truncate — fail loudly instead (pool sizing /
                // pool negotiation is the pipeline's pending spec work).
                return Err(Error::Element {
                    element: ctx.element(),
                    message: format!(
                        "av1dec: {}x{} {} frame needs {} bytes but pool slots hold {} — \
                         raise the pipeline pool slot size",
                        p.frame.width,
                        p.frame.height,
                        p.pixfmt,
                        p.need,
                        buf.memory.capacity()
                    ),
                });
            }
            // Safe to pop now: a slot is in hand and it fits.
            let p = self.pending.pop_front().unwrap();
            let dst = buf.memory.as_mut_full();
            let mut off = 0;
            for plane in &p.frame.planes {
                dst[off..off + plane.len()].copy_from_slice(plane);
                off += plane.len();
            }
            buf.memory.set_len(p.need);
            buf.pts = p.pts;
            buf.duration = p.duration;
            ctx.out(SRC_PAD).push(buf);
        }
        Ok(true)
    }
}

impl Default for Av1Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Av1Dec {
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
            let frames = match self.session.decode_temporal_unit(inbuf.memory.data()) {
                Ok(f) => f,
                Err(e) => {
                    // Per-buffer error scope: warn, drop, resync at the next keyframe
                    // (spec: Supervision). A corrupt unit that desynced the arithmetic
                    // decoder recovers when the next KEY frame rebuilds every slot.
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: format!("av1dec: temporal unit dropped: {e:?}"),
                        },
                    });
                    continue;
                }
            };
            // pts rides the input buffer. One temporal unit usually yields one shown
            // frame (the container-per-packet case); when it yields several, they
            // share the unit's timestamp — a demuxer that wants distinct pts per
            // output splits the unit upstream.
            for frame in frames {
                let Some((pixfmt, need)) = pixfmt_for(&frame) else {
                    // A shape our vocabulary cannot name — drop it loudly, same scope.
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: format!(
                                "av1dec: {}x{} output dropped — {}-bit / {}-plane is not in the \
                                 raw-video vocabulary (only 8-bit i420 / gray8 are wired)",
                                frame.width,
                                frame.height,
                                frame.bit_depth,
                                frame.planes.len()
                            ),
                        },
                    });
                    continue;
                };
                if !self.announced {
                    let mut fields = vec![
                            (F_WIDTH, ValueDesc::Int(frame.width as i64)),
                            (F_HEIGHT, ValueDesc::Int(frame.height as i64)),
                            (F_PIXFMT, ValueDesc::Id(pixfmt)),
                        ];
                        color_passthrough(ctx, SINK_PAD, &mut fields);
                        ctx.announce_format(SRC_PAD, FAMILY, &fields);
                    self.announced = true;
                }
                self.pending.push_back(PendingFrame {
                    frame,
                    pixfmt,
                    need,
                    pts: inbuf.pts,
                    duration: inbuf.duration,
                });
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Flush/seek: drop the carry and all reference state — the stream resumes
            // at a keyframe, which rebuilds every slot (spec: flush/seek). A fresh
            // session also re-establishes the sequence header from the next unit.
            Event::FlushStart => {
                self.pending.clear();
                self.session = SpecDecodeSession::new();
            }
            // Drain any carried frames at EOS so the last pictures are not lost.
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
