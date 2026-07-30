//! `vp9dec` — the VP9 decode element (spec: Milestone applications §5; Formats —
//! dynamic caps). VP9 coded chunks arrive on the sink pad — **one container packet
//! per buffer**, the demuxer contract (mkv `V_VP9`, IVF) — and raw planar video
//! leaves on the src pad, one frame per buffer, planes packed Y then U then V (the
//! `video/raw` layout the profluens-video helpers describe).
//!
//! A container packet is not always a single coded frame: VP9's Annex B
//! **superframe** framing lets an encoder consolidate several coded frames (a hidden
//! alt-ref plus its visible companion, say) into one chunk, with a trailing index.
//! The backing decoder ([`oxideav_vp9`] 0.0.12) does not split superframes, so this
//! element does — a trivial trailing-byte parse ([`superframe`]) — and decodes each
//! enclosed frame in turn, so container packets Just Work.
//!
//! Frame dimensions live in the frame header, not in a static descriptor, so the src
//! pad advertises a broad, `dynamic` `video/raw` template and **announces** the
//! concrete format at runtime — via [`Ctx::announce_format`] — when the first frame
//! decodes (the flacdec / vp8dec pattern, spec: Formats — dynamic caps).
//!
//! Error scope is per-buffer (spec: Supervision — a bit-flipped frame is weather, not
//! a verdict): an enclosed frame that fails to decode posts a bus `Warning` and is
//! dropped; decoding resumes at the next keyframe.
//!
//! # Decoder subset (loudly enforced by [`oxideav_vp9`] 0.0.12)
//!
//! The adopted 0.0.12 decoder is **intra-only**: key frames and intra-only frames
//! decode end-to-end (byte-exact against a corpus spanning 4:2:0 / 4:4:4, 8/10/12-bit,
//! lossless, multi-tile and segmentation AQ). **Inter (P-)frames**, and the
//! `show_existing_frame` re-display, return [`oxideav_vp9::Error::Unsupported`] because
//! they need reference-buffer state that 0.0.12 does not implement. Such frames are
//! treated exactly like a corrupt frame here — a bus `Warning`, then dropped — so a
//! keyframe-only stream (every frame a key frame; e.g. `ffmpeg … -g 1`) decodes fully,
//! while a normal GOP-coded stream decodes only its key frames until upstream gains
//! inter support. See `lib.rs` for the adoption rationale and the tracked debt.

use oxideav_vp9::{decode_intra_frame, Vp9DecodedFrame};

use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_video::color;
use profluens_core::time::Timestamp;

use crate::superframe;

// `video/raw` family/field/value names. Kept as literals (not a dep on
// profluens-video) so pf-vp9 stays core-only, exactly as pf-vp8 does for
// `video/raw`; the pipeline interns by string, so the ids line up with any video
// peer using the same names.
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";

// The pixel-format ids VP9 can produce. `i420` (8-bit 4:2:0) is the common path and
// matches `video/src/format.rs`'s `PixelFormat::I420`. The others are new literals
// this element introduces because the backing decoder genuinely emits them: `i444`
// (8-bit 4:4:4), plus 10/12-bit high-bit-depth variants whose samples are packed as
// little-endian `u16` pairs (`to_planar_bytes`).
const PIXFMT_I420: &str = "i420";
const PIXFMT_I422: &str = "i422";
const PIXFMT_I440: &str = "i440";
const PIXFMT_I444: &str = "i444";
const PIXFMT_I420_10: &str = "i420_10";
const PIXFMT_I422_10: &str = "i422_10";
const PIXFMT_I440_10: &str = "i440_10";
const PIXFMT_I444_10: &str = "i444_10";
const PIXFMT_I420_12: &str = "i420_12";
const PIXFMT_I422_12: &str = "i422_12";
const PIXFMT_I440_12: &str = "i440_12";
const PIXFMT_I444_12: &str = "i444_12";

const SRC_PAD: PadId = PadId(1);
/// The sink pad's local index (pads are `[sink, src]`).
const SINK_PAD: PadId = PadId(0);

static PIXFMT_VALUES: [ValueDesc; 12] = [
    ValueDesc::Id(PIXFMT_I420),
    ValueDesc::Id(PIXFMT_I422),
    ValueDesc::Id(PIXFMT_I440),
    ValueDesc::Id(PIXFMT_I444),
    ValueDesc::Id(PIXFMT_I420_10),
    ValueDesc::Id(PIXFMT_I422_10),
    ValueDesc::Id(PIXFMT_I440_10),
    ValueDesc::Id(PIXFMT_I444_10),
    ValueDesc::Id(PIXFMT_I420_12),
    ValueDesc::Id(PIXFMT_I422_12),
    ValueDesc::Id(PIXFMT_I440_12),
    ValueDesc::Id(PIXFMT_I444_12),
];

// A broad `video/raw` template: any dimensions, any VP9-producible pixel format. The
// concrete width/height/pixfmt are announced at runtime from the first decoded frame
// header — the src pad is `dynamic` for exactly this reason.
static SRC_FIELDS: [FieldDesc; 7] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
    // Colorimetry passthrough (spec: Formats; ITU-T H.273 via profluens-video's
    // color vocab): announced only when the demuxer declared it upstream.
    FieldDesc { field: color::FIELD_MATRIX, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_RANGE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_TRANSFER, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_PRIMARIES, allowed: ConstraintDesc::Any, preferred: None },
];
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }];
// One VP9 coded chunk per buffer, as a demuxer (mkv V_VP9, IVF) hands them out.
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
    [OfferDesc { family: "vp9", fields: &SINK_COLOR_FIELDS }];

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
        dynamic: true, // dimensions/pixfmt are data-dependent — announced at runtime
        validate: None,
    },
];

// COLD: `make_default` boxes one element instance at pipeline construction, never per frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "vp9dec",
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
    // Default-constructs trivially: dimensions come from the frame header, not props.
    make_default: Some(|| Box::new(Vp9Dec::new())),
};

/// The categorical `video/raw` `pixfmt` id for a decoded frame's geometry + depth.
/// Mirrors `to_planar_bytes` packing: 8-bit ids are byte planes, `_10` / `_12` ids are
/// little-endian `u16` pairs.
fn pixfmt_name(frame: &Vp9DecodedFrame) -> &'static str {
    // (subsampling_x, subsampling_y) → chroma geometry:
    //   (true,  true ) = 4:2:0    (false, true ) = 4:4:0
    //   (true,  false) = 4:2:2    (false, false) = 4:4:4
    let (ssx, ssy) = (frame.subsampling_x, frame.subsampling_y);
    match (frame.bit_depth, ssx, ssy) {
        (8, true, true) => PIXFMT_I420,
        (8, true, false) => PIXFMT_I422,
        (8, false, true) => PIXFMT_I440,
        (8, false, false) => PIXFMT_I444,
        (10, true, true) => PIXFMT_I420_10,
        (10, true, false) => PIXFMT_I422_10,
        (10, false, true) => PIXFMT_I440_10,
        (10, false, false) => PIXFMT_I444_10,
        (_, true, true) => PIXFMT_I420_12,
        (_, true, false) => PIXFMT_I422_12,
        (_, false, true) => PIXFMT_I440_12,
        (_, false, false) => PIXFMT_I444_12,
    }
}

/// A decoded frame waiting for a pool slot — the backpressure carry, bounded to one
/// frame: `process` never decodes another until this drains, so a slow sink cannot
/// make the decoder run ahead and balloon the heap.
struct PendingFrame {
    planar: Vec<u8>,
    width: i64,
    height: i64,
    pixfmt: &'static str,
    pts: Timestamp,
    duration: Timestamp,
}

/// Decodes VP9 coded chunks (splitting Annex B superframes) to packed planar video.
pub struct Vp9Dec {
    announced: bool,
    pending: Option<PendingFrame>,
    /// Reused superframe split buffer — `split_superframe_into` clears and refills it
    /// each chunk, so the per-frame path allocates no range list once it has grown to
    /// the largest superframe's frame count (≤ 8, the §B.2.2 3-bit field).
    ranges: Vec<(usize, usize)>,
}

impl Vp9Dec {
    // COLD: one-time element setup — the reused range buffer starts empty and is
    // refilled in place per chunk, never reallocated on the per-frame path.
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        Self {
            announced: false,
            pending: None,
            ranges: Vec::new(),
        }
    }

    /// Copy the pending decoded frame into a pool buffer and emit it. `false` means the
    /// pool is exhausted — leave it pending (backpressure).
    ///
    /// The `to_planar_bytes` packing is the adoption's known cost (see lib.rs): the
    /// upstream decoder owns its output planes, so until a `decode_frame_into` lands
    /// this element pays one packed-planar copy per frame — bounded, and the only copy
    /// on the path.
    fn emit_pending(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        let Some(p) = self.pending.take() else { return Ok(true) };
        let need = p.planar.len();
        let Some(mut buf) = ctx.try_alloc(SRC_PAD) else {
            self.pending = Some(p);
            return Ok(false);
        };
        if buf.memory.capacity() < need {
            // Pool slots are sized by the pipeline; a frame that cannot fit any slot
            // would silently truncate — fail loudly instead (pool sizing / pool
            // negotiation is the pipeline's pending spec work).
            return Err(Error::Element {
                element: ctx.element(),
                message: format!(
                    "vp9dec: {}x{} {} frame needs {need} bytes but pool slots hold {} — \
                     raise the pipeline pool slot size",
                    p.width,
                    p.height,
                    p.pixfmt,
                    buf.memory.capacity()
                ),
            });
        }
        let dst = buf.memory.as_mut_full();
        dst[..need].copy_from_slice(&p.planar);
        buf.memory.set_len(need);
        buf.pts = p.pts;
        buf.duration = p.duration;
        ctx.out(SRC_PAD).push(buf);
        Ok(true)
    }

    /// Decode one enclosed coded frame. On success announces the format (once) and
    /// stashes the packed-planar picture as the pending carry; on the decoder's
    /// `Unsupported` (an inter / `show_existing` frame this 0.0.12 subset cannot
    /// decode) or any error, posts a bus `Warning` and drops — per-buffer error scope.
    fn decode_one(&mut self, ctx: &mut Ctx, frame: &[u8], pts: Timestamp, duration: Timestamp) {
        let decoded = match decode_intra_frame(frame) {
            Ok(f) => f,
            Err(e) => {
                let element = ctx.element();
                ctx.post(BusMessage::Warning {
                    element,
                    error: Error::Element {
                        element,
                        message: format!("vp9dec: frame dropped: {e}"),
                    },
                });
                return;
            }
        };
        let pixfmt = pixfmt_name(&decoded);
        if !self.announced {
            let mut fields = vec![
                    (F_WIDTH, ValueDesc::Int(decoded.width as i64)),
                    (F_HEIGHT, ValueDesc::Int(decoded.height as i64)),
                    (F_PIXFMT, ValueDesc::Id(pixfmt)),
                ];
                color_passthrough(ctx, SINK_PAD, &mut fields);
                ctx.announce_format(SRC_PAD, FAMILY, &fields);
            self.announced = true;
        }
        self.pending = Some(PendingFrame {
            planar: decoded.to_planar_bytes(),
            width: decoded.width as i64,
            height: decoded.height as i64,
            pixfmt,
            pts,
            duration,
        });
    }
}

impl Default for Vp9Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Vp9Dec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        loop {
            // Drain the carry first; if the pool is out of slots, stop pulling input —
            // un-popped buffers stay in the input batch for the next pass.
            if !self.emit_pending(ctx)? {
                return Ok(Flow::Ok);
            }
            let Some(inbuf) = inputs.pop() else { break };
            // A container packet may be an Annex B superframe carrying several coded
            // frames; split it (0.0.12 has no split) and decode each in order. The
            // packet's pts/duration ride the whole chunk; the visible frame (typically
            // the last / only one) carries it downstream.
            let chunk = inbuf.memory.data();
            // Split into the reused `ranges` buffer (no per-chunk allocation), then
            // iterate by index so the `&mut self` decode calls below borrow freely —
            // `(usize, usize)` ranges are `Copy`, and the buffer is not mutated in the
            // loop body.
            superframe::split_superframe_into(chunk, &mut self.ranges);
            let n = self.ranges.len();
            for i in 0..n {
                let (start, end) = self.ranges[i];
                // Emit the carry from a previous enclosed frame before decoding the
                // next, so a multi-frame superframe never carries more than one
                // pending picture (bounded heap).
                if !self.emit_pending(ctx)? {
                    // Pool exhausted mid-superframe: this is the rare hidden-frame
                    // case. The remaining enclosed frames are lost, but that only
                    // happens under sustained backpressure with a superframe present;
                    // a single-frame packet (the overwhelmingly common case) never
                    // reaches here. Warn once and move on to the next packet.
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: "vp9dec: pool exhausted mid-superframe; \
                                      dropped remaining enclosed frames"
                                .into(),
                        },
                    });
                    break;
                }
                let is_last = i + 1 == n;
                // Only the last enclosed frame inherits the packet's timing; hidden
                // alt-refs (earlier enclosed frames) are presentation-less anyway.
                let (pts, dur) = if is_last {
                    (inbuf.pts, inbuf.duration)
                } else {
                    (Timestamp::ZERO, Timestamp::ZERO)
                };
                self.decode_one(ctx, &chunk[start..end], pts, dur);
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Flush/seek: drop the carry. The intra-only decoder holds no cross-frame
            // reference state, so there is nothing else to reset — the stream resumes
            // at a keyframe (spec: flush/seek).
            Event::FlushStart => {
                self.pending = None;
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
/// `profluens_video::color`). Absent/unknown fields stay unannounced — the
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
        let Some(profluens_core::format::Value::Id(vid)) = fmt.get(fid) else { continue };
        let Some(stat) = ctx.value_name(vid).and_then(statify) else { continue };
        out.push((field, ValueDesc::Id(stat)));
    }
}
