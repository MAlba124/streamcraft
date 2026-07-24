//! `h265dec` — the H.265 / HEVC decode element (spec: Milestone applications §5;
//! Formats — dynamic caps). One **Annex B access unit per buffer** arrives on the
//! sink pad — the coded-picture-per-buffer demuxer contract, start-code-delimited
//! NAL units (VPS/SPS/PPS out-of-band plus one coded picture's slices) — and raw
//! I420 video leaves on the src pad, one frame per buffer, planes packed Y then Cb
//! then Cr (the `video/raw` layout the streamcraft-video helpers describe).
//!
//! Frame dimensions live in the SPS, not in a static descriptor, so the src pad
//! advertises a broad, `dynamic` `video/raw` template and **announces** the concrete
//! format at runtime — via [`Ctx::announce_format`] — when the first frame decodes
//! (the flacdec/vp8dec pattern, spec: Formats — dynamic caps).
//!
//! **Output ordering and pts (read this).** HEVC decode order is *not* presentation
//! order: a coded picture can reference a later-presented one (B-frames / open GOP),
//! so the decoder buffers a §7.4.3.2.1 `sps_max_num_reorder_pics`-deep window and
//! emits in `PicOrderCntVal` (output) order. This element does the same via the
//! upstream [`SequenceDecoder`], and **re-attaches the input pts values in ascending
//! order** to the output-order frames (a min-heap, exactly the upstream registry
//! decoder's scheme): the smallest pending pts goes to the first frame emitted. That
//! is correct whenever the container delivers pts monotonically increasing with
//! presentation — which is the normal case — but it is a *reordering* of pts across
//! the reorder window, not a passthrough. Non-output pictures (`pic_output_flag ==
//! 0`, RASL-skipped) are decoded (they may be referenced) but never emitted.
//!
//! Error scope is per-buffer (spec: Supervision — a bit-flipped access unit is
//! weather, not a verdict): an access unit that fails to decode posts a bus `Warning`
//! and is dropped; decode resumes at the next parameter-set + IRAP. A push that
//! desyncs the decoder object is recovered by resetting it (a fresh
//! [`SequenceDecoder`]) — the stream continues at the next keyframe, which re-sends
//! its parameter sets.
//!
//! Output is I420 (8-bit 4:2:0) only. The upstream decoder also reconstructs Main10 /
//! 4:2:2 / 4:4:4 / monochrome pictures correctly, but this element has no `video/raw`
//! pixfmt vocabulary for those yet, so a non-8-bit or non-4:2:0 picture is dropped
//! with a bus `Warning` rather than mis-packed (the honest failure). Adding
//! `i420_10le` / `i422` / `i444` / `gray8` pixfmt literals + `to_planar_le16` packing
//! is the follow-up when a downstream consumer needs them.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use oxideav_h265::{DecodedFrame, SequenceDecoder};

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
use streamcraft_core::time::Timestamp;

// `video/raw` family/field/value names. Kept as literals (not a dep on
// streamcraft-video) so sc-h265 stays core-only, exactly as sc-vp8 does for
// `video/raw`; the pipeline interns by string, so the ids line up with any video
// peer using the same names.
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
const PIXFMT_I420: &str = "i420";

/// The reorder-window depth used before any SPS is seen (an SPS then supplies
/// `sps_max_num_reorder_pics`). Matches the upstream registry decoder's default.
const DEFAULT_REORDER: usize = 8;

const SRC_PAD: PadId = PadId(1);

static PIXFMT_VALUES: [ValueDesc; 1] = [ValueDesc::Id(PIXFMT_I420)];

// A broad `video/raw` template: any dimensions, I420 only (the only pixfmt this
// element packs). The concrete width/height are announced at runtime from the SPS —
// the src pad is `dynamic` for exactly this reason.
static SRC_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
];
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }];
// One HEVC Annex B access unit per buffer, as a demuxer (mkv V_MPEGH/ISO/HEVC in
// Annex B framing) hands them out.
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("h265/annexb")];

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

static DESC: ElementDesc = ElementDesc {
    name: "h265dec",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Default-constructs trivially: dimensions come from the SPS, not props.
    make_default: Some(|| Box::new(H265Dec::new())),
};

/// A decoded frame in output order, waiting for a pool slot — the backpressure
/// carry. `process` refills this from the reorder window but never lets it grow
/// unbounded: it stops pulling input while a slot is unavailable, so a slow sink
/// cannot make the decoder run ahead and balloon the heap.
struct ReadyFrame {
    /// The decoded, output-order picture (packed into I420 lazily in `emit_ready`,
    /// only once a pool slot is available — so a slow sink caps the packed copies at
    /// one, not the whole reorder window).
    frame: DecodedFrame,
    /// The input pts re-attached in output order (§ the module doc's caveat).
    pts: Timestamp,
}

/// Decodes HEVC Annex B access units to packed I420.
pub struct H265Dec {
    seq: SequenceDecoder,
    announced: bool,
    /// Decoded pictures not yet released from the reorder window, sorted on demand
    /// by `(cvs_index, poc)` — output (PicOrderCntVal) order.
    reorder: Vec<DecodedFrame>,
    /// Min-heap of pending input pts values, re-attached to output-order frames in
    /// ascending order (§ the module doc's reordering caveat).
    pts_queue: BinaryHeap<Reverse<i64>>,
    /// Output-order frames released from the reorder window, awaiting a pool slot,
    /// front first — the backpressure carry.
    ready: std::collections::VecDeque<ReadyFrame>,
}

impl H265Dec {
    pub fn new() -> Self {
        Self {
            seq: SequenceDecoder::new(),
            announced: false,
            reorder: Vec::new(),
            pts_queue: BinaryHeap::new(),
            ready: std::collections::VecDeque::new(),
        }
    }

    /// Move newly decoded pictures into the reorder window and release every frame
    /// that is guaranteed next in output order into `ready`. `flush` releases the
    /// whole window (EOS / seek boundary). Mirrors the upstream registry decoder's
    /// `drain`.
    fn drain(&mut self, flush: bool) {
        self.reorder.extend(self.seq.take_decoded());
        self.reorder.sort_by_key(|f| (f.cvs_index, f.poc));
        let depth = if flush {
            0
        } else {
            self.seq
                .max_num_reorder_pics()
                .map_or(DEFAULT_REORDER, |n| n as usize)
        };
        while self.reorder.len() > depth {
            let frame = self.reorder.remove(0);
            // A non-output picture consumed no input pts (it is never presented);
            // don't pull one off the heap for it.
            if !frame.output {
                continue;
            }
            let pts = self
                .pts_queue
                .pop()
                .map(|r| Timestamp::from_nanos(r.0 as u64))
                .unwrap_or(Timestamp::NONE);
            self.ready.push_back(ReadyFrame { frame, pts });
        }
    }

    /// Emit as many ready frames as the pool allows. Returns `false` when the pool is
    /// exhausted — the un-emitted frames stay in `ready` (backpressure), and the
    /// caller must stop pulling input.
    fn emit_ready(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        while let Some(front) = self.ready.front() {
            let width = front.frame.picture.width_luma() as i64;
            let height = front.frame.picture.height_luma() as i64;
            // Pack to I420 (only once we're committed to emitting this frame). A
            // non-8-bit / non-4:2:0 picture has no `video/raw` pixfmt here yet — drop
            // it with a bus warning rather than mis-pack (spec: Supervision; module
            // doc's format limitation).
            let Some(planes) = pack_i420(&front.frame) else {
                let dropped = self.ready.pop_front().expect("front just observed");
                let element = ctx.element();
                let cat = dropped.frame.picture.chroma_array_type();
                let bd = dropped.frame.picture.bit_depth_luma();
                ctx.post(BusMessage::Warning {
                    element,
                    error: Error::Element {
                        element,
                        message: format!(
                            "h265dec: dropped a {width}x{height} frame — only i420 (8-bit \
                             4:2:0) output is wired (this frame: chroma_array_type={cat}, \
                             bit_depth={bd})"
                        ),
                    },
                });
                continue;
            };
            if !self.announced {
                ctx.announce_format(
                    SRC_PAD,
                    FAMILY,
                    &[
                        (F_WIDTH, ValueDesc::Int(width)),
                        (F_HEIGHT, ValueDesc::Int(height)),
                        (F_PIXFMT, ValueDesc::Id(PIXFMT_I420)),
                    ],
                );
                self.announced = true;
            }
            let need = planes.len();
            let Some(mut buf) = ctx.try_alloc(SRC_PAD) else {
                return Ok(false);
            };
            if buf.memory.capacity() < need {
                // Pool slots are sized by the pipeline; a frame that cannot fit any
                // slot would silently truncate — fail loudly instead (pool sizing is
                // the pipeline's pending spec work).
                return Err(Error::Element {
                    element: ctx.element(),
                    message: format!(
                        "h265dec: {width}x{height} I420 frame needs {need} bytes but pool \
                         slots hold {} — raise the pipeline pool slot size",
                        buf.memory.capacity()
                    ),
                });
            }
            let f = self.ready.pop_front().expect("front just observed");
            let dst = buf.memory.as_mut_full();
            dst[..need].copy_from_slice(&planes);
            buf.memory.set_len(need);
            buf.pts = f.pts;
            ctx.out(SRC_PAD).push(buf);
        }
        Ok(true)
    }
}

impl Default for H265Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for H265Dec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        loop {
            // Drain the ready carry first; if the pool is out of slots, stop pulling
            // input — un-popped buffers stay in the input batch for the next pass.
            if !self.emit_ready(ctx)? {
                return Ok(Flow::Ok);
            }
            let Some(inbuf) = inputs.pop() else { break };
            // One buffer = one Annex B access unit. Record its pts for output-order
            // re-attachment, then decode.
            if let Some(ns) = inbuf.pts.nanos() {
                self.pts_queue.push(Reverse(ns as i64));
            }
            // Move anything the push already decoded into the reorder window BEFORE a
            // possible error, so a partially-consumed picture is not lost on reset.
            if let Err(e) = self.seq.push_annexb(inbuf.memory.data()) {
                // Per-buffer error scope: warn, drop, reset so the next parameter-set
                // + keyframe resyncs (spec: Supervision). A push leaves the decoder
                // "unspecified" on error, so a fresh decoder is the safe recovery — but
                // the already-decoded frames pending in `reorder` are valid and MUST
                // survive the reset (they are output, just not yet released).
                self.drain(false);
                let element = ctx.element();
                ctx.post(BusMessage::Warning {
                    element,
                    error: Error::Element {
                        element,
                        message: format!("h265dec: access unit dropped: {e}"),
                    },
                });
                self.seq = SequenceDecoder::new();
                // The dropped AU's pts is now orphaned; the ascending re-attach still
                // matches the surviving frames in order, so leave the heap as is.
                continue;
            }
            self.drain(false);
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Flush/seek: drop all decoder + reorder state — the stream resumes at a
            // keyframe (with its parameter sets), which rebuilds everything (spec:
            // flush/seek). Pending pts are dropped with the frames they belonged to.
            Event::FlushStart => {
                self.seq = SequenceDecoder::new();
                self.reorder.clear();
                self.ready.clear();
                self.pts_queue.clear();
                self.announced = false;
            }
            // Drain the whole reorder window at EOS so no picture is lost, then flush
            // the carry into the pool.
            Event::Eos => {
                // Decode any picture still being assembled, then release the window.
                // Even if the final assemble errors, the already-decoded frames in the
                // reorder window are valid and must still drain (below).
                if let Err(e) = self.seq.flush() {
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: format!("h265dec: EOS flush: {e}"),
                        },
                    });
                    self.seq = SequenceDecoder::new();
                }
                self.drain(true);
                let _ = self.emit_ready(ctx)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// Pack a decoded picture into I420 (8-bit 4:2:0) — `Y|Cb|Cr`. Returns `None` for a
/// non-8-bit or non-4:2:0 picture (Main10 / 4:2:2 / 4:4:4 / monochrome), which this
/// element does not have a `video/raw` pixfmt for yet (module doc).
fn pack_i420(f: &DecodedFrame) -> Option<Vec<u8>> {
    // 4:2:0 (ChromaArrayType == 1), 8-bit both components. `to_planar_u8` already
    // enforces the 8-bit half of this; we add the chroma-format check.
    if f.picture.chroma_array_type() != 1 {
        return None;
    }
    f.picture.to_planar_u8()
}
