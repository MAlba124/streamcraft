//! `h264dec` — the H.264 / AVC decode element (spec: Milestone applications §5;
//! Formats — dynamic caps). H.264 Annex B access units arrive on the sink pad —
//! **one access unit per buffer**, the demuxer contract — and raw I420 video
//! leaves on the src pad, one frame per buffer, planes packed Y then U then V (the
//! `video/raw` layout the streamcraft-video helpers describe).
//!
//! Unlike VP8 (RFC 6386 §9.1: one keyframe carries the dimensions and every frame
//! is self-delimiting), H.264 dimensions live in the SPS (ITU-T H.264 §7.3.2.1),
//! reordering is possible (B-frames — decode order ≠ display order), and one input
//! access unit may be split across several slice NALs. The backing decoder
//! ([`oxideav_h264::h264_decoder::H264CodecDecoder`]) owns all of that: it is a
//! stateful push decoder (`send_packet` → `receive_frame`*) that assembles slices
//! into pictures, runs the §C.4 DPB bumping process, and hands frames back in
//! **display (POC) order**. This element is a thin adapter around that trait.
//!
//! Frame dimensions are data-dependent (SPS), so the src pad advertises a broad,
//! `dynamic` `video/raw` template and **announces** the concrete format at runtime
//! — via [`Ctx::announce_format`] — when the first frame decodes (the flacdec /
//! vp8dec pattern, spec: Formats — dynamic caps).
//!
//! Error scope is per-buffer (spec: Supervision — a bit-flipped access unit is
//! weather, not a verdict): an access unit that fails to decode posts a bus
//! `Warning` and is dropped; the backing decoder resynchronises at the next IDR.
//! FlushStart rebuilds the decoder from scratch (a seek resumes at an IDR, which
//! re-primes SPS/PPS + the reference DPB); EOS flushes the DPB so trailing
//! reordered pictures are not lost.

use oxideav_h264::h264_decoder::H264CodecDecoder;

use oxideav_core::{CodecId, Decoder, Error as OxError, Frame, Packet, TimeBase, VideoFrame};

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
// streamcraft-video) so sc-h264 stays core-only, exactly as sc-vp8 does; the
// pipeline interns by string, so the ids line up with any video peer using the
// same names.
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
const PIXFMT_I420: &str = "i420";

const SRC_PAD: PadId = PadId(1);

static PIXFMT_VALUES: [ValueDesc; 1] = [ValueDesc::Id(PIXFMT_I420)];

// A broad `video/raw` template: any dimensions, I420 only. The concrete
// width/height are announced at runtime from the SPS — the src pad is `dynamic`
// for exactly this reason. This element wires only 8-bit 4:2:0 output (the
// Baseline / Main / High-4:2:0 common case); a stream whose SPS selects 4:2:2,
// 4:4:4 or >8-bit samples decodes in the backing library but is refused here
// per-buffer (see `emit_pending`) rather than mislabelled as i420.
static SRC_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
];
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }];
// One H.264 Annex B access unit per buffer, as a demuxer (mkv `V_MPEG4/ISO/AVC`
// unwrapped to Annex B, an MPEG-TS/ES stream, a raw `.264`/`.h264` file split at
// access-unit boundaries) hands them out. The canonical family name is the
// bitstream framing: `h264/annexb` (start-code delimited, ITU-T H.264 Annex B).
// AVCC (length-prefixed, ISO/IEC 14496-15) is a distinct framing and would be a
// separate `h264/avcc` family with an `extradata`-driven length size — not v1.
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("h264/annexb")];

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
        dynamic: true, // dimensions are data-dependent (SPS) — announced at runtime
        validate: None,
    },
];

static DESC: ElementDesc = ElementDesc {
    name: "h264dec",
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
    make_default: None,
};

/// A decoded frame waiting for a pool slot — the backpressure carry, bounded to
/// one frame: `process` never pulls another picture out of the decoder until this
/// drains, so a slow sink cannot make the decoder run ahead and balloon the heap
/// (spec: backpressure — output elements alloc-then-yield, never unbounded).
struct PendingFrame {
    frame: VideoFrame,
    pts: Timestamp,
    duration: Timestamp,
}

/// Decodes H.264 Annex B access units to packed 8-bit I420.
pub struct H264Dec {
    dec: H264CodecDecoder,
    announced: bool,
    pending: Option<PendingFrame>,
    /// The pts of the access unit currently being fed. H.264 output is in
    /// **display order** and one access unit may yield zero or several frames
    /// (reordering / DPB delay), so a naive input-pts→output-frame pairing is
    /// wrong. v1 policy (documented, honest): stamp each emitted frame with the
    /// pts of the access unit whose `send_packet` produced it. For streams
    /// without B-frames (Baseline, or Main/High with `max_num_reorder_frames ==
    /// 0`) that is the exact display pts; for reordered streams it is the
    /// decode-order pts and downstream A/V sync should prefer container-supplied
    /// display timestamps until a full DTS/PTS reorder buffer lands (backlog).
    feed_pts: Timestamp,
    feed_duration: Timestamp,
}

impl H264Dec {
    pub fn new() -> Self {
        Self {
            dec: H264CodecDecoder::new(CodecId::new("h264")),
            announced: false,
            pending: None,
            feed_pts: Timestamp::NONE,
            feed_duration: Timestamp::ZERO,
        }
    }

    /// Copy the pending decoded frame into a pool buffer and emit it. `false`
    /// means the pool is exhausted — leave it pending (backpressure).
    ///
    /// The plane copy is the adoption's known cost (see lib.rs): the backing
    /// decoder owns its output plane `Vec`s, so this element pays one packed-I420
    /// copy per frame. It is also where the output contract is enforced: exactly
    /// three 8-bit planes (Yuv420P). A stream whose SPS selects a non-4:2:0 or
    /// deeper-than-8-bit format produces a different plane shape upstream — refuse
    /// it loudly here rather than emit bytes mislabelled `i420`.
    fn emit_pending(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        let Some(p) = self.pending.take() else { return Ok(true) };

        // `image_planes()` excludes the trailing palette / significant-bits
        // side-channel records oxideav-core frames may carry.
        let planes = p.frame.image_planes();
        if planes.len() != 3 {
            let element = ctx.element();
            ctx.post(BusMessage::Warning {
                element,
                error: Error::Element {
                    element,
                    message: format!(
                        "h264dec: frame dropped — {} image planes, only packed 8-bit I420 \
                         (3-plane 4:2:0) output is wired",
                        planes.len()
                    ),
                },
            });
            return Ok(true);
        }
        let (y, u, v) = (&planes[0], &planes[1], &planes[2]);
        // Derive the picture geometry from the plane strides. The backing
        // decoder emits MB-aligned dimensions (§7.4.2.1.1: PicWidthInMbs * 16)
        // with stride == width (no per-row padding) for 8-bit planes, so the
        // stride is the luma width and `data.len()/stride` the height. An 8-bit
        // 4:2:0 frame has chroma stride == luma stride / 2; a wider (u16-packed)
        // plane would violate that — treat it as the non-i420 refusal above.
        let width = y.stride;
        if width == 0 || y.data.len() % width != 0 {
            let element = ctx.element();
            ctx.post(BusMessage::Warning {
                element,
                error: Error::Element {
                    element,
                    message: "h264dec: frame dropped — degenerate luma plane geometry".into(),
                },
            });
            return Ok(true);
        }
        let height = y.data.len() / width;
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);
        // Byte-exact 8-bit 4:2:0 shape check: reject anything that is not packed
        // Yuv420P (this catches >8-bit LE-u16 planes, 4:2:2 and 4:4:4).
        if u.stride != cw
            || v.stride != cw
            || u.data.len() != cw * ch
            || v.data.len() != cw * ch
        {
            let element = ctx.element();
            ctx.post(BusMessage::Warning {
                element,
                error: Error::Element {
                    element,
                    message: format!(
                        "h264dec: frame dropped — chroma geometry {}x{} (strides {}/{}) is not \
                         packed 8-bit I420 for a {width}x{height} luma; 4:2:2/4:4:4/>8-bit output \
                         is decoded upstream but not wired here",
                        u.data.len(),
                        v.data.len(),
                        u.stride,
                        v.stride
                    ),
                },
            });
            return Ok(true);
        }

        let need = y.data.len() + u.data.len() + v.data.len();
        let Some(mut buf) = ctx.try_alloc(SRC_PAD) else {
            self.pending = Some(p);
            return Ok(false);
        };
        if buf.memory.capacity() < need {
            // Pool slots are sized by the pipeline; a frame that cannot fit any
            // slot would silently truncate — fail loudly instead (pool sizing is
            // the pipeline's pending spec work; the default 128 KiB slot caps
            // frames at ~320×256 I420).
            return Err(Error::Element {
                element: ctx.element(),
                message: format!(
                    "h264dec: {width}x{height} I420 frame needs {need} bytes but pool slots hold \
                     {} — raise the pipeline pool slot size",
                    buf.memory.capacity()
                ),
            });
        }
        if !self.announced {
            ctx.announce_format(
                SRC_PAD,
                FAMILY,
                &[
                    (F_WIDTH, ValueDesc::Int(width as i64)),
                    (F_HEIGHT, ValueDesc::Int(height as i64)),
                    (F_PIXFMT, ValueDesc::Id(PIXFMT_I420)),
                ],
            );
            self.announced = true;
        }
        let dst = buf.memory.as_mut_full();
        let (ylen, ulen) = (y.data.len(), u.data.len());
        dst[..ylen].copy_from_slice(&y.data);
        dst[ylen..ylen + ulen].copy_from_slice(&u.data);
        dst[ylen + ulen..need].copy_from_slice(&v.data);
        buf.memory.set_len(need);
        buf.pts = p.pts;
        buf.duration = p.duration;
        ctx.out(SRC_PAD).push(buf);
        Ok(true)
    }

    /// Pull every ready picture out of the backing decoder into `pending`,
    /// emitting each to the pool as slots allow. Returns `false` when the pool
    /// ran out mid-drain (a picture is left carried) so `process` stops pulling
    /// input. `Error::NeedMore` / `Error::Eof` are the decoder's "no more frames
    /// right now" signals, not failures.
    fn drain_frames(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        loop {
            if !self.emit_pending(ctx)? {
                return Ok(false);
            }
            match self.dec.receive_frame() {
                Ok(Frame::Video(vf)) => {
                    self.pending = Some(PendingFrame {
                        frame: vf,
                        pts: self.feed_pts,
                        duration: self.feed_duration,
                    });
                }
                // A non-video frame from an H.264 stream should not happen; skip
                // it rather than abort the pipeline.
                Ok(_) => continue,
                Err(OxError::NeedMore) | Err(OxError::Eof) => return Ok(true),
                Err(e) => {
                    // Per-buffer error scope: warn, drop, resync at the next IDR
                    // (spec: Supervision).
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: format!("h264dec: decode error: {e:?}"),
                        },
                    });
                    return Ok(true);
                }
            }
        }
    }
}

impl Default for H264Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for H264Dec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        loop {
            // Drain ready pictures first; if the pool is out of slots, stop
            // pulling input — un-popped buffers stay in the batch for the next
            // pass (backpressure).
            if !self.drain_frames(ctx)? {
                return Ok(Flow::Ok);
            }
            let Some(inbuf) = inputs.pop() else { break };
            self.feed_pts = inbuf.pts;
            self.feed_duration = inbuf.duration;
            // One Annex B access unit per buffer. `TimeBase` here is cosmetic —
            // streamcraft carries timing on the buffer, not the packet — so a
            // unit base keeps the library's internal pts rescale a no-op.
            let packet = Packet::new(0, TimeBase::new(1, 1), inbuf.memory.data().to_vec());
            if let Err(e) = self.dec.send_packet(&packet) {
                // Per-buffer error scope (spec: Supervision).
                let element = ctx.element();
                ctx.post(BusMessage::Warning {
                    element,
                    error: Error::Element {
                        element,
                        message: format!("h264dec: access unit dropped: {e:?}"),
                    },
                });
                continue;
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Flush/seek: drop the carry and rebuild the decoder — the stream
            // resumes at an IDR, which re-primes SPS/PPS and every reference
            // picture (spec: flush/seek). A fresh decoder is the cleanest reset;
            // `H264CodecDecoder` also has a `reset()` but a full rebuild has no
            // stale-state risk and is not on a hot path.
            Event::FlushStart => {
                self.pending = None;
                self.dec = H264CodecDecoder::new(CodecId::new("h264"));
                // `announced` is intentionally NOT cleared: dimensions do not
                // change across a seek within one stream, and re-announcing an
                // identical format is needless churn.
            }
            // Drain the DPB at EOS so trailing reordered pictures (B-frame delay)
            // are not lost.
            Event::Eos => {
                let _ = self.dec.flush();
                let _ = self.drain_frames(ctx)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}
