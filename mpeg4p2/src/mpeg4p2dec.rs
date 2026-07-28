//! `mpeg4p2dec` — the MPEG-4 Part 2 (Visual) Advanced Simple Profile decode
//! element (spec: Milestone applications §5; Formats — dynamic caps). Coded VOPs
//! arrive on the sink pad — the AVI demuxer's `mpeg4/asp` family (or raw `bytes`)
//! hands out **one container frame per buffer**, which may hold one coded VOP or,
//! for DivX packed bitstream, a P-VOP + the following B-VOP plus a stuffing tail
//! (see [`crate::packed`]). Raw I420 video leaves on the src pad, one frame per
//! buffer, planes packed Y then U then V (the `video/raw` layout).
//!
//! Persistent stream parameters (VOL/VOS headers, §6.2) arrive in-band at the
//! start of an XviD elementary stream and are parsed from the first buffer(s);
//! a demuxer may also supply them as negotiated `extradata` (not required here).
//!
//! Error scope is per-buffer (spec: Supervision — a bit-flipped VOP is weather,
//! not a verdict): a VOP that fails to decode posts a bus `Warning` and is
//! dropped; decoding resumes at the next start code / keyframe. Tools this v1 does
//! not implement (quarter-pel, GMC/sprite, interlaced, data-partitioning) are
//! detected in the VOL and warned once, then the affected frames drop rather than
//! decode wrong.

use crate::decoder::{DecodeResult, Decoder};
use crate::headers::{self, startcode, VolHeader};
use crate::packed;

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
use streamcraft_video::color;

// `video/raw` family/field/value names (kept as literals, like sc-vp8/sc-h264).
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
const PIXFMT_I420: &str = "i420";

const SRC_PAD: PadId = PadId(1);
const SINK_PAD: PadId = PadId(0);

static PIXFMT_VALUES: [ValueDesc; 1] = [ValueDesc::Id(PIXFMT_I420)];

static SRC_FIELDS: [FieldDesc; 7] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
    FieldDesc { field: color::FIELD_MATRIX, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_RANGE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_TRANSFER, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_PRIMARIES, allowed: ConstraintDesc::Any, preferred: None },
];
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }];

// Sink: the AVI demuxer announces `mpeg4/asp`; a raw `.m4v` split at VOP
// boundaries is `bytes`. Both carry one coded unit (container frame) per buffer.
static SINK_COLOR_FIELDS: [FieldDesc; 4] = [
    FieldDesc { field: color::FIELD_MATRIX, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_RANGE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_TRANSFER, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: color::FIELD_PRIMARIES, allowed: ConstraintDesc::Any, preferred: None },
];
static SINK_OFFERS: [OfferDesc; 2] = [
    OfferDesc { family: "mpeg4/asp", fields: &SINK_COLOR_FIELDS },
    OfferDesc { family: "bytes", fields: &[] },
];

static PADS: [PadDesc; 2] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &SINK_OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &SRC_OFFERS, dynamic: true, validate: None },
];

// `make_default` boxes the element once at registry construction (spec: Plugins) — cold.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "mpeg4p2dec",
    pads: &PADS,
    props: &[],
    // Active: an ASP decode (IDCT + MC over 1280×544) is milliseconds, far beyond
    // the inline passive budget (spec: Scheduling). Its own thread group
    // pipelines decode against demux and display.
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(Mpeg4p2Dec::new())),
};

/// A decoded frame awaiting a pool slot — the backpressure carry.
struct PendingFrame {
    pic: crate::frame::Picture,
    pts: Timestamp,
    duration: Timestamp,
}

/// The MPEG-4 Part 2 ASP decode element.
pub struct Mpeg4p2Dec {
    /// The decode core, built once the VOL header is parsed.
    dec: Option<Decoder>,
    vol: VolHeader,
    vol_parsed: bool,
    announced: bool,
    warned_unsupported: bool,
    /// Decoded frames waiting for a pool slot. A packed buffer can yield two
    /// displayable VOPs (P + B), so the carry is a small queue, not one slot; it
    /// drains front-first so display order is preserved.
    pending: std::collections::VecDeque<PendingFrame>,
}

impl Mpeg4p2Dec {
    pub fn new() -> Self {
        Mpeg4p2Dec {
            dec: None,
            vol: VolHeader::default(),
            vol_parsed: false,
            announced: false,
            warned_unsupported: false,
            pending: std::collections::VecDeque::new(),
        }
    }

    /// Parse any VOL/VOS headers present at the front of `headers` (or a whole
    /// buffer). Builds the decode core once dimensions are known.
    fn try_parse_headers(&mut self, data: &[u8]) {
        // Walk start codes; parse the VOL when found.
        let mut i = 0usize;
        while let Some((off, code)) = headers::find_start_code(data, i) {
            if (startcode::VOL_MIN..=startcode::VOL_MAX).contains(&code) {
                let body_start = (off + 4) * 8; // after 00 00 01 XX
                let mut r = crate::bits::BitReader::at(data, body_start);
                let mut vol = VolHeader::default();
                if headers::parse_vol(&mut r, &mut vol).is_some() && vol.width > 0 && vol.height > 0 {
                    self.vol = vol;
                    self.vol_parsed = true;
                    self.dec = Some(Decoder::new(self.vol.clone()));
                }
            }
            if code == startcode::VOP_START {
                break; // headers end at the first VOP
            }
            i = off + 3;
        }
    }

    /// Warn once about VOL tools this decoder does not implement.
    //
    // Runs at most once per stream (guarded by `warned_unsupported`); `notes` holds a
    // few `&'static str` labels — a one-time setup allocation, not per-frame — cold.
    #[allow(clippy::disallowed_methods)]
    fn warn_unsupported(&mut self, ctx: &mut Ctx) {
        if self.warned_unsupported {
            return;
        }
        let mut notes = Vec::new();
        if self.vol.quarter_sample {
            notes.push("quarter-pel (qpel)");
        }
        if self.vol.sprite_enable != 0 {
            notes.push("GMC/sprite");
        }
        if self.vol.interlaced {
            notes.push("interlaced");
        }
        if self.vol.data_partitioned {
            notes.push("data-partitioned/RVLC");
        }
        if notes.is_empty() {
            return;
        }
        self.warned_unsupported = true;
        let element = ctx.element();
        ctx.post(BusMessage::Warning {
            element,
            error: Error::Element {
                element,
                message: format!(
                    "mpeg4p2dec: stream uses unimplemented tool(s): {} — affected frames drop",
                    notes.join(", ")
                ),
            },
        });
    }

    /// Drain as many carried frames as pool slots allow, front-first. Returns
    /// `false` when the pool ran dry with frames still queued (backpressure).
    fn drain_pending(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        while !self.pending.is_empty() {
            if !self.emit_one(ctx)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn emit_one(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        let Some(p) = self.pending.pop_front() else { return Ok(true) };
        let (w, h) = (p.pic.width, p.pic.height);
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let need = w * h + 2 * cw * ch;
        let Some(mut buf) = ctx.try_alloc(SRC_PAD) else {
            self.pending.push_front(p);
            return Ok(false);
        };
        if buf.memory.capacity() < need {
            return Err(Error::Element {
                element: ctx.element(),
                message: format!(
                    "mpeg4p2dec: {w}x{h} I420 frame needs {need} bytes but pool slots hold {} — \
                     raise the pipeline pool slot size",
                    buf.memory.capacity()
                ),
            });
        }
        if !self.announced {
            let mut fields = vec![
                (F_WIDTH, ValueDesc::Int(w as i64)),
                (F_HEIGHT, ValueDesc::Int(h as i64)),
                (F_PIXFMT, ValueDesc::Id(PIXFMT_I420)),
            ];
            color_passthrough(ctx, SINK_PAD, &mut fields);
            ctx.announce_format(SRC_PAD, FAMILY, &fields);
            self.announced = true;
        }
        // Crop the MB-aligned planes down to the display dimensions.
        let dst = buf.memory.as_mut_full();
        let mut o = 0;
        for row in 0..h {
            let src = &p.pic.y[row * p.pic.lstride..row * p.pic.lstride + w];
            dst[o..o + w].copy_from_slice(src);
            o += w;
        }
        for row in 0..ch {
            let src = &p.pic.u[row * p.pic.cstride..row * p.pic.cstride + cw];
            dst[o..o + cw].copy_from_slice(src);
            o += cw;
        }
        for row in 0..ch {
            let src = &p.pic.v[row * p.pic.cstride..row * p.pic.cstride + cw];
            dst[o..o + cw].copy_from_slice(src);
            o += cw;
        }
        buf.memory.set_len(need);
        buf.pts = p.pts;
        buf.duration = p.duration;
        ctx.out(SRC_PAD).push(buf);
        Ok(true)
    }

    /// Decode one coded VOP unit (VOP bytes, headers already parsed) into the
    /// pending carry. Warns-and-drops on any failure.
    fn decode_one(&mut self, ctx: &mut Ctx, vop: &[u8], pts: Timestamp, duration: Timestamp) {
        let Some(dec) = self.dec.as_mut() else { return };
        // Parse the VOP header (start code is 00 00 01 B6 at vop[0..4]).
        let body = (4) * 8;
        let mut r = crate::bits::BitReader::at(vop, body);
        let vh = match headers::parse_vop(&mut r, dec.vol()) {
            Some(v) => v,
            None => {
                warn(ctx, "malformed VOP header");
                return;
            }
        };
        match dec.decode_vop(&vh, &mut r) {
            Some(DecodeResult::Picture(pic)) | Some(DecodeResult::Repeat(pic)) => {
                self.pending.push_back(PendingFrame { pic, pts, duration });
            }
            Some(DecodeResult::None) | None => {
                warn(ctx, "VOP decode failed or used an unimplemented tool");
            }
        }
    }
}

impl Default for Mpeg4p2Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Mpeg4p2Dec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        loop {
            // Drain the carry first; a dry pool stops us pulling input, leaving
            // un-popped buffers for the next pass (backpressure).
            if !self.drain_pending(ctx)? {
                return Ok(Flow::Ok);
            }
            let Some(inbuf) = inputs.pop() else { break };
            let pts = inbuf.pts;
            let duration = inbuf.duration;
            // Parse headers from the first buffer(s) if not yet ready.
            if !self.vol_parsed {
                self.try_parse_headers(inbuf.memory.data());
                if self.vol_parsed {
                    self.warn_unsupported(ctx);
                }
            }
            if !self.vol_parsed {
                continue; // no VOL yet — cannot decode; drop until headers arrive
            }
            // Walk the coded VOP units (packed-bitstream aware) straight out of the
            // input buffer and decode each in place — no copy, no allocation. The
            // cursor holds only offsets; `decode_one`/`drain_pending` borrow `self`
            // and `ctx`, disjoint from the input buffer `data` borrows.
            let data = inbuf.memory.data();
            let mut cursor = packed::VopCursor::new(data);
            while let Some(vop) = cursor.next(data) {
                if vop.len() < 5 {
                    continue;
                }
                self.decode_one(ctx, vop, pts, duration);
                // Emit eagerly so the carry stays small; if the pool is dry the
                // frames simply queue until the next pass.
                if !self.drain_pending(ctx)? {
                    // Stop pulling more input this pass.
                    return Ok(Flow::Ok);
                }
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FlushStart => {
                self.pending.clear();
                if let Some(dec) = self.dec.as_mut() {
                    dec.flush();
                }
            }
            Event::Eos => {
                let _ = self.drain_pending(ctx)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

fn warn(ctx: &mut Ctx, msg: &str) {
    let element = ctx.element();
    ctx.post(BusMessage::Warning {
        element,
        error: Error::Element {
            element,
            message: format!("mpeg4p2dec: {msg}"),
        },
    });
}

/// Colorimetry passthrough (spec: Formats — dynamic caps): forward the container's
/// announced colour fields from the negotiated sink format onto `video/raw`.
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
