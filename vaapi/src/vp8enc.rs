//! `vaapivp8enc` — VA-API hardware VP8 encode: `video/raw` (NV12/I420) in, raw
//! VP8 frames (RFC 6386) out on the `vp8` family — exactly what `mkvmuxn`'s
//! `V_VP8` lane and the software `vp8dec` consume.
//!
//! VP8 is the easy one of this crate's encoders: the driver authors the *entire*
//! frame — uncompressed data chunk header (RFC 6386 §9.1), compressed header and
//! coefficient partitions — so there are no packed headers and no bitstream
//! writing here, only parameter buffers.
//!
//! ## Reference policy (RFC 6386 §9.7)
//! Keyframe every `gop` frames (a keyframe refreshes all three reference
//! buffers by definition); inter frames predict from LAST only and refresh it,
//! while `copy_buffer_to_golden/alternate = 1` keeps GOLDEN/ALTREF trailing the
//! last frame — every buffer stays defined without maintaining separate golden
//! logic, matching the single-reference GOP the other encoders use.
//!
//! `qp` here is the VP8 **quantizer index** (0..127, RFC 6386 §9.6), not an
//! H.26x QP — default 40. `bitrate` > 0 switches to driver VBR.

use std::collections::VecDeque;
use std::path::PathBuf;

use streamcraft_core::batch::Inputs;
use streamcraft_core::buffer::BufferFlags;
use streamcraft_core::bus::BusMessage;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, PropDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{Constraint, ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::enc::{self, EncEngine, InFormat};
use crate::ffi;
use crate::probe;
use crate::va::{self, Buffer as VaBuffer, Display};

const SINK_PAD: PadId = PadId(0);
const SRC_PAD: PadId = PadId(1);

/// Bound on frames staged as heap bytes while the output pool is dry (see
/// `vaapih264enc` — the shared-pool deadlock rationale).
const PENDING_MAX: usize = 32;

static PIXFMT_VALUES: [ValueDesc; 2] = [ValueDesc::Id("nv12"), ValueDesc::Id("i420")];

static SINK_FIELDS: [FieldDesc; 4] = [
    FieldDesc { field: enc::F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: enc::F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: enc::F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
    FieldDesc { field: enc::F_FPS, allowed: ConstraintDesc::Any, preferred: None },
];
static SINK_OFFERS: [OfferDesc; 1] =
    [OfferDesc { family: enc::RAW_FAMILY, fields: &SINK_FIELDS }];

static OUT_FIELDS: [FieldDesc; 2] = [
    FieldDesc { field: enc::F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: enc::F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
];
static SRC_OFFERS: [OfferDesc; 2] = [
    OfferDesc { family: "vp8", fields: &OUT_FIELDS },
    // The `bytes` escape (raw concatenated VP8 frames — framing is the sink's
    // problem, same contract as the mkv demux's escape offer).
    OfferDesc::any("bytes"),
];

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
        dynamic: true,
        validate: None,
    },
];

static PROPS: [PropDesc; 3] = [
    // VP8 quantizer index (0..127, RFC 6386 §9.6); used when bitrate == 0.
    PropDesc { name: "qp", allowed: Constraint::Any, live: false },
    // Target bitrate in kbit/s; 0 (default) = constant quantizer.
    PropDesc { name: "bitrate", allowed: Constraint::Any, live: false },
    // Keyframe period in frames.
    PropDesc { name: "gop", allowed: Constraint::Any, live: false },
];

static DESC: ElementDesc = ElementDesc {
    name: "vaapivp8enc",
    pads: &PADS,
    props: &PROPS,
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(VaapiVp8Enc::new())),
};

struct OutFrame {
    bytes: Vec<u8>,
    pts: Timestamp,
    duration: Timestamp,
    key: bool,
}

struct Session {
    engine: EncEngine,
    gop_pos: u32,
}

/// ```text
/// +---------------------+
/// |____               __|
/// |sink| VaapiVp8Enc |src|--> vp8
/// |^^^^               ^^|
/// +---------------------+
/// ```
pub struct VaapiVp8Enc {
    device: Option<PathBuf>,
    display: Option<Display>,
    session: Option<Session>,

    qindex: u8,
    bitrate_kbps: u32,
    gop_len: u32,

    in_fmt: Option<InFormat>,
    /// The link fixated the `bytes` escape — announce bare `bytes`.
    bytes_out: bool,
    announced: bool,
    pending: VecDeque<OutFrame>,
    disabled: bool,
}

// SAFETY: `SchedHint::Active` single-thread confinement, as for the other VA
// elements — the raw `VADisplay` never leaves this element's scheduler thread.
unsafe impl Send for VaapiVp8Enc {}

impl Default for VaapiVp8Enc {
    fn default() -> Self {
        Self::new()
    }
}

impl VaapiVp8Enc {
    pub fn new() -> Self {
        VaapiVp8Enc {
            device: None,
            display: None,
            session: None,
            qindex: 40,
            bitrate_kbps: 0,
            gop_len: 60,
            in_fmt: None,
            bytes_out: false,
            announced: false,
            pending: VecDeque::new(),
            disabled: false,
        }
    }

    /// Quantizer index (0..127, RFC 6386 §9.6). Chainable; the `qp` prop overrides.
    pub fn with_qindex(mut self, q: u8) -> Self {
        self.qindex = q.min(127);
        self
    }

    /// Keyframe period in frames (1 = all-keyframe). The `gop` prop overrides.
    pub fn with_gop(mut self, gop: u32) -> Self {
        self.gop_len = gop.clamp(1, 1024);
        self
    }

    /// Target bitrate in kbit/s (0 = constant quantizer). The `bitrate` prop
    /// overrides.
    pub fn with_bitrate(mut self, kbps: u32) -> Self {
        self.bitrate_kbps = kbps;
        self
    }

    fn warn(&self, ctx: &mut Ctx, message: String) {
        let element = ctx.element();
        ctx.post(BusMessage::Warning {
            element,
            error: Error::Element { element, message },
        });
    }

    fn ensure_display(&mut self, ctx: &mut Ctx) -> bool {
        if self.display.is_some() {
            return true;
        }
        let Some(caps) = probe::probe() else {
            if !self.disabled {
                self.disabled = true;
                self.warn(ctx, "vaapivp8enc: no VA-API device available".into());
            }
            return false;
        };
        match Display::open(&caps.device) {
            Ok(d) => {
                self.device = Some(caps.device.clone());
                self.display = Some(d);
                true
            }
            Err(e) => {
                if !self.disabled {
                    self.disabled = true;
                    self.warn(ctx, format!("vaapivp8enc: cannot open VA display: {e}"));
                }
                false
            }
        }
    }

    fn ensure_session(&mut self, ctx: &mut Ctx, fmt: InFormat) -> bool {
        if let Some(s) = &self.session {
            if s.engine.width == fmt.width && s.engine.height == fmt.height {
                return true;
            }
            self.session = None;
            self.announced = false;
        }
        let Some(display) = &self.display else { return false };
        let rc_mode = if self.bitrate_kbps > 0 { ffi::VA_RC_VBR } else { ffi::VA_RC_CQP };
        let engine = match EncEngine::new(
            display,
            ffi::VAProfileVP8Version0_3,
            ffi::VAEntrypointEncSlice,
            rc_mode,
            fmt.width,
            fmt.height,
            16, // VP8 codes whole macroblocks (RFC 6386 §2)
        ) {
            Ok(e) => e,
            Err(e) => {
                self.warn(ctx, format!("vaapivp8enc: VA session setup failed: {e}"));
                return false;
            }
        };
        self.session = Some(Session { engine, gop_pos: 0 });
        true
    }

    fn drain_pending(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        while let Some(front) = self.pending.front() {
            let Some(mut buf) = ctx.try_alloc(SRC_PAD) else { return Ok(false) };
            if buf.memory.capacity() < front.bytes.len() {
                return Err(Error::Element {
                    element: ctx.element(),
                    message: format!(
                        "vaapivp8enc: encoded frame of {} bytes exceeds pool slot \
                         capacity {} — raise the pipeline pool slot size",
                        front.bytes.len(),
                        buf.memory.capacity()
                    ),
                });
            }
            let front = self.pending.pop_front().expect("front exists");
            buf.memory.as_mut_full()[..front.bytes.len()].copy_from_slice(&front.bytes);
            buf.memory.set_len(front.bytes.len());
            buf.pts = front.pts;
            buf.duration = front.duration;
            buf.flags = if front.key { BufferFlags::KEYFRAME } else { BufferFlags::DELTA };
            ctx.out(SRC_PAD).push(buf);
        }
        Ok(true)
    }

    fn encode_frame(
        &mut self,
        ctx: &mut Ctx,
        data: &[u8],
        pts: Timestamp,
        duration: Timestamp,
    ) -> Result<(), Error> {
        let Some(fmt) = self.in_fmt else { return Ok(()) };
        let result = {
            let (Some(display), Some(s)) = (&self.display, &mut self.session) else {
                return Ok(());
            };
            let is_kf = s.gop_pos == 0;
            encode_one(display, s, fmt, self.qindex, self.bitrate_kbps, self.gop_len, is_kf, data)
                .map(|bytes| (bytes, is_kf))
        };
        let (bytes, is_kf) = match result {
            Ok(v) => v,
            Err(e) => {
                self.warn(ctx, format!("vaapivp8enc: encode failed: {e} — frame dropped"));
                return Ok(());
            }
        };

        if !self.announced {
            if self.bytes_out {
                ctx.announce_format(SRC_PAD, "bytes", &[]);
            } else {
                ctx.announce_format(
                    SRC_PAD,
                    "vp8",
                    &[
                        (enc::F_WIDTH, ValueDesc::Int(fmt.width as i64)),
                        (enc::F_HEIGHT, ValueDesc::Int(fmt.height as i64)),
                    ],
                );
            }
            self.announced = true;
        }
        self.pending.push_back(OutFrame { bytes, pts, duration, key: is_kf });

        let gop_len = self.gop_len.clamp(1, 1024);
        let s = self.session.as_mut().expect("session exists");
        s.engine.advance_recon();
        s.gop_pos += 1;
        if s.gop_pos >= gop_len {
            s.gop_pos = 0;
        }
        Ok(())
    }
}

/// Upload + submit + sync + drain one VP8 frame; the coded-buffer bytes are the
/// complete frame (the driver writes all VP8 headers).
#[allow(clippy::too_many_arguments)]
fn encode_one(
    display: &Display,
    s: &mut Session,
    fmt: InFormat,
    qindex: u8,
    bitrate_kbps: u32,
    gop_len: u32,
    is_kf: bool,
    data: &[u8],
) -> va::VaResult<Vec<u8>> {
    s.engine.upload(display, data, fmt.pixfmt)?;

    let engine = &s.engine;
    let ctxva = engine.context();
    let mut hold: Vec<VaBuffer> = Vec::with_capacity(8);
    let mut ids: Vec<ffi::VABufferID> = Vec::with_capacity(8);

    let coded = VaBuffer::new_empty(display, ctxva, ffi::VAEncCodedBufferType, engine.coded_buf_size())?;

    if is_kf {
        let seq = ffi::VAEncSequenceParameterBufferVP8 {
            frame_width: engine.width,
            frame_height: engine.height,
            frame_width_scale: 0,
            frame_height_scale: 0,
            error_resilient: 0,
            kf_auto: 0, // we force keyframes ourselves
            kf_min_dist: 0,
            kf_max_dist: gop_len,
            bits_per_second: bitrate_kbps * 1000,
            intra_period: gop_len,
            reference_frames: [ffi::VA_INVALID_SURFACE; 4],
            va_reserved: [0; ffi::VA_PADDING_LOW],
        };
        let b = VaBuffer::new_struct(display, ctxva, ffi::VAEncSequenceParameterBufferType, &seq)?;
        ids.push(b.id());
        hold.push(b);
        if bitrate_kbps > 0 {
            for payload in [rc_misc_bytes(bitrate_kbps), framerate_misc_bytes(fmt.fps)] {
                let b = VaBuffer::new_typed_bytes(
                    display,
                    ctxva,
                    ffi::VAEncMiscParameterBufferType,
                    &payload,
                )?;
                ids.push(b.id());
                hold.push(b);
            }
        }
    }

    // ref_flags (va_enc_vp8.h bit order): force_kf:1, no_ref_last:1, no_ref_gf:1,
    // no_ref_arf:1, temporal_id:8, first_ref:2, second_ref:2.
    let ref_flags: u32 = if is_kf {
        1 | (1 << 1) | (1 << 2) | (1 << 3) // force_kf + no references
    } else {
        0
    };
    // pic_flags bit order: frame_type:1, version:3, show_frame:1, color_space:1,
    // recon_filter_type:2, loop_filter_type:2, auto_partitions:1,
    // num_token_partitions:2, clamping_type:1, segmentation_enabled:1,
    // update_mb_segmentation_map:1, update_segment_feature_data:1,
    // loop_filter_adj_enable:1, refresh_entropy_probs:1, refresh_golden_frame:1,
    // refresh_alternate_frame:1, refresh_last:1, copy_buffer_to_golden:2,
    // copy_buffer_to_alternate:2, sign_bias_golden:1, sign_bias_alternate:1,
    // mb_no_coeff_skip:1, forced_lf_adjustment:1.
    let mut pic_flags: u32 = 0;
    if !is_kf {
        pic_flags |= 1; // frame_type = inter (RFC 6386 §9.1 key_frame bit inverted)
    }
    pic_flags |= 1 << 4; // show_frame
    pic_flags |= 1 << 26; // mb_no_coeff_skip (allow skip-coded macroblocks)
    if !is_kf {
        pic_flags |= 1 << 20; // refresh_last (LAST tracks the newest frame)
        pic_flags |= 1 << 21; // copy_buffer_to_golden = 1 (GOLDEN := LAST, §9.7.2)
        pic_flags |= 1 << 23; // copy_buffer_to_alternate = 1 (ALTREF := LAST)
    }
    // A keyframe refreshes everything by definition (§9.7); iHD wants the refresh
    // bits set to match.
    if is_kf {
        pic_flags |= (1 << 18) | (1 << 19) | (1 << 20); // refresh golden/alt/last
    }

    // Loop-filter strength scaled from the quantizer (both live in 0..63 /
    // 0..127 ranges; a mid-strength filter tracks quantization noise well enough
    // for a fixed-QP v1 — RFC 6386 §9.4 leaves the level fully encoder-chosen).
    let lf = ((qindex as i32) / 2).clamp(1, 63) as i8;

    let pic = ffi::VAEncPictureParameterBufferVP8 {
        reconstructed_frame: engine.recon_cur(),
        ref_last_frame: if is_kf { ffi::VA_INVALID_SURFACE } else { engine.recon_ref() },
        ref_gf_frame: if is_kf { ffi::VA_INVALID_SURFACE } else { engine.recon_ref() },
        ref_arf_frame: if is_kf { ffi::VA_INVALID_SURFACE } else { engine.recon_ref() },
        coded_buf: coded.id(),
        ref_flags,
        pic_flags,
        loop_filter_level: [lf; 4],
        ref_lf_delta: [0; 4],
        mode_lf_delta: [0; 4],
        sharpness_level: 0,
        clamp_qindex_high: 127,
        clamp_qindex_low: 0,
        va_reserved: [0; ffi::VA_PADDING_LOW],
    };
    let b = VaBuffer::new_struct(display, ctxva, ffi::VAEncPictureParameterBufferType, &pic)?;
    ids.push(b.id());
    hold.push(b);

    // Quantizer indices (RFC 6386 §9.6): one base index per segment (segmentation
    // off → all four identical), per-class deltas zero.
    let q = ffi::VAQMatrixBufferVP8 {
        quantization_index: [qindex as u16; 4],
        quantization_index_delta: [0; 5],
        va_reserved: [0; ffi::VA_PADDING_LOW],
    };
    let b = VaBuffer::new_struct(display, ctxva, ffi::VAQMatrixBufferType, &q)?;
    ids.push(b.id());
    hold.push(b);

    engine.submit(display, &ids)?;
    drop(hold);

    let mut bytes = Vec::new();
    coded.read_coded(&mut bytes)?;
    Ok(bytes)
}

fn misc_bytes<T: Copy>(misc_type: u32, value: &T) -> Vec<u8> {
    let size = std::mem::size_of::<T>();
    let mut out = Vec::with_capacity(4 + size);
    out.extend_from_slice(&misc_type.to_ne_bytes());
    // SAFETY: T is a plain #[repr(C)] parameter struct; reading its bytes is fine.
    out.extend_from_slice(unsafe {
        std::slice::from_raw_parts(value as *const T as *const u8, size)
    });
    out
}

fn rc_misc_bytes(bitrate_kbps: u32) -> Vec<u8> {
    let rc = ffi::VAEncMiscParameterRateControl {
        bits_per_second: bitrate_kbps * 1000,
        target_percentage: 95,
        window_size: 1000,
        initial_qp: 0,
        min_qp: 4,
        basic_unit_size: 0,
        rc_flags: 0,
        ICQ_quality_factor: 0,
        max_qp: 127,
        quality_factor: 0,
        target_frame_size: 0,
        va_reserved: [0; ffi::VA_PADDING_LOW],
    };
    misc_bytes(ffi::VAEncMiscParameterTypeRateControl, &rc)
}

fn framerate_misc_bytes(fps: (u32, u32)) -> Vec<u8> {
    let (num, den) = if fps.0 > 0 { fps } else { (30, 1) };
    let fr = ffi::VAEncMiscParameterFrameRate {
        framerate: (den.min(0xFFFF) << 16) | num.min(0xFFFF),
        framerate_flags: 0,
        va_reserved: [0; ffi::VA_PADDING_LOW],
    };
    misc_bytes(ffi::VAEncMiscParameterTypeFrameRate, &fr)
}

impl Element for VaapiVp8Enc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        if let Some(Value::Int(v)) = ctx.prop("qp") {
            self.qindex = v.clamp(0, 127) as u8;
        }
        if let Some(Value::Int(v)) = ctx.prop("bitrate") {
            self.bitrate_kbps = v.clamp(0, 500_000) as u32;
        }
        if let Some(Value::Int(v)) = ctx.prop("gop") {
            self.gop_len = v.clamp(1, 1024) as u32;
        }
        self.bytes_out = ctx
            .negotiated(SRC_PAD)
            .and_then(|f| ctx.family_name(f.family))
            .is_some_and(|name| name == "bytes");
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.ensure_display(ctx) {
            while inputs.pop().is_some() {}
            return Ok(Flow::Ok);
        }
        if self.in_fmt.is_none() {
            if let Some(f) = ctx.negotiated(SINK_PAD).cloned() {
                self.in_fmt = enc::read_in_format(ctx, &f);
            }
        }
        loop {
            // A dry output pool must not stop input consumption (shared-pool
            // deadlock with a downstream muxer — see vaapih264enc's process);
            // staged output is heap bytes, bounded by PENDING_MAX.
            self.drain_pending(ctx)?;
            if self.pending.len() >= PENDING_MAX {
                return Ok(Flow::Ok);
            }
            let Some(inbuf) = inputs.pop() else { break };
            let Some(fmt) = self.in_fmt else {
                self.warn(ctx, "vaapivp8enc: raw frame before any format — dropped".into());
                continue;
            };
            if inbuf.memory.data().len() < fmt.frame_bytes() {
                self.warn(
                    ctx,
                    format!(
                        "vaapivp8enc: short frame ({} < {} bytes) — dropped",
                        inbuf.memory.data().len(),
                        fmt.frame_bytes()
                    ),
                );
                continue;
            }
            if !self.ensure_session(ctx, fmt) {
                continue;
            }
            let data = inbuf.memory.data().to_vec();
            self.encode_frame(ctx, &data, inbuf.pts, inbuf.duration)?;
        }
        self.drain_pending(ctx)?;
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FormatChange(f) => {
                if let Some(fmt) = enc::read_in_format(ctx, f) {
                    self.in_fmt = Some(fmt);
                }
            }
            Event::FlushStart => {
                self.pending.clear();
                if let Some(s) = &mut self.session {
                    s.gop_pos = 0; // post-seek output opens on a keyframe
                }
            }
            Event::Eos => {
                let _ = self.drain_pending(ctx)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.pending.clear();
        self.session = None;
        self.display = None;
        self.in_fmt = None;
        self.announced = false;
    }
}
