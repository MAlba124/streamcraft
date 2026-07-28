//! `vaapih265enc` — VA-API hardware H.265/HEVC encode: `video/raw` (NV12/I420)
//! in, H.265 out as **Annex-B** access units (`h265/annexb`) or **HVCC**
//! length-prefixed units with an `hvcC` head buffer (`h265/hvcc`, `mkvmuxn`'s
//! lane) — the family is whichever the link fixated.
//!
//! ## Structure (ITU-T H.265, 09/2023)
//! - **GOP**: IDR (`IDR_W_RADL`) + trailing P (`TRAIL_R`), single reference, no
//!   reordering; POC counts frames since the IDR (§8.3.1), so
//!   `slice_pic_order_cnt_lsb` never wraps within a GOP (`gop` ≤ 256,
//!   `MaxPicOrderCntLsb` = 256).
//! - **One short-term RPS in the SPS** (§7.3.7: one negative pic, Δ=1, used):
//!   P-slice headers then select it with a single flag and carry no per-slice
//!   RPS — the smallest legal P header.
//! - **Coding structure**: Main profile, CTB 32 / min CB 8, TB 4..32, no SAO, no
//!   AMP, no TMVP — the conservative feature set iHD's EncSlice path accepts
//!   everywhere.
//! - **Headers**: VPS (§7.3.2.1), SPS (§7.3.2.2) and PPS (§7.3.2.3) are authored
//!   here and submitted as one packed *sequence* header (VPS+SPS+PPS
//!   concatenated, the layout ffmpeg/gstreamer use); slice segment headers
//!   (§7.3.6.1) are packed per frame.
//! - **Rate control**: `bitrate` = 0 → CQP at `qp`; > 0 → driver VBR (which on
//!   iHD requires `cu_qp_delta_enabled`, mirrored into the PPS we author).

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

use crate::bitwriter::{annexb_nal, BitWriter};
use crate::enc::{self, EncEngine, InFormat};
use crate::ffi;
use crate::h264parse;
use crate::probe;
use crate::va::{self, Buffer as VaBuffer, Display};

const SINK_PAD: PadId = PadId(0);
const SRC_PAD: PadId = PadId(1);

const FAM_ANNEXB: &str = "h265/annexb";
const FAM_HVCC: &str = "h265/hvcc";

/// Bound on frames staged as heap bytes while the output pool is dry (see
/// `vaapih264enc` — the shared-pool deadlock rationale).
const PENDING_MAX: usize = 32;

// H.265 NAL unit types (§7.4.2.2, Table 7-1).
const NAL_TRAIL_R: u8 = 1;
const NAL_IDR_W_RADL: u8 = 19;
const NAL_VPS: u8 = 32;
const NAL_SPS: u8 = 33;
const NAL_PPS: u8 = 34;

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
static SRC_OFFERS: [OfferDesc; 3] = [
    OfferDesc { family: FAM_ANNEXB, fields: &OUT_FIELDS },
    OfferDesc { family: FAM_HVCC, fields: &OUT_FIELDS },
    // The `bytes` escape: a generic byte sink gets the Annex-B stream.
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

static PROPS: [PropDesc; 5] = [
    // Constant-QP quantizer (0..51); used when bitrate == 0.
    PropDesc { name: "qp", allowed: Constraint::Any, live: false },
    // Target bitrate in kbit/s; 0 (default) = CQP mode. **Live** within a
    // rate-controlled session (see vaapih264enc); mode itself is fixed at build.
    PropDesc { name: "bitrate", allowed: Constraint::Any, live: true },
    // Keyframe (IDR) period in frames, 1..=256 (MaxPicOrderCntLsb bound).
    PropDesc { name: "gop", allowed: Constraint::Any, live: false },
    // **Live trigger**: any nonzero write forces the next frame to be an IDR.
    PropDesc { name: "force-keyframe", allowed: Constraint::Any, live: true },
    // HRD decoder-buffer size in ms at the current bitrate (0 = driver default).
    PropDesc { name: "hrd", allowed: Constraint::Any, live: true },
];

// Cold: make_default boxes one element when the registry builds the default, not per-frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "vaapih265enc",
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
    make_default: Some(|| Box::new(VaapiH265Enc::new())),
};

/// H.265 Table A.8 ladder as (general_level_idc = 30×level, MaxLumaPs, MaxLumaSr).
const LEVELS: [(u8, u32, u64); 8] = [
    (60, 122_880, 3_686_400),        // 2
    (63, 245_760, 7_372_800),        // 2.1
    (90, 552_960, 16_588_800),       // 3
    (93, 983_040, 33_177_600),       // 3.1
    (120, 2_228_224, 66_846_720),    // 4
    (123, 2_228_224, 133_693_440),   // 4.1
    (150, 8_912_896, 267_386_880),   // 5
    (153, 8_912_896, 534_773_760),   // 5.1
];

fn pick_level(coded_w: u32, coded_h: u32, fps: (u32, u32)) -> u8 {
    let ps = (coded_w * coded_h) as u64;
    let fps_int = if fps.0 > 0 { fps.0.div_ceil(fps.1) } else { 30 } as u64;
    let sr = ps * fps_int;
    for (idc, max_ps, max_sr) in LEVELS {
        if ps <= max_ps as u64 && sr <= max_sr {
            return idc;
        }
    }
    156 // 5.2
}

struct OutFrame {
    bytes: Vec<u8>,
    pts: Timestamp,
    duration: Timestamp,
    key: bool,
}

struct Session {
    engine: EncEngine,
    /// VPS/SPS/PPS as complete Annex-B NALs.
    vps_nal: Vec<u8>,
    sps_nal: Vec<u8>,
    pps_nal: Vec<u8>,
    packed_mask: u32,
    gop_pos: u32,
    /// CTBs per picture (32×32), for the slice parameter.
    ctu_count: u32,
    level_idc: u8,
}

/// ```text
/// +----------------------+
/// |____                __|
/// |sink| VaapiH265Enc |src|--> h265/annexb | h265/hvcc
/// |^^^^                ^^|
/// +----------------------+
/// ```
pub struct VaapiH265Enc {
    device: Option<PathBuf>,
    display: Option<Display>,
    session: Option<Session>,

    qp: u8,
    bitrate_kbps: u32,
    gop_len: u32,
    /// HRD buffer depth in ms (0 = no HRD misc buffer sent).
    hrd_ms: u32,
    /// A live bitrate/hrd change arrived: resubmit RC/HRD misc with the next frame.
    rc_dirty: bool,
    /// Warned once about a live bitrate write into a CQP session.
    warned_rc_mode: bool,

    in_fmt: Option<InFormat>,
    hvcc: bool,
    /// The link fixated the `bytes` escape — emit Annex-B, announce bare `bytes`.
    bytes_out: bool,
    announced: bool,
    pending: VecDeque<OutFrame>,
    disabled: bool,
}

// SAFETY: `SchedHint::Active` single-thread confinement — the raw `VADisplay`
// never leaves this element's scheduler thread (same argument as the decoder).
unsafe impl Send for VaapiH265Enc {}

impl Default for VaapiH265Enc {
    fn default() -> Self {
        Self::new()
    }
}

impl VaapiH265Enc {
    pub fn new() -> Self {
        VaapiH265Enc {
            device: None,
            display: None,
            session: None,
            qp: 26,
            bitrate_kbps: 0,
            gop_len: 60,
            hrd_ms: 0,
            rc_dirty: false,
            warned_rc_mode: false,
            in_fmt: None,
            hvcc: false,
            bytes_out: false,
            announced: false,
            pending: VecDeque::new(),
            disabled: false,
        }
    }

    /// Constant-QP quantizer (0..51). Chainable; the `qp` prop overrides.
    pub fn with_qp(mut self, qp: u8) -> Self {
        self.qp = qp.min(51);
        self
    }

    /// Keyframe (IDR) period in frames (1 = all-intra). The `gop` prop overrides.
    pub fn with_gop(mut self, gop: u32) -> Self {
        self.gop_len = gop.clamp(1, 256);
        self
    }

    /// Target bitrate in kbit/s (0 = CQP). The `bitrate` prop overrides.
    pub fn with_bitrate(mut self, kbps: u32) -> Self {
        self.bitrate_kbps = kbps;
        self
    }

    /// HRD decoder-buffer depth in milliseconds at the target bitrate (0 = off).
    /// The `hrd` prop overrides.
    pub fn with_hrd_ms(mut self, ms: u32) -> Self {
        self.hrd_ms = ms.min(10_000);
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
                self.warn(ctx, "vaapih265enc: no VA-API device available".into());
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
                    self.warn(ctx, format!("vaapih265enc: cannot open VA display: {e}"));
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
        if !fmt.width.is_multiple_of(2) || !fmt.height.is_multiple_of(2) {
            // §7.4.3.2.1: conformance-window offsets count in SubWidthC/SubHeightC
            // (2-sample) units for 4:2:0.
            self.warn(
                ctx,
                format!(
                    "vaapih265enc: odd dimensions {}x{} unsupported (4:2:0 \
                     conformance window units are 2 samples)",
                    fmt.width, fmt.height
                ),
            );
            return false;
        }

        let profile = ffi::VAProfileHEVCMain;
        let entrypoint = ffi::VAEntrypointEncSlice;
        let rc_mode = if self.bitrate_kbps > 0 { ffi::VA_RC_VBR } else { ffi::VA_RC_CQP };
        let packed_mask =
            va::config_attribute(display, profile, entrypoint, ffi::VAConfigAttribEncPackedHeaders)
                .ok()
                .flatten()
                .unwrap_or(0);

        // Surfaces/coded picture aligned to 16 — a multiple of MinCbSizeY (8) as
        // §7.4.3.2.1 requires; partial CTBs at the edges are legal in HEVC.
        let engine = match EncEngine::new(
            display, profile, entrypoint, rc_mode, fmt.width, fmt.height, 16,
        ) {
            Ok(e) => e,
            Err(e) => {
                self.warn(ctx, format!("vaapih265enc: VA session setup failed: {e}"));
                return false;
            }
        };

        let level_idc = pick_level(engine.coded_w, engine.coded_h, fmt.fps);
        let brc = self.bitrate_kbps > 0;
        let vps_nal = write_vps(level_idc);
        let sps_nal = write_sps(&engine, fmt, level_idc);
        let pps_nal = write_pps(self.qp, brc);
        let ctu_count = engine.coded_w.div_ceil(32) * engine.coded_h.div_ceil(32);

        self.session = Some(Session {
            engine,
            vps_nal,
            sps_nal,
            pps_nal,
            packed_mask,
            gop_pos: 0,
            ctu_count,
            level_idc,
        });
        true
    }

    fn drain_pending(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        while let Some(front) = self.pending.front() {
            let Some(mut buf) = ctx.try_alloc(SRC_PAD) else { return Ok(false) };
            if buf.memory.capacity() < front.bytes.len() {
                return Err(Error::Element {
                    element: ctx.element(),
                    message: format!(
                        "vaapih265enc: encoded frame of {} bytes exceeds pool slot \
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
            let is_idr = s.gop_pos == 0;
            let rc = RcUpdate { dirty: self.rc_dirty, hrd_ms: self.hrd_ms };
            encode_one(display, s, fmt, self.qp, self.bitrate_kbps, self.gop_len, is_idr, rc, data)
                .map(|annexb| (annexb, is_idr))
        };
        let (annexb, is_idr) = match result {
            Ok(v) => v,
            Err(e) => {
                self.warn(ctx, format!("vaapih265enc: encode failed: {e} — frame dropped"));
                return Ok(());
            }
        };
        self.rc_dirty = false; // the RC/HRD update rode this frame's submission

        if !self.announced {
            if self.bytes_out {
                ctx.announce_format(SRC_PAD, "bytes", &[]);
            } else {
                let family = if self.hvcc { FAM_HVCC } else { FAM_ANNEXB };
                ctx.announce_format(
                    SRC_PAD,
                    family,
                    &[
                        (enc::F_WIDTH, ValueDesc::Int(fmt.width as i64)),
                        (enc::F_HEIGHT, ValueDesc::Int(fmt.height as i64)),
                    ],
                );
            }
            self.announced = true;
            if self.hvcc {
                let s = self.session.as_ref().expect("session exists");
                self.pending.push_back(OutFrame {
                    bytes: hvcc_record(&s.vps_nal, &s.sps_nal, &s.pps_nal, s.level_idc),
                    pts: Timestamp::NONE,
                    duration: Timestamp::NONE,
                    key: true,
                });
            }
        }

        let bytes = if self.hvcc { annexb_to_hvcc(&annexb) } else { annexb };
        self.pending.push_back(OutFrame { bytes, pts, duration, key: is_idr });

        let gop_len = self.gop_len.clamp(1, 256);
        let s = self.session.as_mut().expect("session exists");
        s.engine.advance_recon();
        s.gop_pos += 1;
        if s.gop_pos >= gop_len {
            s.gop_pos = 0;
        }
        Ok(())
    }
}

/// A pending rate-control reconfiguration (see `vaapih264enc`'s twin).
#[derive(Clone, Copy)]
struct RcUpdate {
    dirty: bool,
    hrd_ms: u32,
}

/// Upload + submit + sync + drain one frame → assembled Annex-B access unit.
#[allow(clippy::too_many_arguments)]
fn encode_one(
    display: &Display,
    s: &mut Session,
    fmt: InFormat,
    qp: u8,
    bitrate_kbps: u32,
    gop_len: u32,
    is_idr: bool,
    rc: RcUpdate,
    data: &[u8],
) -> va::VaResult<Vec<u8>> {
    s.engine.upload(display, data, fmt.pixfmt)?;

    let engine = &s.engine;
    let ctxva = engine.context();
    let poc = s.gop_pos as i32; // POC == frames since the IDR (§8.3.1, reset at IDR)
    let brc = bitrate_kbps > 0;

    let mut hold: Vec<VaBuffer> = Vec::with_capacity(10);
    let mut ids: Vec<ffi::VABufferID> = Vec::with_capacity(10);

    let coded = VaBuffer::new_empty(display, ctxva, ffi::VAEncCodedBufferType, engine.coded_buf_size())?;

    if is_idr {
        let seq = seq_param(engine, fmt, gop_len, bitrate_kbps, s.level_idc);
        let b = VaBuffer::new_struct(display, ctxva, ffi::VAEncSequenceParameterBufferType, &seq)?;
        ids.push(b.id());
        hold.push(b);
    }

    // RC/HRD misc buffers ride every IDR and any frame carrying a live update
    // (VA accepts misc parameters on any picture; sequence param stays IDR-only).
    if brc && (is_idr || rc.dirty) {
        let mut payloads = vec![rc_misc_bytes(bitrate_kbps), framerate_misc_bytes(fmt.fps)];
        if rc.hrd_ms > 0 {
            payloads.push(hrd_misc_bytes(bitrate_kbps, rc.hrd_ms));
        }
        for payload in payloads {
            let b = VaBuffer::new_typed_bytes(display, ctxva, ffi::VAEncMiscParameterBufferType, &payload)?;
            ids.push(b.id());
            hold.push(b);
        }
    }

    if is_idr {
        if s.packed_mask & ffi::VA_ENC_PACKED_HEADER_SEQUENCE != 0 {
            // One packed sequence header carrying VPS+SPS+PPS (the ffmpeg/gst
            // layout — the driver inserts the bytes verbatim before the frame).
            let mut seq_bytes =
                Vec::with_capacity(s.vps_nal.len() + s.sps_nal.len() + s.pps_nal.len());
            seq_bytes.extend_from_slice(&s.vps_nal);
            seq_bytes.extend_from_slice(&s.sps_nal);
            seq_bytes.extend_from_slice(&s.pps_nal);
            let hdr = ffi::VAEncPackedHeaderParameterBuffer {
                type_: ffi::VAEncPackedHeaderSequence,
                bit_length: seq_bytes.len() as u32 * 8,
                has_emulation_bytes: 1,
                va_reserved: [0; ffi::VA_PADDING_LOW],
            };
            let b = VaBuffer::new_struct(display, ctxva, ffi::VAEncPackedHeaderParameterBufferType, &hdr)?;
            ids.push(b.id());
            hold.push(b);
            let b = VaBuffer::new_typed_bytes(
                display,
                ctxva,
                ffi::VAEncPackedHeaderDataBufferType,
                &seq_bytes,
            )?;
            ids.push(b.id());
            hold.push(b);
        }
    }

    let pic = pic_param(engine, poc, is_idr, qp, brc, coded.id());
    let b = VaBuffer::new_struct(display, ctxva, ffi::VAEncPictureParameterBufferType, &pic)?;
    ids.push(b.id());
    hold.push(b);

    if s.packed_mask & ffi::VA_ENC_PACKED_HEADER_SLICE != 0 {
        let (sh_bytes, sh_bits) = write_slice_header(is_idr, poc);
        let hdr = ffi::VAEncPackedHeaderParameterBuffer {
            type_: ffi::VAEncPackedHeaderSlice,
            bit_length: sh_bits,
            has_emulation_bytes: 0,
            va_reserved: [0; ffi::VA_PADDING_LOW],
        };
        let b = VaBuffer::new_struct(display, ctxva, ffi::VAEncPackedHeaderParameterBufferType, &hdr)?;
        ids.push(b.id());
        hold.push(b);
        let b = VaBuffer::new_typed_bytes(display, ctxva, ffi::VAEncPackedHeaderDataBufferType, &sh_bytes)?;
        ids.push(b.id());
        hold.push(b);
    }

    let slice = slice_param(engine, s.ctu_count, poc, is_idr);
    let b = VaBuffer::new_struct(display, ctxva, ffi::VAEncSliceParameterBufferType, &slice)?;
    ids.push(b.id());
    hold.push(b);

    engine.submit(display, &ids)?;
    drop(hold);

    let mut coded_bytes = Vec::new();
    coded.read_coded(&mut coded_bytes)?;

    let packed_seq = s.packed_mask & ffi::VA_ENC_PACKED_HEADER_SEQUENCE != 0;
    Ok(if is_idr && !packed_seq {
        let mut v = Vec::with_capacity(
            s.vps_nal.len() + s.sps_nal.len() + s.pps_nal.len() + coded_bytes.len(),
        );
        v.extend_from_slice(&s.vps_nal);
        v.extend_from_slice(&s.sps_nal);
        v.extend_from_slice(&s.pps_nal);
        v.extend_from_slice(&coded_bytes);
        v
    } else {
        coded_bytes
    })
}

// --- VA parameter builders ------------------------------------------------------------

fn seq_param(
    engine: &EncEngine,
    fmt: InFormat,
    gop: u32,
    bitrate_kbps: u32,
    level_idc: u8,
) -> ffi::VAEncSequenceParameterBufferHEVC {
    // seq_fields bit order (va_enc_hevc.h): chroma_format_idc:2,
    // separate_colour_plane:1, bit_depth_luma_minus8:3, bit_depth_chroma_minus8:3,
    // scaling_list:1, strong_intra_smoothing:1, amp:1, sao:1, pcm:1,
    // pcm_loop_filter_disabled:1, temporal_mvp:1, low_delay_seq:1, hierachical:1.
    let seq_fields: u32 = 1 // chroma_format_idc = 1 (4:2:0)
        | 1 << 16; // low_delay_seq (decode order == display order)
    ffi::VAEncSequenceParameterBufferHEVC {
        general_profile_idc: 1, // Main
        general_level_idc: level_idc,
        general_tier_flag: 0,
        intra_period: gop,
        intra_idr_period: gop,
        ip_period: 1,
        bits_per_second: bitrate_kbps * 1000,
        pic_width_in_luma_samples: engine.coded_w as u16,
        pic_height_in_luma_samples: engine.coded_h as u16,
        seq_fields,
        log2_min_luma_coding_block_size_minus3: 0, // MinCb 8
        log2_diff_max_min_luma_coding_block_size: 2, // CTB 32
        log2_min_transform_block_size_minus2: 0,   // min TB 4
        log2_diff_max_min_transform_block_size: 3, // max TB 32
        max_transform_hierarchy_depth_inter: 2,
        max_transform_hierarchy_depth_intra: 2,
        pcm_sample_bit_depth_luma_minus1: 0,
        pcm_sample_bit_depth_chroma_minus1: 0,
        log2_min_pcm_luma_coding_block_size_minus3: 0,
        log2_max_pcm_luma_coding_block_size_minus3: 0,
        vui_parameters_present_flag: 0,
        vui_fields: 0,
        aspect_ratio_idc: 0,
        sar_width: 0,
        sar_height: 0,
        vui_num_units_in_tick: if fmt.fps.0 > 0 { fmt.fps.1 } else { 1 },
        vui_time_scale: if fmt.fps.0 > 0 { fmt.fps.0 } else { 30 },
        min_spatial_segmentation_idc: 0,
        max_bytes_per_pic_denom: 0,
        max_bits_per_min_cu_denom: 0,
        scc_fields: 0,
        va_reserved: [0; ffi::VA_PADDING_MEDIUM - 1],
    }
}

fn hevc_pic(surface: ffi::VASurfaceID, poc: i32, flags: u32) -> ffi::VAPictureHEVC {
    ffi::VAPictureHEVC {
        picture_id: surface,
        pic_order_cnt: poc,
        flags,
        va_reserved: [0; ffi::VA_PADDING_LOW],
    }
}

fn pic_param(
    engine: &EncEngine,
    poc: i32,
    is_idr: bool,
    qp: u8,
    brc: bool,
    coded_buf: ffi::VABufferID,
) -> ffi::VAEncPictureParameterBufferHEVC {
    let mut refs = [ffi::VAPictureHEVC::invalid(); 15];
    if !is_idr {
        refs[0] = hevc_pic(engine.recon_ref(), poc - 1, ffi::VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE);
    }
    // pic_fields bit order (va_enc_hevc.h): idr_pic_flag:1, coding_type:3,
    // reference_pic_flag:1, dependent_slice_segments_enabled:1,
    // sign_data_hiding_enabled:1, constrained_intra_pred:1, transform_skip:1,
    // cu_qp_delta_enabled:1, weighted_pred:1, weighted_bipred:1,
    // transquant_bypass:1, tiles_enabled:1, entropy_coding_sync:1, …
    let coding_type: u32 = if is_idr { 1 } else { 2 }; // 1 = I, 2 = P
    let pic_fields: u32 = (is_idr as u32)
        | coding_type << 1
        | 1 << 4 // reference_pic_flag
        | (brc as u32) << 9; // cu_qp_delta_enabled (iHD BRC requirement)
    ffi::VAEncPictureParameterBufferHEVC {
        decoded_curr_pic: hevc_pic(engine.recon_cur(), poc, 0),
        reference_frames: refs,
        coded_buf,
        collocated_ref_pic_index: 0xFF, // no TMVP
        last_picture: 0,
        pic_init_qp: qp,
        diff_cu_qp_delta_depth: if brc { 2 } else { 0 },
        pps_cb_qp_offset: 0,
        pps_cr_qp_offset: 0,
        num_tile_columns_minus1: 0,
        num_tile_rows_minus1: 0,
        column_width_minus1: [0; 19],
        row_height_minus1: [0; 21],
        log2_parallel_merge_level_minus2: 0,
        ctu_max_bitsize_allowed: 0,
        num_ref_idx_l0_default_active_minus1: 0,
        num_ref_idx_l1_default_active_minus1: 0,
        slice_pic_parameter_set_id: 0,
        nal_unit_type: if is_idr { NAL_IDR_W_RADL } else { NAL_TRAIL_R },
        pic_fields,
        hierarchical_level_plus1: 0,
        va_byte_reserved: 0,
        scc_fields: 0,
        va_reserved: [0; ffi::VA_PADDING_HIGH - 1],
    }
}

fn slice_param(
    engine: &EncEngine,
    ctu_count: u32,
    poc: i32,
    is_idr: bool,
) -> ffi::VAEncSliceParameterBufferHEVC {
    let mut l0 = [ffi::VAPictureHEVC::invalid(); 15];
    if !is_idr {
        l0[0] = hevc_pic(engine.recon_ref(), poc - 1, ffi::VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE);
    }
    // slice_fields bit order (va_enc_hevc.h): last_slice_of_pic:1,
    // dependent_slice_segment:1, colour_plane_id:2, slice_temporal_mvp_enabled:1,
    // slice_sao_luma:1, slice_sao_chroma:1, num_ref_idx_active_override:1, …
    let slice_fields: u32 = 1; // last_slice_of_pic (single slice per picture)
    ffi::VAEncSliceParameterBufferHEVC {
        slice_segment_address: 0,
        num_ctu_in_slice: ctu_count,
        slice_type: if is_idr { 2 } else { 1 }, // 2 = I, 1 = P (§7.4.7.1)
        slice_pic_parameter_set_id: 0,
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        ref_pic_list0: l0,
        ref_pic_list1: [ffi::VAPictureHEVC::invalid(); 15],
        luma_log2_weight_denom: 0,
        delta_chroma_log2_weight_denom: 0,
        delta_luma_weight_l0: [0; 15],
        luma_offset_l0: [0; 15],
        delta_chroma_weight_l0: [[0; 2]; 15],
        chroma_offset_l0: [[0; 2]; 15],
        delta_luma_weight_l1: [0; 15],
        luma_offset_l1: [0; 15],
        delta_chroma_weight_l1: [[0; 2]; 15],
        chroma_offset_l1: [[0; 2]; 15],
        max_num_merge_cand: 5,
        slice_qp_delta: 0,
        slice_cb_qp_offset: 0,
        slice_cr_qp_offset: 0,
        slice_beta_offset_div2: 0,
        slice_tc_offset_div2: 0,
        slice_fields,
        pred_weight_table_bit_offset: 0,
        pred_weight_table_bit_length: 0,
        va_reserved: [0; ffi::VA_PADDING_MEDIUM - 2],
    }
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
        min_qp: 1,
        basic_unit_size: 0,
        rc_flags: 0,
        ICQ_quality_factor: 0,
        max_qp: 51,
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

/// HRD buffer model sized as `hrd_ms` milliseconds at the target bitrate,
/// starting half full (the conventional low-latency initial fullness).
fn hrd_misc_bytes(bitrate_kbps: u32, hrd_ms: u32) -> Vec<u8> {
    let bits = (bitrate_kbps as u64 * 1000).saturating_mul(hrd_ms as u64) / 1000;
    let buffer_size = bits.min(u32::MAX as u64) as u32;
    let hrd = ffi::VAEncMiscParameterHRD {
        initial_buffer_fullness: buffer_size / 2,
        buffer_size,
        va_reserved: [0; ffi::VA_PADDING_LOW],
    };
    misc_bytes(ffi::VAEncMiscParameterTypeHRD, &hrd)
}

// --- Bitstream authoring (ITU-T H.265 §7.3) -------------------------------------------

/// The 2-byte H.265 NAL header (§7.3.1.2): type, layer 0, temporal id 0.
fn nal_header(nal_type: u8) -> [u8; 2] {
    [nal_type << 1, 1]
}

/// `profile_tier_level(1, 0)` (§7.3.3) — Main profile, Main tier, no sub-layers.
fn write_ptl(w: &mut BitWriter, level_idc: u8) {
    w.u(0, 2); // general_profile_space
    w.flag(false); // general_tier_flag
    w.u(1, 5); // general_profile_idc = Main
    // Compatibility flags: Main (bit 1) + Main 10 (bit 2) — a Main stream is
    // decodable by Main 10 decoders (A.3.2).
    w.u(0x6000_0000, 32);
    w.flag(true); // general_progressive_source_flag
    w.flag(false); // general_interlaced_source_flag
    w.flag(false); // general_non_packed_constraint_flag
    w.flag(true); // general_frame_only_constraint_flag
    w.u(0, 22); // general_reserved_zero_43bits (22 + 21)
    w.u(0, 21);
    w.flag(false); // general_inbld_flag (reserved)
    w.u(level_idc as u32, 8); // general_level_idc
}

/// VPS (§7.3.2.1) as a complete Annex-B NAL.
fn write_vps(level_idc: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.u(0, 4); // vps_video_parameter_set_id
    w.flag(true); // vps_base_layer_internal_flag
    w.flag(true); // vps_base_layer_available_flag
    w.u(0, 6); // vps_max_layers_minus1
    w.u(0, 3); // vps_max_sub_layers_minus1
    w.flag(true); // vps_temporal_id_nesting_flag
    w.u(0xFFFF, 16); // vps_reserved_0xffff_16bits
    write_ptl(&mut w, level_idc);
    w.flag(false); // vps_sub_layer_ordering_info_present_flag
    w.ue(1); // vps_max_dec_pic_buffering_minus1[0]
    w.ue(0); // vps_max_num_reorder_pics[0]
    w.ue(0); // vps_max_latency_increase_plus1[0]
    w.u(0, 6); // vps_max_layer_id
    w.ue(0); // vps_num_layer_sets_minus1
    w.flag(false); // vps_timing_info_present_flag
    w.flag(false); // vps_extension_flag
    w.rbsp_trailing_bits();
    annexb_nal(&nal_header(NAL_VPS), &w.into_bytes())
}

/// SPS (§7.3.2.2) as a complete Annex-B NAL. One short-term RPS (Δ=1, used) so P
/// slice headers select it with a flag.
fn write_sps(engine: &EncEngine, fmt: InFormat, level_idc: u8) -> Vec<u8> {
    // Conformance-window offsets in SubWidthC/SubHeightC (2-sample) units
    // (§7.4.3.2.1).
    let crop_r = (engine.coded_w - engine.width) / 2;
    let crop_b = (engine.coded_h - engine.height) / 2;
    let mut w = BitWriter::new();
    w.u(0, 4); // sps_video_parameter_set_id
    w.u(0, 3); // sps_max_sub_layers_minus1
    w.flag(true); // sps_temporal_id_nesting_flag
    write_ptl(&mut w, level_idc);
    w.ue(0); // sps_seq_parameter_set_id
    w.ue(1); // chroma_format_idc (4:2:0)
    w.ue(engine.coded_w); // pic_width_in_luma_samples
    w.ue(engine.coded_h); // pic_height_in_luma_samples
    let cropping = crop_r > 0 || crop_b > 0;
    w.flag(cropping); // conformance_window_flag
    if cropping {
        w.ue(0); // conf_win_left_offset
        w.ue(crop_r); // conf_win_right_offset (already in SubWidthC units)
        w.ue(0); // conf_win_top_offset
        w.ue(crop_b); // conf_win_bottom_offset (already in SubHeightC units)
    }
    w.ue(0); // bit_depth_luma_minus8
    w.ue(0); // bit_depth_chroma_minus8
    w.ue(4); // log2_max_pic_order_cnt_lsb_minus4 → MaxPicOrderCntLsb 256
    w.flag(false); // sps_sub_layer_ordering_info_present_flag
    w.ue(1); // sps_max_dec_pic_buffering_minus1
    w.ue(0); // sps_max_num_reorder_pics
    w.ue(0); // sps_max_latency_increase_plus1
    w.ue(0); // log2_min_luma_coding_block_size_minus3 (MinCb 8)
    w.ue(2); // log2_diff_max_min_luma_coding_block_size (CTB 32)
    w.ue(0); // log2_min_luma_transform_block_size_minus2 (min TB 4)
    w.ue(3); // log2_diff_max_min_luma_transform_block_size (max TB 32)
    w.ue(2); // max_transform_hierarchy_depth_inter
    w.ue(2); // max_transform_hierarchy_depth_intra
    w.flag(false); // scaling_list_enabled_flag
    w.flag(false); // amp_enabled_flag
    w.flag(false); // sample_adaptive_offset_enabled_flag
    w.flag(false); // pcm_enabled_flag
    w.ue(1); // num_short_term_ref_pic_sets
    // st_ref_pic_set(0) (§7.3.7): one negative pic at Δ=1, used by curr.
    w.ue(1); // num_negative_pics
    w.ue(0); // num_positive_pics
    w.ue(0); // delta_poc_s0_minus1[0] → ΔPOC = 1
    w.flag(true); // used_by_curr_pic_s0_flag[0]
    w.flag(false); // long_term_ref_pics_present_flag
    w.flag(false); // sps_temporal_mvp_enabled_flag
    w.flag(false); // strong_intra_smoothing_enabled_flag
    // VUI with timing when fps is known (§E.2.1: one tick per frame in H.265).
    let have_fps = fmt.fps.0 > 0;
    w.flag(have_fps); // vui_parameters_present_flag
    if have_fps {
        w.flag(false); // aspect_ratio_info_present_flag
        w.flag(false); // overscan_info_present_flag
        w.flag(false); // video_signal_type_present_flag
        w.flag(false); // chroma_loc_info_present_flag
        w.flag(false); // neutral_chroma_indication_flag
        w.flag(false); // field_seq_flag
        w.flag(false); // frame_field_info_present_flag
        w.flag(false); // default_display_window_flag
        w.flag(true); // vui_timing_info_present_flag
        w.u(fmt.fps.1, 32); // vui_num_units_in_tick
        w.u(fmt.fps.0, 32); // vui_time_scale
        w.flag(false); // vui_poc_proportional_to_timing_flag
        w.flag(false); // vui_hrd_parameters_present_flag
        w.flag(false); // bitstream_restriction_flag
    }
    w.flag(false); // sps_extension_present_flag
    w.rbsp_trailing_bits();
    annexb_nal(&nal_header(NAL_SPS), &w.into_bytes())
}

/// PPS (§7.3.2.3) as a complete Annex-B NAL.
fn write_pps(qp: u8, brc: bool) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.ue(0); // pps_pic_parameter_set_id
    w.ue(0); // pps_seq_parameter_set_id
    w.flag(false); // dependent_slice_segments_enabled_flag
    w.flag(false); // output_flag_present_flag
    w.u(0, 3); // num_extra_slice_header_bits
    w.flag(false); // sign_data_hiding_enabled_flag
    w.flag(false); // cabac_init_present_flag
    w.ue(0); // num_ref_idx_l0_default_active_minus1
    w.ue(0); // num_ref_idx_l1_default_active_minus1
    w.se(qp as i32 - 26); // init_qp_minus26
    w.flag(false); // constrained_intra_pred_flag
    w.flag(false); // transform_skip_enabled_flag
    w.flag(brc); // cu_qp_delta_enabled_flag (driver BRC patches CU QPs)
    if brc {
        w.ue(2); // diff_cu_qp_delta_depth
    }
    w.se(0); // pps_cb_qp_offset
    w.se(0); // pps_cr_qp_offset
    w.flag(false); // pps_slice_chroma_qp_offsets_present_flag
    w.flag(false); // weighted_pred_flag
    w.flag(false); // weighted_bipred_flag
    w.flag(false); // transquant_bypass_enabled_flag
    w.flag(false); // tiles_enabled_flag
    w.flag(false); // entropy_coding_sync_enabled_flag
    w.flag(false); // pps_loop_filter_across_slices_enabled_flag
    w.flag(false); // deblocking_filter_control_present_flag
    w.flag(false); // pps_scaling_list_data_present_flag
    w.flag(false); // lists_modification_present_flag
    w.ue(0); // log2_parallel_merge_level_minus2
    w.flag(false); // slice_segment_header_extension_present_flag
    w.flag(false); // pps_extension_present_flag
    w.rbsp_trailing_bits();
    annexb_nal(&nal_header(NAL_PPS), &w.into_bytes())
}

/// Slice segment header bits (§7.3.6.1) for the packed slice header. Unlike
/// H.264 — where CABAC alignment opens the slice *data* (§7.3.4) — H.265 ends
/// the header proper with `byte_alignment()` (§7.3.6.1), so the packed header
/// carries it and `bit_length` is byte-aligned; the driver's entropy coder
/// starts on the boundary. Returns (bytes incl. start code + NAL header, bits).
fn write_slice_header(is_idr: bool, poc: i32) -> (Vec<u8>, u32) {
    let mut w = BitWriter::new();
    let nal_type = if is_idr { NAL_IDR_W_RADL } else { NAL_TRAIL_R };
    for b in [0u8, 0, 0, 1, nal_type << 1, 1] {
        w.u(b as u32, 8);
    }
    w.flag(true); // first_slice_segment_in_pic_flag
    if is_idr {
        w.flag(false); // no_output_of_prior_pics_flag (IRAP NAL types 16..23)
    }
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(if is_idr { 2 } else { 1 }); // slice_type (I / P)
    if !is_idr {
        w.u((poc & 0xFF) as u32, 8); // slice_pic_order_cnt_lsb (log2 max = 8)
        w.flag(true); // short_term_ref_pic_set_sps_flag (the SPS's single RPS)
        // num_short_term_ref_pic_sets == 1 → no index bits.
        w.flag(false); // num_ref_idx_active_override_flag
        w.ue(0); // five_minus_max_num_merge_cand (MaxNumMergeCand = 5)
    }
    w.se(0); // slice_qp_delta
    // PPS: no chroma offset presence, deblocking control absent, loop filter
    // across slices disabled, no tiles/entropy sync → byte_alignment() (§7.3.6.1:
    // alignment_bit_equal_to_one, then zeros to the byte boundary) ends the header.
    w.rbsp_trailing_bits(); // identical bit pattern to byte_alignment()
    let bits = w.bit_len();
    (w.into_bytes(), bits)
}

// --- Output framing helpers -----------------------------------------------------------

/// H.265 NAL type from a raw NAL's first header byte (§7.3.1.2).
fn h265_nal_type(raw: &[u8]) -> u8 {
    (raw[0] >> 1) & 0x3F
}

/// Build the `hvcC` record (ISO/IEC 14496-15 §8.3.3.1) from our parameter sets.
// Cold: codec-config record assembled once at announce, not per-frame.
#[allow(clippy::disallowed_methods)]
fn hvcc_record(vps_nal: &[u8], sps_nal: &[u8], pps_nal: &[u8], level_idc: u8) -> Vec<u8> {
    let vps = &vps_nal[4..];
    let sps = &sps_nal[4..];
    let pps = &pps_nal[4..];
    let mut v = Vec::with_capacity(23 + 15 + vps.len() + sps.len() + pps.len());
    v.push(1); // configurationVersion
    v.push(1); // profile_space(2)=0 | tier(1)=0 | profile_idc(5)=1 (Main)
    v.extend_from_slice(&0x6000_0000u32.to_be_bytes()); // compatibility flags
    // constraint indicator flags (48 bits): progressive + frame-only.
    v.extend_from_slice(&[0x90, 0, 0, 0, 0, 0]);
    v.push(level_idc); // general_level_idc
    v.extend_from_slice(&[0xF0, 0x00]); // reserved(4)=1111 + min_spatial_segmentation_idc
    v.push(0xFC); // reserved(6) + parallelismType = 0
    v.push(0xFD); // reserved(6) + chromaFormat = 1 (4:2:0)
    v.push(0xF8); // reserved(5) + bitDepthLumaMinus8 = 0
    v.push(0xF8); // reserved(5) + bitDepthChromaMinus8 = 0
    v.extend_from_slice(&[0, 0]); // avgFrameRate = unspecified
    // constantFrameRate(2)=0 | numTemporalLayers(3)=1 | temporalIdNested(1)=1 |
    // lengthSizeMinusOne(2)=3.
    v.push(0x0F);
    v.push(3); // numOfArrays: VPS, SPS, PPS
    for (nal_type, nal) in [(NAL_VPS, vps), (NAL_SPS, sps), (NAL_PPS, pps)] {
        v.push(0x80 | nal_type); // array_completeness=1 + NAL_unit_type
        v.extend_from_slice(&1u16.to_be_bytes()); // numNalus
        v.extend_from_slice(&(nal.len() as u16).to_be_bytes());
        v.extend_from_slice(nal);
    }
    v
}

/// Convert an Annex-B access unit into HVCC framing (4-byte lengths, parameter
/// sets dropped — they live in the `hvcC` record). Start-code scanning is
/// codec-agnostic, so the H.264 splitter does the framing; only the NAL-type
/// read differs.
fn annexb_to_hvcc(au: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(au.len() + 8);
    for nal in h264parse::split_nals(au) {
        if matches!(h265_nal_type(nal.raw), NAL_VPS | NAL_SPS | NAL_PPS) {
            continue;
        }
        out.extend_from_slice(&(nal.raw.len() as u32).to_be_bytes());
        out.extend_from_slice(nal.raw);
    }
    out
}

impl Element for VaapiH265Enc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        if let Some(Value::Int(v)) = ctx.prop("qp") {
            self.qp = v.clamp(0, 51) as u8;
        }
        if let Some(Value::Int(v)) = ctx.prop("bitrate") {
            self.bitrate_kbps = v.clamp(0, 500_000) as u32;
        }
        if let Some(Value::Int(v)) = ctx.prop("gop") {
            self.gop_len = v.clamp(1, 256) as u32;
        }
        if let Some(Value::Int(v)) = ctx.prop("hrd") {
            self.hrd_ms = v.clamp(0, 10_000) as u32;
        }
        let family = ctx
            .negotiated(SRC_PAD)
            .and_then(|f| ctx.family_name(f.family))
            .unwrap_or(FAM_ANNEXB);
        self.hvcc = family == FAM_HVCC;
        self.bytes_out = family == "bytes";
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
                self.warn(ctx, "vaapih265enc: raw frame before any format — dropped".into());
                continue;
            };
            if inbuf.memory.data().len() < fmt.frame_bytes() {
                self.warn(
                    ctx,
                    format!(
                        "vaapih265enc: short frame ({} < {} bytes) — dropped",
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
            // Live congestion-control knobs (see vaapih264enc for the rationale).
            Event::PropChanged { name: "bitrate", value: Value::Int(v) } => {
                let v = *v;
                if self.session.is_some() && self.bitrate_kbps == 0 {
                    if !self.warned_rc_mode {
                        self.warned_rc_mode = true;
                        self.warn(
                            ctx,
                            "vaapih265enc: live bitrate change ignored — session is CQP \
                             (start with bitrate > 0 for a rate-controlled session)"
                                .into(),
                        );
                    }
                } else if v > 0 {
                    self.bitrate_kbps = v.clamp(1, 500_000) as u32;
                    self.rc_dirty = true;
                } else {
                    self.warn(ctx, "vaapih265enc: live bitrate 0 ignored (cannot switch to CQP)".into());
                }
            }
            Event::PropChanged { name: "hrd", value: Value::Int(v) } => {
                self.hrd_ms = (*v).clamp(0, 10_000) as u32;
                self.rc_dirty = true;
            }
            Event::PropChanged { name: "force-keyframe", value: Value::Int(v) } if *v != 0 => {
                if let Some(s) = &mut self.session {
                    s.gop_pos = 0; // next frame is an IDR (POC resets with it)
                }
            }
            Event::FlushStart => {
                self.pending.clear();
                if let Some(s) = &mut self.session {
                    s.gop_pos = 0; // post-seek output opens on an IDR
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_ladder_picks_sane_levels() {
        assert_eq!(pick_level(1920, 1088, (30, 1)), 120, "1080p30 → L4");
        assert_eq!(pick_level(1920, 1088, (60, 1)), 123, "1080p60 → L4.1");
        assert_eq!(pick_level(1280, 720, (30, 1)), 93, "720p30 → L3.1");
        assert_eq!(pick_level(320, 240, (30, 1)), 60, "tiny → L2");
    }

    #[test]
    fn hvcc_record_shape() {
        let vps = [0u8, 0, 0, 1, 0x40, 0x01, 0xAA];
        let sps = [0u8, 0, 0, 1, 0x42, 0x01, 0xBB, 0xCC];
        let pps = [0u8, 0, 0, 1, 0x44, 0x01, 0xDD];
        let rec = hvcc_record(&vps, &sps, &pps, 120);
        assert_eq!(rec[0], 1, "configurationVersion");
        assert_eq!(rec[1], 1, "Main profile");
        assert_eq!(rec[12], 120, "level (after 1+1+4+6 header bytes)");
        assert_eq!(rec[22], 3, "three arrays");
        assert_eq!(rec[23], 0x80 | 32, "VPS array header");
        let vps_len = u16::from_be_bytes([rec[26], rec[27]]) as usize;
        assert_eq!(vps_len, 3, "VPS NAL length");
        assert_eq!(&rec[28..28 + vps_len], &vps[4..]);
    }

    #[test]
    fn h265_nal_types_read_from_two_byte_header() {
        assert_eq!(h265_nal_type(&[32 << 1, 1]), 32);
        assert_eq!(h265_nal_type(&[19 << 1, 1]), 19);
        assert_eq!(h265_nal_type(&[1 << 1, 1]), 1);
    }
}
