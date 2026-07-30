//! `vaapih265dec` — the VA-API hardware H.265 / HEVC decode element. One Annex-B
//! access unit per buffer arrives on the sink (`h265/annexb`, the demuxer contract);
//! planar **I420** frames leave on the `video/raw` src pad, one per buffer — matching
//! the software `h265dec` (`pf-h265`) so the sdl3 sink negotiates identically for
//! both decoders.
//!
//! Like [`crate::h264dec`], VA-API is a *slice-level* API: the driver runs CABAC and
//! reconstruction, but this element parses the parameter sets and slice segment
//! headers ([`crate::h265parse`], ITU-T H.265), maintains the decoded-picture buffer
//! (DPB) and reference lists itself, fills the `VAPictureParameterBufferHEVC` /
//! `VASliceParameterBufferHEVC` structs, and drives `vaBeginPicture` /
//! `vaRenderPicture` / `vaEndPicture`. The GPU decodes; the host does bitstream
//! framing and reference bookkeeping.
//!
//! **Reference management (§8.3.2–§8.3.4).** HEVC uses POC + Reference Picture Sets,
//! not H.264's sliding window / MMCO. For each picture the slice's short-term RPS
//! (deltas from the current POC) and long-term references select which DPB pictures
//! stay live; every DPB picture in *no* current RPS is unmarked and its surface
//! recycled. `ReferenceFrames[]` and the per-slice `RefPicList[2][15]` are built from
//! those sets, tagged `RPS_ST_CURR_BEFORE` / `RPS_ST_CURR_AFTER` / `RPS_LT_CURR` so
//! the driver forms its own lists (§8.3.4).
//!
//! **POC boundary.** Main profile, 8-bit, 4:2:0, progressive — the two BluRay-class
//! x265 movies this exists for. Main10, 4:2:2/4:4:4, range/screen-content extensions,
//! scaling lists, PCM, tiles/WPP, and dependent slice segments are recognized and
//! warn-dropped (a loud one-time warning per class), never mis-decoded. Reorder pts
//! uses a feed-order FIFO (same documented caveat as the software decoder).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

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
use profluens_core::log;
use profluens_core::log::Level;
use profluens_core::time::Timestamp;

use crate::ffi;
use crate::gpuframe::{
    self, GpuFrame, GpuFrameChannel, GpuFrameHeader, FrameToken, FAMILY_VIDEO_GPU, F_LAYOUT,
    LAYOUT_NV12_DMABUF,
};
use crate::h265parse::{self, Pps, SliceHeader, SliceType, Sps};
use crate::probe;
use crate::va::{Buffer as VaBuffer, Config, Context, Display, ExportedSurface, MappedImage, Surfaces};

// `video/raw` family/field/value names, kept as literals (core-only, like the
// software `h265dec`) — the pipeline interns by string so they line up with peers.
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
// I420 (planar Y|Cb|Cr) — matches the software `h265dec` src, so the sdl3 sink
// negotiates the same format for the hardware and software decoders. VA surfaces are
// NV12; the readback de-interleaves the CbCr plane into planar Cb/Cr (`emit_pending`).
const PIXFMT_I420: &str = "i420";

const SRC_PAD: PadId = PadId(1);

static PIXFMT_VALUES: [ValueDesc; 1] = [ValueDesc::Id(PIXFMT_I420)];

// Broad `video/raw` template: any dimensions, I420 only. Concrete width/height are
// announced at runtime from the SPS, so the src pad is `dynamic`.
static SRC_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
];
// Both offers on the src template: a readback (`new`) decoder announces `video/raw`, a
// zero-copy (`new_zerocopy`) decoder announces `video/gpu` (the shared
// [`gpuframe::GPU_OFFER`] — NV12 DMA-BUF). The static template lists both; the runtime
// `announce_format` picks the one matching the element's mode, and negotiation with the
// explicitly-wired sink resolves the intersection (a `video/raw` sink cannot intersect the
// `video/gpu` offer, so a software peer is never mis-linked to a GPU-only frame).
static SRC_OFFERS: [OfferDesc; 2] =
    [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }, gpuframe::GPU_OFFER];
// One HEVC Annex-B access unit per buffer, same sink family as the software decoder
// (`h265/annexb`). The demuxer reframes hvcC → Annex-B upstream.
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
        dynamic: true,
        validate: None,
    },
];

// Cold: make_default boxes one element when the registry builds the default, not per-frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "vaapih265dec",
    pads: &PADS,
    props: &[],
    // Active: a hardware decode round-trip (submit + sync + readback) is well beyond
    // the inline passive budget, and its own thread pipelines against demux + display.
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(VaapiH265Dec::new())),
};

/// A DPB entry: a decoded picture held for prediction and/or output (§8.3.2).
#[derive(Clone)]
struct DpbEntry {
    surface: ffi::VASurfaceID,
    poc: i32,
    long_term: bool,
    /// Membership in the *current* picture's RPS, recomputed per picture — decides the
    /// `VA_PICTURE_HEVC_RPS_*` flag the driver sees for this reference.
    rps_flag: u32,
}

/// A decoded surface awaiting readback + emit — the backpressure carry (bounded to
/// one); the output queue drains through it in POC order.
struct PendingOut {
    surface: ffi::VASurfaceID,
    pts: Timestamp,
    duration: Timestamp,
    /// Cropped display dimensions.
    width: u32,
    height: u32,
    /// Crop origin in the coded surface.
    crop_x: u32,
    crop_y: u32,
    /// Coded surface dimensions (readback maps at this size).
    coded_w: u32,
    coded_h: u32,
}

/// Lazily-built VA-API state: created on the first usable SPS, torn down + rebuilt on
/// a dimension change.
struct VaState {
    _config: Config,
    context: Context,
    // Held for ownership only: the surface set must outlive the context (which renders
    // into it). Field order = drop order — `context` drops first, then these surfaces,
    // the order libva requires.
    _surfaces: Surfaces,
    /// Free surface pool (ids not currently a DPB ref or a pending output).
    free: Vec<ffi::VASurfaceID>,
    /// Conformance-cropped *display* dimensions (announced + emitted, matching the
    /// software decoder and the ffmpeg reference).
    width: u32,
    height: u32,
    /// Cropped-region top-left origin in the coded surface (§7.4.3.2.1). Almost always
    /// (0, 0), but a stream may crop from the top/left.
    crop_x: u32,
    crop_y: u32,
    /// Coded (CTB-aligned) surface dimensions — the readback maps at this size.
    coded_w: u32,
    coded_h: u32,
}

/// The VA-API H.265 decode element.
pub struct VaapiH265Dec {
    device: Option<PathBuf>,
    display: Option<Display>,
    va: Option<VaState>,

    // Parameter-set caches (by id). HEVC allows 16 SPS / 64 PPS (§7.4.3.2/.3).
    sps: [Option<Sps>; 16],
    pps: [Option<Pps>; 64],

    // DPB + POC.
    dpb: Vec<DpbEntry>,
    poc: h265parse::PocState,
    /// Pictures decoded but not yet output, in (gop, POC) order.
    output_queue: VecDeque<PendingOut>,
    /// surface → (gop, POC), for ordering the output queue (§C.5-style bumping). POC
    /// restarts at each IRAP-with-NoRaslOutput, so the gop counter keeps the key
    /// globally monotonic across a cut (the h264dec output-order lesson).
    output_key: std::collections::HashMap<ffi::VASurfaceID, (u32, i32)>,
    /// Monotonic coded-video-sequence counter, bumped at each IRAP boundary.
    gop: u32,
    /// The one picture currently in readback (backpressure carry).
    pending: Option<PendingOut>,

    /// After a CRA/BLA random-access point, the following RASL pictures reference
    /// pictures that precede the RAP and were never decoded — skip them (§8.1.3).
    skip_rasl: bool,
    /// True until the first IRAP is seen — the decoder must start at a clean RAP.
    awaiting_irap: bool,

    announced: bool,
    dims: Option<(u32, u32)>,
    alloc_stalled: bool,
    /// Set once a fatal setup error was reported, to avoid a warning storm.
    disabled: bool,
    /// One-time warning latches for each refused coding-tool class (no per-frame spam).
    warned_unsupported: bool,
    warned_multi_slice: bool,

    // --- Zero-copy (DMA-BUF export) mode -------------------------------------------------
    /// When `Some`, `emit_pending` exports the decoded surface as DMA-BUF(s) and pushes a
    /// [`GpuFrame`] into this channel instead of reading NV12 back to the CPU. `None` is the
    /// default readback mode (unchanged for the CLI and software peers). Set by
    /// [`new_zerocopy`](Self::new_zerocopy).
    zerocopy: Option<Arc<GpuFrameChannel>>,
    /// Monotonic token stamped on each exported frame, pairing the in-band `video/gpu`
    /// buffer with its out-of-band [`GpuFrame`].
    next_token: FrameToken,
    /// Surfaces exported to the GUI and not yet released (presented). Keyed by surface id →
    /// its token. A surface here is **in flight**: its dma-buf is being sampled, so it must
    /// NOT return to the free pool (a new decode into it would tear the displayed picture).
    /// The sink's `release(token)` (drained in `process`) clears the entry and frees it.
    in_flight: std::collections::HashMap<ffi::VASurfaceID, FrameToken>,
    /// One-time latch: the export path failed (driver refused PRIME_2) — fall back to
    /// readback for the rest of the run rather than spamming warnings or dropping frames.
    export_failed: bool,
}

// SAFETY: the element is scheduled `SchedHint::Active` — it runs on a single dedicated
// scheduler thread and never shares its `Display` (the raw `VADisplay` pointer). All
// VA-API calls are made from that one thread. The `Send` bound `Element` requires is
// satisfied by construction; the compiler cannot see the confinement through the raw
// pointer, so assert it here at the single ownership boundary (mirrors h264dec).
unsafe impl Send for VaapiH265Dec {}

impl Default for VaapiH265Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl VaapiH265Dec {
    pub fn new() -> Self {
        Self::with_zerocopy(None)
    }

    /// Build the decoder in **zero-copy** mode: after `vaSyncSurface`, `emit_pending`
    /// exports the decoded surface as DMA-BUF(s) (`vaExportSurfaceHandle`) and pushes a
    /// [`GpuFrame`] into `channel` for the player's EGL frame-slot sink to import — no CPU
    /// readback, no pixel pool. The default [`new`](Self::new) readback mode is unchanged
    /// (the CLI and software peers keep getting planar I420). See [`GpuFrameChannel`] for
    /// the surface-lifetime handshake.
    pub fn new_zerocopy(channel: Arc<GpuFrameChannel>) -> Self {
        Self::with_zerocopy(Some(channel))
    }

    // Cold: constructor, runs once per element; DPB grows in-place afterward.
    #[allow(clippy::disallowed_methods)]
    fn with_zerocopy(zerocopy: Option<Arc<GpuFrameChannel>>) -> Self {
        VaapiH265Dec {
            device: None,
            display: None,
            va: None,
            sps: std::array::from_fn(|_| None),
            pps: std::array::from_fn(|_| None),
            dpb: Vec::new(),
            poc: h265parse::PocState::new(),
            output_queue: VecDeque::new(),
            output_key: std::collections::HashMap::new(),
            gop: 0,
            pending: None,
            skip_rasl: false,
            awaiting_irap: true,
            announced: false,
            dims: None,
            alloc_stalled: false,
            disabled: false,
            warned_unsupported: false,
            warned_multi_slice: false,
            zerocopy,
            next_token: 1,
            in_flight: std::collections::HashMap::new(),
            export_failed: false,
        }
    }

    fn warn(&self, ctx: &mut Ctx, message: String) {
        let element = ctx.element();
        ctx.post(BusMessage::Warning {
            element,
            error: Error::Element { element, message },
        });
    }

    /// Lazily open the VA display (probe-selected device). Returns false if no device
    /// is available (the element then warn-drops everything).
    fn ensure_display(&mut self, ctx: &mut Ctx) -> bool {
        if self.display.is_some() {
            return true;
        }
        let Some(caps) = probe::probe() else {
            if !self.disabled {
                self.disabled = true;
                self.warn(ctx, "vaapih265dec: no VA-API device available".into());
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
                    self.warn(ctx, format!("vaapih265dec: cannot open VA display: {e}"));
                }
                false
            }
        }
    }

    /// Build (or rebuild) the VA config/context/surfaces for an SPS's dimensions.
    // Cold: VA config/context/surface setup, done once per dimension at announce.
    #[allow(clippy::disallowed_methods)]
    fn ensure_va(&mut self, ctx: &mut Ctx, sps: &Sps) -> bool {
        // Announced / emitted dims are the conformance-cropped *display* size (matching
        // the software decoder + reference); the surface allocation is the CTB-aligned
        // *coded* size below. A mismatch here shifts every row and destroys the picture.
        let width = sps.width().max(16);
        let height = sps.height().max(16);
        let (crop_x, crop_y) = sps.crop_origin();
        // Surfaces + context must cover the *coded* picture: whole CTBs
        // (PicWidthInCtbsY × CtbSizeY / PicHeightInCtbsY × CtbSizeY, §7.4.3.2.1);
        // the conformance window is display metadata, not storage. Allocating only
        // the cropped size makes iHD's vaCreateContext fail (the h264dec 1080/1088
        // lesson).
        let ctb = 1u32 << sps.ctb_log2_size_y();
        let coded_w = (sps.pic_width_in_ctbs() * ctb).max(16);
        let coded_h = (sps.pic_height_in_ctbs() * ctb).max(16);
        if let Some(va) = &self.va {
            if va.width == width && va.height == height {
                return true;
            }
            // Dimension change: tear down (drop resets refs) and re-announce.
            self.va = None;
            self.dpb.clear();
            self.output_queue.clear();
            self.pending = None;
            self.announced = false;
        }
        let Some(display) = &self.display else { return false };

        // Surface budget: every simultaneous holder must fit — the DPB (≤
        // sps_max_dec_pic_buffering), a full reorder queue, the readback carry, the
        // picture being decoded, and slack. HEVC's max DPB is 16 refs.
        let dpb_slots = (sps.sps_max_dec_pic_buffering_minus1 + 1).clamp(2, 16);
        // Zero-copy adds a *presentation* budget: a surface exported to the GUI stays out of
        // the free pool until the sink releases its token (anti-tear). With a triple-buffered
        // frame slot the GUI holds ≤ 3 at once; +PRESENTATION_SLACK covers the release-drain
        // latency (the decoder drains released tokens once per `process`). Fewer than this and
        // the decoder blocks on export backpressure; the bound is what keeps it from tearing.
        const PRESENTATION_SLACK: u32 = 6;
        let extra = if self.zerocopy.is_some() { PRESENTATION_SLACK } else { 0 };
        let count = dpb_slots * 2 + 4 + extra;

        let profile = ffi::VAProfileHEVCMain; // Main 8-bit 4:2:0.
        let config = match Config::new_decode(display, profile) {
            Ok(c) => c,
            Err(e) => {
                self.warn(ctx, format!("vaapih265dec: vaCreateConfig failed: {e}"));
                return false;
            }
        };
        let surfaces = match Surfaces::new_nv12(display, coded_w, coded_h, count) {
            Ok(s) => s,
            Err(e) => {
                self.warn(ctx, format!("vaapih265dec: vaCreateSurfaces failed: {e}"));
                return false;
            }
        };
        let context =
            match Context::new(display, &config, coded_w as i32, coded_h as i32, &surfaces) {
                Ok(c) => c,
                Err(e) => {
                    self.warn(ctx, format!("vaapih265dec: vaCreateContext failed: {e}"));
                    return false;
                }
            };
        let free = surfaces.ids().to_vec();
        self.va = Some(VaState {
            _config: config,
            context,
            _surfaces: surfaces,
            free,
            width,
            height,
            crop_x,
            crop_y,
            coded_w,
            coded_h,
        });
        self.dims = Some((width, height));
        true
    }

    fn acquire_surface(&mut self) -> Option<ffi::VASurfaceID> {
        self.va.as_mut().and_then(|va| va.free.pop())
    }

    fn surfaces_exhausted(&self) -> bool {
        self.va.as_ref().is_some_and(|va| va.free.is_empty())
    }

    fn release_surface(&mut self, id: ffi::VASurfaceID) {
        if let Some(va) = &mut self.va {
            if !va.free.contains(&id) {
                va.free.push(id);
            }
        }
    }

    /// Whether any live holder still needs `id`'s pixels: the DPB, the reorder queue,
    /// the readback carry, or — in zero-copy mode — the GUI, which is still sampling the
    /// surface's exported dma-buf until it releases the token (`in_flight`). The last is the
    /// anti-tear gate: an evicted-but-in-flight surface stays out of the free pool.
    fn surface_referenced(&self, id: ffi::VASurfaceID) -> bool {
        self.dpb.iter().any(|e| e.surface == id)
            || self.output_queue.iter().any(|o| o.surface == id)
            || self.pending.as_ref().is_some_and(|o| o.surface == id)
            || self.in_flight.contains_key(&id)
    }

    /// The single return-to-free gate: a surface goes back on the free list only once
    /// *nothing* references it — a DPB-evicted surface that is still queued for output
    /// must not be handed out as a decode target while a frame awaits readback from it.
    fn release_if_unreferenced(&mut self, id: ffi::VASurfaceID) {
        if !self.surface_referenced(id) {
            self.release_surface(id);
        }
    }

    /// Decode one access unit's picture (warn-drop on any malformed / out-of-subset
    /// input, never panic).
    fn decode_au(&mut self, ctx: &mut Ctx, au: &[u8], pts: Timestamp, duration: Timestamp) {
        let nals = h265parse::split_nals(au);

        // 1) Cache parameter sets.
        for nal in &nals {
            match nal.nal_type {
                h265parse::NAL_SPS => {
                    if let Some(sps) = h265parse::parse_sps(nal.raw) {
                        let id = sps.sps_id as usize;
                        if id < 16 {
                            self.sps[id] = Some(sps);
                        }
                    }
                }
                h265parse::NAL_PPS => {
                    if let Some(pps) = h265parse::parse_pps(nal.raw) {
                        let id = pps.pps_id as usize;
                        if id < 64 {
                            self.pps[id] = Some(pps);
                        }
                    }
                }
                _ => {}
            }
        }

        // 2) Find slice NALs (VCL, §7.4.2.2). The first slice carries the picture.
        let slice_nals: Vec<_> =
            nals.iter().filter(|n| h265parse::is_slice_nal(n.nal_type)).collect();
        if slice_nals.is_empty() {
            return; // parameter-set / SEI / AUD-only AU: nothing to output
        }
        let first = slice_nals[0];
        let nal_type = first.nal_type;
        let temporal_id = first.temporal_id;
        let is_idr = h265parse::is_idr(nal_type);
        let is_irap = h265parse::is_irap(nal_type);

        // 3) RASL handling (§8.1.3): after a CRA/BLA that begins decoding, RASL
        // pictures reference pre-RAP pictures that were never decoded — drop them.
        if self.awaiting_irap && !is_irap {
            return; // wait for a clean random-access point
        }
        if h265parse::is_rasl(nal_type) && self.skip_rasl {
            return;
        }

        // 4) Peek the PPS id from the first slice, then resolve SPS/PPS.
        let Some(pps_id) = peek_pps_id(first.raw, is_irap) else {
            self.warn(ctx, "vaapih265dec: could not read slice pps_id — AU dropped".into());
            return;
        };
        let Some(pps) = self.pps.get(pps_id as usize).and_then(|p| p.clone()) else {
            self.warn(ctx, "vaapih265dec: slice references unknown PPS — dropped".into());
            return;
        };
        let Some(sps) = self.sps.get(pps.sps_id as usize).and_then(|s| s.clone()) else {
            self.warn(ctx, "vaapih265dec: PPS references unknown SPS — dropped".into());
            return;
        };

        // 5) Refuse out-of-subset streams (Main 8-bit 4:2:0 only).
        if !sps.supported() {
            if !self.warned_unsupported {
                self.warned_unsupported = true;
                self.warn(
                    ctx,
                    "vaapih265dec: unsupported SPS (only Main 8-bit 4:2:0 is wired; Main10 / \
                     4:2:2 / 4:4:4 / range or screen-content extensions refused) — dropped"
                        .into(),
                );
            }
            return;
        }

        // 6) Parse every slice segment header.
        let mut headers: Vec<SliceHeader> = Vec::with_capacity(slice_nals.len());
        for n in &slice_nals {
            let Some(sh) = h265parse::parse_slice_header(n.raw, n.nal_type, &sps, &pps) else {
                // A dependent slice segment (multi-slice tiled picture) parses to None
                // in our subset — refuse the whole AU rather than decode a partial pic.
                if !self.warned_multi_slice {
                    self.warned_multi_slice = true;
                    self.warn(
                        ctx,
                        "vaapih265dec: malformed or dependent-slice-segment header — AU dropped \
                         (multi-segment tiled pictures are out of this subset)"
                            .into(),
                    );
                }
                return;
            };
            headers.push(sh);
        }
        let sh0 = &headers[0];

        // 7) Ensure VA objects for these dimensions.
        if !self.ensure_va(ctx, &sps) {
            return;
        }

        // 8) IRAP boundary: an IRAP with NoRaslOutputFlag starts a fresh coded video
        // sequence. IDR/BLA always clear the DPB; a CRA that begins decoding does too.
        let no_rasl_output = is_idr || h265parse::is_bla(nal_type) || self.awaiting_irap;
        if is_irap && no_rasl_output {
            self.reset_dpb();
            self.gop = self.gop.wrapping_add(1);
        }
        // After a CRA/BLA random-access, following RASL pictures must be skipped.
        self.skip_rasl = is_irap && !is_idr;
        self.awaiting_irap = false;

        // 9) POC (§8.3.1).
        let poc = self.poc.compute(&sps, sh0, is_idr, is_irap && no_rasl_output, temporal_id);

        // 10) Reference Picture Set (§8.3.2): mark which DPB pictures the current
        // picture keeps, and unmark/evict the rest.
        self.apply_rps(&sps, sh0, poc, is_idr);

        // 11) Acquire the target surface.
        let Some(target) = self.acquire_surface() else {
            self.warn(ctx, "vaapih265dec: no free surface — AU dropped".into());
            return;
        };

        // 12) Build + submit the picture.
        let (width, height, crop_x, crop_y, coded_w, coded_h) = self
            .va
            .as_ref()
            .map(|v| (v.width, v.height, v.crop_x, v.crop_y, v.coded_w, v.coded_h))
            .unwrap();
        if let Err(e) =
            self.submit_picture(&sps, &pps, &headers, &slice_nals, target, poc, nal_type)
        {
            self.warn(ctx, format!("vaapih265dec: VA submit failed: {e} — dropped"));
            self.release_surface(target);
            return;
        }

        // 13) Insert the just-decoded picture into the DPB as a short-term reference
        // (every decoded picture is a potential reference in HEVC until unmarked by a
        // later RPS; §8.3.2). Non-reference pictures are pruned by the next RPS.
        self.dpb.push(DpbEntry { surface: target, poc, long_term: false, rps_flag: 0 });

        // 14) Output queue (POC order) + bump.
        if sh0.pic_output_flag {
            self.output_queue.push_back(PendingOut {
                surface: target,
                pts,
                duration,
                width,
                height,
                crop_x,
                crop_y,
                coded_w,
                coded_h,
            });
            self.reorder_output(target, poc);
        }
        self.maybe_bump(&sps);
    }

    /// Apply the current picture's Reference Picture Set (§8.3.2): tag DPB entries
    /// that the slice's short-term / long-term RPS keeps "used by current picture"
    /// with the matching `VA_PICTURE_HEVC_RPS_*` flag, and evict every DPB picture in
    /// no current set (it can never be referenced again in this CVS).
    fn apply_rps(&mut self, sps: &Sps, sh: &SliceHeader, cur_poc: i32, is_idr: bool) {
        // Clear last picture's membership tags.
        for e in &mut self.dpb {
            e.rps_flag = 0;
        }
        if is_idr {
            // IDR: DPB already cleared in decode_au; nothing to keep.
            return;
        }

        let max_poc_lsb = sps.max_poc_lsb();

        // Short-term "before" (POC < cur) and "after" (POC > cur), by delta (§8.3.2).
        // Iterate the RPS fields directly (same used-by-curr filter as `curr_before()` /
        // `curr_after()`) so no per-frame `Vec<i32>` is allocated; the deltas visited and
        // their order are identical to the helper's output.
        for (delta, _) in sh
            .st_rps
            .delta_poc_s0
            .iter()
            .zip(sh.st_rps.used_s0.iter())
            .filter(|(_, u)| **u)
        {
            let ref_poc = cur_poc + delta; // delta is negative here
            if let Some(e) = self.dpb.iter_mut().find(|e| e.poc == ref_poc) {
                e.rps_flag = ffi::VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE;
                e.long_term = false;
            }
        }
        for (delta, _) in sh
            .st_rps
            .delta_poc_s1
            .iter()
            .zip(sh.st_rps.used_s1.iter())
            .filter(|(_, u)| **u)
        {
            let ref_poc = cur_poc + delta; // delta is positive here
            if let Some(e) = self.dpb.iter_mut().find(|e| e.poc == ref_poc) {
                e.rps_flag = ffi::VA_PICTURE_HEVC_RPS_ST_CURR_AFTER;
                e.long_term = false;
            }
        }
        // Long-term references (§8.3.2): match by POC LSB (± MSB cycle when present).
        for lt in &sh.long_term_refs {
            if !lt.used_by_curr_pic {
                continue;
            }
            let target_lsb = lt.poc_lsb as i32;
            let found = if lt.delta_poc_msb_present {
                let cur_lsb = ((cur_poc % max_poc_lsb) + max_poc_lsb) % max_poc_lsb;
                let target =
                    cur_poc - lt.delta_poc_msb_cycle as i32 * max_poc_lsb - (cur_lsb - target_lsb);
                self.dpb.iter_mut().find(|e| e.poc == target)
            } else {
                self.dpb
                    .iter_mut()
                    .find(|e| (e.poc % max_poc_lsb + max_poc_lsb) % max_poc_lsb == target_lsb)
            };
            if let Some(e) = found {
                e.rps_flag = ffi::VA_PICTURE_HEVC_RPS_LT_CURR;
                e.long_term = true;
            }
        }

        // Evict every DPB picture in no current set that is not still awaiting output —
        // those can never be referenced again (§8.3.2). A picture still queued for
        // output stays (its surface is pinned there), just not as a reference.
        let evicted: Vec<ffi::VASurfaceID> = self
            .dpb
            .iter()
            .filter(|e| e.rps_flag == 0)
            .map(|e| e.surface)
            .collect();
        self.dpb.retain(|e| e.rps_flag != 0);
        for id in evicted {
            self.release_if_unreferenced(id);
        }
    }

    /// Reorder the just-pushed output entry into (gop, POC) order within the queue.
    fn reorder_output(&mut self, surface: ffi::VASurfaceID, poc: i32) {
        self.output_key.insert(surface, (self.gop, poc));
        let mut i = self.output_queue.len().saturating_sub(1);
        while i > 0 {
            let a = self.output_queue[i - 1].surface;
            let b = self.output_queue[i].surface;
            let ka = *self.output_key.get(&a).unwrap_or(&(u32::MAX, i32::MAX));
            let kb = *self.output_key.get(&b).unwrap_or(&(u32::MAX, i32::MAX));
            if ka > kb {
                self.output_queue.swap(i - 1, i);
                i -= 1;
            } else {
                break;
            }
        }
    }

    /// Bump the smallest-(gop,POC) output entry into `pending` when the reorder queue
    /// exceeds the DPB reorder budget (§C.5-style).
    fn maybe_bump(&mut self, sps: &Sps) {
        let budget = (sps.sps_max_dec_pic_buffering_minus1 as usize).max(1) + 1;
        while self.output_queue.len() > budget && self.pending.is_none() {
            if let Some(out) = self.output_queue.pop_front() {
                self.pending = Some(out);
            } else {
                break;
            }
        }
    }

    /// Fill and submit the VA picture: pic param + per-slice params + slice data,
    /// wrapped in Begin/Render/End.
    #[allow(clippy::too_many_arguments)]
    fn submit_picture(
        &mut self,
        sps: &Sps,
        pps: &Pps,
        headers: &[SliceHeader],
        slice_nals: &[&h265parse::Nal],
        target: ffi::VASurfaceID,
        cur_poc: i32,
        nal_type: u8,
    ) -> Result<(), crate::va::VaError> {
        let display = self.display.as_ref().unwrap();
        let va = self.va.as_ref().unwrap();
        let context = &va.context;

        // --- Picture parameter buffer (va_dec_hevc.h VAPictureParameterBufferHEVC) ---
        let mut pic = zeroed_pic_param_hevc();
        pic.CurrPic = ffi::VAPictureHEVC {
            picture_id: target,
            pic_order_cnt: cur_poc,
            flags: 0,
            va_reserved: [0; ffi::VA_PADDING_LOW],
        };
        // ReferenceFrames[15] from the DPB (§8.3.2). Only pictures in the current RPS
        // carry a live flag; the rest are INVALID (the driver ignores them). A stable
        // DPB→ReferenceFrames index map lets per-slice RefPicList point back in.
        pic.ReferenceFrames = [ffi::VAPictureHEVC::invalid(); 15];
        // DPB→ReferenceFrames index map. At most 15 entries (the `.take(15)` below), so a
        // fixed (surface, index) stack array replaces the per-picture heap HashMap; the
        // per-slice lookup is a linear scan over ≤ 15 pairs (see `build_ref_pic_list`).
        let mut ref_index: [(ffi::VASurfaceID, u8); 15] = [(0, 0); 15];
        let mut ref_index_len = 0usize;
        for (i, e) in self.dpb.iter().filter(|e| e.rps_flag != 0).take(15).enumerate() {
            let mut flags = e.rps_flag;
            if e.long_term {
                flags |= ffi::VA_PICTURE_HEVC_LONG_TERM_REFERENCE;
            }
            pic.ReferenceFrames[i] = ffi::VAPictureHEVC {
                picture_id: e.surface,
                pic_order_cnt: e.poc,
                flags,
                va_reserved: [0; ffi::VA_PADDING_LOW],
            };
            ref_index[ref_index_len] = (e.surface, i as u8);
            ref_index_len += 1;
        }
        let ref_index = &ref_index[..ref_index_len];

        pic.pic_width_in_luma_samples = sps.pic_width_in_luma_samples as u16;
        pic.pic_height_in_luma_samples = sps.pic_height_in_luma_samples as u16;
        pic.pic_fields = pack_pic_fields(sps, pps);
        pic.sps_max_dec_pic_buffering_minus1 = sps.sps_max_dec_pic_buffering_minus1 as u8;
        pic.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8 as u8;
        pic.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8 as u8;
        pic.pcm_sample_bit_depth_luma_minus1 = sps.pcm_sample_bit_depth_luma_minus1 as u8;
        pic.pcm_sample_bit_depth_chroma_minus1 = sps.pcm_sample_bit_depth_chroma_minus1 as u8;
        pic.log2_min_luma_coding_block_size_minus3 =
            sps.log2_min_luma_coding_block_size_minus3 as u8;
        pic.log2_diff_max_min_luma_coding_block_size =
            sps.log2_diff_max_min_luma_coding_block_size as u8;
        pic.log2_min_transform_block_size_minus2 = sps.log2_min_transform_block_size_minus2 as u8;
        pic.log2_diff_max_min_transform_block_size =
            sps.log2_diff_max_min_transform_block_size as u8;
        pic.log2_min_pcm_luma_coding_block_size_minus3 =
            sps.log2_min_pcm_luma_coding_block_size_minus3 as u8;
        pic.log2_diff_max_min_pcm_luma_coding_block_size =
            sps.log2_diff_max_min_pcm_luma_coding_block_size as u8;
        pic.max_transform_hierarchy_depth_intra = sps.max_transform_hierarchy_depth_intra as u8;
        pic.max_transform_hierarchy_depth_inter = sps.max_transform_hierarchy_depth_inter as u8;
        pic.init_qp_minus26 = pps.init_qp_minus26 as i8;
        pic.diff_cu_qp_delta_depth = pps.diff_cu_qp_delta_depth as u8;
        pic.pps_cb_qp_offset = pps.pps_cb_qp_offset as i8;
        pic.pps_cr_qp_offset = pps.pps_cr_qp_offset as i8;
        pic.log2_parallel_merge_level_minus2 = pps.log2_parallel_merge_level_minus2 as u8;
        pic.num_tile_columns_minus1 = pps.num_tile_columns_minus1 as u8;
        pic.num_tile_rows_minus1 = pps.num_tile_rows_minus1 as u8;
        // column_width/row_height are meaningful only when tiles are enabled AND
        // non-uniform. Uniform / no-tiles streams leave them zero (the driver derives).
        if pps.tiles_enabled_flag && !pps.uniform_spacing_flag {
            for (i, &w) in pps.column_width_minus1.iter().take(19).enumerate() {
                pic.column_width_minus1[i] = w as u16;
            }
            for (i, &h) in pps.row_height_minus1.iter().take(21).enumerate() {
                pic.row_height_minus1[i] = h as u16;
            }
        }
        pic.slice_parsing_fields = pack_slice_parsing_fields(sps, pps, nal_type, headers);
        pic.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4 as u8;
        pic.num_short_term_ref_pic_sets = sps.num_short_term_ref_pic_sets as u8;
        pic.num_long_term_ref_pic_sps = sps.num_long_term_ref_pics_sps as u8;
        pic.num_ref_idx_l0_default_active_minus1 = pps.num_ref_idx_l0_default_active_minus1 as u8;
        pic.num_ref_idx_l1_default_active_minus1 = pps.num_ref_idx_l1_default_active_minus1 as u8;
        pic.pps_beta_offset_div2 = pps.pps_beta_offset_div2 as i8;
        pic.pps_tc_offset_div2 = pps.pps_tc_offset_div2 as i8;
        pic.num_extra_slice_header_bits = pps.num_extra_slice_header_bits as u8;
        // st_rps_bits: the in-slice short-term-RPS bit count (0 when it came from SPS),
        // taken from the first slice — all slices of one picture share it.
        pic.st_rps_bits = headers[0].st_rps_bits;

        let pic_buf =
            VaBuffer::new_struct(display, context, ffi::VAPictureParameterBufferType, &pic)?;

        context.begin(target)?;
        context.render(&[pic_buf.id()])?;

        // --- Per-slice: slice param + slice data ---
        let mut keep: Vec<VaBuffer> = Vec::with_capacity(slice_nals.len() * 2);
        let last = headers.len().saturating_sub(1);
        for (i, (sh, nal)) in headers.iter().zip(slice_nals.iter()).enumerate() {
            let sp = self.build_slice_param(sps, sh, nal.raw, i == last, ref_index);
            let sp_buf =
                VaBuffer::new_struct(display, context, ffi::VASliceParameterBufferType, &sp)?;
            let data_buf = VaBuffer::new_data(display, context, nal.raw)?;
            context.render(&[sp_buf.id(), data_buf.id()])?;
            keep.push(sp_buf);
            keep.push(data_buf);
        }
        context.end()?;
        Ok(())
    }

    /// Fill a `VASliceParameterBufferHEVC` from a slice header + its NAL, resolving
    /// `RefPicList[2][15]` into `ReferenceFrames[]` indices (§8.3.4).
    fn build_slice_param(
        &self,
        sps: &Sps,
        sh: &SliceHeader,
        raw_nal: &[u8],
        last_slice: bool,
        ref_index: &[(ffi::VASurfaceID, u8)],
    ) -> ffi::VASliceParameterBufferHEVC {
        // RefPicList0/1 (§8.3.4): concatenate the RPS current-before/after/long lists
        // in spec order, cycle to fill the active count, then apply any explicit
        // list_entry modification. Entries are ReferenceFrames[] indices; 0xFF unused.
        let (ref_pic_list, collocated_ref_idx) = self.build_ref_pic_list(sh, ref_index);
        let long_slice_flags = pack_long_slice_flags(sh, last_slice);

        let mut sp: ffi::VASliceParameterBufferHEVC = zeroed_slice_param_hevc();
        sp.slice_data_size = raw_nal.len() as u32;
        sp.slice_data_offset = 0;
        sp.slice_data_flag = ffi::VA_SLICE_DATA_FLAG_ALL;
        sp.slice_data_byte_offset = sh.slice_data_byte_offset;
        sp.slice_segment_address = sh.slice_segment_address;
        sp.RefPicList = ref_pic_list;
        sp.long_slice_flags = long_slice_flags;
        sp.collocated_ref_idx = collocated_ref_idx;
        sp.num_ref_idx_l0_active_minus1 = sh.num_ref_idx_l0_active_minus1 as u8;
        sp.num_ref_idx_l1_active_minus1 = sh.num_ref_idx_l1_active_minus1 as u8;
        sp.slice_qp_delta = sh.slice_qp_delta as i8;
        sp.slice_cb_qp_offset = sh.slice_cb_qp_offset as i8;
        sp.slice_cr_qp_offset = sh.slice_cr_qp_offset as i8;
        sp.slice_beta_offset_div2 = sh.slice_beta_offset_div2 as i8;
        sp.slice_tc_offset_div2 = sh.slice_tc_offset_div2 as i8;
        sp.five_minus_max_num_merge_cand = sh.five_minus_max_num_merge_cand as u8;
        sp.num_entry_point_offsets = sh.num_entry_point_offsets.min(u16::MAX as u32) as u16;
        sp.entry_offset_to_subset_array = 0;
        sp.slice_data_num_emu_prevn_bytes = sh.num_emu_prev_bytes;

        // Weighted prediction (§7.4.7.3 → va_dec_hevc.h). Luma weight/offset are the
        // parsed deltas verbatim. Chroma is the subtle case: VA's `ChromaOffsetL{0,1}`
        // is the *derived spec variable* `ChromaOffset`, NOT the parsed
        // `delta_chroma_offset` syntax element (va_dec_hevc.h: "corresponds to HEVC
        // spec variable of the same name"). §7.4.7.3 (eq. 7-56):
        //   ChromaWeight  = (1 << ChromaLog2WeightDenom) + delta_chroma_weight
        //   ChromaOffset  = Clip3(-128, 127,
        //                     delta_chroma_offset - ((128 * ChromaWeight)
        //                     >> ChromaLog2WeightDenom) + 128)
        // With delta_chroma_weight == 0 this reduces to delta_chroma_offset (why an
        // identity-chroma-weight slice decodes correctly even from the raw value, but a
        // non-zero weight does not).
        let _ = sps;
        if let Some(t) = &sh.pred_weights {
            sp.luma_log2_weight_denom = t.luma_log2_weight_denom as u8;
            sp.delta_chroma_log2_weight_denom = t.delta_chroma_log2_weight_denom as i8;
            let chroma_denom = (t.luma_log2_weight_denom as i32
                + t.delta_chroma_log2_weight_denom)
                .clamp(0, 7);
            let derive_chroma_offset = |delta_w: i32, delta_off: i32| -> i8 {
                let chroma_weight = (1i32 << chroma_denom) + delta_w;
                let v = delta_off - ((128 * chroma_weight) >> chroma_denom) + 128;
                v.clamp(-128, 127) as i8
            };
            for (i, e) in t.l0.iter().take(15).enumerate() {
                sp.delta_luma_weight_l0[i] = e.delta_luma_weight as i8;
                sp.luma_offset_l0[i] = e.luma_offset as i8;
                for c in 0..2 {
                    sp.delta_chroma_weight_l0[i][c] = e.delta_chroma_weight[c] as i8;
                    sp.ChromaOffsetL0[i][c] =
                        derive_chroma_offset(e.delta_chroma_weight[c], e.chroma_offset[c]);
                }
            }
            for (i, e) in t.l1.iter().take(15).enumerate() {
                sp.delta_luma_weight_l1[i] = e.delta_luma_weight as i8;
                sp.luma_offset_l1[i] = e.luma_offset as i8;
                for c in 0..2 {
                    sp.delta_chroma_weight_l1[i][c] = e.delta_chroma_weight[c] as i8;
                    sp.ChromaOffsetL1[i][c] =
                        derive_chroma_offset(e.delta_chroma_weight[c], e.chroma_offset[c]);
                }
            }
        }
        sp
    }

    /// Build `RefPicList[2][15]` (§8.3.4): default construction from the current RPS,
    /// with the slice's explicit `list_entry` modifications applied. Returns the list
    /// and the resolved `collocated_ref_idx` (0xFF when temporal MVP is off).
    fn build_ref_pic_list(
        &self,
        sh: &SliceHeader,
        ref_index: &[(ffi::VASurfaceID, u8)],
    ) -> ([[u8; 15]; 2], u8) {
        let mut list = [[0xFFu8; 15]; 2];
        if !sh.slice_type.is_inter() {
            return (list, 0xFF);
        }

        // Candidate pools by RPS membership, in POC order (§8.3.4):
        //   before: RPS_ST_CURR_BEFORE, POC descending toward cur (nearest first)
        //   after:  RPS_ST_CURR_AFTER,  POC ascending away from cur
        //   long:   RPS_LT_CURR
        // Every pool is drawn through `ref_index` (≤ 15 entries, built with .take(15) in
        // submit_picture), so the combined candidate list is ≤ 15. Fixed stack buffers
        // (with a filled-length counter) replace the per-slice heap Vecs; a defensive
        // capacity guard keeps the same overflow-free behavior the Vecs had.
        let mut before: [(i32, u8); 16] = [(0, 0); 16];
        let mut before_len = 0usize;
        for e in self.dpb.iter().filter(|e| e.rps_flag == ffi::VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE) {
            if let Some(&(_, i)) = ref_index.iter().find(|(s, _)| *s == e.surface) {
                if before_len < before.len() {
                    before[before_len] = (e.poc, i);
                    before_len += 1;
                }
            }
        }
        let before = &mut before[..before_len];
        before.sort_by_key(|a| std::cmp::Reverse(a.0)); // descending (nearest below cur first)
        let mut after: [(i32, u8); 16] = [(0, 0); 16];
        let mut after_len = 0usize;
        for e in self.dpb.iter().filter(|e| e.rps_flag == ffi::VA_PICTURE_HEVC_RPS_ST_CURR_AFTER) {
            if let Some(&(_, i)) = ref_index.iter().find(|(s, _)| *s == e.surface) {
                if after_len < after.len() {
                    after[after_len] = (e.poc, i);
                    after_len += 1;
                }
            }
        }
        let after = &mut after[..after_len];
        after.sort_by_key(|a| a.0); // ascending (nearest above cur first)
        let mut long: [u8; 16] = [0; 16];
        let mut long_len = 0usize;
        for e in self.dpb.iter().filter(|e| e.rps_flag == ffi::VA_PICTURE_HEVC_RPS_LT_CURR) {
            if let Some(&(_, i)) = ref_index.iter().find(|(s, _)| *s == e.surface) {
                if long_len < long.len() {
                    long[long_len] = i;
                    long_len += 1;
                }
            }
        }
        let long = &long[..long_len];

        // RefPicListTemp0 = before ++ after ++ long ; Temp1 = after ++ before ++ long.
        let mut temp0: [u8; 48] = [0; 48];
        let mut temp0_len = 0usize;
        for i in before
            .iter()
            .map(|(_, i)| *i)
            .chain(after.iter().map(|(_, i)| *i))
            .chain(long.iter().copied())
        {
            if temp0_len < temp0.len() {
                temp0[temp0_len] = i;
                temp0_len += 1;
            }
        }
        let temp0 = &temp0[..temp0_len];
        let mut temp1: [u8; 48] = [0; 48];
        let mut temp1_len = 0usize;
        for i in after
            .iter()
            .map(|(_, i)| *i)
            .chain(before.iter().map(|(_, i)| *i))
            .chain(long.iter().copied())
        {
            if temp1_len < temp1.len() {
                temp1[temp1_len] = i;
                temp1_len += 1;
            }
        }
        let temp1 = &temp1[..temp1_len];

        let n0 = (sh.num_ref_idx_l0_active_minus1 as usize + 1).min(15);
        fill_ref_list(&mut list[0], temp0, n0, &sh.ref_list_mods.l0, sh.ref_list_mods.l0_present);
        if sh.slice_type == SliceType::B {
            let n1 = (sh.num_ref_idx_l1_active_minus1 as usize + 1).min(15);
            fill_ref_list(
                &mut list[1],
                temp1,
                n1,
                &sh.ref_list_mods.l1,
                sh.ref_list_mods.l1_present,
            );
        }

        // collocated_ref_idx is the *index into RefPicList[collocated_from_l0 ? 0 : 1]*
        // (va_dec_hevc.h:421 "index to RefPicList[0][] or RefPicList[1][]") — i.e. the
        // slice's `collocated_ref_idx` syntax element verbatim, NOT the resolved
        // `ReferenceFrames[]` index. (Resolving it into ReferenceFrames — as an earlier
        // version did — broke TMVP for any picture with more than one reference, since
        // RefPicList[0][0] there is not 0: exactly the multi-ref P-frame drift.)
        let collocated_ref_idx = if sh.slice_temporal_mvp_enabled_flag {
            (sh.collocated_ref_idx as u8).min(14)
        } else {
            0xFF
        };
        (list, collocated_ref_idx)
    }

    /// Emit the pending decoded surface, dispatching on the output mode: zero-copy
    /// DMA-BUF export ([`emit_pending_zerocopy`](Self::emit_pending_zerocopy)) when built
    /// with [`new_zerocopy`](Self::new_zerocopy) and the export has not fallen back, else
    /// the CPU readback ([`emit_pending_readback`](Self::emit_pending_readback)). `false` =
    /// backpressure (pool exhausted / presentation budget full) — leave the frame carried.
    fn emit_pending(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        if self.zerocopy.is_some() && !self.export_failed {
            return self.emit_pending_zerocopy(ctx);
        }
        self.emit_pending_readback(ctx)
    }

    /// Copy the pending decoded surface into a pool buffer as planar **I420** and emit.
    /// `false` = pool exhausted (backpressure). VA surfaces are NV12: the Y plane
    /// copies straight; the interleaved CbCr plane de-interleaves into planar Cb then
    /// Cr (matching the software `h265dec`'s `i420` output).
    fn emit_pending_readback(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        if self.pending.is_none() {
            self.pending = self.output_queue.pop_front();
        }
        let Some(out) = self.pending.take() else { return Ok(true) };

        let dpy = match &self.display {
            Some(d) => d.raw(),
            None => return Ok(true),
        };

        // Display (cropped) dims — what we announce + emit. The chroma crop origin is
        // half the luma origin in 4:2:0.
        let width = out.width;
        let height = out.height;
        let (crop_x, crop_y) = (out.crop_x, out.crop_y);
        // I420: Y (w×h) + Cb (⌈w/2⌉×⌈h/2⌉) + Cr (⌈w/2⌉×⌈h/2⌉).
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);
        let y_len = (width * height) as usize;
        let c_len = (cw * ch) as usize;
        let need = y_len + 2 * c_len;

        // Map the *coded* surface (the driver's derived image reports coded dims); the
        // repack below lifts the cropped display region out of it.
        let image = match MappedImage::acquire(dpy, out.surface, out.coded_w, out.coded_h) {
            Ok(im) => im,
            Err(e) => {
                self.warn(ctx, format!("vaapih265dec: readback failed: {e} — frame dropped"));
                self.recycle_output_surface(out.surface);
                return Ok(true);
            }
        };
        if image.num_planes() < 2 {
            self.warn(ctx, "vaapih265dec: mapped image is not 2-plane NV12 — dropped".into());
            drop(image);
            self.recycle_output_surface(out.surface);
            return Ok(true);
        }

        let Some(mut buf) = ctx.try_alloc(SRC_PAD) else {
            if !self.alloc_stalled {
                self.alloc_stalled = true;
                let s = ctx.pool_stats();
                log!(
                    &*ctx,
                    Level::Debug,
                    "alloc_stall",
                    outstanding = s.outstanding,
                    max_slots = s.max_slots,
                    acquires = s.acquires,
                    recycles = s.recycles,
                );
            }
            drop(image);
            self.pending = Some(out);
            return Ok(false);
        };
        self.alloc_stalled = false;
        if buf.memory.capacity() < need {
            drop(image);
            self.recycle_output_surface(out.surface);
            return Err(Error::Element {
                element: ctx.element(),
                message: format!(
                    "vaapih265dec: {width}x{height} I420 frame needs {need} bytes but pool slots \
                     hold {} — raise the pipeline pool slot size",
                    buf.memory.capacity()
                ),
            });
        }

        if !self.announced {
            log!(&*ctx, Level::Debug, "announce", width = width, height = height);
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

        // Repack the *cropped display region* (§7.4.3.2.1): Y straight; CbCr
        // de-interleaved (NV12 → I420). The crop origin skips into each source row —
        // luma by crop_x, chroma by crop_x/2 (4:2:0). Rows start at crop_y / crop_y/2.
        let cx = crop_x as usize;
        let cx_c = (crop_x / 2) as usize;
        let dst = buf.memory.as_mut_full();
        {
            let (y_dst, chroma_dst) = dst.split_at_mut(y_len);
            let (cb_dst, cr_dst) = chroma_dst.split_at_mut(c_len);
            for row in 0..height {
                // Read crop_x + width luma bytes from the source row, use the tail.
                let Some(src) = image.row(0, crop_y + row, cx + width as usize) else {
                    drop(image);
                    self.recycle_output_surface(out.surface);
                    self.warn(ctx, "vaapih265dec: Y plane geometry overrun — dropped".into());
                    return Ok(true);
                };
                let d = &mut y_dst[(row * width) as usize..((row + 1) * width) as usize];
                d.copy_from_slice(&src[cx..]);
            }
            for row in 0..ch {
                // Plane 1 is interleaved CbCr: (crop_x/2 + cw) pairs → ×2 bytes.
                let Some(src) = image.row(1, crop_y / 2 + row, (cx_c + cw as usize) * 2) else {
                    drop(image);
                    self.recycle_output_surface(out.surface);
                    self.warn(ctx, "vaapih265dec: CbCr plane geometry overrun — dropped".into());
                    return Ok(true);
                };
                let cb = &mut cb_dst[(row * cw) as usize..((row + 1) * cw) as usize];
                let cr = &mut cr_dst[(row * cw) as usize..((row + 1) * cw) as usize];
                for x in 0..cw as usize {
                    cb[x] = src[2 * (cx_c + x)];
                    cr[x] = src[2 * (cx_c + x) + 1];
                }
            }
        }
        buf.memory.set_len(need);
        buf.pts = out.pts;
        buf.duration = out.duration;
        drop(image);
        log!(&*ctx, Level::Trace, "frame", pts = buf.pts);
        ctx.out(SRC_PAD).push(buf);

        self.recycle_output_surface(out.surface);
        Ok(true)
    }

    /// The zero-copy emit: export the pending surface as DMA-BUF(s), push a [`GpuFrame`]
    /// out-of-band, and emit a tiny `video/gpu` buffer (the token + display geometry) the
    /// frame-slot sink pairs and imports via EGL. The surface is marked **in flight** and
    /// deliberately NOT recycled — the sink returns it (via a released token, drained in
    /// `process`) only after presenting. `false` = the presentation budget is full (every
    /// spare surface is in flight) — backpressure, exactly like a pool-exhausted readback.
    ///
    /// On an export failure (the driver refuses `vaExportSurfaceHandle`/PRIME_2) the mode
    /// latches back to readback for the rest of the run (`export_failed`): the current frame
    /// is retried through the readback path on the next call, so no frame is lost.
    fn emit_pending_zerocopy(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        // Backpressure: if there is no spare surface to leave in flight, do not export — the
        // decoder must keep a working set for the DPB + the picture in decode. `free` being
        // empty here means every surface is a ref/pending/in-flight; carry the frame and let
        // a release free one (the same shape as the readback pool-exhausted path).
        if self.surfaces_exhausted() {
            // Only actually a stall if nothing else can free a surface (all in flight / DPB).
            // The scheduler re-enters after a release drain; carrying is the backpressure.
            if self.pending.is_none() && self.output_queue.is_empty() {
                return Ok(true); // nothing to emit
            }
        }

        if self.pending.is_none() {
            self.pending = self.output_queue.pop_front();
        }
        let Some(out) = self.pending.take() else { return Ok(true) };

        let dpy = match &self.display {
            Some(d) => d.raw(),
            None => return Ok(true),
        };

        // Export the surface (vaSyncSurface + vaExportSurfaceHandle, PRIME_2/COMPOSED/RO).
        let exported = match ExportedSurface::export(dpy, out.surface) {
            Ok(e) => e,
            Err(e) => {
                // Latch to readback for the rest of the run; retry THIS frame there.
                self.export_failed = true;
                self.warn(
                    ctx,
                    format!(
                        "vaapih265dec: DMA-BUF export failed ({e}); falling back to CPU readback \
                         for the rest of the run"
                    ),
                );
                eprintln!(
                    "pfplay-ui: zero-copy VA-API export unavailable ({e}) — falling back to CPU \
                     readback (video path = SDL texture upload)"
                );
                self.pending = Some(out); // retry via readback next call
                return Ok(true);
            }
        };

        // dup the fds so the importer (the sink) owns its own copies; the ExportedSurface
        // drop below closes the originals.
        let dup_fds = match exported.dup_fds() {
            Ok(f) => f,
            Err(e) => {
                self.warn(ctx, format!("vaapih265dec: dup(dma-buf fd) failed: {e} — frame dropped"));
                // The surface never went in flight — recycle it and move on.
                self.recycle_output_surface(out.surface);
                return Ok(true);
            }
        };

        if !self.announced {
            log!(&*ctx, Level::Debug, "announce_gpu", width = out.width, height = out.height);
            ctx.announce_format(
                SRC_PAD,
                FAMILY_VIDEO_GPU,
                &[
                    (F_WIDTH, ValueDesc::Int(out.width as i64)),
                    (F_HEIGHT, ValueDesc::Int(out.height as i64)),
                    (F_LAYOUT, ValueDesc::Id(LAYOUT_NV12_DMABUF)),
                ],
            );
            self.announced = true;
            eprintln!("pfplay-ui: video path = zero-copy VA-API/EGL (vaapih265dec DMA-BUF export)");
        }

        // The tiny in-band buffer: just the header (token + geometry). Allocate from the
        // pool; a `video/gpu` slot is ~40 bytes, so any pool slot fits. Backpressure on the
        // buffer alloc (rare — the header is minuscule) carries the frame like readback does.
        let Some(mut buf) = ctx.try_alloc(SRC_PAD) else {
            if !self.alloc_stalled {
                self.alloc_stalled = true;
                let s = ctx.pool_stats();
                log!(
                    &*ctx,
                    Level::Debug,
                    "alloc_stall",
                    outstanding = s.outstanding,
                    max_slots = s.max_slots,
                );
            }
            // Nothing went in flight; drop the dup'd fds (close them) and carry the frame.
            for fd in dup_fds {
                // SAFETY: fd is an owned dup we just made; close exactly once.
                unsafe {
                    gpuframe::close_raw_fd(fd);
                }
            }
            self.pending = Some(out);
            return Ok(false);
        };
        self.alloc_stalled = false;

        let token = self.next_token;
        self.next_token = self.next_token.wrapping_add(1);

        let header = GpuFrameHeader {
            token,
            coded_w: out.coded_w,
            coded_h: out.coded_h,
            disp_w: out.width,
            disp_h: out.height,
            crop_x: out.crop_x,
            crop_y: out.crop_y,
        };
        let bytes = header.to_bytes();
        let dst = buf.memory.as_mut_full();
        if dst.len() < bytes.len() {
            // A pool slot smaller than the ~40-byte header would be a pipeline mis-config.
            for fd in dup_fds {
                // SAFETY: owned dup; close once.
                unsafe {
                    gpuframe::close_raw_fd(fd);
                }
            }
            self.recycle_output_surface(out.surface);
            return Err(Error::Element {
                element: ctx.element(),
                message: format!(
                    "vaapih265dec: video/gpu header needs {} bytes but pool slots hold {}",
                    bytes.len(),
                    dst.len()
                ),
            });
        }
        dst[..bytes.len()].copy_from_slice(&bytes);
        buf.memory.set_len(bytes.len());
        buf.pts = out.pts;
        buf.duration = out.duration;

        // Push the out-of-band GpuFrame (dup'd fds + geometry) and mark the surface in flight
        // BEFORE emitting the buffer, so a fast sink that pops the token cannot race ahead of
        // the in-flight marking.
        let frame = GpuFrame {
            token,
            fds: dup_fds,
            drm_format: exported.drm_format,
            drm_modifier: exported.drm_modifier,
            coded_w: exported.coded_w,
            coded_h: exported.coded_h,
            disp_w: out.width,
            disp_h: out.height,
            crop_x: out.crop_x,
            crop_y: out.crop_y,
            planes: exported.planes.clone(),
        };
        // The ExportedSurface's own fds close here (drop) — the importer has its dups.
        drop(exported);
        if let Some(ch) = &self.zerocopy {
            ch.push(frame);
        }
        self.in_flight.insert(out.surface, token);

        log!(&*ctx, Level::Trace, "gpu_frame", pts = buf.pts, token = token);
        ctx.out(SRC_PAD).push(buf);

        // Do NOT recycle: the surface is in flight (a ref via `in_flight`). Only clear the
        // output-order key; the release drain frees the surface once the sink presents it.
        self.output_key.remove(&out.surface);
        Ok(true)
    }

    /// Drain the tokens the sink has released (presented) and return each token's surface to
    /// the free pool, if nothing else references it. Called once per `process` in zero-copy
    /// mode — this is the surface-lifetime handshake's producer half.
    fn drain_presentation_releases(&mut self) {
        let Some(ch) = self.zerocopy.clone() else { return };
        for token in ch.drain_released() {
            let surface = self.in_flight.iter().find(|(_, &t)| t == token).map(|(&s, _)| s);
            if let Some(surface) = surface {
                self.in_flight.remove(&surface);
                self.release_if_unreferenced(surface);
            }
        }
    }

    fn recycle_output_surface(&mut self, surface: ffi::VASurfaceID) {
        self.output_key.remove(&surface);
        self.release_if_unreferenced(surface);
    }

    fn reset_dpb(&mut self) {
        let ids: Vec<_> = self.dpb.drain(..).map(|e| e.surface).collect();
        for id in ids {
            // Not release_surface: an IRAP resets the DPB while earlier-GOP frames may
            // still sit in the reorder queue awaiting readback.
            self.release_if_unreferenced(id);
        }
    }

    /// Emit staged output beyond the reorder budget (steady-state drain in process()).
    fn emit_over_budget(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        if self.pending.is_some() && !self.emit_pending(ctx)? {
            return Ok(false);
        }
        Ok(true)
    }
}

impl Element for VaapiH265Dec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.ensure_display(ctx) {
            while inputs.pop().is_some() {}
            return Ok(Flow::Ok);
        }
        // Zero-copy: reclaim surfaces the sink has presented (released) before doing anything
        // else — this is what keeps the export path from starving on the presentation budget.
        self.drain_presentation_releases();
        loop {
            if !self.emit_over_budget(ctx)? {
                return Ok(Flow::Ok);
            }
            // A free surface must exist before the next AU: a momentarily-exhausted
            // pool is *backpressure* (leave input in the ring; emitting frees
            // surfaces), never a drop (the h264dec realtime-pacing lesson).
            while self.surfaces_exhausted() {
                if self.pending.is_none() && self.output_queue.is_empty() {
                    break; // held by the DPB alone — decode_au's drop is the fuse
                }
                if !self.emit_pending(ctx)? {
                    return Ok(Flow::Ok);
                }
            }
            let Some(inbuf) = inputs.pop() else { break };
            let au = inbuf.memory.data().to_vec();
            self.decode_au(ctx, &au, inbuf.pts, inbuf.duration);
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FlushStart => {
                // Drop refs + carries + POC; keep the VA context (dims unchanged).
                // Output entries drain first so reset_dpb frees their surfaces.
                let outs: Vec<_> = self
                    .pending
                    .take()
                    .into_iter()
                    .chain(self.output_queue.drain(..))
                    .map(|o| o.surface)
                    .collect();
                self.output_key.clear();
                // Zero-copy: a flush discards the pre-seek pictures, so surfaces still in
                // flight to the GUI can be reclaimed — the GUI's slot will be overwritten by
                // post-seek frames, and a discarded frame's brief reuse is never displayed.
                // (The sink still owns and closes its dup'd fds; clearing here only lets the
                // decoder reuse the *surface*.) Drain any late releases first, then clear.
                self.drain_presentation_releases();
                let in_flight: Vec<_> = self.in_flight.keys().copied().collect();
                self.in_flight.clear();
                for id in outs.into_iter().chain(in_flight) {
                    self.release_if_unreferenced(id);
                }
                self.reset_dpb();
                self.poc.reset();
                self.skip_rasl = false;
                self.awaiting_irap = true; // resume at a clean random-access point
                self.alloc_stalled = false;
            }
            Event::Eos => {
                self.drain_presentation_releases();
                while self.pending.is_some() || !self.output_queue.is_empty() {
                    if !self.emit_pending(ctx)? {
                        break;
                    }
                    self.drain_presentation_releases();
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // Signal the sink so a GUI blocked on a token wakes; the sink's own drop closes any
        // dup'd fds it still holds.
        if let Some(ch) = &self.zerocopy {
            ch.close();
        }
        self.in_flight.clear();
        self.pending = None;
        self.output_queue.clear();
        self.dpb.clear();
        self.va = None;
        self.display = None;
    }
}

// --- free-function helpers ------------------------------------------------------------

/// Fill one VA `RefPicList` row from a temp candidate list (cycling to the active
/// count, §8.3.4), applying explicit `list_entry` modifications when present.
fn fill_ref_list(dst: &mut [u8; 15], temp: &[u8], active: usize, mods: &[u32], mods_present: bool) {
    if temp.is_empty() {
        return;
    }
    if mods_present && !mods.is_empty() {
        for (i, slot) in dst.iter_mut().take(active.min(15)).enumerate() {
            let idx = *mods.get(i).unwrap_or(&0) as usize;
            *slot = *temp.get(idx).unwrap_or(&temp[0]);
        }
    } else {
        for (i, slot) in dst.iter_mut().take(active.min(15)).enumerate() {
            *slot = temp[i % temp.len()];
        }
    }
}

/// Peek the slice_pic_parameter_set_id of a VCL NAL: first_slice_segment_in_pic_flag
/// (1), [no_output_of_prior_pics_flag (1) if IRAP], then slice_pic_parameter_set_id
/// (ue). De-emulates a short prefix for safety.
fn peek_pps_id(nal: &[u8], is_irap: bool) -> Option<u32> {
    if nal.len() < 3 {
        return None;
    }
    let (body, body_len) = de_emulate_prefix(&nal[2..]);
    let mut r = crate::h264parse::BitReader::new(&body[..body_len]);
    let _first = r.flag();
    if is_irap {
        let _no_output = r.flag();
    }
    Some(r.ue())
}

/// De-emulate just enough of an RBSP body for the pps_id peek (§7.3.1.1). Returns a
/// fixed 16-byte buffer and the filled length (the peek never needs more than the first
/// 16 de-emulated bytes), so the per-AU heap `Vec` is gone — allocation source only, the
/// bytes produced are identical.
fn de_emulate_prefix(body: &[u8]) -> ([u8; 16], usize) {
    let mut out = [0u8; 16];
    let mut out_len = 0usize;
    let mut zeros = 0;
    let mut i = 0;
    while i < body.len() && out_len < 16 {
        let b = body[i];
        if zeros >= 2 && b == 0x03 && i + 1 < body.len() && body[i + 1] <= 0x03 {
            zeros = 0;
            i += 1;
            continue;
        }
        out[out_len] = b;
        out_len += 1;
        zeros = if b == 0 { zeros + 1 } else { 0 };
        i += 1;
    }
    (out, out_len)
}

fn zeroed_pic_param_hevc() -> ffi::VAPictureParameterBufferHEVC {
    // SAFETY: plain POD (integers + arrays of integers); all-zero is a valid initial
    // state, then filled field by field.
    unsafe { std::mem::zeroed() }
}

fn zeroed_slice_param_hevc() -> ffi::VASliceParameterBufferHEVC {
    // SAFETY: as above. RefPicList is then overwritten with 0xFF-filled rows.
    unsafe { std::mem::zeroed() }
}

/// Pack `VAPictureParameterBufferHEVC.pic_fields` (va_dec_hevc.h:71).
fn pack_pic_fields(sps: &Sps, pps: &Pps) -> u32 {
    let mut v = 0u32;
    v |= (sps.chroma_format_idc & 0x3) << ffi::HEVC_PIC_CHROMA_FORMAT_IDC_SHIFT;
    if sps.separate_colour_plane_flag {
        v |= ffi::HEVC_PIC_SEPARATE_COLOUR_PLANE_FLAG;
    }
    if sps.pcm_enabled_flag {
        v |= ffi::HEVC_PIC_PCM_ENABLED_FLAG;
    }
    if sps.scaling_list_enabled_flag {
        v |= ffi::HEVC_PIC_SCALING_LIST_ENABLED_FLAG;
    }
    if pps.transform_skip_enabled_flag {
        v |= ffi::HEVC_PIC_TRANSFORM_SKIP_ENABLED_FLAG;
    }
    if sps.amp_enabled_flag {
        v |= ffi::HEVC_PIC_AMP_ENABLED_FLAG;
    }
    if sps.strong_intra_smoothing_enabled_flag {
        v |= ffi::HEVC_PIC_STRONG_INTRA_SMOOTHING_ENABLED_FLAG;
    }
    if pps.sign_data_hiding_enabled_flag {
        v |= ffi::HEVC_PIC_SIGN_DATA_HIDING_ENABLED_FLAG;
    }
    if pps.constrained_intra_pred_flag {
        v |= ffi::HEVC_PIC_CONSTRAINED_INTRA_PRED_FLAG;
    }
    if pps.cu_qp_delta_enabled_flag {
        v |= ffi::HEVC_PIC_CU_QP_DELTA_ENABLED_FLAG;
    }
    if pps.weighted_pred_flag {
        v |= ffi::HEVC_PIC_WEIGHTED_PRED_FLAG;
    }
    if pps.weighted_bipred_flag {
        v |= ffi::HEVC_PIC_WEIGHTED_BIPRED_FLAG;
    }
    if pps.transquant_bypass_enabled_flag {
        v |= ffi::HEVC_PIC_TRANSQUANT_BYPASS_ENABLED_FLAG;
    }
    if pps.tiles_enabled_flag {
        v |= ffi::HEVC_PIC_TILES_ENABLED_FLAG;
    }
    if pps.entropy_coding_sync_enabled_flag {
        v |= ffi::HEVC_PIC_ENTROPY_CODING_SYNC_ENABLED_FLAG;
    }
    if pps.pps_loop_filter_across_slices_enabled_flag {
        v |= ffi::HEVC_PIC_PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG;
    }
    if pps.loop_filter_across_tiles_enabled_flag {
        v |= ffi::HEVC_PIC_LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG;
    }
    if sps.pcm_loop_filter_disabled_flag {
        v |= ffi::HEVC_PIC_PCM_LOOP_FILTER_DISABLED_FLAG;
    }
    // NoPicReorderingFlag / NoBiPredFlag are optimizations the driver may use; leave 0
    // (safe — the driver then makes no assumption).
    v
}

/// Pack `VAPictureParameterBufferHEVC.slice_parsing_fields` (va_dec_hevc.h:139).
fn pack_slice_parsing_fields(sps: &Sps, pps: &Pps, nal_type: u8, headers: &[SliceHeader]) -> u32 {
    let mut v = 0u32;
    if pps.lists_modification_present_flag {
        v |= ffi::HEVC_SLP_LISTS_MODIFICATION_PRESENT_FLAG;
    }
    if sps.long_term_ref_pics_present_flag {
        v |= ffi::HEVC_SLP_LONG_TERM_REF_PICS_PRESENT_FLAG;
    }
    if sps.sps_temporal_mvp_enabled_flag {
        v |= ffi::HEVC_SLP_SPS_TEMPORAL_MVP_ENABLED_FLAG;
    }
    if pps.cabac_init_present_flag {
        v |= ffi::HEVC_SLP_CABAC_INIT_PRESENT_FLAG;
    }
    if pps.output_flag_present_flag {
        v |= ffi::HEVC_SLP_OUTPUT_FLAG_PRESENT_FLAG;
    }
    if pps.dependent_slice_segments_enabled_flag {
        v |= ffi::HEVC_SLP_DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG;
    }
    if pps.pps_slice_chroma_qp_offsets_present_flag {
        v |= ffi::HEVC_SLP_PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG;
    }
    if sps.sample_adaptive_offset_enabled_flag {
        v |= ffi::HEVC_SLP_SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG;
    }
    if pps.deblocking_filter_override_enabled_flag {
        v |= ffi::HEVC_SLP_DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG;
    }
    if pps.pps_deblocking_filter_disabled_flag {
        v |= ffi::HEVC_SLP_PPS_DISABLE_DEBLOCKING_FILTER_FLAG;
    }
    if pps.slice_segment_header_extension_present_flag {
        v |= ffi::HEVC_SLP_SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG;
    }
    if h265parse::is_irap(nal_type) {
        v |= ffi::HEVC_SLP_RAP_PIC_FLAG;
    }
    if h265parse::is_idr(nal_type) {
        v |= ffi::HEVC_SLP_IDR_PIC_FLAG;
    }
    // IntraPicFlag: the whole picture has only intra (I) slices.
    if headers.iter().all(|h| h.slice_type == SliceType::I) {
        v |= ffi::HEVC_SLP_INTRA_PIC_FLAG;
    }
    v
}

/// Pack `VASliceParameterBufferHEVC.long_slice_flags` (va_dec_hevc.h:390).
fn pack_long_slice_flags(sh: &SliceHeader, last_slice: bool) -> u32 {
    let mut v = 0u32;
    if last_slice {
        v |= ffi::HEVC_LSF_LAST_SLICE_OF_PIC;
    }
    if sh.dependent_slice_segment_flag {
        v |= ffi::HEVC_LSF_DEPENDENT_SLICE_SEGMENT_FLAG;
    }
    v |= (sh.slice_type.va_value() & 0x3) << ffi::HEVC_LSF_SLICE_TYPE_SHIFT;
    // color_plane_id: 0 for 4:2:0 (no separate colour plane) → nothing to set.
    if sh.slice_sao_luma_flag {
        v |= ffi::HEVC_LSF_SLICE_SAO_LUMA_FLAG;
    }
    if sh.slice_sao_chroma_flag {
        v |= ffi::HEVC_LSF_SLICE_SAO_CHROMA_FLAG;
    }
    if sh.mvd_l1_zero_flag {
        v |= ffi::HEVC_LSF_MVD_L1_ZERO_FLAG;
    }
    if sh.cabac_init_flag {
        v |= ffi::HEVC_LSF_CABAC_INIT_FLAG;
    }
    if sh.slice_temporal_mvp_enabled_flag {
        v |= ffi::HEVC_LSF_SLICE_TEMPORAL_MVP_ENABLED_FLAG;
    }
    if sh.slice_deblocking_filter_disabled_flag {
        v |= ffi::HEVC_LSF_SLICE_DEBLOCKING_FILTER_DISABLED_FLAG;
    }
    if sh.collocated_from_l0_flag {
        v |= ffi::HEVC_LSF_COLLOCATED_FROM_L0_FLAG;
    }
    if sh.slice_loop_filter_across_slices_enabled_flag {
        v |= ffi::HEVC_LSF_SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_ref_list_cycles_to_active_count() {
        // §8.3.4: when the temp list is shorter than the active count, it cycles.
        let mut dst = [0xFFu8; 15];
        let temp = [3u8, 5];
        fill_ref_list(&mut dst, &temp, 4, &[], false);
        assert_eq!(&dst[..4], &[3, 5, 3, 5]);
        assert_eq!(dst[4], 0xFF, "entries beyond active stay invalid");
    }

    #[test]
    fn fill_ref_list_applies_modifications() {
        // With list_entry present, each slot takes temp[list_entry[i]].
        let mut dst = [0xFFu8; 15];
        let temp = [10u8, 20, 30];
        fill_ref_list(&mut dst, &temp, 3, &[2, 0, 1], true);
        assert_eq!(&dst[..3], &[30, 10, 20]);
    }

    #[test]
    fn long_slice_flags_carry_slice_type_and_last() {
        let mut sh = stub_sh(SliceType::B);
        sh.collocated_from_l0_flag = false;
        let v = pack_long_slice_flags(&sh, true);
        assert_ne!(v & ffi::HEVC_LSF_LAST_SLICE_OF_PIC, 0);
        // slice_type B == 0 → bits [3:2] are zero.
        assert_eq!((v >> ffi::HEVC_LSF_SLICE_TYPE_SHIFT) & 0x3, 0);
        assert_eq!(v & ffi::HEVC_LSF_COLLOCATED_FROM_L0_FLAG, 0);

        let sh_p = stub_sh(SliceType::P);
        let vp = pack_long_slice_flags(&sh_p, false);
        assert_eq!((vp >> ffi::HEVC_LSF_SLICE_TYPE_SHIFT) & 0x3, 1, "P slice_type == 1");
    }

    fn stub_sh(t: SliceType) -> SliceHeader {
        SliceHeader {
            first_slice_segment_in_pic_flag: true,
            no_output_of_prior_pics_flag: false,
            pps_id: 0,
            dependent_slice_segment_flag: false,
            slice_segment_address: 0,
            slice_type: t,
            pic_output_flag: true,
            pic_order_cnt_lsb: 0,
            st_rps: h265parse::ShortTermRps::default(),
            st_rps_bits: 0,
            long_term_refs: Vec::new(),
            slice_temporal_mvp_enabled_flag: false,
            slice_sao_luma_flag: false,
            slice_sao_chroma_flag: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            ref_list_mods: h265parse::RefListMods::default(),
            mvd_l1_zero_flag: false,
            cabac_init_flag: false,
            collocated_from_l0_flag: true,
            collocated_ref_idx: 0,
            pred_weights: None,
            five_minus_max_num_merge_cand: 0,
            slice_qp_delta: 0,
            slice_cb_qp_offset: 0,
            slice_cr_qp_offset: 0,
            deblocking_filter_override_flag: false,
            slice_deblocking_filter_disabled_flag: false,
            slice_beta_offset_div2: 0,
            slice_tc_offset_div2: 0,
            slice_loop_filter_across_slices_enabled_flag: true,
            num_entry_point_offsets: 0,
            slice_data_byte_offset: 0,
            num_emu_prev_bytes: 0,
        }
    }
}
