//! `vaapih264dec` — the VA-API hardware H.264 decode element. One Annex-B access
//! unit per buffer arrives on the sink (`h264/annexb`, the demuxer contract);
//! tight-packed NV12 frames leave on the `video/raw` src pad, one per buffer.
//!
//! Unlike the software `h264dec` (which wraps a full-decode library), VA-API is a
//! slice-level API: this element parses the parameter sets and slice *headers*
//! (`crate::h264parse`, ITU-T H.264), maintains the decoded-picture buffer (DPB)
//! and reference lists itself (§8.2.4 / §8.2.5), fills the
//! `VAPictureParameterBufferH264` / `VASliceParameterBufferH264` structs, and drives
//! `vaBeginPicture` / `vaRenderPicture` / `vaEndPicture`. The GPU does entropy
//! decode, motion compensation and reconstruction; the host does bitstream framing
//! and reference bookkeeping.
//!
//! POC boundary (see crate docs): 8-bit 4:2:0 progressive; POC types 0/1/2; sliding
//! window + MMCO 5, other MMCO ops best-effort; ref-list default construction
//! (§8.2.4.2) + modifications (§8.2.4.3). Interlaced/MBAFF/field, >8-bit, non-4:2:0
//! and FMO are recognized and warn-dropped, not decoded. Reorder pts uses a
//! feed-order FIFO (same documented caveat as the software decoder).

use std::collections::VecDeque;
use std::path::PathBuf;

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

use std::sync::Arc;

use crate::ffi;
use crate::gpuframe::{
    self, GpuFrame, GpuFrameChannel, GpuFrameHeader, FrameToken, FAMILY_VIDEO_GPU, F_LAYOUT,
    LAYOUT_NV12_DMABUF,
};
use crate::h264parse::{self, MmcoOp, Pps, PredWeightTable, RefListMod, SliceHeader, SliceType, Sps};
use crate::probe;
use crate::va::{Buffer as VaBuffer, Config, Context, Display, ExportedSurface, MappedImage, Surfaces};

// `video/raw` family/field/value names, kept as literals (core-only, like the
// software h264dec) — the pipeline interns by string so they line up with peers.
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
const PIXFMT_NV12: &str = "nv12";

const SRC_PAD: PadId = PadId(1);

static PIXFMT_VALUES: [ValueDesc; 1] = [ValueDesc::Id(PIXFMT_NV12)];

// Broad `video/raw` template: any dimensions, NV12 only (what VA-API surfaces
// yield). Concrete width/height are announced at runtime from the SPS, so the src
// pad is `dynamic`.
static SRC_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
];
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }];
// One H.264 Annex-B access unit per buffer, same sink family as the software
// decoder (`h264/annexb`). AVCC framing is a distinct family and out of scope.
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
        dynamic: true,
        validate: None,
    },
];

// Cold: make_default boxes one element when the registry builds the default, not per-frame.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "vaapih264dec",
    pads: &PADS,
    props: &[],
    // Active: a hardware decode round-trip (submit + sync + readback) is well
    // beyond the inline passive budget, and its own thread pipelines against demux
    // and display.
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(VaapiH264Dec::new())),
};

/// A DPB entry: a reference picture held for prediction (§8.2.5).
#[derive(Clone)]
struct DpbEntry {
    surface: ffi::VASurfaceID,
    frame_num: i32,
    /// FrameNumWrap, recomputed each picture for ref-list ordering (§8.2.4.1).
    frame_num_wrap: i32,
    poc: i32,
    top_poc: i32,
    bottom_poc: i32,
    long_term: bool,
    long_term_frame_idx: i32,
}

/// A decoded surface awaiting readback + emit — the backpressure carry (bounded to
/// one), and the DPB output queue drains through it in POC order.
struct PendingOut {
    surface: ffi::VASurfaceID,
    pts: Timestamp,
    duration: Timestamp,
    width: u32,
    height: u32,
    /// Coded (macroblock-aligned) surface dimensions — the dma-buf plane geometry the
    /// zero-copy export reports is at this size (the readback path maps at the cropped size,
    /// so it ignores these). H.264 frame-cropping is always relative to a (0,0) top-left
    /// origin in this subset, so no crop_x/crop_y is carried (both 0).
    coded_w: u32,
    coded_h: u32,
}

/// Lazily-built VA-API state: created on the first usable SPS, torn down + rebuilt
/// on a dimension change.
struct VaState {
    _config: Config,
    context: Context,
    // Held for ownership only: the surface set must outlive the context (which
    // renders into it). Field order = drop order, so `context` drops first, then
    // these surfaces — the order libva requires.
    _surfaces: Surfaces,
    /// Free surface pool (ids not currently a DPB ref or a pending output).
    free: Vec<ffi::VASurfaceID>,
    width: u32,
    height: u32,
    /// Coded (macroblock-aligned) surface dimensions — the zero-copy export's dma-buf
    /// plane geometry is at this size (the readback path maps at the cropped size).
    coded_w: u32,
    coded_h: u32,
}

/// The VA-API H.264 decode element.
pub struct VaapiH264Dec {
    // Device + VA objects, opened/created lazily.
    device: Option<PathBuf>,
    display: Option<Display>,
    va: Option<VaState>,

    // Parameter-set caches (by id).
    sps: [Option<Sps>; 32],
    pps: [Option<Pps>; 256],

    // DPB + POC + reference state.
    dpb: Vec<DpbEntry>,
    poc: h264parse::PocState,
    /// Pictures decoded and reference-marked but not yet output, in (gop, POC) order.
    output_queue: VecDeque<PendingOut>,
    /// surface → (gop, POC), for ordering the output queue (§C.4-style bumping).
    /// POC resets to 0 at every IDR (§8.2.1), so raw POC alone would sort a new
    /// GOP's pictures *ahead* of the previous GOP's still-queued tail — the old
    /// scene would flash back after the cut. The gop counter makes the key
    /// globally monotonic in display order.
    output_key: std::collections::HashMap<ffi::VASurfaceID, (u32, i32)>,
    /// Monotonic coded-video-sequence counter, bumped at each IDR.
    gop: u32,
    /// The one picture currently in readback (backpressure carry).
    pending: Option<PendingOut>,

    announced: bool,
    dims: Option<(u32, u32)>,
    alloc_stalled: bool,
    /// Set once a fatal setup error was reported, to avoid a warning storm.
    disabled: bool,

    // --- Zero-copy (DMA-BUF export) mode -------------------------------------------------
    /// When `Some`, `emit_pending` exports the decoded surface as DMA-BUF(s) and pushes a
    /// [`GpuFrame`] into this channel instead of NV12 readback. `None` is the default
    /// readback mode (unchanged for the CLI and software peers). See
    /// [`new_zerocopy`](Self::new_zerocopy) and [`crate::gpuframe`] for the full flow.
    zerocopy: Option<Arc<GpuFrameChannel>>,
    /// Monotonic token pairing an in-band `video/gpu` buffer with its [`GpuFrame`].
    next_token: FrameToken,
    /// Surfaces exported to the GUI and not yet released (presented): surface id → token.
    /// A surface here is **in flight** — its dma-buf is being sampled, so it must not return
    /// to the free pool (a new decode into it would tear the display).
    in_flight: std::collections::HashMap<ffi::VASurfaceID, FrameToken>,
    /// One-time latch: the export path failed (driver refused PRIME_2) → fall back to
    /// readback for the rest of the run.
    export_failed: bool,
}

// SAFETY: the element is scheduled `SchedHint::Active` — it runs on a single
// dedicated scheduler thread and never shares its `Display` (which carries the raw
// `VADisplay` pointer) with any other thread. The pipeline moves an element to its
// worker thread once at start and drives it only from there; VA-API calls are all
// made from that one thread. The `Send` bound the `Element` trait requires is
// therefore satisfied by construction, but the compiler cannot see the confinement
// through the raw pointer — assert it here, at the single ownership boundary.
unsafe impl Send for VaapiH264Dec {}

impl Default for VaapiH264Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl VaapiH264Dec {
    pub fn new() -> Self {
        Self::with_zerocopy(None)
    }

    /// Build the decoder in **zero-copy** mode: `emit_pending` exports the decoded surface
    /// as DMA-BUF(s) (`vaExportSurfaceHandle`) and pushes a [`GpuFrame`] into `channel` for
    /// the player's EGL frame-slot sink to import — no CPU readback. The default
    /// [`new`](Self::new) readback mode (planar/NV12 for the CLI + software peers) is
    /// unchanged. See [`GpuFrameChannel`] for the surface-lifetime handshake.
    pub fn new_zerocopy(channel: Arc<GpuFrameChannel>) -> Self {
        Self::with_zerocopy(Some(channel))
    }

    // Cold: constructor, runs once per element; DPB grows in-place afterward.
    #[allow(clippy::disallowed_methods)]
    fn with_zerocopy(zerocopy: Option<Arc<GpuFrameChannel>>) -> Self {
        VaapiH264Dec {
            device: None,
            display: None,
            va: None,
            sps: std::array::from_fn(|_| None),
            pps: std::array::from_fn(|_| None),
            dpb: Vec::new(),
            poc: h264parse::PocState::new(),
            output_queue: VecDeque::new(),
            output_key: std::collections::HashMap::new(),
            gop: 0,
            pending: None,
            announced: false,
            dims: None,
            alloc_stalled: false,
            disabled: false,
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

    /// Lazily open the VA display (probe-selected device). Returns false if no
    /// device is available (the element then warn-drops everything).
    fn ensure_display(&mut self, ctx: &mut Ctx) -> bool {
        if self.display.is_some() {
            return true;
        }
        let Some(caps) = probe::probe() else {
            if !self.disabled {
                self.disabled = true;
                self.warn(ctx, "vaapih264dec: no VA-API device available".into());
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
                    self.warn(ctx, format!("vaapih264dec: cannot open VA display: {e}"));
                }
                false
            }
        }
    }

    /// Build (or rebuild) the VA config/context/surfaces for an SPS's dimensions.
    // Cold: VA config/context/surface setup, done once per dimension at announce.
    #[allow(clippy::disallowed_methods)]
    fn ensure_va(&mut self, ctx: &mut Ctx, sps: &Sps) -> bool {
        let width = sps.width().max(16);
        let height = sps.height().max(16);
        // Surfaces and the context must cover the *coded* picture — whole
        // macroblocks (§7.4.2.1.1: PicWidthInMbs×16 / FrameHeightInMbs×16);
        // frame cropping is display metadata, not storage. A 1080-line stream
        // codes 1088 rows: allocating the cropped size makes iHD's
        // vaCreateContext fail with VA_STATUS_ERROR_ALLOCATION_FAILED.
        let coded_w = (sps.width_in_mbs() * 16).max(16);
        let coded_h = (sps.height_in_mbs() * 16).max(16);
        // Reuse if dimensions match.
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

        // Surface budget: every holder must fit simultaneously — the DPB
        // (≤ max_num_ref refs), a *full* reorder queue (maybe_bump's budget:
        // max_num_ref+1), the readback carry, the picture being decoded, and
        // slack. Undersizing doesn't just stall: under a realtime-pacing sink
        // the exhaustion path then emits the queue front *below* the reorder
        // budget, and a later-decoded smaller-POC B-frame displays after a
        // newer frame — visible as old-frame glitches.
        let dpb_slots = (sps.max_num_ref_frames + 1).clamp(2, 16);
        // Zero-copy adds a *presentation* budget: a surface exported to the GUI stays out of
        // the free pool until the sink releases its token (anti-tear). +PRESENTATION_SLACK
        // covers the triple-buffered slot's ≤3 in-flight frames plus release-drain latency;
        // fewer and the decoder blocks on export backpressure (never tears).
        const PRESENTATION_SLACK: u32 = 6;
        let extra = if self.zerocopy.is_some() { PRESENTATION_SLACK } else { 0 };
        let count = dpb_slots * 2 + 4 + extra;

        let profile = ffi::VAProfileH264High; // High ⊇ Main/CB for VLD on Intel.
        let config = match Config::new_decode(display, profile) {
            Ok(c) => c,
            Err(e) => {
                self.warn(ctx, format!("vaapih264dec: vaCreateConfig failed: {e}"));
                return false;
            }
        };
        let surfaces = match Surfaces::new_nv12(display, coded_w, coded_h, count) {
            Ok(s) => s,
            Err(e) => {
                self.warn(ctx, format!("vaapih264dec: vaCreateSurfaces failed: {e}"));
                return false;
            }
        };
        let context = match Context::new(display, &config, coded_w as i32, coded_h as i32, &surfaces) {
            Ok(c) => c,
            Err(e) => {
                self.warn(ctx, format!("vaapih264dec: vaCreateContext failed: {e}"));
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
            coded_w,
            coded_h,
        });
        self.dims = Some((width, height));
        true
    }

    /// Acquire a free surface, recycling one from the output queue tail if the pool
    /// is momentarily dry (should not happen with the +5 margin, but keeps the
    /// decoder from wedging on a driver that holds surfaces longer).
    fn acquire_surface(&mut self) -> Option<ffi::VASurfaceID> {
        self.va.as_mut().and_then(|va| va.free.pop())
    }

    /// No surface free for the next picture — `process()` treats this as
    /// backpressure (emit to free one, or leave input in the ring), so
    /// `decode_au`'s drop path is only ever the pathological fuse.
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

    /// Whether any live holder still needs `id`'s pixels: the DPB (a prediction
    /// source), the reorder queue, the readback carry, or — in zero-copy mode — the GUI
    /// still sampling its exported dma-buf (`in_flight`, the anti-tear gate).
    fn surface_referenced(&self, id: ffi::VASurfaceID) -> bool {
        self.dpb.iter().any(|e| e.surface == id)
            || self.output_queue.iter().any(|o| o.surface == id)
            || self.pending.as_ref().is_some_and(|o| o.surface == id)
            || self.in_flight.contains_key(&id)
    }

    /// The single return-to-free gate: a surface goes back on the free list only
    /// once *nothing* references it. Releasing a DPB-evicted surface that is still
    /// queued for output would hand it out as a decode target while a frame still
    /// awaits readback from it — aliased pixels and a corrupted free-list ledger.
    fn release_if_unreferenced(&mut self, id: ffi::VASurfaceID) {
        if !self.surface_referenced(id) {
            self.release_surface(id);
        }
    }

    /// Decode one access unit's picture. Returns Ok(()) on success or warn-drop.
    fn decode_au(&mut self, ctx: &mut Ctx, au: &[u8], pts: Timestamp, duration: Timestamp) {
        let nals = h264parse::split_nals(au);

        // 1) Cache parameter sets.
        for nal in &nals {
            match nal.unit_type {
                h264parse::NAL_SPS => {
                    if let Some(sps) = h264parse::parse_sps(nal.raw) {
                        let id = sps.seq_parameter_set_id as usize;
                        if id < 32 {
                            self.sps[id] = Some(sps);
                        }
                    }
                }
                h264parse::NAL_PPS => {
                    if let Some(pps) = h264parse::parse_pps(nal.raw) {
                        let id = pps.pic_parameter_set_id as usize;
                        if id < 256 {
                            self.pps[id] = Some(pps);
                        }
                    }
                }
                _ => {}
            }
        }

        // 2) Find slice NALs. The first slice of the AU carries the picture header.
        let slice_nals: Vec<_> = nals
            .iter()
            .filter(|n| matches!(n.unit_type, h264parse::NAL_SLICE_NON_IDR | h264parse::NAL_SLICE_IDR))
            .collect();
        if slice_nals.is_empty() {
            // Parameter-set-only AU (or unknown): nothing to output; keep the pts
            // for the next picture only if a slice follows — here there is none, so
            // record nothing (parameter-set AUs carry no timing).
            return;
        }

        // Look up the PPS/SPS in force from the first slice header.
        let first = slice_nals[0];
        // Peek the pps_id: parse a provisional header needs SPS+PPS. Read pps_id via
        // a light parse (first_mb ue, slice_type ue, pps_id ue).
        let pps_id = peek_pps_id(first.raw);
        let Some(pps) = pps_id.and_then(|id| self.pps.get(id as usize).and_then(|p| p.clone())) else {
            self.warn(ctx, "vaapih264dec: slice references unknown PPS — dropped".into());
            return;
        };
        let Some(sps) = self.sps.get(pps.seq_parameter_set_id as usize).and_then(|s| s.clone()) else {
            self.warn(ctx, "vaapih264dec: PPS references unknown SPS — dropped".into());
            return;
        };

        // Refuse unsupported coding tools (POC boundary).
        if sps.separate_colour_plane_flag
            || sps.bit_depth_luma_minus8 != 0
            || sps.bit_depth_chroma_minus8 != 0
            || sps.chroma_format_idc != 1
        {
            self.warn(
                ctx,
                "vaapih264dec: unsupported SPS (only 8-bit 4:2:0 is wired) — dropped".into(),
            );
            return;
        }

        // Parse every slice header.
        let mut headers: Vec<SliceHeader> = Vec::with_capacity(slice_nals.len());
        for n in &slice_nals {
            let Some(sh) = h264parse::parse_slice_header(n.raw, n.unit_type, n.ref_idc, &sps, &pps)
            else {
                self.warn(ctx, "vaapih264dec: malformed slice header — AU dropped".into());
                return;
            };
            if sh.field_pic_flag {
                self.warn(ctx, "vaapih264dec: field/interlaced coding not supported — dropped".into());
                return;
            }
            headers.push(sh);
        }
        let sh0 = &headers[0];
        let is_idr = first.unit_type == h264parse::NAL_SLICE_IDR;
        let ref_idc = first.ref_idc;

        // 3) Ensure VA objects for these dimensions.
        if !self.ensure_va(ctx, &sps) {
            return;
        }

        // 4) IDR resets DPB + POC (§8.2.1, §8.2.5.3) and starts a new coded video
        // sequence — the output-order key's gop component (see `output_key`).
        if is_idr {
            self.reset_dpb();
            self.poc.reset();
            self.gop = self.gop.wrapping_add(1);
        }

        // 5) POC for this picture (§8.2.1).
        let (top_poc, bottom_poc) = self.poc.compute(&sps, sh0, is_idr);
        let poc = top_poc.min(bottom_poc);

        // 6) Recompute FrameNumWrap for the current DPB (§8.2.4.1) for ref lists.
        let max_frame_num = sps.max_frame_num() as i32;
        let cur_frame_num = sh0.frame_num as i32;
        for e in &mut self.dpb {
            e.frame_num_wrap = if !e.long_term && e.frame_num > cur_frame_num {
                e.frame_num - max_frame_num
            } else {
                e.frame_num
            };
        }

        // 7) Acquire the target surface.
        let Some(target) = self.acquire_surface() else {
            self.warn(ctx, "vaapih264dec: no free surface — AU dropped".into());
            return;
        };

        // 8) Build + submit the picture.
        let (width, height, coded_w, coded_h) =
            self.va.as_ref().map(|v| (v.width, v.height, v.coded_w, v.coded_h)).unwrap();
        if let Err(e) = self.submit_picture(
            &sps, &pps, &headers, &slice_nals, target, top_poc, bottom_poc, ref_idc,
        ) {
            self.warn(ctx, format!("vaapih264dec: VA submit failed: {e} — dropped"));
            self.release_surface(target);
            return;
        }

        // 9) Reference marking / DPB update (§8.2.5) — only reference pictures.
        if ref_idc != 0 {
            self.mark_and_insert(&sps, sh0, target, cur_frame_num, poc, top_poc, bottom_poc, is_idr);
        }

        // 10) Output-queue insertion (POC order) and bumping.
        self.output_queue.push_back(PendingOut {
            surface: target,
            pts,
            duration,
            width,
            height,
            coded_w,
            coded_h,
        });
        // Keep the output queue (gop, POC)-sorted by pairing surface→key: we sort
        // by the matching DPB/output ordering. Simplest correct model for our
        // subset: reorder the output queue by key of the just-decoded picture set,
        // tracked in a parallel table.
        self.reorder_output(target, poc);

        // If this picture is NOT a reference, its surface must be freed after
        // output. If it IS a reference, the surface belongs to the DPB and is freed
        // when evicted; the output queue only reads it (readback), never frees it.
        // We record ref-ness via membership in self.dpb (checked at drain time).

        // Bump if the DPB / output queue is over budget (§C.4-style, POC-minimal).
        self.maybe_bump(&sps);
    }

    /// Reorder the just-pushed output entry into (gop, POC) order within the queue.
    /// The key is carried on a side table keyed by surface; the gop component keeps
    /// pictures from before an IDR ahead of the new sequence (POC restarts at 0).
    fn reorder_output(&mut self, surface: ffi::VASurfaceID, poc: i32) {
        self.output_key.insert(surface, (self.gop, poc));
        // Insertion sort the tail into place by (gop, POC).
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

    /// Return the smallest-POC output entry when the queue exceeds the reorder
    /// budget, moving it into `pending` (readback carry). Non-reference surfaces get
    /// freed after their readback; reference surfaces stay owned by the DPB.
    fn maybe_bump(&mut self, sps: &Sps) {
        // Bound the reorder depth to the DPB size; MPEG streams rarely reorder more.
        let budget = (sps.max_num_ref_frames as usize).max(1) + 1;
        while self.output_queue.len() > budget && self.pending.is_none() {
            if let Some(out) = self.output_queue.pop_front() {
                self.pending = Some(out);
            } else {
                break;
            }
        }
    }

    /// Fill and submit the VA picture: pic param + IQ matrix + per-slice params +
    /// slice data, wrapped in Begin/Render/End.
    #[allow(clippy::too_many_arguments)]
    fn submit_picture(
        &mut self,
        sps: &Sps,
        pps: &Pps,
        headers: &[SliceHeader],
        slice_nals: &[&h264parse::Nal],
        target: ffi::VASurfaceID,
        top_poc: i32,
        bottom_poc: i32,
        ref_idc: u8,
    ) -> Result<(), crate::va::VaError> {
        let display = self.display.as_ref().unwrap();
        let va = self.va.as_ref().unwrap();
        let context = &va.context;

        // --- Picture parameter buffer (§ va.h VAPictureParameterBufferH264) ---
        let mut pic = zeroed_pic_param();
        pic.CurrPic = ffi::VAPictureH264 {
            picture_id: target,
            frame_idx: headers[0].frame_num,
            flags: if ref_idc != 0 {
                ffi::VA_PICTURE_H264_SHORT_TERM_REFERENCE
            } else {
                0
            },
            TopFieldOrderCnt: top_poc,
            BottomFieldOrderCnt: bottom_poc,
            va_reserved: [0; ffi::VA_PADDING_LOW],
        };
        // ReferenceFrames[16] from the DPB (§8.2.4.2 ordering not required here; the
        // driver matches by picture_id/frame_idx). Unused slots = INVALID.
        pic.ReferenceFrames = [ffi::VAPictureH264::invalid(); 16];
        for (i, e) in self.dpb.iter().take(16).enumerate() {
            pic.ReferenceFrames[i] = ffi::VAPictureH264 {
                picture_id: e.surface,
                frame_idx: if e.long_term {
                    e.long_term_frame_idx as u32
                } else {
                    e.frame_num as u32
                },
                flags: if e.long_term {
                    ffi::VA_PICTURE_H264_LONG_TERM_REFERENCE
                } else {
                    ffi::VA_PICTURE_H264_SHORT_TERM_REFERENCE
                },
                TopFieldOrderCnt: e.top_poc,
                BottomFieldOrderCnt: e.bottom_poc,
                va_reserved: [0; ffi::VA_PADDING_LOW],
            };
        }
        pic.picture_width_in_mbs_minus1 = (sps.width_in_mbs() - 1) as u16;
        pic.picture_height_in_mbs_minus1 = (sps.height_in_mbs() - 1) as u16;
        pic.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8 as u8;
        pic.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8 as u8;
        pic.num_ref_frames = sps.max_num_ref_frames as u8;
        pic.seq_fields = pack_seq_fields(sps);
        pic.pic_init_qp_minus26 = pps.pic_init_qp_minus26 as i8;
        pic.chroma_qp_index_offset = pps.chroma_qp_index_offset as i8;
        pic.second_chroma_qp_index_offset = pps.second_chroma_qp_index_offset as i8;
        pic.pic_fields = pack_pic_fields(sps, pps, headers[0].field_pic_flag, ref_idc != 0);
        pic.frame_num = headers[0].frame_num as u16;

        let pic_buf = VaBuffer::new_struct(display, context, ffi::VAPictureParameterBufferType, &pic)?;

        // --- IQ matrix: flat 16 (§8.5.9 Flat_4x4_16 / Flat_8x8_16 default) ---
        let iq = ffi::VAIQMatrixBufferH264 {
            ScalingList4x4: [[16u8; 16]; 6],
            ScalingList8x8: [[16u8; 64]; 2],
            va_reserved: [0; ffi::VA_PADDING_LOW],
        };
        let iq_buf = VaBuffer::new_struct(display, context, ffi::VAIQMatrixBufferType, &iq)?;

        context.begin(target)?;
        context.render(&[pic_buf.id(), iq_buf.id()])?;

        // --- Per-slice: slice param + slice data ---
        // Keep the buffers alive until after End; store them so Drop runs later.
        let mut keep: Vec<VaBuffer> = Vec::with_capacity(slice_nals.len() * 2);
        for (sh, nal) in headers.iter().zip(slice_nals.iter()) {
            let (list0, list1, n0, n1) = self.build_ref_lists(sps, sh, top_poc.min(bottom_poc));
            let sp = self.build_slice_param(sh, nal.raw, &list0, &list1, n0, n1);
            let sp_buf =
                VaBuffer::new_struct(display, context, ffi::VASliceParameterBufferType, &sp)?;
            let data_buf = VaBuffer::new_data(display, context, nal.raw)?;
            context.render(&[sp_buf.id(), data_buf.id()])?;
            keep.push(sp_buf);
            keep.push(data_buf);
        }
        context.end()?;
        // keep + pic_buf + iq_buf drop here (after End): the driver copied them.
        Ok(())
    }

    /// Build RefPicList0/1 (§8.2.4.2 default construction + §8.2.4.3 modifications).
    /// Returns the two 32-entry VA lists and their active counts.
    fn build_ref_lists(
        &self,
        _sps: &Sps,
        sh: &SliceHeader,
        cur_poc: i32,
    ) -> ([ffi::VAPictureH264; 32], [ffi::VAPictureH264; 32], usize, usize) {
        let (mut list0, mut list1) = default_ref_lists(&self.dpb, sh.slice_type, cur_poc);

        // §8.2.4.3 modifications (reordering).
        apply_ref_list_mods(&mut list0, &sh.ref_list_mods_l0, &self.dpb, sh.frame_num as i32, _sps);
        if sh.slice_type == SliceType::B {
            apply_ref_list_mods(&mut list1, &sh.ref_list_mods_l1, &self.dpb, sh.frame_num as i32, _sps);
        }

        let n0 = (sh.num_ref_idx_l0_active_minus1 as usize + 1).min(32);
        let n1 = (sh.num_ref_idx_l1_active_minus1 as usize + 1).min(32);

        let va0 = to_va_list(&list0);
        let va1 = to_va_list(&list1);
        (va0, va1, n0, n1)
    }

    /// Fill a VASliceParameterBufferH264 from a slice header + its NAL.
    fn build_slice_param(
        &self,
        sh: &SliceHeader,
        raw_nal: &[u8],
        list0: &[ffi::VAPictureH264; 32],
        list1: &[ffi::VAPictureH264; 32],
        n0: usize,
        n1: usize,
    ) -> ffi::VASliceParameterBufferH264 {
        // slice_data_bit_offset: bits from the start of the NAL (including the 1-byte
        // header) to the start of slice_data(), counted in the *de-emulated* domain
        // as VA requires (va.h:3665). header_bits_in_rbsp counts bits after the NAL
        // header within the RBSP body; add the 8 header bits.
        let bit_offset = sh.header_bits_in_rbsp as u16 + 8;

        // Explicit weighted prediction (§7.3.3.2 → va.h): forward the parsed
        // table. pic_fields announces the PPS's weighted_pred_flag, so the
        // driver *uses* this table — an all-zero one multiplies every
        // prediction by 0 (green P/B frames, I frames untouched). References
        // without explicit weights get the identity default `1 << denom`,
        // offset 0 (§8.4.2.3.2).
        let mut w = WeightFill::default();
        if let Some(t) = &sh.pred_weights {
            w.fill(t, n0, n1);
        }

        ffi::VASliceParameterBufferH264 {
            slice_data_size: raw_nal.len() as u32,
            slice_data_offset: 0,
            slice_data_flag: ffi::VA_SLICE_DATA_FLAG_ALL,
            slice_data_bit_offset: bit_offset,
            first_mb_in_slice: sh.first_mb_in_slice as u16,
            slice_type: sh.slice_type.va_value(),
            direct_spatial_mv_pred_flag: sh.direct_spatial_mv_pred_flag as u8,
            num_ref_idx_l0_active_minus1: (n0.max(1) - 1) as u8,
            num_ref_idx_l1_active_minus1: (n1.max(1) - 1) as u8,
            cabac_init_idc: sh.cabac_init_idc as u8,
            slice_qp_delta: sh.slice_qp_delta as i8,
            disable_deblocking_filter_idc: sh.disable_deblocking_filter_idc as u8,
            slice_alpha_c0_offset_div2: sh.slice_alpha_c0_offset_div2 as i8,
            slice_beta_offset_div2: sh.slice_beta_offset_div2 as i8,
            RefPicList0: *list0,
            RefPicList1: *list1,
            luma_log2_weight_denom: w.luma_denom,
            chroma_log2_weight_denom: w.chroma_denom,
            luma_weight_l0_flag: w.l0_present,
            luma_weight_l0: w.luma_weight_l0,
            luma_offset_l0: w.luma_offset_l0,
            chroma_weight_l0_flag: w.l0_present,
            chroma_weight_l0: w.chroma_weight_l0,
            chroma_offset_l0: w.chroma_offset_l0,
            luma_weight_l1_flag: w.l1_present,
            luma_weight_l1: w.luma_weight_l1,
            luma_offset_l1: w.luma_offset_l1,
            chroma_weight_l1_flag: w.l1_present,
            chroma_weight_l1: w.chroma_weight_l1,
            chroma_offset_l1: w.chroma_offset_l1,
            va_reserved: [0; ffi::VA_PADDING_LOW],
        }
    }

    /// Reference marking + DPB insertion (§8.2.5). Sliding window (§8.2.5.3) for the
    /// implicit case; MMCO ops (§8.2.5.4) for the adaptive case (5 = reset;
    /// 1/2/3/4/6 best-effort).
    #[allow(clippy::too_many_arguments)]
    fn mark_and_insert(
        &mut self,
        sps: &Sps,
        sh: &SliceHeader,
        surface: ffi::VASurfaceID,
        frame_num: i32,
        poc: i32,
        top_poc: i32,
        bottom_poc: i32,
        is_idr: bool,
    ) {
        let mut long_term = false;
        let mut long_term_frame_idx = -1;

        if is_idr {
            // IDR marking (§8.2.5.1): the DPB was already cleared. long_term_reference
            // marks this IDR as long-term.
            if sh.long_term_reference_flag {
                long_term = true;
                long_term_frame_idx = 0;
            }
        } else if sh.adaptive_ref_pic_marking_mode_flag {
            self.apply_mmco(sps, &sh.mmco_ops, frame_num, poc);
            // MMCO 6 assigns the current picture a long-term idx.
            for op in &sh.mmco_ops {
                if op.op == 6 {
                    long_term = true;
                    long_term_frame_idx = op.arg1 as i32;
                }
            }
        } else {
            // Sliding window (§8.2.5.3): if the DPB is full of short-term refs, evict
            // the one with the smallest FrameNumWrap.
            let num_short = self.dpb.iter().filter(|e| !e.long_term).count();
            let num_long = self.dpb.iter().filter(|e| e.long_term).count();
            let max_refs = (sps.max_num_ref_frames as usize).max(1);
            if num_short + num_long >= max_refs {
                self.evict_smallest_frame_num_wrap(frame_num, sps);
            }
        }

        self.dpb.push(DpbEntry {
            surface,
            frame_num,
            frame_num_wrap: frame_num,
            poc,
            top_poc,
            bottom_poc,
            long_term,
            long_term_frame_idx,
        });
    }

    fn evict_smallest_frame_num_wrap(&mut self, cur_frame_num: i32, sps: &Sps) {
        let max_frame_num = sps.max_frame_num() as i32;
        let mut idx = None;
        let mut smallest = i32::MAX;
        for (i, e) in self.dpb.iter().enumerate() {
            if e.long_term {
                continue;
            }
            let wrap = if e.frame_num > cur_frame_num {
                e.frame_num - max_frame_num
            } else {
                e.frame_num
            };
            if wrap < smallest {
                smallest = wrap;
                idx = Some(i);
            }
        }
        if let Some(i) = idx {
            let e = self.dpb.remove(i);
            self.release_if_unreferenced(e.surface);
        }
    }

    /// Apply the adaptive MMCO op list (§8.2.5.4). Op 5 resets the DPB (§8.2.5.4.5).
    fn apply_mmco(&mut self, _sps: &Sps, ops: &[MmcoOp], frame_num: i32, _poc: i32) {
        for op in ops {
            match op.op {
                1 => {
                    // Unmark a short-term picture: PicNumX = CurrPicNum - (diff+1).
                    let pic_num_x = frame_num - (op.arg1 as i32 + 1);
                    self.remove_dpb_where(|e| !e.long_term && e.frame_num == pic_num_x);
                }
                2 => {
                    // Unmark a long-term picture by LongTermPicNum.
                    let ltpn = op.arg1 as i32;
                    self.remove_dpb_where(|e| e.long_term && e.long_term_frame_idx == ltpn);
                }
                3 => {
                    // Short-term → long-term (assign LongTermFrameIdx = arg2).
                    let pic_num_x = frame_num - (op.arg1 as i32 + 1);
                    let lt = op.arg2 as i32;
                    for e in &mut self.dpb {
                        if !e.long_term && e.frame_num == pic_num_x {
                            e.long_term = true;
                            e.long_term_frame_idx = lt;
                        }
                    }
                }
                4 => {
                    // MaxLongTermFrameIdx = arg1 - 1; drop longs above it.
                    let max_lt = op.arg1 as i32 - 1;
                    self.remove_dpb_where(|e| e.long_term && e.long_term_frame_idx > max_lt);
                }
                5 => {
                    // Reset (§8.2.5.4.5): clear DPB, reset POC — IDR-like.
                    self.reset_dpb();
                    self.poc.reset();
                }
                _ => {}
            }
        }
    }

    fn reset_dpb(&mut self) {
        let ids: Vec<_> = self.dpb.drain(..).map(|e| e.surface).collect();
        for id in ids {
            // Not `release_surface`: an IDR resets the DPB while earlier GOP
            // frames may still sit in the reorder queue awaiting readback.
            self.release_if_unreferenced(id);
        }
    }

    /// Remove every DPB entry matching `pred`, returning each removed surface to
    /// the free list once nothing else references it (the MMCO unmark ops — a
    /// bare `retain` would leak the surfaces out of the ledger permanently).
    fn remove_dpb_where(&mut self, pred: impl Fn(&DpbEntry) -> bool) {
        let mut removed = Vec::new();
        self.dpb.retain(|e| {
            if pred(e) {
                removed.push(e.surface);
                false
            } else {
                true
            }
        });
        for id in removed {
            self.release_if_unreferenced(id);
        }
    }

    /// Emit the pending decoded surface, dispatching on the output mode: zero-copy DMA-BUF
    /// export ([`emit_pending_zerocopy`](Self::emit_pending_zerocopy)) when built with
    /// [`new_zerocopy`](Self::new_zerocopy) and export has not fallen back, else CPU readback
    /// ([`emit_pending_readback`](Self::emit_pending_readback)). `false` = backpressure.
    fn emit_pending(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        if self.zerocopy.is_some() && !self.export_failed {
            return self.emit_pending_zerocopy(ctx);
        }
        self.emit_pending_readback(ctx)
    }

    /// Copy the pending decoded surface into a pool buffer as tight NV12 and emit.
    /// `false` = pool exhausted (backpressure).
    fn emit_pending_readback(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        // Refill pending from the output queue if empty (bumping already moved the
        // over-budget head; here we also pull when EOS/flush drained the budget).
        if self.pending.is_none() {
            self.pending = self.output_queue.pop_front();
        }
        let Some(out) = self.pending.take() else { return Ok(true) };

        // The raw display handle is a Copy pointer valid for the element's lifetime,
        // so taking it here (rather than a &self.display borrow) lets us keep
        // mutating self while a mapped image is live.
        let dpy = match &self.display {
            Some(d) => d.raw(),
            None => return Ok(true),
        };

        let width = out.width;
        let height = out.height;
        let need = (width * height + width * height.div_ceil(2)) as usize;

        // Map the decoded surface (derive → NV12, or GetImage fallback).
        let image = match MappedImage::acquire(dpy, out.surface, width, height) {
            Ok(im) => im,
            Err(e) => {
                self.warn(ctx, format!("vaapih264dec: readback failed: {e} — frame dropped"));
                self.recycle_output_surface(out.surface);
                return Ok(true);
            }
        };
        if image.num_planes() < 2 {
            self.warn(ctx, "vaapih264dec: mapped image is not 2-plane NV12 — dropped".into());
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
            // Put the picture back as pending (the image drops; re-acquired next spin).
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
                    "vaapih264dec: {width}x{height} NV12 frame needs {need} bytes but pool slots \
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
                    (F_PIXFMT, ValueDesc::Id(PIXFMT_NV12)),
                ],
            );
            self.announced = true;
        }

        // Repack driver strides → tight NV12: Y (width×height), then interleaved
        // CbCr (width×ceil(height/2)).
        let dst = buf.memory.as_mut_full();
        let y_len = (width * height) as usize;
        let c_rows = height.div_ceil(2);
        {
            let (y_dst, c_dst) = dst.split_at_mut(y_len);
            for row in 0..height {
                let Some(src) = image.row(0, row, width as usize) else {
                    drop(image);
                    self.recycle_output_surface(out.surface);
                    self.warn(ctx, "vaapih264dec: Y plane geometry overrun — dropped".into());
                    return Ok(true);
                };
                let d = &mut y_dst[(row * width) as usize..((row + 1) * width) as usize];
                d.copy_from_slice(src);
            }
            for row in 0..c_rows {
                let Some(src) = image.row(1, row, width as usize) else {
                    drop(image);
                    self.recycle_output_surface(out.surface);
                    self.warn(ctx, "vaapih264dec: CbCr plane geometry overrun — dropped".into());
                    return Ok(true);
                };
                let d = &mut c_dst[(row * width) as usize..((row + 1) * width) as usize];
                d.copy_from_slice(src);
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
    /// out-of-band, and emit a tiny `video/gpu` buffer (token + display geometry) the
    /// frame-slot sink pairs and imports via EGL. The surface is marked in flight and NOT
    /// recycled until the sink releases its token (drained in `process`). `false` = the
    /// presentation budget is full (backpressure). On export failure the mode latches back
    /// to readback (`export_failed`) and the frame is retried there — never lost.
    fn emit_pending_zerocopy(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        if self.surfaces_exhausted() && self.pending.is_none() && self.output_queue.is_empty() {
            return Ok(true); // nothing to emit
        }
        if self.pending.is_none() {
            self.pending = self.output_queue.pop_front();
        }
        let Some(out) = self.pending.take() else { return Ok(true) };

        let dpy = match &self.display {
            Some(d) => d.raw(),
            None => return Ok(true),
        };

        let exported = match ExportedSurface::export(dpy, out.surface) {
            Ok(e) => e,
            Err(e) => {
                self.export_failed = true;
                self.warn(
                    ctx,
                    format!(
                        "vaapih264dec: DMA-BUF export failed ({e}); falling back to CPU readback \
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

        let dup_fds = match exported.dup_fds() {
            Ok(f) => f,
            Err(e) => {
                self.warn(ctx, format!("vaapih264dec: dup(dma-buf fd) failed: {e} — frame dropped"));
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
            eprintln!("pfplay-ui: video path = zero-copy VA-API/EGL (vaapih264dec DMA-BUF export)");
        }

        let Some(mut buf) = ctx.try_alloc(SRC_PAD) else {
            if !self.alloc_stalled {
                self.alloc_stalled = true;
                let s = ctx.pool_stats();
                log!(&*ctx, Level::Debug, "alloc_stall", outstanding = s.outstanding, max_slots = s.max_slots);
            }
            for fd in dup_fds {
                // SAFETY: owned dup we just made; close exactly once.
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
            crop_x: 0,
            crop_y: 0,
        };
        let bytes = header.to_bytes();
        let dst = buf.memory.as_mut_full();
        if dst.len() < bytes.len() {
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
                    "vaapih264dec: video/gpu header needs {} bytes but pool slots hold {}",
                    bytes.len(),
                    dst.len()
                ),
            });
        }
        dst[..bytes.len()].copy_from_slice(&bytes);
        buf.memory.set_len(bytes.len());
        buf.pts = out.pts;
        buf.duration = out.duration;

        let frame = GpuFrame {
            token,
            fds: dup_fds,
            drm_format: exported.drm_format,
            drm_modifier: exported.drm_modifier,
            coded_w: exported.coded_w,
            coded_h: exported.coded_h,
            disp_w: out.width,
            disp_h: out.height,
            crop_x: 0,
            crop_y: 0,
            planes: exported.planes.clone(),
        };
        drop(exported); // closes the originals; the importer has the dups
        if let Some(ch) = &self.zerocopy {
            ch.push(frame);
        }
        self.in_flight.insert(out.surface, token);

        log!(&*ctx, Level::Trace, "gpu_frame", pts = buf.pts, token = token);
        ctx.out(SRC_PAD).push(buf);

        // In flight — do NOT recycle; only clear the output-order key.
        self.output_key.remove(&out.surface);
        Ok(true)
    }

    /// Drain tokens the sink has presented and return each token's surface to the free pool
    /// (if nothing else references it). Called once per `process` in zero-copy mode.
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

    /// After a picture is output, free its surface *only if it is not a live DPB
    /// reference*. Reference surfaces are freed when evicted.
    fn recycle_output_surface(&mut self, surface: ffi::VASurfaceID) {
        self.output_key.remove(&surface);
        self.release_if_unreferenced(surface);
    }
}

impl Element for VaapiH264Dec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.ensure_display(ctx) {
            // No device: consume and drop input so the graph does not stall.
            while inputs.pop().is_some() {}
            return Ok(Flow::Ok);
        }
        // Zero-copy: reclaim surfaces the sink has presented before anything else — what
        // keeps the export path from starving on the presentation budget.
        self.drain_presentation_releases();
        loop {
            // Emit any staged output; stop pulling input if the pool is dry.
            if !self.emit_over_budget(ctx)? {
                return Ok(Flow::Ok);
            }
            // A free surface must exist before the next AU is consumed: a
            // momentarily-exhausted surface pool is *backpressure* (leave the
            // input in the ring; emitting is what frees surfaces), never a
            // drop. Under a realtime-pacing sink the DPB + reorder queue pin
            // every surface — dropping here loses all AUs until the next IDR
            // and the video freezes while audio plays on.
            while self.surfaces_exhausted() {
                if self.pending.is_none() && self.output_queue.is_empty() {
                    break; // held by the DPB alone — decode_au's drop is the fuse
                }
                if !self.emit_pending(ctx)? {
                    return Ok(Flow::Ok); // pool dry — re-cranked on slot return
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
                // Drop refs + carries + POC; keep VA context (dims unchanged).
                // Output entries drain *first* (so reset_dpb sees their surfaces
                // unreferenced and actually frees them — clearing the queue after
                // would leak every queued surface on each seek).
                let outs: Vec<_> = self
                    .pending
                    .take()
                    .into_iter()
                    .chain(self.output_queue.drain(..))
                    .map(|o| o.surface)
                    .collect();
                self.output_key.clear();
                // Zero-copy: a flush discards the pre-seek pictures, so in-flight surfaces
                // can be reclaimed (the GUI's slot is overwritten by post-seek frames; a
                // discarded frame's brief surface reuse is never displayed). The sink still
                // owns + closes its dup'd fds; clearing here only frees the *surface*.
                self.drain_presentation_releases();
                let in_flight: Vec<_> = self.in_flight.keys().copied().collect();
                self.in_flight.clear();
                for id in outs.into_iter().chain(in_flight) {
                    self.release_if_unreferenced(id);
                }
                self.reset_dpb();
                self.poc.reset();
                self.alloc_stalled = false;
                // announced stays true: dims don't change across a seek.
            }
            Event::Eos => {
                // Bump every remaining picture out in POC order.
                self.drain_presentation_releases();
                while self.pending.is_some() || !self.output_queue.is_empty() {
                    if !self.emit_pending(ctx)? {
                        break; // pool dry; the scheduler re-cranks
                    }
                    self.drain_presentation_releases();
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // RAII: dropping va/display destroys context/surfaces/config/display.
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

impl VaapiH264Dec {
    /// Emit staged output beyond the reorder budget (steady-state drain in
    /// process()). `false` = pool dry.
    fn emit_over_budget(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        // Emit the carry first.
        if self.pending.is_some() && !self.emit_pending(ctx)? {
            return Ok(false);
        }
        Ok(true)
    }
}

// --- free-function helpers ------------------------------------------------------------

/// The VA slice-parameter weight arrays, prepared from a parsed
/// [`PredWeightTable`]. Defaults (all zero, `l0/l1_present = 0`) are what a
/// slice *without* an explicit table submits — the driver then ignores them
/// (non-weighted P, or implicit-B, where it derives weights from POC itself).
#[derive(Default)]
struct WeightFill {
    luma_denom: u8,
    chroma_denom: u8,
    l0_present: u8,
    l1_present: u8,
    luma_weight_l0: [i16; 32],
    luma_offset_l0: [i16; 32],
    chroma_weight_l0: [[i16; 2]; 32],
    chroma_offset_l0: [[i16; 2]; 32],
    luma_weight_l1: [i16; 32],
    luma_offset_l1: [i16; 32],
    chroma_weight_l1: [[i16; 2]; 32],
    chroma_offset_l1: [[i16; 2]; 32],
}

impl WeightFill {
    /// Expand the parsed table over the active reference counts: explicit
    /// entries verbatim, absent ones as the §8.4.2.3.2 identity default
    /// (`1 << denom`, offset 0) so untouched references predict unweighted.
    fn fill(&mut self, t: &PredWeightTable, n0: usize, n1: usize) {
        self.luma_denom = t.luma_log2_weight_denom as u8;
        self.chroma_denom = t.chroma_log2_weight_denom as u8;
        let ld = 1i16 << t.luma_log2_weight_denom.min(7);
        let cd = 1i16 << t.chroma_log2_weight_denom.min(7);

        self.l0_present = 1;
        for i in 0..n0.min(32) {
            let e = t.l0.get(i).copied().unwrap_or_default();
            let (lw, lo) = e.luma.unwrap_or((ld as i32, 0));
            self.luma_weight_l0[i] = lw as i16;
            self.luma_offset_l0[i] = lo as i16;
            let ch = e.chroma.unwrap_or([(cd as i32, 0); 2]);
            for c in 0..2 {
                self.chroma_weight_l0[i][c] = ch[c].0 as i16;
                self.chroma_offset_l0[i][c] = ch[c].1 as i16;
            }
        }
        if !t.l1.is_empty() {
            self.l1_present = 1;
            for i in 0..n1.min(32) {
                let e = t.l1.get(i).copied().unwrap_or_default();
                let (lw, lo) = e.luma.unwrap_or((ld as i32, 0));
                self.luma_weight_l1[i] = lw as i16;
                self.luma_offset_l1[i] = lo as i16;
                let ch = e.chroma.unwrap_or([(cd as i32, 0); 2]);
                for c in 0..2 {
                    self.chroma_weight_l1[i][c] = ch[c].0 as i16;
                    self.chroma_offset_l1[i][c] = ch[c].1 as i16;
                }
            }
        }
    }
}

/// Peek the pps_id of a slice NAL (first_mb ue, slice_type ue, pps_id ue).
fn peek_pps_id(nal: &[u8]) -> Option<u32> {
    if nal.len() < 2 {
        return None;
    }
    // De-emulate a small prefix is unnecessary for the first few ue codes; parse
    // directly on the body (emulation bytes only appear after 00 00, unlikely this
    // early). Use the full de-emulation for safety via h264parse's reader.
    let body = de_emulate_prefix(&nal[1..]);
    let mut r = h264parse::BitReader::new(&body);
    let _first_mb = r.ue();
    let _slice_type = r.ue();
    Some(r.ue())
}

/// De-emulate just enough of an RBSP body for the pps_id peek.
fn de_emulate_prefix(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len().min(16));
    let mut zeros = 0;
    let mut i = 0;
    while i < body.len() && out.len() < 16 {
        let b = body[i];
        if zeros >= 2 && b == 0x03 && i + 1 < body.len() && body[i + 1] <= 0x03 {
            zeros = 0;
            i += 1;
            continue;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
        i += 1;
    }
    out
}

fn zeroed_pic_param() -> ffi::VAPictureParameterBufferH264 {
    // SAFETY: the struct is plain POD (integers + arrays of integers); all-zero is
    // a valid (empty) initial state we then fill field by field.
    unsafe { std::mem::zeroed() }
}

/// Pack seq_fields bits (va.h VAPictureParameterBufferH264 union layout).
fn pack_seq_fields(sps: &Sps) -> u32 {
    let mut v = 0u32;
    v |= (sps.chroma_format_idc & 0x3) << 0;
    v |= (sps.separate_colour_plane_flag as u32) << 2; // residual_colour_transform_flag
    v |= (sps.gaps_in_frame_num_value_allowed_flag as u32) << 3;
    v |= (sps.frame_mbs_only_flag as u32) << 4;
    v |= (sps.mb_adaptive_frame_field_flag as u32) << 5;
    v |= (sps.direct_8x8_inference_flag as u32) << 6;
    // bit 7 = MinLumaBiPredSize8x8 (level-derived; 0 is safe).
    v |= (sps.log2_max_frame_num_minus4 & 0xf) << 8;
    v |= (sps.pic_order_cnt_type & 0x3) << 12;
    v |= (sps.log2_max_pic_order_cnt_lsb_minus4 & 0xf) << 14;
    v |= (sps.delta_pic_order_always_zero_flag as u32) << 18;
    v
}

/// Pack pic_fields bits (va.h VAPictureParameterBufferH264 union layout).
fn pack_pic_fields(sps: &Sps, pps: &Pps, field_pic: bool, is_ref: bool) -> u32 {
    let _ = sps;
    let mut v = 0u32;
    v |= (pps.entropy_coding_mode_flag as u32) << 0;
    v |= (pps.weighted_pred_flag as u32) << 1;
    v |= (pps.weighted_bipred_idc & 0x3) << 2;
    v |= (pps.transform_8x8_mode_flag as u32) << 4;
    v |= (field_pic as u32) << 5;
    v |= (pps.constrained_intra_pred_flag as u32) << 6;
    v |= (pps.bottom_field_pic_order_in_frame_present_flag as u32) << 7; // pic_order_present_flag
    v |= (pps.deblocking_filter_control_present_flag as u32) << 8;
    v |= (pps.redundant_pic_cnt_present_flag as u32) << 9;
    v |= (is_ref as u32) << 10; // reference_pic_flag
    v
}

/// Convert a slice of DPB entries into a 32-entry VA ref list (unused = INVALID).
fn to_va_list(list: &[&DpbEntry]) -> [ffi::VAPictureH264; 32] {
    let mut out = [ffi::VAPictureH264::invalid(); 32];
    for (i, e) in list.iter().take(32).enumerate() {
        out[i] = ffi::VAPictureH264 {
            picture_id: e.surface,
            frame_idx: if e.long_term {
                e.long_term_frame_idx as u32
            } else {
                e.frame_num as u32
            },
            flags: if e.long_term {
                ffi::VA_PICTURE_H264_LONG_TERM_REFERENCE
            } else {
                ffi::VA_PICTURE_H264_SHORT_TERM_REFERENCE
            },
            TopFieldOrderCnt: e.top_poc,
            BottomFieldOrderCnt: e.bottom_poc,
            va_reserved: [0; ffi::VA_PADDING_LOW],
        };
    }
    out
}

/// Default reference-list construction (§8.2.4.2): the initial RefPicList0/1
/// before any `ref_pic_list_modification`. P/SP: short-term by descending
/// FrameNumWrap then long-term ascending (§8.2.4.2.1). B: list0 = short-term with
/// POC<cur descending, then POC>cur ascending, then long-term; list1 the mirror,
/// with the §8.2.4.2.4 first-two swap when the two lists coincide (§8.2.4.2.3-4).
fn default_ref_lists<'a>(
    dpb: &'a [DpbEntry],
    slice_type: SliceType,
    cur_poc: i32,
) -> (Vec<&'a DpbEntry>, Vec<&'a DpbEntry>) {
    let mut list0: Vec<&DpbEntry> = Vec::new();
    let mut list1: Vec<&DpbEntry> = Vec::new();
    match slice_type {
        SliceType::P | SliceType::Sp => {
            let mut short: Vec<&DpbEntry> = dpb.iter().filter(|e| !e.long_term).collect();
            short.sort_by(|a, b| b.frame_num_wrap.cmp(&a.frame_num_wrap));
            let mut long: Vec<&DpbEntry> = dpb.iter().filter(|e| e.long_term).collect();
            long.sort_by_key(|e| e.long_term_frame_idx);
            list0.extend(short);
            list0.extend(long);
        }
        SliceType::B => {
            let mut less: Vec<&DpbEntry> =
                dpb.iter().filter(|e| !e.long_term && e.poc < cur_poc).collect();
            less.sort_by(|a, b| b.poc.cmp(&a.poc));
            let mut greater: Vec<&DpbEntry> =
                dpb.iter().filter(|e| !e.long_term && e.poc > cur_poc).collect();
            greater.sort_by_key(|e| e.poc);
            let mut long: Vec<&DpbEntry> = dpb.iter().filter(|e| e.long_term).collect();
            long.sort_by_key(|e| e.long_term_frame_idx);

            list0.extend(less.iter().copied());
            list0.extend(greater.iter().copied());
            list0.extend(long.iter().copied());

            list1.extend(greater.iter().copied());
            list1.extend(less.iter().copied());
            list1.extend(long.iter().copied());
            if list1.len() > 1
                && list0.len() == list1.len()
                && list0.iter().zip(list1.iter()).all(|(a, b)| a.surface == b.surface)
            {
                list1.swap(0, 1);
            }
        }
        SliceType::I | SliceType::Si => {}
    }
    (list0, list1)
}

/// Apply §8.2.4.3 ref-list modifications by reordering entries by PicNum /
/// LongTermPicNum. Best-effort for our subset (short-term ops 0/1; long-term op 2).
fn apply_ref_list_mods<'a>(
    list: &mut Vec<&'a DpbEntry>,
    mods: &[RefListMod],
    dpb: &'a [DpbEntry],
    cur_frame_num: i32,
    sps: &Sps,
) {
    if mods.is_empty() {
        return;
    }
    let max_frame_num = sps.max_frame_num() as i32;
    let mut pred = cur_frame_num;
    let mut refined: Vec<&DpbEntry> = Vec::new();
    for m in mods {
        match m.op {
            0 | 1 => {
                // Short-term: compute picNum target relative to pred.
                let abs_diff = m.value as i32 + 1;
                let mut pic_num = if m.op == 0 { pred - abs_diff } else { pred + abs_diff };
                if pic_num < 0 {
                    pic_num += max_frame_num;
                } else if pic_num >= max_frame_num {
                    pic_num -= max_frame_num;
                }
                pred = pic_num;
                if let Some(e) = dpb.iter().find(|e| !e.long_term && e.frame_num == pic_num) {
                    refined.push(e);
                }
            }
            2 => {
                // Long-term: value = long_term_pic_num.
                let lt = m.value as i32;
                if let Some(e) = dpb.iter().find(|e| e.long_term && e.long_term_frame_idx == lt) {
                    refined.push(e);
                }
            }
            _ => {}
        }
    }
    if !refined.is_empty() {
        // Prepend the refined (reordered) entries; append the rest of the default
        // list not already present. A faithful §8.2.4.3 shifts in place, but for our
        // subset (P/B with short-term refs) this yields the same active window.
        let mut new_list = refined;
        for e in list.iter() {
            if !new_list.iter().any(|x| x.surface == e.surface) {
                new_list.push(e);
            }
        }
        *list = new_list;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(surface: u32, frame_num: i32, poc: i32) -> DpbEntry {
        DpbEntry {
            surface,
            frame_num,
            frame_num_wrap: frame_num,
            poc,
            top_poc: poc,
            bottom_poc: poc,
            long_term: false,
            long_term_frame_idx: -1,
        }
    }

    #[test]
    fn p_slice_ref_list_is_descending_frame_num_wrap() {
        // §8.2.4.2.1: short-term refs by descending FrameNumWrap.
        let dpb = vec![entry(10, 0, 0), entry(11, 2, 4), entry(12, 1, 2)];
        let (list0, list1) = default_ref_lists(&dpb, SliceType::P, 6);
        let order: Vec<i32> = list0.iter().map(|e| e.frame_num_wrap).collect();
        assert_eq!(order, vec![2, 1, 0], "P list0 must descend by FrameNumWrap");
        assert!(list1.is_empty(), "P slices have no list1");
    }

    #[test]
    fn b_slice_ref_lists_split_by_poc() {
        // §8.2.4.2.3: list0 = POC<cur desc then POC>cur asc; list1 = the mirror.
        let dpb = vec![
            entry(1, 0, 0),  // POC 0  (< cur)
            entry(2, 1, 2),  // POC 2  (< cur)
            entry(3, 2, 8),  // POC 8  (> cur)
            entry(4, 3, 6),  // POC 6  (> cur)
        ];
        let cur_poc = 4;
        let (list0, list1) = default_ref_lists(&dpb, SliceType::B, cur_poc);
        let l0: Vec<i32> = list0.iter().map(|e| e.poc).collect();
        let l1: Vec<i32> = list1.iter().map(|e| e.poc).collect();
        // list0: [2, 0] (POC<cur descending) then [6, 8] (POC>cur ascending).
        assert_eq!(l0, vec![2, 0, 6, 8]);
        // list1: [6, 8] (POC>cur ascending) then [2, 0] (POC<cur descending).
        assert_eq!(l1, vec![6, 8, 2, 0]);
    }

    #[test]
    fn output_order_survives_idr_poc_reset() {
        // POC restarts at 0 on an IDR (§8.2.1). Raw-POC ordering would sort the
        // new GOP's pictures ahead of the previous GOP's still-queued tail —
        // the old scene would flash back after the cut. The (gop, poc) key must
        // keep queue order [old GOP..., new GOP...].
        let mut dec = VaapiH264Dec::new();
        let push = |dec: &mut VaapiH264Dec, surface: u32, poc: i32| {
            dec.output_queue.push_back(PendingOut {
                surface,
                pts: Timestamp::ZERO,
                duration: Timestamp::ZERO,
                width: 16,
                height: 16,
                coded_w: 16,
                coded_h: 16,
            });
            dec.reorder_output(surface, poc);
        };
        // Old GOP tail, decode order with a B reorder (POC 202 before 200).
        push(&mut dec, 1, 202);
        push(&mut dec, 2, 200);
        // IDR: new coded video sequence, POC restarts.
        dec.gop = dec.gop.wrapping_add(1);
        push(&mut dec, 3, 0);
        push(&mut dec, 4, 2);
        let order: Vec<u32> = dec.output_queue.iter().map(|o| o.surface).collect();
        assert_eq!(
            order,
            vec![2, 1, 3, 4],
            "old GOP (POC-sorted) must drain before the new GOP's POC-0 IDR"
        );
    }
}
