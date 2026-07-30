//! `mosaic` — the NVR's live wall: N decoded I420 camera feeds composited into
//! one I420 grid frame (spec: Aggregation — fan-in heads; the `MkvMuxN` static
//! sink-pad convention while dynamic sink growth is core-side pending work).
//!
//! Model: a **latest-frame compositor**, not an aligner. Each sink pad's most
//! recent frame is retained (a refcount, no copy); pad 0 is the metronome —
//! every new frame there composites the wall and emits one output frame at that
//! frame's pts (the SDL sink paces on it downstream). Cameras that lag or died
//! simply repeat their last picture; cells never seen stay black. Aggregation
//! by alignment (`InputPolicy::All`) is deliberately NOT used: a live wall must
//! not stall because one camera stopped.
//!
//! Scaling is unweighted area averaging per plane (box reconstruction — each
//! output pixel is the mean of its source footprint `[x·W/w, (x+1)·W/w)`; see
//! Wolberg, *Digital Image Warping*, §5.3 box/Fourier-window reconstruction),
//! integer-only and allocation-free. For downscale it is the cheap filter that
//! does not shimmer (point sampling aliases bar edges and text); for identity
//! or upscale the footprint degenerates to one sample = nearest. GPU-quality
//! resampling belongs to the renderer roadmap, not this element.
//!
//! Input format: each pad's camera geometry is fixed at construction — the
//! app already parsed every camera's SPS for the recorder branch, so the wall's
//! geometry is app policy (spec: no-bins). Both the **crop** rectangle and the
//! **coded** (macroblock-aligned, H.264 §7.4.2.1.1) plane dimensions matter:
//! the decoder emits coded-size planes (640×360 content arrives as 640×368
//! planes), so sampling must index with the coded stride but stay inside the
//! crop — mixing them up displaces every chroma read (measured: bars fine,
//! horizontal features tinted green/brown — the classic shifted-chroma look).
//! A mid-stream resolution change is not supported in v1 (the recorder rotates
//! on it; the wall would need a re-announce).

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::memory::Memory;
use profluens_core::time::Timestamp;

/// The static pad table's sink budget (grids up to 4×4).
pub const MAX_CAMS: usize = 16;

// `video/raw` vocabulary — literals, the same interning convention as the
// decoders and the SDL sink (ids line up by string).
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
const PIXFMT_I420: &str = "i420";

static PIXFMT_VALUES: [ValueDesc; 1] = [ValueDesc::Id(PIXFMT_I420)];
static VIDEO_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
];
static VIDEO_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &VIDEO_FIELDS }];

const fn sink_pad(name: &'static str) -> PadDesc {
    // `dynamic`: the upstream decoder announces its dimensions at runtime.
    PadDesc { name, direction: Direction::Sink, offers: &VIDEO_OFFERS, dynamic: true, validate: None }
}

static PADS: [PadDesc; MAX_CAMS + 1] = [
    sink_pad("sink_0"),
    sink_pad("sink_1"),
    sink_pad("sink_2"),
    sink_pad("sink_3"),
    sink_pad("sink_4"),
    sink_pad("sink_5"),
    sink_pad("sink_6"),
    sink_pad("sink_7"),
    sink_pad("sink_8"),
    sink_pad("sink_9"),
    sink_pad("sink_10"),
    sink_pad("sink_11"),
    sink_pad("sink_12"),
    sink_pad("sink_13"),
    sink_pad("sink_14"),
    sink_pad("sink_15"),
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &VIDEO_OFFERS,
        dynamic: true, // grid dimensions announced at runtime
        validate: None,
    },
];

const SRC: PadId = PadId(MAX_CAMS as u32);

static DESC: ElementDesc = ElementDesc {
    name: "mosaic",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active,
    // Any: compose on whatever arrived — a live wall never waits for alignment.
    inputs: InputPolicy::Any,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: true,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// One camera's fixed geometry (see module docs): the visible crop and the
/// decoder's coded plane dimensions (H.264 §7.4.2.1.1 MB alignment).
#[derive(Clone, Copy, Debug)]
pub struct CamGeom {
    pub crop_w: u32,
    pub crop_h: u32,
    pub coded_w: u32,
    pub coded_h: u32,
}

pub struct Mosaic {
    /// Per-pad camera geometry, fixed at construction (see module docs).
    cams: Vec<CamGeom>,
    cell_w: u32,
    cell_h: u32,
    cols: u32,
    rows: u32,
    /// The latest decoded frame per pad — a held refcount, replaced on arrival.
    latest: Vec<Option<Memory>>,
    announced: bool,
    /// Composed frames emitted / metronome ticks skipped pool-dry (taps cover
    /// the rest; these are the wall's own health counters).
    pub emitted: u64,
    pub skipped_dry: u64,
}

impl Mosaic {
    /// A wall for `cams` (per-pad geometry, pad order) rendered into
    /// `cell_w`×`cell_h` cells on a near-square grid.
    pub fn new(cams: Vec<CamGeom>, cell_w: u32, cell_h: u32) -> Mosaic {
        let n = cams.len();
        assert!((1..=MAX_CAMS).contains(&n), "mosaic supports 1..={MAX_CAMS} cams, got {n}");
        // Near-square grid: cols = ⌈√n⌉, rows = ⌈n/cols⌉.
        let cols = (n as f64).sqrt().ceil() as u32;
        let rows = (n as u32).div_ceil(cols);
        // Even cell dimensions keep the 4:2:0 chroma grid aligned per cell.
        let cell_w = cell_w & !1;
        let cell_h = cell_h & !1;
        Mosaic {
            latest: vec![None; n],
            cams,
            cell_w,
            cell_h,
            cols,
            rows,
            announced: false,
            emitted: 0,
            skipped_dry: 0,
        }
    }

    pub fn out_dims(&self) -> (u32, u32) {
        (self.cols * self.cell_w, self.rows * self.cell_h)
    }

    /// Area-average resample of one plane into a cell of the output plane
    /// (unweighted box reconstruction — Wolberg §5.3): each output pixel is the
    /// integer mean of its source footprint `[x·sw/dw, (x+1)·sw/dw)` ×
    /// `[y·sh/dh, (y+1)·sh/dh)` (degenerates to nearest at identity/upscale).
    /// `src` is a plane of row stride `sstride` from which the `sw`×`sh` crop
    /// (top-left origin) is sampled; `dst` is the whole output plane of stride
    /// `dstride`, the cell starting at (`dx`, `dy`) sized `dw`×`dh`.
    #[allow(clippy::too_many_arguments)]
    fn scale_plane(
        src: &[u8],
        sstride: usize,
        sw: usize,
        sh: usize,
        dst: &mut [u8],
        dstride: usize,
        dx: usize,
        dy: usize,
        dw: usize,
        dh: usize,
    ) {
        for y in 0..dh {
            let sy0 = y * sh / dh;
            let sy1 = ((y + 1) * sh / dh).max(sy0 + 1);
            let drow = &mut dst[(dy + y) * dstride + dx..(dy + y) * dstride + dx + dw];
            for (x, d) in drow.iter_mut().enumerate() {
                let sx0 = x * sw / dw;
                let sx1 = ((x + 1) * sw / dw).max(sx0 + 1);
                let mut acc = 0u32;
                for sy in sy0..sy1 {
                    let srow = &src[sy * sstride + sx0..sy * sstride + sx1];
                    for &p in srow {
                        acc += p as u32;
                    }
                }
                *d = (acc / ((sy1 - sy0) * (sx1 - sx0)) as u32) as u8;
            }
        }
    }

    /// Composite every retained frame into `out` (an I420 canvas, planes
    /// pre-set to black: Y=16, U=V=128 — ITU-R BT.601 nominal black).
    fn composite(&self, out: &mut [u8]) {
        let (ow, oh) = self.out_dims();
        let (ow, oh) = (ow as usize, oh as usize);
        let (y_len, c_len) = (ow * oh, ow / 2 * (oh / 2));
        for (i, mem) in self.latest.iter().enumerate() {
            let Some(mem) = mem else { continue };
            let g = self.cams[i];
            // Plane offsets index with the CODED dimensions (the decoder's
            // MB-aligned layout); sampling stays inside the crop (module docs —
            // the shifted-chroma bug lived exactly here).
            let (pw, ph) = (g.coded_w as usize, g.coded_h as usize);
            let (sw, sh) = (g.crop_w as usize, g.crop_h as usize);
            let data = mem.data();
            if data.len() < pw * ph * 3 / 2 || sw > pw || sh > ph {
                continue; // malformed frame — leave the cell as-is
            }
            let (cx, cy) = (
                (i as u32 % self.cols * self.cell_w) as usize,
                (i as u32 / self.cols * self.cell_h) as usize,
            );
            let (cw, ch) = (self.cell_w as usize, self.cell_h as usize);
            let (sy_plane, rest) = data.split_at(pw * ph);
            let (su_plane, sv_plane) = rest.split_at(pw / 2 * (ph / 2));
            let (dy_plane, rest) = out.split_at_mut(y_len);
            let (du_plane, dv_plane) = rest.split_at_mut(c_len);
            Self::scale_plane(sy_plane, pw, sw, sh, dy_plane, ow, cx, cy, cw, ch);
            Self::scale_plane(su_plane, pw / 2, sw / 2, sh / 2, du_plane, ow / 2, cx / 2, cy / 2, cw / 2, ch / 2);
            Self::scale_plane(sv_plane, pw / 2, sw / 2, sh / 2, dv_plane, ow / 2, cx / 2, cy / 2, cw / 2, ch / 2);
        }
    }
}

impl Element for Mosaic {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Feed: retain each pad's newest frame. With one cam the scheduler
        // delivers through `inputs`; with several, per-pad via `take_input_on`
        // (the MkvMuxN dual-path convention).
        let mut tick = false; // pad 0 delivered → compose after the sweep
        let mut tick_pts = Timestamp::NONE;
        if self.cams.len() == 1 {
            while let Some(buf) = inputs.pop() {
                tick = true;
                tick_pts = buf.pts;
                self.latest[0] = Some(buf.memory);
            }
        }
        for i in 0..self.cams.len() {
            let mut batch = ctx.take_input_on(PadId(i as u32));
            while let Some(buf) = batch.pop_front() {
                if i == 0 {
                    tick = true;
                    tick_pts = buf.pts;
                }
                self.latest[i] = Some(buf.memory);
            }
            ctx.recycle_input(batch);
        }
        if !tick {
            return Ok(Flow::Ok);
        }

        if !self.announced {
            let (ow, oh) = self.out_dims();
            ctx.announce_format(
                SRC,
                FAMILY,
                &[
                    (F_WIDTH, ValueDesc::Int(ow as i64)),
                    (F_HEIGHT, ValueDesc::Int(oh as i64)),
                    (F_PIXFMT, ValueDesc::Id(PIXFMT_I420)),
                ],
            );
            self.announced = true;
        }

        let (ow, oh) = self.out_dims();
        let need = (ow * oh) as usize * 3 / 2;
        let Some(mut buf) = ctx.try_alloc(SRC) else {
            // Live wall: a dropped tick is a skipped repaint, never a stall.
            self.skipped_dry += 1;
            return Ok(Flow::Ok);
        };
        if buf.memory.capacity() < need {
            return Err(Error::Resource(format!(
                "mosaic: {ow}x{oh} I420 canvas needs {need} B, pool slot holds {} — raise the pool",
                buf.memory.capacity()
            )));
        }
        {
            let dst = &mut buf.memory.as_mut_full()[..need];
            // Nominal black canvas (BT.601 Y=16, chroma neutral 128).
            let ylen = (ow * oh) as usize;
            dst[..ylen].fill(16);
            dst[ylen..].fill(128);
            self.composite(dst);
        }
        buf.memory.set_len(need);
        if self.emitted == 0 {
            if let Some(path) = std::env::var_os("PF_MOSAIC_DUMP") {
                // Diagnostic escape hatch (env-gated, first composed frame
                // only): raw I420 dump for offline inspection — a debug
                // one-shot, not a streaming path (clippy.toml exception).
                #[allow(clippy::disallowed_methods)]
                let _ = std::fs::write(path, buf.memory.data());
            }
        }
        buf.pts = tick_pts;
        ctx.out(SRC).push(buf);
        self.emitted += 1;
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}
