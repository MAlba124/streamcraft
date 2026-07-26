//! Shared plumbing for the VA-API encoder elements (`vaapih264enc` /
//! `vaapih265enc` / `vaapivp8enc`): the per-codec elements own their parameter
//! buffers and bitstream headers; everything codec-neutral lives here.
//!
//! ## The encode cycle (one frame)
//! 1. [`EncEngine::upload`] copies the tight-packed raw frame (NV12 or I420) into
//!    the NV12 **input surface** via [`UploadImage`] (derive on iHD → zero-copy
//!    map), replicating edge pixels into the alignment padding so the padded
//!    macroblocks predict cheaply instead of encoding garbage.
//! 2. The element builds its codec parameter buffers (sequence/picture/slice +
//!    packed headers) against the **current reconstruction surface** and a coded
//!    buffer.
//! 3. [`EncEngine::submit`] runs `vaBeginPicture(input)` → `vaRenderPicture` →
//!    `vaEndPicture` → `vaSyncSurface(input)` — synchronous per frame, the same
//!    accepted device-wait the decode element makes (`SchedHint::Active`, its own
//!    thread; hardware encode is single-digit ms per 1080p frame).
//! 4. The element drains the coded buffer ([`va::Buffer::read_coded`]) and emits.
//!
//! ## Reconstruction surfaces
//! Two, rotated: the frame being encoded reconstructs into `recon_cur()`, and
//! P-frames predict from `recon_ref()` (the previous frame's reconstruction) —
//! the IDR/P single-reference GOP every element here uses. After a successful
//! frame the element calls [`EncEngine::advance_recon`] to swap the pair.

use streamcraft_core::ctx::Ctx;
use streamcraft_core::format::{FixedFormat, Value};

use crate::ffi;
use crate::va::{self, Config, Context, Display, Surfaces, UploadImage, VaResult};

// `video/raw` vocabulary (same literals as the decoder's announce and the video
// crate's offers — the pipeline interns by string).
pub(crate) const RAW_FAMILY: &str = "video/raw";
pub(crate) const F_WIDTH: &str = "width";
pub(crate) const F_HEIGHT: &str = "height";
pub(crate) const F_PIXFMT: &str = "pixfmt";
pub(crate) const F_FPS: &str = "fps";

/// The raw input layouts the encoders accept (both 8-bit 4:2:0; the hardware
/// input surface is NV12, I420 is interleaved during upload).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RawPixFmt {
    Nv12,
    I420,
}

impl RawPixFmt {
    pub(crate) fn from_caps_name(s: &str) -> Option<RawPixFmt> {
        match s {
            "nv12" => Some(RawPixFmt::Nv12),
            "i420" => Some(RawPixFmt::I420),
            _ => None,
        }
    }
}

/// The negotiated raw-video input format an encoder configures from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct InFormat {
    pub width: u32,
    pub height: u32,
    pub pixfmt: RawPixFmt,
    /// Frames per second as (num, den); (0, _) when the upstream announced none.
    pub fps: (u32, u32),
}

impl InFormat {
    /// Tight-packed byte size of one frame (the size the upstream producers emit:
    /// 4:2:0 with ceil-rounded chroma planes, the `video/geometry` convention).
    pub(crate) fn frame_bytes(&self) -> usize {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        w * h + 2 * cw * ch
    }
}

/// Read a fixated/announced `video/raw` format into an [`InFormat`] — the same
/// field-by-field read the SDL3 sink does. `None` while required fields are
/// missing or the pixfmt is one we do not accept.
pub(crate) fn read_in_format(ctx: &Ctx, f: &FixedFormat) -> Option<InFormat> {
    let int = |field: &str| -> Option<i64> {
        ctx.field_id(field).and_then(|id| f.get(id)).and_then(|v| match v {
            Value::Int(i) => Some(i),
            _ => None,
        })
    };
    let width = int(F_WIDTH)? as u32;
    let height = int(F_HEIGHT)? as u32;
    if width == 0 || height == 0 {
        return None;
    }
    let pix = ctx
        .field_id(F_PIXFMT)
        .and_then(|id| f.get(id))
        .and_then(|v| match v {
            Value::Id(vid) => ctx.value_name(vid),
            _ => None,
        })
        .and_then(RawPixFmt::from_caps_name)?;
    let fps = ctx
        .field_id(F_FPS)
        .and_then(|id| f.get(id))
        .and_then(|v| match v {
            Value::Rat(num, den) if num > 0 && den > 0 => Some((num as u32, den as u32)),
            _ => None,
        })
        .unwrap_or((0, 1));
    Some(InFormat { width, height, pixfmt: pix, fps })
}

pub(crate) fn align_up(v: u32, a: u32) -> u32 {
    v.div_ceil(a) * a
}

/// The VA objects for one encode session at fixed dimensions. Field order = drop
/// order: context before surfaces before config (the teardown order libva wants).
pub(crate) struct EncEngine {
    context: Context,
    // Held for ownership: the surface set must outlive the context that renders
    // into it (drop order top-down, same as the decoder's VaState).
    _surfaces: Surfaces,
    _config: Config,
    /// The raw source surface frames upload into (surface[0]).
    input: ffi::VASurfaceID,
    /// The reconstruction pair (surfaces[1], surfaces[2]); `cur` indexes the slot
    /// the in-flight/next frame reconstructs into.
    recon: [ffi::VASurfaceID; 2],
    cur: usize,
    /// Full (alignment-padded) surface dimensions — what the context was created
    /// with and what parameter buffers describe as the coded picture.
    pub coded_w: u32,
    pub coded_h: u32,
    /// True (cropped) frame dimensions.
    pub width: u32,
    pub height: u32,
}

impl EncEngine {
    /// Stand up config (+ rate-control mode) / surfaces / context for
    /// `width`×`height` frames, padding the coded picture to `align`.
    pub(crate) fn new(
        display: &Display,
        profile: ffi::VAProfile,
        entrypoint: ffi::VAEntrypoint,
        rc_mode: u32,
        width: u32,
        height: u32,
        align: u32,
    ) -> VaResult<EncEngine> {
        let coded_w = align_up(width.max(2), align);
        let coded_h = align_up(height.max(2), align);
        let config = Config::new_encode_rc(display, profile, entrypoint, rc_mode)?;
        // 3 surfaces: input + the reconstruction pair.
        let surfaces = Surfaces::new_nv12(display, coded_w, coded_h, 3)?;
        let context =
            Context::new(display, &config, coded_w as i32, coded_h as i32, &surfaces)?;
        let ids = surfaces.ids();
        let (input, recon) = (ids[0], [ids[1], ids[2]]);
        Ok(EncEngine {
            context,
            _surfaces: surfaces,
            _config: config,
            input,
            recon,
            cur: 0,
            coded_w,
            coded_h,
            width,
            height,
        })
    }

    pub(crate) fn context(&self) -> &Context {
        &self.context
    }

    /// The surface the current frame reconstructs into.
    pub(crate) fn recon_cur(&self) -> ffi::VASurfaceID {
        self.recon[self.cur]
    }

    /// The previous frame's reconstruction — the P-frame reference.
    pub(crate) fn recon_ref(&self) -> ffi::VASurfaceID {
        self.recon[1 - self.cur]
    }

    /// Rotate the reconstruction pair after a successfully encoded frame.
    pub(crate) fn advance_recon(&mut self) {
        self.cur = 1 - self.cur;
    }

    /// Copy one tight-packed raw frame into the input surface. Handles both
    /// accepted layouts (NV12 pass-through rows; I420 interleaves Cb/Cr pairs) and
    /// replicates the right/bottom edge into the alignment padding.
    pub(crate) fn upload(&self, display: &Display, data: &[u8], fmt: RawPixFmt) -> VaResult<()> {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let (pad_w, pad_h) = (self.coded_w as usize, self.coded_h as usize);
        let (pad_cw, pad_ch) = (pad_w / 2, pad_h / 2);

        let mut img = UploadImage::acquire(display.raw(), self.input, self.coded_w, self.coded_h)?;

        // Luma: rows 0..h from the source (right edge replicated to pad_w), then
        // rows h..pad_h replicating the last source row.
        for row in 0..pad_h {
            let src_row = row.min(h - 1);
            let Some(dst) = img.row_mut(0, row as u32, pad_w) else { break };
            let src = &data[src_row * w..src_row * w + w];
            dst[..w].copy_from_slice(src);
            let edge = src[w - 1];
            dst[w..].fill(edge);
        }

        // Chroma: NV12 plane 1 is interleaved CbCr, pad_cw pairs per row.
        let y_len = w * h;
        for row in 0..pad_ch {
            let src_row = row.min(ch - 1);
            let Some(dst) = img.row_mut(1, row as u32, pad_cw * 2) else { break };
            match fmt {
                RawPixFmt::Nv12 => {
                    let src = &data[y_len + src_row * cw * 2..y_len + src_row * cw * 2 + cw * 2];
                    dst[..cw * 2].copy_from_slice(src);
                }
                RawPixFmt::I420 => {
                    let cb = &data[y_len + src_row * cw..y_len + src_row * cw + cw];
                    let cr_base = y_len + cw * ch;
                    let cr = &data[cr_base + src_row * cw..cr_base + src_row * cw + cw];
                    for i in 0..cw {
                        dst[i * 2] = cb[i];
                        dst[i * 2 + 1] = cr[i];
                    }
                }
            }
            // Replicate the last chroma pair across the width padding.
            let (last_cb, last_cr) = (dst[cw * 2 - 2], dst[cw * 2 - 1]);
            for i in cw..pad_cw {
                dst[i * 2] = last_cb;
                dst[i * 2 + 1] = last_cr;
            }
        }

        img.finish()
    }

    /// Submit one encoded picture and wait for it: `vaBeginPicture` on the input
    /// surface, render all of `bufs`, `vaEndPicture`, then `vaSyncSurface` so the
    /// coded buffer is safe to map.
    pub(crate) fn submit(&self, display: &Display, bufs: &[ffi::VABufferID]) -> VaResult<()> {
        self.context.begin(self.input)?;
        self.context.render(bufs)?;
        self.context.end()?;
        va::sync_surface(display.raw(), self.input)
    }

    /// A worst-case coded buffer size for one frame: raw 4:2:0 size + headroom.
    /// A hardware encoder output larger than the raw input means something is
    /// deeply wrong; the +4 KiB covers headers on tiny frames.
    pub(crate) fn coded_buf_size(&self) -> u32 {
        self.coded_w * self.coded_h * 3 / 2 + 4096
    }
}

#[cfg(test)]
mod tests {
    use super::align_up;

    #[test]
    fn align_rounds_up() {
        assert_eq!(align_up(1920, 16), 1920);
        assert_eq!(align_up(1080, 16), 1088);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(33, 32), 64);
    }
}
