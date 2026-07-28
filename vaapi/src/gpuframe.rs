//! The **GPU-frame descriptor** — the tiny payload a zero-copy VA-API decoder emits in
//! place of pixels, and the wire format it rides downstream in.
//!
//! # Why a serialized byte payload (and not a metadata side-channel)
//!
//! streamcraft's [`Buffer`](streamcraft_core::buffer::Buffer) carries a byte
//! [`Memory`](streamcraft_core::memory::Memory) and POD timestamps — there is no generic
//! out-of-band metadata slot, and the `ExternalMemory`/dma-buf `Memory` variant is a
//! documented stub, not yet plumbed through the scheduler/pool. Rather than grow the core
//! buffer model for one consumer, the zero-copy path carries the descriptor *as the
//! buffer's bytes*: the decoder emits a ~64-byte buffer on a new `video/gpu` src family
//! (so it never negotiates against a software `video/raw` sink), and the frame-slot sink —
//! its sole, explicitly-wired consumer — deserializes it and hands the fds to the EGL
//! importer. The fds themselves are **not** in the bytes (an fd is a process-local integer,
//! not portable data); they travel out-of-band through the shared [`GpuFrameChannel`],
//! keyed by the `token` that *is* in the bytes. This keeps the streaming thread's hand-off
//! to the GUI tiny (a few dozen bytes + a lock-free channel push).
//!
//! # The dma-buf lifetime, end to end
//!
//! 1. decode → `vaSyncSurface` → [`ExportedSurface::export`](crate::va::ExportedSurface)
//!    (`vaExportSurfaceHandle`, PRIME_2, COMPOSED, READ_ONLY);
//! 2. `dup()` the fds (the importer owns *its* copies), build a [`GpuFrame`] carrying the
//!    dup'd fds + plane geometry, push it into the [`GpuFrameChannel`] under `token`, and
//!    emit a `video/gpu` buffer whose bytes are `token` + the display geometry;
//! 3. the sink pops the `GpuFrame` by `token`, hands the fds to `eglCreateImageKHR`
//!    (`EGL_LINUX_DMA_BUF_EXT`), samples the external texture, then closes the fds;
//! 4. the *surface* (GPU memory) is kept live by the decoder's pool "in-flight" marking
//!    until the sink signals presentation over the same channel (see the decoder's release
//!    path) — the fd close in step 3 only releases the importer's handles, not the surface.

use std::sync::{Arc, Condvar, Mutex};

use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};

use crate::va::ExportedPlane;

/// The magic + version leading the `video/gpu` byte payload — a guard so a mis-wired
/// software sink (or a future format bump) fails loudly rather than sampling garbage.
pub const GPU_FRAME_MAGIC: u32 = 0x5343_4750; // "SCGP"
pub const GPU_FRAME_VERSION: u32 = 1;

/// The `video/gpu` family a zero-copy VA-API decoder announces on its src pad — distinct
/// from `video/raw` so it only ever links to a consumer that opts in (the frame-slot sink's
/// zero-copy variant), never a software sink expecting CPU pixels.
pub const FAMILY_VIDEO_GPU: &str = "video/gpu";
/// The one categorical value on `video/gpu`: the surface layout the importer must handle.
pub const F_LAYOUT: &str = "layout";
pub const LAYOUT_NV12_DMABUF: &str = "nv12-dmabuf";
/// The `width` field on `video/gpu` (the display/cropped size, matching `video/raw`).
pub const F_WIDTH: &str = "width";
/// The `height` field on `video/gpu`.
pub const F_HEIGHT: &str = "height";

/// The shared `video/gpu` src offer template both zero-copy decoders (`vaapih264dec` /
/// `vaapih265dec`) list on their src pad — any dimensions, `nv12-dmabuf` layout. Defined
/// once here so the two decoders declare an identical offer (the frame-slot sink's
/// zero-copy sink offer must intersect exactly one family).
pub static LAYOUT_VALUES: [ValueDesc; 1] = [ValueDesc::Id(LAYOUT_NV12_DMABUF)];
pub static GPU_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_LAYOUT, allowed: ConstraintDesc::Set(&LAYOUT_VALUES), preferred: None },
];
/// The `video/gpu` offer, ready to drop into a decoder's `SRC_OFFERS` array.
pub static GPU_OFFER: OfferDesc = OfferDesc { family: FAMILY_VIDEO_GPU, fields: &GPU_FIELDS };

/// A monotonic token pairing a `video/gpu` buffer (in-band) with its [`GpuFrame`]
/// (out-of-band, in the [`GpuFrameChannel`]). Wraps at u64 — a player would run for
/// centuries at any frame rate before it did.
pub type FrameToken = u64;

/// The out-of-band GPU frame: the dup'd DMA-BUF fds + the geometry the EGL importer needs.
/// Owned by whoever pops it from the [`GpuFrameChannel`]; that owner **must** either import
/// and then [`close`](Self::close_fds) the fds, or [`close`](Self::close_fds) them on the
/// drop/error path — a leaked fd pins GPU memory.
pub struct GpuFrame {
    /// Pairs with the in-band `video/gpu` buffer's `token`.
    pub token: FrameToken,
    /// The dup'd DMA-BUF fds (one per object); the popper owns and closes them.
    pub fds: Vec<std::os::unix::io::RawFd>,
    /// DRM FourCC of the composed layer (`DRM_FORMAT_NV12`).
    pub drm_format: u32,
    /// DRM format modifier (`DRM_FORMAT_MOD_INVALID` ⇒ importer omits the modifier attrs).
    pub drm_modifier: u64,
    /// Coded (allocation) width/height — the dma-buf plane geometry is at this size.
    pub coded_w: u32,
    pub coded_h: u32,
    /// Cropped display width/height — what the sampler's texture rect covers.
    pub disp_w: u32,
    pub disp_h: u32,
    /// Crop origin in the coded surface (§7.4.3.2.1; almost always (0,0)).
    pub crop_x: u32,
    pub crop_y: u32,
    /// Per-plane object-index/offset/pitch (NV12 → 2: Y then interleaved CbCr).
    pub planes: Vec<ExportedPlane>,
}

impl GpuFrame {
    /// Close every owned fd (idempotent-ish: clears the vec so a second call is a no-op).
    /// Call after `eglCreateImageKHR` has taken its own reference, or on any error path.
    pub fn close_fds(&mut self) {
        for fd in self.fds.drain(..) {
            // SAFETY: fd is an owned dup'd DMA-BUF fd; closed exactly once (the drain
            // empties the vec, so Drop won't re-close).
            unsafe {
                close(fd);
            }
        }
    }
}

impl Drop for GpuFrame {
    fn drop(&mut self) {
        self.close_fds();
    }
}

extern "C" {
    fn close(fd: std::ffi::c_int) -> std::ffi::c_int;
}

/// Close a raw fd (the error-path cleanup for a dup'd DMA-BUF fd that never made it into a
/// [`GpuFrame`]). Ignores the result — a cleanup path has nowhere to report it.
///
/// # Safety
/// `fd` must be a live, owned file descriptor closed exactly once.
pub unsafe fn close_raw_fd(fd: std::os::unix::io::RawFd) {
    close(fd);
}

/// The shared producer→consumer hand-off for [`GpuFrame`]s + the presentation-release
/// return path. One is created per zero-copy player run and cloned (`Arc`) into both the
/// decoder (producer) and the frame-slot sink (consumer).
///
/// # The two directions
///
/// * **forward** ([`push`](Self::push) / [`take`](Self::take)): the decoder deposits a
///   `GpuFrame` keyed by token; the sink takes it when it uploads the paired `video/gpu`
///   buffer. Bounded implicitly by the decoder's surface pool — a `push` for a surface the
///   pool can't yet reuse simply doesn't happen until the release below frees it.
/// * **release** ([`release`](Self::release) / [`drain_released`](Self::drain_released)):
///   after the sink has *presented* a token's frame (its GL commands submitted), it releases
///   the token; the decoder drains released tokens and only then returns the underlying
///   surface to its free pool. This is the anti-tear guarantee: a surface is never a new
///   decode target while the GUI is still sampling its dma-buf.
///
/// A missing release (GUI stalls / exits) is bounded by the pool depth — the decoder blocks
/// (backpressure) rather than reusing a surface early, so a lost token stalls, never tears.
pub struct GpuFrameChannel {
    inner: Mutex<ChannelInner>,
    /// Signalled on every `push` so a sink blocked waiting for a token wakes promptly.
    cv: Condvar,
}

struct ChannelInner {
    /// Frames deposited by the decoder, not yet taken by the sink. Small (≤ pool depth).
    pending: Vec<GpuFrame>,
    /// Tokens the sink has presented, not yet drained by the decoder.
    released: Vec<FrameToken>,
    /// Set on shutdown so a blocked `take` returns instead of hanging.
    closed: bool,
}

impl GpuFrameChannel {
    /// A fresh channel. Share via `Arc` between the decoder and the sink.
    // Cold: constructor, one channel per zero-copy pipeline; buffers grow in-place.
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(ChannelInner {
                pending: Vec::new(),
                released: Vec::new(),
                closed: false,
            }),
            cv: Condvar::new(),
        })
    }

    /// Producer: deposit a decoded frame for the sink to pick up by `token`.
    pub fn push(&self, frame: GpuFrame) {
        let mut g = self.inner.lock().unwrap();
        g.pending.push(frame);
        drop(g);
        self.cv.notify_all();
    }

    /// Consumer: take the frame for `token`, if it has arrived (non-blocking). Removes and
    /// returns it; the caller then owns (and must close) its fds.
    pub fn take(&self, token: FrameToken) -> Option<GpuFrame> {
        let mut g = self.inner.lock().unwrap();
        let pos = g.pending.iter().position(|f| f.token == token)?;
        Some(g.pending.swap_remove(pos))
    }

    /// Consumer: signal that `token`'s frame has been presented — the decoder may now reuse
    /// its surface. (The `GpuFrame`'s fds are the sink's to close; this returns only the
    /// token.)
    pub fn release(&self, token: FrameToken) {
        let mut g = self.inner.lock().unwrap();
        g.released.push(token);
    }

    /// Producer: drain the tokens the sink has released since the last call (the decoder
    /// turns each back into a free surface).
    pub fn drain_released(&self) -> Vec<FrameToken> {
        let mut g = self.inner.lock().unwrap();
        std::mem::take(&mut g.released)
    }

    /// Consumer: on a **flush/seek**, orphan every still-`pending` frame — its `video/gpu`
    /// pipeline buffer was discarded by the flush, so the sink will never pace/take it.
    /// Release each token (the decoder reclaims the surface) and drop the frames (closing
    /// their fds). Without this a seek leaks one surface per un-paced pre-seek frame, and the
    /// decoder's pool eventually starves and freezes. Idempotent; safe to call every flush.
    pub fn flush_pending(&self) {
        let mut g = self.inner.lock().unwrap();
        // Collect first so the `drain` borrow ends before pushing to `released`.
        let orphaned: Vec<GpuFrame> = g.pending.drain(..).collect();
        for f in &orphaned {
            g.released.push(f.token);
        }
        // `orphaned` drops here → each GpuFrame closes its fds.
    }

    /// Mark the channel closed (shutdown) and wake any waiter.
    pub fn close(&self) {
        let mut g = self.inner.lock().unwrap();
        g.closed = true;
        drop(g);
        self.cv.notify_all();
    }

    /// Whether the channel is closed.
    pub fn is_closed(&self) -> bool {
        self.inner.lock().unwrap().closed
    }
}

/// The fixed in-band `video/gpu` byte payload: a self-describing little-endian record the
/// sink reads to (a) pair the buffer with its out-of-band [`GpuFrame`] via `token` and
/// (b) know the display geometry to letterbox even before the fd import. Deliberately POD
/// and versioned; **no fds** (those go out-of-band). 40 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuFrameHeader {
    pub token: FrameToken,
    pub coded_w: u32,
    pub coded_h: u32,
    pub disp_w: u32,
    pub disp_h: u32,
    pub crop_x: u32,
    pub crop_y: u32,
}

/// The serialized size of a [`GpuFrameHeader`]: magic(4) + version(4) + token(8) + six
/// u32 geometry = 40 bytes.
pub const GPU_FRAME_HEADER_LEN: usize = 4 + 4 + 8 + 6 * 4;

impl GpuFrameHeader {
    /// Serialize into a fresh little-endian byte vector (the `video/gpu` buffer body).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(GPU_FRAME_HEADER_LEN);
        b.extend_from_slice(&GPU_FRAME_MAGIC.to_le_bytes());
        b.extend_from_slice(&GPU_FRAME_VERSION.to_le_bytes());
        b.extend_from_slice(&self.token.to_le_bytes());
        b.extend_from_slice(&self.coded_w.to_le_bytes());
        b.extend_from_slice(&self.coded_h.to_le_bytes());
        b.extend_from_slice(&self.disp_w.to_le_bytes());
        b.extend_from_slice(&self.disp_h.to_le_bytes());
        b.extend_from_slice(&self.crop_x.to_le_bytes());
        b.extend_from_slice(&self.crop_y.to_le_bytes());
        b
    }

    /// Parse from a `video/gpu` buffer body. `None` on a short buffer or a magic/version
    /// mismatch (a mis-wired software payload, or a version skew) — never a panic.
    pub fn from_bytes(b: &[u8]) -> Option<GpuFrameHeader> {
        if b.len() < GPU_FRAME_HEADER_LEN {
            return None;
        }
        let rd_u32 = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let rd_u64 = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        if rd_u32(0) != GPU_FRAME_MAGIC || rd_u32(4) != GPU_FRAME_VERSION {
            return None;
        }
        Some(GpuFrameHeader {
            token: rd_u64(8),
            coded_w: rd_u32(16),
            coded_h: rd_u32(20),
            disp_w: rd_u32(24),
            disp_h: rd_u32(28),
            crop_x: rd_u32(32),
            crop_y: rd_u32(36),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips() {
        let h = GpuFrameHeader {
            token: 0xDEAD_BEEF_1234_5678,
            coded_w: 1920,
            coded_h: 1088,
            disp_w: 1920,
            disp_h: 1080,
            crop_x: 0,
            crop_y: 0,
        };
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), GPU_FRAME_HEADER_LEN);
        assert_eq!(GpuFrameHeader::from_bytes(&bytes), Some(h));
    }

    #[test]
    fn header_rejects_bad_magic_and_short() {
        let mut bytes = GpuFrameHeader {
            token: 1,
            coded_w: 4,
            coded_h: 4,
            disp_w: 4,
            disp_h: 4,
            crop_x: 0,
            crop_y: 0,
        }
        .to_bytes();
        assert!(GpuFrameHeader::from_bytes(&bytes[..GPU_FRAME_HEADER_LEN - 1]).is_none());
        bytes[0] ^= 0xFF; // corrupt the magic
        assert!(GpuFrameHeader::from_bytes(&bytes).is_none());
    }

    #[test]
    fn channel_push_take_release() {
        let ch = GpuFrameChannel::new();
        assert!(ch.take(7).is_none());
        // A GpuFrame with no fds (nothing to close) exercises the token routing.
        ch.push(GpuFrame {
            token: 7,
            fds: Vec::new(),
            drm_format: crate::ffi::DRM_FORMAT_NV12,
            drm_modifier: crate::ffi::DRM_FORMAT_MOD_INVALID,
            coded_w: 16,
            coded_h: 16,
            disp_w: 16,
            disp_h: 16,
            crop_x: 0,
            crop_y: 0,
            planes: Vec::new(),
        });
        let f = ch.take(7).expect("token 7 present");
        assert_eq!(f.token, 7);
        assert!(ch.take(7).is_none(), "taken once");
        ch.release(7);
        assert_eq!(ch.drain_released(), vec![7]);
        assert!(ch.drain_released().is_empty());
    }
}
