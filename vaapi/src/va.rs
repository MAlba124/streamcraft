//! Safe RAII wrappers over the raw [`crate::ffi`] surface. Every VA-API object
//! that must be released (display, config, context, surface set, buffer, mapped
//! image) is owned by a Rust value whose `Drop` calls the matching `vaDestroy…`;
//! fallible driver calls return a [`VaError`] carrying the `VAStatus` and the call
//! site, stringified through `vaErrorStr`.
//!
//! Threading: `VADisplay` is single-threaded in this design. The decode element
//! owns its `Display` on its own scheduler thread (`SchedHint::Active`) and never
//! shares it, so the wrappers are deliberately **not** `Send`/`Sync` — the raw
//! pointer arm of the display makes that automatic. All `unsafe` in the crate is
//! confined to this module and [`crate::ffi`].

use std::ffi::{c_int, CStr};
use std::fmt;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr;

use crate::ffi;

/// A VA-API call that returned a non-success `VAStatus`, tagged with the call site.
#[derive(Clone)]
pub struct VaError {
    pub status: ffi::VAStatus,
    pub ctx: &'static str,
}

impl VaError {
    /// The driver's human-readable message for this status (`vaErrorStr`).
    pub fn message(&self) -> String {
        // SAFETY: vaErrorStr returns a static NUL-terminated string for any status
        // (unknown statuses map to "unknown error"); never null in practice.
        unsafe {
            let p = ffi::vaErrorStr(self.status);
            if p.is_null() {
                return format!("VAStatus {:#x}", self.status);
            }
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

impl fmt::Display for VaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed: {} ({:#x})", self.ctx, self.message(), self.status)
    }
}

impl fmt::Debug for VaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for VaError {}

pub type VaResult<T> = Result<T, VaError>;

/// Turn a raw `VAStatus` into a `Result`, tagging failures with the call site.
fn check(status: ffi::VAStatus, ctx: &'static str) -> VaResult<()> {
    if status == ffi::VA_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(VaError { status, ctx })
    }
}

/// An initialized VA-API display over a DRM render node. Holds the `File` for the
/// node so its fd stays valid for the display's lifetime; `Drop` terminates the
/// display *before* the `File` closes the fd.
pub struct Display {
    // Field order = drop order: dpy first, then file. `vaTerminate` runs before
    // the fd is closed.
    dpy: ffi::VADisplay,
    _file: File,
    version: (i32, i32),
}

impl Display {
    /// Open a DRM node (e.g. `/dev/dri/renderD128`) and initialize a VA display on
    /// it. Fails if the node cannot be opened, the driver cannot attach, or
    /// `vaInitialize` reports an error (no usable driver for that node).
    pub fn open(path: &Path) -> VaResult<Display> {
        // Read-*write*: DRM allocation ioctls (GEM buffer creation) need a
        // writable fd. A read-only node initializes and answers every query,
        // then fails the first real allocation — vaCreateContext returns
        // VA_STATUS_ERROR_ALLOCATION_FAILED with nothing else wrong.
        let file = File::options().read(true).write(true).open(path).map_err(|_| VaError {
            status: -1,
            ctx: "open(drm node)",
        })?;
        // SAFETY: fd is valid for the File's lifetime (kept in the struct).
        let dpy = unsafe { ffi::vaGetDisplayDRM(file.as_raw_fd()) };
        if dpy.is_null() {
            return Err(VaError { status: -1, ctx: "vaGetDisplayDRM" });
        }
        let mut major: std::ffi::c_int = 0;
        let mut minor: std::ffi::c_int = 0;
        // SAFETY: dpy is a fresh non-null display; out pointers are valid locals.
        let st = unsafe { ffi::vaInitialize(dpy, &mut major, &mut minor) };
        if let Err(e) = check(st, "vaInitialize") {
            // The display attached but init failed; terminate it so the driver
            // releases whatever it partially set up before we drop the fd.
            // SAFETY: dpy is the display we just failed to fully init.
            unsafe {
                ffi::vaTerminate(dpy);
            }
            return Err(e);
        }
        Ok(Display {
            dpy,
            _file: file,
            version: (major, minor),
        })
    }

    /// The raw display handle, for the wrappers below and the element's decode
    /// commands. Callers must not outlive the `Display`.
    pub fn raw(&self) -> ffi::VADisplay {
        self.dpy
    }

    /// The negotiated VA-API version reported by `vaInitialize` (major, minor).
    pub fn version(&self) -> (i32, i32) {
        self.version
    }

    /// The driver's vendor string (`vaQueryVendorString`), e.g. the iHD banner.
    pub fn vendor(&self) -> String {
        // SAFETY: dpy is a live, initialized display; the returned string is
        // driver-owned static storage, copied out immediately.
        unsafe {
            let p = ffi::vaQueryVendorString(self.dpy);
            if p.is_null() {
                return String::new();
            }
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

impl Drop for Display {
    fn drop(&mut self) {
        // SAFETY: single owner; dpy is live until here. Ignore the status — a Drop
        // has nowhere to propagate a terminate error.
        unsafe {
            ffi::vaTerminate(self.dpy);
        }
    }
}

/// A decode `VAConfig` (profile + entrypoint + RT-format attributes).
pub struct Config {
    dpy: ffi::VADisplay,
    id: ffi::VAConfigID,
}

impl Config {
    /// Create a VLD (decode) config for `profile` with the YUV420 RT format.
    pub fn new_decode(display: &Display, profile: ffi::VAProfile) -> VaResult<Config> {
        let mut attrib = ffi::VAConfigAttrib {
            type_: ffi::VAConfigAttribRTFormat,
            value: ffi::VA_RT_FORMAT_YUV420,
        };
        let mut id: ffi::VAConfigID = ffi::VA_INVALID_ID;
        // SAFETY: display live; attrib is a valid single-element array; id out ptr valid.
        let st = unsafe {
            ffi::vaCreateConfig(
                display.raw(),
                profile,
                ffi::VAEntrypointVLD,
                &mut attrib,
                1,
                &mut id,
            )
        };
        check(st, "vaCreateConfig")?;
        Ok(Config { dpy: display.raw(), id })
    }

    /// Create an *encode* config (NV12/YUV420) at the given encode entrypoint
    /// (`VAEntrypointEncSlice` or the low-power `VAEntrypointEncSliceLP`). Used by
    /// the encode capability smoke test (the encoder elements use
    /// [`new_encode_rc`](Self::new_encode_rc), which also pins the RC mode).
    pub fn new_encode(
        display: &Display,
        profile: ffi::VAProfile,
        entrypoint: ffi::VAEntrypoint,
    ) -> VaResult<Config> {
        let mut attrib = ffi::VAConfigAttrib {
            type_: ffi::VAConfigAttribRTFormat,
            value: ffi::VA_RT_FORMAT_YUV420,
        };
        let mut id: ffi::VAConfigID = ffi::VA_INVALID_ID;
        // SAFETY: display live; attrib is a valid single-element array; id out ptr valid.
        let st = unsafe {
            ffi::vaCreateConfig(display.raw(), profile, entrypoint, &mut attrib, 1, &mut id)
        };
        check(st, "vaCreateConfig(encode)")?;
        Ok(Config { dpy: display.raw(), id })
    }

    /// Create an encode config that also pins the rate-control mode
    /// (`VA_RC_CQP`/`VA_RC_CBR`/`VA_RC_VBR`) — the driver validates the mode against
    /// what the entrypoint supports at config time, so an unsupported mode fails
    /// here rather than mid-stream.
    pub fn new_encode_rc(
        display: &Display,
        profile: ffi::VAProfile,
        entrypoint: ffi::VAEntrypoint,
        rc_mode: u32,
    ) -> VaResult<Config> {
        let mut attribs = [
            ffi::VAConfigAttrib {
                type_: ffi::VAConfigAttribRTFormat,
                value: ffi::VA_RT_FORMAT_YUV420,
            },
            ffi::VAConfigAttrib {
                type_: ffi::VAConfigAttribRateControl,
                value: rc_mode,
            },
        ];
        let mut id: ffi::VAConfigID = ffi::VA_INVALID_ID;
        // SAFETY: display live; attribs is a valid 2-element array; id out ptr valid.
        let st = unsafe {
            ffi::vaCreateConfig(
                display.raw(),
                profile,
                entrypoint,
                attribs.as_mut_ptr(),
                attribs.len() as std::ffi::c_int,
                &mut id,
            )
        };
        check(st, "vaCreateConfig(encode+rc)")?;
        Ok(Config { dpy: display.raw(), id })
    }

    pub fn id(&self) -> ffi::VAConfigID {
        self.id
    }
}

/// `vaSyncSurface` — block until every pending operation targeting `surface`
/// completes (decode readback and encode coded-buffer drains both fence on this).
pub fn sync_surface(dpy: ffi::VADisplay, surface: ffi::VASurfaceID) -> VaResult<()> {
    // SAFETY: caller guarantees dpy/surface are live (element-lifetime handles).
    check(unsafe { ffi::vaSyncSurface(dpy, surface) }, "vaSyncSurface")
}

/// Query one config attribute for `profile`×`entrypoint` (`vaGetConfigAttributes`).
/// `None` when the driver reports the attribute unsupported.
pub fn config_attribute(
    display: &Display,
    profile: ffi::VAProfile,
    entrypoint: ffi::VAEntrypoint,
    attrib_type: ffi::VAConfigAttribType,
) -> VaResult<Option<u32>> {
    let mut attrib = ffi::VAConfigAttrib { type_: attrib_type, value: 0 };
    // SAFETY: display live; attrib is a valid single-element in/out array.
    let st = unsafe {
        ffi::vaGetConfigAttributes(display.raw(), profile, entrypoint, &mut attrib, 1)
    };
    check(st, "vaGetConfigAttributes")?;
    Ok(if attrib.value == ffi::VA_ATTRIB_NOT_SUPPORTED {
        None
    } else {
        Some(attrib.value)
    })
}

impl Drop for Config {
    fn drop(&mut self) {
        // SAFETY: id valid until destroyed once; dpy outlives this (element drop order).
        unsafe {
            ffi::vaDestroyConfig(self.dpy, self.id);
        }
    }
}

/// A fixed set of decode target surfaces (NV12, YUV420), owned and released as one.
pub struct Surfaces {
    dpy: ffi::VADisplay,
    ids: Vec<ffi::VASurfaceID>,
}

impl Surfaces {
    /// Allocate `count` NV12 surfaces of `width`×`height`.
    pub fn new_nv12(display: &Display, width: u32, height: u32, count: u32) -> VaResult<Surfaces> {
        let mut attrib = ffi::VASurfaceAttrib {
            type_: ffi::VASurfaceAttribPixelFormat,
            flags: ffi::VA_SURFACE_ATTRIB_SETTABLE,
            value: ffi::VAGenericValue {
                type_: ffi::VAGenericValueTypeInteger,
                value: ffi::VAGenericValueUnion {
                    i: ffi::VA_FOURCC_NV12 as i32,
                },
            },
        };
        let mut ids = vec![ffi::VA_INVALID_SURFACE; count as usize];
        // SAFETY: display live; ids has room for `count`; attrib is one valid entry.
        let st = unsafe {
            ffi::vaCreateSurfaces(
                display.raw(),
                ffi::VA_RT_FORMAT_YUV420,
                width,
                height,
                ids.as_mut_ptr(),
                count,
                &mut attrib,
                1,
            )
        };
        check(st, "vaCreateSurfaces")?;
        Ok(Surfaces { dpy: display.raw(), ids })
    }

    pub fn ids(&self) -> &[ffi::VASurfaceID] {
        &self.ids
    }
}

impl Drop for Surfaces {
    fn drop(&mut self) {
        if self.ids.is_empty() {
            return;
        }
        // SAFETY: ids are live surfaces owned by this set; destroyed exactly once.
        unsafe {
            ffi::vaDestroySurfaces(self.dpy, self.ids.as_mut_ptr(), self.ids.len() as i32);
        }
    }
}

/// A decode `VAContext` bound to a config + its render targets.
pub struct Context {
    dpy: ffi::VADisplay,
    id: ffi::VAContextID,
}

impl Context {
    /// Create a progressive decode context sized `width`×`height` over `surfaces`.
    pub fn new(
        display: &Display,
        config: &Config,
        width: i32,
        height: i32,
        surfaces: &Surfaces,
    ) -> VaResult<Context> {
        let mut targets = surfaces.ids().to_vec();
        let mut id: ffi::VAContextID = ffi::VA_INVALID_ID;
        // SAFETY: display/config live; targets is a valid array of that length.
        let st = unsafe {
            ffi::vaCreateContext(
                display.raw(),
                config.id(),
                width,
                height,
                ffi::VA_PROGRESSIVE,
                targets.as_mut_ptr(),
                targets.len() as i32,
                &mut id,
            )
        };
        check(st, "vaCreateContext")?;
        Ok(Context { dpy: display.raw(), id })
    }

    pub fn id(&self) -> ffi::VAContextID {
        self.id
    }

    // --- The per-picture decode command sequence -------------------------------------

    /// `vaBeginPicture` — target the surface the coming render buffers decode into.
    pub fn begin(&self, target: ffi::VASurfaceID) -> VaResult<()> {
        // SAFETY: context + target are live for this call.
        check(unsafe { ffi::vaBeginPicture(self.dpy, self.id, target) }, "vaBeginPicture")
    }

    /// `vaRenderPicture` — hand the driver one or more parameter/data buffers.
    pub fn render(&self, buffers: &[ffi::VABufferID]) -> VaResult<()> {
        let mut bufs = buffers.to_vec();
        // SAFETY: context live; bufs is a valid array of the given length.
        check(
            unsafe { ffi::vaRenderPicture(self.dpy, self.id, bufs.as_mut_ptr(), bufs.len() as i32) },
            "vaRenderPicture",
        )
    }

    /// `vaEndPicture` — commit the picture for asynchronous decode.
    pub fn end(&self) -> VaResult<()> {
        // SAFETY: context live for this call.
        check(unsafe { ffi::vaEndPicture(self.dpy, self.id) }, "vaEndPicture")
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: id valid, destroyed once; must precede Surfaces drop (element field
        // order guarantees it).
        unsafe {
            ffi::vaDestroyContext(self.dpy, self.id);
        }
    }
}

/// A single `VABuffer` (a picture/IQ/slice parameter blob or slice data), released
/// on drop. Built by copying a `#[repr(C)]` value or a byte slice into VA storage.
pub struct Buffer {
    dpy: ffi::VADisplay,
    id: ffi::VABufferID,
}

impl Buffer {
    /// Create a buffer of `type_` holding one element `value` (a `#[repr(C)]`
    /// parameter struct). The driver copies the bytes.
    pub fn new_struct<T: Copy>(
        display: &Display,
        context: &Context,
        type_: ffi::VABufferType,
        value: &T,
    ) -> VaResult<Buffer> {
        let mut id: ffi::VABufferID = ffi::VA_INVALID_ID;
        // SAFETY: value is a live &T of the exact size we pass; the driver copies it.
        let st = unsafe {
            ffi::vaCreateBuffer(
                display.raw(),
                context.id(),
                type_,
                std::mem::size_of::<T>() as u32,
                1,
                value as *const T as *mut std::ffi::c_void,
                &mut id,
            )
        };
        check(st, "vaCreateBuffer(struct)")?;
        Ok(Buffer { dpy: display.raw(), id })
    }

    /// Create a slice-data buffer from raw bitstream bytes (the NAL, emulation
    /// bytes included). `num_elements` is the byte count, element size 1.
    pub fn new_data(display: &Display, context: &Context, data: &[u8]) -> VaResult<Buffer> {
        let mut id: ffi::VABufferID = ffi::VA_INVALID_ID;
        // SAFETY: data is a live slice; the driver copies `data.len()` bytes.
        let st = unsafe {
            ffi::vaCreateBuffer(
                display.raw(),
                context.id(),
                ffi::VASliceDataBufferType,
                data.len() as u32,
                1,
                data.as_ptr() as *mut std::ffi::c_void,
                &mut id,
            )
        };
        check(st, "vaCreateBuffer(data)")?;
        Ok(Buffer { dpy: display.raw(), id })
    }

    /// Create a buffer of `type_` from raw bytes the driver copies — packed-header
    /// data (`VAEncPackedHeaderDataBufferType`) and misc-parameter payloads
    /// (`VAEncMiscParameterBufferType`, whose leading `u32` is the misc type).
    pub fn new_typed_bytes(
        display: &Display,
        context: &Context,
        type_: ffi::VABufferType,
        data: &[u8],
    ) -> VaResult<Buffer> {
        let mut id: ffi::VABufferID = ffi::VA_INVALID_ID;
        // SAFETY: data is a live slice; the driver copies `data.len()` bytes.
        let st = unsafe {
            ffi::vaCreateBuffer(
                display.raw(),
                context.id(),
                type_,
                data.len() as u32,
                1,
                data.as_ptr() as *mut std::ffi::c_void,
                &mut id,
            )
        };
        check(st, "vaCreateBuffer(typed bytes)")?;
        Ok(Buffer { dpy: display.raw(), id })
    }

    /// Create an empty driver-filled buffer of `size` bytes — the encode *coded*
    /// buffer (`VAEncCodedBufferType`): data is NULL at creation (the map contract
    /// vaMapBuffer documents) and the driver writes the bitstream into it.
    pub fn new_empty(
        display: &Display,
        context: &Context,
        type_: ffi::VABufferType,
        size: u32,
    ) -> VaResult<Buffer> {
        let mut id: ffi::VABufferID = ffi::VA_INVALID_ID;
        // SAFETY: NULL data + size is the documented "allocate, driver fills" form.
        let st = unsafe {
            ffi::vaCreateBuffer(
                display.raw(),
                context.id(),
                type_,
                size,
                1,
                ptr::null_mut(),
                &mut id,
            )
        };
        check(st, "vaCreateBuffer(empty)")?;
        Ok(Buffer { dpy: display.raw(), id })
    }

    /// Read an encode *coded* buffer's bitstream out: map it, walk the
    /// [`VACodedBufferSegment`](ffi::VACodedBufferSegment) chain, append every
    /// segment's bytes to `out`, unmap. The target surface must be synced first
    /// (`vaSyncSurface`); the map blocks otherwise on some drivers.
    pub fn read_coded(&self, out: &mut Vec<u8>) -> VaResult<()> {
        let mut base: *mut std::ffi::c_void = ptr::null_mut();
        // SAFETY: self.id is a live coded buffer on this display.
        check(unsafe { ffi::vaMapBuffer(self.dpy, self.id, &mut base) }, "vaMapBuffer(coded)")?;
        // Walk the segment list. Bound the walk (a sane encoder yields 1–2
        // segments; 64 is "the driver handed us a cycle").
        let mut seg = base as *const ffi::VACodedBufferSegment;
        let mut hops = 0;
        while !seg.is_null() && hops < 64 {
            // SAFETY: seg points at a driver-owned VACodedBufferSegment within the
            // mapped buffer; buf points at `size` readable bytes for the map's
            // lifetime (vaMapBuffer contract for VAEncCodedBufferType).
            unsafe {
                let s = &*seg;
                if !s.buf.is_null() && s.size > 0 {
                    out.extend_from_slice(std::slice::from_raw_parts(
                        s.buf as *const u8,
                        s.size as usize,
                    ));
                }
                seg = s.next as *const ffi::VACodedBufferSegment;
            }
            hops += 1;
        }
        // SAFETY: mapped above; unmap exactly once.
        check(unsafe { ffi::vaUnmapBuffer(self.dpy, self.id) }, "vaUnmapBuffer(coded)")
    }

    pub fn id(&self) -> ffi::VABufferID {
        self.id
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: id is a live buffer, destroyed once; dpy outlives it.
        unsafe {
            ffi::vaDestroyBuffer(self.dpy, self.id);
        }
    }
}

/// A surface's decoded pixels made CPU-visible: a derived-or-created `VAImage` plus
/// its mapped base pointer. `Drop` unmaps the buffer and destroys the image.
///
/// Access planes through [`plane`]; the wrapper never hands out the raw base
/// pointer, only bounded row slices, so out-of-bounds reads are impossible from
/// safe code.
pub struct MappedImage {
    dpy: ffi::VADisplay,
    image: ffi::VAImage,
    base: *mut u8,
}

impl MappedImage {
    /// Sync `surface`, obtain a CPU-mappable NV12 image over it, and map it.
    ///
    /// Tries `vaDeriveImage` first (zero-copy on Intel iHD); if the driver refuses
    /// to derive (some drivers/formats do), falls back to `vaCreateImage` +
    /// `vaGetImage` into a fresh NV12 image of the given dimensions.
    ///
    /// Takes the raw display handle (a `Copy` pointer) rather than a `&Display`
    /// borrow so the caller (the decode element) can keep mutating its own state
    /// while a mapped image is live — the handle stays valid for the element's
    /// lifetime, which strictly outlives any `MappedImage`.
    pub fn acquire(
        dpy: ffi::VADisplay,
        surface: ffi::VASurfaceID,
        width: u32,
        height: u32,
    ) -> VaResult<MappedImage> {
        // SAFETY: surface is a live decode target on this display.
        check(unsafe { ffi::vaSyncSurface(dpy, surface) }, "vaSyncSurface")?;

        let mut image: ffi::VAImage = unsafe { std::mem::zeroed() };
        image.image_id = ffi::VA_INVALID_ID;
        // SAFETY: image is a valid out struct; surface is synced.
        let derive = unsafe { ffi::vaDeriveImage(dpy, surface, &mut image) };
        let derived_nv12 =
            derive == ffi::VA_STATUS_SUCCESS && image.format.fourcc == ffi::VA_FOURCC_NV12;

        if !derived_nv12 {
            if derive == ffi::VA_STATUS_SUCCESS {
                // Derived, but not NV12 — release it and fall through to GetImage.
                // SAFETY: image_id is the live derived image.
                unsafe {
                    ffi::vaDestroyImage(dpy, image.image_id);
                }
            }
            let mut fmt = ffi::VAImageFormat {
                fourcc: ffi::VA_FOURCC_NV12,
                byte_order: 1, // VA_LSB_FIRST
                bits_per_pixel: 12,
                depth: 0,
                red_mask: 0,
                green_mask: 0,
                blue_mask: 0,
                alpha_mask: 0,
                va_reserved: [0; ffi::VA_PADDING_LOW],
            };
            let mut created: ffi::VAImage = unsafe { std::mem::zeroed() };
            // SAFETY: fmt/created are valid out structs; dimensions are the picture's.
            check(
                unsafe {
                    ffi::vaCreateImage(
                        dpy,
                        &mut fmt,
                        width as i32,
                        height as i32,
                        &mut created,
                    )
                },
                "vaCreateImage",
            )?;
            // SAFETY: created is a live NV12 image; surface synced; region == picture.
            let st = unsafe {
                ffi::vaGetImage(dpy, surface, 0, 0, width, height, created.image_id)
            };
            if let Err(e) = check(st, "vaGetImage") {
                // SAFETY: created is live; release before propagating.
                unsafe {
                    ffi::vaDestroyImage(dpy, created.image_id);
                }
                return Err(e);
            }
            image = created;
        }

        let mut base: *mut std::ffi::c_void = ptr::null_mut();
        // SAFETY: image.buf is the image's backing buffer; base out ptr valid.
        let st = unsafe { ffi::vaMapBuffer(dpy, image.buf, &mut base) };
        if let Err(e) = check(st, "vaMapBuffer") {
            // SAFETY: image is live; release before propagating.
            unsafe {
                ffi::vaDestroyImage(dpy, image.image_id);
            }
            return Err(e);
        }
        Ok(MappedImage {
            dpy,
            image,
            base: base as *mut u8,
        })
    }

    /// Image width/height in pixels (as the driver filled them in).
    pub fn dims(&self) -> (u32, u32) {
        (self.image.width as u32, self.image.height as u32)
    }

    /// Number of planes the image carries (NV12 → 2: Y, then interleaved CbCr).
    pub fn num_planes(&self) -> u32 {
        self.image.num_planes
    }

    /// Byte pitch (stride) of plane `p`.
    pub fn pitch(&self, p: usize) -> u32 {
        self.image.pitches[p]
    }

    /// One row of plane `p`: `len` bytes starting at the plane offset + `row*pitch`.
    /// Bounds-checked against `data_size`; returns `None` on overrun (a
    /// driver-reported geometry we don't trust).
    pub fn row(&self, plane: usize, row: u32, len: usize) -> Option<&[u8]> {
        if plane >= self.image.num_planes as usize || plane >= 3 {
            return None;
        }
        let off = self.image.offsets[plane] as usize + row as usize * self.image.pitches[plane] as usize;
        if off + len > self.image.data_size as usize {
            return None;
        }
        // SAFETY: base is the mapped buffer of `data_size` bytes; [off, off+len) is
        // in range (checked above); lifetime tied to &self (the map is live).
        Some(unsafe { std::slice::from_raw_parts(self.base.add(off), len) })
    }
}

impl Drop for MappedImage {
    fn drop(&mut self) {
        // SAFETY: buf is mapped and image live; unmap then destroy, exactly once.
        unsafe {
            ffi::vaUnmapBuffer(self.dpy, self.image.buf);
            ffi::vaDestroyImage(self.dpy, self.image.image_id);
        }
    }
}

/// One exported DMA-BUF plane: which object (fd) it lives in, its byte offset in that
/// object, and its row pitch. Copied out of the driver's [`ffi::VADRMPRIMESurfaceDescriptor`]
/// so the descriptor (with its raw fds) never has to travel downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExportedPlane {
    /// Index into [`ExportedSurface::fds`] — which DMA-BUF this plane's bytes are in.
    pub object_index: u32,
    /// Byte offset of the plane's first row within that object.
    pub offset: u32,
    /// Row stride in bytes.
    pub pitch: u32,
}

/// A decoded VA surface exported as DMA-BUF(s) via `vaExportSurfaceHandle`
/// (`VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2`, COMPOSED layers). This is the zero-copy
/// display hand-off: the decoded pixels stay in GPU memory; only these fds + plane geometry
/// travel to the GUI, which imports them with `EGL_LINUX_DMA_BUF_EXT`.
///
/// **fd ownership.** The driver hands one fd per DMA-BUF object; this wrapper owns them and
/// `close()`s each on drop. The producer serializes the frame by [`dup`](Self::dup_fds)ing
/// the fds into the outgoing buffer (so the importer owns *its* copies and closes them after
/// `eglCreateImageKHR`), and drops this wrapper — closing the originals — right after. The
/// surface's *pixels* are kept alive by the decoder's surface-pool refcount (the "in-flight"
/// marking), independent of these fds: an fd pins the GEM object, but the decoder must still
/// not reuse the surface for a new decode until the GUI signals it presented (see
/// `h265dec`'s in-flight release channel).
pub struct ExportedSurface {
    /// One fd per DMA-BUF object (owned; closed on drop).
    pub fds: Vec<std::os::unix::io::RawFd>,
    /// The surface FourCC (`VA_FOURCC_NV12`).
    pub fourcc: u32,
    /// The DRM FourCC of the composed layer (`DRM_FORMAT_NV12`).
    pub drm_format: u32,
    /// The DRM format modifier of object 0 (`DRM_FORMAT_MOD_INVALID` when the driver
    /// reported none — the importer then omits the modifier attributes).
    pub drm_modifier: u64,
    /// Coded surface width/height (the descriptor reports the full coded size; the display
    /// crop is applied by the importer via the sampled texture rect).
    pub coded_w: u32,
    pub coded_h: u32,
    /// One entry per plane of the composed layer (NV12 → 2: Y then interleaved CbCr).
    pub planes: Vec<ExportedPlane>,
}

impl ExportedSurface {
    /// Sync `surface`, then export it as DMA-BUF(s) (COMPOSED NV12, read-only). The caller
    /// must keep the *surface* alive (pool refcount) until the importer is done sampling —
    /// this only governs the fd lifetime, not the GPU memory's reuse.
    // `dpy` is the raw `VADisplay` handle every VA-API entry point in this module takes; the
    // element owns it on its single scheduler thread and guarantees it outlives the call
    // (same invariant as `sync_surface`/`MappedImage::acquire`). Clippy's raw-ptr-arg lint
    // fires on the whole module for this; kept confined here rather than making the type
    // `unsafe` (which would infect the safe RAII surface).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn export(dpy: ffi::VADisplay, surface: ffi::VASurfaceID) -> VaResult<ExportedSurface> {
        // The decode into this surface may still be in flight; fence it first (same barrier
        // the readback path takes before mapping).
        // SAFETY: surface is a live decode target on this display.
        check(unsafe { ffi::vaSyncSurface(dpy, surface) }, "vaSyncSurface(export)")?;

        let mut desc: ffi::VADRMPRIMESurfaceDescriptor = unsafe { std::mem::zeroed() };
        // COMPOSED_LAYERS: one layer whose drm_format is the whole-surface FourCC
        // (DRM_FORMAT_NV12) with one plane per component — the shape an EGL NV12
        // `EGL_LINUX_DMA_BUF_EXT` import consumes (VA-API surface export flow,
        // va_drmcommon.h). READ_ONLY: the GUI only samples.
        // SAFETY: desc is a valid out struct; surface is synced; PRIME_2 is the mem type
        // vaExportSurfaceHandle fills a VADRMPRIMESurfaceDescriptor for.
        let st = unsafe {
            ffi::vaExportSurfaceHandle(
                dpy,
                surface,
                ffi::VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                ffi::VA_EXPORT_SURFACE_READ_ONLY | ffi::VA_EXPORT_SURFACE_COMPOSED_LAYERS,
                &mut desc as *mut _ as *mut std::ffi::c_void,
            )
        };
        check(st, "vaExportSurfaceHandle")?;

        let n_obj = (desc.num_objects as usize).min(ffi::VA_DRM_PRIME_MAX_OBJECTS);
        let mut fds = Vec::with_capacity(n_obj);
        for o in desc.objects.iter().take(n_obj) {
            fds.push(o.fd as std::os::unix::io::RawFd);
        }
        // COMPOSED export → exactly one layer; its planes index into objects[].
        let drm_modifier = desc.objects.first().map(|o| o.drm_format_modifier).unwrap_or(0);
        let (drm_format, planes) = if desc.num_layers >= 1 {
            let layer = &desc.layers[0];
            let np = (layer.num_planes as usize).min(ffi::VA_DRM_PRIME_MAX_PLANES);
            let planes: Vec<ExportedPlane> = (0..np)
                .map(|i| ExportedPlane {
                    object_index: layer.object_index[i],
                    offset: layer.offset[i],
                    pitch: layer.pitch[i],
                })
                .collect();
            (layer.drm_format, planes)
        } else {
            (ffi::DRM_FORMAT_NV12, Vec::new())
        };

        Ok(ExportedSurface {
            fds,
            fourcc: desc.fourcc,
            drm_format,
            drm_modifier,
            coded_w: desc.width,
            coded_h: desc.height,
            planes,
        })
    }

    /// `dup()` every owned fd, transferring ownership to the caller — the serialize step:
    /// the importer receives these duplicates (which it closes after `eglCreateImageKHR`),
    /// while this wrapper still closes the originals on drop. Returns the dup'd fds in the
    /// same order as [`fds`](Self::fds); a dup failure yields the fds dup'd so far (the
    /// caller closes them and treats the frame as un-importable).
    pub fn dup_fds(&self) -> std::io::Result<Vec<std::os::unix::io::RawFd>> {
        let mut out = Vec::with_capacity(self.fds.len());
        for &fd in &self.fds {
            // SAFETY: fd is a live DMA-BUF fd owned by this surface; F_DUPFD_CLOEXEC gives
            // the dup the close-on-exec flag (a leaked fd across exec would pin GPU memory).
            let dup = unsafe { libc_dup_cloexec(fd) };
            if dup < 0 {
                for f in out {
                    // SAFETY: f is a fd we just dup'd and own; close exactly once.
                    unsafe {
                        close_fd(f);
                    }
                }
                return Err(std::io::Error::last_os_error());
            }
            out.push(dup);
        }
        Ok(out)
    }
}

impl Drop for ExportedSurface {
    fn drop(&mut self) {
        for &fd in &self.fds {
            // SAFETY: each fd was handed to us by vaExportSurfaceHandle and is owned here;
            // closed exactly once (dup_fds hands out copies, never these).
            unsafe {
                close_fd(fd);
            }
        }
    }
}

// Minimal libc shims so the crate keeps its "no non-core dependency" posture (it already
// hand-rolls the whole libva FFI). `fcntl`/`close` are the two POSIX calls the fd hand-off
// needs; both are in the C runtime this crate already links (via libva-drm).
extern "C" {
    fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    fn close(fd: c_int) -> c_int;
}

/// `F_DUPFD_CLOEXEC` (fcntl.h, Linux) — dup to the lowest free fd ≥ arg with CLOEXEC set.
const F_DUPFD_CLOEXEC: c_int = 1030;

/// `dup(fd)` with the close-on-exec flag, via `fcntl(fd, F_DUPFD_CLOEXEC, 0)` — the
/// race-free CLOEXEC dup (a plain `dup` + `fcntl(F_SETFD)` leaves a window where a
/// concurrent `exec` would inherit the fd). Returns the new fd, or < 0 on error.
///
/// # Safety
/// `fd` must be a live file descriptor.
unsafe fn libc_dup_cloexec(fd: c_int) -> c_int {
    fcntl(fd, F_DUPFD_CLOEXEC, 0)
}

/// `close(fd)`, ignoring the result (a Drop has nowhere to report it).
///
/// # Safety
/// `fd` must be a live, owned file descriptor closed exactly once.
unsafe fn close_fd(fd: c_int) {
    close(fd);
}

/// The **write-path** inverse of [`MappedImage`]: a CPU-writable NV12 view over an
/// encode *input* surface. `vaDeriveImage` when the driver allows it (zero-copy on
/// Intel iHD — rows land straight in the surface); otherwise a staging
/// `vaCreateImage` whose pixels reach the surface via `vaPutImage` in
/// [`finish`](Self::finish).
///
/// The caller writes rows through [`row_mut`] and **must** call `finish()`; a plain
/// drop releases the mapping without the PutImage upload, so a staged (non-derived)
/// write would be lost — fine for error paths, wrong for success paths.
pub struct UploadImage {
    dpy: ffi::VADisplay,
    surface: ffi::VASurfaceID,
    image: ffi::VAImage,
    base: *mut u8,
    derived: bool,
    finished: bool,
}

impl UploadImage {
    /// Obtain a writable NV12 image over `surface` (`width`×`height` is the
    /// surface's full allocation) and map it.
    pub fn acquire(
        dpy: ffi::VADisplay,
        surface: ffi::VASurfaceID,
        width: u32,
        height: u32,
    ) -> VaResult<UploadImage> {
        // The surface may still be the source of an in-flight encode; sync before
        // scribbling over it (same fence the readback path takes).
        // SAFETY: surface is a live surface on this display.
        check(unsafe { ffi::vaSyncSurface(dpy, surface) }, "vaSyncSurface(upload)")?;

        let mut image: ffi::VAImage = unsafe { std::mem::zeroed() };
        image.image_id = ffi::VA_INVALID_ID;
        // SAFETY: image is a valid out struct; surface is synced.
        let derive = unsafe { ffi::vaDeriveImage(dpy, surface, &mut image) };
        let derived = derive == ffi::VA_STATUS_SUCCESS && image.format.fourcc == ffi::VA_FOURCC_NV12;

        if !derived {
            if derive == ffi::VA_STATUS_SUCCESS {
                // Derived but not NV12 — release and stage through CreateImage.
                // SAFETY: image_id is the live derived image.
                unsafe {
                    ffi::vaDestroyImage(dpy, image.image_id);
                }
            }
            let mut fmt = ffi::VAImageFormat {
                fourcc: ffi::VA_FOURCC_NV12,
                byte_order: 1, // VA_LSB_FIRST
                bits_per_pixel: 12,
                depth: 0,
                red_mask: 0,
                green_mask: 0,
                blue_mask: 0,
                alpha_mask: 0,
                va_reserved: [0; ffi::VA_PADDING_LOW],
            };
            let mut created: ffi::VAImage = unsafe { std::mem::zeroed() };
            // SAFETY: fmt/created are valid out structs; dims are the surface's.
            check(
                unsafe { ffi::vaCreateImage(dpy, &mut fmt, width as i32, height as i32, &mut created) },
                "vaCreateImage(upload)",
            )?;
            image = created;
        }

        let mut base: *mut std::ffi::c_void = ptr::null_mut();
        // SAFETY: image.buf is the image's backing buffer; base out ptr valid.
        let st = unsafe { ffi::vaMapBuffer(dpy, image.buf, &mut base) };
        if let Err(e) = check(st, "vaMapBuffer(upload)") {
            // SAFETY: image is live; release before propagating.
            unsafe {
                ffi::vaDestroyImage(dpy, image.image_id);
            }
            return Err(e);
        }
        Ok(UploadImage {
            dpy,
            surface,
            image,
            base: base as *mut u8,
            derived,
            finished: false,
        })
    }

    /// Byte pitch (stride) of plane `p` — rows must be written at this stride, not
    /// the tight width.
    pub fn pitch(&self, p: usize) -> u32 {
        self.image.pitches[p]
    }

    /// One writable row of plane `p`: `len` bytes at the plane offset + `row*pitch`.
    /// Bounds-checked against `data_size` like the readback path.
    pub fn row_mut(&mut self, plane: usize, row: u32, len: usize) -> Option<&mut [u8]> {
        if plane >= self.image.num_planes as usize || plane >= 3 {
            return None;
        }
        let off =
            self.image.offsets[plane] as usize + row as usize * self.image.pitches[plane] as usize;
        if off + len > self.image.data_size as usize {
            return None;
        }
        // SAFETY: base maps `data_size` bytes; [off, off+len) is in range (checked);
        // exclusive &mut self guards aliasing; lifetime tied to the live map.
        Some(unsafe { std::slice::from_raw_parts_mut(self.base.add(off), len) })
    }

    /// Commit the written pixels: unmap, and for the staged (non-derived) path push
    /// them into the surface with `vaPutImage`. Consumes the view.
    pub fn finish(mut self) -> VaResult<()> {
        self.finished = true;
        // SAFETY: buf is mapped (acquire succeeded); unmap exactly once here — the
        // Drop impl skips it for finished views.
        check(unsafe { ffi::vaUnmapBuffer(self.dpy, self.image.buf) }, "vaUnmapBuffer(upload)")?;
        let put = if self.derived {
            Ok(())
        } else {
            let (w, h) = (self.image.width as u32, self.image.height as u32);
            // SAFETY: image holds staged pixels; surface is a live target; regions
            // are the full image/surface extent.
            check(
                unsafe {
                    ffi::vaPutImage(self.dpy, self.surface, self.image.image_id, 0, 0, w, h, 0, 0, w, h)
                },
                "vaPutImage",
            )
        };
        // SAFETY: image is live and unmapped; destroy exactly once (Drop skips).
        unsafe {
            ffi::vaDestroyImage(self.dpy, self.image.image_id);
        }
        put
    }
}

impl Drop for UploadImage {
    fn drop(&mut self) {
        if self.finished {
            return; // finish() already unmapped + destroyed
        }
        // Error-path cleanup: release the mapping and image without uploading.
        // SAFETY: buf is mapped and image live; unmap then destroy, exactly once.
        unsafe {
            ffi::vaUnmapBuffer(self.dpy, self.image.buf);
            ffi::vaDestroyImage(self.dpy, self.image.image_id);
        }
    }
}
