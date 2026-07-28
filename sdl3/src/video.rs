//! Thin safe layer over `sdl3-sys` — exactly the calls the sink (and later the
//! scope UI) needs, nothing more. All `unsafe` lives here.
//!
//! Presentation path: one streaming **IYUV** texture (`SDL_PIXELFORMAT_IYUV` *is*
//! tightly-packed planar I420, chroma dims `ceil(w/2) × ceil(h/2)` — the same
//! ffmpeg-convention layout `streamcraft-video` uses), uploaded per frame with
//! `SDL_UpdateYUVTexture` and drawn with letterboxed logical presentation. YUV→RGB
//! happens on the GPU via SDL's renderer — the CPU conversion the old shm sink
//! carried is gone. Gray8 presents as IYUV with constant-128 chroma planes
//! (Y'CbCr with Cb=Cr=128 is exactly the achromatic axis — ITU-R BT.601 §2.5.1).
//!
//! Threading: everything here runs on the calling element's thread. SDL officially
//! prefers video on the process main thread; on Linux (Wayland/X11 drivers — this
//! crate's posture, like `sc-pipewire`) the video subsystem is an in-process
//! protocol client and works from a single non-main thread. One window per
//! process — matching the one-video-sink-per-pipeline reality.

#![allow(unsafe_code)]

use std::ffi::{c_int, CStr, CString};

use sdl3_sys::everything::*;

use streamcraft_core::error::Error;

/// The last SDL error as a `String` (empty when SDL has none).
fn sdl_error() -> String {
    // SAFETY: SDL_GetError returns a pointer to a thread-local NUL-terminated
    // string, valid until the next SDL call on this thread; we copy it out.
    unsafe { CStr::from_ptr(SDL_GetError()) }.to_string_lossy().into_owned()
}

fn resource(what: &str) -> Error {
    Error::Resource(format!("sdl3: {what}: {}", sdl_error()))
}

/// What [`VideoWindow::pump`] observed since the last call.
#[derive(Default, Clone, Copy)]
pub struct Pump {
    /// The user asked to close the window (window close / quit).
    pub closed: bool,
}

/// A window + renderer + (lazily sized) streaming IYUV texture.
///
/// Two shapes: the classic path holds an `SDL_Renderer` + streaming texture and
/// converts YUV via SDL's fixed function; the GPU path holds only the window (the
/// [`GpuRenderer`](crate::gpu::GpuRenderer) owns the device/swapchain claimed on this
/// window). A window is created without a classic renderer when built for GPU, since
/// a claimed GPU device and an `SDL_Renderer` on the same window conflict.
pub struct VideoWindow {
    window: *mut SDL_Window,
    /// The classic-path renderer, or null when the window was opened for the GPU
    /// backend (the GPU device owns presentation then).
    renderer: *mut SDL_Renderer,
    texture: *mut SDL_Texture,
    tex_w: c_int,
    tex_h: c_int,
    /// The streaming texture's pixel format (IYUV for i420/gray8, NV12 for
    /// hardware-decoder output) — a format change recreates the texture.
    tex_fmt: SDL_PixelFormat,
    /// Constant-128 chroma plane for gray8 presentation, sized `cw*ch` on demand.
    flat_chroma: Vec<u8>,
}

// SAFETY: the raw SDL pointers are used only by the owning element, which lives on
// one streaming thread at a time (the pipeline may move it between runs). SDL video
// is single-threaded, not thread-bound-at-creation on the Linux drivers this crate
// targets; `VideoWindow` is Send (moved whole), never Sync (no sharing).
unsafe impl Send for VideoWindow {}

impl VideoWindow {
    /// Init SDL video and open a resizable window with a classic `SDL_Renderer`
    /// (the fixed-function YUV presentation path). Errors (no display, no driver) are
    /// for the caller to degrade on — a headless box must not fail a pipeline.
    pub fn open(title: &str, width: u32, height: u32) -> Result<Self, Error> {
        let mut w = Self::open_windowed(title, width, height)?;
        // SAFETY: `w.window` is a live window we own; creating its classic renderer.
        unsafe {
            let renderer = SDL_CreateRenderer(w.window, std::ptr::null());
            if renderer.is_null() {
                // Drop of `w` tears down the window + subsystem ref.
                return Err(resource("create renderer"));
            }
            w.renderer = renderer;
        }
        Ok(w)
    }

    /// Init SDL video and open a resizable window with **no** classic renderer — for
    /// the owned GPU backend, which claims the window with its own device. The caller
    /// must attach a [`GpuRenderer`](crate::gpu::GpuRenderer) via [`raw_window`].
    // COLD: one-time window construction — `flat_chroma` starts empty and is grown
    // on demand (and reused) in `present_gray8`, never reallocated per frame.
    #[allow(clippy::disallowed_methods)]
    pub fn open_windowed(title: &str, width: u32, height: u32) -> Result<Self, Error> {
        let title = CString::new(title).unwrap_or_default();
        // Prefer Wayland (fall back to X11) — must precede the video-subsystem init that
        // selects the driver.
        crate::prefer_wayland_driver();
        // SAFETY: plain C calls; SDL_InitSubSystem is refcounted per subsystem.
        unsafe {
            if !SDL_InitSubSystem(SDL_INIT_VIDEO) {
                return Err(resource("init video"));
            }
            let window = SDL_CreateWindow(
                title.as_ptr(),
                width.max(1) as c_int,
                height.max(1) as c_int,
                SDL_WINDOW_RESIZABLE,
            );
            if window.is_null() {
                SDL_QuitSubSystem(SDL_INIT_VIDEO);
                return Err(resource("create window"));
            }
            Ok(Self {
                window,
                renderer: std::ptr::null_mut(),
                texture: std::ptr::null_mut(),
                tex_w: 0,
                tex_h: 0,
                tex_fmt: SDL_PIXELFORMAT_IYUV,
                flat_chroma: Vec::new(),
            })
        }
    }

    /// The raw `SDL_Window` — for the GPU backend to claim with its device. The
    /// window outlives the [`GpuRenderer`](crate::gpu::GpuRenderer): the renderer
    /// unclaims on drop, this window destroys on its own drop.
    pub fn raw_window(&self) -> *mut SDL_Window {
        self.window
    }

    /// (Re)create the streaming texture when the frame geometry or pixel format
    /// changes (IYUV for i420/gray8, NV12 for hardware-decoder output).
    fn ensure_texture(&mut self, w: c_int, h: c_int, fmt: SDL_PixelFormat) -> Result<(), Error> {
        if !self.texture.is_null() && self.tex_w == w && self.tex_h == h && self.tex_fmt == fmt {
            return Ok(());
        }
        // SAFETY: destroying a live texture we own; creating its replacement.
        unsafe {
            if !self.texture.is_null() {
                SDL_DestroyTexture(self.texture);
                self.texture = std::ptr::null_mut();
            }
            let tex = SDL_CreateTexture(
                self.renderer,
                fmt,
                SDL_TEXTUREACCESS_STREAMING,
                w,
                h,
            );
            if tex.is_null() {
                return Err(resource("create texture"));
            }
            // Letterbox the frame into whatever size the user drags the window to.
            SDL_SetRenderLogicalPresentation(self.renderer, w, h, SDL_LOGICAL_PRESENTATION_LETTERBOX);
            self.texture = tex;
            self.tex_w = w;
            self.tex_h = h;
            self.tex_fmt = fmt;
        }
        Ok(())
    }

    /// Present one tightly-packed NV12 frame (Y `w*h`, then one interleaved CbCr
    /// plane `2*cw × ch`; `cw = ceil(w/2)`, `ch = ceil(h/2)`) — the layout every
    /// VA-API/hardware decoder emits. SDL renders NV12 textures natively, so no
    /// CPU conversion happens on this path.
    pub fn present_nv12(&mut self, data: &[u8], width: usize, height: usize) -> Result<bool, Error> {
        if width == 0 || height == 0 {
            return Ok(true);
        }
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);
        let y_size = width * height;
        let uv_size = 2 * cw * ch;
        if data.len() < y_size + uv_size {
            return Ok(false);
        }
        let (w, h) = (width as c_int, height as c_int);
        self.ensure_texture(w, h, SDL_PIXELFORMAT_NV12)?;
        // SAFETY: the texture is streaming NV12 of exactly (w, h); the Y and the
        // interleaved UV plane cover the full rect at the given pitches (length
        // checked above).
        unsafe {
            if !SDL_UpdateNVTexture(
                self.texture,
                std::ptr::null(),
                data.as_ptr(),
                w,
                data[y_size..].as_ptr(),
                (2 * cw) as c_int,
            ) {
                return Err(resource("upload frame (nv12)"));
            }
            SDL_RenderClear(self.renderer);
            if !SDL_RenderTexture(self.renderer, self.texture, std::ptr::null(), std::ptr::null()) {
                return Err(resource("draw frame"));
            }
            SDL_RenderPresent(self.renderer);
        }
        Ok(true)
    }

    /// Present one tightly-packed I420 frame (Y `w*h`, Cb `cw*ch`, Cr `cw*ch`;
    /// `cw = ceil(w/2)`, `ch = ceil(h/2)`). `false` (nothing drawn, no error) on a
    /// short/malformed frame — a caller never indexes out of bounds.
    pub fn present_i420(&mut self, data: &[u8], width: usize, height: usize) -> Result<bool, Error> {
        if width == 0 || height == 0 {
            return Ok(true);
        }
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);
        let y_size = width * height;
        let c_size = cw * ch;
        if data.len() < y_size + 2 * c_size {
            return Ok(false);
        }
        let y = &data[..y_size];
        let u = &data[y_size..y_size + c_size];
        let v = &data[y_size + c_size..y_size + 2 * c_size];
        self.present_planes(width, height, y, u, v)
    }

    /// Present one gray8 frame (`w*h` luma bytes) as IYUV with flat chroma.
    pub fn present_gray8(&mut self, data: &[u8], width: usize, height: usize) -> Result<bool, Error> {
        if width == 0 || height == 0 {
            return Ok(true);
        }
        if data.len() < width * height {
            return Ok(false);
        }
        let c_size = width.div_ceil(2) * height.div_ceil(2);
        if self.flat_chroma.len() < c_size {
            self.flat_chroma = vec![128u8; c_size];
        }
        let chroma = std::mem::take(&mut self.flat_chroma); // split the borrow
        let r = self.present_planes(width, height, &data[..width * height], &chroma, &chroma);
        self.flat_chroma = chroma;
        r
    }

    fn present_planes(
        &mut self,
        width: usize,
        height: usize,
        y: &[u8],
        u: &[u8],
        v: &[u8],
    ) -> Result<bool, Error> {
        let (w, h) = (width as c_int, height as c_int);
        self.ensure_texture(w, h, SDL_PIXELFORMAT_IYUV)?;
        let cw = width.div_ceil(2) as c_int;
        // SAFETY: the texture is streaming IYUV of exactly (w, h); the plane
        // pointers cover the full rect at the given pitches (checked by callers).
        unsafe {
            if !SDL_UpdateYUVTexture(
                self.texture,
                std::ptr::null(),
                y.as_ptr(),
                w,
                u.as_ptr(),
                cw,
                v.as_ptr(),
                cw,
            ) {
                return Err(resource("upload frame"));
            }
            SDL_RenderClear(self.renderer);
            if !SDL_RenderTexture(self.renderer, self.texture, std::ptr::null(), std::ptr::null()) {
                return Err(resource("draw frame"));
            }
            SDL_RenderPresent(self.renderer);
        }
        Ok(true)
    }

    /// Drain pending window events (between frames — cheap, no waiting). Close is
    /// reported; everything else (resize, focus, …) SDL handles internally.
    pub fn pump(&mut self) -> Pump {
        let mut out = Pump::default();
        // SAFETY: SDL_PollEvent fills the union; we read only the discriminant.
        unsafe {
            let mut ev: SDL_Event = std::mem::zeroed();
            while SDL_PollEvent(&mut ev) {
                let ty = SDL_EventType(ev.r#type);
                if ty == SDL_EVENT_QUIT || ty == SDL_EVENT_WINDOW_CLOSE_REQUESTED {
                    out.closed = true;
                }
            }
        }
        out
    }
}

impl Drop for VideoWindow {
    fn drop(&mut self) {
        // SAFETY: tearing down objects we own, then dropping our subsystem ref. The
        // classic renderer/texture are null for a GPU-backed window (the GpuRenderer,
        // dropped before this, already unclaimed the window from its device).
        unsafe {
            if !self.texture.is_null() {
                SDL_DestroyTexture(self.texture);
            }
            if !self.renderer.is_null() {
                SDL_DestroyRenderer(self.renderer);
            }
            SDL_DestroyWindow(self.window);
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }
    }
}
