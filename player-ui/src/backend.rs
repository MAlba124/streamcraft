//! The player's SDL3 backend — the crate's only `unsafe` module.
//!
//! This is a *video-augmented* copy of `profluens-scope`'s `ui::backend::Backend`
//! (we do not modify the scope crate; its `Backend` exposes no video layer). It keeps
//! the exact scope flow — refcounted SDL video init, one window + `SDL_Renderer`, the
//! font-atlas texture, the per-frame `Input` event pump, and the `DrawList` flush via
//! `SDL_RenderGeometryRaw` — and adds ONE thing: a **streaming YUV texture** the media
//! frame is uploaded to and drawn *under* the UI each frame.
//!
//! # Frame composition order (spec: profluens.md UI `<update>` §1209)
//!
//! Per presented frame:
//! 1. clear to black;
//! 2. if a decoded frame is available, upload its planes to the streaming YUV texture
//!    (`SDL_UpdateYUVTexture` for I420, `SDL_UpdateNVTexture` for NV12 — the classic
//!    `SDL_Renderer` does the YUV→RGB CSC on the GPU, no custom shader needed) and
//!    `SDL_RenderTexture` it **letterboxed** into the window (aspect preserved, black
//!    bars);
//! 3. flush the scope UI `DrawList` on top (transport bar, menus, stats) via
//!    `SDL_RenderGeometryRaw`, exactly as scope's backend does;
//! 4. `SDL_RenderPresent`.
//!
//! The video texture reuses the letterbox math the sink's `VideoWindow` used
//! (`SDL_RenderTexture` with a computed destination rect), but here the UI must draw at
//! *window* coordinates unscaled — so we do NOT use `SDL_SetRenderLogicalPresentation`
//! (that would also scale the UI); we compute the letterbox destination rect ourselves
//! and pass it to `SDL_RenderTexture`.
//!
//! SDL calls are cited at their point of use. The `unsafe` here is the same posture as
//! `scope::ui::backend` and `pf_sdl3::video`: SDL video on the Linux drivers this targets
//! is a single-threaded in-process protocol client, and every handle is owned by the one
//! thread that holds the `Backend`.

#![allow(unsafe_code)]

use std::ffi::{c_int, c_void, CStr, CString};

use sdl3_sys::everything::*;

use profluens_scope::ui::draw::{Batch, DrawList, TexId};
use profluens_scope::ui::font::Font;
use profluens_scope::ui::{Input, Key, Mods, MouseButton};

use crate::gl::GlVideo;
use pf_vaapi::gpuframe::GpuFrame;

use crate::framesink::SlotPix;

/// The last SDL error as an owned `String`.
fn sdl_error() -> String {
    // SAFETY: SDL_GetError returns a thread-local NUL-terminated string valid until the
    // next SDL call; we copy it out immediately.
    unsafe { CStr::from_ptr(SDL_GetError()) }.to_string_lossy().into_owned()
}

/// A backend init/runtime error (kept local — like scope, we do not thread core's Error
/// through the UI layer).
#[derive(Debug)]
pub struct BackendError(pub String);

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pfplay-ui: {}", self.0)
    }
}
impl std::error::Error for BackendError {}

fn err(what: &str) -> BackendError {
    BackendError(format!("{what}: {}", sdl_error()))
}

/// A drag-and-drop event: the user dropped a file onto the window this frame.
#[derive(Clone, Debug, Default)]
pub struct FrameEvents {
    /// A file path from `SDL_EVENT_DROP_FILE`, if one arrived this frame.
    pub dropped_file: Option<String>,
    /// `F` fullscreen toggle was requested (via the SDL keycode, since scope's `Key`
    /// enum has no `F` — handled here in the backend where raw keycodes are available).
    pub toggle_fullscreen: bool,
}

/// The active video presentation path. Chosen once at [`Backend::open`]: the zero-copy GLES
/// backend when the caller asked for it AND a GLES/EGL DMA-BUF-import context stood up, else
/// the classic SDL_Renderer + streaming-YUV-texture path (which also handles CPU/software
/// frames). The event pump, window, and font live on the [`Backend`] regardless.
enum VideoPath {
    /// SDL_Renderer + a streaming YUV texture the decoder's CPU frame is uploaded to (the
    /// original path; also the fallback when GL/EGL is unavailable or the decoder is software).
    SdlRenderer {
        renderer: *mut SDL_Renderer,
        font_tex: *mut SDL_Texture,
        /// The streaming video texture (lazily created, recreated on geometry/format change).
        video_tex: *mut SDL_Texture,
        vid_w: c_int,
        vid_h: c_int,
        vid_fmt: SDL_PixelFormat,
        /// Reused flat-128 chroma plane for presenting Gray8 as IYUV (achromatic axis).
        flat_chroma: Vec<u8>,
    },
    /// The zero-copy GLES external-image backend ([`crate::gl::GlVideo`]): a decoded VA
    /// surface's DMA-BUF is imported straight into a `GL_TEXTURE_EXTERNAL_OES` and sampled —
    /// no CPU readback, no texture upload.
    Gl(GlVideo),
}

/// SDL window + event pump + font + one [`VideoPath`] (SDL_Renderer or zero-copy GLES).
pub struct Backend {
    window: *mut SDL_Window,
    video: VideoPath,
    /// Persistent input state (level fields survive between frames).
    input: Input,
    /// Reusable per-frame geometry arena (bump; reset each render) — the scope arena type.
    arena: profluens_scope::ui::Arena,
    fullscreen: bool,
}

// SAFETY: the raw SDL handles are used only from the thread that owns the Backend; SDL
// video on the Linux drivers this targets is a single-threaded in-process protocol client
// (same posture as scope::ui::backend and pf_sdl3::video). Backend is moved whole, never
// shared, so Send but not Sync.
unsafe impl Send for Backend {}

impl Backend {
    /// Open a window titled `title` at `width`×`height` and upload `font`'s atlas, choosing
    /// the video path. When `want_gl` is set (the caller decoded video zero-copy), a GLES
    /// window is opened and the [`crate::gl::GlVideo`] external-image backend is attempted;
    /// if the GLES/EGL DMA-BUF-import context does not stand up, it falls back to the
    /// SDL_Renderer path automatically (so a machine without the extensions still plays,
    /// via the CPU texture upload). When `want_gl` is clear, the SDL_Renderer path is used
    /// directly (software decode, or no zero-copy decoder).
    ///
    /// Headless: with `SDL_VIDEODRIVER=dummy` the GLES context creation fails (the dummy
    /// driver has no GL), so `want_gl` cleanly degrades to the SDL_Renderer path — the
    /// property `--headless-frames` relies on.
    pub fn open(
        title: &str,
        width: u32,
        height: u32,
        font: &Font,
        want_gl: bool,
    ) -> Result<Self, BackendError> {
        let title_c = CString::new(title).unwrap_or_default();
        // SAFETY: plain C calls; SDL_InitSubSystem is refcounted per subsystem.
        unsafe {
            // Force SDL's GL contexts onto **EGL** (not GLX) when we intend the zero-copy
            // GLES/EGL backend: with both DISPLAY and WAYLAND_DISPLAY set, SDL may pick
            // X11/GLX, where `eglCreateImageKHR` and the DMA-BUF-import extensions do not
            // exist. EGL — on X11 and on Wayland — exposes them. Set before video init so the
            // driver honours it. Harmless when we fall back to the SDL_Renderer path.
            if want_gl {
                SDL_SetHint(c"SDL_VIDEO_FORCE_EGL".as_ptr(), c"1".as_ptr());
            }
            if !SDL_InitSubSystem(SDL_INIT_VIDEO) {
                return Err(err("init video"));
            }
            // Log the chosen video driver (wayland/x11/…) — the EGL path needs a driver that
            // provides an EGL display; this line makes a failed import diagnosable at a glance.
            let drv = SDL_GetCurrentVideoDriver();
            if !drv.is_null() {
                eprintln!("pfplay-ui: SDL video driver = {}", CStr::from_ptr(drv).to_string_lossy());
            }
            // The window flags: OPENGL when we may use the GLES backend (a renderer and a GL
            // context cannot both own the window; the flag is harmless if we fall back and
            // create a renderer, which SDL allows on an OPENGL window).
            let flags = if want_gl {
                SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL
            } else {
                SDL_WINDOW_RESIZABLE
            };
            let window = SDL_CreateWindow(
                title_c.as_ptr(),
                width.max(1) as c_int,
                height.max(1) as c_int,
                flags,
            );
            if window.is_null() {
                SDL_QuitSubSystem(SDL_INIT_VIDEO);
                return Err(err("create window"));
            }

            // Try the zero-copy GLES backend first when asked. On any failure it returns None
            // and we fall through to the SDL_Renderer path (logged inside GlVideo::create).
            let video = if want_gl {
                match GlVideo::create(window, font) {
                    Some(gl) => Some(VideoPath::Gl(gl)),
                    None => None,
                }
            } else {
                None
            };

            let video = match video {
                Some(v) => v,
                None => match Self::open_sdl_renderer(window, font) {
                    Ok(v) => v,
                    Err(e) => {
                        SDL_DestroyWindow(window);
                        SDL_QuitSubSystem(SDL_INIT_VIDEO);
                        return Err(e);
                    }
                },
            };

            let input = Input {
                window_w: width as f32,
                window_h: height as f32,
                ..Input::default()
            };

            Ok(Self {
                window,
                video,
                input,
                arena: profluens_scope::ui::Arena::with_capacity(1 << 16),
                fullscreen: false,
            })
        }
    }

    /// Build the SDL_Renderer video path on `window` (the fallback / software path).
    ///
    /// # Safety
    /// `window` is a live window we own; called once during `open`.
    unsafe fn open_sdl_renderer(
        window: *mut SDL_Window,
        font: &Font,
    ) -> Result<VideoPath, BackendError> {
        // Force a GPU renderer — the YUV→RGB conversion and the letterbox scaling must
        // run on the GPU, never the CPU "software" fallback (which was ~70% of playback
        // CPU in profiling). Prefer Vulkan, then OpenGL, then SDL's auto-pick (which still
        // orders hardware before software); a passed driver name pins the choice.
        let renderer = {
            let mut r = SDL_CreateRenderer(window, c"vulkan".as_ptr());
            if r.is_null() {
                r = SDL_CreateRenderer(window, c"opengl".as_ptr());
            }
            if r.is_null() {
                r = SDL_CreateRenderer(window, std::ptr::null());
            }
            r
        };
        if renderer.is_null() {
            return Err(err("create renderer"));
        }
        // Which backend SDL chose — "software" means all the YUV→RGB + letterbox scaling
        // runs on the CPU (a big cost); "opengl"/"vulkan"/… means the GPU does it.
        let rname = SDL_GetRendererName(renderer);
        if !rname.is_null() {
            eprintln!(
                "pfplay-ui: video path = SDL texture upload (SDL render driver = {})",
                std::ffi::CStr::from_ptr(rname).to_string_lossy()
            );
        }
        // Alpha blending so the font atlas + translucent transport bar composite over
        // the video.
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

        let font_tex = match upload_font_atlas(renderer, font) {
            Ok(t) => t,
            Err(e) => {
                SDL_DestroyRenderer(renderer);
                return Err(e);
            }
        };
        Ok(VideoPath::SdlRenderer {
            renderer,
            font_tex,
            video_tex: std::ptr::null_mut(),
            vid_w: 0,
            vid_h: 0,
            vid_fmt: SDL_PIXELFORMAT_IYUV,
            flat_chroma: Vec::new(),
        })
    }

    /// Whether the zero-copy GLES backend is the active video path.
    pub fn is_gl(&self) -> bool {
        matches!(self.video, VideoPath::Gl(_))
    }

    /// Import one zero-copy decoded frame's DMA-BUF into the GLES external texture. No-op
    /// (returns `false`) unless the GLES backend is active. Returns whether the frame was
    /// imported (a failed import keeps the previously-displayed frame).
    ///
    /// # Safety
    /// The GL context (owned by this backend) must be current on the calling thread.
    pub unsafe fn import_gpu_frame(&mut self, frame: GpuFrame) -> bool {
        match &mut self.video {
            VideoPath::Gl(gl) => gl.import_frame(frame),
            // No GL path — the frame's fds close on drop (nothing imported them).
            VideoPath::SdlRenderer { .. } => false,
        }
    }

    /// Pump SDL's event queue into the [`Input`] snapshot for this frame and return it,
    /// plus the player-specific [`FrameEvents`] (drag-drop, fullscreen toggle). Edge
    /// fields describe only this frame; level fields (mouse pos, held buttons, window
    /// size) persist. This mirrors scope's `begin_frame`, extended with `SDL_EVENT_DROP_FILE`
    /// and the raw `F` keycode.
    pub fn begin_frame(&mut self) -> (Input, FrameEvents) {
        self.input.begin_frame();
        let mut fe = FrameEvents::default();
        // SAFETY: SDL_PollEvent fills the event union; we read only the fields matching
        // the discriminant we checked.
        unsafe {
            let mut ev: SDL_Event = std::mem::zeroed();
            while SDL_PollEvent(&mut ev) {
                let ty = SDL_EventType(ev.r#type);
                match ty {
                    SDL_EVENT_QUIT | SDL_EVENT_WINDOW_CLOSE_REQUESTED => {
                        self.input.quit = true;
                    }
                    SDL_EVENT_MOUSE_MOTION => {
                        self.input.mouse_x = ev.motion.x;
                        self.input.mouse_y = ev.motion.y;
                    }
                    SDL_EVENT_MOUSE_BUTTON_DOWN | SDL_EVENT_MOUSE_BUTTON_UP => {
                        if let Some(b) = map_button(ev.button.button) {
                            let i = Input::button_index(b);
                            self.input.mouse_x = ev.button.x;
                            self.input.mouse_y = ev.button.y;
                            if ev.button.down {
                                self.input.mouse_down[i] = true;
                                self.input.mouse_pressed[i] = true;
                            } else {
                                self.input.mouse_down[i] = false;
                                self.input.mouse_released[i] = true;
                            }
                        }
                    }
                    SDL_EVENT_MOUSE_WHEEL => {
                        self.input.wheel += ev.wheel.y;
                    }
                    SDL_EVENT_KEY_DOWN => {
                        let k = ev.key.key;
                        // `F` toggles fullscreen — routed out-of-band because scope's `Key`
                        // enum has no letter keys; the player's own keymap (Space/M/Tab/i/
                        // Q/Esc/arrows/Home) is read from `Input::keys` and `Input::text`.
                        if k == SDLK_F {
                            fe.toggle_fullscreen = true;
                        }
                        if let Some(kk) = map_key(k) {
                            let m = ev.key.r#mod;
                            let mods = Mods {
                                ctrl: (m & SDL_KMOD_CTRL) != SDL_Keymod(0),
                                shift: (m & SDL_KMOD_SHIFT) != SDL_Keymod(0),
                                alt: (m & SDL_KMOD_ALT) != SDL_Keymod(0),
                            };
                            self.input.keys.push((kk, mods));
                        }
                        // Space and letter shortcuts arrive as text too; push the char so
                        // the app's keymap can read `Input::text`.
                        push_char_for_key(&mut self.input.text, k);
                    }
                    SDL_EVENT_DROP_FILE => {
                        // SAFETY: `data` is a NUL-terminated UTF-8 C string owned by SDL,
                        // valid for the duration of this event handling.
                        if !ev.drop.data.is_null() {
                            let s = CStr::from_ptr(ev.drop.data);
                            fe.dropped_file = Some(s.to_string_lossy().into_owned());
                        }
                    }
                    SDL_EVENT_WINDOW_RESIZED | SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED => {
                        let mut w: c_int = 0;
                        let mut h: c_int = 0;
                        SDL_GetWindowSize(self.window, &mut w, &mut h);
                        self.input.window_w = w as f32;
                        self.input.window_h = h as f32;
                    }
                    _ => {}
                }
            }
        }
        (self.input.clone(), fe)
    }

    /// The current window size in logical pixels (level state, queried directly so the
    /// first frame is right).
    pub fn window_size(&self) -> (f32, f32) {
        let mut w: c_int = 0;
        let mut h: c_int = 0;
        // SAFETY: reads the size of a window we own.
        unsafe {
            SDL_GetWindowSize(self.window, &mut w, &mut h);
        }
        (w as f32, h as f32)
    }

    /// Toggle real (desktop) fullscreen.
    pub fn toggle_fullscreen(&mut self) {
        self.fullscreen = !self.fullscreen;
        // SAFETY: setting fullscreen on a window we own.
        unsafe {
            SDL_SetWindowFullscreen(self.window, self.fullscreen);
        }
    }

    /// Compose one presented frame: black clear, the letterboxed video (if one is present),
    /// then the UI `DrawList` on top, then present. Dispatches on the [`VideoPath`]: the GLES
    /// backend draws the imported external texture + the DrawList geometry and
    /// `SDL_GL_SwapWindow`s; the SDL_Renderer path draws the uploaded YUV texture +
    /// `SDL_RenderGeometryRaw` batches and `SDL_RenderPresent`s. The UI geometry comes from
    /// the internal per-frame arena (reset first) — no steady-state vertex allocation.
    ///
    /// Returns `true` if a video frame was drawn (so a caller can assert non-empty output in
    /// the headless proof).
    pub fn render(&mut self, draw_video: bool, dl: &DrawList) -> bool {
        let (ww, wh) = self.window_size();
        self.arena.reset();
        match &mut self.video {
            VideoPath::Gl(gl) => {
                // SAFETY: the GL context is current (this backend owns it, single-threaded);
                // the DrawList batch slices live in `self.arena` until the next reset.
                unsafe { gl.render(ww, wh, dl, &mut self.arena) }
            }
            VideoPath::SdlRenderer { renderer, font_tex, video_tex, vid_w, vid_h, .. } => {
                let (renderer, font_tex, video_tex, vid_w, vid_h) =
                    (*renderer, *font_tex, *video_tex, *vid_w, *vid_h);
                let mut drew_video = false;
                // SAFETY: `renderer` is a live handle we own; every texture is one we create
                // and track; the UI batch slices live in `self.arena` until the next reset.
                unsafe {
                    SDL_SetRenderDrawColorFloat(renderer, 0.0, 0.0, 0.0, 1.0);
                    SDL_RenderClear(renderer);

                    if draw_video && !video_tex.is_null() && vid_w > 0 && vid_h > 0 {
                        let dst = letterbox(ww, wh, vid_w as f32, vid_h as f32);
                        SDL_RenderTexture(renderer, video_tex, std::ptr::null(), &dst);
                        drew_video = true;
                    }

                    let batches = dl.build(&mut self.arena);
                    for b in &batches {
                        apply_clip(renderer, b);
                        let tex = match b.tex {
                            TexId::Font => font_tex,
                            TexId::None => std::ptr::null_mut(),
                        };
                        render_batch(renderer, tex, b);
                    }
                    SDL_SetRenderClipRect(renderer, std::ptr::null());
                    SDL_RenderPresent(renderer);
                }
                drew_video
            }
        }
    }

    /// (Re)create the streaming video texture on a geometry/format change and upload the
    /// frame's planes (the SDL_Renderer path only — a no-op returning `false` under the GLES
    /// backend, which imports DMA-BUFs instead). Returns `false` (nothing drawn, no panic) on
    /// a short/malformed frame so a caller never indexes out of bounds.
    ///
    /// # Safety
    /// Called only from the GUI thread; the SDL_Renderer, when present, is live.
    pub unsafe fn upload_video_raw(&mut self, w: usize, h: usize, pix: SlotPix, bytes: &[u8]) -> bool {
        let VideoPath::SdlRenderer {
            renderer, video_tex, vid_w, vid_h, vid_fmt, flat_chroma, ..
        } = &mut self.video
        else {
            return false; // GLES path doesn't upload CPU frames
        };
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let want_fmt = match pix {
            SlotPix::Nv12 => SDL_PIXELFORMAT_NV12,
            SlotPix::I420 | SlotPix::Gray8 => SDL_PIXELFORMAT_IYUV,
        };
        if w == 0
            || h == 0
            || !ensure_video_tex(*renderer, video_tex, vid_w, vid_h, vid_fmt, w as c_int, h as c_int, want_fmt)
        {
            return false;
        }
        let video_tex = *video_tex;
        match pix {
            SlotPix::Nv12 => {
                let y_size = w * h;
                let uv_size = 2 * cw * ch;
                if bytes.len() < y_size + uv_size {
                    return false;
                }
                // SDL_UpdateNVTexture: Y plane at pitch w, interleaved CbCr at pitch 2*cw.
                SDL_UpdateNVTexture(
                    video_tex,
                    std::ptr::null(),
                    bytes.as_ptr(),
                    w as c_int,
                    bytes[y_size..].as_ptr(),
                    (2 * cw) as c_int,
                )
            }
            SlotPix::I420 => {
                let y_size = w * h;
                let c_size = cw * ch;
                if bytes.len() < y_size + 2 * c_size {
                    return false;
                }
                let y = bytes.as_ptr();
                let u = bytes[y_size..].as_ptr();
                let v = bytes[y_size + c_size..].as_ptr();
                // SDL_UpdateYUVTexture: three planes at pitches w, cw, cw.
                SDL_UpdateYUVTexture(video_tex, std::ptr::null(), y, w as c_int, u, cw as c_int, v, cw as c_int)
            }
            SlotPix::Gray8 => {
                let y_size = w * h;
                if bytes.len() < y_size {
                    return false;
                }
                // Present luma-only as IYUV with flat 128 chroma (achromatic axis).
                let c_size = cw * ch;
                if flat_chroma.len() < c_size {
                    flat_chroma.resize(c_size, 128);
                }
                let chroma = flat_chroma.as_ptr();
                SDL_UpdateYUVTexture(
                    video_tex,
                    std::ptr::null(),
                    bytes.as_ptr(),
                    w as c_int,
                    chroma,
                    cw as c_int,
                    chroma,
                    cw as c_int,
                )
            }
        }
    }

    /// Did the user request to close the window since the last `begin_frame`?
    pub fn should_close(&self) -> bool {
        self.input.quit
    }
}

/// (Re)create the streaming YUV texture when the geometry or pixel format changes (the
/// SDL_Renderer path). Free function so it can borrow the `VideoPath::SdlRenderer` fields
/// piecewise without a whole-`self` borrow.
///
/// # Safety
/// `renderer` must be live.
#[allow(clippy::too_many_arguments)]
unsafe fn ensure_video_tex(
    renderer: *mut SDL_Renderer,
    video_tex: &mut *mut SDL_Texture,
    vid_w: &mut c_int,
    vid_h: &mut c_int,
    vid_fmt: &mut SDL_PixelFormat,
    w: c_int,
    h: c_int,
    fmt: SDL_PixelFormat,
) -> bool {
    if !video_tex.is_null() && *vid_w == w && *vid_h == h && *vid_fmt == fmt {
        return true;
    }
    if !video_tex.is_null() {
        SDL_DestroyTexture(*video_tex);
        *video_tex = std::ptr::null_mut();
    }
    let tex = SDL_CreateTexture(renderer, fmt, SDL_TEXTUREACCESS_STREAMING, w, h);
    if tex.is_null() {
        return false;
    }
    // Linear scaling so the letterboxed video is smooth at non-integer window scales.
    SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_LINEAR);
    *video_tex = tex;
    *vid_w = w;
    *vid_h = h;
    *vid_fmt = fmt;
    true
}

impl Drop for Backend {
    fn drop(&mut self) {
        // SAFETY: tearing down objects we own, then dropping our subsystem ref. The GLES
        // backend (VideoPath::Gl) tears its own GL objects + context down in its Drop, run
        // when `self.video` drops below.
        unsafe {
            if let VideoPath::SdlRenderer { renderer, font_tex, video_tex, .. } = &self.video {
                if !video_tex.is_null() {
                    SDL_DestroyTexture(*video_tex);
                }
                if !font_tex.is_null() {
                    SDL_DestroyTexture(*font_tex);
                }
                SDL_DestroyRenderer(*renderer);
            }
            // `self.video` (incl. a VideoPath::Gl → GlVideo::drop → SDL_GL_DestroyContext)
            // drops here as part of Backend's fields; then the window + subsystem.
            SDL_DestroyWindow(self.window);
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }
    }
}

/// The letterbox destination rect: the largest rect of the frame's aspect ratio that fits
/// inside `(ww, wh)`, centred (equal black bars). This is the aspect-preserving fit
/// `SDL_LOGICAL_PRESENTATION_LETTERBOX` computes internally, done by hand so the UI can
/// draw at unscaled window coordinates over it. Shared with the GLES backend
/// ([`crate::gl`]), which places its video quad at the same rect.
pub(crate) fn letterbox(ww: f32, wh: f32, fw: f32, fh: f32) -> SDL_FRect {
    if fw <= 0.0 || fh <= 0.0 || ww <= 0.0 || wh <= 0.0 {
        return SDL_FRect { x: 0.0, y: 0.0, w: ww, h: wh };
    }
    let scale = (ww / fw).min(wh / fh);
    let (dw, dh) = (fw * scale, fh * scale);
    SDL_FRect { x: (ww - dw) * 0.5, y: (wh - dh) * 0.5, w: dw, h: dh }
}

/// Create the RGBA8 font-atlas texture and upload the glyph pixels (verbatim scope).
///
/// # Safety
/// `renderer` must be a live renderer.
unsafe fn upload_font_atlas(
    renderer: *mut SDL_Renderer,
    font: &Font,
) -> Result<*mut SDL_Texture, BackendError> {
    let (w, h) = font.atlas_size();
    let tex = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STATIC,
        w as c_int,
        h as c_int,
    );
    if tex.is_null() {
        return Err(err("create font texture"));
    }
    SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_LINEAR);
    SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND);
    let pixels = font.atlas_rgba();
    let pitch = (w * 4) as c_int;
    if !SDL_UpdateTexture(tex, std::ptr::null(), pixels.as_ptr() as *const c_void, pitch) {
        let e = err("upload font atlas");
        SDL_DestroyTexture(tex);
        return Err(e);
    }
    Ok(tex)
}

/// Set (or clear) the scissor rect for a batch (verbatim scope).
///
/// # Safety
/// `renderer` must be live.
unsafe fn apply_clip(renderer: *mut SDL_Renderer, b: &Batch) {
    match b.clip {
        Some(c) => {
            let rect = SDL_Rect {
                x: c.x.floor() as c_int,
                y: c.y.floor() as c_int,
                w: c.w.ceil().max(0.0) as c_int,
                h: c.h.ceil().max(0.0) as c_int,
            };
            SDL_SetRenderClipRect(renderer, &rect);
        }
        None => {
            SDL_SetRenderClipRect(renderer, std::ptr::null());
        }
    }
}

/// Issue one `SDL_RenderGeometryRaw` for a batch (verbatim scope).
///
/// # Safety
/// `renderer` live; `tex` a live texture or NULL; the batch slices are valid and their
/// strides match the raw-geometry contract.
unsafe fn render_batch(renderer: *mut SDL_Renderer, tex: *mut SDL_Texture, b: &Batch) {
    if b.indices.is_empty() {
        return;
    }
    SDL_RenderGeometryRaw(
        renderer,
        tex,
        b.xy.as_ptr(),
        (2 * std::mem::size_of::<f32>()) as c_int,
        b.color.as_ptr() as *const SDL_FColor,
        (4 * std::mem::size_of::<f32>()) as c_int,
        b.uv.as_ptr(),
        (2 * std::mem::size_of::<f32>()) as c_int,
        b.num_vertices as c_int,
        b.indices.as_ptr() as *const c_void,
        b.indices.len() as c_int,
        std::mem::size_of::<u16>() as c_int,
    );
}

/// Map an SDL mouse button constant to scope's [`MouseButton`].
fn map_button(b: u8) -> Option<MouseButton> {
    match b as u32 {
        x if x == SDL_BUTTON_LEFT as u32 => Some(MouseButton::Left),
        x if x == SDL_BUTTON_MIDDLE as u32 => Some(MouseButton::Middle),
        x if x == SDL_BUTTON_RIGHT as u32 => Some(MouseButton::Right),
        _ => None,
    }
}

/// Map an SDL keycode to scope's [`Key`] (the navigation keys scope's widgets and the
/// player's transport use — arrows, Home/End, Esc, Tab).
fn map_key(k: SDL_Keycode) -> Option<Key> {
    match k {
        SDLK_RETURN | SDLK_KP_ENTER => Some(Key::Enter),
        SDLK_ESCAPE => Some(Key::Escape),
        SDLK_TAB => Some(Key::Tab),
        SDLK_BACKSPACE => Some(Key::Backspace),
        SDLK_UP => Some(Key::Up),
        SDLK_DOWN => Some(Key::Down),
        SDLK_LEFT => Some(Key::Left),
        SDLK_RIGHT => Some(Key::Right),
        SDLK_PAGEUP => Some(Key::PageUp),
        SDLK_PAGEDOWN => Some(Key::PageDown),
        SDLK_HOME => Some(Key::Home),
        SDLK_END => Some(Key::End),
        _ => None,
    }
}

/// Push the printable char for the player's letter/space shortcuts (Space, Q, M, I) into
/// the frame's `Input::text`, so the app keymap reads them uniformly from one place. We do
/// NOT enable `SDL_StartTextInput` (this is not a text field), so `SDL_EVENT_TEXT_INPUT`
/// never fires — we synthesize the small set of shortcut chars from the keycode here.
fn push_char_for_key(text: &mut String, k: SDL_Keycode) {
    let c = match k {
        SDLK_SPACE => ' ',
        SDLK_Q => 'q',
        SDLK_M => 'm',
        SDLK_I => 'i',
        _ => return,
    };
    text.push(c);
}
