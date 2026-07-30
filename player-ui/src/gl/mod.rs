//! The **zero-copy GLES video backend** — the on-GPU display path that samples a
//! VA-API-decoded DMA-BUF directly, never touching the CPU (the mpv/gstreamer-gl model).
//!
//! # The pipeline this backend closes
//!
//! `vaapi decoder (zero-copy) → vaExportSurfaceHandle → DMA-BUF fds → [this backend] →
//! eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT) → glEGLImageTargetTexture2DOES(GL_TEXTURE_
//! EXTERNAL_OES) → samplerExternalOES → letterboxed quad → SDL_GL_SwapWindow`. The decoded
//! frame's pixels live in GPU memory the entire time; the only bytes that crossed the
//! process/thread boundary were the ~40-byte descriptor + a handful of fds
//! ([`pf_vaapi::gpuframe`]).
//!
//! # Extensions + the flow, cited at point of use
//!
//! * **`EGL_KHR_image_base` + `EGL_EXT_image_dma_buf_import`** — `eglCreateImageKHR` with
//!   `target = EGL_LINUX_DMA_BUF_EXT` and the `EGL_DMA_BUF_PLANE*_{FD,OFFSET,PITCH,
//!   MODIFIER_*}_EXT` attribute list wraps the exported planes as an `EGLImageKHR`
//!   (Khronos EGL_EXT_image_dma_buf_import spec). `ctx = EGL_NO_CONTEXT` (the image is
//!   context-independent).
//! * **`OES_EGL_image_external`** — `glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES,
//!   image)` binds that image to an external texture; a `samplerExternalOES` in the shader
//!   samples it and **the driver applies the YCbCr→RGB conversion** (the surface's BT.601/709
//!   matrix), so no manual CSC is needed for the composed-NV12 import. (If a driver instead
//!   handed us raw 2-plane NV12 — R8 luma + RG8 chroma — the shader would do BT.709 in-shader;
//!   see [`NV12_TWO_PLANE_FALLBACK`] for that documented, not-yet-active alternative.)
//!
//! # What is validated here vs on-GPU
//!
//! The whole external-sampler path needs a real EGL/GLES driver, which the sandbox's dummy
//! SDL video driver does not provide. So [`GlVideo::create`] degrades gracefully: if a GLES
//! context, the EGL display, or any required extension/symbol is missing, it returns `None`
//! and the caller keeps the [`crate::backend::Backend`] SDL_Renderer path. The struct
//! plumbing, the EGL attribute-list builder ([`dmabuf_egl_attribs`]), and the letterbox math
//! are unit-tested; the live import + draw are the user's on-GPU step.

mod loader;

use std::ffi::{c_int, c_void, CString};

use sdl3_sys::everything::*;

use loader::*;
use pf_vaapi::gpuframe::GpuFrame;
use profluens_scope::ui::draw::{Batch, DrawList, TexId};
use profluens_scope::ui::font::Font;

use crate::backend::letterbox;

/// A one-time note documenting the 2-plane-NV12 alternative import, kept OFF by default.
/// Most Mesa/iHD drivers hand a COMPOSED NV12 DMA-BUF that `samplerExternalOES` converts to
/// RGB itself (the active path). Should a driver instead present the two planes as separate
/// DRM formats (`DRM_FORMAT_R8` + `DRM_FORMAT_GR88`), the importer would create two
/// `GL_TEXTURE_2D` (R8 luma, RG8 chroma) and the fragment shader would do BT.709:
/// `R = Y + 1.5748·(Cr−0.5)`, etc. This is documented so the user can flip it fast if the
/// external sampler yields wrong colours; it is not wired because the COMPOSED export we ask
/// for is the standard, widely-working shape.
pub const NV12_TWO_PLANE_FALLBACK: bool = false;

/// The video fragment shader: sample the external (DMA-BUF-backed, driver-CSC'd) texture.
/// `#extension GL_OES_EGL_image_external : require` is mandatory for `samplerExternalOES`.
const VIDEO_FS: &str = r#"#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES uTex;
in vec2 vUv;
out vec4 oColor;
void main() { oColor = texture(uTex, vUv); }
"#;

/// The video vertex shader: a fullscreen-ish quad placed at the letterbox rect (positions
/// already in clip space from the CPU), passing UVs through.
const VIDEO_VS: &str = r#"#version 300 es
precision mediump float;
in vec2 aPos;
in vec2 aUv;
out vec2 vUv;
void main() { vUv = aUv; gl_Position = vec4(aPos, 0.0, 1.0); }
"#;

/// The UI shaders: draw scope's tessellated `DrawList` geometry (screen-pixel positions →
/// clip space via a uniform, per-vertex RGBA, optional font-atlas sample). One program does
/// both solid (`uUseTex = 0`) and textured (`uUseTex = 1`) batches.
const UI_VS: &str = r#"#version 300 es
precision mediump float;
uniform vec2 uViewport;        // window size in pixels
in vec2 aPos;                  // pixel coords, top-left origin
in vec2 aUv;
in vec4 aColor;
out vec2 vUv;
out vec4 vColor;
void main() {
    vUv = aUv;
    vColor = aColor;
    // pixel (0..w, 0..h, top-left) → clip (-1..1, +1..-1)
    vec2 ndc = vec2(aPos.x / uViewport.x * 2.0 - 1.0,
                    1.0 - aPos.y / uViewport.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"#;

const UI_FS: &str = r#"#version 300 es
precision mediump float;
uniform sampler2D uTex;
uniform int uUseTex;
in vec2 vUv;
in vec4 vColor;
out vec4 oColor;
void main() {
    vec4 c = vColor;
    if (uUseTex == 1) { c *= texture(uTex, vUv); }
    oColor = c;
}
"#;

/// An imported frame: the live external texture + its EGLImage, held until the next frame
/// replaces it (so the GL commands sampling it have completed by swap). Owns the imported
/// [`GpuFrame`] so its fds stay open exactly as long as the image references them.
struct ImportedFrame {
    image: EGLImageKHR,
    /// The surface geometry, for the letterbox + UV crop.
    coded_w: u32,
    coded_h: u32,
    disp_w: u32,
    disp_h: u32,
    crop_x: u32,
    crop_y: u32,
}

/// The GLES zero-copy video backend. Created on the SDL window after a GLES-3 context is set
/// up; `None` from [`create`](Self::create) means the fallback SDL_Renderer path must be used.
pub struct GlVideo {
    gl: Gl,
    egl_display: EGLDisplay,
    ctx: SDL_GLContext,
    window: *mut SDL_Window,
    // The external-OES video texture (reused; re-pointed at a new EGLImage each frame).
    video_tex: GLuint,
    video_prog: GLuint,
    video_pos_loc: GLint,
    video_uv_loc: GLint,
    // The UI program + its attribute/uniform locations + streaming VBO/IBO.
    ui_prog: GLuint,
    ui_pos_loc: GLint,
    ui_uv_loc: GLint,
    ui_color_loc: GLint,
    ui_viewport_loc: GLint,
    ui_usetex_loc: GLint,
    ui_tex_loc: GLint,
    ui_vbo_pos: GLuint,
    ui_vbo_uv: GLuint,
    ui_vbo_color: GLuint,
    ui_ibo: GLuint,
    font_tex: GLuint,
    /// The frame currently imported (its EGLImage + the GpuFrame keeping its fds alive).
    current: Option<(ImportedFrame, GpuFrame)>,
}

impl GlVideo {
    /// Try to stand up the GLES zero-copy backend on `window`. Sets the GLES-3 attributes,
    /// creates the context, loads GL+EGL, verifies the DMA-BUF-import extensions, compiles
    /// the shaders, and uploads the font atlas. Returns `None` (→ SDL_Renderer fallback) on
    /// any failure, logging the reason.
    ///
    /// # Safety
    /// `window` must be a live SDL window created WITHOUT a renderer bound (the GL context
    /// and an `SDL_Renderer` cannot both own the window). Call once, on the GUI thread.
    pub unsafe fn create(window: *mut SDL_Window, font: &Font) -> Option<GlVideo> {
        // Request a GLES 3.0 context (samplerExternalOES via the ESSL3 ext). GLES2 + the
        // extension also works, but ESSL3 shaders are cleaner; if 3.0 is unavailable the
        // context creation fails and we fall back.
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES.0 as c_int);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

        let ctx = SDL_GL_CreateContext(window);
        if ctx.is_null() {
            log_fallback("could not create a GLES 3 context");
            return None;
        }
        if !SDL_GL_MakeCurrent(window, ctx) {
            log_fallback("SDL_GL_MakeCurrent failed");
            SDL_GL_DestroyContext(ctx);
            return None;
        }
        // Sync presentation to the display refresh. Without this `SDL_GL_SwapWindow` returns
        // immediately, so the loop presents unthrottled and out of phase with the display —
        // the frame the eye sees updates at an uneven cadence (some frames held an extra
        // refresh, tearing), which reads as "repeated frames". Vsync throttles the loop to the
        // refresh and makes the cadence even. Try adaptive (-1, tear-free late-swap fallback),
        // then plain vsync (1); both are best-effort (a compositor may override).
        if !SDL_GL_SetSwapInterval(-1) {
            let _ = SDL_GL_SetSwapInterval(1);
        }

        let Some(gl) = Gl::load() else {
            log_fallback("a required GL/EGL symbol is missing (glEGLImageTargetTexture2DOES / \
                eglCreateImageKHR — no OES_EGL_image_external or EGL_EXT_image_dma_buf_import)");
            SDL_GL_DestroyContext(ctx);
            return None;
        };

        // The EGL display SDL is running on (present only for the EGL/GLES path; a GLX/WGL
        // context returns NULL and we fall back).
        let egl_display = SDL_EGL_GetCurrentDisplay();
        if egl_display.is_null() {
            log_fallback("no current EGLDisplay (not an EGL-backed GL context)");
            SDL_GL_DestroyContext(ctx);
            return None;
        }

        // Require the two GL/EGL extensions by name (belt-and-braces beyond symbol presence).
        if !gl_extension_supported("GL_OES_EGL_image_external")
            && !gl_extension_supported("GL_OES_EGL_image_external_essl3")
        {
            log_fallback("GL_OES_EGL_image_external not advertised");
            SDL_GL_DestroyContext(ctx);
            return None;
        }

        // Compile the two programs.
        let Some(video_prog) = link_program(&gl, VIDEO_VS, VIDEO_FS) else {
            log_fallback("video (external-sampler) shader failed to compile/link");
            SDL_GL_DestroyContext(ctx);
            return None;
        };
        let Some(ui_prog) = link_program(&gl, UI_VS, UI_FS) else {
            log_fallback("UI shader failed to compile/link");
            (gl.UseProgram)(0);
            SDL_GL_DestroyContext(ctx);
            return None;
        };

        let video_pos_loc = attrib(&gl, video_prog, "aPos");
        let video_uv_loc = attrib(&gl, video_prog, "aUv");
        let ui_pos_loc = attrib(&gl, ui_prog, "aPos");
        let ui_uv_loc = attrib(&gl, ui_prog, "aUv");
        let ui_color_loc = attrib(&gl, ui_prog, "aColor");
        let ui_viewport_loc = uniform(&gl, ui_prog, "uViewport");
        let ui_usetex_loc = uniform(&gl, ui_prog, "uUseTex");
        let ui_tex_loc = uniform(&gl, ui_prog, "uTex");

        // The reused external-OES texture (its image is set per frame).
        let mut video_tex: GLuint = 0;
        (gl.GenTextures)(1, &mut video_tex);
        (gl.BindTexture)(GL_TEXTURE_EXTERNAL_OES, video_tex);
        (gl.TexParameteri)(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        (gl.TexParameteri)(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        (gl.TexParameteri)(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        (gl.TexParameteri)(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Streaming VBOs/IBO for the UI geometry.
        let mut bufs = [0u32; 4];
        (gl.GenBuffers)(4, bufs.as_mut_ptr());
        let [ui_vbo_pos, ui_vbo_uv, ui_vbo_color, ui_ibo] = bufs;

        // The font atlas as an RGBA8 GL_TEXTURE_2D.
        let font_tex = upload_font_atlas(&gl, font);

        eprintln!("pfplay-ui: video path = zero-copy VA-API/EGL (GLES external-image backend live)");

        Some(GlVideo {
            gl,
            egl_display,
            ctx,
            window,
            video_tex,
            video_prog,
            video_pos_loc,
            video_uv_loc,
            ui_prog,
            ui_pos_loc,
            ui_uv_loc,
            ui_color_loc,
            ui_viewport_loc,
            ui_usetex_loc,
            ui_tex_loc,
            ui_vbo_pos,
            ui_vbo_uv,
            ui_vbo_color,
            ui_ibo,
            font_tex,
            current: None,
        })
    }

    /// Import one decoded frame's DMA-BUF(s) into the external texture. Replaces the
    /// previously-imported frame (destroying its EGLImage, which lets the decoder's surface
    /// be released once this frame's token is `release`d by the caller). Returns `false`
    /// (frame skipped, keep the old one) on an import failure, logging once-ish.
    ///
    /// # Safety
    /// The GL context must be current; `frame`'s fds must be live (they are — the decoder
    /// dup'd them and this owns the `GpuFrame`).
    pub unsafe fn import_frame(&mut self, frame: GpuFrame) -> bool {
        let attribs = dmabuf_egl_attribs(&frame);
        let image = (self.gl.eglCreateImageKHR)(
            self.egl_display,
            EGL_NO_CONTEXT,
            EGL_LINUX_DMA_BUF_EXT,
            std::ptr::null_mut(),
            attribs.as_ptr(),
        );
        if image == EGL_NO_IMAGE_KHR {
            let e = (self.gl.eglGetError)();
            eprintln!(
                "pfplay-ui: eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT) failed (EGL error {e:#x}); \
                 frame skipped — check the DMA-BUF fourcc/modifier the driver exported"
            );
            // frame drops here → its fds close; the old imported frame stays displayed.
            return false;
        }
        // Bind the image to the external texture.
        (self.gl.BindTexture)(GL_TEXTURE_EXTERNAL_OES, self.video_tex);
        (self.gl.EGLImageTargetTexture2DOES)(GL_TEXTURE_EXTERNAL_OES, image);

        let imported = ImportedFrame {
            image,
            coded_w: frame.coded_w,
            coded_h: frame.coded_h,
            disp_w: frame.disp_w,
            disp_h: frame.disp_h,
            crop_x: frame.crop_x,
            crop_y: frame.crop_y,
        };
        // Destroy the previous image + drop its GpuFrame (closing the old fds) only AFTER the
        // new one is bound, so there is always a valid texture to sample.
        if let Some((old, _old_frame)) = self.current.take() {
            (self.gl.eglDestroyImageKHR)(self.egl_display, old.image);
        }
        self.current = Some((imported, frame));
        true
    }

    /// Whether a video frame is currently imported (so `render` draws the video quad).
    pub fn has_video(&self) -> bool {
        self.current.is_some()
    }

    /// Compose one frame: black clear, the letterboxed external-texture video (if imported),
    /// then the UI `DrawList`, then `SDL_GL_SwapWindow`. `(ww, wh)` is the drawable size.
    /// Returns `true` if the video quad was drawn.
    ///
    /// # Safety
    /// GL context current; `dl`'s batch slices live for the call.
    pub unsafe fn render(&mut self, ww: f32, wh: f32, dl: &DrawList, arena: &mut profluens_scope::ui::Arena) -> bool {
        let gl = &self.gl;
        (gl.Viewport)(0, 0, ww as GLsizei, wh as GLsizei);
        (gl.ClearColor)(0.0, 0.0, 0.0, 1.0);
        (gl.Clear)(GL_COLOR_BUFFER_BIT);
        (gl.Disable)(GL_SCISSOR_TEST);

        let mut drew_video = false;
        if let Some((frame, _)) = &self.current {
            self.draw_video_quad(ww, wh, frame);
            drew_video = true;
        }

        // UI on top (alpha-blended geometry).
        self.draw_ui(ww, wh, dl, arena);

        SDL_GL_SwapWindow(self.window);
        drew_video
    }

    /// Draw the letterboxed video quad from the external texture. UVs crop the display
    /// region out of the (possibly larger) coded surface.
    ///
    /// # Safety
    /// Context current; `self.current` is `Some(frame)`.
    unsafe fn draw_video_quad(&self, ww: f32, wh: f32, frame: &ImportedFrame) {
        let gl = &self.gl;
        let dst = letterbox(ww, wh, frame.disp_w as f32, frame.disp_h as f32);
        // Letterbox rect (pixels, top-left) → clip space.
        let x0 = dst.x / ww * 2.0 - 1.0;
        let x1 = (dst.x + dst.w) / ww * 2.0 - 1.0;
        let y0 = 1.0 - dst.y / wh * 2.0;
        let y1 = 1.0 - (dst.y + dst.h) / wh * 2.0;
        // UV crop: the display region within the coded surface (0..1). crop origin + display
        // extent over the coded extent. Almost always (0,0)..(disp/coded).
        let (cw, ch) = (frame.coded_w.max(1) as f32, frame.coded_h.max(1) as f32);
        let u0 = frame.crop_x as f32 / cw;
        let v0 = frame.crop_y as f32 / ch;
        let u1 = (frame.crop_x + frame.disp_w) as f32 / cw;
        let v1 = (frame.crop_y + frame.disp_h) as f32 / ch;
        // Two triangles (TL, TR, BR, BL) as a strip: TL, BL, TR, BR.
        let verts: [f32; 8] = [x0, y0, x0, y1, x1, y0, x1, y1];
        let uvs: [f32; 8] = [u0, v0, u0, v1, u1, v0, u1, v1];

        (gl.UseProgram)(self.video_prog);
        (gl.ActiveTexture)(GL_TEXTURE0);
        (gl.BindTexture)(GL_TEXTURE_EXTERNAL_OES, self.video_tex);
        // Positions.
        (gl.BindBuffer)(GL_ARRAY_BUFFER, self.ui_vbo_pos);
        (gl.BufferData)(
            GL_ARRAY_BUFFER,
            std::mem::size_of_val(&verts) as isize,
            verts.as_ptr() as *const c_void,
            GL_STREAM_DRAW,
        );
        (gl.EnableVertexAttribArray)(self.video_pos_loc as GLuint);
        (gl.VertexAttribPointer)(self.video_pos_loc as GLuint, 2, GL_FLOAT, GL_FALSE, 0, std::ptr::null());
        // UVs.
        (gl.BindBuffer)(GL_ARRAY_BUFFER, self.ui_vbo_uv);
        (gl.BufferData)(
            GL_ARRAY_BUFFER,
            std::mem::size_of_val(&uvs) as isize,
            uvs.as_ptr() as *const c_void,
            GL_STREAM_DRAW,
        );
        (gl.EnableVertexAttribArray)(self.video_uv_loc as GLuint);
        (gl.VertexAttribPointer)(self.video_uv_loc as GLuint, 2, GL_FLOAT, GL_FALSE, 0, std::ptr::null());
        (gl.Disable)(GL_BLEND);
        (gl.DrawArrays)(GL_TRIANGLE_STRIP, 0, 4);
    }

    /// Draw scope's tessellated UI `DrawList` as GLES geometry (one draw per batch), matching
    /// the SDL_Renderer backend's composition (alpha-over the video).
    ///
    /// # Safety
    /// Context current; batch slices valid for the call.
    unsafe fn draw_ui(&mut self, ww: f32, wh: f32, dl: &DrawList, arena: &mut profluens_scope::ui::Arena) {
        let batches = dl.build(arena);
        let gl = &self.gl;
        (gl.UseProgram)(self.ui_prog);
        (gl.Uniform2f)(self.ui_viewport_loc, ww, wh);
        (gl.Uniform1i)(self.ui_tex_loc, 0); // texture unit 0
        (gl.Enable)(GL_BLEND);
        (gl.BlendFunc)(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        (gl.PixelStorei)(GL_UNPACK_ALIGNMENT, 1);

        for b in &batches {
            if b.indices.is_empty() {
                continue;
            }
            // Scissor (clip) rect — GL's origin is bottom-left, so flip Y.
            match b.clip {
                Some(c) => {
                    (gl.Enable)(GL_SCISSOR_TEST);
                    let x = c.x.floor().max(0.0) as GLint;
                    let y = (wh - (c.y + c.h)).floor().max(0.0) as GLint;
                    (gl.Scissor)(x, y, c.w.ceil().max(0.0) as GLsizei, c.h.ceil().max(0.0) as GLsizei);
                }
                None => (gl.Disable)(GL_SCISSOR_TEST),
            }
            let use_tex = matches!(b.tex, TexId::Font);
            (gl.Uniform1i)(self.ui_usetex_loc, if use_tex { 1 } else { 0 });
            if use_tex {
                (gl.ActiveTexture)(GL_TEXTURE0);
                (gl.BindTexture)(GL_TEXTURE_2D, self.font_tex);
            }
            // Upload the batch's arrays into the streaming buffers.
            self.upload_and_draw_ui_batch(b);
        }
        (gl.Disable)(GL_SCISSOR_TEST);
    }

    /// Upload one UI batch's pos/uv/color/index arrays and draw it.
    ///
    /// # Safety
    /// Context current; program bound; batch slices valid.
    unsafe fn upload_and_draw_ui_batch(&self, b: &Batch<'_>) {
        let gl = &self.gl;
        (gl.BindBuffer)(GL_ARRAY_BUFFER, self.ui_vbo_pos);
        (gl.BufferData)(
            GL_ARRAY_BUFFER,
            std::mem::size_of_val(b.xy) as isize,
            b.xy.as_ptr() as *const c_void,
            GL_STREAM_DRAW,
        );
        (gl.EnableVertexAttribArray)(self.ui_pos_loc as GLuint);
        (gl.VertexAttribPointer)(self.ui_pos_loc as GLuint, 2, GL_FLOAT, GL_FALSE, 0, std::ptr::null());

        (gl.BindBuffer)(GL_ARRAY_BUFFER, self.ui_vbo_uv);
        (gl.BufferData)(
            GL_ARRAY_BUFFER,
            std::mem::size_of_val(b.uv) as isize,
            b.uv.as_ptr() as *const c_void,
            GL_STREAM_DRAW,
        );
        (gl.EnableVertexAttribArray)(self.ui_uv_loc as GLuint);
        (gl.VertexAttribPointer)(self.ui_uv_loc as GLuint, 2, GL_FLOAT, GL_FALSE, 0, std::ptr::null());

        (gl.BindBuffer)(GL_ARRAY_BUFFER, self.ui_vbo_color);
        (gl.BufferData)(
            GL_ARRAY_BUFFER,
            std::mem::size_of_val(b.color) as isize,
            b.color.as_ptr() as *const c_void,
            GL_STREAM_DRAW,
        );
        (gl.EnableVertexAttribArray)(self.ui_color_loc as GLuint);
        (gl.VertexAttribPointer)(self.ui_color_loc as GLuint, 4, GL_FLOAT, GL_FALSE, 0, std::ptr::null());

        (gl.BindBuffer)(GL_ELEMENT_ARRAY_BUFFER, self.ui_ibo);
        (gl.BufferData)(
            GL_ELEMENT_ARRAY_BUFFER,
            std::mem::size_of_val(b.indices) as isize,
            b.indices.as_ptr() as *const c_void,
            GL_STREAM_DRAW,
        );
        (gl.DrawElements)(GL_TRIANGLES, b.indices.len() as GLsizei, GL_UNSIGNED_SHORT, std::ptr::null());
    }
}

impl Drop for GlVideo {
    fn drop(&mut self) {
        // SAFETY: tearing down objects we own on the GUI thread; context still current.
        unsafe {
            let gl = &self.gl;
            if let Some((old, _)) = self.current.take() {
                (gl.eglDestroyImageKHR)(self.egl_display, old.image);
            }
            (gl.DeleteTextures)(1, &self.video_tex);
            (gl.DeleteTextures)(1, &self.font_tex);
            let bufs = [self.ui_vbo_pos, self.ui_vbo_uv, self.ui_vbo_color, self.ui_ibo];
            (gl.DeleteBuffers)(4, bufs.as_ptr());
            SDL_GL_DestroyContext(self.ctx);
        }
    }
}

/// Build the `eglCreateImageKHR` attribute list for an NV12 (or single/dual-plane) DMA-BUF
/// import (EGL_EXT_image_dma_buf_import). NULL-terminated (`EGL_NONE`). The modifier
/// attributes are omitted when the driver reported `DRM_FORMAT_MOD_INVALID` (passing INVALID
/// is an error on some drivers). Pure data — unit-tested without a GL context.
pub fn dmabuf_egl_attribs(frame: &GpuFrame) -> Vec<EGLint> {
    let mut a = Vec::with_capacity(32);
    a.push(EGL_WIDTH);
    a.push(frame.coded_w as EGLint);
    a.push(EGL_HEIGHT);
    a.push(frame.coded_h as EGLint);
    a.push(EGL_LINUX_DRM_FOURCC_EXT);
    a.push(frame.drm_format as EGLint);

    let has_mod = frame.drm_modifier != pf_vaapi::ffi::DRM_FORMAT_MOD_INVALID;
    let mod_lo = (frame.drm_modifier & 0xFFFF_FFFF) as EGLint;
    let mod_hi = (frame.drm_modifier >> 32) as EGLint;

    // Per-plane attribute keys, plane 0 and plane 1 (NV12 = 2 planes).
    const PLANE_FD: [EGLint; 2] = [EGL_DMA_BUF_PLANE0_FD_EXT, EGL_DMA_BUF_PLANE1_FD_EXT];
    const PLANE_OFF: [EGLint; 2] = [EGL_DMA_BUF_PLANE0_OFFSET_EXT, EGL_DMA_BUF_PLANE1_OFFSET_EXT];
    const PLANE_PITCH: [EGLint; 2] = [EGL_DMA_BUF_PLANE0_PITCH_EXT, EGL_DMA_BUF_PLANE1_PITCH_EXT];
    const PLANE_MOD_LO: [EGLint; 2] =
        [EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT];
    const PLANE_MOD_HI: [EGLint; 2] =
        [EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT];

    for (i, plane) in frame.planes.iter().take(2).enumerate() {
        // Each plane's fd is the object it lives in (COMPOSED export: all planes may share
        // one object, or split across two). The importer received one fd per object.
        let fd = frame
            .fds
            .get(plane.object_index as usize)
            .copied()
            .unwrap_or(-1);
        a.push(PLANE_FD[i]);
        a.push(fd as EGLint);
        a.push(PLANE_OFF[i]);
        a.push(plane.offset as EGLint);
        a.push(PLANE_PITCH[i]);
        a.push(plane.pitch as EGLint);
        if has_mod {
            a.push(PLANE_MOD_LO[i]);
            a.push(mod_lo);
            a.push(PLANE_MOD_HI[i]);
            a.push(mod_hi);
        }
    }
    a.push(EGL_NONE);
    a
}

/// Whether SDL reports a GL extension present (`SDL_GL_ExtensionSupported`).
///
/// # Safety
/// A GL context must be current.
unsafe fn gl_extension_supported(name: &str) -> bool {
    let c = CString::new(name).unwrap_or_default();
    SDL_GL_ExtensionSupported(c.as_ptr())
}

/// Compile + link a vertex/fragment program; `None` on any GL error, logging the info log.
///
/// # Safety
/// A GL context must be current.
unsafe fn link_program(gl: &Gl, vs: &str, fs: &str) -> Option<GLuint> {
    let v = compile(gl, GL_VERTEX_SHADER, vs)?;
    let f = compile(gl, GL_FRAGMENT_SHADER, fs)?;
    let prog = (gl.CreateProgram)();
    (gl.AttachShader)(prog, v);
    (gl.AttachShader)(prog, f);
    (gl.LinkProgram)(prog);
    let mut ok: GLint = 0;
    (gl.GetProgramiv)(prog, GL_LINK_STATUS, &mut ok);
    (gl.DeleteShader)(v);
    (gl.DeleteShader)(f);
    if ok == 0 {
        let mut log = [0u8; 512];
        let mut len: GLsizei = 0;
        (gl.GetProgramInfoLog)(prog, 512, &mut len, log.as_mut_ptr() as *mut GLchar);
        eprintln!(
            "pfplay-ui: GL program link failed: {}",
            String::from_utf8_lossy(&log[..len.max(0) as usize])
        );
        return None;
    }
    Some(prog)
}

/// Compile one shader; `None` on a compile error (info log printed).
///
/// # Safety
/// A GL context must be current.
unsafe fn compile(gl: &Gl, kind: GLenum, src: &str) -> Option<GLuint> {
    let s = (gl.CreateShader)(kind);
    let ptr = src.as_ptr() as *const GLchar;
    let len = src.len() as GLint;
    (gl.ShaderSource)(s, 1, &ptr, &len);
    (gl.CompileShader)(s);
    let mut ok: GLint = 0;
    (gl.GetShaderiv)(s, GL_COMPILE_STATUS, &mut ok);
    if ok == 0 {
        let mut log = [0u8; 512];
        let mut l: GLsizei = 0;
        (gl.GetShaderInfoLog)(s, 512, &mut l, log.as_mut_ptr() as *mut GLchar);
        eprintln!(
            "pfplay-ui: GL shader compile failed: {}",
            String::from_utf8_lossy(&log[..l.max(0) as usize])
        );
        (gl.DeleteShader)(s);
        return None;
    }
    Some(s)
}

/// Look up an attribute location (returns -1 if optimized out — the caller tolerates it).
///
/// # Safety
/// Context current; `prog` linked.
unsafe fn attrib(gl: &Gl, prog: GLuint, name: &str) -> GLint {
    let c = CString::new(name).unwrap_or_default();
    (gl.GetAttribLocation)(prog, c.as_ptr())
}

/// Look up a uniform location.
///
/// # Safety
/// Context current; `prog` linked.
unsafe fn uniform(gl: &Gl, prog: GLuint, name: &str) -> GLint {
    let c = CString::new(name).unwrap_or_default();
    (gl.GetUniformLocation)(prog, c.as_ptr())
}

/// Upload scope's RGBA8 font atlas as a `GL_TEXTURE_2D`.
///
/// # Safety
/// Context current.
unsafe fn upload_font_atlas(gl: &Gl, font: &Font) -> GLuint {
    let (w, h) = font.atlas_size();
    let pixels = font.atlas_rgba();
    let mut tex: GLuint = 0;
    (gl.GenTextures)(1, &mut tex);
    (gl.BindTexture)(GL_TEXTURE_2D, tex);
    (gl.PixelStorei)(GL_UNPACK_ALIGNMENT, 1);
    (gl.TexImage2D)(
        GL_TEXTURE_2D,
        0,
        GL_RGBA8 as GLint,
        w as GLsizei,
        h as GLsizei,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        pixels.as_ptr() as *const c_void,
    );
    (gl.TexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    (gl.TexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    (gl.TexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    (gl.TexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    tex
}

/// Log the reason the zero-copy GL path was declined and that the SDL_Renderer path is used.
fn log_fallback(why: &str) {
    eprintln!("pfplay-ui: zero-copy GLES backend unavailable ({why}); video path = SDL texture upload");
}

#[cfg(test)]
mod tests {
    use super::*;
    use pf_vaapi::gpuframe::GpuFrame;
    use pf_vaapi::ExportedPlane;

    fn nv12_frame(modifier: u64) -> GpuFrame {
        GpuFrame {
            token: 1,
            fds: vec![7], // one object, both planes index it
            drm_format: pf_vaapi::ffi::DRM_FORMAT_NV12,
            drm_modifier: modifier,
            coded_w: 1920,
            coded_h: 1088,
            disp_w: 1920,
            disp_h: 1080,
            crop_x: 0,
            crop_y: 0,
            planes: vec![
                ExportedPlane { object_index: 0, offset: 0, pitch: 1920 },
                ExportedPlane { object_index: 0, offset: 1920 * 1088, pitch: 1920 },
            ],
        }
    }

    #[test]
    fn egl_attribs_carry_two_planes_and_modifier() {
        let f = nv12_frame(0x0100_0000_0000_0001); // a non-INVALID modifier
        let a = dmabuf_egl_attribs(&f);
        // Ends in EGL_NONE.
        assert_eq!(*a.last().unwrap(), EGL_NONE);
        // Width/height/fourcc present.
        assert!(a.windows(2).any(|w| w == [EGL_WIDTH, 1920]));
        assert!(a.windows(2).any(|w| w == [EGL_HEIGHT, 1088]));
        assert!(a.windows(2).any(|w| w == [EGL_LINUX_DRM_FOURCC_EXT, pf_vaapi::ffi::DRM_FORMAT_NV12 as EGLint]));
        // Both plane fd keys present, pointing at fd 7 (object 0).
        assert!(a.windows(2).any(|w| w == [EGL_DMA_BUF_PLANE0_FD_EXT, 7]));
        assert!(a.windows(2).any(|w| w == [EGL_DMA_BUF_PLANE1_FD_EXT, 7]));
        // Plane 1 offset is the Y-plane size.
        assert!(a.windows(2).any(|w| w == [EGL_DMA_BUF_PLANE1_OFFSET_EXT, (1920 * 1088) as EGLint]));
        // Modifier attributes present (non-INVALID).
        assert!(a.iter().any(|&k| k == EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT));
        assert!(a.iter().any(|&k| k == EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT));
        // The fds must not be closed by attrib-building (GpuFrame drop closes fd 7 — but 7 is
        // a fake fd in this test; drop's close(7) is harmless / ignored).
        std::mem::forget(f); // don't close the fake fd 7 (it isn't ours)
    }

    #[test]
    fn egl_attribs_omit_invalid_modifier() {
        let f = nv12_frame(pf_vaapi::ffi::DRM_FORMAT_MOD_INVALID);
        let a = dmabuf_egl_attribs(&f);
        assert!(!a.iter().any(|&k| k == EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT));
        assert!(!a.iter().any(|&k| k == EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT));
        std::mem::forget(f);
    }
}
