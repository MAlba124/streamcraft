//! GLES 3 + EGL function-pointer loading and the constants the zero-copy path needs.
//!
//! sdl3-sys exposes the *SDL* GL/EGL entry points (`SDL_GL_CreateContext`,
//! `SDL_GL_GetProcAddress`, `SDL_EGL_GetProcAddress`, …) but **not** the GL or EGL API
//! itself — those are the driver's, loaded at runtime by name. So this module declares the
//! handful of GLES 3 core calls (texture/shader/draw), the two `OES_EGL_image_external`
//! extension entry points, and the three `EGL_KHR_image_base` / `EGL_EXT_image_dma_buf_import`
//! entry points as fn-pointer types, and loads them through SDL's loaders (`SDL_GL_*` for GL,
//! `SDL_EGL_*` for EGL). Every name + constant is cited at its declaration.
//!
//! Extensions used (verified at runtime in [`super::GlVideo::create`]):
//! * `OES_EGL_image_external` — sampling a DMA-BUF-backed image as `GL_TEXTURE_EXTERNAL_OES`
//!   with `samplerExternalOES` (the driver applies YCbCr→RGB); Khronos OES ext #87.
//! * `EGL_KHR_image_base` — `eglCreateImageKHR` / `eglDestroyImageKHR`.
//! * `EGL_EXT_image_dma_buf_import` — the `EGL_LINUX_DMA_BUF_EXT` target + the
//!   `EGL_DMA_BUF_PLANE*_{FD,OFFSET,PITCH,MODIFIER_*}_EXT` attributes.

// A hand-rolled GL/EGL binding surface (like `pf_vaapi::ffi`): the full set of constants +
// the two-plane-NV12 fallback's R8/RG8 formats + a couple of reserved entry points are kept
// documented even when the active COMPOSED-NV12 path doesn't touch every one, so `dead_code`
// is allowed here exactly as it is on the libva FFI.
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code)]

use std::ffi::{c_char, c_int, c_uint, c_void};

use sdl3_sys::everything::{SDL_EGL_GetProcAddress, SDL_GL_GetProcAddress};

// --- GL / EGL scalar typedefs (khrplatform.h / gl2.h / egl.h) --------------------------
pub type GLenum = c_uint;
pub type GLuint = c_uint;
pub type GLint = c_int;
pub type GLsizei = c_int;
pub type GLbitfield = c_uint;
pub type GLboolean = u8;
pub type GLchar = c_char;
pub type GLfloat = f32;
/// `EGLDisplay` / `EGLContext` / `EGLImageKHR` are opaque `void*` (egl.h).
pub type EGLDisplay = *mut c_void;
pub type EGLContext = *mut c_void;
pub type EGLClientBuffer = *mut c_void;
pub type EGLImageKHR = *mut c_void;
/// `EGLint` is a 32-bit signed int; the DMA-BUF import attrib list is a NULL-terminated
/// (`EGL_NONE`) array of these key/value pairs (egl.h / eglext.h).
pub type EGLint = i32;
pub type EGLenum = c_uint;
pub type EGLBoolean = c_uint;

// --- GL constants (gl2.h / gl2ext.h) ---------------------------------------------------
pub const GL_TEXTURE_2D: GLenum = 0x0DE1;
/// `GL_TEXTURE_EXTERNAL_OES` (OES_EGL_image_external) — the target a DMA-BUF EGLImage binds
/// to; sampled with `samplerExternalOES` (the driver's YCbCr conversion runs here).
pub const GL_TEXTURE_EXTERNAL_OES: GLenum = 0x8D65;
pub const GL_TEXTURE0: GLenum = 0x84C0;
pub const GL_TEXTURE1: GLenum = 0x84C1;
pub const GL_TEXTURE_MIN_FILTER: GLenum = 0x2801;
pub const GL_TEXTURE_MAG_FILTER: GLenum = 0x2800;
pub const GL_TEXTURE_WRAP_S: GLenum = 0x2802;
pub const GL_TEXTURE_WRAP_T: GLenum = 0x2803;
pub const GL_LINEAR: GLint = 0x2601;
pub const GL_CLAMP_TO_EDGE: GLint = 0x812F;
pub const GL_RED: GLenum = 0x1903;
pub const GL_RG: GLenum = 0x8227;
pub const GL_R8: GLenum = 0x8229;
pub const GL_RG8: GLenum = 0x822B;
pub const GL_RGBA: GLenum = 0x1908;
pub const GL_RGBA8: GLenum = 0x8058;
pub const GL_UNSIGNED_BYTE: GLenum = 0x1401;
pub const GL_FLOAT: GLenum = 0x1406;
pub const GL_FALSE: GLboolean = 0;
pub const GL_TRUE: GLboolean = 1;
pub const GL_COLOR_BUFFER_BIT: GLbitfield = 0x0000_4000;
pub const GL_TRIANGLES: GLenum = 0x0004;
pub const GL_TRIANGLE_STRIP: GLenum = 0x0005;
pub const GL_UNSIGNED_SHORT: GLenum = 0x1403;
pub const GL_ARRAY_BUFFER: GLenum = 0x8892;
pub const GL_ELEMENT_ARRAY_BUFFER: GLenum = 0x8893;
pub const GL_STREAM_DRAW: GLenum = 0x88E0;
pub const GL_VERTEX_SHADER: GLenum = 0x8B31;
pub const GL_FRAGMENT_SHADER: GLenum = 0x8B30;
pub const GL_COMPILE_STATUS: GLenum = 0x8B81;
pub const GL_LINK_STATUS: GLenum = 0x8B82;
pub const GL_BLEND: GLenum = 0x0BE2;
pub const GL_SRC_ALPHA: GLenum = 0x0302;
pub const GL_ONE_MINUS_SRC_ALPHA: GLenum = 0x0303;
pub const GL_SCISSOR_TEST: GLenum = 0x0C11;
pub const GL_UNPACK_ALIGNMENT: GLenum = 0x0CF5;

// --- EGL constants (egl.h / eglext.h) --------------------------------------------------
pub const EGL_NONE: EGLint = 0x3038;
pub const EGL_NO_CONTEXT: EGLContext = std::ptr::null_mut();
pub const EGL_NO_IMAGE_KHR: EGLImageKHR = std::ptr::null_mut();
/// `EGL_LINUX_DMA_BUF_EXT` (EGL_EXT_image_dma_buf_import) — the `target` passed to
/// `eglCreateImageKHR` to wrap externally-allocated DMA-BUF planes.
pub const EGL_LINUX_DMA_BUF_EXT: EGLenum = 0x3270;
pub const EGL_WIDTH: EGLint = 0x3057;
pub const EGL_HEIGHT: EGLint = 0x3056;
pub const EGL_LINUX_DRM_FOURCC_EXT: EGLint = 0x3271;
pub const EGL_DMA_BUF_PLANE0_FD_EXT: EGLint = 0x3272;
pub const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EGLint = 0x3273;
pub const EGL_DMA_BUF_PLANE0_PITCH_EXT: EGLint = 0x3274;
pub const EGL_DMA_BUF_PLANE1_FD_EXT: EGLint = 0x3275;
pub const EGL_DMA_BUF_PLANE1_OFFSET_EXT: EGLint = 0x3276;
pub const EGL_DMA_BUF_PLANE1_PITCH_EXT: EGLint = 0x3277;
pub const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EGLint = 0x3443;
pub const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EGLint = 0x3444;
pub const EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT: EGLint = 0x3445;
pub const EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT: EGLint = 0x3446;
/// `EGL_IMAGE_PRESERVED_KHR` (EGL_KHR_image_base) — keep the buffer contents on import.
pub const EGL_IMAGE_PRESERVED_KHR: EGLint = 0x30D2;

// --- GL / EGL fn-pointer types ---------------------------------------------------------
// GLES 3 core (loaded via SDL_GL_GetProcAddress).
type PFNglGenTextures = unsafe extern "C" fn(GLsizei, *mut GLuint);
type PFNglDeleteTextures = unsafe extern "C" fn(GLsizei, *const GLuint);
type PFNglBindTexture = unsafe extern "C" fn(GLenum, GLuint);
type PFNglTexParameteri = unsafe extern "C" fn(GLenum, GLenum, GLint);
type PFNglActiveTexture = unsafe extern "C" fn(GLenum);
type PFNglTexImage2D = unsafe extern "C" fn(
    GLenum,
    GLint,
    GLint,
    GLsizei,
    GLsizei,
    GLint,
    GLenum,
    GLenum,
    *const c_void,
);
type PFNglViewport = unsafe extern "C" fn(GLint, GLint, GLsizei, GLsizei);
type PFNglClearColor = unsafe extern "C" fn(GLfloat, GLfloat, GLfloat, GLfloat);
type PFNglClear = unsafe extern "C" fn(GLbitfield);
type PFNglEnable = unsafe extern "C" fn(GLenum);
type PFNglDisable = unsafe extern "C" fn(GLenum);
type PFNglBlendFunc = unsafe extern "C" fn(GLenum, GLenum);
type PFNglScissor = unsafe extern "C" fn(GLint, GLint, GLsizei, GLsizei);
type PFNglPixelStorei = unsafe extern "C" fn(GLenum, GLint);
type PFNglDrawArrays = unsafe extern "C" fn(GLenum, GLint, GLsizei);
type PFNglDrawElements = unsafe extern "C" fn(GLenum, GLsizei, GLenum, *const c_void);
// Shader / program / buffer objects (GLES 2+).
type PFNglCreateShader = unsafe extern "C" fn(GLenum) -> GLuint;
type PFNglShaderSource =
    unsafe extern "C" fn(GLuint, GLsizei, *const *const GLchar, *const GLint);
type PFNglCompileShader = unsafe extern "C" fn(GLuint);
type PFNglGetShaderiv = unsafe extern "C" fn(GLuint, GLenum, *mut GLint);
type PFNglGetShaderInfoLog =
    unsafe extern "C" fn(GLuint, GLsizei, *mut GLsizei, *mut GLchar);
type PFNglDeleteShader = unsafe extern "C" fn(GLuint);
type PFNglCreateProgram = unsafe extern "C" fn() -> GLuint;
type PFNglAttachShader = unsafe extern "C" fn(GLuint, GLuint);
type PFNglLinkProgram = unsafe extern "C" fn(GLuint);
type PFNglGetProgramiv = unsafe extern "C" fn(GLuint, GLenum, *mut GLint);
type PFNglGetProgramInfoLog =
    unsafe extern "C" fn(GLuint, GLsizei, *mut GLsizei, *mut GLchar);
type PFNglUseProgram = unsafe extern "C" fn(GLuint);
type PFNglGetUniformLocation = unsafe extern "C" fn(GLuint, *const GLchar) -> GLint;
type PFNglGetAttribLocation = unsafe extern "C" fn(GLuint, *const GLchar) -> GLint;
type PFNglUniform1i = unsafe extern "C" fn(GLint, GLint);
type PFNglUniform2f = unsafe extern "C" fn(GLint, GLfloat, GLfloat);
type PFNglUniform4f = unsafe extern "C" fn(GLint, GLfloat, GLfloat, GLfloat, GLfloat);
type PFNglGenBuffers = unsafe extern "C" fn(GLsizei, *mut GLuint);
type PFNglDeleteBuffers = unsafe extern "C" fn(GLsizei, *const GLuint);
type PFNglBindBuffer = unsafe extern "C" fn(GLenum, GLuint);
type PFNglBufferData = unsafe extern "C" fn(GLenum, isize, *const c_void, GLenum);
type PFNglEnableVertexAttribArray = unsafe extern "C" fn(GLuint);
type PFNglDisableVertexAttribArray = unsafe extern "C" fn(GLuint);
type PFNglVertexAttribPointer =
    unsafe extern "C" fn(GLuint, GLint, GLenum, GLboolean, GLsizei, *const c_void);
type PFNglGetError = unsafe extern "C" fn() -> GLenum;
// OES_EGL_image_external (loaded via SDL_GL_GetProcAddress).
type PFNglEGLImageTargetTexture2DOES = unsafe extern "C" fn(GLenum, *mut c_void);
// EGL_KHR_image_base + EGL_EXT_image_dma_buf_import (loaded via SDL_EGL_GetProcAddress).
type PFNeglCreateImageKHR = unsafe extern "C" fn(
    EGLDisplay,
    EGLContext,
    EGLenum,
    EGLClientBuffer,
    *const EGLint,
) -> EGLImageKHR;
type PFNeglDestroyImageKHR = unsafe extern "C" fn(EGLDisplay, EGLImageKHR) -> EGLBoolean;
type PFNeglGetError = unsafe extern "C" fn() -> EGLint;

/// The resolved GL + EGL entry points. `create` returns `None` if any required symbol is
/// missing (an old driver / a non-EGL GL context) — the caller then uses the SDL_Renderer
/// fallback.
pub struct Gl {
    pub GenTextures: PFNglGenTextures,
    pub DeleteTextures: PFNglDeleteTextures,
    pub BindTexture: PFNglBindTexture,
    pub TexParameteri: PFNglTexParameteri,
    pub ActiveTexture: PFNglActiveTexture,
    pub TexImage2D: PFNglTexImage2D,
    pub Viewport: PFNglViewport,
    pub ClearColor: PFNglClearColor,
    pub Clear: PFNglClear,
    pub Enable: PFNglEnable,
    pub Disable: PFNglDisable,
    pub BlendFunc: PFNglBlendFunc,
    pub Scissor: PFNglScissor,
    pub PixelStorei: PFNglPixelStorei,
    pub DrawArrays: PFNglDrawArrays,
    pub DrawElements: PFNglDrawElements,
    pub CreateShader: PFNglCreateShader,
    pub ShaderSource: PFNglShaderSource,
    pub CompileShader: PFNglCompileShader,
    pub GetShaderiv: PFNglGetShaderiv,
    pub GetShaderInfoLog: PFNglGetShaderInfoLog,
    pub DeleteShader: PFNglDeleteShader,
    pub CreateProgram: PFNglCreateProgram,
    pub AttachShader: PFNglAttachShader,
    pub LinkProgram: PFNglLinkProgram,
    pub GetProgramiv: PFNglGetProgramiv,
    pub GetProgramInfoLog: PFNglGetProgramInfoLog,
    pub UseProgram: PFNglUseProgram,
    pub GetUniformLocation: PFNglGetUniformLocation,
    pub GetAttribLocation: PFNglGetAttribLocation,
    pub Uniform1i: PFNglUniform1i,
    pub Uniform2f: PFNglUniform2f,
    pub Uniform4f: PFNglUniform4f,
    pub GenBuffers: PFNglGenBuffers,
    pub DeleteBuffers: PFNglDeleteBuffers,
    pub BindBuffer: PFNglBindBuffer,
    pub BufferData: PFNglBufferData,
    pub EnableVertexAttribArray: PFNglEnableVertexAttribArray,
    pub DisableVertexAttribArray: PFNglDisableVertexAttribArray,
    pub VertexAttribPointer: PFNglVertexAttribPointer,
    pub GetError: PFNglGetError,
    pub EGLImageTargetTexture2DOES: PFNglEGLImageTargetTexture2DOES,
    pub eglCreateImageKHR: PFNeglCreateImageKHR,
    pub eglDestroyImageKHR: PFNeglDestroyImageKHR,
    pub eglGetError: PFNeglGetError,
}

/// Load one GL function by name via `SDL_GL_GetProcAddress`; `None` if the driver has no
/// such symbol.
///
/// # Safety
/// The caller must transmute the returned pointer to the correct fn-pointer type (the
/// declared `PFN…` at the field), and a current GL context must exist.
unsafe fn gl_sym(name: &[u8]) -> Option<*const c_void> {
    // name must be NUL-terminated (we pass byte-string literals ending in \0).
    let p = SDL_GL_GetProcAddress(name.as_ptr() as *const c_char);
    p.map(|f| f as *const c_void)
}

/// Load one EGL function by name via `SDL_EGL_GetProcAddress`.
///
/// # Safety
/// As [`gl_sym`], for EGL entry points; an EGL-backed context/display must exist.
unsafe fn egl_sym(name: &[u8]) -> Option<*const c_void> {
    let p = SDL_EGL_GetProcAddress(name.as_ptr() as *const c_char);
    p.map(|f| f as *const c_void)
}

impl Gl {
    /// Resolve every required GL + EGL symbol. Returns `None` (→ SDL_Renderer fallback) if
    /// any is missing. A current GLES context must already be made current on the thread.
    ///
    /// # Safety
    /// A GLES context created by SDL must be current on the calling thread.
    pub unsafe fn load() -> Option<Gl> {
        // Each `?` short-circuits to None (missing symbol → no zero-copy path). The
        // transmutes are sound because each name maps to the C signature its PFN type
        // declares (verbatim from gl2.h / gl2ext.h / eglext.h).
        macro_rules! gl {
            ($name:literal) => {
                match gl_sym(concat!($name, "\0").as_bytes()) {
                    Some(p) => std::mem::transmute(p),
                    None => {
                        eprintln!("pfplay-ui: gl: SDL_GL_GetProcAddress returned NULL for {}", $name);
                        return None;
                    }
                }
            };
        }
        macro_rules! egl {
            ($name:literal) => {
                match egl_sym(concat!($name, "\0").as_bytes()) {
                    Some(p) => std::mem::transmute(p),
                    None => {
                        eprintln!("pfplay-ui: gl: SDL_EGL_GetProcAddress returned NULL for {} (SDL not on the EGL path? force EGL)", $name);
                        return None;
                    }
                }
            };
        }
        Some(Gl {
            GenTextures: gl!("glGenTextures"),
            DeleteTextures: gl!("glDeleteTextures"),
            BindTexture: gl!("glBindTexture"),
            TexParameteri: gl!("glTexParameteri"),
            ActiveTexture: gl!("glActiveTexture"),
            TexImage2D: gl!("glTexImage2D"),
            Viewport: gl!("glViewport"),
            ClearColor: gl!("glClearColor"),
            Clear: gl!("glClear"),
            Enable: gl!("glEnable"),
            Disable: gl!("glDisable"),
            BlendFunc: gl!("glBlendFunc"),
            Scissor: gl!("glScissor"),
            PixelStorei: gl!("glPixelStorei"),
            DrawArrays: gl!("glDrawArrays"),
            DrawElements: gl!("glDrawElements"),
            CreateShader: gl!("glCreateShader"),
            ShaderSource: gl!("glShaderSource"),
            CompileShader: gl!("glCompileShader"),
            GetShaderiv: gl!("glGetShaderiv"),
            GetShaderInfoLog: gl!("glGetShaderInfoLog"),
            DeleteShader: gl!("glDeleteShader"),
            CreateProgram: gl!("glCreateProgram"),
            AttachShader: gl!("glAttachShader"),
            LinkProgram: gl!("glLinkProgram"),
            GetProgramiv: gl!("glGetProgramiv"),
            GetProgramInfoLog: gl!("glGetProgramInfoLog"),
            UseProgram: gl!("glUseProgram"),
            GetUniformLocation: gl!("glGetUniformLocation"),
            GetAttribLocation: gl!("glGetAttribLocation"),
            Uniform1i: gl!("glUniform1i"),
            Uniform2f: gl!("glUniform2f"),
            Uniform4f: gl!("glUniform4f"),
            GenBuffers: gl!("glGenBuffers"),
            DeleteBuffers: gl!("glDeleteBuffers"),
            BindBuffer: gl!("glBindBuffer"),
            BufferData: gl!("glBufferData"),
            EnableVertexAttribArray: gl!("glEnableVertexAttribArray"),
            DisableVertexAttribArray: gl!("glDisableVertexAttribArray"),
            VertexAttribPointer: gl!("glVertexAttribPointer"),
            GetError: gl!("glGetError"),
            // OES_EGL_image_external — a GL entry point, but only present when the ext is.
            EGLImageTargetTexture2DOES: gl!("glEGLImageTargetTexture2DOES"),
            // EGL entry points — loaded through the EGL loader.
            eglCreateImageKHR: egl!("eglCreateImageKHR"),
            eglDestroyImageKHR: egl!("eglDestroyImageKHR"),
            eglGetError: egl!("eglGetError"),
        })
    }
}
