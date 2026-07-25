//! [`GpuRenderer`] — the SDL3 GPU device / pipeline / texture plumbing for the owned
//! video render pipeline. All `unsafe` for the GPU path lives here (mirroring the
//! classic path's `video.rs`), with a SAFETY note per block (workspace lint policy).
//!
//! Per frame the renderer:
//!   1. acquires a command buffer and (via `SDL_WaitAndAcquireGPUSwapchainTexture`)
//!      the window's swapchain texture;
//!   2. uploads the YCbCr planes through one cycled transfer buffer into R8/R8G8
//!      plane textures inside a copy pass;
//!   3. runs one render pass drawing a fullscreen triangle (no vertex buffers — the
//!      vertex shader synthesises positions from `gl_VertexIndex`), the fragment
//!      shader applying the colorimetry-aware pipeline from `color.rs`;
//!   4. letterboxes by computing the fitted rect itself (`SDL_SetGPUViewport`) — the
//!      renderer owns presentation geometry now.
//!
//! Plane layouts:
//!
//! - i420 — 3× R8_UNORM: Y (w×h), U (cw×ch), V (cw×ch), `cw=ceil(w/2)`.
//! - nv12 — R8_UNORM Y + one R8G8_UNORM interleaved CbCr (cw×ch); the shader reads
//!   Cb from `.r` and Cr from `.g` of that one texture (chroma_mode = Interleaved).
//! - gray8 — Y only; chroma handled in-shader via the flat-chroma mode flag.
//!
//! Chroma siting: linear samplers give bilinear chroma upsampling with siting
//! treated as CENTER; MPEG-2-style left-siting is a noted follow-up (H.273
//! ChromaSampleLocType is not yet plumbed through the format).

#![allow(unsafe_code)]

use std::ffi::{c_void, CStr};

use sdl3_sys::everything::*;

use streamcraft_core::error::Error;

use super::color::{ChromaMode, Colorimetry, FragUniforms};
use super::{VIDEO_FRAG_SPV, VIDEO_VERT_SPV};

/// The plane pixel formats a frame uses (mirrors the sink's `Pix`, but named for the
/// GPU-texture arrangement rather than the container format).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlaneFormat {
    /// Three R8 planes: Y, U, V.
    I420,
    /// R8 Y + R8G8 interleaved CbCr.
    Nv12,
    /// R8 Y only (flat chroma in-shader).
    Gray8,
}

fn sdl_error(what: &str) -> Error {
    // SAFETY: SDL_GetError returns a thread-local NUL-terminated string valid until
    // the next SDL call on this thread; we copy it out immediately.
    let msg = unsafe { CStr::from_ptr(SDL_GetError()) }.to_string_lossy().into_owned();
    Error::Resource(format!("sdl3 gpu: {what}: {msg}"))
}

/// The plane textures for the current geometry — recreated on a dimension / pixfmt
/// change.
struct PlaneTextures {
    width: u32,
    height: u32,
    fmt: PlaneFormat,
    /// Y plane (R8_UNORM, w×h).
    y: *mut SDL_GPUTexture,
    /// Chroma plane 0: U (i420, R8, cw×ch) or interleaved CbCr (nv12, R8G8, cw×ch).
    /// Null for gray8.
    c0: *mut SDL_GPUTexture,
    /// Chroma plane 1: V (i420, R8). Null for nv12/gray8 (nv12 reuses c0).
    c1: *mut SDL_GPUTexture,
}

/// The owned GPU renderer. Holds the device (which owns the window's swapchain), the
/// shaders/pipeline/sampler, a cycled upload transfer buffer, and the current plane
/// textures. Construct with [`GpuRenderer::new`]; call [`present`] per frame.
pub struct GpuRenderer {
    device: *mut SDL_GPUDevice,
    /// The window the device presents into (owned by the caller — the sink's
    /// `VideoWindow`; we only claimed it and must release on drop).
    window: *mut SDL_Window,
    pipeline: *mut SDL_GPUGraphicsPipeline,
    sampler: *mut SDL_GPUSampler,
    swapchain_fmt: SDL_GPUTextureFormat,
    /// One reusable upload transfer buffer, grown as planes grow, cycled per frame.
    transfer: *mut SDL_GPUTransferBuffer,
    transfer_cap: u32,
    planes: Option<PlaneTextures>,
}

// SAFETY: the raw SDL/GPU pointers are used only by the owning element on its single
// streaming thread (the pipeline may move the element whole between runs). SDL_GPU
// objects are not thread-bound-at-creation on the Vulkan backend this crate targets;
// `GpuRenderer` is Send (moved whole), never Sync (never shared).
unsafe impl Send for GpuRenderer {}

impl GpuRenderer {
    /// Create the GPU device (SPIR-V, no debug), claim the window, build the pipeline
    /// and sampler. Any failure returns `Err` so the sink can fall back to the classic
    /// `SDL_Renderer` path — a device-less box must still finish the pipeline.
    ///
    /// # Safety
    /// `window` must be a live `SDL_Window` created on this thread with the video
    /// subsystem initialised (as by `VideoWindow::open_windowed`), and must outlive
    /// the returned renderer (the renderer unclaims it on drop). SDL picks the backend
    /// (Vulkan on this Intel box; see the report).
    pub unsafe fn new(window: *mut SDL_Window) -> Result<Self, Error> {
        // SAFETY: plain C calls on the caller-provided live window; every returned
        // pointer is null-checked before use.
        unsafe {
            let device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, false, std::ptr::null());
            if device.is_null() {
                return Err(sdl_error("create device"));
            }
            if !SDL_ClaimWindowForGPUDevice(device, window) {
                let e = sdl_error("claim window");
                SDL_DestroyGPUDevice(device);
                return Err(e);
            }
            let swapchain_fmt = SDL_GetGPUSwapchainTextureFormat(device, window);

            let mut this = GpuRenderer {
                device,
                window,
                pipeline: std::ptr::null_mut(),
                sampler: std::ptr::null_mut(),
                swapchain_fmt,
                transfer: std::ptr::null_mut(),
                transfer_cap: 0,
                planes: None,
            };
            // `this`'s Drop releases the window + device on any pipeline failure.
            this.build_pipeline()?;
            Ok(this)
        }
    }

    /// Create a windowless GPU renderer for offscreen rendering (the golden test).
    /// `SDL_CreateGPUDevice` needs no window, but the video subsystem must be
    /// initialised first (an SDL requirement), so the caller inits SDL_INIT_VIDEO.
    /// Returns `Err` on any device / pipeline failure so the test can skip cleanly.
    #[cfg(test)]
    pub fn new_headless() -> Result<Self, Error> {
        // SAFETY: init the video subsystem (refcounted), then create the device.
        unsafe {
            if !SDL_InitSubSystem(SDL_INIT_VIDEO) {
                return Err(sdl_error("init video (headless)"));
            }
            let device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, false, std::ptr::null());
            if device.is_null() {
                SDL_QuitSubSystem(SDL_INIT_VIDEO);
                return Err(sdl_error("create device (headless)"));
            }
            // No window claim: offscreen only. Pick a plausible swapchain-ish format
            // for the pipeline color target — the offscreen path renders into an
            // R8G8B8A8_UNORM texture, so build the pipeline for that format.
            let mut this = GpuRenderer {
                device,
                window: std::ptr::null_mut(),
                pipeline: std::ptr::null_mut(),
                sampler: std::ptr::null_mut(),
                swapchain_fmt: SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
                transfer: std::ptr::null_mut(),
                transfer_cap: 0,
                planes: None,
            };
            this.build_pipeline()?;
            Ok(this)
        }
    }

    /// The GPU backend driver name as a `&'static str`, for the configure log line.
    /// SDL_GPU's driver set is fixed and small; unknown names collapse to "other" so
    /// the logger's `&'static str` field constraint is satisfied without leaking.
    pub fn driver(&self) -> &'static str {
        // SAFETY: device is live; SDL returns a static string.
        let name = unsafe {
            let p = SDL_GetGPUDeviceDriver(self.device);
            if p.is_null() {
                return "unknown";
            }
            CStr::from_ptr(p).to_string_lossy().into_owned()
        };
        match name.as_str() {
            "vulkan" => "vulkan",
            "direct3d12" => "direct3d12",
            "metal" => "metal",
            _ => "other",
        }
    }

    /// Build the shader modules, the graphics pipeline (fullscreen triangle, no vertex
    /// buffers, one color target in the swapchain format), and the linear sampler.
    unsafe fn build_pipeline(&mut self) -> Result<(), Error> {
        let vert = self.create_shader(VIDEO_VERT_SPV, SDL_GPU_SHADERSTAGE_VERTEX, 0, 0)?;
        // Fragment: 3 sampled textures (Y, U, V), 1 uniform buffer (the color block).
        let frag = self.create_shader(VIDEO_FRAG_SPV, SDL_GPU_SHADERSTAGE_FRAGMENT, 3, 1);
        let frag = match frag {
            Ok(f) => f,
            Err(e) => {
                SDL_ReleaseGPUShader(self.device, vert);
                return Err(e);
            }
        };

        // One color target in the swapchain format, no blending (opaque video).
        let color_target = SDL_GPUColorTargetDescription {
            format: self.swapchain_fmt,
            ..Default::default()
        };
        let pipeline_info = SDL_GPUGraphicsPipelineCreateInfo {
            vertex_shader: vert,
            fragment_shader: frag,
            primitive_type: SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
            target_info: SDL_GPUGraphicsPipelineTargetInfo {
                color_target_descriptions: &color_target,
                num_color_targets: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let pipeline = SDL_CreateGPUGraphicsPipeline(self.device, &pipeline_info);
        // Shaders can be released once the pipeline is built (it retains what it needs).
        SDL_ReleaseGPUShader(self.device, vert);
        SDL_ReleaseGPUShader(self.device, frag);
        if pipeline.is_null() {
            return Err(sdl_error("create pipeline"));
        }
        self.pipeline = pipeline;

        // Linear sampler: bilinear chroma upsampling, clamp to edge (no wrap bleed).
        let sampler_info = SDL_GPUSamplerCreateInfo {
            min_filter: SDL_GPU_FILTER_LINEAR,
            mag_filter: SDL_GPU_FILTER_LINEAR,
            mipmap_mode: SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
            address_mode_u: SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_v: SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_w: SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            ..Default::default()
        };
        let sampler = SDL_CreateGPUSampler(self.device, &sampler_info);
        if sampler.is_null() {
            return Err(sdl_error("create sampler"));
        }
        self.sampler = sampler;
        Ok(())
    }

    unsafe fn create_shader(
        &self,
        code: &[u8],
        stage: SDL_GPUShaderStage,
        num_samplers: u32,
        num_uniform_buffers: u32,
    ) -> Result<*mut SDL_GPUShader, Error> {
        let info = SDL_GPUShaderCreateInfo {
            code_size: code.len(),
            code: code.as_ptr(),
            entrypoint: c"main".as_ptr(),
            format: SDL_GPU_SHADERFORMAT_SPIRV,
            stage,
            num_samplers,
            num_uniform_buffers,
            ..Default::default()
        };
        let sh = SDL_CreateGPUShader(self.device, &info);
        if sh.is_null() {
            return Err(sdl_error("create shader"));
        }
        Ok(sh)
    }

    /// (Re)create the plane textures for a new geometry / pixel format.
    unsafe fn ensure_planes(
        &mut self,
        width: u32,
        height: u32,
        fmt: PlaneFormat,
    ) -> Result<(), Error> {
        if let Some(p) = &self.planes {
            if p.width == width && p.height == height && p.fmt == fmt {
                return Ok(());
            }
        }
        self.release_planes();
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);

        let mk = |device: *mut SDL_GPUDevice, w: u32, h: u32, f: SDL_GPUTextureFormat| {
            let info = SDL_GPUTextureCreateInfo {
                r#type: SDL_GPU_TEXTURETYPE_2D,
                format: f,
                usage: SDL_GPU_TEXTUREUSAGE_SAMPLER,
                width: w,
                height: h,
                layer_count_or_depth: 1,
                num_levels: 1,
                sample_count: SDL_GPU_SAMPLECOUNT_1,
                ..Default::default()
            };
            SDL_CreateGPUTexture(device, &info)
        };

        let y = mk(self.device, width, height, SDL_GPU_TEXTUREFORMAT_R8_UNORM);
        if y.is_null() {
            return Err(sdl_error("create Y texture"));
        }
        let (c0, c1) = match fmt {
            PlaneFormat::I420 => {
                let u = mk(self.device, cw, ch, SDL_GPU_TEXTUREFORMAT_R8_UNORM);
                let v = mk(self.device, cw, ch, SDL_GPU_TEXTUREFORMAT_R8_UNORM);
                if u.is_null() || v.is_null() {
                    SDL_ReleaseGPUTexture(self.device, y);
                    if !u.is_null() {
                        SDL_ReleaseGPUTexture(self.device, u);
                    }
                    if !v.is_null() {
                        SDL_ReleaseGPUTexture(self.device, v);
                    }
                    return Err(sdl_error("create chroma textures"));
                }
                (u, v)
            }
            PlaneFormat::Nv12 => {
                let uv = mk(self.device, cw, ch, SDL_GPU_TEXTUREFORMAT_R8G8_UNORM);
                if uv.is_null() {
                    SDL_ReleaseGPUTexture(self.device, y);
                    return Err(sdl_error("create CbCr texture"));
                }
                (uv, std::ptr::null_mut())
            }
            PlaneFormat::Gray8 => (std::ptr::null_mut(), std::ptr::null_mut()),
        };
        self.planes = Some(PlaneTextures { width, height, fmt, y, c0, c1 });
        Ok(())
    }

    fn release_planes(&mut self) {
        if let Some(p) = self.planes.take() {
            // SAFETY: textures we own; releasing on the owning device.
            unsafe {
                if !p.y.is_null() {
                    SDL_ReleaseGPUTexture(self.device, p.y);
                }
                if !p.c0.is_null() {
                    SDL_ReleaseGPUTexture(self.device, p.c0);
                }
                if !p.c1.is_null() {
                    SDL_ReleaseGPUTexture(self.device, p.c1);
                }
            }
        }
    }

    /// Ensure the upload transfer buffer holds at least `size` bytes (grown, cycled).
    unsafe fn ensure_transfer(&mut self, size: u32) -> Result<(), Error> {
        if !self.transfer.is_null() && self.transfer_cap >= size {
            return Ok(());
        }
        if !self.transfer.is_null() {
            SDL_ReleaseGPUTransferBuffer(self.device, self.transfer);
            self.transfer = std::ptr::null_mut();
        }
        let info = SDL_GPUTransferBufferCreateInfo {
            usage: SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            size,
            ..Default::default()
        };
        let tb = SDL_CreateGPUTransferBuffer(self.device, &info);
        if tb.is_null() {
            return Err(sdl_error("create transfer buffer"));
        }
        self.transfer = tb;
        self.transfer_cap = size;
        Ok(())
    }

    /// Present one frame: upload the planes, run the color pipeline, letterbox, and
    /// swap. `data` is the tightly-packed frame in the plane layout of `fmt`.
    /// Colorimetry drives the fragment uniform block. Returns `Ok(true)` when drawn,
    /// `Ok(false)` when the frame was too short / the swapchain was unavailable
    /// (a transient, not an error).
    pub fn present(
        &mut self,
        data: &[u8],
        width: usize,
        height: usize,
        fmt: PlaneFormat,
        colorimetry: Colorimetry,
    ) -> Result<bool, Error> {
        if width == 0 || height == 0 {
            return Ok(true);
        }
        let (w, h) = (width as u32, height as u32);
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);
        let y_size = width * height;
        // Byte sizes of each plane region and their tightly-packed layout in `data`.
        let (c0_bytes, c1_bytes, chroma_mode) = match fmt {
            PlaneFormat::I420 => (cw * ch, cw * ch, ChromaMode::Planar),
            PlaneFormat::Nv12 => (2 * cw * ch, 0, ChromaMode::Interleaved),
            PlaneFormat::Gray8 => (0, 0, ChromaMode::Flat),
        };
        if data.len() < y_size + c0_bytes + c1_bytes {
            return Ok(false); // short frame — never index out of bounds
        }

        // SAFETY: the whole GPU frame is a sequence of null-checked SDL calls; the
        // upload copies exactly the bounds-checked plane regions above; the render
        // pass targets the acquired swapchain texture only.
        unsafe { self.present_inner(data, w, h, cw as u32, ch as u32, fmt, chroma_mode, colorimetry) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn present_inner(
        &mut self,
        data: &[u8],
        w: u32,
        h: u32,
        cw: u32,
        ch: u32,
        fmt: PlaneFormat,
        chroma_mode: ChromaMode,
        colorimetry: Colorimetry,
    ) -> Result<bool, Error> {
        self.ensure_planes(w, h, fmt)?;

        // Total upload = Y + chroma, tightly packed (transfer buffer layout matches
        // `data`'s layout so we can memcpy once).
        let y_size = (w * h) as usize;
        let (c0_bytes, c1_bytes) = match fmt {
            PlaneFormat::I420 => ((cw * ch) as usize, (cw * ch) as usize),
            PlaneFormat::Nv12 => ((2 * cw * ch) as usize, 0),
            PlaneFormat::Gray8 => (0, 0),
        };
        let total = y_size + c0_bytes + c1_bytes;
        self.ensure_transfer(total as u32)?;

        // Map the transfer buffer (cycle=true → fresh backing, never blocks on the
        // previous frame's upload — the SDL_GPU cycling idiom) and copy the planes.
        let map = SDL_MapGPUTransferBuffer(self.device, self.transfer, true);
        if map.is_null() {
            return Err(sdl_error("map transfer buffer"));
        }
        std::ptr::copy_nonoverlapping(data.as_ptr(), map as *mut u8, total);
        SDL_UnmapGPUTransferBuffer(self.device, self.transfer);

        // Acquire the command buffer + swapchain texture for this frame.
        let cmd = SDL_AcquireGPUCommandBuffer(self.device);
        if cmd.is_null() {
            return Err(sdl_error("acquire command buffer"));
        }
        let mut swap: *mut SDL_GPUTexture = std::ptr::null_mut();
        let mut sw: u32 = 0;
        let mut sh_px: u32 = 0;
        if !SDL_WaitAndAcquireGPUSwapchainTexture(cmd, self.window, &mut swap, &mut sw, &mut sh_px) {
            // Real failure: cancel and report.
            SDL_CancelGPUCommandBuffer(cmd);
            return Err(sdl_error("acquire swapchain texture"));
        }
        if swap.is_null() {
            // Too many frames in flight / minimised — not an error; drop this frame.
            SDL_CancelGPUCommandBuffer(cmd);
            return Ok(false);
        }

        // --- Copy pass: transfer buffer → plane textures (one region per plane). ---
        let copy = SDL_BeginGPUCopyPass(cmd);
        let planes = self.planes.as_ref().unwrap();
        let upload = |copy: *mut SDL_GPUCopyPass,
                      offset: u32,
                      tex: *mut SDL_GPUTexture,
                      tw: u32,
                      th: u32| {
            let src = SDL_GPUTextureTransferInfo {
                transfer_buffer: self.transfer,
                offset,
                // Tightly packed: 0 means "use the region w/h" (pixels_per_row=w).
                pixels_per_row: 0,
                rows_per_layer: 0,
            };
            let dst = SDL_GPUTextureRegion {
                texture: tex,
                w: tw,
                h: th,
                d: 1,
                ..Default::default()
            };
            SDL_UploadToGPUTexture(copy, &src, &dst, false);
        };
        upload(copy, 0, planes.y, w, h);
        match fmt {
            PlaneFormat::I420 => {
                upload(copy, y_size as u32, planes.c0, cw, ch);
                upload(copy, (y_size + c0_bytes) as u32, planes.c1, cw, ch);
            }
            PlaneFormat::Nv12 => {
                // R8G8 texture of cw×ch: the transfer holds 2*cw*ch bytes tightly.
                upload(copy, y_size as u32, planes.c0, cw, ch);
            }
            PlaneFormat::Gray8 => {}
        }
        SDL_EndGPUCopyPass(copy);

        // --- Render pass: fullscreen triangle, letterboxed. ---
        let color_target = SDL_GPUColorTargetInfo {
            texture: swap,
            clear_color: SDL_FColor { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            load_op: SDL_GPU_LOADOP_CLEAR, // black bars behind the letterbox
            store_op: SDL_GPU_STOREOP_STORE,
            ..Default::default()
        };
        let pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, std::ptr::null());

        SDL_BindGPUGraphicsPipeline(pass, self.pipeline);

        // Letterbox: fit the frame's aspect into the swapchain rect (we own geometry).
        let vp = fitted_viewport(w, h, sw, sh_px);
        SDL_SetGPUViewport(pass, &vp);

        // Bind the three sampled textures per pixel format. The shader reads Cb/Cr
        // according to chroma_mode (CHROMA_PLANAR/INTERLEAVED/FLAT), so:
        //   i420  → slot1 = U (R8), slot2 = V (R8): shader reads .r from each.
        //   nv12  → slot1 = interleaved CbCr (R8G8): shader reads .r=Cb, .g=Cr; slot2
        //           is bound but unused (Y as a harmless, valid-sampler placeholder).
        //   gray8 → chroma ignored (flat mode); slots 1/2 = Y placeholders.
        // Every slot must be a valid SAMPLER texture (SDL asserts non-null bindings),
        // hence the Y placeholder for the unused slots.
        let (t1, t2): (*mut SDL_GPUTexture, *mut SDL_GPUTexture) = match fmt {
            PlaneFormat::I420 => (planes.c0, planes.c1),
            PlaneFormat::Nv12 => (planes.c0, planes.y),
            PlaneFormat::Gray8 => (planes.y, planes.y),
        };
        let bindings = [
            SDL_GPUTextureSamplerBinding { texture: planes.y, sampler: self.sampler },
            SDL_GPUTextureSamplerBinding { texture: t1, sampler: self.sampler },
            SDL_GPUTextureSamplerBinding { texture: t2, sampler: self.sampler },
        ];
        SDL_BindGPUFragmentSamplers(pass, 0, bindings.as_ptr(), 3);

        // Push the color uniform block (fragment slot 0).
        let uniforms = FragUniforms::build(colorimetry, chroma_mode);
        let bytes = uniforms.as_bytes();
        SDL_PushGPUFragmentUniformData(
            cmd,
            0,
            bytes.as_ptr() as *const c_void,
            bytes.len() as u32,
        );

        // Fullscreen triangle: 3 vertices, no buffers.
        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(pass);

        if !SDL_SubmitGPUCommandBuffer(cmd) {
            return Err(sdl_error("submit command buffer"));
        }
        Ok(true)
    }

    /// Render one frame to an offscreen RGBA texture and download it (for the
    /// window-free GPU golden test). Returns the tightly-packed RGBA8 bytes. Uses the
    /// same pipeline as [`present`] but a fixed R8G8B8A8_UNORM target instead of the
    /// swapchain. Only compiled for tests.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        fmt: PlaneFormat,
        colorimetry: Colorimetry,
        out_w: u32,
        out_h: u32,
    ) -> Result<Vec<u8>, Error> {
        // SAFETY: a self-contained GPU frame; every pointer is null-checked; the
        // download reads exactly out_w*out_h*4 bytes into a sized transfer buffer.
        unsafe { self.render_offscreen_inner(data, width, height, fmt, colorimetry, out_w, out_h) }
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn render_offscreen_inner(
        &mut self,
        data: &[u8],
        w: u32,
        h: u32,
        fmt: PlaneFormat,
        colorimetry: Colorimetry,
        out_w: u32,
        out_h: u32,
    ) -> Result<Vec<u8>, Error> {
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        self.ensure_planes(w, h, fmt)?;
        let y_size = (w * h) as usize;
        let (c0_bytes, c1_bytes, chroma_mode) = match fmt {
            PlaneFormat::I420 => ((cw * ch) as usize, (cw * ch) as usize, ChromaMode::Planar),
            PlaneFormat::Nv12 => ((2 * cw * ch) as usize, 0, ChromaMode::Interleaved),
            PlaneFormat::Gray8 => (0, 0, ChromaMode::Flat),
        };
        let total = y_size + c0_bytes + c1_bytes;
        self.ensure_transfer(total as u32)?;
        let map = SDL_MapGPUTransferBuffer(self.device, self.transfer, true);
        if map.is_null() {
            return Err(sdl_error("map transfer buffer (offscreen)"));
        }
        std::ptr::copy_nonoverlapping(data.as_ptr(), map as *mut u8, total);
        SDL_UnmapGPUTransferBuffer(self.device, self.transfer);

        // Offscreen color target (sampler+color_target so we can also download it).
        let target_info = SDL_GPUTextureCreateInfo {
            r#type: SDL_GPU_TEXTURETYPE_2D,
            format: SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
            usage: SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER,
            width: out_w,
            height: out_h,
            layer_count_or_depth: 1,
            num_levels: 1,
            sample_count: SDL_GPU_SAMPLECOUNT_1,
            ..Default::default()
        };
        let target = SDL_CreateGPUTexture(self.device, &target_info);
        if target.is_null() {
            return Err(sdl_error("create offscreen target"));
        }
        // A download transfer buffer for the RGBA readback.
        let dl_size = out_w * out_h * 4;
        let dl_info = SDL_GPUTransferBufferCreateInfo {
            usage: SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
            size: dl_size,
            ..Default::default()
        };
        let dl = SDL_CreateGPUTransferBuffer(self.device, &dl_info);
        if dl.is_null() {
            SDL_ReleaseGPUTexture(self.device, target);
            return Err(sdl_error("create download buffer"));
        }

        let cmd = SDL_AcquireGPUCommandBuffer(self.device);
        if cmd.is_null() {
            SDL_ReleaseGPUTexture(self.device, target);
            SDL_ReleaseGPUTransferBuffer(self.device, dl);
            return Err(sdl_error("acquire command buffer (offscreen)"));
        }
        // Upload planes.
        let copy = SDL_BeginGPUCopyPass(cmd);
        let planes = self.planes.as_ref().unwrap();
        let upload = |copy: *mut SDL_GPUCopyPass, offset: u32, tex: *mut SDL_GPUTexture, tw: u32, th: u32| {
            let src = SDL_GPUTextureTransferInfo {
                transfer_buffer: self.transfer,
                offset,
                pixels_per_row: 0,
                rows_per_layer: 0,
            };
            let dst = SDL_GPUTextureRegion { texture: tex, w: tw, h: th, d: 1, ..Default::default() };
            SDL_UploadToGPUTexture(copy, &src, &dst, false);
        };
        upload(copy, 0, planes.y, w, h);
        match fmt {
            PlaneFormat::I420 => {
                upload(copy, y_size as u32, planes.c0, cw, ch);
                upload(copy, (y_size + c0_bytes) as u32, planes.c1, cw, ch);
            }
            PlaneFormat::Nv12 => upload(copy, y_size as u32, planes.c0, cw, ch),
            PlaneFormat::Gray8 => {}
        }
        SDL_EndGPUCopyPass(copy);

        // Render pass into the offscreen target (viewport = full target, no letterbox).
        let color_target = SDL_GPUColorTargetInfo {
            texture: target,
            clear_color: SDL_FColor { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            load_op: SDL_GPU_LOADOP_CLEAR,
            store_op: SDL_GPU_STOREOP_STORE,
            ..Default::default()
        };
        let pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, std::ptr::null());
        SDL_BindGPUGraphicsPipeline(pass, self.pipeline);
        let vp = SDL_GPUViewport {
            x: 0.0,
            y: 0.0,
            w: out_w as f32,
            h: out_h as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        SDL_SetGPUViewport(pass, &vp);
        let (t1, t2) = match fmt {
            PlaneFormat::I420 => (planes.c0, planes.c1),
            PlaneFormat::Nv12 => (planes.c0, planes.y),
            PlaneFormat::Gray8 => (planes.y, planes.y),
        };
        let bindings = [
            SDL_GPUTextureSamplerBinding { texture: planes.y, sampler: self.sampler },
            SDL_GPUTextureSamplerBinding { texture: t1, sampler: self.sampler },
            SDL_GPUTextureSamplerBinding { texture: t2, sampler: self.sampler },
        ];
        SDL_BindGPUFragmentSamplers(pass, 0, bindings.as_ptr(), 3);
        let uniforms = FragUniforms::build(colorimetry, chroma_mode);
        let bytes = uniforms.as_bytes();
        SDL_PushGPUFragmentUniformData(cmd, 0, bytes.as_ptr() as *const c_void, bytes.len() as u32);
        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(pass);

        // Download the target to the transfer buffer.
        let dlcopy = SDL_BeginGPUCopyPass(cmd);
        let region = SDL_GPUTextureRegion { texture: target, w: out_w, h: out_h, d: 1, ..Default::default() };
        let dst = SDL_GPUTextureTransferInfo {
            transfer_buffer: dl,
            offset: 0,
            pixels_per_row: 0,
            rows_per_layer: 0,
        };
        SDL_DownloadFromGPUTexture(dlcopy, &region, &dst);
        SDL_EndGPUCopyPass(dlcopy);

        // Submit + fence so the download is complete before we map it.
        let fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
        if fence.is_null() {
            SDL_ReleaseGPUTexture(self.device, target);
            SDL_ReleaseGPUTransferBuffer(self.device, dl);
            return Err(sdl_error("submit+fence (offscreen)"));
        }
        SDL_WaitForGPUFences(self.device, true, &fence, 1);
        SDL_ReleaseGPUFence(self.device, fence);

        let mapped = SDL_MapGPUTransferBuffer(self.device, dl, false);
        if mapped.is_null() {
            SDL_ReleaseGPUTexture(self.device, target);
            SDL_ReleaseGPUTransferBuffer(self.device, dl);
            return Err(sdl_error("map download buffer"));
        }
        let mut out = vec![0u8; dl_size as usize];
        std::ptr::copy_nonoverlapping(mapped as *const u8, out.as_mut_ptr(), dl_size as usize);
        SDL_UnmapGPUTransferBuffer(self.device, dl);

        SDL_ReleaseGPUTexture(self.device, target);
        SDL_ReleaseGPUTransferBuffer(self.device, dl);
        Ok(out)
    }
}

/// Compute the letterboxed viewport: the largest rect with the frame's aspect ratio
/// that fits inside the `dst_w × dst_h` swapchain, centred. We own presentation
/// geometry now (the classic path used SDL's logical-presentation letterbox).
fn fitted_viewport(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> SDL_GPUViewport {
    let (sw, sh) = (src_w as f32, src_h as f32);
    let (dw, dh) = (dst_w as f32, dst_h as f32);
    if sw <= 0.0 || sh <= 0.0 || dw <= 0.0 || dh <= 0.0 {
        return SDL_GPUViewport { x: 0.0, y: 0.0, w: dw, h: dh, min_depth: 0.0, max_depth: 1.0 };
    }
    let src_aspect = sw / sh;
    let dst_aspect = dw / dh;
    let (vw, vh) = if dst_aspect > src_aspect {
        // Window wider than the frame: pillarbox (limit width).
        (dh * src_aspect, dh)
    } else {
        // Window taller: letterbox (limit height).
        (dw, dw / src_aspect)
    };
    SDL_GPUViewport {
        x: (dw - vw) * 0.5,
        y: (dh - vh) * 0.5,
        w: vw,
        h: vh,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

impl Drop for GpuRenderer {
    fn drop(&mut self) {
        self.release_planes();
        // SAFETY: releasing objects we own on the owning device, then unclaiming the
        // window (the sink still owns the SDL_Window itself and destroys it later),
        // then destroying the device. Ordering: pipeline/sampler/transfer before the
        // device; window unclaim before device destroy.
        unsafe {
            if !self.transfer.is_null() {
                SDL_ReleaseGPUTransferBuffer(self.device, self.transfer);
            }
            if !self.sampler.is_null() {
                SDL_ReleaseGPUSampler(self.device, self.sampler);
            }
            if !self.pipeline.is_null() {
                SDL_ReleaseGPUGraphicsPipeline(self.device, self.pipeline);
            }
            if !self.window.is_null() {
                SDL_ReleaseWindowFromGPUDevice(self.device, self.window);
            }
            SDL_DestroyGPUDevice(self.device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_pillarbox_math() {
        // 16:9 frame in a 4:3-ish window (wider dst aspect? no: 800/600=1.33 < 1.78)
        // → letterbox (limit height). Frame 1920×1080 into 800×600.
        let vp = fitted_viewport(1920, 1080, 800, 600);
        assert!((vp.w - 800.0).abs() < 0.5, "letterbox fills width: {}", vp.w);
        assert!(vp.h < 600.0, "letterbox shrinks height: {}", vp.h);
        assert!((vp.h - 450.0).abs() < 0.5, "16:9 into 800 wide → 450 tall: {}", vp.h);
        assert!(vp.x.abs() < 0.5 && vp.y > 0.0, "centred vertically");

        // 16:9 frame into a very wide window → pillarbox (limit width).
        let vp2 = fitted_viewport(1920, 1080, 2000, 600);
        assert!((vp2.h - 600.0).abs() < 0.5, "pillarbox fills height: {}", vp2.h);
        assert!(vp2.w < 2000.0, "pillarbox shrinks width: {}", vp2.w);
        assert!(vp2.x > 0.0, "centred horizontally");
    }

    #[test]
    fn letterbox_exact_fit() {
        // Same aspect → full viewport, no bars.
        let vp = fitted_viewport(640, 480, 1280, 960);
        assert!((vp.w - 1280.0).abs() < 0.5 && (vp.h - 960.0).abs() < 0.5);
        assert!(vp.x.abs() < 0.5 && vp.y.abs() < 0.5);
    }

    use super::super::color::Colorimetry;

    /// Build a constant i420 frame (`w×h`) with the given coded (Y,Cb,Cr) bytes.
    fn const_i420(w: usize, h: usize, y: u8, cb: u8, cr: u8) -> Vec<u8> {
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let mut v = vec![y; w * h];
        v.extend(std::iter::repeat_n(cb, cw * ch));
        v.extend(std::iter::repeat_n(cr, cw * ch));
        v
    }

    /// The center pixel's RGB from a downloaded RGBA8 buffer.
    fn center_rgb(buf: &[u8], w: u32, h: u32) -> (u8, u8, u8) {
        let x = w / 2;
        let y = h / 2;
        let i = ((y * w + x) * 4) as usize;
        (buf[i], buf[i + 1], buf[i + 2])
    }

    /// GPU golden test: render known YCbCr→RGB points through the real pipeline on a
    /// device and assert the downloaded colors. Skips cleanly (prints + returns) when
    /// no GPU device is available, so no-GPU machines stay green.
    #[test]
    fn gpu_ycbcr_goldens() {
        let mut r = match GpuRenderer::new_headless() {
            Ok(r) => r,
            Err(e) => {
                println!("gpu_ycbcr_goldens: no GPU device ({e:?}) — skipping");
                return;
            }
        };
        println!("gpu_ycbcr_goldens: running on driver={}", r.driver());

        let (w, h) = (16u32, 16u32);
        let (ow, oh) = (16u32, 16u32);
        let c = Colorimetry::resolve(1080, Some("bt709"), Some("limited"), None, None);

        // 1. Studio white (235,128,128) → (255,255,255) ± 2.
        let white = const_i420(w as usize, h as usize, 235, 128, 128);
        let out = r
            .render_offscreen(&white, w, h, PlaneFormat::I420, c, ow, oh)
            .expect("render white");
        let (rr, gg, bb) = center_rgb(&out, ow, oh);
        assert!(rr >= 253 && gg >= 253 && bb >= 253, "white ≈ 255: {rr},{gg},{bb}");

        // 2. Studio black (16,128,128) → (0,0,0) + 2.
        let black = const_i420(w as usize, h as usize, 16, 128, 128);
        let out = r
            .render_offscreen(&black, w, h, PlaneFormat::I420, c, ow, oh)
            .expect("render black");
        let (rr, gg, bb) = center_rgb(&out, ow, oh);
        assert!(rr <= 2 && gg <= 2 && bb <= 2, "black ≈ 0: {rr},{gg},{bb}");

        // 3. Mid grey (126,128,128) → equal R=G=B, in the mid range (the sRGB re-encode
        //    of the BT.1886-linearised mid grey lands well inside 0..255).
        let grey = const_i420(w as usize, h as usize, 126, 128, 128);
        let out = r
            .render_offscreen(&grey, w, h, PlaneFormat::I420, c, ow, oh)
            .expect("render grey");
        let (rr, gg, bb) = center_rgb(&out, ow, oh);
        assert!(rr.abs_diff(gg) <= 2 && gg.abs_diff(bb) <= 2, "grey neutral: {rr},{gg},{bb}");
        assert!(rr > 60 && rr < 230, "grey mid range: {rr}");

        // 4. Saturated red: full-range-ish red in bt709 limited. Coded Cr high, Cb low,
        //    Y moderate → R dominant, G/B small. We assert the HUE (R is the largest
        //    channel and clearly dominant) rather than an exact triple.
        let red = const_i420(w as usize, h as usize, 81, 90, 240);
        let out = r
            .render_offscreen(&red, w, h, PlaneFormat::I420, c, ow, oh)
            .expect("render red");
        let (rr, gg, bb) = center_rgb(&out, ow, oh);
        assert!(rr > gg && rr > bb, "red channel dominant: {rr},{gg},{bb}");
        assert!(rr > 180, "red is bright: {rr}");
    }

    /// Gray8 through the GPU path: flat chroma → neutral grey output.
    #[test]
    fn gpu_gray8_is_neutral() {
        let mut r = match GpuRenderer::new_headless() {
            Ok(r) => r,
            Err(e) => {
                println!("gpu_gray8_is_neutral: no GPU device ({e:?}) — skipping");
                return;
            }
        };
        let (w, h) = (16u32, 16u32);
        let c = Colorimetry::resolve(1080, None, None, None, None);
        // gray8: just w*h luma bytes at mid grey.
        let gray = vec![180u8; (w * h) as usize];
        let out = r
            .render_offscreen(&gray, w, h, PlaneFormat::Gray8, c, w, h)
            .expect("render gray8");
        let (rr, gg, bb) = center_rgb(&out, w, h);
        assert!(rr.abs_diff(gg) <= 2 && gg.abs_diff(bb) <= 2, "gray8 neutral: {rr},{gg},{bb}");
    }
}
