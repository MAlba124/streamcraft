//! `scplay-ui` — the SDL3 immediate-mode GUI media player, streamcraft's flagship
//! front-end (spec: streamcraft.md Milestone applications §5 + the UI `<update>` §1209 —
//! "use SDL3 with a custom UI library on top … the UI must be the same philosophy as the
//! rest of SC (very performant, per-frame arenas)").
//!
//! # Architecture (one window, clock-paced video)
//!
//! A media player cannot have two SDL windows fighting, so this crate owns **the** window
//! and pulls video across the thread boundary through a lock-light frame slot:
//!
//! - the **streaming thread** runs [`sc_play::Player`] to EOS; its video branch ends in a
//!   [`FrameSlotSink`](framesink::FrameSlotSink) — an active, clock-paced sink that paces
//!   each frame on the pipeline clock exactly like [`sc_sdl3::Sdl3VideoSink`] but, instead
//!   of presenting, copies the released frame into a shared [`FrameSlot`](framesink::FrameSlot);
//! - the **GUI thread** (the main thread — SDL wants the event pump there) owns the window,
//!   uploads the latest slot frame to a streaming YUV texture, letterboxes it, draws the
//!   transport/menu/stats UI on top with scope's immediate-mode toolkit, and presents at
//!   its own vsync.
//!
//! Because the audio device sink is the pipeline clock, video *releases* on the audio
//! timeline (A/V sync preserved); the GUI merely presents whatever is currently released.
//!
//! # The two video paths
//!
//! When the video decodes on the GPU (VA-API) and a GLES/EGL DMA-BUF-import context stands
//! up, the player runs a **zero-copy** path: the decoder exports each decoded surface as a
//! DMA-BUF ([`sc_vaapi::gpuframe`]) that the GLES backend ([`gl`]) imports straight into a
//! `GL_TEXTURE_EXTERNAL_OES` and samples — the decoded pixels never touch the CPU. Otherwise
//! (software decode, or no EGL DMABUF import) it uses the classic [`FrameSlot`] → SDL_Renderer
//! YUV-texture-upload path. The choice is made once at startup and logged (`video path = …`).
//!
//! # Modules
//! - [`framesink`] — the clock-paced frame-slot sink + the shared triple-buffer slot (CPU
//!   path) and the zero-copy `video/gpu` variant that routes DMA-BUF frames to the GUI;
//! - [`backend`]   — the video-augmented copy of scope's SDL backend (the one `unsafe`
//!   module): the SDL_Renderer YUV texture + letterbox under scope's `DrawList` flush, OR the
//!   GLES external-image path;
//! - [`gl`]        — the zero-copy GLES backend: `eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT)` +
//!   `samplerExternalOES` + the UI DrawList as GLES geometry;
//! - [`ui`]        — the per-frame immediate-mode player UI (transport, menus, stats);
//! - [`app`]       — the glue: build the player, spawn the streaming thread, run the loop.

pub mod app;
pub mod backend;
pub mod framesink;
pub mod gl;
pub mod ui;

pub use app::{run, Config};
pub use framesink::{FrameSlot, FrameSlotSink, SlotPix};
