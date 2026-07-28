//! `sc-sdl3` — SDL3 windowing/presentation for streamcraft (spec: scraft-scope
//! update — "use SDL3 … swap our custom wayland and vulkan stuff to use sdl3
//! instead. Graphics sucks and we shouldn't waste time on it.").
//!
//! This crate replaces the hand-written `sc-wayland` shm client *and* the
//! clean-room `sc-vk` Vulkan renderer with one thin layer over SDL3: window,
//! events, and a GPU-accelerated presentation path. Two backends exist. The
//! **owned GPU render pipeline** ([`gpu`]) uses custom SPIR-V shaders doing
//! colorimetry-aware YCbCr→RGB (with a tone-map slot for HDR) on the SDL3 GPU API —
//! all color science is ours, since `SDL_UpdateYUVTexture` lets SDL pick the matrix
//! and cannot tone-map. The classic `SDL_Renderer` streaming-texture path
//! ([`video`]) is the fixed-function fallback used when GPU device creation fails,
//! and the backend the scope UI stays on. This crate is also where the scope UI's
//! draw backend lives (immediate mode over `SDL_RenderGeometryRaw`, per-frame
//! vertex arenas).
//!
//! The one sanctioned dependency is `sdl3-sys` — pure auto-generated bindings, no
//! logic (the `ash`/`pipewire` pattern for plugin crates), linking the system
//! libSDL3 via pkg-config. All `unsafe` is confined to [`video`] and [`gpu`], with a
//! SAFETY note per block (workspace lint policy).
//!
//! Elements:
//! - [`Sdl3VideoSink`] (`sdl3videosink`) — clock-paced windowed video sink with
//!   QoS, dynamic-caps geometry, degrade-to-drops on headless boxes.

pub mod gpu;
pub mod sink;
pub mod video;

pub use sink::Sdl3VideoSink;

use streamcraft_core::element::Element;
use streamcraft_core::registry::Registry;

/// Register this crate's elements for name-based construction (spec: Plugins —
/// `streamcraft launch … ! sdl3videosink`). Typed `use` + constructor stays
/// primary; the descriptor is `&'static`, taken from a throwaway default instance.
pub fn register(registry: &mut Registry) {
    registry.register(Sdl3VideoSink::new().desc());
}

/// Ask SDL to prefer the **Wayland** video driver when it is available, falling back to X11.
/// SDL picks its video driver at video-subsystem init by walking this comma-separated priority
/// list, so this MUST run before `SDL_InitSubSystem(SDL_INIT_VIDEO)` (every init site calls it).
/// Set at the default hint priority, so an explicit `SDL_VIDEO_DRIVER` environment variable
/// still wins — a user who forces `x11` (or anything else) is honoured. Idempotent.
pub(crate) fn prefer_wayland_driver() {
    // SAFETY: a plain C call taking two `'static` NUL-terminated strings; SDL reads the hint
    // registry lazily at init and this needs no prior `SDL_Init`.
    unsafe {
        sdl3_sys::everything::SDL_SetHint(
            sdl3_sys::everything::SDL_HINT_VIDEO_DRIVER,
            c"wayland,x11".as_ptr(),
        );
    }
}
