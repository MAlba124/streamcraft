//! `sc-sdl3` — SDL3 windowing/presentation for streamcraft (spec: scraft-scope
//! update — "use SDL3 … swap our custom wayland and vulkan stuff to use sdl3
//! instead. Graphics sucks and we shouldn't waste time on it.").
//!
//! This crate replaces the hand-written `sc-wayland` shm client *and* the
//! clean-room `sc-vk` Vulkan renderer with one thin layer over SDL3: window,
//! events, and a GPU-accelerated presentation path (streaming IYUV texture —
//! YUV→RGB on the GPU, no CPU conversion pass). It is also where the scope UI's
//! draw backend will live (immediate mode over `SDL_RenderGeometryRaw`,
//! per-frame vertex arenas).
//!
//! The one sanctioned dependency is `sdl3-sys` — pure auto-generated bindings, no
//! logic (the `ash`/`pipewire` pattern for plugin crates), linking the system
//! libSDL3 via pkg-config. All `unsafe` is confined to [`video`], with a SAFETY
//! note per block (workspace lint policy).
//!
//! Elements:
//! - [`Sdl3VideoSink`] (`sdl3videosink`) — clock-paced windowed video sink with
//!   QoS, dynamic-caps geometry, degrade-to-drops on headless boxes.

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
