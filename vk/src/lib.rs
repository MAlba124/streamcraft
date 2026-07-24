//! sc-vk — the clean-room GPU video renderer (spec: Milestone applications §5; the
//! project's algorithm-citation rule).
//!
//! Architecture: **Vulkan renders into exported DMA-BUFs; the hand-written Wayland
//! client (`sc-wayland`) presents them via `zwp_linux_dmabuf_v1`.** There is no
//! libwayland-client, no WSI swapchain, and no libplacebo — presentation stays on our
//! own wire-protocol code, and every algorithm is implemented from its published
//! source (see `REFERENCES.md`): the colour math from Rec. ITU-R BT.601-7, the
//! protocol from the vendored XML, the Vulkan usage from the Khronos specification.
//!
//! Per frame: memcpy the packed I420 planes into a host-visible storage buffer, one
//! compute dispatch (`shaders/yuv2rgb.comp`, committed SPIR-V alongside the GLSL
//! source) converts into a per-slot linear XRGB8888 framebuffer whose memory was
//! allocated dma-buf-exportable, fence-wait, then attach/commit the imported
//! `wl_buffer` — release-gated triple buffering, exactly like the shm swapchain.
//!
//! v1 limitations (documented follow-ups in `REFERENCES.md`): LINEAR modifier only;
//! output at the video's own dimensions (no scaling — the citable scaler ladder,
//! Lanczos/Mitchell–Netravali, comes with window-driven resize); fence-per-frame
//! CPU-throttled sync (explicit-sync protocol integration later); 8-bit i420 in,
//! XRGB8888 out (10-bit/HDR needs the wider vocabulary first).
//!
//! The one sanctioned dependency is `ash` — pure Vulkan bindings, `loaded` (dlopens
//! libvulkan at runtime, no build-time linkage). All raw calls live in the audited
//! [`gpu`] module; the rest of the crate denies `unsafe`.

#![deny(unsafe_code)]

pub mod gpu;
pub mod sink;

pub use sink::VkVideoSink;

use streamcraft_core::element::Element;
use streamcraft_core::registry::Registry;

/// Register this crate's elements for name-based construction (spec: Plugins —
/// `streamcraft launch … ! vkvideosink`).
pub fn register(registry: &mut Registry) {
    registry.register(VkVideoSink::new().desc());
}
