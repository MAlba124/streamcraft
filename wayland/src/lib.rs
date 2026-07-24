//! sc-wayland — a hand-written Wayland shm video sink (spec: Milestone applications §5 —
//! play a video file; the display half). Zero deps beyond `streamcraft-core`,
//! `streamcraft-video`, and `libc` (used *only* for the four syscalls std does not expose:
//! `memfd_create`, `ftruncate`, `mmap`/`munmap`, and `sendmsg` with `SCM_RIGHTS`). The
//! Wayland wire protocol is hand-rolled over `std::os::unix::net::UnixStream`.
//!
//! Module map:
//! * [`wire`] — the wire format: message builder/serialiser + a total (never-panicking)
//!   event parser (header, args, string/array padding).
//! * [`protocol`] — interface names, request/event opcodes, enum constants, each cited
//!   against `wayland/spec/{wayland,xdg-shell,linux-dmabuf-unstable-v1}.xml`.
//! * [`sys`] — the one audited `unsafe` module (memfd / mmap / SCM_RIGHTS fd passing).
//! * [`convert`] — i420 / gray8 → XRGB8888, integer BT.601, one pass into the shm buffer.
//! * [`client`] — a minimal Wayland client: connect, registry, window, shm swapchain,
//!   present, event pump. Additively, it can also *import and present exported
//!   DMA-BUFs* via `zwp_linux_dmabuf_v1` (the GPU present path sc-vk drives) — a parallel
//!   code path alongside the shm swapchain, gated by the same `wl_buffer.release` reuse.
//! * [`sink`] — the [`WaylandVideoSink`] element (active, clock-paced, QoS, dynamic caps).
//!
//! The protocol specs live in the tree (`wayland/spec/`), per project convention.

// `unsafe` is permitted only in the audited `sys` module (memfd / mmap / SCM_RIGHTS),
// following core's `memory`/`ring` pattern — `deny` (not `forbid`) so that one module can
// locally `#![allow(unsafe_code)]`.
#![deny(unsafe_code)]

pub mod client;
pub mod convert;
pub mod protocol;
pub mod sink;
pub mod sys;
pub mod wire;

pub use client::WaylandClient;
pub use sink::WaylandVideoSink;

use streamcraft_core::element::Element;
use streamcraft_core::registry::Registry;

/// Register this crate's elements for name-based construction (spec: Plugins —
/// `streamcraft launch … ! waylandvideosink`). Typed `use` + constructor stays
/// primary; the descriptor is `&'static`, taken from a throwaway default instance.
pub fn register(registry: &mut Registry) {
    registry.register(WaylandVideoSink::new().desc());
}
