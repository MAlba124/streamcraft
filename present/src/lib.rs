//! `pf-present` — a minimal, zero-allocation Wayland presenter for profluens, spoken directly
//! on the display socket (no libwayland, no SDL).
//!
//! ## Why
//!
//! libwayland-client marshals every request through a heap-allocated `wl_closure`; at video
//! rates the present path (`wl_surface.attach`/`damage`/`commit` + `frame` callbacks) is
//! thousands of tiny allocs a second (measured ~13K/30s in a HEVC capture). This crate speaks
//! the wire format itself ([`wire`]), serializing into a **reused** send buffer and recycling
//! `wl_buffer`s on release, so the steady-state present path allocates nothing.
//!
//! ## Model (spec: pf-scope + the zero-copy VA→dmabuf path)
//!
//! A window is an `xdg_toplevel` with two child surfaces the compositor composites (ideally on
//! hardware overlay planes):
//! - a **video** surface backed by a `wl_buffer` imported from the VA-API decode surface's
//!   dmabuf (`zwp_linux_dmabuf_v1`) — zero-copy;
//! - a **GUI** `wl_subsurface` backed by a `wl_shm` buffer the crate **software-renders** (the
//!   scope UI / controls; ARGB with alpha over the video), updated only on change.
//!
//! No GPU shaders are needed: the compositor does the blend. See [`conn`] for the socket +
//! `SCM_RIGHTS` fd passing and [`protocol`] for the interface opcodes.

pub mod conn;
pub mod control;
pub mod protocol;
pub mod raster;
pub mod rawsink;
pub mod sink;
pub mod ui;
pub mod window;
pub mod wire;

pub use control::{PlayerControl, UiCommand};
pub use rawsink::WaylandRawSink;
pub use sink::WaylandVideoSink;
