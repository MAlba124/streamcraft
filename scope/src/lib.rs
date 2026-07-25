//! streamcraft-scope — the inspector (spec: Introspection protocol and scraft-scope).
//!
//! Two modes over one binary protocol:
//! - **Attach**: the `scraft-scope` binary connects to any running streamcraft app
//!   by socket — zero code in the target beyond core's `introspect` feature.
//! - **Embed**: [`Scope::spawn`] runs the same UI in-process on its own thread
//!   for dev builds — one line in `main`, same protocol underneath.
//!
//! The UI is a **custom immediate-mode toolkit over SDL3** (spec update: "use SDL3
//! with a custom UI library on top … the UI must be the same philosophy as the rest
//! of SC — very performant, per-frame arenas"). It is not Slint; the earlier Slint
//! plan was dropped. See [`ui`] for the reusable widget layer (draw list on
//! `SDL_RenderGeometryRaw`, per-frame bump arena, embedded bitmap-font text, core
//! widgets) and `src/bin/ui_demo.rs` for a fake-data inspector demo.
//!
//! Shows (all live): graph view, latency panel, event/log feeds, debugging
//! (pause/step, counters, property editing, buffer-metadata peeking), and an MCP
//! server exposing the same protocol as agent tools. Wiring the protocol client and
//! those panels onto this UI layer is a later integration phase.

#![allow(dead_code)]

pub mod ui;

use streamcraft_core::pipeline::Pipeline;

/// In-process inspector handle (embed mode).
pub struct Scope {
    _priv: (),
}

impl Scope {
    /// Spawn the inspector UI in-process on its own thread, attached to `pipeline`
    /// via the introspection protocol.
    pub fn spawn(_pipeline: &Pipeline) -> Self {
        todo!("spec: scraft-scope — embed mode")
    }
}
