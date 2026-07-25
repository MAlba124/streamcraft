//! streamcraft-scope — the inspector (spec: Introspection protocol and scraft-scope).
//!
//! Two modes over one binary protocol:
//! - **Attach**: the `scraft-scope` binary connects to any running streamcraft app
//!   by socket — zero code in the target beyond core's `introspect` feature.
//! - **Embed**: [`Scope::spawn`] runs the same Slint UI in-process on its own thread
//!   for dev builds — one line in `main`, same protocol underneath.
//!
//! Shows (all live): graph view, latency panel, event/log feeds, debugging
//! (pause/step, counters, property editing, buffer-metadata peeking), and an MCP
//! server exposing the same protocol as agent tools.

#![allow(dead_code)]

pub mod layout;

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
