//! profluens-scope — the inspector (spec: Introspection protocol and pf-scope).
//!
//! Two modes over one binary protocol:
//! - **Attach**: the `pf-scope` binary connects to any running profluens app
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
//! v1 shows, all live: the graph view ([`layout`] + per-edge negotiated formats +
//! per-node counters), per-element throughput, the merged events-and-logs feed, and
//! pause/resume. Latency histograms, property editing, buffer peeking, and the MCP
//! server ride the same protocol in later phases.

#![allow(dead_code)]

pub mod app;
pub mod client;
pub mod layout;
pub mod ui;

use profluens_core::pipeline::Pipeline;

/// In-process inspector handle (embed mode).
pub struct Scope {
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Scope {
    /// Spawn the inspector UI in-process on its own thread, attached to `pipeline`
    /// via the introspection protocol on a private temp socket.
    ///
    /// Call after the topology is built (the same convention as
    /// `Pipeline::serve_introspection`). Embed and attach modes share the whole
    /// client + UI path, so the two cannot drift. Note: some platforms restrict
    /// window creation to the main thread; attach mode is the primary path, embed
    /// is best-effort for dev builds.
    pub fn spawn(pipeline: &mut Pipeline) -> std::io::Result<Scope> {
        let path = std::env::temp_dir().join(format!("pf-scope-{}.sock", std::process::id()));
        pipeline
            .serve_introspection(&path)
            .map_err(|e| std::io::Error::other(format!("serve_introspection: {e:?}")))?;
        let thread = std::thread::Builder::new()
            .name("pf-scope".into())
            .spawn(move || {
                if let Err(e) = app::run(&path, app::AppOpts::default()) {
                    eprintln!("pf-scope (embed): {e}");
                }
            })?;
        Ok(Scope { thread: Some(thread) })
    }

    /// Block until the inspector window is closed.
    pub fn join(mut self) {
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
