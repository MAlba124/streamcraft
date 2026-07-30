//! [`PlayerControl`] — the thread-safe hand-off between the presenter window (input) and the
//! app driving the pipeline. The window pushes [`UiCommand`]s (from pointer/keyboard), the app
//! drains them and applies them to the pipeline's pause/seek/stop handles; the app publishes
//! `duration` + `paused` back so the HUD can draw a real timeline + play/pause glyph.
//!
//! Kept a neutral command queue (no pipeline types) so `pf-present` stays free of the player
//! plumbing — the app owns the handles and does the mapping.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// A user action from the presenter window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UiCommand {
    /// Toggle pause/resume (click the video or the play glyph, space/`p`).
    TogglePause,
    /// Seek to a fraction `0.0..=1.0` of the duration (click/drag the timeline).
    SeekFraction(f32),
    /// End playback (window close, `q`/Esc).
    Quit,
}

/// Shared window↔app control state. Create with [`new`](Self::new); clone the `Arc` into the
/// sink (window side) and the app's control loop.
#[derive(Default)]
pub struct PlayerControl {
    /// Window → app: pending commands, drained by the app's control loop.
    commands: Mutex<Vec<UiCommand>>,
    /// App → window: media duration in ns (`0` = unknown), for the HUD timeline.
    duration_ns: AtomicU64,
    /// App → window: whether playback is paused, for the HUD play/pause glyph.
    paused: AtomicBool,
}

impl PlayerControl {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Window side: queue a command (the app drains it).
    pub fn push(&self, cmd: UiCommand) {
        if let Ok(mut q) = self.commands.lock() {
            q.push(cmd);
        }
    }

    /// App side: take all queued commands.
    pub fn drain(&self) -> Vec<UiCommand> {
        // COLD-ish: at most a handful of user actions between polls; the Vec grows once.
        #[allow(clippy::disallowed_methods)]
        match self.commands.lock() {
            Ok(mut q) => std::mem::take(&mut *q),
            Err(_) => Vec::new(),
        }
    }

    /// App side: publish the media duration (ns).
    pub fn set_duration_ns(&self, ns: u64) {
        self.duration_ns.store(ns, Ordering::Relaxed);
    }
    /// Window side: the media duration (ns), or `0` if unknown.
    pub fn duration_ns(&self) -> u64 {
        self.duration_ns.load(Ordering::Relaxed)
    }

    /// App side: publish the pause state.
    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Relaxed);
    }
    /// Window side: whether playback is paused.
    pub fn paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }
}
