//! What the engine tells the application, and how it refuses.
//!
//! Both types are deliberately small. The engine's contract with a music player is the one
//! rodio's queue had: the app polls on a timer, reads [`Engine::queue_len`] and
//! [`Engine::position`], and is *told* only about the things a poll cannot reconstruct — that a
//! boundary happened, that a duration became known, that something went wrong.
//!
//! [`Engine::queue_len`]: crate::Engine::queue_len
//! [`Engine::position`]: crate::Engine::position

use std::time::Duration;

/// Something the engine did that the application could not have observed by polling.
///
/// Drained by [`Engine::poll_events`](crate::Engine::poll_events), which is meant to be called
/// from the app's existing tick (musikkspiller's 250 ms one). Events queue up between polls and
/// are delivered in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    /// A track became the current one from a **cold or explicit** start: the first
    /// [`enqueue`](crate::Engine::enqueue) into an idle engine, or any
    /// [`play_now`](crate::Engine::play_now).
    TrackStarted,
    /// The queue **advanced**: the previous track ended and the one behind it became current.
    /// This is the gapless boundary — by the time it is delivered the next track's audio is
    /// already queued behind the previous one's tail.
    TrackChanged,
    /// The queue drained: nothing is current and nothing is queued
    /// ([`queue_len`](crate::Engine::queue_len) is 0).
    ///
    /// **Playout may still be draining.** The last track's tail is in the output ring and keeps
    /// playing — deliberately, because cutting it off would clip the end of every album. Ask
    /// [`buffered`](crate::Engine::buffered) how much is left if you need to know when the room
    /// actually goes quiet.
    ///
    /// Only *natural* exhaustion emits this. [`stop`](crate::Engine::stop),
    /// [`clear_queue`](crate::Engine::clear_queue) and [`play_now`](crate::Engine::play_now) do
    /// not — the app asked for those and does not need to be told, and an app that auto-advances
    /// its own playlist on `TrackEnded` would otherwise fight its own Stop button.
    TrackEnded,
    /// The current track's duration, published once when it becomes current. Absent for a
    /// stream whose container declares none (a bare ADTS AAC, a still-downloading Ogg).
    DurationKnown(Duration),
    /// A track failed — to open (bad path, unsupported container, no audio track) or while
    /// playing (a decode error, the output closing underneath it).
    ///
    /// A failed **queued** track is discarded and the current one plays on untouched. A failed
    /// **current** track is followed by the queue advancing, or by [`TrackEnded`] if the queue
    /// is empty.
    ///
    /// [`TrackEnded`]: EngineEvent::TrackEnded
    Error { message: String },
}

/// Why a request was refused. Every variant is a *refusal*, not a failure: nothing has changed
/// and the call can be retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineError {
    /// A track is already queued behind the current one.
    ///
    /// This is rodio's `append` guard reproduced exactly: the queue is one deep, because its
    /// only purpose is to have the *next* track pre-rolled, and a deeper queue would mean
    /// holding several fully-built pipelines (threads, pools, decoded-ahead audio) for tracks
    /// the user may skip past. Wait for [`EngineEvent::TrackChanged`] and enqueue again.
    AlreadyQueued,
    /// The engine is shutting down — its [`Engine`](crate::Engine) has been dropped, or is
    /// being dropped on another thread. Nothing was queued.
    Shutdown,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::AlreadyQueued => {
                f.write_str("a track is already queued (the queue is one deep)")
            }
            EngineError::Shutdown => f.write_str("the engine is shutting down"),
        }
    }
}

impl std::error::Error for EngineError {}
