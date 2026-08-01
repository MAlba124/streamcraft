//! `pf-player-engine` — a **gapless track queue** over one shared audio output (spec:
//! `gapless.md`, Phase 2). The backend a music player consumes in place of rodio's
//! `Player`/queue.
//!
//! ```ignore
//! let engine = Engine::new()?;                  // owns the device, canonical f32/48k/2ch
//! engine.play_now(Track::file("01.flac"));
//! // ~5 s before the end, the way the app's tick already decides to:
//! engine.enqueue(Track::file("02.flac"))?;      // built + pre-rolled immediately
//! // …the boundary happens by itself, sample-continuously.
//! for ev in engine.poll_events() { /* TrackChanged, DurationKnown, Error … */ }
//! ```
//!
//! # What makes it gapless
//!
//! One `AudioOut` for the life of the application: one device, one ring, one clock, one fixed
//! format. A track is a whole `Pipeline` whose sink is a *producer* attached to that output. At
//! end of stream that producer **detaches without draining** — up to half a second of its audio
//! is still in the ring and still playing — and the next track's producer attaches behind it.
//! The seam is a scheduler pass against a 500 ms runway, so it is sample-continuous, and
//! `pf-pipewire`'s own tests prove that seam byte-exact at dozens of split points.
//!
//! This crate is what turns that mechanism into a queue: it opens the next track early, holds
//! it fully decoded behind a gate, and at the boundary opens that gate. Its own
//! acceptance test (`tests/gapless.rs`) plays one continuous signal split across two files and
//! compares the device's recording with the two files decoded separately and concatenated —
//! byte for byte, at several split points and at two sample rates.
//!
//! # The shape of the API
//!
//! Deliberately rodio's, because that is the contract the application already has:
//!
//! | rodio | here |
//! |---|---|
//! | `sink.append(src)` | [`Engine::enqueue`] — refuses if one is already queued |
//! | `sink.len()` | [`Engine::queue_len`] — playing + queued; the transition/end signal |
//! | `sink.get_pos()` | [`Engine::position`] |
//! | `sink.try_seek(d)` | [`Engine::seek`] |
//! | `sink.set_volume(v)` | [`Engine::set_volume`] — ramped, on the device |
//! | `sink.pause()`/`play()` | [`Engine::pause`] / [`Engine::resume`] |
//! | `sink.stop()` | [`Engine::stop`] |
//!
//! Everything is callable from any thread and returns immediately. The two operations that
//! genuinely block — opening a file (probe, container head, seek index, pre-roll) and tearing a
//! pipeline down — happen on the engine's own worker thread, so [`enqueue`](Engine::enqueue) and
//! [`play_now`](Engine::play_now) return before the disk has been touched and report their
//! outcome as an [`EngineEvent`].
//!
//! # Position, and what it means
//!
//! [`position`](Engine::position) is **audible** time within the current track: the device's
//! write-side counter minus its measured output delay, floored at the last seek target and
//! monotonic within the track. It reads zero until the current track's audio actually starts
//! coming out of the speakers — which, for the width of a gapless handoff, is while the previous
//! track's tail is still playing. A track with a [`stretch`](Track::stretch) stage reports
//! **source** time instead, from the stretcher.
//!
//! # Allocation
//!
//! This is application-side controller code and it allocates freely: labels, event queues, one
//! boxed sink and one boxed closure per track. Every module here carries the documented
//! `clippy.toml` exception for that, with one exception of its own — the gate element, which is
//! on the audio path and allocates nothing at all.

pub mod event;
mod gate;
mod inner;
pub mod track;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

pub use event::{EngineError, EngineEvent};
pub use track::{GrowingSource, Track, TrackSource};

/// Re-exported so an application driving a progressive download does not need a direct
/// dependency on the element crate. Minted by [`Track::growing`].
pub use profluens_elements::io::FrontierHandle;
/// Re-exported for [`Engine::with_output`] — the canonical format an engine's output must
/// render, and the configuration used by [`Engine::new`].
pub use pf_pipewire::{AudioOut, AudioOutConfig, AudioOutHandle, CanonicalFormat};

use inner::Shared;

/// A gapless track queue over one audio output.
///
/// Owns the device for its whole life. Dropping it stops everything, joins every playback
/// thread with bounded patience, and closes the output last.
pub struct Engine {
    shared: Arc<Shared>,
    /// Dropped **last**, after every playback thread is gone: closing the output kills the ring
    /// underneath any producer still attached to it.
    out: Option<AudioOut>,
    worker: Option<JoinHandle<()>>,
    worker_done: Arc<AtomicBool>,
}

impl Engine {
    /// Open the audio device and start the engine.
    ///
    /// The output is opened at the canonical format — f32, 48 kHz, stereo, half a second of ring
    /// — which every track's chain is forced to converge on. That fixed format is not a
    /// simplification but the mechanism: because the device never re-latches, track *n+1*'s
    /// producer can attach behind track *n*'s in-flight bytes.
    pub fn new() -> Result<Engine, profluens_core::error::Error> {
        Ok(Engine::with_output(AudioOut::open(AudioOutConfig::default())?))
    }

    /// Start the engine over an output the caller opened — the injection seam.
    ///
    /// Tests use it with `pf_pipewire::out::testing::open_capture`, whose virtual device is
    /// stepped by hand and records every sample; that is how gaplessness is proven as a byte
    /// comparison rather than a listening session. A future CoreAudio backend arrives the same
    /// way.
    ///
    /// The output **must** render [`CanonicalFormat::default()`]. A mismatch is not a panic and
    /// not silent: the first track to negotiate against it fails with the sink's own error, and
    /// an [`EngineEvent::Error`] is queued here immediately so the app learns why before it has
    /// even tried.
    pub fn with_output(out: AudioOut) -> Engine {
        let shared = Shared::new(out.handle());
        if out.format() != CanonicalFormat::default() {
            shared.queue_error(format!(
                "the audio output renders {:?} but every track chain converges on {:?}; \
                 no track will be able to attach",
                out.format(),
                CanonicalFormat::default()
            ));
        }
        let worker_done = Arc::new(AtomicBool::new(false));
        let worker = {
            let shared = Arc::clone(&shared);
            let done = Arc::clone(&worker_done);
            std::thread::Builder::new()
                .name("pf-engine-build".into())
                .spawn(move || {
                    shared.work();
                    done.store(true, Ordering::Release);
                })
                .ok()
        };
        Engine { shared, out: Some(out), worker, worker_done }
    }

    // --- the queue --------------------------------------------------------------------------

    /// Replace everything — the current track and the queue — with `t`.
    ///
    /// Returns immediately. The device is parked in this call, so the room goes quiet at once;
    /// the outgoing track's pipeline is torn down and the audio still queued in the output ring
    /// is dropped on the worker thread, so the new track does not begin behind half a second of
    /// the old one. Emits [`EngineEvent::TrackStarted`] when it actually starts, or
    /// [`EngineEvent::Error`] if it cannot be opened.
    pub fn play_now(&self, t: Track) {
        self.shared.play_now(t);
    }

    /// Queue `t` behind the current track — the gapless append.
    ///
    /// The track is opened and pre-rolled **immediately**: by the time the boundary arrives its
    /// pipeline is running, its decoder has filled the chain, and all that is left is to let it
    /// through. Call it the way the app already decides to (rodio's ~5 s lead); earlier is
    /// cheaper than later, because the pre-roll costs threads and pool slots but no device work
    /// at all.
    ///
    /// With nothing playing, this *is* the current track and starts at once — which is what
    /// makes an enqueue that arrives after a very short track has already ended do the obvious
    /// thing.
    ///
    /// Refuses with [`EngineError::AlreadyQueued`] if one is already queued; the queue is one
    /// deep, exactly as rodio's guard was.
    pub fn enqueue(&self, t: Track) -> Result<(), EngineError> {
        self.shared.enqueue(t)
    }

    /// Drop the queued track, if any. The current track is untouched and keeps playing.
    pub fn clear_queue(&self) {
        self.shared.clear_queue();
    }

    /// Playing + queued — rodio's `len()`.
    ///
    /// A track counts from the moment it is accepted, not from the moment its file finishes
    /// opening, so this is a stable signal to poll: it steps 2 → 1 at a boundary and reaches 0
    /// when the queue drains.
    pub fn queue_len(&self) -> usize {
        self.shared.queue_len()
    }

    /// Whether a track's pipeline is actually running (as opposed to still being opened).
    pub fn is_playing(&self) -> bool {
        self.shared.is_playing()
    }

    // --- transport --------------------------------------------------------------------------

    /// Pause playback. The device holds its buffer and renders silence, and the current track's
    /// pipeline parks — nothing is lost, and [`resume`](Self::resume) continues from exactly
    /// where it stopped. A queued track stays pre-rolled and unaffected.
    pub fn pause(&self) {
        self.shared.set_paused(true);
    }

    /// Continue from where [`pause`](Self::pause) stopped.
    pub fn resume(&self) {
        self.shared.set_paused(false);
    }

    /// Flip paused ↔ playing; returns `true` if now paused.
    pub fn toggle(&self) -> bool {
        self.shared.toggle()
    }

    pub fn is_paused(&self) -> bool {
        self.shared.is_paused()
    }

    /// Stop everything: the current track, the queue, and the audio already queued in the
    /// output ring.
    ///
    /// Returns immediately, and the room is quiet immediately — the device is parked in this
    /// call. Also clears the paused state: "stopped" and "paused" are different answers, and a
    /// stop that left the latch set would make the next track start silently. Emits no
    /// [`EngineEvent::TrackEnded`]: the application asked for this and does not need to be told,
    /// and an app that auto-advances its playlist on that event would otherwise fight its own
    /// Stop button.
    pub fn stop(&self) {
        self.shared.stop();
    }

    /// Seek the current track. A no-op when nothing is playing, or when the container offers no
    /// time→byte mapping at all (a bare ADTS stream, a still-downloading Ogg).
    ///
    /// The target is resolved through the track's seek index and the engine seeks to the
    /// **resolved cue**, never to the request — the index floors to the preceding resume-safe
    /// position, and rebasing to the request while content resumes earlier would leave the
    /// pipeline permanently behind its own clock.
    ///
    /// # Seeking to the end
    ///
    /// The target is **clamped to a quarter of a second before the end** when the track declares
    /// a duration. Seeking to (or past) the duration therefore means *play the last quarter
    /// second, then advance* — it is not an error, is not ignored, and does not skip.
    ///
    /// Without the clamp it did skip, and legitimately: an end-of-file target resolves to an
    /// end-of-file byte, the source reads nothing, and the pipeline reaches end of stream at
    /// once, so the queue advances exactly as it does at a natural boundary. What the user asked
    /// for was to hear the end of the track, so that is what this delivers. A track shorter than
    /// the epsilon clamps to zero and plays in full.
    ///
    /// The policy is the engine's, not [`SeekIndex`](profluens_core::pipeline::SeekIndex)'s,
    /// which still resolves precisely what it is asked to.
    ///
    /// Seeking during a gapless handoff is allowed and safe, with one documented consequence:
    /// the output ring is a single byte stream, so a seek issued in the instant after the
    /// previous track detached but before its tail has played out drops that tail too. The user
    /// is jumping — the previous track's last few hundred milliseconds are not what they asked
    /// to hear — and the alternative would put a branch in the real-time pull path.
    pub fn seek(&self, to: Duration) {
        self.shared.seek(to);
    }

    /// Audible position within the current track; zero when nothing is playing. See the module
    /// docs for the exact policy (delay-compensated, seek-floored, monotonic; source time for a
    /// stretched track).
    pub fn position(&self) -> Duration {
        self.shared.position()
    }

    /// The current track's duration, when its container declares one.
    pub fn duration(&self) -> Option<Duration> {
        self.shared.duration()
    }

    /// How much audio is still queued ahead of the device — the playback **runway**.
    ///
    /// Readable across a track boundary, when neither the outgoing nor the incoming producer
    /// holds the ring, which is precisely when it matters. After
    /// [`EngineEvent::TrackEnded`] this is how long the room keeps playing.
    pub fn buffered(&self) -> Duration {
        self.shared.output().buffered()
    }

    // --- master output ------------------------------------------------------------------------

    /// Master volume: linear gain, clamped to `[0.0, 4.0]`, applied in the device's render
    /// callback over a ~5 ms ramp so a change never clicks. NaN is ignored.
    ///
    /// This is the app's volume slider. Per-track ReplayGain is a different knob and lives in
    /// the track's own chain ([`Track::with_gain_db`]).
    pub fn set_volume(&self, v: f32) {
        self.shared.output().set_volume(v);
    }

    pub fn volume(&self) -> f32 {
        self.shared.output().volume()
    }

    /// Mute/unmute, ramped like the volume. Unmuting returns to the stored volume.
    pub fn set_muted(&self, m: bool) {
        self.shared.output().set_muted(m);
    }

    pub fn is_muted(&self) -> bool {
        self.shared.output().is_muted()
    }

    /// Park (or unpark) the device entirely — a real power saving when the app is not playing,
    /// and the hook a platform audio-session deactivation belongs on.
    ///
    /// **Freeze, not drain**: anything left in the ring stays there and plays when the output is
    /// unparked, and the position stops advancing. An accidental park during playback therefore
    /// stutters — recoverable and obvious — rather than silently discarding audio.
    pub fn set_idle(&self, idle: bool) {
        self.shared.output().set_idle(idle);
    }

    pub fn is_idle(&self) -> bool {
        self.shared.output().is_idle()
    }

    /// Change the current track's ReplayGain correction live, in decibels. Returns `false` if
    /// nothing is playing or the track was built without a gain stage (build it with
    /// [`Track::with_gain_db`] — `Some(0.0)` asks for the stage at unity).
    pub fn set_track_gain_db(&self, db: f32) -> bool {
        self.shared.set_track_gain_db(db)
    }

    /// The underlying output handle, for an application that wants the raw device view
    /// (measured output delay, attach state).
    pub fn output(&self) -> &AudioOutHandle {
        self.shared.output()
    }

    // --- events -------------------------------------------------------------------------------

    /// Take everything the engine has to say since the last call, in order. Meant for the app's
    /// existing tick.
    ///
    /// Also the engine's housekeeping beat: it joins the threads of tracks that have finished,
    /// so a long-running app reaps them promptly even while the build worker is idle.
    pub fn poll_events(&self) -> Vec<EngineEvent> {
        self.shared.reap();
        self.shared.drain_events()
    }
}

impl Drop for Engine {
    /// Stop everything, join every thread, then close the device — in that order, because
    /// closing it first would fail the writes of any producer still attached and turn an orderly
    /// shutdown into a burst of errors.
    ///
    /// **Bounded patience, and what happens past it.** `std` has no timed join, so each wait
    /// polls a completion flag against a deadline. Nothing is left dangling if a deadline
    /// passes, because the two things a thread can be stuck on both resolve on their own:
    ///
    /// * A **playback thread** parked on a full ring is woken by the output being dropped (which
    ///   closes the ring and fails its write), so the [`JOIN_GRACE`](inner::JOIN_GRACE) wait
    ///   after that drop is the one that collects a straggler. Its stop is also re-asserted on
    ///   every reap, covering the window in which `run()` erases a stop it was given too early.
    /// * The **build worker** may be inside a growing source's open, which waits up to ten
    ///   seconds for a download frontier that may never move. It is detached rather than waited
    ///   out; it holds only `Arc`s, its build's result is discarded by the shutdown check, and
    ///   the process can exit while it winds down.
    fn drop(&mut self) {
        // Park the device, take both slots, stop them, and close the job queue.
        self.shared.begin_shutdown();

        // The worker: bounded, then detached.
        if let Some(w) = self.worker.take() {
            if inner::wait_until(inner::WORKER_PATIENCE, || {
                self.worker_done.load(Ordering::Acquire)
            }) {
                let _ = w.join();
            }
        }

        // Playback threads, politely.
        if !self.shared.shutdown_join(inner::JOIN_PATIENCE) {
            // Closing the output wakes anything parked on a full ring by failing its write.
            drop(self.out.take());
            let _ = self.shared.shutdown_join(inner::JOIN_GRACE);
        }
        drop(self.out.take());
    }
}

// The engine is a controller object: every field is `Send + Sync` by construction (Arc'd shared
// state behind mutexes and atomics, plus handles that are all `Clone + Send + Sync`), so this is
// a compile-time assertion rather than a promise.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Engine>();
    assert_send_sync::<EngineEvent>();
    assert_send_sync::<EngineError>();
};
