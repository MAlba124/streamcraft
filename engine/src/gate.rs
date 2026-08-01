//! [`TrackSink`] — the pipewire producer sink with an engine-owned **gate** in front of it.
//!
//! The engine pre-rolls the *next* track: it builds that track's pipeline and lets it run
//! immediately, so that at the boundary there is nothing left to do but let it through. A
//! running pipeline, however, decodes — and its sink would push that audio straight into the
//! shared output the current track is still playing out of. Two things go wrong at once:
//!
//! 1. **Attach is exclusive.** `AudioOutHandle::attach` is a loud error while another producer
//!    holds the ring, and rightly so: two sinks interleaving PCM into one byte ring is garbage
//!    audio that is very hard to trace back. A pre-rolled sink that writes a byte therefore
//!    *fails its pipeline*.
//! 2. **The pause latch is shared.** `PipeWireAudioSink` in attached mode reads and writes the
//!    shared `AudioOut`'s counters — including its pause flag. So the obvious way to hold a
//!    pre-rolled pipeline back, `Pipeline::start_paused`, does the one thing it must not: the
//!    scheduler delivers `Event::Paused` to every element of the *queued* track, its sink
//!    stores `paused = true` on the **shared** playback state, and the **currently playing**
//!    track goes silent.
//!
//! The gate solves both with one boolean. While it is closed:
//!
//! * `process()` returns without consuming its input. The staged buffers stay staged, the
//!   upstream ring fills, the decoder backpressures to a halt, and the group parks on the
//!   scheduler's idle event-count — the pipeline is *fully decoded ahead and idle*, which is
//!   exactly the state a pre-roll wants. Nothing is ever written, so nothing ever attaches.
//!   (This is the same shape the sink itself uses while paused: return `Flow::Ok`, consume
//!   nothing, let the input stay staged for the first pass after the hold ends.)
//! * `Event::Paused` / `Event::Resumed` are **swallowed**, so a queued track can never move the
//!   shared device's pause latch. Every other event is forwarded unchanged.
//!
//! Opening the gate is the whole boundary: the group's next pass consumes, the sink attaches
//! behind the outgoing track's in-flight tail, and audio continues. The scheduler's idle park
//! has a 10 ms backstop tick, so the gate is noticed within 10 ms even with no explicit wake —
//! against the ~500 ms of tail already in the ring, which is the budget the handoff actually
//! has. (The engine nudges the pause transport's condvar as it opens the gate anyway, so in
//! practice the pass runs immediately.)
//!
//! # Transparency
//!
//! [`desc`](Element::desc) returns the **inner sink's** descriptor rather than one of this
//! module's own. That is deliberate: the pad name, the `audio/raw` offer menu, the
//! `SchedHint::Active` hint and the latency descriptor are exactly what the canonical chain and
//! the scheduler must see, and re-declaring them here would create a second copy to drift out
//! of sync with the first. The consequence — a dump or an introspection client names this
//! element `pipewireaudiosink` — is honest: it *is* that sink, plus a gate.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use profluens_core::batch::Inputs;
use profluens_core::clock::Clock;
use profluens_core::ctx::Ctx;
use profluens_core::element::{Element, ElementDesc, Flow};
use profluens_core::error::Error;
use profluens_core::event::Event;

use pf_pipewire::PipeWireAudioSink;

struct GateShared {
    open: AtomicBool,
    /// How many `FlushStart` events this track's sink has **finished** handling.
    ///
    /// The engine needs this because a ring flush is *deferred*: `Producer::flush` publishes a
    /// flush point and bumps a generation, and the real-time consumer drops everything up to it
    /// on its next pull. So a flush is invisible in `AudioOutHandle::buffered` until the device
    /// renders again — and when the engine flushes, it has deliberately parked the device first,
    /// so the device will *not* render again until a new track is activated.
    ///
    /// That leaves no way to know the flush actually reached the sink before tearing its
    /// pipeline down — and tearing it down first hands the producer back unflushed, leaving the
    /// abandoned track's audio in the ring to be played as the *next* track's first half second.
    /// Counting the events here is that proof, at a cost of one relaxed store per seek.
    flushes: AtomicU64,
    /// How many passes have arrived at this sink carrying audio — **whether or not the gate let
    /// them through**.
    ///
    /// This is what "pre-rolled" actually means, and the engine waits for it before announcing a
    /// track. A pipeline that has merely been *spawned* is not ready for anything: `run()` is
    /// still linking, negotiating and starting elements on its own thread, and a seek issued in
    /// that window is either silently dropped (the group has not yet sampled the seek
    /// generation) or lands on a decoder that has not read its header yet and kills the track.
    /// One relaxed store per pass buys the invariant that a track the engine calls playing can
    /// survive the very next call made against it.
    staged: AtomicU64,
}

/// The engine's half of a track's gate. Cloneable, and safe to drive from any thread.
#[derive(Clone)]
pub(crate) struct Gate(Arc<GateShared>);

impl Gate {
    pub(crate) fn closed() -> Gate {
        Gate(Arc::new(GateShared {
            open: AtomicBool::new(false),
            flushes: AtomicU64::new(0),
            staged: AtomicU64::new(0),
        }))
    }

    /// How many flushes this track's sink has completed. Compare a reading taken before a seek
    /// with later readings to know the flush has landed.
    pub(crate) fn flushes(&self) -> u64 {
        self.0.flushes.load(Ordering::Acquire)
    }

    /// Whether audio has reached this track's sink yet — the real "pre-rolled" test. See
    /// `GateShared::staged`.
    pub(crate) fn has_staged(&self) -> bool {
        self.0.staged.load(Ordering::Acquire) > 0
    }

    /// Let the track through. `Release` pairs with the sink thread's `Acquire`, so the pass that
    /// observes the open gate also observes everything the engine published before opening it.
    pub(crate) fn open(&self) {
        self.0.open.store(true, Ordering::Release);
    }

    /// Hold the track back. Used to pre-roll, and again to silence a track being retired: a
    /// closed gate stops it feeding the ring within one pass, without waiting for the pipeline
    /// to actually stop.
    pub(crate) fn close(&self) {
        self.0.open.store(false, Ordering::Release);
    }

    pub(crate) fn is_open(&self) -> bool {
        self.0.open.load(Ordering::Acquire)
    }

    /// Record that a `FlushStart` has been fully handled by the sink.
    fn note_flush(&self) {
        self.0.flushes.fetch_add(1, Ordering::Release);
    }

    /// Record that a pass carrying audio reached the sink. See `GateShared::staged`.
    fn note_staged(&self) {
        self.0.staged.fetch_add(1, Ordering::Release);
    }
}

/// A [`PipeWireAudioSink`] that only runs while its [`Gate`] is open.
pub(crate) struct TrackSink {
    inner: PipeWireAudioSink,
    gate: Gate,
}

impl TrackSink {
    pub(crate) fn new(inner: PipeWireAudioSink, gate: Gate) -> TrackSink {
        TrackSink { inner, gate }
    }
}

impl Element for TrackSink {
    fn desc(&self) -> &'static ElementDesc {
        self.inner.desc()
    }

    fn preroll(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        self.inner.preroll(ctx)
    }

    fn provide_clock(&mut self) -> Option<Arc<dyn Clock>> {
        self.inner.provide_clock()
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        self.inner.start(ctx)
    }

    fn process(&mut self, ctx: &mut Ctx, inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !inputs.is_empty() {
            // Before the gate, deliberately: a pre-rolled track is held back precisely so that
            // it *has* decoded ahead, and the engine's readiness wait must see that. Cheap
            // (`is_empty` consumes nothing) and off the sample path.
            self.gate.note_staged();
        }
        if !self.gate.is_open() {
            // Consume nothing: the input stays staged and is picked up by the first pass after
            // the gate opens, exactly as the sink's own pause path does. Returning here is what
            // keeps a pre-rolled track from attaching to an output another track owns.
            return Ok(Flow::Ok);
        }
        self.inner.process(ctx, inputs)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::Paused | Event::Resumed) && !self.gate.is_open() {
            // The pause latch behind these lives on the *shared* output. A track that is not
            // the audible one must not touch it — see the module docs. The engine drives the
            // device's pause state directly and re-applies it when this track is activated, so
            // nothing is lost by dropping these here.
            return Ok(());
        }
        let handled = self.inner.event(ctx, event);
        if matches!(event, Event::FlushStart) {
            // Published *after* the inner sink has run its flush, so an engine that observes the
            // count has observed `Producer::flush` — see `GateShared::flushes`.
            self.gate.note_flush();
        }
        handled
    }

    fn stop(&mut self, ctx: &mut Ctx) {
        // Always forwarded: the inner sink's `stop` hands the ring's producer end back, and
        // failing to do that would leak the application's audio output for good.
        self.inner.stop(ctx);
    }
}
