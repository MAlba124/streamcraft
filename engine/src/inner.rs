//! The state machine: two slots, one build worker, one thread per playing track.
//!
//! Everything that can change the engine's mind lives in this file, on purpose — the whole
//! value of the design is that the transitions are auditable in one place.
//!
//! # The state
//!
//! Two slots, and nothing else:
//!
//! | | slot 0 — **current** | slot 1 — **queued** |
//! |---|---|---|
//! | `Empty` | nothing is playing | nothing is queued |
//! | `Building{seq}` | a build is in flight, destined to become current | …to become the next track |
//! | `Live` | a pipeline is running, gate **open** | a pipeline is running, gate **closed** (pre-rolled) |
//!
//! [`Engine::queue_len`](crate::Engine::queue_len) is simply the number of non-`Empty` slots,
//! which is what makes it the app's transition signal exactly as rodio's `len()` was: it counts
//! a track from the moment it is accepted, not from the moment its file finishes opening.
//!
//! **Invariant:** slot 0 `Empty` ⟹ slot 1 `Empty`. Every path that empties slot 0 shifts slot 1
//! down in the same critical section (`advance_locked`), so the queue can never hold a
//! track with nothing in front of it.
//!
//! **A slot index is not a build's identity** — the sequence number is. `advance_locked` shifts a
//! `Building` marker from slot 1 to slot 0, so a build in flight can change slots while it is
//! opening; everything that looks a build up does so with [`State::awaiting`]. See its docs for
//! what going by the index instead cost.
//!
//! # The transitions
//!
//! ```text
//!   play_now ──► [park device] ─► retire both slots (flush ring) ─► slot0 := Building
//!   enqueue  ──► slot1 := Building   (or slot0, when the engine is idle)
//!   build ok ──► install: slot := Live;  slot 0 also ⇒ activate + TrackStarted/TrackChanged
//!   build err ─► slot := Empty + Error;  slot 0 also ⇒ advance
//!   run() ends ► bury; slot 0 ⇒ advance (slot1 shifts down; Live ⇒ activate + TrackChanged,
//!                                        Building ⇒ re-labelled Changed, Empty ⇒ TrackEnded)
//!   stop     ──► [park device] ─► retire both slots (flush ring), no event
//! ```
//!
//! *Activate* is the boundary, and it is three stores: open the gate, put the device's pause
//! latch where the user's pause state says it belongs, and nudge the pause transport so the
//! track's idle-parked sink group runs its next pass immediately.
//!
//! # Threads, and the rules that keep them honest
//!
//! * One **worker**. It owns every blocking operation: `Player::open_canonical` (probe, head
//!   walk, seek index, preroll — all app-side IO) and the retire path's bounded flush wait.
//!   Public API methods only ever *post* to it, so they never block on a disk.
//! * One **runner per track**, blocked in `Pipeline::run()`. When `run()` returns, that thread
//!   performs the boundary itself: there is no supervisor to wake, so the gapless advance is not
//!   queued behind whatever the worker happens to be doing.
//! * The runner also drains its pipeline's bus, because it is the only thread that can:
//!   `Pipeline` is `!Sync` and `bus()` borrows it.
//!
//! Rules, held to throughout:
//!
//! 1. **Never join a thread while holding a lock.** Finished runners go to the
//!    graveyard and are joined by `Shared::reap`, always outside every lock.
//! 2. **Lock order is `state` → (`events` | `graveyard`), never the reverse.** `jobs` is never
//!    held together with any other lock.
//! 3. **Only atomics and handle pokes happen under `state`.** Every call made while it is held
//!    (`PauseHandle`, `StopHandle`, `SeekHandle`, `AudioControl`, `AudioOutHandle`) is a handful
//!    of atomic stores and at most a condvar notify.
//!
//! # Two traps in the substrate this file is built around
//!
//! * **`StopHandle::stop()` issued before `run()` starts is erased**, because `run()`'s first
//!   statement clears the stop flag. A track can be retired before its runner has got that far.
//!   Two defences: the runner checks a `cancel` flag *instead of* calling `run()` at all, and
//!   `Shared::reap` re-issues the stop on every pass for as long as a corpse is unburied — so
//!   a stop lost to that window is re-applied within milliseconds.
//! * **`run()` applies `start_paused` after the thread is spawned**, so a resume issued in that
//!   window would be erased and the track would never play. Nothing here uses `start_paused`;
//!   the gate element is the pre-roll hold, and a *pause* (which `run()` never clears) is
//!   the only transport state ever pre-applied.

// Control-plane module: labels, event queues, the job list and the graveyard are ordinary heap
// collections, and one boxed sink + one boxed closure are built per track at open time. None of
// it is on a media path or inside `process()` — the documented `clippy.toml` exception.
#![allow(clippy::disallowed_methods)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pf_pipewire::{AudioControl, AudioOutHandle, PipeWireAudioSink};
use pf_play::chain::ChainHandles;
use pf_play::{ChainSpec, Player, SinkChoice, SinkSpec};
use profluens_core::bus::BusMessage;
use profluens_core::error::Error;
use profluens_core::id::ElementId;
use profluens_core::pipeline::{PauseHandle, SeekHandle, SeekIndex, StopHandle};
use profluens_core::props::PropHandle;
use profluens_core::time::Timestamp;

use crate::gate::{Gate, TrackSink};
use crate::{EngineError, EngineEvent, Track};

// --- tuning ------------------------------------------------------------------------------

/// How many undelivered events the engine keeps before it starts dropping the oldest.
///
/// An application that polls (the whole contract) never approaches this; one that stops polling
/// entirely is already broken, and an unbounded queue would turn that bug into an OOM. Dropping
/// the *oldest* keeps the most recent view of reality, which is what a UI would redraw from.
const EVENT_CAP: usize = 1024;

/// How long the retire path waits for a discontinuity's ring flush to reach the sink before
/// giving up and stopping the pipeline anyway.
///
/// The flush is delivered on the sink group's very next pass, so this is orders of magnitude
/// more than it needs; it exists so that a pipeline which has already died (or was stopped
/// before it ever ran) cannot stall the worker. Overshooting costs at worst the previous track's
/// buffered tail surviving into the next one — never a hang.
const FLUSH_SETTLE: Duration = Duration::from_millis(250);

/// How long the retire path lets the device run to *apply* a published flush before parking it
/// again. One device block is enough; this only has to cover a device that is being scheduled
/// unusually slowly. A device that is not running at all (parked idle, or a test that has
/// stopped stepping) simply times out, and the consequence is bounded — see the retire path.
const FLUSH_DRAIN: Duration = Duration::from_millis(150);

/// How long the retire path waits for a stopped track to actually let go of the audio output.
///
/// Two seconds is far beyond the measured cost of unwinding a pipeline (core documents ~10 ms
/// from `stop()` to `run()` returning when idle, ~60 ms when paused); it is a bound, not an
/// expectation. Exceeding it means the next track's first write may fail the exclusive attach,
/// which surfaces as an `EngineEvent::Error` rather than as silence.
const RELEASE_PATIENCE: Duration = Duration::from_secs(2);

/// Poll interval for the bounded waits in this file.
const POLL: Duration = Duration::from_millis(2);

/// How long [`Shared::shutdown_join`] waits for playback threads before it stops waiting
/// politely and closes the output underneath them.
pub(crate) const JOIN_PATIENCE: Duration = Duration::from_secs(2);

/// How long it waits *after* the output is closed — which fails any parked write immediately,
/// so this only has to cover an unwind.
pub(crate) const JOIN_GRACE: Duration = Duration::from_millis(500);

/// How long [`Engine::drop`](crate::Engine) waits for the build worker to finish what it is
/// doing. The bound that matters is a growing source's open, which waits up to
/// `pf_play::source::OPEN_TIMEOUT` (10 s) for a download frontier that may never move; the
/// worker is detached rather than waited out, and its result is discarded by the shutdown check
/// in [`Shared::install`].
pub(crate) const WORKER_PATIENCE: Duration = Duration::from_secs(3);

/// How close to the end of a track a seek is allowed to land — the **near-end clamp** (see
/// [`Shared::seek`]).
///
/// Long enough that the landing is unmistakably audible rather than a click, short enough that
/// "seek to the end" still means the end. A quarter of a second is roughly the shortest interval
/// a listener reliably hears as a piece of music rather than an artefact, and it is half the
/// output ring, so the clamped tail is typically already buffered when the boundary is
/// announced.
pub(crate) const TAIL_EPSILON: Duration = Duration::from_millis(250);

// --- the pieces --------------------------------------------------------------------------

/// Which event a slot-0 install should announce.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Announce {
    /// A cold or explicit start: the first `enqueue` into an idle engine, or a `play_now`.
    Started,
    /// The queue advanced onto this track while it was still building.
    Changed,
}

/// What retiring a track should do about the audio still queued in the output ring.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Retire {
    /// Leave the ring alone. For a queued track (which never attached, so none of the ring is
    /// its) and for a track being discarded during shutdown.
    Quiet,
    /// Drop what is in the ring. Only ever asked of the *attached* track, and only for a
    /// deliberate discontinuity — see [`Shared::retire`].
    FlushRing,
}

/// A track with a pipeline actually running.
struct Live {
    seq: u64,
    label: String,
    gate: Gate,
    pause: PauseHandle,
    stop: StopHandle,
    seek: SeekHandle,
    props: PropHandle,
    /// The `audiogain` stage, when this track asked for one.
    gain: Option<ElementId>,
    /// Source-time position, when this track has an `audiostretch` stage. Boxed rather than
    /// typed so this crate does not have to depend on `profluens-audio` just to name
    /// `StretchPosition`.
    stretch_pos: Option<Box<dyn Fn() -> Timestamp + Send + Sync>>,
    index: SeekIndex,
    duration: Option<Timestamp>,
    /// Latched once this track's producer has actually claimed the output. Until then the
    /// shared position still belongs to the *previous* epoch, and reporting it would show the
    /// outgoing track's clock under the incoming track's name for the width of the handoff.
    heard: AtomicBool,
    /// Set before [`StopHandle::stop`], and checked by the runner *instead of* calling `run()`
    /// — the defence against a stop issued before `run()` has cleared its own flag.
    cancel: Arc<AtomicBool>,
    /// Set by the runner as its last act. The graveyard joins only what has set this.
    done: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

/// A retired track, waiting to be joined.
struct Corpse {
    stop: StopHandle,
    cancel: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

impl Corpse {
    /// Re-assert the stop. See the module docs: a stop that landed in the window before
    /// `run()` cleared the flag was erased, and this is what un-erases it.
    fn nudge(&self) {
        self.cancel.store(true, Ordering::Release);
        self.stop.stop();
    }
}

/// One of the engine's two queue positions.
enum Slot {
    Empty,
    Building { seq: u64, announce: Announce },
    Live(Box<Live>),
}

impl Slot {
    fn is_empty(&self) -> bool {
        matches!(self, Slot::Empty)
    }

    /// Empty this slot, returning whatever was running in it. A `Building` marker is simply
    /// dropped — the build in flight discovers it was superseded when it tries to install.
    fn take(&mut self) -> Option<Box<Live>> {
        match std::mem::replace(self, Slot::Empty) {
            Slot::Live(l) => Some(l),
            _ => None,
        }
    }
}

/// The mutex-protected heart.
struct State {
    slots: [Slot; 2],
    /// The app's pause intent. Also mirrored into [`Shared::user_paused`] for lock-free reads;
    /// this copy is the authoritative one, written only under the lock so that a pause landing
    /// concurrently with a track boundary cannot be lost by either.
    paused: bool,
}

impl State {
    /// Empty both slots, returning what was running in each with its slot index.
    fn take_all(&mut self) -> Vec<(usize, Box<Live>)> {
        let mut out = Vec::with_capacity(2);
        for i in 0..2 {
            if let Some(l) = self.slots[i].take() {
                out.push((i, l));
            }
        }
        out
    }

    /// Which slot holds a running track with this sequence number.
    fn slot_of(&self, seq: u64) -> Option<usize> {
        (0..2).find(|&i| matches!(&self.slots[i], Slot::Live(l) if l.seq == seq))
    }

    /// Which slot is waiting on the build with this sequence number, and how that slot wants it
    /// announced.
    ///
    /// **The sequence number is a build's identity; the slot index is not.** A queued track
    /// whose build is still in flight is *moved* into slot 0 by [`Shared::advance_locked`] when
    /// the current track ends — that is exactly what re-labelling a `Building` marker
    /// `Announce::Changed` there is for. A build that looked itself up by the index it was
    /// posted with therefore found a slot that had moved on, concluded it had been superseded,
    /// and threw itself away — leaving slot 0 marked `Building` for ever, with `queue_len()`
    /// stuck at 1 and neither `TrackChanged` nor `TrackEnded` ever emitted again. Looking up by
    /// `seq` is what lets the build follow its track.
    fn awaiting(&self, seq: u64) -> Option<(usize, Announce)> {
        (0..2).find_map(|i| match &self.slots[i] {
            Slot::Building { seq: s, announce } if *s == seq => Some((i, *announce)),
            _ => None,
        })
    }
}

/// Work posted to the build worker.
enum Job {
    /// Open `track` for whichever slot is holding `seq` **when the build finishes** — see
    /// [`State::awaiting`]; the slot can change underneath a build in flight.
    Build { seq: u64, track: Track },
    Retire { live: Box<Live>, mode: Retire },
}

struct JobQueue {
    queue: VecDeque<Job>,
    closed: bool,
}

// --- the shared object -------------------------------------------------------------------

pub(crate) struct Shared {
    out: AudioOutHandle,
    /// Direct control of the *shared device's* pause latch, independent of any pipeline.
    ///
    /// Obtained from a throwaway `PipeWireAudioSink::with_output`, which is the only public
    /// route to it — the handle it hands out is on the output's counters, not the element's, so
    /// the element itself is dropped immediately (and, never having attached, drops inertly).
    /// This is what makes silencing the room a single relaxed store, available even when no
    /// track exists.
    audio: AudioControl,
    state: Mutex<State>,
    events: Mutex<VecDeque<EngineEvent>>,
    graveyard: Mutex<Vec<Corpse>>,
    jobs: Mutex<JobQueue>,
    job_cv: Condvar,
    /// Lock-free mirror of `State::paused`, for [`Engine::is_paused`](crate::Engine::is_paused).
    user_paused: AtomicBool,
    shutdown: AtomicBool,
    next_seq: AtomicU64,
}

impl Shared {
    pub(crate) fn new(out: AudioOutHandle) -> Arc<Shared> {
        // A sink built only to borrow the shared output's control handle; it never reaches a
        // pipeline, never attaches, and its `Drop` is a no-op in that state.
        let audio = PipeWireAudioSink::with_output(out.clone()).control();
        Arc::new(Shared {
            out,
            audio,
            state: Mutex::new(State {
                slots: [Slot::Empty, Slot::Empty],
                paused: false,
            }),
            events: Mutex::new(VecDeque::new()),
            graveyard: Mutex::new(Vec::new()),
            jobs: Mutex::new(JobQueue { queue: VecDeque::new(), closed: false }),
            job_cv: Condvar::new(),
            user_paused: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            next_seq: AtomicU64::new(1),
        })
    }

    pub(crate) fn output(&self) -> &AudioOutHandle {
        &self.out
    }

    fn seq(&self) -> u64 {
        self.next_seq.fetch_add(1, Ordering::Relaxed)
    }

    // --- events ---------------------------------------------------------------------------

    fn emit(&self, ev: EngineEvent) {
        let mut q = self.events.lock().unwrap_or_else(|e| e.into_inner());
        if q.len() >= EVENT_CAP {
            q.pop_front();
        }
        q.push_back(ev);
    }

    fn emit_duration(&self, d: Option<Timestamp>) {
        if let Some(ns) = d.and_then(|t| t.nanos()) {
            self.emit(EngineEvent::DurationKnown(Duration::from_nanos(ns)));
        }
    }

    pub(crate) fn drain_events(&self) -> Vec<EngineEvent> {
        let mut q = self.events.lock().unwrap_or_else(|e| e.into_inner());
        q.drain(..).collect()
    }

    /// Queue an error the engine noticed outside any track's lifecycle (a misconfigured
    /// output). Public to the crate so construction can report before a track exists.
    pub(crate) fn queue_error(&self, message: String) {
        self.emit(EngineEvent::Error { message });
    }

    // --- the job worker -------------------------------------------------------------------

    fn post(&self, job: Job) {
        let mut q = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        if q.closed {
            return; // shutting down: the job would never run, and dropping it retires nothing
        }
        q.queue.push_back(job);
        drop(q);
        self.job_cv.notify_one();
    }

    pub(crate) fn close_jobs(&self) {
        let mut q = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        q.closed = true;
        drop(q);
        self.job_cv.notify_all();
    }

    /// The build worker's whole life: take a job, do it, reap. Exits when the queue is closed
    /// and drained.
    pub(crate) fn work(self: &Arc<Self>) {
        loop {
            let job = {
                let mut q = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if let Some(j) = q.queue.pop_front() {
                        break Some(j);
                    }
                    if q.closed {
                        break None;
                    }
                    q = self.job_cv.wait(q).unwrap_or_else(|e| e.into_inner());
                }
            };
            let Some(job) = job else { break };
            match job {
                Job::Retire { live, mode } => self.retire(*live, mode),
                Job::Build { seq, track } => {
                    // Skip a build no slot is waiting for any more. This is what makes a burst
                    // of `play_now` calls cheap: every superseded job costs one atomic-free
                    // slot read instead of a full probe + head walk + preroll.
                    if self.still_wanted(seq) {
                        match self.build(seq, track) {
                            Ok(live) => self.install(seq, live),
                            Err(msg) => self.build_failed(seq, msg),
                        }
                    }
                }
            }
            self.reap();
        }
        self.reap();
    }

    /// Is any slot still waiting for this build? Checked before paying for the open.
    fn still_wanted(&self, seq: u64) -> bool {
        if self.shutdown.load(Ordering::Acquire) {
            return false;
        }
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.awaiting(seq).is_some()
    }

    // --- building -------------------------------------------------------------------------

    /// Open a track and start its pipeline, gate closed. Blocking — worker thread only.
    ///
    /// The returned track is *running and pre-rolled*: its decoder has filled the chain up to
    /// the gate and its groups are parked. Installing it into slot 0 then costs only
    /// [`Shared::activate`], which is why a boundary is measured in microseconds against a
    /// half-second of tail.
    fn build(self: &Arc<Self>, seq: u64, track: Track) -> Result<Box<Live>, String> {
        let label = track.label();
        let Track { source, gain_db, stretch } = track;
        let spec = source.into_spec()?;

        let gate = Gate::closed();
        let sink = TrackSink::new(PipeWireAudioSink::with_output(self.out.clone()), gate.clone());
        let chain = ChainSpec { gain_db, stretch, sink: SinkSpec::Injected(Box::new(sink)) };
        // Video goes nowhere: this is a music engine, and a video track in the container must
        // not grab a window (or a decoder) on its way past.
        let mut player = Player::open_canonical(spec, chain, SinkChoice::Drop)
            .map_err(|e| format!("{label}: {e}"))?;

        // `audio_chain()` is `Some` exactly when the canonical audio chain was built, which is
        // a stricter and more useful predicate than `any_track_linked()`: a video-only file
        // links its video to the drop sink and would otherwise pass as playable, then play
        // silence.
        let Some(handles) = player.audio_chain() else {
            return Err(format!("{label}: no playable audio track"));
        };
        let gain = handles.gain;
        let stretch_pos = stretch_reader(handles);
        let index = player.seek_index().clone();
        let duration = player.duration();
        let pause = player.pipeline.pause_handle();
        let stop = player.pipeline.stop_handle();
        let seek = player.pipeline.seek_handle();
        let props = player.pipeline.prop_handle();
        // Explicitly *not* start-paused: see the module docs. The gate is the hold.
        player.pipeline.start_paused(false);

        let cancel = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let runner_shared = Arc::clone(self);
        let runner_cancel = Arc::clone(&cancel);
        let runner_done = Arc::clone(&done);
        let thread = std::thread::Builder::new()
            .name(format!("pf-engine-track-{seq}"))
            .spawn(move || {
                // A track retired before its thread got here never runs at all — the stop it
                // was given would have been erased by `run()`'s own flag reset.
                let result = if runner_cancel.load(Ordering::Acquire) {
                    Ok(())
                } else {
                    player.run()
                };
                let errors = drain_bus(&player);
                // Tear the pipeline down before announcing: the sink hands the producer back in
                // `stop()` (and, on an unclean unwind, in `Drop`), and the next track must not
                // race that.
                drop(player);
                runner_shared.on_finished(seq, result, errors);
                runner_done.store(true, Ordering::Release);
            })
            .map_err(|e| format!("{label}: cannot spawn a playback thread: {e}"))?;

        Ok(Box::new(Live {
            seq,
            label,
            gate,
            pause,
            stop,
            seek,
            props,
            gain,
            stretch_pos,
            index,
            duration,
            heard: AtomicBool::new(false),
            cancel,
            done,
            thread: Some(thread),
        }))
    }

    /// Place a freshly built track in whichever slot is still waiting for it, or throw it away
    /// if nothing is any more. The slot is looked up by sequence number, not remembered from the
    /// post — the queue can have advanced onto this very build while it was opening (see
    /// [`State::awaiting`]).
    fn install(self: &Arc<Self>, seq: u64, live: Box<Live>) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        // Superseded while we were opening it, or the engine is going away: throw it out.
        let Some((slot, announce)) =
            st.awaiting(seq).filter(|_| !self.shutdown.load(Ordering::Acquire))
        else {
            drop(st);
            self.retire(*live, Retire::Quiet);
            return;
        };
        if slot == 0 {
            self.activate(&live, st.paused);
            self.emit(match announce {
                Announce::Started => EngineEvent::TrackStarted,
                Announce::Changed => EngineEvent::TrackChanged,
            });
            self.emit_duration(live.duration);
        }
        st.slots[slot] = Slot::Live(live);
    }

    fn build_failed(self: &Arc<Self>, seq: u64, msg: String) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let ours = st.awaiting(seq);
        self.emit(EngineEvent::Error { message: msg });
        let Some((slot, _)) = ours else {
            return; // superseded: no slot is waiting for this build any more
        };
        st.slots[slot] = Slot::Empty;
        if slot == 0 {
            // The track that was to become current never existed. Promote the queue rather than
            // stall — a pre-rolled pipeline with nothing in front of it would sit gated for
            // ever, and the invariant "slot 0 empty ⟹ slot 1 empty" would be broken.
            self.advance_locked(&mut st);
        }
    }

    // --- the boundary ---------------------------------------------------------------------

    /// Make `live` the audible track. Three stores; called with `state` held.
    ///
    /// `paused` is passed by value rather than read from the state, so this can be called from
    /// inside a `&mut` borrow of the slot it is activating.
    fn activate(&self, live: &Live, paused: bool) {
        live.gate.open();
        if paused {
            self.audio.pause();
            live.pause.pause();
        } else {
            // Clears a device park left by a previous `stop()`/`play_now()`.
            self.audio.resume();
            // Doubles as the wake: `resume` notifies the pause transport's condvar, which is
            // what this track's idle-parked sink group is waiting on — so the pass that acts on
            // the open gate runs now rather than at the scheduler's 10 ms backstop tick.
            live.pause.resume();
        }
    }

    /// Slot 0 is empty: shift the queue down and announce whatever lands there.
    fn advance_locked(self: &Arc<Self>, st: &mut State) {
        debug_assert!(st.slots[0].is_empty(), "advance with a live current track");
        let paused = st.paused;
        st.slots[0] = std::mem::replace(&mut st.slots[1], Slot::Empty);
        match &mut st.slots[0] {
            Slot::Live(next) => {
                self.activate(next, paused);
                let duration = next.duration;
                self.emit(EngineEvent::TrackChanged);
                self.emit_duration(duration);
            }
            // Still opening. It will be announced as a change, not a start, when it lands.
            Slot::Building { announce, .. } => *announce = Announce::Changed,
            Slot::Empty => self.emit(EngineEvent::TrackEnded),
        }
    }

    /// A runner's `run()` returned.
    fn on_finished(self: &Arc<Self>, seq: u64, result: Result<(), Error>, errors: Vec<String>) {
        for message in errors {
            self.emit(EngineEvent::Error { message });
        }
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let slot = st.slot_of(seq);
        if let Err(e) = result {
            let label = slot
                .and_then(|i| match &st.slots[i] {
                    Slot::Live(l) => Some(l.label.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            self.emit(EngineEvent::Error { message: format!("{label}: playback failed: {e:?}") });
        }
        // Not in either slot: this track was already retired (stopped, replaced, cleared), and
        // its teardown is somebody else's business.
        let Some(slot) = slot else { return };
        if self.shutdown.load(Ordering::Acquire) {
            return;
        }
        let Some(live) = st.slots[slot].take() else { return };
        self.bury(*live);
        if slot == 1 {
            // A queued track finished while still gated: it decoded to nothing at all, so its
            // group reached quiescence with no staged input for the gate to hold back. Drop it
            // — the current track is untouched.
            return;
        }
        self.advance_locked(&mut st);
    }

    // --- retirement -----------------------------------------------------------------------

    /// Tear a track down. Worker thread (or shutdown) only — the flush arm waits.
    fn retire(&self, live: Live, mode: Retire) {
        // First and always: stop it feeding the ring. One store, acted on within a pass.
        live.gate.close();
        if mode == Retire::FlushRing {
            // The caller has already parked the device, so nothing of this track can still be
            // *heard*. What remains is the audio queued in the ring — up to half a second of a
            // track the user just left, which would otherwise be the first thing they hear of
            // the next one. Only the attached producer can drop it, and a seek is how it is
            // asked to: `FlushStart` makes the sink call `Producer::flush`.
            //
            // That flush is deferred by design — it publishes a flush point the real-time
            // consumer honours on its next pull — so it cannot be confirmed by watching
            // `buffered()`, least of all while the device is parked and not pulling at all.
            // What is confirmed is that the sink *handled* the event, which is exactly when the
            // flush point became published; from then on the bytes are unreachable whenever the
            // device next runs. Waiting for it before stopping is the whole point: a pipeline
            // torn down first would hand the producer back with the flush never issued.
            //
            // Nothing refills the ring meanwhile: the gate is shut, so the re-primed pipeline's
            // sink consumes nothing.
            let before = live.gate.flushes();
            live.seek.seek(0, Timestamp::ZERO);
            if wait_until(FLUSH_SETTLE, || live.gate.flushes() > before) {
                // The flush point is published, so the only thing the device can pull now is
                // the silence past it — the gate is shut on this track, the queued track's gate
                // was shut before this job was posted, and the worker is serial, so nothing can
                // write to the ring until this returns. Let it run just long enough to apply it.
                //
                // Skipping this would be a subtle, permanent bug rather than a cosmetic one:
                // until the consumer's next pull, the dropped bytes still count toward
                // `Producer::len`, and that is precisely what `AudioOutHandle::attach` measures
                // the *next* track's position epoch against. Every subsequent track would report
                // a position behind by the length of the tail nobody ever heard.
                //
                // Only after a confirmed flush: unparking with the stale audio still reachable
                // would play the very thing the park was there to prevent.
                self.audio.resume();
                wait_until(FLUSH_DRAIN, || self.out.buffered().is_zero());
                self.audio.pause();
            }
        }
        live.cancel.store(true, Ordering::Release);
        live.stop.stop();
        // Do not return until this track has actually let go of the output.
        //
        // `StopHandle::stop` is asynchronous: it wakes the pipeline's groups, which then unwind
        // and call `stop()` on the sink, and *that* is where the producer end goes back. Attach
        // is exclusive, so a next track built and activated in the meantime would fail its very
        // first write — which is not a glitch but a dead track, and an intermittent one, because
        // it is a race between a teardown and a file open. The worker is serial, so waiting here
        // is exactly what orders the two; no public method is blocked by it.
        //
        // Either signal will do: the output being free is the precondition we actually need, and
        // the runner having finished implies it. (Retiring a *queued* track while the current one
        // plays only ever satisfies the second — the current track is legitimately still
        // attached.)
        if !self.shutdown.load(Ordering::Acquire) {
            wait_until(RELEASE_PATIENCE, || {
                live.done.load(Ordering::Acquire) || !self.out.is_attached()
            });
        }
        self.bury(live);
    }

    /// Hand a finished-or-doomed track to the graveyard for joining.
    fn bury(&self, mut live: Live) {
        let Some(thread) = live.thread.take() else { return };
        let mut g = self.graveyard.lock().unwrap_or_else(|e| e.into_inner());
        g.push(Corpse {
            stop: live.stop.clone(),
            cancel: Arc::clone(&live.cancel),
            done: Arc::clone(&live.done),
            thread,
        });
    }

    /// Join every playback thread that has finished, and re-assert the stop on the rest.
    ///
    /// Called from the worker after every job and from
    /// [`poll_events`](crate::Engine::poll_events) — the app's own tick — so a retired track is
    /// reaped promptly even when the worker is idle. Never called with a lock held.
    pub(crate) fn reap(&self) {
        let ready = {
            let mut g = self.graveyard.lock().unwrap_or_else(|e| e.into_inner());
            let mut ready = Vec::new();
            let mut i = 0;
            while i < g.len() {
                if g[i].done.load(Ordering::Acquire) {
                    ready.push(g.swap_remove(i));
                } else {
                    g[i].nudge();
                    i += 1;
                }
            }
            ready
        };
        for c in ready {
            let _ = c.thread.join();
        }
    }

    /// How many playback threads are still unburied.
    pub(crate) fn graveyard_len(&self) -> usize {
        self.graveyard.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    // --- the transport, as the public API calls it ------------------------------------------

    pub(crate) fn play_now(self: &Arc<Self>, track: Track) {
        // Silence the room *now*: one relaxed store the render callback reads on its next
        // block. Everything after this is bookkeeping the user cannot hear.
        self.audio.pause();
        let seq = self.seq();
        let taken = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let taken = st.take_all();
            st.slots[0] = Slot::Building { seq, announce: Announce::Started };
            taken
        };
        self.retire_all(taken);
        self.post(Job::Build { seq, track });
    }

    pub(crate) fn enqueue(self: &Arc<Self>, track: Track) -> Result<(), EngineError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(EngineError::Shutdown);
        }
        let seq = self.seq();
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if !st.slots[1].is_empty() {
                return Err(EngineError::AlreadyQueued);
            }
            // Nothing playing: this *is* the current track. That is what makes an enqueue after
            // a short track has already ended start playing immediately, instead of queueing
            // behind nothing.
            let slot = usize::from(!st.slots[0].is_empty());
            st.slots[slot] = Slot::Building { seq, announce: Announce::Started };
        }
        self.post(Job::Build { seq, track });
        Ok(())
    }

    pub(crate) fn clear_queue(self: &Arc<Self>) {
        let taken = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            st.slots[1].take()
        };
        if let Some(live) = taken {
            live.gate.close();
            // `Quiet`: a queued track never attached, so none of the ring is its — flushing
            // would destroy the *current* track's audio.
            self.post(Job::Retire { live, mode: Retire::Quiet });
        }
    }

    pub(crate) fn stop(self: &Arc<Self>) {
        self.audio.pause();
        let taken = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            // Stop is a full reset, pause included: "stopped" and "paused" are different
            // answers, and leaving the latch set would make the next track start silently.
            st.paused = false;
            self.user_paused.store(false, Ordering::Release);
            st.take_all()
        };
        self.retire_all(taken);
    }

    /// Post a retire for each track taken out of the slots. Slot 0's track owns the ring; the
    /// queued one does not.
    fn retire_all(&self, taken: Vec<(usize, Box<Live>)>) {
        for (slot, live) in taken {
            live.gate.close();
            let mode = if slot == 0 { Retire::FlushRing } else { Retire::Quiet };
            self.post(Job::Retire { live, mode });
        }
    }

    pub(crate) fn set_paused(&self, paused: bool) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.apply_paused(&mut st, paused);
    }

    pub(crate) fn toggle(&self) -> bool {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let want = !st.paused;
        self.apply_paused(&mut st, want);
        want
    }

    /// The pause state is written **under `state`**, so a pause landing in the same instant as a
    /// track boundary is either fully before it (the incoming track is activated paused) or
    /// fully after it (the incoming track is paused by this call). It cannot be lost between.
    fn apply_paused(&self, st: &mut State, paused: bool) {
        st.paused = paused;
        self.user_paused.store(paused, Ordering::Release);
        if paused {
            self.audio.pause();
            if let Slot::Live(l) = &st.slots[0] {
                l.pause.pause();
            }
        } else {
            self.audio.resume();
            if let Slot::Live(l) = &st.slots[0] {
                l.pause.resume();
            }
        }
    }

    pub(crate) fn is_paused(&self) -> bool {
        self.user_paused.load(Ordering::Acquire)
    }

    /// Seek the current track, with the **near-end clamp** applied.
    ///
    /// # The clamp, and why it lives here
    ///
    /// A seek target is resolved into a byte offset through the track's index. For a container
    /// with no cue table — a FLAC, so most of a music library — that is a proportional estimate,
    /// so a target at the duration resolves to a byte at the end of the file. The source then
    /// reads nothing, the pipeline EOSes immediately, and the boundary machinery correctly does
    /// what it does at any end of stream: it advances the queue. Every layer behaved, and the
    /// user heard their track vanish and the next one start.
    ///
    /// So the target is clamped to `duration - TAIL_EPSILON` whenever the duration is known.
    /// **Seeking to (or past) the end means "play the last quarter-second, then advance"** —
    /// audible, and the obvious reading of dragging a scrubber to the far right. Past the
    /// duration is not an error and not ignored: it is the same gesture, and it lands in the
    /// same place. A track shorter than the epsilon clamps to zero, which is the same policy
    /// taken to its limit — you hear the whole thing.
    ///
    /// It belongs to the engine and not to `SeekIndex`. The index answers a mechanical question
    /// ("which byte is this time") and is used by the introspection server and by seeking code
    /// that means exactly what it asks for; this is a *user interface* policy about what a
    /// gesture ought to sound like, and the engine is the layer that owns those.
    pub(crate) fn seek(&self, to: Duration) {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Slot::Live(live) = &st.slots[0] else { return };
        let duration = live.duration.unwrap_or(Timestamp::NONE);
        let mut ns = u64::try_from(to.as_nanos()).unwrap_or(u64::MAX);
        if let Some(end) = duration.nanos() {
            ns = ns.min(end.saturating_sub(TAIL_EPSILON.as_nanos() as u64));
        }
        let target = Timestamp::from_nanos(ns);
        // Seek to the *resolved* cue, never the request: the index floors to the preceding
        // resume-safe position, and rebasing running time to the request while content resumes
        // earlier leaves the pipeline permanently behind its own clock.
        if let Some((byte, landed)) = live.index.resolve(target, duration) {
            live.seek.seek(byte, landed);
            // Diagnostics only, and free unless armed: the generation is published, so every
            // millisecond after this belongs to the pipeline and the device, not to us. See
            // `pf_pipewire::probe` and `examples/seek_latency.rs`.
            pf_pipewire::probe::mark(pf_pipewire::probe::Stage::Dispatched);
        }
    }

    pub(crate) fn position(&self) -> Duration {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Slot::Live(live) = &st.slots[0] else { return Duration::ZERO };
        // A stretched track's device time is not its source time — a 1.5× audiobook has heard
        // 90 seconds of device audio at the two-minute mark. The stretcher counts the input it
        // consumed, which is what a progress bar over the file must show.
        if let Some(read) = &live.stretch_pos {
            return Duration::from_nanos(read().nanos().unwrap_or(0));
        }
        if !live.heard.load(Ordering::Acquire) {
            if !self.out.is_attached() {
                // Our epoch has not opened yet; the shared position still describes the track
                // that just ended.
                return Duration::ZERO;
            }
            live.heard.store(true, Ordering::Release);
        }
        self.out.position()
    }

    pub(crate) fn duration(&self) -> Option<Duration> {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Slot::Live(live) = &st.slots[0] else { return None };
        live.duration.and_then(|t| t.nanos()).map(Duration::from_nanos)
    }

    pub(crate) fn queue_len(&self) -> usize {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.slots.iter().filter(|s| !s.is_empty()).count()
    }

    /// Whether a track is actually running (as opposed to still opening).
    pub(crate) fn is_playing(&self) -> bool {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        matches!(&st.slots[0], Slot::Live(_))
    }

    /// Change the current track's ReplayGain correction live.
    pub(crate) fn set_track_gain_db(&self, db: f32) -> bool {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Slot::Live(live) = &st.slots[0] else { return false };
        let Some(id) = live.gain else { return false };
        let linear = pf_play::chain::gain_db_to_linear(db);
        live.props.set(id, "gain", ChainHandles::gain_value(linear)).is_ok()
    }

    // --- shutdown --------------------------------------------------------------------------

    pub(crate) fn begin_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.audio.pause();
        let taken = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            st.take_all()
        };
        // Inline, not via the worker: the worker is being wound down, and a teardown that
        // needed it would deadlock against a build already in flight.
        for (_, live) in taken {
            self.retire(*live, Retire::Quiet);
        }
        self.close_jobs();
    }

    /// Wait for every playback thread to finish, joining as they do. Returns `true` if the
    /// graveyard emptied within `patience`.
    pub(crate) fn shutdown_join(&self, patience: Duration) -> bool {
        let deadline = Instant::now() + patience;
        loop {
            self.reap();
            if self.graveyard_len() == 0 {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(POLL);
        }
    }
}

// --- helpers -------------------------------------------------------------------------------

/// The stretcher's source-time reader, type-erased.
///
/// `ChainHandles::stretch_position` is a `profluens_audio::StretchPosition`; capturing it in a
/// closure keeps this crate's dependency list to the four crates it genuinely needs, and the
/// closure is built once per track at open time.
fn stretch_reader(h: &ChainHandles) -> Option<Box<dyn Fn() -> Timestamp + Send + Sync>> {
    let pos = h.stretch_position.as_ref()?.clone();
    Some(Box::new(move || pos.position()))
}

/// Everything the pipeline's bus has to say that an application should hear.
///
/// Only `Error` is forwarded. `Warning` and `Qos` are the bus's explicitly *droppable* classes —
/// diagnostics for a developer, not facts a music player can act on — and forwarding them would
/// bury the one message that matters. The synthetic element id `run()` posts its own return
/// error under is skipped: that error is already reported from `run()`'s result.
fn drain_bus(player: &Player) -> Vec<String> {
    let mut errors = Vec::new();
    while let Some(msg) = player.pipeline.bus().try_recv() {
        if let BusMessage::Error { element, error } = msg {
            if element != ElementId(u32::MAX) {
                errors.push(format!("{error:?}"));
            }
        }
    }
    errors
}

/// Poll `cond` until it holds or `patience` elapses. Returns whether it held.
pub(crate) fn wait_until(patience: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + patience;
    loop {
        if cond() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(POLL);
    }
}
