//! `growingfilesrc` — reads a file a *separate writer is still appending to*
//! (spec: Milestone applications §1; IO). The progressive-download case: an app
//! downloads a podcast episode to disk while this pipeline plays it back from the
//! same path, so the readable region grows under the reader.
//!
//! It is [`FileSrc`](super::FileSrc) plus one idea — a **frontier**: a monotonic
//! watermark, published by the writer through a [`FrontierHandle`], saying how many
//! leading bytes of the file are safely readable. Everything else (reactor
//! registration, pooled positioned reads up to the credit budget, the power-of-two
//! re-sequencing window, the seek seq-floor) is the same machinery, and the two
//! elements should be read side by side.
//!
//! **The frontier wait costs nothing.** When the reader catches up with the writer
//! there is simply no read to submit, so `process()` returns with no output and no
//! IO in flight — which is exactly the scheduler's definition of an idle pass, and
//! the group parks on the idle eventcount (`PauseShared::park_idle`). Nothing here
//! spins, and no new core machinery was needed. The writer's `advance` is *not* a
//! wake source (a [`FrontierHandle`] is a plain `Arc`, deliberately unaware of any
//! pipeline), so the park ends on the scheduler's 10 ms backstop tick instead of on
//! a published wake: watermark growth is observed within one tick. A run dominated
//! by frontier waits therefore shows a high `SchedulerStats::tick_expiries` — here
//! that is the design, not the defect that counter usually reports.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, PropDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{Constraint, OfferDesc};
use profluens_core::id::PadId;
use profluens_core::io::{FileHandle, IoResult};
use profluens_core::log;
use profluens_core::log::Level;
use profluens_core::time::Timestamp;

/// A file is an untyped byte stream — offer the open `bytes` family, no fields.
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

/// The file to read. Structural (`live: false`): opening it is a `start()`-time act.
/// Note that `parse("growingfilesrc path=…")` can set the path but *cannot* hand over
/// a [`FrontierHandle`] — see [`DESC`]'s `make_default` note.
static PROPS: [PropDesc; 1] = [PropDesc {
    name: "path",
    allowed: Constraint::Any,
    live: false,
}];

// COLD: make_default boxes one instance per registry-created element, never per buffer.
//
// A name-constructed instance has no reachable `FrontierHandle` — the handle the
// constructor returns is dropped on the spot. That is not a hang: the element treats
// "every handle dropped without finish/abort" as an abandoned download and winds down
// (see `End::Abandoned`), so `parse("growingfilesrc path=x ! …")` produces a clean
// zero-byte EOS with a bus warning rather than waiting forever at frontier 0. The
// typed `GrowingFileSrc::new(path)` constructor is the only functional path.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "growingfilesrc",
    pads: &PADS,
    props: &PROPS,
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(GrowingFileSrc::new("").0)),
};

// --- the frontier ----------------------------------------------------------------

/// The writer↔reader cell behind [`FrontierHandle`]. Four atomics; no lock, because the
/// reader only ever *loads* and the writer only ever moves the watermark forward.
struct FrontierShared {
    /// Bytes of the file that are safely readable, counted from 0. Monotonic.
    downloaded: AtomicU64,
    /// The final file length, meaningful only once `finished` is set.
    total: AtomicU64,
    /// The writer is done growing the file.
    finished: AtomicBool,
    /// The writer failed and will never finish.
    aborted: AtomicBool,
}

/// The writer's end of the frontier — `Send + Sync + Clone`, so the downloading task
/// (thread, executor, callback) publishes progress from wherever it lives.
///
/// # Trust model
///
/// The one contract is **`downloaded()` ≤ the bytes actually and durably readable from
/// the file**. Call [`advance`](Self::advance) only *after* the corresponding write is
/// visible to a reader in another thread — i.e. after `write_all` has returned (and
/// after `sync_data`/`flush` if the writer buffers). The element then guarantees it
/// never emits a byte that was above the watermark at the moment its read was
/// submitted, so a reader can never observe a half-written append.
///
/// Violating the contract (advancing ahead of the bytes) is not memory-unsafe and does
/// not corrupt the stream — the element clamps every read to the watermark *and* to
/// what the kernel actually returned, so an over-promise simply degrades to a short
/// read. The element then parks and waits for the watermark to move again before
/// retrying, so a buggy writer costs latency, never a hot loop and never garbage
/// bytes. If such a writer also calls [`finish`](Self::finish) on a file that is
/// shorter than the declared total, the element reports a truncation warning on the
/// bus and ends the stream at the bytes it really had.
#[derive(Clone)]
pub struct FrontierHandle(Arc<FrontierShared>);

impl FrontierHandle {
    /// Publish that the first `downloaded` bytes of the file are readable. Monotonic:
    /// a value below the current watermark is ignored, so out-of-order reports from a
    /// concurrent downloader can never walk the frontier backwards.
    pub fn advance(&self, downloaded: u64) {
        self.0.downloaded.fetch_max(downloaded, Ordering::Release);
    }

    /// The file will not grow any further and is `total` bytes long. Implies a final
    /// [`advance`](Self::advance) to `total` — everything up to the declared end is by
    /// definition readable. Idempotent.
    pub fn finish(&self, total: u64) {
        self.0.total.store(total, Ordering::Relaxed);
        self.0.downloaded.fetch_max(total, Ordering::Relaxed);
        // Release last: a reader that sees `finished` sees `total` and the watermark.
        self.0.finished.store(true, Ordering::Release);
    }

    /// The download failed; no more bytes will ever arrive. The element posts one
    /// [`BusMessage::Error`] and then ends its stream, so the pipeline winds down
    /// through the ordinary EOS path instead of hanging at the frontier.
    pub fn abort(&self) {
        self.0.aborted.store(true, Ordering::Release);
    }

    /// The current watermark.
    pub fn downloaded(&self) -> u64 {
        self.0.downloaded.load(Ordering::Acquire)
    }

    /// Whether [`finish`](Self::finish) has been called.
    pub fn is_finished(&self) -> bool {
        self.0.finished.load(Ordering::Acquire)
    }
}

impl FrontierShared {
    fn downloaded(&self) -> u64 {
        self.downloaded.load(Ordering::Acquire)
    }

    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire)
    }

    /// The final length. Only meaningful once [`is_finished`](Self::is_finished).
    fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }
}

// --- the element -----------------------------------------------------------------

/// Why the stream ended. Set once, reported once, then the element only drains.
#[derive(Clone, Copy, PartialEq, Eq)]
enum End {
    /// `finish()` was reached and every byte up to the declared total was emitted.
    Complete,
    /// `abort()`: the download failed.
    Aborted,
    /// Every [`FrontierHandle`] was dropped without `finish()` or `abort()` — the
    /// downloader is gone (it panicked, or was never wired up at all).
    Abandoned,
    /// `finish()` declared a total the file does not actually contain.
    Truncated,
}

/// One slot of [`GrowingFileSrc::window`]: the read submitted with `seq & wmask == i`.
#[derive(Default)]
struct Slot {
    /// The byte offset that read was submitted at — what the short-read repair resumes
    /// from, and what makes the emit order offset-exact.
    offset: u64,
    /// The frontier watermark **at the moment this read was submitted**. Emission is
    /// clamped to it, never to the (possibly larger) watermark at completion time: only
    /// bytes below the submission-time watermark are known to have been fully written
    /// *before* the read executed. See `emit`.
    limit: u64,
    /// Its completed buffer, once the reactor hands it back. `None` until then.
    buf: Option<profluens_core::buffer::Buffer>,
}

pub struct GrowingFileSrc {
    path: PathBuf,
    shared: Arc<FrontierShared>,
    file: FileHandle,
    started: bool,
    /// Where the *next read* is submitted from. Runs ahead of `done_upto` by whatever
    /// is in flight, and is rewound to `done_upto` by `abandon_from`.
    offset: u64,
    /// One past the last byte pushed downstream — the contiguous emit watermark. EOS,
    /// the short-read repair and the seek all key off this rather than off `offset`.
    done_upto: u64,
    end: Option<End>,
    end_reported: bool,
    in_flight: u32,
    seq: u64,
    /// Reads submitted before a seek carry a `user` (their `seq`) below this floor;
    /// their completions land at the *old* offset, so they are discarded.
    valid_from: u64,
    /// **Re-sequencing window** — identical in purpose to `FileSrc`'s: up to `credits`
    /// positioned reads are in flight and the [`Reactor`](profluens_core::io::Reactor)
    /// promises nothing about completion order, so completions park here keyed on
    /// `user` (= their `seq`) and leave in `seq` order. Sized once in `start()` to the
    /// credit budget, rounded up to a power of two so indexing is a mask.
    window: Vec<Slot>,
    /// `window.len() - 1`.
    wmask: u64,
    /// How many window slots hold a completed buffer — a fast-out so the many empty
    /// scheduler passes never re-probe the window.
    parked: u32,
    /// The `seq` of the next read due downstream.
    next_emit: u64,
    /// **The anti-spin gate**, armed when a read comes back with zero usable bytes from
    /// a region the watermark said was readable — i.e. the writer over-promised.
    ///
    /// Without it that case spins at full tilt: submitting a read and reaping its
    /// completion *is* scheduler progress, so the group never parks, and it would retry
    /// the empty read as fast as the CPU allows. While armed, no read is submitted, so
    /// the pass has no output and no IO in flight — an idle pass, which parks. The gate
    /// is then cleared **by that same parking pass** (see the `idle_pass` computation in
    /// `process`), so the retry lands one 10 ms tick later.
    ///
    /// Retrying at all, rather than waiting for the watermark to move, is deliberate: a
    /// writer whose file quietly catches up to an already-published watermark would
    /// otherwise never be noticed, turning a contract violation into a hang. This way it
    /// costs one wasted `pread` per tick and nothing else.
    stalled: bool,
}

impl GrowingFileSrc {
    /// Build a source for `path` together with the [`FrontierHandle`] the writer uses
    /// to publish progress. The file need not exist yet at construction time, and may
    /// be zero-length when the pipeline starts.
    // COLD: one-time construction. The re-sequencing window starts empty and is sized
    // once in `start()`, when the credit budget is known — never per buffer.
    #[allow(clippy::disallowed_methods)]
    pub fn new(path: impl AsRef<Path>) -> (Self, FrontierHandle) {
        let shared = Arc::new(FrontierShared {
            downloaded: AtomicU64::new(0),
            total: AtomicU64::new(0),
            finished: AtomicBool::new(false),
            aborted: AtomicBool::new(false),
        });
        let handle = FrontierHandle(Arc::clone(&shared));
        let me = Self {
            path: path.as_ref().to_path_buf(),
            shared,
            file: FileHandle(0),
            started: false,
            offset: 0,
            done_upto: 0,
            end: None,
            end_reported: false,
            in_flight: 0,
            seq: 0,
            valid_from: 0,
            window: Vec::new(),
            wmask: 0,
            parked: 0,
            next_emit: 0,
            stalled: false,
        };
        (me, handle)
    }

    /// This `seq`'s window slot. `None` only if a reactor completed a read outside the
    /// credit budget it was given — a contract violation, reported rather than silently
    /// reordered.
    #[inline]
    fn slot_of(&self, seq: u64) -> Option<usize> {
        let len = self.window.len() as u64;
        (len > 0 && seq >= self.next_emit && seq < self.next_emit + len)
            .then_some((seq & self.wmask) as usize)
    }

    /// Abandon every read from `seq` on — those parked in the window and those still in
    /// flight, whose completions then land below the `valid_from` floor and are dropped
    /// — and rewind submission to the first byte not yet emitted. A seek, the short-read
    /// repair and the frontier clamp all need this: each invalidates the offset every
    /// later read was submitted at. `in_flight` is deliberately untouched — the parked
    /// buffers were already counted down, and the still-flying ones must stay counted
    /// until their completions arrive.
    fn abandon_from(&mut self, seq: u64) {
        for slot in &mut self.window {
            slot.buf = None; // buffers recycle on drop
        }
        self.parked = 0;
        self.valid_from = seq;
        self.next_emit = seq;
        self.offset = self.done_upto;
    }

    /// Release one in-order read downstream, clamped to the frontier. `at` is the offset
    /// it was submitted from and `limit` the watermark then in force. Returns `true`
    /// when this read ended the run of contiguous data — because it was clamped, came up
    /// short, or came back empty — so the caller stops draining.
    fn emit(
        &mut self,
        ctx: &mut Ctx,
        mut buf: profluens_core::buffer::Buffer,
        at: u64,
        limit: u64,
    ) -> bool {
        let cap = buf.memory.capacity();
        // Two independent bounds, and both matter. `limit - at` is the frontier clamp:
        // the reactor read the whole buffer capacity (there is no length-carrying read
        // op — see the module note on `submit_read`), so a read issued near the frontier
        // legitimately returns bytes the writer had not yet declared safe, and those
        // must not reach the stream. `buf.memory.len()` is what the kernel actually
        // transferred: `IoResult::Ok(n)` is "bytes transferred", not "buffer filled".
        let allowed = limit.saturating_sub(at).min(cap as u64) as usize;
        let n = buf.memory.len().min(allowed);

        if n == 0 {
            // Nothing usable at `at`. For a *growing* file this is emphatically not
            // end-of-stream — the bytes have not arrived yet — so, unlike `FileSrc`,
            // an empty read never ends the stream on its own.
            drop(buf);
            self.abandon_from(self.seq); // rewinds `offset` to `done_upto` == `at`
            if self.shared.is_finished() && at < self.shared.total() {
                // The writer declared a total the file does not contain. Nothing will
                // ever fill the gap; end the stream at the bytes we really got.
                self.end = Some(End::Truncated);
            } else {
                self.stalled = true;
            }
            return true;
        }

        self.next_emit += 1;
        self.done_upto = at + n as u64;
        buf.memory.set_len(n);
        log!(&*ctx, Level::Debug, "read", bytes = n);
        ctx.out(PadId(0)).push(buf);
        if n < cap {
            // Clamped at the frontier, or a genuinely short read. Either way every read
            // behind this one was submitted `cap - n` bytes too far ahead — abandon them
            // and resume exactly after the bytes that did go out, instead of leaving a
            // hole in the stream.
            self.abandon_from(self.seq);
            return true;
        }
        false
    }

    /// Post the one bus message this ending owes the application, once.
    // COLD: runs at most once per stream; the `format!` is not on any per-buffer path.
    #[allow(clippy::disallowed_methods)]
    fn report_end(&mut self, ctx: &mut Ctx, end: End) {
        let element = ctx.element();
        match end {
            End::Complete => {
                log!(&*ctx, Level::Info, "eos", offset = self.done_upto);
            }
            End::Aborted => {
                let error = Error::Element {
                    element,
                    message: format!(
                        "growingfilesrc: the writer aborted the download of {} after \
                         {} of {} byte(s) — ending the stream",
                        self.path.display(),
                        self.done_upto,
                        self.shared.downloaded(),
                    ),
                };
                ctx.post(BusMessage::Error { element, error });
            }
            End::Abandoned => {
                let error = Error::Element {
                    element,
                    message: format!(
                        "growingfilesrc: every FrontierHandle for {} was dropped without \
                         finish() or abort() after {} byte(s) — treating the download as \
                         abandoned and ending the stream",
                        self.path.display(),
                        self.done_upto,
                    ),
                };
                ctx.post(BusMessage::Warning { element, error });
            }
            End::Truncated => {
                let error = Error::Element {
                    element,
                    message: format!(
                        "growingfilesrc: {} is shorter than the {} byte(s) the writer \
                         declared — only {} byte(s) were readable (the frontier contract \
                         is watermark <= bytes actually written)",
                        self.path.display(),
                        self.shared.total(),
                        self.done_upto,
                    ),
                };
                ctx.post(BusMessage::Warning { element, error });
            }
        }
    }
}

impl Element for GrowingFileSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // A parsed `path=` overrides the constructor value (the string rides
        // `Value::Id`, resolved by name off the value vocabulary).
        if let Some(profluens_core::format::Value::Id(id)) = ctx.prop("path") {
            if let Some(s) = ctx.value_name(id) {
                self.path = PathBuf::from(s);
            }
        }
        // The file must *exist* — an app that pipes a download into a pipeline creates
        // (or truncates) it before starting — but it may perfectly well be empty, and
        // usually is. Length is deliberately not consulted here: the frontier, not the
        // file's current size, defines what may be read.
        let f = File::open(&self.path)
            .map_err(|e| Error::Resource(format!("open {}: {e}", self.path.display())))?;
        self.file = ctx.io().register(f);
        // One-time: the re-sequencing window is exactly as deep as the in-flight read
        // budget. COLD: sized once per run, never per buffer.
        #[allow(clippy::disallowed_methods)]
        {
            let depth = (ctx.io().credits() as usize).max(1).next_power_of_two();
            self.window.resize_with(depth, Slot::default);
            self.wmask = depth as u64 - 1;
        }
        self.started = true;
        // **Length reporting.** `FileSrc` reads the size once at open and logs it; here
        // there is no size to report yet — the file is still growing, so its length is
        // *unknown* until the writer calls `finish()`. That is the honest answer for a
        // controller too: a seek bar built on this source shows an open-ended duration
        // until `FrontierHandle::is_finished()`, and `downloaded()` is the meaningful
        // live quantity in the meantime. The final length is logged at EOS.
        log!(&*ctx, Level::Debug, "open", frontier = self.shared.downloaded());
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.started {
            return Err(Error::Todo("growingfilesrc not started"));
        }
        // Where this pass started, for the `idle_pass` verdict at the bottom. Both are
        // monotonic within a pass: `done_upto` only moves on an emit, `seq` only on a
        // submission.
        let (done_before, seq_before) = (self.done_upto, self.seq);

        // Out-of-band endings the writer can announce at any moment. Checked before the
        // drain so an abort with reads in flight discards them instead of emitting them.
        if self.end.is_none() {
            if self.shared.is_aborted() {
                self.end = Some(End::Aborted);
            } else if !self.shared.is_finished() && Arc::strong_count(&self.shared) == 1 {
                // Only this element holds the cell, so no `FrontierHandle` exists and
                // none can appear (a handle is only ever cloned from another handle).
                // The download can therefore never progress or terminate itself.
                self.end = Some(End::Abandoned);
            }
        }

        // A. Drain completions. One already next in line goes straight downstream; one
        // that arrived early parks in the window until its predecessors land.
        loop {
            let Some(c) = ctx.io().next_completion() else { break };
            self.in_flight -= 1;
            if c.user < self.valid_from {
                continue; // read submitted before a seek: wrong offset, discard
            }
            if self.end.is_some() {
                continue; // winding down: drain the mailbox, emit nothing (buf recycles)
            }
            match c.result {
                IoResult::Ok(_) => {
                    // A reactor completing a read outside the credit budget it was given
                    // is a contract violation, not a stream condition.
                    let Some(slot) = self.slot_of(c.user) else {
                        return Err(Error::Todo(
                            "growingfilesrc: completion outside the credit window",
                        ));
                    };
                    if self.parked == 0 && c.user == self.next_emit {
                        let (at, limit) = (self.window[slot].offset, self.window[slot].limit);
                        if self.emit(ctx, c.buf, at, limit) {
                            break; // clamped, short or empty: the rest of the mailbox is stale
                        }
                    } else {
                        self.window[slot].buf = Some(c.buf);
                        self.parked += 1;
                    }
                }
                IoResult::Cancelled => {
                    // A cancelled read leaves a hole its window slot would block on
                    // forever. Treat it as a resync point. The scheduler does not
                    // currently cancel, so this is the safety net, not a hot path.
                    if self.slot_of(c.user).is_some() {
                        self.abandon_from(self.seq);
                        break;
                    }
                }
                IoResult::Err(k) => {
                    // The file went away or the read failed outright (the hostile case:
                    // a writer that truncates and unlinks under us). Report it against
                    // this element and fail the run — there is no recovery, and the
                    // frontier says nothing about a broken fd.
                    let element = ctx.element();
                    // COLD: one message on the fatal path.
                    #[allow(clippy::disallowed_methods)]
                    let err = Error::Element {
                        element,
                        message: format!(
                            "growingfilesrc: read of {} failed at offset {}: {k:?}",
                            self.path.display(),
                            self.done_upto,
                        ),
                    };
                    ctx.post(BusMessage::Error { element, error: err.clone() });
                    return Err(err);
                }
            }
        }

        // B. Release whatever the parked completions made contiguous, in submission order.
        while self.parked > 0 && self.end.is_none() {
            let Some(slot) = self.slot_of(self.next_emit) else { break };
            let Some(buf) = self.window[slot].buf.take() else { break };
            self.parked -= 1;
            let (at, limit) = (self.window[slot].offset, self.window[slot].limit);
            if self.emit(ctx, buf, at, limit) {
                break;
            }
        }

        // Sample the frontier for the submission gate — after the drain, because the
        // writer may have advanced while we were emitting. Two acquire loads per pass,
        // no lock.
        let frontier = self.shared.downloaded();
        let finished = self.shared.is_finished();

        // C. Top up positioned reads — bounded by credits, pool availability, window
        // depth, **and the frontier**. `self.offset < frontier` is the whole of the
        // "never read past what the writer has published" rule on the submission side;
        // `Slot::limit` + `emit`'s clamp is the other half, for the one read that
        // straddles the frontier.
        if self.end.is_none() && !self.stalled {
            let credits = ctx.io().credits();
            let horizon = self.next_emit + self.window.len() as u64;
            while self.in_flight < credits && self.seq < horizon && self.offset < frontier {
                let Some(buf) = ctx.try_alloc(PadId(0)) else {
                    break; // pool full → backpressure
                };
                let cap = buf.memory.capacity() as u64;
                let user = self.seq;
                let Some(slot) = self.slot_of(user) else { break };
                self.window[slot].offset = self.offset;
                self.window[slot].limit = frontier;
                self.seq += 1;
                ctx.io().submit_read(self.file, self.offset, buf, user);
                // A read that straddles the frontier will be clamped on completion, which
                // invalidates the offset of everything submitted behind it. Stop the burst
                // here rather than issue reads we are certain to abandon.
                let straddles = self.offset + cap > frontier;
                self.offset += cap;
                self.in_flight += 1;
                if straddles {
                    break;
                }
            }
        }

        // D. Release the stall gate on the pass that will park. "Nothing emitted, nothing
        // submitted, nothing in flight" is precisely the scheduler's own idle-pass test
        // (`!progressed && reactor.is_idle()` — see `run_group`), so clearing here means
        // the retry happens after exactly one 10 ms backstop tick, never sooner.
        let idle_pass =
            self.done_upto == done_before && self.seq == seq_before && self.in_flight == 0;
        if self.stalled && idle_pass {
            self.stalled = false;
        }

        // E. Termination.
        if self.end.is_none() && finished && self.done_upto >= self.shared.total() {
            self.end = Some(End::Complete);
        }
        let Some(end) = self.end else {
            // Still streaming. If nothing was submitted and nothing emitted this pass we
            // are sitting at the frontier: the scheduler sees an idle pass with no IO in
            // flight and parks the group (see the module note) — no spinning, and the
            // watermark is re-read on the next tick.
            return Ok(Flow::Ok);
        };
        if !self.end_reported {
            self.end_reported = true;
            self.report_end(ctx, end);
        }
        // Mirror `FileSrc`: only claim EOS once every completion has been reaped, so the
        // reactor mailbox is empty when the group checks for quiescence.
        if self.in_flight == 0 { Ok(Flow::Eos) } else { Ok(Flow::Ok) }
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::FlushStart) {
            // Seek. Reads already in flight will complete at the old offset —
            // `abandon_from` raises the `seq` floor so their completions are discarded,
            // and empties the re-sequencing window of any that already completed.
            //
            // **Beyond-frontier policy.** A target past the watermark is *not* an error
            // and never fails the seek: the element simply adopts the offset and waits
            // at the frontier exactly as it would mid-stream, resuming the moment the
            // watermark covers the target. This is deliberate — the application's
            // downloader owns out-of-range fetches (musikkspiller reissues an HTTP range
            // request and writes the bytes in), and from the element's side those bytes
            // just arrive. Erroring here would break the one workflow the element
            // exists for: seeking ahead in a partially downloaded episode.
            //
            // The one case that resolves immediately is a target at or past the end of
            // an already-finished file: nothing more is coming, so that is EOS, which is
            // what `FileSrc` does when a seek lands past the end.
            if let Some(t) = ctx.seek_target() {
                self.done_upto = t.to_byte;
                self.abandon_from(self.seq); // rewinds `offset` to the target
                self.stalled = false;
                self.end = None;
                self.end_reported = false;
                if self.shared.is_finished() && t.to_byte >= self.shared.total() {
                    self.end = Some(End::Complete);
                }
            }
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.started = false; // the reactor owns and closes the file
    }
}
