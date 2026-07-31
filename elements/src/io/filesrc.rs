//! `filesrc` — reads a file into pooled buffers via the reactor
//! (spec: Milestone applications §1; IO). Reactor-native: it registers its file,
//! keeps up to `credits` positioned reads in flight, and emits completed buffers.
//! Positioned reads (explicit offset) make it seek-ready.

use std::fs::File;
use std::path::{Path, PathBuf};

use profluens_core::batch::Inputs;
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

/// The file to read (spec: Plugins — `parse("filesrc path=…")`). A path is a free-form
/// string: `Constraint::Any` (paths ride `Value::Id`; see [`Pipeline::set_str`]).
/// Structural (`live: false`): opening the file is a `start()`-time act.
///
/// [`Pipeline::set_str`]: profluens_core::pipeline::Pipeline::set_str
static PROPS: [PropDesc; 1] = [PropDesc {
    name: "path",
    allowed: Constraint::Any,
    live: false,
}];

// COLD: make_default boxes one instance per registry-created element, never per buffer.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "filesrc",
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
    make_default: Some(|| Box::new(FileSrc::new(""))),
};

/// One slot of [`FileSrc::window`]: the read submitted with `seq & wmask == i`.
#[derive(Default)]
struct Slot {
    /// The byte offset that read was submitted at, recorded at submission — what the
    /// short-read repair resumes from, and what makes the emit order offset-exact.
    offset: u64,
    /// Its completed buffer, once the reactor hands it back. `None` until then.
    buf: Option<profluens_core::buffer::Buffer>,
}

pub struct FileSrc {
    path: PathBuf,
    file: FileHandle,
    started: bool,
    offset: u64,
    eof: bool,
    in_flight: u32,
    seq: u64,
    /// Reads submitted before a seek carry a `user` (their `seq`) below this floor; their
    /// completions land at the *old* offset, so they are discarded (spec: flush/seek).
    valid_from: u64,
    /// **Re-sequencing window.** Up to `credits` positioned reads are in flight at once, and
    /// the [`Reactor`](profluens_core::io::Reactor) contract promises nothing about the order
    /// they complete in — `IoUringReactor` says so outright ("positioned (offset-carrying) ops
    /// make completion order irrelevant"), because io_uring serves a page-cache hit inline and
    /// punts a miss to an io-wq worker. Pushing in arrival order would therefore permute the
    /// byte stream. Completions park here keyed on `user` (= their `seq`) and leave in `seq`
    /// order; `elements/tests/filesrc_reactor_contract.rs` holds the line.
    ///
    /// Sized once in `start()` to the credit budget — the exact bound on in-flight reads —
    /// so it never allocates per buffer, and indexed `seq & wmask` (a ring: only the
    /// `next_emit .. next_emit + len` window can ever be occupied, which the submit gate
    /// enforces).
    window: Vec<Slot>,
    /// `window.len() - 1`. The window is rounded up to a power of two so indexing is a mask
    /// rather than a `u64` division on the per-buffer path.
    wmask: u64,
    /// How many window slots currently hold a completed buffer. Purely a fast-out: the
    /// scheduler calls `process()` many times per buffer (it drives to quiescence), and
    /// without this the emit loop re-probed the window on every one of those empty passes —
    /// measured at ~5% of the filesrc→filesink path's user instructions.
    parked: u32,
    /// The `seq` of the next read due downstream.
    next_emit: u64,
}

impl FileSrc {
    // COLD: one-time construction. The re-sequencing window starts empty and is sized once in
    // `start()`, when the credit budget is known — never per buffer.
    #[allow(clippy::disallowed_methods)]
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            file: FileHandle(0),
            started: false,
            offset: 0,
            eof: false,
            in_flight: 0,
            seq: 0,
            valid_from: 0,
            window: Vec::new(),
            wmask: 0,
            parked: 0,
            next_emit: 0,
        }
    }

    /// This `seq`'s window slot. `None` only if a reactor completed a read outside the credit
    /// budget it was given — a contract violation, reported rather than silently reordered.
    #[inline]
    fn slot_of(&self, seq: u64) -> Option<usize> {
        let len = self.window.len() as u64;
        (len > 0 && seq >= self.next_emit && seq < self.next_emit + len)
            .then_some((seq & self.wmask) as usize)
    }

    /// Abandon every read from `seq` on — those parked in the window and those still in
    /// flight, whose completions then land below the `valid_from` floor and are dropped.
    /// Both a seek and the short-read repair need this: each invalidates the offset every
    /// later read was submitted at. `in_flight` is deliberately untouched — the parked
    /// buffers were already counted down, and the still-flying ones must stay counted until
    /// their completions arrive (the same accounting rule that keeps `discard_buffers` off
    /// the IO mailbox).
    fn abandon_from(&mut self, seq: u64) {
        for slot in &mut self.window {
            slot.buf = None; // buffers recycle on drop
        }
        self.parked = 0;
        self.valid_from = seq;
        self.next_emit = seq;
    }

    /// Release one in-order read downstream. `submitted_at` is the offset it was read from.
    /// Returns `true` when this read ended the run of contiguous data — EOF, or a short read
    /// that invalidated every offset submitted behind it — so the caller stops draining.
    fn emit(
        &mut self,
        ctx: &mut Ctx,
        buf: profluens_core::buffer::Buffer,
        submitted_at: u64,
    ) -> bool {
        let n = buf.memory.len();
        let cap = buf.memory.capacity();
        if n == 0 {
            // EOF. Every later read was submitted past the end, so abandon them.
            drop(buf);
            self.abandon_from(self.seq);
            self.eof = true;
            return true;
        }
        self.next_emit += 1;
        log!(&*ctx, Level::Debug, "read", bytes = n);
        ctx.out(PadId(0)).push(buf);
        if n < cap {
            // A short read: `IoResult::Ok(n)` is documented as "bytes transferred", not "the
            // buffer was filled" (`read_at` is one `pread(2)`, free to come up short on a
            // network filesystem or a signal). Every later read was therefore submitted
            // `cap - n` bytes too far ahead — abandon them and resume exactly after the bytes
            // we did get, instead of leaving a hole in the stream.
            self.abandon_from(self.seq);
            self.offset = submitted_at + n as u64;
            return true;
        }
        false
    }
}

impl Element for FileSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // A parsed `path=` overrides the constructor value (spec: Plugins — the string
        // rides `Value::Id`, resolved by name off the value vocabulary). Falls back to
        // the constructor path when unset (the typed `FileSrc::new(path)` path).
        if let Some(profluens_core::format::Value::Id(id)) = ctx.prop("path") {
            if let Some(s) = ctx.value_name(id) {
                self.path = PathBuf::from(s);
            }
        }
        let f = File::open(&self.path)
            .map_err(|e| Error::Resource(format!("open {}: {e}", self.path.display())))?;
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        self.file = ctx.io().register(f);
        // One-time: the re-sequencing window is exactly as deep as the in-flight read budget.
        // COLD: sized once per run, never per buffer.
        #[allow(clippy::disallowed_methods)]
        {
            let depth = (ctx.io().credits() as usize).max(1).next_power_of_two();
            self.window.resize_with(depth, Slot::default);
            self.wmask = depth as u64 - 1;
        }
        self.started = true;
        log!(&*ctx, Level::Debug, "open", len = len);
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.started {
            return Err(Error::Todo("filesrc not started"));
        }

        // Drain completions. A completion that is already next in line goes straight
        // downstream (the case every in-order reactor produces, and the one that must cost
        // nothing); one that arrived early parks in the window until its predecessors land.
        loop {
            let completion = ctx.io().next_completion();
            let c = match completion {
                Some(c) => c,
                None => break,
            };
            self.in_flight -= 1;
            if c.user < self.valid_from {
                continue; // read submitted before a seek: wrong offset, discard (buf recycles)
            }
            match c.result {
                IoResult::Ok(_) => {
                    // A reactor completing a read outside the credit budget it was given is a
                    // contract violation, not a stream condition — refuse rather than silently
                    // reorder or drop.
                    let Some(slot) = self.slot_of(c.user) else {
                        return Err(Error::Todo("filesrc: completion outside the credit window"));
                    };
                    if self.parked == 0 && c.user == self.next_emit {
                        let at = self.window[slot].offset;
                        if self.emit(ctx, c.buf, at) {
                            break; // EOF or short read: the rest of the mailbox is stale
                        }
                    } else {
                        self.window[slot].buf = Some(c.buf);
                        self.parked += 1;
                    }
                }
                IoResult::Cancelled => {
                    // A cancelled read leaves a hole its window slot would otherwise block
                    // on forever (nothing will ever fill it, and the emit cursor cannot pass
                    // an unfilled slot). Treat it as a resync point: abandon everything from
                    // here and resume at the offset the cancelled read was aimed at. The
                    // scheduler does not currently cancel, so this is the safety net, not a
                    // hot path.
                    if let Some(slot) = self.slot_of(c.user) {
                        let at = self.window[slot].offset;
                        self.abandon_from(self.seq);
                        self.offset = at;
                        break;
                    }
                }
                IoResult::Err(k) => return Err(Error::Resource(format!("read: {k:?}"))),
            }
        }

        // Release whatever the parked completions made contiguous, in submission order.
        while self.parked > 0 {
            let Some(slot) = self.slot_of(self.next_emit) else { break };
            let Some(buf) = self.window[slot].buf.take() else { break };
            self.parked -= 1;
            let at = self.window[slot].offset;
            if self.emit(ctx, buf, at) {
                break;
            }
        }

        // Top up positioned reads up to credits, pool availability, and window depth (a read
        // may only be submitted while its emit slot is free — the window is what bounds how
        // far ahead of `next_emit` submission may run).
        let credits = ctx.io().credits();
        let horizon = self.next_emit + self.window.len() as u64;
        while !self.eof && self.in_flight < credits && self.seq < horizon {
            let buf = match ctx.try_alloc(PadId(0)) {
                Some(b) => b,
                None => break, // pool full → backpressure
            };
            let cap = buf.memory.capacity() as u64;
            let user = self.seq;
            let Some(slot) = self.slot_of(user) else { break };
            self.window[slot].offset = self.offset;
            self.seq += 1;
            ctx.io().submit_read(self.file, self.offset, buf, user);
            self.offset += cap;
            self.in_flight += 1;
        }

        if self.eof && self.in_flight == 0 {
            log!(&*ctx, Level::Info, "eos", offset = self.offset);
            Ok(Flow::Eos)
        } else {
            Ok(Flow::Ok)
        }
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::FlushStart) {
            // Seek (spec: flush/seek): resume reading at the requested byte offset. Reads
            // already in flight will complete at the old offset — `abandon_from` raises the
            // `seq` floor so their completions are discarded, and empties the re-sequencing
            // window of any that already completed. `submit_read` takes the offset
            // explicitly, so the fd's cursor is irrelevant; nothing else to reset.
            if let Some(t) = ctx.seek_target() {
                self.abandon_from(self.seq);
                self.offset = t.to_byte;
                self.eof = false;
            }
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.started = false; // the reactor owns and closes the file
    }
}
