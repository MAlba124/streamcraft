//! The scan engine: a fixed array of state-machine slots driven against a [`Reactor`].
//!
//! One pass of the loop is: admit files into free slots (open, stat, submit the prefix
//! read), hand the gathered submissions to the reactor, run it once, route completions back
//! to their slots by [`ElementId`], and advance each machine — which either submits a
//! follow-up read or parses, emits, and frees the slot for the next path. That is the same
//! gather/submit/run/route shape the scheduler drives elements with; it is spelled out here
//! because a tag scan has no pipeline (cf. `elements/tests/io_hygiene.rs`, the workspace's
//! other hand-driven reactor).
//!
//! **Slot index is identity.** Slot `i` registers its file as `ElementId(i)` and reads it
//! through `FileHandle(i)`, so routing a completion is an array index, not a lookup. It is
//! also the close path: [`Reactor::set_file`] keys on that `u32` and *replaces* the previous
//! registration, dropping (and therefore closing) the `File` it held. The last file in each
//! slot stays open until the `Scanner` drops — at most `in_flight` descriptors, which is why
//! that number is capped rather than "one per path".
//!
//! ## Blocking calls
//!
//! `open(2)` and `stat(2)` have no [`Submission`] kind — the contract is `Read`/`Write`/
//! `Recv` — so they are blocking `std::fs` calls here. This is app-side setup code with no
//! scheduler group to stall (see the crate docs and `play/src/head.rs`), the case
//! `clippy.toml`'s header names as the legitimate exception. Everything that touches *file
//! content* goes through the reactor.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use profluens_core::buffer::{Buffer, BufferFlags};
use profluens_core::event::TagSink;
use profluens_core::id::{ElementId, FormatId};
use profluens_core::io::{
    Completion, FileHandle, IoResult, OpId, OpKind, Reactor, ReactorFactory, Submission,
    SyncReactor,
};
use profluens_core::memory::{Arena, Memory, Pool};
use profluens_core::time::Timestamp;

use crate::sniff::{sniff, Format};
use crate::store::ArenaSink;
use crate::{
    aiff, ape, flac, mkv, mp3, mp4, mpc, ogg, tail, wav, wv, FileMeta, Props, ScanConfig,
    ScanError, ScanOutcome, ScanResult, DEFAULT_PREFIX,
};

/// Ceiling on a single metadata extent read. A FLAC block chain declares its own length, so
/// a corrupt (or hostile) file can claim gigabytes of "metadata"; past this the scan keeps
/// whatever it already has rather than reading — and allocating — what the file demands.
/// 32 MiB is far beyond any real cover art.
const MAX_METADATA: u64 = 32 << 20;

/// How many follow-up reads one file may cost. FLAC needs a second read when cover art
/// overruns the prefix and, rarely, a third when another block follows the art; four bounds
/// a pathological chain without ever tripping on a real file.
const MAX_EXTENTS: u32 = 4;

/// How many oversized metadata-extent buffers a scanner parks for reuse, the total bytes it
/// will hold in them, and the largest single buffer worth parking.
///
/// **Why this exists.** [`Pool::acquire_exact`] serves a request larger than the pool's slot
/// size from a right-sized heap box that is deliberately *unlinked* from the pool
/// (`core/src/memory.rs`: a heap-exact buffer counted against the slot budget starves every
/// `try_acquire` client). Unlinked means it is **freed on drop, not recycled** — so every file
/// whose metadata extent overruns the 128 KiB slot pays a fresh allocation, and because the
/// box is created with `vec![0u8; n]` the kernel-or-libc zeroes bytes that the very next
/// `pread` overwrites in full.
///
/// That is not a rare outsized-art case: measured over a 1170-file library it was **315 files
/// (27%)**, mean extent 359 KiB, **115.7 MB zeroed per scan** — and `__memset_avx2_unaligned_erms`
/// was **21.8% of a warm single-threaded scan's cycles**, the single hottest symbol in the
/// profile, ahead of every parser.
///
/// So the scanner keeps its own small free list of them, which is the pattern this struct
/// already uses for [`Slot::bounce`]: a buffer that is pure reuse is held rather than handed
/// back. Retention is bounded by bytes rather than by count, because these buffers' sizes span
/// two orders of magnitude and a count-based cap would bound nothing.
///
/// Handing a parked buffer to the next file is safe even if the caller kept a picture out of
/// it: [`Memory`] is refcounted and [`Memory::as_mut_full`] is uniqueness-gated, so a still-shared
/// backing is copy-on-written before the read lands rather than written under the sharer.
///
/// **Depth is [`ScanConfig::slots`], not a constant.** Up to `in_flight` files hold an extent
/// buffer at the same instant, so a list shallower than that guarantees misses under load no
/// matter how much budget it has — measured: a depth of 8 against 16 in-flight left 83 of 315
/// extents still allocating, and they were the *large* ones (571 KiB mean against a 359 KiB
/// population mean), because best-fit correctly keeps handing out the small buffers.
const SPARE_EXTENT_BYTES: usize = 8 << 20;
const SPARE_EXTENT_MAX: usize = 2 << 20;

/// How much larger than the request a parked buffer may be and still be reused, as a right
/// shift of the request (`0` = the buffer may be up to twice the read). See
/// [`Scanner::acquire_extent`]: the reactor reads the buffer's whole capacity, so this bounds
/// wasted **IO**, not wasted memory.
///
/// Chosen by sweeping it against the two quantities it trades off — bytes read per file, and
/// allocations per scan — over the 1170-file library (`ceiling`, `KiB read/file`, `allocations
/// on a first scan`, `on a re-scan`; no reuse at all is the 277.2 / 751 / 632 row):
///
/// ```text
///   1.125x   276.9   683   480
///   1.25x    274.7   652   443
///   1.5x     271.4   613   402
///   2x       273.6   514   287     <- here
///   unbounded 287.1  255    57
/// ```
///
/// Unbounded reuse is much the best on allocations and the only row that reads *more* than
/// doing nothing (+3.6%), which is the wrong currency for this crate to spend. Everything at
/// or below `2x` reads **less** than the unoptimised scanner — the relationship is not
/// monotonic because a buffer that overshoots sometimes swallows a follow-up extent that would
/// otherwise have cost its own op — so `2x` is the point that improves both axes at once.
const SPARE_SLACK_SHIFT: u32 = 0;

/// Which reactor backend a [`Scanner`] ended up with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReactorKind {
    /// Core's dependency-free positioned-IO backend.
    Sync,
    /// `profluens_elements::io::IoUringReactor` (feature `io-uring`).
    IoUring,
    /// Whatever a caller-supplied [`ReactorFactory`] built (the test hook).
    Custom,
}

/// A parallel tag scanner: one per thread.
///
/// Owns a reactor, a scratch [`Arena`], a fixed array of state-machine slots, and a
/// [`Pool`] — which may be shared with the scanners on other threads ([`Scanner::with_pool`]).
pub struct Scanner {
    cfg: ScanConfig,
    reactor: Box<dyn Reactor>,
    kind: ReactorKind,
    pool: Pool,
    /// Per-file scratch for interned tag text. Reset after every callback returns; `reset`
    /// takes `&mut self`, so the borrow checker — not a comment — guarantees no `Tags` is
    /// still alive when it happens.
    arena: Arena,
    slots: Vec<Slot>,
    /// Submissions gathered this pass. Drained by the reactor, so its capacity survives
    /// (ZERO-COPY.md stage 4.2 — `submit` takes `&mut Vec` precisely so it can be reused).
    subs: Vec<Submission>,
    /// Completion buffer, reused across passes for the same reason.
    comps: Vec<(ElementId, Completion)>,
    /// Oversized metadata-extent buffers parked for the next file that needs one — see
    /// [`SPARE_EXTENT_BYTES`]. Bounded by that budget, so a scan's resident footprint stays
    /// the pool plus a constant.
    spare: Vec<Memory>,
    /// Bytes currently parked in `spare`. Tracked rather than recomputed so the budget check
    /// on the hot release path is an integer compare.
    spare_bytes: usize,
    next_op: u64,
}

/// What a slot is waiting for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Free.
    Idle,
    /// Filling the prefix buffer (file offset 0).
    Prefix,
    /// Filling the extent buffer (a positioned follow-up read).
    Extent,
    /// Filling the tail buffer: the last [`ScanConfig::tail_len`] bytes of the file.
    ///
    /// A phase of its own rather than another extent because the formats that need it need
    /// it *as well as* the head window, not instead of it: an MP3's ID3v2 tag and its ID3v1
    /// trailer are both tags, an Ogg stream's identification header and its last page's
    /// granule position are both properties. One buffer each, one read each, parsed
    /// together in [`Scanner::finish`].
    Tail,
}

/// The fill in progress: `want` bytes into the phase's destination buffer, starting at file
/// offset `base`, `got` of them present.
#[derive(Clone, Copy, Default)]
struct Fill {
    base: u64,
    want: usize,
    got: usize,
    /// The outstanding op reads into the bounce buffer, to be appended at `got` — the
    /// short-read top-up path.
    bouncing: bool,
}

struct Slot {
    phase: Phase,
    /// Reused across files: `clear` + `push` keeps the allocation, so a scan of ten thousand
    /// paths allocates one path buffer per slot, not one per file.
    path: PathBuf,
    meta: FileMeta,
    format: Format,
    /// Offset of the format's first byte — non-zero only behind an ID3v2 tag.
    body: u64,
    prefix: Option<Memory>,
    extent: Option<Memory>,
    extent_base: u64,
    /// The last `tail_len` bytes of the file, for the formats whose metadata lives at both
    /// ends (see [`Phase::Tail`]).
    tail: Option<Memory>,
    /// File offset of `tail`'s first byte.
    tail_base: u64,
    /// Whether the tail read has been issued: the planner must ask for it exactly once.
    tail_done: bool,
    /// Kept between files: the top-up destination for a short read, acquired on first need.
    bounce: Option<Memory>,
    fill: Fill,
    extents: u32,
}

impl Slot {
    fn new() -> Self {
        Self {
            phase: Phase::Idle,
            path: PathBuf::new(),
            meta: FileMeta { len: 0, mtime: SystemTime::UNIX_EPOCH },
            format: Format::Unknown,
            body: 0,
            prefix: None,
            extent: None,
            extent_base: 0,
            tail: None,
            tail_base: 0,
            tail_done: false,
            bounce: None,
            fill: Fill::default(),
            extents: 0,
        }
    }

    /// Free the slot, returning its buffers to the pool. The path allocation and the bounce
    /// buffer are kept — both are pure reuse.
    fn release(&mut self) {
        self.phase = Phase::Idle;
        self.prefix = None;
        self.extent = None;
        self.extent_base = 0;
        self.tail = None;
        self.tail_base = 0;
        self.tail_done = false;
        self.format = Format::Unknown;
        self.body = 0;
        self.fill = Fill::default();
        self.extents = 0;
    }

    /// The head window: the extent when there is one (every extent this engine issues starts
    /// at file offset 0 *except* MP4's, which is why [`Slot::extent_base`] travels with it),
    /// otherwise the prefix.
    fn window(&self) -> Option<&Memory> {
        self.extent.as_ref().or(self.prefix.as_ref())
    }

    /// `(bytes, file offset of their first byte)` for the head window.
    fn view(&self) -> (&[u8], u64) {
        match self.extent.as_ref() {
            Some(e) => (e.data(), self.extent_base),
            None => (self.prefix.as_ref().map(|p| p.data()).unwrap_or(&[]), 0),
        }
    }

    /// `(bytes, file offset of their first byte)` for a window ending at EOF.
    ///
    /// The tail buffer when one was read; otherwise the head window, but **only** when it
    /// already holds the whole file — which is why a fixture measured in kilobytes costs one
    /// read and not two.
    fn tail_view(&self) -> (&[u8], u64) {
        if let Some(t) = self.tail.as_ref() {
            return (t.data(), self.tail_base);
        }
        match self.view() {
            (w, 0) if w.len() as u64 >= self.meta.len => (w, 0),
            _ => (&[], 0),
        }
    }

    /// Every window this slot holds, as `(bytes, file offset of their first byte)`: the prefix,
    /// and the extent when there is one. Written into `out`, returning how many are live, so
    /// the hot path builds no `Vec`.
    ///
    /// Matroska is the one format that needs *both at once* rather than the extent superseding
    /// the prefix ([`Slot::view`]): its Segment index — the SeekHead that says where the tags
    /// are — is at the front of the file, while the tag block it names may only be in the
    /// extent. Every other format's extent either starts at 0 or is self-describing.
    fn views<'a>(&'a self, out: &mut [(&'a [u8], u64); 2]) -> usize {
        out[0] = (self.prefix.as_ref().map(|p| p.data()).unwrap_or(&[]), 0);
        match self.extent.as_ref() {
            Some(e) => {
                out[1] = (e.data(), self.extent_base);
                2
            }
            None => 1,
        }
    }

    /// Whether a window ending at EOF is available without another read.
    fn has_tail(&self) -> bool {
        self.tail.is_some() || matches!(self.view(), (w, 0) if w.len() as u64 >= self.meta.len)
    }

    /// The slot the current phase reads into. `&mut Option<_>` rather than `&mut Memory` so
    /// the buffer can be *taken* for the submission (it travels to the reactor and back with
    /// the completion) without a placeholder buffer standing in for it.
    fn dest(&mut self) -> Option<&mut Option<Memory>> {
        match self.phase {
            Phase::Prefix => Some(&mut self.prefix),
            Phase::Extent => Some(&mut self.extent),
            Phase::Tail => Some(&mut self.tail),
            Phase::Idle => None,
        }
    }
}

/// A [`TagSink`] that discards: used for the planning walk, which runs before the engine
/// knows whether it has all the bytes and must not emit anything the parse pass will emit
/// again.
struct NullSink;

impl TagSink for NullSink {
    fn text(&mut self, _key: &str, _value: &str) {}
    fn picture(&mut self, _mime: &str, _data: &[u8]) {}
}

impl Scanner {
    /// A scanner with its own pool (slot size [`DEFAULT_PREFIX`]).
    pub fn new(cfg: ScanConfig) -> Self {
        let pool = default_pool(cfg, 1);
        Self::with_pool(cfg, pool)
    }

    /// A scanner drawing buffers from `pool` — [`Pool`] is `Send + Sync` and cheap to clone,
    /// so the scanners on every thread can share one (see [`crate::scan_parallel`]). The
    /// pool's slot size becomes the prefix-read size.
    pub fn with_pool(cfg: ScanConfig, pool: Pool) -> Self {
        let (reactor, kind) = default_reactor();
        Self::build(cfg, pool, reactor, kind)
    }

    /// A scanner whose reactor comes from `factory` — the test hook, mirroring
    /// [`Pipeline::set_reactor_factory`](profluens_core::pipeline::Pipeline::set_reactor_factory).
    /// A factory that fails falls back to [`SyncReactor`], exactly like the `io-uring` path.
    pub fn with_reactor(cfg: ScanConfig, pool: Pool, factory: &ReactorFactory) -> Self {
        match (**factory)() {
            Ok(r) => Self::build(cfg, pool, r, ReactorKind::Custom),
            Err(_) => Self::with_pool(cfg, pool),
        }
    }

    // One-time construction: the slot array, the submission/completion buffers and the boxed
    // reactor are built once per thread and reused for every file thereafter (spec:
    // allocation discipline — setup is the sanctioned exception).
    #[allow(clippy::disallowed_methods)]
    fn build(cfg: ScanConfig, pool: Pool, reactor: Box<dyn Reactor>, kind: ReactorKind) -> Self {
        let n = cfg.slots();
        let mut slots = Vec::with_capacity(n);
        slots.resize_with(n, Slot::new);
        Self {
            cfg,
            reactor,
            kind,
            pool,
            arena: Arena::default(),
            slots,
            subs: Vec::with_capacity(n),
            comps: Vec::with_capacity(n),
            spare: Vec::with_capacity(n),
            spare_bytes: 0,
            next_op: 0,
        }
    }

    /// Which backend this scanner is driving.
    pub fn reactor_kind(&self) -> ReactorKind {
        self.kind
    }

    /// The pool this scanner draws from (clone it to share with another scanner).
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// Scan every path, invoking `emit` once per path, in completion order (**not** input
    /// order — the whole point is that a slow file does not hold up the others).
    ///
    /// `emit` is called synchronously, and everything it is handed is borrowed for the
    /// duration of the call: the arena behind [`Tags`](crate::Tags) is reset the instant it
    /// returns, and the pool buffer behind a picture is recycled.
    pub fn scan<P, E>(&mut self, paths: impl IntoIterator<Item = P>, mut emit: E)
    where
        P: AsRef<Path>,
        E: FnMut(ScanOutcome<'_>),
    {
        let mut paths = paths.into_iter();
        let mut drained = false;
        let mut stalls = 0u32;

        loop {
            // --- admit: fill free slots from the path stream ---
            while !drained {
                let Some(idx) = self.free_slot() else { break };
                let Some(path) = paths.next() else {
                    drained = true;
                    break;
                };
                self.admit(idx, path.as_ref(), &mut emit);
            }

            let busy = self.busy();
            if busy == 0 {
                if drained {
                    break;
                }
                continue;
            }

            // --- submit + run ---
            self.reactor.submit(&mut self.subs);
            let mut comps = std::mem::take(&mut self.comps);
            self.reactor.run_once(&mut comps);

            if comps.is_empty() {
                // A reactor that returns nothing twice running with ops outstanding is not
                // going to make progress (a backend refusing the op, a cancelled ring).
                // Fail the in-flight files rather than spin forever.
                stalls += 1;
                if stalls > 1 {
                    self.comps = comps;
                    for i in 0..self.slots.len() {
                        if self.slots[i].phase != Phase::Idle {
                            self.fail(i, ScanError::Cancelled, &mut emit);
                        }
                    }
                    continue;
                }
            } else {
                stalls = 0;
            }

            // --- route completions back to their slots, and advance ---
            for (element, c) in comps.drain(..) {
                let idx = element.0 as usize;
                if idx < self.slots.len() {
                    self.complete(idx, c, &mut emit);
                }
            }
            self.comps = comps;
        }
    }

    fn free_slot(&self) -> Option<usize> {
        self.slots.iter().position(|s| s.phase == Phase::Idle)
    }

    fn busy(&self) -> usize {
        self.slots.iter().filter(|s| s.phase != Phase::Idle).count()
    }

    /// Open + stat one path and submit its prefix read. Failures are emitted immediately and
    /// leave the slot free.
    // `File::open`/`metadata` are blocking, and deliberately so: there is no reactor op for
    // either (`OpKind` is Read/Write/Recv), and this is app-side setup with no scheduler
    // group to stall — the exception `clippy.toml`'s header names. See the module docs.
    #[allow(clippy::disallowed_methods)]
    fn admit(&mut self, idx: usize, path: &Path, emit: &mut impl FnMut(ScanOutcome<'_>)) {
        // Reuse the slot's path allocation rather than cloning the caller's.
        self.slots[idx].path.clear();
        self.slots[idx].path.push(path);

        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => return self.fail(idx, ScanError::Open(e.kind()), emit),
        };
        let md = match file.metadata() {
            Ok(m) => m,
            Err(e) => return self.fail(idx, ScanError::Open(e.kind()), emit),
        };
        if !md.is_file() {
            // A directory opens fine on Linux and fails only at read time; reject it here so
            // the caller gets one honest error instead of an EISDIR from the reactor.
            return self.fail(idx, ScanError::Open(std::io::ErrorKind::InvalidInput), emit);
        }
        let meta = FileMeta {
            len: md.len(),
            // A filesystem without mtime is not a scan failure — it costs an indexer its
            // change detection, nothing more.
            mtime: md.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        };
        self.slots[idx].meta = meta;

        if meta.len == 0 {
            // Nothing to read: an empty file is not an error, it is simply not audio.
            self.slots[idx].format = Format::Unknown;
            self.slots[idx].phase = Phase::Prefix; // occupied, so `finish` may release it
            return self.finish(idx, emit);
        }

        // Registering under `ElementId(idx)` replaces — and closes — whatever this slot held
        // for the previous file (see the module docs).
        self.reactor.set_file(ElementId(idx as u32), file);

        let Some(mem) = self.acquire_prefix() else {
            // Every buffer is out; leave the slot idle and retry once a file completes.
            // Reachable only if the pool is shared and momentarily drained — `busy() > 0`
            // then guarantees a completion is coming.
            self.slots[idx].phase = Phase::Idle;
            return;
        };
        let want = mem.capacity().min(usize::try_from(meta.len).unwrap_or(usize::MAX));
        self.slots[idx].prefix = Some(mem);
        self.slots[idx].phase = Phase::Prefix;
        self.slots[idx].fill = Fill { base: 0, want, got: 0, bouncing: false };
        self.submit(idx, 0, false);
    }

    /// A prefix buffer. `try_acquire` first so a shared pool applies backpressure; when
    /// nothing is in flight there is no completion to wait for, so fall back to
    /// `acquire_exact`, which cannot fail (it drops to a right-sized buffer outside the slot
    /// budget) — otherwise a drained pool would deadlock the scan.
    fn acquire_prefix(&mut self) -> Option<Memory> {
        if let Some(m) = self.pool.try_acquire() {
            return Some(m);
        }
        if self.busy() == 0 {
            return Some(self.pool.acquire_exact(self.pool.slot_size()));
        }
        None
    }

    /// Queue the read for slot `idx`'s current fill. `bounce` selects the top-up path.
    fn submit(&mut self, idx: usize, offset: u64, bounce: bool) {
        let mem = if bounce {
            let remaining = self.slots[idx].fill.want - self.slots[idx].fill.got;
            match self.slots[idx].bounce.take() {
                Some(m) if m.capacity() >= remaining => m,
                // `acquire_exact` never fails: a slot when the request meaningfully fills
                // one, a size-classed small buffer when it does not, a right-sized box above
                // both (core/src/memory.rs: `acquire_exact`).
                _ => self.pool.acquire_exact(remaining.min(self.pool.slot_size())),
            }
        } else {
            match self.slots[idx].dest().and_then(Option::take) {
                Some(m) => m,
                None => return,
            }
        };
        self.slots[idx].fill.bouncing = bounce;
        // Same packing core's `Io::next_op_id` uses: element in the high half, sequence in
        // the low one, so an op id names its owner.
        let op = OpId(((idx as u64) << 32) | (self.next_op & 0xFFFF_FFFF));
        self.next_op += 1;
        self.subs.push(Submission {
            op,
            element: ElementId(idx as u32),
            kind: OpKind::Read,
            file: FileHandle(idx as u32),
            offset,
            buf: wrap(mem),
            user: u64::from(bounce),
        });
    }

    /// A completion came back for slot `idx`.
    fn complete(&mut self, idx: usize, c: Completion, emit: &mut impl FnMut(ScanOutcome<'_>)) {
        if self.slots[idx].phase == Phase::Idle {
            return; // a straggler for a file already finished (a cancelled op, say)
        }
        let mem = c.buf.memory;
        let n = match c.result {
            IoResult::Ok(n) => n,
            IoResult::Cancelled => return self.fail(idx, ScanError::Cancelled, emit),
            IoResult::Err(k) => return self.fail(idx, ScanError::Io(k), emit),
        };

        if self.slots[idx].fill.bouncing {
            // The rare path: the reactor came up short of the buffer, so the remainder was
            // read into a spare and is appended here. This is the *only* copy the engine
            // makes — a full read lands straight in the destination buffer and is parsed
            // there. `IoResult::Ok(n)` is documented as "bytes transferred", not "buffer
            // filled" (core/src/io.rs), and `filesrc`'s own contract test builds a reactor
            // that halves every read, so this path is real, not defensive.
            let got = self.slots[idx].fill.got;
            let want = self.slots[idx].fill.want;
            let take = n.min(want.saturating_sub(got));
            let src = mem.data();
            if let Some(dst) = self.slots[idx].dest().and_then(Option::as_mut) {
                dst.as_mut_full()[got..got + take].copy_from_slice(&src[..take]);
                dst.set_len(got + take);
            }
            self.slots[idx].fill.got = got + take;
            self.slots[idx].bounce = Some(mem);
        } else {
            self.slots[idx].fill.got = n;
            let phase = self.slots[idx].phase;
            match phase {
                Phase::Prefix => self.slots[idx].prefix = Some(mem),
                Phase::Extent => self.slots[idx].extent = Some(mem),
                Phase::Tail => self.slots[idx].tail = Some(mem),
                Phase::Idle => return,
            }
        }

        let fill = self.slots[idx].fill;
        let at_eof = fill.base + fill.got as u64 >= self.slots[idx].meta.len;
        if fill.got >= fill.want || at_eof || n == 0 {
            // `n == 0` is EOF even when the file claims to be longer (it shrank under us);
            // stopping here is what keeps a lying `stat` from looping forever.
            self.advance(idx, emit);
        } else {
            let next = fill.base + fill.got as u64;
            self.submit(idx, next, true);
        }
    }

    /// The current fill is complete: plan the next read, or parse and emit.
    fn advance(&mut self, idx: usize, emit: &mut impl FnMut(ScanOutcome<'_>)) {
        match self.slots[idx].phase {
            Phase::Prefix => {
                let head = self.slots[idx].prefix.as_ref().map(|m| m.data()).unwrap_or(&[]);
                let s = sniff(head);
                self.slots[idx].format = s.format;
                self.slots[idx].body = s.body;
                self.plan(idx, emit);
            }
            Phase::Extent | Phase::Tail => self.plan(idx, emit),
            Phase::Idle => {}
        }
    }

    /// Decide what this file still needs, and start that read — or parse and emit.
    fn plan(&mut self, idx: usize, emit: &mut impl FnMut(ScanOutcome<'_>)) {
        // A planned read that turns out to be empty must fall through to `finish`, not
        // silently return: the slot would then be occupied with no op outstanding, and the
        // scan would sit there until the stall detector failed the file.
        let started = match self.next_read(idx) {
            Next::Extent { at, len } => self.start_extent(idx, at, len),
            Next::Tail => self.start_tail(idx),
            Next::Done => false,
        };
        if !started {
            self.finish(idx, emit);
        }
    }

    /// The head-window read this file still wants, if any. The tail is decided separately by
    /// [`Scanner::next_read`], so a format whose extent budget ran out still gets it.
    fn next_extent(&self, idx: usize) -> Option<(u64, usize)> {
        if self.slots[idx].extents >= MAX_EXTENTS {
            return None;
        }
        let slot = &self.slots[idx];
        let file_len = slot.meta.len;
        match slot.format {
            // WAV: the chunk walk reports the offset of the first chunk it could not reach.
            // In the common "tagger appended an INFO list after a huge `data` chunk" layout
            // that is the tail of the file, and one read of `tail_len` from that chunk
            // boundary finds it.
            Format::Wav => {
                let (window, base) = slot.view();
                // Only one follow-up for WAV: a second would mean the file has chunks
                // scattered past a 64 KiB tail, which no writer produces.
                match slot.extents {
                    0 => {
                        let mut props = Props::default();
                        wav::walk(window, base, file_len, &mut props, &mut NullSink)
                            .map(|at| (at, self.cfg.tail_len.min(64 << 20)))
                    }
                    _ => None,
                }
            }
            // FLAC: the block chain declares its own extent. Everything from the `fLaC`
            // marker to the end of the last block is read in one go, so cover art that
            // overruns the prefix costs exactly one more op.
            Format::Flac => {
                let window = slot.window();
                match window.map(|w| (w, flac::plan(w.data(), slot.body, file_len))) {
                    // `end > what we already have` is the loop guard: a block whose declared
                    // length runs past the end of the file asks for the same bytes for ever,
                    // so a request no larger than the window means "this is all there is".
                    Some((w, flac::Plan::Need { end }))
                        if end > w.data().len() as u64 && end <= MAX_METADATA =>
                    {
                        Some((0u64, usize::try_from(end).unwrap_or(usize::MAX)))
                    }
                    _ => None,
                }
            }
            // MP3: the ID3v2 tag states its own length, and the first audio frame — which is
            // where the duration comes from — sits immediately behind it. One range covers
            // both, so an oversized tag costs one read, not two.
            Format::Mp3 => {
                let (window, base) = slot.view();
                let end = mp3::head_end(window, file_len);
                // `head_short` is the trigger, `head_end` the size: see `mp3::head_short`.
                // Both hold only when `end > window.len()`, so the request always advances.
                (base == 0 && mp3::head_short(window, file_len) && end <= MAX_METADATA)
                    .then(|| (0u64, usize::try_from(end).unwrap_or(usize::MAX)))
            }
            // MP4: `locate_moov` answers "here", "read this range and ask again", or "no".
            // The middle arm is the non-`faststart` layout, where the walk stepped over a
            // multi-gigabyte `mdat` and the trailing `moov` is one positioned read away.
            Format::Mp4 => {
                let (window, base) = slot.view();
                match mp4::locate(window, base, file_len) {
                    mp4::Plan::Need { at, len } => {
                        // Clamped: an unread *tail* range is "everything from here to EOF",
                        // which on a large file is the file. Re-feeding a clamped chunk is
                        // legal — the protocol's answers are relative to whatever it is
                        // handed — and each hop starts strictly later, so the walk advances.
                        let want = len.min(MAX_METADATA);
                        // …with one exception the protocol cannot fix: a `moov` bigger than
                        // `MAX_METADATA` answers `Beyond` at the range we just clamped, for
                        // ever. Asking only when the request reaches bytes we do not already
                        // hold turns that into one read and a partial answer, not four.
                        (at != base || want > window.len() as u64)
                            .then(|| (at, usize::try_from(want).unwrap_or(usize::MAX)))
                    }
                    _ => None,
                }
            }
            // AIFF: the same shape as WAV and for the same reason — the audio (`SSND`) is
            // skipped by size, so a text or `ID3 ` chunk a tagger appended behind it is one
            // positioned read from the offset the chunk walk reports. One follow-up only:
            // a second would mean chunks scattered past a 64 KiB tail, which no writer emits.
            Format::Aiff => match slot.extents {
                0 => {
                    let (window, base) = slot.view();
                    aiff::plan(window, base, file_len)
                        .map(|at| (at, self.cfg.tail_len.min(64 << 20)))
                }
                _ => None,
            },
            // Ogg: walk page headers to the end of the last header packet. Only a comment
            // packet carrying cover art overruns the prefix, and then the request is exact.
            Format::OggOpus | Format::OggVorbis | Format::OggFlac | Format::OggSpeex => {
                let (window, base) = slot.view();
                if base != 0 {
                    return None;
                }
                let want = ogg::want(window, slot.body, slot.format);
                match ogg::plan(window, slot.body, file_len, want) {
                    Some(end) if end > window.len() as u64 && end <= ogg::MAX_META => {
                        Some((0u64, usize::try_from(end).unwrap_or(usize::MAX)))
                    }
                    _ => None,
                }
            }
            // Matroska: the front SeekHead names where every metadata master lives, so the
            // request is a position the file itself chose rather than a guess. Almost always
            // `None` — the reference library's 24 WebM files all keep Info, Tracks and Tags in
            // the first 600 bytes — but a muxer that wrote its tags behind the frames costs one
            // positioned read, and a tag block bigger than that read costs a second.
            Format::Mkv => {
                // Two reads, not the global four. A slot holds **one** extent, so a second
                // positioned read *replaces* the first — and Matroska is the only format whose
                // metadata can sit in two places far enough apart not to fit one range. Asking
                // a third time would therefore discard what the second read found and chase
                // the two forever. `plan` already merges the masters into a single range
                // whenever they fit one; when they do not, the earliest wins and the rest is
                // dropped, on the same principle `ogg.rs` states — tags without a cover beat
                // no tags at all.
                if slot.extents >= 2 {
                    return None;
                }
                let mut buf = [(&[][..], 0u64); 2];
                let n = slot.views(&mut buf);
                let (at, len) = mkv::plan(&buf[..n], file_len)?;
                // …but only when the request reaches bytes we do not already hold. A SeekHead
                // position that never resolves to a readable element would otherwise ask for
                // the same range until the extent budget ran out.
                let held = slot
                    .extent
                    .as_ref()
                    .is_some_and(|e| at >= slot.extent_base && at + len <= slot.extent_base + e.data().len() as u64);
                (!held && len > 0).then(|| (at, usize::try_from(len).unwrap_or(usize::MAX)))
            }
            // The APEv2 family: every property is in a fixed header at the front of the file
            // and every tag is in the trailing block, so the prefix and the tail read cover
            // them between them and there is never an extent.
            Format::Ape | Format::WavPack | Format::Musepack => None,
            // Recognised only: nothing to read past the prefix.
            Format::Unknown => None,
        }
    }

    /// What this file still needs. The head window is planned first; the tail read (for the
    /// formats whose metadata also lives at EOF) is issued once the head is settled, so the
    /// two never race for the slot's single [`Fill`].
    fn next_read(&self, idx: usize) -> Next {
        if let Some((at, len)) = self.next_extent(idx) {
            return Next::Extent { at, len };
        }
        let slot = &self.slots[idx];
        let wanted = wants_tail(slot.format) && !slot.tail_done && !slot.has_tail();
        if wanted && slot.meta.len > 0 && self.cfg.tail_len > 0 {
            return Next::Tail;
        }
        Next::Done
    }

    /// Read the last `tail_len` bytes of the file into the slot's tail buffer. Returns
    /// whether an op was actually submitted.
    fn start_tail(&mut self, idx: usize) -> bool {
        // Marked before the size checks below, so no path can ask for the tail twice.
        self.slots[idx].tail_done = true;
        let file_len = self.slots[idx].meta.len;
        let want = u64::try_from(self.cfg.tail_len).unwrap_or(u64::MAX).min(file_len);
        let at = file_len - want;
        let Ok(want) = usize::try_from(want) else { return false };
        if want == 0 {
            return false;
        }
        let mem = self.pool.acquire_exact(want);
        self.slots[idx].tail_base = at;
        self.slots[idx].tail = Some(mem);
        self.slots[idx].phase = Phase::Tail;
        self.slots[idx].fill = Fill { base: at, want, got: 0, bouncing: false };
        self.submit(idx, at, false);
        true
    }

    /// Read `[at, at + len)` into the slot's extent buffer. Returns whether an op was
    /// actually submitted.
    fn start_extent(&mut self, idx: usize, at: u64, len: usize) -> bool {
        let room = usize::try_from(self.slots[idx].meta.len.saturating_sub(at)).unwrap_or(usize::MAX);
        let want = len.min(room);
        if want == 0 {
            return false;
        }
        // A metadata extent is sized by the file, not by the pool: `acquire_exact` serves it
        // from a slot when it meaningfully fills one, from a size class when it is small, and
        // from a right-sized box above the slot size — the last being the heap allocation a
        // large cover art costs, outside the slot budget so it cannot starve the scan
        // (core/src/memory.rs: `acquire_exact`). `acquire_extent` checks the parked buffers
        // from previous files before paying for that box (see `SPARE_EXTENT_BYTES`).
        let mem = self.acquire_extent(want);
        self.slots[idx].extent_base = at;
        self.slots[idx].extent = Some(mem);
        self.slots[idx].phase = Phase::Extent;
        self.slots[idx].extents += 1;
        self.slots[idx].fill = Fill { base: at, want, got: 0, bouncing: false };
        self.submit(idx, at, false);
        true
    }

    /// A buffer for a metadata extent of `want` bytes: a parked one when it fits, otherwise
    /// the pool's.
    ///
    /// **Best fit, not first fit.** The parked sizes span 128 KiB to a few MiB; handing the
    /// largest buffer to the smallest request would leave the big requests to allocate while
    /// a big buffer sat used by a small one. Scanning the ≤ [`ScanConfig::slots`] entries to
    /// find the smallest that fits costs a handful of compares against a zeroing `calloc`.
    ///
    /// Only requests past the slot size look here at all: at or below it `acquire_exact`
    /// already recycles, through the pool's slot free-list or its size-classed small buckets,
    /// and parking one of those buffers here would be *taking it out* of that rotation.
    ///
    /// **A reused buffer must not be much bigger than the read.** `SyncReactor` reads
    /// `Memory::capacity()` bytes — `core/src/io.rs:581`, `file.read_at(s.buf.memory.as_mut_full(), …)`
    /// — not the length the planner asked for, so handing a 1 MiB spare to a 300 KiB extent
    /// reads 700 KiB of audio nobody wants. Unbounded best fit measured **+9.9 KiB read per
    /// file (+3.6%)**, which is the wrong currency to spend in a crate whose entire design
    /// argument is about IO. [`SPARE_SLACK_SHIFT`] caps the excess at a fraction of the
    /// request, so the worst case is bounded in the only unit that matters.
    fn acquire_extent(&mut self, want: usize) -> Memory {
        if want > self.pool.slot_size() {
            let ceiling = want.saturating_add(want >> SPARE_SLACK_SHIFT);
            let best = self
                .spare
                .iter()
                .enumerate()
                .filter(|(_, m)| m.capacity() >= want && m.capacity() <= ceiling)
                .min_by_key(|(_, m)| m.capacity())
                .map(|(i, _)| i);
            if let Some(i) = best {
                let mut mem = self.spare.swap_remove(i);
                self.spare_bytes -= mem.capacity();
                // A parked buffer still carries the previous file's length; the fill machinery
                // sets the real one from the completion, but leaving a stale `len` visible
                // between here and then would let a short-circuiting path parse another file's
                // bytes.
                mem.set_len(0);
                return mem;
            }
        }
        self.pool.acquire_exact(want)
    }

    /// Park slot `idx`'s extent buffer for reuse, if it is one of the heap-exact ones and the
    /// budget has room. Anything else is left to drop — which is how a pooled buffer gets
    /// back to the pool.
    ///
    /// Called just before [`Slot::release`], i.e. after `emit` has returned and every borrow
    /// of the buffer is dead.
    fn recycle_extent(&mut self, idx: usize) {
        let Some(mem) = self.slots[idx].extent.take() else { return };
        let cap = mem.capacity();
        // `<= slot_size` is the pool's own rotation (see `acquire_extent`); the two caps keep
        // one pathological file's multi-MiB extent from becoming this scanner's resident set.
        if cap <= self.pool.slot_size()
            || cap > SPARE_EXTENT_MAX
            || self.spare.len() >= self.cfg.slots()
            || self.spare_bytes + cap > SPARE_EXTENT_BYTES
        {
            return;
        }
        self.spare_bytes += cap;
        self.spare.push(mem);
    }

    /// Parse what was read, emit, reset the arena, free the slot.
    fn finish(&mut self, idx: usize, emit: &mut impl FnMut(ScanOutcome<'_>)) {
        {
            let slot = &self.slots[idx];
            let arena = &self.arena;
            // The window pictures may be sliced out of: the extent when there is one (for
            // FLAC, MP3 and MP4 it supersedes the prefix), otherwise the prefix.
            let window = slot.window();
            let mut sink = ArenaSink::new(arena, &self.pool, window);
            let mut props = Props::default();
            let file_len = slot.meta.len;

            match slot.format {
                Format::Wav => {
                    if let Some(p) = slot.prefix.as_ref() {
                        wav::walk(p.data(), 0, file_len, &mut props, &mut sink);
                    }
                    if let Some(e) = slot.extent.as_ref() {
                        wav::walk(e.data(), slot.extent_base, file_len, &mut props, &mut sink);
                    }
                }
                // AIFF: the same two-window shape as WAV — the prefix holds `COMM` and
                // whatever tag chunks precede the audio, the extent (when there was one)
                // holds a tag chunk the tagger appended behind it.
                //
                // The pass is the **outer** loop, not the inner one: an `ID3 ` chunk behind a
                // 500 MB `SSND` lands in the extent while a `NAME` chunk sits in the prefix,
                // and the sink returns the first value for a key. Walking each window in
                // full before the next would hand precedence to whichever window came first
                // rather than to ID3 (see `aiff::Pass::ORDER`).
                Format::Aiff => {
                    for pass in aiff::Pass::ORDER {
                        if let Some(p) = slot.prefix.as_ref() {
                            aiff::walk_pass(
                                p.data(),
                                0,
                                file_len,
                                &mut props,
                                pass,
                                arena,
                                &mut sink,
                            );
                        }
                        if let Some(e) = slot.extent.as_ref() {
                            aiff::walk_pass(
                                e.data(),
                                slot.extent_base,
                                file_len,
                                &mut props,
                                pass,
                                arena,
                                &mut sink,
                            );
                        }
                    }
                }
                // The APEv2 family: properties from a fixed header at the front, tags from
                // the trailing APEv2/ID3v1 pair. Header first, so its facts win — and so the
                // APEv2 block's ReplayGain items fill in what no header carries.
                Format::Ape | Format::WavPack | Format::Musepack => {
                    let (head, base) = slot.view();
                    let head = if base == 0 { head } else { &[][..] };
                    props = match slot.format {
                        Format::Ape => ape::props(head, slot.body),
                        Format::WavPack => wv::props(head, slot.body),
                        _ => mpc::props(head, slot.body),
                    };
                    let (tail_window, tail_base) = slot.tail_view();
                    let trailers = tail::measure(tail_window, tail_base, file_len);
                    tail::emit(tail_window, tail_base, file_len, trailers, arena, &mut sink);
                }
                Format::Flac => {
                    if let Some(w) = window {
                        props = flac::props(w.data(), slot.body);
                        // The sink's `src` is this same buffer, so every picture pf-flac hands
                        // over is a sub-view of it — cover art never leaves the read buffer.
                        flac::emit_tags(w.data(), slot.body, &mut sink);
                    }
                }
                Format::Mp3 => {
                    let (head, base) = slot.view();
                    let head = if base == 0 { head } else { &[][..] };
                    let (tail, tail_base) = slot.tail_view();
                    let v2 = mp3::v2_len(head);
                    // v2 first: `get` returns the first value for a key, so the richest,
                    // unambiguously-encoded tag wins over the trailing ones.
                    mp3::emit_v2(head, v2, arena, &mut sink);
                    let trailers = tail::measure(tail, tail_base, file_len);
                    tail::emit(tail, tail_base, file_len, trailers, arena, &mut sink);
                    // Tag bytes at either end are not audio: counting them would inflate the
                    // constant-bitrate estimate by exactly their size.
                    let audio = file_len.saturating_sub(v2).saturating_sub(trailers.total());
                    props = mp3::props(head, v2, audio);
                }
                Format::Mp4 => {
                    let (w, base) = slot.view();
                    if let mp4::Plan::Here { at, len } = mp4::locate(w, base, file_len) {
                        props = mp4::parse(w, at, len, arena, &mut sink);
                    }
                }
                Format::OggOpus | Format::OggVorbis | Format::OggFlac | Format::OggSpeex => {
                    let (w, base) = slot.view();
                    let w = if base == 0 { w } else { &[][..] };
                    let head = ogg::scan(w, slot.body, slot.format, arena, &mut sink);
                    props = head.props;
                    // The last page's granule position is the only statement of duration an
                    // Ogg file makes (RFC 3533 §6); it supersedes a FLAC STREAMINFO count,
                    // which describes the encoded stream rather than the file as it ends.
                    let (tail, _) = slot.tail_view();
                    if let Some(ns) = ogg::duration(tail, &head, slot.format) {
                        props.duration_ns = Some(ns);
                        props.duration_exact = true;
                    }
                }
                // Matroska: both windows, for the reason `Slot::views` documents — the Segment
                // index is at the front of the file, the tag block it names may be at the back.
                Format::Mkv => {
                    let mut buf = [(&[][..], 0u64); 2];
                    let n = slot.views(&mut buf);
                    props = mkv::parse(&buf[..n], &mut sink);
                }
                // Anything unrecognised: format and file facts are still worth reporting (see
                // the crate docs' coverage table).
                Format::Unknown => {}
            }

            emit(ScanOutcome {
                path: &slot.path,
                result: Ok(ScanResult {
                    format: slot.format,
                    file: slot.meta,
                    props,
                    tags: sink.finish(),
                }),
            });
        }
        // Every borrow of the arena died with the block above — which is precisely why
        // `reset` takes `&mut self`.
        self.arena.reset();
        self.recycle_extent(idx);
        self.slots[idx].release();
    }

    /// Emit a failure for `idx` and free the slot. Used both for a file that never got off
    /// the ground (open/stat) and for one that failed mid-read.
    fn fail(&mut self, idx: usize, err: ScanError, emit: &mut impl FnMut(ScanOutcome<'_>)) {
        emit(ScanOutcome { path: &self.slots[idx].path, result: Err(err) });
        self.recycle_extent(idx);
        self.slots[idx].release();
    }
}

/// The next read a file needs, or that it needs none.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Next {
    /// A positioned head-window read (`[at, at + len)`).
    Extent { at: u64, len: usize },
    /// The last `tail_len` bytes of the file.
    Tail,
    /// Everything is in hand: parse and emit.
    Done,
}

/// Whether a format's metadata also lives at the end of the file.
///
/// MP3 keeps ID3v1 and APEv2 there (see `mp3.rs`), and so do the three formats built around
/// that same trailing pair — Monkey's Audio, WavPack and Musepack (see `tail.rs`). Every Ogg
/// mapping keeps its duration there: an Ogg file states no length anywhere but in the granule
/// position of its last page (RFC 3533 §6).
///
/// WAV's trailing `LIST`/`INFO` and AIFF's trailing `ID3 ` chunk are *not* here: their offsets
/// are known exactly from the chunk walk, so they are extents, not blind tail reads.
fn wants_tail(format: Format) -> bool {
    matches!(
        format,
        Format::Mp3
            | Format::Ape
            | Format::WavPack
            | Format::Musepack
            | Format::OggOpus
            | Format::OggVorbis
            | Format::OggFlac
            | Format::OggSpeex
    )
}

/// The default pool for `scanners` scanners: [`DEFAULT_PREFIX`] slots, enough for every
/// slot's prefix, extent and tail buffer at once (an MP3 with an oversized ID3v2 tag holds
/// all three), plus a margin for the bounce buffers a short-reading reactor needs. Sized
/// rather than unbounded so a shared pool still applies backpressure; `acquire_exact`'s
/// size-classed and right-sized fallbacks mean overshooting the cap costs a recycled small
/// buffer, never a deadlock.
pub(crate) fn default_pool(cfg: ScanConfig, scanners: usize) -> Pool {
    let slots = scanners.max(1) * cfg.slots() * 3 + 4;
    Pool::bounded(DEFAULT_PREFIX, u32::try_from(slots).unwrap_or(u32::MAX))
}

// One-time: the reactor is boxed once per thread (spec: allocation discipline — setup is the
// sanctioned exception; `ReactorFactory` itself is defined as returning a `Box<dyn Reactor>`).
#[allow(clippy::disallowed_methods)]
fn default_reactor() -> (Box<dyn Reactor>, ReactorKind) {
    #[cfg(feature = "io-uring")]
    {
        // The ring is per-thread and may legitimately fail to set up (an old kernel, seccomp,
        // an exhausted `RLIMIT_MEMLOCK`); a tag scan is not worth failing over.
        if let Ok(r) = profluens_elements::io::IoUringReactor::new() {
            return (Box::new(r), ReactorKind::IoUring);
        }
    }
    (Box::new(SyncReactor::new()), ReactorKind::Sync)
}

/// Wrap pool memory as the [`Buffer`] a [`Submission`] carries. All-POD metadata: a read has
/// no timestamps and no format.
fn wrap(memory: Memory) -> Buffer {
    Buffer {
        memory,
        pts: Timestamp::ZERO,
        dts: Timestamp::ZERO,
        duration: Timestamp::ZERO,
        flags: BufferFlags::empty(),
        format: FormatId(0),
        sync: None,
    }
}
