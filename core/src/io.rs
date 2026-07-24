//! The reactor submit/complete contract (spec: IO: built for the io_uring era).
//!
//! Elements never touch a reactor directly. Inside `process()` they use the [`Io`]
//! handle from `ctx.io()` to **register** files, **submit** read/write ops, and
//! **drain completions** — the completion-shaped API the whole design hangs off.
//!
//! [`Reactor`] is the backend contract. Core ships [`SyncReactor`] — a
//! dependency-free, synchronous, positioned-IO backend (`read_at`/`write_at`),
//! executed by the scheduler between element passes. A truly-async backend
//! (io_uring on Linux, in `streamcraft-elements` behind a feature) implements the
//! same trait and is injected via `Pipeline::set_reactor` — element code is
//! unchanged. Positioned IO (explicit offsets) keeps ops order-independent and
//! makes sources seek-ready.
//!
//! Reactors also own **streaming page-cache hygiene** ([`StreamHygiene`]): a large
//! sequential file run must not accumulate gigabytes of page cache (read pages we
//! will never revisit, dirty write pages the kernel lazily holds). Measured on a
//! 1.6 GB → 1.4 GB remux under a 3 GiB cgroup `MemoryMax`: the run froze ~5 s
//! mid-file while direct reclaim drained ~1.2 GiB of accumulated dirty pages.
//! The fix is the classic pair of `posix_fadvise(2)` / `sync_file_range(2)` hints,
//! issued a few times per 64 MiB of forward progress — noise on the fast path.
//!
//! `unsafe` is permitted here (joining the audited set: `memory`, `ring`): the
//! hygiene syscalls have no libc to call through (core is dependency-free), so
//! [`fadvise`] and [`sync_file_range`] enter the kernel via `asm!("syscall")`.
//! Both take only an fd + integer arguments — no userspace memory crosses the
//! boundary, so the blast radius is "wrong advice", not corruption.
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::fs::File;
use std::io::ErrorKind;
use std::os::unix::fs::FileExt;
use std::os::unix::io::{AsRawFd, RawFd};

use crate::buffer::Buffer;
use crate::id::ElementId;

/// Packs `(element, seq)` so flush/seek/shutdown can cancel by element prefix
/// (spec: cancellation by identity).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OpId(pub u64);

/// A file registered with the reactor. Milestone: one file per element, so the
/// handle is just the element id; multi-file handles carry a sub-index later.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FileHandle(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpKind {
    Read,
    Write,
}

/// The outcome of a completed op.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IoResult {
    /// Bytes transferred (0 on a read == EOF).
    Ok(usize),
    /// The op was cancelled before it ran.
    Cancelled,
    Err(ErrorKind),
}

/// A finished op handed back to the submitting element (spec: IO). The buffer rides
/// back with the completion — a completed read *is* a ready `Buffer`.
pub struct Completion {
    pub op: OpId,
    pub user: u64,
    pub result: IoResult,
    pub buf: Buffer,
}

/// A submitted op, forwarded from an element's outbox to the reactor. Public so
/// out-of-core reactor backends (e.g. io_uring) can consume it.
pub struct Submission {
    pub op: OpId,
    pub element: ElementId,
    pub kind: OpKind,
    pub file: FileHandle,
    pub offset: u64,
    pub buf: Buffer,
    pub user: u64,
}

/// The element-facing IO handle, borrowed from `Ctx` for the duration of a call.
/// Submissions are queued into the element's outbox and forwarded to the reactor by
/// the scheduler; completions are drained from its inbox.
pub struct Io<'c> {
    element: ElementId,
    registration: &'c mut Option<File>,
    inbox: &'c mut Vec<Completion>,
    outbox: &'c mut Vec<Submission>,
    next_op: &'c mut u64,
    credits: u32,
}

impl<'c> Io<'c> {
    pub(crate) fn new(
        element: ElementId,
        registration: &'c mut Option<File>,
        inbox: &'c mut Vec<Completion>,
        outbox: &'c mut Vec<Submission>,
        next_op: &'c mut u64,
        credits: u32,
    ) -> Self {
        Self { element, registration, inbox, outbox, next_op, credits }
    }

    /// In-flight budget: how many more ops this element may keep outstanding
    /// (spec: submission credits = pool slots + downstream space).
    pub fn credits(&self) -> u32 {
        self.credits
    }

    /// Register a file with the reactor (at `start`), returning its handle.
    pub fn register(&mut self, file: File) -> FileHandle {
        *self.registration = Some(file);
        FileHandle(self.element.0)
    }

    fn next_op_id(&mut self) -> OpId {
        let id = OpId(((self.element.0 as u64) << 32) | *self.next_op);
        *self.next_op += 1;
        id
    }

    /// The scheduler takes the outbox whole every pass, so the first push of a
    /// pass lands in a fresh (capacity-0) vec. Pre-size it for the element's
    /// credit budget — the natural per-pass submission bound — so a credits-deep
    /// burst costs one allocation, not log2(credits) doublings (ZERO-COPY.md
    /// stage 4.2; full reuse needs the outbox to travel back, a `Ctx` change).
    fn reserve_outbox(&mut self) {
        if self.outbox.is_empty() {
            self.outbox.reserve((self.credits as usize).clamp(4, 64));
        }
    }

    /// Submit a read of up to `buf.capacity()` bytes at `offset` into `buf`.
    pub fn submit_read(&mut self, file: FileHandle, offset: u64, buf: Buffer, user: u64) -> OpId {
        let op = self.next_op_id();
        self.reserve_outbox();
        self.outbox.push(Submission {
            op,
            element: self.element,
            kind: OpKind::Read,
            file,
            offset,
            buf,
            user,
        });
        op
    }

    /// Submit a write of `buf`'s used bytes at `offset`.
    pub fn submit_write(&mut self, file: FileHandle, offset: u64, buf: Buffer, user: u64) -> OpId {
        let op = self.next_op_id();
        self.reserve_outbox();
        self.outbox.push(Submission {
            op,
            element: self.element,
            kind: OpKind::Write,
            file,
            offset,
            buf,
            user,
        });
        op
    }

    /// Drain the next completed op for this element, if any.
    pub fn next_completion(&mut self) -> Option<Completion> {
        if self.inbox.is_empty() {
            None
        } else {
            Some(self.inbox.remove(0))
        }
    }
}

// --- streaming page-cache hygiene: raw syscall shims -------------------------
//
// Syscall numbers verified against the kernel's x86_64 table
// (`arch/x86/entry/syscalls/syscall_64.tbl`):
//
//     221  common  fadvise64        sys_fadvise64
//     277  common  sync_file_range  sys_sync_file_range
//
// On x86_64 `posix_fadvise(3)` is implemented by the `fadvise64` syscall with the
// argument order `(fd, offset, len, advice)` — the alpha/arm 32-bit register-pair
// reorderings (`fadvise64_64`, `sync_file_range2`) do not exist here.

/// `POSIX_FADV_SEQUENTIAL` (`man 2 posix_fadvise`): the application expects to
/// read the file sequentially — the kernel doubles the readahead window.
/// Value from `include/uapi/linux/fadvise.h` (x86_64; only s390x deviates, and
/// only for `DONTNEED`/`NOREUSE`).
pub const POSIX_FADV_SEQUENTIAL: i32 = 2;
/// `POSIX_FADV_DONTNEED` (`man 2 posix_fadvise`): the data will not be accessed
/// again — free the cached pages. **Dirty or under-writeback pages are silently
/// left in place** (the kernel may kick off their writeback but does not wait),
/// which is why the write path syncs a window *before* dropping it. Partial
/// pages at either edge of the range are preserved, so byte-exact offsets are
/// safe to pass.
pub const POSIX_FADV_DONTNEED: i32 = 4;
/// `SYNC_FILE_RANGE_WRITE` (`man 2 sync_file_range`): initiate write-out of the
/// range's dirty pages not already under write-out, **without waiting** — an
/// asynchronous writeback kick. Value from `include/uapi/linux/fs.h`. (This is a
/// cache-hygiene accelerator only; the man page's durability warnings concern
/// uses this module does not make.)
pub const SYNC_FILE_RANGE_WRITE: u32 = 2;

/// Four-argument Linux syscall via the `syscall` instruction (x86_64 SysV ABI:
/// nr in `rax`, args in `rdi/rsi/rdx/r10`; the kernel clobbers `rcx`/`r11`).
/// Returns the raw result: `>= 0` success, `-errno` failure.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn syscall4(nr: u64, a1: u64, a2: u64, a3: u64, a4: u64) -> i64 {
    let ret: i64;
    // SAFETY: both syscalls used here (`fadvise64`, `sync_file_range`) take only
    // integer arguments — no pointers into userspace — so the kernel cannot write
    // to our memory; the clobbered registers are declared.
    unsafe {
        core::arch::asm!(
            "syscall",
            inlateout("rax") nr => ret,
            in("rdi") a1,
            in("rsi") a2,
            in("rdx") a3,
            in("r10") a4,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    ret
}

/// `fadvise64(2)` — declare an access pattern / drop cached pages for
/// `[offset, offset + len)` (`len == 0` means "to end of file").
/// Non-Linux or non-x86_64 targets compile this to a successful no-op: hygiene
/// is advice, and a platform without the shim just keeps default kernel caching.
pub fn fadvise(fd: RawFd, offset: u64, len: u64, advice: i32) -> std::io::Result<()> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let ret = syscall4(221, fd as u64, offset, len, advice as u64);
        if ret < 0 {
            return Err(std::io::Error::from_raw_os_error(-ret as i32));
        }
        Ok(())
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = (fd, offset, len, advice);
        Ok(())
    }
}

/// `sync_file_range(2)` — writeback control for `[offset, offset + nbytes)`
/// (`nbytes == 0` means "to end of file"). Same no-op fallback as [`fadvise`].
pub fn sync_file_range(fd: RawFd, offset: u64, nbytes: u64, flags: u32) -> std::io::Result<()> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let ret = syscall4(277, fd as u64, offset, nbytes, flags as u64);
        if ret < 0 {
            return Err(std::io::Error::from_raw_os_error(-ret as i32));
        }
        Ok(())
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = (fd, offset, nbytes, flags);
        Ok(())
    }
}

/// Hygiene window: how much forward progress accumulates before the kernel is
/// hinted. 64 MiB is far above any readahead window (default 128 KiB, a few MiB
/// on tuned systems), so dropping behind the position never races readahead; it
/// keeps the steady-state resident footprint at ~2 windows per streamed file
/// (≤ 192 MiB for a reader + writer pair — noise under any sane memory cap); and
/// it amortizes to ~3 syscalls per 64 MiB — unmeasurable against the IO itself.
pub const HYGIENE_WINDOW: u64 = 64 << 20;

/// Per-file sequential-streaming cache hygiene (`man 2 posix_fadvise`,
/// `man 2 sync_file_range` — see the constants above for exact semantics).
///
/// Fed with *completed* op ranges (in file order), it issues, per [window] of
/// forward progress:
///
/// - **reads**: `POSIX_FADV_DONTNEED` on everything behind the read position —
///   pages read once and left behind are clean, so they drop immediately.
/// - **writes**: the two-window dance. `SYNC_FILE_RANGE_WRITE` kicks async
///   writeback of the just-completed window, and `DONTNEED` drops the window
///   *behind the previous sync point* — whose writeback started a full window
///   ago and has had a window's worth of IO time to finish. `DONTNEED` on
///   still-dirty pages silently does nothing (see above), so a slow disk
///   degrades to "cache shrinks a little later", never to data loss.
///
/// **Monotonicity guard**: hygiene assumes strictly forward streaming. Each side
/// tracks its high-water mark; one op starting below it (a backward seek, an
/// overwrite) disables that side *permanently for this file* — a random-access
/// pattern (a future seeking source) must not be pessimized into cache thrash,
/// and re-enabling after a seek would need to know which dropped ranges will be
/// revisited, which nothing here can know. All calls are best-effort: hints
/// never fail the pipeline.
pub struct StreamHygiene {
    window: u64,
    read: ReadSide,
    write: WriteSide,
}

struct ReadSide {
    monotonic: bool,
    started: bool,
    /// End of the highest completed read.
    end: u64,
    /// Everything below this has been `DONTNEED`ed.
    dropped: u64,
}

struct WriteSide {
    monotonic: bool,
    started: bool,
    /// End of the highest completed write.
    end: u64,
    /// Start of the newest window: `[synced, end)` has not been writeback-kicked yet.
    synced: u64,
    /// Everything below this has been `DONTNEED`ed; `[dropped, synced)` is the
    /// previous window, syncing in the background.
    dropped: u64,
}

impl StreamHygiene {
    pub fn new() -> Self {
        Self::with_window(HYGIENE_WINDOW)
    }

    /// Same policy with a custom window — tests use small windows to exercise
    /// the sync/drop branches on small files.
    pub fn with_window(window: u64) -> Self {
        Self {
            window: window.max(1),
            read: ReadSide { monotonic: true, started: false, end: 0, dropped: 0 },
            write: WriteSide { monotonic: true, started: false, end: 0, synced: 0, dropped: 0 },
        }
    }

    pub fn set_window(&mut self, window: u64) {
        self.window = window.max(1);
    }

    /// Permanently disable read-side hygiene (backend saw a non-monotonic read).
    pub fn disable_read(&mut self) {
        self.read.monotonic = false;
    }

    /// Permanently disable write-side hygiene (backend saw a non-monotonic write).
    pub fn disable_write(&mut self) {
        self.write.monotonic = false;
    }

    /// Record a completed read of `[offset, offset + len)`.
    pub fn on_read(&mut self, fd: RawFd, offset: u64, len: usize) {
        if !self.read.monotonic || len == 0 {
            return;
        }
        if !self.read.started {
            // First op fixes the origin: a source that starts mid-file (a seek
            // target) must not drop pages below data it never streamed.
            self.read.started = true;
            self.read.dropped = offset;
            self.read.end = offset;
        }
        if offset < self.read.end {
            self.read.monotonic = false; // backward seek: hygiene off for this file
            return;
        }
        self.read.end = offset + len as u64;
        if self.read.end - self.read.dropped >= self.window {
            // Clean, consumed-once pages: drop the whole region behind the read
            // position (edge partial pages are preserved by the kernel).
            let _ = fadvise(fd, self.read.dropped, self.read.end - self.read.dropped, POSIX_FADV_DONTNEED);
            self.read.dropped = self.read.end;
        }
    }

    /// Record a completed write of `[offset, offset + len)`.
    pub fn on_write(&mut self, fd: RawFd, offset: u64, len: usize) {
        if !self.write.monotonic || len == 0 {
            return;
        }
        if !self.write.started {
            self.write.started = true;
            self.write.dropped = offset;
            self.write.synced = offset;
            self.write.end = offset;
        }
        if offset < self.write.end {
            self.write.monotonic = false;
            return;
        }
        self.write.end = offset + len as u64;
        if self.write.end - self.write.synced >= self.window {
            // Kick async writeback of the just-completed window now, so its pages
            // are clean by the time it becomes "the window behind us"...
            let _ = sync_file_range(fd, self.write.synced, self.write.end - self.write.synced, SYNC_FILE_RANGE_WRITE);
            // ...and drop the previous window, whose writeback was kicked a full
            // window ago (never DONTNEED freshly-dirty pages — it no-ops).
            if self.write.synced > self.write.dropped {
                let _ = fadvise(fd, self.write.dropped, self.write.synced - self.write.dropped, POSIX_FADV_DONTNEED);
                self.write.dropped = self.write.synced;
            }
            self.write.synced = self.write.end;
        }
    }
}

impl Default for StreamHygiene {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds a reactor. Each thread group creates its own on its own thread, so the
/// factory is `Send + Sync` but the reactors it produces need not be `Send`
/// (io_uring holds thread-bound rings/pointers). Spec: IO — the reactor is the
/// portability boundary, one per thread group.
pub type ReactorFactory =
    std::sync::Arc<dyn Fn() -> std::io::Result<Box<dyn Reactor>> + Send + Sync>;

/// A reactor backend: executes submitted IO ops and returns their completions
/// (spec: IO — the reactor is the portability boundary). Each thread group owns one.
pub trait Reactor {
    /// Register an element's file (called at `start`).
    fn set_file(&mut self, element: ElementId, file: File);
    /// Enqueue submitted ops.
    fn submit(&mut self, subs: Vec<Submission>);
    /// Cancel a pending op by id (flush/seek/shutdown path).
    fn cancel(&mut self, op: OpId);
    /// Whether the reactor is fully idle: nothing queued *and* nothing in flight.
    /// (An async backend may have submitted ops the kernel hasn't completed yet.)
    fn is_idle(&self) -> bool;
    /// Make progress: execute/reap ops, returning completions tagged with the
    /// owning element.
    fn run_once(&mut self) -> Vec<(ElementId, Completion)>;
}

/// The default synchronous positioned-IO reactor (dependency-free). Files are
/// registered per element; submissions execute in FIFO order on [`run_once`].
///
/// Every registered file gets `POSIX_FADV_SEQUENTIAL` (elements stream) and a
/// [`StreamHygiene`] tracker fed from completed ops — executing in submission
/// order, this backend's completions are naturally monotonic for sequential
/// elements, and one backward seek flips that file's hygiene off (see
/// [`StreamHygiene`]).
pub struct SyncReactor {
    files: HashMap<u32, ReactorFile>,
    pending: Vec<Submission>,
    cancelled: Vec<OpId>,
    /// Drain scratch for [`run_once`]: `pending` swaps in here so both vecs keep
    /// their capacity across passes (ZERO-COPY.md stage 4.2 — the per-pass
    /// `mem::take` + drop was measured churn on the movie remux).
    scratch: Vec<Submission>,
    hygiene_window: u64,
}

struct ReactorFile {
    file: File,
    hygiene: StreamHygiene,
}

impl SyncReactor {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            pending: Vec::new(),
            cancelled: Vec::new(),
            scratch: Vec::new(),
            hygiene_window: HYGIENE_WINDOW,
        }
    }

    /// Override the hygiene window (bytes) for current and future registrations —
    /// tests shrink it to exercise the sync/drop branches on small files.
    pub fn set_hygiene_window(&mut self, bytes: u64) {
        self.hygiene_window = bytes;
        for f in self.files.values_mut() {
            f.hygiene.set_window(bytes);
        }
    }
}

impl Default for SyncReactor {
    fn default() -> Self {
        Self::new()
    }
}

impl Reactor for SyncReactor {
    fn set_file(&mut self, element: ElementId, file: File) {
        // Registered files stream (`man 2 posix_fadvise`): double the readahead
        // window up front. Harmless on a write-only file (readahead never fires),
        // and on the platforms where the shim is a no-op. Best-effort: advice.
        let _ = fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_SEQUENTIAL);
        self.files.insert(
            element.0,
            ReactorFile { file, hygiene: StreamHygiene::with_window(self.hygiene_window) },
        );
    }

    fn submit(&mut self, mut subs: Vec<Submission>) {
        // Steady state `pending` is empty here (run_once drains every pass): adopt
        // the larger allocation instead of copying into a smaller one, then append
        // whatever the swap left behind (nothing, in that common case).
        if self.pending.is_empty() && self.pending.capacity() < subs.capacity() {
            std::mem::swap(&mut self.pending, &mut subs);
        }
        self.pending.append(&mut subs);
    }

    fn cancel(&mut self, op: OpId) {
        self.cancelled.push(op);
    }

    fn is_idle(&self) -> bool {
        // The synchronous backend completes everything within run_once, so it is
        // idle exactly when nothing is queued.
        self.pending.is_empty()
    }

    fn run_once(&mut self) -> Vec<(ElementId, Completion)> {
        // Swap `pending` into the drain scratch so its capacity survives the pass
        // (`scratch` is always fully drained below, so this never reorders ops).
        debug_assert!(self.scratch.is_empty());
        std::mem::swap(&mut self.pending, &mut self.scratch);
        // The returned Vec is forced fresh by the trait (the scheduler consumes it
        // by value); see ZERO-COPY.md stage 4.2 for the caller-buffer variant.
        let mut out = Vec::with_capacity(self.scratch.len());
        for mut s in self.scratch.drain(..) {
            let result = if self.cancelled.contains(&s.op) {
                IoResult::Cancelled
            } else {
                match self.files.get_mut(&s.file.0) {
                    None => IoResult::Err(ErrorKind::NotFound),
                    Some(rf) => match s.kind {
                        OpKind::Read => match rf.file.read_at(s.buf.memory.as_mut_full(), s.offset) {
                            Ok(n) => {
                                s.buf.memory.set_len(n);
                                rf.hygiene.on_read(rf.file.as_raw_fd(), s.offset, n);
                                IoResult::Ok(n)
                            }
                            Err(e) => IoResult::Err(e.kind()),
                        },
                        OpKind::Write => {
                            let n = s.buf.memory.len();
                            match rf.file.write_all_at(s.buf.memory.data(), s.offset) {
                                Ok(()) => {
                                    rf.hygiene.on_write(rf.file.as_raw_fd(), s.offset, n);
                                    IoResult::Ok(n)
                                }
                                Err(e) => IoResult::Err(e.kind()),
                            }
                        }
                    },
                }
            };
            out.push((
                s.element,
                Completion { op: s.op, user: s.user, result, buf: s.buf },
            ));
        }
        self.cancelled.clear();
        out
    }
}
