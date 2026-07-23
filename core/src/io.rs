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

use std::collections::HashMap;
use std::fs::File;
use std::io::ErrorKind;
use std::os::unix::fs::FileExt;

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

    /// Submit a read of up to `buf.capacity()` bytes at `offset` into `buf`.
    pub fn submit_read(&mut self, file: FileHandle, offset: u64, buf: Buffer, user: u64) -> OpId {
        let op = self.next_op_id();
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
pub struct SyncReactor {
    files: HashMap<u32, File>,
    pending: Vec<Submission>,
    cancelled: Vec<OpId>,
}

impl SyncReactor {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            pending: Vec::new(),
            cancelled: Vec::new(),
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
        self.files.insert(element.0, file);
    }

    fn submit(&mut self, subs: Vec<Submission>) {
        self.pending.extend(subs);
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
        let subs = std::mem::take(&mut self.pending);
        let mut out = Vec::with_capacity(subs.len());
        for mut s in subs {
            let result = if self.cancelled.contains(&s.op) {
                IoResult::Cancelled
            } else {
                match self.files.get(&s.file.0) {
                    None => IoResult::Err(ErrorKind::NotFound),
                    Some(f) => match s.kind {
                        OpKind::Read => match f.read_at(s.buf.memory.as_mut_full(), s.offset) {
                            Ok(n) => {
                                s.buf.memory.set_len(n);
                                IoResult::Ok(n)
                            }
                            Err(e) => IoResult::Err(e.kind()),
                        },
                        OpKind::Write => {
                            let n = s.buf.memory.len();
                            match f.write_all_at(s.buf.memory.data(), s.offset) {
                                Ok(()) => IoResult::Ok(n),
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
