//! The reactor submit/complete contract (spec: IO: built for the io_uring era).
//!
//! Elements never touch a reactor directly. Inside `process()` they use the
//! [`Io`] handle from `ctx.io()` to **register** files, **submit** read/write ops,
//! and **drain completions** — the completion-shaped API the whole design hangs off.
//!
//! This module ships a **synchronous, positioned-IO** [`Reactor`] backend: ops carry
//! an explicit offset (`read_at`/`write_at`), so they are order-independent and
//! seek-ready, and are executed by the scheduler between element passes. It is
//! dependency-free and has no concurrency. A truly-async backend (a thread pool,
//! then io_uring with registered buffers and one `io_uring_enter` per wakeup)
//! replaces the executor behind this identical interface — element code is unchanged.

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
enum OpKind {
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
    Err(std::io::ErrorKind),
}

/// A finished op handed back to the submitting element (spec: IO). The buffer rides
/// back with the completion — a completed read *is* a ready `Buffer`.
pub struct Completion {
    pub op: OpId,
    pub user: u64,
    pub result: IoResult,
    pub buf: Buffer,
}

/// A submitted-but-not-yet-executed op. Internal to the reactor plumbing.
pub(crate) struct Submission {
    op: OpId,
    element: ElementId,
    kind: OpKind,
    file: FileHandle,
    offset: u64,
    buf: Buffer,
    user: u64,
}

/// The element-facing IO handle, borrowed from `Ctx` for the duration of a call.
/// Submissions are queued into the element's outbox and forwarded to the reactor by
/// the scheduler; completions are drained from its inbox.
pub struct Io<'c> {
    element: ElementId,
    registration: &'c mut Option<std::fs::File>,
    inbox: &'c mut Vec<Completion>,
    outbox: &'c mut Vec<Submission>,
    next_op: &'c mut u64,
    credits: u32,
}

impl<'c> Io<'c> {
    pub(crate) fn new(
        element: ElementId,
        registration: &'c mut Option<std::fs::File>,
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
    pub fn register(&mut self, file: std::fs::File) -> FileHandle {
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

/// The synchronous positioned-IO reactor, owned by the scheduler. Files are
/// registered per element; submissions execute in FIFO order on [`run_once`].
pub struct Reactor {
    files: Vec<Option<std::fs::File>>,
    pending: Vec<Submission>,
    cancelled: Vec<OpId>,
}

impl Reactor {
    pub fn new(n_elements: usize) -> Self {
        Self {
            files: (0..n_elements).map(|_| None).collect(),
            pending: Vec::new(),
            cancelled: Vec::new(),
        }
    }

    pub fn set_file(&mut self, element: ElementId, file: std::fs::File) {
        self.files[element.0 as usize] = Some(file);
    }

    pub(crate) fn submit(&mut self, subs: Vec<Submission>) {
        self.pending.extend(subs);
    }

    /// Cancel a pending op by id (flush/seek/shutdown path).
    pub fn cancel(&mut self, op: OpId) {
        self.cancelled.push(op);
    }

    pub fn pending_is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Execute all pending ops, returning their completions tagged with the owning
    /// element. Positioned IO, so completion order does not affect correctness.
    pub fn run_once(&mut self) -> Vec<(ElementId, Completion)> {
        let subs = std::mem::take(&mut self.pending);
        let mut out = Vec::with_capacity(subs.len());
        for mut s in subs {
            let result = if self.cancelled.contains(&s.op) {
                IoResult::Cancelled
            } else {
                match self.files[s.file.0 as usize].as_ref() {
                    None => IoResult::Err(std::io::ErrorKind::NotFound),
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
