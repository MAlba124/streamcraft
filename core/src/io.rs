//! The reactor submit/complete contract (spec: IO: built for the io_uring era).
//! Core defines only the contract; the concrete io_uring (Linux) and epoll-fallback
//! reactors live in `streamcraft-elements` behind features. One reactor per thread
//! group, shared by every IO element placed there.

use crate::buffer::Buffer;

/// Portable raw-fd alias for the skeleton. The platform target is Linux; real code
/// uses `std::os::fd::RawFd` (also `i32`).
pub type RawFd = i32;

/// Packs `(element, seq)` so flush/seek/shutdown can cancel by element prefix
/// (spec: cancellation by identity).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OpId(pub u64);

pub struct Completion {
    pub op: OpId,
    /// Raw errno-style; structured at the element edge.
    pub result: i32,
    /// A completed read lands directly as a ready `Buffer` (into pool memory).
    pub buffer: Option<Buffer>,
}

/// Handle an `Active` element uses to drive IO (spec: IO).
pub struct Io {
    _priv: (),
}

impl Io {
    /// In-flight budget = pool slots + downstream queue space. Backpressure into the
    /// kernel: no credits → nothing submitted.
    pub fn credits(&self) -> u32 {
        todo!("spec: IO")
    }

    pub fn read(&mut self, _fd: RawFd, _into: Buffer, _offset: u64) -> OpId {
        todo!("spec: IO")
    }

    pub fn cancel(&mut self, _op: OpId) {
        todo!("spec: IO")
    }

    /// Drained inside `process()`; completions group into a batch per wakeup.
    pub fn next(&mut self) -> Option<Completion> {
        todo!("spec: IO")
    }
}
