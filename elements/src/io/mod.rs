//! IO elements.
//!
//! Milestone 1 uses blocking `std::fs` IO inline on the driving thread. The
//! io_uring / epoll reactor (spec: IO) later replaces the blocking calls behind
//! these same elements — the element code above the submission layer is unchanged.

mod filesink;
mod filesrc;

pub use filesink::FileSink;
pub use filesrc::FileSrc;
