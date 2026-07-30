//! IO elements.
//!
//! Elements are reactor-native: they submit reads/writes through `ctx.io()` and
//! drain completions in `process()` (spec: IO). The scheduler runs them against a
//! [`profluens_core::io::Reactor`] — the dependency-free `SyncReactor` by default,
//! or the hand-rolled [`IoUringReactor`] (Linux, `io-uring` feature) injected via
//! `Pipeline::set_reactor`.

mod filesink;
mod filesrc;
#[cfg(feature = "io-uring")]
mod uring;

pub use filesink::FileSink;
pub use filesrc::FileSrc;
#[cfg(feature = "io-uring")]
pub use uring::IoUringReactor;
