//! Test elements (spec: Testing — seedable patterns, assertion sinks).
//!
//! [`TestSrc`] generates a reproducible byte pattern with no IO, and [`TestSink`]
//! folds every received byte into an FNV-1a hash and counts them, so a golden test
//! can assert the pipeline transported the data with no loss, duplication, or
//! reordering. Both are also the only elements that don't touch files or sockets,
//! so they exercise the scheduler's non-IO path.

mod testsink;
mod testsrc;

pub use testsink::{TestSink, TestSinkStats};
pub use testsrc::TestSrc;

/// FNV-1a offset basis.
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One FNV-1a folding step. Shared by [`TestSink`] and golden tests so both compute
/// the same digest.
#[inline]
pub fn fold(hash: u64, byte: u8) -> u64 {
    (hash ^ byte as u64).wrapping_mul(FNV_PRIME)
}

/// The reproducible byte at stream position `index` produced by [`TestSrc`].
#[inline]
pub fn pattern_byte(index: u64) -> u8 {
    (index.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(23) >> 24) as u8
}
