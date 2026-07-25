//! Sequence-number arithmetic: 16-bit wrapping comparison and extended
//! (unwrapped) sequence numbers (RFC 3550 appendix A.1).
//!
//! **STUB — implementation is agent A's scope.** API sketch; adjust as the
//! RFC demands (this module is owned by one agent, signatures are free).

#![allow(dead_code, unused_variables)]

/// Tracks the wrap cycles of a 16-bit sequence to produce a monotonically
/// comparable extended sequence (A.1's `cycles + seq`).
#[derive(Debug, Default)]
pub struct ExtendedSeq;

impl ExtendedSeq {
    /// Feed the next received 16-bit sequence; returns the extended value.
    pub fn extend(&mut self, seq: u16) -> u64 {
        todo!("agent A: RFC 3550 A.1")
    }
}
