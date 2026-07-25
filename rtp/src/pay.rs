//! Payloaders: codec access units → RTP payloads. The send path's other half
//! — implemented now for **round-trip tests** against [`crate::depay`] (the
//! cheapest strong correctness pin); send *elements* are a later session.
//!
//! **STUBS — implementation is agent B's scope.**

pub mod h264;
pub mod opus;
