//! Depayloaders: RTP payloads → codec access units.
//!
//! **STUBS — implementation is agent B's scope.** Pure functions/state
//! machines over parsed packet fields — no framework types, no sockets. The
//! element wrappers feed them via [`crate::packet::RtpPacket`].

pub mod h264;
pub mod opus;
