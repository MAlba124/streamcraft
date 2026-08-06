//! The element layer: thin pipeline wiring over the pure protocol modules.
//!
//! `udpsrc ! rtpsession ! rtp<fmt>depay ! <decoder>` — the receive chain. The
//! elements own sockets, pads, and clock mapping; all wire logic lives in
//! [`crate::packet`]/[`crate::jitter`]/[`crate::rtcp`]/[`crate::depay`].

pub mod depay;
pub mod pay;
pub mod session;
pub mod udpsink;
pub mod udpsrc;

pub use depay::{RtpH264Depay, RtpOpusDepay};
pub use pay::RtpH264Pay;
pub use session::{RtpSession, RtpStreamDesc};
pub use udpsink::UdpSink;
pub use udpsrc::UdpSrc;
