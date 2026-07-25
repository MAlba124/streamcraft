//! sc-rtp — RTP/RTCP (RFC 3550) from scratch, receive-first.
//!
//! Layered like the other protocol crates (sc-http, sc-flac): **pure protocol
//! modules** with no framework types — [`packet`] (the wire view), [`seq`]
//! (extended sequence arithmetic), [`jitter`] (the reorder/dejitter state
//! machine), [`rtcp`] (compound reports, the SR NTP↔RTP sync anchor),
//! [`depay`]/[`pay`] (payload formats: H.264 per RFC 6184, Opus per
//! RFC 7587) — and thin **elements** on top wiring them into pipelines
//! (`udpsrc ! rtpsession ! rtp*depay ! <decoder>`).
//!
//! Specs are checked into `spec/`: RFC 3550 (RTP/RTCP), RFC 3551 (the AV
//! profile's static payload types), RFC 6184 (H.264), RFC 7587 (Opus).
//! Clean-room: written against the RFCs alone; every non-obvious rule cites
//! its section at the point of use.

pub mod depay;
pub mod jitter;
pub mod packet;
pub mod pay;
pub mod rtcp;
pub mod seq;
