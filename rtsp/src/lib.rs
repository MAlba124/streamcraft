//! sc-rtsp — RTSP 1.0 (RFC 2326) client + SDP (RFC 8866) parser, from scratch.
//!
//! Layered like the other protocol crates (sc-http, sc-rtp): **pure protocol
//! modules** with no framework types — [`sdp`] (the DESCRIBE payload:
//! session/media sections, `rtpmap`/`fmtp`/`control` attributes, plus the
//! RFC 4648 base64 H.264 `sprop-parameter-sets` needs), [`client`] (a
//! blocking request/response client over `TcpStream`: OPTIONS/DESCRIBE/
//! SETUP/PLAY/PAUSE/TEARDOWN/GET_PARAMETER, CSeq/Session tracking, Basic +
//! Digest auth per RFC 2617 with MD5 from RFC 1321 in an in-tree `md5`
//! module), and
//! [`interleaved`] (the `$`-framed TCP channel demuxer of RFC 2326 §10.12)
//! — for a thin `rtspsrc` element (not in this crate yet) to wire into
//! pipelines (`rtspsrc ! rtpsession ! rtp*depay ! <decoder>`).
//!
//! Specs are checked into `spec/`: RFC 2326 (RTSP 1.0), RFC 8866 (SDP),
//! RFC 2617 (Basic/Digest auth), RFC 1321 (MD5), RFC 4648 (base64).
//! Clean-room: written against the RFCs alone; every non-obvious rule cites
//! its section at the point of use.

pub mod client;
pub mod interleaved;
mod md5;
pub mod sdp;
