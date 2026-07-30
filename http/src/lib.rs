//! pf-http — a hand-written HTTP source plugin for profluens.
//!
//! Milestone 2: [`HttpSrc`] downloads a file over HTTP(S) so `HttpSrc ! FileSink`
//! writes it to disk (spec: Milestone applications §2). The HTTP layer is custom
//! and std-only — no networking crate — and deliberately a **separate plugin**:
//! `https://` brings the one external dependency, TLS (rustls on the pure-Rust
//! graviola provider, see [`tls`]), which must not leak into the framework crates.
//!
//! It implements HTTP/1.1 against the specs checked into `spec/`: **RFC 9112**
//! (HTTP/1.1 message syntax) and **RFC 9110** (HTTP semantics — status codes,
//! header fields).

mod httpsrc;
mod tls;

pub use httpsrc::HttpSrc;
