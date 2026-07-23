//! sc-http — a hand-written, dependency-free HTTP source plugin for streamcraft.
//!
//! Milestone 2: [`HttpSrc`] downloads a file over plain HTTP so `HttpSrc ! FileSink`
//! writes it to disk (spec: Milestone applications §2). Custom and std-only — no
//! networking crate — and deliberately a **separate plugin**: HTTPS will eventually
//! need a TLS library, and the framework crates must not drag that in.
//!
//! It implements HTTP/1.1 against the specs checked into `spec/`: **RFC 9112**
//! (HTTP/1.1 message syntax) and **RFC 9110** (HTTP semantics — status codes,
//! header fields).
//!
//! v1 limitations (documented on [`HttpSrc`]): `http://` only (no TLS); reads the
//! body to EOF relying on `Connection: close` — `Content-Length` and chunked
//! transfer-encoding are not yet interpreted.

mod httpsrc;

pub use httpsrc::HttpSrc;
