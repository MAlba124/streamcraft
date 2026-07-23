//! Network elements.
//!
//! Milestone 2: [`HttpSrc`] downloads a file over plain HTTP so `httpsrc ! filesink`
//! writes it to disk (spec: Milestone applications §2). Unlike the reactor-native
//! IO elements, this first version does **blocking** networking directly on a
//! [`std::net::TcpStream`] inside the element: it is an `Active` element, so it owns
//! its group's thread and may block in `process()`. Routing sockets through the
//! reactor (completion-shaped, like `filesrc`/`filesink`) is a follow-up once the
//! reactor's socket path lands.
//!
//! v1 limitations, all documented on [`HttpSrc`]:
//! - `http://` only — no TLS/`https`.
//! - Reads the body to EOF (relies on `Connection: close`); `Content-Length` and
//!   chunked transfer-encoding are not interpreted.

mod httpsrc;

pub use httpsrc::HttpSrc;
