//! The introspection protocol + server (spec: Introspection protocol and
//! scraft-scope). A compact, length-prefixed binary protocol of POD frames served
//! over a Unix socket, feature-gated in core: it costs nothing until a client
//! connects, and observation stays observation — every reply is a snapshot of data
//! the pipeline already maintains, never a lock on a streaming path.
//!
//! Module map (spec: scraft-scope — the inspector):
//! - [`wire`] — the pinned wire format: frame headers, rows, primitives, and BOTH
//!   encode and decode helpers (the scope/ GUI client consumes this as public API).
//! - [`strtab`] — the per-connection string table (names sent once as `StrDef`,
//!   referenced by dense u32 thereafter).
//! - [`snapshot`] — an owned, interner-free image of the topology, built on the app
//!   thread so the server never touches the pipeline's live interners.
//! - [`tap`] — the two hot-path taps: [`tap::BusTapRow`] (a POD image of a
//!   [`BusMessage`](crate::bus::BusMessage)) and [`tap::LogTapRegistry`] (log-record
//!   fan-out to subscribed clients).
//! - [`server`] — the accept thread + per-client blocking loop.

pub mod server;
pub mod snapshot;
pub mod strtab;
pub mod tap;
pub mod wire;

pub use server::{Handles, IntrospectServer, IntrospectShared};
