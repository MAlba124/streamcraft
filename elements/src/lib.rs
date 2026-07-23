//! streamcraft-elements — the built-in, pure-Rust elements.
//!
//! Milestone 1: [`io::FileSrc`] ! [`io::FileSink`] (spec: Milestone applications).
//! The reference elements are kept exemplary — element authors copy the nearest one
//! (spec: Writing elements), so the nearest one must be perfect.
//!
//! (The HTTP source lives in its own `sc-http` plugin crate, not here — it will grow
//! a TLS dependency that must not leak into the framework crates.)

pub mod flow;
pub mod io;
pub mod testing;
