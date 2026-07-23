//! streamcraft-core — the dependency-free heart of streamcraft.
//!
//! The spec lives in `streamcraft.md` at the repo root; module ↔ spec map:
//!
//! - [`time`], [`id`], [`buffer`], [`batch`] — the vocabulary (spec: Buffer, Batching)
//! - [`memory`] — pools, allocator vtable, refcounted views (spec: Memory)
//! - [`format`] — the closed constraint algebra + solver (spec: Formats and negotiation)
//! - [`element`], [`ctx`] — the element contract (spec: Elements and pads, Aggregation)
//! - [`sched`] — thread groups, queues, reactors (spec: Scheduling, Queue internals)
//! - [`clock`] — clocks, `ClockWait`, slaving (spec: Clocking and synchronization)
//! - [`event`], [`bus`] — the two communication channels (spec: Events, queries, and the bus)
//! - [`pipeline`] — topology owner and public API (spec: The pipeline API)
//! - [`io`] — the reactor submit/complete contract (spec: IO)
//! - [`log`], [`counters`] — observability (spec: Debuggability)

// `unsafe` is permitted only in `memory` and the SPSC ring (audited, loom+miri
// covered) once they land. `deny` (not `forbid`) so those modules can locally
// `#[allow(unsafe_code)]`.
#![deny(unsafe_code)]
// Temporary: the skeleton defines the type vocabulary before the modules are wired
// together. Remove as the build order in streamcraft.md is worked through.
#![allow(dead_code)]

pub mod batch;
pub mod buffer;
pub mod bus;
pub mod clock;
pub mod counters;
pub mod ctx;
pub mod element;
pub mod error;
pub mod event;
pub mod format;
pub mod id;
pub mod io;
pub mod log;
pub mod memory;
pub mod pipeline;
pub mod sched;
pub mod time;
