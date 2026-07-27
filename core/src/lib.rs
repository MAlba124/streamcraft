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
//! - [`log`], [`counters`] — observability (spec: Debuggability, Taps)
//! - [`props`] — dynamic element properties (spec: Dynamic element properties)
//! - [`registry`] — the opt-in element-by-name table + parse-launch (spec: Plugins)
//! - [`harness`] — the threadless, mock-clocked single-element test rig (spec: Testing —
//!   Element harness; it *is* the inline caller)

// The scratch `Arena` implements the nightly `std::alloc::Allocator` trait so it can back a
// `Vec<T, &Arena>` — letting decoders allocate per-frame scratch from the pipeline's
// per-`process()` arena (reset by the scheduler) instead of the heap. Workspace pins nightly.
#![feature(allocator_api)]
// `unsafe` is permitted only in the audited modules: `memory` and the SPSC ring
// (loom+miri covered), plus `io`'s raw-syscall shims (integer-only args, no
// userspace pointers — see io.rs's justification header). `deny` (not `forbid`)
// so those modules can locally `#[allow(unsafe_code)]`.
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
pub mod harness;
pub mod id;
/// The introspection protocol + server (spec: Introspection protocol and scraft-scope).
/// Feature-gated; zero cost — and zero code — when `introspect` is off.
#[cfg(feature = "introspect")]
pub mod introspect;
pub mod io;
pub mod log;
pub mod memory;
pub mod pipeline;
pub mod props;
pub mod registry;
pub mod ring;
pub mod sched;
pub mod time;
