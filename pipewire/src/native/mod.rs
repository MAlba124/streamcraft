//! A **from-scratch, native-protocol PipeWire client** — no `libpipewire`, no C bindings. It
//! speaks the daemon's wire protocol directly over the `pipewire-0` Unix socket, in the same
//! raw-wire, zero-per-message-allocation, data-oriented style as `pf-present` speaks Wayland.
//!
//! # Why
//!
//! `pf-pipewire`'s [`crate::PipeWireAudioSink`] currently binds `libpipewire` (the `pipewire`
//! crate). That pulls a large C dependency and its per-message marshalling allocations into a
//! codebase whose whole thesis is hand-written, allocation-disciplined media plumbing
//! ([[alloc-ban]], [[streamcraft-rearchitecture]]) — the same reason `pf-present` dropped
//! libwayland/SDL. This module is the replacement's foundation.
//!
//! # Layers (bottom-up)
//!
//! - [`pod`] — the SPA POD serialization format (every value on the wire is a POD). A reused-
//!   buffer builder + a zero-copy parser. The single most-reused primitive.
//! - [`wire`] — the 16-byte message header framing (`id`, `opcode|size`, `seq`, `n_fds`) around
//!   a POD `Struct` payload.
//! - [`conn`] — the `AF_UNIX` connection with `SCM_RIGHTS` fd passing (buffer memfds, activation
//!   eventfds) and reused send/recv buffers.
//! - [`proto`] — the Core/Client/Registry interfaces + the [`proto::PwClient`] handshake and
//!   registry enumeration.
//!
//! # Status
//!
//! **Milestone 1 (this):** connect, handshake, sync round-trips, enumerate the registry —
//! validated against the live daemon by the `native_info` example. This proves the framing +
//! POD codec against a real server.
//!
//! **Milestone 2 (in progress — [`audio`]):** the ClientNode real-time path. `Core.CreateObject`
//! a client node, `Update`/`PortUpdate`, `Format`/`Buffers` negotiation (POD `Object`/`Choice`),
//! `AddMem` memfd mapping, `PortUseBuffers`/`PortSetIO`, and the `Transport`/`SetActivation`
//! records all work and are validated against the live daemon through `Command Start`. The final
//! per-cycle driver→client RT trigger needs session-manager-*managed* linking (a bare client-node
//! is not adapter-wrapped); see [`audio`]'s module docs for the exact remaining work. Porting
//! [`crate::PipeWireAudioSink`] onto this (reusing the lock-free [`crate::ring`] + the clock/
//! pause/seek contract) and dropping `libpipewire` follow once playback is closed out.
//!
//! See `pipewire/REFERENCES.md` for the protocol/POD references these modules implement.

pub mod audio;
pub mod conn;
pub mod pod;
pub mod proto;
pub mod spa;
pub mod wire;

pub use audio::{play, AudioConfig};
pub use conn::Connection;
pub use proto::{Global, PwClient, PwError, ServerInfo};
pub use spa::SampleFormat;
