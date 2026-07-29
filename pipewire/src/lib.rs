//! pf-pipewire — a PipeWire audio sink plugin for profluens (spec: Milestone
//! applications — play an audio file). It lives outside core so the core stays
//! dependency-free; here we bind the official `pipewire` crate (libpipewire) rather than
//! reimplement a device backend.
//!
//! [`PipeWireAudioSink`] is an active element that renders interleaved PCM through
//! PipeWire, configuring itself from the runtime `audio/raw` format a decoder announces
//! (spec: Formats — dynamic caps) and pacing the graph by backpressure.
//!
//! [`native`] is the from-scratch native-protocol client (no libpipewire) that will replace
//! the `libpipewire` binding below — see its module docs for the migration plan.

pub mod native;
mod ring;
mod sink;

pub use sink::{AudioControl, PipeWireAudioSink};
