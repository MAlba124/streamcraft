//! sc-pipewire — a PipeWire audio sink plugin for streamcraft (spec: Milestone
//! applications — play an audio file). It lives outside core so the core stays
//! dependency-free; here we bind the official `pipewire` crate (libpipewire) rather than
//! reimplement a device backend.
//!
//! [`PipeWireAudioSink`] is an active element that renders interleaved PCM through
//! PipeWire, configuring itself from the runtime `audio/raw` format a decoder announces
//! (spec: Formats — dynamic caps) and pacing the graph by backpressure.

mod ring;
mod sink;

pub use sink::PipeWireAudioSink;
