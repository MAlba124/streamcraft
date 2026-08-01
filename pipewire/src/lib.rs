//! pf-pipewire — a PipeWire audio sink plugin for profluens (spec: Milestone
//! applications — play an audio file). It lives outside core so the core stays
//! dependency-free; here we bind the official `pipewire` crate (libpipewire) rather than
//! reimplement a device backend.
//!
//! [`PipeWireAudioSink`] is an active element that renders interleaved PCM through
//! PipeWire, configuring itself from the runtime `audio/raw` format a decoder announces
//! (spec: Formats — dynamic caps) and pacing the graph by backpressure.
//!
//! [`AudioOut`] is the **application-owned** audio output the gapless design is built on
//! (spec: gapless.md, Phase 2): one device, one ring, one clock, outliving any individual
//! pipeline. A sink built with [`PipeWireAudioSink::with_output`] attaches to it, streams,
//! and detaches at EOS *without draining*, so the next track's audio queues behind the tail
//! still in flight and the boundary is sample-continuous. The single-owner constructor is
//! untouched and remains the default for every non-gapless pipeline.
//!
//! [`native`] is the from-scratch native-protocol client (no libpipewire) that will replace
//! the `libpipewire` binding below — see its module docs for the migration plan.

//! [`probe`] is the seek-latency diagnostic: one timestamp per layer between a seek being
//! issued and the first re-primed sample being pulled by the device callback. Off (and
//! effectively free) unless a caller arms it.

pub mod native;
pub mod out;
pub mod probe;
mod pw_backend;
mod ring;
mod sink;

pub use out::{AudioOut, AudioOutConfig, AudioOutHandle, CanonicalFormat, SampleFormat};
pub use sink::{AudioControl, AudioDeviceClock, PipeWireAudioSink};
