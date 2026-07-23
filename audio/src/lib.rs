//! streamcraft-audio — POD audio format descriptions and typed *views* over the
//! opaque buffer: sample formats, rates, channel layouts, planar spans
//! (spec: Crate layout). No new buffer types, no traits — free functions and plain
//! structs. Views borrow, so misuse is a compile error and they cost nothing over
//! raw offset math. Operate on whole batches (SIMD across the span).
//!
//! TODO(milestone 3+): sample views feeding the hand-written FLAC/Opus codecs.

#![allow(dead_code)]

pub mod format;
pub mod wav;

pub use format::{
    AudioFormat, AudioFrameRef, SampleFormat, FAMILY, FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE,
    RAW_ANY_OFFER,
};
pub use wav::{parse_wav_header, write_pcm_wav, WavError, WavHeader, WavParse};
