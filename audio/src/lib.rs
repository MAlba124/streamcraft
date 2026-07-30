//! profluens-audio — POD audio format descriptions and typed *views* over the
//! opaque buffer: sample formats, rates, channel layouts, planar spans
//! (spec: Crate layout). No new buffer types, no traits — free functions and plain
//! structs. Views borrow, so misuse is a compile error and they cost nothing over
//! raw offset math. Operate on whole batches (SIMD across the span).
//!
//! TODO(milestone 3+): sample views feeding the hand-written FLAC/Opus codecs.

#![feature(allocator_api)] // per-`process()` arena scratch (`ctx.scratch() -> &Arena: Allocator`)
#![allow(dead_code)]

pub mod convert;
pub mod convert_element;
pub mod downmix_element;
pub mod format;
pub mod quality;
pub mod resample;
pub mod resample_element;
pub mod wav;

pub use convert::{
    convert_interleaved, convert_interleaved_vec, converted_len, downmix_to_stereo, remap_channels,
};
pub use convert_element::AudioConvert;
pub use downmix_element::AudioDownmix;
pub use resample::{gcd, output_len, ChannelResampler, PolyphaseFilter};
pub use resample_element::AudioResample;
pub use format::{
    AudioFormat, AudioFrameRef, SampleFormat, FAMILY, FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE,
    RAW_ANY_OFFER,
};
pub use wav::{parse_wav_header, write_pcm_wav, WavError, WavHeader, WavParse};

use profluens_core::element::Element;
use profluens_core::registry::Registry;

/// Register this crate's elements for name-based construction (spec: Plugins —
/// `parse("… ! wavparse ! audioconvert format=s16 ! …")`). Typed `use` + constructor
/// stays primary. `wavparse` is config-free; `audioconvert` reads its target `format`
/// prop and `audioresample` its target `rate` prop in `start()` (defaults S16 / 48 kHz).
/// Descriptors are `&'static`, taken from a throwaway default instance.
pub fn register(registry: &mut Registry) {
    registry.register(WavParse::new().desc());
    registry.register(AudioConvert::new(SampleFormat::S16).desc());
    registry.register(AudioResample::new(48_000).desc());
    registry.register(AudioDownmix::new().desc());
}
