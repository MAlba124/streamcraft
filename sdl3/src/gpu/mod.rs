//! The owned GPU render pipeline: custom SPIR-V shaders doing colorimetry-aware
//! YCbCr→RGB (with a tone-map slot for HDR) on the SDL3 GPU API, replacing the
//! sink's fixed-function `SDL_UpdateYUVTexture` path.
//!
//! Rationale (spec / standing rules): `SDL_UpdateYUVTexture`/`SDL_UpdateNVTexture`
//! let SDL choose the YCbCr matrix and ignore the real colorimetry, and HDR tone
//! mapping is impossible there. SDL keeps windowing / device / swapchain (its GPU
//! API); the shaders and all color science are ours (see [`color`]). Clean-room:
//! every transform cites its standard at the point of use.
//!
//! Split:
//!   * [`color`] — pure-Rust color science (matrices, primaries, uniform block).
//!   * [`renderer`] — [`GpuRenderer`], the SDL_GPU device/pipeline/texture plumbing.
//!
//! The baked SPIR-V blobs live next to the GLSL sources in `sdl3/shaders/` and are
//! embedded via `include_bytes!` (no runtime shader compiler) — regenerate with
//! `sdl3/tools/bake_shaders.sh`.

pub mod color;
pub mod renderer;

pub use color::{ChromaMode, Colorimetry, FragUniforms, Matrix, Primaries, Range, Transfer};
pub use renderer::GpuRenderer;

/// The baked fullscreen-triangle vertex shader (SPIR-V).
pub const VIDEO_VERT_SPV: &[u8] = include_bytes!("../../shaders/video.vert.spv");
/// The baked color-pipeline fragment shader (SPIR-V).
pub const VIDEO_FRAG_SPV: &[u8] = include_bytes!("../../shaders/video.frag.spv");

/// SPIR-V magic number, first word of every valid module (little-endian on disk).
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Read the first little-endian `u32` of a blob (the SPIR-V magic, for validation).
pub fn first_word_le(blob: &[u8]) -> Option<u32> {
    if blob.len() < 4 {
        return None;
    }
    Some(u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baked_spirv_blobs_start_with_magic() {
        assert_eq!(
            first_word_le(VIDEO_VERT_SPV),
            Some(SPIRV_MAGIC),
            "vertex .spv must start with the SPIR-V magic 0x07230203 (rerun bake_shaders.sh)"
        );
        assert_eq!(
            first_word_le(VIDEO_FRAG_SPV),
            Some(SPIRV_MAGIC),
            "fragment .spv must start with the SPIR-V magic 0x07230203 (rerun bake_shaders.sh)"
        );
        // SPIR-V is a stream of 32-bit words → length must be a multiple of 4.
        assert_eq!(VIDEO_VERT_SPV.len() % 4, 0);
        assert_eq!(VIDEO_FRAG_SPV.len() % 4, 0);
    }
}
