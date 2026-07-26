//! sc-vaapi — VA-API hardware-accelerated video decode **and encode** for
//! streamcraft.
//!
//! This crate wraps the system **libva** (the Video Acceleration API) to offload
//! H.264 decode and H.264/H.265/VP8 encode onto the GPU's fixed-function video
//! engine. Unlike the pure-Rust codec crates, VA-API is a *device boundary* — the
//! same class as `sc-pipewire`'s libpipewire — so a small, hand-rolled FFI surface
//! is the sanctioned tool. There is no `-sys` crate and no bindgen: [`ffi`]
//! transcribes the functions and structs these paths need directly from the libva
//! 2.23 headers, with `size_of` assertions locking every struct's ABI, and [`va`]
//! wraps them in RAII types so `unsafe` never escapes those two modules.
//!
//! ## What it provides
//! - [`probe`] — a cached capability probe (`SC_NO_VAAPI` / `SC_VAAPI_DEVICE`
//!   honored) reporting which decode/encode families the driver accelerates.
//! - [`register`] — adds each element **only when its entrypoint exists**, so a
//!   machine without the hardware never advertises it.
//! - [`video_decoder_for`] — negotiation-driven construction (`"h264/annexb"` →
//!   [`VaapiH264Dec`]).
//! - [`h264dec::VaapiH264Dec`] — the decoder: parses Annex-B parameter sets + slice
//!   headers ([`h264parse`], ITU-T H.264), maintains the DPB and reference lists,
//!   drives `vaBeginPicture`/`vaRenderPicture`/`vaEndPicture`, and reads the decoded
//!   surface back as tight-packed NV12 into pool memory.
//! - [`h264enc::VaapiH264Enc`] / [`h265enc::VaapiH265Enc`] / [`vp8enc::VaapiVp8Enc`]
//!   — the encoders: `video/raw` (NV12/I420) in, hardware bitstream out
//!   (Annex-B and AVCC/HVCC framings for the NAL codecs, raw frames for VP8, plus
//!   a `bytes` escape). The NAL encoders author their own SPS/PPS(/VPS) and slice
//!   headers ([`bitwriter`], packed VA headers); every encoder is gated in CI-able
//!   hardware tests by a round trip through this workspace's *independent*
//!   pure-Rust decoders with a PSNR floor (`tests/hw_encode_roundtrip.rs`), and
//!   the muxed output is validated against ffprobe/ffmpeg/mpv.
//!
//! ## POC boundaries (stated honestly)
//! - **IN**: H.264 Baseline / Main / High, **8-bit 4:2:0**, progressive frames,
//!   CAVLC + CABAC (the GPU does entropy), intra + inter (the GPU does MC), Annex-B
//!   framing, POC types 0 / 1 / 2, sliding-window DPB + MMCO 5 (other MMCO ops
//!   best-effort), reference-list default construction + modifications, NV12
//!   readback via `vaDeriveImage` (with a `vaGetImage` fallback).
//! - **OUT**: interlaced / field / MBAFF, >8-bit or non-4:2:0, FMO (slice groups),
//!   AVCC sink framing, and — on this class of hardware — AV1 (no VA-API profile).
//!   `h265`/`vp9` families are *probed* (so the capability table is honest) but only
//!   `vaapih264dec` has an implementation in this POC; the others are follow-up.
//! - **Reorder pts** uses a feed-order FIFO — the same documented caveat as the
//!   software `h264dec`; downstream A/V sync should prefer container timestamps.
//!
//! The whole VA-API surface is single-threaded: the element owns its `Display` on
//! its own scheduler thread (`SchedHint::Active`) and never shares it.

pub mod bitwriter;
mod enc;
pub mod ffi;
pub mod h264dec;
pub mod h264enc;
pub mod h264parse;
pub mod h265enc;
pub mod probe;
pub mod va;
pub mod vp8enc;

pub use h264dec::VaapiH264Dec;
pub use h264enc::VaapiH264Enc;
pub use h265enc::VaapiH265Enc;
pub use probe::{probe, VaCaps};
pub use vp8enc::VaapiVp8Enc;

use streamcraft_core::element::Element;
use streamcraft_core::registry::Registry;

/// Register this crate's elements for name-based construction (spec: Plugins).
/// Gated on the probe: `vaapih264dec` is only registered when the VA-API driver
/// advertises H.264 VLD decode — a machine without the hardware never sees the
/// element in the registry, so autoplug never picks it.
pub fn register(registry: &mut Registry) {
    if let Some(caps) = probe::probe() {
        if caps.supports("h264/annexb") {
            registry.register(VaapiH264Dec::new().desc());
        }
        // h265/vp9 decode families are probed for the capability table but have no
        // element impl in this POC — do not register stubs (follow-up).
        if caps.supports_encode("h264/annexb") {
            registry.register(VaapiH264Enc::new().desc());
        }
        if caps.supports_encode("h265/annexb") {
            registry.register(h265enc::VaapiH265Enc::new().desc());
        }
        if caps.supports_encode("vp8") {
            registry.register(vp8enc::VaapiVp8Enc::new().desc());
        }
    }
}

/// Construct a hardware video decoder for a negotiated decode family, when the
/// device supports it. Returns `None` for unsupported/unimplemented families.
pub fn video_decoder_for(family: &str) -> Option<Box<dyn Element>> {
    let caps = probe::probe()?;
    match family {
        "h264/annexb" if caps.supports("h264/annexb") => Some(Box::new(VaapiH264Dec::new())),
        _ => None,
    }
}
