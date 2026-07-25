//! `sc-aac` — AAC decode for streamcraft, backed by **oxideav-aac** (pure Rust).
//!
//! ## Adoption verdict (rubric: `sc-vp8`'s lib.rs; judged by source, not blurbs)
//! `oxideav-aac 0.1.6` (crates.io, 2026-07): **zero `unsafe`**, no build.rs, one
//! dependency (`oxideav-core`, already in-tree via `sc-h264`), ISO/IEC 14496-3
//! citations at point of use throughout. The published decoder covers AAC-LC and
//! HE-AAC (SBR) with M/S, intensity, TNS, PNS, LTP; git HEAD adds LATM/PS/ER —
//! re-evaluate on the next release. The `oxideav-core` `Decoder` trait impl only
//! accepts self-syncing ADTS/LOAS byte streams, so [`AacDec`] drives the raw
//! `StreamDecoder::decode_raw_data_block` + `asc::AudioSpecificConfig` API —
//! exactly shaped for container-delivered raw access units.
//!
//! Elements:
//! - [`AacDec`] (`aacdec`) — ASC head + raw AUs in, interleaved s16 `audio/raw`
//!   out (announced via dynamic caps from decoded geometry).

pub mod aacdec;

pub use aacdec::AacDec;

use streamcraft_core::element::Element;
use streamcraft_core::registry::Registry;

/// Register this crate's elements for name-based construction (spec: Plugins).
pub fn register(registry: &mut Registry) {
    registry.register(AacDec::new().desc());
}
