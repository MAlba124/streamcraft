//! pf-mp3 — the MPEG-1/2 Audio Layer III (MP3) codec plugin.
//!
//! Like pf-vp8, this crate is **not** hand-written: it wraps
//! [`oxideav-mp3`](https://github.com/OxideAV/oxideav-mp3), a pure-Rust MPEG-1 / MPEG-2
//! / MPEG-2.5 Audio Layer III decoder (and CBR/VBR encoder + demuxer we do not use),
//! adopted after review (2026-07-24) under the same rubric as pf-vp8 (see
//! `vp8/src/lib.rs`). The spec's codec taboo is FFI walls — a foreign allocator,
//! threading model, and timestamp semantics behind a boundary our batches can't cross
//! (spec: First-party codecs; Non-goals). `oxideav-mp3` has none of that: **pure Rust,
//! zero `unsafe`, no `build.rs`, MIT**, and its decode chain cross-references ISO/IEC
//! 11172-3 / 13818-3 section-by-section exactly like our own codec crates. Its only
//! dependency is `oxideav-core` (the shared `Packet` / `AudioFrame` / `Decoder`
//! vocabulary); we touch only those plain types, never its arena.
//!
//! ## Adoption evidence (the gate)
//!
//! MP3 decode is spec-exact enough that two conforming decoders agree to within the
//! float-rounding regime. The pinned **0.1.3** decoder was validated against an ffmpeg
//! (libmp3lame) oracle across the mainstream matrix — 44.1 / 48 kHz, mono +
//! joint-stereo, 128 / 320 kbps CBR + VBR (`-q:a 2`) — decoding through
//! [`Mp3CoreDecoder`](oxideav_mp3::codec_decoder::Mp3CoreDecoder) and scoring the
//! interleaved PCM against ffmpeg's own decode of the same file: **84–102 dB SNR**
//! (16-bit PCM's own dynamic-range floor is ~96 dB), max error a few LSBs. All the
//! oracle files carried a leading ID3v2 tag, which the framer handles. That clears the
//! ISO 11172-4 "conforming decoders agree at full scale" bar by a wide margin.
//!
//! ## Scope and debt (tracked in PLAN.md)
//!
//! - **Decoder only.** `oxideav-mp3` also ships a CBR/VBR encoder and a `Read + Seek`
//!   container demuxer; neither is wired here. profluens frames the byte stream
//!   itself (via the crate's [`FrameWalker`](oxideav_mp3::frame::FrameWalker)) so no
//!   `Seek` source is required — a plain `filesrc` byte stream decodes.
//! - **Published 0.1.3 vs upstream HEAD.** Our lock pins the published `0.1.3`
//!   (registry commit `e758435`); upstream `main` is ~18 commits ahead. None of the
//!   MPEG-1 decode path differs — the HEAD-only fixes are MPEG-2.5 scalefactor-band
//!   tables, 8 kHz mixed-block rendering, and a free-format sub-header panic guard
//!   (plus output-invariant perf and docs). Mainstream MP3 (MPEG-1, 44.1/48 kHz) is
//!   unaffected; a bump to those fixes lands with the next published release.
//! - **Gapless trim.** Xing/LAME encoder-delay / padding trimming is **out of v1**: the
//!   decoder emits every reconstructed sample (the ~half-frame codec-delay priming and
//!   the frame-padded tail included), so output is a few hundred samples longer than a
//!   gapless reference. Documented on [`Mp3Dec`], not a defect.
//! - **`decode_into` pool memory.** As with pf-vp8, the upstream trait decoder returns
//!   owned planar `Vec`s; [`Mp3Dec`] pays one interleave-into-pool copy per frame. A
//!   `decode_into` upstream contribution would remove it without changing this element.
//! - **Stale crate-level doc.** `oxideav-mp3`'s own `lib.rs` header still reads
//!   "clean-room rebuild in progress" and carries a dead `Error::NotImplemented`; the
//!   decoder is in fact complete and tested (its README and module docs are accurate).

pub mod mp3dec;

pub use mp3dec::Mp3Dec;

use profluens_core::element::Element;
use profluens_core::registry::Registry;

/// Register this crate's elements for name-based construction (spec: Plugins —
/// `parse("… ! mp3dec ! …")`). Typed `use` + constructor stays primary; this powers
/// `profluens launch` and one-liner tests. The descriptor is `&'static`, taken from a
/// throwaway default instance; [`Mp3Dec`] is config-free (rate/channels come from the
/// frame header, announced at runtime).
pub fn register(registry: &mut Registry) {
    registry.register(Mp3Dec::new().desc());
}
