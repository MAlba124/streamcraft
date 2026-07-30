//! pf-opus — the Opus (RFC 6716 / RFC 7845) codec plugin, backed by the pure-Rust
//! [`oxideav-opus`](https://github.com/OxideAV/oxideav-opus).
//!
//! # Adoption: git master, not the published crate
//!
//! The adoption rubric is `vp8/src/lib.rs`: pure Rust, zero `unsafe`, no `build.rs`, MIT,
//! RFC-annotated, and — the deciding gate — *it must actually decode*.
//!
//! The **published `0.0.13`** does not: its own description calls it an "orphan-rebuild scaffold
//! pending clean-room re-implementation", and RFC 6716 vectors score ≤ 0 dB SNR (CELT/hybrid emit
//! silence, SILK noise). So it was rejected for months. The working codec — a genuine clean-room
//! SILK + CELT + hybrid decoder (and encoder) rebuilt consulting only the RFCs + black-box
//! `opusdec`/`opusenc` — lives on **git master**, never published. We pin it by rev in
//! `Cargo.toml` (its sibling `oxideav-core` is pinned to the matching master via the workspace
//! `[patch.crates-io]`).
//!
//! **Conformance gate — PASS.** `examples/conformance.rs` decodes all 12 canonical RFC 6716 test
//! vectors through one stateful `OpusDecoder` and best-offset-SNRs against the reference `.dec`:
//! worst vector **20.9 dB**, SILK vectors **96–103 dB** (bit-exact), CELT/hybrid **21–40 dB**
//! (float-noise floor — perceptually transparent, as expected for a non-bit-exact float codec vs
//! the libopus reference). Run it with:
//!
//! ```text
//! curl -LO https://opus-codec.org/testvectors/opus_testvectors.tar.gz && tar xzf …
//! cargo run -p pf-opus --example conformance -- ./opus_testvectors
//! ```
//!
//! # [`OpusDec`]
//!
//! Decodes one Opus packet per buffer to interleaved 48 kHz `s16` PCM, announcing the channel
//! count via dynamic caps (from `OpusHead`, else the first packet), trimming the RFC 7845 §4.2
//! pre-skip and applying the header output-gain. Sample-rate conversion (Opus is always 48 kHz on
//! decode) is a downstream `audioresample`.
//!
//! # [`OpusEnc`] / [`OpusEncoder`] — transcode-to-Opus
//!
//! [`OpusEncoder`] is the unified top-level orchestrator (mode selection + bandwidth selection +
//! CBR rate control) that drives `oxideav-opus`'s per-mode packet encoders; [`OpusEnc`] wires it
//! as an element (48 kHz `s16` PCM in → `opus` packets out). This milestone implements **CELT-only**
//! mode (RFC 6716 §4.3 — the MDCT music/low-delay path): mono + stereo, the four CELT frame sizes,
//! all CELT bandwidths, CBR. It is the natural first transcode target — CELT takes 48 kHz `i16`
//! directly (no encode-side resampling) and never touches the SILK excitation path, so the shared
//! `Excitation`/`DecodeBump` arena gotcha from `OPUS_ENCODER_HANDOFF.md` stays dormant here.
//!
//! **Validation — PASS.** `examples/encode_roundtrip.rs` gates the encoder the workspace's three
//! ways: (1) round-trip through our own [`OpusDec`] (best-offset SNR), (2) the perceptual NMR gate
//! (`profluens-audio`'s `PerceptualAnalyzer` — the right metric for a masking codec, per
//! `audio/PERCEPTUAL_QUALITY.md`), and (3) an **external oracle** — mux to Ogg-Opus (RFC 7845) and
//! decode with **ffmpeg/libopus**, confirming our packets are valid standard Opus libopus agrees
//! on. Tones/sweeps land at −22 to −34 dB NMR (transparent); the Ogg-Opus file decodes cleanly in
//! ffmpeg/ffprobe. Run it with `cargo run -p pf-opus --example encode_roundtrip`.
//!
//! # Codec backend — libopus (default) or pure-Rust
//!
//! Both [`OpusEnc`] and [`OpusDec`] run through the **reference libopus** C codec (statically linked
//! via `libopus-sys`, the `libopus` feature — default): reference-grade quality, allocation-free on
//! the hot path. `--no-default-features` swaps in the pure-Rust `oxideav-opus` codec instead (the
//! CELT-only `OpusEncoder` + the conformant pure-Rust decoder). Codecs are the sanctioned place for
//! a C dependency in this workspace — quality/tuning that clean-room Rust can't match — kept behind
//! the plugin boundary (the core never sees C). Each backend is selected by a small `EncBackend` /
//! `DecBackend` trait so the element code is backend-agnostic.
//!
//! A full FLAC→Opus transcode measures **2.1 allocs/packet** on the libopus backend vs 20.1 on the
//! pure-Rust one (`examples/transcode_alloc_check.rs`) — libopus contributes ~0, so the residual is
//! flacdec (pool-backed) + scheduler.
//!
//! **Pure-Rust encoder status** (still exercised by the examples/tests directly): the CELT hot path
//! is ~90% allocation-free (226 → 22 allocs/packet stereo; `examples/encode_alloc_check.rs`,
//! `encode_alloc_profile.rs`; see `vendor/oxideav-opus/PROFLUENS-PATCHES.md`), transparent but ~7–14
//! dB behind libopus on tonal content. Its follow-ups (SILK/Hybrid modes, VBR, closing the last
//! allocations) live in `OPUS_ENCODER_HANDOFF.md`.

mod encoder;
/// Safe wrappers over the statically-linked reference libopus (`libopus` feature). Public so
/// tools (e.g. `examples/quality_search.rs`) can encode/decode directly, outside the element.
#[cfg(feature = "libopus")]
pub mod libopus;
mod opusdec;
mod opusenc;

pub use encoder::{Application, EncoderConfig, OpusEncoder, Signal, OPUS_RATE};
pub use opusdec::OpusDec;
pub use opusenc::OpusEnc;

use profluens_core::element::Element;
use profluens_core::registry::Registry;

/// Register this plugin's elements (spec: Plugins). `opusdec` decodes Opus packets → `s16` PCM;
/// `opusenc` encodes 48 kHz `s16` PCM → Opus (CELT-only) packets.
pub fn register(registry: &mut Registry) {
    registry.register(OpusDec::new().desc());
    registry.register(OpusEnc::new().desc());
}
