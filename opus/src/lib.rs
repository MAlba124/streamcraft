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
//! # Encode (transcode-to-Opus) — follow-up
//!
//! `oxideav-opus` master also carries a clean-room **encoder** (SILK/CELT/hybrid, CBR/VBR, LBRR
//! FEC — see its CHANGELOG), but there is no unified top-level `OpusEncoder`: the encode side is
//! per-mode building blocks (`encode_silk_frame`, `celt_frame_encode`, `compose_packet`, …) that
//! need mode-selection + bitrate-control orchestration to become an `OpusEnc` element. Wiring
//! that — plus the perceptual-quality gate in `audio/PERCEPTUAL_QUALITY.md` to grade the output —
//! is the next milestone for the transcode path.

mod opusdec;

pub use opusdec::OpusDec;

use profluens_core::element::Element;
use profluens_core::registry::Registry;

/// Register this plugin's elements (spec: Plugins). `opusdec` decodes Opus packets → `s16` PCM.
pub fn register(registry: &mut Registry) {
    registry.register(OpusDec::new().desc());
}
