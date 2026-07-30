# Opus encoder — handoff brief for the "transcode-to-Opus" milestone

**Status (2026-07-30):** the Opus *decoder* is wired, conformant, and allocation-free (`OpusDec`,
see `vendor/oxideav-opus/PROFLUENS-PATCHES.md`). The *encoder* is **not wired** — but it is **not
broken either**. The confusion in older notes ("encoder is low-level building blocks only") is half
right: the per-mode packet encoders are real and unit-tested, what's missing is the **top-level
orchestration** that turns a config + PCM stream into Opus packets, plus wiring as an element.

This doc is the brief for an agent picking that up. Read it fully before touching code.

---

## What actually exists (and appears validated)

`vendor/oxideav-opus/` is git-master, vendored. The encode side is substantial:

**Per-mode packet encoders** (the pieces a top-level encoder would drive):
- `silk_encoder.rs` — `SilkEncoderMono::new(bandwidth)` / `SilkEncoderStereo::new(bandwidth)`, with
  `encode_packet(&[f32]) -> EncodedSilkPacket` and `encode_packet_cbr(...)`.
- `celt_packet_encode.rs` — `CeltEncoder::new(bandwidth, frame_tenths_ms)`,
  `encode_packet(&[i16], payload_bytes) -> (Vec<u8>, CeltFrameEncodeInfo)`.
- `hybrid_packet_encode.rs` — `HybridEncoderMono::new(bandwidth, frame_tenths_ms, stereo)`,
  `encode_packet(&[i16], payload_bytes) -> Vec<u8>` and `encode_packet_elected(...)` (VBR-elected size).
- `silk_packet_encode.rs` — `encode_silk_only_packet_mono/stereo` (+ `_with_lbrr` FEC variants).

**Sub-stage encoders + primitives:** `range_encoder.rs`, `packet_compose.rs`
(`compose_packet`, `compose_packet_code3`, `encode_length` — §3.2 multi-frame packing),
`celt_frame_encode.rs`, `celt_band_encode.rs`, `celt_pvq_encode.rs`, `celt_energy_encode.rs`,
`celt_alloc_encode.rs`, `celt_laplace.rs` (`ec_laplace_encode`), and the SILK sub-stages
(`silk_gains.rs::encode`, `silk_lsf_stage2.rs::encode`, `silk_lsf_interp.rs::encode_index`,
`silk_ltp.rs::encode`, `silk_excitation.rs::encode`, `silk_frame.rs::encode_*`,
`silk_lcg_seed.rs::encode_lcg_seed`).

**Validation:** `celt_packet_encode.rs`, `silk_encoder.rs`, and `silk_packet_encode.rs` carry
**in-crate round-trip tests** (encode → decode through this crate's own decoder → compare), so those
blocks are individually exercised. **Verify** hybrid + LBRR-FEC round-trip coverage yourself — it was
not confirmed in this survey. There is **no** end-to-end encoder conformance against a libopus/ffmpeg
oracle yet.

---

## What's actually missing — this is the work

There is **no `OpusEncoder` type at all**. (`lib.rs`'s doc line "[`Encoder`] entry points still
return `Error::NotImplemented`" is **stale** — it describes a scaffold that was replaced by the
building blocks above; there is no `Encoder`/`OpusEncoder` struct to grep for.) The gaps, roughly in
dependency order:

1. **A unified `OpusEncoder`** taking `(sample_rate, channels, target_bitrate, application)` and a
   PCM stream, producing Opus packets. This is the orchestrator that ties everything below together.
   Mirror the structure of libopus `opus_encoder.c` conceptually (clean-room — do NOT read its source;
   see the rules section).
2. **Mode selection** (SILK vs CELT vs Hybrid, per frame) driven by target bitrate + audio bandwidth
   + signal type (speech vs music). RFC 6716 §2 Table 1 gives the bitrate/bandwidth → mode guidance.
   A minimal first cut can hard-select a mode from an explicit "application" hint (VOIP → SILK,
   AUDIO → CELT/Hybrid) before adding content analysis.
3. **Bitrate / rate control.** The per-mode encoders take an explicit `payload_bytes` budget per
   packet — nothing converts a target bitrate + frame duration into that budget, nor does CBR/VBR/
   CVBR across packets (the bit-reservoir). Start with CBR (`payload_bytes = bitrate * frame_s / 8`),
   then add the VBR-elected path (`encode_packet_elected` already exists for hybrid).
4. **Config/TOC orchestration at the top.** Map `(mode, bandwidth, frame_size)` → the §3.1 32-config
   TOC, and pack multi-frame packets (§3.2 code 1/2/3 via `packet_compose.rs`) when the frame
   duration exceeds one internal frame.
5. **Input convention.** SILK takes `&[f32]`, CELT/Hybrid take `&[i16]` — the unified encoder needs
   one input type + conversion at the boundary. Decide and document one.
6. **Encode-side resampling.** Input at an arbitrary rate → resample to the internal rates (SILK
   8/12/16 kHz, CELT 48 kHz). The decode side has `silk_resampler.rs` (upsampler); the encode side
   needs the *down*sampler. Check whether one already exists before writing it.
7. **Wire it as a `pf-opus` element** (`OpusEnc`) in `opus/src/lib.rs` (only `OpusDec` is registered
   today), following the `opusdec` element pattern (dynamic caps, pool buffers, no per-frame alloc).
   Add it to autoplug/transcode paths. This is the actual "transcode-to-Opus" deliverable.

---

## ⚠️ GOTCHA from the allocation-free decode refactor — READ THIS

The alloc-free work (2026-07-30) made several **decode transients** draw from a per-thread bump
arena (`bump::DecodeBump`) that is reset **only by the decoder's entry points**. One converted
struct is **shared with the encode path**:

- **`silk_excitation.rs::Excitation`** — its `e_q23` / `pulses_per_block` / `lsb_count_per_block`
  fields are now `Vec<T, crate::bump::DecodeBump>`, and **`Excitation::encode` constructs one** (I
  had to convert its field-builders to the bump so the struct type-checks). The encode path never
  calls `bump::reset()`, so **every encoded frame's `Excitation` accumulates in the 16 MiB bump
  region and is never freed → unbounded growth → an `AllocError` panic within an encode session.**
  Today this is dormant (nothing calls `Excitation::encode` outside unit tests, which are short), but
  it will bite the moment the encoder is wired.

  **Fix (do this first, before building the encoder):** make `Excitation` **generic over the
  allocator** — `struct Excitation<A: Allocator = Global>` — so `decode` returns
  `Excitation<DecodeBump>` and `encode` builds `Excitation<Global>` (or a caller-provided scratch).
  This is the clean shared-struct solution and preserves the decoder's alloc-free property. Avoid
  calling `bump::reset()` from encode (it would clash if encode/decode ever interleave on a thread).

- **Audit for other shared structs** you end up constructing on the encode path that were bumped on
  the decode side (grep `crate::bump::DecodeBump` and check each type's construction sites). Most
  bumped structs are decode-only (`BandDecodeResult`, `CeltFrameOutput`, `TfDecode`, `StereoFrameI16`,
  `OpusPacket` frame list), but confirm rather than assume.

---

## Validation plan (hold the encoder to the workspace's correctness bar)

1. **Round-trip through our own `OpusDec`**: encode a signal, decode it back, measure SNR. Extend the
   existing per-module round-trip tests to a whole-encoder test.
2. **External oracle** (the workspace rule — see the `external-player-validation` discipline): decode
   the packets we emit with **libopus/ffmpeg** and confirm they're valid + close in SNR. A pure-Rust
   encoder that only round-trips through its own decoder can hide a shared bug.
3. **Perceptual gating**: the transcode-QA harness already exists — `audio/quality.rs`
   (`PerceptualAnalyzer`, NMR + bark loudness, alloc-free) + `audio/PERCEPTUAL_QUALITY.md`. Gate
   encoder quality on it (perceptual codecs put noise below the masking threshold, so **don't gate on
   raw SNR alone** — see that doc).

---

## Rules that apply (non-negotiable in this workspace)

- **Clean-room:** implement from RFC 6716 (§2 modes, §3 packet structure, §4 the normative encode
  guidance) + the RFC's Appendix A reference listing (staged under `docs/`). Do **not** read
  libopus / other encoder source. Cite the RFC section at each non-trivial algorithm site
  (`algorithm-citation-rule`).
- **No allocations on the hot path** — the `OpusEnc` element must encode into reused/pool buffers,
  no per-frame heap traffic, exactly like `OpusDec`. Fix the `Excitation` bump gotcha the *right*
  way (generic allocator), not by leaking into the decode arena.
- **Reactor IO** if the element ever touches IO (`ctx.io()`, never std blocking).

## Quick file map
- Decoder + patches: `vendor/oxideav-opus/PROFLUENS-PATCHES.md`, `opus/src/opusdec.rs`.
- Conformance harness to mirror for the encoder: `opus/examples/conformance.rs`,
  `opus/examples/alloc_check.rs`.
- Encode building blocks: the `*_encode.rs` files + `silk_encoder.rs` + `packet_compose.rs` above.
