# Opus encoder — handoff brief for the "transcode-to-Opus" milestone

**Status (2026-07-30, updated):** the Opus *decoder* is wired, conformant, and allocation-free
(`OpusDec`). The *encoder* is now **wired for CELT-only mode** — `OpusEncoder` (orchestrator) +
`OpusEnc` (element), validated round-trip + perceptual + ffmpeg oracle. SILK-only and Hybrid modes
are the remaining work; this doc is the brief for picking them up. Read it fully before touching
code.

## Milestone 1 — CELT-only: DONE ✅

- **`opus/src/encoder.rs` — `OpusEncoder`**: the unified orchestrator (mode selection + bandwidth
  selection + CBR rate control) driving `oxideav-opus`'s `CeltEncoder`. Mono + stereo, the four
  CELT frame sizes, all CELT bandwidths, CBR. `EncoderConfig` + `Application` (Voip/Audio/LowDelay,
  all → CELT-only for now; the mode-select fn is the hook SILK/Hybrid attach to). Input:
  interleaved **48 kHz `i16`** (CELT-native; other rates resample upstream).
- **`opus/src/opusenc.rs` — `OpusEnc` element**: `audio/raw` s16/48 kHz/1–2 ch sink →
  one-`opus`-packet-per-buffer src. Reframes to full CELT frames, emits into `alloc_exact` pool
  buffers, EOS zero-pads the tail. Registered in `lib.rs`; autoplugs into the mkv/mp4/ogg muxers
  (they accept `opus`). No per-frame alloc on the pf-opus side (the CELT range coder still allocates
  internally — the alloc-free follow-up, item 4 below).
- **Validation** (`opus/examples/encode_roundtrip.rs` + `opus/tests/pipeline.rs`): round-trip
  through our `OpusDec`, the perceptual NMR gate, and the **ffmpeg/libopus external oracle** on a
  hand-muxed Ogg-Opus file — all PASS. Tones/sweeps −22 to −34 dB NMR (transparent). A full
  `tonesrc → opusenc → opusdec → sink` scheduler test covers the element's caps negotiation +
  reframing.

**Why CELT-only first:** it takes 48 kHz `i16` directly (no encode-side resampler) and never
touches the SILK excitation path, so the ⚠️ gotcha below stays **dormant** — the whole
Milestone-1 deliverable ships without needing to touch the shared decoder struct.

**Design note — the orchestrator lives in `pf-opus`, not the vendored crate.** The vendored
`oxideav-opus` modules are all `pub`, so `OpusEncoder` builds on `CeltEncoder` / `Bandwidth` /
`compose_packet` etc. without diverging the vendored snapshot further (keeps `PROFLUENS-PATCHES.md`
minimal). SILK/Hybrid should follow suit: drive `SilkEncoderMono/Stereo` + `HybridEncoderMono` from
`opus/src/encoder.rs`, adding `ModeImpl::Silk`/`ModeImpl::Hybrid` variants.

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

## What's left — the remaining work (SILK/Hybrid + polish)

Milestone 1 delivered the orchestrator + element for CELT-only (items 1, 4, 5, 7 of the original
plan, and item 3 for CBR). The remaining work, roughly in dependency order:

1. **⚠️ Fix the `Excitation` generic-allocator gotcha FIRST** (see the section below). CELT-only
   dodged it; SILK-only and Hybrid **construct `Excitation` on the encode path**, so without the fix
   the first SILK/Hybrid encode session leaks into the 16 MiB decode bump arena → `AllocError`
   panic. Scope confirmed this session: `Excitation` (and `SilkFrameDecoded` which holds it) must
   become generic over the allocator (`= Global` default) — ~47 mentions across silk_decode.rs,
   silk_decode_core.rs, silk_synthesis.rs, silk_packet_encode.rs, decoder.rs. The synthesis (the one
   `.excitation.e_q23()` reader, `silk_decode_core::decode_core`) is **decode-only**, so encode
   never reads the returned excitation's arena buffers — but it still *constructs* them, which is
   the leak. Verify `alloc_check.rs` still reports 0 decode allocs after the refactor.
2. **SILK-only mode.** Add `ModeImpl::Silk` driving `SilkEncoderMono`/`SilkEncoderStereo`
   (`silk_encoder.rs`). Needs the **encode-side 48→8/12/16 kHz downsampler** (item 5) — SILK takes
   internal-rate `&[f32]`, not 48 kHz `i16`. SILK has no per-packet rate control (quality-driven
   quantizer); use `encode_packet_cbr` (pads to a target) and pick budgets with headroom.
3. **Hybrid mode.** Add `ModeImpl::Hybrid` driving `HybridEncoderMono` — it carries its own
   48→16 kHz decimator, so it takes 48 kHz `i16` like CELT (no external resampler), but only mono
   exists today (`HybridEncoderMono`). Verify its + LBRR-FEC round-trip coverage (unconfirmed).
4. **Real mode selection** (RFC 6716 §2 Table 1): bitrate + bandwidth + speech/music analysis →
   SILK / Hybrid / CELT per frame, replacing the CELT-only `select_mode` in `encoder.rs`.
5. **Encode-side downsampler** for SILK (48→8/12/16 kHz). Check `silk_resampler.rs` (it's the decode
   *up*sampler) and the Hybrid module's private `Decimator48To16` before writing a new one.
6. **VBR/CVBR rate control** (the bit reservoir) — CELT `encode_packet` already takes a per-packet
   budget, and `HybridEncoderMono::encode_packet_elected` exists; add cross-packet VBR on top of the
   CBR baseline in `cbr_payload_bytes`.
7. **Multi-frame packets** (§3.2 code 1/2/3 via `packet_compose.rs`) for frame durations > one
   internal frame (40/60 ms SILK packets), and the alloc-free follow-up (pool-back the CELT range
   coder so `OpusEnc` does zero per-frame heap traffic, like the decoder's alloc_check trajectory).

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

**Already wired for CELT-only** in `examples/encode_roundtrip.rs` — mirror it for SILK/Hybrid: gate
each new mode on the same three checks below. (Note the pure-tone SNR alignment lesson: correlation
alignment is phase-ambiguous on periodic signals, so a low tone-SNR next to a very negative NMR is a
measurement artifact — brute-force the best-SNR offset and gate on the perceptual NMR, not raw SNR.)


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
- **Encoder (Milestone 1, CELT-only):** `opus/src/encoder.rs` (`OpusEncoder`/`EncoderConfig`/
  `Application`), `opus/src/opusenc.rs` (`OpusEnc` element).
- **Encoder validation:** `opus/examples/encode_roundtrip.rs` (round-trip + perceptual + ffmpeg
  Ogg-Opus oracle), `opus/tests/pipeline.rs` (full-scheduler `tonesrc→opusenc→opusdec→sink`).
- Decoder + patches: `vendor/oxideav-opus/PROFLUENS-PATCHES.md`, `opus/src/opusdec.rs`.
- Decode conformance/alloc harnesses to mirror: `opus/examples/conformance.rs`,
  `opus/examples/alloc_check.rs`.
- Encode building blocks: the `*_encode.rs` files + `silk_encoder.rs` + `packet_compose.rs` above.
