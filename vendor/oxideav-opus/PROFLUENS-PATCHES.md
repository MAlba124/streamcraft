# Profluens patches to `oxideav-opus`

Vendored from `oxideav-opus` **git master** (rev `f7dd682d049864d0f12592a90d41af5bedb26823`) —
the working clean-room SILK+CELT+hybrid decoder+encoder, which the published `0.0.13`
("orphan-rebuild scaffold") is not (see `opus/src/lib.rs`). Pinned as a path dependency via
`opus/Cargo.toml`; its `oxideav-core = "0.1"` resolves to git master `0.1.34` through the
workspace `[patch.crates-io]` (the master API `CodecInfo::payload_magic` etc. is unpublished).

Two changes from upstream; both offered upstream, kept here until they land. **Both are output-
preserving** — the RFC 6716 12-vector conformance gate stays byte-identical (worst 20.94 dB) at
every step.

## `OpusDecoder::decode_packet_into` — no per-packet heap allocation (src/decoder.rs)

**Why.** Upstream `decode_packet(&[u8]) -> DecodedAudio { pcm: Vec<i16>, .. }` allocates a **fresh
`Vec<i16>` for every packet**. In a streaming decoder that is steady-state heap traffic on the hot
path (~50 packets/s), which this workspace bans (spec: performance #1 — no per-frame allocation;
`clippy.toml`). `pf-opus`'s `opusdec` element must decode into a reused buffer, exactly as `pf-aac`
decodes into `ctx.scratch()` + a reused `pending_pcm` and `pf-flac` uses `pull_into(&mut Vec)`.

**Change.** The decode core `decode_parsed_packet` was factored into
`decode_into_buffers(&mut self, parsed, pcm: &mut Vec<i16>, frame_outcomes: Option<&mut Vec<FrameOutcome>>)`,
which `clear()`s and refills the caller's buffers (no allocation of its own), and returns the
channel count. Two thin wrappers sit on it:

- `decode_parsed_packet` (unchanged behaviour) allocates fresh buffers and returns `DecodedAudio`,
  so the public `decode_packet` / `decode_self_delimited_packet` API is byte-for-byte identical.
- **`pub fn decode_packet_into(&mut self, packet, pcm: &mut Vec<i16>) -> Result<u8, Error>`** (new)
  fills the caller's `pcm` (reused across packets) and passes `None` for `frame_outcomes`, so a
  steady decode allocates nothing.

No decode logic changed — the §4.5.2 state resets and the §4.5 frame loop are moved verbatim; the
only edits are the two `Vec::with_capacity` → `clear()`/`reserve()` and the `&mut pcm`/`&pcm` call
sites → `pcm`/`pcm.as_mut_slice()`/`pcm.as_slice()`.

**Verification.** The RFC 6716 12-vector conformance gate (`opus/examples/conformance.rs`) still
passes unchanged (worst 20.9 dB, SILK 96–103 dB) — decoding through the reused-buffer path via
`decode_packet_into`, proving the refactor is behaviour-preserving.

## Allocation-free decode — per-packet `DecodeBump` arena (src/bump.rs + ~15 decode files)

**Why.** `decode_packet_into` removed the *output* Vec churn, but the SILK/CELT decode graph still
allocated **~3270 transient `Vec`s per packet** internally (PVQ, LSF/LPC/LTP, MDCT synthesis, band
decode, resampler, frame parsing, …) — the same banned steady-state heap traffic, one layer down.
A `#[global_allocator]` scoped arena reset per packet **SEGV'd**: the decoder keeps inter-packet
state it allocates *during* decode, and the reset freed it.

**Change (output-preserving).** `src/bump.rs` adds **`DecodeBump`**, a zero-sized `Allocator`
(`#![feature(allocator_api)]`) over a per-thread bump region, rewound by `bump::reset()` once per
packet at each decoder entry point *before* the §3.2 parse. Transient decode buffers are typed
`Vec<T, DecodeBump>`; persistent decoder state (synthesis overlap/history, energy, PLC, carries)
stays `Vec<T>` on the global allocator. **The two are different types, so the compiler forbids
storing a transient into a persistent field** — a per-packet reset can therefore never free live
state (the SEGV is unrepresentable). Escape chains (`BandDecodeResult.x → CeltFrameOutput.x →
synthesis`, the SILK `decode_core → synthesize → layer` chain, `Excitation`, `OpusPacket` frame
list, …) were parameterized compiler-guided until they type-check.

Two exceptions to the arena: (1) the PVQ `V(N,K)` combinatorics leaf (`celt_pvq_v.rs`), called
thousands of times per packet, uses a **reused thread-local scratch** instead — a bump would
accumulate every call until the packet boundary and overflow. (2) `celt_mdct_synthesis`'s carried
overlap/history state uses **reserve-max-once**: `new()` reserves the 2-channel capacity, and a
channel switch rearranges it *in place* (`resize`/`copy_within`/`truncate` within the reserved
capacity, logical length preserved) instead of reallocating.

**Result.** Steady-state (constant geometry) decode is **allocation-free** (0.0000 allocs/packet;
`opus/examples/alloc_check.rs` measures all 12 vectors). Across the full 12-vector run the mean is
**0.031 allocs/packet (down from 3270, −99.999%)**; the residual is transition-only persistent-state
resizes on the concealment path (PLC history on a mono↔stereo toggle + the rare redundant-frame
path), which must stay on the global allocator. Conformance byte-identical throughout.

## Encode alloc-free (CELT path) — 226 → 22 allocs/packet (−90%)

The mirror of the decode work, for the wired **CELT-only encoder** (`OpusEncoder`/`OpusEnc` in
`pf-opus`). `opus/examples/encode_alloc_check.rs` counts steady-state allocs/packet;
`encode_alloc_profile.rs` is a backtrace-capturing allocator that ranks the hottest sites (the same
method the decode work used). Baseline was **175/packet mono, 226 stereo**. Two output-preserving
changes (round-trip + perceptual + ffmpeg/libopus oracle all **byte-identical** after):

1. **PVQ search + quant scratch → stack arrays** (`celt_pvq_encode.rs`). `op_pvq_search`'s `iy`/`y`
   and `alg_quant`'s `signs` were `vec![…]` per coded band — **~90% of all encode allocations**
   (profiler: `op_pvq_search` 133/pkt + `alg_quant` 72/pkt). A band's PVQ dimension `N` is bounded
   by `PVQ_V_N_MAX` = 352, so they are now fixed stack arrays sliced to `n` (zero heap; `alg_quant`
   is a non-nested leaf, so no depth concern). This alone took 226 → ~26.
2. **Persistent range coder** (`range_encoder.rs` + `celt_packet_encode.rs`). `CeltEncoder` now
   holds one `RangeEncoder`, `reset()` (capacity retained) per packet instead of `RangeEncoder::new()`,
   and finalizes via the new **`finish_fixed_into(&mut self, size, out)`** — non-consuming, appends
   the payload straight into the caller's reused buffer after the TOC byte (`finish_fixed` now
   delegates to it). Removes the per-packet output `Vec` + packet `Vec` + range-buffer growth.

**Result.** **18 allocs/pkt mono, 22 stereo** (−90%). The residual tail is per-*frame* analysis
transients across `celt_analysis` (`forward_mdct`, `normalise_bands`, `transient_analysis`),
`celt_frame_encode`, `celt_energy_encode`, and `celt_laplace` — small counts each, not per-band.
Reaching literal 0 needs the decode side's `DecodeBump` treatment (an `EncodeBump` arena reset once
per packet at `encode_celt_frame`'s top, transients → `Vec<T, EncodeBump>`) or per-frame reused
scratch on the persistent `CeltEncoderState`; best done with the RFC vectors on hand to re-verify
bit-exactness. NB: two `silk_packet_encode` end-to-end tests fail on **pristine HEAD** (unrelated to
this CELT work — the handoff's flagged unconfirmed SILK/hybrid encode coverage).
