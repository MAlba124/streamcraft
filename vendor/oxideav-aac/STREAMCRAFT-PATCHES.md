# streamcraft patches on top of oxideav-aac 0.1.6 (crates.io)

Vendored via `[patch.crates-io]` in the workspace root. Minimal diffs, intended
to be offered upstream.

## 1. IMDCT/MDCT: libm cosine per term of a naive O(N²) transform

`filterbank::imdct` (and its analysis pair `forward_mdct`) evaluated
`cos((2π/N)·(n+n0)·(k+1/2))` with a libm call per (n, k) — 2·10⁶ cosine calls
per length-2048 frame per channel, ~200M/s for stereo 48 kHz. Measured with
perf during movie playback: **40.9% of the whole process in `__cos_fma`**; the
audio decoder pegged a core and could not hold realtime (audible stutter, and
the demuxer rate-locked the video branch to the starved audio ring).

Fix: the inner loop walks the angle by a constant step, so the cosine sequence
satisfies the Chebyshev three-term recurrence
`cos((k+1)θ+φ) = 2cosθ·cos(kθ+φ) − cos((k−1)θ+φ)` — one multiply-add per term,
two libm calls per row. f64 recurrence drift over ≤1024 steps is ~1e−13
relative, far inside conformance tolerances (the crate's own reference suite
gates it).

The real fix upstream is an N·log N IMDCT (FFT decomposition); this patch just
removes the libm wall while keeping the transform's shape and tests intact.

## 2. IMDCT: O(N log N) synthesis via DCT-IV + quarter-length FFT

Patch 1 removed the libm calls but kept the O(N²) direct sum — re-profiled
during hardware-decoded movie playback (perf + simprof, 2026-07-25) it was
still **50.5% of the entire process's cycles** (~2·10⁶ multiply-adds per long
frame per channel).

`filterbank::ImdctPlan` now computes the §4.6.11.3.1 IMDCT in O(N log N):
the spec kernel at phase `n0 = N/4 + 1/2` is a DCT-IV evaluated at
`p = n + N/4` (extension symmetries scatter the M = N/2 DCT-IV values over
the N outputs — the Princen–Bradley TDAC structure; fast form after Duhamel,
Mahieux & Petit, ICASSP 1991), and the DCT-IV reduces to a Q = N/4-point
complex FFT (radix-2 Cooley–Tukey) with pre/post twiddles. Plans (twiddle +
bit-reversal tables) are built once per `Filterbank`.

The literal O(N²) sum stays in-tree as the executable reference;
`fast_imdct_matches_the_reference_sum` pins the two against each other over
dense random spectra at N = 8/256/2048 (observed error ~1e−13, asserted
< 1e−9). The crate suite (727 tests incl. conformance) and sc-aac's ffmpeg
SNR oracle are unchanged. Playback CPU (30 s movie, hw video decode):
22769 → 16427 cycle samples — the audio decoder no longer appears in the
top functions.

## 3. Inverse-quantization x^(4/3) lookup table

`dequant::inverse_quantize` computed `|x|^(4/3)` as `|x|·|x|.cbrt()` per spectral
coefficient (~1024 per frame per channel). Re-profiled once the streamcraft
video path went fully zero-copy (VA-API surface → EGL/GLES, no CPU touch), this
`cbrt` was the single hottest scalar op in audio decode.

Fix: precompute `|x|^(4/3)` for every reachable quantized magnitude
(`0..=8191`, the 13-bit escape range of §4.6.3.3) into a 64 KiB table built once
via `OnceLock`; a magnitude past the table (unreachable from the wire) still
falls back to the direct `cbrt`. Each table entry is the *same* `a·a.cbrt()` the
scalar path produced, so the dequantized spectrum — and every downstream sample
— is byte-for-byte identical; the conformance suite is unaffected. Profiled
`cbrt` self-samples 79 → 2.

## 4. Per-frame allocation churn in the §4.6.11 filterbank

With the video path zero-copy and patches 1–3 in, the remaining audio-decode
cost was dominated by per-frame allocation (`mi_malloc`/`mi_free`/`memset` spread
across the decode), not arithmetic. The filterbank rebuilt constant data and
allocated transform scratch on every channel of every frame:

* **Composite windows were reassembled per frame.** `Filterbank::long_window`
  is a pure function of `(left_shape, right_shape, kind)` — 12 constant vectors —
  yet each call did a 2048-wide zeroed allocation, two `window_halves` calls
  (each cloning cached half-windows and reversing them, even the SHORT halves an
  `OnlyLong` frame never uses), and piecewise copy loops. Now a free function
  backed by a `OnceLock` table: computed once per combo, thereafter returned by
  reference. `window_halves` and `half_window` likewise now return `&'static`
  (cached; no per-call clone/reverse) — the EIGHT_SHORT path hit `window_halves`
  eight times per frame per channel.
* **IMDCT scratch is reused.** `ImdctPlan::imdct` allocated its Q-point FFT
  buffer and M-length DCT-IV buffer (one zeroed) every call; both are now held
  in the plan and reused (every entry is overwritten before use, so no clearing
  is needed). Method threaded `&self` → `&mut self` down from `synthesize`.
* **In-place windowing.** `long_windowed` scaled the IMDCT output by the window
  into a *second* 2048-wide `Vec`; it now multiplies the (about-to-be-dropped)
  IMDCT output in place and returns it.

All results are bit-identical (constant data cached; scratch fully overwritten;
same multiplies), so the 1548-test suite incl. `pcm_byte_exact` conformance and
the `fast_imdct_matches_the_reference_sum` oracle pass unchanged.

## 5. Planar decode + fused interleave-into-caller-buffer (no intermediate PCM Vec)

Re-profiled with patch 4 in, `__memcpy` was the #1 leaf (4.2%) — the audio PCM
handoff, not the (zero-copy) video. The decoder rendered a fresh interleaved
`Vec<i16>` per frame (`decode_raw_data_block` → `pcm::interleave_s16`), and the
streamcraft `aacdec` element then copied that `Vec` into its pipeline pool slot
with a per-sample little-endian write: two passes and an allocation for data
that is produced and consumed once.

Two additive pieces (no behaviour change to existing callers):

* `decode::StreamDecoder::decode_raw_data_block_planar` returns the reordered
  per-channel `f64` time signals (`decode::PlanarFrame`) — the exact data
  `decode_raw_data_block` builds right before it interleaves. The original
  `decode_raw_data_block` is now a thin wrapper (`_planar` + `interleave_s16`),
  so `DecodedFrame`/`decode_all`/LATM/`codec_decoder` are byte-for-byte
  unchanged.
* `pcm::interleave_s16_le_into(channels, dst)` fuses the interleave and the
  `i16 → little-endian bytes` write into one pass straight into a caller buffer,
  with no intermediate `Vec<i16>`. `interleave_le_into_matches_the_two_step_path`
  pins it byte-for-byte against `interleave_s16` + `to_le_bytes`.

The streamcraft element carries the planar frame through its backpressure carry
and renders the PCM directly into the pool slot at emit time — removing one
per-frame allocation and one full copy. Bit-identical (same `to_s16`, same
interleave order, same little-endian layout); 1551 tests pass (1548 + 3 new).

## 6. Spectrum / scalefactor Huffman: linear scan → peek-lookup table

With patches 1–5 in, the §4.A Huffman decode dominated audio-decode CPU (16 kHz
profile: `hcod_sf_decode` 3.4% + `hcod4/2/1` + `decode_index_to_tuple`, ~8%
combined). Each `hcodN_decode` / `hcod_sf_decode` **linearly scanned the whole
codebook (81–121 entries) for every bit read** — `O(entries · max_len)` per
symbol; `hcod_sf` at 19 bits is up to ~2300 comparisons per scalefactor delta.

New `huffman_table::PrefixTable` implements the standard canonical-prefix
peek-lookup: a `2^max_len` table maps the next `max_len` bits to
`(codeword_length, symbol_index)`, so a decode is peek → index → consume —
`O(1)` per symbol. The 12 decoders (`hcod1..hcod11`, `hcod_sf`) now build their
table once (`OnceLock`) from the *same* `HCOD*` statics and call
`PrefixTable::decode`; the end-of-stream tail is handled by peeking the
available bits and left-justifying (a codeword too long for the remaining bits
is `UnexpectedEnd`, exactly as the scan failed). Every decoded index is
identical, so the `hcodN_is_complete` regression tests, the per-book
encode/decode round-trips, and `pcm_byte_exact` conformance all pass unchanged
(1553 total; +2 `PrefixTable` unit tests). Memory: the tables are built lazily
for books actually used; the widest (`hcod_sf`, 2^19 × 4 B = 2 MiB) is a
one-time static.

## 7. Per-frame scratch from a caller-supplied arena allocator

A heaptrack profile (system allocator) of playback showed the decode path
issuing ~10 K small allocations/second — the per-frame `Vec`/`Vec<Vec>` scratch
of the spectral/scalefactor reconstruction. Several were removed outright first
(`read_and_apply_signs` → stack `[bool;4]`; pre-sized scalefactor vecs;
`reconstruct_pre_pair` no longer clones the whole `SpectralData` on the
common no-pulse path — it borrows). The rest are genuine per-frame scratch, so
the decode path is now **generic over a scratch allocator** and the streamcraft
pipeline hands it a frame arena.

`#![feature(allocator_api)]`. The reconstruction entry points gained an
allocator-parameterised form: `dequant::rescale_spectrum_in<A>`,
`decoded_spectrum::quant_to_spec<A>`, `element_decode::decode_sce_in<A>` /
`decode_cpe_in<A>`, and `decode::StreamDecoder::decode_raw_data_block_planar<A>`
— each `A: Allocator + Copy`. The rescaled `Vec<Vec<f64>>` (consumed by
`quant_to_spec` and dropped inside the decode) is allocated from `A`; `spec`
and the PCM output escape into the PNS/TNS/filterbank tail and stay heap `Vec`s.
The original signatures remain as thin `Global` wrappers, so `decode_frame` /
`decode_all` / every test are byte-for-byte unchanged (1553 pass).

`A` is a **trait** (`std::alloc::Allocator`) — the crate names no streamcraft
type and takes no streamcraft dependency, so the patch stays upstreamable.
streamcraft's `Ctx::scratch()` arena implements `Allocator` (a ~20-line
`unsafe impl Allocator for &Arena` in `core/src/memory.rs`; `deallocate` is a
no-op, the scheduler `reset`s the arena after each `process()`), and the
`sc-aac` element passes `ctx.scratch()` into
`decode_raw_data_block_planar` — so the AAC decoder's per-frame scratch now
lives in the pipeline's recycled arena with no steady-state heap traffic.
Currently backs the `rescaled` buffer; the same generic seam extends to the
other transients (`abs`, the band-indexed tables, `spec`) without further
signature churn.
