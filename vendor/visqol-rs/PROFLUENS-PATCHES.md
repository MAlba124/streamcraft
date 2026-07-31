# Profluens patches to `visqol-rs`

Vendored from `visqol-rs` **0.3.1** (crates.io, Apache-2.0 — a pure-Rust port of Google's ViSQOL
v3). Used by `pf-opus`'s `qa` feature as the MOS-LQO perceptual metric for the transcode
quality-search tools (`opus/examples/{quality_search,transcode}.rs`). Vendored (not a crates.io dep)
so we can carry the speedup below; excluded from the workspace, a path dep from `opus/Cargo.toml`.

## Fused gammatone cascade — `gammatone_filterbank.rs::apply_filter`

**Why.** A `perf` profile of a ViSQOL run (`simprof`) put **~53% of all cycles** in
`signal_filter::filter_signal` — the gammatone filterbank — plus a visible `__memset` share.

**The waste.** The 4th-order gammatone filter is a cascade of four 2nd-order sections. The original
`apply_filter` ran them as **four separate full-signal passes per band**, and `filter_signal`
**allocated and zeroed a fresh `Vec<f64>` each pass** — so per call (32 bands × 4 stages): 128
allocations, 128 signal-length memsets, and the whole signal read+written four times per band.

**The change (output-preserving).** Fuse the cascade into **one pass per band**: the four sections'
Direct-Form-II-transposed states live in local variables (registers), and stage *k*'s output feeds
stage *k+1* for the *same* sample before advancing. The signal is read once and the result written
straight to the output row — no intermediate buffers, no per-pass allocation/memset. The float
operations are **the identical operations in the identical order**, so the output is bit-for-bit the
same. No SIMD, no `unsafe`, no nightly features.

**Result.** End-to-end **MOS-LQO is bit-identical** to crates.io 0.3.1 on every probe; the 15-point
ViSQOL quality search dropped from **8.6 s → 4.1 s (2.1×)** in release — i.e. the gammatone is no
longer a bottleneck (its cost was dominated by the allocations + memory traffic the fusion removes).

**Verification.** The in-crate `gammatone_filterbank::tests::gammatone_filterbank` unit test (inline
data) passes unchanged. (`gammatone_spectrogram_builder::tests::test_spec_builder` fails on **both**
the pristine and patched crate — it loads `test_data/…wav`, which 0.3.1 excludes from packaging
(`exclude = ["test_data"]`), so it can't run from the published source. Unrelated to this patch.)

**Next hot spots** (same profile), if more speed is wanted: `convolution_2d` (~17%, a 2D FIR — a
separable-kernel rewrite or SIMD would help) and the `rustfft` spectrogram FFTs.

## Convolution + NSIM allocation storm — `convolution_2d.rs`, `neurogram_similiarity_index_measure.rs`

**Why.** A `heaptrack` profile of a transcode quality-search put the allocation *count* at ~110 M
calls for one song. `hot_stacks --metric allocations` traced **61%** of every allocation to
`perform_valid_2d_conv_with_boundary::flatten_matrix::push → grow_one` and **~6%** to patch clones
in `measure_patch_similarity` — none of the DSP hot path reuses buffers, and the worst offender grew
a `Vec` one element at a time.

**The waste.**
1. `flatten_matrix(&padded_matrix)` — every 2-D convolution (several per patch, thousands of patches)
   built a fresh `Vec<f64>` with `Vec::new()` + a per-element `push`, i.e. a full
   `grow_one`/realloc storm, to flatten a matrix that is **already contiguous row-major** (it comes
   straight from `Array2::zeros`, C-order, and the conv indexes it as `row*ncols + col`).
2. `flatten_matrix(fir_filter)` — same `Vec::new()` growth for the tiny FIR filter, once per conv.
3. `ref_patch.clone() * ref_patch.clone()` (×3 in NSIM) — two full-array clones per product where a
   reference multiply needs none.

**The change (output-preserving).**
1. Borrow the padded matrix's backing store zero-copy: `padded_matrix.as_slice().expect(...)` instead
   of `flatten_matrix(&padded_matrix)`. Bit-identical (same row-major order), zero alloc, zero copy.
2. `Vec::with_capacity(rows*cols)` in `flatten_matrix` (kept for the Fortran-order filter, whose
   row-major flatten is *not* `as_slice`), so the filter flatten is one sized alloc, not a grow storm.
3. `&*ref_patch * &*ref_patch` — element-wise product by reference; identical f64 products, one
   result array instead of clone + clone + result. (The conv fn borrows these `&mut` but never
   mutates them, so the reference product is exact.)

**Result.** On the same input + same converged search result (`fixtures/out/audio.flac`, both runs
picked 44.2 kbps), `heaptrack --record-only`:

| | pristine 0.3.1 | patched | Δ |
|---|---|---|---|
| allocation calls | 4,275,608 | 2,025,008 | **−53%** |
| temporary allocations | 780,779 | 229,481 | **−71%** |
| peak heap | 374.3 M | 382.7 M | ~flat (the flatten `Vec`s were transient — freed at once, never the peak) |

The top remaining allocator was then a *sized* `from_iter`/`with_capacity` — `ndarray`'s
`&a * &b` / `&a - &b` result arrays in `measure_patch_similarity` — i.e. one alloc per functional-style
matrix op, no longer a per-element grow storm. That's what the arena patch below removes.

**Verification.** In-crate inline-data tests unchanged: `convolution_2d::tests::{convolve_with_window,
perform_padding, copy_with_zeros}`, `neurogram_similiarity_index_measure::tests::test_neurogram_measure`,
and `gammatone_filterbank::tests::gammatone_filterbank` all pass. (The `*_data/*.wav`-loading tests fail
on pristine 0.3.1 too — 0.3.1 `exclude`s `test_data` from the package — unrelated.) Always capture with
`heaptrack --record-only` (defer symbol interpretation to `heaptrack_print`).

## Dropped `ndarray-linalg` + `ndarray-stats` — `Cargo.toml`, `math_utils.rs`, `spectrogram.rs`

**Why.** `ndarray-linalg` 0.17 was a **declared-but-entirely-unused** dependency — no `.solve`/`.inv`/
`.eig`/`.svd`/`.dot` anywhere — that drags in a BLAS/LAPACK backend (openblas/cauchy), the slow, fragile
part of the dep tree. `ndarray-stats` 0.6 was used at exactly **four** sites, all `QuantileExt::{min,max}`
(NaN-aware peak-find over finite dB magnitudes).

**The change.** Delete both deps. Replace the four `.min()`/`.max()` calls with `math_utils::{min_of,
max_of}` (`iter.copied().reduce(f64::min|max)` — exact for this NaN-free data). `cargo tree -p pf-opus
--features qa` confirms the whole `ndarray-linalg`/`ndarray-stats`/BLAS/LAPACK subtree is gone; `ndarray`
core stays.

## Arena-backed NSIM/convolution hot loop — `convolution_2d.rs`, `neurogram_similiarity_index_measure.rs`, threaded through the selector/manager

**Why.** After the flatten fix, the top allocator (~1.4 M of 2.0 M) was `ndarray`'s per-op result arrays
in `measure_patch_similarity` — one heap `Array2` for every `&a * &b`, `&a - &b`, plus `out_matrix`/
`padded_matrix` per convolution — none reused, minted afresh for every patch in the O(patches × window)
alignment loop. visqol-rs has no buffer reuse of its own.

**The change.** Thread a **caller-owned [`profluens_core::memory::Arena`]** (bump allocator) down the
call chain — `VisqolManager::run` → `compute_results` → `visqol::calculate_similarity` → the two
`ComparisonPatchesSelector` methods → the `PatchSimilarityComparator::measure_patch_similarity` trait
method. Inside the NSIM/conv code, every transient matrix is now carved from the arena as an
`ArrayViewMut2` (`arena_mat`) and filled with `Zip` (`binop_into`/`map_into`) instead of allocated by
`&a <op> &b`; the convolution writes its output and padding into arena views too, and is generic over
storage (`ArrayBase<S, Ix2>`) so a conv result feeds the next conv with no owned copy. The selector
`arena.reset()`s **per patch** at each of the three call sites — the previous patch's scratch is dead
(its result is the owned `PatchSimilarityResult`), so the arena reuses its chunks and the loop is
allocation-free in steady state. Each parallel-search thread creates its own arena (`Arena` is `!Sync`);
`opus/examples/transcode.rs` owns one per `visqol_mos` call. Bit-identical: the per-element ops and their
order are unchanged (verified by the exact-value inline tests + identical search convergence).

**Result.** Same fixture + same converged output (44.2 kbps) across all three builds, `heaptrack
--record-only`:

| build | allocation calls | temporary allocs | peak heap | vs pristine |
|---|---|---|---|---|
| pristine 0.3.1 | 4,275,608 | 780,779 | 374.3 M | — |
| flatten + clone fix | 2,025,008 | 229,481 | 382.7 M | −53% |
| **+ arena (this)** | **633,788** | **170,794** | 361.7 M | **−85%** |

The remaining ~475 K is the NSIM per-patch *reductions and results* — `mean_axis`/`std_axis` and the
three `to_vec()` outputs that make up the returned `PatchSimilarityResult`, which by definition outlive
the per-patch arena reset — the near-irreducible floor short of redesigning `PatchSimilarityResult` to
borrow. The big per-op matrix scratch (the "100 M allocs for one song" storm) is gone.

**Dep.** This adds a `profluens-core` path dep to the vendored crate (`../../core`) — acceptable since
visqol-rs is vendored specifically for profluens. `Arena`/`ArrayViewMut2::from_shape` glue lives in
`convolution_2d.rs` (`arena_mat`, `binop_into`, `map_into`); the two unsafe `from_raw_parts_mut`s are the
standard "typed view over a fresh, about-to-be-fully-written arena region" (f64 has no invalid bit
patterns; the arena guarantees disjoint regions).

## Allocation-free comparison — the whole remaining per-patch surface

**Why.** After the rounds above, one 30 s ViSQOL comparison — the unit a `transcode --target` /
`--recommend` search runs 15 times, in parallel — still made **7,029 heap allocations and churned
2,494 MiB**. Everything left scaled with the patch count or the alignment slide window, i.e. it was
paid again for every patch of every grid point.

**Measuring it.** `opus/examples/visqol_alloc_check.rs` is the harness this round was driven by:
libopus-encode/decode the WAV fixture, score the pair twice against a counting global allocator (the
first run warms the SVR model, FFT planner, gammatone coefficients and thread-local scratch; the
second is the steady-state number), and print allocations, bytes, **peak live bytes**, and the exact
MOS bits. `VISQOL_ALLOC_PROFILE=1` samples backtraces and prints the call sites, so each round's next
target is attributable without heaptrack. Knobs: `VISQOL_ALLOC_SECS` (default 30, loops a short
fixture), `VISQOL_ALLOC_BITRATE`, `VISQOL_ALLOC_SAMPLE`.

**The changes**, in descending order of what they removed:

1. **NSIM per-patch results → inline `BandValues`** (−740). ViSQOL has 21 or 32 bands, but
   `PatchSimilarityResult` held three `Vec<f64>`s — the one thing `measure_patch_similarity` produces
   that must outlive the per-patch arena reset, so three heap allocations per patch per call site.
   They are now a `[f64; 32] + len` that `Deref`s to `&[f64]` (callers index/iterate unchanged;
   `Serialize` emits the same sequence). The `std_axis(Axis(1), 1.0)` that fed one of them is
   replaced by `band_std_dev`, which reproduces ndarray 0.16's Welford recurrence element for element
   (`delta`, `mean += delta/(i+1)`, `sum_sq = (x-mean).mul_add(delta, sum_sq)`, `sqrt(sum_sq/(n-1))`)
   — bit-identical, and it drops ndarray's two result `Array1`s as well.
2. **The alignment stage's transforms → arena** (−3,100, and ~1.8 GiB of churn). `forward_1d_*`,
   `inverse_1d*`, `calculate_hilbert`, `calculate_upper_env`, `calculate_best_lag` and
   `calculate_fft_pointwise_product` all minted `Vec`s per call (~30 per `globally_align`, ~74 calls
   per comparison); they now carve from a bump arena. Several copies vanished outright rather than
   moving: `freq_from_time_domain` zero-pads in its thread-local complex staging buffer instead of
   resizing the caller's `Vec` (so no padded copy per transform), `inverse_1d_conj_sym` returns the
   real time-domain buffer directly (the old form built an all-real complex vector and copied the
   real parts back out), and `calculate_best_lag` indexes the wrap-around correlation window in place
   instead of materialising it as `to_vec` + `to_vec` + `append`.
3. **The fine-realignment loop → reusable buffers** (−900). The per-patch sliced audio, the aligned
   pair and the two spectrograms were freshly allocated every patch. `slice_into` /
   `align_and_truncate_into` / `globally_align_into` / `GammatoneSpectrogramBuilder::build_into` now
   fill caller-owned storage that the loop reuses, so it only grows on the first patch or two.
   `build_into` reclaims the destination `Spectrogram`'s backing `Vec` (`into_raw_vec_and_offset` →
   `resize` → `from_shape_vec`) and refills `center_freq_bands` in place instead of cloning it.
4. **`build_degraded_patch` → arena** (−1,880). The degraded candidate patch was
   `slice(..).to_owned()` plus, for offsets spilling past the end, an `Array2::zeros` and a
   `concatenate` — three allocations per call inside the O(patches × window) slide loop. It writes an
   arena matrix now. The `PatchSimilarityComparator` trait went generic over patch storage (shared
   borrows) to accept it, which also removed the `ref_patches[i].clone()` per backtrace step.
5. **Reference patches → zero-copy views** (−80). `create_patches_from_indices` returns
   `Vec<ArrayView2>`; a patch is a column range of the reference spectrogram and every consumer is
   generic over storage, so copying each one out was pure cost.
6. **DP tables → flat** (−148). `cumulative_similarity_dp`/`backtrace` were `Vec<Vec<_>>` — one
   allocation per reference patch for rows that all have the same length.

**Peak memory — the trap in this round.** Bump arenas trade allocation count for *retention*: naively
arena-backing the alignment took peak live heap from 290 MiB to 538 MiB, because the one whole-signal
`globally_align` (a 30 s clip needs a 4 M-point FFT — hundreds of MiB of transform scratch) can never
free the middle of an arena. Two fixes, both in `globally_align_into`: the envelopes go in a separate
`out` arena from the transform `scratch` (which is reset between the three phases), and the scratch is
handed one up-front chunk sized for the largest phase — a bump arena only reuses a chunk across
`reset` if the next phase's allocations still fit it, and the cross-correlation's buffers are twice the
envelope's, so without the hint all three phases stacked. `compute_results` gives that one call its own
pair of arenas, dropped immediately, so the caller's long-lived per-thread arena never grows to
full-signal size.

**Peak memory — where it actually goes.** With the above, the whole-signal alignment's scratch is a
*single* 234 MiB chunk: no dead scratch stacks up, so there is nothing a mark/release ("sub-arena")
API could reclaim — the peak is one phase's simultaneously-live data. What *did* reduce it was
removing a redundant live buffer: `calculate_fft_pointwise_product` wrote the spectrum product into a
third `fft_points`-long complex buffer when its left operand is dead the moment the product is
formed. Multiplying in place drops one 64 MiB spectrum from a 4 M-point transform: **peak live
279 → 215 MiB, maxrss 563 → 497 MiB.** (The chunk hint above follows: `2*16 + 8` bytes per point,
not `3*16 + 8`.)

**Result.** One 30 s comparison (`fixtures/out/audio.wav` looped, libopus-degraded):

| | allocations | bytes churned | peak live | wall (2 comparisons) | maxrss |
|---|---|---|---|---|---|
| before this round | 7,029 | 2,494 MiB | 290 MiB | 8.0–8.8 s | 583 MiB |
| **after** | **51** | **265 MiB** | **215 MiB** | **7.0–7.9 s** | **497 MiB** |

Nothing left scales with the patch count: the 51 are the per-comparison spectrograms, the one global
alignment, the DP tables, the arena chunks and the `SimilarityResult` outputs. Per-patch and
per-slide-offset work allocates **nothing**.

**Verification.** The MOS-LQO is **bit-identical** — `4.73136043548583984` on the 30 s fixture, checked
after every individual change, and identical bit patterns at four degradation levels (8/12/16/24 kbps:
`4012e24020000000`, `4012c36120000000`, `4012420480000000`, `4012ecbb80000000`). A full
`transcode --target 4.5` search produces a **byte-identical** `.opus` (`769f03f8…`) and the same 15-point
rate-distortion table. In-crate tests: **37 pass**, the same 11 fail as on pristine 0.3.1 (they load
`test_data/*.wav`, which 0.3.1 `exclude`s from the package).

## The envelope's Hilbert transform was a no-op — `envelope.rs`

**Why.** A `perf` profile of one comparison put ~13% of cycles in `rustfft`. Reading the call chain
to see whether a real-input transform was worth it turned up something better: two of the three FFTs
were computing the identity.

**The finding.** `calculate_upper_env` is written as an analytic-signal envelope — zero the negative
frequencies, double the positive ones, inverse-transform, take `.norm()`. But
`fast_fourier_transform::inverse_1d` keeps only the **real** part of the inverse transform and leaves
the imaginary part at zero, so the `.norm()` reduces to `|re|` — and the real part of the analytic
signal is the input itself. The imaginary part, the Hilbert transform that would make this an
envelope, is discarded. (Upstream 0.3.1 says so in a comment: *"This makes very little sense but oh
well…"*.) The whole two-FFT round trip therefore computes `|2·(x − mean) − 1e-6| + mean`.

Measured on `fixtures/out/audio.wav`: the FFT formulation and that closed form agree to **3.3e-16
absolute, 1.3e-15 relative** — pure round-trip rounding.

**The change.** `calculate_upper_env` is the closed form, a single pass with no transform, no scratch
arena and no `FftManager`. The FFT formulation stays as `calculate_upper_env_via_hilbert` under
`#[cfg(test)]`, the executable reference; `matches_hilbert_reference` asserts both that the envelopes
agree (< 1e-14) and that `calculate_best_lag` picks the identical lag from them — the envelope's only
consumer is that lag. `globally_align_into` loses two of its three phases, so its scratch arena now
sizes for the cross-correlation alone.

**Result.** MOS-LQO **bit-identical across a nine-point 6–96 kbps degradation sweep**, and
`transcode --target 4.5` still emits a byte-identical `.opus`. Wall time for two 30 s comparisons
**7.0–7.9 s → 5.9–7.3 s**; maxrss **497 → 429 MiB** (the 2 M-point rustfft plan's twiddle tables are
never built). The 15-point parallel search drops 1.4 s → 1.1 s. In-crate tests 38 pass (the same 11
`test_data`-loading failures as pristine 0.3.1).

**Caveat, and why the reference is kept.** This is bit-identical *by measurement*, not by
construction: the envelope differs from the old one at the 1e-16 level, and it feeds an integer
arg-max, which is insensitive to that unless two lag candidates tie to within 1e-15. If `inverse_1d`
is ever fixed to carry the imaginary part — which is what reference ViSQOL intends — this fast path
becomes wrong and `calculate_upper_env_via_hilbert` must take over. The doc comment says so at the
call site.

**Profile after** (perf, self time): gammatone filterbank 36.5%, NSIM convolution 27.6% (kernel +
padding), `rustfft` 9%, ndarray `Zip`/`map_inplace` 11.5%. A real-input (r2c/c2r) cross-correlation
would take roughly half the remaining 9% and shrink the scratch chunk from `40·F` to ~`24·F` bytes,
but needs a new dependency and is genuinely not bit-identical — held.

## NSIM convolution round — dead padding memset, row copies, fused passes

**Why.** With the envelope FFTs gone the profile was gammatone 36.5%, NSIM convolution 27.6% (kernel
plus its padding), ndarray `Zip`/`map_inplace` 11.5%, `rustfft` 9%. The middle two are the NSIM core,
which runs five convolutions and seventeen element-wise passes per patch comparison — inside the
O(patches × window) alignment slide loop.

**The changes**, all bit-identical by construction:

1. **The padding zero-fill is dead work.** `copy_matrix_within_padding` memset the whole padded
   matrix, but `add_matrix_boundary` then writes *every* padded cell — the row loop covers the whole
   first and last row, the column loop the whole first and last column (re-writing the four corners,
   which is what makes them well-defined). Nothing reads a padded cell before it is assigned.
   Confirmed by filling with `NaN` instead and getting a bit-identical MOS. `add_matrix_boundary` now
   skips the fill; the `pub` zero-padding form keeps it (its test asserts zeros).
2. **The padded copy is row-at-a-time** (`copy_from_slice` when both rows are contiguous, which is
   the common case — most convolution inputs are the previous stage's `arena_mat`), instead of an
   element-by-element `[(i, j)]` loop through ndarray's indexing.
3. **`compute_sim_map`'s element-wise work is fused into three passes** separated by the
   convolutions, which are the only stages with a cross-element dependency. Nine of the intermediate
   matrices existed only to carry a value to the next line and are now locals; the intensity term and
   the final map share one buffer. Element-wise stages are independent per element, so fusing leaves
   each element's own sequence of `f64` operations exactly as it was.

**Result** (`perf stat`, two 30 s comparisons, the two builds run **interleaved** — this box drifts
several percent with thermal state, so sequential before/after runs overstate a win by ~10%):

| | instructions | cycles |
|---|---|---|
| before | 60.51 G | ~21.81 G |
| **after** | **57.54 G** (−4.9%) | **~21.30 G** (−2.3%) |

`memset` leaves the profile entirely; ndarray `Zip` drops 11.5% → 2.5%; the padding helper
3.7% → 1.8%. The cycle win is smaller than the profile shares suggest because what these passes
freed is mostly memory bandwidth the convolution kernel was not waiting on.

**Verification.** MOS-LQO bit-identical across the 6–96 kbps sweep; `transcode --target 4.5`
byte-identical; in-crate tests 38 pass / same 11 pre-existing `test_data` failures.

**Profile after**: gammatone 39.7%, NSIM convolution kernel 24.7%, `rustfft` ~9%, everything else
below 3%. Both leaders are already structure-of-arrays and AVX2 — the gammatone keeps every
coefficient and every filter state as a `[f64; NUM_BANDS]` lane per band, and the convolution
vectorises along the contiguous frame axis — so further gains there need an algorithmic change, not a
layout one. (The 3×3 NSIM window is *not* separable — a separable rewrite would need
`w[0][1] = 0.08367` where the actual value is `0.08383` — so the 9→6 MAC form would shift scores.)

## Specialised 3×3 convolution kernel — `convolution_2d.rs`

**Why.** After the round above the convolution kernel was 24.7% of cycles, running at roughly
1 flop/cycle — about a tenth of what AVX2 can retire. The generic kernel walks the filter backwards
through `filter[filter_index]` with `filter_index.saturating_sub(1)`: a serial scalar chain, a load
per tap, and an `(f_row + o_row) * i_c_c` multiply per tap.

**The change.** The NSIM smoothing window is fixed at 3×3, so every convolution in the hot path is
3×3. `conv2d_valid_3x3` specialises it: the nine taps are hoisted into a `[f64; 9]` indexed by a
constant once the (constant-bound) loops unroll, and the three source-row bases are computed once per
output row. The generic kernel stays for any other filter shape. The accumulation order — `f_col`
outer, `f_row` inner, taps from `filter[8]` down to `filter[0]` — is exactly the generic kernel's, so
the sums are bit-identical.

**Result** (interleaved runs, two 30 s comparisons):

| | instructions | cycles |
|---|---|---|
| before | 57.54 G | ~21.49 G |
| **after** | **45.95 G** (−20.1%) | **~19.74 G** (−8.1%) |

Five of five paired runs favoured the specialisation. Across the whole convolution round
(`aa9314a` → here): **instructions −24.1%, cycles −9.9%**; the 15-point parallel search 1.2 s → 1.0 s.

**Verification.** MOS-LQO bit-identical across the nine-point 6–96 kbps sweep; `transcode --target
4.5` byte-identical; tests 38 pass / same 11 pre-existing failures.

**A measurement note for future rounds.** This box drifts several percent with thermal state, enough
to invent or hide a win of this size. Compare builds **interleaved** (`for i in 1..n { run A; run B }`)
and read `cycles:u`, not wall time; `instructions:u` is deterministic to 8 significant figures and is
the better signal when the two disagree. A sequential before/after measurement of the previous round
reported −12.7% cycles where the interleaved figure is −2.3%.
