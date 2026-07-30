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

The top remaining allocator is now a *sized* `from_iter`/`with_capacity` — `ndarray`'s
`&a * &b` / `&a - &b` result arrays in `measure_patch_similarity` — i.e. one alloc per functional-style
matrix op, no longer a per-element grow storm. Eliminating those needs genuine buffer reuse (an arena
threaded through NSIM), a larger change deferred for now.

**Verification.** In-crate inline-data tests unchanged: `convolution_2d::tests::{convolve_with_window,
perform_padding, copy_with_zeros}`, `neurogram_similiarity_index_measure::tests::test_neurogram_measure`,
and `gammatone_filterbank::tests::gammatone_filterbank` all pass. (The `*_data/*.wav`-loading tests fail
on pristine 0.3.1 too — 0.3.1 `exclude`s `test_data` from the package — unrelated.) Always capture with
`heaptrack --record-only` (defer symbol interpretation to `heaptrack_print`).
