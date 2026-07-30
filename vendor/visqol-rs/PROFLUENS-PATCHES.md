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
