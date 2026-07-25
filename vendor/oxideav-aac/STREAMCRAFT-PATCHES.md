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
