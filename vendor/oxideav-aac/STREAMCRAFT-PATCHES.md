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
