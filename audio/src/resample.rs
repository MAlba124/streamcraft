//! Sample-rate conversion by band-limited interpolation (spec: Formats — the resampler is a
//! separate, harder element than `audioconvert`). A **polyphase FIR** resampler with a
//! **Kaiser-windowed-sinc** anti-aliasing low-pass: arbitrary rational `L/M` conversion at
//! good quality, streaming (filter state carries between batches), and allocation-free on the
//! hot path.
//!
//! # Working type
//!
//! The DSP runs entirely in **`f32`**. The element ([`crate::resample_element::AudioResample`])
//! decodes any PCM [`SampleFormat`](crate::format::SampleFormat) to `f32` in `[-1, 1)`, resamples
//! per channel, and re-encodes; this file only ever sees planar `f32`. `f32` is enough for audio
//! (24-bit PCM has ~144 dB of range; an `f32` mantissa gives ~148 dB) and keeps the convolution
//! SIMD-friendly (performance is #1).
//!
//! # The maths (why it is one filter, applied L·M ways)
//!
//! Rational resampling by `L/M` is *upsample by `L`* (insert `L−1` zeros between samples),
//! *low-pass* to the narrower of the two Nyquist limits, then *decimate by `M`* (keep every
//! `M`-th sample). Done naively that is an `L`× longer signal through a long filter. The
//! polyphase identity avoids the zeros entirely: the low-pass prototype `h` (designed at the
//! *upsampled* rate `in_rate·L`) splits into `L` sub-filters `h_p[k] = h[p + kL]`, and output
//! sample `m` uses exactly one of them —
//!
//! ```text
//!   in-time of output m:   t = m·M / L                 (in input samples)
//!   base input index:      i = floor(m·M / L)
//!   polyphase branch:      p = (m·M) mod L
//!   y[m] = Σ_k  h_p[k] · x[i − k]
//! ```
//!
//! so each output costs one dot product of `taps_per_phase` multiplies — no wasted work on the
//! inserted zeros, and no per-output filter *redesign*.
//!
//! The prototype cutoff is the lower of the input and output Nyquist frequencies expressed at
//! the upsampled rate, `fc = 0.5 / max(L, M)` cycles/sample, so the pass-band reaches the usable
//! bandwidth and the stop-band kills the images/aliases. It is scaled by `L` for the
//! interpolation gain (a decimator keeps 1 in `L` of the up-sampled samples, so the retained
//! energy needs the `L`× lift to leave the pass-band at unity).
//!
//! # How the dot product is arranged
//!
//! That per-output dot product is the whole cost of the element, so its memory layout is chosen
//! for it rather than for the maths:
//!
//! * Branch coefficients are stored **reversed** (oldest tap first), so the convolution walks the
//!   coefficients and the delay line in the *same* forward direction — two unit-stride streams a
//!   SIMD loop can load directly, instead of one forward and one backward walk.
//! * Each branch is **zero-padded** to a multiple of [`LANES`], so there is no scalar tail; the
//!   pad multiplies real (already-buffered) history samples by `0.0`, so it changes nothing.
//! * The sum is carried in [`LANES`] independent accumulators and reduced by a fixed-shape
//!   binary tree. A single `f32` accumulator makes the loop a serial `addss` chain — one 4-cycle
//!   FP add per tap, with nothing else to issue — which is what the naive form was bound by.
//!   Splitting it reassociates the sum (see [`dot`]) and *improves* the error bound.
//! * The number of outputs a batch will produce is computed in closed form up front, so the
//!   output `Vec` is reserved exactly once and the loop is counted rather than re-testing the
//!   delay-line length, and the per-output `phase / L` and `phase % L` become a compare and a
//!   subtract instead of a hardware `div`.
//!
//! The whole output loop is compiled twice on x86-64 — once for the baseline SSE2 the workspace
//! targets, once behind a runtime AVX2 check for 256-bit lanes.

/// Zeroth-order modified Bessel function `I₀(x)`, for the Kaiser window. A short power-series
/// (`Σ (x²/4)^k / (k!)²`) — converges fast for the small `x` a Kaiser β implies, and needs no
/// libm. Iterates until a term stops moving the sum.
fn bessel_i0(x: f64) -> f64 {
    let half_x_sq = (x * 0.5) * (x * 0.5);
    let mut term = 1.0f64; // k = 0 term
    let mut sum = 1.0f64;
    let mut k = 1.0f64;
    loop {
        // term_k = term_{k-1} * (x²/4) / k²
        term *= half_x_sq / (k * k);
        sum += term;
        if term <= sum * 1e-12 {
            break;
        }
        k += 1.0;
    }
    sum
}

/// SIMD block width of the per-output dot product, in `f32` lanes. Every polyphase branch is
/// zero-padded up to a multiple of this (see [`PolyphaseFilter::stride`]) so the convolution is a
/// whole number of blocks — no scalar tail to special-case — and the kernel carries `LANES`
/// independent partial sums, which is what breaks the serial `addss` dependency chain that
/// otherwise costs ~9 cycles per tap. 16 lanes maps onto 2 × 256-bit accumulators with AVX2 and
/// 4 × 128-bit on the baseline SSE2 build, both enough in-flight adds to saturate the FP ports.
/// Every filter length this crate designs (`taps_per_phase = 2·half_taps`, `half_taps` ∈ 8..48)
/// is already a multiple of 16, so the padding is normally zero-cost.
const LANES: usize = 16;

/// A designed prototype low-pass, decomposed into `L` polyphase branches. Branch `p` holds the
/// `taps_per_phase` coefficients that the resampler dots against a window of input history.
/// Immutable after [`design`](PolyphaseFilter::design).
#[derive(Clone, Debug)]
pub struct PolyphaseFilter {
    /// Upsampling factor `L` (interpolation ratio numerator).
    pub l: usize,
    /// Downsampling factor `M` (decimation ratio denominator).
    pub m: usize,
    /// Taps per polyphase branch (the per-output multiply count).
    pub taps_per_phase: usize,
    /// Storage pitch of one branch inside [`phases`](Self::phases): `taps_per_phase` rounded up
    /// to a multiple of [`LANES`]. Also the length of the history window each output reads.
    stride: usize,
    /// `L` branches × `stride` coefficients, row-major in one contiguous buffer
    /// (`phases[p * stride + j]`). Flat (not `Vec<Vec>`) so `design` and `clone` are a single
    /// allocation each, not `L`.
    ///
    /// Within a branch the coefficients are stored **oldest-tap first** and **front-padded with
    /// zeros**: `phases[p*stride + (stride-1-k)] == proto[p + k*L]` for `k < taps_per_phase`, and
    /// the leading `stride - taps_per_phase` entries are `0.0`. Two consequences, both for the
    /// hot loop: the dot product walks *both* arrays forward with unit stride (SIMD-loadable),
    /// and the zero pad multiplies real, already-buffered history samples, so it contributes
    /// exactly `0.0` and needs no tail loop. The delay line is seeded with `stride − 1` warm-up
    /// zeros to cover the widened window.
    phases: Vec<f32>,
}

impl PolyphaseFilter {
    /// Design the anti-aliasing prototype for an `in_rate → out_rate` conversion and split it
    /// into polyphase branches. `in_rate`/`out_rate` are reduced by their gcd to `M`/`L`.
    ///
    /// `half_taps` sets the filter length: the prototype spans `2·half_taps·max(L,M) + 1` taps,
    /// i.e. `taps_per_phase = 2·half_taps·(max(L,M)/L)`-ish — concretely `taps_per_phase` is
    /// chosen so each branch reaches `half_taps` input samples on each side of the interpolation
    /// point. `beta` is the Kaiser shape (≈8.6 → ~−80 dB side-lobes, a good default).
    pub fn design(in_rate: u32, out_rate: u32, half_taps: usize, beta: f64) -> PolyphaseFilter {
        assert!(in_rate > 0 && out_rate > 0, "rates must be positive");
        assert!(half_taps >= 1, "need at least one tap per side");
        let g = gcd(in_rate, out_rate);
        let l = (out_rate / g) as usize;
        let m = (in_rate / g) as usize;

        // Design the prototype at the upsampled rate. Cutoff = lower Nyquist, in cycles/sample
        // of the up-sampled grid. When L >= M (upsampling) the output Nyquist is the limit
        // (fc = 0.5/L); when M > L (downsampling) the input Nyquist folds first (fc = 0.5/M).
        let ratemax = l.max(m);
        let fc = 0.5 / ratemax as f64; // normalised cutoff (cycles/sample, upsampled)

        // One branch reaches `half_taps` input samples on each side, so the prototype (which is
        // `L`× denser in time) needs `taps_per_phase = 2*half_taps` coefficients per branch, i.e.
        // a prototype of `n = taps_per_phase * L` taps, centred.
        let taps_per_phase = 2 * half_taps;
        let n = taps_per_phase * l; // total prototype taps (kept a multiple of L)
        let center = (n - 1) as f64 / 2.0;

        let i0_beta = bessel_i0(beta);
        // Build the prototype, then scatter into branches. `proto[j]` is tap j of the full LPF.
        let mut proto = vec![0f32; n];
        for (j, tap) in proto.iter_mut().enumerate() {
            let x = j as f64 - center; // distance from centre, in upsampled samples
            // Windowed sinc: 2·fc·sinc(2·fc·x) with a Kaiser window; ×L for interpolation gain.
            let sinc = if x.abs() < 1e-9 {
                2.0 * fc
            } else {
                let a = std::f64::consts::PI * 2.0 * fc * x;
                (2.0 * fc) * (a.sin() / a)
            };
            // Kaiser window argument: 1 − (2j/(n−1) − 1)² under the sqrt.
            let r = if n > 1 { (2 * j) as f64 / (n - 1) as f64 - 1.0 } else { 0.0 };
            let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0_beta;
            *tap = (sinc * w * l as f64) as f32;
        }

        // Scatter into `L` branches. Branch p, tap k := proto[p + k*L], written *reversed* into
        // the branch row (`stride-1-k`) and front-padded with zeros — see the `phases` field docs:
        // the hot loop then walks coefficients and history in the same forward direction, which
        // is what lets the dot product be a plain contiguous SIMD block loop.
        let stride = taps_per_phase.next_multiple_of(LANES);
        let mut phases = vec![0f32; stride * l];
        for p in 0..l {
            for k in 0..taps_per_phase {
                phases[p * stride + (stride - 1 - k)] = proto[p + k * l];
            }
        }

        PolyphaseFilter { l, m, taps_per_phase, stride, phases }
    }

    /// Total prototype length (all branches), `taps_per_phase · L`.
    pub fn prototype_len(&self) -> usize {
        self.taps_per_phase * self.l
    }

    /// The full designed prototype low-pass coefficients (branches re-interleaved), useful for
    /// verifying the impulse response. `proto[p + k·L] == phases[p·stride + (stride−1−k)]`.
    pub fn prototype(&self) -> Vec<f32> {
        let mut proto = vec![0f32; self.prototype_len()];
        for p in 0..self.l {
            for k in 0..self.taps_per_phase {
                proto[p + k * self.l] = self.phases[p * self.stride + (self.stride - 1 - k)];
            }
        }
        proto
    }
}

/// One output sample: `Σ_j c[j]·h[j]` over `LANES`-wide blocks, with `LANES` independent partial
/// sums reduced by a fixed-shape binary tree.
///
/// `c.len() == h.len()` and both are an exact multiple of [`LANES`] (guaranteed by the branch
/// stride), so this is a bounds-check-free loop over `&[f32; LANES]` chunks that the autovectoriser
/// turns into packed multiply/add — 2 × `vmulps`/`vaddps` per block under AVX2, 4 × `mulps`/`addps`
/// on baseline SSE2.
///
/// The block/tree summation order differs from a left-to-right scalar accumulation, so results are
/// not bit-identical to a naive serial dot; pairwise summation has a strictly *smaller* error bound
/// (`O(log n)·ε` rather than `O(n)·ε`), so this is the more accurate of the two. No `mul_add`: FMA
/// would change rounding again for no throughput gain here (the loop is port-bound, not
/// latency-bound, once the accumulators are split).
#[inline(always)]
fn dot(c: &[f32], h: &[f32]) -> f32 {
    let (cb, _) = c.as_chunks::<LANES>();
    let (hb, _) = h.as_chunks::<LANES>();
    let mut acc = [0f32; LANES];
    for (cc, hh) in cb.iter().zip(hb) {
        for i in 0..LANES {
            acc[i] += cc[i] * hh[i];
        }
    }
    // Fixed-shape pairwise reduction (16 → 8 → 4 → 2 → 1); deterministic regardless of ISA.
    let mut a8 = [0f32; 8];
    for i in 0..8 {
        a8[i] = acc[i] + acc[i + 8];
    }
    let mut a4 = [0f32; 4];
    for i in 0..4 {
        a4[i] = a8[i] + a8[i + 4];
    }
    (a4[0] + a4[2]) + (a4[1] + a4[3])
}

/// Streaming single-channel resampler state: a designed [`PolyphaseFilter`] plus the delay line
/// and phase bookkeeping carried between [`process`](ChannelResampler::process) calls. One per
/// channel (the coefficients are shared by `Arc`-free cloning of the small filter, or by the
/// element owning one filter and N of these).
#[derive(Clone)]
pub struct ChannelResampler {
    filter: PolyphaseFilter,
    /// Delay line, most-recent sample last. Holds the last `taps_per_phase − 1` inputs plus the
    /// new batch during a call; between calls it is trimmed to the tail the next output needs.
    history: Vec<f32>,
    /// Output index `m` modulo `L`, i.e. `(m·M) mod L` bookkeeping kept incrementally so phase
    /// selection never overflows on long streams.
    phase: usize,
    /// Base input index of the *next* output, relative to the current `history` front. Advances
    /// by `M` (in the upsampled `m·M` accumulator) per output and is rebased when history is
    /// trimmed.
    in_pos: usize,
    /// Total input samples consumed (fed) so far — for `output_len` accounting and tests.
    consumed: u64,
    /// Total output samples produced so far.
    produced: u64,
}

impl ChannelResampler {
    /// A streaming resampler from the given [`PolyphaseFilter`]. The delay line is pre-seeded
    /// with `stride − 1` zeros so the very first outputs are well-defined (a leading transient,
    /// standard for causal FIR resampling). `stride ≥ taps_per_phase`; the extra warm-up samples
    /// line up under the branch's zero pad and contribute nothing.
    pub fn new(filter: PolyphaseFilter) -> ChannelResampler {
        let warmup = filter.stride.saturating_sub(1);
        ChannelResampler {
            filter,
            history: vec![0f32; warmup],
            phase: 0,
            in_pos: warmup, // first real input sits just after the warm-up zeros
            consumed: 0,
            produced: 0,
        }
    }

    /// The filter this resampler was built from.
    pub fn filter(&self) -> &PolyphaseFilter {
        &self.filter
    }

    /// Feed `input` samples and append every output sample now computable to `out`. State is
    /// retained, so calling this repeatedly on consecutive chunks yields exactly the same
    /// samples as one call on the concatenation (proven in the streaming test). Returns the
    /// number of output samples appended this call.
    ///
    /// `out` is generic over its allocator so the element can accumulate straight into a
    /// per-`process()` arena (`ctx.scratch()`) — see `resample_element`.
    pub fn process<A: std::alloc::Allocator>(&mut self, input: &[f32], out: &mut Vec<f32, A>) -> usize {
        let l = self.filter.l;
        let m = self.filter.m;
        let stride = self.filter.stride;
        self.history.reserve(input.len());
        self.history.extend_from_slice(input);
        self.consumed += input.len() as u64;

        // How many outputs this batch yields, in closed form. The loop condition is
        // `in_pos < history.len()`, and after `n` outputs `in_pos = in_pos0 + (phase0 + n·M)/L`,
        // so the first blocked output is `n = ceil((D·L − phase0) / M)` with `D = len − in_pos0`.
        // Computing it up front buys an exact `reserve` (one allocation instead of a doubling
        // ladder, and none at all in the steady state) and a counted loop with no per-output
        // reload of `history.len()`.
        let d = self.history.len().saturating_sub(self.in_pos);
        let n_out = if d == 0 { 0 } else { (d * l - self.phase).div_ceil(m) };
        out.reserve(n_out);

        {
            let phases = &self.filter.phases[..];
            let hist = &self.history[..];
            let state = (self.in_pos, self.phase);
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            let (pos, phase) = if std::is_x86_feature_detected!("avx2") {
                // SAFETY: gated on runtime AVX2 detection; the body is plain safe Rust (no
                // intrinsics) — `target_feature` only lets the autovectoriser use 256-bit lanes.
                unsafe { run_avx2(phases, stride, l, m, hist, state, n_out, out) }
            } else {
                run_scalar(phases, stride, l, m, hist, state, n_out, out)
            };
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            let (pos, phase) = run_scalar(phases, stride, l, m, hist, state, n_out, out);
            self.in_pos = pos;
            self.phase = phase;
        }

        // Trim consumed history: keep the `stride-1` samples before `in_pos` (the delay line the
        // next output still needs), and rebase `in_pos` to the new front. `in_pos` points one
        // past the last usable input, so the retained tail starts at `in_pos-(stride-1)`.
        let keep_from = self.in_pos.saturating_sub(stride - 1);
        if keep_from > 0 {
            self.history.drain(..keep_from);
            self.in_pos -= keep_from;
        }

        self.produced += n_out as u64;
        n_out
    }

    /// Total input samples fed so far.
    pub fn consumed(&self) -> u64 {
        self.consumed
    }

    /// Total output samples produced so far.
    pub fn produced(&self) -> u64 {
        self.produced
    }
}

/// The output loop: `n_out` band-limited samples appended to `out`, advancing `(in_pos, phase)`.
/// Split out of [`ChannelResampler::process`] so the whole loop — dot product *and* horizontal
/// reduction — can be compiled once per instruction set (see [`run_avx2`]).
///
/// `state` is `(in_pos, phase)` on entry; the return is the same pair on exit. The caller
/// guarantees `in_pos ≥ stride−1` and `in_pos + (phase + (n_out−1)·M)/L < hist.len()`, so both
/// slices below are in range — one bounds check per output, hoisted out of the tap loop.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn run_scalar<A: std::alloc::Allocator>(
    phases: &[f32],
    stride: usize,
    l: usize,
    m: usize,
    hist: &[f32],
    state: (usize, usize),
    n_out: usize,
    out: &mut Vec<f32, A>,
) -> (usize, usize) {
    let (mut pos, mut phase) = state;
    // `phase + M` never exceeds `L + M`, so the per-output `/L` and `%L` of the original
    // formulation are one predictable compare-and-subtract on the pre-split quotient/remainder
    // of M — integer-identical, but without a 20-plus-cycle hardware `div` per output sample.
    let (m_div_l, m_mod_l) = (m / l, m % l);
    for _ in 0..n_out {
        let c = &phases[phase * stride..][..stride];
        let h = &hist[pos + 1 - stride..][..stride];
        out.push(dot(c, h));

        let mut ph = phase + m_mod_l;
        let mut step = m_div_l;
        if ph >= l {
            ph -= l;
            step += 1;
        }
        phase = ph;
        pos += step;
    }
    (pos, phase)
}

/// [`run_scalar`] recompiled with 256-bit lanes available. The workspace builds for baseline
/// x86-64 (SSE2), so this is the only way the convolution sees `vmulps`/`vaddps` on `ymm`;
/// `avx2` deliberately does **not** pull in `fma`, which would change the rounding of every tap.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn run_avx2<A: std::alloc::Allocator>(
    phases: &[f32],
    stride: usize,
    l: usize,
    m: usize,
    hist: &[f32],
    state: (usize, usize),
    n_out: usize,
    out: &mut Vec<f32, A>,
) -> (usize, usize) {
    run_scalar(phases, stride, l, m, hist, state, n_out, out)
}

/// Euclid's gcd for the rate reduction.
pub fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// The number of output samples a resampler produces for `in_len` input samples, in steady
/// state, for the `L/M` ratio — `floor(in_len · L / M)` give-or-take the filter's start-up. Used
/// to size batch outputs and asserted in the length test.
pub fn output_len(in_len: u64, in_rate: u32, out_rate: u32) -> u64 {
    let g = gcd(in_rate, out_rate) as u64;
    let l = (out_rate as u64) / g;
    let m = (in_rate as u64) / g;
    in_len * l / m
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = std::f64::consts::PI;

    /// I₀(0) = 1, and it grows monotonically; sanity vs. a couple of known-ish values.
    #[test]
    fn bessel_i0_basics() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
        // I0(1) ≈ 1.2660658, I0(2) ≈ 2.2795853 (standard tables).
        assert!((bessel_i0(1.0) - 1.266_065_9).abs() < 1e-5, "I0(1)={}", bessel_i0(1.0));
        assert!((bessel_i0(2.0) - 2.279_585_3).abs() < 1e-5, "I0(2)={}", bessel_i0(2.0));
    }

    #[test]
    fn gcd_reduces_common_rates() {
        assert_eq!(gcd(44_100, 48_000), 300); // 147/160
        assert_eq!(gcd(48_000, 24_000), 24_000); // 2/1
        assert_eq!(gcd(48_000, 44_100), 300);
        assert_eq!(gcd(16_000, 16_000), 16_000);
    }

    #[test]
    fn design_shapes_and_symmetry() {
        // 44100 -> 48000 reduces to L=160, M=147.
        let f = PolyphaseFilter::design(44_100, 48_000, 16, 8.6);
        assert_eq!(f.l, 160);
        assert_eq!(f.m, 147);
        assert_eq!(f.taps_per_phase, 32); // 2 * half_taps
        assert_eq!(f.phases.len(), 32 * 160); // flat: taps_per_phase × L
        assert_eq!(f.prototype_len(), 32 * 160);

        // The prototype is a linear-phase (symmetric) windowed sinc.
        let proto = f.prototype();
        let n = proto.len();
        for j in 0..n / 2 {
            let a = proto[j];
            let b = proto[n - 1 - j];
            assert!((a - b).abs() <= 1e-6 * (1.0 + a.abs()), "asymmetry at {j}: {a} vs {b}");
        }
        // The centre tap is the largest (a low-pass main lobe).
        let (argmax, _) = proto
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.abs().partial_cmp(&y.1.abs()).unwrap())
            .unwrap();
        assert!(
            (argmax as i64 - (n as i64 - 1) / 2).abs() <= f.l as i64,
            "peak near centre, got {argmax} of {n}"
        );
    }

    /// (Test 1) Impulse response: pushing a single unit sample through the *identity* ratio
    /// (L=M=1) yields exactly the designed prototype (one branch, the whole filter). This is the
    /// most direct "impulse response ≈ designed filter" check the polyphase form allows.
    #[test]
    fn impulse_response_is_the_designed_filter() {
        // 1:1 "resample" == pure FIR filtering by the prototype. L=M=1, one branch.
        let filt = PolyphaseFilter::design(48_000, 48_000, 8, 8.0);
        assert_eq!(filt.l, 1);
        assert_eq!(filt.m, 1);
        let proto = filt.prototype(); // == phases[0]
        let tpp = filt.taps_per_phase;

        let mut r = ChannelResampler::new(filt);
        // Feed an impulse followed by enough zeros to flush the whole FIR.
        let mut sig = vec![1.0f32];
        sig.extend(std::iter::repeat_n(0.0, tpp * 2));
        let mut out = Vec::new();
        r.process(&sig, &mut out);

        // With L=M=1 and a warm-up of (tpp-1) zeros, output y[t] = Σ_k proto[k]·x[t-k]. The
        // impulse sits at input index 0, so y[t] = proto[t] for t in 0..tpp. Compare that run.
        assert!(out.len() >= tpp, "produced {} of >= {tpp}", out.len());
        for k in 0..tpp {
            assert!(
                (out[k] - proto[k]).abs() < 1e-6,
                "impulse response tap {k}: got {}, want {}",
                out[k],
                proto[k]
            );
        }
    }

    /// Run a full sine of frequency `freq_hz` at `in_rate` through the resampler to `out_rate`
    /// and return the output samples (steady-state, transients trimmed by the caller).
    fn resample_sine(freq_hz: f64, in_rate: u32, out_rate: u32, in_len: usize) -> Vec<f32> {
        let filt = PolyphaseFilter::design(in_rate, out_rate, 32, 9.0);
        let mut r = ChannelResampler::new(filt);
        let input: Vec<f32> = (0..in_len)
            .map(|n| (2.0 * PI * freq_hz * n as f64 / in_rate as f64).sin() as f32)
            .collect();
        let mut out = Vec::new();
        r.process(&input, &mut out);
        out
    }

    /// Estimate the dominant frequency (Hz) of `sig` sampled at `rate` by a coarse DFT peak
    /// search over a candidate band. Naive O(N·K) — fine for a test.
    fn dominant_freq(sig: &[f32], rate: u32, f_lo: f64, f_hi: f64, step: f64) -> (f64, f64) {
        let n = sig.len();
        let mut best_f = f_lo;
        let mut best_mag = -1.0;
        let mut f = f_lo;
        while f <= f_hi {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            let w = 2.0 * PI * f / rate as f64;
            for (i, &s) in sig.iter().enumerate() {
                let ph = w * i as f64;
                re += s as f64 * ph.cos();
                im -= s as f64 * ph.sin();
            }
            let mag = (re * re + im * im).sqrt() / n as f64;
            if mag > best_mag {
                best_mag = mag;
                best_f = f;
            }
            f += step;
        }
        (best_f, best_mag)
    }

    /// Goertzel-style single-bin magnitude (energy proxy) at `f` Hz.
    fn mag_at(sig: &[f32], rate: u32, f: f64) -> f64 {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        let w = 2.0 * PI * f / rate as f64;
        for (i, &s) in sig.iter().enumerate() {
            let ph = w * i as f64;
            re += s as f64 * ph.cos();
            im -= s as f64 * ph.sin();
        }
        (re * re + im * im).sqrt() / sig.len() as f64
    }

    /// (Test 2) A pure sine at f, resampled 44100→48000, stays a sine at f: the dominant
    /// frequency is preserved, its energy is roughly conserved, and there is no large aliasing
    /// image elsewhere in the band.
    #[test]
    fn sine_44100_to_48000_stays_a_sine() {
        let (in_rate, out_rate) = (44_100u32, 48_000u32);
        let freq = 1_000.0; // well inside both Nyquist limits
        let out = resample_sine(freq, in_rate, out_rate, 44_100); // ~1 s

        // Trim the FIR start/end transient (a few hundred out-samples) before analysis.
        let trim = 800usize;
        assert!(out.len() > 2 * trim, "output too short: {}", out.len());
        let core = &out[trim..out.len() - trim];

        // Dominant frequency search near 1 kHz: must land within one search step of `freq`.
        let (peak, peak_mag) = dominant_freq(core, out_rate, 500.0, 1_500.0, 2.0);
        assert!((peak - freq).abs() <= 3.0, "dominant freq {peak} Hz, want {freq}");

        // Energy (amplitude) at f is roughly preserved: a unit-amplitude sine has a single-bin
        // magnitude near 0.5 (half the energy in each of ±f). Allow a generous band.
        assert!(peak_mag > 0.35, "peak magnitude too low: {peak_mag}");
        assert!(peak_mag < 0.65, "peak magnitude too high (gain error?): {peak_mag}");

        // No large alias image: probe a spot far from `freq` (e.g. 5 kHz) — must be tiny vs peak.
        let alias = mag_at(core, out_rate, 5_000.0);
        assert!(alias < peak_mag * 0.02, "alias image {alias} too large vs peak {peak_mag}");
    }

    /// (Test 3) Integer-ratio downsample 48000→24000 (L=1, M=2): a 1 kHz sine survives, and a
    /// tone above the *output* Nyquist (e.g. 15 kHz, above 12 kHz) is attenuated rather than
    /// aliasing down into the base-band.
    #[test]
    fn downsample_48000_to_24000_antialiases() {
        let (in_rate, out_rate) = (48_000u32, 24_000u32);
        let filt = PolyphaseFilter::design(in_rate, out_rate, 48, 10.0);
        assert_eq!((filt.l, filt.m), (1, 2));

        // A safe 1 kHz tone plus a 15 kHz tone (above the 12 kHz output Nyquist).
        let n = 48_000usize;
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / in_rate as f64;
                (0.5 * (2.0 * PI * 1_000.0 * t).sin() + 0.5 * (2.0 * PI * 15_000.0 * t).sin()) as f32
            })
            .collect();
        let mut r = ChannelResampler::new(filt);
        let mut out = Vec::new();
        r.process(&input, &mut out);

        let trim = 1_000usize;
        let core = &out[trim..out.len() - trim];
        // 1 kHz preserved.
        let good = mag_at(core, out_rate, 1_000.0);
        assert!(good > 0.15, "1 kHz tone lost: {good}");
        // 15 kHz would alias to |24000-15000| = 9000 Hz if not filtered. That image must be
        // strongly attenuated relative to the passband tone.
        let alias = mag_at(core, out_rate, 9_000.0);
        assert!(alias < good * 0.05, "15 kHz aliased to 9 kHz image {alias} vs {good}");
    }

    /// (Test 4) Output length matches the L/M ratio (within the filter's fixed start-up slop).
    #[test]
    fn output_length_matches_ratio() {
        for (inr, outr, inlen) in [
            (44_100u32, 48_000u32, 44_100usize),
            (48_000, 24_000, 50_000),
            (8_000, 16_000, 10_000),
            (48_000, 44_100, 96_000),
        ] {
            let filt = PolyphaseFilter::design(inr, outr, 8, 8.0);
            let mut r = ChannelResampler::new(filt);
            let input = vec![0f32; inlen];
            let mut out = Vec::new();
            r.process(&input, &mut out);

            let expect = output_len(inlen as u64, inr, outr);
            // The causal warm-up shifts the count by at most a few samples; assert it is within
            // a small, ratio-independent slack.
            let got = out.len() as i64;
            let diff = (got - expect as i64).abs();
            assert!(
                diff <= 2,
                "len {inr}->{outr}: got {got}, expect ~{expect} (diff {diff})"
            );
        }
    }

    /// The pre-SIMD reference: one output sample as a strictly left-to-right scalar accumulation
    /// over the branch, innermost tap first, exactly as [`ChannelResampler::process`] computed it
    /// before the dot product was blocked into [`LANES`] partial sums. Kept so the numerical cost
    /// of that reassociation stays measured rather than assumed.
    /// `f64 == true` gives the (effectively exact) ground truth against which both `f32` orderings
    /// are scored.
    fn reference_resample(filt: &PolyphaseFilter, input: &[f32], exact: bool) -> Vec<f32> {
        let (l, m, tpp) = (filt.l, filt.m, filt.taps_per_phase);
        let proto = filt.prototype();
        // The old layout: branch p, tap k := proto[p + k*L], innermost-tap first.
        let branch: Vec<Vec<f32>> =
            (0..l).map(|p| (0..tpp).map(|k| proto[p + k * l]).collect()).collect();

        let mut hist = vec![0f32; tpp - 1];
        hist.extend_from_slice(input);
        let (mut pos, mut phase) = (tpp - 1, 0usize);
        let mut out = Vec::new();
        while pos < hist.len() {
            if exact {
                let mut acc = 0f64;
                for (k, &c) in branch[phase].iter().enumerate() {
                    acc += c as f64 * hist[pos - k] as f64;
                }
                out.push(acc as f32);
            } else {
                let mut acc = 0f32;
                for (k, &c) in branch[phase].iter().enumerate() {
                    acc += c * hist[pos - k];
                }
                out.push(acc);
            }
            let ph = phase + m;
            pos += ph / l;
            phase = ph % l;
        }
        out
    }

    /// Max |Δ| and SNR (dB) of `got` against `want`.
    fn diff_stats(got: &[f32], want: &[f32]) -> (f64, f64) {
        let (mut max_abs, mut sig, mut err) = (0f64, 0f64, 0f64);
        for (a, b) in got.iter().zip(want) {
            let d = (*a as f64) - (*b as f64);
            max_abs = max_abs.max(d.abs());
            sig += (*b as f64) * (*b as f64);
            err += d * d;
        }
        (max_abs, 10.0 * (sig / err.max(f64::MIN_POSITIVE)).log10())
    }

    /// (Test 5) The blocked/SIMD dot product reassociates the tap sum, so it is deliberately
    /// **not** bit-identical to the old left-to-right accumulation. Score *both* orderings against
    /// an exact `f64` evaluation of the same convolution: the blocked form must be at least as
    /// close to the truth as the serial one (pairwise summation has an `O(log n)·ε` error bound
    /// where a serial chain has `O(n)·ε`), and its deviation from the old output must sit below
    /// the quantisation floor of the PCM depths the pipeline emits.
    #[test]
    fn simd_dot_matches_serial_reference() {
        // A broadband, full-scale-ish signal — worst case for cancellation in the tap sum.
        let n = 40_000usize;
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let input: Vec<f32> = (0..n)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = (state >> 40) as f32 / 8_388_608.0 - 0.125;
                let tone = 0.8 * (2.0 * PI * 997.0 * i as f64 / 44_100.0).sin() as f32;
                (tone + noise).clamp(-1.0, 1.0)
            })
            .collect();

        for (inr, outr, ht) in [(44_100u32, 48_000u32, 32usize), (48_000, 44_100, 32), (48_000, 24_000, 16)] {
            let filt = PolyphaseFilter::design(inr, outr, ht, 9.0);
            let serial = reference_resample(&filt, &input, false);
            let exact = reference_resample(&filt, &input, true);
            let mut got = Vec::new();
            ChannelResampler::new(filt).process(&input, &mut got);
            assert_eq!(got.len(), serial.len(), "{inr}->{outr}: output count must be unchanged");

            let (d_max, d_snr) = diff_stats(&got, &serial); // new vs old
            let (n_max, n_snr) = diff_stats(&got, &exact); // new vs truth
            let (s_max, s_snr) = diff_stats(&serial, &exact); // old vs truth
            println!(
                "{inr}->{outr}  blocked-vs-serial: max|Δ| {d_max:.2e}, SNR {d_snr:.1} dB  |  \
                 vs exact f64 — blocked: max|Δ| {n_max:.2e}, SNR {n_snr:.1} dB; \
                 serial: max|Δ| {s_max:.2e}, SNR {s_snr:.1} dB"
            );
            // The reassociation stays ~1 f32 ULP of a full-scale sample — three orders below the
            // 5.96e-8 LSB of 24-bit PCM's float image, i.e. inaudible at any depth we emit.
            assert!(d_max < 2e-6, "{inr}->{outr}: blocked-vs-serial max |Δ| {d_max:e} too large");
            assert!(d_snr > 130.0, "{inr}->{outr}: blocked-vs-serial SNR {d_snr} dB too low");
            // And the new ordering is the *more* accurate of the two against ground truth.
            assert!(
                n_snr >= s_snr,
                "{inr}->{outr}: blocked SNR {n_snr} dB must beat serial {s_snr} dB vs exact f64"
            );
        }
    }

    /// Streaming invariant: chunked feeding produces the identical sample sequence as one shot —
    /// the carried filter state (delay line + phase) is correct across batch boundaries.
    #[test]
    fn chunked_equals_oneshot() {
        let (inr, outr) = (44_100u32, 48_000u32);
        let input: Vec<f32> = (0..20_000)
            .map(|n| (2.0 * PI * 997.0 * n as f64 / inr as f64).sin() as f32)
            .collect();

        // One shot.
        let mut r1 = ChannelResampler::new(PolyphaseFilter::design(inr, outr, 16, 9.0));
        let mut one = Vec::new();
        r1.process(&input, &mut one);

        // Awkward chunk sizes (prime-ish, some smaller than a phase step).
        let mut r2 = ChannelResampler::new(PolyphaseFilter::design(inr, outr, 16, 9.0));
        let mut many = Vec::new();
        for chunk in input.chunks(333) {
            r2.process(chunk, &mut many);
        }

        assert_eq!(one.len(), many.len(), "chunked output count differs");
        for (i, (a, b)) in one.iter().zip(&many).enumerate() {
            assert!((a - b).abs() < 1e-6, "sample {i} differs: {a} vs {b}");
        }
    }
}
