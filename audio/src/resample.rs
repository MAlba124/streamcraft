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

/// A designed prototype low-pass, decomposed into `L` polyphase branches. `phases[p]` holds the
/// `taps_per_phase` coefficients of branch `p`, innermost-tap first (so it dots directly against
/// the most-recent-first history window). Immutable after [`design`](PolyphaseFilter::design).
#[derive(Clone, Debug)]
pub struct PolyphaseFilter {
    /// Upsampling factor `L` (interpolation ratio numerator).
    pub l: usize,
    /// Downsampling factor `M` (decimation ratio denominator).
    pub m: usize,
    /// Taps per polyphase branch (the per-output multiply count).
    pub taps_per_phase: usize,
    /// `L` branches × `taps_per_phase` coefficients, row-major (`phases[p][k]`).
    phases: Vec<Vec<f32>>,
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

        // Scatter into `L` branches. Branch p, tap k := proto[p + k*L]; the innermost tap
        // (k=0, nearest the interpolation instant) comes first so it dots a most-recent-first
        // history window with no index arithmetic in the hot loop.
        let mut phases = vec![vec![0f32; taps_per_phase]; l];
        for (p, branch) in phases.iter_mut().enumerate() {
            for (k, coeff) in branch.iter_mut().enumerate() {
                *coeff = proto[p + k * l];
            }
        }

        PolyphaseFilter { l, m, taps_per_phase, phases }
    }

    /// Total prototype length (all branches), `taps_per_phase · L`.
    pub fn prototype_len(&self) -> usize {
        self.taps_per_phase * self.l
    }

    /// The full designed prototype low-pass coefficients (branches re-interleaved), useful for
    /// verifying the impulse response. `proto[p + k·L] == phases[p][k]`.
    pub fn prototype(&self) -> Vec<f32> {
        let mut proto = vec![0f32; self.prototype_len()];
        for (p, branch) in self.phases.iter().enumerate() {
            for (k, &c) in branch.iter().enumerate() {
                proto[p + k * self.l] = c;
            }
        }
        proto
    }
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
    /// with `taps_per_phase − 1` zeros so the very first outputs are well-defined (a leading
    /// transient, standard for causal FIR resampling).
    pub fn new(filter: PolyphaseFilter) -> ChannelResampler {
        let warmup = filter.taps_per_phase.saturating_sub(1);
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
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) -> usize {
        let l = self.filter.l;
        let m = self.filter.m;
        let tpp = self.filter.taps_per_phase;
        self.history.extend_from_slice(input);
        self.consumed += input.len() as u64;

        let before = out.len();
        // Produce while the base index has `tpp` samples of history behind it (indices
        // `in_pos-(tpp-1)..=in_pos` all present). `in_pos` is an index into `history`.
        while self.in_pos < self.history.len() {
            let branch = &self.filter.phases[self.phase];
            // Dot the branch against history[in_pos], history[in_pos-1], … (most-recent-first).
            let mut acc = 0f32;
            // `in_pos >= tpp-1` always holds here (seeded warm-up), so the window is in range.
            let base = self.in_pos;
            for (k, &c) in branch.iter().enumerate() {
                acc += c * self.history[base - k];
            }
            out.push(acc);

            // Advance to the next output: m -> m+1 means (m·M) grows by M, so phase += M and
            // the base input index steps by the integer part of the phase overflow.
            let ph = self.phase + m;
            self.in_pos += ph / l;
            self.phase = ph % l;
        }

        // Trim consumed history: keep the `tpp-1` samples before `in_pos` (the delay line the
        // next output still needs), and rebase `in_pos` to the new front. `in_pos` points one
        // past the last usable input, so the retained tail starts at `in_pos-(tpp-1)`.
        let keep_from = self.in_pos.saturating_sub(tpp - 1);
        if keep_from > 0 {
            self.history.drain(..keep_from);
            self.in_pos -= keep_from;
        }

        let n = out.len() - before;
        self.produced += n as u64;
        n
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
        assert_eq!(f.phases.len(), 160);
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
