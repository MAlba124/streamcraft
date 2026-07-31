use crate::math_utils;
use num::complex::Complex64;
use num::Zero;
use rustfft::{Fft, FftPlanner};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// Constants
const MIN_FFT_SIZE: usize = 32;

/// Process-wide FFT plan cache.
///
/// `rustfft` caches a size's plan inside its planner, so one planner per `FftManager` was ~38K
/// plan-building allocations/song. A *thread-local* planner fixed that, but a quality search runs
/// twelve worker threads over fifteen comparisons and each thread then built its own twiddle tables
/// — `new_with_avx`'s `sincos` was 2.2% of a comparison, so ~3% of a sweep spent computing the same
/// tables twelve times. `Arc<dyn Fft<f64>>` is `Send + Sync` (rustfft requires it of the trait), so
/// the plans are shared instead. The lock is taken once per transform and the critical section is a
/// hash lookup after the first build; a sweep takes it a few thousand times across twelve threads.
type Plan = Arc<dyn Fft<f64>>;

fn plan_cache() -> &'static Mutex<HashMap<(usize, bool), Plan>> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, bool), Plan>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The cached plan for `size`, built on first use. `inverse` picks the direction.
fn plan(size: usize, inverse: bool) -> Plan {
    let mut cache = plan_cache().lock().expect("fft plan cache poisoned");
    if let Some(cached) = cache.get(&(size, inverse)) {
        return Arc::clone(cached);
    }
    let built = PLANNER.with(|planner| {
        let mut planner = planner.borrow_mut();
        if inverse {
            planner.plan_fft_inverse(size)
        } else {
            planner.plan_fft_forward(size)
        }
    });
    cache.insert((size, inverse), Arc::clone(&built));
    built
}

thread_local! {
    /// Builder for [`plan`]; `rustfft`'s planner is not `Sync`, so it stays thread-local and only
    /// its *output* is shared.
    static PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::<f64>::new());
    /// Per-thread reusable rustfft out-of-place scratch buffer.
    static FFT_SCRATCH: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
    /// Per-thread reusable complex I/O buffer (the real→complex staging / inverse output). The FFT
    /// is called per patch during alignment (~42% of a quality search's allocations went to the
    /// per-call `vec![Complex64; …]`s these replace); the buffers only grow.
    static FFT_CBUF: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
}

/// Wrapper around the `rustfft` library to perform basic fft operations.
pub struct FftManager {
    /// Length of the fft
    pub fft_size: usize,
    /// Scale factor to apply after inverse fft
    inverse_fft_scale: f64,
    /// Number of samples to apply fft on
    pub samples_per_channel: usize,
}

impl FftManager {
    /// Creates a new fft manager, computes internal variables from `samples_per_channel`
    pub fn new(samples_per_channel: usize) -> Self {
        let fft_size = math_utils::next_pow_two(samples_per_channel).max(MIN_FFT_SIZE);

        Self {
            fft_size,
            samples_per_channel,
            inverse_fft_scale: 1.0f64 / (fft_size as f64),
        }
    }

    /// Zero-pads `time_channel` if necessary, transforms its contents into the frequency domain and stores it in `freq_channel`
    ///
    /// `time_channel` is borrowed **shared**: the zero-padding happens in the thread-local complex
    /// staging buffer rather than by resizing the caller's `Vec`, so callers can pass a short signal
    /// (or a borrowed slice) directly instead of first copying it into a padded owned `Vec`. Padding
    /// with `0.0` before staging and staging before padding with `Complex64::zero()` write the same
    /// `(0, 0)` elements, so the transform input is bit-identical.
    pub fn freq_from_time_domain(&mut self, time_channel: &[f64], freq_channel: &mut [Complex64]) {
        let real_to_complex = plan(self.fft_size, false);
        // Reused thread-local buffers rather than a fresh `Vec` per transform. `complex_time_domain`
        // stages `time_channel` as real-valued complex (`re = x, im = 0` — the exact
        // `float_vec_to_real_valued_complex_vec`); the scratch is sized by the plan.
        FFT_CBUF.with(|cb| {
            let mut complex_time_domain = cb.borrow_mut();
            complex_time_domain.clear();
            complex_time_domain.extend(time_channel.iter().map(|&x| Complex64::new(x, 0.0)));
            complex_time_domain.resize(self.fft_size, Complex64::zero());
            FFT_SCRATCH.with(|sb| {
                let mut scratch_buffer = sb.borrow_mut();
                scratch_buffer.clear();
                scratch_buffer.resize(real_to_complex.get_outofplace_scratch_len(), Complex64::zero());
                real_to_complex.process_outofplace_with_scratch(
                    &mut complex_time_domain[..],
                    freq_channel,
                    &mut scratch_buffer[..],
                );
            });
        });
    }

    /// Zero-pads `freq_channel` if necessary, transforms its contents into the time domain and stores it in `time_channel`
    ///
    /// `time_channel` is a pre-sized slice that receives the leading `time_channel.len()` real
    /// parts. (It used to be a `&mut Vec` that was cleared and extended with all `fft_size` real
    /// parts, of which every caller then read only the first `samples_per_channel` — same values,
    /// no growth.)
    pub fn time_from_freq_domain(
        &mut self,
        freq_channel: &mut [Complex64],
        time_channel: &mut [f64],
    ) {
        let complex_to_real = plan(self.fft_size, true);

        // `complex_td` is the inverse transform's *output* — `process_outofplace` overwrites it, so
        // its previous contents are irrelevant and it only needs to be `fft_size` long. Reuse
        // thread-local buffers, then write the real parts back into `time_channel` in place (the
        // original allocated a fresh scratch, a fresh complex buffer, and a fresh output `Vec`).
        FFT_CBUF.with(|cb| {
            let mut complex_td = cb.borrow_mut();
            complex_td.clear();
            complex_td.resize(self.fft_size, Complex64::zero());
            FFT_SCRATCH.with(|sb| {
                let mut scratch_buffer = sb.borrow_mut();
                scratch_buffer.clear();
                scratch_buffer.resize(complex_to_real.get_outofplace_scratch_len(), Complex64::zero());
                complex_to_real.process_outofplace_with_scratch(
                    freq_channel,
                    &mut complex_td[..],
                    &mut scratch_buffer[..],
                );
            });
            for (out, c) in time_channel.iter_mut().zip(complex_td.iter()) {
                *out = c.re;
            }
        });
    }

    /// Multiplies each element in `time_channel` by `self.inverse_fft_scale`
    pub fn apply_reverse_fft_scaling(&self, time_channel: &mut [f64]) {
        time_channel.iter_mut().for_each(|x| {
            *x *= self.inverse_fft_scale;
        });
    }
}
