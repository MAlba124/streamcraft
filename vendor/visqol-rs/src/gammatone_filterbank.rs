use crate::constants;

/// Bank of gammatone filters on each frame of a time domain signal to construct a spectrogram representation.
/// This implementation is fixed to a 4th order filterbank.
pub struct GammatoneFilterbank<const NUM_BANDS: usize> {
    pub min_freq: f64,

    filter_conditions_1: [[f64; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
    filter_conditions_2: [[f64; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
    filter_conditions_3: [[f64; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
    filter_conditions_4: [[f64; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],

    filter_coeff_a0: Vec<f64>,
    filter_coeff_a11: Vec<f64>,
    filter_coeff_a12: Vec<f64>,
    filter_coeff_a13: Vec<f64>,
    filter_coeff_a14: Vec<f64>,
    filter_coeff_a2: Vec<f64>,
    filter_coeff_b0: Vec<f64>,
    filter_coeff_b1: Vec<f64>,
    filter_coeff_b2: Vec<f64>,
    filter_coeff_gain: Vec<f64>,
}

impl<const NUM_BANDS: usize> GammatoneFilterbank<NUM_BANDS> {
    /// Creates a new gammatone filterbank with the desired number of frequency bands and the minimum frequency.
    pub fn new(min_freq: f64) -> Self {
        Self {
            min_freq,
            filter_conditions_1: [[0.0; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
            filter_conditions_2: [[0.0; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
            filter_conditions_3: [[0.0; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
            filter_conditions_4: [[0.0; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
            filter_coeff_a0: Vec::new(),
            filter_coeff_a11: Vec::new(),
            filter_coeff_a12: Vec::new(),
            filter_coeff_a13: Vec::new(),
            filter_coeff_a14: Vec::new(),
            filter_coeff_a2: Vec::new(),
            filter_coeff_b0: Vec::new(),
            filter_coeff_b1: Vec::new(),
            filter_coeff_b2: Vec::new(),
            filter_coeff_gain: Vec::new(),
        }
    }

    /// Sets all internal states of the filterbank to 0.
    pub fn reset_filter_conditions(&mut self) {
        self.filter_conditions_1 = [[0.0, 0.0]; NUM_BANDS];
        self.filter_conditions_2 = [[0.0, 0.0]; NUM_BANDS];
        self.filter_conditions_3 = [[0.0, 0.0]; NUM_BANDS];
        self.filter_conditions_4 = [[0.0, 0.0]; NUM_BANDS];
    }

    /// Populates the filter coefficients with `filter_coeffs`.
    pub fn set_filter_coefficients(&mut self, filter_coeffs: &ndarray::Array2<f64>) {
        self.filter_coeff_a0 = filter_coeffs.column(0).to_vec();
        self.filter_coeff_a11 = filter_coeffs.column(1).to_vec();
        self.filter_coeff_a12 = filter_coeffs.column(2).to_vec();
        self.filter_coeff_a13 = filter_coeffs.column(3).to_vec();
        self.filter_coeff_a14 = filter_coeffs.column(4).to_vec();
        self.filter_coeff_a2 = filter_coeffs.column(5).to_vec();
        self.filter_coeff_b0 = filter_coeffs.column(6).to_vec();
        self.filter_coeff_b1 = filter_coeffs.column(7).to_vec();
        self.filter_coeff_b2 = filter_coeffs.column(8).to_vec();
        self.filter_coeff_gain = filter_coeffs.column(9).to_vec();
    }

    /// Applies the gammatone filterbank on the time-domain signal `signal`, producing a Gammetone spectrogram.
    ///
    /// Profluens patch (PROFLUENS-PATCHES.md): the 4th-order filter is a cascade of four 2nd-order
    /// sections. The original ran them as four separate full-signal passes per band — each
    /// allocating and zeroing a fresh `Vec<f64>` (128 allocations + memsets per call) and re-reading
    /// the whole signal four times. This **fuses the cascade into a single pass per band**: the four
    /// sections' states stay in registers and stage `k`'s output feeds stage `k+1` for the same
    /// sample, so the signal is read once and written once with no intermediate buffers. The float
    /// operations are the identical ops in the identical order, so the output is bit-for-bit the
    /// same (this profiled at ~53% of a ViSQOL run). `filter_signal` (below) is kept unused for API
    /// compatibility.
    ///
    /// `build` now uses [`filter_frame_energy_into`](Self::filter_frame_energy_into) directly, so
    /// `apply_filter` / `apply_filter_into` remain only as the reference the bit-exact energy test
    /// checks against — hence `allow(dead_code)`.
    #[inline(always)]
    #[allow(dead_code)]
    pub fn apply_filter(&mut self, input_signal: &[f64]) -> ndarray::Array2<f64> {
        let mut output = ndarray::Array2::<f64>::zeros((NUM_BANDS, input_signal.len()));
        self.apply_filter_into(input_signal, &mut output);
        output
    }

    /// Like [`apply_filter`](Self::apply_filter) but writes into a caller-provided
    /// `(NUM_BANDS, input_signal.len())` buffer, so the spectrogram builder reuses a single
    /// allocation across every frame instead of allocating a fresh `Array2` per frame (~45K/song,
    /// the bulk of ViSQOL's allocation byte-churn). Each output row is fully overwritten.
    #[allow(dead_code)]
    pub fn apply_filter_into(&mut self, input_signal: &[f64], output: &mut ndarray::Array2<f64>) {
        for band in 0..NUM_BANDS {
            let g = self.filter_coeff_gain[band];
            // Section numerators (`a`) — stage 1 scaled by 1/gain; the shared `a0`/`a2` and the
            // per-stage middle tap `a1x` are the only differences. Denominator taps `b1`/`b2` are
            // shared (matching `filter_signal`, which ignores `denom[0]`).
            let (a0, a2c) = (self.filter_coeff_a0[band], self.filter_coeff_a2[band]);
            let n1 = [a0 / g, self.filter_coeff_a11[band] / g, a2c / g];
            let n2 = [a0, self.filter_coeff_a12[band], a2c];
            let n3 = [a0, self.filter_coeff_a13[band], a2c];
            let n4 = [a0, self.filter_coeff_a14[band], a2c];
            let (b1, b2) = (self.filter_coeff_b1[band], self.filter_coeff_b2[band]);

            // Carried Direct-Form-II-transposed state for each of the 4 sections.
            let mut z1 = self.filter_conditions_1[band];
            let mut z2 = self.filter_conditions_2[band];
            let mut z3 = self.filter_conditions_3[band];
            let mut z4 = self.filter_conditions_4[band];

            let mut row = output.row_mut(band);
            let row = row.as_slice_mut().expect("contiguous output row");
            for (dst, &x) in row.iter_mut().zip(input_signal) {
                let y1 = n1[0] * x + z1[0];
                z1[0] = n1[1] * x + z1[1] - b1 * y1;
                z1[1] = n1[2] * x - b2 * y1;

                let y2 = n2[0] * y1 + z2[0];
                z2[0] = n2[1] * y1 + z2[1] - b1 * y2;
                z2[1] = n2[2] * y1 - b2 * y2;

                let y3 = n3[0] * y2 + z3[0];
                z3[0] = n3[1] * y2 + z3[1] - b1 * y3;
                z3[1] = n3[2] * y2 - b2 * y3;

                let y4 = n4[0] * y3 + z4[0];
                z4[0] = n4[1] * y3 + z4[1] - b1 * y4;
                z4[1] = n4[2] * y3 - b2 * y4;

                *dst = y4;
            }

            self.filter_conditions_1[band] = z1;
            self.filter_conditions_2[band] = z2;
            self.filter_conditions_3[band] = z3;
            self.filter_conditions_4[band] = z4;
        }
    }

    /// Per-band filter energy `Σ y4²` for one frame (state starts at zero), computed directly — the
    /// only thing the spectrogram builder needs from the filtered frame (RMS = `sqrt(energy / n)`),
    /// so the full filtered `Array2` is never materialised. The band loop is the *inner* loop over
    /// structure-of-arrays state, which vectorises: under AVX2 it runs 4 bands per instruction. The
    /// float ops are the identical per-band cascade in the identical order (no FMA contraction), so
    /// the result is bit-for-bit equal to `apply_filter`-then-sum-of-squares.
    pub fn filter_frame_energy_into(&self, input: &[f64], energy: &mut [f64; NUM_BANDS]) {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: gated on runtime AVX2 detection *and* on the coefficient vectors being sized
            // — unlike the crate's other `_avx2` wrappers (which only let the autovectoriser widen
            // safe Rust), this one is hand-written intrinsics whose four-wide loads have no bounds
            // check. `energy` is `[f64; NUM_BANDS]` and every store is at `band + 4 <= NUM_BANDS`.
            if std::is_x86_feature_detected!("avx2") && self.coefficients_are_sized() {
                unsafe { self.filter_frame_energy_avx2(input, energy) };
                return;
            }
        }
        self.filter_frame_energy_kernel(input, energy);
    }

    /// Every coefficient vector holds at least `NUM_BANDS` entries — the invariant
    /// [`set_filter_coefficients`](Self::set_filter_coefficients) establishes, and the precondition
    /// of the unchecked loads in [`filter_frame_energy_avx2`](Self::filter_frame_energy_avx2). False
    /// before it has run, in which case the checked kernel takes over and panics exactly as it
    /// always did.
    #[cfg(target_arch = "x86_64")]
    fn coefficients_are_sized(&self) -> bool {
        [
            &self.filter_coeff_a0,
            &self.filter_coeff_a11,
            &self.filter_coeff_a12,
            &self.filter_coeff_a13,
            &self.filter_coeff_a14,
            &self.filter_coeff_a2,
            &self.filter_coeff_b1,
            &self.filter_coeff_b2,
            &self.filter_coeff_gain,
        ]
        .iter()
        .all(|coefficients| coefficients.len() >= NUM_BANDS)
    }

    /// The per-frame energy cascade in explicit AVX2 intrinsics, blocked four bands at a time.
    ///
    /// The autovectorised [`filter_frame_energy_kernel`](Self::filter_frame_energy_kernel) runs the
    /// band loop innermost over `[f64; NUM_BANDS]` state, which vectorises cleanly but makes nine
    /// full-width arrays live across it — 72 AVX2 registers' worth against the sixteen the ISA has.
    /// Every section's state was therefore stored to and reloaded from the stack on *every sample*:
    /// 183 stack-touching `vmov`s against 38 vector arithmetic instructions. Blocking the bands so
    /// one block's state fits in registers is not expressible in safe Rust here — written that way
    /// LLVM refuses to keep `[f64; 4]` locals in registers across the sample loop and rebuilds them
    /// with `vunpcklpd`/`vinsertf128` instead, which measured 45% *worse*. With intrinsics the state
    /// is register-resident by construction: the same 38 arithmetic instructions against 6 stack
    /// accesses.
    ///
    /// Deliberately no FMA (`_mm256_fmadd_pd`), though this CPU has it: fusing would round once
    /// where the scalar reference rounds twice, and the energies must stay bit-for-bit equal to
    /// `apply_filter` (asserted by `frame_energy_matches_apply_filter` for both band counts).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn filter_frame_energy_avx2(&self, input: &[f64], energy: &mut [f64; NUM_BANDS]) {
        use std::arch::x86_64::*;

        let mut band = 0;
        while band + 4 <= NUM_BANDS {
            let ld = |src: &[f64]| unsafe { _mm256_loadu_pd(src.as_ptr().add(band)) };
            let g = ld(&self.filter_coeff_gain);
            let a0 = ld(&self.filter_coeff_a0);
            let a2 = ld(&self.filter_coeff_a2);
            let n10 = _mm256_div_pd(a0, g);
            let n11 = _mm256_div_pd(ld(&self.filter_coeff_a11), g);
            let n12 = _mm256_div_pd(a2, g);
            let a12 = ld(&self.filter_coeff_a12);
            let a13 = ld(&self.filter_coeff_a13);
            let a14 = ld(&self.filter_coeff_a14);
            let b1 = ld(&self.filter_coeff_b1);
            let b2 = ld(&self.filter_coeff_b2);

            let z = _mm256_setzero_pd();
            let (mut z10, mut z11) = (z, z);
            let (mut z20, mut z21) = (z, z);
            let (mut z30, mut z31) = (z, z);
            let (mut z40, mut z41) = (z, z);
            let mut acc = z;

            for &x in input {
                let xv = _mm256_set1_pd(x);

                let y1 = _mm256_add_pd(_mm256_mul_pd(n10, xv), z10);
                z10 = _mm256_sub_pd(
                    _mm256_add_pd(_mm256_mul_pd(n11, xv), z11),
                    _mm256_mul_pd(b1, y1),
                );
                z11 = _mm256_sub_pd(_mm256_mul_pd(n12, xv), _mm256_mul_pd(b2, y1));

                let y2 = _mm256_add_pd(_mm256_mul_pd(a0, y1), z20);
                z20 = _mm256_sub_pd(
                    _mm256_add_pd(_mm256_mul_pd(a12, y1), z21),
                    _mm256_mul_pd(b1, y2),
                );
                z21 = _mm256_sub_pd(_mm256_mul_pd(a2, y1), _mm256_mul_pd(b2, y2));

                let y3 = _mm256_add_pd(_mm256_mul_pd(a0, y2), z30);
                z30 = _mm256_sub_pd(
                    _mm256_add_pd(_mm256_mul_pd(a13, y2), z31),
                    _mm256_mul_pd(b1, y3),
                );
                z31 = _mm256_sub_pd(_mm256_mul_pd(a2, y2), _mm256_mul_pd(b2, y3));

                let y4 = _mm256_add_pd(_mm256_mul_pd(a0, y3), z40);
                z40 = _mm256_sub_pd(
                    _mm256_add_pd(_mm256_mul_pd(a14, y3), z41),
                    _mm256_mul_pd(b1, y4),
                );
                z41 = _mm256_sub_pd(_mm256_mul_pd(a2, y3), _mm256_mul_pd(b2, y4));

                acc = _mm256_add_pd(acc, _mm256_mul_pd(y4, y4));
            }
            _mm256_storeu_pd(energy.as_mut_ptr().add(band), acc);
            band += 4;
        }
        if band < NUM_BANDS {
            self.filter_frame_energy_tail(input, energy, band);
        }
    }

    /// Scalar remainder for a band count that is not a multiple of four (speech mode has 21).
    #[cfg(target_arch = "x86_64")]
    fn filter_frame_energy_tail(&self, input: &[f64], energy: &mut [f64; NUM_BANDS], from: usize) {
        for band in from..NUM_BANDS {
            let g = self.filter_coeff_gain[band];
            let (n10, n11, n12) = (
                self.filter_coeff_a0[band] / g,
                self.filter_coeff_a11[band] / g,
                self.filter_coeff_a2[band] / g,
            );
            let (a0, a2) = (self.filter_coeff_a0[band], self.filter_coeff_a2[band]);
            let (a12, a13, a14) = (
                self.filter_coeff_a12[band],
                self.filter_coeff_a13[band],
                self.filter_coeff_a14[band],
            );
            let (b1, b2) = (self.filter_coeff_b1[band], self.filter_coeff_b2[band]);
            let (mut z10, mut z11, mut z20, mut z21) = (0.0, 0.0, 0.0, 0.0);
            let (mut z30, mut z31, mut z40, mut z41) = (0.0, 0.0, 0.0, 0.0);
            let mut acc = 0.0;
            for &x in input {
                let y1 = n10 * x + z10;
                z10 = n11 * x + z11 - b1 * y1;
                z11 = n12 * x - b2 * y1;
                let y2 = a0 * y1 + z20;
                z20 = a12 * y1 + z21 - b1 * y2;
                z21 = a2 * y1 - b2 * y2;
                let y3 = a0 * y2 + z30;
                z30 = a13 * y2 + z31 - b1 * y3;
                z31 = a2 * y2 - b2 * y3;
                let y4 = a0 * y3 + z40;
                z40 = a14 * y3 + z41 - b1 * y4;
                z41 = a2 * y3 - b2 * y4;
                acc += y4 * y4;
            }
            energy[band] = acc;
        }
    }

    #[inline(always)]
    fn filter_frame_energy_kernel(&self, input: &[f64], energy: &mut [f64; NUM_BANDS]) {
        // Gather coefficients into fixed-size arrays so the band loop has no `Vec` bounds checks or
        // aliasing and LLVM can vectorise it. Stage-1 numerators are pre-divided by the gain (`n1x`);
        // stages 2–4 share `a0`/`a2` and differ only in the middle tap (`a12`/`a13`/`a14`).
        let mut n10 = [0.0f64; NUM_BANDS];
        let mut n11 = [0.0f64; NUM_BANDS];
        let mut n12 = [0.0f64; NUM_BANDS];
        let mut a0 = [0.0f64; NUM_BANDS];
        let mut a12 = [0.0f64; NUM_BANDS];
        let mut a13 = [0.0f64; NUM_BANDS];
        let mut a14 = [0.0f64; NUM_BANDS];
        let mut a2 = [0.0f64; NUM_BANDS];
        let mut b1 = [0.0f64; NUM_BANDS];
        let mut b2 = [0.0f64; NUM_BANDS];
        for b in 0..NUM_BANDS {
            let g = self.filter_coeff_gain[b];
            n10[b] = self.filter_coeff_a0[b] / g;
            n11[b] = self.filter_coeff_a11[b] / g;
            n12[b] = self.filter_coeff_a2[b] / g;
            a0[b] = self.filter_coeff_a0[b];
            a12[b] = self.filter_coeff_a12[b];
            a13[b] = self.filter_coeff_a13[b];
            a14[b] = self.filter_coeff_a14[b];
            a2[b] = self.filter_coeff_a2[b];
            b1[b] = self.filter_coeff_b1[b];
            b2[b] = self.filter_coeff_b2[b];
        }

        // Direct-Form-II-transposed state for the four cascaded sections, one lane per band, starting
        // from zero — each frame is filtered independently (matches `build`'s per-frame reset).
        let mut z10 = [0.0f64; NUM_BANDS];
        let mut z11 = [0.0f64; NUM_BANDS];
        let mut z20 = [0.0f64; NUM_BANDS];
        let mut z21 = [0.0f64; NUM_BANDS];
        let mut z30 = [0.0f64; NUM_BANDS];
        let mut z31 = [0.0f64; NUM_BANDS];
        let mut z40 = [0.0f64; NUM_BANDS];
        let mut z41 = [0.0f64; NUM_BANDS];
        let mut acc = [0.0f64; NUM_BANDS];

        for &x in input {
            for b in 0..NUM_BANDS {
                let y1 = n10[b] * x + z10[b];
                z10[b] = n11[b] * x + z11[b] - b1[b] * y1;
                z11[b] = n12[b] * x - b2[b] * y1;

                let y2 = a0[b] * y1 + z20[b];
                z20[b] = a12[b] * y1 + z21[b] - b1[b] * y2;
                z21[b] = a2[b] * y1 - b2[b] * y2;

                let y3 = a0[b] * y2 + z30[b];
                z30[b] = a13[b] * y2 + z31[b] - b1[b] * y3;
                z31[b] = a2[b] * y2 - b2[b] * y3;

                let y4 = a0[b] * y3 + z40[b];
                z40[b] = a14[b] * y3 + z41[b] - b1[b] * y4;
                z41[b] = a2[b] * y3 - b2[b] * y4;

                acc[b] += y4 * y4;
            }
        }
        *energy = acc;
    }
}

#[cfg(test)]
mod tests {
    use crate::equivalent_rectangular_bandwidth;
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;

    use super::*;
    #[test]
    fn gammatone_filterbank() {
        let fs = 48000;
        const NUM_BANDS: usize = 32;
        let min_freq = 50.0f64;

        let ten_samples = vec![0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9];

        let (mut filter_coeffs, _) = equivalent_rectangular_bandwidth::make_filters::<NUM_BANDS>(
            fs,
            min_freq,
            fs as f64 / 2.0,
        );

        filter_coeffs.invert_axis(Axis(0));

        let epsilon = 0.0001;

        // Check if filtering works as intended.
        let mut filterbank = GammatoneFilterbank::<{ NUM_BANDS }>::new(min_freq);
        filterbank.reset_filter_conditions();
        filterbank.set_filter_coefficients(&filter_coeffs);

        let filtered_signal = filterbank.apply_filter(&ten_samples);

        // Check dimensions
        assert_eq!(filtered_signal.ncols(), 10);
        assert_eq!(filtered_signal.nrows(), 32);

        // Check individual elements
        let expected_output = [1.028e-10, 6.15143e-10, 2.14718e-09];

        for (&res, ex) in expected_output.iter().zip(filtered_signal) {
            assert_abs_diff_eq!(res, ex, epsilon = epsilon);
        }
    }

    /// The fused (vectorised, AVX2-across-bands) energy kernel must equal `apply_filter` followed by
    /// a per-band sum of squares — bit-for-bit, since the spectrogram RMS is derived from it.
    ///
    /// Run for both band counts ViSQOL configures: 32 (audio) exercises whole 4-band vectors only,
    /// 21 (speech) also exercises the scalar tail the vector loop leaves behind.
    #[test]
    fn frame_energy_matches_apply_filter() {
        check_frame_energy::<32>();
        check_frame_energy::<21>();
    }

    fn check_frame_energy<const NUM_BANDS: usize>() {
        let fs = 48000;
        let min_freq = 50.0f64;
        let (mut filter_coeffs, _) = equivalent_rectangular_bandwidth::make_filters::<NUM_BANDS>(
            fs,
            min_freq,
            fs as f64 / 2.0,
        );
        filter_coeffs.invert_axis(Axis(0));

        // A non-trivial frame so the 4-stage state genuinely evolves.
        let signal: Vec<f64> = (0..96)
            .map(|i| (i as f64 * 0.37).sin() * 0.5 + (i as f64 * 0.11).cos() * 0.3)
            .collect();

        let mut fb = GammatoneFilterbank::<{ NUM_BANDS }>::new(min_freq);
        fb.reset_filter_conditions();
        fb.set_filter_coefficients(&filter_coeffs);

        // Reference: exactly what `build` did before the fused kernel.
        let filtered = fb.apply_filter(&signal);
        let mut reference = [0.0f64; NUM_BANDS];
        for (b, r) in reference.iter_mut().enumerate() {
            *r = filtered.row(b).iter().map(|&e| e * e).sum();
        }

        let mut energy = [0.0f64; NUM_BANDS];
        fb.filter_frame_energy_into(&signal, &mut energy);

        for b in 0..NUM_BANDS {
            assert_eq!(
                energy[b], reference[b],
                "{NUM_BANDS} bands: band {b} energy differs from apply_filter"
            );
        }
    }
}
