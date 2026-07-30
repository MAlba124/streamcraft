//! SILK resampler delay constants — RFC 6716 §4.2.9.
//!
//! The §4.2.9 resampler itself is *non-normative*: the spec explicitly
//! states "the resampler itself is non-normative, and a decoder can use
//! any method it wants to perform the resampling." What IS normative is
//! the **maximum decoder-side delay allocation** (Table 54), which
//! exists so that an encoder can apply a matching pre-delay to the MDCT
//! layer and keep SILK and CELT aligned across a mode switch (§4.5).
//!
//! This module owns:
//!
//! * The §4.2.9 Table 54 delay allocation per SILK audio bandwidth
//!   ([`silk_resampler_delay_ms`], [`silk_resampler_delay_samples_at`]).
//! * The five supported decoder output sample rates per §4.2.9
//!   ([`SUPPORTED_OUTPUT_RATES_HZ`], [`is_supported_output_rate`]).
//! * The internal SILK sample rate per audio bandwidth implied by the
//!   §4.2.1 / §4.2.7.x decode pipeline ([`silk_internal_rate_hz`],
//!   [`silk_frame_samples_internal`]).
//!
//! * The actual sample-rate conversion: [`SilkUpsampler`], the SILK
//!   internal rate → 48 kHz resampler in the exact fixed-point
//!   arithmetic of the RFC 6716 §A reference listing (1 ms delay
//!   compensation + 2× allpass upsampling + fractional-phase 8-tap
//!   FIR interpolation). §4.2.9 makes the filter non-normative, but
//!   the reference decoder's filter decides where SILK audio lands on
//!   the 48 kHz timeline relative to the CELT layer of a Hybrid frame
//!   and to the RFC 7845 pre-skip (the encoder pre-delays the MDCT
//!   layer to match it, §4.5); reproducing it exactly is what makes
//!   the decoded SILK waveform sample-align with reference decodes.
//!   Table 54 remains the spec's stated encoder-side delay target and
//!   is kept as the documented constants above.
//!
//! All numeric values are transcribed from RFC 6716 §4.2 (Table 54
//! plus the §4.2.9 prose, plus the SILK-internal sample rates implied
//! by the §4.2.7.x decode pipeline) and the §A reference listing's
//! resampler tables. No external library source was consulted,
//! paraphrased, or used as a cross-check oracle.

use crate::Bandwidth;

/// The five decoder output sample rates the §4.2.9 prose names as
/// supported (8, 12, 16, 24, 48 kHz). These are the rates the reference
/// implementation "is able to resample to … within or near this delay
/// constraint" per §4.2.9.
///
/// An application is free to ask for any output rate — the resampler
/// is non-normative — but staying inside this list is what the spec
/// promises will fit inside the Table 54 delay allocation.
pub const SUPPORTED_OUTPUT_RATES_HZ: &[u32] = &[8_000, 12_000, 16_000, 24_000, 48_000];

/// The 48 kHz "common" rate the §4.2.9 delay table is referenced
/// against (Table 54: "the maximum resampler delay in samples at
/// 48 kHz"). Also the rate at which the CELT MDCT operates, so the
/// natural rate to keep SILK and CELT aligned.
pub const REFERENCE_RATE_HZ: u32 = 48_000;

/// Table 54 NB delay, in milliseconds: 0.538 ms.
///
/// "NB is given a smaller decoder delay allocation than MB and WB to
/// allow a higher-order filter when resampling to 8 kHz in both the
/// encoder and decoder."
pub const SILK_RESAMPLER_DELAY_MS_NB: f64 = 0.538;

/// Table 54 MB delay, in milliseconds: 0.692 ms.
pub const SILK_RESAMPLER_DELAY_MS_MB: f64 = 0.692;

/// Table 54 WB delay, in milliseconds: 0.706 ms.
pub const SILK_RESAMPLER_DELAY_MS_WB: f64 = 0.706;

/// Return the §4.2.9 Table 54 normative resampler delay (in
/// milliseconds) for the given SILK audio bandwidth, or `None` if the
/// bandwidth never reaches the §4.2.9 resampler (SWB and FB are CELT-
/// or Hybrid-only at the SILK layer; they don't appear in Table 54).
///
/// The returned value is the spec's *maximum* allocation: a decoder
/// is free to use a resampler with less delay (or to use no
/// resampler at all when the output rate matches the internal SILK
/// rate). A decoder that wants *more* delay must compensate by
/// delaying the MDCT layer by the same extra amount.
pub fn silk_resampler_delay_ms(bw: Bandwidth) -> Option<f64> {
    match bw {
        Bandwidth::Nb => Some(SILK_RESAMPLER_DELAY_MS_NB),
        Bandwidth::Mb => Some(SILK_RESAMPLER_DELAY_MS_MB),
        Bandwidth::Wb => Some(SILK_RESAMPLER_DELAY_MS_WB),
        Bandwidth::Swb | Bandwidth::Fb => None,
    }
}

/// Return the §4.2.9 Table 54 normative resampler delay expressed as a
/// sample count at `output_rate_hz`. Rounded to the nearest whole
/// sample because §4.2.9 cautions that "the actual output rate may not
/// be 48 kHz, it may not be possible to achieve exactly these delays
/// while using a whole number of input or output samples."
///
/// Returns `None` for a SWB or FB bandwidth (which never reaches the
/// §4.2.9 SILK resampler) or for a zero `output_rate_hz`.
pub fn silk_resampler_delay_samples_at(bw: Bandwidth, output_rate_hz: u32) -> Option<u32> {
    if output_rate_hz == 0 {
        return None;
    }
    let delay_ms = silk_resampler_delay_ms(bw)?;
    // ms × rate_hz / 1000 → samples. Round half away from zero.
    let samples = (delay_ms * (output_rate_hz as f64) / 1000.0).round();
    // Clamp to u32 — the value can't realistically overflow at any
    // sensible rate (0.706 ms × 48 kHz ≈ 34 samples), but be defensive.
    if !samples.is_finite() || samples < 0.0 || samples > u32::MAX as f64 {
        return None;
    }
    Some(samples as u32)
}

/// Whether `rate_hz` is one of the §4.2.9 "supported output sampling
/// rates" enumerated in the spec (8, 12, 16, 24, 48 kHz). Any other
/// rate is still acceptable per §4.2.9's non-normative resampler
/// clause but is outside the Table 54 delay guarantee.
pub fn is_supported_output_rate(rate_hz: u32) -> bool {
    SUPPORTED_OUTPUT_RATES_HZ.contains(&rate_hz)
}

/// Return the internal SILK sample rate, in Hz, for the given audio
/// bandwidth.
///
/// The SILK §4.2.7.x decode pipeline operates at:
///
/// * NB → 8 000 Hz
/// * MB → 12 000 Hz
/// * WB → 16 000 Hz
///
/// SWB and FB never reach the SILK layer (they're CELT-only or run as
/// the upper half of a Hybrid frame whose lower half is WB SILK), so
/// they return `None`. The §4.2.9 resampler turns this internal rate
/// into whatever output rate the application asked for (commonly
/// 48 kHz to align with CELT).
pub fn silk_internal_rate_hz(bw: Bandwidth) -> Option<u32> {
    match bw {
        Bandwidth::Nb => Some(8_000),
        Bandwidth::Mb => Some(12_000),
        Bandwidth::Wb => Some(16_000),
        Bandwidth::Swb | Bandwidth::Fb => None,
    }
}

/// Return the number of internal-rate samples in one SILK frame of the
/// given bandwidth × duration, or `None` if the bandwidth doesn't
/// reach SILK or the duration isn't a SILK frame length.
///
/// `silk_frame_duration_tenths_ms` is in the same tenths-of-a-ms unit
/// as `OpusTocByte::frame_size_tenths_ms`: 100 = 10 ms, 200 = 20 ms.
/// SILK frames are always 10 ms or 20 ms per §4.2.2.
///
/// The result is the count of post-SILK pre-resampler samples that
/// flow into the §4.2.9 stage for one SILK frame. For example:
///
/// * NB 20 ms → 8 000 Hz × 0.020 s = 160 samples.
/// * MB 20 ms → 12 000 Hz × 0.020 s = 240 samples.
/// * WB 10 ms → 16 000 Hz × 0.010 s = 160 samples.
pub fn silk_frame_samples_internal(
    bw: Bandwidth,
    silk_frame_duration_tenths_ms: u16,
) -> Option<u32> {
    let rate = silk_internal_rate_hz(bw)?;
    match silk_frame_duration_tenths_ms {
        100 => Some(rate / 100), // 10 ms = rate / 100
        200 => Some(rate / 50),  // 20 ms = rate / 50
        _ => None,
    }
}

/// Return the number of output-rate samples in one SILK frame of the
/// given bandwidth × duration after resampling to `output_rate_hz`.
///
/// Computed as `output_rate_hz × duration_ms / 1000`, rounded to the
/// nearest whole sample. Returns `None` if the bandwidth doesn't reach
/// SILK, the duration isn't a SILK frame length, or the output rate is
/// zero. Out-of-range arithmetic returns `None`.
///
/// This is a convenience for callers sizing the post-resampler output
/// buffer; the actual sample-rate conversion filter itself stays
/// non-normative.
pub fn silk_frame_samples_at_output(
    bw: Bandwidth,
    silk_frame_duration_tenths_ms: u16,
    output_rate_hz: u32,
) -> Option<u32> {
    if output_rate_hz == 0 {
        return None;
    }
    // Validate bandwidth reaches SILK by computing the internal count
    // (which already vets the bandwidth and the duration); the result
    // itself isn't used — we just need the early `None`.
    silk_frame_samples_internal(bw, silk_frame_duration_tenths_ms)?;
    let ms = match silk_frame_duration_tenths_ms {
        100 => 10.0_f64,
        200 => 20.0_f64,
        _ => return None,
    };
    let s = (ms * (output_rate_hz as f64) / 1000.0).round();
    if !s.is_finite() || s < 0.0 || s > u32::MAX as f64 {
        return None;
    }
    Some(s as u32)
}

// ---------------------------------------------------------------------
// §4.2.9 resampler (SILK internal rate → 48 kHz), fixed point.
// ---------------------------------------------------------------------

/// Which SILK reconstruction path feeds the §4.2.9 resampler. The
/// reference resampler is path-independent (mono and stereo share one
/// filter; the §4.2.8 one-sample delay is applied by the caller's
/// two-sample output buffering), so this only tags the state for
/// diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilkChannelPath {
    /// A mono SILK channel (SILK-only mono, Hybrid mono, FEC mono).
    Mono,
    /// One channel of an unmixed stereo pair (left or right).
    Stereo,
}

/// The §A reference listing's decoder-side delay compensation, in input
/// samples, for resampling {8, 12, 16} kHz → 48 kHz (the `delay_matrix`
/// row/column for each SILK internal rate at a 48 kHz output).
fn decoder_input_delay(fs_in_khz: usize) -> usize {
    match fs_in_khz {
        8 => 0,
        12 => 4,
        _ => 7, // 16 kHz
    }
}

/// Interpolation FIR half-tables for the fractional-phase stage of the
/// §4.2.9 upsampler (12 phases × 4 taps; the second half of each 8-tap
/// filter mirrors the table at `11 − index`). Transcribed from the
/// RFC 6716 §A reference listing's resampler coefficient tables.
const FRAC_FIR_12: [[i16; 4]; 12] = [
    [189, -600, 617, 30567],
    [117, -159, -1070, 29704],
    [52, 221, -2392, 28276],
    [-4, 529, -3350, 26341],
    [-48, 758, -3956, 23973],
    [-80, 905, -4235, 21254],
    [-99, 972, -4222, 18278],
    [-107, 967, -3957, 15143],
    [-103, 896, -3487, 11950],
    [-91, 773, -2865, 8798],
    [-71, 611, -2143, 5784],
    [-46, 425, -1375, 2996],
];

/// First (even-phase) 2× allpass coefficient triple, Q16 fractions.
const UP2_HQ_0: [i32; 3] = [1746, 14986, 39083 - 65536];
/// Second (odd-phase) 2× allpass coefficient triple, Q16 fractions.
const UP2_HQ_1: [i32; 3] = [6854, 25769, 55542 - 65536];

/// Number of carried interpolation-history samples (the 8-tap FIR).
const ORDER_FIR_12: usize = 8;
/// Batch size in milliseconds (the reference processes 10 ms at a time,
/// restarting the fractional-index accumulator per batch).
const MAX_BATCH_MS: usize = 10;

/// A stateful streaming resampler from one SILK internal rate
/// (8/12/16 kHz) to the 48 kHz decoder output rate — the §4.2.9
/// resampler, in the exact fixed-point arithmetic of the RFC 6716 §A
/// reference listing.
///
/// §4.2.9 makes the filter non-normative, but the *reference decoder's*
/// filter decides where SILK audio lands on the 48 kHz timeline
/// relative to the CELT layer of a Hybrid frame and to the RFC 7845
/// pre-skip (the encoder pre-delays the MDCT layer to match it, §4.5).
/// Reproducing it exactly is what makes the decoded SILK waveform
/// sample-align with reference decodes.
///
/// Structure (the listing's `UF` path for all three SILK rates):
///
/// 1. a 1 ms input delay-compensation buffer (per-rate delay so every
///    mode has equal total delay),
/// 2. 2× upsampling by a pair of 3-section allpass chains (even / odd
///    output phases) with a Q10 internal state, and
/// 3. fractional-phase 8-tap FIR interpolation from the 2× signal to
///    the output rate, in 10 ms batches with an 8-sample carried
///    history.
#[derive(Debug, Clone)]
pub struct SilkUpsampler {
    bandwidth: Bandwidth,
    path: SilkChannelPath,
    fs_in_khz: usize,
    fs_out_khz: usize,
    input_delay: usize,
    batch_size: usize,
    inv_ratio_q16: i32,
    /// 2× allpass chain state (Q10), 3 sections × 2 phases.
    s_iir: [i32; 6],
    /// Carried tail of the 2×-upsampled signal for the FIR stage.
    s_fir: [i16; ORDER_FIR_12],
    /// 1 ms delay-compensation buffer (`fs_in_khz` samples used).
    delay_buf: [i16; 48],
}

impl SilkUpsampler {
    /// Construct a resampler for one SILK audio bandwidth on one
    /// reconstruction path, or `None` for SWB / FB (which never reach
    /// the §4.2.9 SILK resampler).
    pub fn new(bandwidth: Bandwidth, path: SilkChannelPath) -> Option<Self> {
        let fs_in_khz = match bandwidth {
            Bandwidth::Nb => 8,
            Bandwidth::Mb => 12,
            Bandwidth::Wb => 16,
            Bandwidth::Swb | Bandwidth::Fb => return None,
        };
        let fs_out_khz = 48;
        // invRatio_Q16 = ((fs_in << (14 + 1)) / fs_out) << 2, rounded up
        // until invRatio × fs_out ≥ fs_in << 1 (the 2× upsampled rate).
        let fs_in_hz = (fs_in_khz as i32) * 1000;
        let fs_out_hz = (fs_out_khz as i32) * 1000;
        let mut inv_ratio_q16 = ((fs_in_hz << 15) / fs_out_hz) << 2;
        while crate::silk_decode_core::smulww(inv_ratio_q16, fs_out_hz) < (fs_in_hz << 1) {
            inv_ratio_q16 += 1;
        }
        Some(Self {
            bandwidth,
            path,
            fs_in_khz,
            fs_out_khz,
            input_delay: decoder_input_delay(fs_in_khz),
            batch_size: fs_in_khz * MAX_BATCH_MS,
            inv_ratio_q16,
            s_iir: [0; 6],
            s_fir: [0; ORDER_FIR_12],
            delay_buf: [0; 48],
        })
    }

    /// The bandwidth this resampler was built for (its input rate).
    pub fn bandwidth(&self) -> Bandwidth {
        self.bandwidth
    }

    /// The reconstruction path this resampler was built for.
    pub fn path(&self) -> SilkChannelPath {
        self.path
    }

    /// The upsampling factor to 48 kHz (6 / 4 / 3 for NB / MB / WB).
    pub fn factor(&self) -> usize {
        self.fs_out_khz / self.fs_in_khz
    }

    /// Clear the carried filter state (a §4.5.2 SILK state reset).
    pub fn reset(&mut self) {
        self.s_iir = [0; 6];
        self.s_fir = [0; ORDER_FIR_12];
        self.delay_buf = [0; 48];
    }

    /// Resample one frame of internal-rate samples to 48 kHz — i16
    /// domain (the reference decoder's native sample type).
    ///
    /// `out.len()` must equal `input.len() × factor` and `input` must
    /// cover at least 1 ms (every SILK frame does).
    pub fn process_i16(&mut self, input: &[i16], out: &mut [i16]) {
        assert_eq!(
            out.len(),
            input.len() * self.factor(),
            "output must be factor × input"
        );
        assert!(input.len() >= self.fs_in_khz, "need at least 1 ms");
        let n_first = self.fs_in_khz - self.input_delay;

        // 1 ms through the delay-compensation buffer…
        let mut head = [0i16; 48];
        head[..self.fs_in_khz].copy_from_slice(&self.delay_buf[..self.fs_in_khz]);
        head[self.input_delay..self.fs_in_khz].copy_from_slice(&input[..n_first]);
        let (out_head, out_rest) = out.split_at_mut(self.fs_out_khz);
        self.iir_fir(&head[..self.fs_in_khz], out_head);
        // …then the rest of the frame directly.
        // …then the rest of the frame, holding back the final
        // `input_delay` samples for the next frame's delay buffer.
        self.iir_fir(&input[n_first..input.len() - self.input_delay], out_rest);
        // Refill the delay buffer with the frame's tail.
        self.delay_buf[..self.input_delay]
            .copy_from_slice(&input[input.len() - self.input_delay..]);
    }

    /// [`Self::process_i16`] with the crate's `f32` sample convention
    /// (`value = i16 / 32768`, exact for reconstruction-chain signals).
    pub fn process(&mut self, input: &[f32], out: &mut [f32]) {
        let input_i16: Vec<i16> = input
            .iter()
            .map(|&v| (v * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16)
            .collect();
        let mut out_i16 = vec![0i16; out.len()];
        self.process_i16(&input_i16, &mut out_i16);
        for (o, v) in out.iter_mut().zip(&out_i16) {
            *o = f32::from(*v) / 32768.0;
        }
    }

    /// One `silk_resampler_private_IIR_FIR` pass: 2× allpass upsampling
    /// into a scratch buffer (8-sample carried history at the front),
    /// then fractional FIR interpolation to the output rate, in
    /// batches of at most 10 ms.
    fn iir_fir(&mut self, input: &[i16], out: &mut [i16]) {
        // Profluens patch: per-call 2× upsampling scratch (carried FIR history at the front),
        // consumed within this pass — from the per-packet bump arena.
        let mut buf = Vec::with_capacity_in(2 * self.batch_size + ORDER_FIR_12, crate::bump::DecodeBump);
        buf.resize(2 * self.batch_size + ORDER_FIR_12, 0i16);
        buf[..ORDER_FIR_12].copy_from_slice(&self.s_fir);
        let mut in_pos = 0usize;
        let mut out_pos = 0usize;
        loop {
            let n = (input.len() - in_pos).min(self.batch_size);
            self.up2_hq(
                &input[in_pos..in_pos + n],
                &mut buf[ORDER_FIR_12..ORDER_FIR_12 + 2 * n],
            );
            let max_index_q16 = (n as i32) << 17;
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let table_index = crate::silk_decode_core::smulwb(index_q16 & 0xffff, 12) as usize;
                let base = (index_q16 >> 16) as usize;
                let mut res_q15: i32 = 0;
                for t in 0..4 {
                    res_q15 = crate::silk_decode_core::smlabb(
                        res_q15,
                        i32::from(buf[base + t]),
                        i32::from(FRAC_FIR_12[table_index][t]),
                    );
                }
                for t in 0..4 {
                    res_q15 = crate::silk_decode_core::smlabb(
                        res_q15,
                        i32::from(buf[base + 4 + t]),
                        i32::from(FRAC_FIR_12[11 - table_index][3 - t]),
                    );
                }
                out[out_pos] = crate::silk_decode_core::sat16(
                    crate::silk_decode_core::rshift_round(res_q15, 15),
                );
                out_pos += 1;
                index_q16 += self.inv_ratio_q16;
            }
            in_pos += n;
            if in_pos < input.len() {
                buf.copy_within(2 * n..2 * n + ORDER_FIR_12, 0);
            } else {
                self.s_fir
                    .copy_from_slice(&buf[2 * n..2 * n + ORDER_FIR_12]);
                break;
            }
        }
        debug_assert_eq!(out_pos, out.len(), "§4.2.9 output count");
    }

    /// 2× upsampling by two 3-section allpass chains with a notch just
    /// above Nyquist (Q10 internal state).
    fn up2_hq(&mut self, input: &[i16], out: &mut [i16]) {
        use crate::silk_decode_core::{rshift_round, sat16, smlawb, smulwb};
        let s = &mut self.s_iir;
        for (k, &x) in input.iter().enumerate() {
            let in32 = i32::from(x) << 10;

            // Even output phase: three allpass sections.
            let y = in32.wrapping_sub(s[0]);
            let x0 = smulwb(y, UP2_HQ_0[0]);
            let out32_1 = s[0].wrapping_add(x0);
            s[0] = in32.wrapping_add(x0);

            let y = out32_1.wrapping_sub(s[1]);
            let x1 = smulwb(y, UP2_HQ_0[1]);
            let out32_2 = s[1].wrapping_add(x1);
            s[1] = out32_1.wrapping_add(x1);

            let y = out32_2.wrapping_sub(s[2]);
            let x2 = smlawb(y, y, UP2_HQ_0[2]);
            let out32_1 = s[2].wrapping_add(x2);
            s[2] = out32_2.wrapping_add(x2);

            out[2 * k] = sat16(rshift_round(out32_1, 10));

            // Odd output phase.
            let y = in32.wrapping_sub(s[3]);
            let x0 = smulwb(y, UP2_HQ_1[0]);
            let out32_1 = s[3].wrapping_add(x0);
            s[3] = in32.wrapping_add(x0);

            let y = out32_1.wrapping_sub(s[4]);
            let x1 = smulwb(y, UP2_HQ_1[1]);
            let out32_2 = s[4].wrapping_add(x1);
            s[4] = out32_1.wrapping_add(x1);

            let y = out32_2.wrapping_sub(s[5]);
            let x2 = smlawb(y, y, UP2_HQ_1[2]);
            let out32_1 = s[5].wrapping_add(x2);
            s[5] = out32_2.wrapping_add(x2);

            out[2 * k + 1] = sat16(rshift_round(out32_1, 10));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // Table 54 transcription self-checks.
    // ----------------------------------------------------------------

    #[test]
    fn table54_delay_ms_matches_rfc_for_nb_mb_wb() {
        // RFC 6716 §4.2.9 Table 54.
        assert_eq!(silk_resampler_delay_ms(Bandwidth::Nb), Some(0.538));
        assert_eq!(silk_resampler_delay_ms(Bandwidth::Mb), Some(0.692));
        assert_eq!(silk_resampler_delay_ms(Bandwidth::Wb), Some(0.706));
    }

    #[test]
    fn table54_excludes_swb_and_fb() {
        // SWB and FB never reach the §4.2.9 SILK resampler (they only
        // exist as CELT-only configs or as the upper half of a Hybrid
        // frame whose SILK half is WB).
        assert_eq!(silk_resampler_delay_ms(Bandwidth::Swb), None);
        assert_eq!(silk_resampler_delay_ms(Bandwidth::Fb), None);
    }

    #[test]
    fn delay_table_is_strictly_increasing_nb_lt_mb_lt_wb() {
        // §4.2.9 prose: "NB is given a smaller decoder delay
        // allocation than MB and WB to allow a higher-order filter".
        // MB sits between NB and WB. Verify the table's monotonicity.
        let nb = silk_resampler_delay_ms(Bandwidth::Nb).unwrap();
        let mb = silk_resampler_delay_ms(Bandwidth::Mb).unwrap();
        let wb = silk_resampler_delay_ms(Bandwidth::Wb).unwrap();
        assert!(nb < mb, "NB ({nb}) should be < MB ({mb})");
        assert!(mb < wb, "MB ({mb}) should be < WB ({wb})");
    }

    // ----------------------------------------------------------------
    // Delay in samples at output rate.
    // ----------------------------------------------------------------

    #[test]
    fn delay_samples_at_48khz_matches_table54_reference() {
        // Table 54 is stated at 48 kHz. Check the round-to-nearest
        // expansion matches the spec's "samples at 48 kHz" framing.
        //
        // NB: 0.538 ms × 48 = 25.824 samples → 26.
        // MB: 0.692 ms × 48 = 33.216 samples → 33.
        // WB: 0.706 ms × 48 = 33.888 samples → 34.
        assert_eq!(
            silk_resampler_delay_samples_at(Bandwidth::Nb, 48_000),
            Some(26)
        );
        assert_eq!(
            silk_resampler_delay_samples_at(Bandwidth::Mb, 48_000),
            Some(33)
        );
        assert_eq!(
            silk_resampler_delay_samples_at(Bandwidth::Wb, 48_000),
            Some(34)
        );
    }

    #[test]
    fn delay_samples_at_internal_rate_makes_sense() {
        // At 8 kHz: NB delay × 8 = 0.538 × 8 = 4.304 → 4.
        assert_eq!(
            silk_resampler_delay_samples_at(Bandwidth::Nb, 8_000),
            Some(4)
        );
        // At 12 kHz: MB delay × 12 = 0.692 × 12 = 8.304 → 8.
        assert_eq!(
            silk_resampler_delay_samples_at(Bandwidth::Mb, 12_000),
            Some(8)
        );
        // At 16 kHz: WB delay × 16 = 0.706 × 16 = 11.296 → 11.
        assert_eq!(
            silk_resampler_delay_samples_at(Bandwidth::Wb, 16_000),
            Some(11)
        );
    }

    #[test]
    fn delay_samples_at_24khz_intermediate_rate() {
        // 24 kHz is one of the §4.2.9 supported output rates.
        // NB: 0.538 × 24 = 12.912 → 13.
        // MB: 0.692 × 24 = 16.608 → 17.
        // WB: 0.706 × 24 = 16.944 → 17.
        assert_eq!(
            silk_resampler_delay_samples_at(Bandwidth::Nb, 24_000),
            Some(13)
        );
        assert_eq!(
            silk_resampler_delay_samples_at(Bandwidth::Mb, 24_000),
            Some(17)
        );
        assert_eq!(
            silk_resampler_delay_samples_at(Bandwidth::Wb, 24_000),
            Some(17)
        );
    }

    #[test]
    fn delay_samples_rejects_swb_fb_and_zero_rate() {
        assert_eq!(
            silk_resampler_delay_samples_at(Bandwidth::Swb, 48_000),
            None
        );
        assert_eq!(silk_resampler_delay_samples_at(Bandwidth::Fb, 48_000), None);
        assert_eq!(silk_resampler_delay_samples_at(Bandwidth::Nb, 0), None);
    }

    // ----------------------------------------------------------------
    // Supported-output-rate dispatch.
    // ----------------------------------------------------------------

    #[test]
    fn supported_output_rates_are_the_five_spec_rates() {
        // §4.2.9: "8, 12, 16, 24, or 48 kHz".
        assert_eq!(
            SUPPORTED_OUTPUT_RATES_HZ,
            &[8_000, 12_000, 16_000, 24_000, 48_000][..]
        );
        for &r in SUPPORTED_OUTPUT_RATES_HZ {
            assert!(is_supported_output_rate(r), "rate {r} should be supported");
        }
        // A few that aren't on the list.
        for r in [0u32, 11_025, 22_050, 32_000, 44_100, 96_000] {
            assert!(
                !is_supported_output_rate(r),
                "rate {r} should NOT be in §4.2.9 list"
            );
        }
    }

    #[test]
    fn reference_rate_is_48khz() {
        // Table 54 is anchored at 48 kHz; CELT also runs at 48 kHz.
        assert_eq!(REFERENCE_RATE_HZ, 48_000);
        assert!(is_supported_output_rate(REFERENCE_RATE_HZ));
    }

    // ----------------------------------------------------------------
    // Internal SILK rate per bandwidth.
    // ----------------------------------------------------------------

    #[test]
    fn internal_silk_rate_per_bandwidth() {
        assert_eq!(silk_internal_rate_hz(Bandwidth::Nb), Some(8_000));
        assert_eq!(silk_internal_rate_hz(Bandwidth::Mb), Some(12_000));
        assert_eq!(silk_internal_rate_hz(Bandwidth::Wb), Some(16_000));
        // SWB / FB don't reach the SILK layer.
        assert_eq!(silk_internal_rate_hz(Bandwidth::Swb), None);
        assert_eq!(silk_internal_rate_hz(Bandwidth::Fb), None);
    }

    #[test]
    fn internal_silk_rate_is_a_supported_output_rate_for_nb_and_wb() {
        // Decoders that don't want resampling can ask for the SILK
        // internal rate directly — for NB and WB this is also a §4.2.9
        // supported output rate (8 kHz, 16 kHz). MB's 12 kHz is also on
        // the supported-output list.
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            let r = silk_internal_rate_hz(bw).unwrap();
            assert!(
                is_supported_output_rate(r),
                "internal SILK rate {r} for {bw:?} should be in §4.2.9 list"
            );
        }
    }

    // ----------------------------------------------------------------
    // Per-frame sample counts.
    // ----------------------------------------------------------------

    #[test]
    fn silk_frame_samples_internal_canonical_cases() {
        // NB internal = 8000 Hz.
        // 10 ms = 80 samples; 20 ms = 160 samples.
        assert_eq!(silk_frame_samples_internal(Bandwidth::Nb, 100), Some(80));
        assert_eq!(silk_frame_samples_internal(Bandwidth::Nb, 200), Some(160));
        // MB internal = 12000 Hz.
        // 10 ms = 120 samples; 20 ms = 240 samples.
        assert_eq!(silk_frame_samples_internal(Bandwidth::Mb, 100), Some(120));
        assert_eq!(silk_frame_samples_internal(Bandwidth::Mb, 200), Some(240));
        // WB internal = 16000 Hz.
        // 10 ms = 160 samples; 20 ms = 320 samples.
        assert_eq!(silk_frame_samples_internal(Bandwidth::Wb, 100), Some(160));
        assert_eq!(silk_frame_samples_internal(Bandwidth::Wb, 200), Some(320));
    }

    #[test]
    fn silk_frame_samples_internal_rejects_non_silk_durations() {
        // 40 ms and 60 ms Opus frames carry MULTIPLE SILK frames; this
        // helper measures ONE SILK frame so 400 / 600 are not valid
        // inputs. 25 / 50 (2.5 / 5 ms) are CELT-only.
        for dur in [0u16, 25, 50, 400, 600, 1234] {
            assert_eq!(
                silk_frame_samples_internal(Bandwidth::Nb, dur),
                None,
                "dur {dur} should be rejected"
            );
            assert_eq!(silk_frame_samples_internal(Bandwidth::Mb, dur), None);
            assert_eq!(silk_frame_samples_internal(Bandwidth::Wb, dur), None);
        }
    }

    #[test]
    fn silk_frame_samples_internal_rejects_swb_and_fb() {
        for bw in [Bandwidth::Swb, Bandwidth::Fb] {
            for dur in [100u16, 200] {
                assert_eq!(
                    silk_frame_samples_internal(bw, dur),
                    None,
                    "{bw:?} {dur} should be rejected"
                );
            }
        }
    }

    #[test]
    fn silk_frame_samples_at_output_48khz_matches_duration() {
        // At 48 kHz: 10 ms = 480 samples; 20 ms = 960 samples;
        // independent of bandwidth.
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            assert_eq!(
                silk_frame_samples_at_output(bw, 100, 48_000),
                Some(480),
                "{bw:?} 10 ms"
            );
            assert_eq!(
                silk_frame_samples_at_output(bw, 200, 48_000),
                Some(960),
                "{bw:?} 20 ms"
            );
        }
    }

    #[test]
    fn silk_frame_samples_at_output_matches_internal_when_rate_matches() {
        // When the output rate equals the internal SILK rate, the
        // output sample count is identical to the internal one (no
        // resampling needed).
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            let r = silk_internal_rate_hz(bw).unwrap();
            for dur in [100u16, 200] {
                assert_eq!(
                    silk_frame_samples_at_output(bw, dur, r),
                    silk_frame_samples_internal(bw, dur),
                    "{bw:?} {dur} at {r} Hz"
                );
            }
        }
    }

    #[test]
    fn silk_frame_samples_at_output_rejects_zero_rate_and_non_silk() {
        assert_eq!(silk_frame_samples_at_output(Bandwidth::Nb, 100, 0), None);
        assert_eq!(
            silk_frame_samples_at_output(Bandwidth::Swb, 100, 48_000),
            None
        );
        assert_eq!(
            silk_frame_samples_at_output(Bandwidth::Nb, 25, 48_000),
            None
        );
        assert_eq!(
            silk_frame_samples_at_output(Bandwidth::Nb, 400, 48_000),
            None
        );
    }

    // ----------------------------------------------------------------
    // SilkUpsampler: construction and geometry.
    // ----------------------------------------------------------------

    #[test]
    fn upsampler_constructs_for_silk_bandwidths_only() {
        for (bw, factor) in [
            (Bandwidth::Nb, 6usize),
            (Bandwidth::Mb, 4),
            (Bandwidth::Wb, 3),
        ] {
            for path in [SilkChannelPath::Mono, SilkChannelPath::Stereo] {
                let up = SilkUpsampler::new(bw, path).unwrap();
                assert_eq!(up.factor(), factor, "{bw:?}");
                assert_eq!(up.bandwidth(), bw);
                assert_eq!(up.path(), path);
            }
        }
        assert!(SilkUpsampler::new(Bandwidth::Swb, SilkChannelPath::Mono).is_none());
        assert!(SilkUpsampler::new(Bandwidth::Fb, SilkChannelPath::Stereo).is_none());
    }

    /// The fractional-ratio accumulator step satisfies the reference
    /// round-up invariant: `invRatio × 48000 ≥ 2 × Fs_in` and it is the
    /// smallest such value ≥ the truncated base ratio.
    #[test]
    fn upsampler_inv_ratio_round_up_invariant() {
        use crate::silk_decode_core::smulww;
        for (bw, fs_in) in [
            (Bandwidth::Nb, 8000i32),
            (Bandwidth::Mb, 12000),
            (Bandwidth::Wb, 16000),
        ] {
            let up = SilkUpsampler::new(bw, SilkChannelPath::Mono).unwrap();
            let r = up.inv_ratio_q16;
            assert!(smulww(r, 48000) >= fs_in << 1, "{bw:?}: ratio too small");
            assert!(
                smulww(r - 1, 48000) < fs_in << 1,
                "{bw:?}: ratio not minimal"
            );
        }
    }

    /// Every frame length produces exactly `factor × len` output
    /// samples, and processing is deterministic.
    #[test]
    fn upsampler_output_count_and_determinism() {
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            let rate = silk_internal_rate_hz(bw).unwrap() as usize;
            for ms in [10usize, 20, 40, 60] {
                let n = rate * ms / 1000;
                let input: Vec<i16> = (0..n).map(|i| ((i * 37) % 2000) as i16 - 1000).collect();
                let mut a = SilkUpsampler::new(bw, SilkChannelPath::Mono).unwrap();
                let mut b = SilkUpsampler::new(bw, SilkChannelPath::Mono).unwrap();
                let mut out_a = vec![0i16; n * a.factor()];
                let mut out_b = vec![0i16; n * b.factor()];
                a.process_i16(&input, &mut out_a);
                b.process_i16(&input, &mut out_b);
                assert_eq!(out_a, out_b, "{bw:?} {ms}ms");
            }
        }
    }

    /// A DC input settles to (nearly) the same DC value once the
    /// allpass/FIR chain has warmed up.
    #[test]
    fn upsampler_passes_dc_after_warmup() {
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            let rate = silk_internal_rate_hz(bw).unwrap() as usize;
            let n = rate / 50; // 20 ms
            let input = vec![8192i16; n];
            let mut up = SilkUpsampler::new(bw, SilkChannelPath::Mono).unwrap();
            let mut out = vec![0i16; n * up.factor()];
            // Two frames: the second is fully warmed up.
            up.process_i16(&input, &mut out);
            up.process_i16(&input, &mut out);
            let tail = &out[out.len() / 2..];
            for (i, &v) in tail.iter().enumerate() {
                assert!(
                    (i32::from(v) - 8192).abs() <= 8,
                    "{bw:?}: DC error at {i}: {v}"
                );
            }
        }
    }

    /// Streaming two 10 ms halves equals one 20 ms call (the carried
    /// IIR/FIR/delay state makes frame boundaries seamless).
    #[test]
    fn upsampler_streaming_equals_batch() {
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            let rate = silk_internal_rate_hz(bw).unwrap() as usize;
            let n = rate / 50; // 20 ms
            let input: Vec<i16> = (0..n)
                .map(|i| (6000.0 * (i as f64 * 0.31).sin()) as i16)
                .collect();
            let mut whole = SilkUpsampler::new(bw, SilkChannelPath::Mono).unwrap();
            let mut split = SilkUpsampler::new(bw, SilkChannelPath::Mono).unwrap();
            let f = whole.factor();
            let mut out_whole = vec![0i16; n * f];
            whole.process_i16(&input, &mut out_whole);
            let mut out_split = vec![0i16; n * f];
            let half = n / 2;
            let (o1, o2) = out_split.split_at_mut(half * f);
            split.process_i16(&input[..half], o1);
            split.process_i16(&input[half..], o2);
            assert_eq!(out_whole, out_split, "{bw:?}");
        }
    }

    /// Reset clears the carried state: decode after reset matches a
    /// fresh resampler.
    #[test]
    fn upsampler_reset_clears_history() {
        let mut up = SilkUpsampler::new(Bandwidth::Wb, SilkChannelPath::Mono).unwrap();
        let noise: Vec<i16> = (0..320).map(|i| ((i * 97) % 4000) as i16 - 2000).collect();
        let mut out = vec![0i16; 960];
        up.process_i16(&noise, &mut out);
        up.reset();
        let mut fresh = SilkUpsampler::new(Bandwidth::Wb, SilkChannelPath::Mono).unwrap();
        let mut out_reset = vec![0i16; 960];
        let mut out_fresh = vec![0i16; 960];
        up.process_i16(&noise, &mut out_reset);
        fresh.process_i16(&noise, &mut out_fresh);
        assert_eq!(out_reset, out_fresh);
    }

    // ----------------------------------------------------------------
    // Cross-checks: delay never exceeds one SILK frame.
    // ----------------------------------------------------------------

    #[test]
    fn delay_is_smaller_than_one_silk_frame_at_every_supported_rate() {
        // Sanity: the §4.2.9 delay allocation is well under 1 ms.
        // For every supported output rate, the delay in samples
        // must be strictly less than the per-SILK-frame sample count
        // (10 ms = the shorter SILK frame). Otherwise the resampler
        // would be holding more than one frame of data.
        for &out_rate in SUPPORTED_OUTPUT_RATES_HZ {
            for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
                let delay = silk_resampler_delay_samples_at(bw, out_rate).unwrap();
                let one_frame = silk_frame_samples_at_output(bw, 100, out_rate).unwrap();
                assert!(
                    (delay as u64) < (one_frame as u64),
                    "{bw:?} delay {delay} >= one 10ms SILK frame {one_frame} at {out_rate} Hz"
                );
            }
        }
    }
}
