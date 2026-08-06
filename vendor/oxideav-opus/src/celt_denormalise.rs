//! CELT §4.3.6 band denormalisation
//! (RFC 6716 §4.3.6, p. 121).
//!
//! The last step of the CELT decoder before the inverse MDCT (§4.3.7)
//! is to *denormalise* the bands. RFC 6716 §4.3.6 (p. 121) states it
//! in one sentence:
//!
//! > "Just as each band was normalized in the encoder, the last step
//! > of the decoder before the inverse MDCT is to denormalize the
//! > bands. Each decoded normalized band is multiplied by the square
//! > root of the decoded energy."
//!
//! ## The two inputs and the one operation
//!
//! By the time control reaches this stage the decoder holds, per band:
//!
//! * the band **shape** — a unit-L2-norm vector of `N` MDCT
//!   coefficients produced by the §4.3.4.2 PVQ shape decoder
//!   ([`crate::celt_pvq_decode::pvq_unit_normalize`]); for a band that
//!   received zero pulses (`K = 0`) the shape is the all-zero vector,
//!   and the §4.3.5 anti-collapse / folding path supplies any nonzero
//!   content. This module consumes whatever unit-norm (or zero) shape
//!   it is handed.
//! * the band **energy** in the base-2 log domain — the
//!   coarse-plus-fine reconstruction of §4.3.2 (`coarse + fine
//!   correction`, in `log2` units of the per-band energy envelope).
//!
//! Denormalisation is the per-coefficient product
//!
//! ```text
//!   X[band][j] = shape[band][j] * sqrt(energy[band])
//! ```
//!
//! where `energy[band]` is the *linear* band energy. Because the
//! energy arrives in the base-2 log domain (`L = log2(energy)`), the
//! linear energy is `energy = 2**L` and its square root is
//!
//! ```text
//!   sqrt(energy) = sqrt(2**L) = 2**(L/2).
//! ```
//!
//! So each coefficient of a band is scaled by the single per-band gain
//! `g = 2**(L/2)`. A unit-norm shape scaled by `g` yields a band whose
//! L2 energy is exactly `g**2 = 2**L = energy`, which is precisely the
//! "energy preservation" property §4.3.5 and §4.3.6 are built around
//! (the encoder normalised each band to unit norm and coded its energy
//! separately; the decoder restores the energy here).
//!
//! ## Scope / the band layout
//!
//! A 20 ms CELT-only frame lays its 21 bands' coefficients end-to-end
//! into a single `960`-bin (per channel) frequency-domain buffer; the
//! per-band bin counts come from Table 55
//! ([`crate::celt_band_layout::celt_band_bins_per_channel`]). Hybrid
//! frames start at band 17 (the SILK layer carries the low bands).
//! [`denormalise_bands`] walks the coded bands in order, writing each
//! band's denormalised coefficients into the contiguous output region
//! that the inverse MDCT then consumes.
//!
//! This module owns only the log-energy → linear-gain conversion and
//! the per-band multiply-into-place. The energy reconstruction
//! (§4.3.2, upstream) and the inverse MDCT (§4.3.7, downstream) live in
//! their own modules; the band *shapes* are produced by the §4.3.4 PVQ
//! decoder. The boundary kept here is exactly the one the RFC draws:
//! "multiplied by the square root of the decoded energy", nothing more.
//!
//! ## Provenance
//!
//! Denormalisation operation (`shape * sqrt(energy)`, per band, as the
//! last step before the inverse MDCT) and the base-2-log energy
//! representation: RFC 6716 §4.3.6 + §4.3.2 (p. 121, p. 108),
//! reproduced from `docs/audio/opus/rfc6716-opus.txt`. The
//! `sqrt(2**L) = 2**(L/2)` identity is elementary algebra over the
//! stated log domain. No external library source was consulted; the
//! operation is stated directly in the standards-track text.

use crate::celt_band_layout::{
    celt_band_bins_per_channel, celt_end_coded_band, celt_first_coded_band, CeltFrameSize,
    CELT_NUM_BANDS,
};

/// Errors returnable by the §4.3.6 denormalisation helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenormaliseError {
    /// A band index was `>= CELT_NUM_BANDS` (21).
    BandOutOfRange {
        /// The offending band index.
        band: usize,
    },
    /// The output slice was shorter than the number of coefficients to
    /// write.
    OutputBufferTooSmall {
        /// Coefficients the band needed to write.
        required: usize,
        /// Length of the slice provided.
        provided: usize,
    },
    /// The shape slice and the output slice had different lengths (the
    /// per-coefficient multiply is element-wise).
    ShapeLengthMismatch {
        /// Length of the supplied unit-norm shape.
        shape_len: usize,
        /// Length the output region expected.
        expected: usize,
    },
}

impl core::fmt::Display for DenormaliseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DenormaliseError::BandOutOfRange { band } => write!(
                f,
                "band index {band} out of range (must be < {CELT_NUM_BANDS}) \
                 per RFC 6716 §4.3 Table 55"
            ),
            DenormaliseError::OutputBufferTooSmall { required, provided } => write!(
                f,
                "denormalisation output buffer too small: need {required} \
                 coefficients, got {provided}"
            ),
            DenormaliseError::ShapeLengthMismatch {
                shape_len,
                expected,
            } => write!(
                f,
                "shape length {shape_len} does not match output region length \
                 {expected} for the §4.3.6 element-wise multiply"
            ),
        }
    }
}

impl std::error::Error for DenormaliseError {}

/// The per-band denormalisation gain `g = sqrt(2**log2_energy)
/// = 2**(log2_energy / 2)` for a band whose decoded energy is
/// `log2_energy` in the base-2 log domain (RFC 6716 §4.3.6 + §4.3.2).
///
/// This is the single scalar every coefficient of the band is
/// multiplied by; a unit-L2-norm shape scaled by this gain has L2
/// energy exactly `2**log2_energy`, restoring the per-band energy the
/// encoder coded separately.
///
/// ```
/// use oxideav_opus::celt_denormalise::denormalise_gain;
/// // log2_energy = 0 → linear energy 1 → gain 1.
/// assert_eq!(denormalise_gain(0.0), 1.0);
/// // log2_energy = 2 → linear energy 4 → gain sqrt(4) = 2.
/// assert!((denormalise_gain(2.0) - 2.0).abs() < 1e-12);
/// // log2_energy = -2 → linear energy 1/4 → gain 1/2.
/// assert!((denormalise_gain(-2.0) - 0.5).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn denormalise_gain(log2_energy: f64) -> f64 {
    // sqrt(2**L) = 2**(L/2). exp2 is the natural primitive for the
    // base-2 log domain the energy envelope lives in (§4.3.2).
    (log2_energy * 0.5).exp2()
}

/// Denormalises one band in place: multiplies its unit-L2-norm `shape`
/// by `sqrt(2**log2_energy)` (RFC 6716 §4.3.6), writing the result into
/// `out`.
///
/// `shape` and `out` must have the same length (the multiply is
/// element-wise). `shape` and `out` may be the same buffer at the call
/// site only if the caller arranges aliasing; this signature keeps
/// them separate so the input shape can be reused (e.g. for the second
/// stereo channel).
///
/// # Errors
///
/// * [`DenormaliseError::ShapeLengthMismatch`] if `shape.len() != out.len()`.
pub fn denormalise_band(
    shape: &[f64],
    log2_energy: f64,
    out: &mut [f64],
) -> Result<(), DenormaliseError> {
    if shape.len() != out.len() {
        return Err(DenormaliseError::ShapeLengthMismatch {
            shape_len: shape.len(),
            expected: out.len(),
        });
    }
    let gain = denormalise_gain(log2_energy);
    for (dst, &s) in out.iter_mut().zip(shape.iter()) {
        *dst = s * gain;
    }
    Ok(())
}

/// Denormalises a whole CELT frame's worth of bands (RFC 6716 §4.3.6),
/// writing the contiguous frequency-domain coefficient buffer the
/// inverse MDCT (§4.3.7) consumes.
///
/// `shapes[k]` is the unit-L2-norm shape of the `k`-th *coded* band
/// (coded bands run `celt_first_coded_band(is_hybrid) ..
/// celt_end_coded_band()` — i.e. `0..21` for CELT-only, `17..21` for
/// Hybrid). `log2_energy[k]` is the matching band's decoded energy in
/// the base-2 log domain. Each shape's length must equal that band's
/// Table-55 bin count for `frame_size`
/// ([`celt_band_bins_per_channel`]).
///
/// The bands are written end-to-end into `out` in band order; `out`
/// must be at least as long as the sum of the coded bands' bin counts
/// (the §4.3 per-channel total,
/// [`crate::celt_band_layout::celt_total_bins_per_channel`]). The
/// number of coefficients actually written is returned.
///
/// # Errors
///
/// * [`DenormaliseError::OutputBufferTooSmall`] if `out` cannot hold
///   every coded band's coefficients.
/// * [`DenormaliseError::ShapeLengthMismatch`] if any `shapes[k]` does
///   not have its band's Table-55 bin count, or if `shapes.len()` /
///   `log2_energy.len()` does not match the coded-band count.
pub fn denormalise_bands(
    shapes: &[&[f64]],
    log2_energy: &[f64],
    frame_size: CeltFrameSize,
    is_hybrid: bool,
    out: &mut [f64],
) -> Result<usize, DenormaliseError> {
    let first = celt_first_coded_band(is_hybrid);
    let end = celt_end_coded_band();
    let coded = end - first;

    if shapes.len() != coded {
        return Err(DenormaliseError::ShapeLengthMismatch {
            shape_len: shapes.len(),
            expected: coded,
        });
    }
    if log2_energy.len() != coded {
        return Err(DenormaliseError::ShapeLengthMismatch {
            shape_len: log2_energy.len(),
            expected: coded,
        });
    }

    let mut offset = 0usize;
    for (k, band) in (first..end).enumerate() {
        // `celt_band_bins_per_channel` returns `Some` for every
        // `band < CELT_NUM_BANDS`, and the loop bound guarantees that;
        // the `expect` documents the invariant rather than handling a
        // reachable case.
        let bins = celt_band_bins_per_channel(band, frame_size)
            .expect("coded band index < CELT_NUM_BANDS by loop bound") as usize;

        let shape = shapes[k];
        if shape.len() != bins {
            return Err(DenormaliseError::ShapeLengthMismatch {
                shape_len: shape.len(),
                expected: bins,
            });
        }

        let end_off = offset + bins;
        if end_off > out.len() {
            return Err(DenormaliseError::OutputBufferTooSmall {
                required: end_off,
                provided: out.len(),
            });
        }

        denormalise_band(shape, log2_energy[k], &mut out[offset..end_off])?;
        offset = end_off;
    }

    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_band_layout::{celt_total_bins_per_channel, CeltFrameSize};

    #[test]
    fn gain_is_sqrt_of_linear_energy() {
        // g = sqrt(2**L). Check a spread of L against an independent
        // sqrt(2.0.powf(L)) computation.
        for &l in &[-8.0, -2.0, -1.0, 0.0, 0.5, 1.0, 3.0, 7.5, 12.0] {
            let expected = 2.0_f64.powf(l).sqrt();
            assert!(
                (denormalise_gain(l) - expected).abs() < 1e-12,
                "L={l}: {} vs {}",
                denormalise_gain(l),
                expected
            );
        }
    }

    #[test]
    fn gain_zero_energy_is_one() {
        // log2_energy = 0 → linear energy = 1 → gain = 1 (identity).
        assert_eq!(denormalise_gain(0.0), 1.0);
    }

    #[test]
    fn denormalised_band_has_target_energy() {
        // A unit-L2-norm shape scaled by sqrt(2**L) must have L2 energy
        // 2**L — the energy-preservation property of §4.3.6.
        let raw = [3.0_f64, 4.0]; // L2 norm = 5
        let norm = (raw[0] * raw[0] + raw[1] * raw[1]).sqrt();
        let shape = [raw[0] / norm, raw[1] / norm]; // unit norm
        let l = 3.0_f64;
        let mut out = [0.0_f64; 2];
        denormalise_band(&shape, l, &mut out).unwrap();
        let energy = out[0] * out[0] + out[1] * out[1];
        assert!(
            (energy - 2.0_f64.powf(l)).abs() < 1e-9,
            "band energy {energy} != 2**{l} = {}",
            2.0_f64.powf(l)
        );
    }

    #[test]
    fn zero_shape_stays_zero() {
        // A K=0 band carries an all-zero shape; denormalisation of zero
        // is zero regardless of the (finite) energy.
        let shape = [0.0_f64; 4];
        let mut out = [9.9_f64; 4];
        denormalise_band(&shape, 5.0, &mut out).unwrap();
        assert_eq!(out, [0.0; 4]);
    }

    #[test]
    fn band_length_mismatch_errors() {
        let shape = [1.0_f64; 3];
        let mut out = [0.0_f64; 4];
        assert_eq!(
            denormalise_band(&shape, 0.0, &mut out),
            Err(DenormaliseError::ShapeLengthMismatch {
                shape_len: 3,
                expected: 4,
            })
        );
    }

    #[test]
    fn bands_celt_only_2p5ms_writes_full_buffer() {
        // CELT-only 2.5 ms: 21 coded bands, 100 bins/channel total.
        let fs = CeltFrameSize::from_frame_tenths_ms(25).unwrap();
        let coded = celt_end_coded_band() - celt_first_coded_band(false);
        assert_eq!(coded, CELT_NUM_BANDS);

        // Build a unit-norm shape per band (all energy in the first bin)
        // sized to its Table-55 bin count.
        let mut owned: Vec<Vec<f64>> = Vec::with_capacity(coded);
        for band in 0..coded {
            let bins = celt_band_bins_per_channel(band, fs).unwrap() as usize;
            let mut v = vec![0.0_f64; bins];
            v[0] = 1.0; // unit norm: a single ±1 coordinate
            owned.push(v);
        }
        let shapes: Vec<&[f64]> = owned.iter().map(|v| v.as_slice()).collect();
        let energies = vec![2.0_f64; coded]; // L=2 → gain 2 every band

        // 2.5 ms CELT-only column sum of Table 55: 8×1 + 4×2 + 3×4 +
        // 2×6 + 8 + 12 + 18 + 22 = 100 bins / channel.
        let total = celt_total_bins_per_channel(fs, false) as usize;
        assert_eq!(total, 100);
        let mut out = vec![0.0_f64; total];
        let written = denormalise_bands(&shapes, &energies, fs, false, &mut out).unwrap();
        assert_eq!(written, total);

        // The first bin of each band carries the gain (2.0); the rest 0.
        let mut offset = 0;
        for band in 0..coded {
            let bins = celt_band_bins_per_channel(band, fs).unwrap() as usize;
            assert!((out[offset] - 2.0).abs() < 1e-12, "band {band} first bin");
            for j in 1..bins {
                assert_eq!(out[offset + j], 0.0, "band {band} bin {j}");
            }
            offset += bins;
        }
    }

    #[test]
    fn bands_hybrid_20ms_starts_at_band_17() {
        // Hybrid frames code only bands 17..21 (the SILK layer carries
        // the low bands); the denormalised buffer holds only those.
        let fs = CeltFrameSize::from_frame_tenths_ms(200).unwrap();
        let first = celt_first_coded_band(true);
        assert_eq!(first, 17);
        let coded = celt_end_coded_band() - first;
        assert_eq!(coded, 4);

        let mut owned: Vec<Vec<f64>> = Vec::with_capacity(coded);
        for (k, band) in (first..celt_end_coded_band()).enumerate() {
            let bins = celt_band_bins_per_channel(band, fs).unwrap() as usize;
            let mut v = vec![0.0_f64; bins];
            v[0] = 1.0;
            owned.push(v);
            let _ = k;
        }
        let shapes: Vec<&[f64]> = owned.iter().map(|v| v.as_slice()).collect();
        let energies = vec![0.0_f64; coded]; // gain 1

        let total = celt_total_bins_per_channel(fs, true) as usize;
        let mut out = vec![0.0_f64; total];
        let written = denormalise_bands(&shapes, &energies, fs, true, &mut out).unwrap();
        assert_eq!(written, total);
    }

    #[test]
    fn bands_output_too_small_errors() {
        let fs = CeltFrameSize::from_frame_tenths_ms(25).unwrap();
        let coded = CELT_NUM_BANDS;
        let mut owned: Vec<Vec<f64>> = Vec::with_capacity(coded);
        for band in 0..coded {
            let bins = celt_band_bins_per_channel(band, fs).unwrap() as usize;
            owned.push(vec![0.0_f64; bins]);
        }
        let shapes: Vec<&[f64]> = owned.iter().map(|v| v.as_slice()).collect();
        let energies = vec![0.0_f64; coded];
        let mut out = vec![0.0_f64; 10]; // far too small
        let r = denormalise_bands(&shapes, &energies, fs, false, &mut out);
        assert!(matches!(
            r,
            Err(DenormaliseError::OutputBufferTooSmall { .. })
        ));
    }

    #[test]
    fn bands_wrong_shape_count_errors() {
        let fs = CeltFrameSize::from_frame_tenths_ms(25).unwrap();
        let shapes: Vec<&[f64]> = vec![&[1.0][..]; 3]; // only 3, need 21
        let energies = vec![0.0_f64; 3];
        let mut out = vec![0.0_f64; 120];
        let r = denormalise_bands(&shapes, &energies, fs, false, &mut out);
        assert!(matches!(
            r,
            Err(DenormaliseError::ShapeLengthMismatch { .. })
        ));
    }

    #[test]
    fn bands_wrong_per_band_shape_len_errors() {
        // Right band count, but one shape has the wrong bin count.
        let fs = CeltFrameSize::from_frame_tenths_ms(25).unwrap();
        let coded = CELT_NUM_BANDS;
        let mut owned: Vec<Vec<f64>> = Vec::with_capacity(coded);
        for band in 0..coded {
            let bins = celt_band_bins_per_channel(band, fs).unwrap() as usize;
            owned.push(vec![0.0_f64; bins]);
        }
        // Corrupt band 5's length.
        owned[5].push(0.0);
        let shapes: Vec<&[f64]> = owned.iter().map(|v| v.as_slice()).collect();
        let energies = vec![0.0_f64; coded];
        let mut out = vec![0.0_f64; 121];
        let r = denormalise_bands(&shapes, &energies, fs, false, &mut out);
        assert!(matches!(
            r,
            Err(DenormaliseError::ShapeLengthMismatch { .. })
        ));
    }

    #[test]
    fn negative_energy_attenuates() {
        // L < 0 → linear energy < 1 → gain < 1 → attenuation.
        let shape = [1.0_f64];
        let mut out = [0.0_f64; 1];
        denormalise_band(&shape, -4.0, &mut out).unwrap();
        // gain = sqrt(2**-4) = 2**-2 = 0.25.
        assert!((out[0] - 0.25).abs() < 1e-12);
    }
}
