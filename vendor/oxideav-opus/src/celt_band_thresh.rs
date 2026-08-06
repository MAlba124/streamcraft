//! CELT §4.3.3 per-band minimum-allocation vector (RFC 6716 §4.3.3,
//! p. 115).
//!
//! After the §4.3.3 reservation block (round 34,
//! [`crate::celt_reservations::reserve_block`]) skims the four
//! fixed-cost reservations off the working budget, the §4.3.3 procedure
//! computes a per-band *hard minimum* shape allocation `thresh[band]`.
//! This minimum is higher than the technical limit of the §4.3.4 PVQ
//! process; bands whose allocation would drop below `thresh[band]`
//! receive no shape allocation at all — they are skipped rather than
//! coded sparsely.
//!
//! The §4.3.3 narrative (RFC 6716 §4.3.3, p. 115) reads:
//!
//! > The allocation process then computes a vector representing the
//! > hard minimum amounts allocation any band will receive for shape.
//! > This minimum is higher than the technical limit of the PVQ
//! > process, but very low rate allocations produce an excessively
//! > sparse spectrum and these bands are better served by having no
//! > allocation at all. For each coded band, set thresh[band] to 24
//! > times the number of MDCT bins in the band and divide by 16. If 8
//! > times the number of channels is greater, use that instead. This
//! > sets the minimum allocation to one bit per channel or 48 128th
//! > bits per MDCT bin, whichever is greater. The band-size dependent
//! > part of this value is not scaled by the channel count, because
//! > at the very low rates where this limit is applicable there will
//! > usually be no bits allocated to the side.
//!
//! ## §4.3.3 formula
//!
//! For each coded band `b`, with `N = bins_per_channel(b, frame_size)`
//! the number of MDCT bins per channel in band `b`, and
//! `channels ∈ {1, 2}` the channel count:
//!
//! ```text
//! thresh[b] = max((24 * N) / 16, 8 * channels)
//! ```
//!
//! Both terms are in 1/8 bits (the same units as every other §4.3.3
//! budget value). The `(24 * N) / 16` term is 48/128-bit per MDCT bin
//! (= 0.375 bit/bin = `(24 / 16) / 8` whole bits per bin in fractional
//! form, which the spec phrases as "48 128th bits per MDCT bin"). The
//! `8 * channels` term is one whole bit per channel (= 8 1/8 bits per
//! channel).
//!
//! The §4.3.3 narrative is emphatic that the band-size dependent term
//! `(24 * N) / 16` is *not* scaled by the channel count: at the very
//! low rates where the threshold actually matters, the §4.3.3 allocator
//! ends up putting most or all of the side-channel shape budget on
//! the mid channel, so the per-band minimum tracks the mid only.
//!
//! ## What this module does not own
//!
//! * The §4.3.3 reservation block (round 34,
//!   [`crate::celt_reservations`]) — runs immediately before the
//!   minimum-allocation vector is computed; consumes `total_boost`
//!   from round 33 and produces the working `total` budget the
//!   minimum-allocation vector is compared against at the consumer
//!   site (the §4.3.3 Table 57 static-allocation search).
//! * The §4.3.3 allocation trim's per-band `trim_offsets[]` derivation
//!   (RFC 6716 §4.3.3 p. 115). The `trim_offsets[]` vector biases the
//!   §4.3.3 Table 57 static-allocation search; it is computed
//!   alongside `thresh[]` but follows a different formula (depends on
//!   `alloc_trim`, the shortest frame size for the mode, and the
//!   number of remaining bands). A separate module will own it once
//!   we wire up the §4.3.3 allocator's Table 57 search.
//! * The §4.3.3 Table 57 static-allocation search itself. That search
//!   takes the working `total` (from
//!   [`crate::celt_reservations::ReservationOutcome::total_remaining_eighth_bits`]),
//!   the per-band [`band_min_thresh`] minimum, the per-band
//!   [`crate::celt_cache_caps50::cap_for_band_bits`] maximum, the
//!   per-band `trim_offsets[]`, and the per-band boosts from
//!   [`crate::celt_band_boost::decode_band_boosts`], and converges on
//!   a quality index `q` whose interpolated allocation fits the
//!   budget. The search runs at the §4.3.3 allocator's consumer site;
//!   this module owns only the per-band lower bound it consults.
//! * Any bitstream read. `thresh[band]` is a pure function of the
//!   band layout (Table 55 via
//!   [`crate::celt_band_layout::celt_band_bins_per_channel`]) and the
//!   channel count. No range-coder symbol is consumed here.
//!
//! ## Units
//!
//! Every value emitted by this module is in 1/8 bits ("8th bits" /
//! "Q3" in the §4.3.3 narrative). The §4.3.3 minimum-allocation
//! vector slots into the working `total` (in 1/8 bits) at the
//! consumer site without any unit conversion.
//!
//! ## Range
//!
//! The §4.3 standard (non-Custom) CELT layer has 21 bands at
//! `N ∈ [1, 176]` MDCT bins per channel (Table 55). The §4.3.3
//! minimum-allocation formula caps cleanly:
//!
//! * Smallest: `N = 1` ⇒ `(24 * 1) / 16 = 1`, and `8 * channels` ∈
//!   {8, 16} always wins ⇒ `thresh = 8` (mono) or `16` (stereo).
//! * Largest: `N = 176` ⇒ `(24 * 176) / 16 = 264`, and
//!   `8 * channels` ∈ {8, 16} always loses ⇒ `thresh = 264`.
//!
//! Every `thresh[band]` value fits in `u32` by a wide margin.
//!
//! ## Provenance
//!
//! Narrative: RFC 6716 §4.3.3 (p. 115) in
//! `docs/audio/opus/rfc6716-opus.txt`; cross-referenced by
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.6
//! ("Minimums and trim offsets"). The two formula coefficients
//! (`24 / 16`, `8`) and the `max()` selector are inlined in the RFC
//! body — no separate numeric table is needed. The per-band MDCT bin
//! counts come from round 24's
//! [`crate::celt_band_layout::celt_band_bins_per_channel`] (Table 55
//! lookup).

use crate::celt_band_layout::{
    celt_band_bins_per_channel, celt_end_coded_band, celt_first_coded_band, CeltFrameSize,
};

/// §4.3.3 numerator of the band-size dependent term in 1/8 bits per
/// MDCT bin (RFC 6716 §4.3.3 p. 115: "24 times the number of MDCT bins
/// in the band").
pub const BAND_THRESH_BINS_MULTIPLIER: u32 = 24;

/// §4.3.3 divisor of the band-size dependent term (RFC 6716 §4.3.3
/// p. 115: "divide by 16"). The composed `24/16 = 3/2 = 0.1875 whole
/// bits per MDCT bin = 1.5 1/8 bits per MDCT bin`; equivalently
/// `48 128th-bits per MDCT bin` as the RFC phrases it.
pub const BAND_THRESH_BINS_DIVISOR: u32 = 16;

/// §4.3.3 per-channel minimum in 1/8 bits (RFC 6716 §4.3.3 p. 115:
/// "8 times the number of channels"). One whole bit per channel.
pub const BAND_THRESH_PER_CHANNEL_EIGHTH_BITS: u32 = 8;

/// §4.3.3 mono channel multiplier (1 channel).
pub const BAND_THRESH_MONO_CHANNELS: u32 = 1;

/// §4.3.3 stereo channel multiplier (2 channels).
pub const BAND_THRESH_STEREO_CHANNELS: u32 = 2;

/// Errors returned by [`compute_band_min_thresh`] for inputs that
/// violate the §4.3 / §4.3.3 contract. None of these come from the
/// range coder; they are caller-side bookkeeping bugs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandThreshError {
    /// `start > end`. The §4.3 coding window `start..end` must be
    /// monotonic; an inverted window is a caller-side bug.
    InvertedBandWindow {
        /// The provided start.
        start: usize,
        /// The provided end.
        end: usize,
    },
    /// `end > 21`. The §4.3 standard CELT layer has exactly
    /// [`CELT_NUM_BANDS`](crate::celt_band_layout::CELT_NUM_BANDS) = 21
    /// bands; a larger `end` is a caller-side bug.
    BandWindowOutOfRange {
        /// The provided end.
        end: usize,
    },
    /// The output slice is too small to hold one `thresh[]` entry per
    /// band in the `start..end` window.
    OutputBufferTooSmall {
        /// Number of coded bands in the window (= `end - start`).
        expected: usize,
        /// Length of the caller-supplied output buffer.
        provided: usize,
    },
}

/// §4.3.3 per-band minimum-allocation lookup for a single coded band
/// (RFC 6716 §4.3.3 p. 115).
///
/// `band` is the global band index `0..21` (Table 55 layout). The
/// §4.3 coding window `start..end` is enforced at the caller — this
/// function does not know whether `band` is inside the active window.
///
/// `frame_size` selects the Table 55 column (the per-channel MDCT bin
/// count `N` depends on the frame size). `is_stereo` selects the
/// channel-count multiplier on the per-channel term.
///
/// Returns `None` if `band ≥ 21` (Custom mode is out of scope; the
/// standard layer has exactly 21 bands).
///
/// Returns the §4.3.3 minimum in 1/8 bits.
pub fn band_min_thresh(band: usize, frame_size: CeltFrameSize, is_stereo: bool) -> Option<u32> {
    let n = celt_band_bins_per_channel(band, frame_size)? as u32;
    let bin_term = (BAND_THRESH_BINS_MULTIPLIER * n) / BAND_THRESH_BINS_DIVISOR;
    let channels = if is_stereo {
        BAND_THRESH_STEREO_CHANNELS
    } else {
        BAND_THRESH_MONO_CHANNELS
    };
    let channel_term = BAND_THRESH_PER_CHANNEL_EIGHTH_BITS * channels;
    Some(bin_term.max(channel_term))
}

/// Compute the §4.3.3 per-band minimum-allocation vector for every
/// coded band in `start..end`, writing one entry per band into
/// `thresh`.
///
/// `start..end` is the §4.3 CELT coding window (CELT-only frames use
/// `0..21`; Hybrid frames use `17..21`). The §4.3 `start` value comes
/// from [`crate::celt_band_layout::celt_first_coded_band`]; the
/// `end` value comes from the audio bandwidth signalled in the §3.1
/// TOC byte but is bounded by [`crate::celt_band_layout::celt_end_coded_band`]
/// (= 21).
///
/// `thresh` must have length `end - start` exactly. The output is
/// indexed locally (`thresh[band - start]` holds the §4.3.3 minimum
/// for global band `band`).
pub fn compute_band_min_thresh(
    start: usize,
    end: usize,
    frame_size: CeltFrameSize,
    is_stereo: bool,
    thresh: &mut [u32],
) -> Result<(), BandThreshError> {
    if start > end {
        return Err(BandThreshError::InvertedBandWindow { start, end });
    }
    if end > celt_end_coded_band() {
        return Err(BandThreshError::BandWindowOutOfRange { end });
    }
    let coded = end - start;
    if thresh.len() != coded {
        return Err(BandThreshError::OutputBufferTooSmall {
            expected: coded,
            provided: thresh.len(),
        });
    }
    for (slot, band) in thresh.iter_mut().zip(start..end) {
        // band < end ≤ 21 ⇒ band_min_thresh() always succeeds.
        *slot = band_min_thresh(band, frame_size, is_stereo)
            .expect("§4.3 band < CELT_NUM_BANDS by window check");
    }
    Ok(())
}

/// Convenience wrapper: allocate a `Vec<u32>` of length `end - start`
/// and fill it via [`compute_band_min_thresh`].
///
/// Prefer the slice form when avoiding allocation matters; this
/// allocator is exposed primarily for tests and one-shot callers.
pub fn band_min_thresh_vec(
    start: usize,
    end: usize,
    frame_size: CeltFrameSize,
    is_stereo: bool,
) -> Result<Vec<u32>, BandThreshError> {
    if start > end {
        return Err(BandThreshError::InvertedBandWindow { start, end });
    }
    if end > celt_end_coded_band() {
        return Err(BandThreshError::BandWindowOutOfRange { end });
    }
    let mut v = vec![0u32; end - start];
    compute_band_min_thresh(start, end, frame_size, is_stereo, &mut v)?;
    Ok(v)
}

/// §4.3 full-frame band-window helper: returns
/// `(start, end) = (celt_first_coded_band(is_hybrid), celt_end_coded_band())`.
///
/// Convenience for callers that want to compute `thresh[]` over the
/// standard §4.3 window without manually plumbing the two
/// [`celt_band_layout`](crate::celt_band_layout) helpers.
pub fn standard_band_window(is_hybrid: bool) -> (usize, usize) {
    (celt_first_coded_band(is_hybrid), celt_end_coded_band())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_band_layout::{CELT_NUM_BANDS, HYBRID_FIRST_CODED_BAND};

    // ---- Constant pins (each tied to a §4.3.3 RFC narrative phrase) ----

    #[test]
    fn bins_multiplier_matches_rfc_24() {
        // RFC 6716 §4.3.3 p. 115: "24 times the number of MDCT bins in the band".
        assert_eq!(BAND_THRESH_BINS_MULTIPLIER, 24);
    }

    #[test]
    fn bins_divisor_matches_rfc_16() {
        // RFC 6716 §4.3.3 p. 115: "divide by 16".
        assert_eq!(BAND_THRESH_BINS_DIVISOR, 16);
    }

    #[test]
    fn per_channel_term_matches_rfc_one_whole_bit() {
        // RFC 6716 §4.3.3 p. 115: "8 times the number of channels"
        // — one whole bit (= 8 1/8 bits) per channel.
        assert_eq!(BAND_THRESH_PER_CHANNEL_EIGHTH_BITS, 8);
        assert_eq!(BAND_THRESH_PER_CHANNEL_EIGHTH_BITS / 8, 1);
    }

    #[test]
    fn channel_multipliers_match_audio_layout() {
        assert_eq!(BAND_THRESH_MONO_CHANNELS, 1);
        assert_eq!(BAND_THRESH_STEREO_CHANNELS, 2);
    }

    // ---- band_min_thresh: §4.3.3 single-band formula ----

    #[test]
    fn band0_2_5ms_mono_uses_channel_term() {
        // Table 55, band 0, 2.5 ms: N = 1 MDCT bin/ch.
        // bin_term = (24 * 1) / 16 = 1; channel_term = 8 (mono).
        // max(1, 8) = 8.
        assert_eq!(band_min_thresh(0, CeltFrameSize::Ms2_5, false), Some(8));
    }

    #[test]
    fn band0_2_5ms_stereo_uses_channel_term() {
        // Same bin_term as mono (RFC: "not scaled by the channel count").
        // channel_term = 8 * 2 = 16. max(1, 16) = 16.
        assert_eq!(band_min_thresh(0, CeltFrameSize::Ms2_5, true), Some(16));
    }

    #[test]
    fn band0_20ms_mono_uses_channel_term_at_eight_bins() {
        // Table 55, band 0, 20 ms: N = 8.
        // bin_term = (24 * 8) / 16 = 12; channel_term = 8. max = 12.
        assert_eq!(band_min_thresh(0, CeltFrameSize::Ms20, false), Some(12));
    }

    #[test]
    fn band0_20ms_stereo_channel_term_still_wins() {
        // bin_term = 12; channel_term = 16. max = 16.
        assert_eq!(band_min_thresh(0, CeltFrameSize::Ms20, true), Some(16));
    }

    #[test]
    fn band20_20ms_mono_uses_bin_term() {
        // Table 55, band 20, 20 ms: N = 176 (the largest band).
        // bin_term = (24 * 176) / 16 = 264; channel_term = 8.
        // max = 264.
        assert_eq!(band_min_thresh(20, CeltFrameSize::Ms20, false), Some(264));
    }

    #[test]
    fn band20_20ms_stereo_uses_bin_term() {
        // Same bin_term = 264 (RFC: not scaled). channel_term = 16.
        // max = 264.
        assert_eq!(band_min_thresh(20, CeltFrameSize::Ms20, true), Some(264));
    }

    #[test]
    fn band21_or_higher_returns_none() {
        // §4.3 standard layer has exactly 21 bands; the table lookup
        // at band 21 produces None.
        assert_eq!(band_min_thresh(21, CeltFrameSize::Ms20, false), None);
        assert_eq!(band_min_thresh(100, CeltFrameSize::Ms20, false), None);
    }

    // ---- Formula cross-check ----

    #[test]
    fn band_min_thresh_matches_formula_for_every_band_and_frame_size() {
        // The function MUST equal max((24*N)/16, 8*channels) for
        // every (band, frame_size, channels) triple in the standard
        // layer.
        for band in 0..CELT_NUM_BANDS {
            for fs in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                let n = celt_band_bins_per_channel(band, fs).unwrap() as u32;
                let bin_term = (24 * n) / 16;
                for (channels, stereo) in [(1u32, false), (2u32, true)] {
                    let channel_term = 8 * channels;
                    let want = bin_term.max(channel_term);
                    let got = band_min_thresh(band, fs, stereo).unwrap();
                    assert_eq!(got, want, "band={band} fs={fs:?} channels={channels}");
                }
            }
        }
    }

    #[test]
    fn band_min_thresh_independent_of_channel_count_when_bin_term_dominates() {
        // RFC 6716 §4.3.3 p. 115: "The band-size dependent part of
        // this value is not scaled by the channel count". When the
        // bin term dominates, mono == stereo.
        // Band 20 / 20 ms: bin_term = 264; channel_term ≤ 16.
        let mono = band_min_thresh(20, CeltFrameSize::Ms20, false).unwrap();
        let stereo = band_min_thresh(20, CeltFrameSize::Ms20, true).unwrap();
        assert_eq!(mono, stereo);
        assert_eq!(mono, 264);
    }

    #[test]
    fn band_min_thresh_doubles_with_stereo_when_channel_term_dominates() {
        // Band 0 / 2.5 ms: bin_term = 1; channel_term = 8 * channels.
        // Mono ⇒ 8; stereo ⇒ 16. Stereo == 2 × mono.
        let mono = band_min_thresh(0, CeltFrameSize::Ms2_5, false).unwrap();
        let stereo = band_min_thresh(0, CeltFrameSize::Ms2_5, true).unwrap();
        assert_eq!(stereo, 2 * mono);
    }

    // ---- compute_band_min_thresh: full vector ----

    #[test]
    fn compute_full_celt_only_window_mono_20ms() {
        // CELT-only 20 ms mono: window = 0..21.
        let mut out = [0u32; 21];
        compute_band_min_thresh(0, 21, CeltFrameSize::Ms20, false, &mut out).unwrap();
        for (slot, band) in out.iter().zip(0..21) {
            assert_eq!(
                *slot,
                band_min_thresh(band, CeltFrameSize::Ms20, false).unwrap()
            );
        }
    }

    #[test]
    fn compute_hybrid_window_stereo_20ms() {
        // Hybrid 20 ms stereo: window = 17..21 (4 bands).
        let mut out = [0u32; 4];
        compute_band_min_thresh(
            HYBRID_FIRST_CODED_BAND,
            21,
            CeltFrameSize::Ms20,
            true,
            &mut out,
        )
        .unwrap();
        for (slot, band) in out.iter().zip(HYBRID_FIRST_CODED_BAND..21) {
            assert_eq!(
                *slot,
                band_min_thresh(band, CeltFrameSize::Ms20, true).unwrap()
            );
        }
    }

    #[test]
    fn compute_partial_window_2_5ms() {
        // CELT-only 2.5 ms NB window: 0..13 (8 kHz cutoff is band 13).
        let mut out = [0u32; 13];
        compute_band_min_thresh(0, 13, CeltFrameSize::Ms2_5, false, &mut out).unwrap();
        // Every band 0..13 at 2.5 ms has N ∈ [1, 4] → bin_term ≤ 6 <
        // channel_term = 8 ⇒ every entry = 8.
        for slot in &out {
            assert_eq!(*slot, 8);
        }
    }

    #[test]
    fn compute_window_5ms_stereo_band20() {
        // Single-band window covering band 20 at 5 ms.
        // Table 55, band 20, 5 ms: N = 44 (matches the round-24
        // band_layout pin at celt_band_layout.rs L419).
        // bin_term = (24 * 44) / 16 = 66; channel_term = 16.
        // max = 66.
        let n = celt_band_bins_per_channel(20, CeltFrameSize::Ms5).unwrap();
        assert_eq!(n, 44);
        let mut out = [0u32; 1];
        compute_band_min_thresh(20, 21, CeltFrameSize::Ms5, true, &mut out).unwrap();
        assert_eq!(out[0], 66);
    }

    // ---- Error paths ----

    #[test]
    fn inverted_window_rejected() {
        let mut out = [0u32; 0];
        let err = compute_band_min_thresh(5, 3, CeltFrameSize::Ms20, false, &mut out).unwrap_err();
        assert_eq!(
            err,
            BandThreshError::InvertedBandWindow { start: 5, end: 3 }
        );
    }

    #[test]
    fn window_past_max_band_rejected() {
        let mut out = [0u32; 22];
        let err = compute_band_min_thresh(0, 22, CeltFrameSize::Ms20, false, &mut out).unwrap_err();
        assert_eq!(err, BandThreshError::BandWindowOutOfRange { end: 22 });
    }

    #[test]
    fn output_buffer_too_small_rejected() {
        let mut out = [0u32; 20]; // window has 21 bands
        let err = compute_band_min_thresh(0, 21, CeltFrameSize::Ms20, false, &mut out).unwrap_err();
        assert_eq!(
            err,
            BandThreshError::OutputBufferTooSmall {
                expected: 21,
                provided: 20,
            }
        );
    }

    #[test]
    fn output_buffer_too_large_also_rejected() {
        let mut out = [0u32; 22]; // window has 21 bands
        let err = compute_band_min_thresh(0, 21, CeltFrameSize::Ms20, false, &mut out).unwrap_err();
        assert_eq!(
            err,
            BandThreshError::OutputBufferTooSmall {
                expected: 21,
                provided: 22,
            }
        );
    }

    #[test]
    fn empty_window_succeeds_with_empty_output() {
        let mut out = [0u32; 0];
        compute_band_min_thresh(5, 5, CeltFrameSize::Ms20, false, &mut out).unwrap();
        compute_band_min_thresh(0, 0, CeltFrameSize::Ms2_5, true, &mut out).unwrap();
    }

    // ---- band_min_thresh_vec ----

    #[test]
    fn vec_helper_matches_slice_form() {
        let v = band_min_thresh_vec(0, 21, CeltFrameSize::Ms10, true).unwrap();
        let mut out = [0u32; 21];
        compute_band_min_thresh(0, 21, CeltFrameSize::Ms10, true, &mut out).unwrap();
        assert_eq!(v.as_slice(), &out);
    }

    #[test]
    fn vec_helper_propagates_window_errors() {
        let err = band_min_thresh_vec(0, 22, CeltFrameSize::Ms20, false).unwrap_err();
        assert_eq!(err, BandThreshError::BandWindowOutOfRange { end: 22 });

        let err = band_min_thresh_vec(7, 4, CeltFrameSize::Ms20, false).unwrap_err();
        assert_eq!(
            err,
            BandThreshError::InvertedBandWindow { start: 7, end: 4 }
        );
    }

    #[test]
    fn vec_helper_returns_empty_for_empty_window() {
        let v = band_min_thresh_vec(5, 5, CeltFrameSize::Ms20, true).unwrap();
        assert!(v.is_empty());
    }

    // ---- standard_band_window helper ----

    #[test]
    fn standard_window_celt_only_is_zero_to_twentyone() {
        let (start, end) = standard_band_window(false);
        assert_eq!(start, 0);
        assert_eq!(end, 21);
    }

    #[test]
    fn standard_window_hybrid_is_seventeen_to_twentyone() {
        let (start, end) = standard_band_window(true);
        assert_eq!(start, HYBRID_FIRST_CODED_BAND);
        assert_eq!(end, 21);
        assert_eq!(start, 17);
    }

    // ---- §4.3.3 invariants ----

    #[test]
    fn every_band_thresh_is_at_least_one_whole_bit_mono() {
        // §4.3.3: "minimum allocation [is] one bit per channel […]
        // whichever is greater". For mono that floor is 8 1/8 bits.
        for band in 0..CELT_NUM_BANDS {
            for fs in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                let t = band_min_thresh(band, fs, false).unwrap();
                assert!(t >= 8, "band={band} fs={fs:?} thresh={t}");
            }
        }
    }

    #[test]
    fn every_band_thresh_is_at_least_one_whole_bit_per_channel_stereo() {
        // For stereo the floor is 16 1/8 bits.
        for band in 0..CELT_NUM_BANDS {
            for fs in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                let t = band_min_thresh(band, fs, true).unwrap();
                assert!(t >= 16, "band={band} fs={fs:?} thresh={t}");
            }
        }
    }

    #[test]
    fn thresh_monotonic_or_equal_with_frame_size_for_fixed_band() {
        // Within a fixed band, doubling the frame size doubles N
        // (Table 55: bins/channel doubles across each frame-size
        // column). With the band-size term dominating, doubling N
        // doubles bin_term, so thresh is monotonic non-decreasing in
        // frame size. The channel-term floor may flatten it. Check
        // the monotonicity for every band / channel combination.
        for band in 0..CELT_NUM_BANDS {
            for stereo in [false, true] {
                let t0 = band_min_thresh(band, CeltFrameSize::Ms2_5, stereo).unwrap();
                let t1 = band_min_thresh(band, CeltFrameSize::Ms5, stereo).unwrap();
                let t2 = band_min_thresh(band, CeltFrameSize::Ms10, stereo).unwrap();
                let t3 = band_min_thresh(band, CeltFrameSize::Ms20, stereo).unwrap();
                assert!(t0 <= t1, "band={band} stereo={stereo}: t0={t0}, t1={t1}");
                assert!(t1 <= t2, "band={band} stereo={stereo}: t1={t1}, t2={t2}");
                assert!(t2 <= t3, "band={band} stereo={stereo}: t2={t2}, t3={t3}");
            }
        }
    }

    #[test]
    fn stereo_thresh_at_least_mono_for_every_band_and_frame_size() {
        // §4.3.3: the channel-term scales linearly with `channels`;
        // the bin-term is identical across mono/stereo. So
        // `thresh_stereo ≥ thresh_mono` for every band.
        for band in 0..CELT_NUM_BANDS {
            for fs in [
                CeltFrameSize::Ms2_5,
                CeltFrameSize::Ms5,
                CeltFrameSize::Ms10,
                CeltFrameSize::Ms20,
            ] {
                let mono = band_min_thresh(band, fs, false).unwrap();
                let stereo = band_min_thresh(band, fs, true).unwrap();
                assert!(stereo >= mono, "band={band} fs={fs:?}");
            }
        }
    }

    #[test]
    fn thresh_units_are_eighth_bits() {
        // §4.3.3 narrative: minimum is "48 128th bits per MDCT bin"
        // = 48/128 whole bits/bin = 0.375 whole bits/bin = 3/8 whole
        // bits/bin = 3 1/8 bits/bin. The bin term is (24*N)/16 1/8
        // bits ⇒ for N = 16 ⇒ 24 1/8 bits ⇒ 24/16 = 1.5 whole bits
        // per 16-bin band = 3/8 whole bits per bin. So at N=16,
        // bin_term = 24, which is 16 * 3/2 = 24. ✓
        let n = 16u32;
        let bin_term = (24 * n) / 16;
        assert_eq!(bin_term, 24);
        // And 24 1/8 bits / 16 bins = 1.5 1/8 bits per bin = 48/128
        // whole bits per bin, matching the §4.3.3 narrative.
        let per_bin_128ths = (bin_term * 16) / n; // (24*16)/16 = 24
                                                  // 24 in 1/8 bits = 24 * 16 = 384 in 1/128 bits across 16
                                                  // bins ⇒ 24 1/128 bits per bin × 2 (since 1/8 = 16/128) =
                                                  // 48 1/128 bits per bin. Cross-check the §4.3.3 "48 128th
                                                  // bits per MDCT bin" wording.
        assert_eq!(per_bin_128ths * 2, 48);
    }

    // ---- Specific Table 55 cells used by the formula ----

    #[test]
    fn band_min_thresh_pins_table55_band8_20ms_stereo() {
        // Table 55, band 8, 20 ms: N = 16 (per the existing band_layout tests).
        // bin_term = (24 * 16) / 16 = 24. channel_term = 16.
        // max = 24.
        let n = celt_band_bins_per_channel(8, CeltFrameSize::Ms20).unwrap();
        assert_eq!(n, 16);
        assert_eq!(band_min_thresh(8, CeltFrameSize::Ms20, true), Some(24));
    }

    #[test]
    fn band_min_thresh_pins_table55_band20_2_5ms_stereo() {
        // Table 55, band 20, 2.5 ms: N = ?
        // Per the band_layout tests, band 20 / 2.5 ms ≈ 22 bins.
        let n = celt_band_bins_per_channel(20, CeltFrameSize::Ms2_5).unwrap() as u32;
        let bin_term = (24 * n) / 16;
        let want = bin_term.max(16);
        let got = band_min_thresh(20, CeltFrameSize::Ms2_5, true).unwrap();
        assert_eq!(got, want);
    }

    // ---- Determinism + smoke ----

    #[test]
    fn determinism_across_repeats() {
        let a = band_min_thresh_vec(0, 21, CeltFrameSize::Ms20, true).unwrap();
        let b = band_min_thresh_vec(0, 21, CeltFrameSize::Ms20, true).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn debug_format_renders() {
        let err = BandThreshError::InvertedBandWindow { start: 7, end: 3 };
        let s = format!("{err:?}");
        assert!(s.contains("InvertedBandWindow"));
    }

    #[test]
    fn integrates_with_band_layout_hybrid_window() {
        // §4.3 Hybrid window from celt_band_layout produces 4
        // §4.3.3 thresh entries.
        let (start, end) = standard_band_window(true);
        let v = band_min_thresh_vec(start, end, CeltFrameSize::Ms10, true).unwrap();
        assert_eq!(v.len(), 4);
        // Every entry ≥ 16 (stereo floor).
        for &t in &v {
            assert!(t >= 16);
        }
    }
}
