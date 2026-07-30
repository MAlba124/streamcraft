//! SILK Normalized LSF interpolation — RFC 6716 §4.2.7.5.5.
//!
//! For a **20 ms** SILK frame, the first half of the frame (the first two
//! subframes) may use normalized LSF coefficients interpolated between the
//! coefficients decoded for the most recent coded frame in the same channel
//! (`n0_Q15[]`) and the ones decoded for the current frame (`n2_Q15[]`,
//! i.e. the §4.2.7.5.4-stabilized output). A Q2 interpolation factor
//! `w_Q2 ∈ 0..=4` follows the LSF coefficient indices in the bitstream and
//! is decoded with the Table 26 PDF. The interpolated first-half vector is
//!
//! ```text
//!     n1_Q15[k] = n0_Q15[k] + (w_Q2 * (n2_Q15[k] - n0_Q15[k]) >> 2)
//! ```
//!
//! Three behaviours from §4.2.7.5.5 are modelled exactly:
//!
//!  * **20 ms, normal.** Decode `w_Q2` from Table 26 and interpolate.
//!  * **20 ms, after an uncoded regular side-channel SILK frame or a
//!    decoder reset (§4.5.2).** The factor is **still decoded** (so the
//!    range coder stays in sync) but its value is ignored and `4` is used
//!    instead — i.e. `n1_Q15[] == n2_Q15[]`, no interpolation.
//!  * **10 ms.** The factor is not present in the bitstream at all, so it
//!    is neither decoded nor stored, and there is no first-half vector.
//!
//! The second half of a 20 ms frame (and the whole of a 10 ms frame)
//! always uses `n2_Q15[]` directly; that is the caller's responsibility —
//! this module only produces the interpolated first-half `n1_Q15[]` and
//! the decoded factor.

use crate::range_decoder::RangeDecoder;
use crate::silk_lsf_stabilize::NlsfStabilized;
use crate::silk_lsf_stage2::D_LPC_MAX;
use crate::Error;

// =====================================================================
// Table 26 — PDF for the Normalized LSF Interpolation Index.
//
// RFC 6716 §4.2.7.5.5: {13, 22, 29, 11, 181}/256 over the five possible
// Q2 factors w_Q2 ∈ {0, 1, 2, 3, 4}.
//
// The §4.1.3.3 `dec_icdf` primitive consumes the inverse-CDF form
// `icdf[k] = 256 - (fl[0] + .. + fh[k])`, terminated by 0:
//   cumulative {13, 35, 64, 75, 256}
//   icdf       {243, 221, 192, 181, 0}
// =====================================================================
const LSF_INTERP_ICDF: &[u8] = &[243, 221, 192, 181, 0];

/// The fixed factor used after a decoder reset or an uncoded regular
/// side-channel SILK frame: the §4.2.7.5.5 procedure decodes (and
/// discards) the bitstream factor and substitutes `4`, which makes
/// `n1_Q15[] == n2_Q15[]`.
const W_Q2_RESET: u8 = 4;

/// Whether the §4.2.7.5.5 interpolation factor is read from the bitstream
/// and, if so, whether its decoded value is honoured or forced to 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsfInterpContext {
    /// A normal 20 ms SILK frame: decode `w_Q2` from Table 26 and use it.
    TwentyMs,
    /// A 20 ms SILK frame immediately after an uncoded regular
    /// side-channel SILK frame or a decoder reset (§4.5.2). The factor is
    /// still decoded to keep the range coder in sync, but its value is
    /// discarded and `4` is used instead.
    TwentyMsAfterResetOrUncoded,
    /// A 10 ms SILK frame: the factor is not present in the bitstream, so
    /// nothing is decoded and there is no interpolated first-half vector.
    TenMs,
}

/// The §4.2.7.5.5 normalized-LSF interpolation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LsfInterpolated {
    /// The decoded Q2 interpolation factor, `0..=4`.
    ///
    /// `None` for a 10 ms frame ([`LsfInterpContext::TenMs`]), where no
    /// factor is present in the bitstream.
    w_q2: Option<u8>,
    len: u8,
    /// The interpolated first-half coefficients `n1_Q15[k]`.
    ///
    /// Only `0..len` entries are populated, and only when a first-half
    /// vector exists (20 ms frames). For 10 ms frames the array is unused.
    n1_q15: [i16; D_LPC_MAX],
    /// `true` when a first-half `n1_Q15[]` vector exists (20 ms frames).
    has_first_half: bool,
}

impl LsfInterpolated {
    /// Run the §4.2.7.5.5 interpolation against the range decoder.
    ///
    /// * `n2` is the current frame's stabilized NLSF vector (§4.2.7.5.4
    ///   output).
    /// * `n0_q15` is the most recent coded frame's NLSF vector in the same
    ///   channel, or `None` if there is no history (which forces the
    ///   reset behaviour regardless of `context`).
    /// * `context` selects whether the factor is present and whether its
    ///   decoded value is honoured.
    ///
    /// For [`LsfInterpContext::TenMs`] no factor is read and the result
    /// carries no first-half vector. For the two 20 ms contexts the factor
    /// is read from Table 26; in
    /// [`LsfInterpContext::TwentyMsAfterResetOrUncoded`] (or whenever
    /// `n0_q15` is `None`) the decoded value is discarded and `4` is used,
    /// yielding `n1_Q15[] == n2_Q15[]`.
    ///
    /// Returns `Error::MalformedPacket` if `n0_q15` is provided but its
    /// length does not match `n2`'s.
    pub fn decode(
        rd: &mut RangeDecoder,
        n2: &NlsfStabilized,
        n0_q15: Option<&[i16]>,
        context: LsfInterpContext,
    ) -> Result<Self, Error> {
        let d_lpc = n2.len();

        if let Some(n0) = n0_q15 {
            if n0.len() != d_lpc {
                return Err(Error::MalformedPacket);
            }
        }

        // 10 ms frames: the factor is not stored at all, and there is no
        // interpolated first-half vector.
        if context == LsfInterpContext::TenMs {
            return Ok(Self {
                w_q2: None,
                len: d_lpc as u8,
                n1_q15: [0i16; D_LPC_MAX],
                has_first_half: false,
            });
        }

        // 20 ms frames: the factor is always read from the bitstream so
        // the range coder stays in sync.
        let decoded_w_q2 = rd.dec_icdf(LSF_INTERP_ICDF, 8) as u8;
        Ok(Self::from_decoded_index(decoded_w_q2, n2, n0_q15, context))
    }

    /// The non-bitstream tail of the §4.2.7.5.5 interpolation for a
    /// 20 ms frame: apply the already-known Table 26 factor
    /// `decoded_w_q2` (as [`Self::decode`] would after reading it) and
    /// compute the first-half vector. Shared by the decode path and
    /// the encode-side frame composition, which knows the factor it is
    /// writing.
    pub fn from_decoded_index(
        decoded_w_q2: u8,
        n2: &NlsfStabilized,
        n0_q15: Option<&[i16]>,
        context: LsfInterpContext,
    ) -> Self {
        let d_lpc = n2.len();
        let n2_q15 = n2.nlsf_q15();

        // After a reset / uncoded side-channel frame, or when there is no
        // prior-frame history at all, the decoded value is discarded and 4
        // is used — making n1_Q15 == n2_Q15.
        let use_reset =
            matches!(context, LsfInterpContext::TwentyMsAfterResetOrUncoded) || n0_q15.is_none();
        let effective_w_q2 = if use_reset { W_Q2_RESET } else { decoded_w_q2 };

        // Compute n1_Q15[k] = n0[k] + (w_Q2 * (n2[k] - n0[k]) >> 2).
        // When n0 is absent (no history), effective_w_q2 == 4, so the
        // formula collapses to n1 == n2; use n2 itself as the base so the
        // identity is exact without needing an n0 array.
        let mut n1_q15 = [0i16; D_LPC_MAX];
        match n0_q15 {
            Some(n0) => {
                for k in 0..d_lpc {
                    let n0k = n0[k] as i32;
                    let n2k = n2_q15[k] as i32;
                    let interp = n0k + ((effective_w_q2 as i32 * (n2k - n0k)) >> 2);
                    n1_q15[k] = interp as i16;
                }
            }
            None => {
                // No history: n1 == n2 (effective factor is 4).
                n1_q15[..d_lpc].copy_from_slice(&n2_q15[..d_lpc]);
            }
        }

        Self {
            w_q2: Some(decoded_w_q2),
            len: d_lpc as u8,
            n1_q15,
            has_first_half: true,
        }
    }

    /// Encode the §4.2.7.5.5 interpolation factor into `re` — the
    /// write-side mirror of the bitstream part of [`Self::decode`].
    ///
    /// For the two 20 ms contexts `w_q2` must be `Some(0..=4)` and is
    /// written from the Table 26 PDF (even in the
    /// [`LsfInterpContext::TwentyMsAfterResetOrUncoded`] context, where
    /// the decoder discards the value and substitutes 4 — the symbol
    /// still occupies the bitstream to keep the range coder in sync).
    /// For [`LsfInterpContext::TenMs`] the factor is absent and `w_q2`
    /// must be `None`.
    pub fn encode_index(
        re: &mut crate::range_encoder::RangeEncoder,
        context: LsfInterpContext,
        w_q2: Option<u8>,
    ) -> Result<(), Error> {
        match (context, w_q2) {
            (LsfInterpContext::TenMs, None) => Ok(()),
            (
                LsfInterpContext::TwentyMs | LsfInterpContext::TwentyMsAfterResetOrUncoded,
                Some(w),
            ) if w <= 4 => {
                re.enc_icdf(w as usize, LSF_INTERP_ICDF, 8);
                Ok(())
            }
            _ => Err(Error::MalformedPacket),
        }
    }

    /// The decoded Q2 interpolation factor in `0..=4`, or `None` for a
    /// 10 ms frame where no factor is present.
    ///
    /// Note that this is the **decoded** value as it appears in the
    /// bitstream; for [`LsfInterpContext::TwentyMsAfterResetOrUncoded`]
    /// the value applied to the interpolation is 4 regardless of what was
    /// decoded (see [`LsfInterpolated::n1_q15`]).
    pub fn w_q2(&self) -> Option<u8> {
        self.w_q2
    }

    /// The interpolated first-half coefficients `n1_Q15[k]`, or `None`
    /// when there is no first-half vector (10 ms frames).
    pub fn n1_q15(&self) -> Option<&[i16]> {
        if self.has_first_half {
            Some(&self.n1_q15[..self.len as usize])
        } else {
            None
        }
    }

    /// Number of populated coefficients (10 for NB / MB, 16 for WB).
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` if there are no coefficients (never happens after a
    /// successful decode of a valid frame).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk_lsf_recon::NlsfReconstructed;
    use crate::silk_lsf_stage2::LsfStage2;
    use crate::toc::Bandwidth;

    // --- Table 26 transcription --------------------------------------

    #[test]
    fn table26_pdf_sums_to_256() {
        let pdf = [13u32, 22, 29, 11, 181];
        assert_eq!(pdf.iter().sum::<u32>(), 256);
        // icdf[k] = 256 - cumsum(pdf[0..=k]); terminated by 0.
        let mut cum = 0u32;
        let mut expected = Vec::new();
        for &p in &pdf {
            cum += p;
            expected.push((256 - cum) as u8);
        }
        assert_eq!(LSF_INTERP_ICDF, expected.as_slice());
        assert_eq!(*LSF_INTERP_ICDF.last().unwrap(), 0);
    }

    #[test]
    fn table26_icdf_monotone_decreasing() {
        for w in LSF_INTERP_ICDF.windows(2) {
            assert!(
                w[0] > w[1],
                "icdf not strictly decreasing: {LSF_INTERP_ICDF:?}"
            );
        }
        // Exactly five possible factors (w_Q2 ∈ {0,1,2,3,4}).
        assert_eq!(LSF_INTERP_ICDF.len(), 5);
    }

    // --- Helpers ------------------------------------------------------

    /// Build a stabilized NLSF vector for a bandwidth / I1 off a synthetic
    /// range-decoder buffer.
    fn stabilized(bandwidth: Bandwidth, i1: u8, buf: &[u8]) -> NlsfStabilized {
        let mut rd = RangeDecoder::new(buf);
        let stage2 = LsfStage2::decode(&mut rd, bandwidth, i1).expect("stage-2");
        let recon =
            NlsfReconstructed::from_stage1_and_stage2(bandwidth, i1, &stage2).expect("recon");
        NlsfStabilized::from_reconstructed(bandwidth, &recon).expect("stabilize")
    }

    /// Re-derive the §4.2.7.5.5 formula directly for the assertion side.
    fn interp_formula(n0: &[i16], n2: &[i16], w_q2: u8) -> Vec<i16> {
        n0.iter()
            .zip(n2.iter())
            .map(|(&a, &b)| {
                let a = a as i32;
                let b = b as i32;
                (a + ((w_q2 as i32 * (b - a)) >> 2)) as i16
            })
            .collect()
    }

    // --- 10 ms: factor not present -----------------------------------

    #[test]
    fn ten_ms_reads_nothing_and_has_no_first_half() {
        let buf = [0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6];
        let n2 = stabilized(Bandwidth::Nb, 3, &buf);
        let mut rd = RangeDecoder::new(&buf);
        let tell_before = rd.tell();
        let interp = LsfInterpolated::decode(&mut rd, &n2, None, LsfInterpContext::TenMs).unwrap();
        // No bits consumed: the factor is not present in 10 ms frames.
        assert_eq!(rd.tell(), tell_before);
        assert_eq!(interp.w_q2(), None);
        assert!(interp.n1_q15().is_none());
        assert_eq!(interp.len(), n2.len());
    }

    // --- 20 ms normal: interpolation ---------------------------------

    #[test]
    fn twenty_ms_interpolates_between_n0_and_n2() {
        let buf2 = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA];
        let buf0 = [0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6];
        let n2 = stabilized(Bandwidth::Nb, 5, &buf2);
        let n0 = stabilized(Bandwidth::Nb, 5, &buf0);
        let n0_vec: Vec<i16> = n0.nlsf_q15().to_vec();

        // Drive the factor decode off a buffer whose first iCDF read lands
        // on a known value; assert the formula end-to-end.
        let factor_buf = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut rd = RangeDecoder::new(&factor_buf);
        let interp =
            LsfInterpolated::decode(&mut rd, &n2, Some(&n0_vec), LsfInterpContext::TwentyMs)
                .unwrap();

        let w = interp.w_q2().expect("20 ms has a factor");
        assert!(w <= 4, "w_Q2 out of range: {w}");
        let expected = interp_formula(&n0_vec, n2.nlsf_q15(), w);
        assert_eq!(interp.n1_q15().unwrap(), expected.as_slice());
    }

    #[test]
    fn twenty_ms_factor_zero_yields_n0() {
        // w_Q2 == 0 → n1 == n0 exactly. Force it by hand via the formula
        // path: pick a buffer that decodes the factor to some value, but
        // verify the identity for the w==0 algebraic case using the
        // formula helper (independent of which value the buffer decodes).
        let buf2 = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA];
        let buf0 = [0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6];
        let n2 = stabilized(Bandwidth::Wb, 0, &buf2);
        let n0 = stabilized(Bandwidth::Wb, 0, &buf0);
        let n0_vec: Vec<i16> = n0.nlsf_q15().to_vec();
        let zero = interp_formula(&n0_vec, n2.nlsf_q15(), 0);
        assert_eq!(zero, n0_vec, "w_Q2 == 0 must reproduce n0 exactly");
    }

    #[test]
    fn twenty_ms_factor_four_yields_n2() {
        // w_Q2 == 4 → n1 == n2 exactly (the >>2 cancels the *4).
        let buf2 = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA];
        let buf0 = [0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6];
        let n2 = stabilized(Bandwidth::Wb, 0, &buf2);
        let n0 = stabilized(Bandwidth::Wb, 0, &buf0);
        let n0_vec: Vec<i16> = n0.nlsf_q15().to_vec();
        let four = interp_formula(&n0_vec, n2.nlsf_q15(), 4);
        assert_eq!(four, n2.nlsf_q15(), "w_Q2 == 4 must reproduce n2 exactly");
    }

    // --- 20 ms after reset / uncoded side channel --------------------

    #[test]
    fn after_reset_decodes_factor_but_uses_four() {
        let buf2 = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA];
        let buf0 = [0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6];
        let n2 = stabilized(Bandwidth::Nb, 7, &buf2);
        let n0 = stabilized(Bandwidth::Nb, 7, &buf0);
        let n0_vec: Vec<i16> = n0.nlsf_q15().to_vec();

        let factor_buf = [0x80u8, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
        // The factor IS read (range coder advances), but its value is
        // ignored and 4 is applied → n1 == n2.
        let mut rd_reset = RangeDecoder::new(&factor_buf);
        let tell_before = rd_reset.tell();
        let interp = LsfInterpolated::decode(
            &mut rd_reset,
            &n2,
            Some(&n0_vec),
            LsfInterpContext::TwentyMsAfterResetOrUncoded,
        )
        .unwrap();
        assert!(
            rd_reset.tell() > tell_before,
            "factor must still be decoded"
        );
        assert_eq!(
            interp.n1_q15().unwrap(),
            n2.nlsf_q15(),
            "reset context must force w_Q2 = 4 → n1 == n2"
        );

        // And the byte position matches a plain decode of the same buffer
        // in the normal context (same number of bits consumed).
        let mut rd_normal = RangeDecoder::new(&factor_buf);
        let _ = LsfInterpolated::decode(
            &mut rd_normal,
            &n2,
            Some(&n0_vec),
            LsfInterpContext::TwentyMs,
        )
        .unwrap();
        assert_eq!(rd_reset.tell(), rd_normal.tell());
    }

    #[test]
    fn no_history_forces_n2_even_in_normal_context() {
        let buf2 = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA];
        let n2 = stabilized(Bandwidth::Wb, 9, &buf2);
        let factor_buf = [0x33u8, 0x66, 0x99, 0xCC, 0xFF, 0x00, 0x55, 0xAA];
        let mut rd = RangeDecoder::new(&factor_buf);
        // n0 == None: even in the normal 20 ms context, the absence of
        // history forces the factor to 4 → n1 == n2.
        let interp =
            LsfInterpolated::decode(&mut rd, &n2, None, LsfInterpContext::TwentyMs).unwrap();
        assert_eq!(interp.n1_q15().unwrap(), n2.nlsf_q15());
        // The factor was still decoded (range coder advanced).
        assert!(interp.w_q2().is_some());
    }

    // --- Length-mismatch rejection -----------------------------------

    #[test]
    fn rejects_n0_length_mismatch() {
        let buf = [0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6];
        let n2_nb = stabilized(Bandwidth::Nb, 0, &buf); // d_LPC = 10
        let wb_buf = [
            0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6, 0x4C, 0x8E,
        ];
        let n0_wb = stabilized(Bandwidth::Wb, 0, &wb_buf); // d_LPC = 16
        let n0_vec: Vec<i16> = n0_wb.nlsf_q15().to_vec();
        let mut rd = RangeDecoder::new(&buf);
        assert!(LsfInterpolated::decode(
            &mut rd,
            &n2_nb,
            Some(&n0_vec),
            LsfInterpContext::TwentyMs
        )
        .is_err());
    }

    // --- n1 stays in range; intermediate values bounded --------------

    #[test]
    fn interpolated_values_stay_in_q15_range() {
        // For every factor 0..=4 the interpolation is a convex combination
        // of two values already in [0, 32767], so the result stays in
        // [0, 32767]. Sweep a handful of (n0, n2) pairs.
        let buf2 = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA];
        let buf0 = [0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6];
        for bandwidth in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            let extra = [0x4C, 0x8E];
            let mut b2 = buf2.to_vec();
            let mut b0 = buf0.to_vec();
            if bandwidth == Bandwidth::Wb {
                b2.extend_from_slice(&extra);
                b0.extend_from_slice(&extra);
            }
            for i1 in 0u8..32 {
                let n2 = stabilized(bandwidth, i1, &b2);
                let n0 = stabilized(bandwidth, i1, &b0);
                let n0_vec: Vec<i16> = n0.nlsf_q15().to_vec();
                for w in 0u8..=4 {
                    let res = interp_formula(&n0_vec, n2.nlsf_q15(), w);
                    for &v in &res {
                        assert!(
                            (0..=i16::MAX).contains(&v),
                            "n1 out of range: bw={bandwidth:?} i1={i1} w={w} v={v}"
                        );
                    }
                }
            }
        }
    }
}
