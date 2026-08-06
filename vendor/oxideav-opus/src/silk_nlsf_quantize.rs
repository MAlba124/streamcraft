//! Normalized-LSF quantisation for the SILK encoder — RFC 6716
//! §5.2.3.5 (analysis) against the normative §4.2.7.5 reconstruction.
//!
//! Given the analysis NLSF vector (from
//! [`crate::silk_lpc_to_nlsf::lpc_to_nlsf_q15`]), pick the stage-1
//! codebook index `I1` and stage-2 residual indices `I2[]` whose
//! DECODED vector lies closest to the target. §5.2.3.5 describes the
//! reference strategy (N-best stage-1 shortlist + Viterbi delayed
//! decision in stage 2); the exact search is an encoder freedom, and
//! this implementation runs an exhaustive analysis-by-synthesis over
//! all 32 stage-1 candidates:
//!
//!  1. For each `I1`, scale the residual `target - cb1_Q8<<7` by the
//!     codebook's IHMW weights (the §4.2.7.5.3 `w_Q9[]` — exactly
//!     what the decoder will divide by), giving the stage-2 residual
//!     target in Q10.
//!  2. Quantize it with the greedy backwards walk of
//!     [`LsfStage2::quantize`] (the §4.2.7.5.2 inverse).
//!  3. Run the candidate through the REAL decode-side reconstruction
//!     ([`NlsfReconstructed`] + [`NlsfStabilized`]) and score the
//!     IHMW-weighted squared error against the target, so the winner
//!     is judged on what the decoder will actually produce.
//!
//! The winner's `(I1, I2[])` plug straight into
//! [`crate::silk_decode::SilkFrameSymbols::lsf_stage1`] /
//! [`SilkFrameSymbols::lsf_stage2_i2`](crate::silk_decode::SilkFrameSymbols::lsf_stage2_i2),
//! and the returned stabilized vector is the `n2_Q15[]` the decoder
//! will carry into §4.2.7.5.5 interpolation and §4.2.7.5.6 LPC
//! reconstruction.
//!
//! All truth is taken from RFC 6716 §4.2.7.5 / §5.2.3.5. No external
//! library source is consulted.

use crate::silk_lsf_recon::{cb1_q8, ihmw_w_q9_for, NlsfReconstructed};
use crate::silk_lsf_stabilize::NlsfStabilized;
use crate::silk_lsf_stage2::{LsfStage2, D_LPC_MAX};
use crate::toc::Bandwidth;
use crate::Error;

/// Result of the §5.2.3.5 NLSF quantisation: the wire indices plus
/// the exact decoder-side reconstruction they produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NlsfQuantized {
    /// §4.2.7.5.1 stage-1 index `I1 ∈ 0..32`.
    pub lsf_stage1: u8,
    /// §4.2.7.5.2 stage-2 indices `I2[k] ∈ [-10, 10]`; `0..d_lpc`
    /// entries are valid.
    pub i2: [i8; D_LPC_MAX],
    /// `d_LPC`: 10 for NB / MB, 16 for WB.
    pub d_lpc: usize,
    /// The stabilized decoder reconstruction `n2_Q15[]` (what
    /// §4.2.7.5.4 hands to interpolation / LPC conversion); `0..d_lpc`
    /// entries are valid.
    pub nlsf_q15: [i16; D_LPC_MAX],
    /// IHMW-weighted squared error of the reconstruction against the
    /// target (the search metric; useful for encoder diagnostics).
    pub weighted_err: u64,
}

impl NlsfQuantized {
    /// The valid prefix of [`Self::i2`].
    pub fn i2(&self) -> &[i8] {
        &self.i2[..self.d_lpc]
    }

    /// The valid prefix of [`Self::nlsf_q15`].
    pub fn nlsf_q15(&self) -> &[i16] {
        &self.nlsf_q15[..self.d_lpc]
    }
}

/// Quantize an analysis NLSF vector (sorted, Q15) to the §4.2.7.5
/// wire indices by exhaustive stage-1 analysis-by-synthesis.
///
/// `target_nlsf_q15.len()` must equal the bandwidth's `d_LPC` (10 for
/// NB / MB, 16 for WB); SWB / FB are rejected.
pub fn quantize_nlsf(
    bandwidth: Bandwidth,
    target_nlsf_q15: &[i16],
) -> Result<NlsfQuantized, Error> {
    // Validate length once via the I1 = 0 codebook row.
    let d_lpc = cb1_q8(bandwidth, 0)?.len();
    if target_nlsf_q15.len() != d_lpc {
        return Err(Error::MalformedPacket);
    }

    let mut best: Option<NlsfQuantized> = None;
    for i1 in 0..32u8 {
        let cb = cb1_q8(bandwidth, i1)?;
        let w_q9 = ihmw_w_q9_for(bandwidth, i1)?;

        // Stage-2 residual target in Q10: the decoder computes
        // NLSF[k] = (cb1[k]<<7) + (res_Q10[k]<<14)/w_Q9[k], so the
        // residual that lands on the target is
        // res_Q10[k] = (target - cb1<<7) * w_Q9[k] / 2^14 (rounded).
        let mut res_target = [0i32; D_LPC_MAX];
        for k in 0..d_lpc {
            let diff = target_nlsf_q15[k] as i64 - ((cb[k] as i64) << 7);
            let scaled = diff * w_q9[k] as i64;
            // Round-to-nearest with floor shift on signed values.
            res_target[k] = ((scaled + (1 << 13)) >> 14) as i32;
        }

        let stage2 = LsfStage2::quantize(bandwidth, i1, &res_target[..d_lpc])?;
        let recon = NlsfReconstructed::from_stage1_and_stage2(bandwidth, i1, &stage2)?;
        let stab = NlsfStabilized::from_reconstructed(bandwidth, &recon)?;

        let mut err: u64 = 0;
        for (k, (&r, &t)) in stab
            .nlsf_q15()
            .iter()
            .zip(target_nlsf_q15.iter())
            .enumerate()
        {
            let d = r as i64 - t as i64;
            err += (d * d) as u64 * w_q9[k] as u64;
        }

        let better = match &best {
            None => true,
            Some(b) => err < b.weighted_err,
        };
        if better {
            let mut i2 = [0i8; D_LPC_MAX];
            i2[..d_lpc].copy_from_slice(stage2.i2());
            let mut nlsf = [0i16; D_LPC_MAX];
            nlsf[..d_lpc].copy_from_slice(stab.nlsf_q15());
            best = Some(NlsfQuantized {
                lsf_stage1: i1,
                i2,
                d_lpc,
                nlsf_q15: nlsf,
                weighted_err: err,
            });
        }
    }

    // 32 candidates always yield a best.
    best.ok_or(Error::MalformedPacket)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;
    use crate::range_encoder::RangeEncoder;

    /// Quantizing an exact stage-1 codebook vector must reconstruct
    /// (almost) itself: the only error left is the §4.2.7.5.4
    /// stabilizer's minimum-distance nudging.
    #[test]
    fn codebook_vectors_quantize_to_themselves() {
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            for i1 in 0..32u8 {
                let cb = cb1_q8(bw, i1).unwrap();
                let target: Vec<i16> = cb.iter().map(|&v| (v as i16) << 7).collect();
                let q = quantize_nlsf(bw, &target).unwrap();
                for (k, (&r, &t)) in q.nlsf_q15().iter().zip(target.iter()).enumerate() {
                    assert!(
                        (r as i32 - t as i32).abs() <= 16,
                        "bw={bw:?} I1={i1} k={k}: {r} vs {t}"
                    );
                }
            }
        }
    }

    /// Perturbed targets stay within a stage-2 quantisation step of
    /// the reconstruction, and the result is strictly sorted.
    #[test]
    fn perturbed_targets_stay_close_and_sorted() {
        for bw in [Bandwidth::Nb, Bandwidth::Wb] {
            let cb = cb1_q8(bw, 9).unwrap();
            let mut target: Vec<i16> = cb.iter().map(|&v| (v as i16) << 7).collect();
            for (k, t) in target.iter_mut().enumerate() {
                // ±≈400 Q15 zig-zag perturbation.
                let sign = if k % 2 == 0 { 1 } else { -1 };
                *t = (*t as i32 + sign * 400).clamp(0, 32767) as i16;
            }
            let q = quantize_nlsf(bw, &target).unwrap();
            for (k, (&r, &t)) in q.nlsf_q15().iter().zip(target.iter()).enumerate() {
                assert!(
                    (r as i32 - t as i32).abs() <= 900,
                    "bw={bw:?} k={k}: {r} vs {t}"
                );
            }
            for w in q.nlsf_q15().windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    /// The chosen indices survive the real §4.2.7.5.2 wire coding: a
    /// range-coder roundtrip reproduces the same stage-2 indices and
    /// the same reconstruction.
    #[test]
    fn indices_roundtrip_the_wire() {
        let bw = Bandwidth::Wb;
        let cb = cb1_q8(bw, 17).unwrap();
        let target: Vec<i16> = cb
            .iter()
            .enumerate()
            .map(|(k, &v)| (((v as i32) << 7) + (k as i32 * 37 % 300) - 150).clamp(0, 32767) as i16)
            .collect();
        let q = quantize_nlsf(bw, &target).unwrap();

        let mut re = RangeEncoder::new();
        let enc_stage2 = LsfStage2::encode(&mut re, bw, q.lsf_stage1, q.i2()).unwrap();
        assert_eq!(enc_stage2.i2(), q.i2());
        let bytes = re.finish();
        let mut rd = RangeDecoder::new(&bytes);
        let dec_stage2 = LsfStage2::decode(&mut rd, bw, q.lsf_stage1).unwrap();
        assert_eq!(dec_stage2.i2(), q.i2());
        assert_eq!(dec_stage2.res_q10(), enc_stage2.res_q10());

        let recon =
            NlsfReconstructed::from_stage1_and_stage2(bw, q.lsf_stage1, &dec_stage2).unwrap();
        let stab = NlsfStabilized::from_reconstructed(bw, &recon).unwrap();
        assert_eq!(stab.nlsf_q15(), q.nlsf_q15());
    }

    /// `from_indices` matches the bitstream paths bit for bit.
    #[test]
    fn from_indices_matches_wire_paths() {
        let bw = Bandwidth::Nb;
        let i2: Vec<i8> = vec![3, -2, 0, 10, -10, 1, -1, 4, -4, 2];
        let direct = LsfStage2::from_indices(bw, 5, &i2).unwrap();
        let mut re = RangeEncoder::new();
        let encoded = LsfStage2::encode(&mut re, bw, 5, &i2).unwrap();
        assert_eq!(direct, encoded);
    }

    /// The exhaustive search actually helps: the winner's weighted
    /// error is a global minimum over I1 (spot-check against every
    /// candidate re-scored independently).
    #[test]
    fn winner_is_global_minimum_over_stage1() {
        let bw = Bandwidth::Nb;
        let target: Vec<i16> = (0..10)
            .map(|k| (2000 + k * 2900) as i16) // synthetic, sorted
            .collect();
        let q = quantize_nlsf(bw, &target).unwrap();
        for i1 in 0..32u8 {
            let cb = cb1_q8(bw, i1).unwrap();
            let w_q9 = ihmw_w_q9_for(bw, i1).unwrap();
            let mut res_target = [0i32; D_LPC_MAX];
            for k in 0..10 {
                let diff = target[k] as i64 - ((cb[k] as i64) << 7);
                res_target[k] = ((diff * w_q9[k] as i64 + (1 << 13)) >> 14) as i32;
            }
            let stage2 = LsfStage2::quantize(bw, i1, &res_target[..10]).unwrap();
            let recon = NlsfReconstructed::from_stage1_and_stage2(bw, i1, &stage2).unwrap();
            let stab = NlsfStabilized::from_reconstructed(bw, &recon).unwrap();
            let mut err: u64 = 0;
            for (k, (&r, &t)) in stab.nlsf_q15().iter().zip(target.iter()).enumerate() {
                let d = r as i64 - t as i64;
                err += (d * d) as u64 * w_q9[k] as u64;
            }
            assert!(q.weighted_err <= err, "I1={i1} beat the winner");
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert!(quantize_nlsf(Bandwidth::Swb, &[0i16; 10]).is_err());
        assert!(quantize_nlsf(Bandwidth::Nb, &[0i16; 16]).is_err());
        assert!(quantize_nlsf(Bandwidth::Wb, &[0i16; 10]).is_err());
    }
}
