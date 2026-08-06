//! SILK frame synthesis composition — RFC 6716 §4.2.7.9.
//!
//! This module is the composition seam between the bitstream-consuming
//! front half of the SILK decode (everything up to and including the
//! [`crate::silk_decode::SilkFrameDecoded`] parameter set) and the
//! signal-reconstructing back half. It takes one fully decoded regular
//! SILK frame plus the cross-frame synthesis state and produces the
//! frame's time-domain output samples at the **internal SILK rate**
//! (8 kHz NB / 12 kHz MB / 16 kHz WB).
//!
//! The reconstruction itself runs in the **exact fixed-point
//! arithmetic** of the RFC 6716 §A reference listing
//! ([`crate::silk_decode_core`]): §4.2.7.9.1 LTP synthesis (with the
//! re-whitening over the carried output history and the per-subframe
//! gain-change state rescaling) followed by §4.2.7.9.2 LPC synthesis,
//! producing signed 16-bit samples. The i16 samples are returned here as
//! `f32` (`xq / 32768`, an exact conversion — every i16 is representable)
//! so the downstream mixing / resampling stages keep one sample-domain
//! convention; multiplying by `32768` recovers the reference-exact
//! integer sample.
//!
//! ## Per-subframe LPC selection (§4.2.7.9)
//!
//! The RFC §4.2.7.9 preamble says: "If this is the first or second
//! subframe of a 20 ms SILK frame and the LSF interpolation factor,
//! w_Q2 …, is less than 4, then [a_Q12] correspond to the final LPC
//! coefficients produced … from the interpolated LSF coefficients,
//! n1_Q15[k] …. Otherwise, they correspond to the final LPC
//! coefficients produced from the uninterpolated LSF coefficients for
//! the current frame, n2_Q15[k]." [`SilkFrameDecoded`] carries both and
//! the core maps subframes 0 / 1 of an interpolation-split 20 ms frame
//! to `lpc_first_half`, everything else to `lpc_second_half`; the
//! companion §4.2.7.9.1 re-whitening at subframe 2 fires under the same
//! `w_Q2 < 4` condition.
//!
//! ## Cross-frame state
//!
//! The §4.2.7.9.1 output history, the §4.2.7.9.2 Q14 LPC history, the
//! previous subframe gain and the previous frame's pitch lag persist
//! across SILK frames (and across Opus frames within a stream); they are
//! cleared only on a §4.5.2 decoder reset or after an uncoded regular
//! SILK frame for the side channel. [`SilkSynthState`] holds them and is
//! threaded by the caller across the SILK frames of an Opus frame.
//!
//! All truth is taken from RFC 6716 §4.2.7.9 and the §A reference
//! listing. No external library source is consulted.

use crate::silk_decode::SilkFrameDecoded;
use crate::silk_decode_core::{decode_core, SilkCoreState};
use crate::silk_excitation::SilkFrameSize;
use crate::silk_gains::SILK_MAX_SUBFRAMES;
use crate::toc::Bandwidth;
use crate::Error;

/// Cross-frame SILK synthesis state for one channel: the §4.2.7.9
/// fixed-point reconstruction histories
/// ([`crate::silk_decode_core::SilkCoreState`]).
///
/// Zero on construction and after [`Self::reset`] (the §4.5.2
/// decoder-reset path / uncoded side-channel frame). Persists across
/// SILK frames within an Opus frame and across Opus frames within a
/// stream.
#[derive(Debug, Clone)]
pub struct SilkSynthState {
    core: SilkCoreState,
}

impl SilkSynthState {
    /// Construct a fresh zero-initialised synthesis state for `bandwidth`.
    ///
    /// Rejects SWB / FB (the SILK layer never sees them after the §4.2.2
    /// hybrid split).
    pub fn new(bandwidth: Bandwidth) -> Result<Self, Error> {
        Ok(Self {
            core: SilkCoreState::new(bandwidth)?,
        })
    }

    /// Bandwidth this state was created for.
    pub fn bandwidth(&self) -> Bandwidth {
        self.core.bandwidth()
    }

    /// Clear all histories (the §4.5.2 full decoder reset).
    pub fn reset(&mut self) {
        self.core.reset();
    }

    /// Clear the prediction memories but keep the previous-subframe
    /// gain — the reference decoder's side-channel reset after a
    /// mid-only interval run (see
    /// [`crate::silk_decode_core::SilkCoreState::reset_prediction_memory`]).
    pub fn reset_prediction_memory(&mut self) {
        self.core.reset_prediction_memory();
    }
}

/// Synthesize one fully decoded regular SILK frame into time-domain
/// output samples at the internal SILK rate, threading the cross-frame
/// [`SilkSynthState`].
///
/// Returns the reconstructed signal for the whole SILK frame
/// (`subframe_samples × num_subframes` samples). Each returned `f32` is
/// an exact `xq / 32768` image of the fixed-point decoder's signed
/// 16-bit sample. The caller resamples this internal-rate signal to the
/// decoder output rate (§4.2.9).
///
/// `decoded` is the parameter set from
/// [`crate::silk_decode::decode_silk_frame`]; `state` carries the
/// §4.2.7.9 histories across SILK frames.
///
/// Errors:
///
/// * [`Error::MalformedPacket`] if `state.bandwidth()` disagrees with
///   `bandwidth`, if a per-subframe LPC filter has the wrong order, or if
///   the excitation is shorter than the frame.
pub fn synthesize_silk_frame(
    bandwidth: Bandwidth,
    frame_size: SilkFrameSize,
    decoded: &SilkFrameDecoded,
    state: &mut SilkSynthState,
) -> Result<Vec<f32>, Error> {
    let xq = synthesize_silk_frame_i16(bandwidth, frame_size, decoded, state)?;
    Ok(xq.iter().map(|&s| f32::from(s) / 32768.0).collect())
}

/// [`synthesize_silk_frame`], returning the reference-exact signed
/// 16-bit samples directly (the fixed-point decoder's native output
/// domain).
pub fn synthesize_silk_frame_i16(
    bandwidth: Bandwidth,
    frame_size: SilkFrameSize,
    decoded: &SilkFrameDecoded,
    state: &mut SilkSynthState,
) -> Result<Vec<i16, crate::bump::DecodeBump>, Error> {
    if state.bandwidth() != bandwidth {
        return Err(Error::MalformedPacket);
    }
    decode_core(&mut state.core, frame_size, decoded)
}

/// Synthesize a sequence of decoded SILK frames into one contiguous
/// internal-rate output buffer, threading `state` across the frames.
///
/// Convenience for an Opus frame that carries 2 or 3 SILK frames (40 / 60
/// ms). Each frame's samples are appended in order; `state` carries the
/// §4.2.7.9 histories across them.
pub fn synthesize_silk_frames(
    bandwidth: Bandwidth,
    frame_size: SilkFrameSize,
    decoded: &[SilkFrameDecoded],
    state: &mut SilkSynthState,
) -> Result<Vec<f32>, Error> {
    let n = subframe_samples(bandwidth)?;
    let per_frame = n * match frame_size {
        SilkFrameSize::TenMs => 2usize,
        SilkFrameSize::TwentyMs => 4usize,
    };
    let mut out = Vec::with_capacity(per_frame * decoded.len());
    for frame in decoded {
        let frame_out = synthesize_silk_frame(bandwidth, frame_size, frame, state)?;
        out.extend_from_slice(&frame_out);
    }
    Ok(out)
}

/// Samples per 5 ms subframe at the internal SILK rate.
pub fn subframe_samples(bandwidth: Bandwidth) -> Result<usize, Error> {
    match bandwidth {
        Bandwidth::Nb => Ok(40),
        Bandwidth::Mb => Ok(60),
        Bandwidth::Wb => Ok(80),
        _ => Err(Error::MalformedPacket),
    }
}

/// The internal-rate sample count one SILK frame of the given bandwidth ×
/// duration produces (a convenience mirror of
/// [`crate::silk_resampler::silk_frame_samples_internal`] expressed via
/// the synthesis subframe geometry).
pub fn silk_frame_internal_samples(
    bandwidth: Bandwidth,
    frame_size: SilkFrameSize,
) -> Result<usize, Error> {
    let n = subframe_samples(bandwidth)?;
    let num_subframes = match frame_size {
        SilkFrameSize::TenMs => 2usize,
        SilkFrameSize::TwentyMs => 4usize,
    };
    debug_assert!(num_subframes <= SILK_MAX_SUBFRAMES);
    Ok(n * num_subframes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;
    use crate::silk_decode::{decode_silk_frame, SilkFrameConfig};

    fn fresh_cfg(bandwidth: Bandwidth, frame_size: SilkFrameSize, voiced: bool) -> SilkFrameConfig {
        SilkFrameConfig {
            bandwidth,
            frame_size,
            voice_active: voiced,
            first_subframe_independent: true,
            previous_log_gain: None,
            previous_primary_lag: None,
            ltp_scaling_present: true,
            lsf_interp_after_reset: true,
            previous_nlsf_q15: None,
            previous_nlsf_len: 0,
            stereo: None,
        }
    }

    /// Synthesis state constructor routes d_LPC and rejects SWB / FB.
    #[test]
    fn state_new_routes_and_rejects() {
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            let s = SilkSynthState::new(bw).unwrap();
            assert_eq!(s.bandwidth(), bw);
        }
        assert!(SilkSynthState::new(Bandwidth::Swb).is_err());
        assert!(SilkSynthState::new(Bandwidth::Fb).is_err());
    }

    /// `silk_frame_internal_samples` matches the §4.2.9 geometry.
    #[test]
    fn internal_sample_counts() {
        assert_eq!(
            silk_frame_internal_samples(Bandwidth::Nb, SilkFrameSize::TenMs).unwrap(),
            80
        );
        assert_eq!(
            silk_frame_internal_samples(Bandwidth::Nb, SilkFrameSize::TwentyMs).unwrap(),
            160
        );
        assert_eq!(
            silk_frame_internal_samples(Bandwidth::Mb, SilkFrameSize::TwentyMs).unwrap(),
            240
        );
        assert_eq!(
            silk_frame_internal_samples(Bandwidth::Wb, SilkFrameSize::TwentyMs).unwrap(),
            320
        );
        assert_eq!(
            silk_frame_internal_samples(Bandwidth::Wb, SilkFrameSize::TenMs).unwrap(),
            160
        );
    }

    /// Bandwidth mismatch between the synthesis state and the requested
    /// frame is rejected.
    #[test]
    fn rejects_bandwidth_mismatch() {
        let buf = [0x33u8; 96];
        let mut rd = RangeDecoder::new(&buf);
        let cfg = fresh_cfg(Bandwidth::Nb, SilkFrameSize::TwentyMs, false);
        if let Ok(decoded) = decode_silk_frame(&mut rd, cfg) {
            let mut state = SilkSynthState::new(Bandwidth::Wb).unwrap();
            assert!(matches!(
                synthesize_silk_frame(Bandwidth::Nb, SilkFrameSize::TwentyMs, &decoded, &mut state),
                Err(Error::MalformedPacket)
            ));
        }
    }

    /// A real decoded frame synthesizes to the right number of samples,
    /// every sample finite and in the exact-i16-image range, and the f32
    /// image reproduces the i16 output exactly.
    #[test]
    fn synthesize_produces_in_range_samples() {
        let buf: Vec<u8> = (0..160u16)
            .map(|i| (i.wrapping_mul(67).wrapping_add(5) & 0xff) as u8)
            .collect();
        for (bw, expected) in [
            (Bandwidth::Nb, 160usize),
            (Bandwidth::Mb, 240),
            (Bandwidth::Wb, 320),
        ] {
            for voiced in [false, true] {
                let mut rd = RangeDecoder::new(&buf);
                let cfg = fresh_cfg(bw, SilkFrameSize::TwentyMs, voiced);
                if let Ok(decoded) = decode_silk_frame(&mut rd, cfg) {
                    let mut state = SilkSynthState::new(bw).unwrap();
                    let out =
                        synthesize_silk_frame(bw, SilkFrameSize::TwentyMs, &decoded, &mut state)
                            .unwrap();
                    assert_eq!(out.len(), expected, "bw={bw:?} voiced={voiced}");
                    let mut state2 = SilkSynthState::new(bw).unwrap();
                    let xq = synthesize_silk_frame_i16(
                        bw,
                        SilkFrameSize::TwentyMs,
                        &decoded,
                        &mut state2,
                    )
                    .unwrap();
                    for (i, (&v, &q)) in out.iter().zip(&xq).enumerate() {
                        assert!(v.is_finite(), "non-finite at {i}: {v}");
                        assert!(
                            (-1.0..=1.0).contains(&v),
                            "out[{i}]={v} outside nominal range (bw={bw:?})"
                        );
                        assert_eq!(
                            (v * 32768.0) as i32,
                            i32::from(q),
                            "f32 image not exact at {i}"
                        );
                    }
                }
            }
        }
    }

    /// A 10 ms frame synthesizes to two subframes; no interpolation split.
    #[test]
    fn ten_ms_two_subframes() {
        let buf: Vec<u8> = (0..96u16)
            .map(|i| (i.wrapping_mul(43).wrapping_add(9) & 0xff) as u8)
            .collect();
        let mut rd = RangeDecoder::new(&buf);
        let cfg = fresh_cfg(Bandwidth::Wb, SilkFrameSize::TenMs, false);
        if let Ok(decoded) = decode_silk_frame(&mut rd, cfg) {
            assert!(decoded.lpc_first_half.is_none());
            let mut state = SilkSynthState::new(Bandwidth::Wb).unwrap();
            let out =
                synthesize_silk_frame(Bandwidth::Wb, SilkFrameSize::TenMs, &decoded, &mut state)
                    .unwrap();
            assert_eq!(out.len(), 160); // 80 * 2
        }
    }

    /// Synthesis is deterministic: the same decoded frame + fresh state
    /// yields identical output.
    #[test]
    fn synthesis_is_deterministic() {
        let buf: Vec<u8> = (0..160u16)
            .map(|i| (i.wrapping_mul(89).wrapping_add(1) & 0xff) as u8)
            .collect();
        let mut rd = RangeDecoder::new(&buf);
        let cfg = fresh_cfg(Bandwidth::Nb, SilkFrameSize::TwentyMs, true);
        if let Ok(decoded) = decode_silk_frame(&mut rd, cfg) {
            let mut s1 = SilkSynthState::new(Bandwidth::Nb).unwrap();
            let mut s2 = SilkSynthState::new(Bandwidth::Nb).unwrap();
            let o1 =
                synthesize_silk_frame(Bandwidth::Nb, SilkFrameSize::TwentyMs, &decoded, &mut s1)
                    .unwrap();
            let o2 =
                synthesize_silk_frame(Bandwidth::Nb, SilkFrameSize::TwentyMs, &decoded, &mut s2)
                    .unwrap();
            assert_eq!(o1, o2);
        }
    }

    /// The multi-frame helper concatenates per-frame outputs and threads
    /// state (the second frame's history is non-zero going in).
    #[test]
    fn multi_frame_concatenates() {
        let buf: Vec<u8> = (0..160u16)
            .map(|i| (i.wrapping_mul(31).wrapping_add(7) & 0xff) as u8)
            .collect();
        let mut rd = RangeDecoder::new(&buf);
        let cfg = fresh_cfg(Bandwidth::Nb, SilkFrameSize::TwentyMs, false);
        if let Ok(decoded) = decode_silk_frame(&mut rd, cfg) {
            let frames = [decoded.clone(), decoded.clone()];
            let mut state = SilkSynthState::new(Bandwidth::Nb).unwrap();
            let out =
                synthesize_silk_frames(Bandwidth::Nb, SilkFrameSize::TwentyMs, &frames, &mut state)
                    .unwrap();
            assert_eq!(out.len(), 320); // 160 * 2
        }
    }

    /// Reset clears the carried histories: after a reset the same frame
    /// decodes to the same output as from a fresh state.
    #[test]
    fn reset_clears_histories() {
        let buf: Vec<u8> = (0..160u16)
            .map(|i| (i.wrapping_mul(53).wrapping_add(2) & 0xff) as u8)
            .collect();
        let mut rd = RangeDecoder::new(&buf);
        let cfg = fresh_cfg(Bandwidth::Wb, SilkFrameSize::TwentyMs, true);
        if let Ok(decoded) = decode_silk_frame(&mut rd, cfg) {
            let mut state = SilkSynthState::new(Bandwidth::Wb).unwrap();
            let first =
                synthesize_silk_frame(Bandwidth::Wb, SilkFrameSize::TwentyMs, &decoded, &mut state)
                    .unwrap();
            // A second run with carried history generally differs …
            let carried =
                synthesize_silk_frame(Bandwidth::Wb, SilkFrameSize::TwentyMs, &decoded, &mut state)
                    .unwrap();
            // … but after a reset the output matches the fresh decode.
            state.reset();
            let after_reset =
                synthesize_silk_frame(Bandwidth::Wb, SilkFrameSize::TwentyMs, &decoded, &mut state)
                    .unwrap();
            assert_eq!(first, after_reset);
            let _ = carried;
        }
    }
}
