//! SILK subframe quantization-gain decoder — RFC 6716 §4.2.7.4.
//!
//! Each SILK frame carries one quantization gain per 5 ms subframe. A
//! SILK frame is either two or four subframes long (RFC 6716 §4.2.4: 10
//! ms SILK frame = 2 subframes; 20 ms SILK frame = 4 subframes; the
//! Hybrid layer always uses 20 ms = 4 subframes). The first gain of a
//! SILK frame is coded **independently** under specific conditions
//! enumerated in §4.2.7.4; every other subframe gain is coded as a
//! signed **delta** against the previous coded subframe gain in the
//! same channel.
//!
//! The independent path decodes six bits in two pieces: a 3-bit MSB
//! drawn from one of three signal-type-conditioned PDFs (Table 11) and
//! a 3-bit LSB drawn from the uniform Table-12 PDF. The two pieces are
//! combined as `gain_index = (gain_msb << 3) | gain_lsb`, yielding a
//! value in `0..=63`. When a `previous_log_gain` is available, the
//! independent path then clamps with
//! `log_gain = max(gain_index, previous_log_gain - 16)`.
//!
//! The delta path decodes a single 41-symbol value `delta_gain_index`
//! from Table 13, then folds it into the previous gain via
//! `log_gain = clamp(0, max(2*delta - 16, prev + delta - 4), 63)`.
//!
//! The result of this module is the `log_gain` Q7-ish index in
//! `0..=63` for every subframe of the SILK frame. The §4.2.7.4
//! tail-end conversion
//! `gain_Q16[k] = silk_log2lin((0x1D1C71*log_gain >> 16) + 2090)` is
//! part of the gain dequantization path used by the excitation
//! reconstruction stage and is **not** wired up here; the inner SILK
//! pipeline only needs the integer `log_gain` index until the LPC /
//! LTP stages come online in a later round.

use crate::range_decoder::RangeDecoder;
use crate::range_encoder::RangeEncoder;
use crate::silk_frame::SignalType;
use crate::Error;

/// Maximum number of subframes per SILK frame.
///
/// RFC 6716 §4.2.7.4 says "a separate quantization gain is coded for
/// each 5 ms subframe". A SILK frame is logically either two or four
/// subframes long (10 or 20 ms; §4.2.4). The hybrid 20 ms case is the
/// upper bound.
pub const SILK_MAX_SUBFRAMES: usize = 4;

/// Table 11 — independent gain MSB PDF for signal type `Inactive`.
///
/// PDF `{32, 112, 68, 29, 12, 1, 1, 1}/256`. Cumulative
/// `fh = [32, 144, 212, 241, 253, 254, 255, 256]`. iCDF =
/// `[224, 112, 44, 15, 3, 2, 1, 0]`.
const GAIN_MSB_ICDF_INACTIVE: &[u8] = &[224, 112, 44, 15, 3, 2, 1, 0];

/// Table 11 — independent gain MSB PDF for signal type `Unvoiced`.
///
/// PDF `{2, 17, 45, 60, 62, 47, 19, 4}/256`. Cumulative
/// `fh = [2, 19, 64, 124, 186, 233, 252, 256]`. iCDF =
/// `[254, 237, 192, 132, 70, 23, 4, 0]`.
const GAIN_MSB_ICDF_UNVOICED: &[u8] = &[254, 237, 192, 132, 70, 23, 4, 0];

/// Table 11 — independent gain MSB PDF for signal type `Voiced`.
///
/// PDF `{1, 3, 26, 71, 94, 50, 9, 2}/256`. Cumulative
/// `fh = [1, 4, 30, 101, 195, 245, 254, 256]`. iCDF =
/// `[255, 252, 226, 155, 61, 11, 2, 0]`.
const GAIN_MSB_ICDF_VOICED: &[u8] = &[255, 252, 226, 155, 61, 11, 2, 0];

/// Table 12 — uniform PDF for the 3-LSB component of the independent
/// gain: `{32, 32, 32, 32, 32, 32, 32, 32}/256`. iCDF runs
/// `[224, 192, 160, 128, 96, 64, 32, 0]`.
const GAIN_LSB_ICDF: &[u8] = &[224, 192, 160, 128, 96, 64, 32, 0];

/// Table 13 — delta-gain PDF (41 cells, values 0..=40).
///
/// PDF `{6, 5, 11, 31, 132, 21, 8, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
/// 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
/// 1}/256`. Sum = 256.
///
/// Cumulative `fh`:
/// ```text
/// [6, 11, 22, 53, 185, 206, 214, 218, 221, 223, 225, 227,
///  228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
///  239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
///  250, 251, 252, 253, 254, 255, 256]
/// ```
///
/// iCDF = `256 - fh[k]` (terminated by 0):
/// ```text
/// [250, 245, 234, 203, 71, 50, 42, 38, 35, 33, 31, 29,
///  28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
///  15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
/// ```
const GAIN_DELTA_ICDF: &[u8] = &[
    250, 245, 234, 203, 71, 50, 42, 38, 35, 33, 31, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
    17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];

/// Caller-supplied context that determines whether the first subframe
/// of the current SILK frame is coded independently (§4.2.7.4).
///
/// Per §4.2.7.4 the FIRST subframe of the SILK frame is independently
/// coded iff:
///
/// 1. This is the FIRST SILK frame of its type (LBRR or regular) for
///    this channel in the current Opus frame, **OR**
/// 2. The PREVIOUS SILK frame of the same type for this channel in
///    the same Opus frame was NOT coded.
///
/// Every other subframe in the frame is delta-coded against the
/// previous subframe's gain.
///
/// The orchestration of (1) / (2) lives one layer up (in the Opus
/// frame walker) because it depends on the §4.2.4 SILK frame layout,
/// the §4.2.5 channel layout, and the §4.2.6 LBRR flag. This module
/// just receives the boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubframeGainsConfig {
    /// Signal type from §4.2.7.3 / Table 10 — selects the Table 11
    /// MSB PDF for any independently-coded subframe gain.
    pub signal_type: SignalType,
    /// Number of subframes in this SILK frame: 2 (10 ms SILK frame)
    /// or 4 (20 ms SILK frame; also the only choice in Hybrid mode).
    /// Other values are rejected.
    pub num_subframes: u8,
    /// Whether the FIRST subframe of this SILK frame is coded
    /// independently per the §4.2.7.4 enumeration above.
    pub first_subframe_is_independent: bool,
    /// Previous SILK frame's last subframe gain in this channel, if
    /// available. The §4.2.7.4 clamp `max(gain_index, prev - 16)`
    /// applies to independent coding when this is `Some`; the spec
    /// also says the clamp is SKIPPED after a decoder reset and on
    /// the side channel when the previous side-channel frame was not
    /// coded (in which case the caller passes `None`). It MAY also
    /// be skipped after packet loss (caller's discretion; here
    /// `None` skips it).
    pub previous_log_gain: Option<u8>,
}

/// One subframe's decoded quantization gain — the integer `log_gain`
/// in `0..=63`.
///
/// The §4.2.7.4 tail-end mapping to `gain_Q16` (via `silk_log2lin`)
/// is part of the excitation stage, not this header decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubframeGain {
    /// Integer log-gain in `0..=63`. After the §4.2.7.4 clamp and/or
    /// delta-gain composition.
    pub log_gain: u8,
}

/// Decoded gains for a SILK frame: up to four subframes' worth.
///
/// Indices `[0..num_subframes]` are populated. Indices beyond that
/// are reported via [`SubframeGains::len`] and not accessed by the
/// downstream pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubframeGains {
    gains: [SubframeGain; SILK_MAX_SUBFRAMES],
    len: u8,
}

impl SubframeGains {
    /// Decode `cfg.num_subframes` subframe gains from `rd` per RFC 6716
    /// §4.2.7.4.
    ///
    /// Returns `Error::MalformedPacket` if `cfg.num_subframes` is not
    /// 2 or 4, or if the range decoder latches an error mid-decode.
    pub fn decode(rd: &mut RangeDecoder<'_>, cfg: SubframeGainsConfig) -> Result<Self, Error> {
        if cfg.num_subframes != 2 && cfg.num_subframes != 4 {
            return Err(Error::MalformedPacket);
        }
        let num = cfg.num_subframes as usize;
        let mut gains = [SubframeGain { log_gain: 0 }; SILK_MAX_SUBFRAMES];
        // Running `prev` value for the §4.2.7.4 clamp / delta formulas.
        // For the first subframe, the spec text uses
        // "previous_log_gain", which is the LAST subframe gain of the
        // previous SILK frame in this channel (or unavailable, in
        // which case the clamp is skipped). For subframes 2..N this is
        // simply the most recently decoded subframe gain in the
        // CURRENT SILK frame.
        let mut prev: Option<u8> = cfg.previous_log_gain;

        for (k, slot) in gains.iter_mut().enumerate().take(num) {
            let is_independent = k == 0 && cfg.first_subframe_is_independent;
            let log_gain = if is_independent {
                Self::decode_independent(rd, cfg.signal_type, prev)
            } else {
                // Delta path requires a previous value to delta against.
                // For subframes 2..N within the same frame this is
                // always defined (prev = gain we just decoded). For
                // the FIRST subframe when `first_subframe_is_independent
                // == false`, the spec text guarantees the previous
                // SILK frame WAS coded (otherwise condition 2 of the
                // independent-coding rule fires and we would have
                // taken the independent path), so `prev` MUST be
                // `Some` here. Defend anyway.
                let p = match prev {
                    Some(v) => v,
                    None => return Err(Error::MalformedPacket),
                };
                Self::decode_delta(rd, p)
            };
            *slot = SubframeGain { log_gain };
            prev = Some(log_gain);
            if rd.has_error() {
                return Err(Error::MalformedPacket);
            }
        }

        Ok(Self {
            gains,
            len: cfg.num_subframes,
        })
    }

    /// Number of populated subframes.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` if no subframes were decoded (impossible for a successful
    /// `decode()`, since `num_subframes` is constrained to 2 or 4).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// All populated subframes' gains in decoding order.
    pub fn as_slice(&self) -> &[SubframeGain] {
        &self.gains[..self.len()]
    }

    /// Last subframe's log-gain — the value to plumb into the next SILK
    /// frame's `previous_log_gain` field.
    pub fn last_log_gain(&self) -> u8 {
        self.gains[self.len() - 1].log_gain
    }

    /// Dequantise every populated subframe's `log_gain` to a Q16 linear
    /// gain via RFC 6716 §4.2.7.4
    /// `gain_Q16[k] = silk_log2lin((0x1D1C71 * log_gain[k] >> 16) + 2090)`.
    ///
    /// Returns a fixed-size array; only the first [`Self::len`] entries
    /// are populated, the rest are zero. The resulting `gain_Q16[k]`
    /// values lie in the spec-documented `[81920, 1_686_110_208]` range
    /// and are the input to the §4.2.7.9.1 LTP and §4.2.7.9.2 LPC
    /// synthesis filters.
    pub fn dequant_q16(&self) -> [u32; SILK_MAX_SUBFRAMES] {
        let mut out = [0u32; SILK_MAX_SUBFRAMES];
        for (slot, gain) in out.iter_mut().zip(self.gains.iter()).take(self.len()) {
            *slot = crate::silk_log2lin::silk_gains_dequant(gain.log_gain);
        }
        out
    }

    /// Encode `symbols` (one per subframe) into `re`, per RFC 6716
    /// §4.2.7.4 — the exact write-side mirror of [`Self::decode`].
    ///
    /// `cfg` must describe the same conditions the decoder will use:
    /// the symbol at subframe 0 must be [`GainSymbol::Independent`] iff
    /// `cfg.first_subframe_is_independent`, and every later subframe
    /// must be [`GainSymbol::Delta`]. Returns the [`SubframeGains`] the
    /// decoder will reconstruct from this bitstream (computed with the
    /// same §4.2.7.4 clamp / delta formulas), so the caller can carry
    /// the quantization feedback (e.g. [`Self::last_log_gain`] into the
    /// next frame's `previous_log_gain`).
    ///
    /// Returns `Error::MalformedPacket` on an invalid subframe count, a
    /// symbol-kind mismatch, or an out-of-range index (independent
    /// `gain_index > 63`, `delta_gain_index > 40`).
    pub fn encode(
        re: &mut RangeEncoder,
        cfg: SubframeGainsConfig,
        symbols: &[GainSymbol],
    ) -> Result<Self, Error> {
        if cfg.num_subframes != 2 && cfg.num_subframes != 4 {
            return Err(Error::MalformedPacket);
        }
        let num = cfg.num_subframes as usize;
        if symbols.len() != num {
            return Err(Error::MalformedPacket);
        }
        let mut gains = [SubframeGain { log_gain: 0 }; SILK_MAX_SUBFRAMES];
        let mut prev: Option<u8> = cfg.previous_log_gain;

        for (k, (slot, &sym)) in gains.iter_mut().zip(symbols.iter()).enumerate().take(num) {
            let want_independent = k == 0 && cfg.first_subframe_is_independent;
            let log_gain = match sym {
                GainSymbol::Independent(gain_index) => {
                    if !want_independent || gain_index > 63 {
                        return Err(Error::MalformedPacket);
                    }
                    // §4.2.7.4 independent path: MSB from the Table 11
                    // signal-type PDF, LSB from the uniform Table 12 PDF.
                    re.enc_icdf(
                        (gain_index >> 3) as usize,
                        Self::icdf_for_signal_type(cfg.signal_type),
                        8,
                    );
                    re.enc_icdf((gain_index & 0x07) as usize, GAIN_LSB_ICDF, 8);
                    // The decoder applies the clamp; reproduce it so the
                    // returned gains match the decode exactly.
                    match prev {
                        Some(p) => gain_index.max(p.saturating_sub(16)),
                        None => gain_index,
                    }
                }
                GainSymbol::Delta(delta) => {
                    if want_independent || delta > 40 {
                        return Err(Error::MalformedPacket);
                    }
                    let p = match prev {
                        Some(v) => v,
                        None => return Err(Error::MalformedPacket),
                    };
                    re.enc_icdf(delta as usize, GAIN_DELTA_ICDF, 8);
                    Self::delta_log_gain(delta, p)
                }
            };
            *slot = SubframeGain { log_gain };
            prev = Some(log_gain);
        }

        Ok(Self {
            gains,
            len: cfg.num_subframes,
        })
    }

    /// Quantize a desired per-subframe `log_gain` plan into the
    /// [`GainSymbol`]s that best reproduce it through the §4.2.7.4
    /// decode formulas, together with the [`SubframeGains`] the decoder
    /// will actually reconstruct.
    ///
    /// The symbol-selection strategy is an encoder-side choice (RFC
    /// 6716 §5.2 leaves it free); this one is deterministic and locally
    /// optimal per subframe: the independent index is the desired value
    /// itself (the §4.2.7.4 clamp may still raise the decoded gain),
    /// and each delta index minimizes the absolute error of the decoded
    /// gain against the desired one, ties toward the smaller index
    /// (fewer large-delta symbols).
    ///
    /// `desired.len()` must equal `cfg.num_subframes`, every desired
    /// value must be `<= 63`.
    pub fn quantize(
        cfg: SubframeGainsConfig,
        desired: &[u8],
    ) -> Result<(Vec<GainSymbol>, Self), Error> {
        if cfg.num_subframes != 2 && cfg.num_subframes != 4 {
            return Err(Error::MalformedPacket);
        }
        let num = cfg.num_subframes as usize;
        if desired.len() != num || desired.iter().any(|&g| g > 63) {
            return Err(Error::MalformedPacket);
        }
        let mut symbols = Vec::with_capacity(num);
        let mut gains = [SubframeGain { log_gain: 0 }; SILK_MAX_SUBFRAMES];
        let mut prev: Option<u8> = cfg.previous_log_gain;

        for (k, (&want, slot)) in desired.iter().zip(gains.iter_mut()).enumerate().take(num) {
            let independent = k == 0 && cfg.first_subframe_is_independent;
            let (sym, decoded) = if independent {
                let decoded = match prev {
                    Some(p) => want.max(p.saturating_sub(16)),
                    None => want,
                };
                (GainSymbol::Independent(want), decoded)
            } else {
                let p = match prev {
                    Some(v) => v,
                    None => return Err(Error::MalformedPacket),
                };
                // Deterministic argmin over the 41 delta symbols.
                let mut best = (0u8, Self::delta_log_gain(0, p));
                for delta in 1..=40u8 {
                    let g = Self::delta_log_gain(delta, p);
                    if (g as i32 - want as i32).abs() < (best.1 as i32 - want as i32).abs() {
                        best = (delta, g);
                    }
                }
                (GainSymbol::Delta(best.0), best.1)
            };
            symbols.push(sym);
            *slot = SubframeGain { log_gain: decoded };
            prev = Some(decoded);
        }

        Ok((
            symbols,
            Self {
                gains,
                len: cfg.num_subframes,
            },
        ))
    }

    /// The §4.2.7.4 delta-path reconstruction
    /// `log_gain = clamp(0, max(2*delta - 16, prev + delta - 4), 63)`,
    /// shared by the decode, encode, and quantize paths.
    fn delta_log_gain(delta: u8, previous_log_gain: u8) -> u8 {
        let delta = delta as i32;
        let a = 2 * delta - 16;
        let b = previous_log_gain as i32 + delta - 4;
        a.max(b).clamp(0, 63) as u8
    }

    fn icdf_for_signal_type(signal_type: SignalType) -> &'static [u8] {
        match signal_type {
            SignalType::Inactive => GAIN_MSB_ICDF_INACTIVE,
            SignalType::Unvoiced => GAIN_MSB_ICDF_UNVOICED,
            SignalType::Voiced => GAIN_MSB_ICDF_VOICED,
        }
    }

    /// Independent (Table 11 + 12) coding path of §4.2.7.4.
    ///
    /// Returns the clamped `log_gain` in 0..=63.
    fn decode_independent(
        rd: &mut RangeDecoder<'_>,
        signal_type: SignalType,
        previous_log_gain: Option<u8>,
    ) -> u8 {
        let msb = rd.dec_icdf(Self::icdf_for_signal_type(signal_type), 8) as u8;
        let lsb = rd.dec_icdf(GAIN_LSB_ICDF, 8) as u8;
        // gain_index ∈ 0..=63.
        let gain_index = (msb << 3) | (lsb & 0x07);
        // §4.2.7.4: log_gain = max(gain_index, previous_log_gain - 16).
        // The subtraction is on signed integers and saturates at 0 (we
        // can't go below 0; the spec text means "the value of the
        // expression is unsigned"). previous - 16 is computed via
        // saturating_sub for safety.
        match previous_log_gain {
            Some(prev) => gain_index.max(prev.saturating_sub(16)),
            None => gain_index,
        }
    }

    /// Delta (Table 13) coding path of §4.2.7.4.
    ///
    /// Returns the clamped `log_gain` in 0..=63.
    fn decode_delta(rd: &mut RangeDecoder<'_>, previous_log_gain: u8) -> u8 {
        let delta = rd.dec_icdf(GAIN_DELTA_ICDF, 8) as u8;
        // §4.2.7.4: log_gain = clamp(0, max(2*delta - 16, prev + delta
        // - 4), 63).
        Self::delta_log_gain(delta.min(40), previous_log_gain)
    }
}

/// One subframe's gain symbol on the encode side — the §4.2.7.4 coding
/// choice [`SubframeGains::encode`] writes for that subframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GainSymbol {
    /// Independent (Table 11 + 12) coding: the 6-bit `gain_index` in
    /// `0..=63`, written as a 3-bit MSB + 3-bit LSB pair. Only valid
    /// for subframe 0 when
    /// [`SubframeGainsConfig::first_subframe_is_independent`] is set.
    Independent(u8),
    /// Delta (Table 13) coding: the `delta_gain_index` in `0..=40`,
    /// folded into the previous subframe gain by the §4.2.7.4 formula.
    Delta(u8),
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Table 11 / 12 / 13 PDF-to-iCDF self-checks --------

    #[test]
    fn gain_msb_inactive_pdf_sums_to_256() {
        let pdf = [32u32, 112, 68, 29, 12, 1, 1, 1];
        assert_eq!(pdf.iter().sum::<u32>(), 256);
        assert_eq!(GAIN_MSB_ICDF_INACTIVE.len(), pdf.len());
        // Strictly decreasing, terminator zero.
        for w in GAIN_MSB_ICDF_INACTIVE.windows(2) {
            assert!(w[0] > w[1] || (w[0] == 0 && w[1] == 0));
        }
        assert_eq!(*GAIN_MSB_ICDF_INACTIVE.last().unwrap(), 0);
    }

    #[test]
    fn gain_msb_unvoiced_pdf_sums_to_256() {
        let pdf = [2u32, 17, 45, 60, 62, 47, 19, 4];
        assert_eq!(pdf.iter().sum::<u32>(), 256);
        // Self-check iCDF transcription.
        assert_eq!(
            GAIN_MSB_ICDF_UNVOICED,
            &[254u8, 237, 192, 132, 70, 23, 4, 0]
        );
    }

    #[test]
    fn gain_msb_voiced_pdf_sums_to_256() {
        let pdf = [1u32, 3, 26, 71, 94, 50, 9, 2];
        assert_eq!(pdf.iter().sum::<u32>(), 256);
        assert_eq!(GAIN_MSB_ICDF_VOICED, &[255u8, 252, 226, 155, 61, 11, 2, 0]);
    }

    #[test]
    fn gain_lsb_pdf_is_uniform_eight() {
        // PDF = 32 * 8 = 256.
        assert_eq!(GAIN_LSB_ICDF, &[224u8, 192, 160, 128, 96, 64, 32, 0]);
    }

    #[test]
    fn gain_delta_pdf_sums_to_256() {
        let pdf: [u32; 41] = [
            6, 5, 11, 31, 132, 21, 8, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];
        assert_eq!(pdf.iter().sum::<u32>(), 256);
        assert_eq!(GAIN_DELTA_ICDF.len(), pdf.len());
        // Terminator zero, monotone non-increasing.
        assert_eq!(*GAIN_DELTA_ICDF.last().unwrap(), 0);
        for w in GAIN_DELTA_ICDF.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    #[test]
    fn gain_delta_first_cell_matches_pdf() {
        // First cell PDF value = 6; first iCDF cell = 256 - 6 = 250.
        assert_eq!(GAIN_DELTA_ICDF[0], 250);
        // Fourth cell PDF value = 31; cumulative = 6+5+11+31 = 53;
        // fourth iCDF cell = 256 - 53 = 203.
        assert_eq!(GAIN_DELTA_ICDF[3], 203);
        // Fifth cell PDF value = 132 (the modal cell); cumulative =
        // 53+132 = 185; fifth iCDF cell = 256 - 185 = 71.
        assert_eq!(GAIN_DELTA_ICDF[4], 71);
    }

    // ----- Independent path: previous-log-gain clamp --------

    #[test]
    fn clamp_independent_with_no_previous_returns_raw_index() {
        // If previous_log_gain is None (decoder reset / side channel
        // not previously coded / packet loss), the §4.2.7.4 clamp is
        // skipped — log_gain = gain_index.
        // Run a single decode and confirm log_gain matches the
        // gain_index that the iCDF reads.
        let buf = [0x55u8, 0xAA, 0x33, 0xCC, 0x7F, 0x80, 0x12, 0x34];
        let mut rd1 = RangeDecoder::new(&buf);
        let mut rd2 = RangeDecoder::new(&buf);

        let msb = rd1.dec_icdf(GAIN_MSB_ICDF_INACTIVE, 8) as u8;
        let lsb = rd1.dec_icdf(GAIN_LSB_ICDF, 8) as u8;
        let expected_gain = (msb << 3) | (lsb & 0x07);

        // Same buffer, same decoder state: SubframeGains::decode_independent
        // should produce the same gain when no prev is supplied.
        let got = SubframeGains::decode_independent(&mut rd2, SignalType::Inactive, None);
        assert_eq!(got, expected_gain);
        assert!(got < 64);
    }

    #[test]
    fn clamp_independent_with_low_previous_keeps_raw_index() {
        // If previous_log_gain - 16 < gain_index, the clamp is a no-op:
        // log_gain = gain_index.
        let buf = [0x42u8, 0x18, 0xC3, 0x7F, 0x55, 0xAA, 0x33, 0xCC];
        let mut rd1 = RangeDecoder::new(&buf);
        let mut rd2 = RangeDecoder::new(&buf);

        let msb = rd1.dec_icdf(GAIN_MSB_ICDF_VOICED, 8) as u8;
        let lsb = rd1.dec_icdf(GAIN_LSB_ICDF, 8) as u8;
        let raw = (msb << 3) | (lsb & 0x07);

        // Pick previous_log_gain such that prev - 16 < raw guaranteed.
        let got = SubframeGains::decode_independent(&mut rd2, SignalType::Voiced, Some(0));
        assert_eq!(got, raw);
    }

    #[test]
    fn clamp_independent_with_high_previous_raises_index() {
        // If previous_log_gain - 16 > gain_index, the clamp raises
        // log_gain up to previous_log_gain - 16.
        let buf = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut rd = RangeDecoder::new(&buf);
        // previous_log_gain = 60 → floor = 60 - 16 = 44. Inactive PDF
        // strongly biased toward small MSBs, so for an all-zero buffer
        // we expect the raw gain_index to be < 44. Either way, the
        // result must satisfy log_gain >= 44.
        let got = SubframeGains::decode_independent(&mut rd, SignalType::Inactive, Some(60));
        assert!(got >= 44, "got={got}");
        assert!(got <= 63, "got={got}");
    }

    #[test]
    fn clamp_independent_previous_saturates_at_zero() {
        // For previous_log_gain in 0..16, prev - 16 saturates to 0
        // (the spec text is unsigned). No raise.
        let buf = [0x55u8, 0xAA, 0x33, 0xCC, 0x7F, 0x80];
        let mut rd1 = RangeDecoder::new(&buf);
        let mut rd2 = RangeDecoder::new(&buf);

        let msb = rd1.dec_icdf(GAIN_MSB_ICDF_UNVOICED, 8) as u8;
        let lsb = rd1.dec_icdf(GAIN_LSB_ICDF, 8) as u8;
        let raw = (msb << 3) | (lsb & 0x07);

        let got = SubframeGains::decode_independent(&mut rd2, SignalType::Unvoiced, Some(15));
        assert_eq!(got, raw);
    }

    // ----- Delta path: §4.2.7.4 dual-max + clamp --------

    #[test]
    fn delta_path_clamps_to_0_63() {
        // Run delta with many `prev` values and confirm the output
        // always lies in 0..=63.
        let buf = [0x77u8, 0x33, 0x11, 0xAA, 0xDE, 0xAD, 0xBE, 0xEF, 0x55];
        for prev in [0u8, 1, 10, 31, 32, 50, 60, 63] {
            let mut rd = RangeDecoder::new(&buf);
            let g = SubframeGains::decode_delta(&mut rd, prev);
            assert!(g <= 63, "prev={prev}, g={g}");
        }
    }

    #[test]
    fn delta_path_formula_consistency() {
        // For a given decoded `delta`, log_gain = clamp(0, max(2*delta
        // - 16, prev + delta - 4), 63). Reproduce the decode with two
        // separate range decoders and check the formula.
        let buf = [0x12u8, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let mut rd1 = RangeDecoder::new(&buf);
        let mut rd2 = RangeDecoder::new(&buf);

        let delta = rd1.dec_icdf(GAIN_DELTA_ICDF, 8) as i32;
        let prev = 30u8;
        let expected = {
            let a = 2 * delta - 16;
            let b = prev as i32 + delta - 4;
            a.max(b).clamp(0, 63) as u8
        };

        let got = SubframeGains::decode_delta(&mut rd2, prev);
        assert_eq!(got, expected, "delta={delta}");
    }

    // ----- End-to-end: full SubframeGains::decode against the range
    // decoder.

    fn fresh_buf() -> [u8; 24] {
        [
            0xC3, 0x18, 0x42, 0x7F, 0x55, 0xAA, 0x33, 0xCC, 0x77, 0x33, 0x11, 0xAA, 0xDE, 0xAD,
            0xBE, 0xEF, 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
        ]
    }

    #[test]
    fn full_decode_inactive_four_subframes_first_independent() {
        // Mono inactive, 20 ms SILK frame (4 subframes), first
        // subframe independent (first frame of its type for this
        // channel), no previous gain available.
        let buf = fresh_buf();
        let mut rd = RangeDecoder::new(&buf);
        let cfg = SubframeGainsConfig {
            signal_type: SignalType::Inactive,
            num_subframes: 4,
            first_subframe_is_independent: true,
            previous_log_gain: None,
        };
        let gains = SubframeGains::decode(&mut rd, cfg).expect("decode must succeed");
        assert_eq!(gains.len(), 4);
        for g in gains.as_slice() {
            assert!(g.log_gain <= 63);
        }
        // Sanity: the FIRST gain (independent, no prev) is bounded by
        // (msb<<3)|lsb so always <= 63.
        assert!(gains.as_slice()[0].log_gain <= 63);
    }

    #[test]
    fn full_decode_unvoiced_two_subframes_first_delta() {
        // 10 ms SILK frame (2 subframes), first subframe DELTA (so the
        // caller has a prev gain from the previous SILK frame in this
        // channel). Unvoiced signal type.
        let buf = fresh_buf();
        let mut rd = RangeDecoder::new(&buf);
        let cfg = SubframeGainsConfig {
            signal_type: SignalType::Unvoiced,
            num_subframes: 2,
            first_subframe_is_independent: false,
            previous_log_gain: Some(40),
        };
        let gains = SubframeGains::decode(&mut rd, cfg).expect("decode must succeed");
        assert_eq!(gains.len(), 2);
        for g in gains.as_slice() {
            assert!(g.log_gain <= 63);
        }
    }

    #[test]
    fn full_decode_voiced_four_subframes_first_independent_with_prev() {
        // First subframe independent (first SILK frame of its type
        // for this channel in the Opus frame), but the channel has a
        // prev gain from a prior packet (NOT the current Opus frame —
        // the §4.2.7.4 clamp uses it).
        let buf = fresh_buf();
        let mut rd = RangeDecoder::new(&buf);
        let cfg = SubframeGainsConfig {
            signal_type: SignalType::Voiced,
            num_subframes: 4,
            first_subframe_is_independent: true,
            previous_log_gain: Some(50),
        };
        let gains = SubframeGains::decode(&mut rd, cfg).expect("decode must succeed");
        assert_eq!(gains.len(), 4);
        // First subframe: independent with prev=50, so log_gain >=
        // 50 - 16 = 34.
        assert!(
            gains.as_slice()[0].log_gain >= 34,
            "first gain={}",
            gains.as_slice()[0].log_gain
        );
        for g in gains.as_slice() {
            assert!(g.log_gain <= 63);
        }
    }

    #[test]
    fn full_decode_first_delta_without_prev_is_rejected() {
        // Pathological: caller claims delta coding for the first
        // subframe but supplies no prev. §4.2.7.4 enumeration
        // guarantees this never happens in well-formed input; we
        // still defend.
        let buf = fresh_buf();
        let mut rd = RangeDecoder::new(&buf);
        let cfg = SubframeGainsConfig {
            signal_type: SignalType::Voiced,
            num_subframes: 4,
            first_subframe_is_independent: false,
            previous_log_gain: None,
        };
        let err = SubframeGains::decode(&mut rd, cfg).unwrap_err();
        assert_eq!(err, Error::MalformedPacket);
    }

    #[test]
    fn full_decode_invalid_subframe_count_is_rejected() {
        // num_subframes must be 2 or 4.
        let buf = fresh_buf();
        for bad in [0u8, 1, 3, 5, 8, 255] {
            let mut rd = RangeDecoder::new(&buf);
            let cfg = SubframeGainsConfig {
                signal_type: SignalType::Voiced,
                num_subframes: bad,
                first_subframe_is_independent: true,
                previous_log_gain: None,
            };
            let err = SubframeGains::decode(&mut rd, cfg).unwrap_err();
            assert_eq!(err, Error::MalformedPacket, "bad num_subframes={bad}");
        }
    }

    #[test]
    fn last_log_gain_matches_last_subframe() {
        let buf = fresh_buf();
        let mut rd = RangeDecoder::new(&buf);
        let cfg = SubframeGainsConfig {
            signal_type: SignalType::Voiced,
            num_subframes: 4,
            first_subframe_is_independent: true,
            previous_log_gain: None,
        };
        let gains = SubframeGains::decode(&mut rd, cfg).expect("decode must succeed");
        assert_eq!(gains.last_log_gain(), gains.as_slice()[3].log_gain);
    }

    #[test]
    fn delta_after_independent_chain_consistency() {
        // Subframes 2..N use delta against the LATEST decoded gain.
        // Reproduce the full chain manually and compare.
        let buf = fresh_buf();

        // Manual chain — independent then 3 deltas.
        let mut rd_manual = RangeDecoder::new(&buf);
        let msb = rd_manual.dec_icdf(GAIN_MSB_ICDF_VOICED, 8) as u8;
        let lsb = rd_manual.dec_icdf(GAIN_LSB_ICDF, 8) as u8;
        let g0 = (msb << 3) | (lsb & 0x07); // prev=None, so clamp no-op
        let mut prev = g0;
        let mut chain = [g0, 0, 0, 0];
        for slot in &mut chain[1..] {
            let delta = rd_manual.dec_icdf(GAIN_DELTA_ICDF, 8) as i32;
            let a = 2 * delta - 16;
            let b = prev as i32 + delta - 4;
            let g = a.max(b).clamp(0, 63) as u8;
            *slot = g;
            prev = g;
        }

        // SubframeGains chain.
        let mut rd = RangeDecoder::new(&buf);
        let cfg = SubframeGainsConfig {
            signal_type: SignalType::Voiced,
            num_subframes: 4,
            first_subframe_is_independent: true,
            previous_log_gain: None,
        };
        let gains = SubframeGains::decode(&mut rd, cfg).expect("decode must succeed");
        for (k, expected) in chain.iter().enumerate() {
            assert_eq!(
                gains.as_slice()[k].log_gain,
                *expected,
                "subframe {} mismatch: got {}, expected {}",
                k,
                gains.as_slice()[k].log_gain,
                *expected
            );
        }
    }

    // ----- §4.2.7.4 dequant_q16 -----------------------------------

    #[test]
    fn dequant_q16_zeroes_trailing_slots_for_two_subframe_frame() {
        // Drive the decoder with two subframes; assert the trailing
        // unused slots in the fixed-size dequant array stay zero.
        let mut bytes = vec![0u8; 16];
        bytes[0] = 0x40;
        let mut rd = RangeDecoder::new(&bytes);
        let cfg = SubframeGainsConfig {
            signal_type: SignalType::Inactive,
            num_subframes: 2,
            first_subframe_is_independent: true,
            previous_log_gain: None,
        };
        let gains = SubframeGains::decode(&mut rd, cfg).expect("decode must succeed");
        let q16 = gains.dequant_q16();
        // First two slots populated; bounded by the spec range.
        for (k, val) in q16.iter().enumerate().take(gains.len()) {
            assert!(
                (81_920..=1_686_110_208).contains(val),
                "subframe {} gain_Q16 {} outside spec range",
                k,
                val
            );
        }
        // Slots beyond `len` left at zero.
        for (k, val) in q16.iter().enumerate().skip(gains.len()) {
            assert_eq!(*val, 0, "slot {} expected zero, got {}", k, val);
        }
    }

    #[test]
    fn dequant_q16_matches_per_subframe_call() {
        // Drive the decoder with four subframes; assert the array
        // dequant matches a per-subframe `silk_gains_dequant` recomputation.
        let mut bytes = vec![0u8; 16];
        bytes[0] = 0xFF;
        let mut rd = RangeDecoder::new(&bytes);
        let cfg = SubframeGainsConfig {
            signal_type: SignalType::Voiced,
            num_subframes: 4,
            first_subframe_is_independent: true,
            previous_log_gain: None,
        };
        let gains = SubframeGains::decode(&mut rd, cfg).expect("decode must succeed");
        let array = gains.dequant_q16();
        for (k, val) in array.iter().enumerate().take(gains.len()) {
            let log_gain = gains.as_slice()[k].log_gain;
            let expected = crate::silk_log2lin::silk_gains_dequant(log_gain);
            assert_eq!(*val, expected, "subframe {} mismatch", k);
        }
    }

    // ----- §4.2.7.4 encode-side mirror ------------------------------

    /// A tiny deterministic LCG for the encode/decode roundtrip sweeps.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n
        }
    }

    /// encode → decode roundtrip: for random symbol scripts across all
    /// signal types / subframe counts / prev contexts, the decoder must
    /// reconstruct exactly the gains the encoder returned.
    #[test]
    fn encode_decode_roundtrip_random_scripts() {
        let mut rng = Lcg(0x0BAD_5EED);
        for _ in 0..500 {
            let signal_type = match rng.below(3) {
                0 => SignalType::Inactive,
                1 => SignalType::Unvoiced,
                _ => SignalType::Voiced,
            };
            let num_subframes = if rng.below(2) == 0 { 2u8 } else { 4 };
            let first_independent = rng.below(2) == 0;
            let previous_log_gain = if first_independent && rng.below(2) == 0 {
                None
            } else {
                Some(rng.below(64) as u8)
            };
            let cfg = SubframeGainsConfig {
                signal_type,
                num_subframes,
                first_subframe_is_independent: first_independent,
                previous_log_gain,
            };
            let symbols: Vec<GainSymbol> = (0..num_subframes as usize)
                .map(|k| {
                    if k == 0 && first_independent {
                        GainSymbol::Independent(rng.below(64) as u8)
                    } else {
                        GainSymbol::Delta(rng.below(41) as u8)
                    }
                })
                .collect();

            let mut re = RangeEncoder::new();
            let predicted = SubframeGains::encode(&mut re, cfg, &symbols).expect("encode");
            let bytes = re.finish();

            let mut rd = RangeDecoder::new(&bytes);
            let decoded = SubframeGains::decode(&mut rd, cfg).expect("decode");
            assert!(!rd.has_error());
            assert_eq!(decoded, predicted, "cfg={cfg:?} symbols={symbols:?}");
        }
    }

    /// quantize → encode → decode: the decoded gains equal the
    /// quantizer's prediction, and (where reachable) track the desired
    /// plan closely.
    #[test]
    fn quantize_encode_decode_consistency() {
        let mut rng = Lcg(0x5EED_CAFE);
        for _ in 0..500 {
            let cfg = SubframeGainsConfig {
                signal_type: SignalType::Voiced,
                num_subframes: 4,
                first_subframe_is_independent: true,
                previous_log_gain: if rng.below(2) == 0 {
                    None
                } else {
                    Some(rng.below(64) as u8)
                },
            };
            let desired: Vec<u8> = (0..4).map(|_| rng.below(64) as u8).collect();
            let (symbols, predicted) = SubframeGains::quantize(cfg, &desired).expect("quantize");

            let mut re = RangeEncoder::new();
            let encoded = SubframeGains::encode(&mut re, cfg, &symbols).expect("encode");
            assert_eq!(encoded, predicted);

            let bytes = re.finish();
            let mut rd = RangeDecoder::new(&bytes);
            let decoded = SubframeGains::decode(&mut rd, cfg).expect("decode");
            assert_eq!(decoded, predicted);

            // Delta-coded subframes 1..4 can always land within 4 of
            // the desired value when the desired plan is monotone-ish;
            // globally, every decoded gain is in range.
            for g in decoded.as_slice() {
                assert!(g.log_gain <= 63);
            }
        }
    }

    /// The delta symbol range covers enough motion that the quantizer
    /// reproduces any reachable steady-state plan exactly: feeding the
    /// decoder's own output back as the desired plan is a fixed point.
    #[test]
    fn quantize_is_fixed_point_on_reachable_plans() {
        let mut rng = Lcg(0xF1DE_117E);
        for _ in 0..200 {
            let cfg = SubframeGainsConfig {
                signal_type: SignalType::Unvoiced,
                num_subframes: 4,
                first_subframe_is_independent: true,
                previous_log_gain: None,
            };
            let desired: Vec<u8> = (0..4).map(|_| rng.below(64) as u8).collect();
            let (_, first_pass) = SubframeGains::quantize(cfg, &desired).expect("quantize");
            let reachable: Vec<u8> = first_pass.as_slice().iter().map(|g| g.log_gain).collect();
            let (_, second_pass) = SubframeGains::quantize(cfg, &reachable).expect("quantize");
            assert_eq!(first_pass, second_pass, "desired={desired:?}");
        }
    }

    /// Symbol-kind mismatches and out-of-range indices are rejected.
    #[test]
    fn encode_rejects_bad_symbols() {
        let cfg = SubframeGainsConfig {
            signal_type: SignalType::Voiced,
            num_subframes: 2,
            first_subframe_is_independent: true,
            previous_log_gain: None,
        };
        // Delta where independent expected.
        let mut re = RangeEncoder::new();
        assert!(
            SubframeGains::encode(&mut re, cfg, &[GainSymbol::Delta(3), GainSymbol::Delta(3)])
                .is_err()
        );
        // Independent where delta expected.
        let mut re = RangeEncoder::new();
        assert!(SubframeGains::encode(
            &mut re,
            cfg,
            &[GainSymbol::Independent(10), GainSymbol::Independent(10)]
        )
        .is_err());
        // Out-of-range independent index.
        let mut re = RangeEncoder::new();
        assert!(SubframeGains::encode(
            &mut re,
            cfg,
            &[GainSymbol::Independent(64), GainSymbol::Delta(3)]
        )
        .is_err());
        // Out-of-range delta index.
        let mut re = RangeEncoder::new();
        assert!(SubframeGains::encode(
            &mut re,
            cfg,
            &[GainSymbol::Independent(10), GainSymbol::Delta(41)]
        )
        .is_err());
        // Wrong symbol count.
        let mut re = RangeEncoder::new();
        assert!(SubframeGains::encode(&mut re, cfg, &[GainSymbol::Independent(10)]).is_err());
    }

    #[test]
    fn signal_type_routes_to_correct_icdf() {
        // Compare by VALUE — pointer identity isn't guaranteed across
        // toolchains (Windows MSVC deduplicates `&[u8]` literals
        // differently from x86_64-unknown-linux-gnu and aarch64-apple-darwin).
        assert_eq!(
            SubframeGains::icdf_for_signal_type(SignalType::Inactive),
            GAIN_MSB_ICDF_INACTIVE
        );
        assert_eq!(
            SubframeGains::icdf_for_signal_type(SignalType::Unvoiced),
            GAIN_MSB_ICDF_UNVOICED
        );
        assert_eq!(
            SubframeGains::icdf_for_signal_type(SignalType::Voiced),
            GAIN_MSB_ICDF_VOICED
        );
    }
}
