//! SILK per-frame header decoding — RFC 6716 §4.2.7.1 through §4.2.7.5.1.
//!
//! Each regular SILK frame begins with a fixed prefix of side-information
//! symbols that drive the subsequent gain / LSF / LTP / excitation
//! stages. This module parses that prefix:
//!
//! * §4.2.7.1 — Stereo prediction weights (mid-channel of a stereo frame
//!   only). Three range-coded indices `n`, `(i0, i1)`, `(i2, i3)` are
//!   combined into a pair of Q13 prediction weights `(w0_Q13, w1_Q13)`
//!   per the formulas at the end of §4.2.7.1.
//! * §4.2.7.2 — Mid-only flag (mid-channel of a stereo frame, when the
//!   side channel is not otherwise required).
//! * §4.2.7.3 — Frame-type symbol, which jointly carries the signal type
//!   ([`SignalType`]) and the quantization-offset type
//!   ([`QuantizationOffsetType`]) per Table 10.
//! * §4.2.7.5.1 — Normalized LSF stage-1 codebook index `I1` (0..32),
//!   PDF chosen from Table 14 by `(bandwidth, signal_type)`.
//!
//! All symbols are read from a [`RangeDecoder`] using the §4.1.3.3
//! inverse-CDF primitive. The PDFs in Tables 6, 8, 9, and 14 are
//! transcribed verbatim from RFC 6716.
//!
//! Higher-level SILK stages (subframe gains, LSF stage-2 residual, LTP
//! parameters, LCG seed, excitation) are out of scope for round 4 — the
//! goal here is to land the entry point onto the SILK frame body and
//! the four structural decisions that everything downstream branches
//! on.

use crate::range_decoder::RangeDecoder;
use crate::range_encoder::RangeEncoder;
use crate::toc::Bandwidth;
use crate::Error;

/// Decoded stereo prediction weights for one mid-channel SILK frame
/// (RFC 6716 §4.2.7.1).
///
/// Both weights are in Q13 fixed-point. By construction
/// `w0_Q13, w1_Q13 ∈ [-13732 - 0.1*(13732 - 10050), 13732 + ...]`,
/// i.e. roughly `[-14_100, +14_100]` after the interpolation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StereoPredictionWeights {
    /// First weight, Q13. The decoded formula:
    /// `w0 = w_Q13[wi0] + (((w_Q13[wi0+1] - w_Q13[wi0])*6554) >> 16)*(2*i1+1) - w1`.
    pub w0_q13: i32,
    /// Second weight, Q13. The decoded formula:
    /// `w1 = w_Q13[wi1] + (((w_Q13[wi1+1] - w_Q13[wi1])*6554) >> 16)*(2*i3+1)`.
    pub w1_q13: i32,
}

/// Signal type carried by the §4.2.7.3 frame-type symbol (Table 10).
///
/// Drives downstream LSF stage-1 PDF selection (§4.2.7.5.1) and the
/// gain MSB PDF (§4.2.7.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Frame-type 0 or 1 (Table 10).
    Inactive,
    /// Frame-type 2 or 3 (Table 10).
    Unvoiced,
    /// Frame-type 4 or 5 (Table 10).
    Voiced,
}

/// Quantization-offset type carried by the §4.2.7.3 frame-type symbol
/// (Table 10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationOffsetType {
    /// Even frame-type values (0, 2, 4).
    Low,
    /// Odd frame-type values (1, 3, 5).
    High,
}

/// Whether the current SILK frame is an LBRR frame or a regular SILK
/// frame, and the VAD state of the corresponding time interval.
///
/// Drives which §4.2.7.3 PDF is used (Table 9) and whether the
/// mid-only flag (§4.2.7.2) appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// Regular SILK frame whose VAD flag is unset for this time
    /// interval. Frame-type symbol uses the "Inactive" PDF in Table 9.
    /// Decoded value is always 0 or 1.
    RegularInactive,
    /// Regular SILK frame whose VAD flag is set for this time
    /// interval. Frame-type symbol uses the "Active" PDF in Table 9.
    /// Decoded value lies in `2..=5`.
    RegularActive,
    /// LBRR frame. Per §4.2.7.3, LBRR frames also use the "Active"
    /// PDF, since every LBRR frame is itself an active-coded frame.
    Lbrr,
}

/// Configuration for the mid-only flag (§4.2.7.2).
///
/// The mid-only flag is present only on a mid-channel SILK frame of a
/// stereo Opus frame when the corresponding side channel is not
/// otherwise required. The caller decides whether this applies and
/// passes the result via [`SilkFrameHeaderConfig::has_mid_only_flag`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilkFrameHeaderConfig {
    /// True if this is the mid-channel SILK frame of a stereo Opus
    /// frame. The stereo prediction weights (§4.2.7.1) are decoded
    /// only when this is true.
    pub stereo_mid_channel: bool,
    /// True if this is a stereo Opus frame at all. When the stereo
    /// bit of the TOC byte is 0, neither the stereo prediction
    /// weights nor the mid-only flag is present.
    pub stereo: bool,
    /// True if the §4.2.7.2 mid-only flag should be decoded. Per
    /// §4.2.7.2, this happens when (a) we are on the mid channel of
    /// a stereo Opus frame, AND (b) the side channel of this time
    /// interval is not otherwise required (regular frame with side
    /// VAD == 0, or LBRR frame with side LBRR == 0).
    pub has_mid_only_flag: bool,
    /// Frame kind for the current SILK frame (regular vs LBRR; if
    /// regular, the VAD state). Drives the §4.2.7.3 PDF selection.
    pub kind: FrameKind,
    /// Audio bandwidth of the SILK signal. Drives the §4.2.7.5.1 PDF
    /// selection (NB / MB share a row in Table 14, WB has its own).
    /// SWB and FB SILK do not exist; the caller passes the SILK-layer
    /// bandwidth post-§4.2.2 split.
    pub bandwidth: Bandwidth,
}

/// SILK frame header — the prefix of side-information that drives the
/// rest of the SILK frame decoder. RFC 6716 §4.2.7.1 through §4.2.7.5.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilkFrameHeader {
    /// Stereo prediction weights (§4.2.7.1) if this is a mid-channel
    /// SILK frame of a stereo Opus frame; `None` otherwise.
    pub stereo_pred: Option<StereoPredictionWeights>,
    /// Mid-only flag (§4.2.7.2): `Some(true)` means the side channel
    /// of this time interval is skipped; `Some(false)` means the side
    /// channel is coded normally; `None` means the flag was not
    /// present.
    pub mid_only_flag: Option<bool>,
    /// Raw §4.2.7.3 frame-type symbol value in `0..=5` (per Table 10).
    pub frame_type: u8,
    /// Decoded signal type (§4.2.7.3, Table 10).
    pub signal_type: SignalType,
    /// Decoded quantization-offset type (§4.2.7.3, Table 10).
    pub qoff_type: QuantizationOffsetType,
    /// Normalized LSF stage-1 codebook index `I1` (§4.2.7.5.1), in
    /// `0..32`.
    pub lsf_stage1: u8,
}

/// Table 6 stage-1 PDF (25 symbols) — `silk_stereo_pred_joint_iCDF`
/// equivalent expressed as an inverse-CDF for the §4.1.3.3 primitive.
///
/// The PDF as stated in RFC 6716 Table 6:
///
/// ```text
/// {7, 2, 1, 1, 1, 10, 24, 8, 1, 1, 3, 23, 92, 23, 3, 1, 1,
///  8, 24, 10, 1, 1, 1, 2, 7}/256
/// ```
///
/// Cumulative `fh[k]` running sum:
/// `[7,9,10,11,12,22,46,54,55,56,59,82,174,197,200,201,202,210,234,
///   244,245,246,247,249,256]`.
/// `icdf[k] = 256 - fh[k]`, terminated by 0:
const STEREO_STAGE1_ICDF: &[u8] = &[
    249, 247, 246, 245, 244, 234, 210, 202, 201, 200, 197, 174, 82, 59, 56, 55, 54, 46, 22, 12, 11,
    10, 9, 7, 0,
];

/// Table 6 stage-2 PDF — `{85, 86, 85}/256`. Cumulative `fh = [85, 171,
/// 256]`. `icdf = [171, 85, 0]`.
const STEREO_STAGE2_ICDF: &[u8] = &[171, 85, 0];

/// Table 6 stage-3 PDF — `{51, 51, 52, 51, 51}/256`. Cumulative
/// `fh = [51, 102, 154, 205, 256]`. `icdf = [205, 154, 102, 51, 0]`.
const STEREO_STAGE3_ICDF: &[u8] = &[205, 154, 102, 51, 0];

/// Table 7 — 16-entry weight table indexed by `wi0` / `wi1 + 1` in the
/// `w0_Q13` / `w1_Q13` computation of §4.2.7.1.
///
/// Last entry is included even though `wi*` only ranges over 0..=14, so
/// that the linear interpolation `w_Q13[wi+1] - w_Q13[wi]` is always
/// defined.
const STEREO_WEIGHT_Q13: [i32; 16] = [
    -13732, -10050, -8266, -7526, -6500, -5000, -2950, -820, 820, 2950, 5000, 6500, 7526, 8266,
    10050, 13732,
];

/// Table 8 mid-only flag PDF `{192, 64}/256`. Cumulative `fh = [192,
/// 256]`. `icdf = [64, 0]`.
const MID_ONLY_ICDF: &[u8] = &[64, 0];

/// Table 9 inactive frame-type PDF `{26, 230, 0, 0, 0, 0}/256` — only
/// indices 0 and 1 ever decode. Cumulative `fh = [26, 256]`. `icdf =
/// [230, 0]`.
const FRAME_TYPE_INACTIVE_ICDF: &[u8] = &[230, 0];

/// Table 9 active frame-type PDF `{0, 0, 24, 74, 148, 10}/256` —
/// indices 2..=5. Cumulative `fh = [0, 0, 24, 98, 246, 256]`. The
/// §4.1.3.3 primitive needs the leading zero-mass cells to be
/// representable; `icdf = [256, 256, 232, 158, 10, 0]` but 256 is not a
/// valid `u8`. Instead, the §4.1.3.3 formulation handles the
/// degenerate "this cell has probability zero" case naturally: the
/// `s * icdf[k]` product equals `rng` when `icdf[k] == ft`, and
/// `val < rng` always holds — so the search just falls through. We
/// approximate the leading-256 entries with their wraparound `0u8`
/// representation, since the §4.1.3.3 primitive compares `val >= next`
/// where `next = s * icdf[k]`; for `icdf[k] = 0` this gives `next = 0`
/// and the loop returns at this index. That is the WRONG behaviour
/// for a leading zero-probability cell. The clean solution is to use
/// a different `ftb` that excludes the zero-probability cells, which
/// gives us a tight inverse-CDF: the active PDF "really" has support
/// over {2, 3, 4, 5}, so transcribe it as a 4-entry table indexed
/// `0..=3` and add the +2 offset in the caller. Cumulative
/// `fh = [24, 98, 246, 256]`. `icdf = [232, 158, 10, 0]`.
const FRAME_TYPE_ACTIVE_ICDF: &[u8] = &[232, 158, 10, 0];

/// Table 14 LSF stage-1 PDF for NB/MB inactive-or-unvoiced. Sum of
/// the 32 cells is 256 by construction. Build cumulative `fh` then
/// `icdf = 256 - fh` with a trailing zero.
const LSF_STAGE1_NB_MB_INACTIVE_PDF: [u8; 32] = [
    44, 34, 30, 19, 21, 12, 11, 3, 3, 2, 16, 2, 2, 1, 5, 2, 1, 3, 3, 1, 1, 2, 2, 2, 3, 1, 9, 9, 2,
    7, 2, 1,
];

/// Table 14 LSF stage-1 PDF for NB/MB voiced.
const LSF_STAGE1_NB_MB_VOICED_PDF: [u8; 32] = [
    1, 10, 1, 8, 3, 8, 8, 14, 13, 14, 1, 14, 12, 13, 11, 11, 12, 11, 10, 10, 11, 8, 9, 8, 7, 8, 1,
    1, 6, 1, 6, 5,
];

/// Table 14 LSF stage-1 PDF for WB inactive-or-unvoiced.
const LSF_STAGE1_WB_INACTIVE_PDF: [u8; 32] = [
    31, 21, 3, 17, 1, 8, 17, 4, 1, 18, 16, 4, 2, 3, 1, 10, 1, 3, 16, 11, 16, 2, 2, 3, 2, 11, 1, 4,
    9, 8, 7, 3,
];

/// Table 14 LSF stage-1 PDF for WB voiced.
const LSF_STAGE1_WB_VOICED_PDF: [u8; 32] = [
    1, 4, 16, 5, 18, 11, 5, 14, 15, 1, 3, 12, 13, 14, 14, 6, 14, 12, 2, 6, 1, 12, 12, 11, 10, 3,
    10, 5, 1, 1, 1, 3,
];

/// Convert a 32-cell length-256 PDF into an iCDF (`256 - fh[k]`) with
/// a trailing zero — the format consumed by [`RangeDecoder::dec_icdf`].
const fn pdf_to_icdf32(pdf: &[u8; 32]) -> [u8; 33] {
    let mut icdf = [0u8; 33];
    let mut acc: u32 = 0;
    let mut k = 0;
    while k < 32 {
        acc += pdf[k] as u32;
        // `256 - acc` fits in u8 as long as acc <= 256, which it is
        // for any well-formed Table-14 row (sum = 256).
        icdf[k] = (256 - acc) as u8;
        k += 1;
    }
    // trailing zero terminator
    icdf[32] = 0;
    icdf
}

/// LSF stage-1 iCDFs derived from the four PDF rows in Table 14.
const LSF_STAGE1_ICDF_NB_MB_INACTIVE: [u8; 33] = pdf_to_icdf32(&LSF_STAGE1_NB_MB_INACTIVE_PDF);
const LSF_STAGE1_ICDF_NB_MB_VOICED: [u8; 33] = pdf_to_icdf32(&LSF_STAGE1_NB_MB_VOICED_PDF);
const LSF_STAGE1_ICDF_WB_INACTIVE: [u8; 33] = pdf_to_icdf32(&LSF_STAGE1_WB_INACTIVE_PDF);
const LSF_STAGE1_ICDF_WB_VOICED: [u8; 33] = pdf_to_icdf32(&LSF_STAGE1_WB_VOICED_PDF);

/// The §4.2.7.1–§4.2.7.3 header symbols that precede the §4.2.7.4
/// subframe gains in the Table-5 read order, plus the derived signal /
/// quantization-offset type.
///
/// Returned by [`SilkFrameHeader::decode_pre_gains`]; the caller reads
/// the §4.2.7.4 gains next, then the §4.2.7.5.1 LSF stage-1 index via
/// [`SilkFrameHeader::decode_lsf_stage1`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilkHeaderPreGains {
    /// §4.2.7.1 stereo prediction weights (mid channel of a stereo Opus
    /// frame); `None` otherwise.
    pub stereo_pred: Option<StereoPredictionWeights>,
    /// §4.2.7.2 mid-only flag; `None` when not present.
    pub mid_only_flag: Option<bool>,
    /// Raw §4.2.7.3 frame-type symbol in `0..=5`.
    pub frame_type: u8,
    /// Decoded signal type (§4.2.7.3, Table 10).
    pub signal_type: SignalType,
    /// Decoded quantization-offset type (§4.2.7.3, Table 10).
    pub qoff_type: QuantizationOffsetType,
}

impl SilkFrameHeader {
    /// Decode the §4.2.7.1–§4.2.7.5.1 header prefix from `rd`.
    ///
    /// The caller is responsible for telling us, via `cfg`, whether
    /// the stereo prediction weights and the mid-only flag are
    /// present, and whether the current frame is regular-inactive,
    /// regular-active, or LBRR. The function does not consult the
    /// §3.1 TOC byte or the §4.2.3/§4.2.4 packet-level header bits.
    /// Decode the §4.2.7.1–§4.2.7.5.1 header fields as a single unit.
    ///
    /// **Note on read order:** this convenience entry reads the LSF
    /// stage-1 index (§4.2.7.5.1) immediately after the frame type
    /// (§4.2.7.3), i.e. it does *not* leave room for the §4.2.7.4
    /// subframe gains that Table 5 places between them. It is therefore
    /// only correct on a bitstream that has no gains symbol, and exists
    /// as a header-field utility / test helper. A full SILK frame decode
    /// must instead use the Table-5-ordered composable entries
    /// [`Self::decode_pre_gains`] (stereo weights + mid-only + frame
    /// type) and [`Self::decode_lsf_stage1`] (the §4.2.7.5.1 index),
    /// reading the §4.2.7.4 gains in between — see
    /// [`crate::silk_decode`].
    pub fn decode(rd: &mut RangeDecoder<'_>, cfg: SilkFrameHeaderConfig) -> Result<Self, Error> {
        let pre = Self::decode_pre_gains(rd, cfg)?;
        let lsf_stage1 = Self::decode_lsf_stage1(rd, cfg.bandwidth, pre.signal_type)?;

        if rd.has_error() {
            return Err(Error::MalformedPacket);
        }

        Ok(Self {
            stereo_pred: pre.stereo_pred,
            mid_only_flag: pre.mid_only_flag,
            frame_type: pre.frame_type,
            signal_type: pre.signal_type,
            qoff_type: pre.qoff_type,
            lsf_stage1,
        })
    }

    /// Decode the §4.2.7.1 stereo prediction weights, §4.2.7.2 mid-only
    /// flag, and §4.2.7.3 frame type — every header symbol that precedes
    /// the §4.2.7.4 subframe gains in the Table-5 read order.
    ///
    /// The returned [`SilkHeaderPreGains`] carries the decoded fields plus
    /// the derived `(signal_type, qoff_type)` needed to choose the gains
    /// PDF and the §4.2.7.5.1 LSF stage-1 PDF. The caller reads the gains
    /// next, then calls [`Self::decode_lsf_stage1`].
    pub fn decode_pre_gains(
        rd: &mut RangeDecoder<'_>,
        cfg: SilkFrameHeaderConfig,
    ) -> Result<SilkHeaderPreGains, Error> {
        // -------- §4.2.7.1 Stereo Prediction Weights --------
        let stereo_pred = if cfg.stereo && cfg.stereo_mid_channel {
            Some(Self::decode_stereo_pred(rd))
        } else {
            None
        };

        // -------- §4.2.7.2 Mid-Only Flag --------
        // Per §4.2.7.2 the flag is present iff (stereo Opus frame) AND
        // (mid channel) AND (side channel not otherwise required). We
        // gate strictly on the caller's `has_mid_only_flag` to keep
        // the LBRR / VAD logic out of the SILK frame decoder.
        let mid_only_flag = if cfg.has_mid_only_flag {
            // Table 8: P(0) = 192/256 = 3/4, P(1) = 64/256 = 1/4.
            // dec_icdf returns the symbol index; index 0 => flag = 0,
            // index 1 => flag = 1 ("mid only").
            let v = rd.dec_icdf(MID_ONLY_ICDF, 8);
            Some(v == 1)
        } else {
            None
        };

        // -------- §4.2.7.3 Frame Type --------
        let frame_type_raw = match cfg.kind {
            FrameKind::RegularInactive => {
                // Inactive PDF — only indices 0 and 1 ever decode.
                rd.dec_icdf(FRAME_TYPE_INACTIVE_ICDF, 8) as u8
            }
            FrameKind::RegularActive | FrameKind::Lbrr => {
                // Active PDF — indices 2..=5. We use a 4-entry iCDF
                // covering the support and shift by +2.
                let k = rd.dec_icdf(FRAME_TYPE_ACTIVE_ICDF, 8) as u8;
                k + 2
            }
        };
        if frame_type_raw > 5 {
            // Should not happen for well-formed PDFs; defend anyway.
            return Err(Error::MalformedPacket);
        }
        let (signal_type, qoff_type) = frame_type_to_signal_qoff(frame_type_raw);

        Ok(SilkHeaderPreGains {
            stereo_pred,
            mid_only_flag,
            frame_type: frame_type_raw,
            signal_type,
            qoff_type,
        })
    }

    /// Decode the §4.2.7.5.1 normalized LSF stage-1 index `I1 ∈ 0..32`.
    ///
    /// The PDF is selected by `(bandwidth, signal_type)` per Table 14;
    /// `signal_type` comes from [`Self::decode_pre_gains`]. In the
    /// Table-5 read order this symbol follows the §4.2.7.4 subframe
    /// gains.
    pub fn decode_lsf_stage1(
        rd: &mut RangeDecoder<'_>,
        bandwidth: Bandwidth,
        signal_type: SignalType,
    ) -> Result<u8, Error> {
        let lsf_icdf: &[u8] = match (bandwidth, signal_type) {
            (Bandwidth::Nb | Bandwidth::Mb, SignalType::Inactive | SignalType::Unvoiced) => {
                &LSF_STAGE1_ICDF_NB_MB_INACTIVE
            }
            (Bandwidth::Nb | Bandwidth::Mb, SignalType::Voiced) => &LSF_STAGE1_ICDF_NB_MB_VOICED,
            (Bandwidth::Wb, SignalType::Inactive | SignalType::Unvoiced) => {
                &LSF_STAGE1_ICDF_WB_INACTIVE
            }
            (Bandwidth::Wb, SignalType::Voiced) => &LSF_STAGE1_ICDF_WB_VOICED,
            // §2 — SILK does not operate on SWB or FB. Hybrid mode
            // splits the signal so that the SILK layer always sees
            // NB / MB / WB only. Reject anything else.
            _ => return Err(Error::MalformedPacket),
        };
        let lsf_stage1 = rd.dec_icdf(lsf_icdf, 8) as u8;
        if lsf_stage1 >= 32 {
            return Err(Error::MalformedPacket);
        }
        Ok(lsf_stage1)
    }

    /// Internal: decode the five sub-symbols of §4.2.7.1 (`n`, `i0`,
    /// `i1`, `i2`, `i3`) and compose them into `(w0_Q13, w1_Q13)`.
    ///
    /// Reads order is exactly the one stated in §4.2.7.1: "let i0
    /// and i1 be indices decoded with the stage-2 and stage-3 PDFs in
    /// Table 6, respectively, and let i2 and i3 be two more indices
    /// decoded with the stage-2 and stage-3 PDFs, all in that order."
    fn decode_stereo_pred(rd: &mut RangeDecoder<'_>) -> StereoPredictionWeights {
        let n = rd.dec_icdf(STEREO_STAGE1_ICDF, 8) as u8;
        let i0 = rd.dec_icdf(STEREO_STAGE2_ICDF, 8) as u8;
        let i1 = rd.dec_icdf(STEREO_STAGE3_ICDF, 8) as u8;
        let i2 = rd.dec_icdf(STEREO_STAGE2_ICDF, 8) as u8;
        let i3 = rd.dec_icdf(STEREO_STAGE3_ICDF, 8) as u8;
        StereoWeightSymbols { n, i0, i1, i2, i3 }.weights()
    }

    // ----- §4.2.7.1–§4.2.7.5.1 encode-side mirrors ------------------

    /// Encode the §4.2.7.1 stereo prediction weights, §4.2.7.2 mid-only
    /// flag, and §4.2.7.3 frame type — the exact write-side mirror of
    /// [`Self::decode_pre_gains`].
    ///
    /// `symbols` supplies the raw symbol choices; `cfg` must describe
    /// the same conditions the decoder will use, and the two must be
    /// consistent (stereo weight symbols present iff `cfg.stereo &&
    /// cfg.stereo_mid_channel`, a mid-only flag present iff
    /// `cfg.has_mid_only_flag`, and a frame type inside the support of
    /// the `cfg.kind`-selected Table 9 PDF). Returns the
    /// [`SilkHeaderPreGains`] the decoder will reconstruct.
    pub fn encode_pre_gains(
        re: &mut RangeEncoder,
        cfg: SilkFrameHeaderConfig,
        symbols: &SilkHeaderSymbols,
    ) -> Result<SilkHeaderPreGains, Error> {
        // -------- §4.2.7.1 Stereo Prediction Weights --------
        let want_stereo = cfg.stereo && cfg.stereo_mid_channel;
        if want_stereo != symbols.stereo.is_some() {
            return Err(Error::MalformedPacket);
        }
        let stereo_pred = match &symbols.stereo {
            Some(s) => {
                if s.n > 24 || s.i0 > 2 || s.i1 > 4 || s.i2 > 2 || s.i3 > 4 {
                    return Err(Error::MalformedPacket);
                }
                re.enc_icdf(s.n as usize, STEREO_STAGE1_ICDF, 8);
                re.enc_icdf(s.i0 as usize, STEREO_STAGE2_ICDF, 8);
                re.enc_icdf(s.i1 as usize, STEREO_STAGE3_ICDF, 8);
                re.enc_icdf(s.i2 as usize, STEREO_STAGE2_ICDF, 8);
                re.enc_icdf(s.i3 as usize, STEREO_STAGE3_ICDF, 8);
                Some(s.weights())
            }
            None => None,
        };

        // -------- §4.2.7.2 Mid-Only Flag --------
        if cfg.has_mid_only_flag != symbols.mid_only_flag.is_some() {
            return Err(Error::MalformedPacket);
        }
        if let Some(flag) = symbols.mid_only_flag {
            re.enc_icdf(flag as usize, MID_ONLY_ICDF, 8);
        }

        // -------- §4.2.7.3 Frame Type --------
        match cfg.kind {
            FrameKind::RegularInactive => {
                if symbols.frame_type > 1 {
                    return Err(Error::MalformedPacket);
                }
                re.enc_icdf(symbols.frame_type as usize, FRAME_TYPE_INACTIVE_ICDF, 8);
            }
            FrameKind::RegularActive | FrameKind::Lbrr => {
                if !(2..=5).contains(&symbols.frame_type) {
                    return Err(Error::MalformedPacket);
                }
                re.enc_icdf((symbols.frame_type - 2) as usize, FRAME_TYPE_ACTIVE_ICDF, 8);
            }
        }
        let (signal_type, qoff_type) = frame_type_to_signal_qoff(symbols.frame_type);

        Ok(SilkHeaderPreGains {
            stereo_pred,
            mid_only_flag: symbols.mid_only_flag,
            frame_type: symbols.frame_type,
            signal_type,
            qoff_type,
        })
    }

    /// Encode the §4.2.7.5.1 normalized LSF stage-1 index `I1 ∈ 0..32`
    /// — the write-side mirror of [`Self::decode_lsf_stage1`]. The PDF
    /// is selected by `(bandwidth, signal_type)` per Table 14.
    pub fn encode_lsf_stage1(
        re: &mut RangeEncoder,
        bandwidth: Bandwidth,
        signal_type: SignalType,
        lsf_stage1: u8,
    ) -> Result<(), Error> {
        if lsf_stage1 >= 32 {
            return Err(Error::MalformedPacket);
        }
        let lsf_icdf: &[u8] = match (bandwidth, signal_type) {
            (Bandwidth::Nb | Bandwidth::Mb, SignalType::Inactive | SignalType::Unvoiced) => {
                &LSF_STAGE1_ICDF_NB_MB_INACTIVE
            }
            (Bandwidth::Nb | Bandwidth::Mb, SignalType::Voiced) => &LSF_STAGE1_ICDF_NB_MB_VOICED,
            (Bandwidth::Wb, SignalType::Inactive | SignalType::Unvoiced) => {
                &LSF_STAGE1_ICDF_WB_INACTIVE
            }
            (Bandwidth::Wb, SignalType::Voiced) => &LSF_STAGE1_ICDF_WB_VOICED,
            _ => return Err(Error::MalformedPacket),
        };
        re.enc_icdf(lsf_stage1 as usize, lsf_icdf, 8);
        Ok(())
    }
}

/// The raw §4.2.7.1 stereo-weight symbol quintuple `(n, i0, i1, i2,
/// i3)` — the five Table-6 indices that jointly select the Q13
/// prediction-weight pair. Used by the encode-side
/// [`SilkFrameHeader::encode_pre_gains`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StereoWeightSymbols {
    /// Stage-1 joint index, `0..=24`.
    pub n: u8,
    /// First stage-2 index, `0..=2`.
    pub i0: u8,
    /// First stage-3 index, `0..=4`.
    pub i1: u8,
    /// Second stage-2 index, `0..=2`.
    pub i2: u8,
    /// Second stage-3 index, `0..=4`.
    pub i3: u8,
}

impl StereoWeightSymbols {
    /// Compose the §4.2.7.1 weight pair from the five indices — the
    /// normative reconstruction shared by the decode and encode paths.
    pub fn weights(&self) -> StereoPredictionWeights {
        let n = self.n as i32;
        let i1 = self.i1 as i32;
        let i3 = self.i3 as i32;
        // §4.2.7.1: wi0 = i0 + 3*(n/5), wi1 = i2 + 3*(n%5); both fall
        // in 0..=14.
        let wi0 = (self.i0 as i32 + 3 * (n / 5)) as usize;
        let wi1 = (self.i2 as i32 + 3 * (n % 5)) as usize;
        // Defensive clamp: the spec guarantees wi* <= 14 for any
        // (n, i0, i2) tuple, but we still saturate to keep the
        // STEREO_WEIGHT_Q13[wi+1] lookup in-bounds even on a
        // pathologically corrupt frame.
        let wi0 = wi0.min(14);
        let wi1 = wi1.min(14);

        // w1 first (w0 depends on w1):
        //   w1 = w_Q13[wi1] + (((w_Q13[wi1+1] - w_Q13[wi1])*6554) >> 16)*(2*i3+1)
        let step1: i32 =
            (((STEREO_WEIGHT_Q13[wi1 + 1] - STEREO_WEIGHT_Q13[wi1]) * 6554) >> 16) * (2 * i3 + 1);
        let w1_q13 = STEREO_WEIGHT_Q13[wi1] + step1;
        //   w0 = w_Q13[wi0] + (((w_Q13[wi0+1] - w_Q13[wi0])*6554) >> 16)*(2*i1+1) - w1
        let step0: i32 =
            (((STEREO_WEIGHT_Q13[wi0 + 1] - STEREO_WEIGHT_Q13[wi0]) * 6554) >> 16) * (2 * i1 + 1);
        let w0_q13 = STEREO_WEIGHT_Q13[wi0] + step0 - w1_q13;
        StereoPredictionWeights { w0_q13, w1_q13 }
    }

    /// Quantize a target §4.2.7.1 weight pair to the nearest coded
    /// quintuple — the deterministic write-side inverse of
    /// [`Self::weights`].
    ///
    /// The §4.2.7.1 codebook is small (25 stage-1 × 3×5 × 3×5 stage-2/3
    /// index combinations = 5625 quintuples), so this is an exhaustive
    /// argmin of the squared Q13 error `(w0 - t0)² + (w1 - t1)²` over
    /// every quintuple, evaluated through the shared normative
    /// reconstruction — no approximation, no reliance on any structure
    /// beyond what [`Self::weights`] itself defines. Ties keep the
    /// first candidate in `(n, i0, i1, i2, i3)` lexicographic order, so
    /// the result is fully deterministic.
    ///
    /// The achieved pair is available as `quantize(t).weights()`; a
    /// target that is exactly representable (any output of
    /// [`Self::weights`]) reconstructs value-exactly.
    pub fn quantize(target: StereoPredictionWeights) -> StereoWeightSymbols {
        let mut best = StereoWeightSymbols {
            n: 0,
            i0: 0,
            i1: 0,
            i2: 0,
            i3: 0,
        };
        let mut best_err = i64::MAX;
        for n in 0..=24u8 {
            for i0 in 0..=2u8 {
                for i1 in 0..=4u8 {
                    for i2 in 0..=2u8 {
                        for i3 in 0..=4u8 {
                            let cand = StereoWeightSymbols { n, i0, i1, i2, i3 };
                            let w = cand.weights();
                            let e0 = (w.w0_q13 - target.w0_q13) as i64;
                            let e1 = (w.w1_q13 - target.w1_q13) as i64;
                            let err = e0 * e0 + e1 * e1;
                            if err < best_err {
                                best_err = err;
                                best = cand;
                            }
                        }
                    }
                }
            }
        }
        best
    }
}

/// The raw pre-gains header symbol choices consumed by the encode-side
/// [`SilkFrameHeader::encode_pre_gains`] — the §4.2.7.1 stereo-weight
/// quintuple (mid channel of a stereo frame only), the §4.2.7.2
/// mid-only flag (when signalled), and the §4.2.7.3 frame-type symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilkHeaderSymbols {
    /// §4.2.7.1 stereo-weight indices; must be `Some` iff the config
    /// has `stereo && stereo_mid_channel`.
    pub stereo: Option<StereoWeightSymbols>,
    /// §4.2.7.2 mid-only flag; must be `Some` iff the config has
    /// `has_mid_only_flag`.
    pub mid_only_flag: Option<bool>,
    /// §4.2.7.3 frame-type symbol in `0..=5`; `0..=1` for
    /// [`FrameKind::RegularInactive`], `2..=5` for
    /// [`FrameKind::RegularActive`] / [`FrameKind::Lbrr`].
    pub frame_type: u8,
}

/// Map a frame-type symbol (0..=5) to `(signal_type, qoff_type)` per
/// RFC 6716 §4.2.7.3 Table 10.
fn frame_type_to_signal_qoff(frame_type: u8) -> (SignalType, QuantizationOffsetType) {
    let signal = match frame_type {
        0 | 1 => SignalType::Inactive,
        2 | 3 => SignalType::Unvoiced,
        4 | 5 => SignalType::Voiced,
        _ => SignalType::Inactive, // unreachable in practice; defensive
    };
    let qoff = if frame_type % 2 == 0 {
        QuantizationOffsetType::Low
    } else {
        QuantizationOffsetType::High
    };
    (signal, qoff)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Table 6 / 7 / 8 / 9 / 14 PDF→iCDF transcription self-checks.
    //
    // These tests do not exercise the range decoder; they confirm
    // the constant tables match the RFC by checking that each PDF row
    // sums to 256 and that consecutive iCDF cells are strictly
    // monotonically decreasing (a §4.1.3.3 precondition).

    #[test]
    fn stereo_stage1_pdf_sums_to_256() {
        let pdf = [
            7, 2, 1, 1, 1, 10, 24, 8, 1, 1, 3, 23, 92, 23, 3, 1, 1, 8, 24, 10, 1, 1, 1, 2, 7,
        ];
        let sum: u32 = pdf.iter().sum();
        assert_eq!(sum, 256);
        assert_eq!(STEREO_STAGE1_ICDF.len(), pdf.len());
        // iCDF strictly monotone decreasing then terminator zero.
        for w in STEREO_STAGE1_ICDF.windows(2) {
            assert!(w[0] > w[1] || (w[0] == 0 && w[1] == 0));
        }
        assert_eq!(*STEREO_STAGE1_ICDF.last().unwrap(), 0);
    }

    #[test]
    fn stereo_stage2_pdf_self_check() {
        assert_eq!(STEREO_STAGE2_ICDF, &[171u8, 85, 0]);
        assert_eq!(STEREO_STAGE3_ICDF, &[205u8, 154, 102, 51, 0]);
    }

    #[test]
    fn mid_only_pdf_self_check() {
        assert_eq!(MID_ONLY_ICDF, &[64u8, 0]);
    }

    #[test]
    fn lsf_stage1_nb_mb_inactive_sums_to_256() {
        let s: u32 = LSF_STAGE1_NB_MB_INACTIVE_PDF
            .iter()
            .map(|&x| x as u32)
            .sum();
        assert_eq!(s, 256);
    }

    #[test]
    fn lsf_stage1_nb_mb_voiced_sums_to_256() {
        let s: u32 = LSF_STAGE1_NB_MB_VOICED_PDF.iter().map(|&x| x as u32).sum();
        assert_eq!(s, 256);
    }

    #[test]
    fn lsf_stage1_wb_inactive_sums_to_256() {
        let s: u32 = LSF_STAGE1_WB_INACTIVE_PDF.iter().map(|&x| x as u32).sum();
        assert_eq!(s, 256);
    }

    #[test]
    fn lsf_stage1_wb_voiced_sums_to_256() {
        let s: u32 = LSF_STAGE1_WB_VOICED_PDF.iter().map(|&x| x as u32).sum();
        assert_eq!(s, 256);
    }

    #[test]
    fn lsf_stage1_icdf_terminator_is_zero() {
        for icdf in [
            &LSF_STAGE1_ICDF_NB_MB_INACTIVE,
            &LSF_STAGE1_ICDF_NB_MB_VOICED,
            &LSF_STAGE1_ICDF_WB_INACTIVE,
            &LSF_STAGE1_ICDF_WB_VOICED,
        ] {
            assert_eq!(icdf[32], 0, "iCDF must terminate with zero");
            assert_eq!(icdf.len(), 33);
            // Strictly decreasing.
            for w in icdf.windows(2) {
                assert!(
                    w[0] >= w[1],
                    "iCDF must be monotone non-increasing: {:?} -> {:?}",
                    w[0],
                    w[1]
                );
            }
        }
    }

    #[test]
    fn stereo_weight_table_is_symmetric() {
        // Table 7 is symmetric around the middle: w[15-k] == -w[k]
        // for k in 0..=7.
        for k in 0..8 {
            assert_eq!(STEREO_WEIGHT_Q13[15 - k], -STEREO_WEIGHT_Q13[k]);
        }
        assert_eq!(STEREO_WEIGHT_Q13[0], -13732);
        assert_eq!(STEREO_WEIGHT_Q13[15], 13732);
    }

    // --- Table 10 frame-type mapping --------

    #[test]
    fn frame_type_to_signal_qoff_table10() {
        let expected = [
            (0, SignalType::Inactive, QuantizationOffsetType::Low),
            (1, SignalType::Inactive, QuantizationOffsetType::High),
            (2, SignalType::Unvoiced, QuantizationOffsetType::Low),
            (3, SignalType::Unvoiced, QuantizationOffsetType::High),
            (4, SignalType::Voiced, QuantizationOffsetType::Low),
            (5, SignalType::Voiced, QuantizationOffsetType::High),
        ];
        for (ft, sig, q) in expected {
            assert_eq!(frame_type_to_signal_qoff(ft), (sig, q));
        }
    }

    // --- End-to-end: decode against a hand-crafted RangeDecoder.
    //
    // We can't easily construct an arbitrary byte sequence that
    // produces a specific symbol pattern without an encoder, but we
    // CAN check round-trip behaviour: every decoded value must
    // satisfy the spec's range bounds, and the function must not
    // latch the corrupt-frame flag for a non-corrupt input.

    fn mono_inactive_cfg(bw: Bandwidth) -> SilkFrameHeaderConfig {
        SilkFrameHeaderConfig {
            stereo_mid_channel: false,
            stereo: false,
            has_mid_only_flag: false,
            kind: FrameKind::RegularInactive,
            bandwidth: bw,
        }
    }

    fn mono_active_cfg(bw: Bandwidth) -> SilkFrameHeaderConfig {
        SilkFrameHeaderConfig {
            stereo_mid_channel: false,
            stereo: false,
            has_mid_only_flag: false,
            kind: FrameKind::RegularActive,
            bandwidth: bw,
        }
    }

    fn stereo_mid_active_cfg(bw: Bandwidth) -> SilkFrameHeaderConfig {
        SilkFrameHeaderConfig {
            stereo_mid_channel: true,
            stereo: true,
            has_mid_only_flag: true,
            kind: FrameKind::RegularActive,
            bandwidth: bw,
        }
    }

    #[test]
    fn mono_inactive_nb_decode_basic() {
        // A long-enough buffer so the range decoder doesn't immediately
        // start zero-extending past EOF.
        let buf = [
            0x55, 0xAA, 0x33, 0xCC, 0x7F, 0x80, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
            0x12, 0x34,
        ];
        let mut rd = RangeDecoder::new(&buf);
        let hdr = SilkFrameHeader::decode(&mut rd, mono_inactive_cfg(Bandwidth::Nb))
            .expect("decode must succeed");
        // No stereo content.
        assert!(hdr.stereo_pred.is_none());
        assert!(hdr.mid_only_flag.is_none());
        // Inactive frame: ft must be 0 or 1.
        assert!(hdr.frame_type <= 1, "ft={}", hdr.frame_type);
        assert_eq!(hdr.signal_type, SignalType::Inactive);
        assert!(hdr.lsf_stage1 < 32);
    }

    #[test]
    fn mono_active_wb_decode_basic() {
        let buf = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];
        let mut rd = RangeDecoder::new(&buf);
        let hdr = SilkFrameHeader::decode(&mut rd, mono_active_cfg(Bandwidth::Wb))
            .expect("decode must succeed");
        assert!(hdr.stereo_pred.is_none());
        assert!(hdr.mid_only_flag.is_none());
        // Active frame: ft must be 2, 3, 4, or 5.
        assert!((2..=5).contains(&hdr.frame_type), "ft={}", hdr.frame_type);
        assert!(matches!(
            hdr.signal_type,
            SignalType::Unvoiced | SignalType::Voiced
        ));
        assert!(hdr.lsf_stage1 < 32);
    }

    #[test]
    fn stereo_mid_active_includes_pred_and_mid_only() {
        let buf = [
            0xC3, 0x18, 0x42, 0x7F, 0x55, 0xAA, 0x33, 0xCC, 0x77, 0x33, 0x11, 0xAA, 0xDE, 0xAD,
            0xBE, 0xEF, 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
        ];
        let mut rd = RangeDecoder::new(&buf);
        let hdr = SilkFrameHeader::decode(&mut rd, stereo_mid_active_cfg(Bandwidth::Mb))
            .expect("decode must succeed");
        let pred = hdr.stereo_pred.expect("stereo prediction must be present");
        // w1 is one interpolated Table-7 entry (~±14_100 after the
        // §4.2.7.1 interpolation step). w0 then subtracts w1 from
        // another interpolated entry, so |w0| can reach ~ 28_000.
        assert!(
            (-30_000..=30_000).contains(&pred.w0_q13),
            "w0={}",
            pred.w0_q13
        );
        assert!(
            (-15_000..=15_000).contains(&pred.w1_q13),
            "w1={}",
            pred.w1_q13
        );
        assert!(hdr.mid_only_flag.is_some());
        assert!((2..=5).contains(&hdr.frame_type), "ft={}", hdr.frame_type);
        assert!(hdr.lsf_stage1 < 32);
    }

    #[test]
    fn stereo_side_no_prediction() {
        // Side-channel SILK frame: NOT mid channel, so no stereo pred
        // weights and no mid-only flag.
        let cfg = SilkFrameHeaderConfig {
            stereo_mid_channel: false,
            stereo: true,
            has_mid_only_flag: false,
            kind: FrameKind::RegularActive,
            bandwidth: Bandwidth::Wb,
        };
        let buf = [
            0x37, 0x91, 0xC4, 0x18, 0xA2, 0x5D, 0x6E, 0xFF, 0x77, 0x33, 0x11, 0xAA,
        ];
        let mut rd = RangeDecoder::new(&buf);
        let hdr = SilkFrameHeader::decode(&mut rd, cfg).expect("decode must succeed");
        assert!(hdr.stereo_pred.is_none());
        assert!(hdr.mid_only_flag.is_none());
    }

    #[test]
    fn lbrr_frame_uses_active_pdf() {
        // LBRR frames decode the frame-type symbol from the "Active"
        // PDF irrespective of the (regular) VAD state.
        let cfg = SilkFrameHeaderConfig {
            stereo_mid_channel: false,
            stereo: false,
            has_mid_only_flag: false,
            kind: FrameKind::Lbrr,
            bandwidth: Bandwidth::Nb,
        };
        let buf = [
            0x55, 0xAA, 0x33, 0xCC, 0x7F, 0x80, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC,
        ];
        let mut rd = RangeDecoder::new(&buf);
        let hdr = SilkFrameHeader::decode(&mut rd, cfg).expect("decode must succeed");
        assert!(
            (2..=5).contains(&hdr.frame_type),
            "lbrr ft must be active: {}",
            hdr.frame_type
        );
    }

    #[test]
    fn pdf_to_icdf_terminates_and_decreases() {
        // The const helper must produce a strictly-decreasing iCDF
        // with a trailing zero, for any well-formed length-32
        // length-256-sum PDF.
        let pdf = [8u8; 32]; // uniform 32-way: sum = 256.
        let icdf = pdf_to_icdf32(&pdf);
        assert_eq!(icdf[32], 0);
        for w in icdf.windows(2) {
            assert!(w[0] >= w[1]);
        }
        // 256 - 8 = 248 (first cumulative-fh subtraction).
        assert_eq!(icdf[0], 248);
        // After all 32 cells: 256 - 32*8 = 0; uniform PDF sums to ft.
        assert_eq!(icdf[31], 0);
    }

    #[test]
    fn stereo_pred_wi_clamped_in_bounds() {
        // Even in the pathological case where rd.has_error() is set,
        // the wi0/wi1 clamps in decode_stereo_pred ensure we never
        // index past STEREO_WEIGHT_Q13[15]. We can't directly inject
        // n=24, but the clamp `.min(14)` is exercised by inspection;
        // exercise it indirectly by running many random buffers.
        for seed in 0..32u8 {
            let buf = [
                seed,
                seed.wrapping_mul(3),
                seed.wrapping_add(7),
                seed ^ 0xA5,
                seed.wrapping_mul(11),
                seed.wrapping_add(13),
                seed ^ 0x5A,
                seed.wrapping_mul(17),
                seed.wrapping_add(19),
                seed ^ 0xC3,
                seed.wrapping_mul(23),
                seed.wrapping_add(29),
                seed ^ 0x3C,
                seed.wrapping_mul(31),
                seed.wrapping_add(37),
                seed ^ 0x55,
            ];
            let mut rd = RangeDecoder::new(&buf);
            let pred = SilkFrameHeader::decode_stereo_pred(&mut rd);
            // w1 is one interpolated table entry (~±14k); w0 is one
            // interpolated entry MINUS w1 (~±28k worst case).
            assert!(
                (-30_000..=30_000).contains(&pred.w0_q13),
                "seed={seed}, w0={}",
                pred.w0_q13
            );
            assert!(
                (-15_000..=15_000).contains(&pred.w1_q13),
                "seed={seed}, w1={}",
                pred.w1_q13
            );
        }
    }

    // ----- §4.2.7.1–§4.2.7.5.1 encode-side mirrors ------------------

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

    /// encode_pre_gains → decode_pre_gains roundtrip over random symbol
    /// scripts covering mono / stereo-mid / mid-only-flag / all frame
    /// kinds: the decoder must reconstruct exactly the header the
    /// encoder predicted.
    #[test]
    fn header_encode_decode_roundtrip_random_scripts() {
        use crate::range_encoder::RangeEncoder;
        let mut rng = Lcg(0x00AC_E0F5_EED5);
        for _ in 0..500 {
            let stereo_mid = rng.below(2) == 0;
            let has_mid_only = stereo_mid && rng.below(2) == 0;
            let kind = match rng.below(3) {
                0 => FrameKind::RegularInactive,
                1 => FrameKind::RegularActive,
                _ => FrameKind::Lbrr,
            };
            let bandwidth = match rng.below(3) {
                0 => Bandwidth::Nb,
                1 => Bandwidth::Mb,
                _ => Bandwidth::Wb,
            };
            let cfg = SilkFrameHeaderConfig {
                stereo_mid_channel: stereo_mid,
                stereo: stereo_mid,
                has_mid_only_flag: has_mid_only,
                kind,
                bandwidth,
            };
            let symbols = SilkHeaderSymbols {
                stereo: stereo_mid.then(|| StereoWeightSymbols {
                    n: rng.below(25) as u8,
                    i0: rng.below(3) as u8,
                    i1: rng.below(5) as u8,
                    i2: rng.below(3) as u8,
                    i3: rng.below(5) as u8,
                }),
                mid_only_flag: has_mid_only.then(|| rng.below(2) == 1),
                frame_type: match kind {
                    FrameKind::RegularInactive => rng.below(2) as u8,
                    _ => 2 + rng.below(4) as u8,
                },
            };
            let lsf_stage1 = rng.below(32) as u8;

            let mut re = RangeEncoder::new();
            let predicted =
                SilkFrameHeader::encode_pre_gains(&mut re, cfg, &symbols).expect("encode");
            SilkFrameHeader::encode_lsf_stage1(
                &mut re,
                bandwidth,
                predicted.signal_type,
                lsf_stage1,
            )
            .expect("encode lsf1");
            let bytes = re.finish();

            let mut rd = RangeDecoder::new(&bytes);
            let decoded = SilkFrameHeader::decode_pre_gains(&mut rd, cfg).expect("decode");
            assert_eq!(decoded, predicted, "symbols={symbols:?}");
            let got_lsf1 =
                SilkFrameHeader::decode_lsf_stage1(&mut rd, bandwidth, decoded.signal_type)
                    .expect("decode lsf1");
            assert_eq!(got_lsf1, lsf_stage1);
            assert!(!rd.has_error());
        }
    }

    /// The encode path rejects symbol/config mismatches and
    /// out-of-support values.
    #[test]
    fn header_encode_rejects_bad_symbols() {
        use crate::range_encoder::RangeEncoder;
        let mono_cfg = SilkFrameHeaderConfig {
            stereo_mid_channel: false,
            stereo: false,
            has_mid_only_flag: false,
            kind: FrameKind::RegularActive,
            bandwidth: Bandwidth::Nb,
        };
        let stereo_syms = SilkHeaderSymbols {
            stereo: Some(StereoWeightSymbols {
                n: 0,
                i0: 0,
                i1: 0,
                i2: 0,
                i3: 0,
            }),
            mid_only_flag: None,
            frame_type: 4,
        };
        // Stereo symbols on a mono config.
        let mut re = RangeEncoder::new();
        assert!(SilkFrameHeader::encode_pre_gains(&mut re, mono_cfg, &stereo_syms).is_err());
        // Inactive frame type on an active kind.
        let mut re = RangeEncoder::new();
        let bad_type = SilkHeaderSymbols {
            stereo: None,
            mid_only_flag: None,
            frame_type: 1,
        };
        assert!(SilkFrameHeader::encode_pre_gains(&mut re, mono_cfg, &bad_type).is_err());
        // Out-of-range stereo indices.
        let mut re = RangeEncoder::new();
        let stereo_cfg = SilkFrameHeaderConfig {
            stereo_mid_channel: true,
            stereo: true,
            has_mid_only_flag: false,
            kind: FrameKind::RegularActive,
            bandwidth: Bandwidth::Nb,
        };
        let bad_stereo = SilkHeaderSymbols {
            stereo: Some(StereoWeightSymbols {
                n: 25,
                i0: 0,
                i1: 0,
                i2: 0,
                i3: 0,
            }),
            mid_only_flag: None,
            frame_type: 4,
        };
        assert!(SilkFrameHeader::encode_pre_gains(&mut re, stereo_cfg, &bad_stereo).is_err());
        // Out-of-range LSF stage-1 index / SWB bandwidth.
        let mut re = RangeEncoder::new();
        assert!(
            SilkFrameHeader::encode_lsf_stage1(&mut re, Bandwidth::Nb, SignalType::Voiced, 32)
                .is_err()
        );
        let mut re = RangeEncoder::new();
        assert!(
            SilkFrameHeader::encode_lsf_stage1(&mut re, Bandwidth::Swb, SignalType::Voiced, 0)
                .is_err()
        );
    }

    /// Enumerate every §4.2.7.1 quintuple in `(n, i0, i1, i2, i3)`
    /// lexicographic order.
    fn all_weight_quintuples() -> Vec<StereoWeightSymbols> {
        let mut v = Vec::with_capacity(5625);
        for n in 0..=24u8 {
            for i0 in 0..=2u8 {
                for i1 in 0..=4u8 {
                    for i2 in 0..=2u8 {
                        for i3 in 0..=4u8 {
                            v.push(StereoWeightSymbols { n, i0, i1, i2, i3 });
                        }
                    }
                }
            }
        }
        v
    }

    /// A representable target (any output of `weights()`) quantizes back
    /// value-exactly: the achieved pair equals the target pair. Sampled
    /// across the whole codebook (stride 7 keeps the exhaustive
    /// `quantize` affordable while touching every index dimension).
    #[test]
    fn stereo_weight_quantize_exact_on_representable_targets() {
        let all = all_weight_quintuples();
        for q in all.iter().step_by(7) {
            let target = q.weights();
            let quant = StereoWeightSymbols::quantize(target);
            assert_eq!(
                quant.weights(),
                target,
                "quintuple {q:?} did not roundtrip through quantize"
            );
        }
    }

    /// `quantize` is a true argmin: for random unrepresentable targets
    /// no quintuple in the codebook achieves a strictly smaller squared
    /// Q13 error than the returned one, and repeated calls are
    /// deterministic.
    #[test]
    fn stereo_weight_quantize_is_argmin_and_deterministic() {
        let all = all_weight_quintuples();
        let err = |w: StereoPredictionWeights, t: StereoPredictionWeights| -> i64 {
            let e0 = (w.w0_q13 - t.w0_q13) as i64;
            let e1 = (w.w1_q13 - t.w1_q13) as i64;
            e0 * e0 + e1 * e1
        };
        // Small deterministic LCG.
        let mut s = 0x5EED_0385u64;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as i32 % 60001) - 30000
        };
        for _ in 0..25 {
            let target = StereoPredictionWeights {
                w0_q13: next(),
                w1_q13: next(),
            };
            let quant = StereoWeightSymbols::quantize(target);
            let best = err(quant.weights(), target);
            for cand in &all {
                assert!(
                    err(cand.weights(), target) >= best,
                    "candidate {cand:?} beats quantize({target:?}) = {quant:?}"
                );
            }
            assert_eq!(StereoWeightSymbols::quantize(target), quant);
        }
    }

    /// Extreme targets saturate to the codebook's extreme reachable
    /// weight pairs (computed from the codebook itself, not hard-coded).
    #[test]
    fn stereo_weight_quantize_saturates_at_extremes() {
        let all = all_weight_quintuples();
        let max_w1 = all.iter().map(|q| q.weights().w1_q13).max().unwrap();
        let min_w1 = all.iter().map(|q| q.weights().w1_q13).min().unwrap();
        // A target far above every reachable w1 (with w0 target 0) must
        // land on a maximal-w1 quintuple, and symmetrically below.
        let hi = StereoWeightSymbols::quantize(StereoPredictionWeights {
            w0_q13: 0,
            w1_q13: 1 << 20,
        });
        assert_eq!(hi.weights().w1_q13, max_w1);
        let lo = StereoWeightSymbols::quantize(StereoPredictionWeights {
            w0_q13: 0,
            w1_q13: -(1 << 20),
        });
        assert_eq!(lo.weights().w1_q13, min_w1);
    }
}
