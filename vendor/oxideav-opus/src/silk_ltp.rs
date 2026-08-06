//! SILK Long-Term Prediction (LTP) parameters — RFC 6716 §4.2.7.6.
//!
//! After the normalized LSF indices (and, for 20 ms frames, the LSF
//! interpolation index), **voiced** SILK frames (see §4.2.7.3) carry the
//! long-term-prediction parameters that drive the pitch filter:
//!
//!  * **§4.2.7.6.1 Pitch lags.** One primary pitch lag per SILK frame —
//!    coded either as an absolute value (a high part from the Table 29
//!    32-entry codebook plus a bandwidth-dependent low part from Table 30)
//!    or relative to the prior frame's lag (a Table 31 delta, where a
//!    decoded value of zero falls back to absolute coding). A "pitch
//!    contour" VQ index (Table 32 PDF; Tables 33–36 codebooks) then
//!    refines the primary lag into a separate lag per subframe, clamped to
//!    the bandwidth's `[lag_min, lag_max]`.
//!  * **§4.2.7.6.2 LTP filter coefficients.** A single 3-entry
//!    "periodicity index" (Table 37) selects one of three codebooks for
//!    the whole SILK frame; then each subframe reads a filter index from
//!    the periodicity-conditioned Table 38 PDF, yielding a 5-tap Q7 filter
//!    from Tables 39–41.
//!  * **§4.2.7.6.3 LTP scaling.** An optional Q14 scale factor (Table 42).
//!    When present it is one of `{15565, 12288, 8192}`; when absent the
//!    default `15565` (≈0.95) is used.
//!
//! This module decodes all three subsections behind [`LtpParameters`] and
//! its [`LtpConfig`]. It does **not** run the §4.2.7.9 LTP synthesis
//! filter — it only produces the decoded parameters that feed it.
//!
//! All truth is taken from RFC 6716 §4.2.7.6 (Tables 29–42); no external
//! library source is consulted.

use crate::range_decoder::RangeDecoder;
use crate::range_encoder::RangeEncoder;
use crate::silk_frame::SignalType;
use crate::toc::Bandwidth;
use crate::Error;

/// Maximum subframes in a SILK frame (4 for 20 ms / Hybrid; 2 for 10 ms).
pub const LTP_MAX_SUBFRAMES: usize = 4;

/// Number of taps in a SILK LTP pitch filter (§4.2.7.6.2).
pub const LTP_FILTER_TAPS: usize = 5;

// =====================================================================
// Table 29 — PDF for the high part of the absolute primary pitch lag.
//
// {3, 3, 6, 11, 21, 30, 32, 19, 11, 10, 12, 13, 13, 12, 11, 9, 8, 7, 6,
//  4, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1}/256  (32 entries)
//
// Stored as the §4.1.3.3 inverse-CDF form (terminated by 0).
// =====================================================================
const LAG_HIGH_ICDF: &[u8] = &[
    253, 250, 244, 233, 212, 182, 150, 131, 120, 110, 98, 85, 72, 60, 49, 40, 32, 25, 19, 15, 13,
    11, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];

// =====================================================================
// Table 30 — PDF + scale + lag range for the low part of the absolute
// primary pitch lag, per audio bandwidth.
// =====================================================================
const LAG_LOW_ICDF_NB: &[u8] = &[192, 128, 64, 0]; // {64,64,64,64}
const LAG_LOW_ICDF_MB: &[u8] = &[213, 171, 128, 85, 43, 0]; // {43,42,43,43,42,43}
const LAG_LOW_ICDF_WB: &[u8] = &[224, 192, 160, 128, 96, 64, 32, 0]; // eight 32s

// =====================================================================
// Table 31 — PDF for the relative (delta) primary pitch lag.
//
// {46, 2, 2, 3, 4, 6, 10, 15, 26, 38, 30, 22, 15, 10, 7, 6, 4, 4, 2, 2,
//  2}/256  (21 entries)
// =====================================================================
const LAG_DELTA_ICDF: &[u8] = &[
    210, 208, 206, 203, 199, 193, 183, 168, 142, 104, 74, 52, 37, 27, 20, 14, 10, 6, 4, 2, 0,
];

// =====================================================================
// Table 32 — PDFs for the subframe pitch-contour VQ index, selected by
// (bandwidth, SILK frame size).
// =====================================================================
const CONTOUR_ICDF_NB_10MS: &[u8] = &[113, 63, 0]; // {143,50,63}
const CONTOUR_ICDF_NB_20MS: &[u8] = &[
    // {68,12,21,17,19,22,30,24,17,16,10}/256
    188, 176, 155, 138, 119, 97, 67, 43, 26, 10, 0,
];
const CONTOUR_ICDF_MBWB_10MS: &[u8] = &[
    // {91,46,39,19,14,12,8,7,6,5,5,4}/256
    165, 119, 80, 61, 47, 35, 27, 20, 14, 9, 4, 0,
];
const CONTOUR_ICDF_MBWB_20MS: &[u8] = &[
    // {33,22,18,16,15,14,14,13,13,10,9,9,8,6,6,6,5,4,4,4,3,3,3,2,2,2,2,2,
    //  2,2,1,1,1,1}/256
    223, 201, 183, 167, 152, 138, 124, 111, 98, 88, 79, 70, 62, 56, 50, 44, 39, 35, 31, 27, 24, 21,
    18, 16, 14, 12, 10, 8, 6, 4, 3, 2, 1, 0,
];

// =====================================================================
// Tables 33–36 — pitch-contour codebooks. lag_cb[contour_index][k] is
// the offset added to the primary lag for subframe k.
// =====================================================================
const CONTOUR_NB_10MS: &[[i8; 2]; 3] = &[[0, 0], [1, 0], [0, 1]];

const CONTOUR_NB_20MS: &[[i8; 4]; 11] = &[
    [0, 0, 0, 0],
    [2, 1, 0, -1],
    [-1, 0, 1, 2],
    [-1, 0, 0, 1],
    [-1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, -1],
    [1, 0, 0, -1],
];

const CONTOUR_MBWB_10MS: &[[i8; 2]; 12] = &[
    [0, 0],
    [0, 1],
    [1, 0],
    [-1, 1],
    [1, -1],
    [-1, 2],
    [2, -1],
    [-2, 2],
    [2, -2],
    [-2, 3],
    [3, -2],
    [-3, 3],
];

const CONTOUR_MBWB_20MS: &[[i8; 4]; 34] = &[
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [-1, 0, 0, 1],
    [0, 0, 0, -1],
    [-1, 0, 1, 2],
    [1, 0, 0, -1],
    [-2, -1, 1, 2],
    [2, 1, 0, -1],
    [-2, 0, 0, 2],
    [-2, 0, 1, 3],
    [2, 1, -1, -2],
    [-3, -1, 1, 3],
    [2, 0, 0, -2],
    [3, 1, 0, -2],
    [-3, -1, 2, 4],
    [-4, -1, 1, 4],
    [3, 1, -1, -3],
    [-4, -1, 2, 5],
    [4, 2, -1, -3],
    [4, 1, -1, -4],
    [-5, -1, 2, 6],
    [5, 2, -1, -4],
    [-6, -2, 2, 6],
    [-5, -2, 2, 5],
    [6, 2, -1, -5],
    [-7, -2, 3, 8],
    [6, 2, -2, -6],
    [5, 2, -2, -5],
    [8, 3, -2, -7],
    [-9, -3, 3, 9],
];

// =====================================================================
// Table 37 — periodicity-index PDF: {77, 80, 99}/256.
// =====================================================================
const PERIODICITY_ICDF: &[u8] = &[179, 99, 0];

// =====================================================================
// Table 38 — LTP filter-index PDFs, selected by periodicity index.
// =====================================================================
const LTP_FILTER_ICDF_P0: &[u8] = &[
    // {185,15,13,13,9,9,6,6}/256
    71, 56, 43, 30, 21, 12, 6, 0,
];
const LTP_FILTER_ICDF_P1: &[u8] = &[
    // {57,34,21,20,15,13,12,13,10,10,9,10,9,8,7,8}/256
    199, 165, 144, 124, 109, 96, 84, 71, 61, 51, 42, 32, 23, 15, 8, 0,
];
const LTP_FILTER_ICDF_P2: &[u8] = &[
    // {15,16,14,12,12,12,11,11,11,10,9,9,9,9,8,8,8,8,7,7,6,6,5,4,5,4,4,4,
    //  3,4,3,2}/256
    241, 225, 211, 199, 187, 175, 164, 153, 142, 132, 123, 114, 105, 96, 88, 80, 72, 64, 57, 50, 44,
    38, 33, 29, 24, 20, 16, 12, 9, 5, 2, 0,
];

// =====================================================================
// Tables 39–41 — LTP filter codebooks (5 signed Q7 taps per index),
// selected by periodicity index. Codebook sizes are 8 / 16 / 32.
// =====================================================================
const LTP_TAPS_P0: &[[i8; LTP_FILTER_TAPS]; 8] = &[
    [4, 6, 24, 7, 5],
    [0, 0, 2, 0, 0],
    [12, 28, 41, 13, -4],
    [-9, 15, 42, 25, 14],
    [1, -2, 62, 41, -9],
    [-10, 37, 65, -4, 3],
    [-6, 4, 66, 7, -8],
    [16, 14, 38, -3, 33],
];

const LTP_TAPS_P1: &[[i8; LTP_FILTER_TAPS]; 16] = &[
    [13, 22, 39, 23, 12],
    [-1, 36, 64, 27, -6],
    [-7, 10, 55, 43, 17],
    [1, 1, 8, 1, 1],
    [6, -11, 74, 53, -9],
    [-12, 55, 76, -12, 8],
    [-3, 3, 93, 27, -4],
    [26, 39, 59, 3, -8],
    [2, 0, 77, 11, 9],
    [-8, 22, 44, -6, 7],
    [40, 9, 26, 3, 9],
    [-7, 20, 101, -7, 4],
    [3, -8, 42, 26, 0],
    [-15, 33, 68, 2, 23],
    [-2, 55, 46, -2, 15],
    [3, -1, 21, 16, 41],
];

const LTP_TAPS_P2: &[[i8; LTP_FILTER_TAPS]; 32] = &[
    [-6, 27, 61, 39, 5],
    [-11, 42, 88, 4, 1],
    [-2, 60, 65, 6, -4],
    [-1, -5, 73, 56, 1],
    [-9, 19, 94, 29, -9],
    [0, 12, 99, 6, 4],
    [8, -19, 102, 46, -13],
    [3, 2, 13, 3, 2],
    [9, -21, 84, 72, -18],
    [-11, 46, 104, -22, 8],
    [18, 38, 48, 23, 0],
    [-16, 70, 83, -21, 11],
    [5, -11, 117, 22, -8],
    [-6, 23, 117, -12, 3],
    [3, -8, 95, 28, 4],
    [-10, 15, 77, 60, -15],
    [-1, 4, 124, 2, -4],
    [3, 38, 84, 24, -25],
    [2, 13, 42, 13, 31],
    [21, -4, 56, 46, -1],
    [-1, 35, 79, -13, 19],
    [-7, 65, 88, -9, -14],
    [20, 4, 81, 49, -29],
    [20, 0, 75, 3, -17],
    [5, -9, 44, 92, -8],
    [1, -3, 22, 69, 31],
    [-6, 95, 41, -12, 5],
    [39, 67, 16, -4, 1],
    [0, -6, 120, 55, -36],
    [-13, 44, 122, 4, -24],
    [81, 5, 11, 3, 7],
    [2, 0, 9, 10, 88],
];

// =====================================================================
// Table 42 — LTP scaling PDF: {128, 64, 64}/256, mapping the decoded
// index to a Q14 scale factor.
// =====================================================================
const LTP_SCALING_ICDF: &[u8] = &[128, 64, 0];
const LTP_SCALING_VALUES_Q14: [u16; 3] = [15565, 12288, 8192];

/// Default Q14 LTP scaling factor (≈0.95) for frames that do not code one.
pub const LTP_SCALING_DEFAULT_Q14: u16 = 15565;

/// Per-bandwidth low-part PDF + scale + lag range from Table 30.
struct LagLowSpec {
    icdf: &'static [u8],
    scale: i32,
    lag_min: i32,
    lag_max: i32,
}

fn lag_low_spec(bandwidth: Bandwidth) -> Result<LagLowSpec, Error> {
    match bandwidth {
        Bandwidth::Nb => Ok(LagLowSpec {
            icdf: LAG_LOW_ICDF_NB,
            scale: 4,
            lag_min: 16,
            lag_max: 144,
        }),
        Bandwidth::Mb => Ok(LagLowSpec {
            icdf: LAG_LOW_ICDF_MB,
            scale: 6,
            lag_min: 24,
            lag_max: 216,
        }),
        Bandwidth::Wb => Ok(LagLowSpec {
            icdf: LAG_LOW_ICDF_WB,
            scale: 8,
            lag_min: 32,
            lag_max: 288,
        }),
        // SILK only operates on NB / MB / WB internal bandwidths.
        Bandwidth::Swb | Bandwidth::Fb => Err(Error::MalformedPacket),
    }
}

/// How the primary pitch lag is coded for this SILK frame (§4.2.7.6.1).
///
/// Absolute coding is used iff this is the first SILK frame of its type
/// for the channel in the current Opus frame, OR the previous SILK frame
/// of the same type was not coded, OR that previous frame was coded but
/// not voiced. Everything else uses relative coding against
/// `previous_lag` — and even relative coding falls back to absolute when
/// the decoded delta is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagCoding {
    /// Absolute coding: high part (Table 29) + low part (Table 30).
    Absolute,
    /// Relative coding: a Table 31 delta against `previous_lag`. A decoded
    /// delta of zero falls back to absolute coding.
    Relative {
        /// The primary pitch lag from the most recent frame in the same
        /// channel. Used as the base for the delta; carried unclamped per
        /// the §4.2.7.6.1 note.
        previous_lag: i32,
    },
}

/// Inputs to [`LtpParameters::decode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtpConfig {
    /// SILK-layer audio bandwidth (NB / MB / WB). Selects the Table 30
    /// low-part codebook + scale + lag range and the Table 32 contour PDF.
    pub bandwidth: Bandwidth,
    /// Signal type from §4.2.7.3. LTP parameters are present only for
    /// [`SignalType::Voiced`] frames; any other value yields an empty
    /// result with no bitstream reads.
    pub signal_type: SignalType,
    /// Number of subframes in this SILK frame: 2 (10 ms) or 4 (20 ms /
    /// Hybrid). Selects the Table 32 contour PDF and the contour codebook
    /// width; other values are rejected.
    pub num_subframes: u8,
    /// How the primary lag is coded (§4.2.7.6.1).
    pub lag_coding: LagCoding,
    /// Whether the §4.2.7.6.3 LTP scaling field is present in the
    /// bitstream. The caller computes this from the §4.2.7.6.3
    /// enumeration (first time interval of the Opus frame for its type,
    /// or an LBRR frame whose prior LBRR frame is not coded). When
    /// `false` the default Q14 factor (`15565`) is used and no symbol is
    /// read.
    pub ltp_scaling_present: bool,
}

/// Decoded §4.2.7.6 LTP parameters for one SILK frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtpParameters {
    /// `true` when this is a voiced frame carrying real LTP parameters.
    /// `false` (with all other fields zeroed / default) for non-voiced
    /// frames, where no LTP symbols are present in the bitstream.
    voiced: bool,
    /// The primary pitch lag for the SILK frame (the unclamped value used
    /// as `previous_lag` for the next frame's relative coding).
    primary_lag: i32,
    /// The decoded pitch-contour VQ index.
    contour_index: u8,
    /// Per-subframe pitch lags, clamped to `[lag_min, lag_max]`. Only
    /// `0..len` entries are populated.
    pitch_lags: [i32; LTP_MAX_SUBFRAMES],
    /// The decoded periodicity index `0..=2` (§4.2.7.6.2).
    periodicity_index: u8,
    /// Per-subframe 5-tap Q7 LTP filter coefficients. Only `0..len` rows
    /// are populated.
    filter_taps_q7: [[i8; LTP_FILTER_TAPS]; LTP_MAX_SUBFRAMES],
    /// The decoded per-subframe filter indices into the periodicity's
    /// codebook. Only `0..len` entries are populated.
    filter_indices: [u8; LTP_MAX_SUBFRAMES],
    /// The Q14 LTP scaling factor (§4.2.7.6.3); the default `15565` when
    /// not coded.
    ltp_scaling_q14: u16,
    len: u8,
}

impl LtpParameters {
    /// Decode the §4.2.7.6 LTP parameters from `rd`.
    ///
    /// For a non-voiced frame this consumes no bits and returns an empty
    /// (`!is_voiced()`) result. For a voiced frame it decodes, in order:
    /// the §4.2.7.6.1 primary lag + contour → per-subframe pitch lags,
    /// the §4.2.7.6.2 periodicity index + per-subframe filter indices →
    /// 5-tap Q7 filters, and finally the §4.2.7.6.3 LTP scaling factor
    /// (when `cfg.ltp_scaling_present`).
    ///
    /// Returns `Error::MalformedPacket` if `cfg.num_subframes` is not 2
    /// or 4, or if `cfg.bandwidth` is not an internal SILK bandwidth
    /// (NB / MB / WB).
    pub fn decode(rd: &mut RangeDecoder<'_>, cfg: LtpConfig) -> Result<Self, Error> {
        if cfg.num_subframes != 2 && cfg.num_subframes != 4 {
            return Err(Error::MalformedPacket);
        }
        let spec = lag_low_spec(cfg.bandwidth)?;
        let num = cfg.num_subframes as usize;

        // Non-voiced frames carry no LTP parameters at all.
        if cfg.signal_type != SignalType::Voiced {
            return Ok(Self {
                voiced: false,
                primary_lag: 0,
                contour_index: 0,
                pitch_lags: [0; LTP_MAX_SUBFRAMES],
                periodicity_index: 0,
                filter_taps_q7: [[0; LTP_FILTER_TAPS]; LTP_MAX_SUBFRAMES],
                filter_indices: [0; LTP_MAX_SUBFRAMES],
                ltp_scaling_q14: LTP_SCALING_DEFAULT_Q14,
                len: cfg.num_subframes,
            });
        }

        // --- §4.2.7.6.1 primary pitch lag -----------------------------
        let primary_lag = Self::decode_primary_lag(rd, &spec, cfg.lag_coding);

        // --- §4.2.7.6.1 pitch contour → per-subframe lags -------------
        let contour_index = Self::decode_contour_index(rd, cfg.bandwidth, num);
        let mut pitch_lags = [0i32; LTP_MAX_SUBFRAMES];
        for (k, slot) in pitch_lags.iter_mut().enumerate().take(num) {
            let offset = contour_offset(cfg.bandwidth, num, contour_index, k) as i32;
            *slot = (primary_lag + offset).clamp(spec.lag_min, spec.lag_max);
        }

        // --- §4.2.7.6.2 LTP filter coefficients -----------------------
        let periodicity_index = rd.dec_icdf(PERIODICITY_ICDF, 8) as u8;
        let mut filter_indices = [0u8; LTP_MAX_SUBFRAMES];
        let mut filter_taps_q7 = [[0i8; LTP_FILTER_TAPS]; LTP_MAX_SUBFRAMES];
        let filter_icdf = ltp_filter_icdf(periodicity_index);
        for k in 0..num {
            let idx = rd.dec_icdf(filter_icdf, 8) as u8;
            filter_indices[k] = idx;
            filter_taps_q7[k] = ltp_filter_taps(periodicity_index, idx);
        }

        // --- §4.2.7.6.3 LTP scaling -----------------------------------
        let ltp_scaling_q14 = if cfg.ltp_scaling_present {
            let s = rd.dec_icdf(LTP_SCALING_ICDF, 8) as usize;
            LTP_SCALING_VALUES_Q14[s]
        } else {
            LTP_SCALING_DEFAULT_Q14
        };

        Ok(Self {
            voiced: true,
            primary_lag,
            contour_index,
            pitch_lags,
            periodicity_index,
            filter_taps_q7,
            filter_indices,
            ltp_scaling_q14,
            len: cfg.num_subframes,
        })
    }

    /// Encode the §4.2.7.6 LTP parameters into `re` — the exact
    /// write-side mirror of [`Self::decode`].
    ///
    /// For a non-voiced `cfg.signal_type`, `symbols` must be `None`: no
    /// bits are written and the empty (`!is_voiced()`) result is
    /// returned. For a voiced frame `symbols` must be `Some` and
    /// supplies every §4.2.7.6 symbol choice; the lag symbol variant
    /// must match `cfg.lag_coding` and every index must lie in its
    /// table's support. Returns the [`LtpParameters`] the decoder will
    /// reconstruct.
    pub fn encode(
        re: &mut RangeEncoder,
        cfg: LtpConfig,
        symbols: Option<&LtpSymbols>,
    ) -> Result<Self, Error> {
        if cfg.num_subframes != 2 && cfg.num_subframes != 4 {
            return Err(Error::MalformedPacket);
        }
        let spec = lag_low_spec(cfg.bandwidth)?;
        let num = cfg.num_subframes as usize;

        if cfg.signal_type != SignalType::Voiced {
            if symbols.is_some() {
                return Err(Error::MalformedPacket);
            }
            return Ok(Self {
                voiced: false,
                primary_lag: 0,
                contour_index: 0,
                pitch_lags: [0; LTP_MAX_SUBFRAMES],
                periodicity_index: 0,
                filter_taps_q7: [[0; LTP_FILTER_TAPS]; LTP_MAX_SUBFRAMES],
                filter_indices: [0; LTP_MAX_SUBFRAMES],
                ltp_scaling_q14: LTP_SCALING_DEFAULT_Q14,
                len: cfg.num_subframes,
            });
        }
        let symbols = symbols.ok_or(Error::MalformedPacket)?;

        // --- §4.2.7.6.1 primary pitch lag -----------------------------
        // `lag_low` support is the Table 30 codebook size (== scale).
        let lag_low_count = spec.scale as u8;
        let encode_absolute =
            |re: &mut RangeEncoder, lag_high: u8, lag_low: u8| -> Result<i32, Error> {
                if lag_high > 31 || lag_low >= lag_low_count {
                    return Err(Error::MalformedPacket);
                }
                re.enc_icdf(lag_high as usize, LAG_HIGH_ICDF, 8);
                re.enc_icdf(lag_low as usize, spec.icdf, 8);
                Ok(lag_high as i32 * spec.scale + lag_low as i32 + spec.lag_min)
            };
        let primary_lag = match (cfg.lag_coding, symbols.lag) {
            (LagCoding::Absolute, LagSymbols::Absolute { lag_high, lag_low }) => {
                encode_absolute(re, lag_high, lag_low)?
            }
            (LagCoding::Relative { previous_lag }, LagSymbols::RelativeDelta { delta_index }) => {
                if delta_index == 0 || delta_index > 20 {
                    // Index 0 is the absolute-coding fallback and must be
                    // written via `RelativeFallback`.
                    return Err(Error::MalformedPacket);
                }
                re.enc_icdf(delta_index as usize, LAG_DELTA_ICDF, 8);
                previous_lag + (delta_index as i32 - 9)
            }
            (LagCoding::Relative { .. }, LagSymbols::RelativeFallback { lag_high, lag_low }) => {
                re.enc_icdf(0, LAG_DELTA_ICDF, 8);
                encode_absolute(re, lag_high, lag_low)?
            }
            _ => return Err(Error::MalformedPacket),
        };

        // --- §4.2.7.6.1 pitch contour → per-subframe lags -------------
        // An iCDF table's terminating 0 entry is itself the last valid
        // symbol, so the symbol count equals the table length.
        let contour_cells = contour_icdf(cfg.bandwidth, num).len();
        if symbols.contour_index as usize >= contour_cells {
            return Err(Error::MalformedPacket);
        }
        re.enc_icdf(
            symbols.contour_index as usize,
            contour_icdf(cfg.bandwidth, num),
            8,
        );
        let mut pitch_lags = [0i32; LTP_MAX_SUBFRAMES];
        for (k, slot) in pitch_lags.iter_mut().enumerate().take(num) {
            let offset = contour_offset(cfg.bandwidth, num, symbols.contour_index, k) as i32;
            *slot = (primary_lag + offset).clamp(spec.lag_min, spec.lag_max);
        }

        // --- §4.2.7.6.2 LTP filter coefficients -----------------------
        if symbols.periodicity_index > 2 {
            return Err(Error::MalformedPacket);
        }
        re.enc_icdf(symbols.periodicity_index as usize, PERIODICITY_ICDF, 8);
        let filter_icdf = ltp_filter_icdf(symbols.periodicity_index);
        let filter_cells = filter_icdf.len();
        let mut filter_indices = [0u8; LTP_MAX_SUBFRAMES];
        let mut filter_taps_q7 = [[0i8; LTP_FILTER_TAPS]; LTP_MAX_SUBFRAMES];
        for k in 0..num {
            let idx = symbols.filter_indices[k];
            if idx as usize >= filter_cells {
                return Err(Error::MalformedPacket);
            }
            re.enc_icdf(idx as usize, filter_icdf, 8);
            filter_indices[k] = idx;
            filter_taps_q7[k] = ltp_filter_taps(symbols.periodicity_index, idx);
        }

        // --- §4.2.7.6.3 LTP scaling -----------------------------------
        if cfg.ltp_scaling_present != symbols.ltp_scaling_index.is_some() {
            return Err(Error::MalformedPacket);
        }
        let ltp_scaling_q14 = match symbols.ltp_scaling_index {
            Some(s) => {
                if s > 2 {
                    return Err(Error::MalformedPacket);
                }
                re.enc_icdf(s as usize, LTP_SCALING_ICDF, 8);
                LTP_SCALING_VALUES_Q14[s as usize]
            }
            None => LTP_SCALING_DEFAULT_Q14,
        };

        Ok(Self {
            voiced: true,
            primary_lag,
            contour_index: symbols.contour_index,
            pitch_lags,
            periodicity_index: symbols.periodicity_index,
            filter_taps_q7,
            filter_indices,
            ltp_scaling_q14,
            len: cfg.num_subframes,
        })
    }

    /// Decode the §4.2.7.6.1 primary pitch lag. Absolute coding combines
    /// a Table 29 high part with a Table 30 low part; relative coding adds
    /// a Table 31 delta to `previous_lag`, falling back to absolute coding
    /// when the decoded delta is zero.
    fn decode_primary_lag(rd: &mut RangeDecoder<'_>, spec: &LagLowSpec, coding: LagCoding) -> i32 {
        match coding {
            LagCoding::Absolute => Self::decode_absolute_lag(rd, spec),
            LagCoding::Relative { previous_lag } => {
                let delta_lag_index = rd.dec_icdf(LAG_DELTA_ICDF, 8) as i32;
                if delta_lag_index == 0 {
                    // Zero delta falls back to absolute coding.
                    Self::decode_absolute_lag(rd, spec)
                } else {
                    // lag = previous_lag + (delta_lag_index - 9); unclamped
                    // per the §4.2.7.6.1 note.
                    previous_lag + (delta_lag_index - 9)
                }
            }
        }
    }

    /// `lag = lag_high * lag_scale + lag_low + lag_min` (§4.2.7.6.1).
    fn decode_absolute_lag(rd: &mut RangeDecoder<'_>, spec: &LagLowSpec) -> i32 {
        let lag_high = rd.dec_icdf(LAG_HIGH_ICDF, 8) as i32;
        let lag_low = rd.dec_icdf(spec.icdf, 8) as i32;
        lag_high * spec.scale + lag_low + spec.lag_min
    }

    /// Decode the §4.2.7.6.1 pitch-contour VQ index from the Table 32 PDF
    /// selected by `(bandwidth, num_subframes)`.
    fn decode_contour_index(rd: &mut RangeDecoder<'_>, bandwidth: Bandwidth, num: usize) -> u8 {
        let icdf = contour_icdf(bandwidth, num);
        rd.dec_icdf(icdf, 8) as u8
    }

    /// `true` if this is a voiced frame carrying real LTP parameters.
    pub fn is_voiced(&self) -> bool {
        self.voiced
    }

    /// The primary pitch lag (unclamped; used as the next frame's
    /// `previous_lag` for relative coding). `0` for non-voiced frames.
    pub fn primary_lag(&self) -> i32 {
        self.primary_lag
    }

    /// The decoded pitch-contour VQ index. `0` for non-voiced frames.
    pub fn contour_index(&self) -> u8 {
        self.contour_index
    }

    /// Per-subframe pitch lags, clamped to the bandwidth's
    /// `[lag_min, lag_max]`. Empty for non-voiced frames.
    pub fn pitch_lags(&self) -> &[i32] {
        if self.voiced {
            &self.pitch_lags[..self.len as usize]
        } else {
            &[]
        }
    }

    /// The decoded periodicity index `0..=2`. `0` for non-voiced frames.
    pub fn periodicity_index(&self) -> u8 {
        self.periodicity_index
    }

    /// Per-subframe 5-tap Q7 LTP filter coefficients. Empty for non-voiced
    /// frames.
    pub fn filter_taps_q7(&self) -> &[[i8; LTP_FILTER_TAPS]] {
        if self.voiced {
            &self.filter_taps_q7[..self.len as usize]
        } else {
            &[]
        }
    }

    /// The decoded per-subframe filter indices. Empty for non-voiced
    /// frames.
    pub fn filter_indices(&self) -> &[u8] {
        if self.voiced {
            &self.filter_indices[..self.len as usize]
        } else {
            &[]
        }
    }

    /// The Q14 LTP scaling factor (§4.2.7.6.3); the default `15565` when
    /// not coded or for non-voiced frames.
    pub fn ltp_scaling_q14(&self) -> u16 {
        self.ltp_scaling_q14
    }

    /// Number of subframes this result was decoded for.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` if there are no subframes (never happens after a successful
    /// decode of a valid frame).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The §4.2.7.6.1 primary-lag symbol choice on the encode side —
/// which coding path [`LtpParameters::encode`] writes and its raw
/// indices. The variant must match the frame's [`LagCoding`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagSymbols {
    /// Absolute coding ([`LagCoding::Absolute`]): the Table 29 high
    /// part (`0..=31`) and the Table 30 bandwidth-dependent low part
    /// (`0..lag_scale`).
    Absolute {
        /// Table 29 high part, `0..=31`.
        lag_high: u8,
        /// Table 30 low part, `0..lag_scale` (4 NB / 6 MB / 8 WB).
        lag_low: u8,
    },
    /// Relative coding ([`LagCoding::Relative`]) with a non-zero
    /// Table 31 delta index (`1..=20`); the decoded lag is
    /// `previous_lag + (delta_index - 9)`.
    RelativeDelta {
        /// Table 31 delta index, `1..=20` (0 is the fallback and must
        /// be written via [`LagSymbols::RelativeFallback`]).
        delta_index: u8,
    },
    /// Relative coding falling back to absolute (§4.2.7.6.1: a zero
    /// delta): the zero Table 31 symbol followed by the absolute pair.
    RelativeFallback {
        /// Table 29 high part, `0..=31`.
        lag_high: u8,
        /// Table 30 low part, `0..lag_scale`.
        lag_low: u8,
    },
}

/// The full §4.2.7.6 symbol script for one voiced SILK frame on the
/// encode side, consumed by [`LtpParameters::encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtpSymbols {
    /// §4.2.7.6.1 primary-lag coding choice.
    pub lag: LagSymbols,
    /// §4.2.7.6.1 pitch-contour VQ index into the Table 33–36 codebook
    /// selected by `(bandwidth, num_subframes)`.
    pub contour_index: u8,
    /// §4.2.7.6.2 periodicity index, `0..=2`.
    pub periodicity_index: u8,
    /// §4.2.7.6.2 per-subframe filter indices into the periodicity's
    /// Table 39–41 codebook; only the first `num_subframes` entries
    /// are used.
    pub filter_indices: [u8; LTP_MAX_SUBFRAMES],
    /// §4.2.7.6.3 LTP-scaling index (`0..=2`); must be `Some` iff
    /// [`LtpConfig::ltp_scaling_present`].
    pub ltp_scaling_index: Option<u8>,
}

/// Table 32 contour PDF (as iCDF) for `(bandwidth, num_subframes)`.
fn contour_icdf(bandwidth: Bandwidth, num: usize) -> &'static [u8] {
    match (bandwidth, num) {
        (Bandwidth::Nb, 2) => CONTOUR_ICDF_NB_10MS,
        (Bandwidth::Nb, _) => CONTOUR_ICDF_NB_20MS,
        // MB or WB.
        (_, 2) => CONTOUR_ICDF_MBWB_10MS,
        (_, _) => CONTOUR_ICDF_MBWB_20MS,
    }
}

/// Table 33–36 contour offset for `(bandwidth, num_subframes, index, k)`.
fn contour_offset(bandwidth: Bandwidth, num: usize, index: u8, k: usize) -> i8 {
    let i = index as usize;
    match (bandwidth, num) {
        (Bandwidth::Nb, 2) => CONTOUR_NB_10MS[i][k],
        (Bandwidth::Nb, _) => CONTOUR_NB_20MS[i][k],
        (_, 2) => CONTOUR_MBWB_10MS[i][k],
        (_, _) => CONTOUR_MBWB_20MS[i][k],
    }
}

/// Table 38 LTP filter-index PDF (as iCDF) for a periodicity index.
fn ltp_filter_icdf(periodicity_index: u8) -> &'static [u8] {
    match periodicity_index {
        0 => LTP_FILTER_ICDF_P0,
        1 => LTP_FILTER_ICDF_P1,
        // Periodicity index is decoded from a 3-entry PDF, so it is
        // always 0..=2.
        _ => LTP_FILTER_ICDF_P2,
    }
}

/// Table 39–41 filter taps for a `(periodicity_index, filter_index)`.
fn ltp_filter_taps(periodicity_index: u8, filter_index: u8) -> [i8; LTP_FILTER_TAPS] {
    let i = filter_index as usize;
    match periodicity_index {
        0 => LTP_TAPS_P0[i],
        1 => LTP_TAPS_P1[i],
        _ => LTP_TAPS_P2[i],
    }
}

// =====================================================================
// Encode-side (analysis) table accessors — §5.2.3.2 pitch analysis and
// §5.2.3.6 LTP quantisation search the same normative tables the
// decoder reads, so the codebook geometry is exposed read-only here.
// =====================================================================

/// The §4.2.7.6.1 primary-lag geometry for a SILK bandwidth:
/// `(lag_min, lag_max, lag_scale)` from Table 30. Rejects SWB / FB.
pub fn lag_range(bandwidth: Bandwidth) -> Result<(i32, i32, i32), Error> {
    let spec = lag_low_spec(bandwidth)?;
    Ok((spec.lag_min, spec.lag_max, spec.scale))
}

/// Number of entries in the Table 33-36 pitch-contour codebook for
/// `(bandwidth, num_subframes)`. `num_subframes` must be 2 or 4;
/// SWB / FB are rejected.
pub fn contour_codebook_len(bandwidth: Bandwidth, num_subframes: usize) -> Result<usize, Error> {
    if num_subframes != 2 && num_subframes != 4 {
        return Err(Error::MalformedPacket);
    }
    lag_low_spec(bandwidth)?; // bandwidth validity
    Ok(match (bandwidth, num_subframes) {
        (Bandwidth::Nb, 2) => CONTOUR_NB_10MS.len(),
        (Bandwidth::Nb, _) => CONTOUR_NB_20MS.len(),
        (_, 2) => CONTOUR_MBWB_10MS.len(),
        (_, _) => CONTOUR_MBWB_20MS.len(),
    })
}

/// The Table 33-36 per-subframe lag offset for a contour codebook
/// entry: `contour_offsets(bw, num, index)[k]` is what the decoder
/// adds to the primary lag for subframe `k` (before the
/// `[lag_min, lag_max]` clamp). Errors on an out-of-range index.
pub fn contour_offsets(
    bandwidth: Bandwidth,
    num_subframes: usize,
    index: u8,
) -> Result<[i8; LTP_MAX_SUBFRAMES], Error> {
    if index as usize >= contour_codebook_len(bandwidth, num_subframes)? {
        return Err(Error::MalformedPacket);
    }
    let mut out = [0i8; LTP_MAX_SUBFRAMES];
    for (k, slot) in out.iter_mut().enumerate().take(num_subframes) {
        *slot = contour_offset(bandwidth, num_subframes, index, k);
    }
    Ok(out)
}

/// Number of entries in the Table 39-41 LTP filter codebook for a
/// §4.2.7.6.2 periodicity index (8 / 16 / 32 coded entries; the
/// §5.2.3.6 prose's "10, 20, and 40 vectors" describes average rates,
/// not these normative table sizes). Errors when
/// `periodicity_index > 2`.
pub fn ltp_filter_codebook_len(periodicity_index: u8) -> Result<usize, Error> {
    Ok(match periodicity_index {
        0 => LTP_TAPS_P0.len(),
        1 => LTP_TAPS_P1.len(),
        2 => LTP_TAPS_P2.len(),
        _ => return Err(Error::MalformedPacket),
    })
}

/// The Table 39-41 5-tap Q7 filter vector for a
/// `(periodicity_index, filter_index)` pair. Errors on out-of-range
/// indices.
pub fn ltp_filter_taps_q7(
    periodicity_index: u8,
    filter_index: u8,
) -> Result<[i8; LTP_FILTER_TAPS], Error> {
    if (filter_index as usize) >= ltp_filter_codebook_len(periodicity_index)? {
        return Err(Error::MalformedPacket);
    }
    Ok(ltp_filter_taps(periodicity_index, filter_index))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PDF → iCDF transcription self-checks -------------------------

    /// Re-derive the iCDF for a PDF and assert it matches, plus that the
    /// PDF sums to 256 and the iCDF is monotone-decreasing terminated by 0.
    fn check_icdf(pdf: &[u32], icdf: &[u8]) {
        assert_eq!(pdf.iter().sum::<u32>(), 256, "PDF must sum to 256");
        let mut cum = 0u32;
        let mut expected = Vec::new();
        for &p in pdf {
            cum += p;
            expected.push((256 - cum) as u8);
        }
        assert_eq!(icdf, expected.as_slice(), "iCDF mismatch");
        assert_eq!(*icdf.last().unwrap(), 0, "iCDF must terminate with 0");
        for w in icdf.windows(2) {
            assert!(w[0] > w[1], "iCDF not strictly decreasing: {icdf:?}");
        }
        assert_eq!(icdf.len(), pdf.len(), "iCDF length must equal PDF length");
    }

    #[test]
    fn table29_high_part_icdf() {
        check_icdf(
            &[
                3, 3, 6, 11, 21, 30, 32, 19, 11, 10, 12, 13, 13, 12, 11, 9, 8, 7, 6, 4, 2, 2, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
            ],
            LAG_HIGH_ICDF,
        );
        assert_eq!(LAG_HIGH_ICDF.len(), 32);
    }

    #[test]
    fn table30_low_part_icdfs() {
        check_icdf(&[64, 64, 64, 64], LAG_LOW_ICDF_NB);
        check_icdf(&[43, 42, 43, 43, 42, 43], LAG_LOW_ICDF_MB);
        check_icdf(&[32, 32, 32, 32, 32, 32, 32, 32], LAG_LOW_ICDF_WB);
    }

    #[test]
    fn table30_scales_and_ranges() {
        let nb = lag_low_spec(Bandwidth::Nb).unwrap();
        assert_eq!((nb.scale, nb.lag_min, nb.lag_max), (4, 16, 144));
        let mb = lag_low_spec(Bandwidth::Mb).unwrap();
        assert_eq!((mb.scale, mb.lag_min, mb.lag_max), (6, 24, 216));
        let wb = lag_low_spec(Bandwidth::Wb).unwrap();
        assert_eq!((wb.scale, wb.lag_min, wb.lag_max), (8, 32, 288));
        assert!(lag_low_spec(Bandwidth::Swb).is_err());
        assert!(lag_low_spec(Bandwidth::Fb).is_err());
    }

    #[test]
    fn table31_delta_icdf() {
        check_icdf(
            &[
                46, 2, 2, 3, 4, 6, 10, 15, 26, 38, 30, 22, 15, 10, 7, 6, 4, 4, 2, 2, 2,
            ],
            LAG_DELTA_ICDF,
        );
        assert_eq!(LAG_DELTA_ICDF.len(), 21);
    }

    #[test]
    fn table32_contour_icdfs() {
        check_icdf(&[143, 50, 63], CONTOUR_ICDF_NB_10MS);
        check_icdf(
            &[68, 12, 21, 17, 19, 22, 30, 24, 17, 16, 10],
            CONTOUR_ICDF_NB_20MS,
        );
        check_icdf(
            &[91, 46, 39, 19, 14, 12, 8, 7, 6, 5, 5, 4],
            CONTOUR_ICDF_MBWB_10MS,
        );
        check_icdf(
            &[
                33, 22, 18, 16, 15, 14, 14, 13, 13, 10, 9, 9, 8, 6, 6, 6, 5, 4, 4, 4, 3, 3, 3, 2,
                2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
            ],
            CONTOUR_ICDF_MBWB_20MS,
        );
    }

    #[test]
    fn table37_periodicity_icdf() {
        check_icdf(&[77, 80, 99], PERIODICITY_ICDF);
    }

    #[test]
    fn table38_ltp_filter_icdfs() {
        check_icdf(&[185, 15, 13, 13, 9, 9, 6, 6], LTP_FILTER_ICDF_P0);
        check_icdf(
            &[57, 34, 21, 20, 15, 13, 12, 13, 10, 10, 9, 10, 9, 8, 7, 8],
            LTP_FILTER_ICDF_P1,
        );
        check_icdf(
            &[
                15, 16, 14, 12, 12, 12, 11, 11, 11, 10, 9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 6, 6, 5, 4,
                5, 4, 4, 4, 3, 4, 3, 2,
            ],
            LTP_FILTER_ICDF_P2,
        );
    }

    #[test]
    fn table42_ltp_scaling_icdf() {
        check_icdf(&[128, 64, 64], LTP_SCALING_ICDF);
        assert_eq!(LTP_SCALING_VALUES_Q14, [15565, 12288, 8192]);
        assert_eq!(LTP_SCALING_DEFAULT_Q14, 15565);
    }

    // --- Contour codebook structural checks ---------------------------

    #[test]
    fn contour_codebook_sizes_match_pdfs() {
        assert_eq!(CONTOUR_NB_10MS.len(), CONTOUR_ICDF_NB_10MS.len());
        assert_eq!(CONTOUR_NB_20MS.len(), CONTOUR_ICDF_NB_20MS.len());
        assert_eq!(CONTOUR_MBWB_10MS.len(), CONTOUR_ICDF_MBWB_10MS.len());
        assert_eq!(CONTOUR_MBWB_20MS.len(), CONTOUR_ICDF_MBWB_20MS.len());
        // Index-0 vectors are all-zero (no offset) in every codebook.
        assert_eq!(CONTOUR_NB_10MS[0], [0, 0]);
        assert_eq!(CONTOUR_NB_20MS[0], [0, 0, 0, 0]);
        assert_eq!(CONTOUR_MBWB_10MS[0], [0, 0]);
        assert_eq!(CONTOUR_MBWB_20MS[0], [0, 0, 0, 0]);
        // Spot-check a few interior rows against the spec tables.
        assert_eq!(CONTOUR_NB_20MS[1], [2, 1, 0, -1]);
        assert_eq!(CONTOUR_MBWB_20MS[33], [-9, -3, 3, 9]);
        assert_eq!(CONTOUR_MBWB_10MS[11], [-3, 3]);
    }

    #[test]
    fn ltp_filter_codebook_sizes_match_pdfs() {
        assert_eq!(LTP_TAPS_P0.len(), LTP_FILTER_ICDF_P0.len());
        assert_eq!(LTP_TAPS_P1.len(), LTP_FILTER_ICDF_P1.len());
        assert_eq!(LTP_TAPS_P2.len(), LTP_FILTER_ICDF_P2.len());
        assert_eq!(LTP_TAPS_P0.len(), 8);
        assert_eq!(LTP_TAPS_P1.len(), 16);
        assert_eq!(LTP_TAPS_P2.len(), 32);
        // Spot-checks against Tables 39–41.
        assert_eq!(LTP_TAPS_P0[0], [4, 6, 24, 7, 5]);
        assert_eq!(LTP_TAPS_P0[7], [16, 14, 38, -3, 33]);
        assert_eq!(LTP_TAPS_P1[0], [13, 22, 39, 23, 12]);
        assert_eq!(LTP_TAPS_P1[15], [3, -1, 21, 16, 41]);
        assert_eq!(LTP_TAPS_P2[0], [-6, 27, 61, 39, 5]);
        assert_eq!(LTP_TAPS_P2[31], [2, 0, 9, 10, 88]);
    }

    // --- Non-voiced: no reads -----------------------------------------

    #[test]
    fn non_voiced_reads_nothing() {
        let buf = [0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6];
        for signal_type in [SignalType::Inactive, SignalType::Unvoiced] {
            let mut rd = RangeDecoder::new(&buf);
            let tell_before = rd.tell();
            let cfg = LtpConfig {
                bandwidth: Bandwidth::Wb,
                signal_type,
                num_subframes: 4,
                lag_coding: LagCoding::Absolute,
                ltp_scaling_present: true,
            };
            let ltp = LtpParameters::decode(&mut rd, cfg).unwrap();
            assert_eq!(rd.tell(), tell_before, "non-voiced must consume no bits");
            assert!(!ltp.is_voiced());
            assert!(ltp.pitch_lags().is_empty());
            assert!(ltp.filter_taps_q7().is_empty());
            assert_eq!(ltp.ltp_scaling_q14(), LTP_SCALING_DEFAULT_Q14);
        }
    }

    // --- Malformed config rejection -----------------------------------

    #[test]
    fn rejects_bad_subframe_count() {
        let buf = [0x00u8; 8];
        let mut rd = RangeDecoder::new(&buf);
        let cfg = LtpConfig {
            bandwidth: Bandwidth::Nb,
            signal_type: SignalType::Voiced,
            num_subframes: 3,
            lag_coding: LagCoding::Absolute,
            ltp_scaling_present: false,
        };
        assert!(LtpParameters::decode(&mut rd, cfg).is_err());
    }

    #[test]
    fn rejects_non_silk_bandwidth() {
        let buf = [0x00u8; 8];
        for bw in [Bandwidth::Swb, Bandwidth::Fb] {
            let mut rd = RangeDecoder::new(&buf);
            let cfg = LtpConfig {
                bandwidth: bw,
                signal_type: SignalType::Voiced,
                num_subframes: 4,
                lag_coding: LagCoding::Absolute,
                ltp_scaling_present: false,
            };
            assert!(LtpParameters::decode(&mut rd, cfg).is_err());
        }
    }

    // --- Voiced end-to-end: structural invariants ---------------------

    /// Independent re-derivation of the absolute primary lag off a fresh
    /// decoder, used to confirm the production decode reads the same
    /// symbols in the same order.
    fn manual_absolute_lag(rd: &mut RangeDecoder<'_>, spec: &LagLowSpec) -> i32 {
        let high = rd.dec_icdf(LAG_HIGH_ICDF, 8) as i32;
        let low = rd.dec_icdf(spec.icdf, 8) as i32;
        high * spec.scale + low + spec.lag_min
    }

    #[test]
    fn voiced_absolute_pitch_lags_in_range_and_match_formula() {
        let buf = [
            0x12, 0x9C, 0x5E, 0xA3, 0x77, 0x10, 0xBB, 0x42, 0x88, 0x01, 0xFE, 0x6D,
        ];
        for bandwidth in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            for &num in &[2usize, 4usize] {
                let spec = lag_low_spec(bandwidth).unwrap();

                // Production decode.
                let mut rd = RangeDecoder::new(&buf);
                let cfg = LtpConfig {
                    bandwidth,
                    signal_type: SignalType::Voiced,
                    num_subframes: num as u8,
                    lag_coding: LagCoding::Absolute,
                    ltp_scaling_present: true,
                };
                let ltp = LtpParameters::decode(&mut rd, cfg).unwrap();

                // Independent re-derivation of the lag pipeline.
                let mut rd2 = RangeDecoder::new(&buf);
                let primary = manual_absolute_lag(&mut rd2, &spec);
                let contour = rd2.dec_icdf(contour_icdf(bandwidth, num), 8) as u8;
                assert_eq!(ltp.primary_lag(), primary);
                assert_eq!(ltp.contour_index(), contour);

                assert_eq!(ltp.pitch_lags().len(), num);
                for (k, &lag) in ltp.pitch_lags().iter().enumerate() {
                    let off = contour_offset(bandwidth, num, contour, k) as i32;
                    let expected = (primary + off).clamp(spec.lag_min, spec.lag_max);
                    assert_eq!(lag, expected, "bw={bandwidth:?} num={num} k={k}");
                    assert!(
                        (spec.lag_min..=spec.lag_max).contains(&lag),
                        "lag {lag} out of [{}, {}]",
                        spec.lag_min,
                        spec.lag_max
                    );
                }

                // Filters: every subframe yields a 5-tap row from the
                // periodicity codebook, and the index is in-range.
                assert_eq!(ltp.filter_taps_q7().len(), num);
                let p = ltp.periodicity_index();
                assert!(p <= 2);
                for (k, &fi) in ltp.filter_indices().iter().enumerate() {
                    let expected = ltp_filter_taps(p, fi);
                    assert_eq!(ltp.filter_taps_q7()[k], expected);
                }
            }
        }
    }

    #[test]
    fn voiced_relative_nonzero_delta_uses_previous_lag() {
        // Craft a decode where the relative delta is non-zero by choosing
        // a buffer; verify the production result equals
        // previous_lag + (delta - 9) and that the contour follows.
        let buf = [
            0x7F, 0x3A, 0x91, 0x04, 0xCD, 0x6E, 0x22, 0xB0, 0x15, 0x88, 0x40, 0xEE,
        ];
        let bandwidth = Bandwidth::Wb;
        let num = 4usize;
        let spec = lag_low_spec(bandwidth).unwrap();
        let previous_lag = 100;

        let mut rd = RangeDecoder::new(&buf);
        let cfg = LtpConfig {
            bandwidth,
            signal_type: SignalType::Voiced,
            num_subframes: num as u8,
            lag_coding: LagCoding::Relative { previous_lag },
            ltp_scaling_present: false,
        };
        let ltp = LtpParameters::decode(&mut rd, cfg).unwrap();

        // Independent re-derivation: read the delta first.
        let mut rd2 = RangeDecoder::new(&buf);
        let delta = rd2.dec_icdf(LAG_DELTA_ICDF, 8) as i32;
        let primary = if delta == 0 {
            manual_absolute_lag(&mut rd2, &spec)
        } else {
            previous_lag + (delta - 9)
        };
        assert_eq!(ltp.primary_lag(), primary);
    }

    #[test]
    fn relative_zero_delta_falls_back_to_absolute() {
        // The fallback path reads the delta symbol, observes zero, then
        // reads the absolute high+low parts. We cannot easily force a
        // particular delta from an arbitrary buffer, so this test pins the
        // *logic*: a decoder that observes delta == 0 must read exactly as
        // many further symbols as the absolute path. We verify by mirroring
        // the production decode against an independent pass that branches on
        // the same observed delta.
        let buf = [
            0xAA, 0x55, 0xF0, 0x0F, 0x3C, 0xC3, 0x99, 0x66, 0x12, 0x34, 0x56, 0x78,
        ];
        let bandwidth = Bandwidth::Mb;
        let num = 2usize;
        let spec = lag_low_spec(bandwidth).unwrap();

        let mut rd = RangeDecoder::new(&buf);
        let cfg = LtpConfig {
            bandwidth,
            signal_type: SignalType::Voiced,
            num_subframes: num as u8,
            lag_coding: LagCoding::Relative { previous_lag: 50 },
            ltp_scaling_present: false,
        };
        let ltp = LtpParameters::decode(&mut rd, cfg).unwrap();

        let mut rd2 = RangeDecoder::new(&buf);
        let delta = rd2.dec_icdf(LAG_DELTA_ICDF, 8) as i32;
        let primary = if delta == 0 {
            manual_absolute_lag(&mut rd2, &spec)
        } else {
            50 + (delta - 9)
        };
        let contour = rd2.dec_icdf(contour_icdf(bandwidth, num), 8) as u8;
        assert_eq!(ltp.primary_lag(), primary);
        assert_eq!(ltp.contour_index(), contour);
        // Whichever branch was taken, the byte position after decoding the
        // lag + contour must agree between the two passes.
        assert!(rd.tell() >= rd2.tell());
    }

    // --- LTP scaling present vs default -------------------------------

    #[test]
    fn ltp_scaling_present_decodes_one_of_three() {
        let buf = [
            0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0,
        ];
        let mut rd = RangeDecoder::new(&buf);
        let cfg = LtpConfig {
            bandwidth: Bandwidth::Nb,
            signal_type: SignalType::Voiced,
            num_subframes: 4,
            lag_coding: LagCoding::Absolute,
            ltp_scaling_present: true,
        };
        let ltp = LtpParameters::decode(&mut rd, cfg).unwrap();
        assert!(LTP_SCALING_VALUES_Q14.contains(&ltp.ltp_scaling_q14()));
    }

    #[test]
    fn ltp_scaling_absent_uses_default_without_reading() {
        // With scaling absent, the decode stops after the filter indices.
        // Compare bit positions: a "present" decode must consume strictly
        // more bits than an otherwise-identical "absent" decode.
        let buf = [
            0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0,
        ];
        let base = LtpConfig {
            bandwidth: Bandwidth::Nb,
            signal_type: SignalType::Voiced,
            num_subframes: 4,
            lag_coding: LagCoding::Absolute,
            ltp_scaling_present: false,
        };

        let mut rd_absent = RangeDecoder::new(&buf);
        let ltp_absent = LtpParameters::decode(&mut rd_absent, base).unwrap();
        assert_eq!(ltp_absent.ltp_scaling_q14(), LTP_SCALING_DEFAULT_Q14);
        let tell_absent = rd_absent.tell();

        let mut rd_present = RangeDecoder::new(&buf);
        let _ = LtpParameters::decode(
            &mut rd_present,
            LtpConfig {
                ltp_scaling_present: true,
                ..base
            },
        )
        .unwrap();
        assert!(
            rd_present.tell() > tell_absent,
            "present scaling must consume more bits than absent"
        );
    }

    // --- Wide sweep: never panics, always in range --------------------

    #[test]
    fn voiced_sweep_never_panics_and_stays_in_range() {
        let buffers: [&[u8]; 3] = [
            &[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99],
            &[0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66],
            &[0x5A, 0xA5, 0x3C, 0xC3, 0x0F, 0xF0, 0x69, 0x96, 0x12, 0x48],
        ];
        for buf in buffers {
            for bandwidth in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
                let spec = lag_low_spec(bandwidth).unwrap();
                for &num in &[2u8, 4u8] {
                    for &present in &[false, true] {
                        for coding in [
                            LagCoding::Absolute,
                            LagCoding::Relative { previous_lag: 80 },
                        ] {
                            let mut rd = RangeDecoder::new(buf);
                            let cfg = LtpConfig {
                                bandwidth,
                                signal_type: SignalType::Voiced,
                                num_subframes: num,
                                lag_coding: coding,
                                ltp_scaling_present: present,
                            };
                            let ltp = LtpParameters::decode(&mut rd, cfg).unwrap();
                            assert_eq!(ltp.pitch_lags().len(), num as usize);
                            for &lag in ltp.pitch_lags() {
                                assert!((spec.lag_min..=spec.lag_max).contains(&lag));
                            }
                            assert!(ltp.periodicity_index() <= 2);
                            assert_eq!(ltp.filter_taps_q7().len(), num as usize);
                        }
                    }
                }
            }
        }
    }

    // ----- §4.2.7.6 encode-side mirror ------------------------------

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

    /// encode → decode roundtrip over random LTP symbol scripts across
    /// every bandwidth / frame size / lag-coding / scaling combination:
    /// the decoder must reconstruct exactly the parameters the encoder
    /// predicted.
    #[test]
    fn ltp_encode_decode_roundtrip_random() {
        use crate::range_encoder::RangeEncoder;
        let mut rng = Lcg(0x17B_CAFE);
        for _ in 0..800 {
            let bandwidth = match rng.below(3) {
                0 => Bandwidth::Nb,
                1 => Bandwidth::Mb,
                _ => Bandwidth::Wb,
            };
            let num_subframes = if rng.below(2) == 0 { 2u8 } else { 4 };
            let relative = rng.below(2) == 1;
            let lag_coding = if relative {
                LagCoding::Relative {
                    previous_lag: 16 + rng.below(250) as i32,
                }
            } else {
                LagCoding::Absolute
            };
            let ltp_scaling_present = rng.below(2) == 1;
            let cfg = LtpConfig {
                bandwidth,
                signal_type: SignalType::Voiced,
                num_subframes,
                lag_coding,
                ltp_scaling_present,
            };

            let lag_low_count = match bandwidth {
                Bandwidth::Nb => 4u32,
                Bandwidth::Mb => 6,
                _ => 8,
            };
            let lag = if relative {
                if rng.below(2) == 0 {
                    LagSymbols::RelativeDelta {
                        delta_index: 1 + rng.below(20) as u8,
                    }
                } else {
                    LagSymbols::RelativeFallback {
                        lag_high: rng.below(32) as u8,
                        lag_low: rng.below(lag_low_count) as u8,
                    }
                }
            } else {
                LagSymbols::Absolute {
                    lag_high: rng.below(32) as u8,
                    lag_low: rng.below(lag_low_count) as u8,
                }
            };
            let contour_cells = match (bandwidth, num_subframes) {
                (Bandwidth::Nb, 2) => 3u32,
                (Bandwidth::Nb, 4) => 11,
                (_, 2) => 12,
                _ => 34,
            };
            let periodicity_index = rng.below(3) as u8;
            let filter_cells = [8u32, 16, 32][periodicity_index as usize];
            let mut filter_indices = [0u8; LTP_MAX_SUBFRAMES];
            for f in filter_indices.iter_mut().take(num_subframes as usize) {
                *f = rng.below(filter_cells) as u8;
            }
            let symbols = LtpSymbols {
                lag,
                contour_index: rng.below(contour_cells) as u8,
                periodicity_index,
                filter_indices,
                ltp_scaling_index: ltp_scaling_present.then(|| rng.below(3) as u8),
            };

            let mut re = RangeEncoder::new();
            let predicted = LtpParameters::encode(&mut re, cfg, Some(&symbols)).expect("encode");
            let bytes = re.finish();

            let mut rd = RangeDecoder::new(&bytes);
            let decoded = LtpParameters::decode(&mut rd, cfg).expect("decode");
            assert!(!rd.has_error());
            assert_eq!(decoded, predicted, "cfg={cfg:?} symbols={symbols:?}");
        }
    }

    /// A non-voiced encode writes nothing and returns the empty result;
    /// supplying symbols for a non-voiced frame is rejected.
    #[test]
    fn ltp_encode_non_voiced_writes_nothing() {
        use crate::range_encoder::RangeEncoder;
        let cfg = LtpConfig {
            bandwidth: Bandwidth::Nb,
            signal_type: SignalType::Unvoiced,
            num_subframes: 4,
            lag_coding: LagCoding::Absolute,
            ltp_scaling_present: false,
        };
        let mut re = RangeEncoder::new();
        let tell0 = re.tell();
        let params = LtpParameters::encode(&mut re, cfg, None).expect("encode");
        assert_eq!(re.tell(), tell0);
        assert!(!params.is_voiced());
        assert_eq!(params.ltp_scaling_q14(), LTP_SCALING_DEFAULT_Q14);

        let mut re = RangeEncoder::new();
        let symbols = LtpSymbols {
            lag: LagSymbols::Absolute {
                lag_high: 0,
                lag_low: 0,
            },
            contour_index: 0,
            periodicity_index: 0,
            filter_indices: [0; LTP_MAX_SUBFRAMES],
            ltp_scaling_index: None,
        };
        assert!(LtpParameters::encode(&mut re, cfg, Some(&symbols)).is_err());
    }

    /// The encode path rejects mismatched lag variants and
    /// out-of-support indices.
    #[test]
    fn ltp_encode_rejects_bad_symbols() {
        use crate::range_encoder::RangeEncoder;
        let cfg_abs = LtpConfig {
            bandwidth: Bandwidth::Nb,
            signal_type: SignalType::Voiced,
            num_subframes: 2,
            lag_coding: LagCoding::Absolute,
            ltp_scaling_present: false,
        };
        let ok = LtpSymbols {
            lag: LagSymbols::Absolute {
                lag_high: 3,
                lag_low: 1,
            },
            contour_index: 0,
            periodicity_index: 0,
            filter_indices: [0; LTP_MAX_SUBFRAMES],
            ltp_scaling_index: None,
        };
        // Relative delta on an absolute config.
        let mut re = RangeEncoder::new();
        let bad = LtpSymbols {
            lag: LagSymbols::RelativeDelta { delta_index: 3 },
            ..ok
        };
        assert!(LtpParameters::encode(&mut re, cfg_abs, Some(&bad)).is_err());
        // Zero delta must use RelativeFallback.
        let cfg_rel = LtpConfig {
            lag_coding: LagCoding::Relative { previous_lag: 100 },
            ..cfg_abs
        };
        let mut re = RangeEncoder::new();
        let bad = LtpSymbols {
            lag: LagSymbols::RelativeDelta { delta_index: 0 },
            ..ok
        };
        assert!(LtpParameters::encode(&mut re, cfg_rel, Some(&bad)).is_err());
        // lag_low out of range for NB (4 cells).
        let mut re = RangeEncoder::new();
        let bad = LtpSymbols {
            lag: LagSymbols::Absolute {
                lag_high: 0,
                lag_low: 4,
            },
            ..ok
        };
        assert!(LtpParameters::encode(&mut re, cfg_abs, Some(&bad)).is_err());
        // Contour index out of range (NB 10 ms has 3 cells).
        let mut re = RangeEncoder::new();
        let bad = LtpSymbols {
            contour_index: 3,
            ..ok
        };
        assert!(LtpParameters::encode(&mut re, cfg_abs, Some(&bad)).is_err());
        // Filter index out of range for periodicity 0 (8 cells).
        let mut re = RangeEncoder::new();
        let mut bad = ok;
        bad.filter_indices[1] = 8;
        assert!(LtpParameters::encode(&mut re, cfg_abs, Some(&bad)).is_err());
        // Scaling index present on a config without scaling.
        let mut re = RangeEncoder::new();
        let bad = LtpSymbols {
            ltp_scaling_index: Some(1),
            ..ok
        };
        assert!(LtpParameters::encode(&mut re, cfg_abs, Some(&bad)).is_err());
    }
}
