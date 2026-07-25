//! Colorimetry vocabulary: the optional `video/raw` color fields, their categorical
//! values, and the H.273 code-point mappings (spec: Formats — the format algebra
//! carries what a correct renderer needs; ITU-T H.273 "Coding-independent code
//! points for video signal type identification" defines the integers containers
//! and bitstreams carry — Matroska's `Colour` element values ARE these code points,
//! RFC 9559 §5.1.4.1.31).
//!
//! All four fields are **optional** on `video/raw` (and pass through the compressed
//! families so a demuxer's knowledge reaches the renderer): an element announces
//! only what it knows; a consumer that needs values applies [`Colorimetry::default_for`]
//! — the industry "unspecified" convention (H.273 code point 2): SD content is
//! treated as BT.601, HD as BT.709, always limited range.

/// Y'CbCr ↔ R'G'B' matrix coefficients (H.273 §8.3, `MatrixCoefficients`).
pub const FIELD_MATRIX: &str = "matrix";
/// Quantization range (H.273 `VideoFullRangeFlag`; mkv `Range`).
pub const FIELD_RANGE: &str = "range";
/// Transfer characteristics (H.273 §8.2, `TransferCharacteristics`).
pub const FIELD_TRANSFER: &str = "transfer";
/// Colour primaries (H.273 §8.1, `ColourPrimaries`).
pub const FIELD_PRIMARIES: &str = "primaries";

/// Y'CbCr matrix coefficients the pipeline distinguishes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Matrix {
    /// ITU-R BT.709-6 §3 (Kr=0.2126, Kb=0.0722) — HD.
    Bt709,
    /// ITU-R BT.601-7 §2.5 (Kr=0.299, Kb=0.114) — SD (both 525- and 625-line).
    Bt601,
    /// ITU-R BT.2020-2 §4 non-constant luminance (Kr=0.2627, Kb=0.0593) — UHD/HDR.
    Bt2020,
    /// RGB stored directly (H.273 code point 0).
    Identity,
}

/// Quantization range.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Range {
    /// "Video"/"MPEG" range: Y′ ∈ [16,235], C ∈ [16,240] at 8 bits (BT.601 §2.5.3).
    Limited,
    /// Full range: [0,255] at 8 bits.
    Full,
}

/// Transfer characteristics.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TransferFn {
    /// The SDR broadcast OETF family (BT.709/BT.601, displayed per BT.1886).
    Bt709,
    /// IEC 61966-2-1 sRGB piecewise curve.
    Srgb,
    /// Perceptual Quantizer, ITU-R BT.2100 Table 4 (SMPTE ST 2084) — HDR.
    Pq,
    /// Hybrid Log-Gamma, ITU-R BT.2100 Table 5 — HDR.
    Hlg,
    /// Linear light (H.273 code point 8).
    Linear,
}

/// Colour primaries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Primaries {
    /// ITU-R BT.709-6 §1 (same as sRGB).
    Bt709,
    /// SD primaries (H.273 5/6: BT.470BG / SMPTE 170M — treated as one here).
    Bt601,
    /// ITU-R BT.2020-2 §1 wide gamut.
    Bt2020,
    /// SMPTE RP 431-2 "DCI-P3".
    DciP3,
}

macro_rules! caps_enum {
    ($ty:ty, $(($v:path, $name:literal)),+ $(,)?) => {
        impl $ty {
            /// The categorical name used in `video/raw` offers/announcements.
            pub const fn caps_name(self) -> &'static str {
                match self { $($v => $name,)+ }
            }
            pub fn from_caps_name(s: &str) -> Option<Self> {
                match s { $($name => Some($v),)+ _ => None }
            }
        }
    };
}

caps_enum!(Matrix, (Matrix::Bt709, "bt709"), (Matrix::Bt601, "bt601"), (Matrix::Bt2020, "bt2020"), (Matrix::Identity, "identity"));
caps_enum!(Range, (Range::Limited, "limited"), (Range::Full, "full"));
caps_enum!(
    TransferFn,
    (TransferFn::Bt709, "bt709"),
    (TransferFn::Srgb, "srgb"),
    (TransferFn::Pq, "pq"),
    (TransferFn::Hlg, "hlg"),
    (TransferFn::Linear, "linear"),
);
caps_enum!(
    Primaries,
    (Primaries::Bt709, "bt709"),
    (Primaries::Bt601, "bt601"),
    (Primaries::Bt2020, "bt2020"),
    (Primaries::DciP3, "dci-p3"),
);

impl Matrix {
    /// Map an H.273 §8.3 `MatrixCoefficients` code point (what mkv's
    /// `Colour\MatrixCoefficients` carries, RFC 9559). `None` = unspecified or a
    /// code point the pipeline doesn't distinguish (the consumer then defaults).
    pub fn from_h273(cp: u8) -> Option<Matrix> {
        Some(match cp {
            0 => Matrix::Identity,
            1 => Matrix::Bt709,
            5 | 6 => Matrix::Bt601, // BT.470BG / SMPTE 170M share the BT.601 matrix
            9 | 10 => Matrix::Bt2020, // non-constant & constant luminance (ncl math here)
            _ => return None, // 2 = unspecified; the rest are exotic (YCgCo, ICtCp…)
        })
    }
}

impl TransferFn {
    /// Map an H.273 §8.2 `TransferCharacteristics` code point.
    pub fn from_h273(cp: u8) -> Option<TransferFn> {
        Some(match cp {
            1 | 6 | 14 | 15 => TransferFn::Bt709, // 709 / 601 / 2020-10/12bit share the OETF
            13 => TransferFn::Srgb,
            16 => TransferFn::Pq,  // SMPTE ST 2084
            18 => TransferFn::Hlg, // ARIB STD-B67
            8 => TransferFn::Linear,
            _ => return None, // 2 = unspecified
        })
    }
}

impl Primaries {
    /// Map an H.273 §8.1 `ColourPrimaries` code point.
    pub fn from_h273(cp: u8) -> Option<Primaries> {
        Some(match cp {
            1 => Primaries::Bt709,
            5 | 6 => Primaries::Bt601,
            9 => Primaries::Bt2020,
            11 | 12 => Primaries::DciP3, // DCI theatre / D65 display variants
            _ => return None, // 2 = unspecified
        })
    }
}

impl Range {
    /// Map mkv's `Colour\Range` (RFC 9559: 1 = broadcast/limited, 2 = full;
    /// 0/3 = unspecified/defined-by-matrix).
    pub fn from_mkv(v: u8) -> Option<Range> {
        Some(match v {
            1 => Range::Limited,
            2 => Range::Full,
            _ => return None,
        })
    }
}

/// The full announced (or defaulted) colorimetry of a raw video stream.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Colorimetry {
    pub matrix: Matrix,
    pub range: Range,
    pub transfer: TransferFn,
    pub primaries: Primaries,
}

impl Colorimetry {
    /// The convention for unannounced colorimetry (H.273 "unspecified", code
    /// point 2): treat HD-and-up as BT.709 and SD as BT.601, always limited
    /// range, SDR transfer, primaries matching the matrix.
    pub fn default_for(height: u32) -> Colorimetry {
        if height >= 720 {
            Colorimetry {
                matrix: Matrix::Bt709,
                range: Range::Limited,
                transfer: TransferFn::Bt709,
                primaries: Primaries::Bt709,
            }
        } else {
            Colorimetry {
                matrix: Matrix::Bt601,
                range: Range::Limited,
                transfer: TransferFn::Bt709,
                primaries: Primaries::Bt601,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_names_round_trip() {
        for m in [Matrix::Bt709, Matrix::Bt601, Matrix::Bt2020, Matrix::Identity] {
            assert_eq!(Matrix::from_caps_name(m.caps_name()), Some(m));
        }
        for r in [Range::Limited, Range::Full] {
            assert_eq!(Range::from_caps_name(r.caps_name()), Some(r));
        }
        for t in [TransferFn::Bt709, TransferFn::Srgb, TransferFn::Pq, TransferFn::Hlg, TransferFn::Linear] {
            assert_eq!(TransferFn::from_caps_name(t.caps_name()), Some(t));
        }
        for p in [Primaries::Bt709, Primaries::Bt601, Primaries::Bt2020, Primaries::DciP3] {
            assert_eq!(Primaries::from_caps_name(p.caps_name()), Some(p));
        }
        assert_eq!(Matrix::from_caps_name("nope"), None);
    }

    #[test]
    fn h273_code_points_map_per_spec() {
        // H.273 §8.3 MatrixCoefficients.
        assert_eq!(Matrix::from_h273(1), Some(Matrix::Bt709));
        assert_eq!(Matrix::from_h273(5), Some(Matrix::Bt601));
        assert_eq!(Matrix::from_h273(6), Some(Matrix::Bt601));
        assert_eq!(Matrix::from_h273(9), Some(Matrix::Bt2020));
        assert_eq!(Matrix::from_h273(0), Some(Matrix::Identity));
        assert_eq!(Matrix::from_h273(2), None, "unspecified");
        // §8.2 TransferCharacteristics.
        assert_eq!(TransferFn::from_h273(16), Some(TransferFn::Pq));
        assert_eq!(TransferFn::from_h273(18), Some(TransferFn::Hlg));
        assert_eq!(TransferFn::from_h273(1), Some(TransferFn::Bt709));
        assert_eq!(TransferFn::from_h273(2), None);
        // §8.1 ColourPrimaries.
        assert_eq!(Primaries::from_h273(9), Some(Primaries::Bt2020));
        assert_eq!(Primaries::from_h273(12), Some(Primaries::DciP3));
        // mkv Range.
        assert_eq!(Range::from_mkv(1), Some(Range::Limited));
        assert_eq!(Range::from_mkv(2), Some(Range::Full));
        assert_eq!(Range::from_mkv(0), None);
    }

    #[test]
    fn unspecified_defaults_follow_resolution() {
        let hd = Colorimetry::default_for(1080);
        assert_eq!(hd.matrix, Matrix::Bt709);
        assert_eq!(hd.range, Range::Limited);
        let sd = Colorimetry::default_for(576);
        assert_eq!(sd.matrix, Matrix::Bt601);
        assert_eq!(sd.primaries, Primaries::Bt601);
        // The boundary: 720p is HD.
        assert_eq!(Colorimetry::default_for(720).matrix, Matrix::Bt709);
    }
}
