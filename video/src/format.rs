//! POD video format vocabulary + the `video/raw` negotiation offer (spec: Formats;
//! Crate layout). Plain structs and free functions — no new buffer types, no traits,
//! mirroring `streamcraft-audio`'s `format` module.
//!
//! The `video/raw` family names four fields — `width` / `height` (Int px), `pixfmt`
//! (a categorical id like `i420`) and `fps` (a `Rat`, e.g. `30000/1001`) — matching the
//! string-keyed offer descriptors the pipeline interns at link time. [`PixelFormat::caps_name`]
//! is the single source of truth for the categorical names, so offers, raw-video parsing,
//! and frame views never drift.

use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use streamcraft_core::time::Rational;

/// The negotiation family for uncompressed (raster) video frames.
pub const FAMILY: &str = "video/raw";
/// Frame width in pixels (`Value::Int`).
pub const FIELD_WIDTH: &str = "width";
/// Frame height in pixels (`Value::Int`).
pub const FIELD_HEIGHT: &str = "height";
/// Pixel format, a categorical id — see [`PixelFormat::caps_name`].
pub const FIELD_PIXFMT: &str = "pixfmt";
/// Frame rate as an exact rational (`Value::Rat`), e.g. `30000/1001` for 29.97 fps.
pub const FIELD_FPS: &str = "fps";

/// A raster pixel format.
///
/// The two planar YUV formats ([`I420`](PixelFormat::I420),
/// [`Nv12`](PixelFormat::Nv12)) are 4:2:0 subsampled: the chroma planes are half width
/// and half height of luma. The two packed formats ([`Rgb24`](PixelFormat::Rgb24),
/// [`Gray8`](PixelFormat::Gray8)) are single-plane.
///
/// See [`crate::geometry`] for how each maps to planes, strides and sizes, including
/// the odd-dimension rounding convention for 4:2:0.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelFormat {
    /// Planar Y'CbCr 4:2:0, three planes (Y, then Cb, then Cr), 8 bits per sample.
    I420,
    /// Planar Y'CbCr 4:2:0 with interleaved chroma: two planes (Y, then Cb/Cr pairs).
    Nv12,
    /// Packed 8-bit R,G,B triples, one plane, 3 bytes per pixel.
    Rgb24,
    /// Single-plane 8-bit luma (greyscale), 1 byte per pixel.
    Gray8,
}

impl PixelFormat {
    /// The categorical name used in `video/raw` offers (`ValueDesc::Id(name)`).
    pub const fn caps_name(self) -> &'static str {
        match self {
            PixelFormat::I420 => "i420",
            PixelFormat::Nv12 => "nv12",
            PixelFormat::Rgb24 => "rgb24",
            PixelFormat::Gray8 => "gray8",
        }
    }

    pub fn from_caps_name(s: &str) -> Option<PixelFormat> {
        Some(match s {
            "i420" => PixelFormat::I420,
            "nv12" => PixelFormat::Nv12,
            "rgb24" => PixelFormat::Rgb24,
            "gray8" => PixelFormat::Gray8,
            _ => return None,
        })
    }

    /// Number of separately-addressed planes in a frame of this format.
    pub const fn plane_count(self) -> usize {
        match self {
            PixelFormat::I420 => 3,  // Y, Cb, Cr
            PixelFormat::Nv12 => 2,  // Y, interleaved CbCr
            PixelFormat::Rgb24 => 1, // packed RGB
            PixelFormat::Gray8 => 1, // packed luma
        }
    }

    /// True for the 4:2:0-subsampled YUV formats, whose chroma planes are half the
    /// luma resolution in each dimension (and so pull in the odd-dimension rounding
    /// convention — see [`crate::geometry`]).
    pub const fn is_subsampled_420(self) -> bool {
        matches!(self, PixelFormat::I420 | PixelFormat::Nv12)
    }
}

/// A concrete raster format: the runtime counterpart of a fixated `video/raw`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VideoFormat {
    pub width: u32,
    pub height: u32,
    pub pixfmt: PixelFormat,
    /// Frame rate. `fps.den == 0` is invalid; the fps grid ([`VideoFormat::frame_pts`])
    /// treats an invalid or zero rate as "no timing".
    pub fps: Rational,
}

impl VideoFormat {
    pub const fn new(width: u32, height: u32, pixfmt: PixelFormat, fps: Rational) -> Self {
        Self {
            width,
            height,
            pixfmt,
            fps,
        }
    }
}

// --- The `video/raw` negotiation offer ------------------------------------------------
//
// A runtime `VideoFormat` cannot be a `'static` offer (the descriptor layer needs
// `&'static` names), so a *parser* whose format is data-dependent advertises the broad
// [`RAW_ANY_OFFER`] and dictates the exact values downstream once runtime caps exist
// (`ctx.announce_format`, exactly like `wavparse`/`audioconvert`). A *fixed-format*
// element declares its own `static` offer using the [`FIELD_*`] constants and
// [`PixelFormat::caps_name`] — see [`crate::testsrc::VideoTestSrc`] for the shape.

static PIXFMT_VALUES: [ValueDesc; 4] = [
    ValueDesc::Id("i420"),
    ValueDesc::Id("nv12"),
    ValueDesc::Id("rgb24"),
    ValueDesc::Id("gray8"),
];

static RAW_ANY_FIELDS: [FieldDesc; 4] = [
    FieldDesc {
        field: FIELD_WIDTH,
        allowed: ConstraintDesc::Any,
        preferred: None,
    },
    FieldDesc {
        field: FIELD_HEIGHT,
        allowed: ConstraintDesc::Any,
        preferred: None,
    },
    FieldDesc {
        field: FIELD_PIXFMT,
        allowed: ConstraintDesc::Set(&PIXFMT_VALUES),
        preferred: None,
    },
    FieldDesc {
        field: FIELD_FPS,
        allowed: ConstraintDesc::Any,
        preferred: None,
    },
];

/// "Any raster video": matches any width/height/fps and any known [`PixelFormat`]. The
/// offer list a raw-video pad advertises when it accepts whatever the graph fixates —
/// the `audio/raw` `RAW_ANY_OFFER` analogue.
pub static RAW_ANY_OFFER: [OfferDesc; 1] = [OfferDesc {
    family: FAMILY,
    fields: &RAW_ANY_FIELDS,
}];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_names_round_trip_for_every_variant() {
        for f in [
            PixelFormat::I420,
            PixelFormat::Nv12,
            PixelFormat::Rgb24,
            PixelFormat::Gray8,
        ] {
            assert_eq!(PixelFormat::from_caps_name(f.caps_name()), Some(f));
        }
        assert_eq!(PixelFormat::from_caps_name("i420"), Some(PixelFormat::I420));
        assert_eq!(PixelFormat::from_caps_name("nope"), None);
    }

    #[test]
    fn plane_counts_match_the_format() {
        assert_eq!(PixelFormat::I420.plane_count(), 3);
        assert_eq!(PixelFormat::Nv12.plane_count(), 2);
        assert_eq!(PixelFormat::Rgb24.plane_count(), 1);
        assert_eq!(PixelFormat::Gray8.plane_count(), 1);
    }

    #[test]
    fn only_yuv_420_is_subsampled() {
        assert!(PixelFormat::I420.is_subsampled_420());
        assert!(PixelFormat::Nv12.is_subsampled_420());
        assert!(!PixelFormat::Rgb24.is_subsampled_420());
        assert!(!PixelFormat::Gray8.is_subsampled_420());
    }
}
