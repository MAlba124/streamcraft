//! Parse-path property plumbing shared by the fixed-format video elements (spec:
//! Plugins). `videotestsrc` / `rawvideoparse` take their `(width, height, pixfmt, fps)`
//! as constructor arguments; the registry builds them with [`DEFAULT_FORMAT`] and then
//! refines that from parsed props in `start()` via [`read_dims`] — the flacenc pattern
//! (ints for width/height, an interned name for pixfmt, a rational for fps). Absent props
//! keep the constructor value, so the typed [`VideoFormat`](crate::format::VideoFormat)
//! constructor and the parse path share one refinement path.

use streamcraft_core::ctx::Ctx;
use streamcraft_core::format::Value;
use streamcraft_core::time::Rational;

use crate::format::{PixelFormat, VideoFormat};

/// The default raster format the registry constructs a fixed-format element with —
/// 320x240 I420 @ 30 fps — before parse-path props (if any) refine it.
pub const DEFAULT_FORMAT: VideoFormat =
    VideoFormat::new(320, 240, PixelFormat::I420, Rational::new(30, 1));

/// Refine `format` from the `width` / `height` / `pixfmt` / `fps` props on `ctx`,
/// each overriding the corresponding field when present and valid; an absent or
/// malformed prop leaves the field unchanged (spec: Plugins — props override the
/// constructor). `width`/`height` are positive ints, `pixfmt` an interned format name
/// ([`PixelFormat::from_caps_name`]), `fps` a rational (`30/1`) with a non-zero
/// denominator. Returns the refined format.
pub fn read_dims(ctx: &Ctx, mut format: VideoFormat) -> VideoFormat {
    if let Some(Value::Int(w)) = ctx.prop("width") {
        if w > 0 {
            format.width = w as u32;
        }
    }
    if let Some(Value::Int(h)) = ctx.prop("height") {
        if h > 0 {
            format.height = h as u32;
        }
    }
    if let Some(Value::Id(id)) = ctx.prop("pixfmt") {
        if let Some(pf) = ctx.value_name(id).and_then(PixelFormat::from_caps_name) {
            format.pixfmt = pf;
        }
    }
    if let Some(Value::Rat(num, den)) = ctx.prop("fps") {
        if den != 0 {
            format.fps = Rational::new(num, den);
        }
    }
    format
}

/// Read a positive-int prop `name` from `ctx`, falling back to `default` when absent or
/// non-positive (spec: Plugins). For `frames` / `seed`, which are not part of the format.
pub fn read_u64(ctx: &Ctx, name: &str, default: u64) -> u64 {
    match ctx.prop(name) {
        Some(Value::Int(v)) if v >= 0 => v as u64,
        _ => default,
    }
}
