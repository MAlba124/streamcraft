//! Frame geometry as plain functions (spec: Formats; Crate layout — POD helpers, no
//! traits). Given a [`PixelFormat`] and `(width, height)`, these compute per-plane
//! `(stride, height, size)` and the total frame size — the offset math a
//! [`VideoFrameRef`](crate::frame::VideoFrameRef) validates against and indexes with.
//!
//! Strides here are the **tight** (minimum) strides: `stride == bytes_per_row`, no
//! padding. A real device may negotiate a wider stride via pool alignment (spec: Pool
//! negotiation is decoupled); a frame view can then be built over that stride
//! explicitly. Tight strides are what a `rawvideoparse` / `videotestsrc` produce.
//!
//! ## The 4:2:0 odd-dimension convention
//!
//! For the 4:2:0-subsampled formats ([`I420`](PixelFormat::I420),
//! [`Nv12`](PixelFormat::Nv12)) each chroma plane is half the luma resolution in each
//! dimension. When a luma dimension is **odd**, we **round the chroma dimension up**:
//! `chroma = (luma + 1) / 2` (i.e. `ceil(luma / 2)`). This is the FFmpeg/GStreamer
//! convention: it guarantees the trailing odd luma column/row still has a covering
//! chroma sample, so no chroma information is silently dropped. E.g. a 3×3 I420 frame
//! has 2×2 chroma planes. The alternative (round down) would leave the edge pixels
//! without chroma; we reject it. Codecs generally require even dimensions anyway; this
//! convention just makes odd sizes well-defined rather than a panic.

use crate::format::PixelFormat;

/// `ceil(n / 2)` without overflow — the 4:2:0 chroma dimension from a luma dimension.
#[inline]
const fn half_up(n: u32) -> u32 {
    (n + 1) / 2
}

/// One plane's tight geometry: its row stride in bytes, its height in rows, and its
/// total byte size (`stride * height`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PlaneGeometry {
    /// Bytes per row (the tight stride: `= bytes_per_row`, no padding).
    pub stride: usize,
    /// Number of rows in this plane.
    pub height: usize,
    /// Total bytes in this plane (`stride * height`).
    pub size: usize,
}

/// The number of planes in a frame of `pixfmt` — the length of [`plane_geometries`].
pub const fn plane_count(pixfmt: PixelFormat) -> usize {
    pixfmt.plane_count()
}

/// The geometry of plane `plane` (0-based) of a `width`×`height` frame in `pixfmt`, or
/// `None` if `plane >= plane_count(pixfmt)`.
///
/// Layouts (tight strides):
/// * `I420`: plane 0 = Y (`w`×`h`), plane 1 = Cb (`ceil(w/2)`×`ceil(h/2)`),
///   plane 2 = Cr (same as Cb).
/// * `Nv12`: plane 0 = Y (`w`×`h`), plane 1 = interleaved CbCr
///   (`2*ceil(w/2)` bytes/row × `ceil(h/2)` rows).
/// * `Rgb24`: one packed plane, `3*w` bytes/row × `h` rows.
/// * `Gray8`: one packed plane, `w` bytes/row × `h` rows.
pub fn plane_geometry(
    pixfmt: PixelFormat,
    width: u32,
    height: u32,
    plane: usize,
) -> Option<PlaneGeometry> {
    let w = width as usize;
    let h = height as usize;
    let cw = half_up(width) as usize; // chroma width (4:2:0)
    let ch = half_up(height) as usize; // chroma height (4:2:0)

    let g = |stride: usize, height: usize| PlaneGeometry {
        stride,
        height,
        size: stride * height,
    };

    Some(match (pixfmt, plane) {
        (PixelFormat::I420, 0) => g(w, h),
        (PixelFormat::I420, 1) | (PixelFormat::I420, 2) => g(cw, ch),
        (PixelFormat::Nv12, 0) => g(w, h),
        // Interleaved Cb/Cr: two chroma bytes per chroma column → stride 2*cw.
        (PixelFormat::Nv12, 1) => g(2 * cw, ch),
        (PixelFormat::Rgb24, 0) => g(3 * w, h),
        (PixelFormat::Gray8, 0) => g(w, h),
        _ => return None,
    })
}

/// Every plane's geometry, in plane order.
pub fn plane_geometries(pixfmt: PixelFormat, width: u32, height: u32) -> Vec<PlaneGeometry> {
    (0..plane_count(pixfmt))
        .map(|p| plane_geometry(pixfmt, width, height, p).expect("plane in range"))
        .collect()
}

/// The byte offset of plane `plane` within a tightly-packed frame (the sum of all
/// earlier planes' sizes), or `None` if the plane is out of range.
pub fn plane_offset(pixfmt: PixelFormat, width: u32, height: u32, plane: usize) -> Option<usize> {
    if plane >= plane_count(pixfmt) {
        return None;
    }
    let mut off = 0;
    for p in 0..plane {
        off += plane_geometry(pixfmt, width, height, p)?.size;
    }
    Some(off)
}

/// Total bytes in a tightly-packed frame: the sum of every plane's size.
pub fn frame_size(pixfmt: PixelFormat, width: u32, height: u32) -> usize {
    (0..plane_count(pixfmt))
        .map(|p| {
            plane_geometry(pixfmt, width, height, p)
                .expect("plane in range")
                .size
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Table-driven: (pixfmt, w, h) -> expected per-plane (stride, height, size), and the
    // total frame size (which must equal the sum of the plane sizes).
    struct Case {
        pixfmt: PixelFormat,
        w: u32,
        h: u32,
        planes: &'static [(usize, usize)], // (stride, height) per plane
    }

    const CASES: &[Case] = &[
        // Even dimensions — the ordinary path.
        Case {
            pixfmt: PixelFormat::I420,
            w: 4,
            h: 4,
            planes: &[(4, 4), (2, 2), (2, 2)],
        },
        Case {
            pixfmt: PixelFormat::Nv12,
            w: 4,
            h: 4,
            planes: &[(4, 4), (4, 2)], // chroma stride = 2*ceil(4/2) = 4
        },
        Case {
            pixfmt: PixelFormat::Rgb24,
            w: 4,
            h: 4,
            planes: &[(12, 4)],
        },
        Case {
            pixfmt: PixelFormat::Gray8,
            w: 4,
            h: 4,
            planes: &[(4, 4)],
        },
        // Odd dimensions — the rounding convention under test: chroma rounds UP.
        Case {
            pixfmt: PixelFormat::I420,
            w: 3,
            h: 3,
            // Y 3x3, chroma ceil(3/2)=2 → 2x2 each.
            planes: &[(3, 3), (2, 2), (2, 2)],
        },
        Case {
            pixfmt: PixelFormat::I420,
            w: 5,
            h: 3,
            // Y 5x3, chroma ceil(5/2)=3 wide, ceil(3/2)=2 tall.
            planes: &[(5, 3), (3, 2), (3, 2)],
        },
        Case {
            pixfmt: PixelFormat::Nv12,
            w: 3,
            h: 5,
            // Y 3x5, chroma stride 2*ceil(3/2)=4, height ceil(5/2)=3.
            planes: &[(3, 5), (4, 3)],
        },
        Case {
            pixfmt: PixelFormat::Rgb24,
            w: 3,
            h: 5,
            planes: &[(9, 5)],
        },
    ];

    #[test]
    fn plane_geometry_table() {
        for c in CASES {
            assert_eq!(
                plane_count(c.pixfmt),
                c.planes.len(),
                "{:?} {}x{} plane count",
                c.pixfmt,
                c.w,
                c.h
            );
            let mut total = 0usize;
            let mut off = 0usize;
            for (i, &(stride, height)) in c.planes.iter().enumerate() {
                let want = PlaneGeometry {
                    stride,
                    height,
                    size: stride * height,
                };
                let got = plane_geometry(c.pixfmt, c.w, c.h, i).expect("plane in range");
                assert_eq!(got, want, "{:?} {}x{} plane {i}", c.pixfmt, c.w, c.h);
                assert_eq!(
                    plane_offset(c.pixfmt, c.w, c.h, i),
                    Some(off),
                    "{:?} {}x{} plane {i} offset",
                    c.pixfmt,
                    c.w,
                    c.h
                );
                off += want.size;
                total += want.size;
            }
            assert_eq!(
                frame_size(c.pixfmt, c.w, c.h),
                total,
                "{:?} {}x{} total = sum of planes",
                c.pixfmt,
                c.w,
                c.h
            );
            // One past the last plane is out of range.
            assert_eq!(plane_geometry(c.pixfmt, c.w, c.h, c.planes.len()), None);
            assert_eq!(plane_offset(c.pixfmt, c.w, c.h, c.planes.len()), None);
        }
    }

    #[test]
    fn odd_420_chroma_rounds_up_not_down() {
        // The documented convention: 3x3 I420 chroma is 2x2 (ceil), never 1x1 (floor).
        let cb = plane_geometry(PixelFormat::I420, 3, 3, 1).unwrap();
        assert_eq!((cb.stride, cb.height), (2, 2), "3x3 chroma rounds up to 2x2");
        // A 1x1 frame still has a 1x1 chroma plane (ceil(1/2) = 1), never 0x0.
        let cb1 = plane_geometry(PixelFormat::I420, 1, 1, 1).unwrap();
        assert_eq!((cb1.stride, cb1.height), (1, 1), "1x1 chroma is 1x1, never empty");
    }

    #[test]
    fn known_total_sizes() {
        // 4:2:0 total = w*h * 3/2 for even dims; RGB = 3*w*h; gray = w*h.
        assert_eq!(frame_size(PixelFormat::I420, 4, 4), 16 + 4 + 4);
        assert_eq!(frame_size(PixelFormat::Nv12, 4, 4), 16 + 8);
        assert_eq!(frame_size(PixelFormat::Rgb24, 4, 4), 48);
        assert_eq!(frame_size(PixelFormat::Gray8, 4, 4), 16);
        // A common test resolution.
        assert_eq!(frame_size(PixelFormat::I420, 16, 16), 256 + 64 + 64);
    }
}
