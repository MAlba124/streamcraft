//! Borrowed, validated views over a raw video frame (spec: typed format views). They
//! borrow, so misuse is a compile error and they add nothing over raw offset math — the
//! `audio/raw`'s `AudioFrameRef` analogue for planar video.
//!
//! [`VideoFrameRef::new`] validates **once** that the byte slice is exactly one
//! tightly-packed frame of `(width, height, pixfmt)` (via [`crate::geometry`]), then
//! hands out per-plane byte slices and their strides by pure offset arithmetic.
//! [`VideoFrameMut`] is the writable counterpart, for a `videotestsrc` /
//! `rawvideoparse` filling pool memory plane by plane.

use crate::format::{PixelFormat, VideoFormat};
use crate::geometry::{frame_size, plane_count, plane_geometry, PlaneGeometry};

/// A read-only view of one tightly-packed video frame's planes.
pub struct VideoFrameRef<'a> {
    bytes: &'a [u8],
    width: u32,
    height: u32,
    pixfmt: PixelFormat,
}

impl<'a> VideoFrameRef<'a> {
    /// Validate once that `bytes` is exactly one tightly-packed frame of
    /// `(width, height, pixfmt)`, then view it. `None` on any length mismatch, so a
    /// caller never indexes past the buffer.
    pub fn new(bytes: &'a [u8], width: u32, height: u32, pixfmt: PixelFormat) -> Option<Self> {
        if bytes.len() != frame_size(pixfmt, width, height) {
            return None;
        }
        Some(Self {
            bytes,
            width,
            height,
            pixfmt,
        })
    }

    /// Convenience: view against a concrete [`VideoFormat`] (ignores `fps`, which is
    /// timing, not layout).
    pub fn from_format(bytes: &'a [u8], format: VideoFormat) -> Option<Self> {
        Self::new(bytes, format.width, format.height, format.pixfmt)
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn pixfmt(&self) -> PixelFormat {
        self.pixfmt
    }

    /// All the frame's bytes (every plane, tightly packed).
    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// The number of planes in this frame.
    pub fn plane_count(&self) -> usize {
        plane_count(self.pixfmt)
    }

    /// The geometry (stride / height / size) of plane `plane`, or `None` if out of range.
    pub fn plane_geometry(&self, plane: usize) -> Option<PlaneGeometry> {
        plane_geometry(self.pixfmt, self.width, self.height, plane)
    }

    /// The tightly-packed bytes of plane `plane` and its row stride, or `None` if out of
    /// range. Pure offset math — the sum of earlier planes' sizes plus this plane's size.
    pub fn plane(&self, plane: usize) -> Option<(&'a [u8], usize)> {
        let (off, g) = self.plane_span(plane)?;
        Some((&self.bytes[off..off + g.size], g.stride))
    }

    /// Row `row` of plane `plane` (`stride` bytes), or `None` if either is out of range.
    pub fn plane_row(&self, plane: usize, row: usize) -> Option<&'a [u8]> {
        let (data, stride) = self.plane(plane)?;
        let start = row.checked_mul(stride)?;
        data.get(start..start + stride)
    }

    /// `(offset, geometry)` of a plane within the frame, or `None` if out of range.
    fn plane_span(&self, plane: usize) -> Option<(usize, PlaneGeometry)> {
        if plane >= self.plane_count() {
            return None;
        }
        let mut off = 0;
        for p in 0..plane {
            off += self.plane_geometry(p)?.size;
        }
        Some((off, self.plane_geometry(plane)?))
    }
}

/// A writable view of one tightly-packed video frame's planes — the producer side.
pub struct VideoFrameMut<'a> {
    bytes: &'a mut [u8],
    width: u32,
    height: u32,
    pixfmt: PixelFormat,
}

impl<'a> VideoFrameMut<'a> {
    /// Validate once that `bytes` is exactly one tightly-packed frame, then view it
    /// mutably. `None` on any length mismatch.
    pub fn new(
        bytes: &'a mut [u8],
        width: u32,
        height: u32,
        pixfmt: PixelFormat,
    ) -> Option<Self> {
        if bytes.len() != frame_size(pixfmt, width, height) {
            return None;
        }
        Some(Self {
            bytes,
            width,
            height,
            pixfmt,
        })
    }

    /// Convenience: view against a concrete [`VideoFormat`].
    pub fn from_format(bytes: &'a mut [u8], format: VideoFormat) -> Option<Self> {
        Self::new(bytes, format.width, format.height, format.pixfmt)
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn pixfmt(&self) -> PixelFormat {
        self.pixfmt
    }

    pub fn plane_count(&self) -> usize {
        plane_count(self.pixfmt)
    }

    pub fn plane_geometry(&self, plane: usize) -> Option<PlaneGeometry> {
        plane_geometry(self.pixfmt, self.width, self.height, plane)
    }

    /// The writable bytes of plane `plane` and its row stride, or `None` if out of range.
    pub fn plane_mut(&mut self, plane: usize) -> Option<(&mut [u8], usize)> {
        let (off, g) = self.plane_span(plane)?;
        Some((&mut self.bytes[off..off + g.size], g.stride))
    }

    /// Writable row `row` of plane `plane`, or `None` if either is out of range.
    pub fn plane_row_mut(&mut self, plane: usize, row: usize) -> Option<&mut [u8]> {
        let (off, g) = self.plane_span(plane)?;
        let start = off.checked_add(row.checked_mul(g.stride)?)?;
        self.bytes.get_mut(start..start + g.stride)
    }

    fn plane_span(&self, plane: usize) -> Option<(usize, PlaneGeometry)> {
        if plane >= self.plane_count() {
            return None;
        }
        let mut off = 0;
        for p in 0..plane {
            off += self.plane_geometry(p)?.size;
        }
        Some((off, self.plane_geometry(plane)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i420_view_splits_into_three_planes() {
        // 4x4 I420: Y is 16 bytes (0x00..), Cb is 4 bytes (0x10..), Cr is 4 (0x20..).
        let mut bytes = vec![0u8; frame_size(PixelFormat::I420, 4, 4)];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        let v = VideoFrameRef::new(&bytes, 4, 4, PixelFormat::I420).expect("exact frame");
        assert_eq!(v.plane_count(), 3);

        let (y, y_stride) = v.plane(0).unwrap();
        assert_eq!(y_stride, 4);
        assert_eq!(y.len(), 16);
        assert_eq!(y[0], 0);

        let (cb, cb_stride) = v.plane(1).unwrap();
        assert_eq!(cb_stride, 2);
        assert_eq!(cb.len(), 4);
        assert_eq!(cb[0], 16, "Cb starts right after the 16-byte Y plane");

        let (cr, _) = v.plane(2).unwrap();
        assert_eq!(cr[0], 20, "Cr starts after Y(16)+Cb(4)");

        assert!(v.plane(3).is_none(), "no fourth plane");
        // Row indexing.
        assert_eq!(v.plane_row(0, 0).unwrap(), &[0, 1, 2, 3]);
        assert_eq!(v.plane_row(0, 1).unwrap(), &[4, 5, 6, 7]);
        assert!(v.plane_row(0, 4).is_none(), "row past the plane height");
    }

    #[test]
    fn ragged_length_is_rejected() {
        let bytes = vec![0u8; frame_size(PixelFormat::I420, 4, 4) - 1];
        assert!(VideoFrameRef::new(&bytes, 4, 4, PixelFormat::I420).is_none());
        let over = vec![0u8; frame_size(PixelFormat::I420, 4, 4) + 1];
        assert!(VideoFrameRef::new(&over, 4, 4, PixelFormat::I420).is_none());
    }

    #[test]
    fn mut_view_writes_each_plane_then_reads_back() {
        let mut bytes = vec![0u8; frame_size(PixelFormat::Nv12, 4, 4)];
        {
            let mut m = VideoFrameMut::new(&mut bytes, 4, 4, PixelFormat::Nv12).unwrap();
            assert_eq!(m.plane_count(), 2);
            let (y, y_stride) = m.plane_mut(0).unwrap();
            assert_eq!(y_stride, 4);
            y.fill(0xAA);
            let (cbcr, c_stride) = m.plane_mut(1).unwrap();
            assert_eq!(c_stride, 4, "NV12 chroma stride is 2*ceil(w/2)");
            cbcr.fill(0x55);
            // Writable rows.
            m.plane_row_mut(0, 0).unwrap().copy_from_slice(&[1, 2, 3, 4]);
        }
        let v = VideoFrameRef::new(&bytes, 4, 4, PixelFormat::Nv12).unwrap();
        assert_eq!(v.plane(0).unwrap().0[4], 0xAA, "Y row 1 still 0xAA");
        assert_eq!(v.plane_row(0, 0).unwrap(), &[1, 2, 3, 4]);
        assert!(v.plane(1).unwrap().0.iter().all(|&b| b == 0x55));
    }

    #[test]
    fn single_plane_formats_have_one_plane() {
        let bytes = vec![7u8; frame_size(PixelFormat::Gray8, 5, 3)];
        let v = VideoFrameRef::new(&bytes, 5, 3, PixelFormat::Gray8).unwrap();
        assert_eq!(v.plane_count(), 1);
        let (p, stride) = v.plane(0).unwrap();
        assert_eq!(stride, 5);
        assert_eq!(p.len(), 15);
        assert!(v.plane(1).is_none());
    }
}
