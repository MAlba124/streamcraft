//! streamcraft-video — POD video format descriptions and typed *views* over the
//! opaque buffer: pixel formats, planes, strides, colorimetry (spec: Crate layout).
//! [`VideoFrameRef::new`] validates once, then exposes planes/strides; no new buffer
//! types, no traits, zero cost over raw offset math — the `-video` mirror of the
//! `-audio` crate. Distinguishes placement (`video/raw` vs `video/dmabuf`) so GPU↔CPU
//! boundaries are ordinary negotiation (spec: Device memory and sync points).
//!
//! The [`VideoTestSrc`] / [`RawVideoParse`] elements and the clock-driven
//! [`VideoCkSink`] bring the milestone-5 no-decoder video path up end-to-end
//! (`filesrc ! rawvideoparse ! videosink`; spec: Milestone applications §5).
//!
//! TODO(milestone 5+): frame views feeding the hand-written VP8/VP9/AV1 codecs.

#![allow(dead_code)]

pub mod format;
pub mod frame;
pub mod geometry;
pub mod parse;
pub mod sink;
pub mod testsrc;

pub use format::{
    PixelFormat, VideoFormat, FAMILY, FIELD_FPS, FIELD_HEIGHT, FIELD_PIXFMT, FIELD_WIDTH,
    RAW_ANY_OFFER,
};
pub use frame::{VideoFrameMut, VideoFrameRef};
pub use geometry::{
    frame_size, plane_count, plane_geometries, plane_geometry, plane_offset, PlaneGeometry,
};
pub use parse::RawVideoParse;
pub use sink::{VideoCkSink, VideoCkSinkStats, VideoRender};
pub use testsrc::{frame_pattern_byte, VideoTestSrc};
