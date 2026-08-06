//! NVR milestone app (spec: Milestone applications): N RTSP cameras in, a
//! directory of rotated, self-contained MKV segments + a live mosaic out. The
//! deliberately-broad stress test: live clocks, fan-out, segmented muxing with
//! back-patches, long-run memory flatness.

pub mod mosaic;
pub mod segment_sink;

pub use mosaic::Mosaic;
pub use segment_sink::MkvSegmentSink;
