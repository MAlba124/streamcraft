//! streamcraft-video — POD video format descriptions and typed *views* over the
//! opaque buffer: pixel formats, planes, strides, colorimetry (spec: Crate layout).
//! `VideoFrameRef::new(&buf, &format)` validates once, then exposes planes/strides;
//! no new buffer types, no traits, zero cost over raw offset math. Distinguishes
//! placement (`video/raw` vs `video/dmabuf`) so GPU↔CPU boundaries are ordinary
//! negotiation (spec: Device memory and sync points).
//!
//! TODO(milestone 5+): frame views feeding the hand-written VP8/VP9/AV1 codecs.

#![allow(dead_code)]
