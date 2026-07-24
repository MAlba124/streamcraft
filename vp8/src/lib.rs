//! sc-vp8 — the VP8 (RFC 6386) codec plugin.
//!
//! Unlike the other codec crates this one is **not** hand-written: it wraps
//! [`oxideav-vp8`](https://github.com/OxideAV/oxideav-vp8), a pure-Rust VP8
//! decoder + encoder adopted after review (2026-07-24). The spec's codec taboo is
//! FFI walls — a foreign allocator, threading model, and timestamp semantics
//! dragged in behind a boundary our batches can't cross (spec: First-party codecs;
//! Non-goals). `oxideav-vp8` has none of that: pure Rust, **zero dependencies**
//! with default features off, no `build.rs`, no `unsafe`, MIT, and its code
//! cross-references RFC 6386 section-by-section exactly like our own codec crates.
//! Adopting it is the same "buy, don't build" call as libpipewire for the device
//! boundary — except this one is still all-Rust, all-safe, and vendorable.
//!
//! Nativization debt, tracked in PLAN.md:
//! - **Decode into pool memory**: the upstream decoder returns owned `Vec` planes;
//!   [`Vp8Dec`] pays one plane copy into pool memory per frame. The fix is a
//!   `decode_frame_into()` upstream PR (or vendored patch) — the element API here
//!   will not change when it lands.
//! - **Official conformance vectors**: upstream validates against a synthetic
//!   corpus plus a bidirectional out-of-process ffmpeg oracle (the exact dev-oracle
//!   pattern our spec prescribes); the webmproject `vp80-*` suite should still run
//!   in our CI before this plugin is trusted for milestone 6.
//! - **Stable SIMD**: upstream's SIMD is nightly `core::simd` (off by default; the
//!   scalar path is byte-exact-tested against it). Our doctrine is stable
//!   `std::arch` — a candidate upstream contribution.

pub mod vp8dec;

pub use vp8dec::Vp8Dec;
