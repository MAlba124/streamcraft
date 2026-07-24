//! sc-h265 — the H.265 / HEVC (ITU-T H.265 | ISO/IEC 23008-2) codec plugin.
//!
//! Like sc-vp8, this crate is **not** hand-written: it wraps
//! [`oxideav-h265`](https://github.com/OxideAV/oxideav-h265), a pure-Rust HEVC
//! decoder adopted after review (2026-07-24). The spec's codec taboo is FFI walls —
//! a foreign allocator, threading model, and timestamp semantics dragged in behind a
//! boundary our batches can't cross (spec: First-party codecs; Non-goals).
//! `oxideav-h265` has none of that: **pure Rust, zero `unsafe`, no `build.rs`, MIT**,
//! and its source cross-references the H.265 recommendation clause-by-clause exactly
//! like our own codec crates. Hand-writing a conformant HEVC decoder (CABAC, intra +
//! inter reconstruction, deblocking, SAO, the DPB / POC / RPS reference machinery,
//! range + screen-content extensions) is a multi-person-year undertaking; adopting a
//! correct all-Rust one is the same "buy, don't build" call as libpipewire for the
//! device boundary — except this one stays all-Rust and vendorable.
//!
//! ## Why it passed the rubric (vet, 2026-07-24)
//!
//! The crates.io blurb still calls it a "bitstream parser + decoder scaffold", but —
//! exactly as with sc-vp8's stale blurb — the README / CHANGELOG / tests / source
//! tell a different, production-complete story, and a byte-exact test against a real
//! x265-produced stream settled it:
//!
//! * **It decodes real HEVC to correct pictures.** A tiny 16×16 Main-profile IDR and
//!   an IDR+P pair, both produced by system **x265** (via ffmpeg) and stripped to
//!   VPS/SPS/PPS/VCL, decode **byte-exact against ffmpeg's own decoder** through
//!   [`oxideav_h265::decode_annexb_sequence`] (0 differing bytes). See
//!   `tests/h265dec.rs`. Upstream's own `tests/annexb_intra_fixtures.rs` extends this
//!   to Main/Main10/4:2:2/4:4:4, WPP, tiles, multi-slice, weighted prediction, PCM,
//!   B-pyramids with POC reorder — all byte-exact against committed expected YUV, plus
//!   self-built conformance pins for the RExt/SCC tail (RDPCM, palette, CCP, ACT,
//!   IBC).
//! * **Direct decode API, no registry needed.** [`oxideav_h265::SequenceDecoder`] /
//!   `decode_annexb_sequence` / [`oxideav_h265::DecodedFrame`] / `Picture::to_planar_u8`
//!   are the whole surface [`H265Dec`] uses. The `oxideav_core` registry glue
//!   (`make_decoder`, `RuntimeContext`) is untouched.
//! * **Never panics on garbage.** Truncations, bit-flips of the fixture, and hundreds
//!   of random start-code-prefixed blobs all return `Err`, never unwind.
//! * **Zero `unsafe`, no `build.rs`, MIT.**
//!
//! Capability actually wired here: **Main / Main-Still / Main10 profiles, 4:2:0
//! 8-bit output**, intra + inter (P and B) pictures, Annex B access-unit-per-buffer
//! input. See [`h265dec`] for the runtime model.
//!
//! ## The one wart, and the nativization debt (tracked in PLAN.md)
//!
//! - **Non-zero dependency, unlike sc-vp8.** At 0.0.9 (the only non-yanked version)
//!   `oxideav-h265` depends unconditionally on `oxideav-core`, which pulls
//!   `serde_json` + `thiserror` + `bytemuck` (and their build-time proc-macro tree)
//!   transitively. `oxideav-core` is used **only** in the crate's `decoder` / `encoder`
//!   / registry modules — not in the decode path we call — but there is no feature
//!   flag to compile those out, so the transitive deps come along for the ride. This
//!   is confined to this plugin crate (core stays zero-dep). The fix is upstream:
//!   feature-gate the registry (or a version that drops the mandatory core dep) — a
//!   candidate contribution; the element API here will not change when it lands.
//! - **Decode into pool memory.** The upstream decoder returns owned `Vec<i32>`
//!   planes; [`H265Dec`] pays one clip-and-narrow copy into pool memory per frame
//!   (`Picture::to_planar_u8`). A `decode_into` upstream PR (or vendored patch) is the
//!   fix; the element API will not change.
//! - **Wider output formats.** Main10 / 4:2:2 / 4:4:4 / monochrome decode correctly
//!   upstream but are dropped here (bus `Warning`) because `video/raw` has no pixfmt
//!   vocabulary for them yet. Adding `i420_10le` / `i422` / `i444` / `gray8` literals
//!   + `to_planar_le16` packing is the follow-up when a consumer needs them.
//! - **Official conformance vectors.** Upstream validates against a staged corpus + a
//!   black-box reference decoder; the JCT-VC HEVC conformance suite should run in our
//!   CI before this plugin is trusted for the video milestones.
//! - **hvcC / length-prefixed input.** Only Annex B access-unit framing is wired
//!   (`h265/annexb`). Length-prefixed (mkv/MP4 hvcC) transport is supported upstream
//!   (`hvcc` module, `split_length_prefixed`) and is the follow-up when a demuxer
//!   emits it.

pub mod h265dec;

pub use h265dec::H265Dec;
