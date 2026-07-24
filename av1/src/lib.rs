//! sc-av1 — the AV1 (AOMedia Video 1) codec plugin.
//!
//! Like sc-vp8, this crate is **not** hand-written: it wraps
//! [`oxideav-av1`](https://github.com/OxideAV/oxideav-av1), a pure-Rust AV1
//! decoder + encoder adopted after review (2026-07-24). The spec's codec taboo is
//! FFI walls — a foreign allocator, threading model, and timestamp semantics
//! dragged in behind a boundary our batches can't cross (spec: First-party codecs;
//! Non-goals). `oxideav-av1` has none of that: pure Rust, no `build.rs`, one line of
//! `unsafe` in the whole tree, MIT, and its code cross-references the AV1 Bitstream &
//! Decoding Process Specification section-by-section exactly like our own codec
//! crates. Adopting it is the same "buy, don't build" call as libpipewire for the
//! device boundary — except this one is still all-Rust, all-safe, and vendorable.
//!
//! ## Why it passes the rubric (evidence, not the crates.io blurb)
//!
//! The crates.io description still reads "orphan-rebuild scaffold pending clean-room
//! re-implementation" and the top of the upstream `lib.rs` still opens with "Status:
//! orphan-rebuild scaffold … the decoder/encoder pipeline is not wired up yet". That
//! header is **stale** — it narrates the crate's first ~146 header-parsing "rounds"
//! while the code has since reached r428 with a complete decode + encode pipeline
//! (sc-vp8 carried the identical stale blurb while being production-complete). Judged
//! by source / tests / CHANGELOG instead:
//!
//! - **It decodes real pictures.** The spec-faithful driver
//!   (`decoder::SpecDecodeSession` / `decode_av1`) runs the §5.11 partition-syntax
//!   walk, the §8.2 symbol decoder over the full §9.4 CDF tables, the §7.11–7.13
//!   reconstruction chain, and the whole §7.4 post chain (deblock, CDEF, superres,
//!   loop restoration, film grain). Upstream's conformance corpus decodes **32+
//!   streams byte-identically to a third-party AV1 decoder** (README r405/r408) — the
//!   full intra surface, KEY + P inter GOPs with `show_existing_frame`, multi-tile,
//!   128×128 superblocks, delta-q, quantizer matrices, scaled references, intra block
//!   copy, and 10/12-bit / 4:2:2 / 4:4:4 output.
//! - **It has an in-crate encoder.** `encoder::encode_key_frame_yuv420{,_with_q}` and
//!   `encode_gop_yuv420_with_q` produce conformance-grade IVF that decodes
//!   byte-identically to the encoder's own reconstruction (and byte-identically to the
//!   *input* on the lossless `q == 0` arm). That gives [`Av1Dec`] a real encode→decode
//!   roundtrip test the way sc-vp8 has, no external fixtures required.
//! - **It never panics on garbage.** Malformed streams surface a typed,
//!   `#[non_exhaustive]` `Error` (never a panic); there are `cargo-fuzz` targets
//!   (`decode`, `obu`, `roundtrip`) and a checked-in fuzz-regression suite upstream.
//!
//! ## The decode unit and the bounded output subset [`Av1Dec`] enforces
//!
//! One **AV1 temporal unit** (a §7.5 low-overhead OBU bytestream — the payload a
//! container hands out per packet: Matroska `V_AV1`, an ISOBMFF `av01` sample, or one
//! IVF record) arrives per buffer on the sink pad, and raw video leaves on the src
//! pad. A temporal unit yields **zero, one, or several** shown frames (an invisible
//! altref updates references and emits nothing; a later `show_existing_frame` unit
//! emits a stored frame) — the upstream `SpecDecodeSession` applies the §7.4 output
//! discipline internally, so [`Av1Dec`] simply emits every frame the session returns.
//! This is the AV1 analogue of the way sc-vp8 handles `show_frame == 0`.
//!
//! The upstream decoder is broad, but our raw-video vocabulary
//! ([`streamcraft_video`]) names only four 8-bit pixel formats. [`Av1Dec`] therefore
//! **wires the two AV1 output shapes that vocabulary can express and rejects the rest
//! loudly** (a bus `Warning` + drop, the per-buffer error scope):
//!
//! | AV1 output                     | streamcraft pixfmt | status                    |
//! |--------------------------------|--------------------|---------------------------|
//! | 8-bit 4:2:0 (`Y,U,V`)          | `i420`             | wired                     |
//! | 8-bit monochrome (`Y`)         | `gray8`            | wired                     |
//! | 8-bit 4:2:2 / 4:4:4            | —                  | rejected (no vocab name)  |
//! | 10/12-bit (2-byte samples)     | —                  | rejected (no vocab name)  |
//!
//! When [`streamcraft_video`] grows `i422` / `i444` / `i420_10` etc., the map here is
//! the only thing that changes — the element contract does not.
//!
//! Nativization debt, tracked in PLAN.md:
//! - **Mandatory `oxideav-core` dep**: unlike sc-vp8 (zero-dep), `oxideav-av1` pulls
//!   `oxideav-core` (→ `serde_json` + `thiserror` + `bytemuck`) unconditionally, only
//!   to register itself into the `oxideav-meta` runtime registry we do not use
//!   (`oxideav_core::register!("av1", …)`). No feature gates it off in 0.1.16. It is
//!   already in our lockfile (shared with the h264/h265/vp9 adoptions) and is the same
//!   family of dep as libpipewire — but a `default-features = false` that drops the
//!   registry bridge would make this crate zero-transitive-dep like sc-vp8. Flagged
//!   for an upstream contribution.
//! - **Decode into pool memory**: the upstream decoder returns owned `Vec` planes;
//!   [`Av1Dec`] pays one plane copy into pool memory per frame (the same cost sc-vp8
//!   pays). A `decode_into()` upstream PR would remove it without changing this API.
//! - **Official conformance vectors**: upstream validates against its own corpus + a
//!   third-party decoder; the AV1 **argon** conformance vectors should still run in our
//!   CI before this plugin is trusted for a video milestone.

pub mod av1dec;

pub use av1dec::Av1Dec;

use streamcraft_core::element::Element;
use streamcraft_core::registry::Registry;

/// Register this crate's elements for name-based construction (spec: Plugins —
/// `parse("… ! av1dec ! …")`). Typed `use` + constructor stays primary; this powers
/// `scraft-launch` and one-liner tests. The descriptor is `&'static`, taken from a
/// throwaway default instance; [`Av1Dec`] is config-free (dimensions come from the
/// sequence header, announced at runtime).
pub fn register(registry: &mut Registry) {
    registry.register(Av1Dec::new().desc());
}
