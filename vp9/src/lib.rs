//! sc-vp9 — the VP9 (v0.7 bitstream spec) codec plugin.
//!
//! Like `sc-vp8`, this crate is **not** hand-written: it wraps
//! [`oxideav-vp9`](https://github.com/OxideAV/oxideav-vp9), a pure-Rust, clean-room VP9
//! decoder adopted after review (2026-07-24). The spec's codec taboo is FFI walls — a
//! foreign allocator, threading model, and timestamp semantics dragged in behind a
//! boundary our batches can't cross (spec: First-party codecs; Non-goals).
//! `oxideav-vp9` has none of that: pure Rust, **zero `unsafe`**, **no `build.rs`**, MIT,
//! and its code cross-references the VP9 v0.7 spec section-by-section exactly like our
//! own codec crates. Adopting it is the same "buy, don't build" call as `sc-vp8`.
//!
//! # Adoption verdict: WIRE (with a loudly-enforced subset)
//!
//! The evaluation judged the **published `0.0.12`** — the version this workspace's
//! `Cargo.lock` pins — not the crate's git HEAD, whose README/CHANGELOG advertise a far
//! more complete codec (inter decode, a pixel-accurate encoder, superframe splitting)
//! that is a **month newer than 0.0.12 and has never been published**. (crates.io's
//! stale "orphan-rebuild scaffold" blurb is wrong in the other direction — 0.0.12 is a
//! real, working decoder — but the git HEAD is equally not what we depend on.) Judged
//! on 0.0.12 itself:
//!
//! | Capability            | 0.0.12 status                                             |
//! |-----------------------|-----------------------------------------------------------|
//! | Intra (key) frames    | **Decode, byte-exact** against an embedded corpus         |
//! | Intra-only frames     | **Decode, byte-exact**                                    |
//! | Inter (P-)frames      | **`Error::Unsupported`** — no reference-buffer state      |
//! | `show_existing_frame` | **`Error::Unsupported`**                                  |
//! | Profiles / chroma     | 4:2:0, 4:2:2, 4:4:0, 4:4:4 (corpus covers 4:2:0 + 4:4:4)  |
//! | Bit depth             | 8 / 10 / 12-bit (10/12 as LE `u16` pairs)                 |
//! | Superframe split      | **Not provided** (no `split_superframe` in 0.0.12)        |
//! | Encoder               | **None** — `encode_vp9` returns `Error::NotImplemented`   |
//! | Robustness            | Garbage in → clean `Err`, never a panic (fuzz contract)   |
//!
//! 0.0.12 genuinely produces **correct pictures from real VP9 intra streams** — a
//! documented, loudly-enforced subset — so per the task rubric it is WIRED rather than
//! rejected. [`Vp9Dec`] decodes key / intra-only frames and treats every frame the
//! decoder cannot handle (inter, `show_existing`, corrupt) as per-buffer weather: a bus
//! `Warning`, then dropped, resync at the next keyframe. A keyframe-only stream
//! (`ffmpeg … -g 1`) therefore decodes fully; a normal GOP-coded stream decodes only
//! its key frames until upstream gains inter support.
//!
//! Because 0.0.12 does not split VP9 **superframes** (Annex B framing — a hidden
//! alt-ref plus its visible companion packed into one container packet), [`Vp9Dec`]
//! splits them itself before decode (a trailing-byte parse — [`superframe`]), so
//! container packets Just Work.
//!
//! # Adoption debt (tracked in PLAN.md)
//!
//! - **Inter decode**: the single biggest gap. 0.0.12 is intra-only; a normal video
//!   plays only its key frames. The upstream git HEAD already implements inter decode
//!   and superframe splitting — re-evaluate (and drop this element's own superframe
//!   split + inter-frame drops) when a version carrying them is **published**.
//! - **Decode into pool memory**: the upstream decoder returns owned planes; [`Vp9Dec`]
//!   pays one packed-planar copy into pool memory per frame. A `decode_frame_into()`
//!   upstream PR (or vendored patch) removes it; the element API here will not change.
//! - **`oxideav-core` dependency**: unlike `oxideav-vp8` (truly zero-dep), 0.0.12
//!   depends unconditionally on `oxideav-core` for a no-op `register!` plugin hook,
//!   pulling `serde_json` / `thiserror` / `bytemuck` transitively. These are already in
//!   the workspace lock via the sibling `sc-h264` / `sc-h265` / `sc-av1` stubs, so no
//!   *new* weight is added — but it is a heavier tail than the vp8 adoption, and no
//!   published version drops it (0.0.12 is the latest). A candidate upstream feature-gate.
//! - **Official conformance vectors**: 0.0.12 validates against its own clean-room
//!   corpus; the webmproject `vp9-*` conformance suite should run in our CI before this
//!   plugin is trusted for a video milestone.

pub mod superframe;
pub mod vp9dec;

pub use vp9dec::Vp9Dec;
