//! pf-h264 — the H.264 / AVC (ITU-T Rec. H.264 | ISO/IEC 14496-10) codec plugin.
//!
//! Like [`pf-vp8`](../pf_vp8/index.html) this crate is **not** hand-written: it
//! wraps [`oxideav-h264`](https://github.com/OxideAV/oxideav-h264), a pure-Rust
//! H.264 decoder + encoder, adopted after review (2026-07-24). The spec's codec
//! taboo is FFI walls — a foreign allocator, threading model, and timestamp
//! semantics dragged in behind a boundary our batches can't cross (spec:
//! First-party codecs; Non-goals). `oxideav-h264` has none of that: pure Rust,
//! **zero `unsafe`**, no `build.rs`, MIT, and its code cross-references the H.264
//! spec clause-by-clause exactly like our own codec crates. Hand-writing a
//! conformant H.264 decoder (CAVLC + CABAC, intra + inter, deblocking, DPB
//! reordering, MMCO) is a multi-year effort; adopting this is the same
//! "buy, don't build" call as libpipewire for the device boundary — except this
//! one is still all-Rust and vendorable.
//!
//! ## Wire-or-reject decision: **WIRE** (evidence-based)
//!
//! The crates.io blurb is deliberately not trusted — at the published `0.1.7` it
//! reads *"under spec-driven rewrite, no decode/encode functionality yet"* and the
//! `lib.rs` header claims the crate is *"currently empty"*. Both are **stale**:
//! the same published `0.1.7` registers a real [`H264CodecDecoder`], ships a large
//! decode + encode integration-test suite (foreman P16x16 decode, B-slice CABAC,
//! multi-slice assembly, conformance corpus, encoder self-roundtrip), and its
//! README status matrix documents I/P/B reconstruction on both entropy paths. The
//! judgement here is by source + tests, not by the blurb (the vp8 rubric).
//!
//! ### Capability matrix (published `oxideav-h264` 0.1.7)
//!
//! | Area | Status in the backing crate | Wired by `H264Dec` |
//! | ---- | --------------------------- | ------------------ |
//! | Intra (I/IDR) decode | yes — Intra 4x4/8x8/16x16, I_PCM, chroma | **yes** |
//! | Inter (P) decode | yes — P_Skip/P_L0/P_8x8, ¼-pel MC, weighted | **yes** |
//! | Inter (B) decode | yes — direct (spatial+temporal), bipred | **yes** |
//! | CAVLC entropy | yes | yes (transparent) |
//! | CABAC entropy | yes | yes (transparent) |
//! | 4:2:0 8-bit output | yes → Yuv420P | **yes** (packed I420) |
//! | 4:2:2 / 4:4:4 output | yes → Yuv422P/Yuv444P | **no** — refused per-buffer |
//! | >8-bit (High10/High) | yes → Yuv*P*Le (u16) | **no** — refused per-buffer |
//! | Profiles | Baseline / Main / High family | inherited; 4:2:0 8-bit only |
//! | DPB reorder / POC output | yes (§C.4 bumping) | inherited (see reorder note) |
//! | Annex B framing | yes (start-code split) | **yes** — `h264/annexb` sink |
//! | AVCC framing | yes (`avcC` extradata) | **no** — v1 is Annex B only |
//!
//! So `H264Dec` wires the mainstream case — Baseline/Main/High **4:2:0 8-bit**
//! streams, intra **and** inter, both entropy coders, Annex B framing — and
//! refuses the exotic pixel layouts per-buffer (a bus `Warning` + drop) rather
//! than mislabelling them `i420`. That is the honest subset: everything the
//! `video/raw` `i420` vocabulary can carry today.
//!
//! ### Framing and reordering, stated honestly
//! - **Input**: one **Annex B access unit per buffer** on the `h264/annexb` sink
//!   (the demuxer contract). The backing decoder tolerates multiple slice NALs
//!   per access unit and assembles them (§7.4.1.2.4); it does not require exactly
//!   one NAL.
//! - **Reordering**: the decoder emits pictures in **display (POC) order** after
//!   internal DPB bumping, so decode-order ≠ output-order for B-frame streams.
//!   `H264Dec` stamps each emitted picture with the pts of the access unit whose
//!   `send_packet` produced it (decode-order pts). For Baseline / no-reorder
//!   streams that equals the display pts; for reordered streams it is a documented
//!   v1 caveat — downstream A/V sync should prefer container display timestamps
//!   until a proper DTS→PTS reorder buffer lands (backlog).
//!
//! ## Dependency situation (heavier than pf-vp8, called out)
//!
//! Unlike `oxideav-vp8` (zero deps with default features off), `oxideav-h264
//! 0.1.7` pulls `oxideav-core` and `thiserror` **mandatorily** — there is no
//! feature to make `oxideav-core` optional (checked across every published 0.1.x;
//! 0.1.7 is the latest). `oxideav-core` in turn pulls `serde_json` + `bytemuck`.
//! The decoder is consumed through `oxideav-core`'s `Decoder` trait
//! (`send_packet`/`receive_frame`/`flush`), so the dep is structural, not
//! incidental. That is the accepted cost of adopting a full H.264 decoder; it is
//! all pure Rust and the whole tree vendors.
//!
//! ## Nativization debt, tracked in PLAN.md
//! - **Decode into pool memory**: the backing decoder returns owned `VideoFrame`
//!   plane `Vec`s; [`H264Dec`] pays one packed-I420 copy into pool memory per
//!   frame. A `receive_frame_into()` (or the crate's `receive_arena_frame`, once
//!   it can target our pool) upstream contribution removes it — the element API
//!   here will not change when it lands.
//! - **PTS reorder buffer**: emit true display-order pts by pairing the DPB output
//!   against a small pts reorder queue keyed on decode order.
//! - **Wider pixel formats**: `i420_10` / `nv12` / 4:2:2 / 4:4:4 output once the
//!   `video/raw` vocabulary and the pipeline pool sizing grow to carry them.
//! - **AVCC sink family** (`h264/avcc`): length-prefixed framing from `avcC`
//!   extradata, for MP4/MKV sources that hand out AVCC rather than Annex B.
//! - **Conformance vectors**: the JVT/ITU-T H.264 conformance suite should run in
//!   CI before this plugin is trusted for the video milestones.

pub mod h264dec;

pub use h264dec::H264Dec;

use profluens_core::element::Element;
use profluens_core::registry::Registry;

/// Register this crate's elements for name-based construction (spec: Plugins —
/// `parse("… ! h264dec ! …")`). Typed `use` + constructor stays primary; this powers
/// `pf-launch` and one-liner tests. The descriptor is `&'static`, taken from a
/// throwaway default instance; [`H264Dec`] is config-free (dimensions come from the
/// SPS, announced at runtime).
pub fn register(registry: &mut Registry) {
    registry.register(H264Dec::new().desc());
}
