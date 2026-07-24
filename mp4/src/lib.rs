//! sc-mp4 — a hand-written, dependency-free ISO base media file format (ISO-BMFF, ISO/IEC
//! 14496-12) MP4 container **demuxer** for streamcraft.
//!
//! An MP4 is a tree of typed, four-CC-keyed **boxes**: a `size` + `type` header, then either
//! child boxes (a container) or a typed payload (a leaf). This crate implements the read side
//! entirely by hand — no libav, no mp4parse — against the spec summary in `spec/NOTES.md`
//! (which records the exact ISO/IEC 14496-12 / 14496-15 editions each `§` citation refers to,
//! since the ISO texts are copyrighted and cannot be vendored the way the FLAC/Matroska/Ogg
//! RFCs are). Review means reading the code against that spec text.
//!
//! Layout mirrors the `sc-mkv` plugin — the nearest analog (a hand-written container with a
//! box/element split + spec notes in-tree): a thin `lib.rs` re-exporting the public surface,
//! submodules for the work, and the spec provenance in `spec/`.
//!
//! ## What exists
//! - [`boxes`] — the hand-written box grammar: the `size`/`type`/`largesize`/`uuid` header
//!   (§4.2), container recursion ([`boxes::for_each_child`]/[`boxes::find_child`]), and the
//!   FullBox leaf parsers the sample-table resolver needs (`mvhd`, `tkhd`, `mdhd`, `stsd`,
//!   `stts`, `ctts` v0/v1, `stsz`/`stz2`, `stsc`, `stco`/`co64`, `stss`, `elst`). Every
//!   parser is fallible and bounds-checked (untrusted input); unit-tested.
//! - [`codec`] — the sample-entry (`stsd`) parser and the four-CC → announce-family map
//!   ([`codec::family_for`]). It **reuses `sc-mkv`'s** `nal_head_from_config` + [`Reframer`]
//!   for the `avcC`/`hvcC` → Annex B reframing (the identical ISO/IEC 14496-15 configuration
//!   record rides both an MKV `CodecPrivate` and an MP4 `avc1`/`hvc1` sample entry — one
//!   parser, tested once; see `spec/NOTES.md`, "Reframer reuse").
//! - [`Mp4Reader`] — the two-phase parse engine ([`reader`]): **resolution** folds the `moov`
//!   sample tables into one file-offset-sorted [`reader::Sample`] list per track
//!   (`(offset, size, dts, pts, sync)`; `pts = dts + ctts` in media ticks, i128-safe, with a
//!   single-entry edit-list shift), then **streaming** slices each sample out of the file by
//!   absolute offset as its bytes arrive on the sink pad. Fragmented (`moof`) files are
//!   detected and errored loudly.
//! - [`Mp4Demux`] — the streamcraft demuxer **element** ([`element`]): a progressive MP4 byte
//!   stream in on `sink`, one **dynamic src pad per track** out, the family-mapped reframing,
//!   and the same **pool-bounded emission discipline** (`try_alloc` + bounded carry +
//!   `alloc_exact` EOS flush) as `MkvDemux` — unbounded `ctx.alloc` in a demuxer is banned
//!   (it OOM'd a real movie). Track discovery is constructor-supplied ([`Mp4Demux::new`]
//!   takes the file head through `moov`), because a mid-pipeline element gets no input during
//!   preroll and the scheduler freezes topology after preroll.
//!
//! ## Codec families (spec: RFC 6381 four-CCs; §8.5.2)
//! `avc1`/`avc3` → `h264/annexb` (avcC reframe; parameter sets once), `hvc1`/`hev1` →
//! `h265/annexb`, `vp09` → `vp9`, `av01` → `av1`, `Opus` → `opus`, `mp4a`(AAC) → `bytes`
//! (raw AAC access units — no decoder in-tree yet), anything else → `bytes`.
//!
//! ## Not yet (v1 scope; see `spec/NOTES.md`)
//! - **Fragmented MP4** (`moof`) — detected + errored; the resolver is `stbl`-only.
//! - **Multi-entry edit lists** — single media-time shift only.
//! - **A muxer** — a separate follow-up (the box grammar is symmetric).
//! - **Codec-init via caps** — the head bytes are a constructor argument for now (mirroring
//!   `MkvDemux`), with the `avcC`/`hvcC` parameter sets emitted as an Annex B head instead.

#![deny(unsafe_code)]

pub mod boxes;
pub mod codec;
pub mod element;
pub mod reader;

pub use codec::{family_for, SampleEntry};
pub use element::Mp4Demux;
pub use reader::{Mp4Error, Mp4Reader, ResolvedSample, Sample, SamplePayload, Track};

// Re-export the shared reframer surface so downstream code has one import path.
pub use sc_mkv::{nal_head_from_config, Reframer, ReframeError};

use streamcraft_core::element::Element;
use streamcraft_core::registry::Registry;

/// Register this crate's element for name-based lookup (spec: Plugins — `--list` and
/// descriptor introspection). Typed `use` + constructor stays the primary path.
///
/// **`Mp4Demux` is not constructible by name yet.** Its descriptor carries `make_default:
/// None`, so [`Registry::parse`](streamcraft_core::registry::Registry::parse) cannot build
/// it — by design: `Mp4Demux` needs the file **head bytes** (through `moov`) at construction
/// (see [`Mp4Demux::new`]), and has no sensible zero-arg default while codec-init negotiation
/// and autoplug are still being built. Registration exposes the `mp4demux` descriptor for
/// `--list`/help and descriptor queries until autoplug lands and can supply the head.
///
/// The `&'static ElementDesc` is taken from a throwaway instance (a bare `Mp4Demux` over an
/// empty head) — only `desc()` is called, so the instance is dropped.
pub fn register(registry: &mut Registry) {
    registry.register(Mp4Demux::new(Vec::new()).desc());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `register` exposes the demuxer under its descriptor name; it is not name-constructible
    /// (`make_default: None`).
    #[test]
    fn register_exposes_the_demuxer() {
        let mut reg = Registry::new();
        register(&mut reg);
        assert_eq!(reg.names(), vec!["mp4demux"], "mp4demux descriptor registered");
        assert!(reg.get("mp4demux").unwrap().make_default.is_none(), "mp4demux not name-constructible");
    }
}
