//! `pf-play` — turn profluens into a general-purpose media player: hand it any supported
//! file and it probes the container, builds the pipeline, and plays it (spec:
//! profluens.md Milestone applications §5; no-bins §453–475 — autoplug is a **controller
//! library**, not a self-modifying graph element; roadmap item 8 — "autoplug controller").
//!
//! The library is the reusable half of the [`pfplay`](../pfplay/index.html) binary: a
//! [`Player`] that probes → prepares the head → builds the seek index → prerolls → autoplugs
//! → runs. The pipeline stays a flat graph; every dynamic decision (which decoder, which
//! sink, what to drop) is controller logic issuing first-class `link` operations, exactly as
//! `sdl3/examples/play_file.rs` did by hand for MKV — this crate generalizes that across
//! containers (MKV/WebM, MP4, Ogg) and elementary streams (FLAC, MP3, WAV).
//!
//! Layers:
//! - [`probe`] — typefind by magic bytes;
//! - [`head`] — per-container source prep (MKV cluster scan, MP4 box walk);
//! - [`seek`] — time→byte [`SeekIndex`](profluens_core::pipeline::SeekIndex) building;
//! - [`autoplug`] — the controller: negotiation-driven decoder/sink selection;
//! - [`player`] — the thin [`Player`] API the CLI drives.
//!
//! ## The `video` feature (default: **on**)
//!
//! `video` carries the entire video stack: the six software decoders (h264, h265, vp8, vp9,
//! av1, mpeg4p2), the VA-API hardware decoders — and with them the system `libva` — the
//! raw-wire Wayland presenter, the subtitle overlay, and the `video/raw` drop sink. It is on
//! by default, so nothing changes for a consumer that does not ask.
//!
//! Switching it off (`pf-play = { …, default-features = false }`) is for an audio-only
//! consumer — a music player — that should neither link `libva` nor compile six video codecs.
//! The feature is **purely additive**: turning it off removes decoders, it does not change how
//! anything else behaves.
//!
//! - Every demuxer stays unconditional (MKV/WebM, MP4, Ogg, AVI), so the *same* files open.
//!   `pf-avi` in particular is not video-gated: an AVI carries MP3/AC-3 audio beside its
//!   MPEG-4 Part 2 video, and that audio must still play.
//! - A video pad in an opened file has no decoder to link, so it takes the same explicit
//!   `bytes` drop sink an extra audio track or an unknown codec has always taken, and reports
//!   `(dropped: no decoder for h264/annexb)`. Nothing accumulates, and the audio plays.
//! - [`SinkChoice`] keeps all four variants in both builds, so a caller's sink policy needs no
//!   `cfg`. `Device`/`External`/`ExternalZeroCopy` are simply never reached for video, because
//!   no video decoder ever links.
//! - Subtitles ride with video: they are composited *into a decoded frame*, so an audio-only
//!   build reports a subtitle pad as `(dropped: no overlay for subtitle/srt)`.
//! - The two video-typed API items that cannot survive the dep going away are
//!   [`Player::player_control`] (the presenter's window channel), which is `video`-gated, and
//!   [`ZeroCopyChannel`], which becomes an uninhabited stand-in so every signature carrying one
//!   keeps its shape.

pub mod autoplug;
pub mod chain;
pub mod head;
pub mod player;
pub mod probe;
pub mod seek;
pub mod source;
pub mod stereo;

pub use autoplug::{SinkChoice, ZeroCopyChannel};
pub use chain::{ChainHandles, ChainSpec, SinkSpec};
pub use player::{Player, SinkPolicy};
pub use probe::Kind;
pub use source::SourceSpec;
pub use stereo::AudioStereo;
