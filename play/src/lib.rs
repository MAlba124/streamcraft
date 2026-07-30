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

pub mod autoplug;
pub mod head;
pub mod player;
pub mod probe;
pub mod seek;

pub use autoplug::SinkChoice;
pub use player::{Player, SinkPolicy};
pub use probe::Kind;
