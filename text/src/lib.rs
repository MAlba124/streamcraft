//! streamcraft-text (`sc-text`) — **subtitle support**: cue parsing + a non-blocking overlay
//! compositor (spec: subtitle support; RFC 9559 §12.7 the Matroska S_TEXT/* codec mappings the
//! demuxer announces).
//!
//! # Shape of the problem
//!
//! Subtitles are unlike every other stream in the graph: **sparse** (a cue every few seconds,
//! not 25–60 frames a second), **duration-bearing** (each cue owns an on-screen span
//! `[pts, pts+dur)`, carried by Matroska `BlockDuration`, not inferred from the next frame),
//! and **text**, not samples. That shape drives every design choice here:
//!
//! * **The overlay is a latest-wins fan-in, never a time-aligned one.** A time-aligned `All`
//!   aggregator (the muxer's model) would stall the video path waiting for a subtitle buffer
//!   that only arrives seconds later — the picture would freeze between cues. So
//!   [`SubtitleOverlay`] is `InputPolicy::Any`: the video pad is the metronome, each frame
//!   composites immediately against whatever cues have already landed in a side table, and the
//!   text pad just updates that table. This is the same reasoning as `nvr`'s live mosaic wall,
//!   and it is what makes the overlay **controller-friendly** — nothing blocks or waits, so a
//!   player's seek/pause/rate ripple through untouched.
//!
//! * **The overlay is format-agnostic; `subparse` earns that.** SubRip, WebVTT and ASS have
//!   three different markups and timelines. [`SubParse`] collapses all three to one normalised
//!   family — [`subparse::EVENTS_FAMILY`] (`subtitle/events`): plain UTF-8 cue text, markup
//!   stripped, one cue per buffer, timing on the buffer's PTS + duration (which the demuxer
//!   already framed from the Block). The overlay only ever sees `subtitle/events`, so it never
//!   grows a per-format branch.
//!
//! # The element / pad / format surface (what an autoplugger wires to)
//!
//! ```text
//!   mkvdemux.src_trackN ──(subtitle/srt|ass|vtt)──▶ subparse.sink
//!   subparse.src ──────────(subtitle/events)──────▶ suboverlay.text
//!   <videodec>.src ────────(video/raw i420|nv12)──▶ suboverlay.video
//!   suboverlay.src ────────(video/raw, passthrough)▶ <videosink>.sink
//! ```
//!
//! * `subparse`: sink offers `subtitle/srt` / `subtitle/vtt` / `subtitle/ass` (+ `bytes`); src
//!   emits `subtitle/events` (+ `bytes`). Pads: `sink`, `src`.
//! * `suboverlay`: sink pads `video` (`video/raw`, I420 + NV12) and `text` (`subtitle/events`);
//!   src pad `src` (`video/raw`, format forwarded from the video sink). `InputPolicy::Any`.
//!
//! # Bitmap subtitles — the PGS path
//!
//! BluRay subtitles are **not** text: `hdmv_pgs_subtitle` (HDMV Presentation Graphics Stream)
//! is an *image* format — each caption is an RLE-encoded bitmap with its own YCrCb+alpha
//! palette and on-screen placement. The crate handles them on a parallel path:
//!
//! ```text
//!   mkvdemux.src_trackN ──(subtitle/pgs)──▶ pgsdec.sink
//!   pgsdec.src ──────────(subtitle/bitmap)─▶ suboverlay.image
//!   <videodec>.src ──────(video/raw)───────▶ suboverlay.video
//! ```
//!
//! * [`pgsdec`](pgsdec::PgsDec) parses a PGS **Display Set** (a run of PCS/WDS/PDS/ODS/END
//!   segments — see [`pgs`]) per mkv Block: it decodes the HDMV run-length bitmap, maps it
//!   through the palette (Y′CrCb→RGB, BT.709) to straight-alpha RGBA, and emits one
//!   [`BITMAP_FAMILY`] (`subtitle/bitmap`) buffer carrying the RGBA + geometry + timing. A
//!   Display Set with no composition objects is a **clear** (erase the current caption).
//! * [`SubtitleOverlay`] composites those bitmaps on its third sink pad `image`, alpha-over
//!   onto each video frame at the caption's position (scaled from the PGS reference size to
//!   the decoded frame), alongside — never instead of — the text path.
//!
//! # v1 scope and future work
//!
//! The **text** path is deliberately **bottom-centre**: no positioning / alignment overrides,
//! no karaoke timing, no per-span colour. The **bitmap** (PGS) path handles the mainstream
//! show/clear Display Sets every BluRay rip uses; palette-update-only sets (fades), forced
//! flags, object cropping, and 2nd+ composition objects are not specially handled (a `warn`,
//! then the first object / a clear — see [`pgs`]). DVB / VobSub bitmap subtitles are not yet
//! decoded (they would each add their own `*dec` element onto the same `subtitle/bitmap`
//! family). Whole-*file* parsing (a standalone `.srt`/`.vtt`/`.ass` through `filesrc`) is
//! implemented as library calls in [`parse`] but not yet fronted by a `subfilesrc` element.
//! These are the documented follow-ups.

pub mod font;
pub mod font_data;
pub mod overlay;
pub mod parse;
pub mod pgs;
pub mod pgsdec;
pub mod subparse;

pub use overlay::SubtitleOverlay;
pub use parse::{parse_ass, parse_srt, parse_vtt, Cue};
pub use pgsdec::{PgsDec, PGS_FAMILY};
pub use subparse::{SubParse, EVENTS_FAMILY};

/// The bitmap-subtitle family a `*dec` (PGS today; DVB/VobSub tomorrow) emits and the overlay's
/// `image` pad consumes: straight-alpha RGBA8888 + geometry + timing carried on the buffer (a
/// small header, see [`pgs::encode_bitmap`]). Fieldless caps — geometry is *per-caption* and so
/// rides the payload, exactly as `subtitle/events` carries its text inline.
pub const BITMAP_FAMILY: &str = "subtitle/bitmap";

use streamcraft_core::element::Element;
use streamcraft_core::registry::Registry;

/// Register this crate's elements for name-based construction (spec: Plugins —
/// `parse("... ! subparse ! suboverlay ! ...")`). Typed `use` + constructor stays primary;
/// this powers `scraft-launch` and one-liner tests. Descriptors are `&'static`, taken from
/// throwaway default instances (both elements default-construct with no props — `subparse`
/// learns its dialect from caps, `suboverlay` learns its geometry from the negotiated video
/// format).
pub fn register(registry: &mut Registry) {
    registry.register(SubParse::new().desc());
    registry.register(PgsDec::new().desc());
    registry.register(SubtitleOverlay::new().desc());
}
