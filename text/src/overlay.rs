//! `suboverlay` — a **non-blocking, controller-friendly** subtitle compositor: it burns the
//! active cue's text onto each decoded video frame, bottom-centre, and passes the frame on
//! (spec: subtitle support; the mosaic fan-in template in `nvr/src/mosaic.rs`).
//!
//! # Why a latest-wins fan-in, not a time-aligned one
//!
//! Subtitles are **sparse and duration-bearing**: a cue arrives once, seconds before the video
//! frames it decorates, and stays on screen for its `[pts, pts+dur)` span. A time-aligned
//! `InputPolicy::All` fan-in (the muxer's model) would stall the whole video path waiting for a
//! subtitle buffer that simply is not coming until the next cue — the picture would freeze. So
//! this element is **`InputPolicy::Any`** (latest-wins, like the live wall): every video frame
//! composites immediately against whatever cues have already arrived, and a text buffer just
//! updates the cue set. The video pad is the metronome; the text pad feeds a side table.
//!
//! This is also what makes it controller-friendly: nothing here blocks or waits, so a player's
//! seek/pause/rate changes ripple through untouched — the overlay only ever reacts to the video
//! frame it is handed, at that frame's PTS.
//!
//! # The cue timeline
//!
//! Received cues (from `subparse` on the `text` pad — plain UTF-8, PTS + duration) land in a
//! small time-sorted ring keyed by `[start, end)`. For each video frame we find the cues active
//! at the frame's PTS and stack their text upward from the bottom margin. Cues whose end is well
//! behind the current video PTS are evicted (the ring stays bounded — a movie has thousands of
//! cues over its length but only a couple on screen at once). A cue with no declared duration
//! (the container dropped `BlockDuration`) falls back to [`DEFAULT_CUE_NS`] so it does not stick
//! forever.
//!
//! # The blit (alpha-over / coverage compositing)
//!
//! Text is rasterised to an **A8 coverage** bitmap (`crate::font`) and composited onto the
//! frame with straight **alpha-over** (Porter & Duff, *Compositing Digital Images*, SIGGRAPH
//! 1984 — `out = src·α + dst·(1−α)` for premultiplied-by-coverage source): white text
//! (luma≈235, chroma→neutral 128) over a 1-px dark outline (luma≈16) so the caption reads over
//! any content. Both 4:2:0 layouts are handled — I420's separate Cb/Cr planes and NV12's
//! interleaved chroma (branch on `pixfmt`). When no cue is active the frame passes through
//! untouched (the buffer's `Memory` is forwarded as a refcount — zero copy).
//!
//! # The bitmap path (PGS)
//!
//! Alongside the text pad, the overlay carries a third sink pad **`image`** speaking
//! [`crate::BITMAP_FAMILY`] (`subtitle/bitmap`) — the RGBA captions `pgsdec` decodes from a
//! BluRay PGS track. A bitmap caption is **latest-wins with its own timeline**, not stacked
//! like text: at most one is on screen. Each buffer carries straight-alpha RGBA + placement +
//! the PGS *reference* video size (a small header, [`crate::pgs::decode_bitmap`]); a
//! zero-bitmap buffer is a **clear** (the current caption's time is up). For each video frame
//! the overlay composites the active bitmap (if any at the frame's PTS) alpha-over onto the
//! frame at the caption's `(x, y)`, **scaling** from the PGS reference size to the decoded
//! frame size (a rip's frame may differ from the 1920×1080 authoring reference) and clamping to
//! the frame. RGB→YUV is per-pixel (BT.709 limited range) for the plane write — the inverse of
//! `pgsdec`'s palette conversion, cited at the point of use. Text and bitmap coexist: a graph
//! feeds only one in practice, but nothing here precludes both.
//!
//! v1 text is bottom-centre; the bitmap path honours the PGS placement. No
//! positioning/karaoke/colour overrides on the text path (future work, noted in the crate docs).

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::PadId;
use profluens_core::log;
use profluens_core::log::Level;
use profluens_core::time::Timestamp;
use profluens_video::format::PixelFormat;
use profluens_video::frame::VideoFrameMut;

use crate::font;
use crate::pgs;

/// The video sink pad (the metronome), the text sink pad (the cue feed), and the image sink pad
/// (the bitmap-caption feed); one src pad.
const VIDEO: PadId = PadId(0);
const TEXT: PadId = PadId(1);
const IMAGE: PadId = PadId(2);
const SRC: PadId = PadId(3);

/// Default on-screen span for a cue whose container dropped its duration (no `BlockDuration`):
/// 3 s, the SubRip-ish default. Only a fallback — a real cue carries its own interval.
pub const DEFAULT_CUE_NS: u64 = 3_000_000_000;

/// How far behind the current video PTS a cue's end must fall before it is evicted from the
/// ring (a small slack so a frame arriving slightly before its cue's real end still shows it).
const EVICT_SLACK_NS: u64 = 200_000_000;

/// Target caption text height as a fraction of frame height (the atlas snaps to the nearest
/// baked size). ~1/12 of the frame keeps two lines legible without dominating the picture.
const TEXT_HEIGHT_FRAC: f32 = 1.0 / 12.0;

/// The `video/raw` families the overlay passes through (I420 + NV12 — both 4:2:0 layouts a
/// decoder emits). Broad `any` offers: dimensions/pixfmt ride the runtime announcement, and
/// the overlay forwards the exact negotiated format downstream (see `event`).
static VIDEO_OFFERS: [OfferDesc; 1] = [OfferDesc::any("video/raw")];
/// The text sink speaks the normalised `subtitle/events` (from `subparse`), plus a `bytes`
/// escape so a raw plain-text peer can drive it directly.
static TEXT_OFFERS: [OfferDesc; 2] =
    [OfferDesc::any(crate::subparse::EVENTS_FAMILY), OfferDesc::any("bytes")];
/// The image sink speaks `subtitle/bitmap` (from `pgsdec`), plus the `bytes` escape.
static IMAGE_OFFERS: [OfferDesc; 2] =
    [OfferDesc::any(crate::BITMAP_FAMILY), OfferDesc::any("bytes")];

static PADS: [PadDesc; 4] = [
    PadDesc {
        name: "video",
        direction: Direction::Sink,
        offers: &VIDEO_OFFERS,
        dynamic: true, // upstream decoder announces its dimensions/pixfmt at runtime
        validate: None,
    },
    PadDesc {
        name: "text",
        direction: Direction::Sink,
        offers: &TEXT_OFFERS,
        dynamic: true,
        validate: None,
    },
    PadDesc {
        name: "image",
        direction: Direction::Sink,
        offers: &IMAGE_OFFERS,
        dynamic: true,
        validate: None,
    },
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &VIDEO_OFFERS,
        dynamic: true, // same format as the video sink — announced/forwarded at runtime
        validate: None,
    },
];

static DESC: ElementDesc = ElementDesc {
    name: "suboverlay",
    pads: &PADS,
    props: &[],
    // Active: a per-frame CPU blit is beyond the inline passive budget, and the fan-in wants
    // its own group (the mosaic precedent).
    sched: SchedHint::Active,
    // Any: composite on whatever arrived — a subtitle overlay never waits for a cue that is
    // seconds away (module docs). The video pad is the metronome.
    inputs: InputPolicy::Any,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // COLD: registry make_default — boxes one element instance at plugin-registration time.
    #[allow(clippy::disallowed_methods)]
    make_default: Some(|| Box::new(SubtitleOverlay::new())),
};

/// A cue held in the overlay's timeline: its interval and the plain text to draw.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RingCue {
    start: Timestamp,
    end: Timestamp,
    text: String,
}

/// The current bitmap caption on the `image` path: a decoded [`pgs::DisplaySet`] and its
/// on-screen interval `[start, end)`. Latest-wins (at most one) — a new bitmap or a clear
/// replaces it. `None` when nothing is showing.
#[derive(Clone, Debug)]
struct BitmapCue {
    start: Timestamp,
    end: Timestamp,
    ds: pgs::DisplaySet,
}

/// The rasterised caption currently on screen, kept across frames.
///
/// A cue is on screen for seconds — 50-75 frames at 25 fps — and its pixels do not change in
/// that time, so re-rasterising per frame is pure waste (it also allocated a fresh ~25 KB
/// coverage bitmap every frame). The cache is keyed by exactly what the rasterisation depends
/// on: the concatenated active-cue text and the baked font size. `key_scratch` is the reused
/// buffer the per-frame key is built into, so the compare costs no allocation either.
#[derive(Default)]
struct CaptionCache {
    /// The active cues' text, newline-separated — the key the bitmaps were built from.
    key: String,
    /// Scratch the next frame's key is assembled into before comparing (reused, never freed).
    key_scratch: String,
    /// The baked font size (px) the bitmaps were rasterised at.
    size_px: f32,
    /// Line-box height for that size — the vertical stacking advance.
    line_h: usize,
    /// One A8 coverage bitmap per *physical* line (a multi-line cue contributes several).
    lines: Vec<font::Bitmap>,
}

/// Burns active subtitle cues onto decoded video frames, bottom-centre. See the module docs.
pub struct SubtitleOverlay {
    /// Time-sorted (by `start`) ring of received cues; evicted once well behind the video PTS.
    cues: Vec<RingCue>,
    /// The current bitmap caption (PGS `image` path), latest-wins with its own interval.
    bitmap: Option<BitmapCue>,
    /// The negotiated video format forwarded downstream once (announced on first frame).
    announced: bool,
    /// Whether the "cues arrived but the video geometry is unreadable" warning has been logged
    /// (once per stream — see `process`).
    warned_no_geom: bool,
    /// The rasterised caption, reused while the active cue set does not change.
    caption: CaptionCache,
    /// Frames composited / passed through (health counters, mirroring the mosaic's).
    pub emitted: u64,
    pub with_caption: u64,
    /// Frames onto which a bitmap caption was composited (health counter).
    pub with_bitmap: u64,
}

impl Default for SubtitleOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl SubtitleOverlay {
    // COLD: one-time constructor — the empty cue ring (Vec::new() = zero capacity, no heap yet).
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        Self {
            cues: Vec::new(),
            bitmap: None,
            announced: false,
            warned_no_geom: false,
            caption: CaptionCache::default(),
            emitted: 0,
            with_caption: 0,
            with_bitmap: 0,
        }
    }

    /// Ingest one received cue buffer (plain UTF-8 text, PTS + duration). A missing/zero
    /// duration falls back to [`DEFAULT_CUE_NS`]; a missing PTS drops the cue (untimed).
    /// Insertion keeps the ring sorted by start time.
    fn push_cue(&mut self, text: String, pts: Timestamp, duration: Timestamp) {
        let Some(start_ns) = pts.nanos() else { return };
        // An empty cue (a blank Block, or an all-markup line) clears nothing and shows nothing;
        // skip it so it does not occupy the ring.
        if text.trim().is_empty() {
            return;
        }
        let dur_ns = match duration.nanos() {
            Some(d) if d > 0 => d,
            _ => DEFAULT_CUE_NS, // container dropped BlockDuration — bounded fallback
        };
        let start = Timestamp::from_nanos(start_ns);
        let end = Timestamp::from_nanos(start_ns.saturating_add(dur_ns));
        let cue = RingCue { start, end, text };
        // Insert sorted by start (cues usually arrive in order, so this is O(1) amortised at
        // the tail; a stray out-of-order cue costs one shift).
        let at = self.cues.partition_point(|c| c.start <= cue.start);
        // Drop an exact re-send. Cues with an equal start sort adjacently, so the scan is over
        // that run only. Without this a demuxer that replays a Block — a seek landing before a
        // cue that is still in the ring, a looping source — stacks the *same* line twice on
        // screen (`active_text` returns both) instead of showing it once.
        if self.cues[..at].iter().rev().take_while(|c| c.start == cue.start).any(|c| *c == cue) {
            return;
        }
        self.cues.insert(at, cue);
    }

    /// Drop the whole cue timeline — the flush/seek reset (see `Element::event`).
    fn flush(&mut self) {
        self.cues.clear();
        self.bitmap = None;
        self.caption.key.clear();
        self.caption.lines.clear();
    }

    /// Evict cues whose end is more than [`EVICT_SLACK_NS`] behind `now` — they can never be
    /// active again on a forward stream, so the ring stays bounded.
    fn evict_stale(&mut self, now: Timestamp) {
        let Some(now_ns) = now.nanos() else { return };
        let cutoff = now_ns.saturating_sub(EVICT_SLACK_NS);
        self.cues.retain(|c| c.end.nanos().map(|e| e > cutoff).unwrap_or(true));
    }

    /// The text of every cue active at `pts`, in start order.
    #[cfg(test)]
    fn active_text(&self, pts: Timestamp) -> Vec<&str> {
        self.cues
            .iter()
            .filter(|c| pts >= c.start && pts < c.end)
            .map(|c| c.text.as_str())
            .collect()
    }

    /// Bring [`CaptionCache`] up to date for presentation time `pts` on a `frame_h`-tall frame,
    /// and report whether anything is on screen.
    ///
    /// The active cues' text is assembled into a **reused** buffer and compared against the key
    /// the cached bitmaps were built from; only a genuine change (a cue starting, one expiring,
    /// a resolution change) re-rasterises. On an unchanged cue — the overwhelming majority of
    /// frames, since a cue lives 50-75 frames at 25 fps — this allocates nothing and does no
    /// glyph work at all.
    ///
    /// Each **physical** line becomes its own bitmap. Cue text keeps the newlines the parsers
    /// preserve (`\n` from SubRip/WebVTT, `\N` from ASS), and rasterising a string containing
    /// `\n` as a single line drew the font's replacement box rather than breaking the line — a
    /// two-line caption came out as `line one□line two` on one row.
    fn refresh_caption(&mut self, pts: Timestamp, frame_h: usize) -> bool {
        let px = (frame_h as f32 * TEXT_HEIGHT_FRAC).max(8.0);
        let size = font::size_nearest(px);
        // Build this frame's key into the scratch buffer (moved out so `self.cues` can be read
        // while it is written; moved back below — no allocation in the steady state).
        let mut key = std::mem::take(&mut self.caption.key_scratch);
        key.clear();
        for c in self.cues.iter().filter(|c| pts >= c.start && pts < c.end) {
            if !key.is_empty() {
                key.push('\n');
            }
            key.push_str(&c.text);
        }
        if key != self.caption.key || self.caption.size_px != size.px {
            self.caption.lines.clear();
            for line in key.split('\n') {
                if line.trim().is_empty() {
                    continue;
                }
                self.caption.lines.push(font::rasterize_line(size, line));
            }
            self.caption.key.clear();
            self.caption.key.push_str(&key);
            self.caption.size_px = size.px;
            self.caption.line_h = font::line_height(size);
        }
        self.caption.key_scratch = key;
        !self.caption.lines.is_empty()
    }

    /// Ingest one `subtitle/bitmap` buffer (a decoded PGS Display Set: RGBA + geometry, PTS +
    /// duration). Latest-wins: a decoded bitmap **replaces** the current one for its `[pts,
    /// pts+dur)` span; a **clear** (empty bitmap) simply drops it. A missing PTS is ignored
    /// (untimed). A missing/zero duration falls back to [`DEFAULT_CUE_NS`] so a caption whose
    /// clear was lost does not stick forever.
    fn push_bitmap(&mut self, ds: pgs::DisplaySet, pts: Timestamp, duration: Timestamp) {
        // A clear erases the current caption (regardless of its own timing).
        if ds.is_clear() {
            self.bitmap = None;
            return;
        }
        let Some(start_ns) = pts.nanos() else { return };
        let dur_ns = match duration.nanos() {
            Some(d) if d > 0 => d,
            _ => DEFAULT_CUE_NS,
        };
        let start = Timestamp::from_nanos(start_ns);
        let end = Timestamp::from_nanos(start_ns.saturating_add(dur_ns));
        self.bitmap = Some(BitmapCue { start, end, ds });
    }

    /// The bitmap caption active at `pts`, if any. Also evicts a bitmap whose span is well
    /// behind `pts` (a lost clear), keeping the state bounded to at most one.
    fn active_bitmap(&mut self, pts: Timestamp) -> Option<&pgs::DisplaySet> {
        if let Some(b) = &self.bitmap {
            let past = pts
                .nanos()
                .zip(b.end.nanos())
                .map(|(now, end)| now > end.saturating_add(EVICT_SLACK_NS))
                .unwrap_or(false);
            if past {
                self.bitmap = None;
            }
        }
        self.bitmap
            .as_ref()
            .filter(|b| pts >= b.start && pts < b.end)
            .map(|b| &b.ds)
    }
}

impl Element for SubtitleOverlay {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.cues.clear();
        self.bitmap = None;
        self.announced = false;
        self.warned_no_geom = false;
        self.caption = CaptionCache::default();
        self.emitted = 0;
        self.with_caption = 0;
        self.with_bitmap = 0;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Drain the text pad first: every cue that has arrived updates the side table before we
        // composite this pass's video frames (so a cue landing in the same batch as its first
        // frame is already visible). Per-pad reads via `take_input_on` (the mosaic convention).
        let mut text_batch = ctx.take_input_on(TEXT);
        while let Some(buf) = text_batch.pop_front() {
            let text = String::from_utf8_lossy(buf.memory.data()).into_owned();
            self.push_cue(text, buf.pts, buf.duration);
        }
        ctx.recycle_input(text_batch);

        // Drain the image pad next: each `subtitle/bitmap` buffer updates the current bitmap
        // caption (latest-wins) before this pass's frames composite. A malformed buffer (bad
        // header / RGBA mismatch) is dropped — untrusted peer data never panics (spec: P0).
        let mut image_batch = ctx.take_input_on(IMAGE);
        while let Some(buf) = image_batch.pop_front() {
            if let Some(ds) = pgs::decode_bitmap(buf.memory.data()) {
                self.push_bitmap(ds, buf.pts, buf.duration);
            }
        }
        ctx.recycle_input(image_batch);

        // Now the video pad — the metronome. Each frame composites the active cues at its PTS.
        let mut video_batch = ctx.take_input_on(VIDEO);
        while let Some(mut buf) = video_batch.pop_front() {
            // Forward the exact negotiated video format downstream, once (spec: dynamic caps —
            // the overlay is format-preserving, so it re-announces the sink's own format).
            if !self.announced {
                if let Some(fixed) = ctx.negotiated(VIDEO) {
                    ctx.forward_format(SRC, fixed.clone());
                    self.announced = true;
                }
            }

            self.evict_stale(buf.pts);

            // Resolve the frame geometry once (both paths write onto the same planes).
            let geom = negotiated_video(ctx);
            // Without geometry there is nothing to blit onto, and the element degrades to a
            // silent passthrough — the exact shape of "my subtitles just don't show up". Say so
            // once: it happens whenever the video sink's caps carry no width/height/pixfmt (a
            // pure transport upstream that does not `forward_format`, e.g. `tee`) or the pixel
            // format is not 4:2:0.
            if geom.is_none() && !self.warned_no_geom && (!self.cues.is_empty() || self.bitmap.is_some())
            {
                self.warned_no_geom = true;
                log!(
                    &*ctx,
                    Level::Warn,
                    "no_video_geometry",
                    detail = "video sink caps carry no usable width/height/pixfmt — \
                              subtitles cannot be composited and frames pass through unchanged",
                );
            }

            // --- text path: stack the active cues' lines bottom-centre ---
            // `refresh_caption` re-rasterises only when the on-screen text actually changed, so
            // the steady state (the same cue for the next 50 frames) is a string compare.
            if let Some((w, h, pixfmt)) = geom {
                if self.refresh_caption(buf.pts, h as usize) {
                    let need = profluens_video::geometry::frame_size(pixfmt, w, h);
                    let line_h = self.caption.line_h;
                    let data = buf.memory.as_mut_full();
                    if data.len() >= need {
                        if let Some(mut frame) =
                            VideoFrameMut::new(&mut data[..need], w, h, pixfmt)
                        {
                            draw_caption(&mut frame, &self.caption.lines, line_h);
                            self.with_caption += 1;
                        }
                    }
                }
            }

            // --- bitmap path: alpha-over the active PGS caption at its (scaled) placement ---
            // Borrow the active DisplaySet, blit, then bump the counter (split so the immutable
            // `active_bitmap` borrow releases before the `&mut self` counter write).
            let mut drew_bitmap = false;
            if let Some((w, h, pixfmt)) = geom {
                if let Some(ds) = self.active_bitmap(buf.pts) {
                    let need = profluens_video::geometry::frame_size(pixfmt, w, h);
                    let data = buf.memory.as_mut_full();
                    if data.len() >= need {
                        if let Some(mut frame) =
                            VideoFrameMut::new(&mut data[..need], w, h, pixfmt)
                        {
                            drew_bitmap = draw_bitmap(&mut frame, ds);
                        }
                    }
                }
            }
            if drew_bitmap {
                self.with_bitmap += 1;
            }
            // Passthrough (no copy): the (possibly painted-in-place) frame goes on unchanged in
            // shape — same format, same PTS.
            let out_pts = buf.pts;
            let mut out = buf;
            out.pts = out_pts;
            ctx.out(SRC).push(out);
            self.emitted += 1;
        }
        ctx.recycle_input(video_batch);
        Ok(Flow::Ok)
    }

    /// A flush (the scheduler's out-of-band seek signal, delivered at a batch boundary — spec:
    /// flush/seek) drops the whole cue timeline: after a seek the demuxer replays subtitle
    /// Blocks from the new position, and a ring still holding the pre-seek cues both stacks
    /// duplicates of any cue that is re-sent and keeps showing cues from the old playhead. Every
    /// other stateful element in the tree resets here (`h264dec`, `aacdec`, `avidemux`, …); the
    /// overlay was the exception.
    fn event(&mut self, _ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::FlushStart) {
            self.flush();
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// The negotiated `(width, height, pixfmt)` on the video sink, or `None` if not yet fixed /
/// an unsupported pixfmt. Read by name off the negotiated caps (the audioconvert pattern).
fn negotiated_video(ctx: &Ctx) -> Option<(u32, u32, PixelFormat)> {
    use profluens_core::format::Value;
    let fixed = ctx.negotiated(VIDEO)?;
    let int = |name: &str| -> Option<i64> {
        ctx.field_id(name).and_then(|id| fixed.get(id)).and_then(|v| match v {
            Value::Int(n) => Some(n),
            _ => None,
        })
    };
    let (w, h) = (int("width")?, int("height")?);
    if w <= 0 || h <= 0 {
        return None;
    }
    let pixfmt = ctx
        .field_id("pixfmt")
        .and_then(|id| fixed.get(id))
        .and_then(|v| match v {
            Value::Id(vid) => ctx.value_name(vid),
            _ => None,
        })
        .and_then(PixelFormat::from_caps_name)?;
    // Only the 4:2:0 layouts are supported (the module docs); a packed format is refused so we
    // never mis-address chroma.
    if !pixfmt.is_subsampled_420() {
        return None;
    }
    Some((w as u32, h as u32, pixfmt))
}

// ============================ the caption blit ============================

/// Straight-alpha luma target for the caption text — near-white in BT.601/709 limited range
/// (Y' 235 is nominal white). The outline is near-black (Y' 16, nominal black).
const TEXT_LUMA: u8 = 235;
const OUTLINE_LUMA: u8 = 16;
/// Neutral chroma (128 = zero Cb/Cr → achromatic), so text reads as white/grey regardless of
/// the underlying picture's colour.
const NEUTRAL_CHROMA: u8 = 128;

/// Draw pre-rasterised `lines` (A8 coverage, one per physical line, top line first) bottom-centre
/// onto `frame`, stacking upward from a bottom margin with alpha-over white-on-dark-outline luma
/// + neutral chroma. `line_h` is the stacking advance for the size they were rasterised at.
///
/// Rasterisation is the caller's ([`SubtitleOverlay::refresh_caption`]) so an unchanged caption
/// is not re-rendered on every frame.
fn draw_caption(frame: &mut VideoFrameMut<'_>, lines: &[font::Bitmap], line_h: usize) {
    let (fw, fh) = (frame.width() as usize, frame.height() as usize);
    if fw == 0 || fh == 0 || line_h == 0 {
        return;
    }
    // Bottom margin ~ half a line; lines stack upward from there.
    let margin = line_h / 2 + 2;
    let n = lines.len();
    for (i, bm) in lines.iter().enumerate() {
        if bm.w == 0 || bm.h == 0 {
            continue;
        }
        // Horizontal centre; clamp so an over-wide line starts at x=0.
        let x0 = ((fw as i64 - bm.w as i64) / 2).max(0) as usize;
        // The i-th line from the top of the block. The block's bottom sits `margin` px above
        // the frame bottom; line `i` (0 = topmost) is `(n-1-i)` lines up from the last line.
        let lines_from_bottom = n - 1 - i;
        let y_bottom = fh.saturating_sub(margin + lines_from_bottom * line_h);
        let y0 = y_bottom.saturating_sub(bm.h);
        blit_coverage(frame, bm, x0, y0);
    }
}

/// Alpha-over one A8 coverage bitmap onto the frame at `(x0, y0)`: for each covered pixel,
/// paint the luma toward white with a 1-px dark outline underneath, and pull chroma toward
/// neutral so the text reads achromatic. Coverage is the source alpha (Porter–Duff over).
fn blit_coverage(frame: &mut VideoFrameMut<'_>, bm: &font::Bitmap, x0: usize, y0: usize) {
    let (fw, fh) = (frame.width() as usize, frame.height() as usize);
    let pixfmt = frame.pixfmt();

    // --- luma plane: outline pass, then the glyph pass ---------------------------------
    // The outline is a dilation of the coverage by 1 px, drawn dark first; the glyph is drawn
    // white on top. Two passes keep it simple and correct at edges.
    if let Some((y_plane, y_stride)) = frame.plane_mut(0) {
        // Outline (dilate by 1): any pixel adjacent to ink gets darkened by the neighbour's
        // coverage. Draw before the glyph so the white ink sits on top.
        for dy in 0..bm.h {
            let py = y0 + dy;
            if py >= fh {
                break;
            }
            for dx in 0..bm.w {
                let px = x0 + dx;
                if px >= fw {
                    break;
                }
                // Max coverage over the 3×3 neighbourhood = the dilated outline mask.
                let mut dil = 0u8;
                for oy in -1i32..=1 {
                    for ox in -1i32..=1 {
                        let sx = dx as i32 + ox;
                        let sy = dy as i32 + oy;
                        if sx >= 0 && sy >= 0 {
                            dil = dil.max(bm.at(sx as usize, sy as usize));
                        }
                    }
                }
                if dil == 0 {
                    continue;
                }
                let idx = py * y_stride + px;
                y_plane[idx] = over(y_plane[idx], OUTLINE_LUMA, dil);
            }
        }
        // Glyph (white) on top.
        for dy in 0..bm.h {
            let py = y0 + dy;
            if py >= fh {
                break;
            }
            for dx in 0..bm.w {
                let px = x0 + dx;
                if px >= fw {
                    break;
                }
                let a = bm.at(dx, dy);
                if a == 0 {
                    continue;
                }
                let idx = py * y_stride + px;
                y_plane[idx] = over(y_plane[idx], TEXT_LUMA, a);
            }
        }
    }

    // --- chroma: pull toward neutral under the (dilated) text so it reads white -----------
    // Chroma is half-resolution (4:2:0): a luma pixel (x, y) maps to chroma (x/2, y/2). Only
    // the chroma cells the caption's (dilated) box can reach are visited — scanning the whole
    // plane made a one-line caption cost O(frame area) instead of O(caption area), which is
    // ~97% of this function's time at 1080p. The per-cell guards below are unchanged, so the
    // painted result is byte-identical; only the iteration bounds shrink.
    let (cx0, cx1, cy0, cy1) = chroma_box(x0, y0, bm.w, bm.h, fw, fh);
    match pixfmt {
        PixelFormat::I420 => {
            for plane in 1..=2 {
                if let Some((c_plane, c_stride)) = frame.plane_mut(plane) {
                    paint_chroma_i420(c_plane, c_stride, bm, x0, y0, (cx0, cx1, cy0, cy1));
                }
            }
        }
        PixelFormat::Nv12 => {
            if let Some((c_plane, c_stride)) = frame.plane_mut(1) {
                paint_chroma_nv12(c_plane, c_stride, bm, x0, y0, (cx0, cx1, cy0, cy1));
            }
        }
        // Non-4:2:0 formats are refused upstream (negotiated_video); unreachable here.
        _ => {}
    }
}

/// The half-open chroma-cell range `(cx0, cx1, cy0, cy1)` a coverage bitmap of `bw × bh` placed
/// at luma `(x0, y0)` can possibly touch, clamped to the frame's chroma plane (`fw × fh` luma).
///
/// A chroma cell `(cx, cy)` reads luma `2cx..2cx+2` and each of those is further **dilated by
/// one** pixel by [`chroma_coverage`], so the reachable luma span is `[x0-2, x0+bw+2)`; halving
/// and rounding outward gives the cell span. Deliberately one cell generous on each side — the
/// per-cell coverage test still decides what is painted, so a slightly wide box costs a handful
/// of zero-coverage probes and can never clip the caption.
fn chroma_box(
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    fw: usize,
    fh: usize,
) -> (usize, usize, usize, usize) {
    let (cw, ch) = (fw.div_ceil(2), fh.div_ceil(2));
    let cx0 = x0.saturating_sub(2) / 2;
    let cy0 = y0.saturating_sub(2) / 2;
    let cx1 = (x0 + bw + 2).div_ceil(2).saturating_add(1).min(cw);
    let cy1 = (y0 + bh + 2).div_ceil(2).saturating_add(1).min(ch);
    (cx0.min(cx1), cx1, cy0.min(cy1), cy1)
}

/// Chroma neutralisation for an I420 plane (Cb or Cr): a covered chroma cell is pulled toward
/// 128 by the max coverage of its four covering luma pixels. `bounds` is the caption's chroma
/// cell box from [`chroma_box`] — outside it the coverage is zero by construction.
fn paint_chroma_i420(
    plane: &mut [u8],
    stride: usize,
    bm: &font::Bitmap,
    x0: usize,
    y0: usize,
    bounds: (usize, usize, usize, usize),
) {
    let (cx0, cx1, cy0, cy1) = bounds;
    for cy in cy0..cy1 {
        for cx in cx0..cx1 {
            let a = chroma_coverage(bm, x0, y0, cx, cy);
            if a == 0 {
                continue;
            }
            let idx = cy * stride + cx;
            if idx < plane.len() {
                plane[idx] = over(plane[idx], NEUTRAL_CHROMA, a);
            }
        }
    }
}

/// Chroma neutralisation for an NV12 interleaved plane (Cb, Cr pairs per chroma column).
/// `bounds` is the caption's chroma cell box from [`chroma_box`].
fn paint_chroma_nv12(
    plane: &mut [u8],
    stride: usize,
    bm: &font::Bitmap,
    x0: usize,
    y0: usize,
    bounds: (usize, usize, usize, usize),
) {
    let (cx0, cx1, cy0, cy1) = bounds;
    for cy in cy0..cy1 {
        for cx in cx0..cx1 {
            let a = chroma_coverage(bm, x0, y0, cx, cy);
            if a == 0 {
                continue;
            }
            let base = cy * stride + cx * 2; // Cb then Cr
            if base + 1 < plane.len() {
                plane[base] = over(plane[base], NEUTRAL_CHROMA, a);
                plane[base + 1] = over(plane[base + 1], NEUTRAL_CHROMA, a);
            }
        }
    }
}

/// The coverage a chroma cell `(cx, cy)` should feel: the max dilated coverage over its four
/// covering luma pixels (`2cx..2cx+2 × 2cy..2cy+2`), offset into the bitmap by `(x0, y0)`.
/// Dilated (3×3 max) so the neutralised region matches the luma outline, not just the glyph.
fn chroma_coverage(bm: &font::Bitmap, x0: usize, y0: usize, cx: usize, cy: usize) -> u8 {
    let mut a = 0u8;
    for ly in (2 * cy)..(2 * cy + 2) {
        for lx in (2 * cx)..(2 * cx + 2) {
            // Map the luma pixel back into bitmap space.
            let bx = lx as i64 - x0 as i64;
            let by = ly as i64 - y0 as i64;
            if bx < 0 || by < 0 {
                continue;
            }
            // 3×3 dilation to match the outline extent.
            for oy in -1i32..=1 {
                for ox in -1i32..=1 {
                    let sx = bx + ox as i64;
                    let sy = by + oy as i64;
                    if sx >= 0 && sy >= 0 {
                        a = a.max(bm.at(sx as usize, sy as usize));
                    }
                }
            }
        }
    }
    a
}

/// Straight alpha-over of an 8-bit `src` sample onto `dst` with 8-bit coverage `a`
/// (Porter–Duff *over*, `out = src·a + dst·(255−a)`, rounded). `a == 255` replaces, `a == 0`
/// leaves `dst`.
#[inline]
fn over(dst: u8, src: u8, a: u8) -> u8 {
    let a = a as u32;
    let out = (src as u32 * a + dst as u32 * (255 - a) + 127) / 255;
    out as u8
}

// ============================ the bitmap (PGS) blit ============================

/// Alpha-over a decoded PGS caption (`ds`, straight-alpha RGBA at a reference size/position)
/// onto `frame`, scaling from the PGS reference video size to the decoded frame's actual size
/// and clamping to the frame. Each source pixel's RGB is converted to Y′CbCr (BT.709 limited
/// range — the inverse of `pgsdec`'s palette conversion) and composited with the pixel's alpha
/// as coverage (Porter & Duff *over*, *Compositing Digital Images*, SIGGRAPH 1984). Returns
/// `true` if any pixel was drawn (a fully-transparent or off-frame caption draws nothing).
///
/// Scaling is **nearest-neighbour**: a subtitle is a hard-edged mask over a solid outline, and
/// the common case (a 1080p rip of a 1080p PGS reference) is 1:1 with no resampling at all;
/// nearest keeps it cheap and avoids softening the outline. Chroma is written at the frame's
/// 4:2:0 resolution — a chroma cell takes the alpha/colour of its covering luma pixels.
fn draw_bitmap(frame: &mut VideoFrameMut<'_>, ds: &pgs::DisplaySet) -> bool {
    let (fw, fh) = (frame.width() as usize, frame.height() as usize);
    if fw == 0 || fh == 0 || ds.width == 0 || ds.height == 0 || ds.rgba.is_empty() {
        return false;
    }
    // The PGS reference frame the (x, y) placement is expressed in. If the composition never
    // declared one (0), assume the caption is already in frame-space (ref == frame).
    let ref_w = if ds.video_width == 0 { fw } else { ds.video_width as usize };
    let ref_h = if ds.video_height == 0 { fh } else { ds.video_height as usize };
    if ref_w == 0 || ref_h == 0 {
        return false;
    }

    // The caption's on-screen rectangle in frame pixels (scale placement + size from the PGS
    // reference to the actual frame). Nearest sampling maps each destination pixel back to a
    // source pixel.
    let dst_x0 = ds.x as usize * fw / ref_w;
    let dst_y0 = ds.y as usize * fh / ref_h;
    let dst_w = (ds.width as usize * fw).div_ceil(ref_w);
    let dst_h = (ds.height as usize * fh).div_ceil(ref_h);
    if dst_w == 0 || dst_h == 0 {
        return false;
    }

    // Precompute a per-destination-row/col source index (nearest), and a coverage/colour lookup
    // so the chroma pass can re-sample the same source. We paint luma first, then chroma.
    let sample = |dx: usize, dy: usize| -> Option<[u8; 4]> {
        // Destination pixel (dst_x0+dx, dst_y0+dy) ← source (sx, sy) nearest.
        let sx = dx * ds.width as usize / dst_w;
        let sy = dy * ds.height as usize / dst_h;
        let si = (sy * ds.width as usize + sx) * 4;
        ds.rgba.get(si..si + 4).map(|p| [p[0], p[1], p[2], p[3]])
    };

    let mut drew = false;
    let pixfmt = frame.pixfmt();

    // --- luma plane ---
    if let Some((y_plane, y_stride)) = frame.plane_mut(0) {
        for dy in 0..dst_h {
            let py = dst_y0 + dy;
            if py >= fh {
                break;
            }
            for dx in 0..dst_w {
                let px = dst_x0 + dx;
                if px >= fw {
                    break;
                }
                let Some([r, g, b, a]) = sample(dx, dy) else { continue };
                if a == 0 {
                    continue;
                }
                let (yv, _, _) = rgb_to_ycbcr_bt709(r, g, b);
                let idx = py * y_stride + px;
                if idx < y_plane.len() {
                    y_plane[idx] = over(y_plane[idx], yv, a);
                    drew = true;
                }
            }
        }
    }

    // --- chroma planes (4:2:0): a chroma cell samples its top-left covering luma pixel ---
    match pixfmt {
        PixelFormat::I420 => {
            // Two separate planes: Cb (1) and Cr (2).
            blit_bitmap_chroma_i420(frame, dst_x0, dst_y0, dst_w, dst_h, &sample);
        }
        PixelFormat::Nv12 => {
            blit_bitmap_chroma_nv12(frame, dst_x0, dst_y0, dst_w, dst_h, &sample);
        }
        _ => {}
    }
    drew
}

/// The half-open chroma-cell range a bitmap caption occupying luma `[dst_x0, dst_x0+dst_w) ×
/// [dst_y0, dst_y0+dst_h)` can touch, clamped to the frame's chroma plane. A chroma cell
/// `(cx, cy)` samples luma `(2cx, 2cy)` only, so the span is a plain halving rounded outward.
fn bitmap_chroma_box(
    dst_x0: usize,
    dst_y0: usize,
    dst_w: usize,
    dst_h: usize,
    fw: usize,
    fh: usize,
) -> (usize, usize, usize, usize) {
    let (cw, ch) = (fw.div_ceil(2), fh.div_ceil(2));
    let cx0 = (dst_x0 / 2).min(cw);
    let cy0 = (dst_y0 / 2).min(ch);
    let cx1 = (dst_x0 + dst_w).div_ceil(2).min(cw);
    let cy1 = (dst_y0 + dst_h).div_ceil(2).min(ch);
    (cx0.min(cx1), cx1, cy0.min(cy1), cy1)
}

/// Composite the caption's chroma onto an I420 frame (separate Cb/Cr planes). A chroma cell at
/// `(cx, cy)` covers luma `(2cx, 2cy)`; we sample that destination pixel's source colour/alpha.
#[allow(clippy::too_many_arguments)]
fn blit_bitmap_chroma_i420(
    frame: &mut VideoFrameMut<'_>,
    dst_x0: usize,
    dst_y0: usize,
    dst_w: usize,
    dst_h: usize,
    sample: &impl Fn(usize, usize) -> Option<[u8; 4]>,
) {
    let (fw, fh) = (frame.width() as usize, frame.height() as usize);
    // Only the chroma cells the caption rectangle covers (see `chroma_box`) — the per-cell
    // guards below are unchanged, so this is a pure iteration-bound shrink.
    let (cx0, cx1, cy0, cy1) = bitmap_chroma_box(dst_x0, dst_y0, dst_w, dst_h, fw, fh);
    for plane in 1..=2 {
        let is_cr = plane == 2;
        if let Some((c_plane, c_stride)) = frame.plane_mut(plane) {
            for cy in cy0..cy1 {
                for cx in cx0..cx1 {
                    // The luma pixel this chroma cell covers (top-left of the 2×2).
                    let (lx, ly) = (2 * cx, 2 * cy);
                    if lx < dst_x0 || ly < dst_y0 {
                        continue;
                    }
                    let (dx, dy) = (lx - dst_x0, ly - dst_y0);
                    if dx >= dst_w || dy >= dst_h {
                        continue;
                    }
                    let Some([r, g, b, a]) = sample(dx, dy) else { continue };
                    if a == 0 {
                        continue;
                    }
                    let (_, cb, cr) = rgb_to_ycbcr_bt709(r, g, b);
                    let c = if is_cr { cr } else { cb };
                    let idx = cy * c_stride + cx;
                    if idx < c_plane.len() {
                        c_plane[idx] = over(c_plane[idx], c, a);
                    }
                }
            }
        }
    }
}

/// Composite the caption's chroma onto an NV12 frame (interleaved Cb/Cr in one plane).
#[allow(clippy::too_many_arguments)]
fn blit_bitmap_chroma_nv12(
    frame: &mut VideoFrameMut<'_>,
    dst_x0: usize,
    dst_y0: usize,
    dst_w: usize,
    dst_h: usize,
    sample: &impl Fn(usize, usize) -> Option<[u8; 4]>,
) {
    let (fw, fh) = (frame.width() as usize, frame.height() as usize);
    let (cx0, cx1, cy0, cy1) = bitmap_chroma_box(dst_x0, dst_y0, dst_w, dst_h, fw, fh);
    if let Some((c_plane, c_stride)) = frame.plane_mut(1) {
        for cy in cy0..cy1 {
            for cx in cx0..cx1 {
                let (lx, ly) = (2 * cx, 2 * cy);
                if lx < dst_x0 || ly < dst_y0 {
                    continue;
                }
                let (dx, dy) = (lx - dst_x0, ly - dst_y0);
                if dx >= dst_w || dy >= dst_h {
                    continue;
                }
                let Some([r, g, b, a]) = sample(dx, dy) else { continue };
                if a == 0 {
                    continue;
                }
                let (_, cb, cr) = rgb_to_ycbcr_bt709(r, g, b);
                let base = cy * c_stride + cx * 2; // Cb then Cr
                if base + 1 < c_plane.len() {
                    c_plane[base] = over(c_plane[base], cb, a);
                    c_plane[base + 1] = over(c_plane[base + 1], cr, a);
                }
            }
        }
    }
}

/// Convert full-range 8-bit RGB to limited-range BT.709 Y′CbCr (ITU-R BT.709 §3; the forward
/// of `pgsdec`'s palette `ycrcb_to_rgb_bt709`). Kr=0.2126, Kb=0.0722; luma to [16,235], chroma
/// to [16,240] centred at 128:
///
/// ```text
///   Y = 16  + 0.18259·R + 0.61423·G + 0.06201·B
///   Cb = 128 − 0.10064·R − 0.33857·G + 0.43922·B
///   Cr = 128 + 0.43922·R − 0.39894·G − 0.04027·B
/// ```
#[inline]
fn rgb_to_ycbcr_bt709(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let (rf, gf, bf) = (r as f32, g as f32, b as f32);
    let y = 16.0 + 0.182_585_9 * rf + 0.614_230_6 * gf + 0.062_007_1 * bf;
    let cb = 128.0 - 0.100_643_7 * rf - 0.338_572 * gf + 0.439_215_7 * bf;
    let cr = 128.0 + 0.439_215_7 * rf - 0.398_942_2 * gf - 0.040_273_5 * bf;
    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}

/// Round-and-clamp a float sample to `u8`.
#[inline]
fn clamp_u8(v: f32) -> u8 {
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        (v + 0.5) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use profluens_video::geometry::frame_size;

    /// Rasterise `lines` the way `refresh_caption` does, for the blit tests.
    fn raster(frame_h: u32, lines: &[&str]) -> (Vec<font::Bitmap>, usize) {
        let size = font::size_nearest((frame_h as f32 * TEXT_HEIGHT_FRAC).max(8.0));
        (
            lines.iter().map(|l| font::rasterize_line(size, l)).collect(),
            font::line_height(size),
        )
    }

    fn overlay_with_cue(text: &str, start_ns: u64, dur_ns: u64) -> SubtitleOverlay {
        let mut o = SubtitleOverlay::new();
        o.push_cue(text.into(), Timestamp::from_nanos(start_ns), Timestamp::from_nanos(dur_ns));
        o
    }

    #[test]
    fn alpha_over_endpoints() {
        assert_eq!(over(100, 235, 0), 100, "zero coverage leaves dst");
        assert_eq!(over(100, 235, 255), 235, "full coverage replaces with src");
        let mid = over(0, 200, 128);
        assert!((99..=101).contains(&(mid as i32)), "half coverage ~= halfway: {mid}");
    }

    #[test]
    fn cue_activity_window_and_default_duration() {
        let o = overlay_with_cue("hi", 1_000_000_000, 2_000_000_000);
        assert!(o.active_text(Timestamp::from_nanos(1_500_000_000)) == vec!["hi"]);
        assert!(o.active_text(Timestamp::from_nanos(999_000_000)).is_empty(), "before start");
        assert!(o.active_text(Timestamp::from_nanos(3_000_000_000)).is_empty(), "at end (half-open)");

        // A zero/absent duration falls back to DEFAULT_CUE_NS.
        let mut o2 = SubtitleOverlay::new();
        o2.push_cue("x".into(), Timestamp::from_nanos(0), Timestamp::NONE);
        assert!(o2.active_text(Timestamp::from_nanos(DEFAULT_CUE_NS - 1)) == vec!["x"]);
        assert!(o2.active_text(Timestamp::from_nanos(DEFAULT_CUE_NS)).is_empty());
    }

    #[test]
    fn eviction_bounds_the_ring() {
        let mut o = SubtitleOverlay::new();
        for i in 0..100u64 {
            o.push_cue(format!("cue {i}"), Timestamp::from_nanos(i * 1_000_000_000), Timestamp::from_nanos(500_000_000));
        }
        assert_eq!(o.cues.len(), 100);
        // Advance the clock past all of them — every cue evicts.
        o.evict_stale(Timestamp::from_nanos(200 * 1_000_000_000));
        assert!(o.cues.is_empty(), "stale cues evicted, ring bounded");
    }

    #[test]
    fn empty_and_untimed_cues_are_dropped() {
        let mut o = SubtitleOverlay::new();
        o.push_cue("   ".into(), Timestamp::from_nanos(0), Timestamp::from_nanos(1_000_000_000));
        o.push_cue("real".into(), Timestamp::NONE, Timestamp::from_nanos(1_000_000_000));
        assert!(o.cues.is_empty(), "blank text and untimed (no PTS) cues are not held");
    }

    /// The load-bearing behaviour: the caption region's luma changes when a cue is drawn and is
    /// untouched otherwise. Composite directly on a flat mid-grey I420 frame.
    #[test]
    fn caption_changes_bottom_region_luma_i420() {
        let (w, h) = (320u32, 180u32);
        let mut buf = vec![128u8; frame_size(PixelFormat::I420, w, h)];
        // A copy with no caption drawn = the baseline.
        let baseline = buf.clone();
        {
            let mut frame = VideoFrameMut::new(&mut buf, w, h, PixelFormat::I420).unwrap();
            let (bms, lh) = raster(h, &["Hello, subtitles!"]);
            draw_caption(&mut frame, &bms, lh);
        }
        assert_ne!(buf, baseline, "drawing a caption changes the frame");

        // The change is confined to the bottom third (bottom-centre placement), not the top.
        let y_stride = w as usize;
        let top_rows = &buf[..(h as usize / 3) * y_stride];
        let base_top = &baseline[..(h as usize / 3) * y_stride];
        assert_eq!(top_rows, base_top, "the top of the frame is untouched");

        let bottom_start = (2 * h as usize / 3) * y_stride;
        let bottom = &buf[bottom_start..(h as usize) * y_stride];
        let base_bottom = &baseline[bottom_start..(h as usize) * y_stride];
        assert_ne!(bottom, base_bottom, "the bottom (caption) region's luma changed");
        // And the ink went brighter than mid-grey (white text): some pixel exceeds 128.
        assert!(bottom.iter().any(|&p| p > 200), "white glyph ink present");
    }

    #[test]
    fn caption_handles_nv12_layout() {
        let (w, h) = (320u32, 180u32);
        // Colourful (non-neutral) chroma so the pull-toward-128 is observable; grey luma.
        let y_len = (w * h) as usize;
        let mut buf = vec![128u8; frame_size(PixelFormat::Nv12, w, h)];
        buf[y_len..].fill(200); // interleaved Cb/Cr away from neutral
        let baseline = buf.clone();
        {
            let mut frame = VideoFrameMut::new(&mut buf, w, h, PixelFormat::Nv12).unwrap();
            let (bms, lh) = raster(h, &["NV12 caption"]);
            draw_caption(&mut frame, &bms, lh);
        }
        assert_ne!(buf, baseline, "NV12 caption is drawn (luma + interleaved chroma)");
        // Luma changed (white glyph + dark outline).
        assert_ne!(buf[..y_len], baseline[..y_len], "NV12 luma painted");
        // The interleaved chroma plane was pulled toward neutral (128) under the text, so it
        // moved *down* from 200 somewhere.
        let chroma_neutralised = buf[y_len..].iter().any(|&c| c < 200);
        assert!(chroma_neutralised, "NV12 interleaved chroma pulled toward neutral under the text");
    }

    // ---- bitmap (PGS `image`) path ----

    /// A synthetic opaque-white bitmap caption, 1:1 reference→frame, composites onto the frame
    /// **only** in its `(x, y, w, h)` rectangle — the luma there brightens toward white, the
    /// rest of the frame is byte-for-byte unchanged.
    #[test]
    fn bitmap_paints_only_its_rectangle_i420() {
        let (w, h) = (64u32, 48u32);
        let mut buf = vec![128u8; frame_size(PixelFormat::I420, w, h)];
        let baseline = buf.clone();
        // A 10×6 opaque-white caption at (20, 30), reference size == frame size (1:1).
        let ds = pgs::DisplaySet {
            rgba: vec![255u8; 10 * 6 * 4], // solid white, fully opaque
            width: 10,
            height: 6,
            x: 20,
            y: 30,
            video_width: w,
            video_height: h,
        };
        let drew = {
            let mut frame = VideoFrameMut::new(&mut buf, w, h, PixelFormat::I420).unwrap();
            draw_bitmap(&mut frame, &ds)
        };
        assert!(drew, "an opaque bitmap draws");

        // Inside the caption rectangle the luma went bright (white ≈ 235); outside is untouched.
        let ys = w as usize;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let inside = (20..30).contains(&x) && (30..36).contains(&y);
                let idx = y * ys + x;
                if inside {
                    assert!(buf[idx] > 200, "caption pixel ({x},{y}) brightened: {}", buf[idx]);
                } else {
                    assert_eq!(buf[idx], baseline[idx], "outside-caption luma ({x},{y}) untouched");
                }
            }
        }
    }

    /// A fully-transparent bitmap draws nothing (alpha 0 everywhere) — the frame is unchanged.
    #[test]
    fn transparent_bitmap_draws_nothing() {
        let (w, h) = (32u32, 32u32);
        let mut buf = vec![100u8; frame_size(PixelFormat::I420, w, h)];
        let baseline = buf.clone();
        let ds = pgs::DisplaySet {
            rgba: vec![0u8; 8 * 8 * 4], // RGBA all zero → alpha 0
            width: 8,
            height: 8,
            x: 4,
            y: 4,
            video_width: w,
            video_height: h,
        };
        let drew = {
            let mut frame = VideoFrameMut::new(&mut buf, w, h, PixelFormat::I420).unwrap();
            draw_bitmap(&mut frame, &ds)
        };
        assert!(!drew, "a fully-transparent bitmap draws nothing");
        assert_eq!(buf, baseline, "transparent caption leaves the frame untouched");
    }

    /// A caption authored against a larger reference frame is **scaled** into a smaller frame:
    /// a 1920-wide reference placement maps into a 96-wide frame at the proportional x.
    #[test]
    fn bitmap_scales_from_reference_size() {
        let (w, h) = (96u32, 54u32);
        let mut buf = vec![64u8; frame_size(PixelFormat::I420, w, h)];
        // Caption at x=960 (mid-screen) in a 1920×1080 reference → x≈48 in the 96-wide frame.
        let ds = pgs::DisplaySet {
            rgba: vec![255u8; 20 * 4 * 4],
            width: 20,
            height: 4,
            x: 960,
            y: 540,
            video_width: 1920,
            video_height: 1080,
        };
        {
            let mut frame = VideoFrameMut::new(&mut buf, w, h, PixelFormat::I420).unwrap();
            assert!(draw_bitmap(&mut frame, &ds), "scaled bitmap draws");
        }
        // The brightened region begins near the horizontal centre (x≈48), not at x=0.
        let ys = w as usize;
        let mid_row = 27usize; // y=540/1080*54 = 27
        let first_bright = (0..w as usize).find(|&x| buf[mid_row * ys + x] > 200);
        assert!(
            matches!(first_bright, Some(x) if (40..56).contains(&x)),
            "caption scaled to mid-frame, first bright x = {first_bright:?}"
        );
    }

    /// The `image` timeline: `push_bitmap` shows a caption for its span; a clear erases it.
    #[test]
    fn bitmap_cue_timeline_and_clear() {
        let mut o = SubtitleOverlay::new();
        let ds = pgs::DisplaySet {
            rgba: vec![255u8; 4],
            width: 1,
            height: 1,
            x: 0,
            y: 0,
            video_width: 100,
            video_height: 100,
        };
        o.push_bitmap(ds, Timestamp::from_nanos(1_000_000_000), Timestamp::from_nanos(2_000_000_000));
        assert!(o.active_bitmap(Timestamp::from_nanos(1_500_000_000)).is_some(), "shown mid-span");
        assert!(o.active_bitmap(Timestamp::from_nanos(999_000_000)).is_none(), "before start");
        // A clear (empty bitmap) erases it immediately.
        o.push_bitmap(pgs::DisplaySet::clear(100, 100), Timestamp::from_nanos(1_600_000_000), Timestamp::NONE);
        assert!(o.active_bitmap(Timestamp::from_nanos(1_500_000_000)).is_none(), "cleared");
    }

    // ---- the chroma bounding box (a pure iteration-bound shrink) ----

    /// The bounded chroma pass must paint **byte-for-byte** what the old whole-plane scan
    /// painted, for **both** 4:2:0 layouts. Reference implementations: visit every chroma cell
    /// in the frame, exactly as `paint_chroma_*` used to.
    ///
    /// The placement sweep is exhaustive over small frames of every width/height parity, with
    /// the caption walked past all four edges and entirely off-frame — that is where a bounds
    /// error would hide. `chroma_box` is deliberately a cell or two generous, so a shrunken
    /// box (the failure that would clip a caption) is what this is guarding.
    #[test]
    fn bounded_chroma_matches_a_whole_plane_scan() {
        fn reference_i420(plane: &mut [u8], stride: usize, bm: &font::Bitmap, x0: usize, y0: usize, fw: usize, fh: usize) {
            for cy in 0..fh.div_ceil(2) {
                for cx in 0..fw.div_ceil(2) {
                    let a = chroma_coverage(bm, x0, y0, cx, cy);
                    if a == 0 {
                        continue;
                    }
                    let idx = cy * stride + cx;
                    if idx < plane.len() {
                        plane[idx] = over(plane[idx], NEUTRAL_CHROMA, a);
                    }
                }
            }
        }
        fn reference_nv12(plane: &mut [u8], stride: usize, bm: &font::Bitmap, x0: usize, y0: usize, fw: usize, fh: usize) {
            for cy in 0..fh.div_ceil(2) {
                for cx in 0..fw.div_ceil(2) {
                    let a = chroma_coverage(bm, x0, y0, cx, cy);
                    if a == 0 {
                        continue;
                    }
                    let idx = cy * stride + cx * 2;
                    if idx + 1 < plane.len() {
                        plane[idx] = over(plane[idx], NEUTRAL_CHROMA, a);
                        plane[idx + 1] = over(plane[idx + 1], NEUTRAL_CHROMA, a);
                    }
                }
            }
        }

        let s = font::size_nearest(14.0);
        let mut checked = 0usize;
        for text in ["Wg|_", "i", "The quick brown fox"] {
            let bm = font::rasterize_line(s, text);
            // Every width/height parity combination, swept exhaustively past all four edges.
            for &(fw, fh) in &[(24usize, 20usize), (25, 21), (26, 20), (25, 22)] {
                let (cw, ch) = (fw.div_ceil(2), fh.div_ceil(2));
                for y0 in 0..fh + 4 {
                    for x0 in 0..fw + 4 {
                        let bounds = chroma_box(x0, y0, bm.w, bm.h, fw, fh);
                        let mut got = vec![90u8; cw * ch];
                        let mut want = got.clone();
                        paint_chroma_i420(&mut got, cw, &bm, x0, y0, bounds);
                        reference_i420(&mut want, cw, &bm, x0, y0, fw, fh);
                        assert_eq!(
                            got, want,
                            "I420 bounded chroma differs from the whole-plane scan at frame \
                             {fw}x{fh} caption ({x0},{y0}) size {}x{}",
                            bm.w, bm.h
                        );
                        let mut got = vec![90u8; cw * 2 * ch];
                        let mut want = got.clone();
                        paint_chroma_nv12(&mut got, cw * 2, &bm, x0, y0, bounds);
                        reference_nv12(&mut want, cw * 2, &bm, x0, y0, fw, fh);
                        assert_eq!(
                            got, want,
                            "NV12 bounded chroma differs from the whole-plane scan at frame \
                             {fw}x{fh} caption ({x0},{y0}) size {}x{}",
                            bm.w, bm.h
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 3000, "the sweep should be broad, got {checked} placements");

        // A large frame at the edge placements, plus the proof that the box really is a
        // shrink: it must not cover the whole plane for a caption occupying a small part.
        let bm = font::rasterize_line(s, "Wg|_");
        for &(fw, fh) in &[(64usize, 48usize), (65, 49), (320, 180)] {
            for &(x0, y0) in &[
                (0usize, 0usize),
                (1, 1),
                (3, 5),
                (fw / 2, fh / 2),
                (fw.saturating_sub(2), fh.saturating_sub(2)),
                (fw + 10, fh + 10), // entirely off-frame
            ] {
                let cw = fw.div_ceil(2);
                let ch = fh.div_ceil(2);
                let mut got = vec![90u8; cw * ch];
                let mut want = got.clone();
                let bounds = chroma_box(x0, y0, bm.w, bm.h, fw, fh);
                paint_chroma_i420(&mut got, cw, &bm, x0, y0, bounds);
                reference_i420(&mut want, cw, &bm, x0, y0, fw, fh);
                assert_eq!(
                    got, want,
                    "bounded chroma differs from the whole-plane scan at frame {fw}x{fh} \
                     caption ({x0},{y0}) size {}x{}",
                    bm.w, bm.h
                );
                if fw >= 320 && x0 < fw {
                    let (cx0, cx1, cy0, cy1) = bounds;
                    assert!(
                        (cx1 - cx0) * (cy1 - cy0) < cw * ch,
                        "the caption box should be smaller than the whole chroma plane"
                    );
                }
            }
        }
    }

    /// The same, for the PGS bitmap path's chroma pass.
    #[test]
    fn bounded_bitmap_chroma_matches_a_whole_plane_scan() {
        let (w, h) = (128u32, 96u32);
        // A 20x10 semi-transparent coloured caption near the bottom-right.
        let ds = pgs::DisplaySet {
            rgba: (0..20 * 10 * 4).map(|i| (i % 251) as u8).collect(),
            width: 20,
            height: 10,
            x: 100,
            y: 80,
            video_width: w,
            video_height: h,
        };
        let mut got = vec![70u8; frame_size(PixelFormat::I420, w, h)];
        let mut want = got.clone();
        {
            let mut f = VideoFrameMut::new(&mut got, w, h, PixelFormat::I420).unwrap();
            assert!(draw_bitmap(&mut f, &ds));
        }
        // Reference: the luma pass is shared, so replay it, then scan every chroma cell.
        {
            let mut f = VideoFrameMut::new(&mut want, w, h, PixelFormat::I420).unwrap();
            let (fw, fh) = (w as usize, h as usize);
            let (ref_w, ref_h) = (fw, fh);
            let dst_x0 = ds.x as usize * fw / ref_w;
            let dst_y0 = ds.y as usize * fh / ref_h;
            let dst_w = (ds.width as usize * fw).div_ceil(ref_w);
            let dst_h = (ds.height as usize * fh).div_ceil(ref_h);
            let sample = |dx: usize, dy: usize| -> Option<[u8; 4]> {
                let sx = dx * ds.width as usize / dst_w;
                let sy = dy * ds.height as usize / dst_h;
                let si = (sy * ds.width as usize + sx) * 4;
                ds.rgba.get(si..si + 4).map(|p| [p[0], p[1], p[2], p[3]])
            };
            if let Some((y_plane, y_stride)) = f.plane_mut(0) {
                for dy in 0..dst_h {
                    let py = dst_y0 + dy;
                    if py >= fh {
                        break;
                    }
                    for dx in 0..dst_w {
                        let px = dst_x0 + dx;
                        if px >= fw {
                            break;
                        }
                        let Some([r, g, b, a]) = sample(dx, dy) else { continue };
                        if a == 0 {
                            continue;
                        }
                        let (yv, _, _) = rgb_to_ycbcr_bt709(r, g, b);
                        let idx = py * y_stride + px;
                        if idx < y_plane.len() {
                            y_plane[idx] = over(y_plane[idx], yv, a);
                        }
                    }
                }
            }
            for plane in 1..=2 {
                let is_cr = plane == 2;
                if let Some((c_plane, c_stride)) = f.plane_mut(plane) {
                    for cy in 0..fh.div_ceil(2) {
                        for cx in 0..fw.div_ceil(2) {
                            let (lx, ly) = (2 * cx, 2 * cy);
                            if lx < dst_x0 || ly < dst_y0 {
                                continue;
                            }
                            let (dx, dy) = (lx - dst_x0, ly - dst_y0);
                            if dx >= dst_w || dy >= dst_h {
                                continue;
                            }
                            let Some([r, g, b, a]) = sample(dx, dy) else { continue };
                            if a == 0 {
                                continue;
                            }
                            let (_, cb, cr) = rgb_to_ycbcr_bt709(r, g, b);
                            let c = if is_cr { cr } else { cb };
                            let idx = cy * c_stride + cx;
                            if idx < c_plane.len() {
                                c_plane[idx] = over(c_plane[idx], c, a);
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(got, want, "bounded bitmap chroma differs from the whole-plane scan");
    }

    // ---- flush / seek ----

    /// A flush (seek) drops the cue timeline. Without this the ring keeps the pre-seek cues:
    /// they are all in the *future* of the new playhead, so `evict_stale` never removes them,
    /// and playing forward shows them again alongside the re-sent copies.
    #[test]
    fn flush_clears_the_cue_timeline() {
        let mut o = SubtitleOverlay::new();
        for i in 0..10u64 {
            o.push_cue(format!("cue {i}"), Timestamp::from_nanos(i * 1_000_000_000), Timestamp::from_nanos(900_000_000));
        }
        o.push_bitmap(
            pgs::DisplaySet { rgba: vec![255; 4], width: 1, height: 1, x: 0, y: 0, video_width: 8, video_height: 8 },
            Timestamp::from_nanos(0),
            Timestamp::from_nanos(1_000_000_000),
        );
        assert_eq!(o.cues.len(), 10);
        assert!(o.bitmap.is_some());
        o.flush();
        assert!(o.cues.is_empty(), "seek drops the stale cue ring");
        assert!(o.bitmap.is_none(), "seek drops the stale bitmap caption");
    }

    /// A cue re-sent verbatim (a seek landing before a cue still in the ring, a looping source)
    /// must not stack on screen twice.
    #[test]
    fn a_resent_cue_is_not_shown_twice() {
        let mut o = SubtitleOverlay::new();
        let (t, d) = (Timestamp::from_nanos(1_000_000_000), Timestamp::from_nanos(2_000_000_000));
        o.push_cue("Hello".into(), t, d);
        o.push_cue("Hello".into(), t, d);
        assert_eq!(o.cues.len(), 1, "the identical re-send is dropped");
        assert_eq!(o.active_text(Timestamp::from_nanos(1_500_000_000)), vec!["Hello"]);
        // Distinct cues at the same start still both show (two speakers on screen at once).
        o.push_cue("- Who's there?".into(), t, d);
        assert_eq!(o.active_text(Timestamp::from_nanos(1_500_000_000)).len(), 2);
    }

    // ---- the caption cache ----

    /// A cue whose text contains a newline (every multi-line SubRip/WebVTT cue, and every ASS
    /// cue with `\N`) must become **two stacked bitmaps**. Rasterising the whole string as one
    /// line rendered the `\n` as the font's replacement box: `line one[]line two` on one row.
    #[test]
    fn a_multi_line_cue_becomes_two_stacked_lines() {
        let mut o = SubtitleOverlay::new();
        o.push_cue(
            "line one\nline two".into(),
            Timestamp::from_nanos(0),
            Timestamp::from_nanos(1_000_000_000),
        );
        assert!(o.refresh_caption(Timestamp::from_nanos(0), 360));
        assert_eq!(o.caption.lines.len(), 2, "one coverage bitmap per physical line");
        assert_eq!(o.caption.lines[0].h, o.caption.lines[1].h, "same line box height");

        // And they land on different rows of the frame.
        let (w, h) = (320u32, 180u32);
        let mut buf = vec![128u8; frame_size(PixelFormat::I420, w, h)];
        {
            let mut f = VideoFrameMut::new(&mut buf, w, h, PixelFormat::I420).unwrap();
            draw_caption(&mut f, &o.caption.lines, o.caption.line_h);
        }
        let ys = w as usize;
        let painted: Vec<usize> = (0..h as usize)
            .filter(|&y| buf[y * ys..(y + 1) * ys].iter().any(|&p| p != 128))
            .collect();
        let (first, last) = (painted[0], *painted.last().unwrap());
        assert!(
            last - first >= o.caption.line_h,
            "the two lines stack at least a line height apart (rows {first}..={last}, line_h {})",
            o.caption.line_h
        );
    }

    /// The baked atlas tops out at 32 px, so `size_nearest` saturates: every frame taller than
    /// ~384 px gets the *same* 32 px text however large the picture is. At 1080p the target
    /// height is 90 px and the caption is drawn at 32 — about a third of the intended size, and
    /// small enough to be hard to read at a normal viewing distance. Recorded here as a known
    /// limitation of the v1 "no scaling, snap to a baked size" model rather than a silent one.
    #[test]
    fn caption_size_saturates_on_tall_frames() {
        let target_720 = 720.0 * TEXT_HEIGHT_FRAC;
        let target_1080 = 1080.0 * TEXT_HEIGHT_FRAC;
        let s720 = font::size_nearest(target_720);
        let s1080 = font::size_nearest(target_1080);
        assert_eq!(s720.px, s1080.px, "720p and 1080p both clamp to the largest baked size");
        assert!(
            s1080.px < target_1080 / 2.0,
            "the drawn size ({}) is less than half the intended {target_1080} px",
            s1080.px
        );
    }

    /// The render-on-change gate: an unchanged cue reuses its bitmaps across frames, and a
    /// changed one re-rasterises. Identity is the coverage buffer's address — a re-render
    /// allocates a fresh `Vec`.
    #[test]
    fn an_unchanged_caption_is_not_re_rasterised() {
        let mut o = SubtitleOverlay::new();
        o.push_cue("Steady".into(), Timestamp::from_nanos(0), Timestamp::from_nanos(2_000_000_000));
        assert!(o.refresh_caption(Timestamp::from_nanos(0), 360));
        let first = o.caption.lines[0].cov.as_ptr();
        for f in 1..50u64 {
            assert!(o.refresh_caption(Timestamp::from_nanos(f * 40_000_000), 360));
            assert_eq!(
                o.caption.lines[0].cov.as_ptr(),
                first,
                "frame {f} re-rasterised an unchanged caption"
            );
        }
        // A second cue coming on screen invalidates the cache.
        o.push_cue(
            "Different".into(),
            Timestamp::from_nanos(1_000_000_000),
            Timestamp::from_nanos(1_000_000_000),
        );
        assert!(o.refresh_caption(Timestamp::from_nanos(1_000_000_000), 360));
        assert_eq!(o.caption.lines.len(), 2, "both cues are on screen now");
        // So does a resolution change. (NB 360 and 1080 would *not* differ: the atlas tops out
        // at 32 px, so every frame taller than ~384 px snaps to the same baked size — see the
        // `caption_size_saturates_on_tall_frames` test.)
        let before = o.caption.size_px;
        assert!(o.refresh_caption(Timestamp::from_nanos(1_000_000_000), 120));
        assert_ne!(o.caption.size_px, before, "a shorter frame picks a different baked size");
        // And an expired cue empties it.
        assert!(!o.refresh_caption(Timestamp::from_nanos(9_000_000_000), 120));
    }
}
