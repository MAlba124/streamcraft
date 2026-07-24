//! [`RawVideoParse`] — chops a raw-video byte stream into frame-sized buffers (spec:
//! Milestone applications §5; the `rawvideoparse` half). Bytes arrive on the sink pad
//! (`filesrc`-style); given the construction-time `(width, height, pixfmt, fps)` it emits
//! exactly one buffer per whole frame on the src pad, stamped with a PTS/duration on the
//! **fps grid**. It carries a partial frame across input buffer boundaries, exactly as
//! `wavparse` carries partial PCM.
//!
//! Both pads speak `video/raw`; the src is `dynamic` and **announces** its concrete
//! output format via [`Ctx::announce_format`] (the `wavparse`/`audioconvert` producer
//! pattern), so downstream re-fixates and reads the format via `ctx.negotiated`. It also
//! offers `bytes` on the sink so it links directly after a `filesrc`.

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{OfferDesc, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::time::{Rational, Timestamp};

use crate::format::{
    PixelFormat, VideoFormat, FAMILY, FIELD_FPS, FIELD_HEIGHT, FIELD_PIXFMT, FIELD_WIDTH,
    RAW_ANY_OFFER,
};
use crate::geometry::frame_size;
use crate::testsrc::frame_pts;

const SINK: PadId = PadId(0);
const SRC: PadId = PadId(1);

// The sink accepts raw `bytes` (a `filesrc` transport) *or* a `video/raw` stream; the
// src announces its concrete `video/raw` output at runtime. `bytes` is listed first on
// the sink so a plain byte upstream links, mirroring how `wavparse` bridges `bytes`.
static SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("bytes"), RAW_ANY_OFFER[0]];
static SRC_OFFERS: [OfferDesc; 1] = [RAW_ANY_OFFER[0]];

static PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &SINK_OFFERS,
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &SRC_OFFERS,
        dynamic: true, // concrete output format is announced at runtime
        validate: None,
    },
];

static DESC: ElementDesc = ElementDesc {
    name: "rawvideoparse",
    pads: &PADS,
    props: &[],
    // Passive: a pure transform that inlines into the upstream group (spec: Scheduling).
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Chunks a raw-video byte stream into frame-sized buffers of a fixed format.
pub struct RawVideoParse {
    format: VideoFormat,
    /// Bytes per whole frame (`frame_size` of the format).
    frame_bytes: usize,
    /// Carry for a partial frame straddling two input buffers (mirrors `wavparse`).
    carry: Vec<u8>,
    /// Frames emitted so far, for the fps-grid PTS/duration.
    frames_out: u64,
    /// Whether the output format has been announced downstream yet.
    announced: bool,
}

impl RawVideoParse {
    /// A parser emitting `format` frames from a raw byte stream.
    pub fn new(format: VideoFormat) -> Self {
        Self {
            frame_bytes: frame_size(format.pixfmt, format.width, format.height),
            format,
            carry: Vec::new(),
            frames_out: 0,
            announced: false,
        }
    }

    /// Convenience constructor from the individual fields.
    pub fn with_dims(
        width: u32,
        height: u32,
        pixfmt: PixelFormat,
        fps: Rational,
    ) -> Self {
        Self::new(VideoFormat::new(width, height, pixfmt, fps))
    }

    pub fn format(&self) -> VideoFormat {
        self.format
    }

    /// Announce the concrete output `video/raw` format on the src pad, once.
    fn announce(&mut self, ctx: &mut Ctx) {
        if self.announced {
            return;
        }
        let f = self.format;
        ctx.announce_format(
            SRC,
            FAMILY,
            &[
                (FIELD_WIDTH, ValueDesc::Int(f.width as i64)),
                (FIELD_HEIGHT, ValueDesc::Int(f.height as i64)),
                (FIELD_PIXFMT, ValueDesc::Id(f.pixfmt.caps_name())),
                (FIELD_FPS, ValueDesc::Rat(f.fps.num, f.fps.den)),
            ],
        );
        self.announced = true;
    }

    /// Push one whole frame (`self.frame_bytes` bytes) on the src pad, stamped on the
    /// fps grid. The frame must fit one pool slot (checked once).
    fn emit_frame(&mut self, ctx: &mut Ctx, frame: &[u8]) -> Result<(), Error> {
        debug_assert_eq!(frame.len(), self.frame_bytes);
        let mut buf = ctx.alloc(SRC);
        let cap = buf.memory.capacity();
        if cap < self.frame_bytes {
            return Err(Error::Resource(format!(
                "rawvideoparse: frame is {} bytes but the pool slot is only {cap}",
                self.frame_bytes
            )));
        }
        buf.memory.as_mut_full()[..self.frame_bytes].copy_from_slice(frame);
        buf.memory.set_len(self.frame_bytes);
        let idx = self.frames_out;
        buf.pts = frame_pts(self.format.fps, idx);
        buf.duration = frame_pts(self.format.fps, idx + 1).saturating_sub(buf.pts);
        self.frames_out += 1;
        ctx.out(SRC).push(buf);
        Ok(())
    }

    /// Consume `bytes` (appended to any carry), emitting every whole frame and keeping
    /// the remainder as the new carry.
    fn feed(&mut self, ctx: &mut Ctx, bytes: &[u8]) -> Result<(), Error> {
        if self.frame_bytes == 0 {
            return Err(Error::Resource("rawvideoparse: zero-size frame format".into()));
        }
        // Fast path: nothing carried and the buffer holds whole frames — no join copy.
        if self.carry.is_empty() {
            let whole = bytes.len() - (bytes.len() % self.frame_bytes);
            let mut off = 0;
            while off < whole {
                self.emit_frame(ctx, &bytes[off..off + self.frame_bytes])?;
                off += self.frame_bytes;
            }
            self.carry.extend_from_slice(&bytes[whole..]);
            return Ok(());
        }
        // Join the carry with the new bytes, then drain whole frames.
        let mut joined = std::mem::take(&mut self.carry);
        joined.extend_from_slice(bytes);
        let whole = joined.len() - (joined.len() % self.frame_bytes);
        let mut off = 0;
        while off < whole {
            self.emit_frame(ctx, &joined[off..off + self.frame_bytes])?;
            off += self.frame_bytes;
        }
        self.carry.clear();
        self.carry.extend_from_slice(&joined[whole..]);
        Ok(())
    }
}

impl Element for RawVideoParse {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.carry.clear();
        self.frames_out = 0;
        self.announced = false;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        self.announce(ctx);
        while let Some(buf) = inputs.pop() {
            self.feed(ctx, buf.memory.data())?;
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // A flush drops the partial-frame carry so the output stays frame-aligned across
        // a seek (spec: flush/seek); the fps-grid counter is left to the seek target.
        if matches!(event, Event::FlushStart) {
            self.carry.clear();
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // A leftover carry is an incomplete final frame (ragged input) — drop it.
        self.carry = Vec::new();
        self.frames_out = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_bytes_matches_geometry() {
        let p = RawVideoParse::with_dims(16, 16, PixelFormat::I420, Rational::new(25, 1));
        assert_eq!(p.frame_bytes, 256 + 64 + 64);
        assert_eq!(p.format().pixfmt, PixelFormat::I420);
    }
}
