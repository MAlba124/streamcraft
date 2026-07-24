//! [`VideoTestSrc`] — a seedable, deterministic raw-video source (spec: Testing —
//! seedable patterns; Milestone applications §5). An **active** source: it emits `count`
//! frames whose content is a pure function of `(seed, frame_index)`, stamps each with a
//! PTS/duration on the **fps grid**, then EOS — so a sink can verify both the schedule
//! and the pixels with no IO and no clock sleeps.
//!
//! Its src pad advertises the broad [`RAW_ANY_OFFER`](crate::format::RAW_ANY_OFFER) (so it
//! links against any `video/raw` peer) and **announces** its concrete construction-time
//! format — `width` / `height` / `pixfmt` / `fps` — the moment it starts, via
//! [`Ctx::announce_format`], exactly like `wavparse`/`audioconvert` on the producer side.
//! Downstream then reads the concrete format via `ctx.negotiated`.

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, PropDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{Constraint, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::time::{Rational, Timestamp};

use crate::format::{
    PixelFormat, VideoFormat, FAMILY, FIELD_FPS, FIELD_HEIGHT, FIELD_PIXFMT, FIELD_WIDTH,
    RAW_ANY_OFFER,
};
use crate::geometry::frame_size;
use crate::props::{read_dims, read_u64, DEFAULT_FORMAT};

const SRC: PadId = PadId(0);

static PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &RAW_ANY_OFFER,
    dynamic: true, // the concrete format is announced at runtime
    validate: None,
}];

/// The construction parameters, for the parse path (spec: Plugins —
/// `parse("videotestsrc width=320 height=240 pixfmt=i420 fps=30/1 frames=90 seed=1 ! …")`).
/// Until a demuxer feeds the format through caps, `videotestsrc` needs these out of band;
/// the parse layer supplies them as props (the typed [`VideoTestSrc::new`] is the other
/// path). All structural (`live: false`): the source is built in `start()`. `width` /
/// `height` / `frames` / `seed` are ints, `pixfmt` an interned format name, `fps` a
/// rational (`30/1`).
static PROPS: [PropDesc; 6] = [
    PropDesc { name: "width", allowed: Constraint::Any, live: false },
    PropDesc { name: "height", allowed: Constraint::Any, live: false },
    PropDesc { name: "pixfmt", allowed: Constraint::Any, live: false },
    PropDesc { name: "fps", allowed: Constraint::Any, live: false },
    PropDesc { name: "frames", allowed: Constraint::Any, live: false },
    PropDesc { name: "seed", allowed: Constraint::Any, live: false },
];

static DESC: ElementDesc = ElementDesc {
    name: "videotestsrc",
    pads: &PADS,
    props: &PROPS,
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Default parameters (320x240 I420 @ 30 fps, 90 frames); override via the props.
    make_default: Some(|| Box::new(VideoTestSrc::new(DEFAULT_FORMAT, 0, 90))),
};

/// The deterministic byte a [`VideoTestSrc`] writes at flat frame offset `pos` of frame
/// `frame_index` for stream `seed`. A pure function of `(seed, frame_index, pos)`, so a
/// sink can regenerate any frame and compare, or checksum it, with no shared state.
#[inline]
pub fn frame_pattern_byte(seed: u64, frame_index: u64, pos: usize) -> u8 {
    // A cheap SplitMix-style avalanche over the three inputs — no external deps.
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(frame_index.wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(pos as u64);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    (x >> 24) as u8
}

/// The PTS of frame `i` on an `fps` grid, as an exact nanosecond timestamp:
/// `round(i * den / num * 1e9)`. Uses `i128` so long streams do not overflow and the
/// grid never drifts. Returns [`Timestamp::NONE`] for an invalid (zero-denominator or
/// non-positive) rate — "no timing".
pub fn frame_pts(fps: Rational, i: u64) -> Timestamp {
    if fps.den == 0 || fps.num <= 0 {
        return Timestamp::NONE;
    }
    // ns = i * (den / num) * 1e9, rounded to nearest.
    let num = fps.num as i128;
    let den = fps.den.unsigned_abs() as i128;
    let numer = i as i128 * den * 1_000_000_000i128;
    let ns = (numer + num / 2) / num;
    Timestamp::from_nanos(ns as u64)
}

/// A deterministic raw-video source: `count` frames of `(width, height, pixfmt)` at
/// `fps`, content seeded by `seed`.
pub struct VideoTestSrc {
    format: VideoFormat,
    seed: u64,
    count: u64,
    produced: u64,
    announced: bool,
    frame_bytes: usize,
}

impl VideoTestSrc {
    /// A source producing `count` frames of `format`, seeded by `seed`.
    pub fn new(format: VideoFormat, seed: u64, count: u64) -> Self {
        Self {
            frame_bytes: frame_size(format.pixfmt, format.width, format.height),
            format,
            seed,
            count,
            produced: 0,
            announced: false,
        }
    }

    /// Convenience constructor from the individual fields.
    pub fn with_dims(
        width: u32,
        height: u32,
        pixfmt: PixelFormat,
        fps: Rational,
        seed: u64,
        count: u64,
    ) -> Self {
        Self::new(VideoFormat::new(width, height, pixfmt, fps), seed, count)
    }

    pub fn format(&self) -> VideoFormat {
        self.format
    }

    /// Announce the concrete output `video/raw` format on the src pad, once (spec:
    /// dynamic caps — producer side).
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
}

impl Element for VideoTestSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Parsed props override the constructor parameters (spec: Plugins). `width` /
        // `height` / `pixfmt` / `fps` refine the format; `frames` / `seed` the schedule.
        self.format = read_dims(ctx, self.format);
        self.count = read_u64(ctx, "frames", self.count);
        self.seed = read_u64(ctx, "seed", self.seed);
        self.frame_bytes = frame_size(self.format.pixfmt, self.format.width, self.format.height);
        self.produced = 0;
        self.announced = false;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Announce the concrete format before the first frame so downstream re-fixates.
        self.announce(ctx);

        if self.produced >= self.count {
            return Ok(Flow::Eos);
        }
        // The frame must fit one pool slot (frame-per-buffer contract for raw video).
        let mut buf = match ctx.try_alloc(SRC) {
            Some(b) => b,
            None => return Ok(Flow::Ok), // pool full → backpressure
        };
        let cap = buf.memory.capacity();
        if cap < self.frame_bytes {
            return Err(Error::Resource(format!(
                "videotestsrc: frame is {} bytes but the pool slot is only {cap}",
                self.frame_bytes
            )));
        }

        let idx = self.produced;
        let dst = &mut buf.memory.as_mut_full()[..self.frame_bytes];
        for (pos, b) in dst.iter_mut().enumerate() {
            *b = frame_pattern_byte(self.seed, idx, pos);
        }
        buf.memory.set_len(self.frame_bytes);
        buf.pts = frame_pts(self.format.fps, idx);
        buf.duration = frame_pts(self.format.fps, idx + 1).saturating_sub(buf.pts);
        self.produced += 1;
        ctx.out(SRC).push(buf);
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_pts_is_exact_on_the_grid() {
        // 30 fps → 1/30 s per frame; frame 30 is exactly 1 second.
        let fps = Rational::new(30, 1);
        assert_eq!(frame_pts(fps, 0), Timestamp::ZERO);
        assert_eq!(frame_pts(fps, 30), Timestamp::from_secs(1));
        assert_eq!(frame_pts(fps, 1), Timestamp::from_nanos(33_333_333));

        // 30000/1001 (29.97) — the drift-prone case. Frame 30000 is exactly 1001 s.
        let ntsc = Rational::new(30000, 1001);
        assert_eq!(frame_pts(ntsc, 30000), Timestamp::from_secs(1001));

        // An invalid rate yields NONE.
        assert!(frame_pts(Rational::new(0, 0), 5).is_none());
    }

    #[test]
    fn pattern_is_deterministic_and_seed_sensitive() {
        // Same (seed, frame, pos) → same byte; different seed usually differs.
        assert_eq!(frame_pattern_byte(1, 2, 3), frame_pattern_byte(1, 2, 3));
        let a: Vec<u8> = (0..64).map(|p| frame_pattern_byte(1, 0, p)).collect();
        let b: Vec<u8> = (0..64).map(|p| frame_pattern_byte(2, 0, p)).collect();
        assert_ne!(a, b, "different seeds produce different frames");
        let f0: Vec<u8> = (0..64).map(|p| frame_pattern_byte(1, 0, p)).collect();
        let f1: Vec<u8> = (0..64).map(|p| frame_pattern_byte(1, 1, p)).collect();
        assert_ne!(f0, f1, "different frame indices differ");
    }
}
