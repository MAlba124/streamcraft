//! [`VideoCkSink`] — a clock-driven test/assertion video sink (spec: Clocking — the
//! timed sink; Testing — assertion sinks). It renders each frame **on the pipeline
//! clock** (`ctx.wait_until(pts)`), so against a `MockClock` a test drives hours of
//! virtual time in milliseconds of wall time and proves nothing renders ahead of its
//! deadline. For every frame it records `(pts, checksum, rendered_at)`, so a test can
//! verify both the schedule and — via the checksum — the pixels a
//! [`VideoTestSrc`](crate::testsrc::VideoTestSrc) produced.
//!
//! Structure mirrors `TimedTestSink` exactly, including the [`VideoCkSinkStats`]
//! handle read after `run()`. Two flush-safety refinements (spec: flush/seek): a
//! `FlushStart` resets the recorded stream cleanly, and a long clock wait bails when
//! [`Ctx::seek_gen`] changes, so the scheduler can run the flush promptly instead of
//! after the whole batch has "played".

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use streamcraft_core::batch::Inputs;
use streamcraft_core::clock::WaitOutcome;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::format::RAW_ANY_OFFER;

const SINK: PadId = PadId(0);

// Accept a `video/raw` stream, or a raw `bytes` transport (so it can sit directly after
// a `rawvideoparse`, which announces `video/raw`, or a byte producer in a smoke test).
use streamcraft_core::format::OfferDesc;
static OFFERS: [OfferDesc; 2] = [RAW_ANY_OFFER[0], OfferDesc::any("bytes")];

static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

static DESC: ElementDesc = ElementDesc {
    name: "videocksink",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // The stats handle is discarded for the parse path (see `VideoCkSink::new_element`).
    make_default: Some(|| Box::new(VideoCkSink::new_element())),
};

/// One rendered frame: the PTS it carried, an FNV-1a checksum of its bytes, and the
/// running time the clock read when it was released for rendering (`rendered_at >= pts`
/// always holds — the sink never renders ahead of the clock).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoRender {
    pub pts: Timestamp,
    pub checksum: u64,
    pub rendered_at: Timestamp,
}

/// FNV-1a over a frame's bytes — the same fold the audio/test sinks use, so a test can
/// recompute the expected digest independently.
pub fn checksum(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(PRIME);
    }
    h
}

struct SinkShared {
    renders: Mutex<Vec<VideoRender>>,
    count: AtomicU64,
    done: AtomicBool,
    interrupted: AtomicBool,
    flushes: AtomicU64,
}

/// A handle to a [`VideoCkSink`]'s results, readable after the sink thread joins (i.e.
/// after `pipeline.run()` returns).
pub struct VideoCkSinkStats(Arc<SinkShared>);

impl VideoCkSinkStats {
    /// Every rendered frame, in receive order.
    pub fn renders(&self) -> Vec<VideoRender> {
        self.0.renders.lock().unwrap().clone()
    }

    /// Number of frames rendered so far.
    pub fn count(&self) -> u64 {
        self.0.count.load(Ordering::Acquire)
    }

    /// Whether the sink saw EOS (the stream ended cleanly).
    pub fn is_done(&self) -> bool {
        self.0.done.load(Ordering::Acquire)
    }

    /// Whether a clock wait was cut short by an interrupt (flush / shutdown).
    pub fn was_interrupted(&self) -> bool {
        self.0.interrupted.load(Ordering::Acquire)
    }

    /// How many `FlushStart`s reset the recorded stream.
    pub fn flushes(&self) -> u64 {
        self.0.flushes.load(Ordering::Acquire)
    }
}

/// An active sink that renders each video frame on the pipeline clock.
pub struct VideoCkSink {
    shared: Arc<SinkShared>,
}

impl VideoCkSink {
    /// Returns the sink (to `add` to a pipeline) and a stats handle to read after `run()`.
    pub fn new() -> (Self, VideoCkSinkStats) {
        let shared = Arc::new(SinkShared {
            renders: Mutex::new(Vec::new()),
            count: AtomicU64::new(0),
            done: AtomicBool::new(false),
            interrupted: AtomicBool::new(false),
            flushes: AtomicU64::new(0),
        });
        let stats = VideoCkSinkStats(Arc::clone(&shared));
        (Self { shared }, stats)
    }

    /// The sink alone, discarding the stats handle — for the registry `make_default`,
    /// which cannot hand a handle back through the parse path (spec: Plugins). A launch
    /// run inspects results via the pipeline's counters/taps or its EOS exit, not the
    /// [`VideoCkSinkStats`] a typed caller would keep.
    pub fn new_element() -> Self {
        Self::new().0
    }
}

impl Element for VideoCkSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // The seek generation at entry: a bump while we render a batch means the frames
        // behind the seek are stale, so we bail and let the scheduler run the flush
        // (spec: flush/seek — blocking sinks bail on ctx.seek_gen()).
        let entry_gen = ctx.seek_gen();
        while let Some(buf) = inputs.pop() {
            // Render on the clock: block until running time reaches this frame's PTS.
            match ctx.wait_until(buf.pts) {
                WaitOutcome::Reached => {}
                WaitOutcome::Interrupted => {
                    self.shared.interrupted.store(true, Ordering::Release);
                    return Ok(Flow::Ok);
                }
            }
            // A seek landed while we waited: drop this and the rest of the batch.
            if ctx.seek_gen() != entry_gen {
                return Ok(Flow::Ok);
            }
            let render = VideoRender {
                pts: buf.pts,
                checksum: checksum(buf.memory.data()),
                rendered_at: ctx.now(),
            };
            self.shared.renders.lock().unwrap().push(render);
            self.shared.count.fetch_add(1, Ordering::AcqRel);
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // A flush resets the recorded stream cleanly (spec: flush/seek): drop everything
        // rendered so the post-seek run is what the test inspects.
        if matches!(event, Event::FlushStart) {
            self.shared.renders.lock().unwrap().clear();
            self.shared.count.store(0, Ordering::Release);
            self.shared.flushes.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.shared.done.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_is_stable_and_content_sensitive() {
        assert_eq!(checksum(b"abc"), checksum(b"abc"));
        assert_ne!(checksum(b"abc"), checksum(b"abd"));
        // Empty frame is the bare offset basis.
        assert_eq!(checksum(&[]), 0xcbf2_9ce4_8422_2325);
    }
}
