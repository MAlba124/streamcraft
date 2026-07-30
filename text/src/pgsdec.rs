//! `pgsdec` — decode a PGS (HDMV Presentation Graphics Stream) subtitle track to the
//! `subtitle/bitmap` family for the overlay (spec: subtitle support; RFC 9559 §12.7
//! `S_HDMV/PGS` Matroska mapping). The image-based subtitle every BluRay rip carries.
//!
//! # Input shape: one Block, one Display Set (the mkv path)
//!
//! On the mkv `S_HDMV/PGS` path the demuxer frames each **Display Set** as its own Block: the
//! payload is the bare PGS segments (PCS, WDS, PDS, ODS, END — no `.sup` `PG`/timestamp
//! header) and the timing is the Block's PTS + `BlockDuration`. So `pgsdec` is a per-Block
//! decode: parse the Display Set ([`crate::pgs::parse_display_set`]), and emit one
//! `subtitle/bitmap` buffer (RGBA + geometry, see [`crate::pgs::encode_bitmap`]) with the
//! Block's timing passed through.
//!
//! ## Reassembling a chunked Block
//!
//! A very large Display Set (a full-width caption's RLE bitmap can be tens of KB) may exceed
//! the demuxer's pool slot size and arrive as **several buffers sharing one PTS**. PGS is
//! self-framing (every segment carries its length, the set ends with an `END` segment), so
//! `pgsdec` accumulates payload bytes until it has seen an `END` and only then decodes. A
//! change of PTS also flushes the accumulator (a new Display Set began), so a dropped `END`
//! never wedges the decoder.
//!
//! # Output: `subtitle/bitmap`
//!
//! One decoded Display Set → one buffer on [`crate::BITMAP_FAMILY`] (`subtitle/bitmap`): a
//! small header (bitmap size, placement, reference video size) + straight-alpha RGBA8888, the
//! Block's PTS + duration the caption's on-screen span. A **clear** Display Set (an empty
//! composition — the caption's time is up) emits a zero-bitmap buffer, so the overlay learns
//! to erase. A `bytes` escape lets a raw byte peer drive the same sink.
//!
//! # Robustness
//!
//! Block bytes are untrusted (spec: a crash on bad input is a P0). A malformed Display Set is
//! **warned-and-dropped** — the accumulator resets and the element carries on; it never panics
//! or emits garbage. Every segment/run length is bounds-checked in [`crate::pgs`].
//!
//! # v1 scope
//!
//! v1 decodes the mainstream show/clear Display Sets. Palette-update-only sets (fades), forced
//! flags, object cropping, and 2nd+ composition objects are not specially handled (a `warn`,
//! then the first object / a clear). See [`crate::pgs`]'s module docs.

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

use crate::pgs;
use crate::BITMAP_FAMILY;

/// The PGS subtitle family the mkv demuxer announces for an `S_HDMV/PGS` track.
pub const PGS_FAMILY: &str = "subtitle/pgs";

/// The single src pad. The sink (pad 0) is drained via `inputs.pop()` (`InputPolicy::Single`),
/// so it needs no named constant.
const SRC: PadId = PadId(1);

/// The last segment type byte (`END`) that closes a Display Set — used to detect a complete
/// accumulation before decoding.
const SEG_END: u8 = 0x80;

/// The sink accepts the PGS family the demuxer announces plus the `bytes` escape (a raw PGS
/// byte peer, e.g. a `.sup` reader stripped of its headers upstream). The src emits
/// `subtitle/bitmap`, also `bytes`-compatible so a byte sink can tap it.
static SINK_OFFERS: [OfferDesc; 2] =
    [OfferDesc::any(PGS_FAMILY), OfferDesc::any("bytes")];
static SRC_OFFERS: [OfferDesc; 2] = [OfferDesc::any(BITMAP_FAMILY), OfferDesc::any("bytes")];

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
        // The output family is fixed (`subtitle/bitmap`), announced once at start.
        dynamic: true,
        validate: None,
    },
];

static DESC: ElementDesc = ElementDesc {
    name: "pgsdec",
    pads: &PADS,
    props: &[],
    // Passive: a per-Block decode with no blocking IO; inlines into the upstream group like
    // `subparse`. The RLE decode is bounded (a caption bitmap), well within the inline budget.
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // COLD: registry make_default — boxes one element instance at plugin-registration time.
    #[allow(clippy::disallowed_methods)]
    make_default: Some(|| Box::new(PgsDec::new())),
};

/// Decodes a PGS subtitle track to `subtitle/bitmap`. See the module docs.
pub struct PgsDec {
    announced: bool,
    /// Accumulated Display-Set bytes for a Block that spanned several pool slots (same PTS).
    accum: Vec<u8>,
    /// The PTS/duration of the Display Set being accumulated (the first chunk's timing).
    accum_pts: Timestamp,
    accum_dur: Timestamp,
    /// Whether the accumulator currently holds any bytes (a set in progress).
    accumulating: bool,
    /// Display sets decoded (health counter).
    pub decoded: u64,
    /// Display sets warned-and-dropped for malformed framing.
    pub dropped: u64,
}

impl Default for PgsDec {
    fn default() -> Self {
        Self::new()
    }
}

impl PgsDec {
    // COLD: one-time constructor — the empty reassembly accumulator (Vec::new() = zero capacity;
    // grown once, then `clear()`ed and reused across Blocks, never reallocated per caption).
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        Self {
            announced: false,
            accum: Vec::new(),
            accum_pts: Timestamp::NONE,
            accum_dur: Timestamp::NONE,
            accumulating: false,
            decoded: 0,
            dropped: 0,
        }
    }

    /// Announce the `subtitle/bitmap` output family once, so downstream re-fixates (spec:
    /// dynamic caps — producer side). No fields: geometry rides the buffer payload.
    fn announce(&mut self, ctx: &mut Ctx) {
        if self.announced {
            return;
        }
        ctx.announce_format(SRC, BITMAP_FAMILY, &[]);
        self.announced = true;
    }

    /// Decode the accumulated Display Set and emit a `subtitle/bitmap` buffer (or warn-and-drop
    /// on malformed framing). Resets the accumulator either way.
    fn flush_set(&mut self, ctx: &mut Ctx) {
        if self.accum.is_empty() {
            self.reset_accum();
            return;
        }
        match pgs::parse_display_set(&self.accum) {
            Ok(ds) => {
                let wire = pgs::encode_bitmap(&ds);
                let mut out = ctx.alloc_exact(SRC, wire.len().max(1));
                out.memory.as_mut_full()[..wire.len()].copy_from_slice(&wire);
                out.memory.set_len(wire.len());
                out.pts = self.accum_pts;
                out.duration = self.accum_dur;
                ctx.out(SRC).push(out);
                self.decoded += 1;
            }
            Err(e) => {
                // Warn-and-drop: a malformed Display Set never kills the pipeline (spec: P0).
                log!(&*ctx, Level::Warn, "pgs_drop", reason = e.reason());
                self.dropped += 1;
            }
        }
        self.reset_accum();
    }

    fn reset_accum(&mut self) {
        self.accum.clear();
        self.accumulating = false;
        self.accum_pts = Timestamp::NONE;
        self.accum_dur = Timestamp::NONE;
    }
}

impl Element for PgsDec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.announced = false;
        self.reset_accum();
        self.decoded = 0;
        self.dropped = 0;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        self.announce(ctx);
        while let Some(buf) = inputs.pop() {
            let data = buf.memory.data();
            // A new PTS means a new Display Set — flush any straggling partial accumulation
            // (a dropped END) so the decoder never wedges on stale bytes.
            if self.accumulating && buf.pts != self.accum_pts {
                self.flush_set(ctx);
            }
            if !self.accumulating {
                self.accum_pts = buf.pts;
                self.accum_dur = buf.duration;
                self.accumulating = true;
            }
            self.accum.extend_from_slice(data);
            // A Display Set closes with an END segment. In the common case (one Block fits one
            // pool slot) this is the last byte of this very buffer, so we decode immediately.
            if ends_with_end_segment(&self.accum) {
                self.flush_set(ctx);
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // On end-of-stream, decode whatever partial set remains (best-effort), then reset.
        if matches!(event, Event::Eos) {
            self.flush_set(ctx);
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.reset_accum();
    }
}

/// Whether `data` ends on a well-framed `END` segment (`0x80, 0x00, 0x00`), i.e. the last
/// segment header is an END and its (zero) payload lands exactly at the end. We only trust a
/// *terminal* END so a stray `0x80` inside an RLE run doesn't prematurely flush; the cheap
/// check is "the final three bytes are an END header with a zero length that closes the buffer".
fn ends_with_end_segment(data: &[u8]) -> bool {
    let n = data.len();
    n >= 3 && data[n - 3] == SEG_END && data[n - 2] == 0 && data[n - 1] == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_terminal_end_segment() {
        assert!(ends_with_end_segment(&[0x16, 0x00, 0x00, 0x80, 0x00, 0x00]));
        assert!(!ends_with_end_segment(&[0x16, 0x00, 0x00]), "no END");
        assert!(!ends_with_end_segment(&[0x80]), "too short");
        // An END *header* with a nonzero declared size is not a terminal END.
        assert!(!ends_with_end_segment(&[0x80, 0x00, 0x05]));
    }

    #[test]
    fn families_are_pgs_in_bitmap_out() {
        assert_eq!(PGS_FAMILY, "subtitle/pgs");
        assert_eq!(BITMAP_FAMILY, "subtitle/bitmap");
        assert_eq!(SINK_OFFERS[0].family, "subtitle/pgs");
        assert_eq!(SRC_OFFERS[0].family, "subtitle/bitmap");
    }
}
