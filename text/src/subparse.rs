//! `subparse` — normalise a subtitle stream to a format-agnostic **`subtitle/events`** stream
//! for the overlay (spec: subtitle support; RFC 9559 §12.7 S_TEXT/* mappings).
//!
//! The overlay must not care whether the source was SubRip, WebVTT or ASS — so this element
//! collapses all three to one shape: **plain UTF-8 cue text, markup stripped, one cue per
//! buffer, timing on the buffer's PTS + duration**. Downstream, the overlay only ever sees
//! `subtitle/events`.
//!
//! ## Input shape: one Block, one cue (the mkv path)
//!
//! On the mkv S_TEXT/* path the demuxer has already framed each cue as its own buffer — the
//! Block payload is exactly one cue's text, and the timing is the Block's PTS +
//! `BlockDuration` (which the demuxer stamps onto `buffer.duration`). So `subparse` on this
//! path is a pure **byte transform**: it passes PTS/duration straight through and only strips
//! markup from the payload, per the negotiated family:
//! - `subtitle/srt` → SubRip cue body (HTML-ish tags dropped),
//! - `subtitle/vtt` → WebVTT cue body (inline spans dropped),
//! - `subtitle/ass` → the ASS Block body's Text field (override blocks dropped, `\N`→newline).
//!
//! It learns the family from the negotiated sink caps (the `flacdec`/`audioconvert` pattern:
//! read it off `ctx.negotiated(sink)` at `start`, refresh on a `FormatChange`). A `bytes`
//! fallback (or an un-negotiated edge) is treated as already-plain text (passthrough).
//!
//! Whole-*file* parsing (a standalone `.srt` fed via `filesrc`, where one buffer is the entire
//! document with its own timeline) is handled by [`crate::parse`]'s `parse_srt`/`parse_vtt`/
//! `parse_ass` — the overlay's example uses the demuxer's per-cue framing, so the element
//! itself implements the streaming one-block-one-cue path; the file path is a library call for
//! a future `subfilesrc`. This keeps the element a light, inline byte transform (spec:
//! Scheduling — Passive).

use profluens_core::batch::Inputs;
use profluens_core::buffer::Buffer;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

use crate::parse;

/// The normalised output family every overlay speaks: plain-UTF-8 cue text on the buffer, PTS
/// + duration the cue interval. Named so an autoplugger links `subparse ! suboverlay.text`.
pub const EVENTS_FAMILY: &str = "subtitle/events";

const SINK: PadId = PadId(0);
const SRC: PadId = PadId(1);

/// The sink accepts the three text subtitle families the mkv demuxer announces (RFC 9559
/// §12.7) plus the `bytes` escape (a raw byte peer, or an already-plain stream). The src emits
/// the normalised `subtitle/events`, also `bytes`-compatible so a byte sink can tap it.
static SINK_OFFERS: [OfferDesc; 4] = [
    OfferDesc::any("subtitle/srt"),
    OfferDesc::any("subtitle/vtt"),
    OfferDesc::any("subtitle/ass"),
    OfferDesc::any("bytes"),
];
static SRC_OFFERS: [OfferDesc; 2] = [OfferDesc::any(EVENTS_FAMILY), OfferDesc::any("bytes")];

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
        // The output family is fixed (`subtitle/events`), announced once at start.
        dynamic: true,
        validate: None,
    },
];

static DESC: ElementDesc = ElementDesc {
    name: "subparse",
    pads: &PADS,
    props: &[],
    // Passive: a light byte→byte transform (strip markup); inlines into the upstream group
    // exactly like `audioconvert` — no blocking, no per-buffer allocation beyond the cue text.
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
    make_default: Some(|| Box::new(SubParse::new())),
};

/// Which markup dialect the negotiated sink carries — fixes how a cue body is stripped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dialect {
    Srt,
    Vtt,
    Ass,
    /// `bytes` / unknown: already plain text, pass through untouched.
    Plain,
}

impl Dialect {
    fn from_family(name: &str) -> Dialect {
        match name {
            "subtitle/srt" => Dialect::Srt,
            "subtitle/vtt" => Dialect::Vtt,
            "subtitle/ass" => Dialect::Ass,
            _ => Dialect::Plain,
        }
    }

    /// Strip one cue's Block body to plain UTF-8 per this dialect.
    // COLD: one owned cue string per sparse subtitle Block (a cue every few seconds), never per
    // video frame — the natural shape of a text cue (the srt/vtt/ass arms allocate likewise). The
    // element then copies this into a pool buffer (`ctx.alloc_exact`) for the wire.
    #[allow(clippy::disallowed_methods)]
    fn strip(self, body: &str) -> String {
        match self {
            Dialect::Srt => parse::cue_body_srt(body),
            Dialect::Vtt => parse::cue_body_vtt(body),
            Dialect::Ass => parse::cue_body_ass_dialogue(body),
            Dialect::Plain => body.to_string(),
        }
    }
}

/// Normalises an mkv S_TEXT/* subtitle stream to `subtitle/events` (one cue per buffer, plain
/// text, timing passed through). See the module docs.
pub struct SubParse {
    dialect: Dialect,
    announced: bool,
}

impl Default for SubParse {
    fn default() -> Self {
        Self::new()
    }
}

impl SubParse {
    pub fn new() -> Self {
        // Until the sink is negotiated, assume plain passthrough (a `bytes` edge, or a test
        // feeding pre-stripped text). `learn_from_sink` overrides once caps arrive.
        Self { dialect: Dialect::Plain, announced: false }
    }

    /// Read the negotiated sink family and set the strip dialect (the flacdec/audioconvert
    /// consumer-side caps pattern). Called at `start` (a link-time-fixed edge) and on a
    /// runtime `FormatChange`.
    fn learn_from_sink(&mut self, ctx: &Ctx) {
        if let Some(fixed) = ctx.negotiated(SINK) {
            if let Some(name) = ctx.family_name(fixed.family) {
                self.dialect = Dialect::from_family(name);
            }
        }
    }

    /// Announce the `subtitle/events` output family once, so downstream re-fixates (spec:
    /// dynamic caps — producer side). No fields: an event stream is bare UTF-8 text.
    fn announce(&mut self, ctx: &mut Ctx) {
        if self.announced {
            return;
        }
        ctx.announce_format(SRC, EVENTS_FAMILY, &[]);
        self.announced = true;
    }
}

impl Element for SubParse {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        self.announced = false;
        self.learn_from_sink(ctx);
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // A late-arriving announcement may have set the family after start().
        self.learn_from_sink(ctx);
        self.announce(ctx);
        while let Some(buf) = inputs.pop() {
            // One Block = one cue: strip its body to plain text, keep the timing verbatim.
            // Non-UTF-8 bytes are lossily decoded rather than dropped (a corrupt cue should
            // still show *something*; subtitle text is UTF-8 by mapping, so this is rare).
            let body = String::from_utf8_lossy(buf.memory.data());
            let text = self.dialect.strip(&body);
            let bytes = text.into_bytes();
            // Emit the normalised cue on the src pad, timing passed straight through. Sized
            // exactly to the cue text — cues are tiny, so a right-sized slot is fine and keeps
            // the buffer's `len` == the cue length (the overlay reads the whole buffer). The
            // `.max(1)` guards `acquire_exact(0)` for an empty cue; we still set_len(0).
            let mut out: Buffer = ctx.alloc_exact(SRC, bytes.len().max(1));
            out.memory.as_mut_full()[..bytes.len()].copy_from_slice(&bytes);
            out.memory.set_len(bytes.len());
            // Timing rides through: the container already framed the cue interval.
            out.pts = buf.pts;
            out.dts = buf.dts;
            out.duration = buf.duration;
            out.flags = buf.flags;
            ctx.out(SRC).push(out);
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::FormatChange(_)) {
            self.learn_from_sink(ctx);
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect_from_family_maps_the_three_text_codecs() {
        assert_eq!(Dialect::from_family("subtitle/srt"), Dialect::Srt);
        assert_eq!(Dialect::from_family("subtitle/vtt"), Dialect::Vtt);
        assert_eq!(Dialect::from_family("subtitle/ass"), Dialect::Ass);
        assert_eq!(Dialect::from_family("bytes"), Dialect::Plain, "bytes → passthrough");
        assert_eq!(Dialect::from_family("subtitle/pgs"), Dialect::Plain, "unknown → passthrough");
    }

    #[test]
    fn strip_normalises_each_dialect() {
        assert_eq!(Dialect::Srt.strip("<i>hi</i>"), "hi");
        assert_eq!(Dialect::Vtt.strip("<c>hi</c> there"), "hi there");
        assert_eq!(
            Dialect::Ass.strip("0,0,Default,,0,0,0,,{\\b1}bold{\\b0}"),
            "bold",
            "ass Block body: text after 8 commas, overrides stripped"
        );
        assert_eq!(Dialect::Plain.strip("already plain"), "already plain");
    }
}
