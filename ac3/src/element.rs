//! The `ac3dec` / `eac3dec` decoder elements — thin wrappers over the shared
//! A/52 decode core ([`crate::frame`], [`crate::parse`]). AC-3 / E-AC-3 bytes
//! arrive on the sink pad; interleaved s16 `audio/raw` leaves on the src pad,
//! decoded frame-by-frame as bytes arrive (mp3dec's incremental-transform shape).
//!
//! The two elements differ only in the **sink family string** they offer
//! (`ac3` vs `eac3`, matching the demuxer's codec-id announcement) and their
//! element name; both drive the same [`Frame`] core, which auto-detects AC-3 vs
//! E-AC-3 per syncframe by bsid (§5.3.2 / §E.1.2.2) — so an `ac3` element still
//! decodes an E-AC-3 syncframe if one appears, and vice versa. The split exists
//! for negotiation/autoplug, not for two decoders.
//!
//! ## Output contract (the downmix/sink negotiate against this)
//! - src offer family **`audio/raw`**; fields `rate` (Int), `channels` (Int),
//!   `sample` = **`s16`** (interleaved signed 16-bit).
//! - **Channel order: L, R, C, LFE, Ls, Rs** (ITU/SMPTE 5.1(side) order). The
//!   downmixer downstream assumes exactly this order; the core permutes A/52's
//!   stream order (Table 5.8) to it in [`Frame::decode`]. Documented here at the
//!   announce site as the brief requires.
//!
//! ## Robustness (P0)
//! A malformed syncframe is warned (bus, capped) and dropped; the framer resyncs
//! on the next `0B77`. Rate/channels are announced from the first decoded frame
//! and a mid-stream change is a loud warn+drop (real streams never change).
//!
//! ## Status
//! The elements + this whole contract (pads, families, `audio/raw` s16 output,
//! ITU channel order) are complete and the pipeline runs end-to-end. The decode
//! *core* is **not yet bit-accurate** — see the crate-level [`crate`] docs for the
//! precise breakdown of what reconstructs correctly vs. the open bit-allocation
//! issue. The output geometry and timeline are correct; the sample values are not.

use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

use crate::frame::Frame;
use crate::parse::{next_frame, overflow_drop, Framed};

const SRC_PAD: PadId = PadId(1);

// `audio/raw` vocabulary — literals, like every audio element (the pipeline
// interns by string, so ids line up with the downmix / audio sink peer).
const FAMILY: &str = "audio/raw";
const F_RATE: &str = "rate";
const F_CHANNELS: &str = "channels";
const F_SAMPLE: &str = "sample";
/// The core emits interleaved signed 16-bit (A/52 §7.11 output word width).
const SAMPLE_S16: &str = "s16";

/// How many syncframes `process()` decodes before yielding its output batch
/// downstream. Small on purpose: bounds the batch so the first audio reaches the
/// sink after ~one frame's decode. An AC-3 frame is 1536 samples (6 blocks × 256,
/// 32 ms at 48 kHz), so a couple still leaves plenty of downstream jitter buffer.
const FRAMES_PER_BATCH: u32 = 2;

static SAMPLE_VALUES: [ValueDesc; 1] = [ValueDesc::Id(SAMPLE_S16)];

// A broad `audio/raw` template: any rate/channels, s16 sample format. Concrete
// values are announced at runtime from the first decoded frame — the src pad is
// `dynamic` for exactly this reason (spec: Formats — dynamic caps).
static SRC_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_RATE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_CHANNELS, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_SAMPLE, allowed: ConstraintDesc::Set(&SAMPLE_VALUES), preferred: None },
];
// Typed `audio/raw` first, then the `bytes` bridge so a byte sink still links and
// the announcement rides it tolerantly (mirrors mp3dec / flacdec).
static SRC_OFFERS: [OfferDesc; 2] =
    [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }, OfferDesc::any("bytes")];

// Sink offers — one per element, keyed by the codec family the demuxer announces
// (`A_AC3` → `ac3`, `A_EAC3` → `eac3`; AVI's 0x2000/0x2001 → the same). The
// `bytes` escape lets a `filesrc` reading a raw `.ac3`/`.eac3` link too.
static AC3_SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("ac3"), OfferDesc::any("bytes")];
static EAC3_SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("eac3"), OfferDesc::any("bytes")];

static AC3_PADS: [PadDesc; 2] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &AC3_SINK_OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &SRC_OFFERS, dynamic: true, validate: None },
];
static EAC3_PADS: [PadDesc; 2] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &EAC3_SINK_OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &SRC_OFFERS, dynamic: true, validate: None },
];

// `make_default` boxes the element once at registry construction (spec: Plugins) — cold.
#[allow(clippy::disallowed_methods)]
static AC3_DESC: ElementDesc = ElementDesc {
    name: "ac3dec",
    pads: &AC3_PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: Some(|| Box::new(Ac3Dec::new())),
};
// `make_default` boxes the element once at registry construction (spec: Plugins) — cold.
#[allow(clippy::disallowed_methods)]
static EAC3_DESC: ElementDesc = ElementDesc {
    name: "eac3dec",
    pads: &EAC3_PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: Some(|| Box::new(Eac3Dec::new())),
};

/// The shared decode engine both elements delegate to — a byte accumulator, the
/// A/52 [`Frame`] core, and the runtime-announced output geometry.
struct Core {
    dec: Frame,
    /// Accumulated bytes not yet framed into whole syncframes.
    buf: Vec<u8>,
    /// Runtime format, learned from the first decoded frame.
    announced: bool,
    rate: u32,
    channels: usize,
    /// PCM samples-per-channel emitted so far — the zero-based PTS grid.
    samples_emitted: u64,
    /// Decoded interleaved i16 PCM not yet emitted — the backpressure carry,
    /// bounded to a batch: the framing loop never decodes ahead while this holds.
    pending: Vec<i16>,
    pending_pos: usize,
    /// Consecutive decode failures, for capped warning spam.
    consecutive_errors: u32,
    /// Syncframes consumed so far, stamping drop warnings for stream forensics.
    frame_index: u64,
    /// Set once we have reported that this stream uses an approximated E-AC-3
    /// advanced tool (SPX/AHT/coupling coords) — so the warning fires once.
    reported_approx: bool,
}

impl Core {
    // One-time element setup: `buf`/`pending` are reused-and-cleared across frames, not
    // re-allocated per frame (spec: performance #1) — cold.
    #[allow(clippy::disallowed_methods)]
    fn new() -> Self {
        Self {
            dec: Frame::new(),
            buf: Vec::new(),
            announced: false,
            rate: 0,
            channels: 0,
            samples_emitted: 0,
            pending: Vec::new(),
            pending_pos: 0,
            consecutive_errors: 0,
            frame_index: 0,
            reported_approx: false,
        }
    }

    fn reset_state(&mut self) {
        self.dec = Frame::new();
        self.buf.clear();
        self.announced = false;
        self.rate = 0;
        self.channels = 0;
        self.samples_emitted = 0;
        self.pending.clear();
        self.pending_pos = 0;
        self.consecutive_errors = 0;
        self.frame_index = 0;
        self.reported_approx = false;
    }

    /// PTS in nanoseconds for the current `samples_emitted` on the sample grid
    /// from a zero base. `NONE` until the rate is known.
    fn pts_now(&self) -> Timestamp {
        if self.rate == 0 {
            return Timestamp::NONE;
        }
        Timestamp::from_nanos((self.samples_emitted * 1_000_000_000) / u64::from(self.rate))
    }

    /// Emit the pending carry as interleaved LE s16 into pool buffers. With
    /// `bounded` it uses `try_alloc` and returns `Ok(false)` when the pool is full
    /// (backpressure); with `!bounded` it uses the unbounded pool to flush the
    /// tail at EOS. Returns `Ok(true)` once fully drained.
    fn emit_pending(&mut self, ctx: &mut Ctx, bounded: bool) -> Result<bool, Error> {
        let ch = self.channels.max(1);
        while self.pending_pos < self.pending.len() {
            let mut buf = if bounded {
                match ctx.try_alloc(SRC_PAD) {
                    Some(b) => b,
                    None => return Ok(false),
                }
            } else {
                ctx.alloc(SRC_PAD)
            };
            // Whole interleaved frames (ch samples) per buffer so a buffer never
            // splits a multichannel sample group.
            let cap_samples = ((buf.memory.capacity() / 2) / ch * ch).max(ch);
            let end = (self.pending_pos + cap_samples).min(self.pending.len());
            let dst = buf.memory.as_mut_full();
            let mut n = 0;
            for &s in &self.pending[self.pending_pos..end] {
                dst[n..n + 2].copy_from_slice(&s.to_le_bytes());
                n += 2;
            }
            buf.memory.set_len(n);
            buf.pts = self.pts_now();
            self.samples_emitted += ((end - self.pending_pos) / ch) as u64;
            buf.duration = self.pts_now().saturating_sub(buf.pts);
            ctx.out(SRC_PAD).push(buf);
            self.pending_pos = end;
        }
        self.pending.clear();
        self.pending_pos = 0;
        Ok(true)
    }

    /// Frame and decode the next whole syncframe from `self.buf`, appending its
    /// interleaved PCM to `self.pending`. Returns `Ok(true)` if a frame was
    /// consumed (whether it decoded or was dropped), `Ok(false)` if more input is
    /// needed. `element_name` labels warnings.
    //
    // The only heap allocation here is the once-per-stream approx-tool warning string
    // (`.to_string()`, guarded by `reported_approx`); the per-frame decode copies nothing
    // to the heap — cold.
    #[allow(clippy::disallowed_methods)]
    fn frame_and_decode_one(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        let (offset, len) = match next_frame(&self.buf) {
            Framed::Frame { offset, len, .. } => (offset, len),
            Framed::NeedMore { offset } => {
                if offset > 0 {
                    self.buf.drain(..offset);
                }
                return Ok(false);
            }
            Framed::NoSync => {
                let drop = overflow_drop(self.buf.len());
                if drop > 0 {
                    self.buf.drain(..drop);
                }
                return Ok(false);
            }
        };
        // Decode straight from the framed slice — `decode` returns an owned
        // `Decoded`, so no borrow of `self.buf` survives it — then consume through
        // the frame's end (dropping pre-sync junk). No per-frame heap copy.
        let decoded = self.dec.decode(&self.buf[offset..offset + len]);
        self.buf.drain(..offset + len);

        let fi = self.frame_index;
        self.frame_index += 1;
        match decoded {
            Ok(decoded) => {
                self.consecutive_errors = 0;
                if !self.announced {
                    self.rate = decoded.info.sample_rate;
                    self.channels = decoded.info.channels;
                    // Announce the runtime format. CHANNEL ORDER is
                    // L, R, C, LFE, Ls, Rs — the downmix contract (see module doc).
                    ctx.announce_format(
                        SRC_PAD,
                        FAMILY,
                        &[
                            (F_RATE, ValueDesc::Int(i64::from(self.rate))),
                            (F_CHANNELS, ValueDesc::Int(self.channels as i64)),
                            (F_SAMPLE, ValueDesc::Id(SAMPLE_S16)),
                        ],
                    );
                    self.announced = true;
                }
                if decoded.info.channels != self.channels || decoded.info.sample_rate != self.rate {
                    // Mid-stream geometry change: real streams never do this;
                    // mislabeling downstream corrupts the interleave — warn+drop.
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: format!(
                                "ac3dec: mid-stream format change {}ch@{}Hz → {}ch@{}Hz — frame {fi} dropped",
                                self.channels, self.rate, decoded.info.channels, decoded.info.sample_rate
                            ),
                        },
                    });
                    return Ok(true);
                }
                if decoded.info.used_approx_tool && !self.reported_approx {
                    self.reported_approx = true;
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: "ac3dec: stream uses an E-AC-3 advanced tool \
                                      (spectral extension / AHT / enhanced coupling); \
                                      decoded via the base tools (approximate in those bands)"
                                .to_string(),
                        },
                    });
                }
                self.pending.extend_from_slice(decoded.pcm);
                Ok(true)
            }
            Err(e) => {
                self.consecutive_errors = self.consecutive_errors.saturating_add(1);
                if self.consecutive_errors <= 3 {
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: format!("ac3dec: syncframe {fi} dropped: {e:?}"),
                        },
                    });
                }
                // Substitute one frame of silence at this frame's grid position so
                // the timeline stays intact (the aacdec lesson: a dropped frame
                // silently shifts all later audio). Only once announced (we know
                // the geometry); before that, just drop.
                if self.announced && self.channels > 0 {
                    self.pending.extend(std::iter::repeat_n(0i16, 1536 * self.channels));
                }
                Ok(true)
            }
        }
    }

    fn process(&mut self, ctx: &mut Ctx, inputs: &mut Inputs<'_>) -> Result<Flow, Error> {
        if !self.emit_pending(ctx, true)? {
            return Ok(Flow::Ok);
        }
        let mut emitted = 0u32;
        loop {
            match self.frame_and_decode_one(ctx)? {
                true => {
                    if !self.emit_pending(ctx, true)? {
                        return Ok(Flow::Ok); // pool filled — carry, yield
                    }
                    emitted += 1;
                    if emitted >= FRAMES_PER_BATCH {
                        return Ok(Flow::Ok); // bound the batch (latency)
                    }
                }
                false => match inputs.pop() {
                    Some(inbuf) => self.buf.extend_from_slice(inbuf.memory.data()),
                    None => return Ok(Flow::Ok),
                },
            }
        }
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FlushStart => {
                // Seek: drop buffered bytes + carry, reset the IMDCT overlap
                // history so decode re-syncs at the next syncframe. The announced
                // format is kept (geometry is stable across a stream).
                self.dec.reset();
                self.buf.clear();
                self.pending.clear();
                self.pending_pos = 0;
                self.consecutive_errors = 0;
            }
            Event::Eos => {
                // Flush the carry and every remaining whole frame through the
                // unbounded pool. A truncated trailing frame stays in `buf`.
                self.emit_pending(ctx, false)?;
                while self.frame_and_decode_one(ctx)? {
                    self.emit_pending(ctx, false)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// AC-3 (ATSC A/52 §5–§7) decoder element — sink family **`ac3`**.
pub struct Ac3Dec {
    core: Core,
}

impl Ac3Dec {
    pub fn new() -> Self {
        Self { core: Core::new() }
    }
}

impl Default for Ac3Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Ac3Dec {
    fn desc(&self) -> &'static ElementDesc {
        &AC3_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.core.reset_state();
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        self.core.process(ctx, &mut inputs)
    }
    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        self.core.event(ctx, event)
    }
    fn stop(&mut self, _ctx: &mut Ctx) {
        self.core.reset_state();
    }
}

/// E-AC-3 / Dolby Digital Plus (ATSC A/52 Annex E) decoder element — sink family
/// **`eac3`**. Shares the [`Frame`] core; the base decode covers a plain DD+ 5.1
/// stream, with SPX/AHT/enhanced-coupling reported-and-approximated.
pub struct Eac3Dec {
    core: Core,
}

impl Eac3Dec {
    pub fn new() -> Self {
        Self { core: Core::new() }
    }
}

impl Default for Eac3Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Eac3Dec {
    fn desc(&self) -> &'static ElementDesc {
        &EAC3_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.core.reset_state();
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        self.core.process(ctx, &mut inputs)
    }
    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        self.core.event(ctx, event)
    }
    fn stop(&mut self, _ctx: &mut Ctx) {
        self.core.reset_state();
    }
}
