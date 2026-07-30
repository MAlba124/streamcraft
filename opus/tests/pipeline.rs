//! End-to-end element test: a 48 kHz `s16` stereo source → `opusenc` → `opusdec` → sink, run
//! through the real scheduler. This exercises what the encoder/element unit tests can't: the
//! `opusenc` sink learning its channel count from the negotiated `audio/raw` caps, announcing its
//! `opus` src caps to `opusdec`, reframing + emitting packets as pool buffers across the scheduler,
//! and EOS draining the tail — i.e. the harness glue, not just the codec.

#![allow(clippy::disallowed_methods)] // test fixtures, not a hot path

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;
use profluens_elements::testing::TestSink;

use pf_opus::{OpusDec, OpusEnc};

const RATE: i64 = 48_000;
const CHANNELS: usize = 2;
/// 25 CELT frames of 20 ms (960 samples/ch) — well past the codec startup.
const FRAMES: usize = 25;
const FRAME_SAMPLES: usize = 960;

// A source announcing `audio/raw` 48 kHz / s16 / stereo, emitting a stereo tone then EOS.
static SRC_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: "rate", allowed: ConstraintDesc::Set(&[ValueDesc::Int(RATE)]), preferred: None },
    FieldDesc { field: "channels", allowed: ConstraintDesc::Set(&[ValueDesc::Int(CHANNELS as i64)]), preferred: None },
    FieldDesc { field: "sample", allowed: ConstraintDesc::Set(&[ValueDesc::Id("s16")]), preferred: None },
];
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc { family: "audio/raw", fields: &SRC_FIELDS }];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &SRC_OFFERS,
    dynamic: true,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "tonesrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct ToneSrc {
    /// Interleaved s16 LE bytes still to emit.
    bytes: Vec<u8>,
    pos: usize,
    announced: bool,
}

impl ToneSrc {
    fn new() -> Self {
        let n = FRAMES * FRAME_SAMPLES;
        let mut bytes = Vec::with_capacity(n * CHANNELS * 2);
        for i in 0..n {
            let t = i as f64 / RATE as f64;
            let l = (0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 16384.0) as i16;
            let r = (0.3 * (2.0 * std::f64::consts::PI * 660.0 * t).sin() * 16384.0) as i16;
            bytes.extend_from_slice(&l.to_le_bytes());
            bytes.extend_from_slice(&r.to_le_bytes());
        }
        Self { bytes, pos: 0, announced: false }
    }
}

impl Element for ToneSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.pos = 0;
        self.announced = false;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.announced {
            ctx.announce_format(
                PadId(0),
                "audio/raw",
                &[
                    ("rate", ValueDesc::Int(RATE)),
                    ("channels", ValueDesc::Int(CHANNELS as i64)),
                    ("sample", ValueDesc::Id("s16")),
                ],
            );
            self.announced = true;
        }
        if self.pos >= self.bytes.len() {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let cap = buf.memory.capacity();
        let n = cap.min(self.bytes.len() - self.pos);
        buf.memory.as_mut_full()[..n].copy_from_slice(&self.bytes[self.pos..self.pos + n]);
        buf.memory.set_len(n);
        self.pos += n;
        ctx.out(PadId(0)).push(buf);
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// The full transcode round-trip through the scheduler: tone → opusenc → opusdec → sink. The sink
/// must see EOS and receive ~the input amount of decoded PCM (opusenc reframes 1:1 with the CELT
/// frame, opusdec restores 960 samples/channel per packet; no OpusHead means no pre-skip trim).
#[test]
fn tonesrc_opusenc_opusdec_roundtrips_through_scheduler() {
    let (sink, stats) = TestSink::new();
    let mut p = Pipeline::new();
    let src = p.add(ToneSrc::new());
    let enc = p.add(OpusEnc::new());
    let dec = p.add(OpusDec::new());
    let snk = p.add(sink);
    p.link((src, "src"), (enc, "sink")).expect("src->opusenc");
    p.link((enc, "src"), (dec, "sink")).expect("opusenc->opusdec");
    p.link((dec, "src"), (snk, "sink")).expect("opusdec->sink");
    p.run().expect("pipeline run");

    assert!(stats.is_done(), "sink saw EOS");
    let input_bytes = (FRAMES * FRAME_SAMPLES * CHANNELS * 2) as u64;
    let frame_bytes = (FRAME_SAMPLES * CHANNELS * 2) as u64;
    let out = stats.bytes();
    // Decoded PCM should match the input closely (whole frames, no pre-skip trim). Allow a small
    // margin for any codec-delay framing.
    assert!(
        out >= input_bytes * 9 / 10 && out <= input_bytes + frame_bytes,
        "decoded {out} bytes vs input {input_bytes} — reframing/emit through the scheduler is off",
    );
}
