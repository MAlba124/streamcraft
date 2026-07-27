//! [`AacDec`] — AAC audio decoder element over `oxideav-aac` (spec: Elements;
//! adoption rubric in `sc-vp8`). Mirrors [`Mp3Dec`]'s contract: compressed audio
//! in, **interleaved s16 `audio/raw`** out, format announced via dynamic caps the
//! moment the first frame decodes.
//!
//! ## Input contract (the container convention both our demuxers speak)
//! One **raw `raw_data_block()` access unit per buffer** (ISO/IEC 14496-3 §4.4.2.1
//! — no ADTS framing; RFC 9559 §12 and MP4 store AAC "as is"), preceded by the
//! **AudioSpecificConfig as the first buffer** (the in-band head / CodecPrivate
//! convention, exactly how `Mp4Demux` and `MkvDemux` deliver codec init data).
//! The ASC (ISO/IEC 14496-3 §1.6.2.1) configures object type, rate and channel
//! layout; each AU then decodes independently (overlap-add history lives in the
//! backing [`StreamDecoder`]).
//!
//! We drive `oxideav-aac`'s raw API (`StreamDecoder::decode_raw_data_block`)
//! rather than its `oxideav-core` `Decoder` trait: the trait's 0.1.6 `send_packet`
//! only accepts self-syncing ADTS/LOAS transports, which container AUs are not.
//!
//! ## Robustness (spec: Supervision — per-buffer error scope)
//! A malformed AU is warned (bus) and dropped; the decoder resyncs on the next AU
//! naturally (AAC frames are independent apart from the overlap-add tail). An
//! unparseable ASC is a loud per-element error — nothing downstream can be right.
//!
//! [`Mp3Dec`]: ../../sc_mp3/index.html

use oxideav_aac::asc::AudioSpecificConfig;
use oxideav_aac::decode::{PlanarFrame, StreamDecoder};
use oxideav_aac::pcm::interleave_s16_le_into;

use streamcraft_core::batch::Inputs;
use streamcraft_core::bus::BusMessage;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::log;
use streamcraft_core::log::Level;
use streamcraft_core::time::Timestamp;

const SRC_PAD: PadId = PadId(1);

// `audio/raw` vocabulary — literals, like every audio element (the pipeline
// interns by string, so ids line up with any audio peer).
const FAMILY: &str = "audio/raw";
const F_RATE: &str = "rate";
const F_CHANNELS: &str = "channels";
const F_SAMPLE: &str = "sample";
const SAMPLE_S16: &str = "s16";

static SAMPLE_VALUES: [ValueDesc; 1] = [ValueDesc::Id(SAMPLE_S16)];
static SRC_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_RATE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_CHANNELS, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_SAMPLE, allowed: ConstraintDesc::Set(&SAMPLE_VALUES), preferred: None },
];
static SRC_OFFERS: [OfferDesc; 2] =
    [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }, OfferDesc::any("bytes")];

// The sink declares the fields a demuxer's "aac" announcement carries
// (rate/channels), even though we read our config from the ASC — declared
// names are what get interned, and an announcement with un-interned names is
// silently dropped (the vocabulary gotcha; see memory/runtime-caps-gap).
static AAC_SINK_FIELDS: [FieldDesc; 2] = [
    FieldDesc { field: F_RATE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_CHANNELS, allowed: ConstraintDesc::Any, preferred: None },
];
static SINK_OFFERS: [OfferDesc; 2] =
    [OfferDesc { family: "aac", fields: &AAC_SINK_FIELDS }, OfferDesc::any("bytes")];

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
        dynamic: false,
        validate: None,
    },
];

static DESC: ElementDesc = ElementDesc {
    name: "aacdec",
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
    make_default: Some(|| Box::new(AacDec::new())),
};

/// The stream geometry learned from the AudioSpecificConfig — the arguments
/// `decode_raw_data_block` needs per AU.
#[derive(Clone, Copy)]
struct Cfg {
    aot: u8,
    fs_index: u8,
    sample_rate: u32,
    channel_configuration: u8,
}

/// A decoded frame waiting for a pool slot (the backpressure carry — spec: the
/// try_alloc + yield rule; a producer must never allocate unboundedly).
///
/// Carries the decoded **planar** `f64` channels rather than an interleaved `Vec<i16>`: the
/// §4.6.11 integer-PCM interleave is deferred to [`AacDec::emit_pending`], which renders it
/// straight into the pool slot (no intermediate `Vec<i16>`, no second copy — streamcraft patch).
struct PendingPcm {
    frame: PlanarFrame,
    pts: Timestamp,
}

/// Decodes raw AAC access units (ASC-configured) to interleaved s16 PCM.
pub struct AacDec {
    dec: StreamDecoder,
    cfg: Option<Cfg>,
    announced: bool,
    pending: Option<PendingPcm>,
    /// Consecutive decode failures, for capped warning spam (warn the first
    /// few, then go quiet until an AU succeeds again).
    consecutive_errors: u32,
    /// Access units consumed so far — stamps drop warnings so a failing AU can
    /// be located in the source stream (diagnostics).
    au_index: u64,
    /// The last successfully decoded frame's geometry, for silence substitution.
    last_geometry: Option<(usize, u32)>,
}

impl AacDec {
    pub fn new() -> Self {
        Self {
            dec: StreamDecoder::new(),
            cfg: None,
            announced: false,
            pending: None,
            consecutive_errors: 0,
            au_index: 0,
            last_geometry: None,
        }
    }

    /// Emit the carried frame if the pool allows. `false` = still carried.
    fn emit_pending(&mut self, ctx: &mut Ctx) -> bool {
        let Some(p) = self.pending.take() else { return true };
        let Some(mut buf) = ctx.try_alloc(SRC_PAD) else {
            self.pending = Some(p);
            return false;
        };
        let channels = &p.frame.channels;
        let channel_count = channels.len();
        let frame_len = channels.first().map_or(0, Vec::len);
        let total_samples = frame_len * channel_count;
        let need = total_samples * 2;
        if buf.memory.capacity() < need {
            // One AAC frame is ≤ ~24 KB (2048 samples × 6 ch × 2 B); any sane
            // slot holds it. Refuse loudly rather than truncate.
            let element = ctx.element();
            ctx.post(BusMessage::Warning {
                element,
                error: Error::Element {
                    element,
                    message: format!(
                        "aacdec: frame needs {need} bytes but pool slots hold {} — \
                         raise the pipeline pool slot size",
                        buf.memory.capacity()
                    ),
                },
            });
            return true; // dropped; keep flowing
        }
        if !self.announced {
            log!(
                &*ctx,
                Level::Debug,
                "announce",
                rate = p.frame.sample_rate,
                channels = channel_count,
            );
            // Announce from *decoded* geometry, not the ASC: SBR doubles the
            // rate relative to the core index (ISO/IEC 14496-3 §4.6.18), and the
            // decoder's output is what downstream actually receives.
            ctx.announce_format(
                SRC_PAD,
                FAMILY,
                &[
                    (F_RATE, ValueDesc::Int(i64::from(p.frame.sample_rate))),
                    (F_CHANNELS, ValueDesc::Int(channel_count as i64)),
                    (F_SAMPLE, ValueDesc::Id(SAMPLE_S16)),
                ],
            );
            self.announced = true;
        }
        // Render the §4.6.11 integer PCM straight into the pool slot — one fused
        // interleave + little-endian write, no intermediate `Vec<i16>` and no second copy.
        let dst = buf.memory.as_mut_full();
        match interleave_s16_le_into(channels, dst) {
            Ok(_) => {}
            Err(e) => {
                // The capacity was already checked above; a mismatch here means the decoder
                // produced ragged channel lengths — drop the frame rather than emit garbage.
                let element = ctx.element();
                ctx.post(BusMessage::Warning {
                    element,
                    error: Error::Element {
                        element,
                        message: format!("aacdec: interleave failed: {e:?}"),
                    },
                });
                return true;
            }
        }
        buf.memory.set_len(need);
        buf.pts = p.pts;
        // Frame duration in ns: samples-per-channel over the output rate.
        if channel_count > 0 && p.frame.sample_rate > 0 {
            buf.duration = Timestamp(
                (frame_len as u64).saturating_mul(1_000_000_000) / u64::from(p.frame.sample_rate),
            );
        }
        log!(&*ctx, Level::Trace, "frame", pts = buf.pts);
        ctx.out(SRC_PAD).push(buf);
        true
    }

    /// Parse the first buffer as the AudioSpecificConfig (ISO/IEC 14496-3
    /// §1.6.2.1) — the in-band head both demuxers deliver.
    fn configure(&mut self, ctx: &mut Ctx, data: &[u8]) -> Result<(), Error> {
        match AudioSpecificConfig::parse(data) {
            Ok((asc, _bits)) => {
                log!(
                    &*ctx,
                    Level::Debug,
                    "configured",
                    aot = asc.aot,
                    rate = asc.sample_rate,
                    channel_configuration = asc.channel_configuration,
                );
                self.cfg = Some(Cfg {
                    aot: asc.aot,
                    fs_index: asc.sampling_frequency_index,
                    sample_rate: asc.sample_rate,
                    channel_configuration: asc.channel_configuration,
                });
                Ok(())
            }
            Err(e) => {
                // Without a config nothing downstream can be right — loud error.
                let element = ctx.element();
                Err(Error::Element {
                    element,
                    message: format!("aacdec: AudioSpecificConfig parse failed: {e:?}"),
                })
            }
        }
    }
}

impl Default for AacDec {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for AacDec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        loop {
            // Backpressure first: a carried frame must land before more input is
            // consumed (spec: the try_alloc + yield rule).
            if !self.emit_pending(ctx) {
                return Ok(Flow::Ok);
            }
            let Some(inbuf) = inputs.pop() else { break };
            let data = inbuf.memory.data();
            let Some(cfg) = self.cfg else {
                // First buffer: the ASC head.
                self.configure(ctx, data)?;
                continue;
            };
            let au = self.au_index;
            self.au_index += 1;
            match self.dec.decode_raw_data_block_planar(
                cfg.aot,
                cfg.fs_index,
                cfg.sample_rate,
                cfg.channel_configuration,
                1, // container AUs carry one raw_data_block each
                data,
            ) {
                Ok(frame) => {
                    self.consecutive_errors = 0;
                    self.last_geometry = Some((frame.channels.len(), frame.sample_rate));
                    self.pending = Some(PendingPcm { frame, pts: inbuf.pts });
                }
                Err(e) => {
                    // Per-buffer error scope: warn (capped, AU-indexed for stream
                    // forensics), then **substitute one frame of silence** at the
                    // AU's pts so the timeline stays intact — a dropped frame
                    // would silently shift all later audio 1024 samples early
                    // (measured: 21 drops turned a whole movie track's SNR to
                    // noise purely by accumulated misalignment).
                    self.consecutive_errors = self.consecutive_errors.saturating_add(1);
                    if self.consecutive_errors <= 3 {
                        let element = ctx.element();
                        ctx.post(BusMessage::Warning {
                            element,
                            error: Error::Element {
                                element,
                                message: format!("aacdec: access unit {au} dropped: {e:?}"),
                            },
                        });
                    }
                    log!(&*ctx, Level::Debug, "au_dropped", au = au);
                    if let Some((channels, sample_rate)) = self.last_geometry {
                        // One frame of silence in planar form — the emit path interleaves it
                        // into the pool slot like any decoded frame (1024 samples/channel,
                        // matching the prior interleaved-silence length).
                        self.pending = Some(PendingPcm {
                            frame: PlanarFrame {
                                channels: vec![vec![0.0f64; 1024]; channels],
                                sample_rate,
                            },
                            pts: inbuf.pts,
                        });
                    }
                }
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Flush/seek: drop the carry and the overlap-add history — decode
            // resumes cleanly at any AU (spec: flush/seek). Config and the
            // announcement survive: the stream's geometry does not change.
            Event::FlushStart => {
                self.pending = None;
                self.dec = StreamDecoder::new();
                self.consecutive_errors = 0;
            }
            // EOS: flush the carried frame with the unbounded allocator — waiting
            // is no longer an option (the same rule every producer follows).
            Event::Eos => {
                if let Some(p) = self.pending.take() {
                    self.pending = Some(p);
                    // One retry via the normal path; if the pool is still dry the
                    // frame is dropped with the teardown (one frame, at EOS).
                    let _ = self.emit_pending(ctx);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-built LC/44.1kHz/stereo ASC: AOT 2 (5 bits), freq index 4
    /// (5 bits... 4 bits), channel config 2, GA bits zero — the canonical
    /// `0x12 0x10` pair every MP4 muxer emits for 44.1 stereo LC.
    #[test]
    fn asc_parses_canonical_lc() {
        let (asc, _) = AudioSpecificConfig::parse(&[0x12, 0x10]).expect("parse");
        assert_eq!(asc.aot, 2, "AAC-LC");
        assert_eq!(asc.sample_rate, 44_100);
        assert_eq!(asc.channel_configuration, 2);
    }
}
