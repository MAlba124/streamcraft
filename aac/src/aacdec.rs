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
use oxideav_aac::decode::StreamDecoder;
use oxideav_aac::pcm::{interleave_s16_into, s16_slice_le_into};

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

// COLD: `make_default` boxes one element instance at pipeline construction, never per frame.
#[allow(clippy::disallowed_methods)]
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

/// The geometry + timestamp of a decoded frame waiting for a pool slot (the backpressure carry —
/// spec: the try_alloc + yield rule; a producer must never allocate unboundedly).
///
/// The decoded samples themselves live in [`AacDec::pending_pcm`] (an owned, reused `Vec<i16>`),
/// interleaved to §4.6.11 integer PCM the moment the frame is decoded. The planar decode borrows
/// the per-`process()` arena (`ctx.scratch()`), so the decoded `PlanarFrame<&Arena>` cannot be
/// parked across `process()` calls — it is consumed into `pending_pcm` (Global) inside the
/// arena-borrow scope, and only this arena-free geometry is carried (streamcraft patch).
#[derive(Clone, Copy)]
struct PendingPcm {
    /// Output channel count of the parked frame.
    channels: usize,
    /// Per-channel sample count of the parked frame.
    frame_len: usize,
    /// Output sample rate of the parked frame (SBR-doubled when active).
    sample_rate: u32,
    pts: Timestamp,
}

/// Decodes raw AAC access units (ASC-configured) to interleaved s16 PCM.
pub struct AacDec {
    dec: StreamDecoder,
    cfg: Option<Cfg>,
    announced: bool,
    /// Geometry + pts of the carried frame; its samples are in [`Self::pending_pcm`]. `None`
    /// when nothing is carried.
    pending: Option<PendingPcm>,
    /// The carried frame's interleaved s16 PCM — an owned buffer reused across frames (allocates
    /// once when it first grows to the frame size, then only re-fills), so parking a frame past a
    /// `process()` boundary keeps no arena memory (spec: the try_alloc + yield rule).
    pending_pcm: Vec<i16>,
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
    // One-time element setup: `pending_pcm` is reused-and-cleared across frames, not
    // re-allocated per frame (spec: performance #1) — cold.
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        Self {
            dec: StreamDecoder::new(),
            cfg: None,
            announced: false,
            pending: None,
            pending_pcm: Vec::new(),
            consecutive_errors: 0,
            au_index: 0,
            last_geometry: None,
        }
    }

    /// Emit the carried frame if the pool allows. `false` = still carried.
    ///
    /// The frame's interleaved s16 samples already live in [`Self::pending_pcm`] (rendered when
    /// the frame decoded, inside the arena-borrow scope); this step copies them into the pool
    /// slot as little-endian bytes — the owned-buffer `try_alloc` + copy path (spec: backpressure
    /// carry). `pending` holds only the arena-free geometry + pts.
    fn emit_pending(&mut self, ctx: &mut Ctx) -> bool {
        let Some(p) = self.pending else { return true };
        let Some(mut buf) = ctx.try_alloc(SRC_PAD) else {
            return false;
        };
        let channel_count = p.channels;
        let frame_len = p.frame_len;
        let need = self.pending_pcm.len() * 2;
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
            // Consume the carry: dropped, keep flowing.
            self.pending = None;
            return true;
        }
        if !self.announced {
            log!(
                &*ctx,
                Level::Debug,
                "announce",
                rate = p.sample_rate,
                channels = channel_count,
            );
            // Announce from *decoded* geometry, not the ASC: SBR doubles the
            // rate relative to the core index (ISO/IEC 14496-3 §4.6.18), and the
            // decoder's output is what downstream actually receives.
            ctx.announce_format(
                SRC_PAD,
                FAMILY,
                &[
                    (F_RATE, ValueDesc::Int(i64::from(p.sample_rate))),
                    (F_CHANNELS, ValueDesc::Int(channel_count as i64)),
                    (F_SAMPLE, ValueDesc::Id(SAMPLE_S16)),
                ],
            );
            self.announced = true;
        }
        // Copy the parked interleaved PCM into the pool slot as little-endian bytes.
        let dst = buf.memory.as_mut_full();
        match s16_slice_le_into(&self.pending_pcm, dst) {
            Ok(_) => {}
            Err(e) => {
                // The capacity was already checked above; a mismatch here can only mean a
                // corrupted carry — drop the frame rather than emit garbage.
                let element = ctx.element();
                ctx.post(BusMessage::Warning {
                    element,
                    error: Error::Element {
                        element,
                        message: format!("aacdec: emit failed: {e:?}"),
                    },
                });
                self.pending = None;
                return true;
            }
        }
        buf.memory.set_len(need);
        buf.pts = p.pts;
        // Frame duration in ns: samples-per-channel over the output rate.
        if channel_count > 0 && p.sample_rate > 0 {
            buf.duration = Timestamp(
                (frame_len as u64).saturating_mul(1_000_000_000) / u64::from(p.sample_rate),
            );
        }
        log!(&*ctx, Level::Trace, "frame", pts = buf.pts);
        ctx.out(SRC_PAD).push(buf);
        self.pending = None;
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
            // Decode the block into the pipeline's per-`process()` arena (`ctx.scratch()`): the
            // whole planar decode — every transient AND the returned `PlanarFrame<&Arena>` PCM —
            // lives in that arena, which the scheduler resets after each `process()`, so the
            // decoder imposes no steady-state heap traffic. The arena-borrowed frame CANNOT be
            // parked across `process()` calls; inside this scope we interleave it into the owned
            // (Global, reused) `self.pending_pcm` and record the arena-free geometry, then drop
            // the frame so the `&ctx` borrow ends before the `&mut ctx` (`ctx.post`) error path.
            let outcome: Result<PendingPcm, oxideav_aac::Error> = {
                let scratch = ctx.scratch();
                match self.dec.decode_raw_data_block_planar(
                    cfg.aot,
                    cfg.fs_index,
                    cfg.sample_rate,
                    cfg.channel_configuration,
                    1, // container AUs carry one raw_data_block each
                    data,
                    scratch,
                ) {
                    Ok(frame) => {
                        let channels = frame.channels.len();
                        let frame_len = frame.channels.first().map_or(0, Vec::len);
                        // Interleave the arena-backed planar channels into the owned reused
                        // buffer here, while the arena is still live; a ragged-length frame is
                        // surfaced as a decode error (silence-substituted) rather than emitting
                        // garbage.
                        match interleave_s16_into(&frame.channels, &mut self.pending_pcm) {
                            Ok(()) => Ok(PendingPcm {
                                channels,
                                frame_len,
                                sample_rate: frame.sample_rate,
                                pts: inbuf.pts,
                            }),
                            Err(e) => Err(e),
                        }
                    }
                    Err(e) => Err(e),
                }
            };
            match outcome {
                Ok(p) => {
                    self.consecutive_errors = 0;
                    self.last_geometry = Some((p.channels, p.sample_rate));
                    self.pending = Some(p);
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
                        // One frame of silence interleaved into the reused buffer (1024
                        // samples/channel, matching a normal frame), parked like any decoded
                        // frame so the emit path copies it into the pool slot unchanged.
                        let frame_len = 1024usize;
                        self.pending_pcm.clear();
                        self.pending_pcm.resize(frame_len * channels, 0i16);
                        self.pending = Some(PendingPcm {
                            channels,
                            frame_len,
                            sample_rate,
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
            // EOS: try once more to land the carried frame — waiting is no longer an
            // option (the same rule every producer follows). One retry via the normal
            // path; if the pool is still dry the frame is dropped with the teardown
            // (one frame, at EOS).
            Event::Eos => {
                let _ = self.emit_pending(ctx);
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
