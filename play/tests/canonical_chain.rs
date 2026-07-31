//! The canonical-output audio chain (`pf_play::chain`) — the per-track half of the gapless
//! design (`gapless.md` Phase 2).
//!
//! One claim is load-bearing and everything here exists to hold it up: **whatever the source
//! is, the sink is offered f32 / 48 000 Hz / 2 channels.** A shared `AudioOut` latches its
//! device format once for the life of the application so consecutive tracks can hand off inside
//! it; a chain that quietly converged on 44 100 Hz because the file happened to be a CD rip
//! would make track two unattachable, and the symptom would be silence, not a link error.
//!
//! So the assertions are on the **negotiated format at the sink**, read out of the sink's own
//! caps (`negotiated_audio_format`), not on a summary string or on the elements the builder
//! says it added. The sinks here are *injected* through
//! [`SinkSpec::Injected`](pf_play::chain::SinkSpec::Injected) — which means every test in this
//! file is also a test of the injectable-sink seam, since a capture sink is exactly the shape
//! the engine's `AudioOut` producer will be.
//!
//! Fixtures are synthesised in-process (`write_pcm_wav`, `pf_flac::FlacEncoder`) rather than
//! shelled out to ffmpeg, so nothing here skips and every expected sample count is exact.

// Tests own their fixtures: temp files, staging `Vec`s, format! in assertions.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::time::Duration;

use pf_play::chain::{
    self, gain_db_to_linear, wire_canonical_audio_chain, ChainHandles, ChainSpec, SinkSpec,
    CANONICAL,
};
use pf_play::{Player, SinkChoice};
use profluens_audio::{AudioFormat, SampleFormat};
use profluens_core::pipeline::Pipeline;

// ---------------------------------------------------------------------------------------
// A raw `audio/raw` source of an arbitrary runtime format.
//
// The workspace's existing test source (`audio/tests/audiogain.rs::raw_src`) pins its caps in
// a `static` descriptor, so it can only ever be one format. This chain has to be proved over a
// *table* of input shapes, so this one advertises the broad family and **announces** its
// concrete format at runtime — the dynamic-caps producer path every decoder uses, which is
// also the path the chain will really be fed from.
// ---------------------------------------------------------------------------------------
mod raw_src {
    use profluens_audio::{
        AudioFormat, FAMILY, FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE,
    };
    use profluens_core::batch::Inputs;
    use profluens_core::ctx::Ctx;
    use profluens_core::element::{
        Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
    };
    use profluens_core::error::Error;
    use profluens_core::event::Event;
    use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
    use profluens_core::id::PadId;
    use profluens_core::time::Timestamp;

    // Every sample-format name is offered so the announcement's `sample` id is interned.
    static SAMPLE_VALUES: [ValueDesc; 5] = [
        ValueDesc::Id("u8"),
        ValueDesc::Id("s16"),
        ValueDesc::Id("s24"),
        ValueDesc::Id("s32"),
        ValueDesc::Id("f32"),
    ];
    static FIELDS: [FieldDesc; 3] = [
        FieldDesc { field: FIELD_RATE, allowed: ConstraintDesc::Any, preferred: None },
        FieldDesc { field: FIELD_CHANNELS, allowed: ConstraintDesc::Any, preferred: None },
        FieldDesc {
            field: FIELD_SAMPLE,
            allowed: ConstraintDesc::Set(&SAMPLE_VALUES),
            preferred: None,
        },
    ];
    static OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &FIELDS }];
    static PADS: [PadDesc; 1] = [PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &OFFERS,
        dynamic: true, // the concrete format is announced at runtime
        validate: None,
    }];
    static DESC: ElementDesc = ElementDesc {
        name: "rawaudiosrc",
        pads: &PADS,
        props: &[],
        sched: SchedHint::Active,
        inputs: InputPolicy::None,
        latency: LatencyDesc {
            min: Timestamp::ZERO,
            max: Timestamp::ZERO,
            is_live: false,
            jitter: Timestamp::ZERO,
        },
        make_default: None,
    };

    pub struct RawAudioSrc {
        format: AudioFormat,
        pcm: Vec<u8>,
        off: usize,
        announced: bool,
    }

    impl RawAudioSrc {
        pub fn new(format: AudioFormat, pcm: Vec<u8>) -> Self {
            Self { format, pcm, off: 0, announced: false }
        }
    }

    impl Element for RawAudioSrc {
        fn desc(&self) -> &'static ElementDesc {
            &DESC
        }
        fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
            self.off = 0;
            self.announced = false;
            Ok(())
        }
        fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
            if !self.announced {
                ctx.announce_format(
                    PadId(0),
                    FAMILY,
                    &[
                        (FIELD_RATE, ValueDesc::Int(self.format.sample_rate as i64)),
                        (FIELD_CHANNELS, ValueDesc::Int(self.format.channels as i64)),
                        (FIELD_SAMPLE, ValueDesc::Id(self.format.format.caps_name())),
                    ],
                );
                self.announced = true;
            }
            if self.off >= self.pcm.len() {
                return Ok(Flow::Eos);
            }
            let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
            let stride = self.format.frame_stride();
            // Whole interchannel frames only, so the chain is never fed a ragged buffer it
            // then has to carry — that path has its own unit test.
            let cap = buf.memory.capacity() / stride * stride;
            let n = cap.min(self.pcm.len() - self.off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&self.pcm[self.off..self.off + n]);
            buf.memory.set_len(n);
            self.off += n;
            ctx.out(PadId(0)).push(buf);
            Ok(Flow::Ok)
        }
        fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
            Ok(())
        }
        fn stop(&mut self, _ctx: &mut Ctx) {}
    }
}

// ---------------------------------------------------------------------------------------
// The capture sink: what a headless `AudioOut` looks like from the chain's side.
//
// It records the **negotiated format on its own sink pad** — the only honest way to ask "what
// did the chain actually deliver" — alongside every PCM byte. `TestSink` cannot do this: it
// speaks `bytes` only, publishes a hash at `stop()`, and never sees caps at all.
// ---------------------------------------------------------------------------------------
mod capture {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use profluens_audio::{negotiated_audio_format, AudioFormat};
    use profluens_core::batch::Inputs;
    use profluens_core::ctx::Ctx;
    use profluens_core::element::{
        Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
    };
    use profluens_core::error::Error;
    use profluens_core::event::Event;
    use profluens_core::format::OfferDesc;
    use profluens_core::id::PadId;
    use profluens_core::time::Timestamp;

    #[derive(Default)]
    struct Recorded {
        pcm: Vec<u8>,
        format: Option<AudioFormat>,
    }

    /// The app-side view of a [`CaptureSink`]. Cheap to clone; safe to poll while running.
    #[derive(Clone, Default)]
    pub struct CaptureStats {
        inner: Arc<Mutex<Recorded>>,
        buffers: Arc<AtomicU64>,
        done: Arc<AtomicBool>,
    }

    impl CaptureStats {
        /// Every payload byte the sink received, in order.
        pub fn pcm(&self) -> Vec<u8> {
            self.inner.lock().unwrap().pcm.clone()
        }
        /// The `audio/raw` format negotiated on the sink pad — the chain's real output format.
        pub fn format(&self) -> Option<AudioFormat> {
            self.inner.lock().unwrap().format
        }
        /// Buffers received so far (live, for a mid-run trigger).
        pub fn buffers(&self) -> u64 {
            self.buffers.load(Ordering::Acquire)
        }
        pub fn is_done(&self) -> bool {
            self.done.load(Ordering::Acquire)
        }
        /// The received bytes read as `f32` samples (the canonical format).
        pub fn samples_f32(&self) -> Vec<f32> {
            self.pcm().as_chunks::<4>().0.iter().copied().map(f32::from_le_bytes).collect()
        }
    }

    static OFFERS: [OfferDesc; 2] = [OfferDesc::any("audio/raw"), OfferDesc::any("bytes")];

    macro_rules! sink_element {
        ($mod_name:ident, $ty:ident, $desc_name:literal, $pad:literal) => {
            /// A capture sink whose input pad is named `$pad`.
            pub mod $mod_name {
                use super::*;

                static PADS: [PadDesc; 1] = [PadDesc {
                    name: $pad,
                    direction: Direction::Sink,
                    offers: &OFFERS,
                    dynamic: false,
                    validate: None,
                }];
                static DESC: ElementDesc = ElementDesc {
                    name: $desc_name,
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
                    make_default: None,
                };

                pub struct $ty {
                    stats: CaptureStats,
                    /// Per-buffer sleep, so a test can act mid-stream. `ZERO` runs flat out.
                    pace: Duration,
                }

                impl $ty {
                    pub fn new(stats: CaptureStats) -> Self {
                        Self { stats, pace: Duration::ZERO }
                    }
                    // Only the `"sink"`-padded variant is ever paced; the macro generates both.
                    #[allow(dead_code)]
                    pub fn paced(stats: CaptureStats, pace: Duration) -> Self {
                        Self { stats, pace }
                    }
                }

                impl Element for $ty {
                    fn desc(&self) -> &'static ElementDesc {
                        &DESC
                    }
                    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
                        if let Some(f) = negotiated_audio_format(ctx, PadId(0)) {
                            self.stats.inner.lock().unwrap().format = Some(f);
                        }
                        Ok(())
                    }
                    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
                        // Re-read every pass: the chain announces its format at runtime, so the
                        // concrete caps land on this pad after `start()`.
                        if let Some(f) = negotiated_audio_format(ctx, PadId(0)) {
                            self.stats.inner.lock().unwrap().format = Some(f);
                        }
                        while let Some(buf) = inputs.pop() {
                            self.stats.inner.lock().unwrap().pcm.extend_from_slice(buf.memory.data());
                            self.stats.buffers.fetch_add(1, Ordering::Release);
                            if !self.pace.is_zero() {
                                std::thread::sleep(self.pace);
                            }
                        }
                        Ok(Flow::Ok)
                    }
                    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
                        match event {
                            Event::Eos => self.stats.done.store(true, Ordering::Release),
                            Event::FormatChange(_) => {
                                if let Some(f) = negotiated_audio_format(ctx, PadId(0)) {
                                    self.stats.inner.lock().unwrap().format = Some(f);
                                }
                            }
                            _ => {}
                        }
                        Ok(())
                    }
                    fn stop(&mut self, _ctx: &mut Ctx) {}
                }
            }
        };
    }

    // The ordinary sink, input pad `"sink"`.
    sink_element!(plain, CaptureSink, "capturesink", "sink");
    // A sink that calls its input something else — the chain must read the pad name off the
    // descriptor rather than assuming `"sink"`.
    sink_element!(oddly_named, OddlyNamedSink, "oddsink", "in");
}

use capture::oddly_named::OddlyNamedSink;
use capture::plain::CaptureSink;
use capture::CaptureStats;

// ---------------------------------------------------------------------------------------
// Fixtures + helpers
// ---------------------------------------------------------------------------------------

/// A constant-amplitude interleaved signal in `format`, `frames` long. Constant because the
/// gain assertions want an exact expected value at every sample, and because a constant
/// survives resampling and the stretcher's overlap-add unchanged — so any deviation is the
/// chain's doing, not the signal's.
fn constant_pcm(format: AudioFormat, frames: usize, level: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(frames * format.frame_stride());
    for _ in 0..frames {
        for ch in 0..format.channels as usize {
            // Give each channel its own level so a channel mix-up is visible, except in mono.
            let v = if format.channels > 1 && ch % 2 == 1 { level * 0.5 } else { level };
            push_sample(&mut out, format.format, v);
        }
    }
    out
}

/// A per-channel ramp: sample `i` of channel `c` is `(i, c)`-distinct, so a channel duplication
/// (mono→stereo) or a lane rotation is provable rather than plausible.
fn ramp_pcm_mono_f32(frames: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(frames * 4);
    for i in 0..frames {
        let v = (i as f32 / frames as f32) * 0.8 - 0.4;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Encode one sample of `fmt` from a float in [-1, 1).
fn push_sample(out: &mut Vec<u8>, fmt: SampleFormat, v: f32) {
    match fmt {
        SampleFormat::F32 => out.extend_from_slice(&v.to_le_bytes()),
        SampleFormat::S16 => out.extend_from_slice(&((v * 32767.0) as i16).to_le_bytes()),
        SampleFormat::S32 => out.extend_from_slice(&((v as f64 * 2147483647.0) as i32).to_le_bytes()),
        SampleFormat::S24 => {
            let s = (v as f64 * 8388607.0) as i32;
            out.extend_from_slice(&s.to_le_bytes()[..3]);
        }
        SampleFormat::U8 => out.push(((v * 127.0) as i32 + 128).clamp(0, 255) as u8),
    }
}

/// Build `rawaudiosrc → <canonical chain> → capture sink`, prerolled and ready to run.
fn build_chain(
    input: AudioFormat,
    pcm: Vec<u8>,
    gain_db: Option<f32>,
    stretch: Option<f32>,
    pace: Duration,
) -> (Pipeline, CaptureStats, ChainHandles) {
    let mut p = Pipeline::new();
    // Small slots: this is audio, and a chain stage that cannot fit one canonical frame in a
    // slot is a failure we want to see immediately, not paper over with megabytes.
    p.set_pool(16 * 1024, 32);
    let src = p.add(raw_src::RawAudioSrc::new(input, pcm));
    let stats = CaptureStats::default();
    let spec = ChainSpec {
        gain_db,
        stretch,
        sink: SinkSpec::Injected(Box::new(CaptureSink::paced(stats.clone(), pace))),
    };
    let handles =
        wire_canonical_audio_chain(&mut p, (src, "src"), spec).expect("the canonical chain wires");
    p.preroll().expect("preroll");
    (p, stats, handles)
}

/// Run a built chain to EOS and return what the sink captured.
fn run(mut p: Pipeline) {
    p.run().expect("the canonical chain must reach EOS cleanly");
}

/// Poll `cond` up to ~4 s (the workspace `settle` idiom); `false` on timeout.
fn settle(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..2000 {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

fn tmp_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("canonical_chain");
    std::fs::create_dir_all(&d).expect("tmp dir");
    d
}

// ---------------------------------------------------------------------------------------
// 1. The invariant: every input shape emerges as f32 / 48 kHz / 2 ch
// ---------------------------------------------------------------------------------------

#[test]
fn every_input_shape_emerges_as_f32_48k_stereo() {
    // The shapes a music library actually contains, plus the awkward ones: a 44.1 kHz mono
    // podcast, a 48 kHz stereo Opus track, a 96 kHz high-res FLAC, a 5.1 movie track, and an
    // 8-bit oddity. Each is the *whole* point of a separate conform stage.
    let cases: &[(&str, u32, u16, SampleFormat)] = &[
        ("44.1k mono s16 (podcast / mp3)", 44_100, 1, SampleFormat::S16),
        ("48k stereo s16 (opus)", 48_000, 2, SampleFormat::S16),
        ("44.1k stereo s16 (CD rip)", 44_100, 2, SampleFormat::S16),
        ("96k stereo s32 (high-res flac)", 96_000, 2, SampleFormat::S32),
        ("48k 5.1 s16 (ac3 movie)", 48_000, 6, SampleFormat::S16),
        ("22.05k mono u8 (the awkward one)", 22_050, 1, SampleFormat::U8),
        ("48k stereo f32 (already canonical)", 48_000, 2, SampleFormat::F32),
    ];
    for (name, rate, channels, fmt) in cases {
        let input = AudioFormat::new(*rate, *channels, *fmt);
        let frames = *rate as usize / 4; // a quarter second of audio
        let pcm = constant_pcm(input, frames, 0.5);
        let (p, stats, _h) = build_chain(input, pcm, None, None, Duration::ZERO);
        run(p);

        assert_eq!(
            stats.format(),
            Some(CANONICAL),
            "{name}: the sink negotiated {:?}, not the canonical {CANONICAL}",
            stats.format()
        );
        let bytes = stats.pcm().len();
        assert!(bytes > 0, "{name}: the chain delivered nothing");
        assert_eq!(
            bytes % CANONICAL.frame_stride(),
            0,
            "{name}: {bytes} bytes is not a whole number of stereo f32 frames"
        );

        // The frame count must track the rate conversion. The polyphase FIR does not flush its
        // tail at EOS (a documented follow-up in `audioresample`), so the output is short by
        // about a filter length — bounded here rather than ignored.
        let out_frames = bytes / CANONICAL.frame_stride();
        let want = frames as u64 * CANONICAL_RATE_U64 / *rate as u64;
        let slack = (want / 50).max(256); // 2 %, floor 256 frames
        assert!(
            out_frames as u64 + slack >= want && (out_frames as u64) <= want + slack,
            "{name}: {out_frames} frames out, expected ~{want} (±{slack})"
        );
    }
}

const CANONICAL_RATE_U64: u64 = chain::CANONICAL_RATE as u64;

// ---------------------------------------------------------------------------------------
// 2. The mono→stereo answer, proved sample for sample
// ---------------------------------------------------------------------------------------

#[test]
fn mono_is_upmixed_to_stereo_rather_than_left_as_one_channel() {
    // 48 kHz f32 in: the convert and the resample stages are both provable identities, so the
    // ONLY thing the chain does here is the channel conform. That makes the assertion exact —
    // every output frame must be the input sample, twice.
    //
    // This is the gap `audiodownmix` explicitly leaves ("mono is left as-is … duplicating it is
    // the converter's job") and `audioconvert`/`audioresample` do not fill (both preserve the
    // channel count). Without `audiostereo` this test sees one channel and a 4-byte stride.
    const FRAMES: usize = 4096;
    let input = AudioFormat::new(48_000, 1, SampleFormat::F32);
    let pcm = ramp_pcm_mono_f32(FRAMES);
    let want: Vec<f32> = pcm.as_chunks::<4>().0.iter().copied().map(f32::from_le_bytes).collect();

    let (p, stats, _h) = build_chain(input, pcm, None, None, Duration::ZERO);
    run(p);

    assert_eq!(stats.format(), Some(CANONICAL));
    let got = stats.samples_f32();
    assert_eq!(got.len(), FRAMES * 2, "mono in must become exactly 2 channels out");
    for (i, w) in want.iter().enumerate() {
        assert_eq!(got[i * 2], *w, "frame {i} left lane");
        assert_eq!(got[i * 2 + 1], *w, "frame {i} right lane must duplicate the mono sample");
    }
}

#[test]
fn a_five_point_one_track_is_folded_to_stereo() {
    // The other end of the conform: 6 channels in, 2 out, and the front pair is *not* simply
    // dropped — BS.775 sums the centre and surrounds in at −3 dB, so the fold is louder than
    // the bare front-left channel.
    const FRAMES: usize = 2048;
    let input = AudioFormat::new(48_000, 6, SampleFormat::F32);
    let mut pcm = Vec::new();
    for _ in 0..FRAMES {
        // L, R, C, LFE, Ls, Rs
        for v in [0.2f32, 0.2, 0.2, 0.9, 0.2, 0.2] {
            pcm.extend_from_slice(&v.to_le_bytes());
        }
    }
    let (p, stats, _h) = build_chain(input, pcm, None, None, Duration::ZERO);
    run(p);

    assert_eq!(stats.format(), Some(CANONICAL));
    let got = stats.samples_f32();
    assert_eq!(got.len(), FRAMES * 2, "6 channels in must become exactly 2 out");
    // L + .707·C + .707·Ls = 0.2 + 0.707·0.2 + 0.707·0.2 ≈ 0.4828. The LFE (0.9) is dropped
    // per BS.775, which is why the answer is well under 1.0.
    let want = 0.2 + std::f32::consts::FRAC_1_SQRT_2 * 0.4;
    for (i, s) in got.iter().enumerate().take(64) {
        assert!((s - want).abs() < 1e-4, "sample {i} = {s}, expected the BS.775 fold {want}");
    }
}

// ---------------------------------------------------------------------------------------
// 3. ReplayGain
// ---------------------------------------------------------------------------------------

#[test]
fn replaygain_decibels_scale_the_delivered_samples() {
    // −6.02 dB is a factor of one half. 48 kHz f32 stereo in, so convert/stereo/resample are
    // all identities and the only thing that touches the samples is the gain stage.
    const FRAMES: usize = 4096;
    const LEVEL: f32 = 0.5;
    let input = AudioFormat::new(48_000, 2, SampleFormat::F32);
    let pcm = constant_pcm(input, FRAMES, LEVEL);

    let (p, stats, h) = build_chain(input, pcm, Some(-6.0206), None, Duration::ZERO);
    let linear = h.gain_linear.expect("a gain stage was requested");
    assert!((linear - 0.5).abs() < 1e-4, "−6.02 dB is a linear half, got {linear}");
    assert!(h.gain.is_some(), "the gain element id must be exposed for live changes");
    run(p);

    let got = stats.samples_f32();
    assert!(!got.is_empty());
    // Channel 0 was written at LEVEL, channel 1 at LEVEL/2 (see `constant_pcm`), and the gain
    // halves both. The declick ramp is *snapped* at `start()`, so there is no fade-in to skip.
    for (i, s) in got.iter().enumerate().take(2048) {
        let want = if i % 2 == 0 { LEVEL * 0.5 } else { LEVEL * 0.25 };
        assert!((s - want).abs() < 1e-4, "sample {i} = {s}, expected {want}");
    }
}

#[test]
fn no_gain_stage_means_the_samples_are_untouched() {
    // The counterpart: `gain_db: None` must not insert an element at all, so an app cannot
    // silently get a volume knob it never asked for (and `set_gain` says so instead of
    // pretending to work).
    const FRAMES: usize = 1024;
    let input = AudioFormat::new(48_000, 2, SampleFormat::F32);
    let pcm = constant_pcm(input, FRAMES, 0.5);
    let (p, stats, h) = build_chain(input, pcm, None, None, Duration::ZERO);
    assert!(h.gain.is_none() && h.gain_linear.is_none());
    assert!(h.stretch.is_none() && h.stretch_position.is_none());
    let props = p.prop_handle();
    assert!(h.set_gain(&props, 0.5).is_err(), "no stage — setting gain must report it");
    assert!(h.set_rate(&props, 1.5).is_err(), "no stage — setting rate must report it");
    run(p);
    let got = stats.samples_f32();
    for (i, s) in got.iter().enumerate().take(512) {
        let want = if i % 2 == 0 { 0.5 } else { 0.25 };
        assert!((s - want).abs() < 1e-6, "sample {i} = {s}, expected an untouched {want}");
    }
}

// ---------------------------------------------------------------------------------------
// 4. Composition: stretch and gain in one chain
// ---------------------------------------------------------------------------------------

#[test]
fn stretch_and_gain_compose_in_one_chain() {
    // The podcast case and the music case at once: play 1.5× faster *and* apply a −6 dB track
    // correction. Both stages must be present, both must act, and the stretcher's source-time
    // handle must come back so a progress bar can follow the episode.
    const FRAMES: usize = 48_000; // one second
    const LEVEL: f32 = 0.5;
    let input = AudioFormat::new(48_000, 2, SampleFormat::F32);
    let pcm = constant_pcm(input, FRAMES, LEVEL);

    let (p, stats, h) = build_chain(input, pcm, Some(-6.0206), Some(1.5), Duration::ZERO);
    assert!(h.stretch.is_some(), "a stretch stage was requested");
    assert!(h.gain.is_some(), "a gain stage was requested");
    let pos = h.stretch_position.clone().expect("the stretcher's position handle");
    // Both optional stages plus the three mandatory ones and the sink.
    assert_eq!(h.stages().len(), 6, "convert, stereo, resample, stretch, gain, sink");
    run(p);

    assert_eq!(stats.format(), Some(CANONICAL), "the stretch/gain stages must not move the format");

    // 1.5× playback means ~2/3 of the frames come out.
    let out_frames = stats.pcm().len() / CANONICAL.frame_stride();
    let want = FRAMES * 2 / 3;
    assert!(
        out_frames > want * 85 / 100 && out_frames < want * 115 / 100,
        "{out_frames} frames out at rate 1.5, expected ~{want}"
    );

    // …and the gain still halved a constant signal (WSOLA cross-fades identical windows, so a
    // constant stays constant through the stretcher).
    let got = stats.samples_f32();
    let mid = (got.len() / 2) & !1; // an even index, i.e. a left sample, away from both edges
    for (i, s) in got.iter().enumerate().skip(mid).take(256) {
        let want = if i % 2 == 0 { LEVEL * 0.5 } else { LEVEL * 0.25 };
        assert!((s - want).abs() < 2e-2, "sample {i} = {s}, expected ~{want}");
    }

    // Source time advanced — the handle is live, not a stub.
    assert!(pos.position().nanos().unwrap_or(0) > 0, "the stretcher reported no source time");
    assert!(pos.source_frames() > 0 && pos.output_frames() > 0);
}

// ---------------------------------------------------------------------------------------
// 5. The injectable-sink seam
// ---------------------------------------------------------------------------------------

#[test]
fn an_injected_sink_whose_pad_is_not_called_sink_still_links() {
    // The chain reads the input pad name off the injected element's own descriptor. An
    // `AudioOut` producer is free to call its pad whatever it likes.
    const FRAMES: usize = 2048;
    let input = AudioFormat::new(44_100, 1, SampleFormat::S16);
    let pcm = constant_pcm(input, FRAMES, 0.5);

    let mut p = Pipeline::new();
    p.set_pool(16 * 1024, 32);
    let src = p.add(raw_src::RawAudioSrc::new(input, pcm));
    let stats = CaptureStats::default();
    let spec = ChainSpec::injected(OddlyNamedSink::new(stats.clone()));
    let h = wire_canonical_audio_chain(&mut p, (src, "src"), spec).expect("chain into 'in'");
    assert_eq!(h.sink_pad, "in", "the chain must use the declared pad name, not assume 'sink'");
    p.preroll().expect("preroll");
    run(p);

    assert_eq!(stats.format(), Some(CANONICAL));
    assert!(!stats.pcm().is_empty());
    assert!(stats.is_done(), "the injected sink saw EOS");
}

#[test]
fn the_chain_reports_the_sink_it_actually_wired() {
    const FRAMES: usize = 512;
    let input = AudioFormat::new(48_000, 2, SampleFormat::F32);
    let pcm = constant_pcm(input, FRAMES, 0.25);
    let (p, _stats, h) = build_chain(input, pcm, None, None, Duration::ZERO);
    // The three mandatory stages plus the injected sink, all distinct elements.
    let ids = [h.convert, h.stereo, h.resample, h.sink];
    for (i, a) in ids.iter().enumerate() {
        for b in ids.iter().skip(i + 1) {
            assert_ne!(a, b, "every chain stage is its own element");
        }
    }
    assert_eq!(h.sink_pad, "sink");
    run(p);
}

// ---------------------------------------------------------------------------------------
// 6. The live property path
// ---------------------------------------------------------------------------------------

#[test]
fn a_live_gain_change_reaches_the_element_through_the_prop_handle() {
    // The engine's volume slider: `ChainHandles::set_gain` → `PropHandle::set` → the element's
    // `PropChanged` at a batch boundary. Paced so the change lands mid-stream, then asserted on
    // the shape of the output rather than on timing: the head must be loud and some suffix
    // must be silent.
    const FRAMES: usize = 48_000;
    let input = AudioFormat::new(48_000, 2, SampleFormat::F32);
    let pcm = constant_pcm(input, FRAMES, 0.5);
    let (mut p, stats, h) =
        build_chain(input, pcm, Some(0.0), None, Duration::from_millis(2));
    let props = p.prop_handle();

    let watcher = stats.clone();
    let run_thread = std::thread::spawn(move || p.run());
    assert!(settle(|| watcher.buffers() >= 4), "the chain never delivered a buffer");
    h.set_gain(&props, 0.0).expect("live gain set");

    run_thread.join().expect("joined").expect("run ok");

    let got = stats.samples_f32();
    assert!(got.len() > 1000, "too little audio to judge");
    let head_peak = got[..256].iter().fold(0f32, |m, s| m.max(s.abs()));
    let tail_peak = got[got.len() - 256..].iter().fold(0f32, |m, s| m.max(s.abs()));
    assert!(head_peak > 0.2, "the head should be at the built gain (unity), peak {head_peak}");
    assert!(
        tail_peak < 1e-3,
        "the tail should be silenced by the live gain=0, peak {tail_peak}"
    );
}

#[test]
fn a_live_mute_reaches_the_element_the_same_way() {
    const FRAMES: usize = 48_000;
    let input = AudioFormat::new(48_000, 2, SampleFormat::F32);
    let pcm = constant_pcm(input, FRAMES, 0.5);
    let (mut p, stats, h) =
        build_chain(input, pcm, Some(0.0), None, Duration::from_millis(2));
    let props = p.prop_handle();

    let watcher = stats.clone();
    let run_thread = std::thread::spawn(move || p.run());
    assert!(settle(|| watcher.buffers() >= 4));
    h.set_mute(&props, true).expect("live mute set");
    run_thread.join().expect("joined").expect("run ok");

    let got = stats.samples_f32();
    let tail_peak = got[got.len() - 256..].iter().fold(0f32, |m, s| m.max(s.abs()));
    assert!(tail_peak < 1e-3, "mute must silence the tail, peak {tail_peak}");
}

// ---------------------------------------------------------------------------------------
// 7. Through the whole controller: `Player::open_canonical`
// ---------------------------------------------------------------------------------------

/// A WAV file of `format`, `frames` long, written to the target tmp dir.
fn wav_fixture(name: &str, format: AudioFormat, frames: usize) -> PathBuf {
    let pcm = constant_pcm(format, frames, 0.5);
    let bytes = profluens_audio::write_pcm_wav(&format, &pcm);
    let path = tmp_dir().join(name);
    std::fs::write(&path, &bytes).expect("write wav fixture");
    path
}

/// A 44.1 kHz **stereo** FLAC, encoded in-process by `pf-flac` (no external tool).
fn flac_fixture(name: &str, frames: usize) -> PathBuf {
    use pf_flac::{FlacEncoder, SampleFormat as FlacFmt};
    const RATE: u32 = 44_100;
    let (mut enc, header) = FlacEncoder::new(RATE, 2, FlacFmt::S16).expect("flac encoder");
    let pcm: Vec<u8> = (0..frames)
        .flat_map(|i| {
            let t = i as f64 / RATE as f64;
            let v = ((t * 440.0 * std::f64::consts::TAU).sin() * 12_000.0) as i16;
            [v.to_le_bytes(), (v / 2).to_le_bytes()]
        })
        .flatten()
        .collect();
    let mut frames_out = Vec::new();
    enc.encode_interleaved(&pcm, &mut frames_out).expect("encode");
    let body = enc.finish();
    let mut head = header.clone();
    let at = pf_flac::streaminfo_offset();
    head[at..at + body.len()].copy_from_slice(&body);
    head.extend_from_slice(&frames_out);
    let path = tmp_dir().join(name);
    std::fs::write(&path, &head).expect("write flac fixture");
    path
}

#[test]
fn open_canonical_conforms_a_mono_wav_end_to_end() {
    // Exercises the WAV branch specifically: `wavparse` emits untyped `bytes`, so the canonical
    // chain is fed by the header-pinned converter rather than by an announcing decoder. Mono
    // in — so this is also the mono→stereo path through the real controller.
    let path = wav_fixture("mono48k.wav", AudioFormat::new(48_000, 1, SampleFormat::S16), 24_000);
    let stats = CaptureStats::default();
    let player = Player::open_canonical(
        pf_play::SourceSpec::path(path.to_str().unwrap()),
        ChainSpec::injected(CaptureSink::new(stats.clone())),
        SinkChoice::Drop,
    )
    .expect("open the wav canonically");
    assert!(player.any_track_linked(), "the wav must link: {:?}", summaries(&player));
    assert!(player.audio_chain().is_some(), "a canonical open exposes its chain handles");
    let mut player = player;
    player.run().expect("play to EOS");

    assert_eq!(stats.format(), Some(CANONICAL), "a mono 48k WAV must arrive as f32/48k/2ch");
    assert!(!stats.pcm().is_empty());
}

#[test]
fn open_canonical_conforms_a_44k_flac_end_to_end() {
    // The decoder path: flacdec (44.1 kHz s16 stereo) → convert → stereo → resample(48k).
    let path = flac_fixture("cd44k.flac", 44_100);
    let stats = CaptureStats::default();
    let mut player = Player::open_canonical(
        pf_play::SourceSpec::path(path.to_str().unwrap()),
        ChainSpec::injected(CaptureSink::new(stats.clone())).with_gain_db(-3.0),
        SinkChoice::Drop,
    )
    .expect("open the flac canonically");
    assert!(player.any_track_linked(), "the flac must link: {:?}", summaries(&player));
    let h = player.audio_chain().expect("chain handles");
    assert!(h.gain.is_some(), "the requested gain stage is in the chain");
    player.run().expect("play to EOS");

    assert_eq!(stats.format(), Some(CANONICAL), "a 44.1k FLAC must arrive as f32/48k/2ch");
    let out_frames = stats.pcm().len() / CANONICAL.frame_stride();
    let want = 44_100u64 * CANONICAL_RATE_U64 / 44_100;
    assert!(
        (out_frames as u64) + 4096 >= want,
        "{out_frames} frames out, expected roughly {want}"
    );
}

#[test]
fn the_legacy_open_path_is_untouched_and_exposes_no_chain() {
    // The regression gate in miniature: `Player::open` still takes the tiered path, still
    // wires a drop sink for `SinkChoice::Drop`, and reports no canonical chain — so nothing
    // pfplay or the GUI does can accidentally pick up the new mode.
    let path = wav_fixture("legacy.wav", AudioFormat::new(44_100, 2, SampleFormat::S16), 8_000);
    let player = Player::open(
        path.to_str().unwrap(),
        pf_play::SinkPolicy { video: SinkChoice::Drop, audio: SinkChoice::Drop },
    )
    .expect("legacy open");
    assert!(player.any_track_linked());
    assert!(player.audio_chain().is_none(), "the tiered path builds no canonical chain");
    assert!(!player.is_growing(), "a plain path is not a growing source");
    let s = summaries(&player);
    assert!(
        s.iter().any(|t| t.contains("drop sink")),
        "the legacy drop wiring must be unchanged, got {s:?}"
    );
}

fn summaries(p: &Player) -> Vec<String> {
    p.tracks().iter().map(|t| t.summary.clone()).collect()
}

// ---------------------------------------------------------------------------------------
// 8. dB → linear, at the API surface
// ---------------------------------------------------------------------------------------

#[test]
fn the_documented_decibel_conversion_is_the_one_the_chain_applies() {
    // Also asserted as a unit test in `chain.rs`; repeated here against the *built* chain so
    // the number a caller reads back out of `ChainHandles` is the number that was applied.
    for (db, want) in [(0.0f32, 1.0f32), (-6.0206, 0.5), (6.0206, 2.0), (-20.0, 0.1)] {
        let input = AudioFormat::new(48_000, 2, SampleFormat::F32);
        let (p, _s, h) = build_chain(input, constant_pcm(input, 64, 0.1), Some(db), None, Duration::ZERO);
        let got = h.gain_linear.expect("gain stage");
        assert!(
            (got - want).abs() < 1e-3,
            "{db} dB built a gain of {got}, expected {want} (= {})",
            gain_db_to_linear(db)
        );
        drop(p);
    }
}
