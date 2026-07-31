//! Mid-file seeks through the **canonical-output chain** (`pf_play::chain`).
//!
//! `elementary_seek.rs` proves that `filesrc → decoder → sink` recovers from a seek. This file
//! proves the same for the chain a music player actually runs — `… → audioconvert → audiostereo
//! → audioresample → sink` — because that chain is longer, is split across scheduler groups
//! differently, and ends in an *injected* sink, which is the shape the engine's shared-`AudioOut`
//! producer has.
//!
//! The measure is deliberately **bytes at the sink after the flush**, never `position()`:
//! position is floored at the seek target, so it reports success the instant the flush lands
//! whether or not a single sample follows it.

// Tests own their fixtures: temp files, staging `Vec`s, `format!` in assertions.
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use pf_play::chain::ChainSpec;
use pf_play::{Player, SinkChoice, SourceSpec};
use profluens_audio::{AudioFormat, SampleFormat};
use profluens_core::time::Timestamp;

use capture::{CaptureSink, CaptureStats};

// ---------------------------------------------------------------------------------------
// A capture sink that also counts flushes and EOS — the two things a seek test must see.
// ---------------------------------------------------------------------------------------
mod capture {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use profluens_core::batch::Inputs;
    use profluens_core::ctx::Ctx;
    use profluens_core::element::{
        Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
    };
    use profluens_core::error::Error;
    use profluens_core::event::Event;
    use profluens_core::format::OfferDesc;
    use profluens_core::time::Timestamp;

    /// The app-side view of a [`CaptureSink`]. Cheap to clone; safe to poll while running.
    #[derive(Clone, Default)]
    pub struct CaptureStats {
        buffers: Arc<AtomicU64>,
        flushes: Arc<AtomicU64>,
        /// PCM received since the most recent flush — i.e. the *post-seek* audio, which is
        /// the only thing worth asserting content against.
        since_flush: Arc<Mutex<Vec<u8>>>,
    }

    impl CaptureStats {
        pub fn buffers(&self) -> u64 {
            self.buffers.load(Ordering::Acquire)
        }
        pub fn flushes(&self) -> u64 {
            self.flushes.load(Ordering::Acquire)
        }
        /// The audio delivered since the last flush, as canonical f32 samples.
        pub fn post_flush_f32(&self) -> Vec<f32> {
            self.since_flush
                .lock()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect()
        }
    }

    static OFFERS: [OfferDesc; 2] = [OfferDesc::any("audio/raw"), OfferDesc::any("bytes")];
    static PADS: [PadDesc; 1] = [PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    }];
    static DESC: ElementDesc = ElementDesc {
        name: "seekcapturesink",
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

    /// A sink that consumes at a fixed pace, so the file cannot run to EOS before the test has
    /// had a chance to seek it — the stand-in for a real device's backpressure.
    pub struct CaptureSink {
        stats: CaptureStats,
        pace: Duration,
    }

    impl CaptureSink {
        pub fn paced(stats: CaptureStats, pace: Duration) -> Self {
            Self { stats, pace }
        }
    }

    impl Element for CaptureSink {
        fn desc(&self) -> &'static ElementDesc {
            &DESC
        }
        fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
            Ok(())
        }
        fn stop(&mut self, _ctx: &mut Ctx) {}
        fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
            // A paced sink must bail the instant a seek is requested, exactly as
            // `pipewireaudiosink` does (it re-checks `ctx.seek_gen()` inside every slice of
            // its blocking ring push). Without this a batch worth seconds of audio keeps the
            // sink inside one `process()` call, and the scheduler cannot deliver the flush
            // until it returns.
            let start_gen = ctx.seek_gen();
            while let Some(buf) = inputs.pop() {
                if ctx.seek_gen() != start_gen {
                    return Ok(Flow::Ok); // seek: stop; the loop top flushes
                }
                self.stats.since_flush.lock().unwrap().extend_from_slice(buf.memory.data());
                self.stats.buffers.fetch_add(1, Ordering::Release);
                if !self.pace.is_zero() {
                    std::thread::sleep(self.pace);
                }
            }
            Ok(Flow::Ok)
        }
        fn event(&mut self, _ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
            if matches!(event, Event::FlushStart) {
                // Everything recorded so far is pre-seek; the seek target's audio starts
                // after this.
                self.stats.since_flush.lock().unwrap().clear();
                self.stats.flushes.fetch_add(1, Ordering::Release);
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------

fn tmp_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("canonical_seek");
    std::fs::create_dir_all(&d).expect("tmp dir");
    d
}

/// A `secs`-long sine WAV in `format`.
fn wav_fixture(name: &str, format: AudioFormat, secs: f64) -> PathBuf {
    let frames = (format.sample_rate as f64 * secs) as usize;
    let mut pcm = Vec::with_capacity(frames * format.frame_stride());
    for i in 0..frames {
        let t = i as f64 / format.sample_rate as f64;
        let v = (t * 440.0 * std::f64::consts::TAU).sin() * 0.5;
        for _ in 0..format.channels {
            match format.format {
                SampleFormat::S16 => pcm.extend_from_slice(&((v * 32767.0) as i16).to_le_bytes()),
                SampleFormat::F32 => pcm.extend_from_slice(&(v as f32).to_le_bytes()),
                _ => unreachable!("fixture formats"),
            }
        }
    }
    let bytes = profluens_audio::write_pcm_wav(&format, &pcm);
    let path = tmp_dir().join(name);
    std::fs::write(&path, &bytes).expect("write wav fixture");
    path
}

/// A `secs`-long 44.1 kHz stereo FLAC, encoded in-process.
fn flac_fixture(name: &str, secs: f64) -> PathBuf {
    use pf_flac::{FlacEncoder, SampleFormat as FlacFmt};
    const RATE: u32 = 44_100;
    let frames = (RATE as f64 * secs) as usize;
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

/// Poll `cond` up to ~8 s; `false` on timeout.
fn settle(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..4_000 {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

// ---------------------------------------------------------------------------------------
// The repro
// ---------------------------------------------------------------------------------------

/// The tone every fixture carries, so a test can say what the audio at a given instant
/// *should* be. 440 Hz, half scale — see `wav_fixture`.
const TONE_HZ: f64 = 440.0;

/// Drive `path` canonically onto a paced capture sink and seek to each of `targets` in turn.
/// Returns, per seek, the resolved landing time and the audio that followed it.
///
/// Deliberately measures **bytes at the sink after the flush**, never `position()`: position
/// is floored at the seek target, so it reports success the instant the flush lands whether or
/// not a single sample follows it. That is exactly how this defect hid.
fn seek_and_collect(path: &Path, targets: &[Duration]) -> Vec<(Timestamp, Vec<f32>)> {
    let stats = CaptureStats::default();
    let mut player = Player::open_canonical(
        SourceSpec::path(path.to_str().unwrap()),
        ChainSpec::injected(CaptureSink::paced(stats.clone(), Duration::from_millis(4))),
        SinkChoice::Drop,
    )
    .expect("open canonically");
    assert!(player.any_track_linked(), "the fixture must link");

    let duration = player.duration().expect("the fixture declares a duration");
    let resolved: Vec<(u64, Timestamp)> = targets
        .iter()
        .map(|to| {
            let t = Timestamp::from_nanos(to.as_nanos() as u64);
            player.seek_index().resolve(t, duration).expect("the index maps the time")
        })
        .collect();

    let seek = player.pipeline.seek_handle();
    let stop = player.pipeline.stop_handle();
    let run = std::thread::spawn(move || player.run());

    let mut out = Vec::new();
    for (byte, at) in resolved {
        assert!(
            settle(|| stats.buffers() > 8 || run.is_finished()),
            "no audio at all before the seek to {at:?}"
        );
        assert!(!run.is_finished(), "the stream ended before the seek to {at:?} could be issued");

        let flushes = stats.flushes();
        seek.seek(byte, at);
        assert!(
            settle(|| stats.flushes() > flushes || run.is_finished()),
            "the flush never reached the sink for the seek to {at:?}"
        );
        // The whole point: audio has to keep coming *after* the flush.
        let enough = 64 * 1024;
        assert!(
            settle(|| stats.post_flush_f32().len() * 4 > enough || run.is_finished()),
            "no audio followed the seek to {at:?} — the pipeline went quiet"
        );
        assert!(
            !run.is_finished(),
            "the pipeline concluded on the seek to {at:?} instead of resuming there"
        );
        out.push((at, stats.post_flush_f32()));
    }
    stop.stop();
    run.join().expect("join").expect("run must not error");
    out
}

/// Assert the audio delivered after a seek really is the audio *at the seek target* — the
/// fixture's 440 Hz tone with the phase the landing time implies, not silence and not the
/// start of the file. Skips the first few frames: the resampler's delay line is re-primed at
/// the flush, so its first output frames ramp.
fn assert_tone_at(landed: Timestamp, samples: &[f32], label: &str) {
    const SKIP_FRAMES: usize = 512;
    const CHECK_FRAMES: usize = 2048;
    let frames = samples.as_chunks::<2>().0; // canonical is stereo
    assert!(
        frames.len() > SKIP_FRAMES + CHECK_FRAMES,
        "{label}: only {} frames after the seek",
        frames.len()
    );
    let t0 = landed.nanos().expect("a resolved landing time") as f64 / 1e9;
    let mut worst = 0f64;
    for (i, f) in frames[SKIP_FRAMES..SKIP_FRAMES + CHECK_FRAMES].iter().enumerate() {
        let t = t0 + (SKIP_FRAMES + i) as f64 / 48_000.0;
        let want = (t * TONE_HZ * std::f64::consts::TAU).sin() * 0.5;
        worst = worst.max((f[0] as f64 - want).abs());
    }
    // Generous: s16 quantisation, the resampler, and a landing time that the index floors to
    // the preceding cue all move samples a little. A wrong *position* is off by ~the full
    // amplitude, so this separates "resumed at the target" from "resumed somewhere else".
    assert!(worst < 0.08, "{label}: post-seek audio is not the tone at {landed:?} (worst {worst:.3})");
}

#[test]
fn a_forward_seek_through_the_canonical_chain_keeps_producing_audio() {
    let path = wav_fixture("fwd48k.wav", AudioFormat::new(48_000, 2, SampleFormat::S16), 8.0);
    let got = seek_and_collect(&path, &[Duration::from_secs(4)]);
    assert_tone_at(got[0].0, &got[0].1, "forward seek");
}

#[test]
fn a_backward_seek_through_the_canonical_chain_keeps_producing_audio() {
    let path = wav_fixture("back48k.wav", AudioFormat::new(48_000, 2, SampleFormat::S16), 8.0);
    let got = seek_and_collect(&path, &[Duration::from_millis(400)]);
    assert_tone_at(got[0].0, &got[0].1, "backward seek");
}

#[test]
fn repeated_seeks_each_resume_at_their_own_target() {
    // One seek working is not the property; the group has to survive being revived over and
    // over, forwards and backwards, long after the source first reached end of file.
    let path = wav_fixture("many48k.wav", AudioFormat::new(48_000, 2, SampleFormat::S16), 8.0);
    let targets = [
        Duration::from_secs(5),
        Duration::from_millis(700),
        Duration::from_secs(6),
        Duration::from_secs(2),
    ];
    let got = seek_and_collect(&path, &targets);
    assert_eq!(got.len(), targets.len());
    for (landed, samples) in &got {
        assert_tone_at(*landed, samples, "repeated seek");
    }
}

#[test]
fn a_seek_through_the_canonical_chain_resumes_a_flac_too() {
    // A decoder (and a 44.1 kHz -> 48 kHz resample) in the chain rather than a raw parser, so
    // this covers the path a music library actually takes.
    let path = flac_fixture("seek44k.flac", 8.0);
    let got = seek_and_collect(&path, &[Duration::from_secs(4), Duration::from_secs(1)]);
    for (_, samples) in &got {
        assert!(samples.len() > 16_384, "the flac went quiet after a seek");
        let peak = samples.iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!(peak > 0.05, "post-seek flac audio is silence (peak {peak})");
    }
}
