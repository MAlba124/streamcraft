//! Opening and playing a file a **separate writer is still appending to** — the progressive
//! download (`pf_play::source`).
//!
//! The interesting failures here are all *timing* failures, so every test drives a real writer
//! thread that appends in chunks while the open and the run proceed. What is being proved:
//!
//! 1. the open returns as soon as the typefind prefix lands, not when the download finishes;
//! 2. no read crosses the frontier, so a parse never sees a torn append;
//! 3. the duration and the seek index are built against the caller's `total_hint`, not against
//!    however much happened to have downloaded at open time — the answer must not change as
//!    bytes arrive;
//! 4. playback runs to a clean EOS with the writer still working, and delivers the *whole*
//!    file, not the prefix that existed at open;
//! 5. a download that is abandoned (every `FrontierHandle` dropped without `finish`/`abort`)
//!    ends the stream instead of parking the pipeline on a watermark that will never move.
//!
//! (5) is why `Player` deliberately does not retain a handle of its own — see the note on
//! `Player::growing`.

// Tests own their fixtures: temp files, staging `Vec`s, format! in assertions.
#![allow(clippy::disallowed_methods)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use pf_play::chain::{ChainSpec, CANONICAL};
use pf_play::{Player, SinkChoice, SourceSpec};
use profluens_audio::{AudioFormat, SampleFormat};
use profluens_elements::io::FrontierHandle;

// ---------------------------------------------------------------------------------------
// A capture sink (see `canonical_chain.rs` for the reasoning — `TestSink` records no caps).
// ---------------------------------------------------------------------------------------
mod capture {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

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
        bytes: u64,
        format: Option<AudioFormat>,
    }

    #[derive(Clone, Default)]
    pub struct CaptureStats {
        inner: Arc<Mutex<Recorded>>,
        buffers: Arc<AtomicU64>,
        done: Arc<AtomicBool>,
    }

    impl CaptureStats {
        pub fn bytes(&self) -> u64 {
            self.inner.lock().unwrap().bytes
        }
        pub fn format(&self) -> Option<AudioFormat> {
            self.inner.lock().unwrap().format
        }
        pub fn buffers(&self) -> u64 {
            self.buffers.load(Ordering::Acquire)
        }
        pub fn is_done(&self) -> bool {
            self.done.load(Ordering::Acquire)
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
        name: "capturesink",
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

    pub struct CaptureSink(pub CaptureStats);

    impl Element for CaptureSink {
        fn desc(&self) -> &'static ElementDesc {
            &DESC
        }
        fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
            Ok(())
        }
        fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
            if let Some(f) = negotiated_audio_format(ctx, PadId(0)) {
                self.0.inner.lock().unwrap().format = Some(f);
            }
            while let Some(buf) = inputs.pop() {
                self.0.inner.lock().unwrap().bytes += buf.memory.data().len() as u64;
                self.0.buffers.fetch_add(1, Ordering::Release);
            }
            Ok(Flow::Ok)
        }
        fn event(&mut self, _ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
            if matches!(event, Event::Eos) {
                self.0.done.store(true, Ordering::Release);
            }
            Ok(())
        }
        fn stop(&mut self, _ctx: &mut Ctx) {}
    }
}

use capture::{CaptureSink, CaptureStats};

// ---------------------------------------------------------------------------------------
// The writer: a stand-in for a podcast downloader.
// ---------------------------------------------------------------------------------------

/// A background "download": appends `bytes` to `path` in `chunk`-sized pieces, publishing the
/// frontier after each write has actually landed, and `finish()`ing at the end.
///
/// The `advance`-after-`write_all` order is the [`FrontierHandle`] contract ("call `advance`
/// only *after* the corresponding write is visible to a reader in another thread") and is the
/// whole reason a reader can never observe a half-written append.
struct Writer {
    handle: std::thread::JoinHandle<()>,
    written: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
}

impl Writer {
    fn spawn(
        path: &Path,
        bytes: Vec<u8>,
        chunk: usize,
        delay: Duration,
        frontier: FrontierHandle,
        finish: bool,
    ) -> Writer {
        let path = path.to_path_buf();
        let written = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (w, s) = (Arc::clone(&written), Arc::clone(&stop));
        let handle = std::thread::spawn(move || {
            let mut f = std::fs::OpenOptions::new().append(true).open(&path).expect("open append");
            let mut at = 0usize;
            while at < bytes.len() {
                if s.load(Ordering::Acquire) {
                    return;
                }
                let n = chunk.min(bytes.len() - at);
                f.write_all(&bytes[at..at + n]).expect("append");
                f.flush().expect("flush");
                at += n;
                // Publish only what is durably readable — the frontier contract.
                w.store(at as u64, Ordering::Release);
                frontier.advance(at as u64);
                if !delay.is_zero() {
                    std::thread::sleep(delay);
                }
            }
            if finish {
                frontier.finish(bytes.len() as u64);
            }
        });
        Writer { handle, written, stop }
    }

    fn written(&self) -> u64 {
        self.written.load(Ordering::Acquire)
    }

    fn join(self) {
        self.stop.store(false, Ordering::Release);
        self.handle.join().expect("writer thread");
    }
}

// ---------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------

fn tmp_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("growing_source");
    std::fs::create_dir_all(&d).expect("tmp dir");
    d
}

/// A 48 kHz mono s16 WAV of `frames` frames, as bytes (not written anywhere yet).
fn wav_bytes(format: AudioFormat, frames: usize) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(frames * format.frame_stride());
    for i in 0..frames {
        // A slow ramp — non-constant, so a truncated or duplicated region is visible.
        let v = ((i % 1000) as f32 / 1000.0 - 0.5) * 0.6;
        for _ in 0..format.channels {
            pcm.extend_from_slice(&((v * 32767.0) as i16).to_le_bytes());
        }
    }
    profluens_audio::write_pcm_wav(&format, &pcm)
}

/// Create the (empty) file a growing source will read, as an app that pipes a download into a
/// pipeline does — `growingfilesrc` requires the path to exist, but not to have content.
fn empty_file(name: &str) -> PathBuf {
    let path = tmp_dir().join(name);
    std::fs::File::create(&path).expect("create the growing file");
    path
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

// ---------------------------------------------------------------------------------------
// 1. The open returns on the prefix, not on the download
// ---------------------------------------------------------------------------------------

#[test]
fn a_growing_file_opens_once_the_typefind_prefix_is_readable() {
    // 2 s of audio delivered in 4 KiB chunks at 1 ms each — the whole download takes seconds,
    // and the open must not wait for it. Typefind needs 64 bytes; the WAV header 44.
    let format = AudioFormat::new(48_000, 1, SampleFormat::S16);
    let bytes = wav_bytes(format, 96_000);
    let total = bytes.len() as u64;
    let path = empty_file("prefix.wav");
    let (source, frontier) = SourceSpec::growing(path.to_str().unwrap(), Some(total));
    let writer = Writer::spawn(&path, bytes, 4096, Duration::from_millis(1), frontier.clone(), true);

    let started = std::time::Instant::now();
    let stats = CaptureStats::default();
    let player = Player::open_canonical(
        source,
        ChainSpec::injected(CaptureSink(stats.clone())),
        SinkChoice::Drop,
    )
    .expect("open must succeed on the prefix alone");
    let open_took = started.elapsed();

    assert!(player.is_growing(), "the player knows it is reading a download");
    assert!(player.any_track_linked(), "the growing wav must link: {:?}", summaries(&player));
    // The open completed while the download was still going — which is the claim.
    assert!(
        writer.written() < total,
        "the download finished before the open did ({} of {total} bytes); \
         the test cannot prove anything about waiting",
        writer.written()
    );
    assert!(open_took < Duration::from_secs(5), "open took {open_took:?}");

    drop(player);
    drop(frontier);
    writer.join();
}

// ---------------------------------------------------------------------------------------
// 2. Duration and index come from the hint, not from the downloaded prefix
// ---------------------------------------------------------------------------------------

#[test]
fn the_duration_and_index_are_built_against_the_total_hint() {
    // Exactly 2 s of audio. Open when only a fraction has arrived and the reported duration
    // must still be 2 s — because a WAV states its length in the header (`data` chunk size),
    // which is in the first 44 bytes, and because the index length is the caller's hint.
    let format = AudioFormat::new(48_000, 1, SampleFormat::S16);
    let frames = 96_000usize;
    let bytes = wav_bytes(format, frames);
    let total = bytes.len() as u64;
    let path = empty_file("hint.wav");
    let (source, frontier) = SourceSpec::growing(path.to_str().unwrap(), Some(total));

    // Deliver a slice of the file up front, then stop — nothing more arrives during the open.
    {
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).expect("open append");
        let n = (total / 4) as usize;
        f.write_all(&bytes[..n]).expect("append");
        f.flush().expect("flush");
        frontier.advance(n as u64);
    }

    let stats = CaptureStats::default();
    let player = Player::open_canonical(
        source,
        ChainSpec::injected(CaptureSink(stats.clone())),
        SinkChoice::Drop,
    )
    .expect("open on a quarter-downloaded file");

    let dur = player.duration().and_then(|d| d.nanos()).expect("a duration from the WAV header");
    let want = frames as u64 * 1_000_000_000 / 48_000;
    assert_eq!(
        dur, want,
        "the duration must be the FILE's 2 s, not the downloaded quarter's 0.5 s"
    );
    assert_eq!(
        player.seek_index().file_len,
        Some(total),
        "the index must be proportional to the hinted final length"
    );

    drop(player);
    drop(frontier);
}

// ---------------------------------------------------------------------------------------
// 3. Playing while the writer appends
// ---------------------------------------------------------------------------------------

#[test]
fn a_growing_wav_plays_through_to_eos_while_the_writer_appends() {
    // The whole workflow: open on the prefix, play, and receive every frame of the finished
    // file — not the prefix that existed at open time.
    let format = AudioFormat::new(48_000, 1, SampleFormat::S16);
    let frames = 48_000usize; // one second
    let bytes = wav_bytes(format, frames);
    let total = bytes.len() as u64;
    let path = empty_file("play.wav");
    let (source, frontier) = SourceSpec::growing(path.to_str().unwrap(), Some(total));
    let writer =
        Writer::spawn(&path, bytes, 8192, Duration::from_millis(1), frontier.clone(), true);

    let stats = CaptureStats::default();
    let mut player = Player::open_canonical(
        source,
        ChainSpec::injected(CaptureSink(stats.clone())),
        SinkChoice::Drop,
    )
    .expect("open");
    player.run().expect("a growing source must reach EOS cleanly");

    assert!(stats.is_done(), "the sink never saw EOS");
    assert_eq!(
        stats.format(),
        Some(CANONICAL),
        "a growing source goes through the same canonical conform as any other"
    );
    // 48 kHz mono in → 48 kHz stereo out, no resampling, so the frame count is exact.
    let out_frames = stats.bytes() / CANONICAL.frame_stride() as u64;
    assert_eq!(
        out_frames, frames as u64,
        "the whole file must play, not the prefix that existed at open"
    );

    drop(frontier);
    writer.join();
}

#[test]
fn a_growing_source_survives_a_writer_slower_than_the_reader() {
    // The reader catching up with the writer is the *normal* state of a progressive download,
    // and `growingfilesrc` parks on it rather than treating an empty read as EOS. A 20 ms
    // per-chunk writer against a drop-speed reader guarantees the catch-up happens repeatedly.
    let format = AudioFormat::new(48_000, 1, SampleFormat::S16);
    let frames = 24_000usize;
    let bytes = wav_bytes(format, frames);
    let total = bytes.len() as u64;
    let path = empty_file("slow.wav");
    let (source, frontier) = SourceSpec::growing(path.to_str().unwrap(), Some(total));
    let writer =
        Writer::spawn(&path, bytes, 8192, Duration::from_millis(20), frontier.clone(), true);

    let stats = CaptureStats::default();
    let mut player = Player::open_canonical(
        source,
        ChainSpec::injected(CaptureSink(stats.clone())),
        SinkChoice::Drop,
    )
    .expect("open");
    player.run().expect("EOS despite the reader outrunning the writer");

    let out_frames = stats.bytes() / CANONICAL.frame_stride() as u64;
    assert_eq!(out_frames, frames as u64, "no frames lost to a frontier stall");

    drop(frontier);
    writer.join();
}

// ---------------------------------------------------------------------------------------
// 4. The abandoned download
// ---------------------------------------------------------------------------------------

#[test]
fn a_download_abandoned_mid_stream_ends_cleanly_instead_of_hanging() {
    // The downloader dies partway through: every `FrontierHandle` is dropped without
    // `finish()`/`abort()`. Without the element's abandonment check this parks forever, waiting
    // for a watermark nothing will ever move — and it is detected through the handle `Arc`'s
    // strong count, which is exactly why `Player` keeps a `bool` rather than a handle. A hidden
    // copy in there would pin the count at two and turn this test into a hang.
    let format = AudioFormat::new(48_000, 1, SampleFormat::S16);
    let bytes = wav_bytes(format, 48_000);
    let path = empty_file("abandoned.wav");
    let (source, frontier) = SourceSpec::growing(path.to_str().unwrap(), Some(bytes.len() as u64));

    // Deliver a usable prefix — enough to open, decode and play for a while.
    {
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).expect("open append");
        let n = bytes.len() / 3;
        f.write_all(&bytes[..n]).expect("append");
        f.flush().expect("flush");
        frontier.advance(n as u64);
    }

    let stats = CaptureStats::default();
    let player = Player::open_canonical(
        source,
        ChainSpec::injected(CaptureSink(stats.clone())),
        SinkChoice::Drop,
    )
    .expect("open");

    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let stop = player.pipeline.stop_handle();
    let mut player = player;
    let run = std::thread::spawn(move || {
        let r = player.run();
        flag.store(true, Ordering::Release);
        r
    });

    // Let the prefix play, *then* kill the downloader. (Dropping the handle before `run()`
    // instead is also handled — the element abandons on its first pass — but then nothing plays
    // at all, which is a less interesting thing to assert.)
    assert!(settle(|| stats.buffers() > 0), "the prefix never reached the sink");
    drop(frontier);

    // The element must wind down on its own, one 10 ms backstop tick after it notices. The stop
    // handle is the test's own safety net so a regression fails rather than hangs the suite.
    let ended = settle(|| done.load(Ordering::Acquire));
    if !ended {
        stop.stop();
    }
    run.join().expect("joined").expect("an abandoned download still ends cleanly");
    assert!(ended, "an abandoned download must end the stream, not park on the frontier");
    assert!(stats.bytes() > 0, "the prefix that did arrive should still have played");
}

// ---------------------------------------------------------------------------------------
// 5. A growing FLAC — the decoder path
// ---------------------------------------------------------------------------------------

#[test]
fn a_growing_flac_opens_and_decodes_while_downloading() {
    let (bytes, frames) = flac_bytes(44_100);
    let total = bytes.len() as u64;
    let path = empty_file("grow.flac");
    let (source, frontier) = SourceSpec::growing(path.to_str().unwrap(), Some(total));
    let writer =
        Writer::spawn(&path, bytes, 4096, Duration::from_millis(1), frontier.clone(), true);

    let stats = CaptureStats::default();
    let mut player = Player::open_canonical(
        source,
        ChainSpec::injected(CaptureSink(stats.clone())),
        SinkChoice::Drop,
    )
    .expect("open a downloading flac");
    assert!(player.any_track_linked(), "{:?}", summaries(&player));
    // STREAMINFO is in the first 42 bytes, so the duration is right from the first chunk.
    let dur = player.duration().and_then(|d| d.nanos()).expect("STREAMINFO duration");
    let want = frames as u64 * 1_000_000_000 / 44_100;
    assert!(
        dur.abs_diff(want) < 20_000_000,
        "duration {dur} ns, expected ~{want} ns from STREAMINFO"
    );
    player.run().expect("EOS");

    assert_eq!(stats.format(), Some(CANONICAL));
    assert!(stats.buffers() > 0, "the flac decoded nothing");

    drop(frontier);
    writer.join();
}

/// A 44.1 kHz stereo FLAC encoded in-process, plus its frame count.
fn flac_bytes(frames: usize) -> (Vec<u8>, usize) {
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
    let mut body_frames = Vec::new();
    enc.encode_interleaved(&pcm, &mut body_frames).expect("encode");
    let body = enc.finish();
    let mut out = header.clone();
    let at = pf_flac::streaminfo_offset();
    out[at..at + body.len()].copy_from_slice(&body);
    out.extend_from_slice(&body_frames);
    (out, frames)
}

fn summaries(p: &Player) -> Vec<String> {
    p.tracks().iter().map(|t| t.summary.clone()).collect()
}
