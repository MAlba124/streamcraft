//! Duration and time-seeking over the **elementary music formats** — a bare MP3, a bare
//! FLAC, and an Ogg-mapped stream — end to end through the controller that a player uses.
//!
//! These formats have no container, so nothing in the byte stream announces how long it is
//! or which byte three minutes in lives at; the answers are dug out of whatever the codec
//! left behind (an MP3's first frame, a FLAC's metadata chain, an Ogg stream's last granule)
//! by [`pf_play::head`] and [`pf_play::seek`] before the pipeline exists. Until that lands,
//! `Player::duration()` is `None`, `SeekIndex::resolve` therefore returns `None`, and a
//! player has no seek bar at all — which is what makes "did the duration come out right"
//! and "does the pipeline survive a seek" the two things worth testing here.
//!
//! ## What is asserted, and against what
//! - **Duration** against `ffprobe`, per format, with the tolerance each format's method
//!   earns: a Xing/LAME frame count and a FLAC `STREAMINFO` sample count are *exact* and are
//!   held to a millisecond; Opus differs from `ffprobe` by exactly its `pre_skip`, which is
//!   RFC 7845 §4's arithmetic and not an error (see [`opus_duration_subtracts_pre_skip`]).
//! - **The index**, by resolving through it and checking where it lands.
//! - **Recovery**, by seeking a running pipeline and watching the timestamps that come out
//!   the far side. The decoders resync on `FlushStart` (both re-sync to the next frame
//!   boundary) and re-base their sample counters to the seek target, so post-seek buffers
//!   carry post-seek times rather than continuing from where playback was interrupted.
//!
//! Fixtures that need `ffmpeg` skip themselves when it is absent; the FLAC one needs no
//! external tool at all, because `pf-flac`'s own encoder can build a stream whose exact
//! frame offsets are known — which is the only way to get a FLAC with a `SEEKTABLE` here,
//! ffmpeg's FLAC muxer writing STREAMINFO, comments and padding but no index.

#![allow(clippy::disallowed_methods)] // tests own their fixtures (spec: allocation discipline)

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use profluens_core::ctx::Ctx;
use profluens_core::batch::Inputs;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;
use profluens_elements::io::FileSrc;

use pf_play::{Kind, Player, SinkChoice, SinkPolicy};

// --- A sink that records what it is handed ---------------------------------------------
//
// The headless `SinkChoice::Drop` path uses `TestSink`, which publishes only a byte count
// and a hash, and only at `stop()` — nothing a running test can read, and no timestamps at
// all. Recording the PTS is the whole point here, so this is the "minimal harness" the
// drop sink cannot be: the same shape as `mkv/tests/seek.rs`'s `RecSink`, narrowed to the
// two families an audio decoder's src pad offers.

static SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("audio/raw"), OfferDesc::any("bytes")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "ptssink",
    pads: &SINK_PADS,
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

/// What the sink saw, in arrival order.
#[derive(Default)]
struct Recorded {
    /// One `(pts_ns, bytes)` per buffer. `None` pts means the decoder had not yet
    /// announced a rate — legitimate for the very first buffers of a stream.
    bufs: Vec<(Option<u64>, usize)>,
    /// How many `FlushStart` events arrived: the seek boundary, observed rather than
    /// guessed. Slicing the record by a buffer *count* would be racing the flush.
    flushes: usize,
    /// Index into `bufs` at each flush, so "the buffers after the seek" is exact.
    flush_at: Vec<usize>,
}

#[derive(Clone, Default)]
struct PtsStats(Arc<Mutex<Recorded>>, Arc<AtomicUsize>);

impl PtsStats {
    /// Buffers received so far — a release-stored count, so a test that observes it knows
    /// the buffers are in the record and not merely in flight.
    fn count(&self) -> usize {
        self.1.load(Ordering::Acquire)
    }
    fn flushes(&self) -> usize {
        self.0.lock().unwrap().flushes
    }
    /// The `(pts_ns, bytes)` pairs recorded after the last `FlushStart`.
    fn after_last_flush(&self) -> Vec<(Option<u64>, usize)> {
        let r = self.0.lock().unwrap();
        let at = r.flush_at.last().copied().unwrap_or(0);
        r.bufs[at.min(r.bufs.len())..].to_vec()
    }
}

/// The recording sink, paced.
///
/// `pace` is a deliberate per-buffer sleep, and it is not decoration: nothing else in this
/// graph runs in real time. A `filesrc → decoder → sink` chain with no device on the end
/// decodes a twelve-second song in a fraction of a second, so an unpaced pipeline can reach
/// EOS before the test has issued its second seek — which is not a bug being caught, just a
/// race being lost. Pacing keeps the stream alive long enough to seek it several times,
/// while staying far faster than playback.
struct PtsSink(PtsStats, Duration);

impl Element for PtsSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut received = 0;
        let mut r = self.0 .0.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            r.bufs.push((buf.pts.nanos(), buf.memory.data().len()));
            received += 1;
        }
        let n = r.bufs.len();
        drop(r);
        self.0 .1.store(n, Ordering::Release);
        if !self.1.is_zero() && received > 0 {
            std::thread::sleep(self.1 * received);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::FlushStart) {
            let mut r = self.0 .0.lock().unwrap();
            r.flushes += 1;
            let at = r.bufs.len();
            r.flush_at.push(at);
        }
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// --- External tools ---------------------------------------------------------------------

fn tool(name: &str) -> Option<PathBuf> {
    let out = Command::new("which").arg(name).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_string()))
}

/// `ffprobe`'s container duration in nanoseconds — the oracle every duration below is
/// checked against.
fn ffprobe_duration_ns(ffprobe: &Path, path: &Path) -> Option<u64> {
    let out = Command::new(ffprobe)
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let secs: f64 = s.trim().parse().ok()?;
    Some((secs * 1e9) as u64)
}

fn tmp_dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&d).expect("create tmp dir");
    d
}

/// A 12-second tone encoded however `args` says. Returns `None` when ffmpeg is absent.
fn ffmpeg_fixture(dir: &Path, name: &str, args: &[&str]) -> Option<PathBuf> {
    let ffmpeg = tool("ffmpeg")?;
    let path = dir.join(name);
    let st = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=12"])
        .args(args)
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(st.success(), "ffmpeg failed to build {name}");
    Some(path)
}

// --- A FLAC that actually has a SEEKTABLE ------------------------------------------------

/// Build a decodable FLAC stream carrying a real `SEEKTABLE` (RFC 9639 §8.5).
///
/// No external encoder writes one here — ffmpeg's FLAC muxer emits STREAMINFO, a comment
/// block and padding, and the reference `flac` tool is not installed — so this uses
/// `pf-flac`'s own encoder and encodes in fixed chunks. Encoding a chunk at a time is what
/// makes the table *correct* rather than plausible: the byte length of each chunk's frames
/// is known as it is produced, so every seek point's offset is measured, never estimated.
///
/// Returns the file bytes and the `(sample, byte offset from the first frame)` points that
/// went into the table, so the test can assert the index against the same numbers.
fn flac_with_seektable(chunk_frames: usize, chunks: usize) -> (Vec<u8>, Vec<(u64, u64)>) {
    use pf_flac::{FlacEncoder, SampleFormat};

    const RATE: u32 = 44_100;
    const CHANNELS: u32 = 2;

    let (mut enc, header) = FlacEncoder::new(RATE, CHANNELS, SampleFormat::S16).unwrap();
    let mut frames = Vec::new();
    let mut points: Vec<(u64, u64)> = Vec::new();
    let mut sample = 0u64;

    for c in 0..chunks {
        // Each seek point names the frame that *starts* the chunk: the sample it begins at
        // and the byte it begins at, both of which are exactly where the encoder is now.
        points.push((sample, frames.len() as u64));
        let pcm: Vec<u8> = (0..chunk_frames)
            .flat_map(|i| {
                // A cheap non-constant signal; constant input would encode to CONSTANT
                // subframes and make every frame the same size, hiding an offset bug.
                let t = (c * chunk_frames + i) as f64 / RATE as f64;
                let v = ((t * 440.0 * std::f64::consts::TAU).sin() * 12_000.0) as i16;
                [v.to_le_bytes(), (v / 2).to_le_bytes()]
            })
            .flatten()
            .collect();
        enc.encode_interleaved(&pcm, &mut frames).unwrap();
        sample += chunk_frames as u64;
    }

    // The finalised STREAMINFO body carries the true total sample count and frame sizes.
    let body = enc.finish();
    let mut head = header.clone();
    let at = pf_flac::streaminfo_offset();
    head[at..at + body.len()].copy_from_slice(&body);

    // The encoder's header ends at STREAMINFO with the last-block flag set (§8.1). Clear
    // it and append the SEEKTABLE as the new last block; the four bytes at offset 4 are
    // that block's `[last:1 | type:7]` header and its 3-byte length.
    assert_eq!(head[4] & 0x7F, 0, "STREAMINFO is the first metadata block");
    head[4] &= 0x7F;
    let mut table = Vec::new();
    for &(sample, offset) in &points {
        table.extend_from_slice(&sample.to_be_bytes());
        table.extend_from_slice(&offset.to_be_bytes());
        table.extend_from_slice(&(chunk_frames as u16).to_be_bytes());
    }
    // One placeholder point (§8.5) — reserved space, and not a seek point.
    table.extend_from_slice(&u64::MAX.to_be_bytes());
    table.extend_from_slice(&[0u8; 10]);

    head.push(0x80 | 3); // last block, type 3 = SEEKTABLE
    head.extend_from_slice(&(table.len() as u32).to_be_bytes()[1..]);
    head.extend_from_slice(&table);

    let audio_start = head.len() as u64;
    let mut file = head;
    file.extend_from_slice(&frames);
    // Report the points file-absolute-ready: offsets stay relative to the first frame,
    // which is what §8.5 defines and what the index adds `audio_start` to.
    let _ = audio_start;
    (file, points)
}

// --- Running a graph and seeking it ------------------------------------------------------

/// Drive `filesrc → dec → ptssink` over `path`, let it get going, then seek to each of
/// `targets` and return the sink's record. Bounded everywhere: no assertion here can hang
/// a test run.
fn run_and_seek(
    path: &Path,
    dec: Box<dyn Element>,
    index: &profluens_core::pipeline::SeekIndex,
    duration: Timestamp,
    targets: &[Timestamp],
) -> (PtsStats, Vec<(Timestamp, u64)>) {
    let stats = PtsStats::default();
    let mut p = Pipeline::new();
    // Small slots so the stream arrives as many buffers, which is what the pacing below
    // has to work with.
    p.set_pool(16 * 1024, 16);
    let src = p.add(FileSrc::new(path.to_str().unwrap()));
    let dec_id = p.add_boxed(dec);
    let snk = p.add(PtsSink(stats.clone(), Duration::from_millis(4)));
    p.link((src, "src"), (dec_id, "sink")).expect("filesrc ! dec");
    p.link((dec_id, "src"), (snk, "sink")).expect("dec ! ptssink");
    p.preroll().expect("preroll");

    let seek = p.seek_handle();
    let stop = p.stop_handle();
    let run = std::thread::spawn(move || p.run());

    let mut landed = Vec::new();
    for &target in targets {
        // Let some audio flow first, so a seek is genuinely interrupting playback.
        let before = stats.count();
        assert!(
            settle(|| stats.count() > before + 2 || run.is_finished()),
            "the pipeline produced no buffers before the seek"
        );
        assert!(
            !run.is_finished(),
            "the stream ended before the seek to {target:?} could be issued — pace the sink"
        );
        let flushes = stats.flushes();
        let (byte, at) = index.resolve(target, duration).expect("the index must map a time");
        landed.push((at, byte));
        seek.seek(byte, at);
        assert!(settle(|| stats.flushes() > flushes), "the flush never reached the sink");
        // …and let the post-seek audio arrive.
        let after = stats.count();
        assert!(
            settle(|| stats.count() > after + 2 || run.is_finished()),
            "no buffers after seeking to {at:?}"
        );
    }
    stop.stop();
    let _ = run.join().expect("the run thread joined");
    (stats, landed)
}

/// Poll until `cond` or ~4 s of real time — the repo's `settle` pattern, given a generous
/// budget because these graphs do real decoding on a loaded build machine.
fn settle(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..2_000 {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

/// Open through the real controller, headless.
fn open(path: &Path) -> Player {
    let policy = SinkPolicy { video: SinkChoice::Drop, audio: SinkChoice::Drop };
    Player::open(path.to_str().unwrap(), policy).expect("Player::open")
}

// --- Duration ----------------------------------------------------------------------------

#[test]
fn vbr_mp3_duration_is_lame_exact() {
    let dir = tmp_dir("elementary_seek");
    // `-q:a 2` is libmp3lame's VBR mode, which writes a Xing header with a frame count, a
    // 100-point TOC and the LAME extension stating the encoder delay and padding.
    let Some(path) = ffmpeg_fixture(&dir, "vbr.mp3", &["-c:a", "libmp3lame", "-q:a", "2"]) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let Some(ffprobe) = tool("ffprobe") else { return };
    let want = ffprobe_duration_ns(&ffprobe, &path).expect("ffprobe duration");

    let player = open(&path);
    assert_eq!(player.kind(), Kind::Mp3);
    let got = player.duration().expect("a VBR MP3 must report a duration").0;
    let delta = got.abs_diff(want);
    assert!(delta < 1_000_000, "duration {got} vs ffprobe {want}: {delta} ns apart");
    // The frame count is exact, so this is not a coarse match: the two agree to the sample.
    assert!(player.seek_index().entries.len() >= 2, "a Xing TOC must give an index");
}

#[test]
fn flac_duration_is_streaminfo_exact() {
    let dir = tmp_dir("elementary_seek");
    let Some(path) = ffmpeg_fixture(&dir, "tone.flac", &["-c:a", "flac"]) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let Some(ffprobe) = tool("ffprobe") else { return };
    let want = ffprobe_duration_ns(&ffprobe, &path).expect("ffprobe duration");

    let player = open(&path);
    assert_eq!(player.kind(), Kind::Flac);
    let got = player.duration().expect("a FLAC must report a duration").0;
    assert!(got.abs_diff(want) < 1_000_000, "duration {got} vs ffprobe {want}");
}

#[test]
fn opus_duration_subtracts_pre_skip() {
    let dir = tmp_dir("elementary_seek");
    let Some(path) = ffmpeg_fixture(&dir, "tone.opus", &["-c:a", "libopus", "-b:a", "96k"]) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let Some(ffprobe) = tool("ffprobe") else { return };
    let want = ffprobe_duration_ns(&ffprobe, &path).expect("ffprobe duration");

    let player = open(&path);
    assert_eq!(player.kind(), Kind::Ogg);
    let got = player.duration().expect("an Ogg Opus stream must report a duration").0;

    // RFC 7845 §4: "the pre-skip is subtracted from the granule position … to determine the
    // duration of the stream". `ffprobe`'s container duration does *not* subtract it, so the
    // two differ by exactly the pre-skip and this figure is the shorter one — deliberately.
    // The pre-skip is at most a couple of Opus frames, so bound the disagreement at 100 ms
    // and assert its sign, which is what tells a real error from this known offset.
    assert!(got <= want, "our duration {got} must not exceed ffprobe's {want}");
    let delta = want - got;
    assert!(delta < 100_000_000, "{delta} ns apart is more than a pre-skip");

    // The duration is the point: without it `resolve` cannot map a time at all, which is
    // exactly the state Ogg was in before.
    let d = Timestamp(got);
    assert!(player.seek_index().resolve(Timestamp(got / 2), d).is_some());
}

#[test]
fn adts_aac_honestly_reports_no_duration() {
    let dir = tmp_dir("elementary_seek");
    let Some(path) = ffmpeg_fixture(&dir, "tone.aac", &["-c:a", "aac", "-f", "adts"]) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let player = open(&path);
    assert_eq!(player.kind(), Kind::AdtsAac);
    // An ADTS stream states no duration anywhere; deriving one means walking every frame
    // header in the file, which is not paid for at open time. `None` is the honest answer,
    // and it keeps `resolve` returning `None` rather than a fabricated byte offset.
    assert_eq!(player.duration(), None);
    assert!(player
        .seek_index()
        .resolve(Timestamp(1_000_000_000), Timestamp::NONE)
        .is_none());
}

#[test]
fn wav_duration_comes_from_the_byte_rate_and_data_size() {
    let dir = tmp_dir("elementary_seek");
    let Some(path) = ffmpeg_fixture(&dir, "tone.wav", &["-c:a", "pcm_s16le"]) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let Some(ffprobe) = tool("ffprobe") else { return };
    let want = ffprobe_duration_ns(&ffprobe, &path).expect("ffprobe duration");

    let player = open(&path);
    assert_eq!(player.kind(), Kind::Wav);
    let got = player.duration().expect("a WAV must report a duration").0;
    assert!(got.abs_diff(want) < 1_000_000, "duration {got} vs ffprobe {want}");

    // PCM's map is exact, so every entry lands on a whole interchannel frame at the byte
    // its own timestamp says — checked here through the real header rather than a fixture.
    let d = Timestamp(got);
    let (byte, at) = player.seek_index().resolve(Timestamp(got / 2), d).expect("mapped");
    assert!(at.0.abs_diff(got / 2) < 200_000_000, "the grid is sub-second: {at:?}");
    assert!(byte > 0 && byte < std::fs::metadata(&path).unwrap().len());
}

// --- Seeking a running pipeline ----------------------------------------------------------

#[test]
fn flac_seektable_gives_an_exact_index_and_seeking_recovers() {
    // ~12 s at 44.1 kHz, a seek point every 8192 samples (~186 ms).
    const CHUNK: usize = 8_192;
    const CHUNKS: usize = 64;
    let (bytes, points) = flac_with_seektable(CHUNK, CHUNKS);
    let dir = tmp_dir("elementary_seek");
    let path = dir.join("seektable.flac");
    std::fs::write(&path, &bytes).expect("write the fixture");

    let player = open(&path);
    let duration = player.duration().expect("STREAMINFO states the sample count");
    let want = (CHUNK * CHUNKS) as u64 * 1_000_000_000 / 44_100;
    assert_eq!(duration.0, want, "an exact sample count, not an estimate");

    // Every real seek point became an entry; the placeholder did not.
    let index = player.seek_index().clone();
    assert_eq!(index.entries.len(), points.len(), "one entry per real seek point");
    // …and each entry is the point's own sample time at its own file offset. The first
    // entry pins the anchor: byte 0 of the table is the first *audio* byte, not byte 0 of
    // the file, and getting that wrong would shift every seek by the metadata's length.
    let audio_start = index.entries[0].1;
    assert!(audio_start > 0 && audio_start < bytes.len() as u64);
    for (i, &(sample, offset)) in points.iter().enumerate() {
        let (time, byte) = index.entries[i];
        assert_eq!(time, sample * 1_000_000_000 / 44_100, "entry {i} time");
        assert_eq!(byte, audio_start + offset, "entry {i} byte");
    }
    // The file really does start a FLAC frame there (§9.1's 14-bit sync code).
    for &(_, byte) in index.entries.iter() {
        let at = byte as usize;
        assert_eq!(bytes[at], 0xFF, "entry at {at} is not a frame start");
        assert_eq!(bytes[at + 1] & 0xFC, 0xF8, "entry at {at} is not a frame start");
    }

    // Resolving floors to the preceding point and reports where it landed.
    let (byte, at) = index.resolve(Timestamp(duration.0 / 2), duration).expect("mapped");
    assert!(at.0 <= duration.0 / 2 && duration.0 / 2 - at.0 < 200_000_000);
    assert!(index.entries.iter().any(|&(t, b)| (t, b) == (at.0, byte)), "landed on a point");

    // …and the graph survives it: seek to 25 %, 50 % and 90 % in turn, and every buffer
    // after the last flush carries a timestamp at or after where that seek landed.
    let targets: Vec<Timestamp> =
        [25u64, 50, 90].iter().map(|p| Timestamp(duration.0 * p / 100)).collect();
    let (stats, landed) = run_and_seek(
        &path,
        Box::new(pf_flac::FlacDec::new()),
        &index,
        duration,
        &targets,
    );
    assert_recovered(&stats, landed.last().unwrap().0, duration);
}

#[test]
fn vbr_mp3_toc_gives_an_index_and_seeking_recovers() {
    let dir = tmp_dir("elementary_seek");
    let Some(path) = ffmpeg_fixture(&dir, "vbr_seek.mp3", &["-c:a", "libmp3lame", "-q:a", "2"])
    else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let player = open(&path);
    let duration = player.duration().expect("a Xing frame count");
    let index = player.seek_index().clone();
    let file_len = std::fs::metadata(&path).unwrap().len();

    // A Xing TOC is a hundred points over the audio region, and the first of them is the
    // first audio frame — past the ID3v2 tag ffmpeg writes, which is the anchoring the
    // bare proportional fallback would have got wrong.
    assert_eq!(index.entries.len(), 100, "one entry per TOC point");
    assert!(index.entries[0].1 > 0, "the first entry is past the tag");
    assert!(index.entries.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1));
    assert!(index.entries.iter().all(|&(_, b)| b < file_len));

    // The TOC is a hundred points, so a landing is within ~1 % of the request; the decoder
    // resyncs to the next frame header from wherever it lands.
    let (_, at) = index.resolve(Timestamp(duration.0 / 2), duration).expect("mapped");
    let off = at.0.abs_diff(duration.0 / 2);
    assert!(off <= duration.0 / 100 + 1, "a TOC landing is within a percent: {off} ns");

    let targets: Vec<Timestamp> =
        [25u64, 50, 90].iter().map(|p| Timestamp(duration.0 * p / 100)).collect();
    let (stats, landed) =
        run_and_seek(&path, Box::new(pf_mp3::Mp3Dec::new()), &index, duration, &targets);
    assert_recovered(&stats, landed.last().unwrap().0, duration);
}

/// The shared post-seek assertion: the pipeline kept producing, and what it produced is
/// stamped at or after where the seek landed rather than continuing from where playback was
/// interrupted.
///
/// The tolerance is one decoded frame's worth (100 ms is generous for both codecs): a
/// decoder resyncs to the *next* frame header after the landing byte, so the first buffer
/// out can be a fraction of a frame later than the target — never earlier, which is the
/// direction that would mean the counter had not been re-based.
fn assert_recovered(stats: &PtsStats, landed: Timestamp, duration: Timestamp) {
    let post = stats.after_last_flush();
    assert!(!post.is_empty(), "no buffers after the final seek");
    assert!(post.iter().any(|&(_, n)| n > 0), "post-seek buffers are all empty");

    let stamped: Vec<u64> = post.iter().filter_map(|&(pts, _)| pts).collect();
    assert!(!stamped.is_empty(), "post-seek buffers carry no timestamps at all");
    let first = stamped[0];
    let floor = landed.0.saturating_sub(100_000_000);
    assert!(
        first >= floor,
        "post-seek pts {first} is before the seek landing {landed:?} — the decoder's sample \
         counter was not re-based to the seek target"
    );
    assert!(
        first <= duration.0 + 100_000_000,
        "post-seek pts {first} is past the end of a {duration:?} stream"
    );
    // …and time keeps moving forward from there.
    assert!(
        stamped.windows(2).all(|w| w[0] <= w[1]),
        "post-seek timestamps must be non-decreasing: {:?}",
        &stamped[..stamped.len().min(8)]
    );
}
