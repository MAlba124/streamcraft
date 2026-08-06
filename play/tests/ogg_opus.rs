//! Ogg-Opus (RFC 7845) end to end through the controller a player uses: the duration the
//! head/tail reads dig out, the chain `Player::build_ogg` wires, the audio that comes out of
//! it, what a seek does to it, and what a truncated file does *not* do to it.
//!
//! ## Why this file exists separately from `elementary_seek.rs`
//! That file already checks that an Ogg-Opus stream reports a *duration* — the number, from a
//! bos page and a tail read, with no pipeline involved. This one checks the other half: that
//! the number now belongs to something you can actually hear. Until the Opus arm landed,
//! `build_ogg` reported `(dropped: no decoder for Opus)` and every one of these files played
//! silence with a working seek bar over it.
//!
//! ## The three things worth asserting, and why they are hard
//! - **Pre-skip** (RFC 7845 §4.2): the first `pre_skip` samples are encoder priming and must
//!   never be emitted. Nothing in the *output* announces where the trim happened, so
//!   [`pre_skip_is_trimmed_at_stream_start`] does not try to detect it in a signal — it
//!   *changes* the number in the header and shows the output moves by exactly that much.
//! - **Timestamps**: Ogg carries none. RFC 3533 §4 is explicit that Ogg "has no concept of
//!   'time'", and a page granule is an end-of-page position, so `opusdec` owns the timeline
//!   and re-bases it to the seek target on `FlushStart` (as `flacdec`/`mp3dec` do). A seek
//!   that did not re-base would show up as post-seek buffers stamped with pre-seek times.
//! - **Hostile input**: a seek lands the source on an arbitrary byte, so mid-page landings are
//!   the *normal* case, not the corrupt one — RFC 3533 §6's capture pattern is what makes them
//!   recoverable. [`truncated_files_never_panic`] cuts the file everywhere and insists the
//!   pipeline stays boring.
//!
//! Fixtures come from `ffmpeg` at run time into `CARGO_TARGET_TMPDIR` (nothing binary is
//! committed); every test skips itself when `ffmpeg` is absent.

#![allow(clippy::disallowed_methods)] // tests own their fixtures (spec: allocation discipline)

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
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
use profluens_core::pipeline::{Pipeline, SeekIndex};
use profluens_core::time::Timestamp;
use profluens_elements::io::FileSrc;

use pf_play::{Kind, Player, SinkChoice, SinkPolicy};

/// Opus always decodes at 48 kHz (RFC 6716 §2), which is also the Ogg-Opus granule rate
/// (RFC 7845 §4), so one granule tick is one interchannel sample and the two never need
/// converting between.
const OPUS_RATE: u64 = 48_000;
/// Decoded output is interleaved `s16` — two bytes per single-channel sample.
const BYTES_PER_SAMPLE: usize = 2;

// --- A sink that records timestamps *and* PCM -------------------------------------------
//
// `TestSink` (the headless `SinkChoice::Drop` path) publishes a byte count and a hash, only
// at `stop()`. Two of the tests below need more than that: the seek test needs per-buffer
// timestamps and the flush boundary, and the pre-skip test needs the actual samples, since
// its whole argument is "these bytes are the other run's bytes, shifted". Same shape as
// `elementary_seek.rs`'s `PtsSink`, plus the payload.

static SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("audio/raw"), OfferDesc::any("bytes")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "pcmsink",
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

#[derive(Default)]
struct Recorded {
    /// One `(pts_ns, bytes)` per buffer, in arrival order.
    bufs: Vec<(Option<u64>, usize)>,
    /// Every PCM byte, concatenated — buffer boundaries are a pool artefact, not signal.
    pcm: Vec<u8>,
    /// How many `FlushStart` events arrived: the seek boundary, observed rather than guessed.
    flushes: usize,
    /// Index into `bufs` at each flush, so "the buffers after the seek" is exact.
    flush_at: Vec<usize>,
}

#[derive(Clone, Default)]
struct PcmStats(Arc<Mutex<Recorded>>, Arc<AtomicUsize>);

impl PcmStats {
    /// Buffers received so far — a release-stored count, so a test that observes it knows the
    /// buffers are in the record and not merely in flight.
    fn count(&self) -> usize {
        self.1.load(Ordering::Acquire)
    }
    fn flushes(&self) -> usize {
        self.0.lock().unwrap().flushes
    }
    fn pcm(&self) -> Vec<u8> {
        self.0.lock().unwrap().pcm.clone()
    }
    /// The `(pts_ns, bytes)` pairs recorded after the last `FlushStart`.
    fn after_last_flush(&self) -> Vec<(Option<u64>, usize)> {
        let r = self.0.lock().unwrap();
        let at = r.flush_at.last().copied().unwrap_or(0);
        r.bufs[at.min(r.bufs.len())..].to_vec()
    }
}

/// The recording sink. `pace` is a per-buffer sleep, and it is not decoration: nothing else
/// in this graph runs in real time, so an unpaced `filesrc → … → sink` chain reaches EOS long
/// before a test can issue its second seek. `Duration::ZERO` (the EOS runs) leaves it flat out.
struct PcmSink(PcmStats, Duration);

impl Element for PcmSink {
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
            let data = buf.memory.data();
            r.bufs.push((buf.pts.nanos(), data.len()));
            r.pcm.extend_from_slice(data);
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

// --- Fixtures ----------------------------------------------------------------------------

fn tool(name: &str) -> Option<PathBuf> {
    let out = Command::new("which").arg(name).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_string()))
}

fn tmp_dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&d).expect("create tmp dir");
    d
}

/// A `secs`-second 440 Hz tone as Ogg-Opus, built by `ffmpeg` into the test's tempdir (no
/// binary fixture is committed). `None` when `ffmpeg` is absent, which every caller treats as
/// "skip me". The source rate is 44.1 kHz deliberately: libopus resamples it to 48 kHz, which
/// is the resampled-input case a music library is actually full of.
fn opus_fixture(dir: &Path, name: &str, secs: u32) -> Option<PathBuf> {
    let ffmpeg = tool("ffmpeg")?;
    let path = dir.join(name);
    let st = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", &format!("sine=frequency=440:sample_rate=44100:duration={secs}")])
        .args(["-c:a", "libopus", "-b:a", "96k"])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(st.success(), "ffmpeg failed to build {name}");
    Some(path)
}

/// `ffprobe`'s container duration in nanoseconds — the external oracle.
fn ffprobe_duration_ns(ffprobe: &Path, path: &Path) -> Option<u64> {
    let out = Command::new(ffprobe)
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .ok()?;
    let secs: f64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some((secs * 1e9) as u64)
}

/// The stream's `(pre_skip, channels)` from its bos page, and the last granule position from
/// its tail — the three numbers RFC 7845 §4's duration arithmetic is made of, read here
/// independently of `pf_play::seek` so the duration test has something to check *against*
/// rather than a restatement of the code under test.
fn head_and_granule(bytes: &[u8]) -> (u16, u8, u64) {
    let bos = pf_ogg::PageHeader::parse(bytes).expect("a bos page at byte 0");
    let head = pf_ogg::parse_opus_head(bos.payload()).expect("an OpusHead identification packet");
    let last = pf_ogg::last_granule(bytes, Some(bos.serial())).expect("a granule in the tail");
    (head.pre_skip, head.channels, last)
}

/// Rewrite the bos page's `OpusHead` pre-skip field (RFC 7845 §5.1: a 16-bit little-endian
/// value at offset 10 of the identification packet) and re-stamp the page CRC (RFC 3533 §6,
/// field 7 — computed over the page with the CRC field zeroed, which is exactly what
/// [`pf_ogg::page_crc`] does). Everything else, audio included, is untouched: the two files
/// differ in precisely one number, which is what makes the comparison in
/// [`pre_skip_is_trimmed_at_stream_start`] an experiment rather than an estimate.
fn with_pre_skip(bytes: &[u8], pre_skip: u16) -> Vec<u8> {
    let bos = pf_ogg::PageHeader::parse(bytes).expect("a bos page at byte 0");
    let page_len = bos.len();
    // A page is the 27-byte fixed header (the segment count is its last byte), then the
    // segment table, then the payload (§6, fields 1-10). The bos page of an Ogg-Opus stream
    // carries exactly one packet — the `OpusHead` — so the payload starts the header.
    let payload_at = pf_ogg::HEADER_FIXED_LEN + bos.segment_table().len();
    let mut out = bytes.to_vec();
    out[payload_at + 10..payload_at + 12].copy_from_slice(&pre_skip.to_le_bytes());
    let crc = pf_ogg::page_crc(&out[..page_len]);
    out[22..26].copy_from_slice(&crc.to_le_bytes());
    // The patch must survive the same parser the pipeline uses, or the test proves nothing.
    let check = pf_ogg::PageHeader::parse(&out).expect("the patched bos page must still verify");
    assert_eq!(
        pf_ogg::parse_opus_head(check.payload()).expect("OpusHead").pre_skip,
        pre_skip,
    );
    out
}

// --- Running the chain -------------------------------------------------------------------

/// The chain under test, exactly as `Player::build_ogg` wires it minus the sink tail:
/// `filesrc → oggdemux → opusdec → <recording sink>`. Small pool slots so the stream arrives
/// as many buffers, which is what the pacing has to work with.
fn build(path: &Path, pace: Duration) -> (Pipeline, PcmStats) {
    let stats = PcmStats::default();
    let mut p = Pipeline::new();
    p.set_pool(16 * 1024, 16);
    let src = p.add(FileSrc::new(path.to_str().unwrap()));
    let demux = p.add(pf_ogg::OggDemux::new());
    let dec = p.add(pf_opus::OpusDec::new());
    let snk = p.add(PcmSink(stats.clone(), pace));
    p.link((src, "src"), (demux, "sink")).expect("filesrc ! oggdemux");
    p.link((demux, "src"), (dec, "sink")).expect("oggdemux ! opusdec");
    p.link((dec, "src"), (snk, "sink")).expect("opusdec ! pcmsink");
    // Mirror the real player: the decoder gets its own pool so decoded PCM does not compete
    // with the container bytes for the shared slots.
    p.set_element_pool(dec, 16 * 1024, 32);
    p.preroll().expect("preroll");
    (p, stats)
}

/// Run the chain to its natural end and return everything the sink saw.
fn run_to_eos(path: &Path) -> (PcmStats, Result<(), Error>) {
    let (mut p, stats) = build(path, Duration::ZERO);
    let r = p.run();
    (stats, r)
}

/// Drive the chain, then seek to each of `targets` in turn, returning the record and where
/// each seek landed. Bounded everywhere: nothing here can hang a test run.
fn run_and_seek(
    path: &Path,
    index: &SeekIndex,
    duration: Timestamp,
    targets: &[Timestamp],
) -> (PcmStats, Vec<(Timestamp, u64)>) {
    let (mut p, stats) = build(path, Duration::from_millis(4));
    let seek = p.seek_handle();
    let stop = p.stop_handle();
    let run = std::thread::spawn(move || p.run());

    let mut landed = Vec::new();
    for &target in targets {
        // Let some audio flow first, so a seek genuinely interrupts playback.
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

/// Poll until `cond` or ~4 s of real time — the repo's `settle` pattern.
fn settle(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..2_000 {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

fn open(path: &Path) -> Player {
    let policy = SinkPolicy { video: SinkChoice::Drop, audio: SinkChoice::Drop };
    Player::open(path.to_str().unwrap(), policy).expect("Player::open")
}

// --- Duration ------------------------------------------------------------------------------

#[test]
fn duration_is_the_last_granule_minus_pre_skip() {
    let dir = tmp_dir("ogg_opus");
    let Some(path) = opus_fixture(&dir, "duration.opus", 5) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let bytes = std::fs::read(&path).expect("read the fixture");
    let (pre_skip, _channels, last_granule) = head_and_granule(&bytes);

    let player = open(&path);
    assert_eq!(player.kind(), Kind::Ogg);
    let got = player.duration().expect("an Ogg-Opus stream must report a duration").0;

    // RFC 7845 §4: "the pre-skip is subtracted from the granule position … to determine the
    // duration". Granules tick at 48 kHz whatever the *input* rate was (§4), so this is exact
    // integer arithmetic on the same grid — not a tolerance.
    let want = (last_granule - pre_skip as u64) * 1_000_000_000 / OPUS_RATE;
    assert_eq!(got, want, "duration must be (granule {last_granule} − pre_skip {pre_skip}) @48k");
    assert!(pre_skip > 0, "libopus always primes the encoder — a zero pre_skip means the header was misread");

    // …and the same figure, checked against an outside opinion. `ffprobe` does *not* subtract
    // the pre-skip, so ours is the shorter one by exactly that much — the sign is the assertion.
    if let Some(ffprobe) = tool("ffprobe") {
        let want_ff = ffprobe_duration_ns(&ffprobe, &path).expect("ffprobe duration");
        assert!(got <= want_ff, "our duration {got} must not exceed ffprobe's {want_ff}");
        assert!(want_ff - got < 100_000_000, "{} ns apart is more than a pre-skip", want_ff - got);
    }
}

// --- Wiring ---------------------------------------------------------------------------------

#[test]
fn autoplug_wires_opus_instead_of_dropping_it() {
    let dir = tmp_dir("ogg_opus");
    let Some(path) = opus_fixture(&dir, "wiring.opus", 2) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let player = open(&path);
    assert_eq!(player.kind(), Kind::Ogg);
    let tracks = player.tracks();
    let summaries: Vec<&str> = tracks.iter().map(|t| t.summary.as_str()).collect();
    assert_eq!(tracks.len(), 1, "one logical bitstream, one track: {summaries:?}");
    let t = &tracks[0];
    assert_eq!(t.pad, "ogg/opus");
    assert!(t.linked, "the Opus track must link: {}", t.summary);
    // The regression this whole file exists for: `(dropped: no decoder for Opus)`.
    assert!(!t.summary.contains("dropped"), "summary still reports a drop: {}", t.summary);
    assert!(t.summary.contains("opusdec"), "summary should name the decoder: {}", t.summary);
    assert!(player.any_track_linked());
}

// --- Playback --------------------------------------------------------------------------------

#[test]
fn plays_through_to_eos() {
    let dir = tmp_dir("ogg_opus");
    let Some(path) = opus_fixture(&dir, "eos.opus", 5) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let bytes = std::fs::read(&path).expect("read the fixture");
    let (pre_skip, channels, last_granule) = head_and_granule(&bytes);

    let (stats, result) = run_to_eos(&path);
    result.expect("the chain must reach EOS cleanly");

    let pcm = stats.pcm();
    assert!(!pcm.is_empty(), "the chain reached EOS having decoded nothing");
    let frames = pcm.len() / (BYTES_PER_SAMPLE * channels as usize);

    // The output length is the RFC 7845 §4 duration — the last granule less the pre-skip — give
    // or take the last packet: `opusdec` emits whole packets and does not apply the granule's
    // end-trim (§4.4), while the encoder's final packet may itself be short of a full frame. So
    // bound it at one libopus frame (60 ms, the largest) either side. The tolerance is the point
    // of the assertion, not a hedge: a *truncated* decode or a *missing* pre-skip trim both move
    // this by far more than a frame, and both did before this chain existed.
    let want = last_granule - pre_skip as u64;
    let slack = OPUS_RATE * 60 / 1000;
    assert!(
        (frames as i64 - want as i64).unsigned_abs() <= slack,
        "decoded {frames} interchannel frames, expected {want} ± {slack}"
    );

    // Every buffer is stamped, and time only moves forwards.
    let stamped: Vec<u64> = stats.after_last_flush().iter().filter_map(|&(p, _)| p).collect();
    assert!(!stamped.is_empty(), "opusdec emitted no timestamps at all");
    assert_eq!(stamped[0], 0, "playback from the top starts at zero, not {}", stamped[0]);
    assert!(stamped.windows(2).all(|w| w[0] <= w[1]), "timestamps must be non-decreasing");
}

// --- Pre-skip ----------------------------------------------------------------------------------

#[test]
fn pre_skip_is_trimmed_at_stream_start() {
    // RFC 7845 §4.2: "the first `pre_skip` samples … MUST be discarded" — encoder priming that
    // is not part of the signal.
    //
    // The trim leaves no mark in the output, so instead of hunting for one this decodes the
    // *same audio twice* with the header's pre-skip differing by a known amount. If the trim is
    // honoured, the second run is short by exactly that many interchannel samples and its
    // output is the first run's output with that much sliced off the front — byte for byte,
    // since both runs decode an identical packet sequence from an identical initial state. If
    // the trim were ignored (or clamped to the first packet, the easy bug), the two runs would
    // be identical instead.
    const EXTRA: u16 = 4_800; // 100 ms at 48 kHz — five libopus packets, so the trim must span them

    let dir = tmp_dir("ogg_opus");
    let Some(base_path) = opus_fixture(&dir, "preskip_base.opus", 3) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let bytes = std::fs::read(&base_path).expect("read the fixture");
    let (pre_skip, channels, _) = head_and_granule(&bytes);

    let patched_path = dir.join("preskip_more.opus");
    std::fs::write(&patched_path, with_pre_skip(&bytes, pre_skip + EXTRA)).expect("write the patch");

    let (base_stats, base_run) = run_to_eos(&base_path);
    base_run.expect("the unpatched stream must reach EOS");
    let (more_stats, more_run) = run_to_eos(&patched_path);
    more_run.expect("the patched stream must reach EOS");

    let base = base_stats.pcm();
    let more = more_stats.pcm();
    let stride = BYTES_PER_SAMPLE * channels as usize; // bytes per interchannel sample
    let cut = EXTRA as usize * stride;
    assert!(base.len() > cut * 4, "the fixture is too short to measure a {EXTRA}-sample trim");

    assert_eq!(
        base.len() - more.len(),
        cut,
        "a {EXTRA}-sample larger pre-skip must remove exactly {EXTRA} interchannel samples \
         ({cut} bytes); got {} bytes",
        base.len() - more.len()
    );
    assert_eq!(
        more, base[cut..],
        "the patched run's output must be the base run's output with the first {EXTRA} \
         interchannel samples removed — the trim is happening somewhere else, or not at all"
    );
    // Guard the guard: the two runs really are different, so the equality above is not two
    // identical buffers agreeing with themselves.
    assert_ne!(&more[..stride * 16], &base[..stride * 16], "the patch changed nothing");
}

// --- Seeking ------------------------------------------------------------------------------------

#[test]
fn seeking_recovers_with_post_seek_timestamps() {
    let dir = tmp_dir("ogg_opus");
    let Some(path) = opus_fixture(&dir, "seek.opus", 12) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let player = open(&path);
    let duration = player.duration().expect("a granule-derived duration");
    let index = player.seek_index().clone();
    let file_len = std::fs::metadata(&path).unwrap().len();

    // Ogg v1 has no keyframe index, so the map is the bare proportional fallback: a duration
    // and a file length, resolved arithmetically rather than from a table. That is exactly what
    // makes this the interesting seek to test — every landing is an arbitrary byte, mid-page and
    // mid-packet, and the demuxer has to resync from there (RFC 3533 §6).
    let (byte, at) = index.resolve(Timestamp(duration.0 / 2), duration).expect("a duration maps");
    assert!(byte < file_len, "the landing byte {byte} is outside a {file_len}-byte file");
    assert!(at.0 <= duration.0, "resolved past the end of the stream");

    let targets: Vec<Timestamp> =
        [25u64, 50, 90].iter().map(|p| Timestamp(duration.0 * p / 100)).collect();
    let (stats, landed) = run_and_seek(&path, &index, duration, &targets);

    let (at, _byte) = *landed.last().unwrap();
    let post = stats.after_last_flush();
    assert!(!post.is_empty(), "no buffers after the final seek");
    assert!(post.iter().any(|&(_, n)| n > 0), "post-seek buffers are all empty");

    let stamped: Vec<u64> = post.iter().filter_map(|&(pts, _)| pts).collect();
    assert!(!stamped.is_empty(), "post-seek buffers carry no timestamps at all");
    // The landing byte is mid-page and mid-packet, so the first packet the demuxer recovers
    // starts a little *after* the target — never before, which is the direction that would mean
    // `opusdec` never re-based its sample counter to the seek target on `FlushStart`.
    let first = stamped[0];
    let floor = at.0.saturating_sub(100_000_000);
    assert!(
        first >= floor,
        "post-seek pts {first} is before the seek landing {at:?} — the decoder's sample counter \
         was not re-based to the seek target"
    );
    assert!(
        first <= duration.0 + 100_000_000,
        "post-seek pts {first} is past the end of a {duration:?} stream"
    );
    assert!(stamped.windows(2).all(|w| w[0] <= w[1]), "post-seek timestamps must be non-decreasing");
}

#[test]
fn a_seek_does_not_replay_pre_seek_audio() {
    // The failure this pins: `oggdemux` holds a whole source chunk of *already-pushed* bytes
    // inside its reader (the page parser runs to a bounded look-ahead, not to the end of the
    // input). Without a `FlushStart` reset those bytes are pre-seek, and they come out after
    // the flush — audibly, as a second or two of the old position before the new one arrives.
    // A timestamp is what catches it: those buffers would be stamped from the *re-based*
    // counter, i.e. at the new position, while carrying the old audio. So instead of listening,
    // assert the demuxer is empty across the flush by seeking to the very end and requiring
    // that what follows is bounded, not a burst.
    let dir = tmp_dir("ogg_opus");
    let Some(path) = opus_fixture(&dir, "noreplay.opus", 12) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let player = open(&path);
    let duration = player.duration().expect("a granule-derived duration");
    let index = player.seek_index().clone();

    // Seek to 95 %: past that point there is at most 5 % of the file left to decode, so a
    // post-seek run that produces substantially more than that has replayed buffered bytes.
    let target = Timestamp(duration.0 * 95 / 100);
    let (stats, _) = run_and_seek(&path, &index, duration, &[target]);
    let post = stats.after_last_flush();
    let bytes: usize = post.iter().map(|&(_, n)| n).sum();
    // 5 % of a 12 s stereo 48 kHz s16 stream is ~115 KB. Allow 4× that for the tail of the
    // final pages plus scheduling slop; a pre-seek replay would be the whole remaining file.
    let budget = (duration.0 as usize / 20) * 4 * OPUS_RATE as usize * 4 / 1_000_000_000;
    assert!(
        bytes <= budget.max(512 * 1024),
        "{bytes} bytes after a 95 % seek — the demuxer replayed pre-seek input"
    );
}

// --- Hostile input ---------------------------------------------------------------------------------

#[test]
fn truncated_files_never_panic() {
    // A decoder parses untrusted input and a crash on bad input is a P0. Cutting an Ogg file
    // leaves a partial final page — and cutting it *early* leaves a partial `OpusHead`, which
    // is the one packet everything downstream is configured from.
    let dir = tmp_dir("ogg_opus");
    let Some(path) = opus_fixture(&dir, "truncate.opus", 3) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let whole = std::fs::read(&path).expect("read the fixture");

    for pct in [1usize, 3, 10, 37, 50, 63, 88, 99] {
        let cut = whole.len() * pct / 100;
        let cut_path = dir.join(format!("truncate_{pct}.opus"));
        std::fs::write(&cut_path, &whole[..cut]).expect("write the cut file");

        // `Player::open` may legitimately fail on a file cut before its header — that is a
        // reported error, not a panic, and it is the whole distinction being tested.
        let policy = SinkPolicy { video: SinkChoice::Drop, audio: SinkChoice::Drop };
        let Ok(mut player) = Player::open(cut_path.to_str().unwrap(), policy) else { continue };
        // Whatever it decides about the duration, running it must terminate without panicking.
        let _ = player.duration();
        let _ = player.run();

        // …and the same file straight through the chain under test.
        let (_stats, _r) = run_to_eos(&cut_path);
    }
}

#[test]
fn corrupted_pages_never_panic() {
    // Garbage inside the stream, rather than a clean cut: the CRC (RFC 3533 §6, field 7) fails,
    // the reader resyncs to the next capture pattern, and `opusdec` warn-drops any packet that
    // survives to it malformed. Neither may take the process down.
    let dir = tmp_dir("ogg_opus");
    let Some(path) = opus_fixture(&dir, "corrupt.opus", 3) else {
        eprintln!("ffmpeg absent — skipping");
        return;
    };
    let whole = std::fs::read(&path).expect("read the fixture");

    for (i, seed) in [7u8, 0xFF, 0x4F].iter().enumerate() {
        let mut bytes = whole.clone();
        // Smear a run of bytes across the middle, and drop a bogus capture pattern in to make
        // the resync scanner chase a false page start.
        let from = bytes.len() / 3 + i * 101;
        for b in bytes[from..(from + 900).min(whole.len())].iter_mut() {
            *b ^= seed;
        }
        let at = bytes.len() / 2;
        bytes[at..at + 4].copy_from_slice(b"OggS");

        let cut_path = dir.join(format!("corrupt_{i}.opus"));
        std::fs::write(&cut_path, &bytes).expect("write the corrupt file");
        let (_stats, _r) = run_to_eos(&cut_path);
    }
}
