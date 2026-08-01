//! **Seeking near the end of a track that has something queued behind it.**
//!
//! The reported bug: "seeking to the end when something is queued skips to the next song." The
//! user drags the scrubber to 99%, expects to hear the last moment of the track, and instead the
//! next one starts immediately.
//!
//! # Why that happened, and what the fix is
//!
//! A seek is resolved through the track's [`SeekIndex`] into a *byte* offset. For a container
//! with no cue table — a FLAC, which is most of a music library — that is a proportional
//! estimate, so a target at 100% of the duration resolves to a byte at (or past) the end of the
//! file. The source then reads nothing, the pipeline EOSes at once, and the boundary machinery
//! does exactly what it is supposed to do at an end of stream: it advances the queue. Nothing
//! was broken; the seek target was simply unlistenable.
//!
//! The engine is the UX layer, so the engine is where it is fixed: a seek target is clamped to
//! `duration - TAIL_EPSILON` when the duration is known. "Seek to the end" therefore means
//! **play the last quarter-second, then advance** — which is both audible and the obvious
//! reading of the gesture. `SeekIndex` semantics are untouched: it still resolves exactly what
//! it is asked to, and callers that genuinely want the last byte still get it.
//!
//! # How the tail is proven audible
//!
//! Not by listening, and not by timing. Each fixture carries its identity and its own timeline
//! *in the samples*: the right channel is a constant, positive in track A and negative in track
//! B, and the left channel is a staircase that steps every 100 ms. Every frame the virtual
//! device records therefore answers two questions exactly — which track it came from, and how
//! far into that track it was. The chain is a pass-through at 48 kHz (`gapless.rs` relies on the
//! same fact), so the markers arrive at the device unmodified and the decode is a comparison,
//! not an estimate.
//!
//! That turns "the user heard the tail" into a frame count over a labelled recording, and
//! "it skipped" into that count being zero.

// Test module: fixtures, recordings and assertions allocate freely — none of this is on a media
// path or inside `process()`.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pf_pipewire::out::testing::{open_capture, CaptureHandle};
use pf_pipewire::AudioOutConfig;
use pf_player_engine::{Engine, EngineEvent, Track};

/// The canonical output frame: 2 channels × 4 bytes of f32.
const STRIDE: usize = 8;

/// The canonical rate. Fixtures are authored at it so the chain's resampler is a pass-through
/// and the markers survive to the device bit-for-bit.
const RATE: usize = 48_000;

const PATIENCE: Duration = Duration::from_secs(20);

/// Track A's length. Long enough that "90% in" is unambiguously past the point the test lets it
/// reach before seeking, and short enough to stay a fast test.
const A_SECS: f64 = 6.0;
const B_SECS: f64 = 3.0;

/// The engine's documented near-end clamp: a seek lands at most this far from the end.
const TAIL_EPSILON: Duration = Duration::from_millis(250);

/// How much of A's tail must actually reach the device for the seek to count as audible.
///
/// Below the clamp itself, deliberately. The clamp fixes the *target*; what arrives is then
/// subject to the proportional byte estimate landing a FLAC frame or two late, so demanding the
/// full 250 ms would be asserting the precision of an estimate rather than the behaviour under
/// test. 120 ms is unmistakably "the track played on" and still an order of magnitude away from
/// the bug, which delivered exactly zero.
const MIN_AUDIBLE_TAIL: Duration = Duration::from_millis(120);

// --- fixtures ------------------------------------------------------------------------------

fn tmp_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("seek_boundary");
    std::fs::create_dir_all(&d).expect("tmp dir");
    d
}

/// Which fixture a recorded frame came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    A,
    B,
    /// The device rendered silence: parked, underrunning, or flushed.
    Silence,
}

/// The right-channel constant that labels a track. Near full scale so the classification is a
/// sign test with an enormous margin, and identical in magnitude for both so neither track is
/// louder than the other.
const LABEL: i16 = 30_000;

/// Left-channel staircase step, in s16 units per 100 ms. 6 seconds is 59 steps, so the largest
/// value is 29 500 — inside i16, and every step is far larger than any conceivable numeric
/// slop.
const STEP: i32 = 500;

/// Frames per staircase step.
const STEP_FRAMES: usize = RATE / 10;

/// Interleaved s16 stereo carrying the track label and a 100 ms timeline.
fn marker_pcm(frames: usize, which: Which) -> Vec<u8> {
    let label = match which {
        Which::A => LABEL,
        Which::B => -LABEL,
        Which::Silence => unreachable!("fixtures are never silence"),
    };
    let mut pcm = Vec::with_capacity(frames * 4);
    for i in 0..frames {
        let l = (STEP * (i / STEP_FRAMES) as i32) as i16;
        pcm.extend_from_slice(&l.to_le_bytes());
        pcm.extend_from_slice(&label.to_le_bytes());
    }
    pcm
}

fn write_flac(name: &str, pcm: &[u8]) -> PathBuf {
    use pf_flac::{FlacEncoder, SampleFormat as FlacFmt};
    let (mut enc, header) = FlacEncoder::new(RATE as u32, 2, FlacFmt::S16).expect("flac encoder");
    let mut frames_out = Vec::new();
    enc.encode_interleaved(pcm, &mut frames_out).expect("encode");
    let body = enc.finish();
    let mut file = header.clone();
    let at = pf_flac::streaminfo_offset();
    file[at..at + body.len()].copy_from_slice(&body);
    file.extend_from_slice(&frames_out);
    let path = tmp_dir().join(name);
    std::fs::write(&path, &file).expect("write flac fixture");
    path
}

fn marker_track(name: &str, secs: f64, which: Which) -> PathBuf {
    write_flac(name, &marker_pcm((RATE as f64 * secs) as usize, which))
}

// --- reading the recording back ----------------------------------------------------------------

/// One decoded frame of the recording.
#[derive(Clone, Copy, Debug)]
struct Mark {
    which: Which,
    /// Position within that track, decoded from the staircase. Meaningless for silence.
    at: Duration,
}

fn f32_at(rec: &[u8], byte: usize) -> f32 {
    f32::from_le_bytes([rec[byte], rec[byte + 1], rec[byte + 2], rec[byte + 3]])
}

/// Decode every frame of a capture into its label and its source position.
///
/// The right channel is `±LABEL/32768` (≈ ±0.916), so the sign test has ~0.9 of margin either
/// way and a silent frame is unambiguous. The left channel divides straight back out to the
/// staircase index.
fn marks(rec: &[u8]) -> Vec<Mark> {
    let mut out = Vec::with_capacity(rec.len() / STRIDE);
    for f in 0..rec.len() / STRIDE {
        let l = f32_at(rec, f * STRIDE);
        let r = f32_at(rec, f * STRIDE + 4);
        let which = if r > 0.5 {
            Which::A
        } else if r < -0.5 {
            Which::B
        } else {
            Which::Silence
        };
        // `l` is `STEP * step_index / 32768`; recover the index and scale it back to time.
        let step = (l * 32_768.0 / STEP as f32).round().max(0.0) as u64;
        out.push(Mark { which, at: Duration::from_millis(step * 100) });
    }
    out
}

fn count(m: &[Mark], which: Which) -> usize {
    m.iter().filter(|x| x.which == which).count()
}

fn frames_to_dur(frames: usize) -> Duration {
    Duration::from_secs_f64(frames as f64 / RATE as f64)
}

/// Everything one near-end seek produced, decoded.
struct Landing {
    /// A-frames recorded from the second half of A — unambiguously post-seek, because the test
    /// never lets A play past the first second before it seeks.
    tail_frames: usize,
    /// Where in A the post-seek audio started.
    landed_at: Option<Duration>,
    /// The first recorded frame belonging to B, if any.
    first_b: Option<usize>,
    /// The last recorded post-seek frame belonging to A, if any.
    last_tail: Option<usize>,
}

/// Anything at or past this point in A can only have come from the seek: the tests seek after
/// letting A play for well under a second.
const POST_SEEK_FLOOR: Duration = Duration::from_millis(2_000);

fn landing(rec: &[u8]) -> Landing {
    let m = marks(rec);
    let tail: Vec<usize> = (0..m.len())
        .filter(|&i| m[i].which == Which::A && m[i].at >= POST_SEEK_FLOOR)
        .collect();
    Landing {
        tail_frames: tail.len(),
        landed_at: tail.first().map(|&i| m[i].at),
        first_b: (0..m.len()).find(|&i| m[i].which == Which::B),
        last_tail: tail.last().copied(),
    }
}

// --- the harness --------------------------------------------------------------------------------

/// A background thread keeping the virtual device consuming, so test bodies stay linear.
/// (`robustness.rs`'s pump, at the same ~10× real time.)
struct Pump {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Pump {
    fn start(cap: CaptureHandle) -> Pump {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            while !flag.load(Ordering::Acquire) {
                cap.step_available(480);
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        Pump { stop, thread: Some(thread) }
    }
}

impl Drop for Pump {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct Rig {
    engine: Engine,
    cap: CaptureHandle,
    events: Vec<EngineEvent>,
    _pump: Pump,
}

impl Rig {
    fn new() -> Rig {
        let (out, cap) = open_capture(AudioOutConfig::default());
        let pump = Pump::start(cap.clone());
        Rig { engine: Engine::with_output(out), cap, events: Vec::new(), _pump: pump }
    }

    fn poll(&mut self) {
        self.events.extend(self.engine.poll_events());
    }

    fn settle(&mut self, mut cond: impl FnMut(&Rig) -> bool) -> bool {
        let deadline = Instant::now() + PATIENCE;
        loop {
            self.poll();
            if cond(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn errors(&self) -> Vec<&str> {
        self.events
            .iter()
            .filter_map(|e| match e {
                EngineEvent::Error { message } => Some(message.as_str()),
                _ => None,
            })
            .collect()
    }

    fn saw(&self, want: fn(&EngineEvent) -> bool) -> bool {
        self.events.iter().any(want)
    }

    /// Play A, queue B behind it, and let A get properly under way — then forget the recording,
    /// so everything captured afterwards belongs to the seek under test.
    fn primed(a: &PathBuf, b: Option<&PathBuf>) -> Rig {
        let mut rig = Rig::new();
        rig.engine.play_now(Track::file(a));
        assert!(
            rig.settle(|r| r.engine.is_playing() && r.cap.available() > 32 * 1024),
            "track A never started (errors: {:?})",
            rig.errors()
        );
        if let Some(b) = b {
            rig.engine.enqueue(Track::file(b)).expect("enqueue B");
            assert!(
                rig.settle(|r| r.engine.queue_len() == 2),
                "B was never accepted (errors: {:?})",
                rig.errors()
            );
        }
        // Far enough in to be sure playback is real, far enough from `POST_SEEK_FLOOR` that no
        // pre-seek frame can be mistaken for a post-seek one.
        assert!(
            rig.settle(|r| r.engine.position() >= Duration::from_millis(600)),
            "A never reached 600 ms (errors: {:?})",
            rig.errors()
        );
        rig.cap.clear();
        rig
    }
}

// --- the matrix ------------------------------------------------------------------------------

/// Seek to `fraction` of A's duration with B queued behind, and report what reached the device.
fn near_end_seek_with_queue(tag: &str, fraction: f64) -> (Landing, Vec<EngineEvent>) {
    let a = marker_track(&format!("sb_{tag}_a.flac"), A_SECS, Which::A);
    let b = marker_track(&format!("sb_{tag}_b.flac"), B_SECS, Which::B);
    let mut rig = Rig::primed(&a, Some(&b));

    rig.engine.seek(Duration::from_secs_f64(A_SECS * fraction));

    // Drive until B is genuinely audible — the user-visible end of the transition — or until
    // the queue has drained, which is the failure mode where nothing of B arrives either.
    let heard_b = rig.settle(|r| {
        count(&marks(&r.cap.captured()), Which::B) >= RATE / 10 || r.engine.queue_len() == 0
    });
    assert!(heard_b, "{tag}: B never became audible (errors: {:?})", rig.errors());
    assert!(rig.errors().is_empty(), "{tag}: {:?}", rig.errors());

    let rec = rig.cap.captured();
    let l = landing(&rec);
    eprintln!(
        "{tag}: seek to {:.0}% of A -> landed at {:?}, {} tail frames ({:.0} ms audible)",
        fraction * 100.0,
        l.landed_at,
        l.tail_frames,
        frames_to_dur(l.tail_frames).as_secs_f64() * 1000.0
    );
    (l, rig.events.clone())
}

/// The whole point: at every near-end target, the track the user was listening to keeps playing
/// for long enough to hear before the queue advances.
fn assert_tail_then_advance(tag: &str, fraction: f64) {
    let (l, events) = near_end_seek_with_queue(tag, fraction);

    assert!(
        frames_to_dur(l.tail_frames) >= MIN_AUDIBLE_TAIL,
        "{tag}: seeking to {:.0}% of A played only {:.0} ms of A's tail before advancing \
         (wanted at least {:.0} ms) — the user hears a skip, not a landing",
        fraction * 100.0,
        frames_to_dur(l.tail_frames).as_secs_f64() * 1000.0,
        MIN_AUDIBLE_TAIL.as_secs_f64() * 1000.0,
    );

    // The landing is in the last part of A, not somewhere earlier: a clamp that overshot
    // backwards would also produce a long "tail", and would be just as wrong.
    let landed = l.landed_at.expect("tail frames exist, so a landing does");
    let earliest = Duration::from_secs_f64(A_SECS).saturating_sub(TAIL_EPSILON * 4);
    assert!(
        landed >= earliest,
        "{tag}: the seek landed at {landed:?}, which is not near the end (expected >= {earliest:?})"
    );

    // Order, not event timing: `TrackChanged` is emitted when A's pipeline finishes, which by
    // design is while up to half a second of A is still in flight in the ring (detach without
    // drain — that overlap *is* the gaplessness). So the assertion that means "the tail was not
    // cut off" is about the audio: every post-seek frame of A precedes every frame of B.
    if let (Some(last_a), Some(first_b)) = (l.last_tail, l.first_b) {
        assert!(
            last_a < first_b,
            "{tag}: B's audio (frame {first_b}) arrived before A's tail finished (frame {last_a})"
        );
    }

    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::TrackChanged)),
        "{tag}: the queue never advanced onto B: {events:?}"
    );
}

#[test]
fn a_seek_to_90_percent_plays_the_tail_before_advancing() {
    assert_tail_then_advance("p90", 0.90);
}

#[test]
fn a_seek_to_95_percent_plays_the_tail_before_advancing() {
    assert_tail_then_advance("p95", 0.95);
}

#[test]
fn a_seek_to_99_percent_plays_the_tail_before_advancing() {
    assert_tail_then_advance("p99", 0.99);
}

#[test]
fn a_seek_to_the_exact_duration_plays_the_last_epsilon_before_advancing() {
    assert_tail_then_advance("p100", 1.00);
}

#[test]
fn a_seek_past_the_duration_plays_the_last_epsilon_before_advancing() {
    // The scrubber cannot produce this, but MPRIS, a remote and a restored resume point all can.
    assert_tail_then_advance("beyond", 1.35);
}

// --- the same targets with nothing queued -------------------------------------------------------

/// With an empty queue the policy has to hold just as exactly, and end in `TrackEnded` rather
/// than `TrackChanged`. This is the case that would otherwise mask a regression: "it stopped"
/// looks like a plausible answer to "seek to the end", so the tail has to be measured here too.
#[test]
fn a_seek_to_the_end_of_a_lone_track_plays_the_tail_then_ends() {
    let a = marker_track("sb_lone_a.flac", A_SECS, Which::A);
    let mut rig = Rig::primed(&a, None);

    rig.engine.seek(Duration::from_secs_f64(A_SECS));
    assert!(
        rig.settle(|r| r.engine.queue_len() == 0 && r.cap.available() < STRIDE),
        "the track never ended (errors: {:?})",
        rig.errors()
    );
    assert!(rig.errors().is_empty(), "{:?}", rig.errors());

    let l = landing(&rig.cap.captured());
    eprintln!(
        "lone: seek to the end -> landed at {:?}, {:.0} ms audible",
        l.landed_at,
        frames_to_dur(l.tail_frames).as_secs_f64() * 1000.0
    );
    assert!(
        frames_to_dur(l.tail_frames) >= MIN_AUDIBLE_TAIL,
        "seeking to the end of a lone track played only {:.0} ms of it",
        frames_to_dur(l.tail_frames).as_secs_f64() * 1000.0
    );
    assert!(l.first_b.is_none(), "nothing was queued, so nothing may follow");
    assert!(
        rig.saw(|e| matches!(e, EngineEvent::TrackEnded)),
        "no TrackEnded: {:?}",
        rig.events
    );
}

// --- the boundary race ---------------------------------------------------------------------------

/// A seek issued in the instant a track boundary is happening must never be *partly* applied.
///
/// `Engine::seek` reads slot 0 under the same lock the advance writes it under, so it sees
/// either the outgoing track or the incoming one, never a mixture. The documented consequence:
/// a seek that loses the race applies to the **new** current track, and one that arrives while
/// the outgoing track's runner has returned but the advance has not yet run is inert (that
/// pipeline is finished; nothing observes its seek generation).
///
/// Both outcomes are acceptable and both are *stable* — what would not be is a panic, a stall,
/// or a queue left inconsistent. This drives the race hard and asserts exactly that.
///
/// # The seek rate, and why it is what it is
///
/// `SEEK_SPACING` is 2 ms: five hundred seeks a second, which is already twenty-odd times what a
/// dragged scrubber, a held arrow key or an MPRIS client can emit, and the pump runs the device
/// at roughly ten times real time on top of that. It is deliberately *not* lower.
///
/// At ~200 µs spacing this test also fails, and for a reason that is not the engine's: the seeks
/// arrive faster than `flacdec` can re-sync, and the decoder eventually reports
/// `Resource("flacdec: BadMagic")` and kills the track. That is a real fragility in the decoder's
/// resync (it lives in `flac/`, not here) and it is worth fixing on its own terms, but it is not
/// reachable from any user gesture and it is not what this test is about. Spacing the seeks
/// keeps this an assertion about the *boundary race* rather than a decoder stress test that
/// would fail for an unrelated reason.
#[test]
fn a_seek_racing_the_boundary_leaves_the_queue_consistent() {
    /// See the doc comment: fast enough to land inside the boundary window many times over,
    /// slow enough not to be testing `flacdec`'s resync instead.
    const SEEK_SPACING: Duration = Duration::from_millis(2);

    let a = marker_track("sb_race_a.flac", 1.5, Which::A);
    let b = marker_track("sb_race_b.flac", B_SECS, Which::B);
    let mut rig = Rig::new();
    rig.engine.play_now(Track::file(&a));
    assert!(rig.settle(|r| r.engine.is_playing()), "A never started");
    rig.engine.enqueue(Track::file(&b)).expect("enqueue B");
    assert!(rig.settle(|r| r.engine.queue_len() == 2), "B was never accepted");

    // Seek straight through the boundary, to a target far enough into A that the advance keeps
    // happening underneath the seeks. Each is legal at any instant; the engine must survive all
    // of them and still end up playing B.
    let deadline = Instant::now() + PATIENCE;
    while !rig.saw(|e| matches!(e, EngineEvent::TrackChanged)) {
        rig.engine.seek(Duration::from_millis(1_200));
        std::thread::sleep(SEEK_SPACING);
        rig.poll();
        assert!(Instant::now() < deadline, "the boundary never happened: {:?}", rig.events);
    }
    // Keep going a little past the advance, so seeks provably land on a track that has *just*
    // become current — the "dispatched to slot 0 after it advanced" case.
    for _ in 0..20 {
        rig.engine.seek(Duration::from_millis(1_200));
        std::thread::sleep(SEEK_SPACING);
    }
    rig.poll();

    // Slot 0 is B now, and it is a whole, healthy track: it can be seeked, it reports B's
    // duration, and it plays to its own end.
    assert_eq!(rig.engine.queue_len(), 1, "after the advance only B should be in the queue");
    let d = rig.engine.duration().expect("B declares a duration");
    assert!(
        d.as_secs_f64() > B_SECS - 0.5 && d.as_secs_f64() < B_SECS + 0.5,
        "slot 0 reports {d:?}, which is not B's duration"
    );
    rig.engine.seek(Duration::from_millis(500));
    assert!(
        rig.settle(|r| r.engine.queue_len() == 0),
        "B never finished after the racing seeks (errors: {:?})",
        rig.errors()
    );
    assert!(rig.errors().is_empty(), "{:?}", rig.errors());
    assert!(rig.saw(|e| matches!(e, EngineEvent::TrackEnded)), "no TrackEnded: {:?}", rig.events);
}

/// A track that is **still opening** when the current one ends must still play.
///
/// Found by the racing-seek test above, which reached this state by killing the current track
/// early enough that the queued one had not finished building. `advance_locked` handles it
/// deliberately — it moves the `Building` marker down into slot 0 and re-labels it as a change
/// rather than a start — so the intent has always been that the build lands there. It could not:
/// the build job carried the slot index it was *posted* with, and looked that slot up when it
/// finished, so a build that had been moved found a slot that no longer wanted it, threw itself
/// away, and left slot 0 marked `Building` for ever. `queue_len()` stuck at 1, `is_playing()`
/// false, and neither `TrackChanged` nor `TrackEnded` ever again: the app's playlist stops dead
/// and only an explicit `play_now` recovers it.
///
/// Reproduced without a race, by holding the build open: B's file is complete on disk, but its
/// download frontier says nothing has arrived, so its open blocks on the worker until this test
/// says otherwise — for as long as it takes A to finish.
#[test]
fn a_queued_track_still_opening_when_the_current_one_ends_still_plays() {
    let a = marker_track("sb_open_a.flac", 2.0, Which::A);
    let b = marker_track("sb_open_b.flac", B_SECS, Which::B);
    let len = std::fs::metadata(&b).expect("stat B").len();

    let mut rig = Rig::new();
    rig.engine.play_now(Track::file(&a));
    // The *event*, not `is_playing()`: at the pump's ~10x rate a short track can start and
    // finish between two polls, and this test must not depend on catching it mid-flight.
    assert!(
        rig.settle(|r| r.saw(|e| matches!(e, EngineEvent::TrackStarted))),
        "A never started (errors: {:?})",
        rig.errors()
    );

    // The "downloader" has published nothing, so B's open parks in `wait_readable`.
    let (queued, frontier) = Track::growing(&b, Some(len)).expect("a UTF-8 path");
    rig.engine.enqueue(queued).expect("enqueue B");

    // A ends first. B is now in slot 0 as a build in flight — accepted, not yet open.
    assert!(
        rig.settle(|r| !r.engine.is_playing()),
        "A never finished (errors: {:?})",
        rig.errors()
    );
    assert_eq!(rig.engine.queue_len(), 1, "B is still queued, it has only not opened yet");
    assert!(
        !rig.saw(|e| matches!(e, EngineEvent::TrackEnded)),
        "the queue has not drained — B is still coming: {:?}",
        rig.events
    );

    // The download lands, so the open completes and B must take over.
    frontier.advance(len);
    frontier.finish(len);

    assert!(
        rig.settle(|r| r.saw(|e| matches!(e, EngineEvent::TrackChanged))),
        "B never took over after its open completed (queue_len {}, playing {}, events {:?})",
        rig.engine.queue_len(),
        rig.engine.is_playing(),
        rig.events
    );
    assert!(rig.engine.is_playing(), "B was announced but is not playing");
    assert!(rig.errors().is_empty(), "{:?}", rig.errors());

    // And it is really B's audio, not silence or a leftover of A.
    assert!(
        rig.settle(|r| count(&marks(&r.cap.captured()), Which::B) >= RATE / 10),
        "B never became audible"
    );
    assert!(
        rig.settle(|r| r.engine.queue_len() == 0),
        "B never finished (errors: {:?})",
        rig.errors()
    );
}

/// The marker scheme itself, so a failure in the tests above is never ambiguous about whether
/// the *measurement* is sound. Plays A alone from the start and checks that the recording
/// decodes as A throughout, with a staircase that advances monotonically to the end.
#[test]
fn the_marker_fixtures_decode_back_exactly() {
    let a = marker_track("sb_selftest.flac", 2.0, Which::A);
    let mut rig = Rig::new();
    rig.engine.play_now(Track::file(&a));
    assert!(rig.settle(|r| r.engine.is_playing()), "A never started");
    assert!(
        rig.settle(|r| r.engine.queue_len() == 0 && r.cap.available() < STRIDE),
        "A never finished (errors: {:?})",
        rig.errors()
    );

    let m = marks(&rig.cap.captured());
    assert_eq!(count(&m, Which::B), 0, "nothing of B exists in this test");
    let a_frames = count(&m, Which::A);
    assert!(
        a_frames > RATE * 3 / 2,
        "only {a_frames} frames decoded as A out of a two-second fixture"
    );
    // The staircase never goes backwards, and it reaches the last step of a 2 s fixture.
    let mut last = Duration::ZERO;
    for x in m.iter().filter(|x| x.which == Which::A) {
        assert!(x.at >= last, "the staircase went backwards: {:?} after {last:?}", x.at);
        last = x.at;
    }
    assert!(last >= Duration::from_millis(1_800), "the staircase only reached {last:?}");
}
