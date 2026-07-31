//! **The acceptance proof.** One continuous signal, cut in two, played as two tracks — and the
//! device's recording compared with the two halves played separately and concatenated, byte for
//! byte.
//!
//! # Why this is a proof and not a demonstration
//!
//! The engine plays through `pf_pipewire::out::testing::open_capture`: a real [`AudioOut`] whose
//! device backend has no device. It runs the *same* `Renderer` the PipeWire real-time callback
//! runs — same ring, same counters, same gain path — but a test steps its clock by hand and it
//! keeps every byte it consumes. So "the boundary lost nothing" becomes an array comparison at a
//! specific byte index, not a listening session.
//!
//! The expected side is built by the engine itself, one track at a time. That is deliberate:
//! any hand-rolled reference would be a second implementation to be wrong in its own way,
//! whereas *this* comparison isolates exactly one variable — whether there was a boundary — and
//! holds every other byte of the code path fixed. What it therefore proves is the precise claim:
//! **the handoff neither loses, duplicates, nor reorders a single sample.**
//!
//! It deliberately does *not* claim the two halves rejoin into the same waveform the unsplit
//! source would have produced. They cannot quite, at 44.1 kHz: each track's resampler starts
//! cold and does not flush its filter tail at EOS (a documented `audioresample` follow-up), so a
//! filter length is lost at each seam. That loss is a property of splitting the file, is
//! identical on both sides of this comparison, and is what the byte-exact equality pins down —
//! the engine adds nothing to it.
//!
//! # The enqueue point
//!
//! The application enqueues the next track about five seconds early. These fixtures are seconds
//! long, not minutes, so the test scales that to **one second before the end** — and one second
//! is not an arbitrary choice: the output ring holds half a second, so a track's `run()` cannot
//! return until the device has consumed all but that half second of it. Enqueueing a full second
//! from the end is therefore provably *before* the boundary, whatever the machine's timing, with
//! half a second of margin. That is what makes the `queue_len()` 2 → 1 assertion deterministic
//! rather than lucky.

// Test module: fixtures, byte vectors and assertions allocate freely — none of this is on a
// media path or inside `process()`.
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use pf_pipewire::out::testing::{open_capture, CaptureHandle};
use pf_pipewire::AudioOutConfig;
use pf_player_engine::{Engine, EngineEvent, Track};

/// Frames the virtual device renders per step. Not a divisor of anything in the fixtures — the
/// ring is a byte stream and the seam must be exact regardless of how the device blocks it up.
const QUANTUM: usize = 1024;

/// The canonical output frame: 2 channels × 4 bytes.
const STRIDE: usize = 8;

/// One second of canonical output.
const SEC_BYTES: usize = 48_000 * STRIDE;

/// How long any single drive loop may run before the test gives up and fails loudly.
const PATIENCE: Duration = Duration::from_secs(60);

// --- fixtures ------------------------------------------------------------------------------

fn tmp_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("gapless");
    std::fs::create_dir_all(&d).expect("tmp dir");
    d
}

/// One continuous, deterministic, non-repeating stereo signal: two incommensurable tones plus a
/// slow sweep, with the right channel phase-shifted so a channel swap is provable rather than
/// plausible. Interleaved s16.
fn signal(rate: u32, frames: usize) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(frames * 4);
    for i in 0..frames {
        let t = i as f64 / rate as f64;
        let sweep = 200.0 + 60.0 * t;
        let l = 0.55 * (t * 220.0 * std::f64::consts::TAU).sin()
            + 0.30 * (t * sweep * std::f64::consts::TAU).sin();
        let r = 0.55 * (t * 331.0 * std::f64::consts::TAU).sin()
            + 0.30 * (t * sweep * std::f64::consts::TAU + 1.1).sin();
        pcm.extend_from_slice(&((l * 15_000.0) as i16).to_le_bytes());
        pcm.extend_from_slice(&((r * 15_000.0) as i16).to_le_bytes());
    }
    pcm
}

/// Encode interleaved s16 stereo PCM as a FLAC file, in process (no external tool).
fn write_flac(name: &str, rate: u32, pcm: &[u8]) -> PathBuf {
    use pf_flac::{FlacEncoder, SampleFormat as FlacFmt};
    let (mut enc, header) = FlacEncoder::new(rate, 2, FlacFmt::S16).expect("flac encoder");
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

/// Cut `total` frames of signal at `split` and write the two halves as FLAC files.
fn split_pair(tag: &str, rate: u32, total: usize, split: usize) -> (PathBuf, PathBuf) {
    let pcm = signal(rate, total);
    let at = split * 4; // 2 channels × 2 bytes
    (
        write_flac(&format!("{tag}_a.flac"), rate, &pcm[..at]),
        write_flac(&format!("{tag}_b.flac"), rate, &pcm[at..]),
    )
}

// --- driving the virtual device ---------------------------------------------------------------

/// Render one block if the ring has one, and report whether it did.
///
/// Deliberately **not** `drain`: taking only a block at a time leaves the producer ahead of the
/// device, so the ring stays full and the boundary happens with half a second of the outgoing
/// track genuinely in flight — the overlap the whole design rests on. `step_available` never
/// pads silence on an underrun, so nothing the test's own pacing does can appear in the
/// recording.
fn pump(cap: &CaptureHandle) -> bool {
    cap.step_available(QUANTUM) > 0
}

/// Step the device until `done`, collecting engine events as they arrive. Panics on timeout —
/// a stall here is the failure the test exists to catch.
fn drive(
    engine: &Engine,
    cap: &CaptureHandle,
    events: &mut Vec<EngineEvent>,
    what: &str,
    mut done: impl FnMut(&Engine, &CaptureHandle, &[EngineEvent]) -> bool,
) {
    let deadline = Instant::now() + PATIENCE;
    loop {
        events.extend(engine.poll_events());
        if done(engine, cap, events) {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for: {what} (events: {events:?})");
        if !pump(cap) {
            // Nothing queued yet: let the decoder get ahead.
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Play one file to the very end and return everything the device rendered.
fn play_one(path: &Path) -> Vec<u8> {
    let (out, cap) = open_capture(AudioOutConfig::default());
    let engine = Engine::with_output(out);
    let mut events = Vec::new();
    engine.play_now(Track::file(path));
    drive(&engine, &cap, &mut events, "the track to start", |e, _, _| e.is_playing());
    drive(&engine, &cap, &mut events, "the track to end", |e, _, _| e.queue_len() == 0);
    // `TrackEnded` fires when the pipeline finishes; the tail is still in the ring.
    drive(&engine, &cap, &mut events, "the tail to play out", |_, c, _| c.available() < STRIDE);
    let captured = cap.captured();
    assert!(!captured.is_empty(), "{} produced no audio at all", path.display());
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::TrackStarted)),
        "no TrackStarted for {}: {events:?}",
        path.display()
    );
    assert!(
        !events.iter().any(|e| matches!(e, EngineEvent::Error { .. })),
        "{} reported an error: {events:?}",
        path.display()
    );
    captured
}

/// What one gapless run observed.
struct Run {
    captured: Vec<u8>,
    events: Vec<EngineEvent>,
    /// `queue_len()` sampled immediately after the enqueue, and again once the boundary was seen.
    queue_after_enqueue: usize,
    queue_after_boundary: usize,
    /// Bytes still queued in the output ring at the instant the boundary was observed.
    ///
    /// Without this the proof would have a hole: a run in which the ring happened to be empty at
    /// the seam would compare equal for a trivial reason (there was nothing to overlap), and
    /// would say nothing about whether a producer can hand over *while audio is in flight* —
    /// which is the entire mechanism.
    ring_at_boundary: usize,
}

/// Play `a`, enqueue `b` one second before `a` ends, and let the boundary happen.
fn play_gapless(a: &Path, b: &Path, a_len: usize) -> Run {
    let (out, cap) = open_capture(AudioOutConfig::default());
    let engine = Engine::with_output(out);
    let mut events = Vec::new();

    engine.play_now(Track::file(a));
    drive(&engine, &cap, &mut events, "track A to start", |e, _, _| e.is_playing());

    // One second from the end — provably before A's `run()` can return, because the ring holds
    // only half a second and A cannot finish until the device has consumed all but that.
    let enqueue_at = a_len.saturating_sub(SEC_BYTES);
    drive(&engine, &cap, &mut events, "A to reach the enqueue point", |_, c, _| {
        c.captured_len() >= enqueue_at
    });
    assert_eq!(engine.queue_len(), 1, "A must still be playing when B is enqueued");
    engine.enqueue(Track::file(b)).expect("enqueue B");
    let queue_after_enqueue = engine.queue_len();

    let mut ring_at_boundary = 0;
    drive(&engine, &cap, &mut events, "the boundary", |_, c, ev| {
        if ev.iter().any(|e| matches!(e, EngineEvent::TrackChanged)) {
            ring_at_boundary = c.available();
            return true;
        }
        false
    });
    let queue_after_boundary = engine.queue_len();

    drive(&engine, &cap, &mut events, "track B to end", |e, _, _| e.queue_len() == 0);
    drive(&engine, &cap, &mut events, "the tail to play out", |_, c, _| c.available() < STRIDE);

    Run {
        captured: cap.captured(),
        events,
        queue_after_enqueue,
        queue_after_boundary,
        ring_at_boundary,
    }
}

/// Compare, and say *where* if they differ.
fn assert_identical(got: &[u8], want: &[u8], case: &str) {
    if got.len() != want.len() {
        panic!(
            "{case}: {} bytes captured, {} expected ({} frames adrift)",
            got.len(),
            want.len(),
            (got.len() as i64 - want.len() as i64) / STRIDE as i64
        );
    }
    if let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{case}: first divergence at byte {at} (frame {}, {:.3} s in): got {:#04x}, want {:#04x}",
            at / STRIDE,
            (at / STRIDE) as f64 / 48_000.0,
            got[at],
            want[at]
        );
    }
}

/// The whole proof, for one source rate and one split point.
fn proof(tag: &str, rate: u32, total: usize, split: usize) {
    let (a, b) = split_pair(tag, rate, total, split);

    // Expected: each half through the same engine, the same canonical chain and the same
    // capture device — one track at a time — concatenated.
    let want_a = play_one(&a);
    let want_b = play_one(&b);
    let mut want = want_a.clone();
    want.extend_from_slice(&want_b);

    let run = play_gapless(&a, &b, want_a.len());

    assert!(
        !run.events.iter().any(|e| matches!(e, EngineEvent::Error { .. })),
        "{tag}: the gapless run reported an error: {:?}",
        run.events
    );
    assert_eq!(run.queue_after_enqueue, 2, "{tag}: enqueue must make the queue two deep");
    assert_eq!(run.queue_after_boundary, 1, "{tag}: the boundary must step the queue 2 -> 1");
    assert!(
        run.events.iter().any(|e| matches!(e, EngineEvent::TrackChanged)),
        "{tag}: no TrackChanged: {:?}",
        run.events
    );
    assert!(
        run.events.iter().any(|e| matches!(e, EngineEvent::TrackEnded)),
        "{tag}: no TrackEnded once the queue drained: {:?}",
        run.events
    );

    // The seam happened with the outgoing track genuinely still in flight — see `Run`.
    assert!(
        run.ring_at_boundary >= 32 * 1024,
        "{tag}: only {} bytes were in flight at the boundary; the overlap was not exercised",
        run.ring_at_boundary
    );
    eprintln!(
        "{tag}: boundary with {} bytes ({:.0} ms) of track A still in the ring",
        run.ring_at_boundary,
        run.ring_at_boundary as f64 / STRIDE as f64 / 48.0
    );

    assert_identical(&run.captured, &want, tag);
    assert!(want.len() > SEC_BYTES, "{tag}: the fixture is too short to prove anything");
}

// --- the tests -----------------------------------------------------------------------------

#[test]
fn a_track_boundary_is_byte_exact_at_48k() {
    // 48 kHz source: the chain's resampler is a pass-through, so this isolates the handoff
    // itself. Split well past the middle, at a sample that is a multiple of nothing.
    proof("s48_a", 48_000, 192_000, 111_109);
}

#[test]
fn a_track_boundary_is_byte_exact_at_48k_second_split_point() {
    // A different, much earlier cut: track A is now shorter than the enqueue lead, so B is
    // queued almost immediately and sits pre-rolled for most of A's playback — the long-lived
    // pre-roll case rather than the just-in-time one.
    proof("s48_b", 48_000, 192_000, 52_631);
}

#[test]
fn a_track_boundary_is_byte_exact_across_the_resampler_at_44100() {
    // 44.1 kHz source: every buffer crosses `audioresample` (44100 -> 48000) on its way to the
    // device, and each track's resampler is its own. The expected side is built the same way, so
    // the equality still holds exactly — see the module docs on what that does and does not
    // claim.
    proof("s441", 44_100, 176_400, 97_231);
}

// --- the discontinuity ----------------------------------------------------------------------

/// A deliberate discontinuity must not leak the abandoned track into the next one.
///
/// This is the other half of the ring's story. At a *natural* boundary the outgoing track's tail
/// is precious and is deliberately preserved. At a `stop` or a `play_now` it is the opposite:
/// half a second of a track the user just left would otherwise be the first thing they hear of
/// the one they chose. The engine flushes it — and this test checks the result at byte level,
/// because "sounds about right" cannot distinguish a flush that landed from one that did not.
#[test]
fn a_discontinuity_does_not_leak_the_previous_track_into_the_next() {
    let long = write_flac("x_long.flac", 48_000, &signal(48_000, 144_000));
    let next = write_flac("x_next.flac", 48_000, &signal(44_100, 48_000));

    // What the next track sounds like from a clean start.
    let want = play_one(&next);

    let (out, cap) = open_capture(AudioOutConfig::default());
    let engine = Engine::with_output(out);
    let mut events = Vec::new();
    engine.play_now(Track::file(&long));
    drive(&engine, &cap, &mut events, "the long track to start", |e, _, _| e.is_playing());
    // Let it get properly under way, with the ring full of audio that is about to be abandoned.
    drive(&engine, &cap, &mut events, "the ring to fill", |_, c, _| {
        c.captured_len() > 128 * 1024 && c.available() > 128 * 1024
    });
    let abandoned = cap.available();
    assert!(abandoned > 128 * 1024, "nothing would have leaked anyway");

    engine.stop();
    cap.clear();
    engine.play_now(Track::file(&next));
    drive(&engine, &cap, &mut events, "the next track to start", |e, _, _| e.is_playing());
    drive(&engine, &cap, &mut events, "the next track to end", |e, _, _| e.queue_len() == 0);
    drive(&engine, &cap, &mut events, "its tail to play out", |_, c, _| c.available() < STRIDE);

    let got = cap.captured();
    assert!(
        got.len() >= want.len(),
        "only {} bytes recorded, the next track alone is {}",
        got.len(),
        want.len()
    );
    // Everything before the next track must be silence — the device was parked, and the
    // abandoned tail was dropped rather than played.
    let head = &got[..got.len() - want.len()];
    if let Some(at) = head.iter().position(|&b| b != 0) {
        panic!(
            "{} bytes of the abandoned track leaked before the new one (first at {at}, {} were queued)",
            head.len() - at,
            abandoned
        );
    }
    assert_identical(&got[got.len() - want.len()..], &want, "post-discontinuity");
}
