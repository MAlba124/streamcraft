//! `growingfilesrc` — playing a file while somebody else is still writing it
//! (the progressive-download case: an app downloads a podcast episode to disk and
//! this pipeline plays it from the same path).
//!
//! What these tests pin down:
//!
//! * the stream is **byte-identical** to the final file, in order, however the
//!   writer chunks its appends (chunk sizes here are deliberately not a multiple of
//!   the pool slot, so nearly every read straddles the frontier and is clamped);
//! * catching up with the writer costs **no CPU** — the group parks on the
//!   scheduler's idle eventcount and re-checks on its 10 ms backstop tick. Proven
//!   from `SchedulerStats::parked_ns`/`parks`, which is exact and unaffected by
//!   other tests running in this binary; `growing_src_park_cpu.rs` carries the
//!   process-CPU cross-check that has to own its own test binary;
//! * a seek **beyond** the frontier waits instead of failing;
//! * every way the writer can go wrong — abort, an over-promising watermark, a
//!   file shorter than the declared total, a read that fails outright — ends the
//!   pipeline promptly, with a bus message, and never with garbage bytes.
//!
//! Test setup does ordinary blocking IO (writing fixtures); the reactor rule the
//! `disallowed_methods` lint enforces is about `Element` code, so this module
//! carries the sanctioned file-wide `#[allow]` with that justification.
#![allow(clippy::disallowed_methods)]

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::ElementId;
use profluens_core::io::{Completion, IoResult, OpId, Reactor, Submission, SyncReactor};
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;
use profluens_elements::io::{FrontierHandle, GrowingFileSrc};

// --- fixtures --------------------------------------------------------------------

/// A scratch path unique to this process and test.
fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("pf_growing_{tag}_{}.bin", std::process::id()))
}

/// Deterministic, position-dependent bytes: a wrong offset shows up as wrong bytes,
/// which a constant or a repeating pattern would hide.
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i as u64).wrapping_mul(0x9E37_79B9) as u8).collect()
}

/// Create (or truncate to) an empty file — what an app does before starting to
/// download into it.
fn create_empty(path: &Path) -> File {
    File::create(path).expect("create the growing file")
}

/// Append bytes and make them visible to a reader in another thread *before* the
/// caller advances the frontier — the writer half of the frontier contract.
fn append(path: &Path, bytes: &[u8]) {
    let mut f = OpenOptions::new().append(true).open(path).expect("open for append");
    f.write_all(bytes).expect("append");
    f.sync_data().expect("sync the append before publishing it");
}

/// Poll until `cond` or ~4 s of real time (the settle pattern from `seek.rs`, with a
/// longer budget: progress here is paced by the 10 ms park tick).
fn settle(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..2000 {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

// --- the recording sink ----------------------------------------------------------

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

static RECORD_DESC: ElementDesc = ElementDesc {
    name: "growingrecordsink",
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

/// Records every received byte, in order, for the whole run — **append-only, even
/// across a seek**.
///
/// It deliberately does *not* clear on `FlushStart`, though a real sink drops its
/// device buffer there. A consumer group parked in its ring `pop` is released by the
/// producer's push, and step A appends and step B processes that batch in the same
/// pass — before the *next* pass's top-of-loop generation check delivers
/// `FlushStart`. Post-seek bytes can therefore legitimately be recorded just ahead of
/// the flush that announces them (in-band correctness is preserved by the batch's
/// `seek_gen` stamp, which drops *stale* data; it says nothing about fresh data
/// racing ahead of the event). A recorder that cleared would eat them intermittently.
/// So the seek tests below identify the post-seek stream by *position* — the run's
/// byte total is exact and stated in each test — rather than by a flush boundary.
struct RecordSink {
    shared: Arc<Mutex<Vec<u8>>>,
}

impl RecordSink {
    fn new() -> (Self, Arc<Mutex<Vec<u8>>>) {
        let shared = Arc::new(Mutex::new(Vec::new()));
        (Self { shared: Arc::clone(&shared) }, shared)
    }
}

impl Element for RecordSink {
    fn desc(&self) -> &'static ElementDesc {
        &RECORD_DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut rec = self.shared.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            rec.extend_from_slice(buf.memory.data());
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// `growingfilesrc path ! recordsink`, plus the handles a test needs.
struct Rig {
    pipeline: Pipeline,
    src: ElementId,
    recorded: Arc<Mutex<Vec<u8>>>,
    frontier: FrontierHandle,
}

fn rig(path: &Path) -> Rig {
    let (src, frontier) = GrowingFileSrc::new(path);
    let (sink, recorded) = RecordSink::new();
    let mut pipeline = Pipeline::new();
    let s = pipeline.add(src);
    let k = pipeline.add(sink);
    pipeline.link((s, "src"), (k, "sink")).expect("link");
    Rig { pipeline, src: s, recorded, frontier }
}

/// Collect every bus message left after a run.
fn drain_bus(p: &Pipeline) -> Vec<BusMessage> {
    let mut out = Vec::new();
    while let Some(m) = p.bus().try_recv() {
        out.push(m);
    }
    out
}

/// Whether the bus carries a message of the wanted kind, attributed to `who`, whose
/// text contains `needle`.
fn has_message(msgs: &[BusMessage], who: ElementId, error: bool, needle: &str) -> bool {
    msgs.iter().any(|m| match m {
        BusMessage::Error { element, error: e } if error => {
            *element == who && format!("{e:?}").contains(needle)
        }
        BusMessage::Warning { element, error: e } if !error => {
            *element == who && format!("{e:?}").contains(needle)
        }
        _ => false,
    })
}

/// `BusMessage` is not `Debug` (it is a transport type, not a diagnostic one), so
/// assertion failures render it by hand.
fn summarize(msgs: &[BusMessage]) -> String {
    let mut s = String::new();
    for m in msgs {
        match m {
            BusMessage::Error { element, error } => {
                s.push_str(&format!("Error({element:?}, {error:?}); "));
            }
            BusMessage::Warning { element, error } => {
                s.push_str(&format!("Warning({element:?}, {error:?}); "));
            }
            BusMessage::Eos => s.push_str("Eos; "),
            _ => s.push_str("<other>; "),
        }
    }
    s
}

// --- 1. the headline case --------------------------------------------------------

/// A writer appends 12 chunks with a pause after each, advancing the frontier once
/// the bytes are on disk; the pipeline reads the file *concurrently* and the sink
/// sees exactly the final file, in order.
///
/// The 40 000-byte chunk is deliberately unrelated to the 128 KiB pool slot: every
/// read the source issues is larger than what the frontier permits, so this exercises
/// the straddle clamp + short-read repair on essentially every buffer rather than on
/// one tail read.
#[test]
fn growing_file_streams_byte_identically_while_the_writer_appends() {
    const CHUNKS: usize = 12;
    const CHUNK: usize = 40_000;
    let path = tmp("append");
    create_empty(&path);
    let data = pattern(CHUNKS * CHUNK);

    let r = rig(&path);
    let (frontier, writer_path, writer_data) = (r.frontier.clone(), path.clone(), data.clone());
    let writer = std::thread::spawn(move || {
        let mut written = 0usize;
        for i in 0..CHUNKS {
            append(&writer_path, &writer_data[i * CHUNK..(i + 1) * CHUNK]);
            written += CHUNK;
            frontier.advance(written as u64);
            std::thread::sleep(Duration::from_millis(5));
        }
        frontier.finish(written as u64);
    });

    let mut p = r.pipeline;
    p.run().expect("run");
    writer.join().expect("writer joined");

    assert_eq!(
        *r.recorded.lock().unwrap(),
        data,
        "the stream must be byte-identical to the final file, in order"
    );
    let _ = std::fs::remove_file(&path);
}

// --- 2. finish() ends the stream -------------------------------------------------

/// `finish()` before the run even starts: the source reads to the declared total and
/// stops. The clean-EOS half of the contract, without any concurrency in the way.
#[test]
fn finish_ends_the_stream_with_the_complete_file() {
    let path = tmp("finish");
    create_empty(&path);
    let data = pattern(300_000);
    append(&path, &data);

    let r = rig(&path);
    r.frontier.advance(data.len() as u64);
    r.frontier.finish(data.len() as u64);
    assert!(r.frontier.is_finished(), "the handle reports the finished state");
    assert_eq!(r.frontier.downloaded(), data.len() as u64, "finish implies a final advance");

    let mut p = r.pipeline;
    p.run().expect("run");

    assert_eq!(*r.recorded.lock().unwrap(), data, "every byte, exactly once");
    assert!(
        drain_bus(&p).iter().any(|m| matches!(m, BusMessage::Eos)),
        "a finished growing file ends the pipeline through the ordinary EOS path"
    );
    let _ = std::fs::remove_file(&path);
}

// --- 3. the frontier wait parks -------------------------------------------------

/// The writer stalls for 500 ms mid-file. Playback must resume when it continues, and
/// **while stalled the group must be parked, not spinning**.
///
/// Measured from `SchedulerStats`: `parked_ns` is only ever incremented from inside
/// `park_idle`, so a group that spun through empty passes would leave it flat. The
/// stall is also the case with no wake source at all — a `FrontierHandle` is a plain
/// `Arc` that knows nothing about any pipeline — so the parks end on the 10 ms
/// backstop tick, and `tick_expiries` climbing with `parks` is that mechanism being
/// exercised rather than a defect.
#[test]
fn a_stalled_writer_leaves_the_group_parked_not_spinning() {
    const HALF: usize = 200_000;
    let path = tmp("stall");
    create_empty(&path);
    let data = pattern(2 * HALF);
    append(&path, &data[..HALF]);

    let r = rig(&path);
    r.frontier.advance(HALF as u64);
    let mut p = r.pipeline;
    let tap = p.tap_handle();
    let recorded = Arc::clone(&r.recorded);
    let run = std::thread::spawn(move || p.run());

    assert!(
        settle(|| recorded.lock().unwrap().len() == HALF),
        "the first half plays out while the writer is still holding the rest"
    );

    // --- the stall ---
    let before = tap.scheduler();
    std::thread::sleep(Duration::from_millis(500));
    let after = tap.scheduler();
    let parked_ms = (after.parked_ns - before.parked_ns) / 1_000_000;
    let parks = after.parks - before.parks;
    let ticks = after.tick_expiries - before.tick_expiries;

    assert_eq!(
        recorded.lock().unwrap().len(),
        HALF,
        "nothing beyond the frontier may be emitted while the writer is stalled"
    );
    // Two groups (source + sink) park through a 500 ms stall, so ~1000 ms is the
    // expectation; 300 ms keeps a wide non-flaky margin while a spin — which parks
    // never, leaving both deltas at zero — fails loudly.
    assert!(
        parked_ms >= 300,
        "the group spent only {parked_ms} ms parked across a 500 ms stall — spinning?"
    );
    assert!(parks >= 20, "only {parks} parks in a 500 ms stall (expect ~50/group at 10 ms)");
    assert!(
        ticks >= 20,
        "only {ticks} of {parks} parks ended on the backstop tick — the frontier wait has no \
         wake source, so the tick is what must be advancing it"
    );

    // --- the writer continues ---
    append(&path, &data[HALF..]);
    r.frontier.advance(data.len() as u64);
    r.frontier.finish(data.len() as u64);

    let result = run.join().expect("the run thread joined");
    result.expect("run");
    assert_eq!(*r.recorded.lock().unwrap(), data, "playback resumed and completed the file");
    let _ = std::fs::remove_file(&path);
}

// --- 4. seek backwards ----------------------------------------------------------

/// Seeking back into already-downloaded territory yields the right bytes from the
/// target. The file is fully downloaded but *not* finished, so the source is still in
/// its growing mode throughout the seek.
#[test]
fn seek_back_into_downloaded_territory_replays_from_the_target() {
    const TOTAL: usize = 400_000;
    const TARGET: usize = 137_000; // unaligned to the pool slot on purpose
    let path = tmp("seekback");
    create_empty(&path);
    let data = pattern(TOTAL);
    append(&path, &data);

    let r = rig(&path);
    r.frontier.advance(TOTAL as u64);
    let mut p = r.pipeline;
    let seek = p.seek_handle();
    let recorded = Arc::clone(&r.recorded);
    let run = std::thread::spawn(move || p.run());

    assert!(settle(|| recorded.lock().unwrap().len() == TOTAL), "the whole downloaded region plays");
    // The source has emitted everything up to the frontier and is parked, so the
    // recording is exactly the file — which makes the post-seek total exact too.
    seek.seek(TARGET as u64, Timestamp::ZERO);
    assert!(
        settle(|| recorded.lock().unwrap().len() == 2 * TOTAL - TARGET),
        "the source must re-read from the seek target up to the frontier"
    );
    r.frontier.finish(TOTAL as u64);

    run.join().expect("the run thread joined").expect("a backward seek must not fail the run");
    let mut expected = data.clone();
    expected.extend_from_slice(&data[TARGET..]);
    assert_eq!(
        *r.recorded.lock().unwrap(),
        expected,
        "the post-seek stream is the file from the target on — the right bytes, from the right \
         offset, exactly once"
    );
    let _ = std::fs::remove_file(&path);
}

// --- 5. seek beyond the frontier ------------------------------------------------

/// Seeking **past** the watermark is a wait, never an error: the source adopts the
/// target and sits at the frontier until the writer catches up — which is exactly
/// what an app whose downloader services out-of-range fetches itself needs.
#[test]
fn seek_beyond_the_frontier_waits_for_the_writer_instead_of_failing() {
    const QUARTER: usize = 100_000;
    const TARGET: usize = 3 * QUARTER;
    let path = tmp("seekahead");
    create_empty(&path);
    let data = pattern(4 * QUARTER);
    append(&path, &data[..QUARTER]);

    let r = rig(&path);
    r.frontier.advance(QUARTER as u64);
    let mut p = r.pipeline;
    let seek = p.seek_handle();
    let tap = p.tap_handle();
    let recorded = Arc::clone(&r.recorded);
    let run = std::thread::spawn(move || p.run());

    assert!(settle(|| recorded.lock().unwrap().len() == QUARTER), "the downloaded quarter plays");

    // Target 300 000 while only 100 000 bytes are published.
    seek.seek(TARGET as u64, Timestamp::ZERO);
    let before = tap.scheduler();
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        !run.is_finished(),
        "a beyond-frontier seek must not end the run — the bytes are simply not there yet"
    );
    assert_eq!(
        recorded.lock().unwrap().len(),
        QUARTER,
        "not one byte may be emitted while the target is still beyond the watermark"
    );
    assert!(
        tap.scheduler().parked_ns > before.parked_ns,
        "the beyond-frontier wait is the ordinary idle park, not a spin"
    );

    // The app's downloader fetches the rest; from here the bytes just arrive.
    append(&path, &data[QUARTER..]);
    r.frontier.advance(data.len() as u64);
    r.frontier.finish(data.len() as u64);

    run.join().expect("the run thread joined").expect("run");
    let mut expected = data[..QUARTER].to_vec();
    expected.extend_from_slice(&data[TARGET..]);
    assert_eq!(
        *r.recorded.lock().unwrap(),
        expected,
        "playback resumed at the seek target once the watermark covered it"
    );
    let _ = std::fs::remove_file(&path);
}

// --- 6. abort ------------------------------------------------------------------

/// A failed download: `abort()` posts one attributed error and then ends the stream,
/// so the pipeline winds down through EOS instead of hanging at the frontier.
#[test]
fn abort_posts_an_error_and_winds_the_pipeline_down() {
    const HAVE: usize = 150_000;
    let path = tmp("abort");
    create_empty(&path);
    let data = pattern(HAVE);
    append(&path, &data);

    let r = rig(&path);
    r.frontier.advance(HAVE as u64);
    let p = r.pipeline;
    let recorded = Arc::clone(&r.recorded);
    let run = std::thread::spawn(move || {
        let mut p = p;
        let result = p.run();
        (result, p)
    });

    assert!(settle(|| recorded.lock().unwrap().len() == HAVE), "the downloaded prefix plays");
    r.frontier.abort();

    assert!(settle(|| run.is_finished()), "abort must end the run, not leave it at the frontier");
    let (result, p) = run.join().expect("the run thread joined");
    result.expect("abort winds down through EOS, so the run itself succeeds");

    let msgs = drain_bus(&p);
    assert!(
        has_message(&msgs, r.src, true, "aborted the download"),
        "abort must put one clear Error on the bus, attributed to the source: {}", summarize(&msgs)
    );
    assert_eq!(*r.recorded.lock().unwrap(), data, "the bytes that did arrive were not lost");
    let _ = std::fs::remove_file(&path);
}

// --- 7. zero-length start, and a watermark that outruns the file ----------------

/// The trust model under stress. The file is empty when the pipeline starts, and the
/// writer then publishes a watermark for bytes it has not written — a contract
/// violation (`downloaded()` must never exceed the bytes actually readable).
///
/// Required behaviour: no garbage (nothing is emitted for the phantom region), no hot
/// loop (the reads that come back empty must still leave the group parked), and
/// self-healing — when the bytes finally land the stream continues **without** the
/// writer having to move the watermark again.
#[test]
fn an_over_promising_watermark_degrades_to_parked_retries_not_garbage() {
    const LEN: usize = 4096;
    let path = tmp("overpromise");
    create_empty(&path); // zero-length at start: legal, and the usual case
    let data = pattern(LEN);

    let r = rig(&path);
    let mut p = r.pipeline;
    let tap = p.tap_handle();
    let recorded = Arc::clone(&r.recorded);
    let run = std::thread::spawn(move || p.run());

    // Claim 4096 readable bytes for a file that has none.
    r.frontier.advance(LEN as u64);
    let before = tap.scheduler();
    std::thread::sleep(Duration::from_millis(250));
    let after = tap.scheduler();

    assert!(
        recorded.lock().unwrap().is_empty(),
        "a watermark ahead of the real bytes must never produce bytes"
    );
    let parked_ms = (after.parked_ns - before.parked_ns) / 1_000_000;
    assert!(
        parked_ms >= 100,
        "only {parked_ms} ms parked over a 250 ms over-promised stall — the empty reads are \
         looping hot instead of being paced by the park tick"
    );

    // The bytes finally arrive. The watermark is NOT advanced again (it already
    // covers them) — the paced retry is what must notice.
    append(&path, &data);
    assert!(
        settle(|| recorded.lock().unwrap().len() == LEN),
        "the retry must pick the bytes up without a further advance()"
    );
    r.frontier.finish(LEN as u64);

    run.join().expect("the run thread joined").expect("run");
    assert_eq!(*r.recorded.lock().unwrap(), data, "and the bytes are the right ones");
    let _ = std::fs::remove_file(&path);
}

// --- 8a. a file shorter than the declared total ---------------------------------

/// Hostile: the writer declares a total the file does not contain (a truncated or
/// half-written download). The source emits the bytes that really exist, warns, and
/// ends the stream — it must not hang waiting for bytes that will never come, and
/// must not invent any.
#[test]
fn a_file_shorter_than_the_declared_total_warns_and_ends_cleanly() {
    const REAL: usize = 4096;
    const CLAIMED: u64 = 8192;
    let path = tmp("truncated");
    create_empty(&path);
    let data = pattern(REAL);
    append(&path, &data);

    let r = rig(&path);
    r.frontier.finish(CLAIMED); // twice what is there
    let mut p = r.pipeline;
    p.run().expect("a short file is a warning, not a failed run");

    assert_eq!(*r.recorded.lock().unwrap(), data, "exactly the bytes the file really had");
    let msgs = drain_bus(&p);
    assert!(
        has_message(&msgs, r.src, false, "is shorter than"),
        "the truncation must be reported on the bus: {}", summarize(&msgs)
    );
    let _ = std::fs::remove_file(&path);
}

// --- 8b. a read that fails outright ---------------------------------------------

/// A reactor decorator that lets `pass` completions through and then fails every
/// read — the file going bad under an already-open fd (a stale handle, a device
/// pulled out from under a download). Unlinking the path cannot be used to provoke
/// this: the source's fd survives it, so the read would simply succeed.
struct FailingReactor {
    inner: SyncReactor,
    remaining: Arc<AtomicU32>,
}

impl Reactor for FailingReactor {
    fn set_file(&mut self, element: ElementId, file: File) {
        self.inner.set_file(element, file);
    }

    fn submit(&mut self, subs: &mut Vec<Submission>) {
        self.inner.submit(subs);
    }

    fn cancel(&mut self, op: OpId) {
        self.inner.cancel(op);
    }

    fn is_idle(&self) -> bool {
        self.inner.is_idle()
    }

    fn run_once(&mut self, out: &mut Vec<(ElementId, Completion)>) {
        self.inner.run_once(out);
        for (_, c) in out.iter_mut() {
            if self.remaining.load(Ordering::Relaxed) == 0 {
                c.buf.memory.set_len(0);
                c.result = IoResult::Err(std::io::ErrorKind::NotFound);
            } else {
                self.remaining.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

/// A failing read is fatal and says so: an attributed `Error` on the bus, a failed
/// run, a bounded exit, no panic.
#[test]
fn a_failing_read_reports_an_error_and_exits_without_hanging() {
    let path = tmp("readerror");
    create_empty(&path);
    let data = pattern(600_000);
    append(&path, &data);

    let r = rig(&path);
    r.frontier.advance(data.len() as u64);
    let mut p = r.pipeline;
    // One good read, then the fd goes bad. The counter is shared so the per-group
    // reactors the factory builds cannot each get their own budget.
    let budget = Arc::new(AtomicU32::new(1));
    p.set_reactor_factory(Arc::new(move || {
        Ok(Box::new(FailingReactor {
            inner: SyncReactor::new(),
            remaining: Arc::clone(&budget),
        }) as Box<dyn Reactor>)
    }));

    let run = std::thread::spawn(move || {
        let result = p.run();
        (result, p)
    });
    assert!(settle(|| run.is_finished()), "a failing read must end the run promptly");
    let (result, p) = run.join().expect("the run thread joined — no panic");

    let err = result.expect_err("a failed read must fail the run");
    assert!(
        format!("{err:?}").contains("read of"),
        "the error must name the failing read: {err:?}"
    );
    let msgs = drain_bus(&p);
    assert!(
        has_message(&msgs, r.src, true, "read of"),
        "the read failure must be attributed to the source on the bus: {}", summarize(&msgs)
    );
    assert!(
        r.recorded.lock().unwrap().len() < data.len(),
        "the stream was cut short, not silently completed"
    );
    let _ = std::fs::remove_file(&path);
}

// --- 9. the abandoned download --------------------------------------------------

/// Dropping every `FrontierHandle` without `finish()` or `abort()` — the downloader
/// task panicked, or a name-constructed `growingfilesrc` never had a handle at all —
/// is treated as an abandoned download: a warning and a clean end, never a hang.
#[test]
fn dropping_every_handle_ends_the_stream_instead_of_hanging() {
    let path = tmp("abandoned");
    create_empty(&path);
    let data = pattern(50_000);
    append(&path, &data);

    let r = rig(&path);
    r.frontier.advance(data.len() as u64);
    let Rig { pipeline, src, recorded, frontier } = r;
    let mut p = pipeline;
    let watcher = Arc::clone(&recorded);
    let run = std::thread::spawn(move || {
        let result = p.run();
        (result, p)
    });

    assert!(settle(|| watcher.lock().unwrap().len() == data.len()), "the published bytes play");
    drop(frontier); // the downloader is gone

    assert!(settle(|| run.is_finished()), "an abandoned download must not hang the pipeline");
    let (result, p) = run.join().expect("the run thread joined");
    result.expect("an abandoned download winds down through EOS");
    assert!(
        has_message(&drain_bus(&p), src, false, "abandoned"),
        "the abandonment must be reported on the bus"
    );
    assert_eq!(*recorded.lock().unwrap(), data, "the bytes that did arrive were delivered");
    let _ = std::fs::remove_file(&path);
}
