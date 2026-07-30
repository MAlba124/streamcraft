//! End-to-end logging wiring (spec: Debuggability — Logging). Drives a real
//! `testsrc ! testsink` pipeline with logging enabled and proves that:
//! - enabling logging spins up the per-element channels + the single drain thread and
//!   the run still completes with byte-identical transport (observation never perturbs
//!   what it observes), and
//! - the disabled default and the enabled path produce the same result, so turning
//!   logging on changes nothing but what is emitted.
//!
//! The record-reaches-drain and gate-suppression proofs live in
//! `profluens-core`'s `ctx` unit tests, which read a `LogDrain` directly (the
//! `Ctx`→`Log` install is crate-internal, so it can't be reached from here). Here we
//! confirm the pipeline actually wires those pieces under a real run and joins the
//! drain thread cleanly on shutdown. Records are formatted to stderr; the test asserts
//! on the transported data, not on that text.

use profluens_core::log::Level;
use profluens_core::pipeline::Pipeline;
use profluens_elements::testing::{fold, pattern_byte, TestSink, TestSrc, FNV_OFFSET};

fn expected_hash(n: u64) -> u64 {
    let mut h = FNV_OFFSET;
    for i in 0..n {
        h = fold(h, pattern_byte(i));
    }
    h
}

fn run_and_digest(n: u64, level: Option<Level>) -> (u64, u64) {
    let (sink, stats) = TestSink::new();
    let mut p = Pipeline::new();
    let src = p.add(TestSrc::new(n));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    if let Some(l) = level {
        p.log_to_stderr(l);
    }
    p.run().expect("run completes with logging wired");
    assert!(stats.is_done(), "sink saw EOS");
    (stats.bytes(), stats.hash())
}

#[test]
fn run_with_trace_logging_is_byte_identical_and_terminates() {
    // Trace enables every level, so testsrc emits `start`/`eos` and each group logs
    // `group_done`; the drain thread must spawn, format them, and join at shutdown
    // without the run hanging or losing data.
    let n = 1_000_003u64; // not a multiple of the buffer size
    let (bytes, hash) = run_and_digest(n, Some(Level::Trace));
    assert_eq!(bytes, n, "every byte arrived exactly once with logging on");
    assert_eq!(hash, expected_hash(n), "content intact and in order");
}

#[test]
fn logging_on_and_off_agree() {
    // Enabling logging must not perturb the stream: same digest either way.
    let n = 300_007u64;
    let off = run_and_digest(n, None);
    let on = run_and_digest(n, Some(Level::Debug));
    assert_eq!(off, on, "logging changed only what is emitted, not the data");
    assert_eq!(off.1, expected_hash(n));
}

#[test]
fn empty_stream_with_logging_terminates() {
    // The drain thread must exit cleanly even when almost nothing is logged.
    let (bytes, hash) = run_and_digest(0, Some(Level::Trace));
    assert_eq!(bytes, 0);
    assert_eq!(hash, FNV_OFFSET);
}
