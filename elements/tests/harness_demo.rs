//! The element [`Harness`] driving the built-in `passthrough` and `testsrc`, asserting
//! the same properties their hand-rolled pipeline tests do (`elements/tests/golden.rs`,
//! `elements/tests/transform.rs`) — but with no `Pipeline`, no threads, no temp files, and
//! no bespoke `PacketSrc`/`RecordSink` scaffolding (spec: Testing — Element harness).
//!
//! Each test is the *inline caller*: `push`/`crank` run `process()` directly and `pull`
//! reads the emitted buffers, so a transform's forwarding and a source's byte pattern are
//! provable in a handful of lines against the very same `pattern_byte`/`fold` golden the
//! pipeline tests use.

use profluens_core::element::Flow;
use profluens_core::harness::Harness;
use profluens_elements::flow::PassThrough;
use profluens_elements::testing::{fold, pattern_byte, TestSrc, FNV_OFFSET};

/// The FNV-1a digest of the first `n` `TestSrc` pattern bytes — the golden the pipeline
/// tests assert against, reused here so the harness proves the *same* content.
fn expected_hash(n: u64) -> u64 {
    let mut h = FNV_OFFSET;
    for i in 0..n {
        h = fold(h, pattern_byte(i));
    }
    h
}

#[test]
fn passthrough_forwards_bytes_unchanged() {
    // The `transform.rs` property (byte-identical forwarding) with no filesrc/filesink/temp
    // files: push three buffers, pull three identical buffers back.
    let mut h = Harness::new(PassThrough::new());
    let inputs: [&[u8]; 3] = [&[1, 2, 3], &[4, 5, 6, 7], &[8]];
    for chunk in inputs {
        let buf = h.alloc(chunk);
        assert_eq!(h.push("sink", buf).unwrap(), Flow::Ok);
        let out = h.pull("src").expect("passthrough emits one buffer per input");
        assert_eq!(out.memory.data(), chunk, "bytes forwarded unchanged");
    }
    assert!(h.pull("src").is_none(), "no stray buffers");
}

#[test]
fn passthrough_batch_forwards_all() {
    // A whole batch through at once — passthrough drains its input in one `process()`.
    use profluens_core::batch::Batch;
    use profluens_core::id::FormatId;
    let mut h = Harness::new(PassThrough::new());
    let mut batch = Batch::new(FormatId(0));
    for chunk in [&[10u8, 11][..], &[12, 13, 14][..]] {
        batch.push(h.alloc(chunk));
    }
    assert_eq!(h.push_batch(batch).unwrap(), Flow::Ok);
    let outs = h.drain_outputs();
    assert_eq!(outs.len(), 2, "both buffers forwarded");
    assert_eq!(outs[0].memory.data(), &[10, 11]);
    assert_eq!(outs[1].memory.data(), &[12, 13, 14]);
}

#[test]
fn testsrc_produces_the_pattern_then_eos() {
    // The `golden.rs` property (testsrc emits exactly `n` pattern bytes, in order) driven
    // inline: crank until EOS, fold every emitted byte, compare to the golden digest.
    let n = 200_003u64; // not a multiple of the pool slot size
    let mut h = Harness::new(TestSrc::new(n));

    let mut hash = FNV_OFFSET;
    let mut total = 0u64;
    loop {
        let flow = h.crank().expect("testsrc process");
        while let Some(buf) = h.pull("src") {
            for &b in buf.memory.data() {
                hash = fold(hash, b);
            }
            total += buf.memory.data().len() as u64;
        }
        if flow == Flow::Eos {
            break;
        }
    }
    assert_eq!(total, n, "every pattern byte produced exactly once");
    assert_eq!(hash, expected_hash(n), "content intact and in order");
}

#[test]
fn testsrc_empty_stream_is_immediate_eos() {
    // `golden.rs::empty_stream` inline: a zero-length source EOSes with no output.
    let mut h = Harness::new(TestSrc::new(0));
    assert_eq!(h.crank().unwrap(), Flow::Eos, "empty source EOSes at once");
    assert!(h.pull("src").is_none(), "no bytes emitted");
}

#[test]
fn testsrc_reads_its_total_prop_via_start() {
    // `testsrc` reads `total` in `start()` (spec: Plugins). The harness installs the prop
    // mailbox, so a `start()`-time `ctx.prop("total")` resolves — here we just prove the
    // element runs to EOS after producing the constructor total, exercising that path.
    let mut h = Harness::new(TestSrc::new(3));
    h.start().expect("start");
    let mut produced = 0u64;
    loop {
        let flow = h.crank().unwrap();
        while let Some(buf) = h.pull("src") {
            produced += buf.memory.data().len() as u64;
        }
        if flow == Flow::Eos {
            break;
        }
    }
    assert_eq!(produced, 3);
}
