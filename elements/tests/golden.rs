//! Golden pipeline tests using the seedable `TestSrc` and checksumming `TestSink`
//! (spec: Testing). No files, no sockets — this exercises the scheduler's non-IO
//! path and verifies the data arrives with no loss, duplication, or reordering.

use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::flow::PassThrough;
use streamcraft_elements::testing::{fold, pattern_byte, TestSink, TestSrc, FNV_OFFSET};

/// The digest `TestSink` should compute for the first `n` pattern bytes.
fn expected_hash(n: u64) -> u64 {
    let mut h = FNV_OFFSET;
    for i in 0..n {
        h = fold(h, pattern_byte(i));
    }
    h
}

#[test]
fn testsrc_to_testsink_transports_exactly() {
    let n = 1_000_003u64; // not a multiple of the buffer size
    let (sink, stats) = TestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TestSrc::new(n));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.run().expect("run");

    assert!(stats.is_done(), "sink saw EOS");
    assert_eq!(stats.bytes(), n, "every byte arrived exactly once");
    assert_eq!(stats.hash(), expected_hash(n), "content intact and in order");
}

#[test]
fn testsrc_passthrough_testsink_matches() {
    // A passive passthrough inlines into testsrc's group; the digest is unchanged.
    let n = 500_009u64;
    let (sink, stats) = TestSink::new();

    let mut p = Pipeline::new();
    let src = p.add(TestSrc::new(n));
    let mid = p.add(PassThrough::new());
    let snk = p.add(sink);
    p.link((src, "src"), (mid, "sink")).expect("link 1");
    p.link((mid, "src"), (snk, "sink")).expect("link 2");
    p.run().expect("run");

    assert_eq!(stats.bytes(), n);
    assert_eq!(stats.hash(), expected_hash(n));
}

#[test]
fn empty_stream() {
    let (sink, stats) = TestSink::new();
    let mut p = Pipeline::new();
    let src = p.add(TestSrc::new(0));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.run().expect("run");
    assert!(stats.is_done());
    assert_eq!(stats.bytes(), 0);
    assert_eq!(stats.hash(), FNV_OFFSET, "no bytes → offset basis unchanged");
}

#[test]
fn counters_track_throughput() {
    let n = 300_007u64;
    let (sink, _stats) = TestSink::new();
    let mut p = Pipeline::new();
    let src = p.add(TestSrc::new(n));
    let snk = p.add(sink);
    p.link((src, "src"), (snk, "sink")).expect("link");
    p.run().expect("run");

    let src_c = p.counters(src);
    let snk_c = p.counters(snk);
    assert_eq!(src_c.bytes_out, n, "source produced exactly n bytes");
    assert_eq!(snk_c.bytes_in, n, "sink consumed exactly n bytes");
    assert!(src_c.buffers_out >= 1);
    assert_eq!(
        snk_c.buffers_in, src_c.buffers_out,
        "sink received the same buffer count the source produced"
    );
    assert_eq!(src_c.bytes_in, 0, "a source has no input");
    assert_eq!(snk_c.bytes_out, 0, "a sink has no output");
}

#[test]
fn dump_dot_shows_topology_and_groups() {
    let (sink, _stats) = TestSink::new();
    let mut p = Pipeline::new();
    let src = p.add(TestSrc::new(0));
    let mid = p.add(PassThrough::new());
    let snk = p.add(sink);
    p.link((src, "src"), (mid, "sink")).unwrap();
    p.link((mid, "src"), (snk, "sink")).unwrap();

    let dot = p.dump_dot();
    assert!(dot.contains("digraph streamcraft"));
    assert!(dot.contains("testsrc") && dot.contains("passthrough") && dot.contains("testsink"));
    assert!(dot.contains("e0 -> e1") && dot.contains("e1 -> e2"));
    // testsrc (active) + passthrough (passive) share group 0; testsink is group 1.
    assert!(dot.contains("cluster_0") && dot.contains("cluster_1"));
}
