//! Milestone 1: `filesrc ! filesink` copies a file byte-identically with zero
//! steady-state payload allocations (spec: Milestone applications §1).

use streamcraft_core::bus::BusMessage;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::{FileSink, FileSrc};

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("sc_m1_{}_{}.bin", tag, std::process::id()));
    p
}

#[test]
fn copies_bytes_identically_with_flat_allocation() {
    let inp = temp_path("in");
    let outp = temp_path("out");

    // ~1 MiB of non-trivial content over 128 KiB slots → several batches.
    let data: Vec<u8> = (0..1_000_003u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(&inp, &data).expect("write input");

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&inp));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");

    // Byte-identical output.
    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got.len(), data.len(), "output length matches");
    assert_eq!(got, data, "output bytes match input");

    // Bounded steady-state allocation: the pool is capped (default 4 slots) and
    // recycles, so the buffer count never grows with file size.
    let report = p.last_report().expect("report");
    assert!(
        report.pool_slot_allocations <= 4,
        "payload buffers bounded by the pool cap, got {}",
        report.pool_slot_allocations
    );
    assert!(
        report.pool_high_water <= 4,
        "never more than the pool cap live at once, got {}",
        report.pool_high_water
    );
    assert!(report.buffers >= 8, "several batches flowed: {}", report.buffers);

    // EOS was posted to the bus.
    assert!(
        matches!(p.bus().try_recv(), Some(BusMessage::Eos)),
        "pipeline posted Eos on completion"
    );

    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&outp);
}

#[test]
fn copies_empty_file() {
    let inp = temp_path("empty_in");
    let outp = temp_path("empty_out");
    std::fs::write(&inp, b"").expect("write empty input");

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&inp));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");

    let got = std::fs::read(&outp).expect("read output");
    assert!(got.is_empty(), "empty in → empty out");

    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&outp);
}
