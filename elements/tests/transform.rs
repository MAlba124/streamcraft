//! Validates the scheduler's passive-inline group path: `filesrc ! passthrough !
//! filesink`. The passive `passthrough` inlines into filesrc's thread group, so the
//! group runner drives a two-element inline chain, and the output must still be
//! byte-identical.

use profluens_core::pipeline::Pipeline;
use profluens_elements::flow::PassThrough;
use profluens_elements::io::{FileSink, FileSrc};

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf_tf_{}_{}.bin", tag, std::process::id()));
    p
}

#[test]
fn passthrough_copy_is_byte_identical() {
    let inp = temp_path("in");
    let outp = temp_path("out");
    let data: Vec<u8> = (0..1_000_003u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(&inp, &data).expect("write input");

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&inp));
    let mid = p.add(PassThrough::new());
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (mid, "sink")).expect("link src->mid");
    p.link((mid, "src"), (sink, "sink")).expect("link mid->sink");
    p.run().expect("run");

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got, data, "passthrough forwarded bytes unchanged");

    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&outp);
}
