//! `filesrc` against the *whole* [`Reactor`] contract, not just `SyncReactor`'s behaviour.
//!
//! The trait promises nothing about completion order, and `IoUringReactor`'s own module doc
//! says so outright ("positioned (offset-carrying) ops make completion order irrelevant") —
//! io_uring reaps CQEs in whatever order the kernel finished them, which for a mix of
//! page-cache hits (served inline) and misses (punted to io-wq) is not submission order.
//! `filesrc` keeps `credits` reads in flight at once, so if it pushes completions in arrival
//! order the byte stream is silently permuted.
//!
//! Likewise `IoResult::Ok(n)` is documented as "bytes transferred", not "the buffer was
//! filled": `read_at` is one `pread(2)`, which is allowed to come up short (network
//! filesystems, a signal landing mid-transfer). `filesrc` advances its read offset by the
//! buffer *capacity*, so a short read leaves a hole.
//!
//! Both shims below are legal `Reactor` implementations. They wrap the real `SyncReactor`
//! and only perturb what the contract leaves free.

use std::fs::File;

use profluens_core::id::ElementId;
use profluens_core::io::{Completion, OpId, Reactor, Submission, SyncReactor};
use profluens_core::pipeline::Pipeline;
use profluens_elements::io::{FileSink, FileSrc};

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf_contract_{}_{}.bin", tag, std::process::id()));
    p
}

/// Reverses each pass's completions — a reactor that finishes the ops it was handed in the
/// opposite order. Nothing in the trait forbids it.
struct ReverseReactor(SyncReactor);

impl Reactor for ReverseReactor {
    fn set_file(&mut self, element: ElementId, file: File) {
        self.0.set_file(element, file)
    }
    fn submit(&mut self, subs: &mut Vec<Submission>) {
        self.0.submit(subs)
    }
    fn cancel(&mut self, op: OpId) {
        self.0.cancel(op)
    }
    fn is_idle(&self) -> bool {
        self.0.is_idle()
    }
    fn run_once(&mut self, out: &mut Vec<(ElementId, Completion)>) {
        self.0.run_once(out);
        out.reverse();
    }
}

/// Truncates every read to half the buffer — a reactor whose `pread` comes up short.
/// `IoResult::Ok(n)` with `0 < n < capacity` is exactly what the trait documents.
struct ShortReadReactor(SyncReactor);

impl Reactor for ShortReadReactor {
    fn set_file(&mut self, element: ElementId, file: File) {
        self.0.set_file(element, file)
    }
    fn submit(&mut self, subs: &mut Vec<Submission>) {
        self.0.submit(subs)
    }
    fn cancel(&mut self, op: OpId) {
        self.0.cancel(op)
    }
    fn is_idle(&self) -> bool {
        self.0.is_idle()
    }
    fn run_once(&mut self, out: &mut Vec<(ElementId, Completion)>) {
        self.0.run_once(out);
        for (_, c) in out.iter_mut() {
            if let profluens_core::io::IoResult::Ok(n) = c.result {
                if n > 1 {
                    let half = n / 2;
                    c.buf.memory.set_len(half);
                    c.result = profluens_core::io::IoResult::Ok(half);
                }
            }
        }
    }
}

fn copy_with<R: Reactor + 'static>(
    tag: &str,
    data: &[u8],
    make: impl Fn() -> R + Send + Sync + 'static,
) -> Vec<u8> {
    let inp = temp_path(&format!("{tag}_in"));
    let outp = temp_path(&format!("{tag}_out"));
    std::fs::write(&inp, data).expect("write input");

    let mut p = Pipeline::new();
    p.set_reactor_factory(std::sync::Arc::new(move || Ok(Box::new(make()) as Box<dyn Reactor>)));
    let src = p.add(FileSrc::new(&inp));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");

    let got = std::fs::read(&outp).expect("read output");
    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&outp);
    got
}

/// ~1 MiB over the default 128 KiB slots, several reads in flight per pass.
fn fixture() -> Vec<u8> {
    (0..1_000_003u32).map(|i| (i % 251) as u8).collect()
}

/// Compare without dumping a megabyte into the failure message.
fn assert_same(got: &[u8], want: &[u8], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length differs");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{what}: first difference at byte {i} (got {}, want {})", got[i], want[i]);
    }
}

#[test]
fn out_of_order_completions_do_not_permute_the_stream() {
    let data = fixture();
    let got = copy_with("reverse", &data, || ReverseReactor(SyncReactor::new()));
    assert_same(&got, &data, "filesrc must re-sequence completions, not trust arrival order");
}

#[test]
fn short_reads_do_not_punch_holes() {
    let data = fixture();
    let got = copy_with("short", &data, || ShortReadReactor(SyncReactor::new()));
    assert_same(&got, &data, "a short read must resume at off+n, not off+capacity");
}
