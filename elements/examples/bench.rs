//! Rough throughput benchmarks. Run in release:
//!   cargo run --release --features io-uring --example bench -p streamcraft-elements
//!
//! Numbers are page-cache-warm (no disk sync) so file copy is an apples-to-apples
//! framework-overhead comparison against `cp`, not a disk benchmark.

use std::time::Instant;

use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::{FileSink, FileSrc};
use streamcraft_elements::testing::{TestSink, TestSrc};

fn gbps(bytes: u64, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

fn main() {
    // 0. Ring primitive: pure lock-free SPSC handoff of u64 across two threads.
    {
        use streamcraft_core::ring::spsc;
        let n = 200_000_000u64;
        let (p, c) = spsc::<u64>(1024);
        let prod = std::thread::spawn(move || {
            for i in 0..n {
                while p.try_push(i).is_err() {}
            }
        });
        let t = Instant::now();
        let cons = std::thread::spawn(move || {
            let mut got = 0u64;
            while got < n {
                if c.try_pop().is_some() {
                    got += 1;
                }
            }
        });
        prod.join().unwrap();
        cons.join().unwrap();
        let dt = t.elapsed().as_secs_f64();
        println!(
            "ring SPSC (u64)          : {:7.1} M items/s   ({:.1} ns/item)",
            n as f64 / dt / 1e6,
            dt * 1e9 / n as f64
        );
    }

    // 1. In-process pipeline throughput: testsrc ! testsink (generate + FNV-hash
    //    every byte + cross a thread boundary via the ring).
    {
        let n = 2u64 << 30; // 2 GiB
        let (sink, stats) = TestSink::new();
        let mut p = Pipeline::new();
        let src = p.add(TestSrc::new(n));
        let snk = p.add(sink);
        p.link((src, "src"), (snk, "sink")).unwrap();
        let t = Instant::now();
        p.run().unwrap();
        let dt = t.elapsed().as_secs_f64();
        assert_eq!(stats.bytes(), n);
        let bufs = p.counters(src).buffers_out;
        println!(
            "pipeline testsrc!testsink: {:7.2} GB/s      ({:.1} ns/buffer, {} buffers)",
            gbps(n, dt),
            dt * 1e9 / bufs as f64,
            bufs
        );
    }

    // 2. File copy vs cp (page-cache warm).
    let n: u64 = 1 << 30; // 1 GiB
    let inp = std::env::temp_dir().join("sc_bench_in.bin");
    let outp = std::env::temp_dir().join("sc_bench_out.bin");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&inp).unwrap();
        let chunk = vec![0xABu8; 1 << 20];
        let mut w = 0u64;
        while w < n {
            let take = ((n - w) as usize).min(chunk.len());
            f.write_all(&chunk[..take]).unwrap();
            w += take as u64;
        }
        f.sync_all().unwrap();
    }

    {
        let mut p = Pipeline::new();
        let s = p.add(FileSrc::new(&inp));
        let k = p.add(FileSink::new(&outp));
        p.link((s, "src"), (k, "sink")).unwrap();
        let t = Instant::now();
        p.run().unwrap();
        println!("file copy (SyncReactor)  : {:7.2} GB/s", gbps(n, t.elapsed().as_secs_f64()));
    }

    #[cfg(feature = "io-uring")]
    {
        use streamcraft_elements::io::IoUringReactor;
        let mut p = Pipeline::new();
        p.set_reactor_factory(std::sync::Arc::new(|| {
            Ok(Box::new(IoUringReactor::new()?) as Box<dyn streamcraft_core::io::Reactor>)
        }));
        let s = p.add(FileSrc::new(&inp));
        let k = p.add(FileSink::new(&outp));
        p.link((s, "src"), (k, "sink")).unwrap();
        let t = Instant::now();
        p.run().unwrap();
        println!("file copy (io_uring)     : {:7.2} GB/s", gbps(n, t.elapsed().as_secs_f64()));
    }

    {
        let t = Instant::now();
        std::process::Command::new("cp").arg(&inp).arg(&outp).status().unwrap();
        println!("file copy (cp reference) : {:7.2} GB/s", gbps(n, t.elapsed().as_secs_f64()));
    }

    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&outp);
}
