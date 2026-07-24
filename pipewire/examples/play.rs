//! Play an audio file: `filesrc ! flacdec ! pipewireaudiosink` (spec: Milestone
//! applications — play an audio file), with a live progress line and pause control.
//!
//! Usage: `cargo run --release -p sc-pipewire --example play -- <file.flac>`
//! Controls (type, then Enter): a blank line toggles pause/resume; `q` quits.
//!
//! Requires a running PipeWire session — it plays to the default output. `flacdec`
//! announces the file's `audio/raw` format at runtime and the sink configures the device
//! from it (spec: Formats — dynamic caps); the graph is paced by the device via
//! backpressure. Pausing renders silence and holds the device buffer, so the whole pipeline
//! backpressures to a stop and resumes exactly where it left off — no audio lost.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sc_flac::{FlacDec, StreamDecoder};
use sc_pipewire::PipeWireAudioSink;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::FileSrc;

/// Best-effort total duration from the FLAC header (STREAMINFO); `None` if unknown. Reads
/// only the file's head — enough for the metadata — and parses it with the same decoder the
/// pipeline uses, so there's no separate FLAC parser to drift.
fn duration_secs(path: &str) -> Option<f64> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut head = vec![0u8; 1 << 17];
    let n = f.read(&mut head).ok()?;
    head.truncate(n);
    let mut sd = StreamDecoder::new();
    sd.push(&head);
    let _ = sd.pull(); // parses the header (populates info) even if the first frame is short
    let info = sd.info()?;
    (info.total_samples != 0 && info.sample_rate != 0)
        .then(|| info.total_samples as f64 / info.sample_rate as f64)
}

fn fmt_time(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: play <file.flac>");
        std::process::exit(2);
    });

    let total = duration_secs(&path);

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&path));
    let dec = p.add(FlacDec::new());
    let sink = PipeWireAudioSink::new();
    let ctrl = sink.control();
    let snk = p.add(sink);
    p.link((src, "src"), (dec, "sink")).expect("filesrc -> flacdec");
    p.link((dec, "src"), (snk, "sink")).expect("flacdec -> pipewireaudiosink");

    println!("playing {path} …  (Enter = pause/resume, q + Enter = quit)");

    // Controls on a helper thread: a blank line toggles pause, `q` quits.
    let stop = p.stop_handle();
    {
        let ctrl = ctrl.clone();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                match stdin.read_line(&mut line) {
                    Ok(0) | Err(_) => break, // stdin closed
                    Ok(_) => {}
                }
                match line.trim() {
                    "q" | "quit" => {
                        ctrl.resume(); // unblock if paused, so the cooperative stop lands
                        stop.stop();
                        break;
                    }
                    _ => {
                        let _ = ctrl.toggle();
                    }
                }
            }
        });
    }

    // Run the pipeline on its own thread; the main thread renders a live progress line.
    let done = Arc::new(AtomicBool::new(false));
    let run = {
        let done = Arc::clone(&done);
        std::thread::spawn(move || {
            let r = p.run();
            done.store(true, Ordering::Release);
            r
        })
    };

    while !done.load(Ordering::Acquire) {
        let pos = ctrl.position_secs();
        let clock = match total {
            Some(t) => format!("{} / {}", fmt_time(pos), fmt_time(t)),
            None => fmt_time(pos),
        };
        print!("\r{}  {}   ", if ctrl.is_paused() { "[paused]" } else { "[playing]" }, clock);
        let _ = std::io::stdout().flush();
        std::thread::sleep(Duration::from_millis(250));
    }

    match run.join().expect("run thread joined") {
        Ok(()) => println!("\ndone"),
        Err(e) => {
            eprintln!("\nerror: {e:?}");
            std::process::exit(1);
        }
    }
}
