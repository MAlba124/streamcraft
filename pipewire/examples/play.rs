//! Play an audio file: `filesrc ! flacdec ! pipewireaudiosink` (spec: Milestone
//! applications — play an audio file).
//!
//! Usage: `cargo run --release -p sc-pipewire --example play -- <file.flac>`
//!
//! Requires a running PipeWire session (your desktop audio server) — it plays to the
//! default output. `flacdec` announces the file's `audio/raw` format at runtime and the
//! sink configures the device from it (spec: Formats — dynamic caps); the graph is paced
//! by the device via backpressure.

use sc_flac::FlacDec;
use sc_pipewire::PipeWireAudioSink;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::FileSrc;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: play <file.flac>");
        std::process::exit(2);
    });

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&path));
    let dec = p.add(FlacDec::new());
    let sink = p.add(PipeWireAudioSink::new());
    p.link((src, "src"), (dec, "sink")).expect("filesrc -> flacdec");
    p.link((dec, "src"), (sink, "sink")).expect("flacdec -> pipewireaudiosink");

    println!("playing {path} …");
    match p.run() {
        Ok(()) => println!("done"),
        Err(e) => {
            eprintln!("error: {e:?}");
            std::process::exit(1);
        }
    }
}
