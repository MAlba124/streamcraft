//! Play an audio file: `filesrc ! flacdec ! pipewireaudiosink` (spec: Milestone
//! applications — play an audio file), with a live progress line, pause, and seeking.
//!
//! Usage: `cargo run --release -p sc-pipewire --example play -- <file.flac>`
//! Controls: `space`/`p` pause·resume · `←`/`→` seek ∓/±5 s · `↑`/`↓` seek ±30 s · `q` quit.
//!
//! Requires a running PipeWire session — it plays to the default output. `flacdec` announces
//! the file's `audio/raw` format at runtime and the sink configures the device from it (spec:
//! Formats — dynamic caps); the graph is paced by the device via backpressure.
//!
//! Pausing renders silence and holds the device buffer, so the whole pipeline backpressures
//! to a stop and resumes exactly where it left off. Seeking (spec: flush/seek) publishes a
//! byte target through the pipeline's `SeekHandle`: the source resumes reading there, the
//! decoder re-syncs to the next frame, and the data already in flight is dropped out-of-band
//! (rather than played out first), so it is responsive. FLAC has no seektable here, so
//! time→byte is a proportional estimate the decoder re-syncs from — accurate to a frame.

use std::io::{Read, Write};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sc_flac::{FlacDec, StreamDecoder};
use sc_pipewire::PipeWireAudioSink;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;
use streamcraft_elements::io::FileSrc;

/// What the app needs to map a wall-clock seek target to a byte offset and a play position.
#[derive(Clone, Copy)]
struct Media {
    total_secs: f64,
    rate: u32,
    file_len: u64,
}

/// Best-effort media info from the FLAC header (STREAMINFO) + file size. Parses the header
/// with the same decoder the pipeline uses, so there is no separate FLAC parser to drift.
/// STREAMINFO is the first block, but a large metadata chain (embedded cover art, padding)
/// can push the *end* of the chain — which the decoder needs before it exposes the header —
/// well past the first read, so this feeds chunks until the header appears (capped, so a
/// non-FLAC or headerless file can't make it read forever). Fields are `0` when unknown
/// (then seeking is disabled and only elapsed time is shown).
fn probe(path: &str) -> Media {
    let file_len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut media = Media { total_secs: 0.0, rate: 0, file_len };
    if let Ok(mut f) = std::fs::File::open(path) {
        let mut sd = StreamDecoder::new();
        let mut chunk = vec![0u8; 256 * 1024];
        let mut read_total = 0u64;
        while sd.info().is_none() && read_total < 32 * 1024 * 1024 {
            let n = match f.read(&mut chunk) {
                Ok(0) | Err(_) => break, // EOF or error
                Ok(n) => n,
            };
            sd.push(&chunk[..n]);
            read_total += n as u64;
            let _ = sd.pull(); // triggers the header parse (info populated once the chain fits)
        }
        if let Some(info) = sd.info() {
            media.rate = info.sample_rate;
            if info.total_samples != 0 && info.sample_rate != 0 {
                media.total_secs = info.total_samples as f64 / info.sample_rate as f64;
            }
        }
    }
    media
}

fn fmt_time(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

/// Put the controlling terminal into cbreak mode (character-at-a-time, no echo) so single
/// keypresses — including arrow escape sequences — arrive immediately, restoring the previous
/// settings on drop. Dependency-free: shells out to `stty` on `/dev/tty`. If there is no tty
/// (stdin piped, e.g. a scripted test), it degrades to a no-op and reads bytes as they come.
struct RawTty {
    saved: Option<String>,
}

impl RawTty {
    fn enable() -> Self {
        let saved = Command::new("sh")
            .args(["-c", "stty -g </dev/tty"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty());
        if saved.is_some() {
            // `isig` is left on, so Ctrl-C still works; we only turn off line-buffering/echo.
            let _ = Command::new("sh")
                .args(["-c", "stty -icanon -echo min 1 time 0 </dev/tty"])
                .status();
        }
        RawTty { saved }
    }
}

impl Drop for RawTty {
    fn drop(&mut self) {
        if let Some(s) = &self.saved {
            let _ = Command::new("sh").args(["-c", &format!("stty {s} </dev/tty")]).status();
        }
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: play <file.flac>");
        std::process::exit(2);
    });

    let media = probe(&path);

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&path));
    let dec = p.add(FlacDec::new());
    let sink = PipeWireAudioSink::new();
    let ctrl = sink.control();
    let snk = p.add(sink);
    p.link((src, "src"), (dec, "sink")).expect("filesrc -> flacdec");
    p.link((dec, "src"), (snk, "sink")).expect("flacdec -> pipewireaudiosink");

    let seek = p.seek_handle();
    let stop = p.stop_handle();

    println!(
        "playing {path} …  (space pause · ←/→ ∓5s · ↑/↓ ±30s · q quit)\r"
    );

    // Controls on a helper thread, reading raw keypresses. Arrow keys are ESC '[' <A-D>.
    let _raw = RawTty::enable();
    {
        let (ctrl, seek, stop) = (ctrl.clone(), seek.clone(), stop.clone());
        let ctrl_keys = ctrl.clone();
        // Seek by `delta` seconds relative to the current position (clamped to the track).
        let seek_by = move |delta: f64| {
            if media.total_secs <= 0.0 {
                return; // unknown duration → can't map time to a byte offset
            }
            let target = (ctrl.position_secs() + delta).clamp(0.0, media.total_secs);
            let to_byte = (target / media.total_secs * media.file_len as f64) as u64;
            // Seeking while paused stays paused: the interruptible sink push lets the flush
            // propagate through the frozen graph, so it re-primes at the new position and the
            // shown time jumps there, without starting playback (spec: flush/seek).
            seek.seek(to_byte, Timestamp::from_nanos((target * 1e9) as u64));
        };
        std::thread::spawn(move || {
            let mut stdin = std::io::stdin();
            let mut b = [0u8; 1];
            let mut get = |buf: &mut [u8; 1]| stdin.read(buf).map(|n| n > 0).unwrap_or(false);
            while get(&mut b) {
                match b[0] {
                    b'q' => {
                        ctrl_keys.resume(); // unblock if paused so the cooperative stop lands
                        stop.stop();
                        break;
                    }
                    b' ' | b'p' => {
                        let _ = ctrl_keys.toggle();
                    }
                    0x1B => {
                        // Escape sequence: expect '[' then a direction byte.
                        if !get(&mut b) || b[0] != b'[' {
                            continue;
                        }
                        if !get(&mut b) {
                            break;
                        }
                        match b[0] {
                            b'C' => seek_by(5.0),    // →
                            b'D' => seek_by(-5.0),   // ←
                            b'A' => seek_by(30.0),   // ↑
                            b'B' => seek_by(-30.0),  // ↓
                            _ => {}
                        }
                    }
                    _ => {}
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
        let clock = if media.total_secs > 0.0 {
            format!("{} / {}", fmt_time(pos), fmt_time(media.total_secs))
        } else {
            fmt_time(pos)
        };
        print!("\r{}  {}   ", if ctrl.is_paused() { "[paused] " } else { "[playing]" }, clock);
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
