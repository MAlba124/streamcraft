//! `scplay-ui` — the SDL3 GUI media player binary (spec: streamcraft.md Milestone
//! applications §5 + the UI `<update>` §1209). Hand it a file; it opens a window, plays
//! the video letterboxed with audio-synced pacing, burns in subtitles (the pipeline does
//! that), and gives a transport bar, track menus, and a stats overlay.
//!
//! ```text
//! scplay-ui [--headless-frames N] FILE
//! ```
//!
//! Keys: Space = pause/resume, ←/→ = seek ∓10 s, Home = seek 0, F = fullscreen,
//! M = mute (best-effort; no live volume prop today), Tab/i = stats overlay,
//! Q/Esc = quit. Drag-and-drop a file onto the window is a bonus (handled in the backend
//! event pump; opening a *new* file mid-run is future work — v1 opens the argv file).
//!
//! `--headless-frames N`: run the pipeline and drive the compositor for N frames into
//! SDL's offscreen dummy driver (no display needed), proving the video-texture +
//! letterbox + DrawList compositing path and reporting fps / paced-drop numbers.

// The adopted h264/hevc decoders churn malloc internally (see sdl3/examples/play_file.rs);
// mimalloc buys playback headroom, exactly like scplay. Binary-only — the library stays
// allocator-agnostic. Gated behind the default `mimalloc` feature: `--no-default-features`
// falls back to the system allocator so heaptrack/valgrind can observe allocations.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::process::ExitCode;

use scplay_ui::app::{run, Config};

fn usage() -> String {
    "usage: scplay-ui [--headless-frames N] FILE".to_string()
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut file: Option<String> = None;
    let mut headless_frames = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--headless-frames" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--headless-frames needs a value (frame count)".to_string())?
                    .parse::<u32>()
                    .map_err(|_| "--headless-frames value must be a non-negative integer".to_string())?;
                headless_frames = Some(v);
            }
            "-h" | "--help" => return Err(usage()),
            other if other.starts_with("--") => {
                return Err(format!("unknown switch '{other}'\n{}", usage()));
            }
            _ => {
                if file.replace(arg.clone()).is_some() {
                    return Err(format!("expected exactly one FILE argument\n{}", usage()));
                }
            }
        }
    }
    let file = file.ok_or_else(usage)?;
    Ok(Config { file, headless_frames })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = match parse_args(&args) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("{msg}");
            return if msg.starts_with("usage:") { ExitCode::SUCCESS } else { ExitCode::from(2) };
        }
    };

    match run(cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
