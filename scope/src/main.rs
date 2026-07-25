//! `scraft-scope` — attach mode: connect to any running streamcraft app that is
//! serving the introspection protocol (feature `introspect` + either
//! `pipeline.serve_introspection(path)` or `STREAMCRAFT_INTROSPECT=<path>`).
//!
//! Usage: `scraft-scope <socket-path> [--frames N]`
//! (`--frames N` renders N frames then exits — headless CI via SDL_VIDEODRIVER=dummy.)

use std::path::PathBuf;

use streamcraft_scope::app::{self, AppOpts};

fn main() {
    let mut path: Option<PathBuf> = None;
    let mut opts = AppOpts::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--frames" => opts.max_frames = args.next().and_then(|s| s.parse().ok()),
            "-h" | "--help" => {
                eprintln!("usage: scraft-scope <socket-path> [--frames N]");
                return;
            }
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => eprintln!("scraft-scope: unexpected arg {other:?}"),
        }
    }
    let Some(path) = path else {
        eprintln!("usage: scraft-scope <socket-path> [--frames N]");
        std::process::exit(2);
    };
    if let Err(e) = app::run(&path, opts) {
        eprintln!("scraft-scope: {e}");
        std::process::exit(1);
    }
}
