//! `pf-scope` — the inspector binary.
//!
//! Two ways in, one UI:
//!
//! - **Attach**: `pf-scope <socket-path>` connects to a running profluens app
//!   that is already serving the protocol (core feature `introspect` + either
//!   `pipeline.serve_introspection(path)` or `PROFLUENS_INTROSPECT=<path>`).
//! - **Launch**: `pf-scope [--] <command> [args…]` spawns the command with
//!   `PROFLUENS_INTROSPECT` pointing at a fresh private socket, waits for the app
//!   to serve it, and attaches — no socket paths to wire up by hand. The target
//!   must be built with core's `introspect` feature or the env var is ignored.
//!   Closing the scope terminates the launched command.
//!
//! `--frames N` renders N frames then exits (headless CI via SDL_VIDEODRIVER=dummy).

use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use profluens_scope::app::{self, AppOpts};

fn usage() -> ! {
    eprintln!(
        "usage: pf-scope [--frames N] <socket-path>\n       pf-scope [--frames N] [--] <command> [args...]"
    );
    std::process::exit(2);
}

/// Kills the launched child when the scope exits (we started it, we stop it).
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn main() {
    let mut opts = AppOpts::default();
    let mut positionals: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--frames" => opts.max_frames = args.next().and_then(|s| s.parse().ok()),
            "-h" | "--help" => usage(),
            "--" => {
                positionals.extend(args);
                break;
            }
            _ => {
                positionals.push(a);
                // Everything after the command is the command's own argv.
                positionals.extend(args);
                break;
            }
        }
    }
    if positionals.is_empty() {
        usage();
    }

    // One positional that is an existing *socket* ⇒ attach; anything else ⇒ launch
    // (a command path is also an existing file, so the file type decides).
    let attach_path = if positionals.len() == 1 {
        use std::os::unix::fs::FileTypeExt;
        let p = PathBuf::from(&positionals[0]);
        match std::fs::metadata(&p) {
            Ok(m) if m.file_type().is_socket() => Some(p),
            _ => None,
        }
    } else {
        None
    };

    let result = match attach_path {
        Some(path) => app::run(&path, opts),
        None => launch_and_attach(&positionals, opts),
    };
    if let Err(e) = result {
        eprintln!("pf-scope: {e}");
        std::process::exit(1);
    }
}

/// Launch mode: spawn `argv` with `PROFLUENS_INTROSPECT` set to a private socket,
/// wait for the app to serve it, attach. The child dies with the scope.
fn launch_and_attach(argv: &[String], opts: AppOpts) -> Result<(), String> {
    let sock = std::env::temp_dir().join(format!("pf-scope-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&sock);

    let child = Command::new(&argv[0])
        .args(&argv[1..])
        .env("PROFLUENS_INTROSPECT", &sock)
        .spawn()
        .map_err(|e| format!("launch {:?}: {e} (not a running app's socket path either)", argv[0]))?;
    let mut guard = ChildGuard(child);
    eprintln!("pf-scope: launched {:?}, waiting for {}", argv.join(" "), sock.display());

    // The app serves the socket when its pipeline reaches run(); building/decoding
    // may take a while, so wait generously — but bail as soon as the child dies.
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if sock.exists() {
            break;
        }
        if let Ok(Some(status)) = guard.0.try_wait() {
            let _ = std::fs::remove_file(&sock);
            return Err(format!(
                "command exited ({status}) before serving the introspection socket — \
                 is it built with core's `introspect` feature?"
            ));
        }
        if Instant::now() > deadline {
            let _ = std::fs::remove_file(&sock);
            return Err("timed out waiting for the introspection socket".into());
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    let result = app::run(&sock, opts);
    drop(guard); // kill + reap the child before removing its socket
    let _ = std::fs::remove_file(&sock);
    result
}
