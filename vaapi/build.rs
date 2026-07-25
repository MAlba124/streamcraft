//! Minimal link setup for the system libva. The FFI declares `#[link(name="va")]`
//! + `#[link(name="va-drm")]`; this script only teaches the linker *where* those
//! shared objects live, by asking `pkg-config` for the library search paths.
//!
//! No `-sys` crate and no bindgen: the surface is small and hand-transcribed
//! (`src/ffi.rs`), so a ~20-line pkg-config shim is the whole build story. It
//! fails soft — if pkg-config is missing or the libs are not registered with it,
//! the `#[link]` attributes still resolve against the default linker paths (in the
//! devshell libva is on the default path anyway); the warning just makes a
//! genuinely-absent libva a legible build error rather than a mystery.

use std::process::Command;

fn main() {
    // Re-run if the pkg-config search path changes (nix devshell entry, etc.).
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    match Command::new("pkg-config")
        .args(["--libs", "libva", "libva-drm"])
        .output()
    {
        Ok(out) if out.status.success() => {
            let flags = String::from_utf8_lossy(&out.stdout);
            for tok in flags.split_whitespace() {
                if let Some(dir) = tok.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={dir}");
                }
                // `-l` flags are emitted by the crate's #[link] attributes; do not
                // duplicate them here (would double-link).
            }
        }
        _ => {
            println!(
                "cargo:warning=pkg-config could not resolve libva/libva-drm; relying on \
                 default linker search paths (set PKG_CONFIG_PATH or install libva-dev)"
            );
        }
    }
}
