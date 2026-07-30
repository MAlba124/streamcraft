//! Compile the vendored libopus C source (`vendor/opus/`) into a **static** `libopus.a` with the
//! `cc` crate and statically link it — no cmake, no autotools, no system library.
//!
//! The exact source set is read from libopus's own authoritative `*_sources.mk` manifests (so a
//! version bump needs no edits here): the scalar-C CELT + common-SILK + float-SILK + top-level
//! Opus sources. SIMD/RTCD (`*_x86_*`, `*_arm_*`) and the fixed-point SILK (`*_FIXED`) variants are
//! intentionally excluded — this is a portable float, scalar reference build (correct everywhere;
//! SIMD is a later opt-in). The `demo`/`compare` tools are not in the manifests, so they never
//! compile.

use std::path::{Path, PathBuf};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/opus");

    let celt_mk = std::fs::read_to_string(root.join("celt_sources.mk")).expect("celt_sources.mk");
    let silk_mk = std::fs::read_to_string(root.join("silk_sources.mk")).expect("silk_sources.mk");
    let opus_mk = std::fs::read_to_string(root.join("opus_sources.mk")).expect("opus_sources.mk");

    // The float, scalar-C library set (no SIMD/RTCD, no fixed-point).
    let mut sources: Vec<String> = Vec::new();
    sources.extend(mk_sources(&celt_mk, "CELT_SOURCES"));
    sources.extend(mk_sources(&silk_mk, "SILK_SOURCES"));
    sources.extend(mk_sources(&silk_mk, "SILK_SOURCES_FLOAT"));
    sources.extend(mk_sources(&opus_mk, "OPUS_SOURCES"));
    sources.extend(mk_sources(&opus_mk, "OPUS_SOURCES_FLOAT"));
    assert!(!sources.is_empty(), "parsed zero libopus sources — the .mk format changed");

    let mut build = cc::Build::new();
    build
        .include(root.join("include")) // public opus*.h
        .include(root.join("celt"))
        .include(root.join("silk"))
        .include(root.join("silk/float"))
        // Required by opus (no autotools config.h): OPUS_BUILD gates the public API build, and
        // VAR_ARRAYS selects C99 variable-length-array scratch — the alloc-free stack path libopus
        // uses instead of per-call malloc (stack_alloc.h errors without one of the three modes).
        .define("OPUS_BUILD", None)
        .define("VAR_ARRAYS", None)
        // `opus_get_version_string()` reports this (else "libopus unknown").
        .define("PACKAGE_VERSION", "\"1.4-profluens-static\"")
        .warnings(false) // upstream C; not our warnings to fix
        .flag_if_supported("-fvisibility=hidden");
    for s in &sources {
        build.file(root.join(s));
    }
    build.compile("opus"); // → libopus.a, emitted as `-l static=opus`

    // libopus calls into libm (sin/cos/log/pow/…). Rust's std pulls it on most targets, but link
    // it explicitly so the static archive resolves regardless.
    println!("cargo:rustc-link-lib=m");

    // Rebuild when the vendored tree or manifests change.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/opus/celt_sources.mk");
    println!("cargo:rerun-if-changed=vendor/opus/silk_sources.mk");
    println!("cargo:rerun-if-changed=vendor/opus/opus_sources.mk");
    for s in &sources {
        rerun_if_changed(&root.join(s));
    }
}

/// Extract the `.c` paths assigned to make variable `var` in a `*_sources.mk` manifest.
///
/// The format is always `VAR = \` followed by ` path.c \` continuation lines until a line without
/// a trailing backslash. The `= ` anchor after the exact name keeps `SILK_SOURCES` from also
/// matching `SILK_SOURCES_FLOAT` / `SILK_SOURCES_X86_RTCD`.
fn mk_sources(mk: &str, var: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_block = false;
    for line in mk.lines() {
        let t = line.trim();
        if !in_block {
            if let Some(rest) = t.strip_prefix(var) {
                if rest.trim_start().starts_with('=') {
                    in_block = true;
                }
            }
            continue;
        }
        let cont = t.ends_with('\\');
        let tok = t.trim_end_matches('\\').trim();
        if tok.ends_with(".c") {
            out.push(tok.to_string());
        }
        if !cont {
            break;
        }
    }
    out
}

fn rerun_if_changed(p: &Path) {
    println!("cargo:rerun-if-changed={}", p.display());
}
