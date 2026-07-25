//! The single integration-test binary for streamcraft-core (spec: Testing — "tests are
//! data", table-driven; one binary so the suite links once and runs fast). Cargo's test
//! autodiscovery is disabled in `Cargo.toml` and this is the one declared `[[test]]`
//! target; sub-suites are `mod`-included here as sibling files, so the whole thing links
//! and runs as ONE binary in well under a second.

// The introspection protocol + server (spec: Introspection protocol and scraft-scope).
// Everything here is behind the `introspect` feature; with the feature off the module
// is empty, so `cargo test -p streamcraft-core` (feature off) still links and passes.
#[path = "introspect.rs"]
mod introspect;
