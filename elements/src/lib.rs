//! profluens-elements — the built-in, pure-Rust elements.
//!
//! Milestone 1: [`io::FileSrc`] ! [`io::FileSink`] (spec: Milestone applications).
//! The reference elements are kept exemplary — element authors copy the nearest one
//! (spec: Writing elements), so the nearest one must be perfect.
//!
//! (The HTTP source lives in its own `pf-http` plugin crate, not here — it will grow
//! a TLS dependency that must not leak into the framework crates.)

pub mod flow;
pub mod io;
pub mod testing;

use profluens_core::element::Element;
use profluens_core::registry::Registry;

/// Register the built-in elements for name-based construction (spec: Plugins —
/// `parse("filesrc path=x ! …")`). The typed `use` + constructor path is primary;
/// this powers `pf-launch` and one-liner tests. Each descriptor is `&'static`
/// (it lives in its element's module), taken here from a throwaway default instance.
pub fn register(registry: &mut Registry) {
    registry.register(io::FileSrc::new("").desc());
    registry.register(io::FileSink::new("").desc());
    registry.register(flow::Queue::new().desc());
    registry.register(testing::TestSrc::new(0).desc());
    registry.register(testing::TestSink::new().0.desc());
}
