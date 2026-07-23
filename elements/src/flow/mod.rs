//! Flow elements — the thread-group boundaries and inline transforms
//! (spec: Scheduling). `queue`, `tee`, and `funnel` will live here; for now a
//! `passthrough` exercises the passive-inline group path.

mod passthrough;

pub use passthrough::PassThrough;
