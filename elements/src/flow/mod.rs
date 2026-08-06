//! Flow elements — the thread-group boundaries and inline transforms
//! (spec: Scheduling). `tee` and `funnel` will live here too; today a `passthrough`
//! exercises the passive-inline group path and [`Queue`] is the explicit thread-
//! decoupling boundary (spec: Scheduling — "wherever the user drops an explicit `queue`
//! element").

mod passthrough;
mod queue;
mod tee;

pub use passthrough::PassThrough;
pub use queue::Queue;
pub use tee::Tee;
