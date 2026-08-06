//! Scheduling: thread groups, the SPSC ring, reactor placement (spec: Scheduling
//! and threading; Queue internals). The single biggest performance lever — passive
//! chains run inline as function calls; real queues only at group boundaries.

/// Compiles the graph into thread groups and drives them. Static, printable,
/// inspectable schedule — no async runtime, no work stealing.
pub struct Scheduler {
    _priv: (),
}

/// A chain of elements running inline on one thread; pinnable to a core, with pools
/// allocated NUMA-local to the pin.
pub struct ThreadGroup {
    _priv: (),
}

/// Bounded single-producer / single-consumer ring — the most-executed structure in
/// the framework. Slots *are* the SoA batch columns; producer and consumer each own
/// a 64-byte cache line; two atomics per batch; edge-triggered wakeups;
/// generation-counter flush (spec: Queue internals). The one audited-`unsafe` area
/// alongside [`crate::memory`]; loom-tested before any pipeline exists. TODO(step 2).
pub struct SpscRing {
    _priv: (),
}

/// Producer-side leak policy, chosen before writing (spec: Queue internals).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LeakPolicy {
    Block,
    DropOldest,
    DropNewest,
}
