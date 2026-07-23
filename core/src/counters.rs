//! Observability: per-element counters and latency reports (spec: Debuggability;
//! Latency). Stored SoA in the framework wrapper (not element code), updated once
//! per batch with plain atomics, near-zero cost when unread.

/// A snapshot of one element's counters.
#[derive(Clone, Debug, Default)]
pub struct CounterSnapshot {
    pub buffers_in: u64,
    pub buffers_out: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub queue_high_water: u32,
    pub drops: u64,
    // TODO(step 7): batch-size and latency histograms.
}

/// Per-path, per-element latency breakdown — where every microsecond goes
/// (spec: Latency). TODO(step 5).
#[derive(Clone, Debug, Default)]
pub struct LatencyReport {
    _priv: (),
}
