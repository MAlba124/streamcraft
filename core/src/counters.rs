//! Observability: per-element counters and latency reports (spec: Debuggability;
//! Latency). Maintained by the scheduler wrapper (not element code) as plain
//! atomics, near-zero cost when unread.

use std::sync::atomic::{AtomicU64, Ordering};

/// A snapshot of one element's counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CounterSnapshot {
    pub buffers_in: u64,
    pub buffers_out: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub queue_high_water: u32,
    pub drops: u64,
    // TODO(step 7): batch-size and latency histograms.
}

/// Live per-element counters, incremented by the scheduler as buffers cross the
/// element's boundaries. One shared instance per element; read via
/// [`snapshot`](Self::snapshot).
#[derive(Default)]
pub struct ElementCounters {
    buffers_in: AtomicU64,
    buffers_out: AtomicU64,
    bytes_in: AtomicU64,
    bytes_out: AtomicU64,
}

impl ElementCounters {
    /// Record a batch consumed by this element.
    pub fn record_in(&self, buffers: u64, bytes: u64) {
        self.buffers_in.fetch_add(buffers, Ordering::Relaxed);
        self.bytes_in.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a batch produced by this element.
    pub fn record_out(&self, buffers: u64, bytes: u64) {
        self.buffers_out.fetch_add(buffers, Ordering::Relaxed);
        self.bytes_out.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            buffers_in: self.buffers_in.load(Ordering::Relaxed),
            buffers_out: self.buffers_out.load(Ordering::Relaxed),
            bytes_in: self.bytes_in.load(Ordering::Relaxed),
            bytes_out: self.bytes_out.load(Ordering::Relaxed),
            queue_high_water: 0,
            drops: 0,
        }
    }
}

/// Per-path, per-element latency breakdown — where every microsecond goes
/// (spec: Latency). TODO(step 5).
#[derive(Clone, Debug, Default)]
pub struct LatencyReport {
    _priv: (),
}
