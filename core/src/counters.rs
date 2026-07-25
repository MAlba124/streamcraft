//! Observability: per-element counters, tap handles, and latency reports (spec:
//! Debuggability; Taps; Latency). Maintained by the scheduler wrapper (not element
//! code) as plain relaxed atomics updated once per batch — near-zero cost when
//! unread.
//!
//! The tap model (spec: Taps): **a tap is a pull of data the pipeline already
//! keeps, never a push callback on a streaming thread.** The pipeline stores
//! *cumulative counts, not rates*, on purpose — the window and the arithmetic are
//! the observer's policy: bitrate is `Δbytes / Δrunning_time` between two
//! snapshots, throughput is `Δbuffers`, backpressure is the queue high-water.

use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use crate::clock::Clock;
use crate::id::ElementId;
use crate::time::Timestamp;

/// A snapshot of one element's counters, taken on the observer's thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CounterSnapshot {
    pub buffers_in: u64,
    pub buffers_out: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    /// Batches consumed / produced — `buffers / batches` is the realized batch size,
    /// the batching-efficiency signal (spec: Batching).
    pub batches_in: u64,
    pub batches_out: u64,
    /// Highest fill (in batches) observed on this element's downstream ring at push
    /// time. At the ring's capacity ⇒ this element is being backpressured.
    pub queue_high_water: u32,
    pub drops: u64,
    // TODO(step 7): latency histograms.
}

/// A lock-free power-of-two latency histogram (spec: Debuggability — latency
/// instrumentation). Bucket `i` counts durations in `[2^i, 2^(i+1))` ns — 32
/// buckets cover 1 ns to ~4.3 s at a constant ~2× resolution (the top bucket
/// absorbs anything longer), which is what perf debugging needs (is it 10 µs or
/// 10 ms?). Recording is one `leading_zeros` + four relaxed `fetch_*`s.
#[derive(Default)]
pub struct LatencyHistogram {
    buckets: [AtomicU64; Self::BUCKETS],
    count: AtomicU64,
    sum_ns: AtomicU64,
    max_ns: AtomicU64,
}

impl LatencyHistogram {
    pub const BUCKETS: usize = 32;

    /// Record one duration. Zero-duration records land in bucket 0.
    pub fn record(&self, ns: u64) {
        let bucket = (63 - (ns | 1).leading_zeros() as usize).min(Self::BUCKETS - 1);
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(ns, Ordering::Relaxed);
        self.max_ns.fetch_max(ns, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> LatencySnapshot {
        let mut buckets = [0u64; Self::BUCKETS];
        for (b, a) in buckets.iter_mut().zip(&self.buckets) {
            *b = a.load(Ordering::Relaxed);
        }
        LatencySnapshot {
            buckets,
            count: self.count.load(Ordering::Relaxed),
            sum_ns: self.sum_ns.load(Ordering::Relaxed),
            max_ns: self.max_ns.load(Ordering::Relaxed),
        }
    }
}

/// An observer-side copy of a [`LatencyHistogram`], with quantile arithmetic (the
/// observer's policy, per the tap model — the pipeline only keeps counts).
#[derive(Clone, Copy, Debug)]
pub struct LatencySnapshot {
    pub buckets: [u64; LatencyHistogram::BUCKETS],
    pub count: u64,
    pub sum_ns: u64,
    pub max_ns: u64,
}

impl LatencySnapshot {
    pub fn mean_ns(&self) -> u64 {
        if self.count == 0 { 0 } else { self.sum_ns / self.count }
    }

    /// The quantile `q` in `[0,1]`, estimated at bucket resolution (the geometric
    /// midpoint of the bucket holding the q-th sample — ~±41% by construction,
    /// which is the granularity that matters for "µs or ms?" questions).
    pub fn quantile_ns(&self, q: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let target = ((self.count as f64) * q.clamp(0.0, 1.0)).ceil().max(1.0) as u64;
        let mut seen = 0u64;
        for (i, &c) in self.buckets.iter().enumerate() {
            seen += c;
            if seen >= target {
                // Geometric midpoint of [2^i, 2^(i+1)): 2^i * sqrt(2) — clamped to the
                // observed max so an estimate never reads above a real sample.
                return (((1u128 << i) as f64 * std::f64::consts::SQRT_2) as u64)
                    .min(self.max_ns);
            }
        }
        self.max_ns
    }
}

/// Live per-element counters, incremented by the scheduler as batches cross the
/// element's boundaries. One shared instance per element, created at `add()` and
/// stable across runs (cumulative), so a [`TapHandle`] taken before `run()` reads
/// them while streaming. Read via [`snapshot`](Self::snapshot).
///
/// The three latency histograms (spec: Debuggability — latency instrumentation)
/// fill only while [`Pipeline::set_tracing`](crate::pipeline::Pipeline::set_tracing)
/// (or `STREAMCRAFT_TRACE=1`) is on — the timestamp reads they need are the cost
/// the flag gates; the histograms themselves are always allocated (≈1.3 KB/element).
#[derive(Default)]
pub struct ElementCounters {
    buffers_in: AtomicU64,
    buffers_out: AtomicU64,
    bytes_in: AtomicU64,
    bytes_out: AtomicU64,
    batches_in: AtomicU64,
    batches_out: AtomicU64,
    queue_high_water: AtomicU32,
    drops: AtomicU64,
    /// Wall time spent inside this element's `process()` per call.
    pub process_ns: LatencyHistogram,
    /// Ring residency: wall time a batch waited between the producer's push and this
    /// (consuming) element's pop — the backpressure/queueing signal.
    pub queue_ns: LatencyHistogram,
    /// Sink wait overshoot: how far past its deadline `ctx.wait_until` actually
    /// returned (running-time domain) — scheduling jitter as the sink experiences it.
    pub wait_lateness_ns: LatencyHistogram,
}

impl ElementCounters {
    /// Record a batch consumed by this element.
    pub fn record_in(&self, buffers: u64, bytes: u64) {
        self.batches_in.fetch_add(1, Ordering::Relaxed);
        self.buffers_in.fetch_add(buffers, Ordering::Relaxed);
        self.bytes_in.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a batch produced by this element.
    pub fn record_out(&self, buffers: u64, bytes: u64) {
        self.batches_out.fetch_add(1, Ordering::Relaxed);
        self.buffers_out.fetch_add(buffers, Ordering::Relaxed);
        self.bytes_out.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record the downstream ring's fill (in batches) observed after a push.
    pub fn record_queue_fill(&self, fill: u32) {
        self.queue_high_water.fetch_max(fill, Ordering::Relaxed);
    }

    /// Record buffers this element produced that the scheduler dropped — today,
    /// output on an unlinked src pad (spec: robustness — an unlinked track is
    /// discarded by policy, never accumulated); later also leaky-queue drops.
    pub fn record_drops(&self, buffers: u64) {
        self.drops.fetch_add(buffers, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            buffers_in: self.buffers_in.load(Ordering::Relaxed),
            buffers_out: self.buffers_out.load(Ordering::Relaxed),
            bytes_in: self.bytes_in.load(Ordering::Relaxed),
            bytes_out: self.bytes_out.load(Ordering::Relaxed),
            batches_in: self.batches_in.load(Ordering::Relaxed),
            batches_out: self.batches_out.load(Ordering::Relaxed),
            queue_high_water: self.queue_high_water.load(Ordering::Relaxed),
            drops: self.drops.load(Ordering::Relaxed),
        }
    }
}

/// A cloneable handle for reading live stats off a running pipeline from any
/// thread (spec: Taps, tier 1) — the same shape as `StopHandle`/`SeekHandle`/
/// `PropHandle`. Snapshots are relaxed loads of counters the scheduler already
/// maintains: zero cost on the streaming path, no locks anywhere.
///
/// Rates are the observer's arithmetic: sample [`snapshot`](Self::snapshot) and
/// [`now`](Self::now) twice; bitrate is `Δbytes / Δnow`.
///
/// Obtain from [`Pipeline::tap_handle`](crate::pipeline::Pipeline::tap_handle)
/// *after* the topology is built (it snapshots the elements present at that point).
#[derive(Clone)]
pub struct TapHandle {
    entries: Arc<[(&'static str, Arc<ElementCounters>)]>,
    /// The pipeline's *selected* clock, read through a shared cell: `run()` may
    /// replace the default with a sink-provided device clock after this handle was
    /// taken, and mixing the old clock with the (device-rebased) base would read a
    /// timeline that exists nowhere (an observer-only lock; never a streaming path).
    clock: Arc<std::sync::Mutex<Arc<dyn Clock>>>,
    /// Signed ns of the current run's base time; [`crate::time::BASE_UNSET`] until
    /// the first `run()`. Signed: a seek rebase can push it negative.
    base: Arc<AtomicI64>,
}

impl TapHandle {
    pub(crate) fn new(
        entries: Vec<(&'static str, Arc<ElementCounters>)>,
        clock: Arc<std::sync::Mutex<Arc<dyn Clock>>>,
        base: Arc<AtomicI64>,
    ) -> Self {
        Self { entries: entries.into(), clock, base }
    }

    /// The number of elements this handle observes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The element's descriptor name (for labelling observer output).
    pub fn element_name(&self, el: ElementId) -> Option<&'static str> {
        self.entries.get(el.0 as usize).map(|(n, _)| *n)
    }

    /// A snapshot of one element's cumulative counters. `None` for an unknown id.
    pub fn snapshot(&self, el: ElementId) -> Option<CounterSnapshot> {
        self.entries.get(el.0 as usize).map(|(_, c)| c.snapshot())
    }

    /// Snapshots of one element's latency histograms — `(process, queue_wait,
    /// sink_wait_lateness)` (spec: Debuggability). All-zero unless tracing is on
    /// (`Pipeline::set_tracing` / `STREAMCRAFT_TRACE=1`). `None` for an unknown id.
    pub fn latency(
        &self,
        el: ElementId,
    ) -> Option<(LatencySnapshot, LatencySnapshot, LatencySnapshot)> {
        self.entries.get(el.0 as usize).map(|(_, c)| {
            (c.process_ns.snapshot(), c.queue_ns.snapshot(), c.wait_lateness_ns.snapshot())
        })
    }

    /// Running time on the pipeline clock — the denominator for rate windows.
    /// [`Timestamp::NONE`] before the first `run()` samples a base time. Not
    /// monotonic across seeks: a backward seek rebases it backward (observers
    /// windowing rates should skip non-increasing samples).
    pub fn now(&self) -> Timestamp {
        let base = self.base.load(Ordering::Acquire);
        if base == crate::time::BASE_UNSET {
            return Timestamp::NONE;
        }
        let clock = Arc::clone(&self.clock.lock().unwrap_or_else(|e| e.into_inner()));
        crate::time::running_time(clock.now().0, base)
    }
}

/// Per-path, per-element latency breakdown — where every microsecond goes (spec:
/// Latency — "computed, per path... a graph traversal over data the pipeline
/// already holds", never a distributed query protocol).
#[derive(Clone, Debug, Default)]
pub struct LatencyReport {
    /// One entry per sink (an element with no outgoing links), worst path first.
    pub paths: Vec<PathLatency>,
}

impl LatencyReport {
    /// The pipeline-wide latency: the worst sink path (what a live pipeline must
    /// compensate end-to-end). Zero for an empty graph.
    pub fn total(&self) -> Timestamp {
        self.paths.first().map_or(Timestamp::ZERO, |p| p.total)
    }
}

/// The worst source→sink path for one sink: its total declared latency and the
/// per-element contributions along that path (in path order, source first).
#[derive(Clone, Debug)]
pub struct PathLatency {
    pub sink: ElementId,
    /// Sum of the declared minimum latencies of every element strictly upstream of
    /// the sink on its worst path — what the sink adds to `base_time + pts` when it
    /// waits (spec: Latency — enforced at sinks).
    pub total: Timestamp,
    /// `(element, declared min latency)` along the worst path, source first,
    /// including the sink itself (whose own latency is reported but not waited on).
    pub per_element: Vec<(ElementId, Timestamp)>,
    /// Whether any element on the path declared itself live (spec: live pipelines
    /// get exactly the delay the graph requires; non-live get throughput mode).
    pub is_live: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_accumulate_and_snapshot() {
        let c = ElementCounters::default();
        c.record_in(4, 1000);
        c.record_in(2, 500);
        c.record_out(6, 1500);
        c.record_queue_fill(3);
        c.record_queue_fill(1); // high-water keeps the max
        let s = c.snapshot();
        assert_eq!(s.buffers_in, 6);
        assert_eq!(s.bytes_in, 1500);
        assert_eq!(s.batches_in, 2);
        assert_eq!(s.buffers_out, 6);
        assert_eq!(s.bytes_out, 1500);
        assert_eq!(s.batches_out, 1);
        assert_eq!(s.queue_high_water, 3);
    }

    #[test]
    fn tap_handle_reads_counters_and_clock() {
        use crate::clock::MockClock;

        let counters = Arc::new(ElementCounters::default());
        let clock = MockClock::new();
        let base = Arc::new(AtomicI64::new(crate::time::BASE_UNSET));
        let tap = TapHandle::new(
            vec![("testsrc", Arc::clone(&counters))],
            Arc::new(std::sync::Mutex::new(Arc::new(clock.clone()) as Arc<dyn Clock>)),
            Arc::clone(&base),
        );

        assert!(tap.now().is_none(), "no run yet — no running time");
        assert_eq!(tap.element_name(ElementId(0)), Some("testsrc"));
        assert!(tap.snapshot(ElementId(1)).is_none(), "unknown element");

        // A "run" starts: base is sampled; the clock then advances 5 ms.
        base.store(crate::time::base_from_clock(clock.now()), Ordering::Release);
        clock.advance(Timestamp::from_millis(5));
        counters.record_out(1, 1_000_000);

        let s = tap.snapshot(ElementId(0)).unwrap();
        assert_eq!(s.bytes_out, 1_000_000);
        assert_eq!(tap.now(), Timestamp::from_millis(5));
        // The observer's arithmetic (spec: bitrate = Δbytes / Δrunning_time):
        let bits_per_sec = s.bytes_out * 8 * 1_000_000_000 / tap.now().0;
        assert_eq!(bits_per_sec, 1_600_000_000);
    }
}
