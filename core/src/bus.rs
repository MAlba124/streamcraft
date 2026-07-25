//! The bus: out-of-band element/framework → application (spec: Events, queries, and
//! the bus). Drained by the app on the app's own thread — no callbacks on streaming
//! threads, ever.
//!
//! **Bounded and class-aware** (spec: "The bus is bounded, with two message classes:
//! droppable (QoS observations, progress chatter) drop-oldest with a counter; critical
//! (errors, EOS, state) never drop — posting one evicts droppables if needed. The
//! streaming path is never blocked."). A real incident motivates the bound: a decoder
//! warn-flood once grew the unbounded `std::mpsc` bus without limit while nobody drained
//! it. The bus is deliberately off every hot path (spec: Queue internals — "the one MPSC
//! structure … deliberately kept off every hot path"), so a plain `Mutex<VecDeque>` +
//! `Condvar` is exactly right: correctness and a hard bound over raw throughput. Posting
//! *never blocks*: a full bus drops the oldest droppable (or, for a droppable arrival,
//! itself) rather than parking a streaming thread.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::error::Error;
use crate::event::TagList;
use crate::format::FixedFormat;
use crate::id::{ElementId, GroupId, LinkId, PadId};
use crate::time::Timestamp;

pub enum BusMessage {
    Error { element: ElementId, error: Error },
    Warning { element: ElementId, error: Error },
    Eos,
    StateChanged { old: State, new: State },
    PadAdded { element: ElementId, pad: PadId, format: FixedFormat },
    ElementAdded { element: ElementId, group: GroupId },
    ElementRemoved { element: ElementId },
    LinkChanged { link: LinkId },
    Tags { element: ElementId, tags: TagList },
    SubgraphJoined { group: GroupId, added_latency: Timestamp },
    LatencyChanged { old: Timestamp, new: Timestamp },
    Qos { sink: ElementId, lateness_ns: i64 },
    /// A branch failed and was isolated (spec: Supervision).
    BranchSealed { group: GroupId, error: Error },
    /// A presentation duration became known — a demuxer parsed it from its headers
    /// (mkv `Info\Duration`). Transport UIs act on this, so it rides the bus (spec:
    /// the bus carries semantics the application acts on); it cannot ride negotiated
    /// formats on a playback path, since announced fields survive fixation only when
    /// the consumer's offers declare them (decoders don't declare `duration`).
    DurationChanged { element: ElementId, ns: u64 },
}

/// The two message classes (spec: Events, queries, and the bus). `Critical` never
/// drops — losing an error, EOS, or a topology change would silently break the app's
/// view of the pipeline. `Droppable` is progress chatter: QoS observations, warnings,
/// tags — a flood of it must never evict something that matters, nor block streaming.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MessageClass {
    Droppable,
    Critical,
}

impl BusMessage {
    /// This message's drop class (spec). Critical: `Error`, `Eos`, `StateChanged`,
    /// `PadAdded`, `ElementAdded/Removed`, `LinkChanged`, `SubgraphJoined`,
    /// `LatencyChanged`, `BranchSealed`, `DurationChanged` — errors, EOS/state,
    /// topology mutations, and rare facts the app acts on. Droppable: `Warning`,
    /// `Qos`, `Tags` — chatter.
    pub fn class(&self) -> MessageClass {
        match self {
            BusMessage::Warning { .. } | BusMessage::Qos { .. } | BusMessage::Tags { .. } => {
                MessageClass::Droppable
            }
            BusMessage::Error { .. }
            | BusMessage::Eos
            | BusMessage::StateChanged { .. }
            | BusMessage::PadAdded { .. }
            | BusMessage::ElementAdded { .. }
            | BusMessage::ElementRemoved { .. }
            | BusMessage::LinkChanged { .. }
            | BusMessage::SubgraphJoined { .. }
            | BusMessage::LatencyChanged { .. }
            | BusMessage::BranchSealed { .. }
            | BusMessage::DurationChanged { .. } => MessageClass::Critical,
        }
    }
}

/// Deliberately smaller than gst's. Pause is a clock op, not a state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Stopped,
    Ready,
    Playing,
}

/// Default bus capacity (messages). Generous, because the bound exists to catch a
/// runaway flood, not to throttle a healthy app that drains at any reasonable cadence.
pub const DEFAULT_CAPACITY: usize = 1024;

/// The shared, bounded queue behind both halves. Guarded by one `Mutex`; a `Condvar`
/// wakes a blocked [`Bus::recv`] when a message lands or the last sender goes away.
struct Shared {
    inner: Mutex<Inner>,
    ready: Condvar,
}

struct Inner {
    queue: VecDeque<BusMessage>,
    capacity: usize,
    /// Live [`BusSender`] count. `recv` returns `None` only once this hits zero *and*
    /// the queue has drained, mirroring `std::mpsc`'s disconnect semantics.
    senders: usize,
    /// Introspection bus taps (spec: The two hot-path touches). Attached by a
    /// [`BusTapPort`], read under this same lock so publishing a tap row is one memcpy
    /// inside the already-held `send` lock. Empty (and untouched) with no client, so a
    /// tapless run pays one `is_empty()` per posted message — and nothing when the
    /// feature is off.
    #[cfg(feature = "introspect")]
    taps: Vec<crate::introspect::tap::BusTapSlot>,
    /// A per-bus monotonic sequence stamped on each tap row under the lock, so a client
    /// sees a total order of bus messages regardless of which tap ring it reads.
    #[cfg(feature = "introspect")]
    tap_seq: u64,
}

impl Inner {
    /// Post `msg`, upholding the bound without ever blocking (spec):
    /// - **Critical** always enqueues; if the bus is full it first evicts the oldest
    ///   droppable, and only if none exists does it grow past capacity (a critical
    ///   message is never dropped, so a bus of nothing-but-critical simply exceeds the
    ///   soft cap — bounded in practice by the finite set of topology/error events).
    /// - **Droppable** enqueues if there is room; at capacity it drops the oldest
    ///   droppable (drop-oldest, spec), or drops *itself* when the bus holds only
    ///   critical messages — either way the counter ticks and ordering of survivors is
    ///   preserved.
    ///
    /// Returns the number of messages dropped by this call (0 or 1).
    fn post(&mut self, msg: BusMessage) -> u64 {
        if self.queue.len() < self.capacity {
            self.queue.push_back(msg);
            return 0;
        }
        // Full. Try to make room by evicting the oldest droppable, preserving the
        // relative order of everything that survives.
        match msg.class() {
            MessageClass::Critical => {
                if self.evict_oldest_droppable() {
                    self.queue.push_back(msg);
                    1
                } else {
                    // All-critical bus: never drop a critical, so grow past the cap.
                    self.queue.push_back(msg);
                    0
                }
            }
            MessageClass::Droppable => {
                if self.evict_oldest_droppable() {
                    self.queue.push_back(msg);
                    1
                } else {
                    // No droppable to evict (bus is all critical) — drop the arrival
                    // itself rather than push a critical message out.
                    1
                }
            }
        }
    }

    /// Remove the oldest droppable message, preserving order of the rest. Returns
    /// whether one was found and removed.
    fn evict_oldest_droppable(&mut self) -> bool {
        if let Some(i) = self
            .queue
            .iter()
            .position(|m| m.class() == MessageClass::Droppable)
        {
            self.queue.remove(i);
            true
        } else {
            false
        }
    }
}

/// The posting half, handed to elements via `Ctx`. Cheap to clone.
pub struct BusSender {
    shared: Arc<Shared>,
    /// Droppable messages lost to the bound, so the receiver can report it (spec: a
    /// slow-draining app degrades observably). Shared across every sender clone.
    dropped: Arc<AtomicU64>,
}

impl Clone for BusSender {
    fn clone(&self) -> Self {
        // A new live sender: count it, so the last drop is detected correctly.
        self.shared.inner.lock().unwrap().senders += 1;
        BusSender {
            shared: self.shared.clone(),
            dropped: self.dropped.clone(),
        }
    }
}

impl BusSender {
    /// Post a message. Never blocks the data path: at capacity it drops the oldest
    /// droppable (or itself) per the class policy, and silently no-ops if the
    /// application has dropped the [`Bus`]. Wakes a blocked [`Bus::recv`].
    pub fn send(&self, msg: BusMessage) {
        let mut inner = self.shared.inner.lock().unwrap();
        // Introspection bus tap (spec: The two hot-path touches). Inside the lock we
        // already hold, and only when a client is attached: stamp a monotonic seq,
        // build a POD row (zero alloc), and copy it into each attached slot. The tap
        // copies, never steals — `inner.post(msg)` below still consumes the original,
        // so the app's `try_recv` is unaffected.
        #[cfg(feature = "introspect")]
        if !inner.taps.is_empty() {
            inner.tap_seq += 1;
            let seq = inner.tap_seq;
            let row = crate::introspect::tap::BusTapRow::from_msg(&msg, seq);
            for slot in inner.taps.iter_mut() {
                slot.publish(row);
            }
        }
        let dropped = inner.post(msg);
        drop(inner);
        if dropped > 0 {
            self.dropped.fetch_add(dropped, Ordering::Relaxed);
        }
        // A message may now be available (or, if it was dropped, nothing changed and a
        // spurious wake is harmless).
        self.shared.ready.notify_one();
    }
}

impl Drop for BusSender {
    fn drop(&mut self) {
        let mut inner = self.shared.inner.lock().unwrap();
        inner.senders -= 1;
        let last = inner.senders == 0;
        drop(inner);
        if last {
            // Wake a `recv` parked on an empty queue so it can observe disconnect.
            self.shared.ready.notify_all();
        }
    }
}

/// The receiving half, owned by the pipeline / drained by the application.
pub struct Bus {
    shared: Arc<Shared>,
    dropped: Arc<AtomicU64>,
}

impl Bus {
    /// A bus with the [`DEFAULT_CAPACITY`] bound.
    pub fn channel() -> (BusSender, Bus) {
        Self::channel_with_capacity(DEFAULT_CAPACITY)
    }

    /// A bus bounded at `capacity` droppable+critical messages. `capacity` is clamped to
    /// at least 1 so a critical message always has somewhere to land.
    pub fn channel_with_capacity(capacity: usize) -> (BusSender, Bus) {
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                queue: VecDeque::new(),
                capacity: capacity.max(1),
                senders: 1,
                #[cfg(feature = "introspect")]
                taps: Vec::new(),
                #[cfg(feature = "introspect")]
                tap_seq: 0,
            }),
            ready: Condvar::new(),
        });
        let dropped = Arc::new(AtomicU64::new(0));
        (
            BusSender {
                shared: shared.clone(),
                dropped: dropped.clone(),
            },
            Bus { shared, dropped },
        )
    }

    /// Non-blocking drain of the oldest message, or `None` if the bus is empty.
    pub fn try_recv(&self) -> Option<BusMessage> {
        self.shared.inner.lock().unwrap().queue.pop_front()
    }

    /// Blocking; app thread only. Returns the oldest message, or `None` once every
    /// [`BusSender`] is gone *and* the queue has drained (mirrors `std::mpsc`).
    pub fn recv(&self) -> Option<BusMessage> {
        let mut inner = self.shared.inner.lock().unwrap();
        loop {
            if let Some(msg) = inner.queue.pop_front() {
                return Some(msg);
            }
            if inner.senders == 0 {
                return None; // disconnected and drained
            }
            inner = self.shared.ready.wait(inner).unwrap();
        }
    }

    /// Droppable messages dropped so far because the bus was full (spec: a slow-draining
    /// app degrades observably, with a counter). Critical messages are never counted
    /// here — they are never dropped.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// A port for the introspection server to attach/detach bus taps (spec: The two
    /// hot-path touches). Sharing the bus `Shared` means attach/detach/take_dropped are
    /// brief locks of the same mutex `send` uses — no second lock on the hot path.
    #[cfg(feature = "introspect")]
    pub(crate) fn tap_port(&self) -> BusTapPort {
        BusTapPort { shared: Arc::clone(&self.shared) }
    }
}

/// A handle for the introspection server to add/remove bus taps on a live bus (spec:
/// The two hot-path touches — "Bus gains tap_port"). Each op briefly locks the same
/// `Inner` mutex `send` holds, so a tap only ever perturbs a message post by the
/// bounded time to memcpy rows into rings.
#[cfg(feature = "introspect")]
pub(crate) struct BusTapPort {
    shared: Arc<Shared>,
}

#[cfg(feature = "introspect")]
impl BusTapPort {
    /// Attach a producer ring; returns its [`TapId`](crate::introspect::tap::TapId) for
    /// later detach. Ids are dense per port from 0.
    pub(crate) fn attach(
        &self,
        tx: crate::ring::Producer<crate::introspect::tap::BusTapRow>,
    ) -> crate::introspect::tap::TapId {
        use crate::introspect::tap::{BusTapSlot, TapId};
        let mut inner = self.shared.inner.lock().unwrap();
        // Dense id from the current max + 1 (taps are 1-2, so a linear scan is fine).
        let id = TapId(inner.taps.iter().map(|s| s.id.0 + 1).max().unwrap_or(0));
        inner.taps.push(BusTapSlot { id, tx, dropped: 0 });
        id
    }

    /// Detach a tap, returning its final drop count.
    pub(crate) fn detach(&self, id: crate::introspect::tap::TapId) -> u64 {
        let mut inner = self.shared.inner.lock().unwrap();
        if let Some(pos) = inner.taps.iter().position(|s| s.id == id) {
            let dropped = inner.taps[pos].dropped;
            inner.taps.remove(pos);
            dropped
        } else {
            0
        }
    }

    /// The current drop count for a tap (for a `Dropped` push when it advances).
    pub(crate) fn take_dropped(&self, id: crate::introspect::tap::TapId) -> u64 {
        let inner = self.shared.inner.lock().unwrap();
        inner
            .taps
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.dropped)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn err() -> Error {
        Error::Todo("test")
    }

    fn warn(n: i64) -> BusMessage {
        // A droppable carrying a marker so ordering is checkable.
        BusMessage::Qos { sink: ElementId(0), lateness_ns: n }
    }

    fn critical(n: i64) -> BusMessage {
        // A critical carrying a marker (LatencyChanged's `new` doubles as the marker).
        BusMessage::LatencyChanged {
            old: Timestamp::ZERO,
            new: Timestamp::from_nanos(n as u64),
        }
    }

    fn qos_marker(m: &BusMessage) -> Option<i64> {
        match m {
            BusMessage::Qos { lateness_ns, .. } => Some(*lateness_ns),
            _ => None,
        }
    }

    fn crit_marker(m: &BusMessage) -> Option<u64> {
        match m {
            BusMessage::LatencyChanged { new, .. } => new.nanos(),
            _ => None,
        }
    }

    #[test]
    fn classification_table() {
        // Every variant lands in the class the spec assigns it.
        use MessageClass::{Critical, Droppable};
        let cases: &[(BusMessage, MessageClass)] = &[
            (BusMessage::Error { element: ElementId(0), error: err() }, Critical),
            (BusMessage::Warning { element: ElementId(0), error: err() }, Droppable),
            (BusMessage::Eos, Critical),
            (BusMessage::StateChanged { old: State::Ready, new: State::Playing }, Critical),
            (
                BusMessage::PadAdded {
                    element: ElementId(0),
                    pad: PadId(0),
                    format: FixedFormat::new(crate::id::FormatId(0)),
                },
                Critical,
            ),
            (BusMessage::ElementAdded { element: ElementId(0), group: GroupId(0) }, Critical),
            (BusMessage::ElementRemoved { element: ElementId(0) }, Critical),
            (BusMessage::LinkChanged { link: LinkId(0) }, Critical),
            // `Tags` (Droppable) is omitted here only because `TagList` has no public
            // constructor yet (TODO(step 5)); its class is covered by `class()` directly.
            (
                BusMessage::SubgraphJoined { group: GroupId(0), added_latency: Timestamp::ZERO },
                Critical,
            ),
            (BusMessage::LatencyChanged { old: Timestamp::ZERO, new: Timestamp::ZERO }, Critical),
            (BusMessage::Qos { sink: ElementId(0), lateness_ns: 0 }, Droppable),
            (BusMessage::BranchSealed { group: GroupId(0), error: err() }, Critical),
        ];
        for (msg, want) in cases {
            assert_eq!(msg.class(), *want);
        }
    }

    #[test]
    fn droppable_drop_oldest_with_exact_counter() {
        let (tx, bus) = Bus::channel_with_capacity(4);
        // Post 10 droppables into a 4-slot bus: 6 drop, oldest-first.
        for i in 0..10 {
            tx.send(warn(i));
        }
        assert_eq!(bus.dropped(), 6, "exact drop count");

        // The survivors are the last four, in order — drop-oldest preserved ordering.
        let mut got = Vec::new();
        while let Some(m) = bus.try_recv() {
            got.push(qos_marker(&m).unwrap());
        }
        assert_eq!(got, vec![6, 7, 8, 9]);
    }

    #[test]
    fn critical_never_lost_evicts_oldest_droppable() {
        let (tx, bus) = Bus::channel_with_capacity(3);
        // Fill with droppables, then post a critical: it must land, evicting the oldest
        // droppable, and nothing critical is ever counted as dropped.
        tx.send(warn(1));
        tx.send(warn(2));
        tx.send(warn(3));
        tx.send(critical(100));
        assert_eq!(bus.dropped(), 1, "one droppable evicted for the critical");

        // Order of survivors: the two newest droppables, then the critical.
        let a = bus.recv().unwrap();
        let b = bus.recv().unwrap();
        let c = bus.recv().unwrap();
        assert_eq!(qos_marker(&a), Some(2));
        assert_eq!(qos_marker(&b), Some(3));
        assert_eq!(crit_marker(&c), Some(100));
        assert!(bus.try_recv().is_none());
    }

    #[test]
    fn critical_grows_past_cap_when_no_droppable_to_evict() {
        let (tx, bus) = Bus::channel_with_capacity(2);
        // An all-critical bus at capacity must still accept a critical — grow, don't drop.
        tx.send(critical(1));
        tx.send(critical(2));
        tx.send(critical(3)); // over the cap
        assert_eq!(bus.dropped(), 0, "no critical is ever dropped");
        let mut got = Vec::new();
        while let Some(m) = bus.try_recv() {
            got.push(crit_marker(&m).unwrap());
        }
        assert_eq!(got, vec![1, 2, 3], "all critical survive, in order");
    }

    #[test]
    fn droppable_into_all_critical_full_bus_drops_itself() {
        let (tx, bus) = Bus::channel_with_capacity(2);
        tx.send(critical(1));
        tx.send(critical(2));
        // No droppable to evict; the arriving droppable drops itself, criticals untouched.
        tx.send(warn(9));
        assert_eq!(bus.dropped(), 1);
        let a = bus.recv().unwrap();
        let b = bus.recv().unwrap();
        assert_eq!(crit_marker(&a), Some(1));
        assert_eq!(crit_marker(&b), Some(2));
        assert!(bus.try_recv().is_none(), "the droppable did not sneak in");
    }

    #[test]
    fn ordering_preserved_for_survivors_mixed() {
        let (tx, bus) = Bus::channel_with_capacity(4);
        // Interleave; then overflow with droppables. Criticals must all remain, in order,
        // and the surviving droppables are the newest, in order.
        tx.send(critical(10));
        tx.send(warn(1));
        tx.send(critical(20));
        tx.send(warn(2));
        // Bus is full (4). Two more droppables evict warn(1) then warn(2).
        tx.send(warn(3));
        tx.send(warn(4));
        assert_eq!(bus.dropped(), 2);

        let mut crits = Vec::new();
        let mut drops = Vec::new();
        while let Some(m) = bus.try_recv() {
            if let Some(c) = crit_marker(&m) {
                crits.push(c);
            } else if let Some(d) = qos_marker(&m) {
                drops.push(d);
            }
        }
        assert_eq!(crits, vec![10, 20], "criticals kept, in order");
        assert_eq!(drops, vec![3, 4], "newest droppables kept, in order");
    }

    #[test]
    fn recv_blocks_and_wakes_across_threads() {
        let (tx, bus) = Bus::channel_with_capacity(8);
        let woke = Arc::new(AtomicU64::new(0));
        let woke2 = woke.clone();
        let bus = Arc::new(bus);
        let bus2 = bus.clone();

        let h = thread::spawn(move || {
            // Blocks until the producer posts.
            let m = bus2.recv().expect("a message arrives");
            assert_eq!(crit_marker(&m), Some(42));
            woke2.fetch_add(1, Ordering::SeqCst);
        });

        // Give the receiver a moment to park, then post.
        thread::yield_now();
        tx.send(critical(42));
        h.join().unwrap();
        assert_eq!(woke.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn recv_returns_none_when_all_senders_drop() {
        let (tx, bus) = Bus::channel_with_capacity(4);
        let tx2 = tx.clone();
        drop(tx);
        drop(tx2);
        // Disconnected and drained → None, not a hang.
        assert!(bus.recv().is_none());
    }

    #[test]
    fn recv_drains_queued_before_reporting_disconnect() {
        let (tx, bus) = Bus::channel_with_capacity(4);
        tx.send(critical(1));
        tx.send(critical(2));
        drop(tx);
        // Queued messages come out first, then None.
        assert_eq!(crit_marker(&bus.recv().unwrap()), Some(1));
        assert_eq!(crit_marker(&bus.recv().unwrap()), Some(2));
        assert!(bus.recv().is_none());
    }

    #[test]
    fn send_after_bus_dropped_is_a_noop() {
        let (tx, bus) = Bus::channel_with_capacity(4);
        drop(bus);
        // The sender still holds the shared queue; posting must not panic (spec: silently
        // drops if the application has dropped the bus — here the queue simply fills and
        // is never drained).
        tx.send(critical(1));
        tx.send(warn(1));
    }
}
