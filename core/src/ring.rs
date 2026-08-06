//! Bounded lock-free single-producer / single-consumer ring — the boundary queue
//! between thread groups (spec: Queue internals). The most-executed data structure
//! in the framework, so its design is spec-level:
//!
//! - Power-of-two capacity; producer and consumer each own a cache line, so the
//!   fast path performs no shared-line writes the other side reads except the one
//!   index it publishes.
//! - Each side additionally owns a **cached copy of the peer index**
//!   (`Producer::cached_head` / `Consumer::cached_tail`), a plain non-atomic cell in
//!   the endpoint struct — private to the sole thread that touches it, never shared.
//!   The full/empty check tests the cache first; only when the cache says full/empty
//!   does it pay one `Acquire` load of the peer's shared line and refresh. A busy ring
//!   therefore performs *zero* shared-line loads on the room/emptiness check until the
//!   cache would block, amortizing that coherence traffic to roughly one line transfer
//!   per ring wrap (spec: Queue internals — cached peer index).
//! - Two atomics per element (Release-publish an index, Acquire-load the peer).
//! - Blocking `push`/`pop` layer parking over the lock-free core. Waking is a
//!   **Dekker handshake**: a `SeqCst` fence orders the item publish against a
//!   parker's re-check, and a per-side `waiting` flag is loaded on the fast path so a
//!   busy ring never touches the mutex. A naive "was it empty?" edge check is a lost
//!   wakeup — the peer advances its index concurrently, so the producer can read
//!   "non-empty", skip the fence, and *then* the consumer drains to empty and parks
//!   with no fence to pair against. So the fence runs on every publish; the cache
//!   speeds the room/emptiness decision, not the wake handshake. (This was measured:
//!   an edge-gated fence deadlocked a two-thread stream within ~10k items.)
//!
//! This is one of the two audited-`unsafe` areas in core (alongside `memory`).

#![allow(unsafe_code)]

use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

// The lock-free core is model-checked with loom under `--cfg loom`: that build
// swaps std's atomics/cell/sync primitives for loom's instrumented versions so
// `loom::model` can explore every interleaving of try_push/try_pop. Real builds
// (`cfg(not(loom))`) use std and carry no dependency. Only the primitives that
// participate in the memory-ordering surface are swapped; the shared-slot access
// is abstracted behind `slot_write`/`slot_read` because loom's `UnsafeCell` has a
// closure-based API (`with`/`with_mut`) unlike std's `get() -> *mut T`.
#[cfg(loom)]
use loom::cell::UnsafeCell;
#[cfg(loom)]
use loom::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
#[cfg(loom)]
use loom::sync::{Arc, Condvar, Mutex};

#[cfg(not(loom))]
use std::cell::UnsafeCell;
#[cfg(not(loom))]
use std::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
#[cfg(not(loom))]
use std::sync::{Arc, Condvar, Mutex};

/// Force producer and consumer indices onto separate cache lines (no false sharing).
#[repr(align(64))]
struct CachePadded<T>(T);

/// Write `val` into a ring slot. Behind a helper because std's `UnsafeCell` exposes
/// a raw `get() -> *mut T` while loom's needs a `with_mut(|p| ..)` closure.
///
/// # Safety
/// Caller must hold exclusive access to `slot` (SPSC: the sole producer writing a
/// free slot) and the slot must not currently hold an initialized value that would
/// be leaked.
#[cfg(not(loom))]
unsafe fn slot_write<T>(slot: &UnsafeCell<MaybeUninit<T>>, val: T) {
    (*slot.get()).write(val);
}
#[cfg(loom)]
unsafe fn slot_write<T>(slot: &UnsafeCell<MaybeUninit<T>>, val: T) {
    slot.with_mut(|p| (*p).write(val));
}

/// Read a value out of a ring slot, taking ownership. Mirror of [`slot_write`].
///
/// # Safety
/// Caller must hold exclusive access to `slot` (SPSC: the sole consumer reading a
/// published slot) and the slot must currently hold an initialized value; the value
/// is moved out, so the caller must not read it again.
#[cfg(not(loom))]
unsafe fn slot_read<T>(slot: &UnsafeCell<MaybeUninit<T>>) -> T {
    (*slot.get()).assume_init_read()
}
#[cfg(loom)]
unsafe fn slot_read<T>(slot: &UnsafeCell<MaybeUninit<T>>) -> T {
    slot.with(|p| (*p).assume_init_read())
}

/// Drop the initialized value in a ring slot in place (drain on `Inner::drop`).
///
/// # Safety
/// Caller must hold exclusive access and the slot must currently hold an
/// initialized value that has not been consumed.
#[cfg(not(loom))]
unsafe fn slot_drop<T>(slot: &UnsafeCell<MaybeUninit<T>>) {
    (*slot.get()).assume_init_drop();
}
#[cfg(loom)]
unsafe fn slot_drop<T>(slot: &UnsafeCell<MaybeUninit<T>>) {
    slot.with_mut(|p| (*p).assume_init_drop());
}

/// Overflow policy for a leaky ring (spec: Queue internals — "leaky modes are a
/// producer-side decision made before writing"). Applies only to the *blocking*
/// [`Producer::push`]; [`Producer::try_push`] is already the caller's own
/// drop-newest primitive (it returns the value on full).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leaky {
    /// Full ring blocks the producer until the consumer drains (the default; what a
    /// plain [`spsc`] ring does). Lossless backpressure — for offline/bulk paths.
    Block,
    /// Full ring drops the *incoming* value and bumps [`Producer::dropped`], never
    /// blocking. For live sources that must not stall (spec: leaky policies for live
    /// sources): a late frame is better dropped than allowed to back up the graph.
    DropNewest,
    // DropOldest is deliberately absent — see the `spsc_leaky` docs for why an SPSC
    // ring can't evict the oldest queued item from the producer side without breaking
    // the single-consumer ownership of `head`.
}

struct Inner<T> {
    buf: Box<[UnsafeCell<MaybeUninit<T>>]>,
    cap: usize,
    mask: usize,
    /// Overflow policy consulted by [`Producer::push`] before it decides to block.
    leaky: Leaky,
    /// Consumer-owned position (monotonic; masked for indexing).
    head: CachePadded<AtomicUsize>,
    /// Producer-owned position (monotonic; masked for indexing).
    tail: CachePadded<AtomicUsize>,
    /// Count of values dropped by a leaky `push` (spec: "drop counts on the
    /// counters"). Producer-owned: only the producer stores it, only the producer
    /// reads it via [`Producer::dropped`], so `Relaxed` suffices — it carries no
    /// ordering, just a monotonic tally. Sits on the producer's line conceptually,
    /// though it's cold (touched only on the leak path).
    dropped: AtomicUsize,
    /// Set when either end is dropped.
    closed: AtomicBool,
    /// A parked consumer / producer wants to be woken (Dekker handshake flags).
    consumer_waiting: AtomicBool,
    producer_waiting: AtomicBool,
    /// Parking gate for the blocking API; untouched on the lock-free fast path.
    gate: Mutex<()>,
    cvar: Condvar,
}

// SAFETY: access is disciplined SPSC — exactly one producer and one consumer, each
// touching only its own index plus the shared slots it is allowed to at that time.
unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send> Sync for Inner<T> {}

impl<T> Inner<T> {
    fn is_empty(&self) -> bool {
        self.head.0.load(Ordering::Acquire) == self.tail.0.load(Ordering::Acquire)
    }

    fn is_full(&self) -> bool {
        let tail = self.tail.0.load(Ordering::Acquire);
        let head = self.head.0.load(Ordering::Acquire);
        tail.wrapping_sub(head) == self.cap
    }

    fn wake(&self) {
        // Serialize against a waiter's under-lock re-check so a notify can't slip in
        // between "decide to wait" and "wait".
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.cvar.notify_all();
    }
}

impl<T> Drop for Inner<T> {
    // Real build: sole owner at drop, so drop any undelivered elements in
    // [head, tail). `get_mut()` gives the final indices without atomics.
    #[cfg(not(loom))]
    fn drop(&mut self) {
        let head = *self.head.0.get_mut();
        let tail = *self.tail.0.get_mut();
        let mut h = head;
        while h != tail {
            let idx = h & self.mask;
            // SAFETY: slots in [head, tail) are initialized and not yet consumed.
            unsafe {
                slot_drop(&self.buf[idx]);
            }
            h = h.wrapping_add(1);
        }
    }

    // Loom build: `AtomicUsize::get_mut()` doesn't exist under loom, and the model
    // only ever exercises `usize` payloads (no drop glue to run), so skip the drain.
    // This keeps the drop out of loom's cell bookkeeping entirely.
    #[cfg(loom)]
    fn drop(&mut self) {}
}

/// The producing end. `Send` (movable to its group's thread) but not `Sync`.
pub struct Producer<T> {
    inner: Arc<Inner<T>>,
    /// Cached copy of the consumer's `head`, private to the producer thread. The
    /// producer knows there is room whenever `tail - cached_head < cap`; only when
    /// that says full does it reload the shared `head` (Acquire) and refresh here.
    /// A plain `Cell` (not atomic) because exactly one thread — the producer — ever
    /// reads or writes it: it establishes no cross-thread ordering, so it stays off
    /// loom's synchronization surface. `cached_head <= real head` always (the
    /// consumer only advances `head`), so a stale value is *conservative*: it can
    /// only make the producer think the ring is fuller than it is, never emptier —
    /// never an overwrite of a live slot.
    cached_head: Cell<usize>,
    /// `Producer` must stay `!Sync` even though `Cell` already implies it; kept
    /// explicit for intent and to match `Consumer`.
    _not_sync: PhantomData<Cell<()>>,
}

/// The consuming end. `Send` but not `Sync`.
pub struct Consumer<T> {
    inner: Arc<Inner<T>>,
    /// Cached copy of the producer's `tail`, private to the consumer thread. The
    /// consumer knows there is an item whenever `head != cached_tail`; only when that
    /// says empty does it reload the shared `tail` (Acquire) and refresh here. Same
    /// reasoning as [`Producer::cached_head`]: `cached_tail <= real tail`, so a stale
    /// value is conservative — it can only make the consumer think the ring is
    /// emptier than it is, never fuller, so it never reads an unpublished slot.
    cached_tail: Cell<usize>,
    _not_sync: PhantomData<Cell<()>>,
}

/// Create a ring holding up to `capacity` elements (rounded up to a power of two).
/// Blocking-on-full (`Leaky::Block`); see [`spsc_leaky`] for live-source policies.
pub fn spsc<T>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    spsc_leaky(capacity, Leaky::Block)
}

/// Create a ring with an explicit overflow [`Leaky`] policy (spec: Queue internals —
/// "leaky modes are a producer-side decision made before writing … dropping never
/// touches the consumer's line"). Additive over [`spsc`]; the fast path, the shared
/// atomics, and the consumer are byte-for-byte identical regardless of policy — the
/// policy is read only by the blocking [`Producer::push`] when it would otherwise
/// park.
///
/// `Leaky::DropNewest` drops the *incoming* value on a full ring and tallies it in
/// [`Producer::dropped`]. This is the only leak an SPSC ring can perform purely
/// producer-side.
///
/// # Why not `DropOldest`
/// Drop-oldest means evicting the *oldest queued* item to make room — advancing
/// `head` past a slot and running that value's destructor. But `head` is the
/// consumer's cache-line-owned index and the slot it points at is the consumer's to
/// read; a producer that bumped `head` and dropped `buf[head]` would (a) write the
/// line the whole design keeps producer-private for zero false sharing, racing the
/// consumer's own `head` store, and (b) potentially drop a value the consumer is
/// mid-read of — a use-after-free. So drop-oldest is *not* a producer-side decision
/// in an SPSC ring; it needs a different mechanism, e.g. a consumer-side "skip
/// stale" (the consumer drops instead of delivering when a generation says so, as
/// the flush path already does via the generation counter), or a two-index scheme
/// (a separate reclaim index the producer may advance, distinct from the consumer's
/// read index). Both are larger changes than this queue owns today, so `DropOldest`
/// is *designed but unbuilt* rather than shipped wrong. It is intentionally not a
/// [`Leaky`] variant so a caller can't select a policy that silently degrades to
/// blocking.
pub fn spsc_leaky<T>(capacity: usize, leaky: Leaky) -> (Producer<T>, Consumer<T>) {
    let cap = capacity.max(1).next_power_of_two();
    let buf = (0..cap)
        .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let inner = Arc::new(Inner {
        buf,
        cap,
        mask: cap - 1,
        leaky,
        head: CachePadded(AtomicUsize::new(0)),
        tail: CachePadded(AtomicUsize::new(0)),
        dropped: AtomicUsize::new(0),
        closed: AtomicBool::new(false),
        consumer_waiting: AtomicBool::new(false),
        producer_waiting: AtomicBool::new(false),
        gate: Mutex::new(()),
        cvar: Condvar::new(),
    });
    (
        Producer {
            inner: Arc::clone(&inner),
            cached_head: Cell::new(0),
            _not_sync: PhantomData,
        },
        Consumer { inner, cached_tail: Cell::new(0), _not_sync: PhantomData },
    )
}

impl<T> Producer<T> {
    /// The ring's slot count (the `capacity` given at creation, rounded up to a
    /// power of two) — how many items can be in flight at once.
    pub fn capacity(&self) -> usize {
        self.inner.cap
    }

    /// Non-blocking push. `Err(val)` if the ring is full.
    pub fn try_push(&self, val: T) -> Result<(), T> {
        // `tail` is producer-owned: `Relaxed` — no other thread stores it, and the
        // ordering that matters (slot write → tail publish) is the Release below.
        let tail = self.inner.tail.0.load(Ordering::Relaxed);

        // Room check off the *cached* head first — no shared-line load when the cache
        // already proves room (`tail - cached_head < cap`). `cached_head <= head`, so
        // this is conservative: it can only under-estimate room, never over. Refresh
        // from the shared `head` (Acquire, pairs with the consumer's Release store of
        // `head`) only when the cache says full, then re-test.
        if tail.wrapping_sub(self.cached_head.get()) == self.inner.cap {
            let head = self.inner.head.0.load(Ordering::Acquire);
            self.cached_head.set(head);
            if tail.wrapping_sub(head) == self.inner.cap {
                return Err(val);
            }
        }

        let idx = tail & self.inner.mask;
        // SAFETY: this slot is free (past head); we are the only producer.
        unsafe {
            slot_write(&self.inner.buf[idx], val);
        }
        // Publish: Release pairs with the consumer's Acquire load of `tail`, making
        // the slot write visible before the index that exposes it.
        self.inner.tail.0.store(tail.wrapping_add(1), Ordering::Release);

        // Dekker handshake, on *every* publish. The `SeqCst` fence orders this publish
        // against a consumer that is about to park, then we wake it iff it flagged
        // itself waiting — no mutex on the busy fast path.
        //
        // This fence is load-bearing and cannot be gated on an "empty edge": a naive
        // "only fence when the ring was empty" scheme is a lost wakeup, because the
        // consumer advances `head` concurrently. The producer would read `head`, see
        // the ring non-empty, skip the fence — and *then* the consumer drains to empty
        // and parks, with no fence to pair against its park re-check. (Empirically
        // reproducible: a two-thread stream deadlocks within ~10k items with the ring
        // full and the consumer asleep.) The cheap win is above — the cached-index
        // room check spares the shared `head` load on the busy path — not here.
        fence(Ordering::SeqCst);
        if self.inner.consumer_waiting.load(Ordering::Relaxed) {
            self.inner.wake();
        }
        Ok(())
    }

    /// Blocking push, subject to the ring's [`Leaky`] policy on full.
    ///
    /// - `Leaky::Block` (the [`spsc`] default): blocks while full; `Err(val)` once the
    ///   consumer is gone.
    /// - `Leaky::DropNewest`: never blocks — on a full ring it drops `val`, bumps
    ///   [`dropped`](Self::dropped), and returns `Ok(())`. It still `Err(val)`s if the
    ///   consumer is gone (nothing will ever drain), so the caller can stop.
    pub fn push(&self, mut val: T) -> Result<(), T> {
        loop {
            match self.try_push(val) {
                Ok(()) => return Ok(()),
                Err(v) => {
                    val = v;
                    if self.inner.closed.load(Ordering::Acquire) {
                        return Err(val); // consumer gone
                    }
                    // Producer-side leak decision, made before ever touching the gate
                    // and never touching the consumer's line (spec: leaky modes). Drop
                    // the incoming value and count it instead of parking.
                    if self.inner.leaky == Leaky::DropNewest {
                        // Relaxed: producer-owned tally, carries no cross-thread order.
                        self.inner.dropped.fetch_add(1, Ordering::Relaxed);
                        drop(val);
                        return Ok(());
                    }
                    let mut guard = self.inner.gate.lock().unwrap();
                    self.inner.producer_waiting.store(true, Ordering::Relaxed);
                    fence(Ordering::SeqCst); // pairs with the consumer's pop fence
                    if self.inner.is_full() && !self.inner.closed.load(Ordering::Acquire) {
                        guard = self.inner.cvar.wait(guard).unwrap();
                    }
                    self.inner.producer_waiting.store(false, Ordering::Relaxed);
                    drop(guard);
                }
            }
        }
    }

    /// Number of values dropped so far by a leaky [`push`](Self::push) on a full ring
    /// (`Leaky::DropNewest`). Producer-readable, monotonic; always `0` for a
    /// `Leaky::Block` ring (spec: Queue internals — drop counts on the counters).
    pub fn dropped(&self) -> usize {
        self.inner.dropped.load(Ordering::Relaxed)
    }

    /// Whether the consumer has been dropped.
    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }

    /// Items currently queued, from the producer's side. Momentary — the consumer
    /// may be draining concurrently, so this can overestimate; used for queue-fill
    /// observability (spec: Taps — backpressure is the queue high-water), never for
    /// flow-control decisions.
    pub fn len(&self) -> usize {
        let tail = self.inner.tail.0.load(Ordering::Relaxed);
        let head = self.inner.head.0.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Consumer<T> {
    /// The ring's slot count — see [`Producer::capacity`].
    pub fn capacity(&self) -> usize {
        self.inner.cap
    }

    /// Non-blocking pop. `None` if the ring is empty.
    pub fn try_pop(&self) -> Option<T> {
        // `head` is consumer-owned: `Relaxed`. The publish ordering that matters
        // (producer's slot write → tail) is picked up by the Acquire refresh below.
        let head = self.inner.head.0.load(Ordering::Relaxed);

        // Emptiness check off the *cached* tail first — no shared-line load while the
        // cache still shows an item (`head != cached_tail`). `cached_tail <= tail`, so
        // this is conservative: it can only under-estimate available items, never
        // over. Refresh from the shared `tail` (Acquire, pairs with the producer's
        // Release store of `tail`) only when the cache says empty, then re-test.
        if head == self.cached_tail.get() {
            let tail = self.inner.tail.0.load(Ordering::Acquire);
            self.cached_tail.set(tail);
            if head == tail {
                return None;
            }
        }

        let idx = head & self.inner.mask;
        // SAFETY: this slot (before tail) is initialized; we are the only consumer.
        let val = unsafe { slot_read(&self.inner.buf[idx]) };
        // Release pairs with the producer's Acquire load of `head`: the slot is fully
        // read before the index frees it for reuse.
        self.inner.head.0.store(head.wrapping_add(1), Ordering::Release);

        // Dekker handshake, on *every* pop — mirror of `try_push`. The `SeqCst` fence
        // orders this slot-free against a producer about to park on full; wake it iff
        // it flagged itself. Cannot be gated on a "full edge" for the same reason the
        // producer's fence can't be gated on an empty edge: the producer advances
        // `tail` concurrently, so a "was it full" snapshot races the producer's park.
        fence(Ordering::SeqCst);
        if self.inner.producer_waiting.load(Ordering::Relaxed) {
            self.inner.wake();
        }
        Some(val)
    }

    /// Whether the producer has been dropped (the ring may still hold undrained
    /// items — keep calling [`try_pop`](Self::try_pop) until it returns `None`).
    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }

    /// Items currently queued, from the consumer's side. Momentary (the producer may
    /// be pushing concurrently); used by the scheduler to distinguish "closed and
    /// drained" from "closed with a tail still queued" when its backlog cap left
    /// items behind.
    pub fn len(&self) -> usize {
        let head = self.inner.head.0.load(Ordering::Relaxed);
        let tail = self.inner.tail.0.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Blocking pop. Blocks while empty; `None` once the producer is gone *and* the
    /// ring has drained.
    pub fn pop(&self) -> Option<T> {
        loop {
            if let Some(v) = self.try_pop() {
                return Some(v);
            }
            if self.inner.closed.load(Ordering::Acquire) {
                // Producer gone; a push may have landed just before it closed.
                return self.try_pop();
            }
            let mut guard = self.inner.gate.lock().unwrap();
            self.inner.consumer_waiting.store(true, Ordering::Relaxed);
            fence(Ordering::SeqCst); // pairs with the producer's push fence
            if self.inner.is_empty() && !self.inner.closed.load(Ordering::Acquire) {
                guard = self.inner.cvar.wait(guard).unwrap();
            }
            self.inner.consumer_waiting.store(false, Ordering::Relaxed);
            drop(guard);
        }
    }

    /// [`pop`](Self::pop) with a deadline: the same blocking wait, but it also gives
    /// up once `timeout` elapses. `None` therefore means *either* "timed out" or
    /// "closed and drained" — tell them apart with [`is_closed`](Self::is_closed)
    /// plus [`is_empty`](Self::is_empty), exactly as the non-blocking path does.
    ///
    /// The scheduler uses this for a head element that has asked to be re-run by a
    /// running time ([`Ctx::wake_at`](crate::ctx::Ctx::wake_at)): the untimed `pop`
    /// wakes only on a push, which strands data that ripens with the *clock*.
    pub fn pop_timeout(&self, timeout: std::time::Duration) -> Option<T> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(v) = self.try_pop() {
                return Some(v);
            }
            if self.inner.closed.load(Ordering::Acquire) {
                return self.try_pop();
            }
            let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) else {
                return None; // deadline reached (a zero timeout degrades to try_pop)
            };
            let mut guard = self.inner.gate.lock().unwrap();
            self.inner.consumer_waiting.store(true, Ordering::Relaxed);
            fence(Ordering::SeqCst); // pairs with the producer's push fence
            if self.inner.is_empty() && !self.inner.closed.load(Ordering::Acquire) {
                guard = self.inner.cvar.wait_timeout(guard, left).unwrap().0;
            }
            self.inner.consumer_waiting.store(false, Ordering::Relaxed);
            drop(guard);
        }
    }
}

impl<T> Drop for Producer<T> {
    fn drop(&mut self) {
        self.inner.closed.store(true, Ordering::Release);
        self.inner.wake();
    }
}

impl<T> Drop for Consumer<T> {
    fn drop(&mut self) {
        self.inner.closed.store(true, Ordering::Release);
        self.inner.wake();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_thread_fifo_and_full() {
        let (p, c) = spsc::<u32>(4);
        assert!(c.try_pop().is_none(), "starts empty");
        for i in 0..4 {
            assert!(p.try_push(i).is_ok());
        }
        assert_eq!(p.try_push(99), Err(99), "full at capacity");
        for i in 0..4 {
            assert_eq!(c.try_pop(), Some(i), "FIFO");
        }
        assert!(c.try_pop().is_none());
    }

    #[test]
    fn wraps_around() {
        let (p, c) = spsc::<usize>(2);
        for round in 0..1000 {
            assert!(p.try_push(round).is_ok());
            assert_eq!(c.try_pop(), Some(round));
        }
    }

    #[test]
    fn blocking_transfer_across_threads() {
        const N: usize = 1_000_000;
        let (p, c) = spsc::<usize>(64);
        let producer = std::thread::spawn(move || {
            for i in 0..N {
                p.push(i).expect("consumer alive");
            }
        });
        let consumer = std::thread::spawn(move || {
            for expected in 0..N {
                let got = c.pop().expect("producer alive until N");
                assert_eq!(got, expected, "FIFO preserved, nothing lost");
            }
            assert!(c.pop().is_none(), "closed and drained");
        });
        producer.join().unwrap();
        consumer.join().unwrap();
    }

    #[test]
    fn timed_pop_gives_up_and_still_sees_a_late_push_and_a_close() {
        let (p, c) = spsc::<usize>(4);
        // Empty and open: the wait runs out and reports nothing, without
        // consuming the ring's ability to deliver later.
        let at = std::time::Instant::now();
        assert_eq!(c.pop_timeout(std::time::Duration::from_millis(30)), None);
        assert!(at.elapsed() >= std::time::Duration::from_millis(25), "it actually waited");
        assert!(!c.is_closed(), "a timeout is not a close");

        // A push landing mid-wait wakes it, exactly like the untimed `pop`.
        let pusher = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            p.push(7).expect("consumer alive");
            // Dropping the producer here closes the ring.
        });
        let at = std::time::Instant::now();
        assert_eq!(c.pop_timeout(std::time::Duration::from_secs(10)), Some(7));
        assert!(at.elapsed() < std::time::Duration::from_secs(5), "woke on the push, not the timeout");
        pusher.join().unwrap();

        // Closed and drained: `None` immediately, and `is_closed` is what tells
        // this apart from the timeout above.
        assert_eq!(c.pop_timeout(std::time::Duration::from_secs(10)), None);
        assert!(c.is_closed());
    }

    #[test]
    fn tiny_capacity_stress() {
        // Capacity 1 makes the ring oscillate empty<->full on every element, so a
        // parked waiter is the common case — the harshest test for the blocking
        // wakeup path (and the lost-wakeup regression).
        const N: usize = 100_000;
        let (p, c) = spsc::<usize>(1);
        let prod = std::thread::spawn(move || {
            for i in 0..N {
                p.push(i).expect("consumer alive");
            }
        });
        let cons = std::thread::spawn(move || {
            for expected in 0..N {
                assert_eq!(c.pop(), Some(expected));
            }
            assert!(c.pop().is_none());
        });
        prod.join().unwrap();
        cons.join().unwrap();
    }

    #[test]
    fn producer_drop_closes_after_drain() {
        let (p, c) = spsc::<i32>(8);
        p.push(1).unwrap();
        p.push(2).unwrap();
        drop(p);
        assert_eq!(c.pop(), Some(1));
        assert_eq!(c.pop(), Some(2));
        assert_eq!(c.pop(), None, "drained + closed");
    }

    #[test]
    fn consumer_drop_unblocks_producer() {
        let (p, c) = spsc::<i32>(1);
        p.push(1).unwrap(); // fills it
        drop(c);
        assert_eq!(p.push(2), Err(2), "consumer gone → push fails instead of hanging");
    }

    #[test]
    fn drop_releases_undelivered_elements() {
        let (p, c) = spsc::<Box<u64>>(8);
        p.try_push(Box::new(1)).unwrap();
        p.try_push(Box::new(2)).unwrap();
        drop(p);
        drop(c); // Inner::drop must drop the two boxed values exactly once
    }

    #[test]
    fn leaky_drop_newest_counts_and_never_blocks() {
        // A full DropNewest ring must not block `push`; it drops the incoming value
        // and tallies it. Capacity rounds up to 2.
        let (p, _c) = spsc_leaky::<usize>(2, Leaky::DropNewest);
        assert_eq!(p.dropped(), 0);
        p.push(0).unwrap();
        p.push(1).unwrap(); // ring now full (cap 2)
        // These would block a `Leaky::Block` ring forever; here they drop + count and
        // return Ok immediately (the test itself completing proves "never blocks").
        p.push(2).unwrap();
        p.push(3).unwrap();
        assert_eq!(p.dropped(), 2, "two overflow values dropped and counted");
    }

    #[test]
    fn leaky_drop_newest_keeps_the_oldest_in_order() {
        // Dropping the *newest* means the two survivors are the *first* two pushed,
        // still in FIFO order — the queued prefix is untouched.
        let (p, c) = spsc_leaky::<usize>(2, Leaky::DropNewest);
        for v in 0..5 {
            p.push(v).unwrap();
        }
        assert_eq!(p.dropped(), 3);
        assert_eq!(c.try_pop(), Some(0));
        assert_eq!(c.try_pop(), Some(1));
        assert_eq!(c.try_pop(), None, "only the first two survived");
    }

    #[test]
    fn block_default_has_zero_drops() {
        // The plain `spsc` constructor is `Leaky::Block`: `dropped()` stays 0.
        let (p, c) = spsc::<usize>(2);
        p.push(0).unwrap();
        assert_eq!(c.try_pop(), Some(0));
        assert_eq!(p.dropped(), 0);
    }
}

// Loom model-checks the lock-free core (`try_push`/`try_pop`) across *every*
// interleaving, verifying the head/tail Acquire/Release protocol, the cached-index
// room/emptiness fast path, the per-publish SeqCst wake fence, and the `*_waiting`
// loads. The blocking `push`/`pop` themselves (Mutex + Condvar) are intentionally
// *not* modelled, but the *lock-free half* of their wake handshake — the part that
// must not lose a wakeup — is modelled directly in `no_lost_wakeup_on_empty_edge`.
//
// Run with: `RUSTFLAGS="--cfg loom" cargo test -p profluens-core --lib loom`
#[cfg(loom)]
mod loom_tests {
    use super::*;

    #[test]
    fn spsc_fifo_no_loss_no_dup() {
        // Op counts are deliberately tiny — loom explores interleavings
        // exponentially (N=4 over cap=2 already runs for minutes). 3 pushes over a
        // capacity-2 ring forces at least one full/empty oscillation (a wrap), which
        // is what exercises the cached_head/cached_tail refresh: the cache goes stale,
        // says full/empty, then reloads the shared index on the room/emptiness check.
        // The dedicated `no_lost_wakeup_*` models cover the fence/flag handshake when
        // a publish/pop races a parking peer.
        const N: usize = 3;

        loom::model(|| {
            let (p, c) = spsc::<usize>(2);

            let producer = loom::thread::spawn(move || {
                for i in 0..N {
                    // Retry on `Err` (ring full): the consumer will drain and make
                    // room. Under loom every retry ordering is explored.
                    while p.try_push(i).is_err() {
                        loom::thread::yield_now();
                    }
                }
            });

            // Consumer collects everything it observes; `try_pop` returns `None`
            // while empty (producer hasn't published yet), so spin until we've
            // seen all N. Loom's bounded scheduler guarantees progress.
            let mut received = Vec::with_capacity(N);
            while received.len() < N {
                if let Some(v) = c.try_pop() {
                    received.push(v);
                } else {
                    loom::thread::yield_now();
                }
            }

            producer.join().unwrap();

            // FIFO with no loss and no duplication: the exact sequence [0, 1, .., N).
            let expected: Vec<usize> = (0..N).collect();
            assert_eq!(received, expected, "FIFO order preserved, nothing lost or duplicated");

            // Ring is fully drained.
            assert!(c.try_pop().is_none(), "nothing left after all items consumed");
        });
    }

    // The load-bearing wakeup proof. The blocking `pop` decides to sleep like this:
    //   1. try_pop() saw empty            2. set consumer_waiting = true
    //   3. fence(SeqCst)                  4. re-check is_empty(); sleep iff still empty
    // and `try_push`, on every publish, does the mirror:
    //   A. publish the item (Release)     B. fence(SeqCst)
    //   C. load consumer_waiting; wake iff true
    // The invariant that must hold across *every* interleaving: the consumer never
    // ends up "asleep" (would-block after step 4) while an item sits unconsumed and
    // the producer skipped the wake at step C. We model exactly steps 1-4 / A-C with
    // plain atomics (no Condvar) and assert the disjunction: either the consumer
    // sees the item at step 4 (won't sleep), or the producer sees the flag at step C
    // (will wake). A lost wakeup is the case where *both* fail. This is precisely why
    // the fence can't be gated on an "empty edge": step B must always run, or a
    // publish that races a parking consumer leaves the two fences unpaired.
    #[test]
    fn no_lost_wakeup_on_empty_edge() {
        loom::model(|| {
            // Capacity-1 ring, starting empty: model the two handshake sequences with
            // the *exact* atomics/orderings the real `try_push` (steps A/B/C) and
            // `pop` (steps 1-4) use — the real methods' FIFO/no-loss is covered by
            // `spsc_fifo_no_loss_no_dup`; here we isolate the flag/fence Dekker so the
            // assertion can observe both verdicts.
            let (p, c) = spsc::<usize>(1);
            let inner_p = Arc::clone(&p.inner);
            let inner_c = Arc::clone(&c.inner);

            let producer = loom::thread::spawn(move || {
                // A: publish into the empty ring (slot then tail-Release), exactly as
                // try_push does. Slot write elided — no consumer reads it here.
                inner_p.tail.0.store(1, Ordering::Release);
                // B + C: the wake handshake try_push runs on *every* publish — the
                // SeqCst fence, then the waiting-flag load.
                fence(Ordering::SeqCst);
                inner_p.consumer_waiting.load(Ordering::Relaxed) // did we see a waiter?
            });

            let consumer = loom::thread::spawn(move || {
                // 1: observe emptiness via the fast path (Relaxed own head, Acquire
                // peer tail — as try_pop does before deciding it's empty).
                if inner_c.head.0.load(Ordering::Relaxed) == inner_c.tail.0.load(Ordering::Acquire)
                {
                    // 2 + 3 + 4: the park decision, exactly as `pop` does it.
                    inner_c.consumer_waiting.store(true, Ordering::Relaxed);
                    fence(Ordering::SeqCst);
                    let still_empty = inner_c.is_empty();
                    // Crucial fidelity to the real code: the flag stays TRUE while the
                    // consumer is parked (it is cleared only *after* cvar.wait returns,
                    // i.e. after a wakeup). So on the "would sleep" branch we leave it
                    // set — that is precisely the state the producer must observe. Only
                    // the not-parking branch clears it (the consumer proceeds to pop).
                    if !still_empty {
                        inner_c.consumer_waiting.store(false, Ordering::Relaxed);
                    }
                    still_empty // `true` == "the consumer would go to sleep now".
                } else {
                    false // saw the item without parking
                }
            });

            let producer_saw_waiter = producer.join().unwrap();
            let consumer_would_sleep = consumer.join().unwrap();

            // Lost wakeup ⇔ consumer sleeps AND producer didn't wake it. Forbidden.
            assert!(
                !(consumer_would_sleep && !producer_saw_waiter),
                "lost wakeup: consumer parked on empty while producer skipped the wake",
            );
            // Sanity: the item is present regardless (nothing lost).
            assert_eq!(c.inner.tail.0.load(Ordering::Acquire), 1, "item was published");
        });
    }

    // Mirror of the above for the full→non-full edge: a producer parked on `full`
    // must be woken by the consumer's pop. Same Dekker structure, opposite sides.
    #[test]
    fn no_lost_wakeup_on_full_edge() {
        loom::model(|| {
            // Capacity-1 ring, pre-filled (tail=1, head=0 ⇒ full): the consumer's pop
            // is the full→non-full edge.
            let (p, c) = spsc::<usize>(1);
            p.try_push(7).expect("empty ring has room");
            let inner_p = Arc::clone(&p.inner);
            let inner_c = Arc::clone(&c.inner);

            let consumer = loom::thread::spawn(move || {
                // Consume: advance head-Release, then the handshake try_pop runs on
                // every pop (fence + producer_waiting load).
                inner_c.head.0.store(1, Ordering::Release);
                fence(Ordering::SeqCst);
                inner_c.producer_waiting.load(Ordering::Relaxed) // saw a waiter?
            });

            let producer = loom::thread::spawn(move || {
                // Producer's park decision when full (as `push` does it). The flag
                // stays set while parked — cleared only after cvar.wait returns — so
                // only the not-parking branch clears it (mirror of the empty-edge
                // model above).
                if inner_p.is_full() {
                    inner_p.producer_waiting.store(true, Ordering::Relaxed);
                    fence(Ordering::SeqCst);
                    let still_full = inner_p.is_full();
                    if !still_full {
                        inner_p.producer_waiting.store(false, Ordering::Relaxed);
                    }
                    still_full
                } else {
                    false
                }
            });

            let consumer_saw_waiter = consumer.join().unwrap();
            let producer_would_sleep = producer.join().unwrap();

            assert!(
                !(producer_would_sleep && !consumer_saw_waiter),
                "lost wakeup: producer parked on full while consumer skipped the wake",
            );
        });
    }

    // DropNewest leak is a pure producer-side decision that never touches the
    // consumer's line — model that the counter tally and the FIFO of surviving items
    // stay consistent under interleaving. Producer pushes 3 into a capacity-1 leaky
    // ring while the consumer drains; every item is either delivered (in order) or
    // counted as dropped, and the two never overlap.
    #[test]
    fn leaky_drop_newest_conserves_items() {
        const N: usize = 3;
        loom::model(|| {
            let (p, c) = spsc_leaky::<usize>(1, Leaky::DropNewest);

            let producer = loom::thread::spawn(move || {
                for i in 0..N {
                    // push() never blocks under DropNewest: Ok whether delivered or
                    // dropped. It only Errs if the consumer vanished.
                    p.push(i).expect("consumer alive");
                }
                p.dropped()
            });

            let mut received = Vec::new();
            // Drain until the producer is done and the ring is empty. Loom bounds
            // progress, so spin-with-yield terminates.
            loop {
                match c.try_pop() {
                    Some(v) => received.push(v),
                    None => {
                        if c.is_closed() {
                            while let Some(v) = c.try_pop() {
                                received.push(v);
                            }
                            break;
                        }
                        loom::thread::yield_now();
                    }
                }
            }
            let dropped = producer.join().unwrap();

            // Conservation: delivered + dropped == N, nothing invented, nothing
            // vanished uncounted.
            assert_eq!(received.len() + dropped, N, "every item delivered or counted");
            // Delivered items are a prefix-preserving subsequence of [0..N): strictly
            // increasing (FIFO, no reorder, no dup).
            for w in received.windows(2) {
                assert!(w[0] < w[1], "delivered items keep FIFO order, no duplicates");
            }
        });
    }
}
