//! Bounded lock-free single-producer / single-consumer ring — the boundary queue
//! between thread groups (spec: Queue internals). The most-executed data structure
//! in the framework, so its design is spec-level:
//!
//! - Power-of-two capacity; producer and consumer each own a cache line, so the
//!   fast path performs no shared-line writes the other side reads except the one
//!   index it publishes.
//! - Two atomics per element (Release-publish an index, Acquire-load the peer).
//! - Blocking `push`/`pop` layer parking over the lock-free core. Waking is a
//!   **Dekker handshake**: a `SeqCst` fence orders the item publish against a
//!   parker's re-check, and a per-side `waiting` flag is loaded on the fast path so
//!   a busy ring never touches the mutex. A naive "was it empty?" edge check is a
//!   lost wakeup — the peer can flip that state between the check and the publish.
//!
//! This is one of the two audited-`unsafe` areas in core (alongside `memory`).

#![allow(unsafe_code)]

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Force producer and consumer indices onto separate cache lines (no false sharing).
#[repr(align(64))]
struct CachePadded<T>(T);

struct Inner<T> {
    buf: Box<[UnsafeCell<MaybeUninit<T>>]>,
    cap: usize,
    mask: usize,
    /// Consumer-owned position (monotonic; masked for indexing).
    head: CachePadded<AtomicUsize>,
    /// Producer-owned position (monotonic; masked for indexing).
    tail: CachePadded<AtomicUsize>,
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
    fn drop(&mut self) {
        // Sole owner at drop: drop any undelivered elements in [head, tail).
        let head = *self.head.0.get_mut();
        let tail = *self.tail.0.get_mut();
        let mut h = head;
        while h != tail {
            let idx = h & self.mask;
            // SAFETY: slots in [head, tail) are initialized and not yet consumed.
            unsafe {
                (*self.buf[idx].get()).assume_init_drop();
            }
            h = h.wrapping_add(1);
        }
    }
}

/// The producing end. `Send` (movable to its group's thread) but not `Sync`.
pub struct Producer<T> {
    inner: Arc<Inner<T>>,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

/// The consuming end. `Send` but not `Sync`.
pub struct Consumer<T> {
    inner: Arc<Inner<T>>,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

/// Create a ring holding up to `capacity` elements (rounded up to a power of two).
pub fn spsc<T>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    let cap = capacity.max(1).next_power_of_two();
    let buf = (0..cap)
        .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let inner = Arc::new(Inner {
        buf,
        cap,
        mask: cap - 1,
        head: CachePadded(AtomicUsize::new(0)),
        tail: CachePadded(AtomicUsize::new(0)),
        closed: AtomicBool::new(false),
        consumer_waiting: AtomicBool::new(false),
        producer_waiting: AtomicBool::new(false),
        gate: Mutex::new(()),
        cvar: Condvar::new(),
    });
    (
        Producer { inner: Arc::clone(&inner), _not_sync: PhantomData },
        Consumer { inner, _not_sync: PhantomData },
    )
}

impl<T> Producer<T> {
    /// Non-blocking push. `Err(val)` if the ring is full.
    pub fn try_push(&self, val: T) -> Result<(), T> {
        let tail = self.inner.tail.0.load(Ordering::Relaxed);
        let head = self.inner.head.0.load(Ordering::Acquire);
        if tail.wrapping_sub(head) == self.inner.cap {
            return Err(val);
        }
        let idx = tail & self.inner.mask;
        // SAFETY: this slot is free (past head); we are the only producer.
        unsafe {
            (*self.inner.buf[idx].get()).write(val);
        }
        self.inner.tail.0.store(tail.wrapping_add(1), Ordering::Release);
        // Dekker handshake: order this publish against a consumer about to park, then
        // wake it iff it flagged itself waiting — no mutex on the busy fast path.
        fence(Ordering::SeqCst);
        if self.inner.consumer_waiting.load(Ordering::Relaxed) {
            self.inner.wake();
        }
        Ok(())
    }

    /// Blocking push. Blocks while full; `Err(val)` once the consumer is gone.
    pub fn push(&self, mut val: T) -> Result<(), T> {
        loop {
            match self.try_push(val) {
                Ok(()) => return Ok(()),
                Err(v) => {
                    val = v;
                    if self.inner.closed.load(Ordering::Acquire) {
                        return Err(val); // consumer gone
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

    /// Whether the consumer has been dropped.
    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }
}

impl<T> Consumer<T> {
    /// Non-blocking pop. `None` if the ring is empty.
    pub fn try_pop(&self) -> Option<T> {
        let head = self.inner.head.0.load(Ordering::Relaxed);
        let tail = self.inner.tail.0.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let idx = head & self.inner.mask;
        // SAFETY: this slot (before tail) is initialized; we are the only consumer.
        let val = unsafe { (*self.inner.buf[idx].get()).assume_init_read() };
        self.inner.head.0.store(head.wrapping_add(1), Ordering::Release);
        // Dekker handshake mirroring try_push: wake a producer parked on "full".
        fence(Ordering::SeqCst);
        if self.inner.producer_waiting.load(Ordering::Relaxed) {
            self.inner.wake();
        }
        Some(val)
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
}
