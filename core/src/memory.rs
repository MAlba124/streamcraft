//! Pool-backed, refcounted memory (spec: Memory; Device memory and sync points).
//!
//! Milestone 1 ships a **safe** recycling pool: a free-list of fixed-size boxed
//! slots, handed out as [`Memory`] and returned on drop. Steady-state streaming
//! reuses slots, so allocation count stays flat regardless of how long it runs
//! (the zero-steady-state-allocation criterion). The unsafe arena / registration /
//! sub-slicing version (spec) replaces the internals behind this same API later.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// A fixed-slot-size buffer pool. Cheap to clone (an `Arc` handle); every clone
/// shares the same free-list and counters.
pub struct Pool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    slot_size: usize,
    max_slots: u32,
    free: Mutex<Vec<Box<[u8]>>>,
    slot_allocations: AtomicU64, // boxes ever heap-allocated (misses)
    acquires: AtomicU64,
    recycles: AtomicU64,
    outstanding: AtomicU64,
    high_water: AtomicU64,
}

impl Pool {
    /// An unbounded pool (grows on demand). Use [`Pool::bounded`] for backpressure.
    pub fn new(slot_size: usize) -> Self {
        Self::bounded(slot_size, u32::MAX)
    }

    /// A pool capped at `max_slots` concurrently-outstanding buffers. Once the cap
    /// is hit, [`Pool::try_acquire`] returns `None` — the backpressure signal.
    pub fn bounded(slot_size: usize, max_slots: u32) -> Self {
        Self {
            inner: Arc::new(PoolInner {
                slot_size,
                max_slots,
                free: Mutex::new(Vec::new()),
                slot_allocations: AtomicU64::new(0),
                acquires: AtomicU64::new(0),
                recycles: AtomicU64::new(0),
                outstanding: AtomicU64::new(0),
                high_water: AtomicU64::new(0),
            }),
        }
    }

    pub fn slot_size(&self) -> usize {
        self.inner.slot_size
    }

    /// Free slots available before hitting the cap (`u32::MAX` for unbounded pools).
    pub fn free_slots(&self) -> u32 {
        let outstanding = self.inner.outstanding.load(Ordering::Relaxed);
        (self.inner.max_slots as u64).saturating_sub(outstanding) as u32
    }

    /// Like [`acquire`](Self::acquire) but respects the cap: `None` when full. The
    /// cap check and the `outstanding` bump happen under the free-list lock, so
    /// concurrent acquirers can never collectively exceed `max_slots`.
    pub fn try_acquire(&self) -> Option<Memory> {
        let mut free = self.inner.free.lock().unwrap();
        let buf = match free.pop() {
            Some(b) => b,
            None => {
                if self.inner.outstanding.load(Ordering::Relaxed) >= self.inner.max_slots as u64 {
                    return None;
                }
                self.inner.slot_allocations.fetch_add(1, Ordering::Relaxed);
                vec![0u8; self.inner.slot_size].into_boxed_slice()
            }
        };
        self.inner.acquires.fetch_add(1, Ordering::Relaxed);
        let now = self.inner.outstanding.fetch_add(1, Ordering::Relaxed) + 1;
        self.inner.high_water.fetch_max(now, Ordering::Relaxed);
        drop(free);
        Some(Memory {
            buf: Some(buf),
            len: 0,
            pool: Arc::clone(&self.inner),
        })
    }

    /// Take a slot from the free-list, or heap-allocate one on a miss.
    pub fn acquire(&self) -> Memory {
        let recycled = self.inner.free.lock().unwrap().pop();
        let buf = match recycled {
            Some(b) => b,
            None => {
                self.inner.slot_allocations.fetch_add(1, Ordering::Relaxed);
                vec![0u8; self.inner.slot_size].into_boxed_slice()
            }
        };
        self.inner.acquires.fetch_add(1, Ordering::Relaxed);
        let now = self.inner.outstanding.fetch_add(1, Ordering::Relaxed) + 1;
        self.inner.high_water.fetch_max(now, Ordering::Relaxed);
        Memory {
            buf: Some(buf),
            len: 0,
            pool: Arc::clone(&self.inner),
        }
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            slot_allocations: self.inner.slot_allocations.load(Ordering::Relaxed),
            acquires: self.inner.acquires.load(Ordering::Relaxed),
            recycles: self.inner.recycles.load(Ordering::Relaxed),
            high_water: self.inner.high_water.load(Ordering::Relaxed),
        }
    }
}

impl Clone for Pool {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl PoolInner {
    fn recycle(&self, buf: Box<[u8]>) {
        self.recycles.fetch_add(1, Ordering::Relaxed);
        let keep = buf.len() == self.slot_size;
        // Decrement outstanding under the free-list lock so it stays consistent with
        // try_acquire's cap check.
        let mut free = self.free.lock().unwrap();
        self.outstanding.fetch_sub(1, Ordering::Relaxed);
        if keep {
            free.push(buf);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Boxes ever heap-allocated. Flat in steady state == the zero-alloc goal.
    pub slot_allocations: u64,
    pub acquires: u64,
    pub recycles: u64,
    pub high_water: u64,
}

/// A pooled byte buffer, returned to its pool on drop. Milestone 1 is a movable
/// owned handle; refcounted fan-out / sub-slicing (spec) come with `tee`.
pub struct Memory {
    buf: Option<Box<[u8]>>,
    len: usize,
    pool: Arc<PoolInner>,
}

impl Memory {
    /// Total slot capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.buf.as_deref().map_or(0, <[u8]>::len)
    }

    /// Bytes currently in use.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The used bytes.
    pub fn data(&self) -> &[u8] {
        let b = self.buf.as_deref().expect("memory is live");
        &b[..self.len]
    }

    /// The full capacity for filling from IO; contents are unspecified.
    pub fn as_mut_full(&mut self) -> &mut [u8] {
        self.buf.as_deref_mut().expect("memory is live")
    }

    /// Set the number of used bytes (clamped to capacity).
    pub fn set_len(&mut self, n: usize) {
        self.len = n.min(self.capacity());
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.pool.recycle(buf);
        }
    }
}

// --- Stubs for the fuller design (spec), not yet wired ---

/// Plain fn-pointer allocator vtable — no generics (spec: Memory). TODO(step 1+).
pub struct Allocator {
    _priv: (),
}

pub struct PoolConfig {
    pub slot_size: usize,
    pub slots: u32,
    /// ≥ 64; page/hugepage for O_DIRECT.
    pub align: usize,
}

/// Optional per-buffer fence for device memory (spec: Device memory and sync
/// points). `None` on system memory — zero cost on the common path.
pub struct SyncPoint {
    _priv: (),
}

/// Per-`process()` bump allocator (spec: Memory — scratch arenas). TODO.
pub struct Arena {
    _priv: (),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recycles_slots_flat_allocation() {
        let pool = Pool::new(1024);
        for _ in 0..100 {
            let mut m = pool.acquire();
            assert_eq!(m.capacity(), 1024);
            m.set_len(10);
            drop(m); // back to the pool
        }
        let s = pool.stats();
        assert_eq!(s.slot_allocations, 1, "one box, recycled 99 times");
        assert_eq!(s.high_water, 1);
        assert_eq!(s.acquires, 100);
        assert_eq!(s.recycles, 100);
    }

    #[test]
    fn concurrent_outstanding_allocates_more() {
        let pool = Pool::new(8);
        let a = pool.acquire();
        let b = pool.acquire(); // both alive → 2 distinct slots
        assert_eq!(pool.stats().slot_allocations, 2);
        assert_eq!(pool.stats().high_water, 2);
        drop(a);
        drop(b);
        // A third acquire now reuses, no new allocation.
        let _c = pool.acquire();
        assert_eq!(pool.stats().slot_allocations, 2);
    }

    #[test]
    fn data_roundtrip() {
        let pool = Pool::new(16);
        let mut m = pool.acquire();
        m.as_mut_full()[..3].copy_from_slice(&[1, 2, 3]);
        m.set_len(3);
        assert_eq!(m.data(), &[1, 2, 3]);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn set_len_clamps_to_capacity() {
        let pool = Pool::new(4);
        let mut m = pool.acquire();
        m.set_len(999);
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn bounded_pool_backpressure() {
        let pool = Pool::bounded(8, 2);
        assert_eq!(pool.free_slots(), 2);
        let a = pool.try_acquire().expect("slot 1");
        let b = pool.try_acquire().expect("slot 2");
        assert_eq!(pool.free_slots(), 0);
        assert!(pool.try_acquire().is_none(), "cap reached → backpressure");
        drop(a);
        assert_eq!(pool.free_slots(), 1);
        let _c = pool.try_acquire().expect("slot freed after drop");
        assert!(pool.try_acquire().is_none());
        drop(b);
    }
}
