//! Pool-backed, refcounted memory (spec: Memory; Device memory and sync points).
//!
//! Milestone 1 ships a **safe** recycling pool: a free-list of fixed-size boxed
//! slots, handed out as [`Memory`] and returned on drop. Steady-state streaming
//! reuses slots, so allocation count stays flat regardless of how long it runs
//! (the zero-steady-state-allocation criterion). The unsafe arena / registration /
//! sub-slicing version (spec) replaces the internals behind this same API later.
//!
//! `unsafe` is permitted here (one of the two audited modules, alongside the SPSC ring):
//! the [`Arena`] bump allocator hands out disjoint `&mut [u8]` regions from reusable
//! chunks.
#![allow(unsafe_code)]

use std::cell::{Cell, UnsafeCell};
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
        // Keep only right-sized slots, and never retain more free slots than the cap: a
        // transient spike from the unbounded `acquire` path must not inflate the free-list
        // permanently (it would read as a steady-state leak). Excess boxes are dropped
        // (freed) here. `bounded` pools cap at `max_slots`; unbounded pools (`max_slots ==
        // u32::MAX`) keep everything, as before.
        let keep = buf.len() == self.slot_size;
        // Decrement outstanding under the free-list lock so it stays consistent with
        // try_acquire's cap check.
        let mut free = self.free.lock().unwrap();
        self.outstanding.fetch_sub(1, Ordering::Relaxed);
        if keep && (free.len() as u64) < self.max_slots as u64 {
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

/// A per-`process()` bump allocator for scratch space (spec: Memory — scratch arenas):
/// carve temporary byte regions off a reusable backing store during one `process()` call,
/// then [`reset`](Arena::reset) it (the scheduler does this between calls) so the same
/// memory is reused next time — no per-`process()` heap traffic in steady state.
///
/// It grows by adding chunks when a call needs more than the current one holds; the chunks
/// are kept across resets and reused, so allocation count goes flat (the
/// zero-steady-state-allocation goal, like [`Pool`]). Chunks are boxed, so a region handed
/// out stays valid even when a later allocation grows the chunk list — the box's contents
/// never move.
///
/// [`alloc`](Arena::alloc) takes `&self` and returns disjoint `&mut [u8]`s, so several
/// scratch regions coexist within one call; [`reset`](Arena::reset) takes `&mut self`, so
/// the borrow checker guarantees no region is still live when the arena is reset.
pub struct Arena {
    /// All chunks; `active` indexes the one currently being filled. Bytes live in
    /// `UnsafeCell` so regions can be mutated through the shared `&self` of `alloc`.
    chunks: UnsafeCell<Vec<Box<[UnsafeCell<u8>]>>>,
    active: Cell<usize>,
    offset: Cell<usize>,
    chunk_size: usize,
}

impl Arena {
    /// Default scratch chunk size in bytes; a call needing more grows a larger chunk.
    pub const DEFAULT_CHUNK: usize = 64 * 1024;

    /// A fresh arena whose chunks default to `chunk_size` bytes (min 1). No chunk is
    /// allocated until the first [`alloc`](Arena::alloc).
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: UnsafeCell::new(Vec::new()),
            active: Cell::new(0),
            offset: Cell::new(0),
            chunk_size: chunk_size.max(1),
        }
    }

    fn make_chunk(cap: usize) -> Box<[UnsafeCell<u8>]> {
        (0..cap).map(|_| UnsafeCell::new(0u8)).collect::<Vec<_>>().into_boxed_slice()
    }

    /// Carve `size` bytes aligned to `align` (a power of two) and return a fresh mutable
    /// region; its contents are unspecified (reused memory). A pointer bump on the
    /// steady-state path; a chunk allocation only when the current chunks are exhausted.
    pub fn alloc(&self, size: usize, align: usize) -> &mut [u8] {
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        loop {
            // Try to carve from an existing chunk. The shared borrow of the chunk list
            // ends with this block, before any growth below.
            let carved: Option<*mut u8> = {
                let chunks = unsafe { &*self.chunks.get() };
                let mut ptr = None;
                while self.active.get() < chunks.len() {
                    let chunk = &chunks[self.active.get()];
                    let base = chunk.as_ptr() as usize;
                    let aligned = (base + self.offset.get()).wrapping_add(align - 1) & !(align - 1);
                    let start = aligned - base;
                    if start + size <= chunk.len() {
                        self.offset.set(start + size);
                        // SAFETY: [start, start+size) is inside this chunk and, because the
                        // offset only advances, disjoint from every region already handed
                        // out — so this `&mut` cannot alias a live one. Bytes are
                        // `UnsafeCell`, so mutating through `&self` is sound.
                        ptr = Some(unsafe { (chunk.as_ptr() as *mut u8).add(start) });
                        break;
                    }
                    if self.active.get() + 1 < chunks.len() {
                        self.active.set(self.active.get() + 1);
                        self.offset.set(0);
                        continue;
                    }
                    break; // no existing chunk fits → grow
                }
                ptr
            };
            if let Some(ptr) = carved {
                // SAFETY: `ptr` is valid and writable for `size` bytes (checked above) and
                // uniquely owned for the returned lifetime.
                return unsafe { std::slice::from_raw_parts_mut(ptr, size) };
            }
            // Grow. No borrow of `chunks` is alive here, and existing regions point into
            // box contents the push does not move, so this cannot invalidate them.
            let cap = self.chunk_size.max(size + align);
            // SAFETY: exclusive access — no reference into `chunks` outlives this block.
            let len = unsafe {
                let v = &mut *self.chunks.get();
                v.push(Self::make_chunk(cap));
                v.len()
            };
            self.active.set(len - 1);
            self.offset.set(0);
        }
    }

    /// Convenience for byte scratch (`align = 1`).
    pub fn alloc_bytes(&self, n: usize) -> &mut [u8] {
        self.alloc(n, 1)
    }

    /// Reset to empty, retaining the chunks for reuse (the scheduler calls this between
    /// `process()` calls). `&mut self`, so no handed-out region can still be live.
    pub fn reset(&mut self) {
        self.active.set(0);
        self.offset.set(0);
    }

    /// Number of chunks allocated. Flat across resets in steady state == the reuse goal.
    pub fn chunk_count(&self) -> usize {
        // SAFETY: shared read; the arena is owned by a single thread.
        unsafe { (*self.chunks.get()).len() }
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new(Self::DEFAULT_CHUNK)
    }
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

    #[test]
    fn bounded_pool_free_list_does_not_retain_beyond_cap() {
        let pool = Pool::bounded(8, 2);
        // Allocate 5 at once via the *unbounded* `acquire` (bypasses the cap) — a transient
        // spike, as an unbounded producer would cause.
        let held: Vec<_> = (0..5).map(|_| pool.acquire()).collect();
        assert_eq!(pool.stats().slot_allocations, 5, "5 distinct boxes while all outstanding");
        drop(held); // recycle all 5 → free-list keeps at most max_slots (2), drops 3
        // Two acquires reuse the retained slots (no new allocation)…
        let a = pool.acquire();
        let b = pool.acquire();
        assert_eq!(pool.stats().slot_allocations, 5, "reused the 2 retained free slots");
        // …a third finds the free-list empty (the excess was freed) → one fresh allocation.
        let c = pool.acquire();
        assert_eq!(pool.stats().slot_allocations, 6, "free-list capped — the spike was not retained");
        drop((a, b, c));
    }

    // --- scratch arena ---

    #[test]
    fn arena_allocs_are_disjoint_and_writable() {
        let arena = Arena::new(1024);
        // Two regions from one call coexist (both borrow &arena) and don't overlap.
        let a = arena.alloc_bytes(16);
        let b = arena.alloc_bytes(16);
        a.fill(0xAA);
        b.fill(0xBB);
        assert!(a.iter().all(|&x| x == 0xAA));
        assert!(b.iter().all(|&x| x == 0xBB), "second region untouched by the first");
        assert_eq!((a.len(), b.len()), (16, 16));
    }

    #[test]
    fn arena_respects_alignment() {
        let arena = Arena::new(4096);
        let _pad = arena.alloc_bytes(1); // misalign the bump offset
        let s = arena.alloc(64, 64);
        assert_eq!(s.as_ptr() as usize % 64, 0, "carved region is 64-aligned");
        assert_eq!(s.len(), 64);
    }

    #[test]
    fn arena_reset_reuses_memory() {
        let mut arena = Arena::new(1024);
        let addr = {
            let a = arena.alloc_bytes(100);
            a.as_ptr() as usize
        }; // region dropped, so reset() (needs &mut) is allowed
        let chunks = arena.chunk_count();
        arena.reset();
        let addr2 = arena.alloc_bytes(100).as_ptr() as usize;
        assert_eq!(addr, addr2, "reset rewinds to the same memory");
        assert_eq!(arena.chunk_count(), chunks, "reuse allocates no new chunk");
    }

    #[test]
    fn arena_grows_across_chunks_keeping_regions_valid() {
        let arena = Arena::new(64); // small chunks: several allocs force growth
        let mut regions: Vec<&mut [u8]> = Vec::new();
        for i in 0..10u8 {
            let r = arena.alloc_bytes(32);
            r.fill(i);
            regions.push(r);
        }
        assert!(arena.chunk_count() >= 2, "grew beyond a single chunk");
        // Every earlier region survived the later chunk allocations intact.
        for (i, r) in regions.iter().enumerate() {
            assert!(r.iter().all(|&x| x == i as u8), "region {i} intact after growth");
        }
    }

    #[test]
    fn arena_reset_after_growth_reuses_all_chunks() {
        let mut arena = Arena::new(64);
        for _ in 0..8 {
            let _ = arena.alloc_bytes(32);
        }
        let grown = arena.chunk_count();
        assert!(grown >= 2);
        arena.reset();
        // The same total again fits in the retained chunks — no further allocation.
        for _ in 0..8 {
            let _ = arena.alloc_bytes(32);
        }
        assert_eq!(arena.chunk_count(), grown, "retained chunks are reused after reset");
    }
}
