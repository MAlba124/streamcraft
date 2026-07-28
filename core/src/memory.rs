//! Pool-backed, refcounted memory (spec: Memory; Device memory and sync points).
//!
//! A **safe** recycling pool (a free-list of fixed-size boxed slots) hands out
//! [`Memory`]: a refcounted `(base, offset, len)` **view**. Clones and
//! [`Memory::slice`] sub-views bump a refcount — fan-out and demuxer slicing never
//! copy payload — and the slot recycles to the pool when the last view drops, so
//! steady-state streaming allocates nothing. Mutable access is uniqueness-gated
//! with copy-on-write for shared backings. Still future (spec): pluggable
//! allocators, registration hooks (io_uring/RDMA/GPU), the `ExternalMemory`
//! (dma-buf) variant, and per-link pools.
//!
//! `unsafe` is permitted here (one of the two audited modules, alongside the SPSC ring):
//! the [`Arena`] bump allocator hands out disjoint `&mut [u8]` regions from reusable
//! chunks.
#![allow(unsafe_code)]

use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// A fixed-slot-size buffer pool. Cheap to clone (an `Arc` handle); every clone
/// shares the same free-list and counters.
pub struct Pool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    slot_size: usize,
    max_slots: u32,
    /// Recycled backings, **Arc and all**: reusing the whole `Arc<MemoryInner>`
    /// (not just the byte box) makes a steady-state acquire/drop cycle allocate
    /// *nothing* — the per-buffer `Arc::new` in `from_box` was a third of a movie
    /// remux's 1.2M allocations. Entries hold `pool: Weak`, so this list creates no
    /// refcount cycle with the pool that owns it.
    free: Mutex<Vec<Arc<MemoryInner>>>,
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
    // Pool constructor (once at pipeline setup); the free-list Vec starts empty (no alloc)
    // and is the recycling domain, not a per-frame allocation.
    #[allow(clippy::disallowed_methods)]
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
        let recycled = free.pop();
        let mem = match recycled {
            // Zero-alloc fast path: the whole backing (Arc included) comes back.
            Some(arc) => Memory { inner: Some(arc), offset: 0, len: 0 },
            None => {
                if self.inner.outstanding.load(Ordering::Relaxed) >= self.inner.max_slots as u64 {
                    return None;
                }
                self.inner.slot_allocations.fetch_add(1, Ordering::Relaxed);
                Memory::from_box(
                    vec![0u8; self.inner.slot_size].into_boxed_slice(),
                    &self.inner,
                )
            }
        };
        self.inner.acquires.fetch_add(1, Ordering::Relaxed);
        let now = self.inner.outstanding.fetch_add(1, Ordering::Relaxed) + 1;
        self.inner.high_water.fetch_max(now, Ordering::Relaxed);
        drop(free);
        Some(mem)
    }

    /// Acquire a buffer of at least `n` usable bytes: a pooled slot when `n` fits and
    /// one is free, otherwise an **exactly-`n`** heap allocation. The right-sized
    /// fallback is the point (spec: Memory — pool negotiation is future work): with
    /// one pipeline-wide slot size, a cold/tail path emitting a 15 KB demuxed sample
    /// must not pay a multi-MiB slot per buffer — that turned a movie transcode into
    /// gigabytes. Odd-sized boxes recycle to nothing (`recycle` keeps only slot-sized
    /// ones), and every heap fallback still shows in `PoolStats::slot_allocations`.
    /// Hot paths should prefer `try_acquire` + backpressure; this is for bounded
    /// flushes (an EOS drain) where yielding is not an option.
    pub fn acquire_exact(&self, n: usize) -> Memory {
        // A pooled slot serves an exact request only when the request meaningfully
        // fills it (n ≥ slot_size/4). Handing a whole multi-MiB slot to a tiny
        // buffer starves the pool's real clients — measured as a mid-movie playback
        // freeze: the demuxer's ~12 KB compressed AUs pinned every free 4 MiB slot,
        // the decoder's carried picture could then never allocate its output (and
        // by design it stops consuming input until it emits), and the demux group
        // won the race for every slot the sink freed. Small requests take the
        // exactly-n heap path instead: freed on drop, never competing with frames.
        // (Spec: Memory — per-link pools are the real fix for slot-size mismatch.)
        if n <= self.inner.slot_size && n >= self.inner.slot_size / 4 {
            if let Some(m) = self.try_acquire() {
                return m;
            }
        }
        // The heap fallback is **unlinked** — it must not consume the pool's slot
        // budget. A heap-exact buffer counted in `outstanding` starves every
        // `try_acquire` client: measured as the playback freeze's true root — the
        // demuxer queued hundreds of heap-exact compressed AUs ahead of the
        // realtime-paced decoder, `outstanding` sat far above `max_slots`, and the
        // decoder's carried picture could never allocate its output *even with all
        // slots free* (it stops consuming input until it emits — deadlock). Heap
        // buffers are bounded by ring capacities; only slot-backed memory is the
        // pool's to budget. `slot_allocations` still records the heap alloc.
        self.inner.slot_allocations.fetch_add(1, Ordering::Relaxed);
        Memory::from_heap(vec![0u8; n.max(1)].into_boxed_slice())
    }

    /// Take a slot from the free-list, or heap-allocate one on a miss.
    pub fn acquire(&self) -> Memory {
        let recycled = self.inner.free.lock().unwrap().pop();
        let mem = match recycled {
            Some(arc) => Memory { inner: Some(arc), offset: 0, len: 0 },
            None => {
                self.inner.slot_allocations.fetch_add(1, Ordering::Relaxed);
                Memory::from_box(
                    vec![0u8; self.inner.slot_size].into_boxed_slice(),
                    &self.inner,
                )
            }
        };
        self.inner.acquires.fetch_add(1, Ordering::Relaxed);
        let now = self.inner.outstanding.fetch_add(1, Ordering::Relaxed) + 1;
        self.inner.high_water.fetch_max(now, Ordering::Relaxed);
        mem
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            slot_allocations: self.inner.slot_allocations.load(Ordering::Relaxed),
            acquires: self.inner.acquires.load(Ordering::Relaxed),
            recycles: self.inner.recycles.load(Ordering::Relaxed),
            high_water: self.inner.high_water.load(Ordering::Relaxed),
            outstanding: self.inner.outstanding.load(Ordering::Relaxed),
            max_slots: self.inner.max_slots as u64,
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
    /// Recycle a whole backing, Arc allocation included (the hot path, from
    /// [`Memory`]'s drop when it holds the sole reference). Keeps only right-sized
    /// slots, and never retains more free entries than the cap: a transient spike
    /// from the unbounded `acquire` path must not inflate the free-list permanently.
    /// A rejected backing is **disarmed** (bytes taken, pool link severed) before it
    /// drops, so `MemoryInner::drop`'s fallback cannot double-decrement the counters.
    fn recycle_arc(&self, mut arc: Arc<MemoryInner>) {
        self.recycles.fetch_add(1, Ordering::Relaxed);
        let keep = arc.buf.len() == self.slot_size;
        // Decrement outstanding under the free-list lock so it stays consistent with
        // try_acquire's cap check.
        let mut free = self.free.lock().unwrap();
        self.outstanding.fetch_sub(1, Ordering::Relaxed);
        if keep && (free.len() as u64) < self.max_slots as u64 {
            free.push(arc);
        } else if let Some(inner) = Arc::get_mut(&mut arc) {
            // Wrong size (an `acquire_exact` heap fallback) or list at cap: free the
            // bytes, and disarm so the fallback drop path is a no-op.
            drop(std::mem::take(&mut inner.buf));
            inner.pool = Weak::new();
        }
    }

    /// The cold fallback (from [`MemoryInner::drop`] — a concurrent-last-drop race or
    /// a CoW discard): only the bytes come back; re-wrapping pays one `Arc::new`.
    fn recycle_box(self: &Arc<Self>, buf: Box<[u8]>) {
        self.recycles.fetch_add(1, Ordering::Relaxed);
        let keep = buf.len() == self.slot_size;
        let mut free = self.free.lock().unwrap();
        self.outstanding.fetch_sub(1, Ordering::Relaxed);
        if keep && (free.len() as u64) < self.max_slots as u64 {
            free.push(Arc::new(MemoryInner { buf, pool: Arc::downgrade(self) }));
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
    /// Slot-backed memories currently alive — what `try_acquire`'s cap gates on.
    pub outstanding: u64,
    /// The cap itself, so `outstanding == max_slots` reads as "pool exhausted".
    pub max_slots: u64,
}

/// The shared backing of one or more [`Memory`] views: the slot bytes plus the pool
/// they recycle to (`Weak`, so pooled free-list entries don't cycle-keep the pool
/// alive). The slot returns on last-view drop (spec: Memory — "recycled on
/// last-ref drop"): normally as the whole Arc via `Memory`'s drop; this fallback
/// only fires on a concurrent-last-drop race or a CoW discard.
struct MemoryInner {
    buf: Box<[u8]>,
    pool: Weak<PoolInner>,
}

impl Drop for MemoryInner {
    fn drop(&mut self) {
        if self.buf.is_empty() {
            return; // disarmed by recycle_arc, or a taken husk — nothing to return
        }
        if let Some(pool) = self.pool.upgrade() {
            pool.recycle_box(std::mem::take(&mut self.buf));
        }
    }
}

/// A **refcounted `(base, offset, len)` view** into pooled bytes (spec: Memory —
/// "sub-buffer slicing is free: a demuxer slicing one 1 MB read into 300 packets
/// does 300 refcount bumps and zero copies"; fan-out "bumps a refcount, never
/// copies payload").
///
/// - [`Clone`] and [`slice`](Self::slice) bump a refcount; the backing slot
///   recycles to its pool when the last view drops.
/// - Mutable access ([`as_mut_full`](Self::as_mut_full)) is uniqueness-gated:
///   a sole owner mutates in place; a shared view **copies-on-write** into a fresh
///   backing first (spec: "CoW only when a downstream wants mutable access to a
///   shared buffer") — readers of the other views never observe the mutation.
pub struct Memory {
    /// `None` is the dead husk [`take`](Self::take) leaves behind — reads empty,
    /// recycles nothing.
    inner: Option<Arc<MemoryInner>>,
    /// Start of this view's window within the backing slot.
    offset: usize,
    /// Used bytes within the window — what [`data`](Self::data) exposes.
    len: usize,
}

impl Drop for Memory {
    fn drop(&mut self) {
        let Some(arc) = self.inner.take() else { return };
        // Sole owner: recycle the WHOLE backing — Arc allocation included — so a
        // steady-state acquire/drop cycle allocates nothing (the per-buffer
        // `Arc::new` in `from_box` was ~a third of a movie remux's allocations).
        // At strong_count 1 we hold the only handle, so no new clone can appear:
        // the check is race-free. Two sharers dropping concurrently can both read
        // count 2 — then neither takes this path and the plain Arc drop below runs
        // `MemoryInner::drop`, whose byte-recycle fallback still returns the slot.
        if Arc::strong_count(&arc) == 1 {
            if let Some(pool) = arc.pool.upgrade() {
                pool.recycle_arc(arc);
                return;
            }
        }
        drop(arc);
    }
}

impl Memory {
    fn from_box(buf: Box<[u8]>, pool: &Arc<PoolInner>) -> Memory {
        Memory {
            inner: Some(Arc::new(MemoryInner { buf, pool: Arc::downgrade(pool) })),
            offset: 0,
            len: 0,
        }
    }

    /// A pool-less heap backing: dropping the last view frees it, nothing recycles,
    /// and it consumes no pool budget (see `acquire_exact`'s heap fallback).
    fn from_heap(buf: Box<[u8]>) -> Memory {
        Memory {
            inner: Some(Arc::new(MemoryInner { buf, pool: Weak::new() })),
            offset: 0,
            len: 0,
        }
    }

    /// This view's window capacity in bytes (the backing slot minus the view's
    /// offset). A fresh pool buffer's window is the whole slot.
    pub fn capacity(&self) -> usize {
        self.inner
            .as_deref()
            .map_or(0, |i| i.buf.len().saturating_sub(self.offset))
    }

    /// Bytes currently in use (within the window).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The used bytes.
    pub fn data(&self) -> &[u8] {
        let i = self.inner.as_deref().expect("memory is live");
        &i.buf[self.offset..self.offset + self.len]
    }

    /// The full window capacity for filling from IO; contents are unspecified.
    ///
    /// Uniqueness-gated: if other views share this backing, the bytes are first
    /// copied into a fresh backing from the same pool (CoW) so the sharers are
    /// unaffected. The hot producer path (a fresh, never-shared pool buffer) takes
    /// the in-place branch — no copy, no allocation.
    pub fn as_mut_full(&mut self) -> &mut [u8] {
        let inner = self.inner.as_mut().expect("memory is live");
        if Arc::get_mut(inner).is_none() {
            // Shared: copy-on-write the whole backing (offset preserved). Cold by
            // design — only a mutator downstream of a tee/slice pays it. A dead pool
            // (all handles dropped mid-flight) degrades to an unpooled backing.
            let mut fresh = match inner.pool.upgrade() {
                Some(p) => Pool { inner: p }.acquire_exact(inner.buf.len()),
                None => Memory {
                    inner: Some(Arc::new(MemoryInner {
                        buf: vec![0u8; inner.buf.len()].into_boxed_slice(),
                        pool: Weak::new(),
                    })),
                    offset: 0,
                    len: 0,
                },
            };
            let fresh_inner =
                Arc::get_mut(fresh.inner.as_mut().expect("fresh")).expect("unshared");
            let n = inner.buf.len().min(fresh_inner.buf.len());
            fresh_inner.buf[..n].copy_from_slice(&inner.buf[..n]);
            *inner = fresh.inner.take().expect("fresh backing");
        }
        let offset = self.offset;
        let i = Arc::get_mut(inner).expect("unique after CoW");
        &mut i.buf[offset..]
    }

    /// Set the number of used bytes (clamped to the window capacity).
    pub fn set_len(&mut self, n: usize) {
        self.len = n.min(self.capacity());
    }

    /// A sub-view of this view's **used** bytes: `[offset, offset + len)` relative
    /// to [`data`](Self::data). A refcount bump — zero copies; the backing stays
    /// alive until every view drops. `None` when out of bounds (offsets come from
    /// untrusted container data — never a panic).
    pub fn slice(&self, offset: usize, len: usize) -> Option<Memory> {
        let end = offset.checked_add(len)?;
        if end > self.len {
            return None;
        }
        Some(Memory {
            inner: self.inner.clone(),
            offset: self.offset + offset,
            len,
        })
    }

    /// Move this view out, leaving a dead husk behind (empty, no backing — recycles
    /// nothing on drop). O(1); lets a SoA batch column hand out its front buffer
    /// without shifting the tail (spec: Batching — queued hop budget).
    pub(crate) fn take(&mut self) -> Memory {
        Memory {
            inner: self.inner.take(),
            offset: std::mem::replace(&mut self.offset, 0),
            len: std::mem::replace(&mut self.len, 0),
        }
    }
}

/// Fan-out: another view of the same bytes — a refcount bump, never a payload copy
/// (spec: Memory — what `tee` does per downstream).
impl Clone for Memory {
    fn clone(&self) -> Self {
        Memory {
            inner: self.inner.clone(),
            offset: self.offset,
            len: self.len,
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
    /// `UnsafeCell` so regions can be mutated through the shared `&self` of `alloc`. Each
    /// chunk is a whole number of page-aligned [`Page`]s (see [`Arena::make_chunk`]).
    chunks: UnsafeCell<Vec<Box<[Page]>>>,
    active: Cell<usize>,
    offset: Cell<usize>,
    chunk_size: usize,
    /// The most-recent allocation as `(chunk index, start offset, size)`, for the bump
    /// "free/grow the last allocation" fast path (the standard optimisation `bumpalo` uses):
    /// a [`deallocate`] of exactly this region rolls the bump pointer back to `start`
    /// (reclaiming it), and a [`grow`] of it extends in place (no allocate-and-copy). This is
    /// what makes an arena-backed `Vec::push` loop *reuse* memory instead of leaking every
    /// grown-past buffer until the next [`reset`]. `None` when the last allocation is unknown
    /// (fresh, post-reset, or after a non-last free). A `(chunk, start)` pair — not a raw
    /// pointer — so `Arena` stays pointer-free and the pointer is recomputed from the chunk
    /// on demand.
    ///
    /// [`deallocate`]: std::alloc::Allocator::deallocate
    /// [`grow`]: std::alloc::Allocator::grow
    last: Cell<Option<(usize, usize, usize)>>,
}

/// The page size the arena rounds and aligns its chunks to (4 KiB — the common x86-64/aarch64
/// base page).
const PAGE: usize = 4096;

/// One page of arena backing. `#[repr(align(4096))]` forces every `Box<[Page]>` chunk onto a
/// page boundary, so a chunk allocation is a whole number of pages *starting on* a page — it
/// maps cleanly onto the allocator's large-object/mmap path, and a page-aligned request (SIMD,
/// O_DIRECT) is satisfiable without cross-page waste. The bytes are `UnsafeCell` so
/// [`Arena::alloc`] can carve mutable regions through its shared `&self`.
#[repr(align(4096))]
struct Page(UnsafeCell<[u8; PAGE]>);

impl Arena {
    /// Default scratch chunk size in bytes (16 pages); a call needing more grows a larger,
    /// still page-rounded chunk.
    pub const DEFAULT_CHUNK: usize = 64 * 1024;

    /// A fresh arena whose chunks default to `chunk_size` bytes (min 1). No chunk is
    /// allocated until the first [`alloc`](Arena::alloc).
    // Arena constructor; the chunk-list Vec starts empty — no chunk is allocated until the
    // first `alloc`, so this is not a per-`process()` cost.
    #[allow(clippy::disallowed_methods)]
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: UnsafeCell::new(Vec::new()),
            active: Cell::new(0),
            offset: Cell::new(0),
            chunk_size: chunk_size.max(1),
            last: Cell::new(None),
        }
    }

    /// The pointer of the last allocation, recomputed from its `(chunk, start)` record; `None`
    /// if there is no known last allocation. Used by the [`Allocator`](std::alloc::Allocator)
    /// fast paths to recognise "is this the region I handed out last?".
    fn last_ptr(&self) -> Option<*mut u8> {
        let (chunk, start, _size) = self.last.get()?;
        // SAFETY: shared read of the chunk list; `chunk` indexes a chunk that still exists
        // (chunks are never removed, only appended / reset-retained), and `start` is within it.
        let chunks = unsafe { &*self.chunks.get() };
        Some(unsafe { (chunks[chunk].as_ptr() as *mut u8).add(start) })
    }

    fn make_chunk(cap: usize) -> Box<[Page]> {
        // Round the requested capacity up to whole pages: chunks are page-sized and (via
        // `Page`'s `repr(align)`) page-aligned. One-time — chunks are retained and reused
        // across `reset`, so this is not a per-`process()` cost.
        let npages = cap.div_ceil(PAGE).max(1);
        (0..npages).map(|_| Page(UnsafeCell::new([0u8; PAGE]))).collect::<Vec<_>>().into_boxed_slice()
    }

    /// Carve `size` bytes aligned to `align` (a power of two) and return a fresh mutable
    /// region; its contents are unspecified (reused memory). A pointer bump on the
    /// steady-state path; a chunk allocation only when the current chunks are exhausted.
    // `&self -> &mut [u8]` is the defining shape of a bump arena (cf. `bumpalo::Bump::alloc`):
    // each call carves a *disjoint* region, so handing out `&mut` from a shared borrow is
    // sound. `clippy::mut_from_ref` can't see that invariant, so allow it here (the module's
    // unsafe is audited); the borrow checker still enforces one region borrow at a time.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc(&self, size: usize, align: usize) -> &mut [u8] {
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        loop {
            // Try to carve from an existing chunk. The shared borrow of the chunk list
            // ends with this block, before any growth below. Also capture `start` so the
            // allocation can be recorded as the "last" for the free/grow fast path.
            let carved: Option<(*mut u8, usize)> = {
                let chunks = unsafe { &*self.chunks.get() };
                let mut hit = None;
                while self.active.get() < chunks.len() {
                    let chunk = &chunks[self.active.get()];
                    let base = chunk.as_ptr() as usize;
                    let aligned = (base + self.offset.get()).wrapping_add(align - 1) & !(align - 1);
                    let start = aligned - base;
                    if start + size <= chunk.len() * PAGE {
                        self.offset.set(start + size);
                        // SAFETY: [start, start+size) is inside this chunk and, because the
                        // offset only advances, disjoint from every region already handed
                        // out — so this `&mut` cannot alias a live one. Bytes are
                        // `UnsafeCell`, so mutating through `&self` is sound.
                        hit = Some((unsafe { (chunk.as_ptr() as *mut u8).add(start) }, start));
                        break;
                    }
                    if self.active.get() + 1 < chunks.len() {
                        self.active.set(self.active.get() + 1);
                        self.offset.set(0);
                        continue;
                    }
                    break; // no existing chunk fits → grow
                }
                hit
            };
            if let Some((ptr, start)) = carved {
                self.last.set(Some((self.active.get(), start, size)));
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
        self.last.set(None);
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

// SAFETY: `&Arena` is a valid `Allocator` — every `allocate` carves a *disjoint* region (the
// offset only advances, cf. [`Arena::alloc`]), so distinct live allocations never alias, and
// the returned block stays valid until [`Arena::reset`] (which takes `&mut self`, so the borrow
// checker guarantees no `&Arena`-backed value is still live at reset). `deallocate` is a no-op:
// a bump arena reclaims nothing per-allocation, only wholesale on `reset`. The impl is on
// `&Arena` (a `Copy` handle) so a `Vec<T, &Arena>` can clone its allocator, exactly as
// `bumpalo` implements `Allocator for &Bump`. This is what lets a decoder's per-frame scratch
// live in the pipeline's recycled `ctx.scratch()` arena instead of the heap.
unsafe impl std::alloc::Allocator for &Arena {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let size = layout.size();
        if size == 0 {
            // A zero-sized allocation returns a dangling-but-aligned, non-null pointer.
            let dangling = std::ptr::NonNull::<u8>::new(layout.align() as *mut u8)
                .ok_or(std::alloc::AllocError)?;
            return Ok(std::ptr::NonNull::slice_from_raw_parts(dangling, 0));
        }
        let region = self.alloc(size, layout.align());
        let ptr = std::ptr::NonNull::new(region.as_mut_ptr()).ok_or(std::alloc::AllocError)?;
        Ok(std::ptr::NonNull::slice_from_raw_parts(ptr, size))
    }

    // Free/reclaim only the **last** allocation (the bump fast path): if `ptr` is exactly the
    // region handed out most recently, roll the bump pointer back to reclaim it; any other
    // free is a no-op (a bump arena reclaims interior regions only on `reset`). This lets a
    // `Vec` that shrinks/drops-then-reallocates give the space straight back.
    //
    // SAFETY: `ptr`/`layout` describe a block this arena returned. The only mutation is the
    // `offset`/`last` cells; nothing dereferences `ptr`.
    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, _layout: std::alloc::Layout) {
        if self.last_ptr() == Some(ptr.as_ptr()) {
            if let Some((_chunk, start, _size)) = self.last.get() {
                self.offset.set(start);
            }
            self.last.set(None);
        }
    }

    // Grow the **last** allocation in place when it still fits its chunk — no allocate-and-copy,
    // and no leaked old buffer. This is what makes `Vec::push` into the arena reuse the same
    // region as it grows (the block stays put; only the bump pointer advances). Any other grow,
    // or one that overflows the chunk, falls back to allocate-new + copy.
    //
    // SAFETY: `ptr`/`old_layout` describe this arena's last block; `new_layout.size() >=
    // old_layout.size()` (the trait's precondition). The in-place path keeps the same pointer,
    // so the first `old_layout.size()` bytes are trivially preserved.
    unsafe fn grow(
        &self,
        ptr: std::ptr::NonNull<u8>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let new_size = new_layout.size();
        // In-place extend only when: `ptr` is the last allocation, it lives in the active
        // chunk, the alignment already satisfied still covers `new_layout`, and the grown size
        // fits the chunk.
        if new_layout.align() <= old_layout.align() && self.last_ptr() == Some(ptr.as_ptr()) {
            if let Some((chunk, start, _size)) = self.last.get() {
                if chunk == self.active.get() {
                    // SAFETY: shared read; `chunk` still exists.
                    let chunks = unsafe { &*self.chunks.get() };
                    let chunk_len = chunks[chunk].len() * PAGE;
                    if start + new_size <= chunk_len {
                        self.offset.set(start + new_size);
                        self.last.set(Some((chunk, start, new_size)));
                        return Ok(std::ptr::NonNull::slice_from_raw_parts(ptr, new_size));
                    }
                }
            }
        }
        // Fallback: fresh region + copy the old bytes. (The old region is no longer "last"
        // after the new allocation, so it is simply left until `reset` — same as any interior
        // free.)
        let fresh = self.allocate(new_layout)?;
        // SAFETY: `fresh` has `new_size >= old_layout.size()` bytes; `ptr` is readable for
        // `old_layout.size()` bytes; the regions are distinct (bump never re-hands a live one).
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                fresh.as_ptr() as *mut u8,
                old_layout.size(),
            );
        }
        Ok(fresh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_share_the_backing_until_the_last_drops() {
        // Fan-out semantics (spec: tee bumps a refcount, never copies): the slot
        // returns to the pool exactly once, when the final view is gone.
        let pool = Pool::bounded(64, 1);
        let mut a = pool.acquire();
        a.as_mut_full()[..4].copy_from_slice(b"abcd");
        a.set_len(4);
        let b = a.clone();
        assert_eq!(b.data(), b"abcd");
        assert!(pool.try_acquire().is_none(), "one slot, still outstanding");
        drop(a);
        assert!(pool.try_acquire().is_none(), "a clone still holds the backing");
        drop(b);
        assert_eq!(pool.stats().recycles, 1, "recycled once, not per view");
        assert!(pool.try_acquire().is_some(), "last view recycled the slot");
    }

    #[test]
    fn slice_is_a_zero_copy_window_and_never_panics_on_bad_ranges() {
        let pool = Pool::new(64);
        let mut m = pool.acquire();
        m.as_mut_full()[..8].copy_from_slice(b"01234567");
        m.set_len(8);
        let s = m.slice(2, 4).expect("in bounds");
        assert_eq!(s.data(), b"2345");
        let ss = s.slice(1, 2).expect("slice of slice");
        assert_eq!(ss.data(), b"34");
        // Untrusted offsets: out-of-bounds and overflowing ranges are None, not panics.
        assert!(m.slice(7, 2).is_none());
        assert!(m.slice(9, 0).is_none());
        assert!(m.slice(usize::MAX, 2).is_none());
        // The backing outlives the original view.
        drop(m);
        assert_eq!(ss.data(), b"34");
    }

    #[test]
    fn mutating_a_shared_view_copies_on_write() {
        let pool = Pool::new(64);
        let mut a = pool.acquire();
        a.as_mut_full()[..4].copy_from_slice(b"aaaa");
        a.set_len(4);
        let b = a.clone();
        // Mutation through `a` must not be visible through `b` (spec: CoW on
        // shared mutable access).
        a.as_mut_full()[..4].copy_from_slice(b"zzzz");
        assert_eq!(a.data(), b"zzzz");
        assert_eq!(b.data(), b"aaaa", "sharer unaffected by the CoW mutation");
        // A unique view mutates in place: no further allocation happens.
        let before = pool.stats().slot_allocations;
        a.as_mut_full()[0] = b'q';
        assert_eq!(pool.stats().slot_allocations, before, "unique path is in-place");
    }

    #[test]
    fn taken_husk_reads_empty_and_recycles_nothing() {
        let pool = Pool::bounded(64, 1);
        let mut m = pool.acquire();
        m.set_len(3);
        let moved = m.take();
        assert_eq!(m.len(), 0);
        assert_eq!(m.capacity(), 0);
        drop(m); // husk: must not recycle
        assert!(pool.try_acquire().is_none(), "moved view still owns the slot");
        drop(moved);
        assert!(pool.try_acquire().is_some());
    }

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
    fn arena_vec_grows_last_allocation_in_place() {
        // A `Vec<T, &Arena>` built by repeated `push` must grow *in place* (the last-allocation
        // fast path), so the arena consumes ~the final capacity — not the sum of every doubling
        // step (which a naive bump would leak). And its data must survive each in-place grow.
        let arena = Arena::new(64 * 1024);
        {
            let mut v: Vec<u64, &Arena> = Vec::new_in(&arena);
            for i in 0..500u64 {
                v.push(i);
            }
            assert_eq!(v.len(), 500);
            assert_eq!(v[0], 0);
            assert_eq!(v[499], 499);
            assert_eq!(v.iter().copied().sum::<u64>(), (0..500u64).sum::<u64>());

            // 500 u64 → final capacity 512 (4096 B). In-place growth leaves the arena at ~that;
            // a leaking bump would sit near 8+16+…+512 elements of extra waste (≈ 12 KB).
            let used = arena.offset.get();
            assert!(used <= 512 * 8 + 64, "grow was not in-place — arena used {used} bytes");
        }
        // Dropping the Vec deallocates the last (and only) allocation → the bump pointer rolls
        // all the way back.
        assert_eq!(arena.offset.get(), 0, "last allocation was not reclaimed on drop");
        assert_eq!(arena.chunk_count(), 1, "in-place growth spilled to extra chunks");
    }

    #[test]
    fn arena_reclaims_only_the_last_allocation() {
        // Freeing the last allocation rewinds the bump; freeing an interior one is a no-op
        // (reclaimed only by `reset`) — exactly a bump arena's contract.
        use std::alloc::{Allocator, Layout};
        let arena = Arena::new(64 * 1024);
        let arena_ref = &arena;
        let l = Layout::from_size_align(64, 8).unwrap();
        let a = arena_ref.allocate(l).unwrap();
        let after_a = arena.offset.get();
        let b = arena_ref.allocate(l).unwrap();
        let after_b = arena.offset.get();
        assert!(after_b > after_a);
        // Freeing `a` (interior — `b` came after) does nothing.
        unsafe { arena_ref.deallocate(a.cast(), l) };
        assert_eq!(arena.offset.get(), after_b, "interior free must not rewind");
        // Freeing `b` (the last) rewinds to before it.
        unsafe { arena_ref.deallocate(b.cast(), l) };
        assert_eq!(arena.offset.get(), after_a, "last free must rewind the bump pointer");
    }

    #[test]
    fn arena_grows_across_chunks_keeping_regions_valid() {
        // Chunks are page-rounded, so force growth with half-page regions (two per chunk).
        let arena = Arena::new(PAGE);
        let mut regions: Vec<&mut [u8]> = Vec::new();
        for i in 0..10u8 {
            let r = arena.alloc_bytes(PAGE / 2);
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
        let mut arena = Arena::new(PAGE);
        for _ in 0..8 {
            let _ = arena.alloc_bytes(PAGE / 2);
        }
        let grown = arena.chunk_count();
        assert!(grown >= 2);
        arena.reset();
        // The same total again fits in the retained chunks — no further allocation.
        for _ in 0..8 {
            let _ = arena.alloc_bytes(PAGE / 2);
        }
        assert_eq!(arena.chunk_count(), grown, "retained chunks are reused after reset");
    }
}
