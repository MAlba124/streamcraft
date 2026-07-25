//! Per-frame bump arena for the scope UI.
//!
//! The immediate-mode UI produces a fresh pile of transient geometry every frame
//! — vertices, indices, glyph quads, temporary format strings. Rather than churn
//! the global allocator each frame (which is exactly the allocation-per-frame the
//! SC ethos rejects), all of it lands in one reusable byte buffer that is *reset*,
//! not freed, at the top of every frame. Steady state does zero heap traffic once
//! the buffer has grown to the frame's high-water mark.
//!
//! This is deliberately independent of core's arena (scope does not reach into
//! core internals — spec: "the UI must be the same philosophy as the rest of SC").
//! It is a plain bump pointer with alignment, no free list, no per-object drop.
//! Only `Copy` POD is stored here, so leaking destructors is a non-issue.

use std::cell::Cell;

/// A reusable bump allocator. Grows to the frame high-water mark, then reuses the
/// same allocation forever; [`reset`](Arena::reset) rewinds the cursor to zero.
pub struct Arena {
    buf: Vec<u8>,
    /// Bump cursor. `Cell` so `alloc*` can take `&self` — the UI hands out arena
    /// slices from many call sites within one frame without threading `&mut`.
    head: Cell<usize>,
    /// High-water mark across all frames since construction (diagnostics only).
    peak: Cell<usize>,
}

impl Arena {
    /// A new arena pre-sized to `capacity` bytes (it still grows on demand).
    pub fn with_capacity(capacity: usize) -> Self {
        Self { buf: vec![0u8; capacity.max(1)], head: Cell::new(0), peak: Cell::new(0) }
    }

    /// Rewind to empty for the next frame. Keeps the backing allocation.
    pub fn reset(&mut self) {
        self.head.set(0);
    }

    /// Bytes handed out since the last [`reset`](Arena::reset).
    pub fn used(&self) -> usize {
        self.head.get()
    }

    /// The largest single-frame footprint observed (for tuning `with_capacity`).
    pub fn peak(&self) -> usize {
        self.peak.get()
    }

    /// Total backing bytes.
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Grow the backing buffer so at least `additional` bytes are free past the
    /// current head. Only ever called when a frame exceeds the current capacity;
    /// after warm-up this never fires. `&mut` because it may reallocate.
    fn grow_to_fit(&mut self, need_end: usize) {
        if need_end <= self.buf.len() {
            return;
        }
        // Geometric growth so warm-up is O(log) reallocations, then never again.
        let mut new_cap = self.buf.len().max(1);
        while new_cap < need_end {
            new_cap *= 2;
        }
        self.buf.resize(new_cap, 0);
    }

    /// Reserve `count` default-initialised `T`s and return them as a mutable slice
    /// that lives until the next [`reset`](Arena::reset). `T` must be `Copy` POD.
    ///
    /// Takes `&mut self` because it may reallocate; the returned slice borrows the
    /// arena for its lifetime, so the borrow checker prevents a second alloc from
    /// invalidating it (exactly the guarantee a bump arena needs).
    pub fn alloc_slice<T: Copy + Default>(&mut self, count: usize) -> &mut [T] {
        let align = std::mem::align_of::<T>();
        let size = std::mem::size_of::<T>();
        // Align the head up to T's alignment (bit trick valid for power-of-two
        // alignments, which all Rust type alignments are).
        let start = (self.head.get() + align - 1) & !(align - 1);
        let bytes = size * count;
        let end = start + bytes;
        self.grow_to_fit(end);
        self.head.set(end);
        if end > self.peak.get() {
            self.peak.set(end);
        }
        // SAFETY: `start` is aligned to `T` and `[start, end)` is inside `buf`
        // (grown to fit above). `T: Copy + Default` is POD, and we initialise every
        // element below, so no uninitialised value is ever read.
        let ptr = unsafe { self.buf.as_mut_ptr().add(start) as *mut T };
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, count) };
        for e in slice.iter_mut() {
            *e = T::default();
        }
        slice
    }

    /// Allocate the four geometry streams a draw-list build needs in **one** arena
    /// bump, returning them together.
    ///
    /// Doing it in a single call sidesteps the borrow checker's objection to
    /// handing out several independent `&'a mut` slices from sequential `alloc`
    /// calls (each would extend the arena borrow to `'a`, conflicting with the
    /// next). Internally we reserve one contiguous region and carve typed
    /// sub-slices from it — all derived from one bump, so their non-overlapping
    /// ranges are provably disjoint.
    ///
    /// `verts` vertices produce `verts*2` xy floats, `verts*4` colour floats,
    /// `verts*2` uv floats; `indices` u16 indices. Everything is zero-initialised.
    pub fn alloc_geometry(&mut self, verts: usize, indices: usize) -> Geometry<'_> {
        let xy_n = verts * 2;
        let col_n = verts * 4;
        let uv_n = verts * 2;
        // Lay out the f32 streams first (4-byte aligned), then the u16 indices
        // (2-byte aligned — 4-byte alignment satisfies it). One region, one bump.
        let f32_count = xy_n + col_n + uv_n;
        // Align head to f32 for the whole region.
        let align = std::mem::align_of::<f32>();
        let start = (self.head.get() + align - 1) & !(align - 1);
        let f32_bytes = f32_count * std::mem::size_of::<f32>();
        // u16 region begins right after the f32 region (already 4-byte aligned end).
        let u16_off = start + f32_bytes;
        let u16_bytes = indices * std::mem::size_of::<u16>();
        let end = u16_off + u16_bytes;
        self.grow_to_fit(end);
        self.head.set(end);
        if end > self.peak.get() {
            self.peak.set(end);
        }
        let base = self.buf.as_mut_ptr();
        // SAFETY: `[start, end)` lies inside `buf` (grown to fit). `start` is
        // 4-aligned (f32); `u16_off` is 4-aligned too (f32 region is a whole number
        // of 4-byte elements), which satisfies u16's 2-byte alignment. The four
        // sub-slices carve non-overlapping ranges of one region, so the mutable
        // aliasing rules hold. Both types are POD; we zero every element below.
        unsafe {
            let f32_ptr = base.add(start) as *mut f32;
            let xy = std::slice::from_raw_parts_mut(f32_ptr, xy_n);
            let col = std::slice::from_raw_parts_mut(f32_ptr.add(xy_n), col_n);
            let uv = std::slice::from_raw_parts_mut(f32_ptr.add(xy_n + col_n), uv_n);
            let idx = std::slice::from_raw_parts_mut(base.add(u16_off) as *mut u16, indices);
            for e in xy.iter_mut() {
                *e = 0.0;
            }
            for e in col.iter_mut() {
                *e = 0.0;
            }
            for e in uv.iter_mut() {
                *e = 0.0;
            }
            for e in idx.iter_mut() {
                *e = 0;
            }
            Geometry { xy, col, uv, idx }
        }
    }
}

/// The four vertex/index streams for one draw-list build, all carved from one
/// arena bump (see [`Arena::alloc_geometry`]).
pub struct Geometry<'a> {
    /// Interleaved `[x, y]` pairs (2 per vertex).
    pub xy: &'a mut [f32],
    /// `[r, g, b, a]` per vertex (4 per vertex).
    pub col: &'a mut [f32],
    /// `[u, v]` pairs (2 per vertex).
    pub uv: &'a mut [f32],
    /// Triangle indices (u16).
    pub idx: &'a mut [u16],
}
