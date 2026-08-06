//! Profluens patch (PROFLUENS-PATCHES.md): a per-thread **bump arena** for the decoder's
//! transient allocations, so a steady decode makes zero system-heap allocations.
//!
//! [`DecodeBump`] is a zero-sized [`Allocator`] backed by a thread-local bump region. Transient
//! decode buffers are typed `Vec<T, DecodeBump>` and draw from it; [`reset`] rewinds the region
//! (freeing them all) and is called once per packet by `decode_into_buffers`. Persistent decoder
//! state stays `Vec<T>` (the global allocator) — and because the two are *different types*, the
//! compiler forbids storing a transient (`DecodeBump`) buffer into a persistent (`Global`) field,
//! so the per-packet reset can never free live state (the failure mode that sank the whole-process
//! arena). `DecodeBump` carries no lifetime, so converting a site is a local type change
//! (`Vec::new()` → `Vec::new_in(DecodeBump)`), not viral `&'a Bump` plumbing.

#![allow(clippy::disallowed_methods)] // this IS the arena; it defines it, not uses one

use core::alloc::{AllocError, Allocator, Layout};
use core::cell::Cell;
use core::ptr::NonNull;

/// Per-thread region size. Comfortably larger than a single Opus packet's transient footprint
/// (120 ms max); mapped once per thread and reused across packets (leaked until thread exit).
const REGION: usize = 16 * 1024 * 1024;

struct Region {
    base: Cell<*mut u8>,
    cursor: Cell<usize>,
    high: Cell<usize>,
}

thread_local! {
    // const-init (all `Cell`s of primitives) → first access doesn't allocate.
    static REGIONS: Region = const {
        Region { base: Cell::new(core::ptr::null_mut()), cursor: Cell::new(0), high: Cell::new(0) }
    };
}

/// Rewind this thread's bump region, freeing every `DecodeBump` allocation made since the last
/// reset. Call once per packet, after the previous packet's transients are all dropped.
pub fn reset() {
    REGIONS.with(|r| r.cursor.set(0));
}

/// Peak region bytes used on this thread — for sizing [`REGION`].
pub fn high_water() -> usize {
    REGIONS.with(|r| r.high.get())
}

/// A zero-sized [`Allocator`] over the thread-local bump region. `Vec<T, DecodeBump>` is a
/// transient buffer freed by the next [`reset`].
#[derive(Clone, Copy, Default, Debug)]
pub struct DecodeBump;

// SAFETY: the backing region is thread-local; `DecodeBump` is only ever used for buffers created
// and dropped on the same decode thread (never sent across threads). Marking it Send+Sync lets
// `Vec<T, DecodeBump>` be Send/Sync like a normal Vec, which the crate's decode signatures need.
unsafe impl Send for DecodeBump {}
unsafe impl Sync for DecodeBump {}

// SAFETY: `allocate` returns a fresh, correctly-aligned, non-overlapping block within the region
// (or errors); `deallocate` is a no-op (the region is bulk-rewound by `reset`).
unsafe impl Allocator for DecodeBump {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        REGIONS.with(|r| {
            let mut base = r.base.get();
            if base.is_null() {
                // Map the region once via the system global allocator (no recursion — this is a
                // plain heap block, not a DecodeBump allocation).
                let l = Layout::from_size_align(REGION, 64).map_err(|_| AllocError)?;
                base = unsafe { std::alloc::alloc(l) };
                if base.is_null() {
                    return Err(AllocError);
                }
                r.base.set(base);
            }
            let align = layout.align().max(1);
            let start = ((base as usize + r.cursor.get()) + (align - 1)) & !(align - 1);
            let end = (start - base as usize) + layout.size().max(1);
            if end > REGION {
                return Err(AllocError); // one packet exceeded REGION — grow the constant
            }
            r.cursor.set(end);
            if end > r.high.get() {
                r.high.set(end);
            }
            let ptr = NonNull::new(start as *mut u8).ok_or(AllocError)?;
            Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
        })
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Bump arena: individual frees are no-ops; `reset` bulk-frees the whole region.
    }
}
