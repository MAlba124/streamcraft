# streamcraft patches on top of oxideav-h264 0.1.7 (crates.io)

Vendored via `[patch.crates-io]` in the workspace root. Minimal diffs, intended
to be offered upstream.

## 1. Reference pictures are never freed (unbounded memory growth)

`RefPicStore` (src/ref_store.rs) has `insert()` and no removal of any kind:
every finalized **reference** picture's `Picture` (full planes + MV grid,
~1.5 MB at 720p) stays resident for the life of the decoder. §8.2.5 reference
marking correctly evicts `DpbEntry`s (`dpb_entries.retain(...)` in
`finalize_in_progress_picture`), but the corresponding storage was never
reclaimed. Playing a movie leaks one reference picture per reference frame —
measured ~1.7 GB in 35 s of 720p25 (heaptrack peak attribution:
`finalize_in_progress_picture` 89.8% of peak).

Fix: `RefPicStore::retain_keys(&[u32])` (drop every slot whose key is not
live) + one call at the end of `finalize_in_progress_picture`'s reference
branch, where `dpb_entries` — marking settled, current entry pushed — is
exactly the live set. Non-reference pictures never enter the store, and the
monotonically-minted key vector keeps only `None` husks for dead slots.

Verified: 5-minute 720p25 playback flat at the DPB working set (was: linear
growth to OOM); the crate's own test suite passes.
