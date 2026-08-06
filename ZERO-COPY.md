# The road to actual zero-copy (and zero-alloc) streaming

Status quo after the session-4c allocation round (`a212b33`, measured on the 1.6 GB
movie remux with heaptrack/simprof): **87.7K allocation calls, 459 MB allocated**
(down from 1.22M / 3.51 GB). That fixed the *pathological* traffic — but the remux
is still neither zero-alloc nor zero-copy. This file is the concrete inventory of
what remains, why, and in what order to take it. Priorities follow the project's:
performance #1; every stage keeps the backpressure discipline intact.

Two independent axes, often conflated:

- **Zero-alloc**: no heap allocation on the steady-state path (the pools already
  give this for *payload* memory; the transport still allocates around it).
- **Zero-copy**: no `memcpy` of payload bytes between the kernel handing us data
  and the kernel taking it back.

## Where every byte is copied today (remux path)

A sample's bytes are touched **four times** in userspace between disk and disk:

```
filesrc read            kernel → pool slot            (unavoidable w/o O_DIRECT+registered bufs)
Mp4Reader::push         pool slot → reader window     COPY 1  (internal Vec<u8> accumulator)
Mp4Demux emit           reader window → pool slot     COPY 2  (emit_bounded memcpy)
MatroskaWriter block    pool slot → cluster staging   COPY 3  (write_simple_block extend)
MkvMux emit             cluster staging → pool slot   COPY 4  (chunked emit)
FileSink write          pool slot → kernel            (write(2); registered-buffer io_uring later)
```

Eliminating copies 1–4 is what "zero-copy" means here. The machinery for it
already exists in core — `Memory` is a refcounted `(base, offset, len)` view and
`Memory::slice()` is free — **nothing uses it yet**. That was the design intent
("a demuxer slicing one 1 MB read into 300 packets does 300 refcount bumps and
zero copies"); the elements just predate the API.

## Stage 1 — demuxer emits slices of its input (kills copies 1+2)

The single biggest change, and the model for every parser element.

- `Mp4Reader` stops owning a `Vec<u8>` window. Instead it **retains the input
  `Memory` buffers** (a `VecDeque<Memory>` rope) and resolves each sample to a
  `(chunk_index, range)` instead of `&[u8]`.
- `Mp4Demux::drain` emits `retained[chunk].slice(range)` — a refcount bump, no
  pool slot, no memcpy. The pool's job shifts from "per-sample buffers" to "IO
  read buffers" (filesrc's), whose lifetime now extends until every sample sliced
  from them has been consumed downstream — **backpressure still works** because
  the pool bounds outstanding *read* buffers, and a slow consumer holding slices
  keeps them outstanding.
- Samples straddling a read-chunk boundary (rare: one per ~128 KB chunk) either
  pay a small copy into a fresh slot (simple, recommended v1) or need
  multi-`Memory` buffers (a `Buffer` today holds exactly one `Memory` — changing
  that ripples everywhere; don't).
- Same restructure applies to `MkvDemux` (its `codec_head` reconstruction and
  laced-block splitting are already range-based internally) and any future
  demuxer. `OggDemux` too.
- **Interaction to watch**: slice-retention means CoW (`as_mut_full`) can now
  actually trigger downstream — a mutating element after a demuxer shares the
  backing with the demuxer's other slices. Decoders don't mutate input; muxers
  don't either. Fine, but assert it stays true.

## Stage 2 — muxer forwards slices, stages only headers (kills copies 3+4)

Known-size clusters (the mpv fix) currently force staging the whole cluster body.
They don't have to — the *size* must be known before the cluster header is
emitted, not the bytes made contiguous:

- `MatroskaWriter` grows a scatter mode: per block it appends the tiny
  SimpleBlock header (ID + size vint + track vint + ts + flags, ~10 bytes) into a
  small reused header buffer and records the payload as a **`Memory` slice**
  (refcount bump). The open cluster is `(header_bytes, Vec<(hdr_range, Memory)>)`
  plus a running size.
- On cluster close: emit Cluster ID + now-known size + interleaved header
  fragments and payload slices as a sequence of `Buffer`s downstream. Memory cost
  of the staged cluster becomes refcounts (≈ one GOP of *retained input* — same
  bytes the demuxer's pool already accounts for, not a second copy of them).
- The element (`MkvMux::emit`) then pushes those buffers as-is instead of
  chunking a contiguous `Vec` through pool slots.
- FLAC path: identical (frames are already whole buffers).

## Stage 3 — sink writes gather-style (keeps the tail zero-copy)

`FileSink` receives a mix of tiny header buffers and large payload slices. The
write path is already payload-zero-copy in userspace: `Submission.buf` carries
the `Buffer` (refcounted `Memory`) by value — audited, no hidden copy. What shows
in the 87.7K profile is only the `Vec<Submission>` *struct* churn (Stage 4.2).
The remaining wins here are syscall- and kernel-side:

- `SyncReactor`: `writev(2)` over a batch of buffer slices (gather IO — one
  syscall per batch, zero userspace copies).
- `IoUringReactor`: chained `writev`/registered buffers; registered pool slots
  (`IORING_REGISTER_BUFFERS` over the pool's slabs) also remove the kernel-side
  pin/copy on the *read* end. This is the reactor work PLAN already lists.

## Stage 4 — transport zero-alloc (the remaining 87.7K)

Independent of the copy work; these are the measured leftovers:

1. **`Batch` SoA column churn** (the dominant share): every scheduler pass,
   `take_output` replaces the batch with `Batch::new` (columns empty); the
   consumed batch's columns free on drop; next pass reallocates them. Fix with
   **shell reuse, locally**: after an element's input batch is drained, the
   scheduler hands the spent shell (empty columns, capacity intact) to that
   element's `Ctx` as the next output-batch donor (`take_input` → `reset` →
   `donate_output_shell`). Mid-chain elements then never allocate columns.
   Sources/ring-fed heads still do (their shells emigrate downstream) — cap that
   with a small per-`Ctx` spare-shell stack filled from arriving input batches;
   the truly general fix (a per-link return ring for shells) is not worth its
   complexity until profiles say so.
2. **Reactor vec churn**: `SyncReactor::run_once` builds a fresh completions
   `Vec` per pass and `submit` extends a submissions `Vec` through
   `into_iter`/`spec_extend`. Both become reused member buffers with `swap`.
3. **Cluster staging growth**: already amortized (one reused buffer); goes away
   entirely with Stage 2.

## Stage 5 — don't process what nobody consumes

The movie remux still demuxes, reframes and pool-copies **every audio sample**
before the scheduler drops it at the unlinked pad. With Stage 1 that cost falls
to a refcount bump + drop, which is fine — but the principled fix is the per-pad
**gate** from the stream-selection design (PLAN): a disabled/unlinked pad skips
sample resolution entirely inside the demuxer. Ties into `SELECT_STREAMS`; don't
build it just for this.

## Status (updated as stages land)

- **Stages 1+2: DONE** (zero-copy agent, merged) — payload flows as `Memory` slices
  end to end; movie remux ~140 → **1438 MiB/s**.
- **Stage 3 (hygiene half): DONE** (IO agent, merged) — fadvise/sync_file_range
  streaming hygiene; memcapped stall gone, dirty plateau 8–63 MiB. `writev` gather
  remains (the next lever for the residual below).
- **Stage 4: DONE** — shell return rings (4.1) + caller-owned reactor vecs (4.2,
  trait changed as proposed). Movie: 314.9K → **87.1K allocation calls**, 397 →
  306 MB allocated. Residual: `Batch::push` column growth in the mux's per-pass
  batches — bounded, amortized; shrinks further with writev-style batching, not
  with more recycling.
- **Stage 5: open** (`Ctx::pad_linked` gate — see PLAN, stream-selection design).

## Order and payoff

| Stage | Effort | Removes | Blocked by |
|---|---|---|---|
| 4 (transport zero-alloc) | S | ~87K allocs/movie | nothing |
| 1 (demux slices) | M | 2 of 4 copies; most pool traffic | nothing (API exists) |
| 2 (mux scatter) | M | remaining 2 copies | Stage 1 (wants slices arriving) |
| 3 (writev/uring) | M | syscall + kernel-copy tail | Stage 2 (wants gather lists) |
| 5 (pad gates) | S | dead-track work | stream-selection design |

Do 4 first (small, measurable, orthogonal), then 1 → 2 → 3 as one arc. After
1+2+4 the remux's steady state is: kernel→slot once, refcounts everywhere else,
allocations O(clusters), not O(samples). Gate each stage with the same harness:
`heaptrack --record-only` on the movie remux, read via `simprof --mcp`, compare
`allocation calls` + `bytes allocated` + the `MB/s` the example prints — and
bytes-copied-per-byte-streamed goes on the perf CI wishlist next to the ring
bench.
