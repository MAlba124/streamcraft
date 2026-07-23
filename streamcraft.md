# Streamcraft

An ultra light weight general purpose data/multimedia streaming/processing graph based
framework. Inspired by GStreamer.

The first prototype is dead — this document describes the rearchitecture from a clean
slate.

## Vision

- Must do clocking, syncing, etc. (what GStreamer is good at) at least as well or better
  than GStreamer.
- Latency must be handled *perfectly*: accounted end-to-end, measurable at runtime, and
  as low as the graph physically allows — never a mystery to be debugged.
- Data oriented: hot paths allocation free, POD for most stuff, SoA layouts for the hot
  structures, batching as the default unit of work.
- Memory is fully controllable: arenas, custom/pluggable allocators, aligned and
  registered memory. The framework never assumes malloc.
- Modern IO must be trivial: writing a source/sink on io_uring-style completion APIs
  should be the natural shape, not a fight against a blocking read-loop mold.
- Pipelines and elements as the core abstractions; pads as plain data (see below), not
  objects.
- Threading is taken care of by the framework — element code never spawns threads.
- Performance is the primary goal — highly performant and light weight above all else.
- Concept of plugins.
- The core must be written in Rust without any external dependencies. Not even
  crossbeam — we control the whole stack, so we write exactly the primitives we need.
- Compile times must be quick (dependency-free core buys most of this).
- No crazy type masturbation (no heavy use of generics or traits — one small object-safe
  element trait, everything else plain structs and enums).
- Prefer simple constructs over GStreamer's OOP machinery.
- Easily testable: the framework itself and individual elements, without threads or real
  time.
- The core defines only an opaque buffer type; helper crates interpret it as video,
  audio, etc.
- Everything easily debuggable and profilable.
- Pipelines highly dynamic, and better at it than GStreamer (dynamic gst pipelines get
  nasty).

## Performance doctrine

Performance is the #1 priority. That only means something if it is operationalized.

### Budgets, not vibes

Every framework mechanism has a stated cost budget, checked by benchmarks in CI:

| Operation                              | Budget (modern x86_64)                  |
|----------------------------------------|-----------------------------------------|
| Inline hop (passive element dispatch)  | ~5 ns + the element's actual work       |
| Queued hop, amortized, batch ≥ 32      | < 20 ns per buffer                      |
| Buffer alloc from pool                 | < 15 ns (bump + refcount init)          |
| Buffer drop / recycle                  | < 15 ns                                 |
| Clock read                             | < 25 ns                                 |
| Latency bookkeeping                    | one clock read per batch                |
| Steady-state syscalls                  | 0 per batch (park/unpark edges only)    |
| Steady-state allocations               | 0 — not "few", zero                     |
| Negotiation re-solve (100-element graph)| microseconds (relink latency depends on it) |

Budgets are enforced like correctness: a bench regression fails CI, and a feature
that can't fit its budget has that argument *in review*, not after users measure it
for us.

### The cost hierarchy

Design questions resolve against this ordering (cheap → expensive). Every structure
in this spec exists to move work up the list:

1. Register/L1 work on data already in cache — *SoA batches exist for this*
2. Predictable branches — *one format per batch exists for this*
3. L2/L3 misses — *pool locality and thread-group affinity exist for this*
4. Atomic RMW on shared lines — *batch-granularity refcounts/counters exist for this*
5. Cross-core cache-line transfer — *SPSC + cache-line ownership exist for this*
6. Syscalls — *edge-triggered park/unpark exists for this*
7. Context switches — *inline thread groups exist for this*
8. Page faults / allocation — *pools, arenas, and mlock exist for this*

### Mechanisms follow measurement

- A `perf/` harness exists from week one: hand-rolled microbenches (no dep in core —
  and we want cycles/instructions/cache-miss counts via `perf_event_open` behind a
  feature, not just wall time) with diff-vs-main reporting.
- Optimization PRs state before/after numbers and which counter moved. "Should be
  faster" doesn't merge.
- Three pinned whole-pipeline references — n-element chain, the A/V playback graph,
  a 64-branch fan-out — with checked-in GStreamer equivalents. The claim is "faster
  than gst at everything, *measured*", reproducible by anyone with `cargo bench`.

### What we refuse to pay for

- Allocation, locks, syscalls, or unbounded work on any streaming path.
- Branch-unpredictable dispatch in inner loops: dispatch once per batch, then loop
  over POD arrays.
- "Small" conveniences on `Buffer`/`Batch`: every field earns its place via a
  milestone. Convenience lives in helper crates and costs only its users.
- Observability tax when off: counters are per-batch atomics, tracing compiles out
  entirely, log macros branch on a static level before formatting anything.

## Lessons from the dead prototype

Not a base to build on, but worth writing down why it went to shit so v2 doesn't repeat
it:

- **Closed enums everywhere.** Data payloads, formats, and element types were all
  hardcoded enums in core, so adding a media library (libav) meant polluting the core
  type system. Core must be format-agnostic and payload-agnostic from day one.
- **Elements owned their downstream peers.** Topology embedded in elements means the
  pipeline can't see the graph, can't dump it, can't relink it, and every multi-output
  element invents its own ad-hoc linking API. The pipeline must own topology.
- **Thread-per-element with rendezvous channels.** A context switch per buffer per hop.
  Threading is a scheduling decision, not a structural one.
- **App-driven `iter()` handshake as the only mode.** A blocking round-trip per
  iteration can't stream. Manual stepping is a great *debug* mode, not the engine.
- **Errors died inside element threads.** Without a bus, the application never learns
  the pipeline broke.
- **Timestamps as an afterthought.** Sync is the whole point; buffers must carry timing
  metadata from the first line of code, even while nothing uses it yet.

## Architecture

Code blocks below are directional sketches — they pin down shapes, ownership, and
what is POD, not final signatures.

### Crate layout

- `streamcraft-core` — buffer, formats, element trait, topology, scheduler, clock,
  events, bus. Zero external dependencies, `#![forbid(unsafe_code)]` except one small
  audited module (buffer pool / ring buffer) with loom+miri coverage.
- `streamcraft-elements` — built-in pure-Rust elements (filesrc, queue, tee, testsrc,
  assertion sinks), per-element feature flags.
- `streamcraft-video`, `streamcraft-audio` — POD format descriptions and typed *views*
  over the opaque buffer (`VideoFrameRef`, strides, channel layouts). No new buffer
  types, no traits — free functions and plain structs.
- `streamcraft-scope` — the Slint inspector GUI + MCP server (§debuggability),
  riding on core's introspection protocol. Fully optional; apps that don't attach
  it pay nothing.
- First-party codec and container crates (`sc-flac`, `sc-opus`, `sc-vp8`, `sc-vp9`,
  `sc-av1`, `sc-mkv`, `sc-ogg`, …) — hand-written, specs checked into the tree
  (§first-party codecs). There is deliberately **no libav/FFmpeg binding** — an FFI
  wrapper is too heavyweight for this framework.

### Buffer: one opaque type, POD metadata

```rust
pub struct Timestamp(u64);   // ns; Timestamp::NONE sentinel. All timeline math in core.
// Interned ids, compared as integers, printable via the interning table:
pub struct FormatId(u32);    // likewise FieldId, ValueId, ElementId, PadId, LinkId

pub struct Buffer {
    memory: Memory,          // refcounted slice, pool-backed
    pub pts: Timestamp,
    pub dts: Timestamp,
    pub duration: Timestamp,
    pub flags: BufferFlags,  // bitflags: KEYFRAME, DISCONT, GAP, ...
    pub format: FormatId,
}
```

- `Memory` is a refcounted slice into a **buffer pool**: arenas allocated up front,
  recycled on last-ref drop. Steady-state streaming does zero allocations; pool
  exhaustion falls back to plain heap so nothing deadlocks.
- Fan-out (tee) bumps a refcount, never copies payload. Copy-on-write only when a
  downstream wants mutable access to a shared buffer.
- Everything but `memory` is POD: cheap to copy, trivially loggable/serializable, which
  directly feeds the debuggability goal.
- Elements that need out-of-band payloads (e.g. a GPU/device buffer handle) get an
  `ExternalMemory` variant: a pointer + vtable-free drop fn. Core still doesn't know
  what's inside.

### Metadata, blobs, and tags (the POD escape hatches)

Three real-world needs puncture a purely fixed-field POD model. Each gets a designed
hole now, instead of an ad-hoc one bolted on later:

- **Codec config blobs** (H.264 SPS/PPS, FLAC STREAMINFO): `FixedFormat` carries an
  optional refcounted, immutable blob reference. Equality uses (length, hash) so
  formats stay cheaply comparable; the blob is set at negotiation/`FormatChange`
  time only — never touched per buffer.
- **Per-buffer metadata** (crop rects, timecodes, HDR dynamic metadata, ROI):
  batches carry an optional parallel `metas: &[MetaRef]` column — null for almost
  every buffer, so the common path pays one perfectly-predicted pointer-width load.
  A `MetaRef` points into a per-pool side arena of `(MetaId, POD payload)` entries,
  recycled with the pool. Open-ended like GstMeta, but flat, POD, allocation-free.
- **Stream tags** (title/artist/language — wav→flac must preserve them): a `Tag`
  in-band event, so tags stay ordered with the stream they describe, mirrored to
  the bus for the app. Payloads are interned-key → `Value` pairs, plus the blob
  reference for cover art.

Rule for all three: the hot path may *carry* them but never *interprets* them. Only
an element that asked for a given `MetaId` pays for reading it.

### Memory: arenas, custom allocators, no malloc assumptions

- **Pools are built on a pluggable allocator** — a plain struct of fn pointers (no
  generics, per the vision): default heap arenas, hugepage-backed, NUMA-node-pinned,
  mmap'd files, shared memory (future IPC transport), and device memory
  (DMA-buf/GPU) so zero-copy capture→encode paths are possible. Core defines the
  allocator contract; helper crates provide the exotic ones.
- **Alignment and padding guarantees**: pool allocations are cache-line aligned (64B
  minimum, configurable up to page/hugepage) and size-padded, so elements can use
  full-width SIMD loads without tail handling and O_DIRECT IO without copies.
- **Registered memory**: pools can pre-register their arenas with external APIs once at
  `Ready` time (io_uring registered buffers, RDMA, GPU pinning). A buffer handed to an
  io_uring filesrc is already a fixed buffer — zero per-IO registration cost.
- **Scratch arenas**: `Ctx` hands elements a bump allocator that resets after each
  `process()` call. Temporary tables, staging copies, string scratch — all free, no
  element-local `Vec` reuse dances, no hidden malloc in hot paths.
- Rule of thumb the API enforces: allocation happens at state changes (`Stopped→Ready`),
  never per buffer. Debug builds can assert this (pool counters + a panicking allocator
  guard around `process()`).

```rust
pub struct Allocator {          // plain fn-pointer vtable — no generics
    ctx: *mut (),
    alloc: unsafe fn(*mut (), Layout) -> *mut u8,
    free: unsafe fn(*mut (), *mut u8, Layout),
    // Called once per arena at Ready: io_uring fixed buffers, RDMA, GPU pinning.
    register: Option<unsafe fn(*mut (), arena: *mut u8, len: usize, target: &RegisterTarget)>,
}

pub struct PoolConfig {
    pub slot_size: usize,
    pub slots: u32,
    pub align: usize,           // ≥ 64; page/hugepage for O_DIRECT
    pub allocator: Allocator,   // heap arena by default
}
```

### Device memory and sync points

Zero-copy GPU paths (VAAPI/V4L2/Vulkan decode → render) are asynchronous in a way
system memory isn't: the producer hands over a buffer whose contents aren't ready
until the device signals. Bolting this on late is how GStreamer got its GL stack;
the hole is designed in now:

- `Buffer` carries an optional **`SyncPoint`** (fence handle + vtable-free
  wait/poll fns). `None` for system memory — zero cost on the common path.
- A buffer is *consumable* when its sync point has signaled. A clock-driven sink's
  readiness is `fence signaled AND render time reached`, and fence waits fold into
  the same reactor/`ClockWait` machinery as everything else — DMA fences are
  pollable fds on Linux, i.e. just another completion.
- CPU access to device memory is explicit: `Memory::map(Read|Write)` can be
  expensive or fail, so passive elements declare mapped-access needs and the solver
  places an explicit download element (visible in the dump) when a CPU-only element
  sits in a GPU path. gst's silent round-trip through system memory is banned.
- Format families distinguish placement (`video/raw` vs `video/dmabuf`), so GPU↔CPU
  boundaries are ordinary negotiation: same loud failure, same opt-in visible
  auto-conversion (upload/download elements).

### Formats and negotiation (caps, without the caps system)

The core inversion vs. GStreamer: negotiation logic never runs inside elements. Pads
*declare* what they support as plain data; the **pipeline solves the whole graph as a
constraint pass at `Stopped→Ready`**. No caps queries recursing through pads, no sticky
events, no per-element negotiation code, no `GstBaseTransform`-style negotiation logic
anywhere.

- **Open vocabulary, closed algebra** — the opposite of GstCaps, which is open in both.
  Format *families* (`video/raw`, `h264/annexb`) and field names (`width`, `rate`,
  `pixfmt`) are interned `u32`s: open set, plugins add freely, core never knows their
  meaning. Field *values* come from a tiny closed set: `Int(i64)`, `Rational(i32,i32)`,
  `Id(u32)`. A pad's per-field constraint is `Any | Eq | Range{min,max,step} | Set`.
  That is the entire constraint language — intersection is a page of integer code, and
  GstCaps' costs (heap-allocated structures, string-keyed lookups, generalized
  fixation) are structurally impossible.
- **Fixed formats live on edges.** The solve produces exactly one fixed format per link
  (`FormatId` + flat POD params blob, memcmp-comparable); it is stored in the topology
  and elements read theirs from `Ctx` in `start()`. Negotiation failure is a
  `Ready`-time bus error naming the link, both sides' offers, and the empty
  intersection — never a mid-stream flow error. The graph dump shows, per edge, what
  was chosen and why.
- **Fixation with preferences**: each side may state a preferred value per field
  (source prefers native resolution, sink prefers display rate); the solver picks from
  the surviving range.
- **Dynamic pads arrive pre-fixed** — a demuxer knows its stream's format, so linking a
  dynamic pad is a compatibility check, not a negotiation round.
- **Mid-stream changes** are an in-band `FormatChange` event carrying a new fixed
  format, ordered with buffers; receivers declared what they accept, and anything
  structural goes through the safe-point relink machinery. Downstream-initiated
  renegotiation (sink window resized) is a pipeline operation requested via the bus.
  Never a RECONFIGURE-style event ping-pong.
- **Link constraints replace capsfilter**: `link()` takes optional per-field
  constraints that narrow the solver on that edge — no dummy element in the graph.
- **Auto-conversion is opt-in and visible**: v1 fails loudly on empty intersection.
  Later, an explicit flag lets the pipeline insert registered converter elements —
  always visible in the dump, never silent magic.
- **Pool negotiation is decoupled.** gst couples allocation to caps via the ALLOCATION
  query — a major complexity source. Here it is a separate, simpler pass after formats
  are fixed: the format determines buffer sizes, link peers state memory requirements
  (alignment, registration, device memory) as POD, and the pipeline picks or creates
  the pool.
- **Escape hatch** for genuinely coupled constraints ("width×height ≤ N macroblocks at
  this profile"): a pad may declare a validation callback the solver consults during
  fixation. It can reject candidates but cannot extend the algebra — the common path
  stays simple.

```rust
pub enum Value { Int(i64), Rat(i32, i32), Id(ValueId) }   // the whole value universe

pub enum Constraint {
    Any,
    Eq(Value),
    Range { min: Value, max: Value, step: Value },
    Set(&'static [Value]),
}

pub struct FieldConstraint {
    pub field: FieldId,             // interned: WIDTH, RATE, PIXFMT, ...
    pub allowed: Constraint,
    pub preferred: Option<Value>,   // fixation hint (native resolution, display rate)
}

pub struct FormatOffer {
    pub family: FormatId,           // "video/raw", "audio/raw", "h264/annexb", ...
    pub fields: &'static [FieldConstraint],
}

pub struct FixedFormat {            // the solve's per-edge output; flat POD, memcmp-eq
    pub family: FormatId,
    len: u8,
    fields: [(FieldId, Value); 16],
}
```

How the solve runs (and why it's fast): each link's intersection is field-wise
interval/set arithmetic over ≤16 integer fields — nanoseconds per link. Propagation
walks the DAG in topological order (transforms forward constraints: "output rate =
input rate"), so one pass fixes most graphs; elements with coupled in/out
constraints get a bounded number of refinement passes. There is no backtracking
search — where gst's general caps can require it, the closed algebra plus explicit
converters can't. Runtime re-solves are incremental: only the affected subgraph
re-runs, against the already-fixed formats at its boundary. Solve time is budgeted
and benchmarked (§performance doctrine) because relink latency depends on it.

### Elements and pads

- One small object-safe trait, roughly:

  ```rust
  pub enum Flow { Ok, NeedMore, Eos }

  pub trait Element: Send {
      fn desc(&self) -> &'static ElementDesc;   // points at a shared static
      fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error>;
      fn process(&mut self, ctx: &mut Ctx, inputs: Inputs<'_>) -> Result<Flow, Error>;
      fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error>;
      fn stop(&mut self, ctx: &mut Ctx);
  }
  ```

  Note `Inputs` of `BatchRef`s, not a single buffer — batching *and* multi-input
  (§aggregation) are in the trait from day one, because retrofitting either means
  touching every element ever written.
  Everything *static* about an element lives in a descriptor, not on the trait:

  ```rust
  pub struct ElementDesc {
      pub name: &'static str,
      pub pads: &'static [PadDesc],
      pub props: &'static [PropDesc],
      pub sched: SchedHint,               // Passive | Active
      pub inputs: InputPolicy,            // §aggregation: who waits, and how
      pub latency: LatencyDesc,           // min/max ns, is_live, jitter
      // Only for the string/parse path (§plugins): default-construct, then apply
      // parsed props. Typed `T::new(...)` is how elements are actually made.
      pub make_default: Option<fn() -> Box<dyn Element>>,
  }

  pub struct PadDesc {
      pub name: &'static str,
      pub direction: Direction,           // Sink | Src
      pub offers: &'static [FormatOffer],
      pub dynamic: bool,
      pub validate: Option<fn(&FixedFormat) -> bool>,  // fixation escape hatch
  }

  pub struct PropDesc {
      pub name: &'static str,
      pub allowed: Constraint,            // reuses the format algebra for validation
      pub live: bool,                     // settable at batch boundaries while Playing
  }
  ```

- **Construction is direct and typed** — `HttpSrc::new(url)`, `Volume::new(1.0)` —
  and `pipeline.add()` takes the constructed element. No string-factory indirection
  on the primary path; "element by name" exists only in the opt-in parse layer
  (§plugins). Typed constructors also make required config required at compile
  time — the "element created but mandatory property never set" runtime failure
  can't exist.

- **`Ctx` is the element's entire world** — output, memory, time, IO, bus. Elements
  hold no channels, no threads, no peers:

  ```rust
  impl Ctx {
      pub fn negotiated(&self, pad: PadId) -> &FixedFormat;
      pub fn alloc(&mut self, pad: PadId) -> Buffer;      // that pad's pool; never malloc
      pub fn scratch(&mut self) -> &mut Arena;            // bump alloc, reset after process()
      pub fn out(&mut self, pad: PadId) -> OutBatch<'_>;  // SoA writer into queue slots
      pub fn send_event(&mut self, pad: PadId, event: Event);
      pub fn post(&mut self, msg: BusMessage);
      pub fn now(&self) -> Timestamp;                     // pipeline clock
      pub fn running_time(&self) -> Timestamp;
      pub fn io(&mut self) -> &mut Io;                    // reactor handle (active elements)
      pub fn add_pad(&mut self, desc: &PadDesc, format: FixedFormat) -> PadId;
  }
  ```
- **Pads are plain data** (`PadDesc`), not objects. Dynamic pads (demuxer discovering
  streams) are announced via the bus; the app or an autoplugger links them.
- The **pipeline owns topology**: `pipeline.link(demux, "video_src", dec, "sink")`.
  This single decision unlocks graph dumping, dynamic relinking, and scheduling.
- Lifecycle is owned by the framework wrapper around the element (start/stop/drop run
  exactly once, in order) — element authors can't get it wrong, no macros required for
  correctness.

### Aggregation: elements with multiple inputs

Muxers, mixers, and compositors are where GStreamer needed a whole late-added base
class (`GstAggregator`) and where naive designs deadlock or freeze on one stalled
branch. Multi-input is in the element contract from day one — and as always, the
*element* declares policy while the *scheduler* implements the waiting:

```rust
pub enum InputPolicy {
    None,                        // pure source
    Single,                      // the common transform case
    Any,                         // process whatever arrived (funnel, interleaver)
    All { by: AlignBy },         // wait for all linked pads (muxer: AlignBy::Dts)
    AllDeadline { by: AlignBy, timeout: Timestamp },  // live mixer: never stall
}

pub struct Inputs<'a> { /* which pads have data, and their batches */ }
impl<'a> Inputs<'a> {
    pub fn single(&self) -> Option<BatchRef<'a>>;           // Single-policy sugar
    pub fn get(&self, pad: PadId) -> Option<BatchRef<'a>>;
    pub fn iter(&self) -> impl Iterator<Item = (PadId, BatchRef<'a>)>;
}
```

- For `All`, the scheduler wakes the element only when every linked, non-EOS sink
  pad has data aligned to the requested point (by dts or running time). The element
  never writes waiting logic, so it can't write it wrong.
- For `AllDeadline`, the wake fires at data-complete *or* deadline, whichever comes
  first; pads that missed show as absent, the element mixes without them, and the
  miss lands in the QoS counters. A stalled live branch structurally cannot freeze
  a mix.
- EOS is per-pad: a finished pad stops participating in `All` waits; the element
  sees it via `event()` and completes when all pads are done.
- Sparse inputs combine with `All` via the GAP protocol (§data-flow rules): a GAP
  batch means "nothing through running time T" and satisfies the wait — a silent
  subtitle track doesn't stall a muxer.

### No bins — templates, groups, and controllers

GStreamer's bin bundles three unrelated jobs into one object, and all three cause pain:
containment (ghost pads proxying every boundary pad), state aggregation (the arcane
child-state rules), and autoplugging (decodebin: a bin that rewires itself from
inside). Streamcraft has no bin object; the topology stays flat. The three jobs get
three separate, simpler tools:

- **Templates** for reuse: a subgraph template is a function that expands into plain
  elements + links in the flat graph, exporting named boundary pads. Composition
  happens at construction time; after expansion the pipeline sees ordinary elements.
  No ghost pads, no proxying, no runtime container.
- **Groups** for organization: every element carries a group path label
  (`"player/decode/video"`). Dumps render clustered by group; bulk operations (remove
  a group, set a property across a group) are atomic pipeline ops over a label query.
  Pure metadata — zero flow-path cost.
- **Controllers** for dynamics: decodebin's actual job — watch `PadAdded`, pick
  decoders, rewire — is bus-listening logic issuing first-class relink operations.
  Ship an autoplug controller as a library, not as a magic self-modifying graph node.
- State aggregation isn't needed at all: state is pipeline-global, pause is a clock
  operation, and subgraph-local drain/re-preroll is internal machinery (see runtime
  configuration below), not app-visible state juggling.

### Batching: the unit of work is a span, not a buffer

GStreamer's unit of flow is one `GstBuffer` per pad-push — per-buffer locking,
per-buffer dispatch, per-buffer everything. Streamcraft's unit is a **batch**:

- `process()` receives and pushes *spans* of buffers. A batch of one is legal and
  normal (a 4K video frame is plenty of work); a batch of 512 audio chunks or network
  packets is where it pays. Element code is written once, over spans, and gets both.

  ```rust
  pub struct BatchRef<'a> {          // SoA: parallel arrays; format-homogeneous
      pub format: FormatId,          // one format per batch — FormatChange ends a batch
      pub memories: &'a [Memory],
      pub pts: &'a [Timestamp],
      pub durations: &'a [Timestamp],
      pub flags: &'a [BufferFlags],
  }

  impl<'a> BatchRef<'a> {
      pub fn len(&self) -> usize;
      pub fn get(&self, i: usize) -> Buffer;   // row view — batch-of-one ergonomics
      pub fn iovecs(&self) -> IoVecs<'a>;      // scatter-gather view for sinks
  }

  // From Ctx::out(pad): rows are written straight into the outgoing queue's
  // slots — building an output batch *is* enqueueing it.
  impl OutBatch<'_> {
      pub fn push(&mut self, buf: Buffer);
      pub fn reserve(&mut self) -> Option<BufferSlot<'_>>;  // in-place construction
  }
  ```
- **Queues transfer batches**: one atomic operation and at most one wakeup per batch,
  not per buffer. Synchronization cost amortizes to near zero at high rates.
- **Batch metadata is SoA**: a batch is `ptss: &[Timestamp]`, `durations: &[Timestamp]`,
  `flags: &[BufferFlags]`, `memories: &[Memory]` — parallel arrays, not an array of
  structs. Scanning timestamps (QoS lateness checks, latency accounting, seek
  scanning, gap detection) touches only the pts array: cache-dense, vectorizable.
  The wrapper's per-buffer latency timestamping is one clock read *per batch* plus a
  vectorized subtract.
- **The latency budget decides batch limits**: batching trades latency for throughput,
  so max batch size/age per queue falls out of the latency computation (§Latency)
  instead of being another knob to mistune. Non-live pipelines batch maximally; a
  10 ms-budget live path gets small batches automatically.
- **Sinks get scatter-gather views**: a batch exposes an iovec view over its memories,
  so network/file sinks do one `writev`/io_uring submission per batch — no coalescing
  copies.
- Helper crates lean in: audio helpers operate on whole batches (SIMD across the span),
  and planar audio/video formats are themselves SoA in memory — the natural fit.

### IO: built for the io_uring era

Blocking `read()` loops are the mold GStreamer casts sources into (`GstBaseSrc::create`,
one blocking call per buffer). Completion-based sources/sinks must be the *easy* shape:

- **Active elements own an event loop.** An active element's `process(None)` isn't
  "produce one buffer" — the element may keep many operations in flight and emit
  whatever completed. The contract is completion-friendly by construction.
- **Everything the framework wants from an element is waitable.** Pipeline wakeups
  (queue space, messages, shutdown) are exposed as an `eventfd`-style handle, and
  `ClockWait` deadlines convert to timeout args/timerfd — so an io_uring element waits
  on its CQ *and* the pipeline in a single `io_uring_enter`. No second thread, no
  polling, no framework-vs-ring tug of war.
- **Pool ↔ ring integration**: registered buffers (see Memory) plus completion-owned
  buffers — a source submits reads directly into pool memory; the completion becomes a
  `Buffer` with zero copies and zero per-IO bookkeeping. Multi-shot receive fills a
  batch per wakeup.
- Core stays dependency-free: core defines the waitable-handle and registration
  contracts; the actual io_uring source/sink lives in `streamcraft-elements` (Linux)
  behind a feature, with an epoll/blocking fallback element sharing the same element
  code above the submission layer.
- Same contracts cover the neighbors: V4L2 (queued/dequeued buffers are
  completion-shaped), ALSA, sockets with `MSG_ZEROCOPY`, RDMA later. If a source/sink
  API hands you completed buffers asynchronously, streamcraft should feel native.

How IO elements *share* rings — never one ring per element (kernel resources, per-ring
registered-buffer tables, one parked thread each):

- **One reactor per thread group, owned by the scheduler.** The ring is a per-thread
  resource like the scratch arena. Elements submit through a `Ctx` submit/complete
  interface and never see the ring; which elements share a reactor is placement
  policy — the same knob as thread grouping. Consolidate all IO onto one reactor
  thread, or isolate a hot NIC source onto its own.
- **The drive loop**: one `io_uring_enter` waiting on the CQ (pipeline wakeups arrive
  as a multishot eventfd poll in the same ring), drain the whole CQ, group completions
  by element — which forms natural batches — run the inline passive chains downstream,
  top up submissions, repeat.
- **Backpressure = submission credits**: an element may keep only as many operations in
  flight as it has pool buffers and downstream queue space. No unbounded in-flight IO,
  and pressure propagates into the kernel (no credits → nothing submitted) where it
  belongs.
- **Cancellation by identity**: SQE `user_data` encodes (element, op), so flush, seek,
  and shutdown cancel by element prefix and drain — flush-interruptibility extended
  into the kernel.
- **Registration follows placement**: pools register arenas with their group's ring at
  `Ready`; a relink that migrates an element to another group re-registers at the safe
  point, invisibly to element code. Sinks running on IO threads fold `ClockWait`
  deadlines into ring timeouts — rendering deadlines and IO completions in one wait.
- **The reactor is the portability boundary**: Linux gets io_uring; the fallback is
  epoll plus a small blocking-op thread pool behind the identical submit/complete
  interface. Element code is written once.

```rust
pub struct OpId(u64);              // packs (element, seq) — enables cancel-by-prefix

impl Io {
    pub fn credits(&self) -> u32;  // in-flight budget = pool slots + downstream space
    pub fn read(&mut self, fd: RawFd, into: Buffer, offset: u64) -> OpId;
    pub fn write_batch(&mut self, fd: RawFd, batch: BatchRef) -> OpId;  // one writev
    pub fn cancel(&mut self, op: OpId);
    pub fn next(&mut self) -> Option<Completion>;   // drained inside process()
}

pub struct Completion {
    pub op: OpId,
    pub result: i32,               // raw errno-style; structured at the element edge
    pub buffer: Option<Buffer>,    // a completed read lands as a ready Buffer
}
```

### Pull-mode boundaries: callback audio

JACK, CoreAudio, and pro-audio APIs invert control: a realtime thread the OS owns
calls *you*, demanding exactly N samples right now. That thread must never touch a
queue that can block, a pool that can miss, or the bus. The reactor model doesn't
cover this, so a third boundary primitive exists (alongside inline calls and SPSC
queues):

- A **wait-free ring** sized from the negotiated latency: the framework side (a
  normal scheduled element) keeps it filled; the foreign callback drains it with a
  bounded number of atomic loads plus a memcpy, and on underrun writes silence and
  bumps a counter. It never waits. Ever.
- The callback cadence *is* the audio clock: the sink publishes position
  observations from the callback (two atomic stores); the clock system consumes
  them on normal threads for master-clock duty and slaving (§clocking).
- RT thread configuration is declared, not hand-rolled: elements state requirements
  (`SCHED_FIFO` priority, no-page-fault) in the descriptor; the scheduler applies
  them to the thread group and *verifies* (mlocked pools, pre-faulted stacks)
  rather than hoping.

### Scheduling and threading

The biggest performance lever, and where we out-lightweight GStreamer:

- Elements declare a hint: `Passive` (pure transform, callable inline) or `Active`
  (blocks — file IO, device, clock-waiting sink).
- The pipeline compiles the graph into **thread groups**: chains of passive elements run
  inline on one thread as plain function calls — zero context switches, zero queues.
  Real queues exist only at group boundaries: around active elements, at branches, and
  wherever the user drops an explicit `queue` element.
- Boundary queues are purpose-built bounded SPSC ring buffers (in core, no deps) with
  configurable capacity and leaky policies (block / drop-oldest / drop-newest) for live
  sources.
- No async runtime, no work stealing. Static, predictable, inspectable schedule — you
  can print which thread runs which elements.
- **Topology-aware placement**: thread groups are pinnable to cores, and a group's
  pools allocate NUMA-local to its pin. Queue producer/consumer fields live on
  separate cache lines (no false sharing); wakeups are adaptive spin-then-park, and
  batching means at most one wakeup per batch anyway. Steady state is zero syscalls:
  no clock reads beyond one per batch, no futex traffic while queues are neither
  empty nor full.
- **The `Passive` contract is enforced, not hoped**: in debug/telemetry builds the
  wrapper times `process()` and flags a passive element exceeding its budget
  (default: a fraction of the tightest downstream deadline) on the bus. A blocking
  "passive" element poisons its whole inline chain — it gets caught in development,
  not production.
- Two driving modes over the same scheduler: `play()` (free-running) and `step()`
  (manual single-iteration stepping for tests and debugging — the one good idea from
  the prototype, kept as a debug mode).

### Queue internals (the SPSC ring)

The boundary queue is the most-executed data structure in the framework; its design
is spec-level, not an implementation detail:

- Power-of-two slot count, and the slots *are* the SoA batch columns (the pts
  array, the memory array, …). Transferring a batch is publishing an index range
  over the ring's columns, not moving structs. `OutBatch` writes land directly in
  the consumer-visible storage — "building the output batch" and "enqueueing it"
  are the same store.
- Producer and consumer each own a cache line — `{head, cached_tail}` /
  `{tail, cached_head}`, 64-byte padded. The cached peer index means the fast path
  performs *zero* shared-line loads until the cached value would block, amortizing
  coherence traffic to roughly one line transfer per ring wrap.
- Two atomics per batch (Release-publish index, Acquire-load peer), independent of
  batch size. Every ordering decision is documented per-field in the source and
  model-checked with loom.
- Wakeups are edge-triggered only: producer signals on empty→non-empty, consumer on
  full→non-full. A busy pipeline performs zero wake syscalls. Parking is adaptive
  spin (~µs) → yield → futex/eventfd, tunable per queue — live audio and bulk
  transcode want different points on the latency/CPU curve.
- Leaky modes (`Block | DropOldest | DropNewest`) are a producer-side decision made
  before writing, with drop counts on the counters; dropping never touches the
  consumer's line.
- Flush runs through a generation counter: a flushed ring rejects stale publishes
  from before the flush. This is what makes seek race-free without locks.
- The bus is the one MPSC structure (many posters), deliberately kept off every hot
  path — see §events for its bounding policy.

### Clocking and synchronization (the flagship)

Where "as good as or better than GStreamer" is won or lost:

- **Clock trait** (monotonic), default `Instant`-backed implementation; audio sinks can
  provide the master clock (audio hardware clock is the usual master). Exactly one
  master per pipeline. A **mock clock** ships in core for tests.
- **Base time + running time**: sinks compute `render_time = base_time + pts` and wait;
  only sinks wait — upstream runs as fast as backpressure allows.
- **`ClockWait` primitive**: interruptible wait (for flush/seek/shutdown), written
  early, small, hammered with mock-clock tests. Miserable to retrofit, cheap to build
  first.

  ```rust
  pub trait Clock: Send + Sync {   // the one deliberately tiny trait
      fn now(&self) -> Timestamp;
  }                                // core ships InstantClock and MockClock

  pub enum WaitOutcome { Reached, Interrupted }

  impl ClockWait {
      pub fn wait_until(&self, deadline: Timestamp) -> WaitOutcome;
      pub fn interrupt(&self);                    // flush/seek/shutdown path
      pub fn as_waitable(&self) -> RawWaitable;   // timerfd-shaped; folds into a ring
  }
  ```
- **QoS**: sinks post plain-struct lateness observations to the bus; adaptive elements
  can subscribe and drop/degrade. Even a minimal version beats gst's QoS on
  debuggability because observations are visible, plain data.
- Pausing is a **clock operation** (freeze/unfreeze), not an element state — sidesteps
  the whole class of gst state-change bugs around PAUSED.

- **Clock slaving, because re-mastering is banned**: a second audio device, a
  network-synced peer, or an inbound RTP/SRT stream each have a timebase that
  *will* drift from the master. Every such boundary runs the same slaving
  mechanism: collect (master_time, local_time) observations, estimate skew with a
  small regression/PI controller, correct at the boundary — audio sinks by
  micro-resampling (a few ppm, inaudible), video sinks by deadline nudging,
  sources by timestamp rate-mapping. Slaving state (current skew, correction rate)
  is in the counters, because "why is this output drifting" must be answerable
  from a dump. Network clocks (PTP-style, or a simple UDP protocol for multi-room
  audio) are just `Clock` impls feeding the same slaving path.

### Latency (first-class, not a footnote)

Latency in GStreamer is where debugging goes to die — queries, min/max ranges, live-ness
flags, and `latency` messages that few people fully understand. Streamcraft treats
latency as a budget that is *computed, enforced, and observable*:

- **Declared, per element**: every element states its processing latency as POD
  (`min_ns`, `max_ns`, plus `is_live` for sources that produce data pegged to real
  time). Passive inline elements default to zero — and because inline hops are function
  calls, that zero is actually true, not an approximation.
- **Computed, per path**: since the pipeline owns topology, it computes the exact
  latency along every source→sink path (element latencies + worst-case queue residency,
  which is known because queues are bounded with known capacities and formats have
  known rates). No distributed query protocol — it's a graph traversal over data the
  pipeline already holds.
- **Enforced at sinks**: sinks render at `base_time + pts + path_latency`. Live
  pipelines get exactly the delay the graph requires and not a nanosecond of slack by
  default; non-live pipelines get throughput mode (no waiting at all).
- **A pipeline-level latency budget knob**: the app can say "target ≤ N ms end-to-end".
  The pipeline sizes queues to fit the budget, switches the relevant queues to leaky
  mode, and reports (on the bus, with per-element numbers) if the graph *cannot* meet
  the budget — at `Ready` time, before data flows, not as mysterious runtime stutter.
- **Measured, always**: every buffer's true source-to-sink latency is observable — the
  wrapper timestamps buffer entry per element (cheap: one clock read into a POD field),
  so runtime histograms of per-element and end-to-end latency are available via the
  counters API and the graph dump shows where the budget actually goes. When latency is
  wrong, the answer is one dump away, never a printf hunt.
- **Recomputed on change**: relinks, dynamic pads, and renegotiation trigger automatic
  recomputation and a bus notification with old/new values. Latency changes are events
  the app can see, not silent drift.
- **Jitter absorption is explicit**: live sources declare their jitter; the pipeline
  places exactly one jitter-absorbing queue at the right spot per path instead of the
  gst pattern of scattering queues until the stutter stops.

### Events, queries, and the bus

GStreamer has four overlapping mechanisms — serialized and non-serialized events,
queries, bus messages, and GObject signals — each with different threading and
blocking semantics, and its deadlocks live in their intersections. Streamcraft has
exactly **two channels, and no query system at all**:

- **In-band events** travel *with* buffers through the same queues, ordered relative
  to them: `Segment`, `FormatChange`, `Eos`, `FlushStart/Stop`. Delivered to
  `Element::event()` at batch boundaries. There are no "non-serialized" events: the
  out-of-band cases gst uses them for (flush, seek initiation) are pipeline
  operations that interrupt via `ClockWait::interrupt()` and reactor cancel — they
  don't race the data path as messages.
- **The bus** carries everything element/framework → application: every element
  error ends up here (errors never die in threads), plus EOS, state completions,
  dynamic pads, latency changes, QoS. It is an MPSC queue drained *by the app on the
  app's own thread* — `try_recv()`, blocking `recv()`, or a waitable handle for any
  event loop. No GLib main loop, no callbacks firing on random streaming threads.
  Posting never blocks the data path.
- **Queries are eliminated, not replaced.** Every gst query becomes data the
  pipeline already owns, read synchronously from topology state: caps → the solved
  `FixedFormat` on each edge; latency → the computed per-path report; position →
  running time from the clock; duration/seekability → declared by sources at `Ready`
  into topology metadata. Nothing traverses the graph at runtime to answer a
  question the solve already answered.
- App → element communication is not special either: property sets and pipeline ops
  (§runtime configuration). There is no signal system; "signals" are bus messages.
- **Topology changes are observable**: `ElementAdded/Removed`, `LinkChanged` — so
  monitors and supervisors can track mutations they didn't initiate. There are no
  `element-added` *callbacks*: the gst use case (configure elements someone else
  created, before they run) doesn't need them here, because templates take typed
  config as arguments and `add_subgraph()` is synchronous — you can enumerate and
  configure the returned group before the join goes live. Framework→app is always
  async bus data; app→framework is always sync calls on the app's thread. The
  callback deadlock class doesn't exist.
- **The bus is bounded**, with two message classes: droppable (QoS observations,
  progress chatter) drop-oldest with a counter; critical (errors, EOS, state) never
  drop — posting one evicts droppables if needed. The streaming path is never
  blocked by an app that forgot to drain its bus.
- **Unlinked dynamic pads have a stated policy**: between `PadAdded` and the app's
  link operation, data buffers up to that pad's queue cap (sized from the latency
  budget), then drops oldest with a counter. A slow-reacting app degrades
  observably — it doesn't stall the demuxer, and it doesn't silently lose the
  stream's start.

```rust
pub enum Event {                    // in-band, ordered with buffers
    Segment { base: Timestamp, rate: f64 },
    FormatChange(FixedFormat),
    Eos,
    FlushStart,
    FlushStop,
}

pub enum BusMessage {               // out-of-band → application; plain data
    Error { element: ElementId, error: Error },
    Warning { element: ElementId, error: Error },
    Eos,
    StateChanged { old: State, new: State },
    PadAdded { element: ElementId, pad: PadId, format: FixedFormat },
    ElementAdded { element: ElementId, group: GroupId },
    ElementRemoved { element: ElementId },
    LinkChanged { link: LinkId },
    Tags { element: ElementId, tags: TagList },
    BranchSealed { group: GroupId, error: Error },      // §supervision
    SubgraphJoined { group: GroupId, added_latency: Timestamp },
    LatencyChanged { old: Timestamp, new: Timestamp },
    Qos { sink: ElementId, lateness_ns: i64 },
}

impl Bus {
    pub fn try_recv(&self) -> Option<BusMessage>;
    pub fn recv(&self) -> BusMessage;             // blocking; app thread only
    pub fn as_waitable(&self) -> RawWaitable;     // integrate into any event loop
}
```
- **State machine**, deliberately smaller than gst's: `Stopped` (no threads, no
  resources) → `Ready` (resources open, negotiated, prerolled) → `Playing`. Pause is a
  clock freeze, see above. Async transitions complete via the bus.
- **Seeking**: flush (interrupts `ClockWait`s, drains queues) → seek to sources → new
  segment downstream → re-preroll. Flush-interruptibility is designed into the queue
  and clock primitives from day one.

### Data-flow rules

Global invariants every element and the scheduler can rely on:

- **The graph is a DAG.** Cycles are rejected at link time. Feedback (echo
  cancellation, sidechain compression) uses an explicit `feedback` element pair — a
  bounded ring seeded with neutral data — keeping loop latency visible and the
  schedule acyclic.
- **Order is per-link**: buffers and in-band events on one link arrive in the order
  sent, always. Cross-link ordering exists only where an aggregation policy creates
  it. Within a link, delivery order is decode order (`dts`); `pts` may reorder
  (B-frames), and only sinks/aggregators care — they size their reorder window from
  the format's declared reorder depth.
- **GAP protocol for sparse streams**: a stream with nothing to say still speaks.
  Sources and parsers emit GAP batches ("no data through running time T" — one POD
  entry, no memory) at a declared heartbeat. Preroll, `All`-aggregation, and
  latency accounting treat GAP as data, so subtitle tracks, muted mics, and blanked
  video never stall anything.
- **Trick modes, v1 semantics**: `Segment.rate` scales the running-time mapping at
  sinks — fast/slow motion for free; audio elements that can do better
  (scaletempo) handle `Segment` themselves. Reverse playback is explicitly out of
  v1: it needs a per-decoder protocol (keyframe-batch reversal). The segment
  carries the field so the design space stays open; sources reject negative rates
  until then.
- **Drain and abort are distinct shutdowns**: *drain* pushes EOS from sources and
  waits (bounded) for sinks and muxers to finalize — trailers written, files valid,
  always. *Abort* flushes, cancels kernel ops, tears down now. `scraft-launch` maps
  Ctrl-C to drain, second Ctrl-C to abort — the recording-app correctness pattern,
  owned by the framework.
- **Pool swap protocol**: when a mid-stream `FormatChange` alters buffer sizes, the
  affected chain runs quiesce-at-safe-point → new pool (+ ring re-registration) →
  swap → resume; the old pool retires when its last in-flight buffer drops.
  Bounded, race-free, counter-visible. Never a stall, never a stale-size buffer
  downstream.

### Dynamic pipelines, done better than gst

Why gst dynamics get nasty: pad blocking, probes, and manual event choreography that
every app reimplements slightly wrong. Since the framework owns topology and the
scheduler knows the safe points between buffers:

- `pipeline.relink(...)` is a first-class operation executed by the scheduler at a safe
  point — no user-visible pad blocking, no probe callbacks, ever.
- The framework does the event bookkeeping internally: drain the old branch, EOS it,
  send segment + format into the new branch.
- Add/remove elements while `Playing` via the same mechanism.
- Stress-tested feature: relink thousands of times under load, under loom/sanitizers.
  "Dynamic pipelines that don't crash" is a headline feature.

### Runtime configuration in every state

Every state is configurable; what varies is the mechanism, never the legality. No
operation's documented answer is "tear it down and rebuild" — that is the gst failure
mode this design defines itself against.

- **`Stopped`**: pure data-structure editing. Broken intermediate graphs are fine;
  nothing validates until `Ready`.
- **`Ready`**: edits trigger incremental re-negotiation, latency recompute, and
  pool/ring re-registration for the affected subgraph; errors hit the bus immediately.
  The cheap place for structural change — resources exist, nothing flows.
- **`Playing`** — three classes, three mechanisms:
  1. **Properties**: declared *live* or *structural* in the element descriptor. Live
     properties apply at batch boundaries via a `Ctx` mailbox — the safe point already
     exists, so no hot-path locks; continuous knobs (volume, thresholds) can be
     per-batch-read atomics. Structural properties (file path, device) restart the
     *element*, never the pipeline.
  2. **Topology**: the safe-point relink operations above. These may recompute thread
     groups — spawn/retire threads, migrate elements between reactors, re-register
     pools — all pipeline bookkeeping, invisible to elements and app.
  3. **Policy**: latency budget, queue capacity/leakiness, pinning — recomputed and
     applied at safe points, old/new values reported on the bus.
- The unifying primitive: a change that can't be absorbed at one batch boundary runs a
  **subgraph-local micro-transition** — drain the affected branch, apply, re-preroll
  it — while the rest of the graph keeps playing. GStreamer makes applications
  choreograph this by hand and everyone gets it wrong; here it is the internal
  machinery every runtime mutation compiles down to.
- Honest caveat: "any state, any change" is the API guarantee, not a free lunch. A
  `Playing`-state relink of an io_uring source implies kernel cancels,
  re-registration, and re-preroll — correct and invisible, not instantaneous. The
  latency recompute and bus notification keep the cost observable.

### Joining and leaving a playing pipeline

Adding a branch (say, a networked source) while `Playing` is one atomic operation
that the framework stages internally. There is no per-element state for the app to
synchronize — gst's `sync_state_with_parent` / base-time / pad-offset choreography
is structural here, not homework:

1. **Build offline**: negotiate the new subgraph against the fixed formats at its
   junction points, allocate pools, register with the assigned reactor,
   spawn/extend thread groups, `start()` elements. The running graph is untouched.
2. **Preroll in isolation**: the branch runs until its first batches are held at
   the junction queue. Connection setup, initial buffering, and jitter estimation
   happen off to the side.
3. **Align the timeline at splice**: the framework stamps the branch with a
   `Segment` whose base is current running time plus the branch's path latency —
   its first buffer renders "now", not "late by T". Element code never computes
   offsets.
4. **Splice at a safe point**: one batch boundary at the junction; downstream
   simply sees a new upstream with correctly mapped running times.
5. **Recompute latency**: if the new live path needs more than sinks currently
   compensate, standing policy decides — grow pipeline latency (one coordinated
   deadline shift, old/new reported) or hold the budget and run the branch leaky.
   Decided and reported at join time, never discovered as stutter.

Failure in steps 1–2 rolls back completely — the branch never existed; error on
the bus. Removal mirrors the join: drain at a safe point, EOS, cancel-by-identity
in the kernel, unregister, retire threads, recompute latency. The join never
re-masters the clock: a late-arriving audio device *slaves* (§clocking); clock
stability while `Playing` is absolute.

### Supervision: failure is a runtime event, not a verdict

For 24/7 pipelines (streaming servers, kiosks, broadcast), element failure is
weather. Handling it is the runtime-configuration machinery, connected:

- **Errors have declared scope**: classified (by the element, with a framework
  default) as `Buffer` (drop and continue — a bit-flipped frame), `Branch` (this
  chain is dead — camera unplugged), or `Pipeline` (unrecoverable).
- On a `Branch` error the framework isolates deterministically: flush the branch,
  cancel its kernel ops, stop its elements, and seal the junction with a GAP
  heartbeat so downstream aggregation keeps working — the mix continues minus one
  input. `BranchSealed` goes on the bus. The rest of the graph never sees more
  than a missing input.
- **Restart is a policy object, not app boilerplate**: a supervisor (shipped as a
  library, like the autoplug controller) watches `BranchSealed` and re-joins
  replacement subgraphs with exponential backoff via the standard join machinery.
  Offline preroll means a flapping camera reconnects glitch-free or not at all —
  never half-connected.
- **Watchdogs are built in**: per-element progress deadlines (input available but
  no batch processed in T → bus warning with that element's counters attached)
  plus the passive-budget enforcement (§scheduling) catch wedged-not-dead states,
  which are worse than crashes.

### The pipeline API (sketch)

```rust
pub enum State { Stopped, Ready, Playing }

impl Pipeline {
    pub fn new() -> Self;               // no registry — that's a parse-layer concern

    // Topology — legal in every state (§runtime configuration).
    // Elements arrive already constructed: pipeline.add(HttpSrc::new(url)).
    pub fn add(&mut self, element: impl Element + 'static) -> ElementId;
    pub fn add_subgraph(&mut self, tpl: &Template) -> Result<GroupId, Error>;
    pub fn link(&mut self, src: (ElementId, &str), sink: (ElementId, &str))
        -> Result<LinkId, Error>;
    pub fn link_filtered(&mut self, src: (ElementId, &str), sink: (ElementId, &str),
        filter: &[FieldConstraint]) -> Result<LinkId, Error>;
    pub fn relink(&mut self, link: LinkId, new_sink: (ElementId, &str)) -> Result<(), Error>;
    pub fn remove(&mut self, el: ElementId) -> Result<(), Error>;   // drains if Playing
    pub fn remove_group(&mut self, g: GroupId) -> Result<(), Error>;

    // State and clock. Pause is a clock op, not a state.
    pub fn set_state(&mut self, s: State) -> Result<(), Error>;     // completion on bus
    pub fn pause(&mut self);
    pub fn resume(&mut self);
    pub fn seek(&mut self, to: Timestamp) -> Result<(), Error>;
    pub fn step(&mut self) -> Result<(), Error>;                    // debug single-step

    // Properties and policy.
    pub fn set(&mut self, el: ElementId, prop: &str, v: Value) -> Result<(), Error>;
    pub fn set_latency_budget(&mut self, budget: Timestamp);

    // Observability.
    pub fn bus(&self) -> &Bus;
    pub fn dump_dot(&self) -> String;
    pub fn latency_report(&self) -> LatencyReport;    // per path, per element
    pub fn counters(&self, el: ElementId) -> CounterSnapshot;
}
```

### Debuggability and profiling

- **Graph dump**: topology → Graphviz dot at any moment, with queue fill levels and
  negotiated formats on edges. Programmatic, not just env-var triggered.
- **Per-element counters** in the framework wrapper (not element code): buffers/bytes
  in/out, queue high-water marks, processing vs. waiting time, batch-size and latency
  histograms. Stored SoA (one array per counter kind across all elements) so a stats
  snapshot is a few contiguous reads. Plain atomics, updated once per batch,
  near-zero cost when unread.
- **Own logging in core** (no `log`/`tracing` dep): per-element targets,
  runtime-adjustable levels, `STREAMCRAFT_DEBUG=element:level` env syntax like
  `GST_DEBUG`. A helper crate can bridge to `tracing`/perfetto spans per buffer per
  element for those who want it.
- **Deterministic replay**: record source output, replay it through the pipeline in
  `step()` mode — turns "glitches after 3 hours" into a reproducible unit test.

### Logging cont'd

<experimental>
Should the `Ctx` take care of logging? Logging should be super light weight. Should logging be
similar to other buffers where we can have sinks (e.g. logsink) that accepts logs and the framework
provides insanely light weight logging facilities? E.g. a filelogsink could use the same IO things
as other elements? Food for thought. This section is still WIP so please fill in or push back on
this.
</experimental>

### Introspection protocol and scraft-scope (the inspector)

Everything the debuggability sections describe — dumps, counters, latency reports,
bus traffic, logs — is exposed through one **introspection protocol**: a compact
binary protocol (length-prefixed POD frames, naturally — same vocabulary as the
counters and bus types) served over a Unix socket / local TCP, feature-gated in
core. Serving it costs nothing until a client connects, and observation stays
observation: reads are snapshots of data the pipeline already maintains, never
locks on streaming paths.

**scraft-scope** is the flagship client: a GUI written in Slint (Rust-native,
lightweight — fits the ethos; no GTK/Electron) that can run two ways:

- **Attach**: a standalone binary connecting to any running streamcraft app by
  socket — zero code in the target beyond the feature flag.
- **Embed**: `Scope::spawn(&pipeline)` runs the same UI in-process on its own
  thread for dev builds — one line in `main`, same protocol underneath, so the two
  modes can't drift apart.

What it shows, all live:

- **Graph view**: the topology rendered (same data as `dump_dot`), with per-edge
  negotiated formats, queue fill bars, batch-size and throughput overlays; groups
  cluster visually; relinks animate as they happen (topology bus messages drive it).
- **Latency panel**: the per-path budget breakdown from `latency_report()` —
  where every microsecond goes, per element, with live histograms and the QoS
  lateness stream.
- **Events and logs**: the bus feed and the log stream, filterable per element
  (the interned-id vocabulary makes filtering cheap and exact, not regex-over-text).
- **Debugging**: pause (clock freeze), single-`step()`, per-element counter
  inspection, live property editing (the `PropDesc` table makes the UI free),
  buffer metadata peeking (sample a batch's POD rows — pts/flags/meta — without
  copying payloads), watchdog and drop alerts surfaced as annotations on the graph.
- **MCP server**: the same protocol exposed as Model Context Protocol tools —
  `get_topology`, `get_latency_report`, `get_counters`, `tail_bus`, `tail_logs`,
  `set_property`, `pause`/`step`, `dump_dot` — so an agent can attach to a live
  pipeline, diagnose "why is this path late", and propose the fix from real data.
  Debugging a streamcraft app with an agent should be *better* than with printf,
  because the agent gets structured truth instead of log soup.

The protocol is versioned and documented from day one — it is also how CI's stress
runs capture state on failure, so the inspector and the test infrastructure are the
same plumbing, kept honest by shared use.

### Testing, benchmarks, stress, fuzzing

All four are written *continuously alongside the framework* — every primitive and
feature lands with its tests and benchmarks in the same change, not as a later phase.
And all four must be extremely fast; a slow suite stops being run, and then it stops
being written.

- **Element harness**: wraps one element with no threads and a mock clock; push
  buffers/events, assert outputs. The passive/inline scheduling model makes the harness
  nearly free — it *is* the inline caller.

  ```rust
  let mut h = Harness::new(Volume::new(), &VOLUME_DESC);   // MockClock built in
  h.fix_format("sink", audio_f32(48_000, 2));
  h.push("sink", batch);                 // runs process() inline — no threads
  h.clock().advance(Timestamp::ms(10));  // sync logic tested in microseconds of wall time
  let out = h.pull("src").unwrap();
  assert_eq!(out.pts[0], Timestamp::ZERO);
  ```

- Golden pipeline tests: seedable testsrc → transforms → assertion sink (checksums,
  timestamps, buffer counts).
- **Fast at an extreme test count** — compile *and* run:
  - One integration-test binary per crate (one `tests/all.rs` including modules), not
    one per file: N test binaries means N link steps, and linking dominates Rust test
    build time. The dependency-free core keeps the edit-compile-test loop in seconds.
  - Tests are **data, not code**, wherever possible: table-driven cases and golden
    files feeding a few generic runners. Ten thousand cases compile as one function
    plus data, not ten thousand `#[test]` bodies. No proc-macro test frameworks —
    rustc time is a budget too.
  - Nothing ever sleeps: MockClock everywhere and the threadless harness mean wall
    time ≈ CPU time even for sync/latency/timeout tests. Anything needing real time,
    real devices, or the network is quarantined in a separate rarely-rebuilt crate.
  - CI tracks suite compile+run time like a benchmark and fails on regression — a
    slow suite is treated as a bug in itself.
- **Benchmarks, written constantly**: every primitive gets its microbench the day it
  exists (queue hop, pool recycle, clock read, solve time, batch push), every feature
  its macrobench (buffers/sec through an n-element chain, inline vs. queued hop cost,
  end-to-end latency, relink churn) — tracked against equivalent GStreamer pipelines,
  failing CI on regression. The whole bench suite stays runnable locally in seconds,
  because a benchmark you don't run before committing is documentation, not a
  benchmark. "Performance is the primary goal" needs numbers, from week one.
- **Stress tests are deterministic and time-compressed**: seeded schedules driving
  hours of virtual clock time in seconds of wall time — relink churn under load,
  flush storms, pool exhaustion, credit starvation, EOS/seek races. A failing seed is
  a reproducible bug report.
- **Fuzzing from day one, in-process and fast**: the fuzz-friendly surfaces are exactly
  the POD ones — the negotiation solver (offer pairs → must never panic, intersection
  must be sound), ring-buffer operation sequences, event orderings into the harness,
  segment/timestamp arithmetic, parse-launch strings. No IO and no deps means
  thousands of execs/sec/core; corpora are checked in, and each fuzz target doubles as
  a table-driven regression test replaying the corpus in normal `cargo test`.
- `loom` (dev-dep only) for queue/clock/pool primitives; miri + sanitizers in CI for
  the one unsafe module. These are CI passes, not part of the default fast loop.

### Plugins

- **A plugin is just a crate.** The primary consumption path is `use` plus typed
  construction: `use scraft_http::HttpSrc;` → `pipeline.add(HttpSrc::new(url))`.
  No registration, no factories, no strings — rustc is the registry, with dead-code
  elimination and compile-checked config for free.
- **`Registry` is the opt-in second layer**, existing only where "element by name" is
  genuinely the point: a plugin crate may additionally expose
  `fn register(&mut Registry)` with its descriptors and `make_default` factories,
  powering `parse()` (scraft-launch, quick tests, bug-report one-liners) and, later,
  dynamically loaded plugins. Applications that don't use string construction never
  touch it.

  ```rust
  impl Registry {
      pub fn register(&mut self, desc: &'static ElementDesc);
      pub fn get(&self, name: &str) -> Option<&ElementDesc>;
      pub fn parse(&self, into: &mut Pipeline, launch: &str) -> Result<GroupId, ParseError>;
  }
  ```
- **Dynamic loading later, feature-gated**: Rust has no stable ABI, so `dlopen` plugins
  need a small versioned C-ABI shim (`extern "C" fn streamcraft_plugin_register`).
  Build it only when a real out-of-tree consumer exists — it's a maintenance tax.
- Registry enables **string pipeline construction**
  (`parse("filesrc path=x ! decode ! sink")`) — a gst-launch equivalent is cheap once
  the registry exists and invaluable for debugging and bug reports.

## Writing elements: patterns and utilities

The element-author experience is a feature. Rules first, then the toolbox:

- **The rules** (enforced where possible): no allocation after `start()` (debug
  builds assert); passive elements never block (watchdog-enforced); scratch via
  `ctx.scratch()` only; state is plain fields on `self`, because exactly one thread
  ever calls you; errors are returned, never logged-and-swallowed.
- **A transform is ~30 lines**: read the negotiated format once in `start()`, loop
  over input rows, write through `OutBatch::reserve()` for in-place construction.
  The harness tests it without a pipeline, and the SoA layout makes the natural
  loop the vectorizable one — the fast way is the obvious way.
- **`ByteAdapter`** (helper crate): the parser's companion. Push refcounted
  memories in; view a contiguous byte window across them (zero-copy when the window
  fits one memory, one scratch-arena splice when it spans); consume with sub-slice
  buffers that keep source memories alive. Every parser needs this; five subtly
  wrong per-element copies is how frameworks rot.
- **Sub-buffer slicing is free**: `Memory` is a refcounted (base, offset, len)
  view, so a demuxer slicing one 1 MB read into 300 packets does 300 refcount
  bumps and zero copies.
- **Typed format views** (helper crates): `VideoFrameRef::new(&buf, &format)`
  validates once, then exposes planes/strides; audio views expose channel spans.
  Views borrow — misuse is a compile error — and add zero cost over raw offset
  math.
- **Reference elements are kept exemplary**: a documented, benchmarked source,
  transform, sink, and aggregator live in the repo as templates. Element authors
  copy the nearest example, so the nearest example must be perfect.

## Non-goals (so the goals stay reachable)

- No async runtime, no work-stealing executor, no futures in core. The static,
  printable schedule is a feature.
- No GObject-style dynamic type/property/signal system. Types are Rust types.
- No fully general caps algebra — the closed value set plus the validation-callback
  escape hatch is a hard, deliberate limit.
- No dynamic plugin ABI until a real out-of-tree consumer needs it.
- No libav/FFmpeg binding in the framework — too heavyweight. Open formats are
  hand-written (§first-party codecs); a system `ffmpeg` binary is at most a dev-time
  test oracle run as a subprocess, never linked in and never a dependency.
- No Windows in v1 — but the reactor contract is completion-shaped precisely so
  IOCP slots in without redesign; macOS/BSD ride the kqueue fallback tier.
- No third-party callbacks on streaming threads. Ever. (The one exception is
  inverted: *we* run inside foreign RT audio callbacks, under the wait-free rules
  of §pull-mode.)

## First-party codecs and containers

The open formats get hand-written, first-party implementations — encoders *and*
decoders: **FLAC, Opus, VP8, VP9, AV1**, and the containers **Matroska/WebM, Ogg**
(EBML as a shared foundation crate), plus WAV/MP4-demux as table stakes. There is no
libav/FFmpeg binding: an FFI wrapper is too heavyweight, and it would drag a foreign
allocator, threading model, and timestamp semantics in behind a wall our batches
can't cross. Hand-written codecs are native citizens: SoA batches in, pool memory out, scratch arenas for
transforms, SIMD via `std::arch` over full batches, zero adapter impedance. This
is also where "performance is the primary goal" gets proven at the highest level —
a decoder whose inner loops were designed for this framework's memory model, not
adapted to it.

Conventions, mandatory for every codec/container crate:

- **The spec lives in the tree**: `sc-flac/spec/rfc9639.txt`, `sc-opus/spec/rfc6716.txt`,
  `sc-vp8/spec/rfc6386.txt`, `sc-mkv/spec/rfc9559.txt` + `rfc8794.txt` (EBML),
  `sc-ogg/spec/rfc3533.txt`, and the AV1/VP9 bitstream PDFs. Code cross-references
  spec sections in comments (`// §4.5.2: residual coding, method 1`), so review
  means reading the implementation *against* the normative text, not against
  someone's blog post. Spec errata and interpretation decisions get documented in
  `spec/NOTES.md` per crate.
- **Conformance vectors in CI**: the official test vector suites (FLAC test files,
  Opus test vectors, AV1 argon suite, Matroska conformance files) run in the normal
  test flow — table-driven, so thousands of vectors are data, not code (§testing).
- **Cross-checked against a reference decode**: conformance vectors are primary; as
  an extra net, a decoder can be diffed against a reference produced *out of process*
  by a system `ffmpeg`/reference binary — a dev-only oracle invoked as a subprocess,
  never linked in, never a dependency. Divergence is a failing test with the
  offending bitstream minimized and archived.
- **Decoders parse untrusted input**: fuzzing is not optional here — every parser
  and decoder has structure-aware fuzz targets from its first commit, and the
  no-alloc/no-unsafe-outside-audited-modules rules apply doubly. A crash on any
  bitstream, however malformed, is a P0.
- **Encoders declare their quality/speed ladder as data** (`PropDesc` presets), and
  every rung is benchmarked for both speed *and* quality (PSNR/SSIM vs. reference
  encoders on a pinned clip set) so regressions in either direction fail CI.
- **Build order follows tractability**: EBML+Matroska/Ogg demux (parsers, weeks) →
  FLAC codec (well-specified, closed-form) → mux support → VP8 → Opus → VP9 → AV1
  (each a serious project; AV1 decode is dav1d-scale and is allowed to take the
  time it takes). Until a given rung exists, that format simply isn't supported —
  the framework never blocks on it, and sink/clock/scheduler bring-up uses raw
  (uncompressed) formats from `testsrc` and rawvideo/rawaudio files, which need no
  decoder at all.

## Milestone applications

Each milestone is a small real program (an `examples/` binary), ordered so that every
one forces a new slice of the framework into existence end-to-end. A milestone counts
as done only when its pass criteria hold — and each one becomes a permanent
integration test and benchmark the day it works.

1. **File copy** — `filesrc ! filesink`.
   Forces: buffer/pool, batches, the scheduler's minimal loop, EOS, bus.
   Pass: byte-identical output; zero allocations in steady state (pool counters
   prove it); throughput within ~2× of `cp`.
2. **Download a video file** — `httpsrc url=… ! filesink`.  Forces: the reactor (io_uring source +
   sink sharing one ring), submission credits/backpressure, cancellation (Ctrl-C mid-download must
   cancel cleanly), error reporting on the bus (DNS failure, 404, reset).  Forces nothing
   clock-related — deliberately: network before clocks.  Pass: saturates a fast local link with
   near-zero CPU; clean abort at any moment.  The HTTP source should be completely custom and
   dependency free and all the parser and stuff should be really fast and tailored to SC's
   architecture. Since this is the first spec SC implements make sure to download the RFCs
   used. Start with HTTP 1.1? This should be a separate plugin since we'll eventually add TLS which
   we need a library for... sadly :(
3. **Convert wav to flac** — `filesrc ! wavparse ! flacenc ! filesink`.
   Forces: format negotiation (solver picks sample format/rate across the chain),
   the `-audio` helper crate, non-live throughput mode (no clock waits, maximal
   batches), and the first hand-written codec: the `sc-flac` encoder
   (§first-party codecs), tags preserved via the `Tag` event.
   Pass: output validates with `flac -t`; faster than `ffmpeg` doing the same
   conversion, single-threaded, and the win must come from batching + zero-copy
   (verified by the counters, not vibes).
4. **Play an audio file** — `filesrc ! wavparse ! alsasink`.
   Forces: the clock system for real — audio device as master clock, sink
   synchronization, preroll, pause/resume (clock freeze), latency computation and
   report, underrun handling.
   Pass: hours of playback with zero underruns and no drift; pause/resume is
   glitch-free; `latency_report()` matches measured output delay.
   Also make a pipewiresink.
5. **Play a video file (no audio)** — `filesrc ! mkvdemux ! vp8dec ! videosink`
   (raw-video bring-up first: `filesrc ! rawvideoparse ! videosink`, which needs no
   decoder and isolates the sink/clock path).
   Forces: the first hand-written video decoder (`sc-vp8`) producing frames straight
   into pool memory, dynamic pads from the demuxer, video sink rendering on clock
   deadlines, QoS (late-frame dropping visible as bus observations).
   Pass: smooth playback at native rate; artificial CPU starvation degrades via
   measured QoS drops, never via freeze or runaway memory.
6. **Play audio + video** — the classic, over WebM (VP9 + Opus): demuxer fanning out
   to both hand-written decode branches, audio-mastered clock, both sinks in sync.
   Forces: multi-path latency alignment (video waits for the audio path's latency),
   seek/flush across branches, format changes mid-stream.
   Pass: lip sync within one video frame over hours (measurable with a test clip of
   beeps+flashes and a loopback capture); seeking lands frame-accurately with both
   branches consistent; this exact example is the doc's front-page code sample.
7. **Live source switch** — audio+video playing, then `pipeline.add_subgraph()` a
   second (networked) source and relink the sinks to it mid-`Playing`, old branch
   drained and removed.
   Forces: the whole §runtime-configuration machinery — offline preroll, timeline
   alignment at splice, latency recompute, kernel cancel, rollback on join failure.
   Pass: the switch is glitch-free on the audio path (no gap, no click), and a
   stress loop switching every few hundred ms survives sanitizers overnight.
8. **`scraft-launch`** — the gst-launch equivalent over `Registry::parse`, with
   `--dump-dot`, `--latency-report`, `--counters` flags.
   Forces: registry, parse syntax, and the observability APIs — and from then on,
   every bug report and benchmark in the project is a one-liner.
9. **Record a webcam to WebM** — `v4l2src ! vp9enc ! webmmux ! filesink`; Ctrl-C
   finalizes.
   Forces: device source (V4L2 is completion-shaped), muxer aggregation
   (`All` by dts), drain-vs-abort shutdown, GAP handling when the camera stalls.
   Pass: the file is *always* playable no matter when you kill it; a stalled
   camera never wedges the muxer.
10. **Multi-room audio** — one source, two networked sinks in sync.
    Forces: network clock, clock slaving at both sinks, drift correction.
    Pass: measured inter-room skew < 1 ms over hours (loopback capture) while both
    device clocks drift freely.
11. **scraft-scope attached to milestone 6** — live graph, latency panel, log/event
    feeds, property editing, and an MCP agent diagnosing an injected latency bug.
    Forces: the introspection protocol end-to-end, the Slint UI, the MCP server.
    Pass: attaching under full load changes the pinned benchmarks by < 1%.

Milestones 1–3 need no clock at all; 4 is where clocking lands; 6 is the
"better-than-GStreamer-or-bust" gate; 7 proves the headline dynamism claim; 9
proves shutdown correctness; 10 proves distributed clocking; 11 proves the
observability story costs nothing.

## Order of attack

1. `Buffer` + pool (allocator contract, alignment, registration hooks) + `Batch` SoA
   layout (including the meta column and `SyncPoint` slot) + `FormatId`/blob refs +
   interning — the vocabulary everything else speaks. Batching, metadata, and
   pluggable memory cannot be retrofitted; they go in first.
2. `ClockWait` + batch-transferring SPSC ring queue (with exported waitable handles) +
   mock clock — the three primitives, loom-tested in isolation before any pipeline
   exists.
3. Element trait (batch-based `process`, `InputPolicy` aggregation contract, scratch
   arenas in `Ctx`) + pads + pipeline-owned topology + link-time negotiation.
4. Scheduler: thread groups, `play()`/`step()`, backpressure, pinning.
5. Events, bus, state machine; then sink synchronization against the clock.
6. Harness + golden tests + benchmarks (earlier if anything above gets hairy).
7. Seeking/flush, dynamic relinking, graph dump, counters.
8. Registry, parse-launch, templates/groups/autoplug controller, helper crates
   (`-video`, `-audio`).
