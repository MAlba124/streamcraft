# streamcraft — roadmap / next stages

A hand-off for the next agent. Priorities of the project: **performance #1, perfect
latency #2.** Core (`streamcraft-core`) stays **zero-dependency** and
`#![deny(unsafe_code)]` (except the audited `memory`/`ring` modules). Everything is
hand-written; the only external deps live in isolated plugin crates (`sc-pipewire` binds
libpipewire — a device backend is the one "buy, don't build"). The design doc is
`streamcraft.md`.

## Where things stand (2026-07-24)

> **Update — session 2 (2026-07-24):** landed **Stage 1 (dynamic pads: branching DAG
> scheduler + preroll)**, **Stage 2 (Clock, wired)**, **Stage 3 (FLAC-in-Ogg)**, **Stage
> 4.1 (downstream re-validation)** + **4.3 (AudioConvert dynamic consumer)**, and the
> parallelizable **pipewire RT-safe ring** and **audioresample**. Workspace is **265 tests
> green**. Remaining: **Stage 4.2** (per-buffer mid-batch boundaries + intra-group passive
> FormatChange), **Stage 5 (registry/parse-launch)**, a real multi-stream `OggDemux` + a
> `multiqueue`, plus TLS-for-http, the perf CI gate / fenceless ring / io_uring, and
> spec-writing.

> **Update — session 3 (2026-07-24):** landed **dynamic element properties** (spec:
> Dynamic element properties) and **taps tier 1** (spec: Taps). New core `props` module:
> a per-element **seqlock `PropTable` mailbox** (the spec-experimental lean, chosen);
> `Pipeline::set` and a cloneable **`PropHandle`** validate on the app's thread against
> `PropDesc.allowed` via `Constraint::accepts`; the scheduler polls a dirty mask at the
> batch boundary (one relaxed load idle) and delivers **`Event::PropChanged`**; elements
> pull-read via `ctx.prop(name)`. Structural (`live: false`) sets while playing are
> rejected loudly — they need the element-restart micro-transition (backlog). Taps:
> counters are created at `add()` and **stable across runs**, extended with
> `batches_in/out` and downstream-ring `queue_high_water`; a cloneable **`TapHandle`**
> gives `snapshot(el)` + running-time `now()` (shared base) so observers compute
> bitrate = Δbytes/Δnow themselves (counters stay **per element**; per-pad still open).
> Perf fixes found in review: `Batch::pop_front` was four `Vec::remove(0)`s — O(n²)
> per batch drain, paid by every element via `Inputs::pop` — now an O(1) drain cursor
> (`Batch` columns went private; `Memory::take` moves payloads out O(1)); idle
> scheduler passes no longer do counter RMWs. Workspace **319 tests green**.

Working, ~261 tests green (`nix develop --command cargo test --workspace`, exit 0):

- **Core**: opaque `Buffer` + pool-backed `Memory`; SoA `Batch` that also carries in-band
  `events`; interned negotiation solver with **two layers** — static link-time
  (`intersect`/`negotiate`) *and* runtime **dynamic caps** (`ctx.announce_format` →
  `Event::FormatChange` on the batch → pipeline re-fixates the peer + delivers `event()`);
  a shared `format::Vocabulary` so elements read their negotiated format **by name**
  (`ctx.field_id`/`value_name`); thread-group scheduler with lock-free SPSC rings
  (loom-checked); pluggable `Reactor` (sync + hand-rolled io_uring); logging (POD records
  → per-element ring → stderr drain, gated by `STREAMCRAFT_DEBUG`); per-element counters;
  `dump_dot`; cooperative cancellation (`StopHandle`); **EOS delivered to elements'
  `event()` in chain order** (sinks drain, muxers flush).
- **Elements** (`streamcraft-elements`): filesrc/filesink (reactor-native, seekable),
  passthrough, testsrc/testsink.
- **Plugin crates**: `sc-http` (custom HTTP/1.1 source), `sc-flac` (encoder + decoder +
  incremental `StreamDecoder`, `FlacEnc`/`FlacDec` elements, RFC 9639 in-tree), `sc-ogg`
  (page reader/writer + single-stream `OggMux`/`OggDemux`, RFC 3533), `streamcraft-audio`
  (`audio/raw` vocab, `AudioFrameRef`, `WavParse`, `AudioConvert` + conversion lib),
  `sc-pipewire` (`PipeWireAudioSink`).
- **Milestones done**: file copy, HTTP download, wav→flac, **play an audio file**
  (`filesrc ! flacdec ! pipewireaudiosink`, paced by the device via backpressure).

The clock is now **wired** (Stage 2): `Pipeline` owns `Arc<dyn Clock>` (default
`InstantClock`; `set_clock` installs a `MockClock`), `run()` samples `base_time`, and
`ctx.now()`/`ctx.wait_until()` give running time + interruptible deadline waits. A timed
sink (`TimedTestSink`) renders on the clock. Follow-up: prompt hard-interrupt of a blocked
clock wait via `StopHandle` (see [[clock-wired]]).

## How to work here

- **Build/test**: `nix develop --command cargo test --workspace` (disable the Bash
  sandbox — nix needs `/nix` + network). The flake devshell sets cc+mold and overrides a
  broken global clang/wild config. Fallback: `nix-shell -p cargo rustc gcc --run 'export
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc; export RUSTFLAGS=" "; cargo test ...'`.
- **Parallelism**: fan independent work out to subagents in **git worktrees** off
  `rearchitecture`; each verifies HEAD, stays in a disjoint file scope, commits with a
  `Co-Authored-By` trailer; merge with `--no-ff` and delete the worktree/branch. Core
  changes are delicate — keep them to one agent (usually the main one).
- **Style**: match surrounding code exactly; tests + benches land with the code; codecs
  check their spec/RFC into the crate's `spec/`.
- `streamcraft.md` has in-flight user WIP (Taps, Dynamic element properties) — don't
  clobber it; add new spec sections around it.

---

## Stage 1 — Dynamic pads (the "god bin" keystone) · HIGH value · HIGH effort · core — ✅ DONE

> Landed in 3 commits: per-pad `Ctx.outs` → **branching DAG scheduler** (`topo_order`/
> `compute_groups`/per-edge rings; `elements/tests/branching.rs`) → **dynamic pads +
> preroll** (`Element::preroll` + `ctx.add_pad` + `Pipeline::preroll` returning the added
> pads; `elements/tests/dynamic_pads.rs`: a demux exposes 2 runtime src pads → 2 sinks,
> `dump_dot` shows the fan-out). Model is **settle-at-preroll, then freeze**. Follow-ups:
> a real multi-stream `OggDemux`, a `multiqueue` to avoid cross-branch head-of-line
> stalls, and registry-driven auto-plug (Stage 5). See `memory/dynamic-pads-design.md`.


**Why now**: dynamic caps just landed and the user is explicitly after `decodebin3`/
`parsebin`-style auto-plugging. Dynamic caps (runtime formats) is done; **dynamic pads**
(one pad per discovered stream) + a **registry** + **runtime relink** compose into god
bins. This is the single biggest unlock.

**Key design call**: don't fight the "static, printable schedule is a feature" ethos.
Model it as a **preroll/setup phase** — a demuxer reads headers, discovers streams, the
bin adds pads and auto-plugs decoders, the schedule is (re)computed — **then frozen** for
the streaming phase. Truly-dynamic mid-stream topology is out of scope; "settles during
preroll, then static" is the target.

**Approach / files**:
- `core/src/element.rs`: `PadDesc.dynamic` exists but is unused. Add **pad templates** —
  an element declares template (dynamic) pads that are *instantiated* at runtime.
- `core/src/ctx.rs`: let an element request a new src pad at runtime (`ctx.add_pad(...)`),
  and post a **pad-added** notification.
- `core/src/bus.rs`: a `PadAdded { element, pad }` message so the app/bin can link it.
- `core/src/pipeline.rs`: runtime `link()` during setup; re-run `compute_groups` /
  rebuild rings after the topology settles. The scheduler currently computes groups/rings
  once in `run()` — the hard part is letting topology grow before the streaming loop
  starts (a `preroll()` that runs sources until pads settle, then builds the schedule).
- Driving use case: **multi-stream `OggDemux`** (one src pad per serial) → two sinks.

**Acceptance**: a demuxer exposing 2 runtime src pads routes each logical stream to its
own downstream chain; `dump_dot` shows the settled topology; existing static pipelines
unaffected.

**Risks**: the scheduler's group/ring model assumes static topology at `run()`. This is
the crux — budget for a `preroll` phase and a schedule rebuild. Start minimal.

## Stage 2 — Clock & synchronization (the "perfect latency" pillar) · HIGH · core — ✅ DONE

**Why now**: latency is priority #2 and nothing uses the clock yet; every real-time and
A/V-sync feature depends on it. Today audio plays because the sink blocks on the device
(backpressure) — correct for audio-only, insufficient for A/V.

**Approach / files**:
- `core/src/pipeline.rs`: the pipeline owns a `Clock` and a `base_time`; a sink can be the
  clock **provider** (e.g. `PipeWireAudioSink` slaves the pipeline clock to the device).
- `core/src/ctx.rs`: `ctx.now()` returns running time (`clock.now() - base_time`);
  `ctx.wait_until(ts)` parks on `ClockWait` until a deadline (interruptible by stop).
- A **timed sink**: renders each buffer at `base_time + pts`, blocking via `ClockWait`.
- `core/src/clock.rs` already has the primitives — wire, don't rewrite.

**Acceptance**: a timed testsink rendering on deadlines with `MockClock` runs hours of
virtual time in milliseconds of wall time (deterministic, no real sleeps); a two-sink
graph stays in sync against a shared clock.

## Stage 3 — FLAC-in-Ogg playback · MED · mostly isolated · good subagent — ✅ DONE

> `OggFlacDeframe` (flac crate) strips the 9-byte mapping-header prefix from packet 0 and
> forwards packets verbatim → native FLAC bytes for `FlacDec`. Bit-exact round-trip via
> `OggMux!OggDemux` in `flac/tests/oggflac_roundtrip.rs`; mapping in `flac/spec/`.


**Why now**: proves container + codec + device end-to-end using pieces we already have,
and is far more achievable than a from-scratch Opus decoder. Milestone:
`filesrc ! oggdemux ! oggflacdeframe ! flacdec ! pipewireaudiosink` playing a `.oga`/
`.ogg` FLAC file.

**Approach**: a small **FLAC-in-Ogg mapping** element in `sc-ogg` or `sc-flac` — packet 0
of the logical stream is the `0x7F "FLAC"` mapping header carrying STREAMINFO; subsequent
packets are FLAC frames. The deframer reconstructs a native FLAC byte stream (`fLaC` +
STREAMINFO metadata block + frames) that `FlacDec`'s `StreamDecoder` already handles, and
announces `audio/raw` (dynamic caps, exactly like `flacdec`).

**Acceptance**: round-trip (`flacenc`-style → mux-to-ogg-flac → deframe → decode) is
bit-exact; a real `.oga` plays.

## Stage 4 — Harden dynamic caps · MED · core — partly done

Three loose ends from Stage-0 dynamic caps (see `memory/runtime-caps-gap.md`):
1. ✅ **Downstream re-validation** — DONE. `deliver_events` re-validates an announced
   `FixedFormat` against the peer sink pad's offers (`Vocabulary::offer_admits`): if the
   peer speaks the announced family but no offer admits the values, it's a loud
   `Error::Element` on the bus; a family the peer doesn't offer (a `bytes`-bridge) installs
   tolerantly. Tested in `elements/tests/dynamic_caps.rs`.
2. ⬜ **Per-buffer mid-batch boundaries**: `FormatChange` is delivered before *all* of a
   batch's buffers — correct for announce-once-at-start, wrong for a change partway
   through one batch. Split the batch at the event position (needs `events` to carry a
   batch position). Also open: **intra-group passive `FormatChange`** — a co-grouped
   passive consumer never sees an upstream's runtime change (only group heads re-fixate).
3. ✅ **`AudioConvert`'s dynamic consumer path** — DONE. `AudioConvert::new(target)` infers
   the full input format by name off the negotiated caps (`learn_from_sink`); no
   `with_input` needed. (`audioresample` landed alongside.)

## Stage 5 — Registry + parse-launch · MED

`use` + typed construction stays primary; add the opt-in `Registry` (`register`/`get`/
`parse`) the spec describes so `parse("filesrc ! oggdemux ! …")` works. This is the
substrate for **auto-plugging** (find a decoder whose sink offer accepts a discovered
format) — i.e. the "god bin" from Stage 1 needs it. Also invaluable for bug-report
one-liners and debugging.

---

## Parallelizable right now (good subagent tasks, isolated scopes)

- ✅ **Sink RT-safety** (`pipewire/`): DONE — the `Mutex<VecDeque<u8>>` hand-off is now a
  byte-specialized lock-free SPSC ring (`pipewire/src/ring.rs`); the RT callback's `pull`
  is wait-free/alloc-free. 12 ring tests.
- ✅ **audioresample** (`audio/`): DONE — polyphase FIR with a Kaiser-windowed-sinc
  anti-aliasing low-pass (`audio/src/resample.rs` + `resample_element.rs`), arbitrary
  rational L/M, streaming state, announces its output rate via dynamic caps. Tests + bench.
- **Performance** (`core/`, `elements/`): a benchmark-regression **CI gate**, plus the two
  known soft spots — a fenceless fast-path variant of the SPSC ring (the SeqCst Dekker
  fence dominates the ~51 ns/item hop) and a deeper-queue / batched-submission io_uring
  reactor.
- **TLS/https for `sc-http`**: the reason `sc-http` is an isolated plugin (needs a TLS
  library — the one sanctioned dependency, like `sc-pipewire`).
- **Spec** (`streamcraft.md`): write the **dynamic-caps two-layer** section and an
  **audio-sink / clocking** section (work *around* the user's WIP blocks).

## Backlog (lower priority / larger)

- **Structural property sets while `Playing`**: the subgraph-local micro-transition
  (drain the element, `stop`/`start`, re-preroll) that `PropHandle` currently refuses;
  also unlocks `Pipeline::remove`/`relink` while playing.
- **Introspection protocol** (spec: scraft-scope): length-prefixed POD frames over a
  socket, feature-gated. The substrate is now uniform (CounterSnapshot/TapHandle, bus,
  log records, dump_dot) — the protocol is a thin frame-encoder over those snapshots.
  Best landed together with the latency instrumentation so the wire format is designed
  once.

- **More codecs** (hand-written, spec in-tree): Opus (`sc-opus`, RFC 6716 — large: SILK +
  CELT), Vorbis, then video (VP8/VP9/AV1).
- **Video path**: `streamcraft-video` vocab (pixel formats, `VideoFrameRef`), a video sink,
  a first video codec — the "play a video file" / "play A+V" milestones (need Stage 2
  clocking for sync).
- **More containers**: MKV/WebM, MP4.
- **scraft-scope inspector**: the introspection protocol (length-prefixed POD frames over a
  socket, feature-gated in core) + a Slint GUI + MCP tools — the logging/counters/bus
  plumbing already exists to feed it.
- **Hard-interrupt cancellation**: fold reactor-cancel + `ClockWait::interrupt` into
  `StopHandle` so a blocked device/clock wait aborts promptly.
- **Testing infra the spec wants**: in-process fuzzing (the negotiation solver, ring op
  sequences, parsers) with checked-in corpora; deterministic time-compressed stress tests
  (seeded schedules, `MockClock`); one integration-test binary per crate to keep link time
  down.

## Quick gap reference (file pointers)

- Dynamic pads: `core/src/{element,ctx,pipeline,bus}.rs` — `PadDesc.dynamic` unused.
- Clock: `core/src/clock.rs` (ready) + `core/src/ctx.rs` (`now()` returns `NONE`).
- Dynamic-caps loose ends: `core/src/pipeline.rs::deliver_events`,
  `audio/src/convert_element.rs`; see `memory/runtime-caps-gap.md`.
- Sink RT ring: `pipewire/src/sink.rs` (`Shared`).
