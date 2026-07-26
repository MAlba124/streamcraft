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

> **Update — session 3b (2026-07-24, parallel agents):** merged four agent branches +
> one main-session adoption. (1) **`video/`**: `video/raw` vocab, plane geometry +
> `VideoFrameRef`/`Mut`, `VideoTestSrc`, `RawVideoParse`, clock-driven `VideoCkSink`
> (chroma rounds **up**, ffmpeg convention). (2) **Stage 5 DONE**: `core/src/registry.rs`
> (panic-free parse), `Pipeline::{add_boxed,set_str}` (string props ride `Value::Id`),
> props+`make_default` on the common elements, `launch/` = **scraft-launch** with
> `--dump-dot/--counters/--log/--list`. (3) **`MkvDemux`**: incremental EBML reader
> (RFC 8794/9559 in `mkv/spec/`), per-track dynamic pads (constructor-supplied header
> bytes for preroll discovery — mid-pipeline elements get no preroll input), all lacing
> modes, A_FLAC→native-FLAC reconstruction feeding `FlacDec` bit-exact. (4) **`vp8/`:
> adopted `oxideav-vp8`** (pure Rust, zero-dep w/ default-features off, no unsafe, MIT,
> RFC-annotated — rubric + nativization debt in `vp8/src/lib.rs`): `Vp8Dec` with
> dynamic-caps announce, per-buffer warn-drop-resync error scope, invisible-frame
> handling. Log records now carry the element **name** (`flacdec#3`). Workspace ~350
> tests green across 67 binaries. Open follow-ups: `decode_frame_into` upstream
> (kill the per-frame plane copy), official `vp80-*` conformance vectors in CI,
> `MkvDemux` V_VP8 family naming alignment with `vp8dec` ("vp8"), display video
> sink backend decision.

> **Update — session 3c (2026-07-24): all five OxideAV video decoders wired.** Four
> parallel agents vetted-then-adopted, all WIRE verdicts, judged by source/tests
> (every crates.io blurb was stale, in both directions): **sc-h264** (0.1.7 is a
> *full* I/P/B CAVLC+CABAC decoder + encoder despite an "empty" blurb; `h264/annexb`
> in, i420 out, DPB display-order via a feed-order pts FIFO — B-frame pts caveat
> documented); **sc-h265** (0.0.9 is production-complete HEVC — Main/Main10, inter,
> SAO, byte-exact vs ffmpeg; `h265/annexb` in, i420 out, honest pts reorder window);
> **sc-av1** (0.1.16 decodes byte-exact vs dav1d; one temporal unit per buffer,
> i420/gray8 out, multi-frame TU carry); **sc-vp9** (0.0.12 is **intra-only** — the
> one real subset; element splits superframes itself; git HEAD has inter+encoder but
> is unpublished — re-evaluate on next release). Non-i420 outputs (10/12-bit,
> 4:2:2/4:4:4) decode upstream but are refused per-buffer until the video vocab +
> pool sizing grow. Shared adoption debt, per-crate lib.rs: mandatory `oxideav-core`
> (serde_json tail — only their *registry glue*, dead code on our paths; upstream
> feature-gate PR like vp8 0.2.x did), decode-into-pool (one plane copy/frame),
> conformance suites in CI, 128 KiB pool slot caps frames ~320×256 (needs a
> pipeline pool-size API). Workspace **71 test binaries green**. Video milestone
> path is now: MkvDemux → *Dec → sink; remaining for milestone 5: a display sink +
> pool sizing + MkvDemux V_VP8/V_MPEG4-ISO-AVC track family naming.

> **Update — session 4 (2026-07-24, core roadmap + 3 agents):** landed the serial core
> list in order, plus three merged agent branches. **(1) Refcounted `Memory`**: views are
> `(Arc<MemoryInner>, offset, len)` — `Clone`/`slice()` are zero-copy, `as_mut_full` is
> uniqueness-gated CoW via `pool.acquire_exact`, `take()` leaves an inner-less husk
> (O(1) batch drain preserved). **(2) Per-element output pools**:
> `Pipeline::set_element_pool(el, slot_size, slots)` — pool negotiation v1; distinct
> recycling domains per override. **(3) Latency system v1**: `latency_report()` is a DP
> over the topo order (per-sink worst path, per-element breakdown, `is_live`); `run()`
> installs each element's upstream path latency into its `Ctx` and `wait_until` folds it
> in — sinks render at `base+pts+path_latency` with zero sink code. **(3b) Clock
> providers**: `Element::provide_clock`, sinks-first selection unless `set_clock` forced
> one; `ClockWait::polling` (Poll variant) for clocks that advance where no notify
> exists; `pipewireaudiosink` provides `AudioDeviceClock` (DAC-rendered frames / rate,
> monotonic across seeks, frozen while paused) — audio-master A/V sync.
> **(4) `set_queue_capacity(el, batches)`**: per-consumer inbound ring depth (how the
> `queue` element gets real depth). **(5) Robustness**: unlinked-src-pad output is
> dropped + counted (drops now real in `CounterSnapshot`; no more dummy drop-sinks);
> `Ctx::forward_format` (Announcement carries Named *or* resolved Fixed payload) so a
> pure transport re-announces a `FormatChange` onward — dynamic caps now cross the
> `queue` end to end. **Merged agents**: threadless single-element `core::harness`
> (fix_format/push/crank/pull/announced, ~180 lines of per-test scaffolding gone);
> bounded class-aware bus (critical never dropped, droppable drop-oldest + counter) +
> `*` wildcard family + the `queue` element itself. Ring agent (fenceless fast path +
> leaky modes + benches) still in flight — leaky-queue integration follows it.
> Remaining from the sweep: per-buffer mid-batch FormatChange boundaries.

> **Update — session 4b (2026-07-24): remux.** `mkvdemux ! mkvmux` **works** for
> flac/vp8/vp9: `MkvMux::from_caps()` (now the registry default → `mkvmux` is
> name-constructible) builds its track from the upstream announcement; flac
> CodecPrivate is absorbed from the in-band native head with params parsed from
> STREAMINFO. Core grew `BufferFlags::DELTA`; `MkvDemux` stamps `KEYFRAME`/`DELTA`
> from the container keyframe bit and the mux preserves it — remuxed video stays
> seekable. **Vocabulary lesson**: a consumer that *reads* announced fields must
> declare them in its offers (declaring interns the names `build_fixed` resolves
> against; a wildcard pad admits everything but interns nothing — the announcement
> silently fails to build). Remaining for remux: av1 (`av1C` from the Sequence
> Header OBU), h264/h265 (Annex B → length-prefixed + config record, or a demux
> passthrough/no-reframe mode — cleaner for remux generally), a frame-preserving
> demux emission mode (frames bigger than a pool slot arrive split and would mux as
> several blocks), multi-track fan-in `MkvMux`, and an `mp4 → mkv` test once
> vp9/av1 passthrough lands end-to-end.

> **Update — session 4c (2026-07-24): the movie remux goal — DONE.** A real 1.6 GB
> H.264 movie remuxes `mp4 → mkv` at disk speed (1.44 GiB in 9.5 s, 151 MiB/s):
> `Mp4Demux::passthrough()` keeps NAL samples length-prefixed as stored (families
> `h264/avcc`/`h265/hvcc`, raw record as in-band head — Matroska's native shape,
> RFC 9559 §12); `MkvMux` takes the first buffer as CodecPrivate verbatim; sample
> sync bits ride as KEYFRAME/DELTA. `mp4/examples/remux_to_mkv.rs` + fixture test
> vs the Mp4Reader oracle. The two-track movie flushed out **two core bugs**:
> the single announcement *slot* (several pads announcing in one process() pass
> lost all but the last — now a queue) and the unlinked-pad drop policy only
> covering the group *tail* (a mid-group demuxer's unwatched audio pad exhausted
> its pool and stalled the run at 72 KB — now every member drops+counts unrouted
> output). Ring agent merged: cached-index SPSC fast path (~120→~90 ns/item
> two-thread), leaky DropNewest, loom models, ring_hop bench — fenceless-as-specced
> proven unsound (Dekker lost-wakeup), the SeqCst wake fence stays. Remaining for
> remux: audio (multi-track fan-in MkvMux + A_AAC/esds), av1C, `Language` tags.
> **Post-playback fixes (same day):** (1) mpv rejected every cluster — unknown-size
> (`0xFF`) Clusters are RFC-legal but ecosystem-hostile (libav accepts, mpv's own
> demuxer doesn't); the writer now stages one cluster (~a GOP) and emits it sized;
> Segment stays unknown-size. **Standing rule saved to memory: muxer output must
> pass headless mpv AND ffprobe — they disagree in practice; our own reader is
> never the only gate.** (2) No `Info\Duration` → players treat the output as a
> live stream; a remux knows the duration before the lazy header, so the demuxer
> announces a `duration` field (passthrough mode only — decode pipelines don't
> intern the name) and the muxer writes the Duration float. ffprobe: 5385.36 s;
> mpv seeks clean. `mkv/examples/check.rs` = full-file structural checker.

> **Update — session 4d (2026-07-24): perf round + observability.** simprof-guided
> allocation fix: remux **1.22M → 87.7K** calls, 3.51 GB → 459 MB (demux copied every
> sample twice to break the reader borrow — now split-borrow + emit from the reader's
> window; the pool now recycles whole `Arc<MemoryInner>`s, `Weak` pool link, disarm on
> reject). `ZERO-COPY.md` (repo root) = staged plan to true zero-copy; a Fable agent
> is on Stages 1–2 (demux slices + mux scatter, mp4/mkv only). **Latency tracing**:
> per-element power-of-2 histograms (process time / ring residency / sink wait
> overshoot), `Pipeline::set_tracing` / `STREAMCRAFT_TRACE=1`, `TapHandle::latency`,
> `streamcraft launch --trace` p50/p99/max table. **Colored logs** on TTY stderr
> (level + stable per-element hue; `NO_COLOR`/`STREAMCRAFT_LOG_COLOR` honored).

> **Update — session 4e (2026-07-24): pause transport + zero-copy merged.** Playback
> pause landed spec-conformant ("pause is a clock op, not a state"): `PauseHandle`
> {pause,resume,toggle} freezes running time; resume re-bases the now-**shared**
> base cell (Ctx derives every deadline from it per call); groups gate at the pass
> top with `Event::Paused`/`Resumed` (pipewireaudiosink holds hardware via its
> silence latch — device clock freezes ⇒ zero re-base shift, InstantClock shifts by
> the pause interval — both correct); a sink whose wait expires mid-pause blocks
> and re-derives after resume — provably nothing renders while paused;
> `start_paused` = preroll-and-hold; stop unsticks a paused pipeline. Known gap: a
> head blocked in a ring pop learns of pause by data starvation, not the event.
> **Zero-copy agent merged** (ZERO-COPY.md stages 1+2): Mp4Reader retains input
> `Memory` chunks, samples resolve to `Memory::slice`; MatroskaWriter scatter
> clusters (size still known at close — no 0xFF regression). Movie remux:
> **~140→1438 MiB/s**; allocation calls now dominated by Stage-4 transport churn
> (Submission vecs + Batch columns, amplified by small buffers) — next core work,
> with an IO-hygiene agent (fadvise/sync_file_range + reactor vec reuse) in
> flight. Agent-requested core APIs worth adding: `Ctx::out_format()`,
> `Ctx::pad_linked(pad)` (unlocks skipping unlinked-track resolution).

> **Update — session 4f (overnight 2026-07-25):** the through-the-night batch.
> **Stage-4 transport zero-alloc closed**: batch shell return rings (columns
> circulate back upstream) + the IO agent's fadvise/sync_file_range hygiene
> (memcapped stall GONE, dirty 1.15 GB → 8–63 MiB) + the Reactor trait's
> caller-owned vecs — movie remux **314.9K → 87.1K allocs**, and the filesink
> rerun stall fixed earlier (unlink-before-create). **Mid-batch event boundaries
> DONE** (the last dynamic-caps gap): positioned events + drain-cursor barrier;
> a mid-process() announce lands between exactly the right two buffers, across
> rings, inline hand-offs, and the queue's forward hop (fan-in stays
> pre-positioned — documented). **Leaky rings integrated**
> (set_queue_leaky(el, DropNewest), drops surfaced in producer counters).
> **Pause UX**: 'p'/'q' stdin control in `streamcraft launch` and `play_file`.
> New Ctx APIs: pad_linked (skip unlinked tracks — wire into demuxers next),
> out_format. **In flight**: multi-track audio remux agent (aac/esds +
> MkvMux::multi fan-in — merge on completion).

> **Update — session 4g (overnight 2026-07-25, morning wrap):** **THE MOVIE REMUXES
> WITH AUDIO.** Multi-track agent merged: mp4 `esds`→AudioSpecificConfig ('aac'
> family, A_AAC CodecPrivate), `MkvMux::multi(n)` fan-in (per-pad caps setup,
> pts-ordered merge, video-anchor clusters, small-run coalescing). Movie → 2-track
> h264+aac MKV, 1544.9 MiB, ffprobe/mpv clean incl. tail seeks, memcapped no stall.
> Honest note from its A/B: today's throughput is disk-bound parity (~205-257
> MiB/s); the earlier 1438 figure was a fully-cached pre-hygiene run.
> deliver_events dynamic-pad OOB panic guarded (tolerant install). **Core
> follow-ups queued from agent findings**: `Ctx::pad_offers` + event-pad-id (drop
> the static sink_0..7 fallback and give dynamic pads real re-validation), fan-in
> idle park (~374K sched_yield/movie), writev gather in FileSink, single-track
> MkvMux aac offer, av1C, the one-batch EOS-race note in MkvMuxN.

> **Update — session 4h (2026-07-25): shell-recycling leaks + Cues.** (1) User
> report "still 140K allocs, walls of 1B–512B" → four leaks in batch-shell
> recycling (`6d2fd91`, 141K → **12.2K** allocs/movie): inert-pad `take_output`
> destroyed a warm spare per idle pass; `take_input_on` minted fresh shells
> (→ public `Ctx::recycle_input`); shared spare pool paired capacities
> pessimally (→ split out-duty/input-duty pools); and the load-bearer — shell
> return rings sized == data ring silently dropped burst returns (→ `2*cap+2`,
> spare retention scales via new `ring::capacity()`). Lesson: instrument
> (capacity-logging run) after the first wrong theory, not the third. EOS
> drain now gathers small pieces (was per-header exact alloc+Arc). Remaining
> 12K: mkv reader `to_vec` (the queued retained-slice conversion), Mp4Reader
> chunk-boundary copies, EOS finalize. (2) **Cues + front SeekHead**
> (`e7bf564`): `MatroskaWriter::enable_cues` (opt-in; MkvMuxN default-on) —
> 128-octet Void reservation, cue-worthy = cluster holds a cue-track keyframe,
> Cues after last Cluster, and the new core **`Event::Patch{offset,data}`**
> (single positioned overwrite; `Ctx::push_event`; FileSink applies — its IO
> was positioned already; queue must forward it like FormatChange if one ever
> sits before a byte sink). Movie: 1078 CuePoints/89.8 min, mpv 0/50%/95%
> clean, ffprobe silent. Benchmark-vs-ffmpeg protocol: `time (cmd && sync)`
> both ways — ffmpeg exits with ~1 GB dirty; its extra index is ~24 KB (not
> the speed story).

> **Update — session 4i (2026-07-25): graphics ride SDL3.** Spec update executed:
> **`wayland/` + `vk/` deleted**, replaced by `sdl3/` (**sc-sdl3**) binding
> `sdl3-sys` (pure bindings, the ash/pipewire pattern; system SDL3 via
> pkg-config, flake provides 3.4.8). `Sdl3VideoSink` keeps the WaylandVideoSink
> contract verbatim; presentation is one streaming `SDL_PIXELFORMAT_IYUV`
> texture (== our tight I420, chroma ceil/2) — YUV→RGB on the GPU, the CPU
> conversion pass and ~150 KB of protocol/renderer plumbing gone. gray8 = IYUV
> with constant-128 chroma. Headless: `SDL_VIDEODRIVER=dummy`. The scope UI
> backend lands in this crate next: **immediate mode** (user-confirmed) over
> `SDL_RenderGeometryRaw`, per-frame vertex arenas. Also fixed en route: the
> demuxer's dynamic pads now carry **per-track offer menus**
> (`codec::offers_for`) — the shared all-families menu let link-time
> intersection admit every decoder, so play_file's try-in-order autoplug put
> h265dec on an h264 track (all AUs dropped at runtime; never seen before
> because the test movies were HEVC). Gate: the remuxed movie plays
> `mkvdemux ! h264dec ! sdl3videosink` in a real window, zero warnings;
> workspace 102 result lines green. **Next: scope** — protocol first
> (feature-gated core, length-prefixed POD frames), then the SDL UI.

> **Update — session 4j (2026-07-25): the playback freeze + the 4 GiB leak.**
> Real-window movie playback froze nondeterministically (~5-30 s) then, once
> cured, leaked to OOM. Two root causes, both found by instrumentation added
> along the way (mkvdemux/h264dec/sdl3videosink log coverage; `play_file
> --stats` 3 s counter deltas; `Ctx::pool_stats` + h264dec's edge-triggered
> `alloc_stall` log; gdb thread dumps): (1) **shared-pool deadlock** —
> `acquire_exact` handed whole 4 MiB slots to ~12 KB AUs *and* its heap
> fallback still counted against `outstanding`, so the output-blocked demuxer's
> queue starved the decoder's `try_alloc` forever (`outstanding=24/24`, all
> counters flat; three groups spin-yielding). Fixed in `ec3c135`: slots only
> for requests ≥ slot_size/4, heap fallback fully **unlinked** from pool
> accounting, plus a per-decoder pool in play_file (`set_element_pool`, the
> pool-negotiation v1). (2) **oxideav-h264 leaks every reference picture** —
> upstream `RefPicStore` has insert and no eviction (~35 MB/s growth; heaptrack
> peak 89.8% in `finalize_in_progress_picture`). Fixed in `c69bed0` by
> vendoring 0.1.7 (`vendor/oxideav-h264`, `[patch.crates-io]`) with a
> documented `retain_keys` sweep (STREAMCRAFT-PATCHES.md, upstream-PR
> candidate); crate's 1288-test suite green, RSS flat 177→188 MiB over 2 min.
> Movie now plays smoothly. **Known follow-ups**: the decoder's internal churn
> (51.8M allocs / 35 s — CABAC temporaries; nativization debt), spin-yield
> groups should park (3 cores busy while stalled), h264dec emits feed-order
> pts (B-frame reorder caveat — sink saw pts go backwards), and mkvdemux
> announces no fps so sink QoS is disarmed.

> **Update — session 4k (2026-07-25): AAC decode adopted.** `sc-aac`/`AacDec`
> over oxideav-aac 0.1.6 (raw `decode_raw_data_block` + ASC API — the Decoder
> trait is ADTS/LOAS-only), MkvDemux A_AAC wiring (family/offers/ASC-head/
> announce), launch registration, `decode_adts` diagnostic example. Gates:
> ffmpeg oracle 72.8 dB @44.1k / 69.5 dB @48k (committed ADTS fixture); the
> movie's real track = timeline-exact (failed AUs substitute silence, AU-indexed
> warnings) but **~15 dB systematic fidelity + 21/1300 `ElementDecodeInvalid`
> clusters** — real-content tools (M/S/TNS/PNS/short windows) are wrong or
> unsupported upstream. **Next AAC step: vendored upstream fix** (the h264-leak
> pattern): failing AU indices are logged, `/tmp/movie_audio.adts` +
> `decode_adts` reproduce, diff per-frame error vs ffmpeg to isolate the tool.
> Also queued: play A+V together (aacdec ! audioconvert? ! pipewireaudiosink
> beside the video chain — clock + latency infra is ready); mp4demux still has
> the shared-offer-menu autoplug weakness mkv got fixed for (d2a2440).

> **Update — session 4l (2026-07-25): A+V plays for real.** play_file wires
> `aacdec ! pipewireaudiosink` beside the video chain (DAC = pipeline clock,
> audio-master sync). Three rounds of whack-a-stall to get there, each caught
> by instrumentation: (1) aacdec's 4 KB frames pinned whole 4 MiB shared slots
> → own 64 KiB pool; (2) the demuxer interleave problem (one thread feeds both
> pads; either full ring blocks the other track) → deep rings both branches;
> (3) **the real one** (`4efa3ab`): perf showed 40.9% of the process in libm
> `__cos_fma` — upstream oxideav-aac computes the IMDCT as a naive O(N²) sum
> with a cos() per term (~200M/s at 48 kHz stereo). Vendored with a Chebyshev
> three-term-recurrence patch (STREAMCRAFT-PATCHES.md #2): crate suite green,
> oracle SNR identical to the digit, audio holds realtime, video back at
> 25 fps. Example also gained mimalloc + thin-LTO release (adopted decoders
> are alloc/call-heavy) and SC_AUDIO_DROP / SC_FORCE_WALL discriminators.
> Soak: minutes of A+V, all counters realtime. Upstream debt list for
> oxideav-aac grows: N·log N IMDCT is the real fix; the ~15 dB real-content
> fidelity + clustered AU failures remain from 4k.

> **Update — session 4m (2026-07-25): introspection protocol + scraft-scope live.**
> Three parallel Opus agents + inline integration. (1) **Core** (`3cc31ec`,
> `core/src/introspect/`, feature `introspect`, still zero-dep/`deny(unsafe_code)`):
> the **SCIP v1.0 wire format** — LE length-prefixed POD frames, 8 B header
> `len/kind/seq`, per-connection string interning (`StrDef`, `&'static str` by
> pointer identity), `row_size`-strided tables for forward compat; request/reply
> Topology (3 tables incl. per-edge `FixedFormat`)/Dot/Counters(+`now_ns`)/Latency/
> LatencyReport/Props, pushed Subscribe streams BusMsg(100 B — the brief's "96" was
> an arithmetic slip)/LogRec(88 B)/Dropped, controls SetProp/Pause/Resume/
> SetLogLevel/SetTracing (Step reserved → Error(6)). Server = accept thread +
> per-client blocking threads; slow client ⇒ `try_push`-drop + Dropped gap frames
> (ring has no drop-oldest, by design). **Bus tap** copies rows under the existing
> send lock (never steals from the app); **log tap** rides `LogDrainThread`;
> `build_logging` wires threshold-off channels (cap 256) when serving so streaming
> cost stays byte-identical. `pipeline.serve_introspection(path)` or
> `STREAMCRAFT_INTROSPECT=<path>` env (zero-code attach). 22 pinned wire/round-trip
> tests. (2) **scope UI** (`755364e`): immediate-mode toolkit on
> `SDL_RenderGeometryRaw` — per-frame bump arena, batched draw list w/ clip stack,
> original CC0 8x8 bitmap font atlas, panels/kv/fill-bar/button/toggle/tabs/log
> view, all window-free-testable (35 tests) + `ui_demo --frames N` headless.
> (3) **layout** (`18e4517`): pure Sugiyama layered DAG (longest-path layering,
> barycenter sweeps, dummy waypoints, port-ordered attach points, group boxes;
> 17 golden/invariant tests). (4) **Integration** (`adcbda7`): protocol client
> (blocking reader thread — framing never torn; `Model` + topo_gen; counters
> polled 100 ms, re-GetTopology on mutation cues), graph/elements/events+logs
> panels, `scraft-scope <sock> [--frames N]`, `Scope::spawn` embed (same app on a
> temp socket). **E2E**: play_pattern + env attach → headless scope decodes
> 2 elements/1 edge + live counters, both exit 0. Deferred: MCP server, step(),
> prop-editing UI, latency panel, TCP, buffer peeking.

> **Update — session 4n (2026-07-25): scope UI feedback round** (`a946001`). User
> feedback: font/zoom/pan/splitters/caps/groups/progress/scrolling. Font: the
> hand-drawn 8x8 replaced by **JetBrains Mono 2.304 (OFL-1.1) baked offline**
> (`scope/tools/bake_font.py`, Pillow via nix-shell) into one AA A8 atlas at 6 px
> sizes (`font_data.rs` + `font_atlas.a8`, committed) — real typeface, zero new
> deps; requested px snaps to nearest baked size, residual scales the quads
> (linear filtering), so graph-zoom text stays smooth. Graph view: wheel-zoom
> anchored at the cursor, drag-pan (4 px click/drag threshold), auto-fit per
> topology + Fit button + zoom readout; **caps inspection** = hover an edge for a
> tooltip with the full negotiated format, click to pin (edge fields were already
> on the wire; `duration` field also feeds transport); groups = per-group hue
> hull fills + borders + labels. Splitters between graph|elements and body|log
> (drag fractions); elements panel wheel-scrolls (`Ui::scroll_body`) with a
> scrollbar. Transport: running-time clock (counters `now_ns`) + progress bar
> against duration when a format announced one (mkv `Info\Duration` rides the
> demuxer pad formats — no core change needed). Camera/hit-test helpers are pure
> + tested (58 scope tests). Known gap: position is *running time* — drifts from
> media time after a seek; honest fix needs a position query (backlog).

> **Update — session 4o (2026-07-25): scope feedback round 2** (`12ab0ca`,
> `38de9c9`). (1) **Duration end-to-end**: new `BusMessage::DurationChanged`
> (ordinal 13, critical) — mkvdemux posts once from `process()` when the
> streaming reader parses `Info\Duration` (preroll is too early: play_file
> prerolls before `run()` starts the server); the server keeps an
> always-attached internal sticky bus tap + new `GetInfo`/`Info` frames
> (0x0025/0x0026, **SCIP v1.1**) so late-attaching clients poll what they
> missed. Verified: 1:29:45 on a 90-min mkv, attach 2s after start. Lesson:
> announced format fields survive fixation only when the consumer's offers
> declare them — a playback path drops `duration` at interning, hence the bus.
> (2) **Rotated-quad lines**: `DrawList::line` diagonals were filled bounding
> rects (the "gray rectangle" fan-out bug) — `Prim.corners` now carries real
> rotated geometry. (3) Graph: **minimap** (bottom-right overview, draggable
> viewport rectangle), group hulls reserve a label strip (nodes no longer
> cover the name), edge labels draw above nodes. (4) **Launcher mode**:
> `scraft-scope <command> [args…]` spawns the target with
> `STREAMCRAFT_INTROSPECT` on a private socket, attaches, kills the child on
> exit (attach mode now requires an actual socket file type). (5) **Dockable
> panes** (`ui/dock.rs`, simprof's dock model): generic tree of splits with
> tabbed leaves → flat layout geometry; drag a tab → ghost + VS Code drop
> zones (center = join tab group, edge = split), dividers drag-resize,
> emptied leaves collapse; `Ui::begin_region` = title-less panel body (tab
> bars replace titles). 68 scope tests green; workspace green; launcher e2e
> headless OK. Dock persistence (save/restore layout) deferred.

> **Update — session 5 (2026-07-25): time-based seeking, end to end** (`a00dfbb`
> core, `a86627a` decoders, `421f90f` http, `d560c08` protocol/UI, `2093bd5`
> mkv + merges). Five tracks, three parallel Opus agents + two inline. **(1)
> Core clock rebase**: `SeekHandle::seek(to_byte, to_time)` rebases running
> time to the target under the pause mutex (`PauseShared::rebase_to`;
> seek-while-paused composes with the resume shift). The base cell went
> **signed** (`Arc<AtomicI64>`, `BASE_UNSET=i64::MIN`; helpers in time.rs) —
> a device clock starts near 0, so `base = now − target` is negative on
> forward seeks. `SeekState.to_frame` → `to_time_ns` (sinks derive their own
> units; pipewire sink computes frames = time × rate). `Ctx::wait_until`
> re-derives per iteration, returns Interrupted on a seek-gen change, and
> parks in 10 ms slices (found gap: parked waits never re-derived; nothing
> called ClockWait::interrupt). `elements/tests/seek.rs`: forward-no-stall,
> on-time backward replay, pause composition, tap position jump, rapid
> seeks. **(2) mkv**: `parse_seek_head`/`parse_cues` (Cues live after the
> last Cluster, found via the front SeekHead; positions Segment-relative),
> `MatroskaReader::resync_streaming()` with a bounded Cluster-id scan (a
> proportional-estimate landing anywhere recovers), demux FlushStart =
> resync + re-emit codec heads + per-video-track keyframe gating. **(3)
> Video decoders** already reset on FlushStart since adoption — verified +
> flush tests added per crate. **(4) Protocol v1.2 + UI**: `SeekIndex`
> (cues floor-lookup, proportional file_len fallback) +
> `Pipeline::set_seek_index`; SEEK frame 0x0027 (clamp→map→rebase→Ack, or
> Error(6) without an index); scope progress bar scrubs (drag marker,
> release seeks); play_file builds the index at open (SeekHead → pread Cues)
> and stdin digits 0-9 seek to n×10%. **(5) HttpSrc rides the Reactor**
> (user request): `OpKind::Recv` streaming reads in SyncReactor (blocking
> read) and io_uring (`IORING_OP_READ` @ offset −1), HttpSrc drains the Io
> mailbox with one sequential read in flight + a `valid_from` stale floor;
> trickle/early-EOF/uring tests. **Fix round (`7b52f4b`), from live testing:**
> (a) *frozen video after seek* — the rebase used the REQUESTED time while
> content resumed at the preceding cue cluster, leaving video permanently
> late by the gap (QoS dropped everything; audio free-runs): seeks now use
> the RESOLVED cue time (`SeekIndex::resolve` returns `(byte, landed)`).
> (b) The rebase anchors to the clock's *current* reading, not `pause_at` —
> a device clock advances during the post-pause drain. (c) **Seek-while-
> paused reworked**: the pause gate parks seek-aware (`park_while_paused`,
> notify on seek), flushes while paused, and bursts passes while they
> progress (re-prime in ms, 100 ms idle ticks after); `ClockWait::
> wait_ticked` bounds every parked wait (incl. MockClock) and waits honour
> stop; sdl3videosink **prerolls the first post-flush frame** (paused seek
> shows its frame); pipewire sink consumes nothing while paused (blocking
> would strand its group across a resume). (d) `TapHandle` reads the clock
> through a shared cell `run()` updates at selection — pre-run handles mixed
> the default clock with the device-rebased base. (e) release profile:
> `debug=\"line-tables-only\"` for heaptrack/perf. E2E exact: playing seek
> lands 2690.0s, paused seek lands 1075.0s, stays paused, resumes clean.
> Known limit: http has no Range-based seek; plain pause/resume still
> excises the device drain (pre-existing, noted). Also landed (`47772fa`):
> **GStreamer-style pipeline diagnostics** — `STREAMCRAFT_DEBUG=pipeline:debug`
> logs add/negotiate (offer menus, failed probes)/link (resolved caps)/dynamic
> pads/clock selection/thread groups at build time; runtime `format_change`
> rides the element ring.

> **Update — session 4p (2026-07-25): scope fix round** (`28ac090`). (1) The
> vanishing-pane dock bug: my collapse moved the sibling node into the parent
> slot, orphaning the sibling's index — dropping a panel onto its own split
> sibling then inserted into an unreachable slot. Fixed the simprof way:
> **repoint the grandparent/root at the sibling, never move nodes**;
> `all_panels()` walks reachable-only. (2) Fan-out overlap: attach slots were
> port-ordered while barycenter stacked targets differently → edges crossed at
> the boundary; now ordered by edge *direction* (first-waypoint y, port as
> tiebreak), one slot per edge. (3) Group hulls overlapped: `Opts.group_gap`
> adds clearance between different-group neighbours in a layer (seed +
> separation). (4) Tab faces get 1px dividers on a darker strip; progress bar
> centred on the top-bar mid-line. 72 scope tests. **Click-to-seek was asked
> for and is NOT wired**: honest seek for A+V mkv needs (a) mkv Cues parsing /
> cluster resync in the demuxer+reader, (b) a time→byte mapping the server can
> use (SeekHandle is byte+frame based, mapping is the caller's job), and (c) a
> clock story for post-seek pts (sinks wait at base+pts — a forward seek stalls
> them; flush/seek was validated on the flac path). That is its own milestone —
> next session candidate.

> **Update — session 4q (2026-07-25): time-based seeking end-to-end.** mkv Cues
> parsing → `SeekIndex` (1078 cue points on the movie), protocol `Seek` (SCIP
> v1.2, scrub-bar click in scope), out-of-band seek generation + signed base
> rebase (`core/src/time.rs` helpers, `PauseShared::rebase_to` anchored to
> `clock.now()`). Hard-won: **always seek to the RESOLVED cue time** (rebasing to
> the requested time while content resumes at the preceding cluster = permanent
> lateness = QoS drops everything → frozen video); seek-while-paused flushes
> immediately, prerolls one frame, stays paused (`ParkWake` gate: wake-on-seek +
> 100 ms Tick + burst re-prime while progressed); `ClockWait::wait_ticked` so
> MockClock waits observe seeks. Also: HttpSrc ported to the Reactor, release
> profile `debug="line-tables-only"` (heaptrack), nightly toolchain for codec
> SIMD, GStreamer-style build-phase pipeline logging (`STREAMCRAFT_DEBUG=
> pipeline:debug`), duration/progress end-to-end in scope.

> **Update — session 4r (2026-07-25): colorimetry + owned GPU renderer + vaapi
> decode live.** Three tracks. (1) **Colorimetry plumbing** (`3d3338b`):
> `video/src/color.rs` vocab (matrix/range/transfer/primaries, H.273 §8.1–8.3
> mappings, height≥720 defaults), mkv `Colour` parse/announce/write-back
> (RFC 9559 §5.1.4.1.31; reader defaults 2/0/2/2 — 0 is *valid* matrix
> identity, never an absent sentinel), passthrough on all five decoders.
> **Interning lesson #2**: an `Any` field declaration interns the field name but
> not its categorical *values* — `build_fixed` dropped the demux announcement
> whole (broke vp8 remux); muxer offers now declare the value names via `Set`
> (the `MUX_SAMPLE_VALUES` precedent), and an unresolvable announcement is a loud
> bus Warning in the scheduler instead of a silent drop. (2) **Owned GPU render
> pipeline** (agent, `82569ef`, merged `39e5272`): SDL3 GPU API (Vulkan) +
> our SPIR-V shaders (offline glslang bake), color science in Rust as uniforms
> (BT.601/709/2020 matrices, BT.1886/sRGB/PQ/HLG EOTFs, 203-nit BT.2408
> normalize + Reinhard tone-map slot v1 — BT.2390 EETF is the named follow-up),
> classic-renderer fallback + `SC_RENDER` override, 16 tests incl. two real
> on-device YCbCr goldens. (3) **sc-vaapi** (agent `81024f8` + fix round
> `72b071c`): hand-rolled libva FFI (~26 fns, size-asserted structs),
> capability-gated registration, working `vaapih264dec` (DPB/POC/ref-lists,
> NV12 readback). Post-merge fixes that made it actually play: DRM node opened
> read-*write* (read-only fd answers every query then fails the first GEM
> allocation at vaCreateContext), surfaces/context at *coded* MB-aligned size,
> one surface-ownership rule (`release_if_unreferenced` — IDR reset was
> aliasing queued surfaces, MMCO unmark ops leaked them, flush leaked the
> queue), and surface exhaustion as *backpressure* not AU drops. play_file
> prefers hw decode (probe-gated, link-through fallback): movie holds 25 fps
> hw-decoded, zero drops. **Encode**: probe reports per-profile
> EncSlice/EncSliceLP (this box: h264+h265), `Config::new_encode` + hw tests
> prove an encode context allocates — `vaapih264enc` element is the follow-up.

> **Update — session 5n (2026-07-26): the NVR milestone app + three core
> liveness bugs it flushed out.** The chosen stress-test application (breadth
> over depth: live clocks, fan-out, fan-in, segmented muxing, long-run memory)
> is LIVE end-to-end, pure-sc: `scraft-nvr` (new `nvr/` crate) records N RTSP
> cameras into rotated **self-contained MKV segments** (Cues + SeekHead + a new
> `MatroskaWriter::reserve_duration`/`duration_patch` back-patch) while a
> **mosaic wall** (fan-in `InputPolicy::Any` latest-frame compositor,
> nearest-neighbour per-plane, cites Wolberg §5.1) renders all cams through one
> `sdl3videosink`. Per cam: `udpsrc ! rtpsession ! rtph264depay ! tee`
> (`elements::flow::Tee` — new, wildcard-offer, refcount fan-out) `!
> mkvsegmentsink` + `! h264dec ! mosaic.sink_i`. `mkvsegmentsink` is
> **reactor-native** (drain-then-swap rotation; one file per element;
> `stop()` teardown finalize via dup'ed fd is the one sanctioned blocking
> exception — now lint-enforced: root `clippy.toml` disallowed-methods bans
> blocking IO in elements). Camera sim: `rtsp_serve --loop` (`CamSrc`: reactor
> reads, pts += loops×duration, in-band SPS/PPS per IDR).
> **Numbers (this box, 4×640x360@25fps sw-decode + record + wall, 120 s):**
> ~2.35 cores, RSS flat ~160 MB, **12000/12000 frames** recorded (zero drops
> at every element), 48/48 segments pass ffprobe + headless mpv + `--start=95%`
> tail, per-cam durations sum to **exactly 120.000 s** (no gaps, stop-path
> finals included). **Core bugs found & fixed:** (1) *production-is-progress* —
> the inline hand-off never set `progressed`, so a source feeding an inline
> buffering tail parked every pass and crawled one buffer per 10 ms tick
> (mp4 ragged_chunking: 4+ min "hang" → 0.07 s); (2) *runtime registration* —
> `take_registration → set_file` was start-only, mid-stream registrations
> (segment rotation) wrote into `Err(NotFound)`; (3) *settle-IO-before-stop* —
> submissions stranded in an outbox by event-path finalize (EOS during stop)
> left a 26 KB hole of zeros mid-file; run_group now flushes outboxes and runs
> the reactor to idle before `stop()`. **Sender-side pool lesson**: the paced
> `udpsink` holding 1.4 KB packets that each pinned a 4 MiB shared slot
> deadlocked `camsrc ! pay ! udpsink` (racy 1–25 packets then all groups park);
> fix is the established per-element pool (`set_element_pool(pay, 2048, 256)`)
> — mpv/ffmpeg "truncated keyframe" symptoms were THIS, not receiver buffers.
> **(4) — closed same session:** *error cascades stop* — a failed group
> (element `Err` or panic, via `catch_unwind` in the group-thread closure) now
> sets the shared stop flag + `bump_idle` before its thread dies, so siblings
> with no data path to the failure (independent cameras) unwind within a tick
> and `run()` surfaces the error instead of joining a healthy endless group
> forever; Ok/EOS exits deliberately do not cascade. Regression test
> `elements/tests/error_stop.rs` (negative-checked: disabling the cascade
> hangs it into its 30 s watchdog). Workspace: **132 result blocks green,
> rc=0**.

> **Update — session 4s (2026-07-25/26): RTP/RTSP network streaming + the
> scheduler finally parks.** The network milestone, receive-first, three
> parallel agents + inline elements. **sc-rtp** (RFCs 3550/3551/6184/7587 in
> tree): packet view (frozen first, scaffold commit `2529eef` — agents built
> against pinned stubs, zero merge conflicts), A.1 extended seq, jitter buffer
> (pure state machine, latency-held gaps, late-uncounts-loss), RTCP SR/RR,
> H.264 depay/pay (single NAL/STAP-A/FU-A; ffmpeg fixtures byte-exact vs the
> encoder's own stream) + Opus; elements: udpsrc (reactor Recv; **SO_RCVTIMEO
> 100 ms — a quiet sender otherwise wedges the group in read(2), stop
> unobservable**: the hard-interrupt reactor-cancel follow-up's first real
> bite), rtpsession (SDP-declared streams → preroll pads; **fan-in elements
> must consume BOTH `inputs` and `take_input_on`** — single-linked-pad heads
> get batches via the param, the MkvMuxN lesson re-learned), depay/pay
> (payloaders **Active** — a demuxer branches only from its group tail),
> clock-paced udpsink (`wait_until(pts)` = ffmpeg's `-re` by construction).
> **sc-rtsp** (RFC 2326/8866/2617 + hand-rolled MD5/base64): SDP parser,
> client (digest auth, control-URL append-not-RFC1808, interleaved demuxer)
> live-validated vs mediamtx; server (Transport parse, 454/455/459/461 paths,
> session timeout) loopback-tested against our own client. **play_file
> rtsp://** runs the client dance app-side (no-bins) into the hw-first decode
> chain: 25 fps vaapi, zero drops, vs mediamtx AND vs **rtsp_serve** (our
> server example, avcC→sprop SDP, per-viewer Play/Teardown pipelines) — the
> pure-sc e2e; ffprobe as the independent second implementation. Also this
> session: vaapi green-frame fix (pred_weight_table forwarded, `0a45d95`) +
> display-order fixes ((gop,poc) key + pigeonhole surface budget, `6aeac39`);
> **aac O(N log N) IMDCT** (vendored patch #2, `2b943ac`: DCT-IV + quarter-FFT,
> imdct was 50.5% of playback cycles → gone, ~28% less total CPU); av1
> assessed (scalar oxideav ≈ 10–15 fps @720p; MC+CDEF are the SIMD targets,
> entropy decode is NOT the wall); and the long-standing **spin→park**
> scheduler fix (agent, `370ef39`: eventcount on the pause gate's plumbing,
> ring-push/pass-end/seek/pause bumps + 10 ms tick backstop; **13.2× less
> playback CPU** — 1.06 cores → 0.08, sys time ÷50, loom green, park_cpu
> regression test). Follow-ups: RTCP RR sending + SR-based A/V sync apply,
> AAC depay (RFC 3640), TCP-interleaved element wiring, RTSP server
> multi-client/PAUSE-resume, av1 SIMD campaign, reactor hard-cancel.

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
2. ◐ **Per-buffer mid-batch boundaries**: `FormatChange` is delivered before *all* of a
   batch's buffers — correct for announce-once-at-start, wrong for a change partway
   through one batch. Split the batch at the event position (needs `events` to carry a
   batch position). ✅ **Intra-group passive `FormatChange`** — DONE (session 3c): non-head
   group members get their input batch's events delivered (with re-validation) before the
   buffers they precede, so `flacdec ! flacenc` inlined in one group re-fixates correctly
   (`flac/tests/transcode.rs`).
3. ✅ **`AudioConvert`'s dynamic consumer path** — DONE. `AudioConvert::new(target)` infers
   the full input format by name off the negotiated caps (`learn_from_sink`); no
   `with_input` needed. (`audioresample` landed alongside.)

> **Update — session 3d (2026-07-24): MILESTONE 5 SHIPPED.** `play_mkv` plays a
> VP8-in-MKV file in a real window (`filesrc ! mkvdemux ! vp8dec ! waylandvideosink`,
> clock-paced, clean EOS), and `streamcraft launch videotestsrc frames=60 !
> waylandvideosink` opens a window from a one-liner. Landed: **sc-wayland** (hand-written
> wire-protocol client + shm sink; BT.601 §-cited per the new **algorithm-citation rule**
> — every algorithm cites its standard/paper, clean-room, see memory), the **streamcraft
> CLI** (launch/dot/inspect/list; switches-first + unquoted pipeline; `registry::describe`
> cards), **parallel-forest transcode** proven (6×60s FLACs in 0.8s, one pipeline,
> ~12 threads; inline-gate livelock fix + `set_pool`), **sc-mp3 wired** (84–102 dB vs
> ffmpeg), **sc-opus REJECTED** (published 0.0.13 fails the official RFC 6716 vectors —
> silence/noise; the desired decoder exists only at unpublished git HEAD; turnkey
> re-vet recipe in `opus/src/lib.rs`). **Decision: containers are hand-written**
> (like sc-mkv/sc-ogg) — the oxideav-mp4 adoption was stopped; agents in flight:
> hand-written `Mp4Demux` (oxideav-mp4 demoted to dev-oracle; muxer is a follow-up
> task) and **sc-vk** (clean-room Vulkan renderer: render into exported DMA-BUFs,
> present via sc-wayland's `zwp_linux_dmabuf_v1`; `ash` only; cited algorithms).

> **Update — session 3e (2026-07-24): real-file playback + the GPU renderer.**
> `play_file` plays a real **1080p HEVC BluRay MKV** (4 tracks: HEVC→h265dec, AAC/PGS
> to drop-sinks) in a window, bounded at ~1.2 GiB RSS — after fixing the OOM family
> it exposed: MkvDemux unbounded `ctx.alloc` → **try_alloc + pending carry +
> consume-only-while-emitting**; `Pool::acquire_exact`/`Ctx::alloc_exact`
> (right-sized heap fallback, never a 4 MiB slot per 15 KB sample); ring-fed group
> heads now cap their input backlog at INLINE_INPUT_CAP (closed-and-drained guard).
> Video decoders went **Active** (a decode is ms, not the inline ns budget; also
> keeps a branching demuxer a legal group tail). **sc-vk landed, built in-session**
> (agents kept OOMing the box — policy now: max ONE background agent): compute-only
> Vulkan → exported LINEAR dma-bufs → presented by sc-wayland's hand-written client
> via `zwp_linux_dmabuf_v1` (salvaged+reviewed from the dead agent's worktree);
> BT.601 §-cited GLSL + committed SPIR-V; byte-exact vs the CPU path on a Quadro
> P620; `vkvideosink` registered, `play_file --vk`. **sc-mp3 salvaged + merged**
> (agent finished, OOM'd before committing). **sc-opus REJECTED** on official
> RFC 6716 vectors (0.0.13 emits noise/silence; recipe in opus/src/lib.rs).
> Test heavy pipelines under `systemd-run --scope -p MemoryMax=3G`. Workspace 90
> binaries green. In flight: hand-written Mp4Demux (containers stay hand-written —
> user decision; oxideav-mp4 is dev-oracle only). Follow-ups: per-link pools, h26x
> pts reorder exactness, Opus re-vet on next upstream publish, explicit-sync +
> scaling/HDR ladder for sc-vk.

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
- **Kill the last tiny allocations** (`mkv/src/reader.rs` mainly; isolated, good
  subagent scope). After `6d2fd91` the transport is ~zero-alloc; the movie remux's
  remaining **12.2K** allocs (heaptrack + simprof `min_size`/`max_size` bands — rerun
  `heaptrack --record-only …/remux_to_mkv` to re-measure) break down as:
  1. **~67% — `MatroskaReader` copies** (`step` `to_vec`s every EBML element payload —
     the ≤16B wall: flags/track numbers/timestamps are 1–2B leaves; `push_frame` /
     `decode_block` `to_vec` each frame payload, the 257B–4KB band). Hit here via the
     example's verify pass, but it equally taxes `MkvDemux` playback. Fix = the
     **retained-slice conversion**, mirroring what `mp4/src/reader.rs` got (session
     4c, `SamplePayload::Slice`): (a) feed via `push(Memory)` retaining input chunks
     in a rope (keep a byte-slice entry point that wraps into one owned chunk for
     tests); (b) decode **leaf values in place** — an EBML uint/float should parse
     straight off the window to a scalar, never materialize as a `Vec<u8>` (this
     alone kills the ≤16B band); copy only blobs that must outlive the window
     (CodecPrivate — once, small); (c) frames become `Memory::slice` of the retained
     chunk, with an Owned fallback only when lacing/reads split a frame across
     chunks (mp4's `CarryPayload` pattern); (d) `MkvDemux` then emits those slices
     direct-to-out (ZERO-COPY Stage 2 style). Gate: bit-exact frames vs today's
     reader on the existing mkv fixtures + the movie verify pass.
  2. **~13% — `Mp4Reader::next_sample` Owned copies** for samples straddling retained
     chunks: plan reads to **end on stsc/stco chunk boundaries** so samples never
     straddle (local to the read planner; simpler than teaching downstream about
     two-piece payloads).
  3. **~10% — EOS finalize** (`drain_out_exact` gathers + their `Arc`s, Cues master):
     cold path, once per stream — leave unless it shows up again.
  Re-measure after (1); expect low-single-digit K. The tools: capture must exit
  cleanly, analyze via `simprof --mcp=PORT` + curl only (see
  `memory/profiling-and-backpressure.md`).
- **Performance** (`core/`, `elements/`): a benchmark-regression **CI gate**, plus the two
  known soft spots — a fenceless fast-path variant of the SPSC ring (the SeqCst Dekker
  fence dominates the ~51 ns/item hop) and a deeper-queue / batched-submission io_uring
  reactor.
- **TLS/https for `sc-http`** — *decided (2026-07-24): rustls.* The sanctioned dependency
  for this plugin (like libpipewire for `sc-pipewire`); core stays at zero deps.
  - **Why rustls**: pure Rust, audited (Cure53/ISRG), and — decisive here — a **sans-IO
    core**: `ClientConnection` never owns the socket (`read_tls`/`write_tls` +
    `reader()`/`writer()` pump buffers), so it drops into today's blocking-`TcpStream`
    `HttpSrc` *and* stays compatible with the io_uring reactor later, where
    stream-owning TLS APIs (native-tls/openssl — C bindings anyway) get awkward.
  - **Crypto provider** (the real decision; rustls 0.23 makes it pluggable): use
    **`rustls-graviola`** — Rust with formally-verified constant-time cores (s2n-bignum
    ports), from the rustls maintainer, x86-64/aarch64 only (fine for our targets), **no
    C compiler in the tree** (the default `aws-lc-rs` compiles C via cmake — audit/FIPS
    pedigree, but a real risk with this repo's worked-around clang toolchain; `ring` has
    maintenance concerns; `rustls-rustcrypto` is unaudited). The `CryptoProvider`
    boundary makes this low-regret — swappable to `aws-lc-rs` later without touching
    element code.
  - **Roots**: `rustls-native-certs` (system trust store), optional bundled
    `webpki-roots` fallback feature.
  - **Integration**: `enum Transport { Plain(TcpStream), Tls(StreamOwned<ClientConnection,
    TcpStream>) }` implementing `Read` — the header/chunked/Content-Length paths in
    `http/src/httpsrc.rs` are untouched (blocking reads are fine; `HttpSrc` is Active).
    Pin ALPN to `http/1.1` (refuse h2); enable session resumption (free win for
    reconnect-on-seek via `Range`). Everything behind an **`https` cargo feature** so the
    plain-HTTP build keeps its current footprint. Client-only, TLS 1.2+1.3.
  - **Verify at adoption time** (not from memory): exact provider crate names/versions
    and graviola's current maturity — check the rustls provider docs when wiring.
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
