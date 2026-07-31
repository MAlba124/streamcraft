# Gapless track transitions — design (APPROVED)

Status: APPROVED 2026-07-31 — user directive: "the most robust, future proof,
most solid implementation." Decisions: **straight to Phase 2** (Phase 1 kept
below as design context only); **canonical format f32 48 kHz stereo**; engine
is a **workspace crate `pf-player-engine`**. Additional robustness mandate:
`AudioOut`'s device backend is a seam (PipeWire impl + a deterministic capture
backend for tests — also the future CoreAudio slot), and gaplessness is proven
by a sample-exact continuity test across a track boundary, not by ear.

Original proposal follows.

Status: PROPOSAL, 2026-07-31. Grounded in a code audit of the as-built pipeline
(citations inline). Goal: reproduce musikkspiller's rodio contract — `append()`
pre-queues the next track ~5 s early, `len()` (queue depth) is the only
track-transition and end-of-playback signal, transitions are sample-continuous
on gapless albums — on top of profluens.

## The constraints that shape the design (audit results)

1. **A `Pipeline` is single-use and frozen at `run()`.** Elements are moved
   into group threads (`pipeline.rs:1593`), `run()` blocks holding `&mut self`,
   and every splice-shaped API (`relink`, `remove`, `add_subgraph`,
   `set_state`) is `todo!()` (`pipeline.rs:1959-1982`). Structural props are
   rejected while playing (`props.rs:155-179`); `filesrc.path` is one.
2. **The pipewire sink latches its device format on first configure
   (`sink.rs:373-379`) and destroys itself on `stop()`** (closes ring, joins
   the PW thread — `sink.rs:534-546`). Its device clock freezes when its
   stream pauses and is never seek-rebased (`sink.rs:273-276`).
3. **The audio glue latches too**: `audioconvert`/`audioresample`/
   `audiodownmix` short-circuit on `if self.input.is_some() { return; }`
   (`convert_element.rs:384` et al.), making their format-change paths
   unreachable after the first learn — and none of the three handles
   `Event::FlushStart`, so partial-frame `carry` survives seeks (a live bug
   independent of gapless; see "Independent fixes").
4. **Two `Pipeline`s run concurrently without conflict.** Separate pools,
   clocks, threads, buses; each pipewire sink owns a private connection and
   the server mixes streams. A paused pipeline's sink is already connected and
   renders silence (`sink.rs:174-178`).
5. **The decisive simplification: the audio sink is ring-paced, not
   clock-paced.** It never calls `wait_until`; pacing is pure ring
   backpressure (`sink.rs:8-10, 476-490`). PTS continuity across a track
   boundary is irrelevant for audio-only playback — only the ring must stay
   fed. (`Event::Segment` exists but is never constructed anywhere; we don't
   need it.)
6. Per-track open cost (probe, head reads, seek index, element construction,
   negotiation) is all app-thread work independent of any running pipeline
   (`player.rs:75-98`) — a "prepare next track" can do it entirely off the
   audio path.

## Rejected: in-pipeline playlist source

A `PlaylistSrc` that swaps files inside one running pipeline rides the
seek/flush machinery — but that machinery *deliberately preserves* decoder
header state (STREAMINFO, mkv tracks, mp3 format latch) because a mid-file
seek carries none; a different file would decode against stale headers. Worse,
a format-mixed library (flac → mp3) changes the *topology* (different decoder
element), which the frozen pipeline cannot express — that's the `todo!()`
wall. Fixing all of it means touching every decoder, all three glue elements,
the sink latch, and the scheduler's static-topology assumption. Maximal
contact surface, and it still fragments into a same-codec fast path. Rejected.

## Phase 1 — app-side crossover (zero framework changes)

A `TrackQueue` engine (this is also musikkspiller's backend seam):

- Current track: its own `Pipeline`, `run()` blocking on a dedicated thread.
- At T−5 s (`append()`): build the next track's `Player` fully (probe, seek
  index, chain), `start_paused`, run it on its own thread. Its sink connects
  to PipeWire immediately and renders silence (constraint 4) — the expensive
  connect happens *before* the boundary.
- Boundary: track 1's `run()` returns (sink has drained), engine immediately
  `resume()`s track 2 and emits `TrackChanged`. Queue depth, position, EOS are
  engine state — mapping 1:1 onto musikkspiller's 250 ms poll loop
  (`len()`, `get_pos()`, `TrackEnded`).

Gap at the boundary: track 1's drain + thread wake + track 2's first quantum
on a *different* PW stream — realistically one or two quanta (~5-40 ms).
Inaudible on silence-separated tracks; audible on true gapless albums. That's
the phase boundary: Phase 1 ships the engine API and correct behavior;
Phase 2 makes the same API sample-accurate underneath.

## Phase 2 — shared `AudioOut` with producer handoff (true gapless)

Refactor the pipewire sink into two pieces:

- **`AudioOut`** (new, app-created, one per app): owns what today lives inside
  the sink element — the PW thread, stream, SPSC byte ring, `Playback`
  counters. Fixed canonical format chosen at creation (f32 48 kHz stereo);
  every track chain is autoplugged to converge on it — `wire_audio_chain`'s
  convert+resample fallback (`autoplug.rs:718-760`) becomes the *forced* mode.
  This dissolves constraint 2's format latch: the device format never changes,
  so there is nothing to re-latch, and PW resamples server-side for the device.
- **`PipeWireAudioSink::with_output(handle)`**: the element becomes a thin
  producer that *attaches* to the AudioOut on `start()` and **detaches without
  draining** on EOS — in-flight ring bytes keep playing. `stop()` no longer
  tears down the stream in shared mode. Single-producer discipline: attach is
  exclusive; track 2's sink attaches only after track 1 detached.

Boundary sequencing: track 1 EOS → its sink detaches (ring still holds up to
~0.5 s of audio) → `run()` returns → engine resumes the pre-rolled track 2 →
its sink attaches and its already-decoded batches land behind track 1's
in-flight bytes on the next scheduler pass. Handoff cost is milliseconds
against a ~500 ms ring: **sample-continuous**. No core scheduler changes, no
Segment events, no decoder or glue modifications — constraint 5 is what makes
this legal.

Position and volume fall out naturally:

- **Position**: `AudioOut` records a frames watermark per attach epoch;
  `position = (frames − epoch_start) / rate`. Seek keeps the existing
  per-pipeline rebase, made epoch-relative. Monotonic device clock keeps
  running across boundaries (it no longer pauses between tracks).
- **Master volume + sleep fade**: an atomic f32 multiply in the AudioOut RT
  callback — exactly rodio's `set_volume`, RT-safe, no element needed.
  **Per-track ReplayGain**: a small gain stage in the per-track chain (tiny
  element, or an optional scale on audioconvert) set at build time from the
  track's stored gain — it dies with the track's pipeline, which is exactly
  the scoping ReplayGain wants.

Existing single-owner sink mode stays as the default constructor — NVR, pfplay
and every current pipeline are untouched; `with_output` is opt-in.

## Independent fixes surfaced by the audit — DONE 2026-07-31

1. ~~glue FlushStart handlers~~ — LANDED (commit 9d3324f: carry cleared,
   resampler delay lines reset, WavParse cursor rebased). If you are reading
   this while chasing a seek bug: the glue is no longer the suspect.
2. ~~loud format-relatch~~ — LANDED (same commit: once-per-change Warning).
3. ~~mid-file seek ended the track~~ — LANDED. The seek bug the note above
   disclaims was the **scheduler**, not the glue: `run_group` left its loop for
   good the moment its head reached end of stream, and the seek-generation check
   that revives a stream lives inside that loop. A player fills its buffers in a
   few hundred ms and then plays for minutes, so by the time anyone seeked, the
   only thread that could re-read the file was gone — the flush still reached the
   sink, which discarded its staged audio, saw its upstream ring closed, and
   declared EOS. End of stream now travels **in band** (`Batch::eos`) and a
   finished non-terminal group parks alive instead of exiting; terminal groups
   still exit at their terminus, and that exit cascades the retirement back up
   the graph, so termination is unchanged in shape. Regression tests:
   `play/tests/canonical_seek.rs`; the engine's inverted pin is now
   `robustness.rs::a_seek_is_followed_by_the_rest_of_the_track`.

## Open questions for review

1. Canonical AudioOut format: f32 48 kHz stereo proposed (PW-native float,
   downmix exists). Alternative: latch to the first track's format and
   recreate AudioOut on mismatch (cheaper CPU for 44.1 k libraries, small gap
   on format boundaries only).
2. Phase 1 first (engine seam unblocked now, gap ~1-2 quanta) then Phase 2 —
   or straight to Phase 2? Phase 1 is small and its engine API is designed to
   be re-plumbed, but it is still throwaway plumbing.
3. Does the engine live in the profluens workspace (a `pf-player-engine`
   crate musikkspiller consumes) or app-side in musikkspiller? Workspace crate
   proposed — pfplay-ui could adopt it too.
