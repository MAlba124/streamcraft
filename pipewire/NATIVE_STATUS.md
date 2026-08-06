# Native PipeWire client — status, what's missing, what's next

A from-scratch PipeWire client (`src/native/`) that speaks the daemon's wire protocol directly
over the `pipewire-0` Unix socket — no `libpipewire`, no C bindings — in the same raw-wire,
zero-per-message-allocation style `pf-present` uses for Wayland. Goal: drop the `pipewire = "0.8"`
C dependency and port `PipeWireAudioSink` onto this.

Everything below was diagnosed live against the running daemon (PipeWire **1.6.5**,
`/run/user/1000/pipewire-0`). Set `STREAMCRAFT_PW_DEBUG=1` to trace the full negotiation.

## File map

| File | Role | State |
| ---- | ---- | ----- |
| `src/native/pod.rs`   | SPA POD builder + parser (the reusable primitive) | done, unit-tested |
| `src/native/wire.rs`  | 16-byte message header framing                    | done, unit-tested |
| `src/native/conn.rs`  | `AF_UNIX` + `SCM_RIGHTS` fd passing, reused buffers | done |
| `src/native/proto.rs` | Core/Client/Registry + `PwClient` handshake        | done, live-validated |
| `src/native/spa.rs`   | SPA/ClientNode constants + Format/Buffers/DSP POD builders | done |
| `src/native/audio.rs` | ClientNode RT path (`native::play`)                | built; blocked on the RT trigger (below) |
| `examples/native_info.rs` | connect → handshake → dump registry            | **works** |
| `examples/native_tone.rs` | export a mono DSP tone node                     | reaches `Command Start`; not audible yet |

## Milestone 1 — control protocol — DONE ✅

Connect, `Core.Hello(3)`, `Client.UpdateProperties`, `Sync`/`Done` round-trips, `Ping`→`Pong`,
`Error` surfacing, `GetRegistry` → enumerate globals. `native_info` prints the server identity and
all registry objects, zero libpipewire. 25 unit tests pass (10 POD/wire + 15 ring).

## Milestone 2 — ClientNode RT audio — BUILT, validated through `Command Start`, NOT audible ⚠️

`native::play(cfg, quit, fill)` runs the whole node on one thread (`poll(2)` the socket + the
activation eventfd). Verified end-to-end against the live daemon:

1. `Core.CreateObject("client-node", …, PW_VERSION_CLIENT_NODE=6)`.
2. `ClientNode.Update` — one output port + a **node-level `EnumFormat`** (required, else WirePlumber
   logs *"no usable format found"* and never proceeds).
3. `ClientNode.PortUpdate` — port `EnumFormat` + the param list the server may get/set.
4. Server fixates the format → `port_set_param(Format)`.
5. We answer with `PortUpdate` carrying **only `Buffers`** (re-sending `Format` restarts negotiation
   in a loop). The `Buffers` object **must** advertise `dataType` = MemFd, or the daemon fails with
   `alloc buffers: Invalid argument` (cross-process buffers need shareable memory).
6. `Core.AddMem` (memfds) → `ClientNode.PortUseBuffers` (buffers mapped) → `PortSetIO(Buffers)`
   (the `spa_io_buffers` hand-off area).
7. `ClientNode.Transport` (our activation record + eventfds) + `ClientNode.SetActivation` (peers).
8. `ClientNode.Command(Start)` → the node is **running** and appears in `pw-top` as a follower of
   the real sink driver (`driver_id` = the sink's node id).

At this point the RT process cycle *would* run: pull a quantum from `fill`, write it into a mapped
buffer, publish via `spa_io_buffers`, and trigger the sink via its activation. The machinery is all
there and correct.

## What's missing (the one wall) 🧱

**The per-cycle driver→client RT trigger never fires.** Our activation sits at
`status=FINISHED(3)`, `state[0].required=1`, `pending=0`, `driver_id=<sink>` — but the sink driver
**never writes our eventfd** and never CASes our status `FINISHED → NOT_TRIGGERED`. So
`process_cycle()` is never entered and no PCM is ever produced. (Confirmed: neither `readfd` nor
`writefd` becomes readable; self-writing our own `readfd` doesn't make it readable either.)

Root cause: that "prepare + trigger the follower" step is only wired up by the session manager's
**managed** linking. A **bare client-node is not adapter-wrapped**, so:

- **WirePlumber will not autoconnect it.** `pw_stream` never exports a bare client-node for normal
  playback — it creates a local **`adapter`** node (audioconvert, `adapt.follower.spa-node`) and
  exports *that*. The adapter converts interleaved→per-channel and presents the single convert
  port WirePlumber's policy expects.
- **A hand-made `pw-link` creates the link but not the driver schedule** — the node joins the
  graph (`driver_id` is set) but the driver's per-cycle follower list is not rebuilt to include us.

So the negotiation, buffers, io, and activation are all correct; the missing piece is being
*managed-linked* so the driver actually schedules us.

## What needs to be done (next, in order)

1. **Make the node linkable so the session manager schedules it.** Pick one:
   - **(a) audioconvert adapter (general).** Present a convert-capable node: advertise
     `SPA_PARAM_PortConfig`, handle the session manager configuring it, and do the
     interleaved→per-channel + rate conversion ourselves. This is what `pw_stream` does via
     libpipewire's adapter — a large piece (the audioconvert/resampler), but it makes *any* format
     work with autoconnect.
   - **(b) DSP mono ports (smaller, uncertain policy).** Present N mono `DSP_F32` ports (one per
     channel) with channel props (`audio.channel=FL/FR`), deinterleave into N buffers per cycle, and
     rely on WirePlumber's pro-audio policy to link them 1:1 to the device's `playback_FL/FR`. No
     conversion, but the output rate must equal the graph rate (resample upstream otherwise). The
     `AudioConfig::dsp` path already builds a single mono DSP port and got furthest through buffer
     negotiation — extend it to the full channel set and verify WirePlumber links it.
   - Verify success by: `pw-top` shows non-zero process time on our node, and (with
     `STREAMCRAFT_PW_DEBUG=1`) the `process cycle` traces increment.
2. **Wire `PipeWireAudioSink` onto `native::play`.** Reuse the lock-free `ring` (the RT `fill`
   callback = `Consumer::pull`) and keep the element contract: `AudioDeviceClock` provider,
   pause/seek (see [[clock-wired]]). Then **drop the `pipewire = "0.8"` dependency** from
   `Cargo.toml` and delete the libpipewire `sink.rs`/`ring.rs` binding.
3. **Move the socket + eventfd onto the reactor** (`ctx.io()`) instead of the blocking `poll(2)` /
   libc loop, per the reactor-IO rule ([[reactor-io-rule]]) — same follow-up as `pf-present`.
4. **Clock/quantum:** `process_cycle` currently derives the per-cycle frame count from the buffer
   `maxsize` with a fallback to `position.clock.duration` (activation offset 656). Once audio flows,
   read the quantum from `spa_io_position` cleanly and feed the device clock.

## Gotchas already paid for (don't re-discover these)

- **A PipeWire dict is a nested `Struct{ Int(n), (String,String)* }`** (`push_dict` /
  `parse_dict_struct`), NOT inline. Inline gave `-EINVAL "invalid message id:1 op:2"` on
  `UpdateProperties`. `Core.Info` / `Registry.Global` props are nested structs too.
- **fd handling must be per-message.** fds were pooled in one global FIFO; a message consuming the
  wrong count made a later `AddMem`'s `take_fd()` return `None` → a *missing* mem (io mapped to
  `0x0`). `conn.rs` now isolates each message's fds by its header `n_fds` (`msg_fds`) and closes
  un-taken ones. Keep this.
- **`Buffers.dataType` must include MemFd** for cross-process links.
- **Device sink ports are mono `DSP_F32`** (`0x206` = `F32P`, `mediaSubtype = dsp = 2`), not
  interleaved — hence the format/stride mismatch on a direct stereo link.
- **`pw_node_activation` head offsets** (ABI-frozen, `private.h`): `status@0`,
  `state[0].required@12`, `pending@16`, `driver_id@552`, `position.clock.duration@656`. Touch only
  the head via SeqCst atomics on the mmap.
- POD `size` = body length excluding the 8-byte header and trailing pad; each POD occupies
  `round_up_8(8+size)`. `Long`/`Double`/`Fd` size=8; `Int`/`Id`/`Bool`/`Float` size=4 (+4 pad).

## References

See `REFERENCES.md`. The 1.6.5 sources consulted for the ClientNode/audio path:
`module-client-node/protocol-native.c`, `remote-node.c`, `src/pipewire/private.h`,
`src/pipewire/buffers.c`, `src/pipewire/stream.c`, and the `spa-0.2` headers.
