# References — native PipeWire protocol client (`src/native/`)

The `src/native/` client implements the PipeWire native wire protocol and the SPA POD
serialization format from scratch (no `libpipewire`). These are the normative references it
was written against. It is a clean-room implementation from the protocol description — no
PipeWire source was copied.

## Wire protocol & POD format

- **PipeWire Native Protocol** — message framing, socket resolution, the connection handshake,
  Core/Client/Registry opcodes, footers.
  <https://docs.pipewire.org/page_native_protocol.html>
- **SPA POD** — the Plain-Old-Data serialization format (headers, alignment, every type's byte
  layout).
  <https://docs.pipewire.org/page_spa_pod.html>
- **SPA Pod (API)** — `struct spa_pod` (`size`/`type`), `SPA_POD_SIZE`, 8-byte rounding.
  <https://docs.pipewire.org/group__spa__pod.html>

## Interface & type enumerations

- **Core / Client / Registry interfaces** — method & event opcodes, interface versions.
  <https://docs.pipewire.org/group__pw__core.html>,
  <https://docs.pipewire.org/group__pw__registry.html>,
  <https://docs.pipewire.org/group__pw__client.html>
- **SPA types** (`enum spa_type`) — the basic POD type ids (None=1 … Choice=19, Pod=20).
  <https://docs.pipewire.org/group__spa__types.html>

## What we implement (verified against the live daemon)

| Layer            | Reference concept                                             | Module      |
| ---------------- | ------------------------------------------------------------ | ----------- |
| POD codec        | `struct spa_pod` header + per-type layout, 8-byte alignment  | `pod.rs`    |
| Message framing  | 16-byte header: `id`, `(opcode<<24)\|size`, `seq`, `n_fds`   | `wire.rs`   |
| Socket + fds     | `AF_UNIX` `pipewire-0`, `SCM_RIGHTS` fd passing              | `conn.rs`   |
| Handshake        | `Core.Hello(3)`, `Client.UpdateProperties`, `Sync`/`Done`    | `proto.rs`  |
| Registry         | `Core.GetRegistry`, `Registry.Global`                        | `proto.rs`  |

Key facts baked into the code (all from the references above):

- Header word 1 packs `opcode` in the high 8 bits and payload `size` in the low 24 bits; `size`
  spans the payload **plus** an optional footer. The client emits no footer; an incoming footer
  is skipped by parsing only the leading payload `Struct`.
- POD `size` is the body length excluding the 8-byte header and excluding trailing padding; on
  the wire every POD occupies `round_up_8(8 + size)` bytes. All multi-byte words are
  **native-endian**.
- A PipeWire dict marshals **inline** as `Int(n_items)` followed by `n` × (`String` key,
  `String` value) — not wrapped in a nested `Struct`.
- `fd` args ride `SCM_RIGHTS`; the `Fd` POD stores only the fd's index within the message.

## ClientNode / audio path (`src/native/spa.rs` + `audio.rs`)

Implemented from the PipeWire 1.6.5 sources + `spa-0.2` headers (ABI-exact values):

- **`pw_client_node`** interface + wire marshalling — `module-client-node/protocol-native.c`,
  `remote-node.c`, `pipewire/extensions/client-node.h`.
  <https://docs.pipewire.org/group__pw__client__node.html>
- **SPA Param** objects — `Format`, `Buffers`, `IO`, `Meta` (POD `Object` + `Choice`);
  `spa/param/*.h`, `spa/param/audio/raw.h`.
  <https://docs.pipewire.org/group__spa__param.html>
- **`spa_io_buffers`** + `spa_io_position`/`spa_io_clock` — `spa/node/io.h`.
  <https://docs.pipewire.org/structspa__io__buffers.html>
- **`pw_node_activation`** (the RT scheduling record: `status`, `state[].required/pending`) —
  `src/pipewire/private.h`. Buffer allocation / `dataType` selection — `src/pipewire/buffers.c`.

Validated against the live daemon through `Command Start` (node join as a follower of the sink
driver). The last step — the per-cycle driver→client eventfd trigger — needs session-manager
*managed* linking; see `audio.rs` module docs. Reference for how `pw_stream` self-adapts (the
adapter factory + `adapt.follower.spa-node`): `src/pipewire/stream.c`.
