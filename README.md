[![Latest release](https://img.shields.io/crates/v/profluens)](https://crates.io/crates/profluens)
[![Docs](https://docs.rs/profluens/badge.svg)](https://docs.rs/profluens)

# Profluens

An ultra light weight, general purpose data/multimedia streaming/processing graph
framework. Inspired by GStreamer, but data-oriented, dependency-free at the core, and
built for performance and perfect latency handling above all else.

> **Status: rearchitecture in progress.** The original prototype was removed; this is
> the clean-slate skeleton. See [`profluens.md`](profluens.md) for the full design
> — vision, architecture, API sketches, milestones, and build order.

## Workspace

| Crate                  | Role                                                              |
|------------------------|------------------------------------------------------------------|
| [`core`](core)         | Buffers, formats, topology, scheduler, clock, events, bus. **Zero dependencies.** |
| [`elements`](elements) | Built-in pure-Rust elements (filesrc, queue, tee, testsrc, sinks). |
| [`audio`](audio)       | POD audio formats + typed views over the opaque buffer.          |
| [`video`](video)       | POD video formats + typed views over the opaque buffer.          |
| [`scope`](scope)       | Slint inspector GUI + MCP server over the introspection protocol. |

First-party hand-written codec/container crates (`pf-flac`, `pf-opus`, `pf-vp9`,
`pf-mkv`, …) land as the build order reaches them.

## Building

```sh
cargo check    # the skeleton compiles; implementations are TODO(step N) per profluens.md
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
