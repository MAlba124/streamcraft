# Ogg Mapping for FLAC

Interpretation notes for FLAC-in-Ogg de-framing, per the codec conventions (spec:
First-party codecs — "Spec errata and interpretation decisions get documented in
`spec/NOTES.md` per crate"). The normative text is the xiph "Ogg Mapping for FLAC"
(<https://xiph.org/flac/ogg_mapping.html>), which layers a FLAC elementary stream on top
of the Ogg container (RFC 3533, `../../ogg/spec/rfc3533.txt`). The FLAC bitstream itself
is `rfc9639.txt` (RFC 9639). This mapping is what [`OggFlacDeframe`](../src/oggflac.rs)
implements; it is the inverse of a muxer that wraps a native FLAC stream into Ogg packets.

## Logical-bitstream layout

An Ogg-mapped FLAC logical bitstream is a sequence of Ogg **packets** (reassembled from
pages by the demuxer, one packet per input unit here). The mapping defines three kinds of
packet, in order:

### 1. The mapping header — the FIRST packet of the logical bitstream

Byte layout (the first byte of the packet is `0x7F`, outside the FLAC frame sync `0xFF`
and the metadata-block-type range, so it identifies the mapping packet):

```text
 0   packet type            0x7F                        (1 byte)
 1   signature              "FLAC"                       (4 bytes)  0x46 0x4C 0x41 0x43
 5   mapping major version  0x01                         (1 byte)
 6   mapping minor version  0x00                         (1 byte)
 7   header packet count    N, BIG-endian                (2 bytes)
 9   native FLAC signature  "fLaC"                       (4 bytes)  0x66 0x4C 0x61 0x43
13   STREAMINFO             4-byte block header + 34-byte body (38 bytes)
```

Quoting the mapping:

- The header packet count is *"a two-byte, big-endian binary number signifying the number
  of header (non-audio) packets, not including this one"*. It may be `0x0000` for
  "unknown". This is the **only** big-endian field the mapping adds; everything from
  offset 9 onward is native FLAC (RFC 9639), whose own multi-byte fields are big-endian
  too, and whose Ogg page headers (§6, RFC 3533) are little-endian.
- The native `"fLaC"` signature and the STREAMINFO metadata block that follow are exactly
  the first bytes of a native `.flac` file. STREAMINFO's block header carries the usual
  last-metadata-block flag; it is set iff there are no further metadata blocks (`N == 0`).
- *"This first packet is the only packet in the first page of the stream. This results in
  a first Ogg page of exactly 79 bytes"* — the mapping packet payload is 51 bytes (9
  mapping-prefix + 4 `fLaC` + 4 STREAMINFO block header + 34 STREAMINFO body), and the 79
  figure adds the 27-byte Ogg page header + 1-byte segment table (28 + 51 = 79). *"This
  first page is marked 'beginning of stream' in the page flags."* De-framing does not
  depend on the page boundary — the demuxer has already reassembled the packet — but it
  relies on the 51-byte mapping packet arriving whole in the first buffer (it fits any
  sane pool slot; the default is 128 KiB).

### 2. The remaining metadata blocks — the next `N` packets

*"Each such packet will contain a single native FLAC metadata block"* — one metadata block
per packet, verbatim (SEEKTABLE, VORBIS_COMMENT, PADDING, …). The last of them carries the
native last-metadata-block flag, exactly as in a native stream. These are forwarded
byte-for-byte.

### 3. The audio frames — every subsequent packet

*"Each packet corresponds to one FLAC audio frame"*; *"this first byte will be always
0xFF"* (the native FLAC frame sync). Forwarded byte-for-byte.

## De-framing decision (what `OggFlacDeframe` does)

De-framing reconstructs the **native FLAC byte stream**: `"fLaC"` + STREAMINFO + any
further metadata blocks + all frames, in order — byte-identical to a native `.flac` file,
which `FlacDec`'s `StreamDecoder` already decodes. Concretely:

- **First packet:** validate the `0x7F "FLAC"` signature and skip the **9-byte** mapping
  prefix (packet type + signature + 2 version bytes + 2 count bytes); emit the remainder
  (`"fLaC"` + STREAMINFO …) verbatim.
- **Every later packet:** emit verbatim.

So after the one 9-byte strip the element simply concatenates every packet payload — the
metadata blocks and frames are *already* native FLAC on the wire; the mapping only prepends
a 9-byte header to the very first packet and repackages the native stream one frame per
packet, both of which are undone by "strip 9 bytes from the first packet, forward the
rest". The element carries raw `bytes` on both pads and does **not** decode audio or
announce `audio/raw`; the downstream `flacdec` announces the concrete format from the
STREAMINFO it now sees at the head of the reconstructed native stream (spec: Formats —
dynamic caps). Milestone pipeline: `filesrc ! oggdemux ! oggflacdeframe ! flacdec !
pipewireaudiosink`.

## Decisions / interpretation

- **Header packet count is advisory, not load-bearing.** Because metadata blocks and
  frames are all forwarded verbatim, de-framing does not need to count the `N` header
  packets to know where audio starts: the native last-metadata-block flag already marks
  the metadata/audio boundary inside the reconstructed stream, and the downstream decoder
  reads it. The count is validated to be well-formed (the field is read) but the boundary
  is not enforced by this element — a smaller, simpler contract that cannot desynchronise.
- **Robustness.** A first packet shorter than 9 bytes, or one lacking the `0x7F "FLAC"`
  signature, is a malformed stream and yields an error (never a panic) — the deframer
  parses untrusted input (spec: a crash on bad input is a P0).
- **Version.** Only mapping major version 1 is defined; the version bytes are read but a
  mismatch is not currently rejected (there is only one version). If a version 2 ever
  changes the prefix layout this becomes load-bearing.
- **Single logical bitstream.** `OggDemux` already filters to the first bos serial, so the
  deframer sees exactly one FLAC logical bitstream. Multi-stream (e.g. FLAC + a second
  codec grouped in one physical Ogg stream) is the demuxer's dynamic-pad follow-up, not
  this element's concern.
